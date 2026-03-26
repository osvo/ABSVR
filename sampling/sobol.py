"""Sobol sampling utilities for benchmark runs."""

import math

import numpy as np
from scipy import stats
from scipy.stats import qmc


def _is_power_of_two(value: int) -> bool:
    """Return ``True`` when ``value`` is a positive power of two."""

    return value > 0 and (value & (value - 1)) == 0


class SobolNormalSampler:
    """Generate Sobol samples transformed to Gaussian space."""

    def __init__(
        self,
        mean: np.ndarray,
        sigma: np.ndarray,
        scramble: bool = False,
    ) -> None:
        self.mean = np.asarray(mean, dtype=float)
        self.sigma = np.asarray(sigma, dtype=float)
        if self.mean.shape != self.sigma.shape:
            raise ValueError("mean and sigma must share the same shape")
        self.dimension = self.mean.size
        self._sampler = qmc.Sobol(d=self.dimension, scramble=scramble)

    def _draw_unit_hypercube(self, n_samples: int) -> np.ndarray:
        """Draw Sobol points, using base-2 draws only when safe."""

        if self._sampler.num_generated == 0:
            if _is_power_of_two(n_samples):
                exponent = int(math.log2(n_samples))
                return self._sampler.random_base2(exponent)

            exponent = int(math.log2(n_samples))
            base_count = 1 << exponent
            base_draw = self._sampler.random_base2(exponent) if base_count else np.empty((0, self.dimension))
            remainder = n_samples - base_count
            if remainder <= 0:
                return base_draw[:n_samples]

            tail_draw = self._sampler.random(remainder)
            if base_draw.size == 0:
                return tail_draw
            return np.concatenate((base_draw, tail_draw), axis=0)

        return self._sampler.random(n_samples)

    def draw(self, n_samples: int) -> np.ndarray:
        """Draw ``n_samples`` from the Sobol sequence mapped to Gaussian space."""

        if n_samples <= 0:
            raise ValueError("n_samples must be a positive integer")
        unit = self._draw_unit_hypercube(n_samples)
        tiny = np.nextafter(0.0, 1.0)
        unit = np.clip(unit, tiny, 1.0 - tiny)
        z = stats.norm.ppf(unit)
        return self.mean + self.sigma * z

    def fast_forward(self, n: int) -> None:
        """Advance the underlying Sobol sequence by ``n`` draws."""

        steps = int(n)
        if steps <= 0:
            return
        try:
            self._sampler.fast_forward(steps)
        except AttributeError as exc:  # pragma: no cover - older SciPy
            raise RuntimeError("Sobol fast_forward is unavailable in this SciPy version.") from exc
