"""Adaptive Bayesian SVR algorithm implementation in Python."""

import json
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable, Iterable, Optional

import numpy as np
from scipy import stats

try:
    from surrogate.svr import (
        svr_predict,
        svr_predict_ensemble,
        train_svr,
        train_svr_ensemble,
    )
    from surrogate.svr.kernel_utils import normalize_kernel_name
    from sampling import SobolNormalSampler
    from utilities import (
        global_random_state,
        lhs_normal,
        lhs_uniform,
        maybe_print_memory_usage,
        reset_global_seed,
    )
    from .ood_scoring import apply_tail_policy, score_standard_distance
except ImportError:
    from ..surrogate.svr import (
        svr_predict,
        svr_predict_ensemble,
        train_svr,
        train_svr_ensemble,
    )
    from ..surrogate.svr.kernel_utils import normalize_kernel_name
    from ..sampling import SobolNormalSampler
    from ..utilities import (
        global_random_state,
        lhs_normal,
        lhs_uniform,
        maybe_print_memory_usage,
        reset_global_seed,
    )
    from ..absvr_core.ood_scoring import (
        apply_tail_policy,
        score_standard_distance,
    )
from .checkpointing import load_checkpoint, save_checkpoint
from .candidate_selection import select_best_candidate
from .learning_function import compute_learning_function
from .probability_estimation import estimate_probability_of_failure
from .problem_setup import load_custom_benchmark, prepare_problem_definition


@dataclass
class AdaptiveSVRResult:
    """Container for the main outputs of :func:`adaptive_svr`."""

    probability_of_failure: float
    reliability_index: float
    total_evaluations: int
    phase_timings: dict[str, float] = field(default_factory=dict)
    phase_counts: dict[str, int] = field(default_factory=dict)
    kernel: str = "gaussian"
    covariance: str = "Gaussian"

    def to_dict(self) -> dict[str, float | int]:
        return {
            "probability_of_failure": float(self.probability_of_failure),
            "reliability_index": float(self.reliability_index),
            "total_evaluations": int(self.total_evaluations),
            "phase_timings": {str(key): float(value) for key, value in self.phase_timings.items()},
            "phase_counts": {str(key): int(value) for key, value in self.phase_counts.items()},
            "kernel": str(self.kernel),
            "covariance": str(self.covariance),
        }


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(float(value))
    except ValueError:
        return default


VM = 3.0  # Initial design of experiments span = ±3 standard deviations
N0 = 15  # Number of initial Latin-hypercube samples
N_MCS = _env_int("ABSVR_N_MCS", 0)  # 0 = auto-scale at runtime based on n_dim
MCS_ENRICH_SIZE = _env_int("ABSVR_MCS_ENRICH_SIZE", int(1e5))  # Candidates per enrichment
MAX_MCS_POOL_SIZE = _env_int("ABSVR_MAX_MCS_POOL_SIZE", int(2e6))  # Hard cap on pool size
COV_PF_TOL = 0.05  # Target coefficient of variation for the failure probability
MIN_BURN_IN = 50  # Minimum iterations before evaluating convergence
BETA_STABILITY_TOL = 5e-3  # 0.5% relative change in beta between consecutive iterations (relaxed from 1e-3)

def _evaluate_convergence(
    pf_history: np.ndarray,
    total_evaluations: int,
    n0: int,
) -> tuple[bool, float]:
    """Check reliability index stability over 3 consecutive iterations.

    Returns ``(converged, beta)`` where *converged* is True when the
    relative change in beta stays below ``BETA_STABILITY_TOL`` for
    three consecutive pairs of iterations.
    """
    stability = False
    beta_curr = float("nan")

    if total_evaluations > (n0 + 7):
        betas = [-stats.norm.ppf(pf_history[total_evaluations - 1 - k]) for k in range(4)]
        beta_curr = betas[0]

        if all(b != 0.0 and np.isfinite(b) for b in betas):
            deltas = [abs(betas[k] - betas[k + 1]) / abs(betas[k + 1]) for k in range(3)]
            stability = all(d < BETA_STABILITY_TOL for d in deltas)

    elif total_evaluations > n0:
        beta_curr = -stats.norm.ppf(pf_history[total_evaluations - 1])

    return stability, beta_curr

PHASE_NAMES: tuple[str, ...] = (
    "checkpoint_io",
    "initial_design_evaluation",
    "initial_pool_generation",
    "ood_scoring",
    "model_training",
    "surrogate_prediction",
    "candidate_scoring",
    "candidate_evaluation",
    "stopping_checks",
    "pool_enrichment",
    "result_export",
)


def _evaluate_fun_single(point: np.ndarray, fun, fun_par) -> float:
    value = np.asarray(fun(point[None, :], fun_par), dtype=float)
    if value.size == 0:
        return float("nan")
    return float(value.reshape(-1)[0])


def _parallel_eval_wrapper(args) -> float:
    point, fun, fun_par = args
    return _evaluate_fun_single(point, fun, fun_par)


def _evaluate_batch(points: Iterable[np.ndarray], fun, fun_par, n_workers: Optional[int]) -> np.ndarray:
    points_list = [np.asarray(pt, dtype=float) for pt in points]
    if not points_list:
        return np.array([], dtype=float)

    if n_workers is None or n_workers <= 1 or len(points_list) == 1:
        stacked = np.vstack(points_list)
        if stacked.shape[0] == 1:
            values = [_evaluate_fun_single(stacked[0], fun, fun_par)]
            return np.asarray(values, dtype=float)
        try:
            vector_values = np.asarray(fun(stacked, fun_par), dtype=float).reshape(-1)
        except Exception:
            vector_values = np.array([])
        if vector_values.size == stacked.shape[0]:
            return vector_values
        values = [_evaluate_fun_single(pt, fun, fun_par) for pt in points_list]
        return np.asarray(values, dtype=float)

    try:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(_parallel_eval_wrapper, (pt, fun, fun_par)) for pt in points_list]
            results = [future.result() for future in futures]
    except Exception:
        results = [_evaluate_fun_single(pt, fun, fun_par) for pt in points_list]
    return np.asarray(results, dtype=float)


def run_adaptive_svr(
    test_example: str,
    max_iter: int,
    *args: int,
    n_workers: Optional[int] = None,
    ensemble_size: int = 1,
    svr_backend: str = "standard",
    svr_kernel: str = "gaussian",
    random_seed: Optional[int] = None,
    checkpoint_path: Optional[str] = None,
    checkpoint_frequency: int = 1,
    resume_from: Optional[str] = None,
    result_output_path: Optional[str] = None,
    custom_benchmark: Optional[str] = None,
    custom_factory: Optional[str] = None,
    problem_definition: Optional[tuple] = None,
    mc_pool_override: Optional[np.ndarray] = None,
    v_pdf_pool_override: Optional[np.ndarray] = None,
    sobol_draw_count_override: Optional[int] = None,
    learning_w_grad: float = 1.0,
    svr_bounds_mode: str | None = None,
    tail_policy: str = "none",
    tail_alpha: float = 1.0,
    svr_optim_method: str = "evidence",

    n0: int = 0,
    vm: float = 3.0,
    cov_pf_tol: float = 0.05,
    gamma: float | None = None,
    poly_degree: int | None = None,
    poly_coef0: float | None = None,
    c_init: float | None = None,
    epsilon_init: float | None = None,
    min_samples: int = 0,
    initial_design: str = "uniform_box",
    svr_trainer: Callable[..., dict[str, Any]] | None = None,
    pruning_callback: Callable[[float, int], bool] | None = None,
    verbose: bool = True,
) -> AdaptiveSVRResult:
    """Run the adaptive SVR workflow for the selected benchmark.

    Parameters
    ----------
    svr_kernel:
        Kernel identifier understood by :func:`train_svr`.
    checkpoint_path:
        Optional path to persist checkpoints during the run.
    checkpoint_frequency:
        Write a checkpoint every ``checkpoint_frequency`` accepted enrichment steps
        (defaults to 1, meaning every step).
    resume_from:
        Optional checkpoint path to resume from. The stored configuration must match
        the parameters supplied to the current call.
    result_output_path:
        Optional JSON file that will receive the final ``AdaptiveSVRResult`` summary.
    custom_benchmark:
        Optional custom benchmark spec (module path or file) resolved via
        ``load_custom_benchmark``.
    custom_factory:
        Optional factory function name for ``custom_benchmark``.
    problem_definition:
        Optional pre-loaded benchmark tuple to bypass ``prepare_problem_definition``.
    mc_pool_override:
        Optional precomputed Monte Carlo pool to reuse (copied before use).
    v_pdf_pool_override:
        Optional precomputed PDF weights for the Monte Carlo pool (copied before use).
    sobol_draw_count_override:
        Optional Sobol draw count used to advance the sampler when a pool is reused.
    learning_w_grad:
        Weight for the gradient term in the learning function (0 disables it).
    svr_bounds_mode:
        Bound profile for the SVR optimizer ("python" or "baseline").
    tail_policy:
        Behaviour of the out-of-distribution adjustment (``"none"``, ``"down"``, ``"explore"``).
    tail_alpha:
        Intensity multiplier applied to the OOD scores when the tail policy is active.

    n0:
        Number of initial Latin-hypercube samples.
    vm:
        Initial design of experiments span (±vm standard deviations).
    cov_pf_tol:
        Target coefficient of variation for the failure probability.
    gamma:
        Initial value for the SVR kernel parameter (theta).
    poly_degree:
        Degree for the polynomial kernel (sets ABSVR_POLY_DEGREE env var).
    poly_coef0:
        Coef0 for the polynomial kernel (sets ABSVR_POLY_COEF0 env var).
    c_init:
        Initial SVR regularization parameter C. Higher values = less regularization.
    epsilon_init:
        Initial ε-insensitive tube width. Larger values ignore small errors.
    min_samples:
        Minimum number of adaptively added points before stability may stop the
        run. Values below the built-in burn-in retain the historical behaviour.
    initial_design:
        ``"uniform_box"`` preserves the historical finite-box LHS;
        ``"normal_lhs"`` stratifies the probability space of independent
        normal variables before applying any distribution transform.
    svr_trainer:
        Optional drop-in replacement for :func:`train_svr`, used for a single
        standard model. This enables auditable deterministic tuning policies
        without changing the historical trainer used by default.
    pruning_callback:
        Function called as callback(pf, step) -> bool. If True, stops execution.
    """

    if svr_backend and svr_backend.lower() != "standard":
        raise ValueError("Only the 'standard' SVR backend is supported.")

    if checkpoint_frequency <= 0:
        raise ValueError("checkpoint_frequency must be a positive integer.")

    initial_design = str(initial_design).strip().lower()
    if initial_design not in {"uniform_box", "normal_lhs"}:
        raise ValueError("initial_design must be 'uniform_box' or 'normal_lhs'.")
    if svr_trainer is not None and int(ensemble_size) != 1:
        raise ValueError("A custom svr_trainer cannot be combined with an ensemble.")
    trainer = svr_trainer or train_svr

    checkpoint_target = Path(checkpoint_path).expanduser() if checkpoint_path else None
    resume_source = Path(resume_from).expanduser() if resume_from else None
    result_target = Path(result_output_path).expanduser() if result_output_path else None

    tail_policy = (tail_policy or "none").lower()
    if tail_policy not in {"none", "down", "explore"}:
        raise ValueError("tail_policy must be one of 'none', 'down', or 'explore'.")
    tail_alpha = float(tail_alpha)
    if tail_alpha < 0.0:
        raise ValueError("tail_alpha must be non-negative.")

    if poly_degree is not None:
        os.environ["ABSVR_POLY_DEGREE"] = str(int(poly_degree))
    if poly_coef0 is not None:
        os.environ["ABSVR_POLY_COEF0"] = str(float(poly_coef0))

    canonical_kernel, covariance_name = normalize_kernel_name(svr_kernel)

    phase_timings: dict[str, float] = {name: 0.0 for name in PHASE_NAMES}
    phase_counts: dict[str, int] = {name: 0 for name in PHASE_NAMES}

    resume_state: Optional[dict[str, Any]] = None
    if resume_source is not None:
        load_start = perf_counter()
        resume_state = load_checkpoint(resume_source)
        load_duration = perf_counter() - load_start
        phase_timings["checkpoint_io"] = phase_timings.get("checkpoint_io", 0.0) + load_duration
        phase_counts["checkpoint_io"] = phase_counts.get("checkpoint_io", 0) + 1
        stored_timings = resume_state.get("phase_timings", {}) if resume_state is not None else {}
        if isinstance(stored_timings, dict):
            for phase, value in stored_timings.items():
                phase_timings[phase] = phase_timings.get(phase, 0.0) + float(value)
        stored_counts = resume_state.get("phase_counts", {}) if resume_state is not None else {}
        if isinstance(stored_counts, dict):
            for phase, value in stored_counts.items():
                phase_counts[phase] = phase_counts.get(phase, 0) + int(value)

    else:
        resume_state = None

    if checkpoint_target is None and resume_source is not None:
        checkpoint_target = resume_source

    if resume_state is not None:
        stored_example = str(resume_state.get("test_example", test_example))
        if stored_example != test_example:
            raise ValueError(
                f"Checkpoint was created for benchmark {stored_example!r}, "
                f"but {test_example!r} was requested."
            )
        stored_args = tuple(resume_state.get("args", []))
        if stored_args != tuple(args):
            raise ValueError(
                f"Checkpoint arguments {stored_args} do not match the requested {tuple(args)}."
            )
        stored_backend = resume_state.get("svr_backend")
        if stored_backend is not None and str(stored_backend).lower() != "standard":
            raise ValueError(f"Checkpoint uses unsupported backend {stored_backend!r}.")
        stored_kernel = str(resume_state.get("svr_kernel", canonical_kernel)).lower()
        if stored_kernel != canonical_kernel:
            raise ValueError(
                f"Checkpoint expects kernel={stored_kernel!r}, but {canonical_kernel!r} was provided."
            )
        stored_covariance = str(resume_state.get("svr_covariance", covariance_name))
        if stored_covariance.lower() != covariance_name.lower():
            raise ValueError(
                f"Checkpoint expects covariance={stored_covariance!r}, but {covariance_name!r} was provided."
            )
        stored_ensemble = int(resume_state.get("ensemble_size", ensemble_size))
        if stored_ensemble != int(ensemble_size):
            raise ValueError(
                f"Checkpoint expects ensemble_size={stored_ensemble}, but {ensemble_size} was provided."
            )
        stored_tail_policy = str(resume_state.get("tail_policy", tail_policy)).lower()
        if stored_tail_policy != tail_policy:
            raise ValueError(
                f"Checkpoint expects tail_policy={stored_tail_policy!r}, but {tail_policy!r} was provided."
            )
        stored_tail_alpha = float(resume_state.get("tail_alpha", tail_alpha))
        if not np.isclose(stored_tail_alpha, tail_alpha, atol=1e-9, rtol=1e-6):
            raise ValueError(
                f"Checkpoint expects tail_alpha={stored_tail_alpha}, but {tail_alpha} was provided."
            )
        stored_initial_design = str(
            resume_state.get("initial_design", initial_design)
        ).lower()
        if stored_initial_design != initial_design:
            raise ValueError(
                f"Checkpoint expects initial_design={stored_initial_design!r}, "
                f"but {initial_design!r} was provided."
            )
        trainer_state = resume_state.get("svr_trainer_state")
        if trainer_state is not None and svr_trainer is not None:
            restore_trainer = getattr(svr_trainer, "load_state_dict", None)
            if restore_trainer is None:
                raise ValueError(
                    "Checkpoint contains custom trainer state, but svr_trainer "
                    "does not implement load_state_dict()."
                )
            restore_trainer(trainer_state)

    def _time_phase(phase: str, func: Callable[..., Any], *func_args, **func_kwargs):
        start = perf_counter()
        try:
            return func(*func_args, **func_kwargs)
        finally:
            duration = perf_counter() - start
            phase_timings[phase] = phase_timings.get(phase, 0.0) + duration
            phase_counts[phase] = phase_counts.get(phase, 0) + 1

    stored_seed = resume_state.get("random_seed") if resume_state is not None else None
    effective_seed = random_seed if random_seed is not None else stored_seed

    if effective_seed is not None:
        reset_global_seed(int(effective_seed))
        seed_used: Optional[int] = int(effective_seed)
    else:
        global_random_state()
        seed_used = None

    if resume_state is not None:
        rng_meta = resume_state.get("rng_state")
        if isinstance(rng_meta, dict):
            try:
                rng = global_random_state()
                bit_generator = str(rng_meta.get("bit_generator", "MT19937"))
                state_vector = np.asarray(rng_meta.get("state", []), dtype=np.uint32)
                pos = int(rng_meta.get("pos", 0))
                has_gauss = int(rng_meta.get("has_gauss", 0))
                cached_gaussian = float(rng_meta.get("cached_gaussian", 0.0))
                rng.set_state((bit_generator, state_vector, pos, has_gauss, cached_gaussian))
            except Exception:
                pass

    if problem_definition is not None:
        mu_x, sigma_x, n_dim, fun_par, fun, _, dist_type = problem_definition
    elif custom_benchmark is not None:
        mu_x, sigma_x, n_dim, fun_par, fun, _, dist_type = load_custom_benchmark(
            custom_benchmark, factory=custom_factory
        )
    else:
        mu_x, sigma_x, n_dim, fun_par, fun, _, dist_type = prepare_problem_definition(test_example, *args)

    sobol_sampler = SobolNormalSampler(mu_x, sigma_x)

    # For lognormal distribution, we sample from N(mu, sigma) then transform via exp()
    is_lognormal = dist_type == "lognormal"

    def _transform_samples(z: np.ndarray) -> np.ndarray:
        """Transform samples from underlying normal to target distribution."""
        if is_lognormal:
            return np.exp(z)
        return z

    def _compute_pdf(x: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Compute joint PDF of samples.

        For lognormal: f(x) = f_Z(z) / x where z = log(x)
        For normal: f(x) = f_Z(x) directly

        Uses log-space computation to avoid overflow in high dimensions.
        """
        # Compute log-PDF in standard normal space for numerical stability
        # log(pdf_z) = -0.5 * ((z - mu) / sigma)^2 - log(sigma * sqrt(2*pi))
        standardized = (z - mu_x) / sigma_x
        log_pdf_per_dim = -0.5 * standardized**2 - np.log(sigma_x * np.sqrt(2 * np.pi))
        log_pdf_joint = np.sum(log_pdf_per_dim, axis=1)

        if is_lognormal:
            # Jacobian for lognormal: f_X(x) = f_Z(z) / prod(x)
            # log(f_X) = log(f_Z) - sum(log(x)) = log(f_Z) - sum(z)
            log_pdf_joint = log_pdf_joint - np.sum(z, axis=1)

        # Convert from log-space with numerical stability (subtract max before exp)
        log_pdf_max = np.max(log_pdf_joint)
        pdf_joint = np.exp(log_pdf_joint - log_pdf_max)
        return pdf_joint

    if resume_state is None:
        # Dimension-adaptive initial DoE size: n0=0 → auto-scale as
        # max(15, 2*n_dim + 1), following surrogate modeling guidelines
        # (Loeppky et al. 2009, Bourinet 2016).
        if n0 <= 0:
            n0 = max(15, 2 * n_dim + 1)
        if initial_design == "normal_lhs":
            doe_z, _ = lhs_normal(mu_x, sigma_x, n0)
        else:
            doe_z, _ = lhs_uniform(mu_x, vm * sigma_x, n0)
        doe = _transform_samples(doe_z)
        design_eval_start = perf_counter()
        g = np.asarray(fun(doe, fun_par), dtype=float).reshape(-1)
        design_eval_duration = perf_counter() - design_eval_start
        phase_timings["initial_design_evaluation"] = (
            phase_timings.get("initial_design_evaluation", 0.0) + design_eval_duration
        )
        phase_counts["initial_design_evaluation"] = phase_counts.get("initial_design_evaluation", 0) + 1
        total_evaluations = doe.shape[0]
        pool_init_start = perf_counter()
        if mc_pool_override is not None:
            mc_pool = np.asarray(mc_pool_override, dtype=float).copy()
            num_mc = mc_pool.shape[0]
            if v_pdf_pool_override is None:
                if is_lognormal:
                    mc_pool_z_for_pdf = np.log(np.maximum(mc_pool, 1e-10))
                else:
                    mc_pool_z_for_pdf = mc_pool
                v_pdf_pool = _compute_pdf(mc_pool, mc_pool_z_for_pdf)
            else:
                v_pdf_pool = np.asarray(v_pdf_pool_override, dtype=float).reshape(-1).copy()
                if v_pdf_pool.size != num_mc:
                    raise ValueError("v_pdf_pool_override size must match mc_pool_override.")
            pool_init_duration = perf_counter() - pool_init_start
            phase_timings["initial_pool_generation"] = (
                phase_timings.get("initial_pool_generation", 0.0) + pool_init_duration
            )
            phase_counts["initial_pool_generation"] = phase_counts.get("initial_pool_generation", 0) + 1
            sobol_draw_count = int(sobol_draw_count_override or num_mc)
            if sobol_draw_count > 0:
                try:
                    sobol_sampler.fast_forward(int(sobol_draw_count))
                except Exception:
                    pass
        else:
            # Dimension-adaptive MC pool size: 500K for n_dim ≤ 6,
            # scaling down for higher dimensions with a floor of 100K.
            if N_MCS > 0:
                n_mcs_effective = N_MCS
            else:
                n_mcs_effective = max(100_000, min(500_000, int(5e5 / max(1, n_dim / 6))))
            sobol_samples_z = sobol_sampler.draw(n_mcs_effective)
            sobol_samples = _transform_samples(sobol_samples_z)
            pool_init_duration = perf_counter() - pool_init_start
            phase_timings["initial_pool_generation"] = (
                phase_timings.get("initial_pool_generation", 0.0) + pool_init_duration
            )
            phase_counts["initial_pool_generation"] = phase_counts.get("initial_pool_generation", 0) + 1
            mc_pool = sobol_samples
            sobol_draw_count = int(mc_pool.shape[0])
            num_mc = mc_pool.shape[0]
            v_pdf_pool = _compute_pdf(mc_pool, sobol_samples_z)
        pf_history = np.zeros(max_iter + doe.shape[0] + 10)
        n_samples_added = 0
        memory_profile = False
        pf_sequence: list[float] = []
        max_iter_reached = False
        n0 = doe.shape[0]
    else:
        doe = np.asarray(resume_state.get("doe", []), dtype=float)
        g = np.asarray(resume_state.get("g", []), dtype=float).reshape(-1)
        if doe.shape[0] != g.shape[0]:
            raise ValueError("Checkpoint DOE and response arrays have mismatched lengths.")
        total_evaluations = max(int(resume_state.get("total_evaluations", doe.shape[0])), doe.shape[0])
        n_samples_added = int(resume_state.get("n_samples_added", 0))
        sobol_draw_count = max(int(resume_state.get("sobol_draw_count", 0)), 0)
        mc_pool = np.asarray(resume_state.get("mc_pool", []), dtype=float)
        num_mc = mc_pool.shape[0]
        v_pdf_pool = np.asarray(resume_state.get("v_pdf_pool", []), dtype=float).reshape(-1)
        if v_pdf_pool.size != num_mc and num_mc:
            # Recompute PDF if needed (for lognormal, compute z from mc_pool)
            if is_lognormal:
                mc_pool_z_for_pdf = np.log(np.maximum(mc_pool, 1e-10))
            else:
                mc_pool_z_for_pdf = mc_pool
            v_pdf_pool = _compute_pdf(mc_pool, mc_pool_z_for_pdf)
        pf_history = np.asarray(resume_state.get("pf_history", []), dtype=float).reshape(-1)
        if pf_history.size == 0:
            pf_history = np.zeros(max_iter + doe.shape[0] + 10)
        pf_sequence = list(np.asarray(resume_state.get("pf_sequence", []), dtype=float).reshape(-1))

        memory_profile = bool(resume_state.get("memory_profile", False))
        n0 = int(resume_state.get("n0", n0))
        max_iter_reached = bool(resume_state.get("max_iter_reached", False))
        learning_w_grad = float(resume_state.get("learning_w_grad", learning_w_grad))
        svr_bounds_mode = resume_state.get("svr_bounds_mode", svr_bounds_mode)
        if n_samples_added < max_iter:
            max_iter_reached = False
        required_pf_len = max(max_iter + n0 + 10, pf_history.shape[0], total_evaluations + max_iter + 10)
        if pf_history.shape[0] < required_pf_len:
            pf_extended = np.zeros(required_pf_len, dtype=float)
            pf_extended[: pf_history.shape[0]] = pf_history
            pf_history = pf_extended
        sobol_draw_count = max(sobol_draw_count, mc_pool.shape[0])
        if sobol_draw_count > 0:
            try:
                sobol_sampler.fast_forward(int(sobol_draw_count))
            except Exception:
                pass
        if not pf_sequence:
            pf_sequence = []

    ensemble_size = max(1, int(ensemble_size))
    if resume_state is not None:
        ensemble_size = int(resume_state.get("ensemble_size", ensemble_size))

    def _capture_rng_state() -> Optional[dict[str, Any]]:
        try:
            rng = global_random_state()
        except Exception:
            return None
        try:
            bit_generator, state_array, pos, has_gauss, cached_gaussian = rng.get_state()
        except Exception:
            return None
        return {
            "bit_generator": str(bit_generator),
            "state": np.asarray(state_array, dtype=np.uint32).tolist(),
            "pos": int(pos),
            "has_gauss": int(has_gauss),
            "cached_gaussian": float(cached_gaussian),
        }

    def _checkpoint_payload() -> dict[str, Any]:
        trainer_state = None
        if svr_trainer is not None:
            capture_trainer = getattr(svr_trainer, "state_dict", None)
            if capture_trainer is not None:
                trainer_state = capture_trainer()
        return {
            "test_example": test_example,
            "args": list(args),
            "ensemble_size": int(ensemble_size),
            "svr_kernel": canonical_kernel,
            "svr_covariance": covariance_name,
            "random_seed": seed_used,
            "n_workers": None if n_workers is None else int(n_workers),
            "tail_policy": tail_policy,
            "tail_alpha": float(tail_alpha),
            "initial_design": initial_design,
            "svr_trainer_state": trainer_state,

            "learning_w_grad": float(learning_w_grad),
            "svr_bounds_mode": svr_bounds_mode,
            "n_samples_added": int(n_samples_added),
            "total_evaluations": int(total_evaluations),
            "sobol_draw_count": int(sobol_draw_count),

            "memory_profile": bool(memory_profile),
            "max_iter_reached": bool(max_iter_reached),
            "n0": int(n0),
            "rng_state": _capture_rng_state(),
            "max_iter": int(max_iter),
            "doe": doe,
            "g": g,
            "mc_pool": mc_pool,
            "v_pdf_pool": v_pdf_pool,
            "pf_history": pf_history,
            "pf_sequence": np.asarray(pf_sequence, dtype=float),
            "phase_timings": {str(key): float(value) for key, value in phase_timings.items()},
            "phase_counts": {str(key): int(value) for key, value in phase_counts.items()},
        }

    last_checkpoint_samples = -1

    def _maybe_checkpoint(force: bool = False) -> None:
        nonlocal last_checkpoint_samples
        if checkpoint_target is None:
            return
        if not force:
            if checkpoint_frequency > 1 and (n_samples_added % checkpoint_frequency) != 0:
                return
            if n_samples_added == last_checkpoint_samples:
                return
        save_start = perf_counter()
        save_checkpoint(checkpoint_target, _checkpoint_payload())
        save_duration = perf_counter() - save_start
        phase_timings["checkpoint_io"] = phase_timings.get("checkpoint_io", 0.0) + save_duration
        phase_counts["checkpoint_io"] = phase_counts.get("checkpoint_io", 0) + 1
        last_checkpoint_samples = n_samples_added

    _maybe_checkpoint(force=True)

    while num_mc < MAX_MCS_POOL_SIZE and not max_iter_reached:
        while True:
            total_evaluations += 1
            effective_ensemble = max(1, int(ensemble_size))

            doe_svr = doe
            mc_svr = mc_pool
            svr_dim = n_dim

            if effective_ensemble > 1:
                ensemble_model = _time_phase(
                    "model_training",
                    train_svr_ensemble,
                    doe_svr,
                    g,
                    svr_dim,
                    n_models=effective_ensemble,
                    bootstrap=True,
                    kernel=canonical_kernel,
                    theta_init=gamma,
                    c_init=c_init,
                    epsilon_init=epsilon_init,
                    bounds_mode=svr_bounds_mode,
                )
                member_models = ensemble_model.get("models", [])
                if member_models:
                    primary_model = member_models[0]
                else:
                    primary_model = _time_phase(
                        "model_training", train_svr, doe_svr, g, svr_dim,
                        kernel=canonical_kernel, theta_init=gamma,
                        c_init=c_init, epsilon_init=epsilon_init,
                    )
                g_predict, g_mse = _time_phase(
                    "surrogate_prediction",
                    svr_predict_ensemble,
                    mc_svr,
                    ensemble_model,
                )
                surrogate_for_gradients = primary_model
            else:
                primary_model = _time_phase(
                    "model_training",
                    trainer,
                    doe_svr,
                    g,
                    svr_dim,
                    kernel=canonical_kernel,
                    theta_init=gamma,
                    c_init=c_init,
                    epsilon_init=epsilon_init,
                    bounds_mode=svr_bounds_mode,
                )
                g_predict, g_mse = _time_phase("surrogate_prediction", svr_predict, mc_svr, primary_model)
                surrogate_for_gradients = primary_model
            pool_std = np.sqrt(np.maximum(0.0, g_mse))
            current_pf = estimate_probability_of_failure(g_predict, g, num_mc)
            pf_history[total_evaluations - 1] = current_pf
            pf_sequence.append(current_pf)

            if pruning_callback is not None:
                if pruning_callback(current_pf, n_samples_added):
                    if verbose:
                        print("Pruned by callback.")
                    max_iter_reached = True
                    break

            # Score candidates and extract the most promising region.
            lf, mc_pool_region, region_indices = _time_phase(
                "candidate_scoring",
                compute_learning_function,
                mc_pool,
                doe,
                g_predict,
                g_mse,
                v_pdf_pool,
                current_pf,
                model=surrogate_for_gradients,
                w_grad=learning_w_grad,
            )

            if tail_policy != "none" and mc_pool_region.size and doe.shape[0] > 1:
                ood_start = perf_counter()
                candidate_scores = score_standard_distance(mc_pool_region, doe)
                lf = apply_tail_policy(lf, candidate_scores, alpha=tail_alpha, policy=tail_policy)
                ood_duration = perf_counter() - ood_start
                phase_timings["ood_scoring"] = phase_timings.get("ood_scoring", 0.0) + ood_duration
                phase_counts["ood_scoring"] = phase_counts.get("ood_scoring", 0) + 1

            # ── Convergence check ────────────────────────────────────
            stop_start = perf_counter()
            stability, beta_curr = _evaluate_convergence(
                pf_history, total_evaluations, n0,
            )
            _pf_now = pf_history[total_evaluations - 1]
            _denom = _pf_now * num_mc
            cov_pf_display = np.sqrt((1.0 - _pf_now) / _denom) if _denom > 0 else float("inf")
            stop_duration = perf_counter() - stop_start
            phase_timings["stopping_checks"] = phase_timings.get("stopping_checks", 0.0) + stop_duration
            phase_counts["stopping_checks"] = phase_counts.get("stopping_checks", 0) + 1

            required_samples = max(MIN_BURN_IN, int(min_samples))
            converged = (n_samples_added >= required_samples) and stability
            if converged or (n_samples_added >= max_iter):
                if n_samples_added >= max_iter and not converged:
                    if verbose:
                        print("Reached maximum number of added samples.")
                    max_iter_reached = True
                break

            (
                doe,
                g,
                mc_pool,
                v_pdf_pool,
                num_mc,
                n_samples_added,
            ) = _time_phase(
                "candidate_evaluation",
                select_best_candidate,
                lf,
                mc_pool_region,
                region_indices,
                doe,
                g,
                mc_pool,
                v_pdf_pool,
                num_mc,
                n_samples_added,
                fun,
                fun_par,
            )
            if verbose:
                print(
                    "Added Point: {count:4d} | Pf: {pf:.4e} | \u03b2: {beta:6.3f} | CoV(Pf): {cov:.4f} | Total Calls: {calls:4d}".format(
                        count=n_samples_added,
                        pf=pf_history[total_evaluations - 1],
                        beta=beta_curr,
                        cov=cov_pf_display,
                        calls=total_evaluations,
                    )
                )
            _maybe_checkpoint()
            memory_profile = maybe_print_memory_usage(memory_profile)

        _maybe_checkpoint(force=True)

        if max_iter_reached:
            break

        current_pf = pf_history[total_evaluations - 1]
        # CoV formula: sqrt((1-pf) / (pf * Num_MCP))
        # Only use MC pool size, not DOE size
        denom = current_pf * num_mc
        if denom <= 0:
            cov_pf = np.inf
        else:
            cov_pf = np.sqrt((1.0 - current_pf) / denom)
        if cov_pf < cov_pf_tol:
            break

        # Enrich the Monte Carlo pool with additional quasi-random samples to
        # maintain exploration pressure once the coefficient of variation stalls.
        enrichment_start = perf_counter()
        additional_z = sobol_sampler.draw(MCS_ENRICH_SIZE)
        additional = _transform_samples(additional_z)
        mc_pool = np.vstack([mc_pool, additional])
        v_pdf_add = _compute_pdf(additional, additional_z)
        v_pdf_pool = np.concatenate([v_pdf_pool, v_pdf_add])
        num_mc = mc_pool.shape[0]
        sobol_draw_count += MCS_ENRICH_SIZE
        enrichment_duration = perf_counter() - enrichment_start
        phase_timings["pool_enrichment"] = phase_timings.get("pool_enrichment", 0.0) + enrichment_duration
        phase_counts["pool_enrichment"] = phase_counts.get("pool_enrichment", 0) + 1
        if verbose:
            print(f"Candidate pool enriched. New size: {num_mc}")
        memory_profile = maybe_print_memory_usage(memory_profile)
        _maybe_checkpoint(force=True)

    pf_final = pf_history[total_evaluations - 1]
    beta = -stats.norm.ppf(pf_final)

    result = AdaptiveSVRResult(
        probability_of_failure=pf_final,
        reliability_index=beta,
        total_evaluations=doe.shape[0],
        phase_timings=phase_timings,
        phase_counts=phase_counts,
        kernel=canonical_kernel,
        covariance=covariance_name,
    )

    _maybe_checkpoint(force=True)

    if result_target is not None:
        result_target.parent.mkdir(parents=True, exist_ok=True)
        export_start = perf_counter()
        initial_payload = json.dumps(result.to_dict(), indent=2)
        result_target.write_text(initial_payload, encoding="utf-8")
        export_duration = perf_counter() - export_start
        phase_timings["result_export"] = phase_timings.get("result_export", 0.0) + export_duration
        phase_counts["result_export"] = phase_counts.get("result_export", 0) + 1
        final_payload = json.dumps(result.to_dict(), indent=2)
        result_target.write_text(final_payload, encoding="utf-8")

    for phase, value in list(phase_timings.items()):
        phase_timings[phase] = float(value)
    for phase, value in list(phase_counts.items()):
        phase_counts[phase] = int(value)

    return result


# Backwards compatibility with the original module name.
adaptive_svr = run_adaptive_svr
