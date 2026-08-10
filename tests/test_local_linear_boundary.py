import numpy as np
import pytest
from scipy.stats import norm

from studies.planar_truss.evaluate_local_linear_boundary import (
    fit_local_linear_boundary,
)


def test_affine_standard_normal_failure_probability_is_analytic():
    rng = np.random.default_rng(17)
    inputs = rng.normal(size=(80, 2))
    response = 3.0 + 2.0 * inputs[:, 0]

    result = fit_local_linear_boundary(
        inputs,
        response,
        weight_power=0.0,
        ridge_alpha=1.0e-10,
    )

    assert result["failure_probability"] == pytest.approx(norm.cdf(-1.5), rel=1.0e-8)
    assert result["reliability_index"] == pytest.approx(1.5, rel=1.0e-8)


def test_affine_fit_requires_more_samples_than_dimensions():
    with pytest.raises(ValueError, match="more samples"):
        fit_local_linear_boundary(
            np.eye(2),
            np.array([1.0, -1.0]),
        )
