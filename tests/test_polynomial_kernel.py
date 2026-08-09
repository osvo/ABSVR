"""Mathematical validity checks for the polynomial SVR kernel."""

from __future__ import annotations

import numpy as np

from surrogate.svr.kernel_utils import compute_kernel


def test_polynomial_gram_matrix_remains_positive_semidefinite_at_large_scale() -> None:
    x = np.random.default_rng(20260808).normal(size=(24, 2))
    gram = compute_kernel(x, x, np.full(2, 1.0e5), "Polynomial")

    np.testing.assert_allclose(gram, gram.T, rtol=0.0, atol=1.0e-8)
    diagonal_scale = np.sqrt(np.diag(gram))
    correlation_form = gram / np.outer(diagonal_scale, diagonal_scale)
    eigenvalues = np.linalg.eigvalsh(0.5 * (correlation_form + correlation_form.T))
    assert eigenvalues[0] >= -1.0e-10


def test_polynomial_kernel_matches_its_untruncated_definition() -> None:
    x = np.array([[2.0, -1.0], [-3.0, 4.0], [5.0, 2.0]])
    theta = np.array([1.0e4, 2.0e4])
    scaled = x * np.sqrt(theta)
    expected = (scaled @ scaled.T / x.shape[1] + 1.0) ** 3
    actual = compute_kernel(x, x, theta, "Polynomial")
    np.testing.assert_allclose(actual, expected, rtol=1.0e-14, atol=1.0e-8)

