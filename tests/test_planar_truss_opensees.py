"""Verification tests for the published 23-bar planar-truss benchmark."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import qmc

from absvr_core.problem_setup import prepare_problem_definition
from benchmarks.planar_truss_opensees import (
    ALL_MEMBERS,
    DIAGONAL_MEMBERS,
    HORIZONTAL_MEMBERS,
    INPUT_MEANS,
    INPUT_STANDARD_DEVIATIONS,
    NODE_COORDINATES,
    REFERENCE_FAILURE_PROBABILITY,
    midspan_displacement_numpy,
    midspan_displacement_opensees,
    midspan_displacement_vectorized,
    planar_truss_limit_state,
    standard_normal_to_physical,
)


def test_published_topology_counts() -> None:
    assert len(NODE_COORDINATES) == 13
    assert len(HORIZONTAL_MEMBERS) == 11
    assert len(DIAGONAL_MEMBERS) == 12
    assert len(ALL_MEMBERS) == 23
    assert len(set(ALL_MEMBERS)) == 23


def test_isoprobabilistic_transform_recovers_published_moments() -> None:
    unit = qmc.Sobol(d=10, scramble=True, seed=314159).random_base2(17)
    from scipy.special import ndtri

    z = ndtri(np.clip(unit, np.finfo(float).tiny, 1.0 - np.finfo(float).eps))
    physical = standard_normal_to_physical(z)

    relative_mean_error = np.abs(physical.mean(axis=0) - INPUT_MEANS) / INPUT_MEANS
    relative_std_error = np.abs(physical.std(axis=0) - INPUT_STANDARD_DEVIATIONS) / INPUT_STANDARD_DEVIATIONS
    assert np.max(relative_mean_error) < 2.0e-3
    assert np.max(relative_std_error) < 8.0e-3


def test_virtual_work_response_matches_direct_numpy_fe() -> None:
    rng = np.random.default_rng(20260808)
    physical = standard_normal_to_physical(rng.normal(size=(12, 10)))
    direct = np.asarray(midspan_displacement_numpy(physical))
    vectorized = np.asarray(midspan_displacement_vectorized(physical))
    np.testing.assert_allclose(vectorized, direct, rtol=2.0e-12, atol=1.0e-14)


def test_opensees_response_matches_independent_numpy_fe() -> None:
    pytest.importorskip("openseespy.opensees")
    z = np.array(
        [
            np.zeros(10),
            [0.5, -0.7, 0.4, -0.2, 0.1, 0.3, -0.6, 0.9, -1.1, 0.2],
            [-1.0, 0.6, -0.3, 0.8, 1.2, -0.4, 0.7, -0.8, 0.5, 1.0],
        ]
    )
    physical = standard_normal_to_physical(z)
    direct = np.asarray(midspan_displacement_numpy(physical))
    opensees = np.asarray(midspan_displacement_opensees(physical))
    np.testing.assert_allclose(opensees, direct, rtol=2.0e-12, atol=1.0e-14)


def test_limit_state_solver_paths_are_equivalent() -> None:
    pytest.importorskip("openseespy.opensees")
    adverse_point = np.concatenate([np.full(4, -1.5), np.full(6, 1.5)])
    z = np.vstack([np.zeros(10), adverse_point])
    g_opensees = planar_truss_limit_state(z, {"solver": "opensees"})
    g_numpy = planar_truss_limit_state(z, {"solver": "numpy"})
    g_vectorized = planar_truss_limit_state(z, {"solver": "vectorized"})
    np.testing.assert_allclose(g_opensees, g_numpy, rtol=2.0e-12, atol=1.0e-14)
    np.testing.assert_allclose(g_vectorized, g_numpy, rtol=2.0e-12, atol=1.0e-14)
    assert g_opensees[0] > 0.0
    assert g_opensees[1] < 0.0


def test_log_ratio_response_preserves_failure_event() -> None:
    from scipy.special import ndtri

    unit = qmc.Sobol(d=10, scramble=True, seed=271828).random_base2(8)
    z = ndtri(np.clip(unit, np.finfo(float).tiny, 1.0 - np.finfo(float).eps))
    difference = planar_truss_limit_state(
        z,
        {"solver": "vectorized", "response": "difference"},
    )
    log_ratio = planar_truss_limit_state(
        z,
        {"solver": "vectorized", "response": "log_ratio"},
    )

    np.testing.assert_array_equal(difference <= 0.0, log_ratio <= 0.0)
    np.testing.assert_allclose(
        log_ratio,
        np.log(1.0 + difference / (0.12 - difference)),
        rtol=1.0e-14,
        atol=1.0e-14,
    )


def test_eg7_problem_definition_uses_standard_normal_space() -> None:
    mu, sigma, n_dim, params, function, pf_ref, distribution = prepare_problem_definition("eg7")
    np.testing.assert_array_equal(mu, np.zeros(10))
    np.testing.assert_array_equal(sigma, np.ones(10))
    assert n_dim == 10
    assert params == {"displacement_limit_m": 0.12, "solver": "opensees"}
    assert function is planar_truss_limit_state
    assert pf_ref == REFERENCE_FAILURE_PROBABILITY
    assert distribution == "normal"
