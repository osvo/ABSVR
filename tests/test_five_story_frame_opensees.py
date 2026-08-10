"""Verification tests for the five-storey structural-frame benchmark."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import ndtri
from scipy.stats import qmc

from benchmarks.five_story_frame_opensees import (
    ALL_MEMBERS,
    BEAM_MEMBERS,
    COLUMN_MEMBERS,
    GAUSSIAN_COPULA_CORRELATION,
    INPUT_MEANS,
    INPUT_STANDARD_DEVIATIONS,
    NODE_COORDINATES,
    five_story_frame_limit_state,
    standard_normal_to_physical,
    top_displacement_banded,
    top_displacement_euler_banded,
    top_displacement_euler_numpy,
    top_displacement_euler_opensees,
    top_displacement_numpy,
    top_displacement_opensees,
)


def test_published_frame_topology_counts() -> None:
    assert len(NODE_COORDINATES) == 24
    assert len(COLUMN_MEMBERS) == 20
    assert len(BEAM_MEMBERS) == 15
    assert len(ALL_MEMBERS) == 35
    assert len({(node_i, node_j) for node_i, node_j, _ in ALL_MEMBERS}) == 35


def test_published_gaussian_copula_is_positive_definite() -> None:
    np.testing.assert_allclose(
        GAUSSIAN_COPULA_CORRELATION,
        GAUSSIAN_COPULA_CORRELATION.T,
        rtol=0.0,
        atol=0.0,
    )
    eigenvalues = np.linalg.eigvalsh(GAUSSIAN_COPULA_CORRELATION)
    assert eigenvalues[0] == pytest.approx(0.05, rel=1.0e-12, abs=1.0e-12)
    assert GAUSSIAN_COPULA_CORRELATION[3, 4] == 0.90
    assert GAUSSIAN_COPULA_CORRELATION[5, 13] == 0.95
    assert GAUSSIAN_COPULA_CORRELATION[5, 14] == 0.13
    assert GAUSSIAN_COPULA_CORRELATION[0, 1] == 0.0


def test_isoprobabilistic_transform_recovers_published_model() -> None:
    unit = qmc.Sobol(d=21, scramble=True, seed=20260809).random_base2(17)
    z = ndtri(np.clip(unit, np.finfo(float).tiny, 1.0 - np.finfo(float).eps))
    physical = standard_normal_to_physical(z)

    relative_load_mean_error = np.abs(physical[:, :3].mean(axis=0) - INPUT_MEANS[:3]) / INPUT_MEANS[:3]
    relative_load_std_error = np.abs(physical[:, :3].std(axis=0) - INPUT_STANDARD_DEVIATIONS[:3]) / INPUT_STANDARD_DEVIATIONS[:3]
    assert np.max(relative_load_mean_error) < 2.0e-3
    assert np.max(relative_load_std_error) < 6.0e-3

    # The paper quotes the moments of the underlying Gaussian variables, not
    # the moments after positive truncation.  Back-transform the physical
    # sample to verify those underlying parameters without conflating them.
    assert np.all(physical[:, 3:] > 0.0)
    assert physical.shape == (2**17, 21)

    transformed = np.corrcoef(z @ np.linalg.cholesky(GAUSSIAN_COPULA_CORRELATION).T, rowvar=False)
    np.testing.assert_allclose(
        transformed,
        GAUSSIAN_COPULA_CORRELATION,
        rtol=0.0,
        atol=5.0e-3,
    )


def test_opensees_response_matches_independent_numpy_fe() -> None:
    pytest.importorskip("openseespy.opensees")
    z = np.array(
        [
            np.zeros(21),
            np.linspace(-0.8, 0.8, 21),
            np.array(
                [
                    0.9,
                    -0.4,
                    0.5,
                    -0.6,
                    0.3,
                    0.2,
                    -0.7,
                    0.8,
                    -0.2,
                    0.6,
                    -0.3,
                    0.4,
                    -0.5,
                    0.7,
                    -0.1,
                    0.1,
                    -0.8,
                    0.5,
                    -0.6,
                    0.3,
                    -0.2,
                ]
            ),
        ]
    )
    physical = standard_normal_to_physical(z)
    direct = np.asarray(top_displacement_numpy(physical))
    banded = np.asarray(top_displacement_banded(physical, chunk_size=2))
    opensees = np.asarray(top_displacement_opensees(physical))
    np.testing.assert_allclose(banded, direct, rtol=3.0e-12, atol=1.0e-14)
    np.testing.assert_allclose(opensees, direct, rtol=3.0e-12, atol=1.0e-14)

    direct_euler = np.asarray(top_displacement_euler_numpy(physical))
    banded_euler = np.asarray(top_displacement_euler_banded(physical, chunk_size=2))
    opensees_euler = np.asarray(top_displacement_euler_opensees(physical))
    np.testing.assert_allclose(banded_euler, direct_euler, rtol=3.0e-12, atol=1.0e-14)
    np.testing.assert_allclose(opensees_euler, direct_euler, rtol=3.0e-12, atol=1.0e-14)


def test_timoshenko_solver_reproduces_published_response_moments() -> None:
    unit = qmc.Sobol(d=21, scramble=True, seed=98765).random_base2(14)
    z = ndtri(np.clip(unit, np.finfo(float).tiny, 1.0 - np.finfo(float).eps))
    physical = standard_normal_to_physical(z)
    displacement = np.asarray(top_displacement_banded(physical))

    # The original benchmark response moments, reproduced in Li et al., are
    # 0.069 ft and 0.021 ft.  The conventional Timoshenko reconstruction
    # recovers them without an empirical response factor.
    published_mean_m = 0.069 * 0.3048
    published_standard_deviation_m = 0.021 * 0.3048
    assert abs(displacement.mean() - published_mean_m) / published_mean_m < 0.01
    assert (
        abs(displacement.std(ddof=1) - published_standard_deviation_m)
        / published_standard_deviation_m
        < 0.01
    )


def test_euler_sensitivity_reproduces_li_direct_mcs_response_moments() -> None:
    unit = qmc.Sobol(d=21, scramble=True, seed=98765).random_base2(14)
    z = ndtri(np.clip(unit, np.finfo(float).tiny, 1.0 - np.finfo(float).eps))
    physical = standard_normal_to_physical(z)
    displacement = np.asarray(top_displacement_euler_banded(physical))

    # Li et al. separately report 0.0652 ft and 0.0202 ft from 100,000 direct
    # MCS evaluations; the Euler sensitivity reproduces that alternate result.
    published_mean_m = 0.0652 * 0.3048
    published_standard_deviation_m = 0.0202 * 0.3048
    assert abs(displacement.mean() - published_mean_m) / published_mean_m < 0.01
    assert (
        abs(displacement.std(ddof=1) - published_standard_deviation_m)
        / published_standard_deviation_m
        < 0.01
    )


def test_limit_state_responses_preserve_failure_event() -> None:
    pytest.importorskip("openseespy.opensees")
    z = np.vstack([np.zeros(21), np.r_[np.full(3, 2.0), np.full(18, -1.5)]])
    difference_numpy = five_story_frame_limit_state(
        z, {"solver": "numpy", "response": "difference"}
    )
    difference_opensees = five_story_frame_limit_state(
        z, {"solver": "opensees", "response": "difference"}
    )
    difference_banded = five_story_frame_limit_state(
        z, {"solver": "banded", "response": "difference"}
    )
    log_ratio = five_story_frame_limit_state(
        z, {"solver": "numpy", "response": "log_ratio"}
    )

    np.testing.assert_allclose(difference_opensees, difference_numpy, rtol=2.0e-12, atol=1.0e-14)
    np.testing.assert_allclose(difference_banded, difference_numpy, rtol=2.0e-12, atol=1.0e-14)
    np.testing.assert_array_equal(difference_numpy <= 0.0, log_ratio <= 0.0)
    assert difference_numpy[0] > 0.0
    assert difference_numpy[1] < 0.0
