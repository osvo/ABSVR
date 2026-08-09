"""OpenSeesPy model of the published 23-bar planar-truss benchmark.

The benchmark follows Schobi et al. (2016), Marelli and Sudret (2018),
and Wang et al. (2021).  A simply supported, six-panel Warren truss has
23 bars, 13 nodes, and 10 independent random inputs.  Failure occurs when
the downward midspan displacement reaches 0.12 m.

ABSVR operates in independent standard-normal space.  This module therefore
maps the first four coordinates to lognormal section and material properties
and the remaining six coordinates to Gumbel loads before launching the finite
element analysis.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Mapping

import numpy as np
from scipy import stats


REFERENCE_FAILURE_PROBABILITY = 1.52e-3
DEFAULT_DISPLACEMENT_LIMIT_M = 0.12

INPUT_NAMES = ("A1", "A2", "E1", "E2", "P1", "P2", "P3", "P4", "P5", "P6")
INPUT_MEANS = np.array(
    [2.0e-3, 1.0e-3, 2.1e11, 2.1e11, 5.0e4, 5.0e4, 5.0e4, 5.0e4, 5.0e4, 5.0e4],
    dtype=float,
)
INPUT_STANDARD_DEVIATIONS = np.array(
    [2.0e-4, 1.0e-4, 2.1e10, 2.1e10, 7.5e3, 7.5e3, 7.5e3, 7.5e3, 7.5e3, 7.5e3],
    dtype=float,
)

# Bottom chord nodes 1-7 and upper chord nodes 8-13, in metres.
NODE_COORDINATES = {
    **{tag: (4.0 * (tag - 1), 0.0) for tag in range(1, 8)},
    **{tag: (2.0 + 4.0 * (tag - 8), 2.0) for tag in range(8, 14)},
}
HORIZONTAL_MEMBERS = tuple(
    [(tag, tag + 1) for tag in range(1, 7)]
    + [(tag, tag + 1) for tag in range(8, 13)]
)
DIAGONAL_MEMBERS = tuple(
    member
    for panel in range(6)
    for member in ((1 + panel, 8 + panel), (2 + panel, 8 + panel))
)
ALL_MEMBERS = HORIZONTAL_MEMBERS + DIAGONAL_MEMBERS
LOAD_NODES = tuple(range(8, 14))
MIDSPAN_NODE = 4


def _as_2d(array: np.ndarray) -> tuple[np.ndarray, bool]:
    values = np.asarray(array, dtype=float)
    was_vector = values.ndim == 1
    if was_vector:
        values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape[1] != len(INPUT_NAMES):
        raise ValueError("Expected an array with shape (n_samples, 10).")
    return values, was_vector


def _lognormal_parameters(mean: np.ndarray, standard_deviation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    variance_ratio = (standard_deviation / mean) ** 2
    sigma_log = np.sqrt(np.log1p(variance_ratio))
    mu_log = np.log(mean) - 0.5 * sigma_log**2
    return mu_log, sigma_log


def _gumbel_parameters(mean: float, standard_deviation: float) -> tuple[float, float]:
    scale = standard_deviation * np.sqrt(6.0) / np.pi
    location = mean - np.euler_gamma * scale
    return float(location), float(scale)


def standard_normal_to_physical(z: np.ndarray) -> np.ndarray:
    """Map independent standard-normal samples to the published marginals."""

    z_values, was_vector = _as_2d(z)
    physical = np.empty_like(z_values)

    mu_log, sigma_log = _lognormal_parameters(
        INPUT_MEANS[:4], INPUT_STANDARD_DEVIATIONS[:4]
    )
    physical[:, :4] = np.exp(mu_log + sigma_log * z_values[:, :4])

    load_location, load_scale = _gumbel_parameters(
        INPUT_MEANS[4], INPUT_STANDARD_DEVIATIONS[4]
    )
    probabilities = stats.norm.cdf(z_values[:, 4:])
    tiny = np.finfo(float).eps
    probabilities = np.clip(probabilities, tiny, 1.0 - tiny)
    physical[:, 4:] = load_location - load_scale * np.log(-np.log(probabilities))

    return physical[0] if was_vector else physical


def _assemble_reduced_stiffness(ea_horizontal: float, ea_diagonal: float) -> tuple[np.ndarray, list[int]]:
    n_dof = 2 * len(NODE_COORDINATES)
    stiffness = np.zeros((n_dof, n_dof), dtype=float)

    for node_i, node_j in ALL_MEMBERS:
        xi, yi = NODE_COORDINATES[node_i]
        xj, yj = NODE_COORDINATES[node_j]
        dx = xj - xi
        dy = yj - yi
        length = float(np.hypot(dx, dy))
        cosine = dx / length
        sine = dy / length
        direction = np.array([cosine, sine, -cosine, -sine], dtype=float)
        ea = ea_horizontal if (node_i, node_j) in HORIZONTAL_MEMBERS else ea_diagonal
        element_stiffness = (ea / length) * np.outer(direction, direction)
        dofs = [2 * (node_i - 1), 2 * (node_i - 1) + 1, 2 * (node_j - 1), 2 * (node_j - 1) + 1]
        stiffness[np.ix_(dofs, dofs)] += element_stiffness

    restrained = {0, 1, 2 * (7 - 1) + 1}
    free = [dof for dof in range(n_dof) if dof not in restrained]
    return stiffness[np.ix_(free, free)], free


def midspan_displacement_numpy(physical_inputs: np.ndarray) -> np.ndarray | float:
    """Independent NumPy FE oracle used to verify the OpenSeesPy model."""

    values, was_vector = _as_2d(physical_inputs)
    displacements = np.empty(values.shape[0], dtype=float)

    for row_index, row in enumerate(values):
        area_horizontal, area_diagonal, young_horizontal, young_diagonal = row[:4]
        stiffness, free = _assemble_reduced_stiffness(
            young_horizontal * area_horizontal,
            young_diagonal * area_diagonal,
        )
        loads = np.zeros(2 * len(NODE_COORDINATES), dtype=float)
        for node, load in zip(LOAD_NODES, row[4:]):
            loads[2 * (node - 1) + 1] = -load
        response = np.linalg.solve(stiffness, loads[free])
        midspan_vertical_dof = 2 * (MIDSPAN_NODE - 1) + 1
        displacements[row_index] = -response[free.index(midspan_vertical_dof)]

    return float(displacements[0]) if was_vector else displacements


@lru_cache(maxsize=1)
def _virtual_work_coefficients() -> tuple[np.ndarray, np.ndarray]:
    """Return load influence coefficients for the two member groups.

    The truss is statically determinate.  Its midspan displacement is therefore
    a sum of one term proportional to ``1 / (E1 A1)`` and another proportional
    to ``1 / (E2 A2)``.  The coefficients are recovered from two independent
    stiffness solutions for each unit load.
    """

    horizontal = np.empty(6, dtype=float)
    diagonal = np.empty(6, dtype=float)
    template = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    for load_index in range(6):
        unit_load = template.copy()
        unit_load[4 + load_index] = 1.0
        equal_ea = float(midspan_displacement_numpy(unit_load))

        doubled_horizontal = unit_load.copy()
        doubled_horizontal[0] = 2.0
        horizontal_doubled = float(midspan_displacement_numpy(doubled_horizontal))

        horizontal[load_index] = 2.0 * (equal_ea - horizontal_doubled)
        diagonal[load_index] = equal_ea - horizontal[load_index]

    return horizontal, diagonal


def midspan_displacement_vectorized(physical_inputs: np.ndarray) -> np.ndarray | float:
    """Fast virtual-work response for large reference simulations."""

    values, was_vector = _as_2d(physical_inputs)
    horizontal, diagonal = _virtual_work_coefficients()
    horizontal_flexibility = 1.0 / (values[:, 0] * values[:, 2])
    diagonal_flexibility = 1.0 / (values[:, 1] * values[:, 3])
    load_effect_horizontal = values[:, 4:] @ horizontal
    load_effect_diagonal = values[:, 4:] @ diagonal
    displacement = (
        load_effect_horizontal * horizontal_flexibility
        + load_effect_diagonal * diagonal_flexibility
    )
    return float(displacement[0]) if was_vector else displacement


def midspan_displacement_opensees(physical_inputs: np.ndarray) -> np.ndarray | float:
    """Evaluate midspan displacement with OpenSeesPy linear truss elements."""

    try:
        import openseespy.opensees as ops
    except (ImportError, RuntimeError) as exc:  # pragma: no cover - optional native package
        raise ImportError(
            "The planar-truss benchmark requires OpenSeesPy. Install "
            "requirements-structural.txt before running eg7."
        ) from exc

    values, was_vector = _as_2d(physical_inputs)
    displacements = np.empty(values.shape[0], dtype=float)

    for row_index, row in enumerate(values):
        area_horizontal, area_diagonal, young_horizontal, young_diagonal = row[:4]

        ops.wipe()
        ops.model("basic", "-ndm", 2, "-ndf", 2)
        for tag, (x_coordinate, y_coordinate) in NODE_COORDINATES.items():
            ops.node(tag, x_coordinate, y_coordinate)
        ops.fix(1, 1, 1)
        ops.fix(7, 0, 1)

        ops.uniaxialMaterial("Elastic", 1, young_horizontal)
        ops.uniaxialMaterial("Elastic", 2, young_diagonal)
        element_tag = 1
        for node_i, node_j in HORIZONTAL_MEMBERS:
            ops.element("truss", element_tag, node_i, node_j, area_horizontal, 1)
            element_tag += 1
        for node_i, node_j in DIAGONAL_MEMBERS:
            ops.element("truss", element_tag, node_i, node_j, area_diagonal, 2)
            element_tag += 1

        ops.timeSeries("Linear", 1)
        ops.pattern("Plain", 1, 1)
        for node, load in zip(LOAD_NODES, row[4:]):
            ops.load(node, 0.0, -float(load))

        ops.constraints("Plain")
        ops.numberer("RCM")
        ops.system("BandSPD")
        ops.test("NormDispIncr", 1.0e-12, 10)
        ops.algorithm("Linear")
        ops.integrator("LoadControl", 1.0)
        ops.analysis("Static")
        analysis_code = ops.analyze(1)
        if analysis_code != 0:
            ops.wipe()
            raise RuntimeError(f"OpenSees analysis failed with code {analysis_code}.")

        displacements[row_index] = -float(ops.nodeDisp(MIDSPAN_NODE, 2))
        ops.wipe()

    return float(displacements[0]) if was_vector else displacements


def planar_truss_limit_state(z: np.ndarray, params: Mapping[str, object] | None = None) -> np.ndarray:
    """Return an equivalent midspan-displacement limit-state response.

    ``response='difference'`` (the default) returns the conventional response
    ``limit - abs(displacement)`` in metres. ``response='log_ratio'`` returns
    ``log(limit / abs(displacement))``.  Both responses have exactly the same
    zero contour and failure classification; the logarithmic form only changes
    the regression target presented to a surrogate.
    """

    settings = dict(params or {})
    displacement_limit = float(
        settings.get("displacement_limit_m", DEFAULT_DISPLACEMENT_LIMIT_M)
    )
    solver = str(settings.get("solver", "opensees")).strip().lower()
    response = str(settings.get("response", "difference")).strip().lower()
    physical = standard_normal_to_physical(z)
    physical_2d, _ = _as_2d(physical)

    if solver == "opensees":
        displacement = np.asarray(midspan_displacement_opensees(physical_2d), dtype=float)
    elif solver == "numpy":
        displacement = np.asarray(midspan_displacement_numpy(physical_2d), dtype=float)
    elif solver in {"vectorized", "virtual_work"}:
        displacement = np.asarray(midspan_displacement_vectorized(physical_2d), dtype=float)
    else:
        raise ValueError("solver must be 'opensees', 'numpy', or 'vectorized'.")

    absolute_displacement = np.abs(displacement).reshape(-1)
    if response == "difference":
        return displacement_limit - absolute_displacement
    if response == "log_ratio":
        return np.log(displacement_limit / absolute_displacement)
    raise ValueError("response must be 'difference' or 'log_ratio'.")


def get_problem_definition() -> tuple:
    """Return the ABSVR problem tuple for the OpenSeesPy benchmark."""

    n_dim = len(INPUT_NAMES)
    return (
        np.zeros(n_dim),
        np.ones(n_dim),
        n_dim,
        {"displacement_limit_m": DEFAULT_DISPLACEMENT_LIMIT_M, "solver": "opensees"},
        planar_truss_limit_state,
        REFERENCE_FAILURE_PROBABILITY,
        "normal",
    )


__all__ = [
    "ALL_MEMBERS",
    "DEFAULT_DISPLACEMENT_LIMIT_M",
    "DIAGONAL_MEMBERS",
    "HORIZONTAL_MEMBERS",
    "INPUT_MEANS",
    "INPUT_NAMES",
    "INPUT_STANDARD_DEVIATIONS",
    "NODE_COORDINATES",
    "REFERENCE_FAILURE_PROBABILITY",
    "get_problem_definition",
    "midspan_displacement_numpy",
    "midspan_displacement_opensees",
    "midspan_displacement_vectorized",
    "planar_truss_limit_state",
    "standard_normal_to_physical",
]
