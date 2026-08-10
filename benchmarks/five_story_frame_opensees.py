"""Five-storey, three-bay structural-frame reliability benchmark.

The benchmark follows the frame in Blatman and Sudret (2010) and Marelli and
Sudret (2018).  It has 24 nodes, 35 two-dimensional Euler-Bernoulli frame
elements, and 21 correlated random inputs.  Failure occurs when the horizontal
displacement of the top-right node reaches 0.05 m.

ABSVR works in independent standard-normal space.  The transformation below
first applies the published Gaussian copula and then the published marginal
distributions.  All finite-element calculations use kN and m consistently.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Mapping

import numpy as np
from scipy.linalg import solveh_banded
from scipy.special import ndtr, ndtri


PUBLISHED_REFERENCE_FAILURE_PROBABILITY = 1.54e-3
DEFAULT_DISPLACEMENT_LIMIT_M = 0.05

INPUT_NAMES = (
    "P1",
    "P2",
    "P3",
    "E4",
    "E5",
    "I6",
    "I7",
    "I8",
    "I9",
    "I10",
    "I11",
    "I12",
    "I13",
    "A14",
    "A15",
    "A16",
    "A17",
    "A18",
    "A19",
    "A20",
    "A21",
)
INPUT_MEANS = np.array(
    [
        133.454,
        88.970,
        71.175,
        2.1738e7,
        2.3796e7,
        8.1344e-3,
        1.1509e-2,
        2.1375e-2,
        2.5961e-2,
        1.0812e-2,
        1.4105e-2,
        2.3279e-2,
        2.5961e-2,
        3.1256e-1,
        3.7210e-1,
        5.0606e-1,
        5.5815e-1,
        2.5302e-1,
        2.9117e-1,
        3.7303e-1,
        4.1860e-1,
    ],
    dtype=float,
)
INPUT_STANDARD_DEVIATIONS = np.array(
    [
        40.040,
        35.590,
        28.470,
        1.9152e6,
        1.9152e6,
        1.0834e-3,
        1.2980e-3,
        2.5961e-3,
        3.0288e-3,
        2.5961e-3,
        3.4615e-3,
        5.6249e-3,
        6.4902e-3,
        5.5815e-2,
        7.4420e-2,
        9.3025e-2,
        1.1163e-1,
        9.3025e-2,
        1.0232e-1,
        1.2093e-1,
        1.9537e-1,
    ],
    dtype=float,
)

# Geometry in metres: 25 ft, 30 ft, 25 ft; first storey 16 ft and four
# additional storeys of 12 ft.  Nodes are numbered left-to-right by level.
X_COORDINATES = np.array([0.0, 7.62, 16.764, 24.384], dtype=float)
Y_COORDINATES = np.array(
    [0.0, 4.8768, 8.5344, 12.1920, 15.8496, 19.5072], dtype=float
)
NODE_COORDINATES = {
    4 * level + column + 1: (float(x), float(y))
    for level, y in enumerate(Y_COORDINATES)
    for column, x in enumerate(X_COORDINATES)
}
BASE_NODES = (1, 2, 3, 4)
TOP_RIGHT_NODE = 24

# Member groups read bottom-to-top from Figure 5 of Marelli and Sudret (2018).
_COLUMN_GROUPS_BY_STOREY = (
    ("C3", "C4", "C4", "C3"),
    ("C2", "C3", "C3", "C2"),
    ("C2", "C3", "C3", "C2"),
    ("C1", "C2", "C2", "C1"),
    ("C1", "C2", "C2", "C1"),
)
_BEAM_GROUPS_BY_FLOOR = (
    ("B3", "B4", "B3"),
    ("B2", "B3", "B2"),
    ("B2", "B3", "B2"),
    ("B1", "B2", "B1"),
    ("B1", "B2", "B1"),
)

COLUMN_MEMBERS = tuple(
    (4 * storey + column + 1, 4 * (storey + 1) + column + 1, group)
    for storey, groups in enumerate(_COLUMN_GROUPS_BY_STOREY)
    for column, group in enumerate(groups)
)
BEAM_MEMBERS = tuple(
    (4 * (floor + 1) + bay + 1, 4 * (floor + 1) + bay + 2, group)
    for floor, groups in enumerate(_BEAM_GROUPS_BY_FLOOR)
    for bay, group in enumerate(groups)
)
ALL_MEMBERS = COLUMN_MEMBERS + BEAM_MEMBERS

# Physical-input indices for E, I, and A in each element group.
ELEMENT_PROPERTY_INDICES = {
    "B1": (3, 9, 17),
    "B2": (3, 10, 18),
    "B3": (3, 11, 19),
    "B4": (3, 12, 20),
    "C1": (4, 5, 13),
    "C2": (4, 6, 14),
    "C3": (4, 7, 15),
    "C4": (4, 8, 16),
}

# Horizontal nodal loads from the first to the fifth floor.
LOAD_NODES = (5, 9, 13, 17, 21)
LOAD_INPUT_INDICES = (2, 1, 0, 0, 0)


def _build_gaussian_copula_correlation() -> np.ndarray:
    correlation = np.eye(len(INPUT_NAMES), dtype=float)
    correlation[3, 4] = correlation[4, 3] = 0.90

    # I6,...,I13 correspond one-to-one with A14,...,A21.
    for element_i in range(8):
        inertia_i = 5 + element_i
        area_i = 13 + element_i
        correlation[inertia_i, area_i] = correlation[area_i, inertia_i] = 0.95
        for element_j in range(element_i + 1, 8):
            inertia_j = 5 + element_j
            area_j = 13 + element_j
            for row, column in (
                (inertia_i, inertia_j),
                (area_i, area_j),
                (inertia_i, area_j),
                (area_i, inertia_j),
            ):
                correlation[row, column] = correlation[column, row] = 0.13
    return correlation


GAUSSIAN_COPULA_CORRELATION = _build_gaussian_copula_correlation()
GAUSSIAN_COPULA_CHOLESKY = np.linalg.cholesky(GAUSSIAN_COPULA_CORRELATION)


def _as_2d(array: np.ndarray) -> tuple[np.ndarray, bool]:
    values = np.asarray(array, dtype=float)
    was_vector = values.ndim == 1
    if was_vector:
        values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape[1] != len(INPUT_NAMES):
        raise ValueError("Expected an array with shape (n_samples, 21).")
    return values, was_vector


def _lognormal_parameters(
    mean: np.ndarray, standard_deviation: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    variance_ratio = (standard_deviation / mean) ** 2
    sigma_log = np.sqrt(np.log1p(variance_ratio))
    mu_log = np.log(mean) - 0.5 * sigma_log**2
    return mu_log, sigma_log


def standard_normal_to_physical(z: np.ndarray) -> np.ndarray:
    """Map independent standard normals through the published Gaussian copula."""

    independent, was_vector = _as_2d(z)
    dependent = independent @ GAUSSIAN_COPULA_CHOLESKY.T
    physical = np.empty_like(dependent)

    load_mu_log, load_sigma_log = _lognormal_parameters(
        INPUT_MEANS[:3], INPUT_STANDARD_DEVIATIONS[:3]
    )
    physical[:, :3] = np.exp(load_mu_log + load_sigma_log * dependent[:, :3])

    # The quoted Gaussian moments are those of the underlying untruncated
    # distributions.  Invert their CDF conditional on each variable being > 0.
    lower_standardized = -INPUT_MEANS[3:] / INPUT_STANDARD_DEVIATIONS[3:]
    lower_probability = ndtr(lower_standardized)
    uniform = ndtr(dependent[:, 3:])
    conditional_probability = lower_probability + uniform * (1.0 - lower_probability)
    tiny = np.finfo(float).eps
    conditional_probability = np.clip(conditional_probability, tiny, 1.0 - tiny)
    physical[:, 3:] = INPUT_MEANS[3:] + INPUT_STANDARD_DEVIATIONS[3:] * ndtri(
        conditional_probability
    )

    return physical[0] if was_vector else physical


def _frame_element_global_stiffness(
    node_i: int,
    node_j: int,
    area: float,
    young_modulus: float,
    moment_of_inertia: float,
) -> np.ndarray:
    xi, yi = NODE_COORDINATES[node_i]
    xj, yj = NODE_COORDINATES[node_j]
    dx = xj - xi
    dy = yj - yi
    length = float(np.hypot(dx, dy))
    cosine = dx / length
    sine = dy / length

    axial = young_modulus * area / length
    bending_12 = 12.0 * young_modulus * moment_of_inertia / length**3
    bending_6 = 6.0 * young_modulus * moment_of_inertia / length**2
    bending_4 = 4.0 * young_modulus * moment_of_inertia / length
    bending_2 = 2.0 * young_modulus * moment_of_inertia / length
    local = np.array(
        [
            [axial, 0.0, 0.0, -axial, 0.0, 0.0],
            [0.0, bending_12, bending_6, 0.0, -bending_12, bending_6],
            [0.0, bending_6, bending_4, 0.0, -bending_6, bending_2],
            [-axial, 0.0, 0.0, axial, 0.0, 0.0],
            [0.0, -bending_12, -bending_6, 0.0, bending_12, -bending_6],
            [0.0, bending_6, bending_2, 0.0, -bending_6, bending_4],
        ],
        dtype=float,
    )
    transformation = np.array(
        [
            [cosine, sine, 0.0, 0.0, 0.0, 0.0],
            [-sine, cosine, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, cosine, sine, 0.0],
            [0.0, 0.0, 0.0, -sine, cosine, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    return transformation.T @ local @ transformation


def _assemble_reduced_system(physical_inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_dof = 3 * len(NODE_COORDINATES)
    stiffness = np.zeros((n_dof, n_dof), dtype=float)

    for node_i, node_j, group in ALL_MEMBERS:
        young_index, inertia_index, area_index = ELEMENT_PROPERTY_INDICES[group]
        element = _frame_element_global_stiffness(
            node_i,
            node_j,
            float(physical_inputs[area_index]),
            float(physical_inputs[young_index]),
            float(physical_inputs[inertia_index]),
        )
        dofs = [
            3 * (node_i - 1),
            3 * (node_i - 1) + 1,
            3 * (node_i - 1) + 2,
            3 * (node_j - 1),
            3 * (node_j - 1) + 1,
            3 * (node_j - 1) + 2,
        ]
        stiffness[np.ix_(dofs, dofs)] += element

    loads = np.zeros(n_dof, dtype=float)
    for node, input_index in zip(LOAD_NODES, LOAD_INPUT_INDICES):
        loads[3 * (node - 1)] = float(physical_inputs[input_index])

    restrained = {3 * (node - 1) + dof for node in BASE_NODES for dof in range(3)}
    free = np.array([dof for dof in range(n_dof) if dof not in restrained], dtype=int)
    return stiffness[np.ix_(free, free)], loads[free]


def top_displacement_numpy(physical_inputs: np.ndarray) -> np.ndarray | float:
    """Independent dense-matrix finite-element implementation."""

    values, was_vector = _as_2d(physical_inputs)
    displacements = np.empty(values.shape[0], dtype=float)
    top_right_horizontal_dof_in_reduced_system = 3 * (TOP_RIGHT_NODE - 1) - 12

    for row_index, row in enumerate(values):
        stiffness, loads = _assemble_reduced_system(row)
        response = np.linalg.solve(stiffness, loads)
        displacements[row_index] = response[top_right_horizontal_dof_in_reduced_system]

    return float(displacements[0]) if was_vector else displacements


@lru_cache(maxsize=1)
def _reduced_stiffness_bands() -> tuple[np.ndarray, int]:
    """Return 16 unit-property stiffness bases in lower-band storage."""

    n_dof = 3 * len(NODE_COORDINATES)
    restrained = {3 * (node - 1) + dof for node in BASE_NODES for dof in range(3)}
    free = np.array([dof for dof in range(n_dof) if dof not in restrained], dtype=int)
    dense_bases: list[np.ndarray] = []

    for group in ELEMENT_PROPERTY_INDICES:
        for property_type in ("EA", "EI"):
            stiffness = np.zeros((n_dof, n_dof), dtype=float)
            for node_i, node_j, member_group in ALL_MEMBERS:
                if member_group != group:
                    continue
                if property_type == "EA":
                    element = _frame_element_global_stiffness(
                        node_i, node_j, area=1.0, young_modulus=1.0, moment_of_inertia=0.0
                    )
                else:
                    element = _frame_element_global_stiffness(
                        node_i, node_j, area=0.0, young_modulus=1.0, moment_of_inertia=1.0
                    )
                dofs = [
                    3 * (node_i - 1),
                    3 * (node_i - 1) + 1,
                    3 * (node_i - 1) + 2,
                    3 * (node_j - 1),
                    3 * (node_j - 1) + 1,
                    3 * (node_j - 1) + 2,
                ]
                stiffness[np.ix_(dofs, dofs)] += element
            dense_bases.append(stiffness[np.ix_(free, free)])

    dense = np.asarray(dense_bases)
    nonzero_rows, nonzero_columns = np.where(np.max(np.abs(dense), axis=0) > 0.0)
    half_bandwidth = int(np.max(np.abs(nonzero_rows - nonzero_columns)))
    banded = np.zeros((dense.shape[0], half_bandwidth + 1, dense.shape[1]), dtype=float)
    for offset in range(half_bandwidth + 1):
        for column in range(dense.shape[1] - offset):
            banded[:, offset, column] = dense[:, column + offset, column]
    return banded, half_bandwidth


def top_displacement_banded(
    physical_inputs: np.ndarray, *, chunk_size: int = 16_384
) -> np.ndarray | float:
    """Fast exact frame response for independent reference simulations.

    The global stiffness is a linear combination of eight ``EA`` and eight
    ``EI`` bases.  The regular frame has lower half-bandwidth 14 after removing
    the fixed-base degrees of freedom, so LAPACK's symmetric positive-definite
    band solver avoids both surrogate approximation and dense 60-by-60 solves.
    """

    values, was_vector = _as_2d(physical_inputs)
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive.")
    stiffness_bases, _ = _reduced_stiffness_bands()
    displacements = np.empty(values.shape[0], dtype=float)
    top_right_horizontal_dof_in_reduced_system = 3 * (TOP_RIGHT_NODE - 1) - 12

    for chunk_start in range(0, values.shape[0], chunk_size):
        chunk_end = min(chunk_start + chunk_size, values.shape[0])
        chunk = values[chunk_start:chunk_end]
        coefficients: list[np.ndarray] = []
        for young_index, inertia_index, area_index in ELEMENT_PROPERTY_INDICES.values():
            coefficients.extend(
                [
                    chunk[:, young_index] * chunk[:, area_index],
                    chunk[:, young_index] * chunk[:, inertia_index],
                ]
            )
        coefficient_matrix = np.asarray(coefficients).T
        banded_stiffness = np.einsum(
            "bi,ikn->bkn", coefficient_matrix, stiffness_bases, optimize=True
        )

        loads = np.zeros((chunk.shape[0], 60), dtype=float)
        for node, input_index in zip(LOAD_NODES, LOAD_INPUT_INDICES):
            loads[:, 3 * (node - 5)] = chunk[:, input_index]

        for local_index in range(chunk.shape[0]):
            response = solveh_banded(
                banded_stiffness[local_index],
                loads[local_index],
                lower=True,
                overwrite_ab=True,
                overwrite_b=True,
                check_finite=False,
            )
            displacements[chunk_start + local_index] = response[
                top_right_horizontal_dof_in_reduced_system
            ]

    return float(displacements[0]) if was_vector else displacements


def top_displacement_opensees(physical_inputs: np.ndarray) -> np.ndarray | float:
    """Evaluate the frame with OpenSeesPy elastic beam-column elements."""

    try:
        import openseespy.opensees as ops
    except (ImportError, RuntimeError) as exc:  # pragma: no cover - optional native package
        raise ImportError(
            "The five-storey frame requires OpenSeesPy. Install "
            "requirements-structural.txt before running this benchmark."
        ) from exc

    values, was_vector = _as_2d(physical_inputs)
    displacements = np.empty(values.shape[0], dtype=float)

    for row_index, row in enumerate(values):
        ops.wipe()
        ops.model("basic", "-ndm", 2, "-ndf", 3)
        for tag, (x_coordinate, y_coordinate) in NODE_COORDINATES.items():
            ops.node(tag, x_coordinate, y_coordinate)
        for node in BASE_NODES:
            ops.fix(node, 1, 1, 1)

        ops.geomTransf("Linear", 1)
        for element_tag, (node_i, node_j, group) in enumerate(ALL_MEMBERS, start=1):
            young_index, inertia_index, area_index = ELEMENT_PROPERTY_INDICES[group]
            ops.element(
                "elasticBeamColumn",
                element_tag,
                node_i,
                node_j,
                float(row[area_index]),
                float(row[young_index]),
                float(row[inertia_index]),
                1,
            )

        ops.timeSeries("Linear", 1)
        ops.pattern("Plain", 1, 1)
        for node, input_index in zip(LOAD_NODES, LOAD_INPUT_INDICES):
            ops.load(node, float(row[input_index]), 0.0, 0.0)

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

        displacements[row_index] = float(ops.nodeDisp(TOP_RIGHT_NODE, 1))
        ops.wipe()

    return float(displacements[0]) if was_vector else displacements


def five_story_frame_limit_state(
    z: np.ndarray, params: Mapping[str, object] | None = None
) -> np.ndarray:
    """Return the displacement-margin or equivalent log-ratio response."""

    settings = dict(params or {})
    displacement_limit = float(
        settings.get("displacement_limit_m", DEFAULT_DISPLACEMENT_LIMIT_M)
    )
    solver = str(settings.get("solver", "opensees")).strip().lower()
    response = str(settings.get("response", "difference")).strip().lower()
    physical = standard_normal_to_physical(z)
    physical_2d, _ = _as_2d(physical)

    if solver == "opensees":
        displacement = np.asarray(top_displacement_opensees(physical_2d), dtype=float)
    elif solver == "numpy":
        displacement = np.asarray(top_displacement_numpy(physical_2d), dtype=float)
    elif solver in {"banded", "reference"}:
        displacement = np.asarray(top_displacement_banded(physical_2d), dtype=float)
    else:
        raise ValueError("solver must be 'opensees', 'numpy', or 'banded'.")

    absolute_displacement = np.abs(displacement).reshape(-1)
    if response == "difference":
        return displacement_limit - absolute_displacement
    if response == "log_ratio":
        return np.log(displacement_limit / absolute_displacement)
    raise ValueError("response must be 'difference' or 'log_ratio'.")


def get_problem_definition() -> tuple:
    """Return the ABSVR problem tuple for the OpenSeesPy frame."""

    n_dim = len(INPUT_NAMES)
    return (
        np.zeros(n_dim),
        np.ones(n_dim),
        n_dim,
        {"displacement_limit_m": DEFAULT_DISPLACEMENT_LIMIT_M, "solver": "opensees"},
        five_story_frame_limit_state,
        PUBLISHED_REFERENCE_FAILURE_PROBABILITY,
        "normal",
    )


__all__ = [
    "ALL_MEMBERS",
    "BASE_NODES",
    "BEAM_MEMBERS",
    "COLUMN_MEMBERS",
    "DEFAULT_DISPLACEMENT_LIMIT_M",
    "ELEMENT_PROPERTY_INDICES",
    "GAUSSIAN_COPULA_CHOLESKY",
    "GAUSSIAN_COPULA_CORRELATION",
    "INPUT_MEANS",
    "INPUT_NAMES",
    "INPUT_STANDARD_DEVIATIONS",
    "LOAD_INPUT_INDICES",
    "LOAD_NODES",
    "NODE_COORDINATES",
    "PUBLISHED_REFERENCE_FAILURE_PROBABILITY",
    "TOP_RIGHT_NODE",
    "five_story_frame_limit_state",
    "get_problem_definition",
    "standard_normal_to_physical",
    "top_displacement_banded",
    "top_displacement_numpy",
    "top_displacement_opensees",
]
