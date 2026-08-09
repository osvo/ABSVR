"""Generate publication figures for the planar-truss reliability benchmark."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon

from benchmarks.planar_truss_opensees import (
    DIAGONAL_MEMBERS,
    HORIZONTAL_MEMBERS,
    LOAD_NODES,
    MIDSPAN_NODE,
    NODE_COORDINATES,
)


BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
GREY = "#777777"


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save(fig: plt.Figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_truss_schematic(output_base: Path) -> None:
    """Plot the finite-element topology and the reliability limit state."""

    _configure_style()
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    for members, color, width in (
        (HORIZONTAL_MEMBERS, BLUE, 2.2),
        (DIAGONAL_MEMBERS, ORANGE, 1.8),
    ):
        for node_i, node_j in members:
            x = [NODE_COORDINATES[node_i][0], NODE_COORDINATES[node_j][0]]
            y = [NODE_COORDINATES[node_i][1], NODE_COORDINATES[node_j][1]]
            ax.plot(x, y, color=color, linewidth=width, solid_capstyle="round")

    coordinates = np.array(list(NODE_COORDINATES.values()))
    ax.scatter(coordinates[:, 0], coordinates[:, 1], s=18, color="black", zorder=3)
    for node, (x_coordinate, y_coordinate) in NODE_COORDINATES.items():
        ax.text(x_coordinate, y_coordinate + 0.16, str(node), ha="center", va="bottom", fontsize=7)

    for load_index, node in enumerate(LOAD_NODES, start=1):
        x_coordinate, y_coordinate = NODE_COORDINATES[node]
        ax.annotate(
            "",
            xy=(x_coordinate, y_coordinate + 0.03),
            xytext=(x_coordinate, y_coordinate + 0.78),
            arrowprops={"arrowstyle": "-|>", "color": "black", "lw": 1.0},
        )
        ax.text(x_coordinate + 0.12, y_coordinate + 0.67, rf"$P_{load_index}$", fontsize=8)

    left_x, left_y = NODE_COORDINATES[1]
    right_x, right_y = NODE_COORDINATES[7]
    ax.add_patch(
        Polygon(
            [(left_x, left_y - 0.04), (left_x - 0.35, left_y - 0.42), (left_x + 0.35, left_y - 0.42)],
            closed=True,
            facecolor="none",
            edgecolor="black",
            linewidth=1.0,
        )
    )
    ax.add_patch(
        Polygon(
            [(right_x, right_y - 0.04), (right_x - 0.35, right_y - 0.35), (right_x + 0.35, right_y - 0.35)],
            closed=True,
            facecolor="none",
            edgecolor="black",
            linewidth=1.0,
        )
    )
    ax.plot([right_x - 0.43, right_x + 0.43], [right_y - 0.46, right_y - 0.46], color="black", lw=0.8)
    ax.scatter([right_x - 0.22, right_x, right_x + 0.22], [right_y - 0.40] * 3, s=12, facecolors="white", edgecolors="black")

    mid_x, mid_y = NODE_COORDINATES[MIDSPAN_NODE]
    ax.annotate(
        "",
        xy=(mid_x, mid_y - 0.58),
        xytext=(mid_x, mid_y - 0.08),
        arrowprops={"arrowstyle": "-|>", "color": GREEN, "lw": 1.5},
    )
    ax.text(mid_x + 0.22, mid_y - 0.38, r"$|u_y|$", color=GREEN, va="center")
    ax.text(12.0, -0.9, r"Failure: $g(\mathbf{x})=0.12-|u_y|\leq0$", ha="center", color=GREEN)

    ax.annotate("", xy=(0, -0.72), xytext=(4, -0.72), arrowprops={"arrowstyle": "|-|", "lw": 0.8})
    ax.text(2, -0.69, "4 m", ha="center", va="bottom", fontsize=8)
    ax.annotate("", xy=(24.55, 0), xytext=(24.55, 2), arrowprops={"arrowstyle": "|-|", "lw": 0.8})
    ax.text(24.68, 1, "2 m", rotation=90, va="center", fontsize=8)

    ax.legend(
        handles=[
            Line2D([0], [0], color=BLUE, lw=2.2, label=r"Chord members: $A_1,E_1$"),
            Line2D([0], [0], color=ORANGE, lw=1.8, label=r"Diagonal members: $A_2,E_2$"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.36),
        ncol=2,
        frameon=False,
    )
    ax.set_aspect("equal")
    ax.set_xlim(-0.8, 25.2)
    ax.set_ylim(-1.05, 3.0)
    ax.axis("off")
    _save(fig, output_base)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def plot_method_comparison(
    comparison_path: Path,
    absvr_campaign_path: Path,
    combined_audit_path: Path,
    uqlab_summary_path: Path,
    reference_path: Path,
    output_base: Path,
) -> None:
    """Plot efficiency/accuracy and repeated-estimate stability."""

    _configure_style()
    with comparison_path.open(newline="", encoding="utf-8") as stream:
        comparison = list(csv.DictReader(stream))
    absvr_campaign = _load_json(absvr_campaign_path)
    combined_audit = _load_json(combined_audit_path)
    uqlab_summary = _load_json(uqlab_summary_path)
    reference = _load_json(reference_path)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15), gridspec_kw={"wspace": 0.34})
    ax = axes[0]
    offsets = {
        "MCS": (-10, 7),
        "FORM": (-11, -13),
        "SORM": (3, 6),
        "AK-MCS": (3, -13),
        "ABSVR1": (-34, 12),
        "ABSVR2": (5, 7),
        "A-bPCE": (7, -14),
    }
    for row in comparison:
        method = row["method"]
        evaluations = float(row["mean_model_evaluations"])
        error = float(row["relative_error_vs_independent_rqmc_percent"])
        if method.startswith("ABSVR (this study"):
            color, marker, size, zorder = BLUE, "o", 52, 4
            label = "ABSVR (confirmed mean)"
            offset = (7, 12)
        elif method.startswith("UQLab native"):
            color, marker, size, zorder = ORANGE, "s", 52, 4
            label = "UQLab AK-MCS (confirmed mean)"
            offset = (7, 10)
        else:
            color, marker, size, zorder = GREY, "o", 25, 2
            label = method
            offset = offsets.get(method, (4, 6))
        ax.scatter(evaluations, error, color=color, marker=marker, s=size, zorder=zorder)
        ax.annotate(label, (evaluations, error), xytext=offset, textcoords="offset points", fontsize=7, color=color)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Mean true limit-state evaluations")
    ax.set_ylabel("Relative error vs independent RQMC (%)")
    ax.set_xlim(35, 2.0e6)
    ax.set_ylim(0.18, 80)
    ax.grid(True, which="both", color="#dddddd", linewidth=0.55)
    ax.text(-0.15, 1.04, "(a)", transform=ax.transAxes, fontweight="bold")

    ax = axes[1]
    absvr_estimates = np.array(
        [run["validation"]["surrogate_pf_mean"] for run in absvr_campaign["runs"]]
    )
    uqlab_estimates = np.array([run["pf"] for run in uqlab_summary["individual_runs"]])
    rng = np.random.default_rng(20260809)
    for position, estimates, color, label in (
        (0, absvr_estimates, BLUE, "ABSVR\n95 calls"),
        (1, uqlab_estimates, ORANGE, "UQLab AK-MCS\n383.4 calls"),
    ):
        jitter = rng.uniform(-0.07, 0.07, size=estimates.size)
        ax.scatter(position + jitter, 1e3 * estimates, color=color, s=24, alpha=0.82, zorder=3)
        audit_key = "absvr" if position == 0 else "uqlab_native_akmcs"
        audit = combined_audit[audit_key]
        mean = 1e3 * float(audit["mean_pf"])
        lower, upper = [
            1e3 * float(value)
            for value in audit["t_interval_95_for_mean_across_algorithm_seeds"]
        ]
        ax.errorbar(
            position,
            mean,
            yerr=[[mean - lower], [upper - mean]],
            fmt="D",
            color="black",
            markerfacecolor="white",
            capsize=4,
            markersize=5,
            linewidth=1.1,
            zorder=4,
        )

    ref_lower, ref_upper = [1e3 * float(value) for value in reference["confidence_interval_95"]]
    ref_mean = 1e3 * float(reference["mean_pf"])
    ax.axhspan(ref_lower, ref_upper, color=GREEN, alpha=0.16, label="RQMC 95% interval")
    ax.axhline(ref_mean, color=GREEN, linestyle="--", linewidth=1.1, label="RQMC mean")
    ax.set_xlim(-0.45, 1.45)
    ax.set_xticks([0, 1], ["ABSVR\n95 calls", "UQLab AK-MCS\n383.4 calls"])
    ax.set_ylabel(r"Estimated $P_f$ ($\times 10^{-3}$)")
    ax.grid(True, axis="y", color="#dddddd", linewidth=0.55)
    ax.legend(loc="upper left", frameon=False, fontsize=7)
    ax.text(-0.15, 1.04, "(b)", transform=ax.transAxes, fontweight="bold")
    _save(fig, output_base)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, default=Path("results/planar_truss/figures"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    plot_truss_schematic(args.output_directory / "planar_truss_schematic")
    plot_method_comparison(
        Path("results/planar_truss/comparison_table_v3.csv"),
        Path("results/planar_truss/confirmation_campaign_v2.json"),
        Path("results/planar_truss/comparison_audit_v3.json"),
        Path("results/planar_truss/uqlab_confirmation/campaign_summary.json"),
        Path("results/planar_truss/reference_qmc.json"),
        args.output_directory / "method_comparison",
    )


if __name__ == "__main__":
    main()
