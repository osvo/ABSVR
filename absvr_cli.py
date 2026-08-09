"""Command-line interface for the adaptive Bayesian SVR workflow."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

try:
    from absvr_core import AdaptiveSVRResult, baseline_profile_kwargs, run_adaptive_svr
except ImportError:
    from .absvr_core import AdaptiveSVRResult, baseline_profile_kwargs, run_adaptive_svr
try:
    from surrogate.svr.kernel_utils import normalize_kernel_name, supported_kernel_names
except ImportError:
    from .surrogate.svr.kernel_utils import normalize_kernel_name, supported_kernel_names  # type: ignore
try:
    from utilities.array_backend import get_backend_name
except Exception:

    def get_backend_name() -> str:  # type: ignore
        return "numpy"

try:
    from utilities import get_default_seed, reset_global_seed, ensure_results_dir, ABSVR_RESULTS_DIR
except Exception:

    def get_default_seed() -> int:  # type: ignore
        return 60

    def reset_global_seed(seed: int = 60) -> None:  # type: ignore
        return None

    def ensure_results_dir() -> Path:  # type: ignore
        base = Path("results") / "surrogate"
        base.mkdir(parents=True, exist_ok=True)
        return base

    ABSVR_RESULTS_DIR = ensure_results_dir()


@dataclass
class _BenchmarkOption:
    identifier: str
    title: str
    extra_prompt: str | None = None


@dataclass
class _RunSettings:
    max_iter: int
    ensemble_size: int
    n_workers: Optional[int]
    random_seed: Optional[int]
    replications: int
    tail_policy: str
    tail_alpha: float

    svr_kernel: str
    profile: str


@dataclass
class _PersistenceSettings:
    checkpoint_path: Optional[str]
    checkpoint_frequency: int
    resume_path: Optional[str]
    result_output_path: Optional[str]


_BENCHMARKS = {
    1: _BenchmarkOption("eg1", "High Nonlinearity Problem (2D)"),
    2: _BenchmarkOption("eg2", "Four-Branch Series System (2D)"),
    3: _BenchmarkOption(
        "eg3",
        "Nonlinear Oscillator (6D)",
    ),
    4: _BenchmarkOption("eg4", "Modified Rastrigin (2D)"),
    5: _BenchmarkOption("eg5", "Decision Function (2D)"),
    6: _BenchmarkOption("eg6", "High-Dimensional Problem (40D)"),
    7: _BenchmarkOption("eg7", "23-Bar Planar Truss, OpenSeesPy (10D)"),
    8: _BenchmarkOption("custom", "Custom Benchmark (Python module or file)"),
}


def _resolve_benchmark_flag(raw: str) -> _BenchmarkOption:
    """Map CLI flag input to one of the known benchmark definitions."""

    key = (raw or "").strip().lower()
    if not key:
        raise SystemExit("A benchmark identifier is required when using non-interactive flags.")

    for index, option in _BENCHMARKS.items():
        if key == option.identifier.lower() or key == str(index):
            return option

    raise SystemExit(
        f"Unknown benchmark {raw!r}. Use one of: "
        + ", ".join(option.identifier for option in _BENCHMARKS.values())
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    """Define the CLI arguments shared by interactive and automated modes."""

    kernels = supported_kernel_names()
    parser = argparse.ArgumentParser(
        description=(
            "Adaptive Bayesian SVR CLI. Run interactively by default or pass --benchmark "
            "to skip prompts and control every parameter via flags."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--profile",
        choices=("baseline", "python"),
        default="baseline",
        help="Preset bundle for workflow knobs (baseline matches reference defaults).",
    )
    parser.add_argument(
        "--benchmark",
        help="Identifier (eg1-eg7, 1-7, or custom) to execute without the interactive menu.",
    )
    parser.add_argument(
        "--custom-benchmark",
        type=str,
        help="Custom benchmark module path or .py file (used with --benchmark custom).",
    )
    parser.add_argument(
        "--custom-factory",
        type=str,
        help="Factory function name inside the custom module (default: get_problem_definition).",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=500,
        help="Maximum number of enrichment iterations (also the interactive default).",
    )
    parser.add_argument(
        "--replications",
        type=int,
        default=1,
        help="Run the same configuration multiple times while sharing the RNG stream.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        help="Seed for the first run; omit to reuse the CLI default (60).",
    )
    parser.add_argument(
        "--tail-policy",
        choices=("none", "down", "explore"),
        default="none",
        help="OOD tail policy (requires --benchmark to take effect).",
    )
    parser.add_argument(
        "--tail-alpha",
        type=float,
        default=1.0,
        help="Tail policy intensity; ignored when --tail-policy is 'none'.",
    )

    parser.add_argument(
        "--svr-kernel",
        choices=kernels,
        default="gaussian",
        help="SVR kernel to use in non-interactive mode.",
    )
    parser.add_argument(
        "--ensemble-size",
        type=int,
        default=1,
        help="Bootstrap ensemble size for SVR training (>=1).",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        help="Worker processes for parallel limit-state evaluations.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        help="Persist checkpoints to this path (single-run mode).",
    )
    parser.add_argument(
        "--checkpoint-frequency",
        type=int,
        default=1,
        help="How often to save checkpoints when --checkpoint-path is set.",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        help="Resume a single run from this checkpoint file.",
    )
    parser.add_argument(
        "--result-output",
        type=str,
        help="Write the final (or aggregated) JSON report to this path.",
    )
    return parser


def _automation_flags_requested(args: argparse.Namespace) -> bool:
    """Detect if the user supplied automation flags without selecting a benchmark."""

    return any(
        (
            args.profile != "baseline",
            args.custom_benchmark is not None,
            args.custom_factory is not None,
            args.replications != 1,
            args.random_seed is not None,
            args.tail_policy != "none",
            args.tail_alpha != 1.0,

            args.svr_kernel != "gaussian",
            args.ensemble_size != 1,
            args.n_workers is not None,
            args.checkpoint_path is not None,
            args.checkpoint_frequency != 1,
            args.resume_from is not None,
            args.result_output is not None,
        )
    )


def _resolve_profile_kwargs(profile: str) -> dict[str, object]:
    """Return workflow kwargs for the selected profile."""

    if profile != "baseline":
        return {}
    profile_kwargs = dict(baseline_profile_kwargs())
    for key in ("tail_policy", "tail_alpha", "random_seed"):
        profile_kwargs.pop(key, None)
    return profile_kwargs

def _prompt_choice() -> int:
    """Prompt the user for a benchmark identifier."""

    print("----------------------------------------------------")
    print("   ABSVR Structural Reliability Analysis Menu")
    print("----------------------------------------------------")
    print("Please select a benchmark to run:")
    for index, option in _BENCHMARKS.items():
        print(f"  [{index}] Benchmark {index}: {option.title}")
    print("----------------------------------------------------")
    try:
        return int(input("Enter your choice (1-7): "))
    except ValueError as exc:
        raise SystemExit(
            "Invalid choice. Please enter an integer between 1 and 7."
        ) from exc


def _prompt_extra_args(option: _BenchmarkOption) -> tuple[int, ...]:
    """Collect any benchmark-specific selections (e.g., oscillator cases)."""

    return ()


def _prompt_custom_benchmark() -> tuple[str, str | None]:
    """Prompt for a custom benchmark module/file and optional factory."""

    _print_heading("Custom benchmark")
    raw_spec = input("  Module path or .py file: ").strip()
    if not raw_spec:
        raise SystemExit("Custom benchmark path/module is required.")
    factory = input("  Factory function [get_problem_definition]: ").strip()
    if not factory:
        factory = None
    return raw_spec, factory


def _prompt_int(prompt: str, default: int, *, min_value: int = 1) -> int:
    raw = input(f"{prompt} [{default}]: ").strip()
    if raw == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise SystemExit(f"Invalid integer for '{prompt}'.") from exc
    if value < min_value:
        raise SystemExit(f"Value for '{prompt}' must be at least {min_value}.")
    return value


def _prompt_float(prompt: str, default: float, *, min_value: float = 0.0, max_value: Optional[float] = None) -> float:
    raw = input(f"{prompt} [{default}]: ").strip()
    if raw == "":
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise SystemExit(f"Invalid number for '{prompt}'.") from exc
    if value < min_value:
        raise SystemExit(f"Value for '{prompt}' must be >= {min_value}.")
    if max_value is not None and value > max_value:
        raise SystemExit(f"Value for '{prompt}' must be <= {max_value}.")
    return value


def _prompt_yes_no(prompt: str, *, default: bool = False) -> bool:
    affirmative = {"y", "yes"}
    negative = {"n", "no"}
    while True:
        suffix = " [Y/n]" if default else " [y/N]"
        raw = input(f"{prompt}{suffix}: ").strip().lower()
        if raw == "":
            return default
        if raw in affirmative:
            return True
        if raw in negative:
            return False
        print("  Unrecognized response. Use 'y' or 'n'.")


def _print_heading(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


def _prompt_menu(title: str, options: list[tuple[str, str]], default_value: str) -> str:
    normalized_options = [(value.lower(), label) for value, label in options]
    values = [value for value, _ in normalized_options]
    if default_value.lower() not in values:
        raise ValueError("Default value must exist in options.")

    default_label = next(label for value, label in normalized_options if value == default_value.lower())

    print()
    print(f"{title}:")
    for idx, (value, label) in enumerate(normalized_options, start=1):
        marker = "  (default)" if value == default_value.lower() else ""
        print(f"  [{idx}] {label}{marker}")

    prompt_text = f"  Selection (ENTER for {default_label}): "

    raw = input(prompt_text).strip().lower()
    if raw == "":
        return default_value.lower()
    if raw.isdigit():
        idx = int(raw)
        if 1 <= idx <= len(normalized_options):
            return normalized_options[idx - 1][0]
        raise SystemExit("Numeric option out of range.")

    for value, label in normalized_options:
        if raw == value or raw == label.lower():
            return value

    raise SystemExit("Unrecognized option. Please re-run the program.")


def _prompt_run_mode() -> bool:
    choice = _prompt_menu(
        "Run mode",
        [
            ("single", "Single run"),
            ("multi", "Multiple runs (30 repetitions with consecutive seeds)"),
        ],
        "single",
    )
    return choice == "multi"


def _prompt_seed(default_seed: int, *, replications: int) -> Optional[int]:
    if replications > 1:
        raw = input(f"  Base random seed (ENTER for default {default_seed}): ").strip()
    else:
        raw = input(f"  Random seed (ENTER for default {default_seed}): ").strip()
    if raw == "":
        if replications > 1:
            print(
                f"    Using base seed {default_seed}; generating {replications} consecutive seeds."
            )
        else:
            print(f"    Using default random seed: {default_seed}")
        return None
    try:
        seed = int(raw)
    except ValueError as exc:
        raise SystemExit("Invalid integer for 'Random seed'.") from exc
    if seed < 0:
        raise SystemExit("Random seed must be a non-negative integer.")
    if replications > 1:
        print(
            f"    Using custom base seed: {seed}. Seeds will increment by 1 for each run."
        )
    else:
        print(f"    Using custom random seed: {seed}")
    return seed


def _create_run_directory(benchmark: str, *, tag: str | None = None) -> Path:
    ensure_results_dir()
    base = ABSVR_RESULTS_DIR
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_benchmark = benchmark.replace(" ", "_")
    parts = [timestamp, safe_benchmark]
    if tag:
        parts.append(tag)
    name = "_".join(part for part in parts if part)
    run_dir = base / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _prompt_tail_options() -> tuple[str, float, str]:
    _print_heading("Tail policy configuration")
    policy = _prompt_menu(
        "Select policy",
        [
            ("none", "No tail adjustment"),
            ("down", "Attenuate tail (down)"),
            ("explore", "Explore tail (explore)"),
        ],
        "none",
    )

    alpha = 1.0
    if policy != "none":
        alpha = _prompt_float("  Alpha intensity", 1.0, min_value=0.0)

    return policy, alpha


def _prompt_kernel(default: str = "gaussian") -> str:
    canonical_default, _ = normalize_kernel_name(default)
    options = [
        ("gaussian", "Gaussian / RBF"),
        ("laplacian", "Laplacian (exponential)"),
        ("polynomial", "Polynomial"),
    ]
    choice = _prompt_menu("SVR kernel", options, canonical_default)
    canonical, _ = normalize_kernel_name(choice)
    return canonical


def _prompt_run_settings(
    default_max_iter: int,
    default_seed: int,
    *,
    multi_run: bool,
    default_replications: int,
) -> _RunSettings:
    _print_heading("Run configuration")
    print("  Sampling always uses sequential strategy (1 point per iteration).")
    max_iter = _prompt_int("  Maximum samples to add", default_max_iter, min_value=1)

    replications = 1
    if multi_run:
        replications = _prompt_int(
            "  Number of repetitions",
            default_replications,
            min_value=1,
        )

    random_seed = _prompt_seed(default_seed, replications=replications)
    tail_policy, tail_alpha = _prompt_tail_options()
    if tail_policy != "none":
        print(
            f"  Tail policy: {tail_policy} (alpha={tail_alpha:.3f})"
        )
    svr_kernel = _prompt_kernel()
    print(f"  SVR kernel: {svr_kernel}")

    return _RunSettings(
        max_iter=max_iter,
        ensemble_size=1,
        n_workers=None,
        random_seed=random_seed,
        replications=replications,
        tail_policy=tail_policy,
        tail_alpha=tail_alpha,

        svr_kernel=svr_kernel,
        profile="baseline",
    )

def _prompt_persistence_settings(*, multi_run: bool, benchmark: str) -> _PersistenceSettings:
    if multi_run:
        _print_heading("Persistence (read-only)")
        print("  Multi-run mode does not support checkpoints or resume.")
        save_results = _prompt_yes_no(
            "  Save aggregated results to the results folder?", default=False
        )
        result_output_path = None
        if save_results:
            run_dir = _create_run_directory(benchmark, tag="multi")
            result_file = run_dir / "summary.json"
            result_output_path = str(result_file)
            print(f"    Aggregated results will be written to: {result_output_path}")
        return _PersistenceSettings(
            checkpoint_path=None,
            checkpoint_frequency=1,
            resume_path=None,
            result_output_path=result_output_path,
        )

    _print_heading("Persistence")

    resume_path: Optional[str] = None
    if _prompt_yes_no("  Resume from an existing checkpoint?", default=False):
        resume_raw = input("    Checkpoint path (.npz): ").strip()
        if resume_raw:
            resume_path = resume_raw

    save_artifacts = _prompt_yes_no(
        "  Automatically save checkpoints and final result to 'results/surrogate/'?",
        default=False,
    )

    checkpoint_path: Optional[str] = None
    checkpoint_frequency = 1
    result_output_path: Optional[str] = None

    if save_artifacts:
        run_dir = _create_run_directory(benchmark)
        checkpoint_path = str(run_dir / "checkpoint.npz")
        result_output_path = str(run_dir / "result.json")

        freq_raw = input("    Save checkpoint every N samples [1]: ").strip()
        if freq_raw:
            try:
                checkpoint_frequency = int(freq_raw)
            except ValueError as exc:
                raise SystemExit("Invalid integer for checkpoint frequency.") from exc
            if checkpoint_frequency <= 0:
                raise SystemExit("Checkpoint frequency must be a positive integer.")

        print(f"    Results folder: {run_dir}")
        print(f"    Checkpoint: {checkpoint_path}")
        print(f"    Final result: {result_output_path}")

    return _PersistenceSettings(
        checkpoint_path=checkpoint_path,
        checkpoint_frequency=checkpoint_frequency,
        resume_path=resume_path,
        result_output_path=result_output_path,
    )


def _execute_workflow(
    option: _BenchmarkOption,
    extra_args: tuple[int, ...],
    settings: _RunSettings,
    persistence: _PersistenceSettings,
    default_seed: int,
    *,
    custom_benchmark: str | None = None,
    custom_factory: str | None = None,
) -> AdaptiveSVRResult:
    profile_kwargs = _resolve_profile_kwargs(settings.profile)
    if settings.replications > 1:
        if persistence.resume_path:
            raise SystemExit("Checkpoint resume is only available for single runs.")
        if persistence.checkpoint_path:
            print("Checkpoint saving is disabled in multi-run mode; ignoring the provided path.")
            persistence.checkpoint_path = None
        persistence.checkpoint_frequency = 1

    print(f"\nInitializing analysis for: {option.title}...")
    print(f"Array backend: {get_backend_name()}")
    if persistence.resume_path:
        print(f"Resuming from checkpoint: {persistence.resume_path}")
    if persistence.checkpoint_path:
        print(
            "Checkpoints will be written to {path} (every {freq} sample{plural}).".format(
                path=persistence.checkpoint_path,
                freq=persistence.checkpoint_frequency,
                plural="" if persistence.checkpoint_frequency == 1 else "s",
            )
        )
    if persistence.result_output_path:
        if settings.replications <= 1:
            print(f"Final results will be exported to: {persistence.result_output_path}")
        else:
            print(f"Aggregated results will be exported to: {persistence.result_output_path}")

    def _print_stat(label: str, value: str) -> None:
        print(f"{label:<32}{value}")

    def _print_phase_line(phase: str, total: float, events_display: str, avg: float) -> None:
        print(f"  {phase:<28}total={total:9.3f}s  events={events_display:>8}  avg={avg:9.4f}s")

    if settings.replications <= 1:
        seed_in_use = settings.random_seed if settings.random_seed is not None else default_seed
        _print_stat("Random seed in use:", str(seed_in_use))
        _print_stat("Sampling strategy:", "Sequential (1 point per iteration)")
        kernel_name, kernel_covariance = normalize_kernel_name(settings.svr_kernel)
        _print_stat("SVR kernel:", f"{kernel_name} ({kernel_covariance})")
        result = run_adaptive_svr(
            option.identifier,
            settings.max_iter,
            *extra_args,
            n_workers=settings.n_workers,
            ensemble_size=settings.ensemble_size,
            svr_kernel=settings.svr_kernel,
            random_seed=settings.random_seed,
            checkpoint_path=persistence.checkpoint_path,
            checkpoint_frequency=persistence.checkpoint_frequency,
            resume_from=persistence.resume_path,
            result_output_path=persistence.result_output_path,
            tail_policy=settings.tail_policy,
            tail_alpha=settings.tail_alpha,

            custom_benchmark=custom_benchmark,
            custom_factory=custom_factory,
            **profile_kwargs,
        )

        print("\n====================================================")
        print("               ANALYSIS COMPLETE")
        print("====================================================")
        _print_stat("Benchmark executed:", option.title)
        print("----------------------------------------------------")
        _print_stat("Probability of Failure (Pf):", f"{result.probability_of_failure:.4e}")
        _print_stat("Reliability Index (Beta):", f"{result.reliability_index:.4f}")
        _print_stat("Total function calls:", f"{result.total_evaluations}")
        print("====================================================")
        if settings.tail_policy != "none":
            _print_stat(
                "Tail policy active:",
                f"{settings.tail_policy} (alpha={settings.tail_alpha:.3f})",
            )
        _print_stat("SVR kernel used:", f"{result.kernel} ({result.covariance})")
        phase_breakdown = [
            (
                phase,
                float(result.phase_timings.get(phase, 0.0)),
                int(result.phase_counts.get(phase, 0)),
            )
            for phase in sorted(result.phase_timings)
            if result.phase_timings.get(phase, 0.0) > 0.0
        ]
        if phase_breakdown:
            print("Phase timings (seconds):")
            for phase, duration, count in phase_breakdown:
                if count > 0:
                    avg = duration / count
                    _print_phase_line(phase, duration, str(count), avg)
                else:
                    _print_phase_line(phase, duration, "0", 0.0)
        if persistence.result_output_path:
            print(f"Results exported to: {persistence.result_output_path}")
        return result

    base_seed = settings.random_seed if settings.random_seed is not None else default_seed
    print("Random number generator: NumPy RandomState (MT19937)")
    _print_stat("Base random seed:", f"{base_seed} (shared across repetitions)")
    if settings.replications > 1:
        _print_stat("Seed handling:", "Continuing the shared RNG stream between runs.")
    _print_stat("Sampling strategy:", "Sequential (1 point per iteration)")
    kernel_name, kernel_covariance = normalize_kernel_name(settings.svr_kernel)
    _print_stat("SVR kernel:", f"{kernel_name} ({kernel_covariance})")
    _print_stat("Total repetitions:", str(settings.replications))

    reset_global_seed(base_seed)

    pf_values: list[float] = []
    beta_values: list[float] = []
    eval_values: list[int] = []
    seeds_used: list[Optional[int]] = []
    last_result: Optional[AdaptiveSVRResult] = None
    phase_timings_list: list[dict[str, float]] = []
    phase_counts_list: list[dict[str, int]] = []
    kernel_records: list[str] = []
    covariance_records: list[str] = []

    for idx in range(settings.replications):
        if idx == 0:
            run_seed: Optional[int] = base_seed
            run_label = f"seed={base_seed}"
        else:
            run_seed = None
            run_label = "continuing shared RNG stream"
        seeds_used.append(run_seed)
        print(f"\n--- Run {idx + 1}/{settings.replications} | {run_label} ---")
        current_result = run_adaptive_svr(
            option.identifier,
            settings.max_iter,
            *extra_args,
            n_workers=settings.n_workers,
            ensemble_size=settings.ensemble_size,
            svr_kernel=settings.svr_kernel,
            random_seed=run_seed,
            tail_policy=settings.tail_policy,
            tail_alpha=settings.tail_alpha,

            custom_benchmark=custom_benchmark,
            custom_factory=custom_factory,
            **profile_kwargs,
        )
        pf_values.append(float(current_result.probability_of_failure))
        beta_values.append(float(current_result.reliability_index))
        eval_values.append(int(current_result.total_evaluations))
        phase_timings_list.append({phase: float(value) for phase, value in current_result.phase_timings.items()})
        phase_counts_list.append({phase: int(value) for phase, value in current_result.phase_counts.items()})
        kernel_records.append(str(current_result.kernel))
        covariance_records.append(str(current_result.covariance))
        print(
            "    Pf={pf:.4e}, Beta={beta:.4f}, Calls={calls}".format(
                pf=current_result.probability_of_failure,
                beta=current_result.reliability_index,
                calls=current_result.total_evaluations,
            )
        )
        last_result = current_result

    pf_arr = np.asarray(pf_values, dtype=float)
    beta_arr = np.asarray(beta_values, dtype=float)
    eval_arr = np.asarray(eval_values, dtype=float)

    pf_mean = float(pf_arr.mean())
    pf_std = float(pf_arr.std(ddof=0))
    pf_min = float(pf_arr.min())
    pf_max = float(pf_arr.max())

    beta_mean = float(beta_arr.mean())
    beta_std = float(beta_arr.std(ddof=0))
    beta_min = float(beta_arr.min())
    beta_max = float(beta_arr.max())

    eval_mean = float(eval_arr.mean())
    eval_std = float(eval_arr.std(ddof=0))
    eval_min = int(eval_arr.min())
    eval_max = int(eval_arr.max())

    phase_summary: dict[str, dict[str, float]] = {}
    phase_order: list[str] = []
    if phase_timings_list:
        phase_order = sorted(
            {phase for timings in phase_timings_list for phase in timings}
            | {phase for counts in phase_counts_list for phase in counts}
        )
        for phase in phase_order:
            totals = np.asarray([timings.get(phase, 0.0) for timings in phase_timings_list], dtype=float)
            counts_array = np.asarray([counts.get(phase, 0) for counts in phase_counts_list], dtype=float)
            total_mean = float(totals.mean())
            total_std = float(totals.std(ddof=0))
            total_min = float(totals.min())
            total_max = float(totals.max())
            events_mean = float(counts_array.mean())
            events_std = float(counts_array.std(ddof=0))
            events_min = int(counts_array.min())
            events_max = int(counts_array.max())
            count_sum = float(counts_array.sum())
            mean_per_event = float(totals.sum() / count_sum) if count_sum > 0 else 0.0
            phase_summary[phase] = {
                "total_mean": total_mean,
                "total_std": total_std,
                "total_min": total_min,
                "total_max": total_max,
                "events_mean": events_mean,
                "events_std": events_std,
                "events_min": events_min,
                "events_max": events_max,
                "mean_per_event": mean_per_event,
            }

    print("\n====================================================")
    print("          MULTI-RUN ANALYSIS SUMMARY")
    print("====================================================")
    _print_stat("Benchmark executed:", option.title)
    _print_stat("Total repetitions:", str(settings.replications))
    if settings.tail_policy != "none":
        _print_stat(
            "Tail policy:",
            f"{settings.tail_policy} (alpha={settings.tail_alpha:.3f})",
        )
    _print_stat("Random number generator:", "NumPy RandomState (MT19937)")
    _print_stat("Initial seed:", str(base_seed))
    _print_stat("SVR kernel:", f"{kernel_name} ({kernel_covariance})")
    if settings.replications > 1:
        _print_stat(
            "Seed handling:",
            "Shared RNG stream; subsequent runs reuse the evolving state.",
        )
    print("----------------------------------------------------")
    _print_stat("Pf mean:", f"{pf_mean:.4e}")
    _print_stat("Pf std deviation:", f"{pf_std:.4e}")
    _print_stat("Pf min / max:", f"{pf_min:.4e} / {pf_max:.4e}")
    print("----------------------------------------------------")
    _print_stat("Beta mean:", f"{beta_mean:.4f}")
    _print_stat("Beta std deviation:", f"{beta_std:.4e}")
    _print_stat("Beta min / max:", f"{beta_min:.4f} / {beta_max:.4f}")
    print("----------------------------------------------------")
    _print_stat("Total calls mean:", f"{eval_mean:.1f}")
    _print_stat("Total calls std dev:", f"{eval_std:.1f}")
    _print_stat("Total calls min / max:", f"{eval_min} / {eval_max}")
    print("====================================================")

    if phase_order:
        print("Phase timing averages (seconds):")
        for phase in phase_order:
            metrics = phase_summary[phase]
            events_display = f"{metrics['events_mean']:.2f}"
            _print_phase_line(phase, metrics["total_mean"], events_display, metrics["mean_per_event"])

    if persistence.result_output_path:
        run_indices = np.arange(1, pf_arr.size + 1, dtype=int)
        runs_payload = []
        for idx, run_idx in enumerate(run_indices):
            seed_value = seeds_used[idx]
            runs_payload.append(
                {
                    "run": int(run_idx),
                    "seed": None if seed_value is None else int(seed_value),
                    "kernel": kernel_records[idx],
                    "covariance": covariance_records[idx],
                    "probability_of_failure": float(pf_arr[idx]),
                    "reliability_index": float(beta_arr[idx]),
                    "total_evaluations": int(eval_arr[idx]),
                    "phase_timings": {
                        phase: float(value) for phase, value in phase_timings_list[idx].items()
                    },
                    "phase_counts": {
                        phase: int(value) for phase, value in phase_counts_list[idx].items()
                    },
                }
            )

        summary_payload = {
            "benchmark": option.identifier,
            "title": option.title,
            "svr_kernel": kernel_name,
            "svr_covariance": kernel_covariance,
            "runs": runs_payload,
            "probability_of_failure_summary": {
                "mean": float(pf_mean),
                "std": float(pf_std),
                "min": float(pf_min),
                "max": float(pf_max),
            },
            "reliability_index_summary": {
                "mean": float(beta_mean),
                "std": float(beta_std),
                "min": float(beta_min),
                "max": float(beta_max),
            },
            "total_evaluations_summary": {
                "mean": float(eval_mean),
                "std": float(eval_std),
                "min": int(eval_min),
                "max": int(eval_max),
            },
            "phase_summary": {
                phase: {
                    "total_mean": float(metrics["total_mean"]),
                    "total_std": float(metrics["total_std"]),
                    "total_min": float(metrics["total_min"]),
                    "total_max": float(metrics["total_max"]),
                    "events_mean": float(metrics["events_mean"]),
                    "events_std": float(metrics["events_std"]),
                    "events_min": int(metrics["events_min"]),
                    "events_max": int(metrics["events_max"]),
                    "mean_per_event": float(metrics["mean_per_event"]),
                }
                for phase, metrics in phase_summary.items()
            },
        }
        export_path = Path(persistence.result_output_path).expanduser()
        export_path.parent.mkdir(parents=True, exist_ok=True)
        export_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
        if settings.replications <= 1:
            print(f"Results exported to: {export_path}")
        else:
            print(f"Aggregated results exported to: {export_path}")

    if last_result is None:
        raise RuntimeError("No runs were executed in multi-run mode.")
    return last_result


def run_cli(max_iterations: int = 500) -> AdaptiveSVRResult:
    """Launch the interactive ABSVR benchmark selection menu."""

    choice = _prompt_choice()
    option = _BENCHMARKS.get(choice)
    if option is None:
        raise SystemExit(
            "Invalid choice. Please run the script again and select a number between 1 and 7."
        )

    custom_benchmark = None
    custom_factory = None
    if option.identifier == "custom":
        custom_benchmark, custom_factory = _prompt_custom_benchmark()

    extra_args = _prompt_extra_args(option)
    default_seed = get_default_seed()
    multi_run = _prompt_run_mode()
    settings = _prompt_run_settings(
        max_iterations,
        default_seed,
        multi_run=multi_run,
        default_replications=30,
    )
    persistence = _prompt_persistence_settings(multi_run=multi_run, benchmark=option.identifier)

    return _execute_workflow(
        option,
        extra_args,
        settings,
        persistence,
        default_seed,
        custom_benchmark=custom_benchmark,
        custom_factory=custom_factory,
    )


def run_cli_from_args(args: argparse.Namespace) -> AdaptiveSVRResult:
    """Execute the workflow directly from parsed CLI flags."""

    option = _resolve_benchmark_flag(args.benchmark)
    custom_benchmark = None
    custom_factory = None

    if option.identifier == "custom":
        if not args.custom_benchmark:
            raise SystemExit("--custom-benchmark is required when using --benchmark custom.")
        custom_benchmark = str(args.custom_benchmark)
        custom_factory = None if args.custom_factory is None else str(args.custom_factory)
        extra_args = ()
    else:
        extra_args = ()

    if args.max_iter < 1:
        raise SystemExit("--max-iter must be at least 1.")
    if args.replications < 1:
        raise SystemExit("--replications must be at least 1.")
    if args.ensemble_size < 1:
        raise SystemExit("--ensemble-size must be at least 1.")
    if args.n_workers is not None and args.n_workers < 1:
        raise SystemExit("--n-workers must be at least 1.")
    if args.random_seed is not None and args.random_seed < 0:
        raise SystemExit("--random-seed must be a non-negative integer.")
    if args.tail_alpha < 0.0:
        raise SystemExit("--tail-alpha must be non-negative.")
    if args.checkpoint_frequency < 1:
        raise SystemExit("--checkpoint-frequency must be at least 1.")

    tail_policy = args.tail_policy

    def _clean_path(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    persistence = _PersistenceSettings(
        checkpoint_path=_clean_path(args.checkpoint_path),
        checkpoint_frequency=int(args.checkpoint_frequency),
        resume_path=_clean_path(args.resume_from),
        result_output_path=_clean_path(args.result_output),
    )

    settings = _RunSettings(
        max_iter=int(args.max_iter),
        ensemble_size=int(args.ensemble_size),
        n_workers=None if args.n_workers is None else int(args.n_workers),
        random_seed=None if args.random_seed is None else int(args.random_seed),
        replications=int(args.replications),
        tail_policy=tail_policy,
        tail_alpha=float(args.tail_alpha),

        svr_kernel=args.svr_kernel,
        profile=str(args.profile),
    )

    default_seed = get_default_seed()
    return _execute_workflow(
        option,
        extra_args,
        settings,
        persistence,
        default_seed,
        custom_benchmark=custom_benchmark,
        custom_factory=custom_factory,
    )


def main(argv: Optional[Sequence[str]] = None) -> AdaptiveSVRResult:
    """Entry point used by ``python -m absvr_cli``."""

    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    automation_requested = _automation_flags_requested(args)

    if args.benchmark is None:
        if automation_requested:
            parser.error("--benchmark is required when using non-interactive flags.")
        return run_cli(max_iterations=args.max_iter)

    return run_cli_from_args(args)


if __name__ == "__main__":
    main()
