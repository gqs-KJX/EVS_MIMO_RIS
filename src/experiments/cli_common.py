"""Shared argparse builders for the experiment runners.

Centralizing the common CPU/worker/memory, IO, Monte-Carlo, and progress flags
keeps every runner's command line consistent and means a new shared flag only
has to be added in one place. Per-runner defaults are passed in explicitly so
existing behavior, defaults, and CSV/result conventions are preserved exactly.
"""

from __future__ import annotations

import argparse
import pathlib


def add_resource_args(
    parser: argparse.ArgumentParser,
    *,
    jobs_default: int = 10,
    blas_threads_default: str = "auto",
    include_respect_existing_blas_env: bool = True,
    include_trim_memory: bool = True,
) -> None:
    """Add the shared CPU-slot / worker / BLAS-thread / memory-budget flags."""
    parser.add_argument(
        "--jobs",
        type=int,
        default=jobs_default,
        help="Total CPU-slot budget (process_workers * blas_threads).",
    )
    parser.add_argument(
        "--process-workers",
        type=int,
        default=None,
        help="Number of memory-heavy worker processes.",
    )
    parser.add_argument(
        "--blas-threads",
        default=blas_threads_default,
        help="Native BLAS threads per worker, or 'auto'.",
    )
    parser.add_argument("--memory-budget-gb", type=float, default=None)
    parser.add_argument("--memory-per-worker-gb", type=float, default=None)
    if include_respect_existing_blas_env:
        parser.add_argument("--respect-existing-blas-env", action="store_true")
    if include_trim_memory:
        parser.add_argument(
            "--trim-memory", action=argparse.BooleanOptionalAction, default=True
        )


def normalize_blas_threads(args: argparse.Namespace) -> argparse.Namespace:
    """Convert a non-'auto' --blas-threads value to int, in place."""
    if str(args.blas_threads).lower() != "auto":
        args.blas_threads = int(args.blas_threads)
    return args


def add_io_args(
    parser: argparse.ArgumentParser,
    *,
    default_out_dir: str,
    seed_default: int = 20260526,
    include_force_rerun: bool = True,
) -> None:
    """Add the shared output-directory / seed / force-rerun flags."""
    parser.add_argument(
        "--out-dir", type=pathlib.Path, default=pathlib.Path(default_out_dir)
    )
    parser.add_argument("--seed", type=int, default=seed_default)
    if include_force_rerun:
        parser.add_argument("--force-rerun", action="store_true")


def add_mc_args(
    parser: argparse.ArgumentParser,
    *,
    n_trials_default: int = 50,
    paper_k_default: int | None = None,
    outlier_threshold_default: float | None = 0.1,
) -> None:
    """Add the shared Monte-Carlo flags (trial count and optional K / outlier)."""
    parser.add_argument("--n-trials", type=int, default=n_trials_default)
    if paper_k_default is not None:
        parser.add_argument("--paper-k", type=int, default=paper_k_default)
    if outlier_threshold_default is not None:
        parser.add_argument(
            "--outlier-threshold-m", type=float, default=outlier_threshold_default
        )


def add_progress_args(
    parser: argparse.ArgumentParser,
    *,
    include_no_plots: bool = True,
    include_profile_memory: bool = True,
) -> None:
    """Add the shared progress / plotting flags."""
    if include_no_plots:
        parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--progress-log", type=pathlib.Path, default=None)
    parser.add_argument("--quiet-progress", action="store_true")
    if include_profile_memory:
        parser.add_argument("--profile-memory", action="store_true")
