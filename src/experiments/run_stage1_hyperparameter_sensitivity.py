"""Paired sensitivity sweep for the Stage-I heuristic constants.

Answers the reviewer question that the frozen ``paper_v3`` campaign does not
cover: how sensitive is the estimator to the association shortlist size, the
association clock weight, the common-geometry confidence weights, and the
acquisition range grids?

The sweep changes nothing but the swept key.  Every value of the swept key
sees the *same* trial seeds, hence the same scene, Jones states and noise
realizations, so differences are paired and attributable to the parameter
alone.  Configuration is built by ``make_paper_config`` exactly as in the
campaign, and each trial runs the frozen ``proposed`` route through
``run_paper_variant``.

Example
-------
    python -m src.experiments.run_stage1_hyperparameter_sensitivity \
        --param stage1_assignment_num_exact_permutations --values 2,3,6 \
        --snr-db -10 --n-trials 480 --jobs 24 \
        --out-dir results/sensitivity/shortlist
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import pathlib
from typing import Any

import numpy as np

from .final_mksc_ccop_common import (
    make_paper_config,
    make_shared_data,
    run_paper_variant,
    write_csv,
)
from .resource_control import apply_thread_limits

ROW_FIELDS = (
    "param",
    "value",
    "trial_id",
    "seed",
    "snr_db",
    "position_error_m",
    "clock_error_ns",
    "outlier",
    "stage1_assignment_margin",
    "runtime_s",
)


def _trial_seeds(root_seed: int, n_trials: int) -> list[int]:
    """Reproduce the campaign's seed spawning exactly."""
    sequence = np.random.SeedSequence(int(root_seed))
    return [
        int(child.generate_state(1, dtype=np.uint32)[0])
        for child in sequence.spawn(int(n_trials))
    ]


def _coerce(text: str) -> Any:
    """Parse a swept value as bool, int, float or string, in that order."""
    low = text.strip().lower()
    if low in {"true", "false"}:
        return low == "true"
    for caster in (int, float):
        try:
            return caster(text)
        except ValueError:
            continue
    return text


def _run_one(task: dict[str, Any]) -> dict[str, Any]:
    apply_thread_limits(int(task["blas_threads"]))
    config = make_paper_config(
        int(task["seed"]),
        float(task["snr_db"]),
        diagnostic_mode=str(task["diagnostic_mode"]),
        overrides={str(task["param"]): task["value"]},
    )
    data = make_shared_data(config)
    row = run_paper_variant(
        "proposed",
        data=data,
        config=config,
        suite="sensitivity",
        x_name=str(task["param"]),
        x_value=task["value"],
        trial_id=int(task["trial_id"]),
        outlier_threshold_m=float(task["outlier_threshold_m"]),
    )
    out = {
        "param": str(task["param"]),
        "value": task["value"],
        "trial_id": int(task["trial_id"]),
        "seed": int(task["seed"]),
        "snr_db": float(task["snr_db"]),
    }
    for key in ROW_FIELDS:
        if key in out:
            continue
        out[key] = row.get(key, "")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--param",
        required=True,
        help="configuration key to sweep, e.g. assignment_clock_weight",
    )
    parser.add_argument(
        "--values",
        required=True,
        help="comma-separated values for the swept key",
    )
    parser.add_argument("--snr-db", type=float, default=-10.0)
    parser.add_argument("--n-trials", type=int, default=480)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument(
        "--diagnostic-mode", choices=("fast", "performance"), default="performance"
    )
    parser.add_argument("--outlier-threshold-m", type=float, default=0.1)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--blas-threads", type=int, default=1)
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    args = parser.parse_args()

    values = [_coerce(v) for v in str(args.values).split(",") if v.strip()]
    seeds = _trial_seeds(args.seed, args.n_trials)
    tasks = [
        {
            "param": args.param,
            "value": value,
            "trial_id": index,
            "seed": seed,
            "snr_db": args.snr_db,
            "diagnostic_mode": args.diagnostic_mode,
            "outlier_threshold_m": args.outlier_threshold_m,
            "blas_threads": args.blas_threads,
        }
        for value in values
        for index, seed in enumerate(seeds)
    ]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "command.txt").write_text(
        json.dumps(vars(args), default=str, indent=2), encoding="utf-8"
    )

    rows: list[dict[str, Any]] = []
    if int(args.jobs) <= 1:
        for position, task in enumerate(tasks, start=1):
            rows.append(_run_one(task))
            print(f"[{position}/{len(tasks)}] {task['param']}={task['value']}", flush=True)
    else:
        with cf.ProcessPoolExecutor(max_workers=int(args.jobs)) as pool:
            for position, row in enumerate(pool.map(_run_one, tasks), start=1):
                rows.append(row)
                if position % 25 == 0 or position == len(tasks):
                    print(f"[{position}/{len(tasks)}]", flush=True)

    write_csv(args.out_dir / "sensitivity_trials.csv", rows, ROW_FIELDS)

    print("\nvalue                         N   median_pos_m   P_cat[%]")
    for value in values:
        subset = [r for r in rows if r["value"] == value]
        errors = np.array(
            [float(r["position_error_m"]) for r in subset if r["position_error_m"] != ""],
            dtype=float,
        )
        cats = [str(r["outlier"]).lower() in {"1", "true"} for r in subset]
        med = float(np.median(errors)) if errors.size else float("nan")
        rate = 100.0 * sum(cats) / max(len(cats), 1)
        print(f"{str(value):<28} {len(subset):>4}   {med:>12.6g}   {rate:>7.2f}")


if __name__ == "__main__":
    main()
