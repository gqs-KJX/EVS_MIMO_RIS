"""Print the latest state from an experiment progress JSONL file."""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any

from .progress_logger import format_eta


def read_progress(path: str | pathlib.Path) -> tuple[dict[str, Any], dict[str, Any] | None]:
    latest: dict[str, Any] | None = None
    latest_error: dict[str, Any] | None = None
    latest_context: dict[str, Any] | None = None
    with pathlib.Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            event = json.loads(line)
            latest = event
            if any(
                event.get(key)
                for key in (
                    "figure",
                    "baseline_or_variant",
                    "snr_db",
                    "trial_id",
                    "seed",
                    "K",
                )
            ):
                latest_context = event
            if event.get("error") or event.get("event_type") == "task_failed":
                latest_error = event
    if latest is None:
        raise ValueError("progress log contains no events")
    if latest_context is not None:
        latest = dict(latest)
        for key in (
            "figure",
            "baseline_or_variant",
            "snr_db",
            "trial_id",
            "seed",
            "K",
        ):
            if latest.get(key, "") == "":
                latest[key] = latest_context.get(key, "")
    return latest, latest_error


def print_progress(path: str | pathlib.Path) -> None:
    latest, latest_error = read_progress(path)
    print(
        f"Latest: {latest.get('event_type')} / {latest.get('status')} "
        f"at {latest.get('timestamp_utc')}"
    )
    print(
        f"Progress: {latest.get('done_tasks')}/{latest.get('total_tasks')} "
        f"({float(latest.get('percent', 0.0)):.1f}%)"
    )
    print(
        f"Elapsed: {format_eta(latest.get('elapsed_s'))}  "
        f"ETA: {format_eta(latest.get('eta_s'))}"
    )
    print(
        "Current: "
        f"figure={latest.get('figure', '')} "
        f"baseline/variant={latest.get('baseline_or_variant', '')} "
        f"snr_db={latest.get('snr_db', '')} "
        f"trial={latest.get('trial_id', '')} "
        f"seed={latest.get('seed', '')} "
        f"K={latest.get('K', '')}"
    )
    if latest_error is not None:
        print(f"Latest error: {latest_error.get('error') or latest_error.get('message')}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--progress-log", type=pathlib.Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    print_progress(args.progress_log)


if __name__ == "__main__":
    main()
