"""Materialize deterministic CCOP presets and seed splits as result artifacts."""

from __future__ import annotations

import argparse
import json
import pathlib
import shlex
import sys

from src.validation_artifacts import validation_environment
from .ccop_validation_presets import PRESETS, seed_splits


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=pathlib.Path("results/ccop_full_validation"),
    )
    parser.add_argument("--force-rerun", action="store_true")
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "seed_splits": args.out_dir / "seed_splits.json",
        "presets": args.out_dir / "presets.json",
        "environment": args.out_dir / "protocol_environment.json",
    }
    if not args.force_rerun and any(path.exists() for path in paths.values()):
        raise FileExistsError("protocol artifacts exist; use --force-rerun")
    command = shlex.join(
        [
            sys.executable,
            "-m",
            "src.experiments.write_ccop_validation_protocol",
            *(argv or sys.argv[1:]),
        ]
    )
    records = {
        "seed_splits": seed_splits(),
        "presets": PRESETS,
        "environment": validation_environment(
            command, repo_root=pathlib.Path(__file__).resolve().parents[3]
        ),
    }
    for key, path in paths.items():
        path.write_text(
            json.dumps(records[key], indent=2, sort_keys=True), encoding="utf-8"
        )
    print(json.dumps({key: str(path) for key, path in paths.items()}, indent=2))


if __name__ == "__main__":
    main()
