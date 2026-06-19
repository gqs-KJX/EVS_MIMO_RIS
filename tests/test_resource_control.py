import os

import numpy as np

from src.experiments.resource_control import (
    apply_thread_limits,
    assert_row_is_light,
    resolve_hybrid_resources,
)


def test_resolve_auto_threads_with_explicit_workers():
    plan = resolve_hybrid_resources(
        jobs=10,
        process_workers=4,
        blas_threads="auto",
        n_tasks=20,
    )
    assert plan["process_workers"] == 4
    assert plan["blas_threads"] >= 2
    assert plan["estimated_cpu_slots"] <= 10


def test_resolve_respects_memory_budget():
    plan = resolve_hybrid_resources(
        jobs=10,
        process_workers=None,
        blas_threads="auto",
        n_tasks=20,
        memory_budget_gb=9.0,
        memory_per_worker_gb=2.5,
    )
    assert plan["process_workers"] == 3


def test_apply_thread_limits_sets_environment(monkeypatch):
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        monkeypatch.delenv(name, raising=False)
    apply_thread_limits(3)
    assert os.environ["OMP_NUM_THREADS"] == "3"
    assert os.environ["MKL_NUM_THREADS"] == "3"
    assert os.environ["OPENBLAS_NUM_THREADS"] == "3"
    assert os.environ["NUMEXPR_NUM_THREADS"] == "3"


def test_assert_row_is_light_removes_ndarrays():
    row = assert_row_is_light({"metric": 1.0, "diag": np.arange(3)})
    assert not any(isinstance(value, np.ndarray) for value in row.values())
