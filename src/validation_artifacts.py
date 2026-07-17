"""Deterministic hashes and environment records for experimental routes.

This module is validation infrastructure only.  It does not alter the channel
model, estimators, objective definitions, or frozen result schemas.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import pathlib
import platform
import struct
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Iterable

import numpy as np


def _hash_bytes(hasher: Any, value: bytes) -> None:
    hasher.update(struct.pack("!Q", len(value)))
    hasher.update(value)


def _update_canonical_hash(hasher: Any, value: Any) -> None:
    """Hash nested numerical state without lossy JSON float conversion."""
    if value is None:
        hasher.update(b"N")
        return
    if isinstance(value, (bool, np.bool_)):
        hasher.update(b"B1" if bool(value) else b"B0")
        return
    if isinstance(value, (int, np.integer)):
        hasher.update(b"I")
        _hash_bytes(hasher, str(int(value)).encode("ascii"))
        return
    if isinstance(value, (float, np.floating)):
        hasher.update(b"F")
        hasher.update(struct.pack("!d", float(value)))
        return
    if isinstance(value, (complex, np.complexfloating)):
        hasher.update(b"C")
        hasher.update(struct.pack("!dd", float(np.real(value)), float(np.imag(value))))
        return
    if isinstance(value, (str, pathlib.Path)):
        hasher.update(b"S")
        _hash_bytes(hasher, str(value).encode("utf-8"))
        return
    if isinstance(value, bytes):
        hasher.update(b"Y")
        _hash_bytes(hasher, value)
        return
    if isinstance(value, np.ndarray):
        array = np.asarray(value)
        if array.dtype.hasobject:
            _update_canonical_hash(hasher, array.tolist())
            return
        contiguous = np.ascontiguousarray(array)
        hasher.update(b"A")
        _hash_bytes(hasher, contiguous.dtype.str.encode("ascii"))
        _update_canonical_hash(hasher, tuple(int(item) for item in contiguous.shape))
        raw = contiguous.view(np.uint8)
        hasher.update(struct.pack("!Q", int(raw.size)))
        hasher.update(memoryview(raw))
        return
    if isinstance(value, dict):
        hasher.update(b"D")
        keys = sorted(value, key=lambda item: str(item))
        hasher.update(struct.pack("!Q", len(keys)))
        for key in keys:
            _update_canonical_hash(hasher, str(key))
            _update_canonical_hash(hasher, value[key])
        return
    if isinstance(value, (list, tuple)):
        hasher.update(b"L" if isinstance(value, list) else b"T")
        hasher.update(struct.pack("!Q", len(value)))
        for item in value:
            _update_canonical_hash(hasher, item)
        return
    raise TypeError(f"unsupported canonical-hash type: {type(value).__name__}")


def canonical_hash(value: Any, *, digest_size: int = 32) -> str:
    """Return a deterministic BLAKE2b hash of nested numerical state."""
    hasher = hashlib.blake2b(digest_size=int(digest_size))
    _update_canonical_hash(hasher, value)
    return hasher.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    """Return the full SHA-256 hash of one exact array representation."""
    contiguous = np.ascontiguousarray(np.asarray(array))
    hasher = hashlib.sha256()
    hasher.update(contiguous.dtype.str.encode("ascii"))
    hasher.update(str(tuple(int(item) for item in contiguous.shape)).encode("ascii"))
    hasher.update(memoryview(contiguous.view(np.uint8)))
    return hasher.hexdigest()


def deterministic_stage1_output(stage1_estimate: dict) -> dict:
    """Drop timing-only fields before hashing the complete Stage-I estimate."""
    return {
        key: value
        for key, value in stage1_estimate.items()
        if not str(key).startswith("stage1_time_")
        and not str(key).endswith("_runtime_s")
    }


def git_value(arguments: list[str], *, repo_root: pathlib.Path) -> str:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    return result.stdout.strip() or ""


def worktree_fingerprint(repo_root: pathlib.Path) -> str:
    """Hash tracked diffs and untracked source files, including CCOP modules."""
    hasher = hashlib.sha256()
    try:
        diff = subprocess.run(
            ["git", "diff", "--binary", "HEAD", "--", "."],
            cwd=repo_root,
            check=True,
            capture_output=True,
            timeout=20.0,
        ).stdout
        _hash_bytes(hasher, diff)
        untracked_text = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10.0,
        ).stdout
        paths = sorted(path for path in untracked_text.splitlines() if path)
        for relative in paths:
            path = repo_root / relative
            if not path.is_file():
                continue
            _hash_bytes(hasher, relative.encode("utf-8"))
            _hash_bytes(hasher, path.read_bytes())
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    return hasher.hexdigest()


def _distribution_version(names: Iterable[str]) -> str:
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return "unavailable"


def validation_environment(command: str, *, repo_root: pathlib.Path) -> dict:
    """Return a reproducibility record without initializing a CUDA context."""
    scipy_version = _distribution_version(["scipy"])
    cupy_version = _distribution_version(
        ["cupy", "cupy-cuda13x", "cupy-cuda12x", "cupy-cuda11x"]
    )
    status = git_value(["status", "--porcelain"], repo_root=repo_root)
    return {
        "command": command,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_value(["rev-parse", "HEAD"], repo_root=repo_root),
        "git_branch": git_value(["branch", "--show-current"], repo_root=repo_root),
        "git_dirty": bool(status and status != "unavailable"),
        "git_status_porcelain": status,
        "worktree_fingerprint": worktree_fingerprint(repo_root),
        "python_executable": sys.executable,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy_version,
        "cupy": cupy_version,
    }
