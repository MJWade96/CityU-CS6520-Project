"""Shared metadata helpers for formal cache artifact manifests."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Mapping


FORMAL_CACHE_CONFIG_VERSION = "formal_cache_design_v1"


def current_code_version() -> str:
    """Return the current git commit when available."""
    try:
        repo_root = Path(__file__).resolve().parents[4]
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_root,
                text=True,
                stderr=subprocess.DEVNULL,
            )
            .strip()
        )
    except Exception:
        return "unknown"


def path_fingerprint(path: str | Path) -> Dict[str, Any]:
    artifact_path = Path(path)
    if not artifact_path.exists():
        return {"path": str(artifact_path), "exists": False}
    stat = artifact_path.stat()
    return {
        "path": str(artifact_path),
        "exists": True,
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def manifest_metadata(
    *,
    key: Mapping[str, Any],
    input_artifacts: Mapping[str, Any],
    parameters: Mapping[str, Any],
    dataset_split: str,
    fingerprint: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "key": dict(key),
        "input_artifacts": dict(input_artifacts),
        "parameters": dict(parameters),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "code_version": current_code_version(),
        "config_version": FORMAL_CACHE_CONFIG_VERSION,
        "dataset_split": dataset_split,
        "fingerprint": dict(fingerprint),
    }
