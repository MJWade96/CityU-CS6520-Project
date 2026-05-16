"""Shared JSON file helpers.

These helpers keep UTF-8 JSON I/O and atomic writes in one place so scripts
do not repeat the same persistence code.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import uuid4


def load_json_safe(path: str | Path) -> Any:
    """Load JSON from a UTF-8 file path."""
    json_path = Path(path)
    with json_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json_atomic(
    path: str | Path,
    payload: Any,
    *,
    indent: int = 2,
    ensure_ascii: bool = False,
) -> Path:
    """Persist JSON through a same-directory temp file and atomic replace."""
    json_path = Path(path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = json_path.with_name(f".{json_path.name}.{uuid4().hex}.tmp")

    try:
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=indent, ensure_ascii=ensure_ascii)
        temp_path.replace(json_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()

    return json_path