"""Download and normalize the MedRAG Textbooks corpus."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

from .corpus_registry import normalize_medrag_record
from .data_paths import TEXTBOOKS_CORPUS_FILE, TEXTBOOKS_DOWNLOAD_DIR
from .json_utils import save_json_atomic


TEXTBOOKS_DATASET_NAME = "MedRAG/textbooks"
TEXTBOOKS_DATASET_SPLIT = "train"
HF_TIMEOUT_SECONDS = "60"
HF_MAX_RETRIES = 5


def sync_textbooks_dataset(
    *,
    output_file: Path = TEXTBOOKS_CORPUS_FILE,
    cache_dir: Path = TEXTBOOKS_DOWNLOAD_DIR,
) -> Dict[str, object]:
    """Download MedRAG/Textbooks and persist a project-local JSON artifact."""
    try:
        from datasets import DownloadConfig
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("Install datasets to download MedRAG/textbooks") from exc

    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", HF_TIMEOUT_SECONDS)
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", HF_TIMEOUT_SECONDS)
    print(
        f"Downloading {TEXTBOOKS_DATASET_NAME} split={TEXTBOOKS_DATASET_SPLIT} "
        f"to cache {cache_dir}",
        flush=True,
    )
    dataset = load_dataset(
        TEXTBOOKS_DATASET_NAME,
        split=TEXTBOOKS_DATASET_SPLIT,
        cache_dir=str(cache_dir),
        download_config=DownloadConfig(
            resume_download=True,
            max_retries=HF_MAX_RETRIES,
            num_proc=1,
        ),
    )

    records: List[Dict[str, object]] = []
    for row in dataset:
        records.append(normalize_medrag_record(row, source_name="textbooks"))

    if not records:
        raise ValueError(f"No records loaded from {TEXTBOOKS_DATASET_NAME}")

    save_json_atomic(output_file, records)
    return {
        "dataset": TEXTBOOKS_DATASET_NAME,
        "split": TEXTBOOKS_DATASET_SPLIT,
        "record_count": len(records),
        "output_file": str(output_file),
        "cache_dir": str(cache_dir),
    }
