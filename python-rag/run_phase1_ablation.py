"""Phase 1 smoke runner for corpus and retrieval ablation wiring.

The constants below intentionally replace command-line arguments so each run is
reproducible from the checked-in script configuration.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Dict, List, Sequence

from app.rag.data.corpus_registry import combine_registered_corpora
from app.rag.data.data_paths import (
    COMBINED_CORPUS_FILE,
    PHASE1_INDEX_DIR,
    PHASE1_RESULTS_DIR,
    TEXTBOOKS_CORPUS_FILE,
    ensure_data_directories,
)
from app.rag.data.json_utils import save_json_atomic
from app.rag.data.medical_corpus.build_vector_index import IndexBuildConfig, build_index
from app.rag.data.textbooks_dataset import sync_textbooks_dataset
from app.rag.evaluation.eval_shared import EvaluationLLMConfig, load_questions
from app.rag.evaluation.naive_rag_eval import evaluate_sync_dataset, load_vector_store
from app.rag.retriever.runtime_config import DEFAULT_HF_EMBEDDING_MODEL


AUTO_SYNC_TEXTBOOKS = True
SAMPLE_SIZE = 5
MAX_RECORDS_PER_SOURCE = 25
K_VALUES = [3, 5, 10]
CORPUS_VARIANTS: Dict[str, Sequence[str]] = {
    "statpearls": ("statpearls",),
    "statpearls_textbooks": ("statpearls", "textbooks"),
}
PLANNED_EMBEDDING_MODELS = [
    DEFAULT_HF_EMBEDDING_MODEL,
    "ncbi/MedCPT-Query-Encoder",
    "BAAI/bge-large-en-v1.5",
]
SMOKE_EMBEDDING_MODELS = [DEFAULT_HF_EMBEDDING_MODEL]
EMBEDDING_DEVICE = "auto"
EMBEDDING_LOCAL_FILES_ONLY = True
INDEX_BATCH_SIZE = 64
INDEX_INSERT_BATCH_SIZE = 256


def _slug(value: str) -> str:
    """Create stable path names without repeating ad-hoc replacements."""
    return (
        value.lower()
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace(" ", "_")
    )


def ensure_phase1_corpora() -> None:
    """Prepare required source corpora before the smoke runner combines variants."""
    ensure_data_directories()
    if not COMBINED_CORPUS_FILE.exists():
        raise FileNotFoundError(
            "StatPearls source is missing. Use the existing statpearls.py processing "
            f"path to generate {COMBINED_CORPUS_FILE} from statpearls_NBK430685."
        )

    if AUTO_SYNC_TEXTBOOKS and not TEXTBOOKS_CORPUS_FILE.exists():
        print("Textbooks source JSON is missing; syncing MedRAG/textbooks now.", flush=True)
        sync_textbooks_dataset()

    missing_sources = [
        str(path)
        for path in (TEXTBOOKS_CORPUS_FILE,)
        if not path.exists()
    ]
    if missing_sources:
        raise FileNotFoundError(
            "Required phase 1 corpus sources are missing: " + ", ".join(missing_sources)
        )


def select_smoke_records(records: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    """Take a balanced source slice so smoke indexes stay cheap and explicit."""
    counts: Dict[str, int] = {}
    selected: List[Dict[str, object]] = []
    for record in records:
        source = str(record.get("source") or "unknown")
        count = counts.get(source, 0)
        if count >= MAX_RECORDS_PER_SOURCE:
            continue
        selected.append(dict(record))
        counts[source] = count + 1
    return selected


def write_smoke_corpus(variant_name: str, sources: Sequence[str]) -> Path:
    """Persist the exact tiny corpus used for a smoke index build."""
    result = combine_registered_corpora(selected_sources=sources)
    records = select_smoke_records(result["records"])
    if not records:
        raise ValueError(f"No smoke records selected for {variant_name}")

    corpus_path = PHASE1_RESULTS_DIR / "corpora" / f"{variant_name}_smoke_corpus.json"
    save_json_atomic(corpus_path, records)
    print(
        f"Prepared smoke corpus {variant_name}: {len(records)} records -> {corpus_path}",
        flush=True,
    )
    return corpus_path


def build_phase1_index(
    *,
    variant_name: str,
    corpus_path: Path,
    embedding_model: str,
) -> Dict[str, object]:
    """Build one small FAISS index and return its metadata for the run summary."""
    index_dir = PHASE1_INDEX_DIR / f"{variant_name}__{_slug(embedding_model)}"
    config = IndexBuildConfig(
        corpus_file=corpus_path,
        index_dir=index_dir,
        embedding_model=embedding_model,
        embedding_device=EMBEDDING_DEVICE,
        batch_size=INDEX_BATCH_SIZE,
        insert_batch_size=INDEX_INSERT_BATCH_SIZE,
        local_files_only=EMBEDDING_LOCAL_FILES_ONLY,
        use_gpu_faiss=False,
        corpus_version=f"phase1-smoke:{variant_name}",
    )
    started_at = time.time()
    metadata = build_index(config)
    metadata["phase1_index_dir"] = str(index_dir)
    metadata["phase1_index_build_time_seconds"] = time.time() - started_at
    return metadata


def evaluate_index(
    *,
    variant_name: str,
    embedding_model: str,
    index_dir: Path,
) -> List[Dict[str, object]]:
    """Evaluate one smoke index over the configured k values."""
    questions = load_questions()[:SAMPLE_SIZE]
    vectorstore = load_vector_store(index_dir)
    rows: List[Dict[str, object]] = []
    for k in K_VALUES:
        print(
            f"Evaluating variant={variant_name}, embedding={embedding_model}, k={k}",
            flush=True,
        )
        result = evaluate_sync_dataset(
            vectorstore=vectorstore,
            llm_config=EvaluationLLMConfig(),
            questions=questions,
            top_k=k,
            run_name="PHASE1_SMOKE",
            evaluation_type="PHASE1_SMOKE_NAIVE_RAG",
            dataset_name=f"Phase 1 smoke ({variant_name}, k={k})",
            script_name="run_phase1_ablation",
        )
        rows.append(
            {
                "corpus_variant": variant_name,
                "embedding_model": embedding_model,
                "k": k,
                "accuracy": result["accuracy"],
                "correct": result["correct"],
                "total_questions": result["total_questions"],
                "elapsed_time": result["elapsed_time"],
                "questions_per_second": result["questions_per_second"],
                "retrieved_docs_first_question": (
                    result["detailed_results"][0].get("retrieved_docs")
                    if result["detailed_results"]
                    else None
                ),
            }
        )
    return rows


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    """Write phase 1 rows in a tabular form without duplicating result logic."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_phase1_smoke() -> Dict[str, object]:
    """Run the small, explicitly non-final phase 1 ablation smoke."""
    ensure_phase1_corpora()
    all_rows: List[Dict[str, object]] = []
    index_metadata: List[Dict[str, object]] = []

    for variant_name, sources in CORPUS_VARIANTS.items():
        corpus_path = write_smoke_corpus(variant_name, sources)
        for embedding_model in SMOKE_EMBEDDING_MODELS:
            metadata = build_phase1_index(
                variant_name=variant_name,
                corpus_path=corpus_path,
                embedding_model=embedding_model,
            )
            index_metadata.append(metadata)
            all_rows.extend(
                evaluate_index(
                    variant_name=variant_name,
                    embedding_model=embedding_model,
                    index_dir=Path(str(metadata["phase1_index_dir"])),
                )
            )

    summary = {
        "run_type": "phase1_smoke_not_formal_results",
        "sample_size": SAMPLE_SIZE,
        "max_records_per_source": MAX_RECORDS_PER_SOURCE,
        "corpus_variants": {k: list(v) for k, v in CORPUS_VARIANTS.items()},
        "planned_embedding_models": PLANNED_EMBEDDING_MODELS,
        "smoke_embedding_models": SMOKE_EMBEDDING_MODELS,
        "k_values": K_VALUES,
        "index_metadata": index_metadata,
        "results": all_rows,
    }
    json_path = PHASE1_RESULTS_DIR / "phase1_smoke_summary.json"
    csv_path = PHASE1_RESULTS_DIR / "phase1_smoke_summary.csv"
    save_json_atomic(json_path, summary)
    write_csv(csv_path, all_rows)
    summary["summary_json"] = str(json_path)
    summary["summary_csv"] = str(csv_path)
    return summary


def main() -> None:
    try:
        summary = run_phase1_smoke()
    except Exception as exc:
        blocker_path = PHASE1_RESULTS_DIR / "phase1_smoke_blocker.json"
        save_json_atomic(
            blocker_path,
            {
                "run_type": "phase1_smoke_not_formal_results",
                "status": "blocked",
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        print(f"Phase 1 smoke blocked; details written to {blocker_path}", flush=True)
        raise

    print("=" * 60)
    print("Phase 1 Smoke Complete")
    print("=" * 60)
    print(f"Summary JSON: {summary['summary_json']}")
    print(f"Summary CSV: {summary['summary_csv']}")
    print("These smoke results prove the pipeline runs; they are not formal ablation results.")


if __name__ == "__main__":
    main()
