"""
Shared evaluation progress and result persistence helpers.

This module standardizes:
- realtime console progress output
- incremental JSON/TXT persistence during evaluation
- final result artifact writing
- optional checkpoint/resume support
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class CheckpointData:
    """Checkpoint data structure for resumable evaluation stages."""

    timestamp: str
    script_name: str
    dataset_name: str
    total_questions: int
    processed_questions: int
    current_top_k: int
    results: List[Dict[str, Any]]
    correct_count: int
    total_count: int
    elapsed_time: float
    config: Dict[str, Any]
    error_message: Optional[str] = None


class EvaluationProgressManager:
    """Manage evaluation progress, checkpoints, and standardized artifacts."""

    def __init__(self, output_dir: str = "./results/evaluation"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_file = self.output_dir / "checkpoint.json"
        self.backup_file = self.output_dir / "checkpoint.backup.json"
        self._current_checkpoint: Optional[CheckpointData] = None

    def create_run_artifacts(
        self,
        run_name: str,
        *,
        timestamp: Optional[str] = None,
    ) -> Dict[str, Path]:
        """Create standardized live/final artifact paths for one evaluation run."""
        run_slug = run_name.lower().replace("-", "_").replace(" ", "_")
        stamp = timestamp or time.strftime("%Y%m%d_%H%M%S")
        return {
            "live_json": self.output_dir / f"{run_slug}_live.json",
            "live_summary": self.output_dir / f"{run_slug}_live_summary.txt",
            "final_json": self.output_dir / f"{run_slug}_{stamp}.json",
            "final_summary": self.output_dir / f"{run_slug}_summary_{stamp}.txt",
        }

    def print_progress(
        self,
        *,
        run_name: str,
        dataset_name: str,
        processed_questions: int,
        total_questions: int,
        correct_count: int,
        elapsed_time: float,
    ) -> None:
        """Print a standardized realtime progress line."""
        accuracy = correct_count / processed_questions if processed_questions > 0 else 0.0
        progress = processed_questions / total_questions if total_questions > 0 else 0.0
        speed = processed_questions / elapsed_time if elapsed_time > 0 else 0.0
        print(
            f"[{run_name}][{dataset_name}] "
            f"{processed_questions}/{total_questions} "
            f"({progress * 100:.1f}%) | "
            f"acc {accuracy:.4f} | "
            f"{speed:.2f} q/s | "
            f"{elapsed_time:.1f}s"
        )

    def build_stage_result(
        self,
        *,
        dataset_name: str,
        total_questions: int,
        processed_questions: int,
        correct_count: int,
        elapsed_time: float,
        detailed_results: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build a standardized stage result payload."""
        stage_result: Dict[str, Any] = {
            "dataset_name": dataset_name,
            "total_questions": total_questions,
            "processed_questions": processed_questions,
            "correct": correct_count,
            "accuracy": correct_count / processed_questions if processed_questions > 0 else 0.0,
            "elapsed_time": elapsed_time,
            "questions_per_second": processed_questions / elapsed_time if elapsed_time > 0 else 0.0,
            "detailed_results": detailed_results,
        }
        if top_k is not None:
            stage_result["top_k"] = top_k
        if extra:
            stage_result.update(extra)
        return stage_result

    def write_live_results(
        self,
        *,
        artifact_paths: Dict[str, Path],
        run_name: str,
        evaluation_type: str,
        config: Dict[str, Any],
        stage_result: Dict[str, Any],
        extra_sections: Optional[Dict[str, Any]] = None,
        status: str = "running",
    ) -> None:
        """Write standardized realtime JSON and TXT artifacts."""
        payload: Dict[str, Any] = {
            "run_name": run_name,
            "evaluation_type": evaluation_type,
            "status": status,
            "updated_at": datetime.now().isoformat(),
            "config": config,
            "current_stage": stage_result,
        }
        if extra_sections:
            payload.update(extra_sections)

        self._write_json(artifact_paths["live_json"], payload)
        artifact_paths["live_summary"].write_text(
            self._build_summary_text(payload),
            encoding="utf-8",
        )

    def write_final_results(
        self,
        *,
        artifact_paths: Dict[str, Path],
        run_name: str,
        evaluation_type: str,
        config: Dict[str, Any],
        stage_results: Dict[str, Any],
        extra_sections: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Path]:
        """Write standardized final JSON and TXT artifacts."""
        payload: Dict[str, Any] = {
            "run_name": run_name,
            "evaluation_type": evaluation_type,
            "status": "completed",
            "updated_at": datetime.now().isoformat(),
            "config": config,
        }
        payload.update(stage_results)
        if extra_sections:
            payload.update(extra_sections)

        self._write_json(artifact_paths["final_json"], payload)
        artifact_paths["final_summary"].write_text(
            self._build_summary_text(payload),
            encoding="utf-8",
        )

        if artifact_paths["live_json"].exists():
            artifact_paths["live_json"].unlink()
        if artifact_paths["live_summary"].exists():
            artifact_paths["live_summary"].unlink()

        return {
            "json": artifact_paths["final_json"],
            "summary": artifact_paths["final_summary"],
        }

    def _write_json(self, path: Path, payload: Dict[str, Any]) -> None:
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _build_summary_text(self, payload: Dict[str, Any]) -> str:
        lines = [
            "Medical RAG Evaluation Progress",
            "=" * 60,
            f"Run: {payload.get('run_name', 'UNKNOWN')}",
            f"Type: {payload.get('evaluation_type', 'UNKNOWN')}",
            f"Status: {payload.get('status', 'unknown')}",
            f"Updated: {payload.get('updated_at', '')}",
            "",
            "Configuration:",
        ]

        config = payload.get("config", {})
        for key, value in config.items():
            lines.append(f"  {key}: {value}")

        for section_name, title in (
            ("current_stage", "Current Stage"),
            ("development_set_evaluation", "Development Set"),
            ("test_set_evaluation", "Test Set"),
            ("evaluation_results", "Evaluation Results"),
        ):
            section = payload.get(section_name)
            if not isinstance(section, dict):
                continue
            lines.extend(self._format_stage_section(title, section))

        if "hyperparameter_search" in payload:
            lines.append("")
            lines.append("Hyperparameter Search:")
            for key, value in payload["hyperparameter_search"].items():
                lines.append(f"  {key}: {value}")

        if "retrieval_recall_at_k" in payload:
            lines.append("")
            lines.append("Retrieval Recall@k:")
            for key, value in payload["retrieval_recall_at_k"].items():
                lines.append(f"  {key}: {value}")

        return "\n".join(lines) + "\n"

    def _format_stage_section(self, title: str, section: Dict[str, Any]) -> List[str]:
        lines = ["", f"{title}:"]
        for key in (
            "dataset_name",
            "top_k",
            "processed_questions",
            "total_questions",
            "correct",
            "accuracy",
            "elapsed_time",
            "questions_per_second",
        ):
            if key in section:
                value = section[key]
                if isinstance(value, float) and key in {"accuracy", "elapsed_time", "questions_per_second"}:
                    if key == "accuracy":
                        value = f"{value:.4f}"
                    else:
                        value = f"{value:.2f}"
                lines.append(f"  {key}: {value}")
        return lines

    def _get_backup_path(self, script_name: Optional[str] = None) -> Path:
        if script_name:
            return self.output_dir / f"checkpoint_{script_name}.backup.json"
        return self.backup_file

    def _get_checkpoint_path(self, script_name: Optional[str] = None) -> Path:
        if script_name:
            return self.output_dir / f"checkpoint_{script_name}.json"
        return self.checkpoint_file

    def has_checkpoint(self, script_name: Optional[str] = None) -> bool:
        checkpoint_path = self._get_checkpoint_path(script_name)
        return checkpoint_path.exists()

    def load_checkpoint(self, script_name: Optional[str] = None) -> Optional[CheckpointData]:
        checkpoint_path = self._get_checkpoint_path(script_name)
        backup_path = self._get_backup_path(script_name)

        if not checkpoint_path.exists():
            if backup_path.exists():
                checkpoint_path = backup_path
            else:
                return None

        try:
            data = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            checkpoint = CheckpointData(**data)
            self._current_checkpoint = checkpoint
            print(
                f"[resume][{checkpoint.script_name}] "
                f"{checkpoint.processed_questions}/{checkpoint.total_questions} | "
                f"acc {checkpoint.correct_count / checkpoint.total_count:.4f}"
                if checkpoint.total_count > 0
                else f"[resume][{checkpoint.script_name}] {checkpoint.processed_questions}/{checkpoint.total_questions}"
            )
            return checkpoint
        except (json.JSONDecodeError, TypeError, KeyError):
            if backup_path.exists() and backup_path != checkpoint_path:
                try:
                    data = json.loads(backup_path.read_text(encoding="utf-8"))
                    checkpoint = CheckpointData(**data)
                    self._current_checkpoint = checkpoint
                    return checkpoint
                except Exception:
                    return None
            return None

    def save_checkpoint(
        self,
        dataset_name: str,
        total_questions: int,
        processed_questions: int,
        current_top_k: int,
        results: List[Dict[str, Any]],
        correct_count: int,
        total_count: int,
        elapsed_time: float,
        config: Dict[str, Any],
        script_name: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> None:
        checkpoint = CheckpointData(
            timestamp=datetime.now().isoformat(),
            script_name=script_name or "unknown",
            dataset_name=dataset_name,
            total_questions=total_questions,
            processed_questions=processed_questions,
            current_top_k=current_top_k,
            results=results,
            correct_count=correct_count,
            total_count=total_count,
            elapsed_time=elapsed_time,
            config=config,
            error_message=error_message,
        )
        self._current_checkpoint = checkpoint

        checkpoint_path = self._get_checkpoint_path(script_name)
        backup_path = self._get_backup_path(script_name)
        if checkpoint_path.exists():
            try:
                shutil.copy2(checkpoint_path, backup_path)
            except Exception:
                pass

        checkpoint_path.write_text(
            json.dumps(asdict(checkpoint), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def clear_checkpoint(self, script_name: Optional[str] = None) -> None:
        checkpoint_path = self._get_checkpoint_path(script_name)
        backup_path = self._get_backup_path(script_name)
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        if backup_path.exists():
            backup_path.unlink()

    def should_resume(self, script_name: Optional[str] = None) -> bool:
        if not self.has_checkpoint(script_name):
            return False

        checkpoint = self.load_checkpoint(script_name)
        if not checkpoint:
            return False

        if checkpoint.processed_questions >= checkpoint.total_questions:
            self.clear_checkpoint(script_name)
            return False

        return True

    def get_resume_info(self, script_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        checkpoint = self.load_checkpoint(script_name)
        if not checkpoint:
            return None
        return {
            "start_from": checkpoint.processed_questions,
            "results": checkpoint.results,
            "correct_count": checkpoint.correct_count,
            "total_count": checkpoint.total_count,
            "elapsed_time": checkpoint.elapsed_time,
            "current_top_k": checkpoint.current_top_k,
            "config": checkpoint.config,
        }


def create_progress_manager(output_dir: str = "./results/evaluation") -> EvaluationProgressManager:
    """Factory function to create a progress manager."""
    return EvaluationProgressManager(output_dir)
