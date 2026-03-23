"""
Evaluation Progress Manager
Handles saving and resuming evaluation progress for long-running tasks

Features:
- Auto-save progress after each question
- Resume from last checkpoint on restart
- Handle interruptions gracefully
- Support multiple evaluation runs
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class CheckpointData:
    """Checkpoint data structure"""
    timestamp: str
    script_name: str
    dataset_name: str
    total_questions: int
    processed_questions: int
    current_top_k: int
    results: List[Dict]
    correct_count: int
    total_count: int
    elapsed_time: float
    config: Dict[str, Any]
    error_message: Optional[str] = None


class EvaluationProgressManager:
    """
    Manages evaluation progress with checkpoint support
    
    Usage:
        progress_mgr = EvaluationProgressManager(output_dir="./results/evaluation")
        
        # Check if we can resume
        if progress_mgr.has_checkpoint():
            checkpoint = progress_mgr.load_checkpoint()
            start_from = checkpoint.processed_questions
            results = checkpoint.results
        else:
            start_from = 0
            results = []
        
        # During evaluation, save after each question
        progress_mgr.save_checkpoint(
            dataset_name="Test Set",
            total_questions=len(questions),
            processed_questions=i,
            current_top_k=top_k,
            results=results,
            correct_count=correct,
            total_count=total,
            elapsed_time=elapsed,
            config=config_dict
        )
        
        # Clear checkpoint when done
        progress_mgr.clear_checkpoint()
    """
    
    def __init__(self, output_dir: str = "./results/evaluation"):
        """
        Initialize progress manager
        
        Args:
            output_dir: Directory to save checkpoints
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_file = self.output_dir / "checkpoint.json"
        self.backup_file = self.output_dir / "checkpoint.backup.json"
        
        # Auto-save interval (in questions)
        self.auto_save_interval = 1
        
        # Current checkpoint in memory
        self._current_checkpoint: Optional[CheckpointData] = None
    
    def _get_backup_path(self, script_name: Optional[str] = None) -> Path:
        """Get backup file path for a specific script"""
        if script_name:
            return self.output_dir / f"checkpoint_{script_name}.backup.json"
        return self.backup_file
    
    def _get_checkpoint_path(self, script_name: Optional[str] = None) -> Path:
        """Get checkpoint file path"""
        if script_name:
            return self.output_dir / f"checkpoint_{script_name}.json"
        return self.checkpoint_file
    
    def has_checkpoint(self, script_name: Optional[str] = None) -> bool:
        """Check if a checkpoint exists"""
        checkpoint_path = self._get_checkpoint_path(script_name)
        return checkpoint_path.exists()
    
    def load_checkpoint(
        self, 
        script_name: Optional[str] = None
    ) -> Optional[CheckpointData]:
        """
        Load checkpoint from disk
        
        Returns:
            CheckpointData if exists, None otherwise
        """
        checkpoint_path = self._get_checkpoint_path(script_name)
        
        if not checkpoint_path.exists():
            # Try backup
            if self.backup_file.exists():
                checkpoint_path = self.backup_file
            else:
                return None
        
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            checkpoint = CheckpointData(**data)
            self._current_checkpoint = checkpoint
            
            print(f"\n[OK] Checkpoint loaded from {checkpoint_path}")
            print(f"  Timestamp: {checkpoint.timestamp}")
            print(f"  Progress: {checkpoint.processed_questions}/{checkpoint.total_questions}")
            print(f"  Current top-k: {checkpoint.current_top_k}")
            print(f"  Accuracy so far: {checkpoint.correct_count}/{checkpoint.total_count}")
            
            return checkpoint
            
        except (json.JSONDecodeError, TypeError, KeyError) as e:
            print(f"Warning: Checkpoint file corrupted: {e}")
            # Try to load from backup
            backup_path = self._get_backup_path(script_name)
            if backup_path.exists() and backup_path != checkpoint_path:
                print(f"  Trying backup: {backup_path}")
                try:
                    with open(backup_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    checkpoint = CheckpointData(**data)
                    self._current_checkpoint = checkpoint
                    print(f"  [OK] Checkpoint loaded from backup")
                    return checkpoint
                except Exception as backup_e:
                    print(f"  Backup also failed: {backup_e}")
            return None
        except Exception as e:
            print(f"Warning: Could not load checkpoint: {e}")
            return None
    
    def save_checkpoint(
        self,
        dataset_name: str,
        total_questions: int,
        processed_questions: int,
        current_top_k: int,
        results: List[Dict],
        correct_count: int,
        total_count: int,
        elapsed_time: float,
        config: Dict[str, Any],
        script_name: Optional[str] = None,
        error_message: Optional[str] = None,
    ):
        """
        Save current progress to checkpoint
        
        Args:
            dataset_name: Name of the dataset being evaluated
            total_questions: Total questions in dataset
            processed_questions: Number of questions processed so far
            current_top_k: Current top-k value
            results: List of detailed results
            correct_count: Number of correct answers
            total_count: Total answers evaluated
            elapsed_time: Elapsed time in seconds
            config: Configuration dictionary
            script_name: Optional script name for checkpoint file
            error_message: Optional error message if interrupted
        """
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
        
        # Create backup of existing checkpoint
        checkpoint_path = self._get_checkpoint_path(script_name)
        backup_path = self._get_backup_path(script_name)
        if checkpoint_path.exists():
            try:
                import shutil
                shutil.copy2(checkpoint_path, backup_path)
            except Exception:
                pass
        
        # Save new checkpoint
        try:
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(asdict(checkpoint), f, indent=2, ensure_ascii=False)
            
            # Print progress
            accuracy = correct_count / total_count if total_count > 0 else 0
            progress = processed_questions / total_questions if total_questions > 0 else 0
            
            print(
                f"\n💾 Progress saved: {processed_questions}/{total_questions} "
                f"({progress*100:.1f}%) | "
                f"Accuracy: {accuracy:.4f} | "
                f"Elapsed: {elapsed_time:.1f}s"
            )
            
        except Exception as e:
            print(f"Warning: Could not save checkpoint: {e}")
    
    def clear_checkpoint(self, script_name: Optional[str] = None):
        """Remove checkpoint file after successful completion"""
        checkpoint_path = self._get_checkpoint_path(script_name)
        backup_path = self._get_backup_path(script_name)
        
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            print(f"✓ Checkpoint cleared: {checkpoint_path}")
        
        if backup_path.exists():
            backup_path.unlink()
            print(f"✓ Backup cleared: {backup_path}")
    
    def should_resume(self, script_name: Optional[str] = None) -> bool:
        """
        Check if we should resume from checkpoint
        
        Returns:
            True if checkpoint exists and user wants to resume
        """
        if not self.has_checkpoint(script_name):
            return False
        
        checkpoint = self.load_checkpoint(script_name)
        if not checkpoint:
            return False
        
        # Check if already completed
        if checkpoint.processed_questions >= checkpoint.total_questions:
            print("\n✓ Evaluation already completed. Starting fresh...")
            self.clear_checkpoint(script_name)
            return False
        
        print(f"\n⚠️  Found interrupted evaluation: {checkpoint.processed_questions}/{checkpoint.total_questions}")
        
        # Auto-resume (can be configured to ask user)
        return True
    
    def get_resume_info(self, script_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get resume information from checkpoint
        
        Returns:
            Dictionary with resume info or None
        """
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
    
    def save_final_results(
        self,
        results: Dict[str, Any],
        filename: str,
        backup: bool = True,
    ):
        """
        Save final evaluation results
        
        Args:
            results: Final results dictionary
            filename: Output filename
            backup: Whether to backup existing file
        """
        output_path = self.output_dir / filename
        
        # Backup existing file
        if backup and output_path.exists():
            backup_path = self.output_dir / f"{filename}.backup"
            try:
                import shutil
                shutil.copy2(output_path, backup_path)
                print(f"✓ Backed up existing results to {backup_path}")
            except Exception as e:
                print(f"Warning: Could not backup results: {e}")
        
        # Save new results
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Results saved to {output_path}")


def create_progress_manager(output_dir: str = "./results/evaluation") -> EvaluationProgressManager:
    """Factory function to create progress manager"""
    return EvaluationProgressManager(output_dir)
