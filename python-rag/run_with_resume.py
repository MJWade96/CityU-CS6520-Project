"""
Smart Runner for Medical RAG Evaluation Scripts

Features:
- Auto-detect interrupted evaluations
- Auto-resume from last checkpoint
- Handle multiple evaluation scripts
- Clean up checkpoints after successful completion

Usage:
    python run_with_resume.py complete_eval
    python run_with_resume.py enhanced_eval
    python run_with_resume.py evaluate_no_rag
    python run_with_resume.py --auto  # Auto-detect and run
"""

import os
import sys
import json
import asyncio
import argparse
import inspect
import traceback
from pathlib import Path
from typing import Optional, Dict, Any, List


def get_script_path(script_name: str) -> Path:
    """Get the full path of evaluation script"""
    script_dir = Path(__file__).parent
    return script_dir / f"{script_name}.py"


def get_checkpoint_script_names(script_name: str) -> List[str]:
    """Return current and legacy checkpoint prefixes for a script."""
    aliases = {
        "complete_eval": ["complete_eval_test", "complete_eval_dev", "complete_eval"],
        "enhanced_eval": ["enhanced_eval_test", "enhanced_eval_dev", "enhanced_eval"],
        "evaluate_no_rag": ["evaluate_no_rag", "no_rag_eval"],
    }
    return aliases.get(script_name, [script_name])


def sanitize_runtime_env() -> None:
    """Remove invalid OpenMP settings that can crash or spam warnings on Linux."""
    raw_value = os.environ.get("OMP_NUM_THREADS")
    if raw_value is None:
        return

    try:
        if int(str(raw_value).strip()) >= 1:
            return
    except (TypeError, ValueError):
        pass

    print(
        f"⚠️  Ignoring invalid OMP_NUM_THREADS={raw_value!r}; "
        "using library default instead."
    )
    os.environ.pop("OMP_NUM_THREADS", None)


def check_checkpoint_status(output_dir: str, script_name: str) -> Dict[str, Any]:
    """
    Check if there's an interrupted evaluation
    
    Returns:
        Dictionary with status information
    """
    output_path = Path(output_dir)

    checkpoint_files = []
    for checkpoint_script in get_checkpoint_script_names(script_name):
        checkpoint_files.extend(output_path.glob(f"checkpoint_{checkpoint_script}.json"))
        checkpoint_files.extend(
            output_path.glob(f"checkpoint_{checkpoint_script}.backup.json")
        )

    if not checkpoint_files:
        return {
            "has_checkpoint": False,
            "message": "No interrupted evaluation found"
        }

    latest_checkpoint = None
    latest_data = None
    latest_time = None

    for cp_file in checkpoint_files:
        if cp_file.exists():
            try:
                with open(cp_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                processed = data.get("processed_questions", 0)
                total = data.get("total_questions", 0)
                if total and processed >= total:
                    continue

                timestamp = data.get("timestamp", "")
                if latest_time is None or timestamp > latest_time:
                    latest_time = timestamp
                    latest_checkpoint = cp_file
                    latest_data = data
            except Exception:
                continue

    if latest_checkpoint and latest_data:
        return {
            "has_checkpoint": True,
            "checkpoint_file": str(latest_checkpoint),
            "timestamp": latest_data.get("timestamp", "unknown"),
            "dataset": latest_data.get("dataset_name", "unknown"),
            "progress": f"{latest_data.get('processed_questions', 0)}/{latest_data.get('total_questions', 0)}",
            "accuracy": f"{latest_data.get('correct_count', 0)}/{latest_data.get('total_count', 0)}",
            "script": latest_data.get("script_name", "unknown"),
            "error": latest_data.get("error_message"),
        }

    return {
        "has_checkpoint": False,
        "message": "No resumable checkpoint found"
    }


def run_script(script_name: str, auto_resume: bool = True):
    """
    Run evaluation script with resume support
    
    Args:
        script_name: Name of the script (without .py extension)
        auto_resume: Whether to automatically resume from checkpoint
    """
    script_path = get_script_path(script_name)
    
    if not script_path.exists():
        print(f"❌ ERROR: Script not found: {script_path}")
        print("\nAvailable scripts:")
        print("  - complete_eval")
        print("  - enhanced_eval")
        print("  - evaluate_no_rag")
        return False
    
    print("=" * 60)
    print(f"Smart Runner for Medical RAG Evaluation")
    print("=" * 60)
    print(f"\n📜 Script: {script_path.name}")
    print(f"📂 Location: {script_path.parent}")
    
    # Determine output directory based on script
    if "complete" in script_name:
        output_dir = script_path.parent / "results" / "evaluation"
    elif "enhanced" in script_name:
        output_dir = script_path.parent / "results" / "evaluation"
    else:
        output_dir = script_path.parent / "results" / "evaluation"
    
    # Check for interrupted evaluations
    if auto_resume:
        status = check_checkpoint_status(str(output_dir), script_name)
        
        if status.get("has_checkpoint"):
            print(f"\n⚠️  Found interrupted evaluation:")
            print(f"  📅 Timestamp: {status.get('timestamp', 'unknown')}")
            print(f"  📊 Dataset: {status.get('dataset', 'unknown')}")
            print(f"  📈 Progress: {status.get('progress', 'unknown')}")
            print(f"  ✅ Accuracy: {status.get('accuracy', 'unknown')}")
            
            if status.get("error"):
                print(f"  ❌ Last error: {status.get('error')}")
            
            print(f"\n💡 The script will automatically resume from the last checkpoint")
        else:
            print(f"\n✓ {status.get('message', 'No checkpoint found')}")
            print("  Starting fresh evaluation...")
    
    print(f"\n{'=' * 60}")
    print("Starting Evaluation...")
    print(f"{'=' * 60}\n")

    sanitize_runtime_env()

    # Import and run the script
    script_dir = str(script_path.parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    
    try:
        # Import the module using importlib for better error handling
        import importlib.util
        spec = importlib.util.spec_from_file_location(script_name, str(script_path))
        if spec is None or spec.loader is None:
            print(f"❌ ERROR: Could not load module spec for {script_name}")
            return False
            
        module = importlib.util.module_from_spec(spec)
        sys.modules[script_name] = module
        spec.loader.exec_module(module)
        
        # Check if module has main function
        if hasattr(module, "main"):
            original_argv = sys.argv[:]
            try:
                sys.argv = [str(script_path)]
                result = module.main()
                if inspect.isawaitable(result):
                    asyncio.run(result)
            finally:
                sys.argv = original_argv
            print(f"\n{'=' * 60}")
            print("✅ Evaluation completed successfully!")
            print(f"{'=' * 60}")
            return True
        else:
            print(f"❌ ERROR: No main() function found in {script_name}")
            return False
            
    except KeyboardInterrupt:
        print(f"\n\n{'=' * 60}")
        print("⚠️  Evaluation interrupted by user")
        print(f"{'=' * 60}")
        print("\n💡 Progress has been saved. Run again to resume:")
        print(f"   python run_with_resume.py {script_name}")
        return False
        
    except Exception as e:
        print(f"\n\n{'=' * 60}")
        print(f"❌ ERROR: Evaluation failed")
        print(f"{'=' * 60}")
        print(f"\nError type: {type(e).__name__}")
        print(f"Error details: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        print(f"\n💡 If progress was saved, you can resume by running again:")
        print(f"   python run_with_resume.py {script_name}")
        return False


def auto_detect_and_run():
    """Auto-detect interrupted evaluations and run them"""
    print("=" * 60)
    print("Auto-Detect Mode")
    print("=" * 60)
    
    # List of evaluation scripts to check
    scripts = [
        "complete_eval",
        "enhanced_eval",
        "evaluate_no_rag",
    ]
    
    output_dir = Path(__file__).parent / "results" / "evaluation"
    
    interrupted_scripts = []
    
    for script in scripts:
        status = check_checkpoint_status(str(output_dir), script)
        if status.get("has_checkpoint"):
            interrupted_scripts.append({
                "script": script,
                "status": status
            })
    
    if not interrupted_scripts:
        print("\n✓ No interrupted evaluations found")
        print("\nAvailable scripts:")
        for script in scripts:
            print(f"  - {script}")
        print("\nRun with: python run_with_resume.py <script_name>")
        return
    
    print(f"\n📋 Found {len(interrupted_scripts)} interrupted evaluation(s):\n")
    
    for i, item in enumerate(interrupted_scripts, 1):
        script = item["script"]
        status = item["status"]
        print(f"{i}. {script}")
        print(f"   Dataset: {status.get('dataset', 'unknown')}")
        print(f"   Progress: {status.get('progress', 'unknown')}")
        print(f"   Timestamp: {status.get('timestamp', 'unknown')}")
        print()
    
    # Ask user which one to resume
    print("Which evaluation to resume?")
    print(f"Enter number (1-{len(interrupted_scripts)}) or 'all' to resume all:")
    
    try:
        choice = input("> ").strip()
        
        if choice.lower() == "all":
            for item in interrupted_scripts:
                print(f"\n{'=' * 60}")
                print(f"Resuming: {item['script']}")
                print(f"{'=' * 60}")
                run_script(item["script"], auto_resume=True)
        elif choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(interrupted_scripts):
                run_script(interrupted_scripts[idx]["script"], auto_resume=True)
            else:
                print("❌ Invalid choice")
        else:
            print("❌ Invalid input")
            
    except Exception as e:
        print(f"❌ Error: {e}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Smart runner for Medical RAG evaluation scripts with resume support"
    )
    parser.add_argument(
        "script",
        nargs="?",
        default=None,
        help="Script name to run (e.g., complete_eval, enhanced_eval)"
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Auto-detect and resume interrupted evaluations"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable auto-resume (start fresh)"
    )
    
    args = parser.parse_args()
    
    if args.auto:
        auto_detect_and_run()
    elif args.script:
        auto_resume = not args.no_resume
        success = run_script(args.script, auto_resume=auto_resume)
        sys.exit(0 if success else 1)
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python run_with_resume.py complete_eval")
        print("  python run_with_resume.py enhanced_eval")
        print("  python run_with_resume.py evaluate_no_rag")
        print("  python run_with_resume.py --auto")
        print("  python run_with_resume.py complete_eval --no-resume")


if __name__ == "__main__":
    main()
