"""
Run the full Telegram pipeline: Build CSV + Extract features + Train + Inference

Usage:
    python run_pipeline.py                          # Full pipeline
    python run_pipeline.py --skip-extraction        # Skip feature extraction
    python run_pipeline.py --extract-only           # Only extract features
"""
import os
import sys
import argparse
import subprocess

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def run_script(script_name, extra_args=None):
    """Run a Python script from this directory."""
    cmd = [sys.executable, os.path.join(SCRIPT_DIR, script_name)]
    if extra_args:
        cmd.extend(extra_args)
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, cwd=PARENT_DIR)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Telegram Full Pipeline (Videos/)")
    parser.add_argument("--skip-extraction", action="store_true",
                        help="Skip feature extraction (use existing features)")
    parser.add_argument("--extract-only", action="store_true",
                        help="Only extract features, don't train")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of sentence folders to process")
    args = parser.parse_args()

    print("=" * 60)
    print("Telegram Pipeline (Videos/)")
    print("=" * 60)

    # Stage 0: Build dataset CSV
    ret = run_script("build_dataset.py")
    if ret != 0:
        print("❌ Dataset CSV build failed!")
        return
    print("✅ Dataset CSV built!")

    # Stage 1: Feature extraction
    if not args.skip_extraction:
        extra = []
        if args.limit:
            extra.extend(["--limit", str(args.limit)])
        ret = run_script("extract_features.py", extra)
        if ret != 0:
            print("❌ Feature extraction failed!")
            return
        print("✅ Feature extraction complete!")

    if args.extract_only:
        return

    # Stage 2: Training
    ret = run_script("train.py", ["--epochs", str(args.epochs)])
    if ret != 0:
        print("❌ Training failed!")
        return
    print("✅ Training complete!")

    # Stage 3: Inference
    ret = run_script("inference.py")
    if ret != 0:
        print("❌ Inference failed!")
        return
    print("✅ Inference complete!")

    print("\n" + "=" * 60)
    print("Telegram Pipeline Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
