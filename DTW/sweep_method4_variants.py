"""
Run a small experiment sweep for Method 4 variants and save a summary table.

This is designed to help you find "best options" without editing existing research scripts.
Start with small limits, then remove limits for full runs.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Dict


def run_cmd(cmd: List[str]) -> int:
    print("\n$ " + " ".join(cmd))
    return subprocess.call(cmd)


def main() -> None:
    base = Path(__file__).resolve().parent
    py = base / "venv310" / "Scripts" / "python.exe"
    if not py.exists():
        py = Path(sys.executable)

    p = argparse.ArgumentParser(description="Sweep Method4 variants.")
    p.add_argument("--limit-folders", type=int, default=10)
    p.add_argument("--limit-users", type=int, default=1)
    p.add_argument("--window-ratios", type=str, default="0.2,0.25,0.3")
    p.add_argument("--features", type=str, default="angles15,bones_angles_len,bones_angles_len_d1")
    p.add_argument("--swap", type=str, default="global")
    args = p.parse_args()

    ratios = [float(x.strip()) for x in args.window_ratios.split(",") if x.strip()]
    feats = [x.strip() for x in args.features.split(",") if x.strip()]

    out_rows: List[Dict] = []
    os.chdir(str(base))

    for feat in feats:
        for r in ratios:
            cmd = [
                str(py),
                "evaluate_recognition_method4_variants.py",
                "--feature",
                feat,
                "--swap",
                args.swap,
                "--window-ratio",
                str(r),
                "--limit-folders",
                str(args.limit_folders),
                "--limit-users",
                str(args.limit_users),
            ]
            rc = run_cmd(cmd)
            if rc != 0:
                out_rows.append({"feature": feat, "window_ratio": r, "swap": args.swap, "error": rc})
                continue

            # Read the most recent summary json matching this tag.
            tag = f"{feat}_{args.swap}_win{r:.2f}".replace(".", "p")
            summary_path = base / "recognition_results" / f"recognition_method4_{tag}_summary.json"
            if summary_path.exists():
                data = json.loads(summary_path.read_text(encoding="utf-8"))
                s = data["summary"]
                out_rows.append({
                    "feature": s["feature"],
                    "swap": s["swap"],
                    "window_ratio": s["params"]["window_ratio"],
                    "top1": s["top1_accuracy"],
                    "top3": s["top3_accuracy"],
                    "top5": s["top5_accuracy"],
                    "total_users": s["total_users"],
                })
            else:
                out_rows.append({"feature": feat, "window_ratio": r, "swap": args.swap, "error": "missing_summary"})

    out_file = base / "recognition_results" / "sweep_method4_variants_summary.json"
    out_file.write_text(json.dumps(out_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote: {out_file}")


if __name__ == "__main__":
    main()

