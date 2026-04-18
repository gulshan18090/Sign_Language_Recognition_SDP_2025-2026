"""
Evaluate recognition accuracy for Method 4 (angle degrees + hand-swap invariance).

This script is research-only and does NOT modify existing evaluation scripts.

Method 4:
  - Extract coordinate features using compare_three_methods.extract_all_frames_and_filter
  - Convert per-frame coords -> joint angles (degrees), normalized to [0..1]
  - Global DTW with Sakoe-Chiba band + swap-invariant cost and scoring

Outputs (written to recognition_results/):
  - recognition_method4_angles_swap.csv
  - recognition_method4_angles_swap_summary.json
  - recognition_method4_angles_swap_report.txt
"""

from __future__ import annotations

import argparse
import csv
import glob
import io
import json
import os
import sys
import time
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

# Fix Unicode output on Windows console
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent
VIDEOS_DIR = BASE_DIR / "Videos"
CACHE_DIR = BASE_DIR / "cache" / "recognition_method4_angles_swap"
RESULTS_DIR = BASE_DIR / "recognition_results"

sys.path.append(str(BASE_DIR))
from compare_three_methods import extract_all_frames_and_filter  # noqa: E402
from compare_four_methods import angles_matrix_from_coords, dtw_align_angles_with_hand_swap  # noqa: E402


def safe_name(text: str, max_len: int = 60) -> str:
    s = unicodedata.normalize("NFKD", str(text)).strip().replace(" ", "_")
    return s[:max_len]


def discover_folders() -> Dict[str, Dict[str, List[Path]]]:
    """Folders in Videos/ that have both translator and user videos."""
    folders: Dict[str, Dict[str, List[Path]]] = {}
    if not VIDEOS_DIR.exists():
        return folders

    for d in sorted(VIDEOS_DIR.iterdir()):
        if not d.is_dir():
            continue
        trans = sorted(d.glob("translator_*.mp4"))
        users = sorted(d.glob("user_*.mp4"))
        if trans and users:
            folders[d.name] = {"translator": trans, "user": users}
    return folders


def _load_npz(path: Path) -> Tuple[np.ndarray, List[int]]:
    data = np.load(path, allow_pickle=True)
    feats = data["features"].astype(np.float32, copy=False)
    idx = data["indices"].tolist()
    return feats, [int(x) for x in idx]


def _save_npz(path: Path, features: np.ndarray, indices: List[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, features=features.astype(np.float32, copy=False), indices=np.array(indices, dtype=np.int32))


def extract_angles_with_cache(video_path: Path, similarity_threshold: float) -> Tuple[np.ndarray, List[int]]:
    """Return (angles_matrix, kept_indices) for a video. Cached on disk."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = safe_name(video_path.parent.name) + "_" + safe_name(video_path.name)
    cache_path = CACHE_DIR / f"{key}_thr{similarity_threshold:.3f}.npz"
    if cache_path.exists():
        return _load_npz(cache_path)

    coords, kept_idx = extract_all_frames_and_filter(str(video_path), similarity_threshold=similarity_threshold)
    if coords.size == 0:
        return np.zeros((0, 30), dtype=np.float32), []
    angles = angles_matrix_from_coords(coords)  # (N,30)
    _save_npz(cache_path, angles, kept_idx)
    return angles, kept_idx


def compare_user_vs_all_translators_method4(
    user_angles: np.ndarray,
    user_idx: List[int],
    translator_data: Dict[str, Tuple[np.ndarray, List[int]]],
    window_ratio: float,
) -> List[Tuple[str, float]]:
    results: List[Tuple[str, float]] = []
    for folder, (t_angles, t_idx) in translator_data.items():
        if t_angles.size == 0:
            continue
        res = dtw_align_angles_with_hand_swap(
            t_angles, user_angles, t_idx, user_idx, window_ratio=window_ratio
        )
        results.append((folder, float(res.score)))
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def precompute_translator_angles(
    folders: Dict[str, Dict[str, List[Path]]],
    similarity_threshold: float,
) -> Dict[str, Tuple[np.ndarray, List[int]]]:
    translator_data: Dict[str, Tuple[np.ndarray, List[int]]] = {}
    total = len(folders)
    for i, (folder, info) in enumerate(folders.items(), start=1):
        trans_path = info["translator"][0]
        print(f"[{i}/{total}] Translator: {folder}")
        angles, idx = extract_angles_with_cache(trans_path, similarity_threshold)
        translator_data[folder] = (angles, idx)
    print(f"\nLoaded translator angle features for {len(translator_data)} folders.\n")
    return translator_data


def run_evaluation(
    folders: Dict[str, Dict[str, List[Path]]],
    translator_data: Dict[str, Tuple[np.ndarray, List[int]]],
    similarity_threshold: float,
    window_ratio: float,
    limit_folders: Optional[int],
    limit_users: Optional[int],
) -> Tuple[List[Dict], Dict]:
    all_results: List[Dict] = []
    top1_correct = 0
    top3_correct = 0
    top5_correct = 0
    total_users = 0

    folder_items = list(folders.items())
    if limit_folders is not None:
        folder_items = folder_items[:limit_folders]

    for f_i, (folder, info) in enumerate(folder_items, start=1):
        user_videos = info["user"]
        if limit_users is not None:
            user_videos = user_videos[:limit_users]

        for user_path in user_videos:
            total_users += 1
            user_name = user_path.name
            print("=" * 70)
            print(f"[Folder {f_i}/{len(folder_items)}] [{total_users}] {folder}")
            print(f"  User video: {user_name}")
            print("  Method: METHOD4_ANGLES_SWAP")

            t0 = time.time()
            user_angles, user_idx = extract_angles_with_cache(user_path, similarity_threshold)
            extract_time = time.time() - t0

            if user_angles.size == 0:
                print("  WARNING: No features extracted, skipping")
                continue

            t1 = time.time()
            rankings = compare_user_vs_all_translators_method4(
                user_angles, user_idx, translator_data, window_ratio=window_ratio
            )
            compare_time = time.time() - t1

            if not rankings:
                print("  WARNING: No rankings produced, skipping")
                continue

            top_folders = [r[0] for r in rankings]
            rank = top_folders.index(folder) + 1 if folder in top_folders else len(top_folders) + 1

            is_top1 = rank == 1
            is_top3 = rank <= 3
            is_top5 = rank <= 5
            if is_top1:
                top1_correct += 1
            if is_top3:
                top3_correct += 1
            if is_top5:
                top5_correct += 1

            result = {
                "folder": folder,
                "user_video": user_name,
                "correct_rank": int(rank),
                "top1": bool(is_top1),
                "top3": bool(is_top3),
                "top5": bool(is_top5),
                "correct_score": float(rankings[rank - 1][1]) if rank <= len(rankings) else 0.0,
                "top1_folder": rankings[0][0],
                "top1_score": float(rankings[0][1]),
                "top5_rankings": [(r[0], round(float(r[1]), 4)) for r in rankings[:5]],
                "extract_time": round(extract_time, 2),
                "compare_time": round(compare_time, 2),
                "user_frames": int(len(user_idx)),
            }
            all_results.append(result)

            status = "CORRECT" if is_top1 else f"WRONG (rank {rank})"
            print(f"\n  Result: {status}")
            print(f"  Correct score: {result['correct_score']:.4f} | Top-1 score: {result['top1_score']:.4f}")
            print("  Top-5 rankings:")
            for k, (rf, rs) in enumerate(result["top5_rankings"], start=1):
                marker = " <<< CORRECT" if rf == folder else ""
                print(f"    {k}. {rs:.4f}  {rf}{marker}")
            print(f"  Time: extract={extract_time:.1f}s, compare={compare_time:.1f}s\n")

    summary = {
        "method": "method4_angles_swap",
        "total_users": int(total_users),
        "total_translators": int(len(translator_data)),
        "top1_correct": int(top1_correct),
        "top3_correct": int(top3_correct),
        "top5_correct": int(top5_correct),
        "top1_accuracy": round(top1_correct / total_users * 100, 2) if total_users else 0.0,
        "top3_accuracy": round(top3_correct / total_users * 100, 2) if total_users else 0.0,
        "top5_accuracy": round(top5_correct / total_users * 100, 2) if total_users else 0.0,
        "params": {
            "similarity_threshold": float(similarity_threshold),
            "window_ratio": float(window_ratio),
        },
    }
    return all_results, summary


def save_results(all_results: List[Dict], summary: Dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = RESULTS_DIR / "recognition_method4_angles_swap.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "folder",
                "user_video",
                "correct_rank",
                "top1",
                "top3",
                "top5",
                "correct_score",
                "top1_folder",
                "top1_score",
                "user_frames",
                "extract_time",
                "compare_time",
            ],
        )
        writer.writeheader()
        for r in all_results:
            row = {k: v for k, v in r.items() if k != "top5_rankings"}
            writer.writerow(row)

    json_path = RESULTS_DIR / "recognition_method4_angles_swap_summary.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "detailed": all_results}, f, ensure_ascii=False, indent=2)

    txt_path = RESULTS_DIR / "recognition_method4_angles_swap_report.txt"
    with txt_path.open("w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("SIGN LANGUAGE RECOGNITION EVALUATION — Method: METHOD4_ANGLES_SWAP\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total user videos tested:    {summary['total_users']}\n")
        f.write(f"Total translator references: {summary['total_translators']}\n\n")
        f.write(f"Top-1 Accuracy: {summary['top1_correct']}/{summary['total_users']} = {summary['top1_accuracy']}%\n")
        f.write(f"Top-3 Accuracy: {summary['top3_correct']}/{summary['total_users']} = {summary['top3_accuracy']}%\n")
        f.write(f"Top-5 Accuracy: {summary['top5_correct']}/{summary['total_users']} = {summary['top5_accuracy']}%\n\n")
        f.write(f"Params: similarity_threshold={summary['params']['similarity_threshold']}  window_ratio={summary['params']['window_ratio']}\n\n")

        wrong = [r for r in all_results if not r["top1"]]
        f.write("-" * 80 + "\n")
        f.write(f"MISCLASSIFICATIONS ({len(wrong)} cases)\n")
        f.write("-" * 80 + "\n\n")
        for r in wrong[:200]:
            f.write(f"User: {r['user_video']}\n")
            f.write(f"Correct folder: {r['folder']}\n")
            f.write(f"Predicted: {r['top1_folder']} (score {r['top1_score']:.4f})\n")
            f.write(f"Correct rank: {r['correct_rank']} (score {r['correct_score']:.4f})\n")
            f.write("Top-5:\n")
            for i, (rf, rs) in enumerate(r["top5_rankings"], start=1):
                marker = " <<< CORRECT" if rf == r["folder"] else ""
                f.write(f"  {i}. {rs:.4f}  {rf}{marker}\n")
            f.write("\n")

    print("\nResults saved to:")
    print(f"  CSV:    {csv_path}")
    print(f"  JSON:   {json_path}")
    print(f"  Report: {txt_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate recognition for Method 4 (angles + swap invariance).")
    p.add_argument("--limit-folders", type=int, default=None, help="Process only first N folders.")
    p.add_argument("--limit-users", type=int, default=None, help="Process only first N user videos per folder.")
    p.add_argument("--window-ratio", type=float, default=0.3, help="DTW window ratio (default 0.3).")
    p.add_argument("--similarity-threshold", type=float, default=0.99, help="Drop redundant frames threshold.")
    args = p.parse_args()

    os.chdir(str(BASE_DIR))
    print("Discovering video folders...")
    folders = discover_folders()
    print(f"Found {len(folders)} folders with translator+user videos\n")

    if args.limit_folders is not None:
        folders = dict(list(folders.items())[: args.limit_folders])
        print(f"(Limited to first {args.limit_folders} folders)\n")

    print("=" * 80)
    print("PHASE 1: Pre-computing translator angle features (cached)")
    print("=" * 80)
    translator_data = precompute_translator_angles(folders, similarity_threshold=args.similarity_threshold)

    print("\n" + "=" * 80)
    print("PHASE 2: Evaluating recognition — Method: METHOD4_ANGLES_SWAP")
    print("=" * 80 + "\n")

    all_results, summary = run_evaluation(
        folders,
        translator_data,
        similarity_threshold=args.similarity_threshold,
        window_ratio=args.window_ratio,
        limit_folders=args.limit_folders,
        limit_users=args.limit_users,
    )
    save_results(all_results, summary)

    print("\n" + "=" * 80)
    print("FINAL RECOGNITION RESULTS")
    print("=" * 80)
    print(f"  Method:             {summary['method']}")
    print(f"  User videos tested: {summary['total_users']}")
    print(f"  Translator refs:    {summary['total_translators']}")
    print()
    print(f"  Top-1 Accuracy: {summary['top1_correct']}/{summary['total_users']} = {summary['top1_accuracy']}%")
    print(f"  Top-3 Accuracy: {summary['top3_correct']}/{summary['total_users']} = {summary['top3_accuracy']}%")
    print(f"  Top-5 Accuracy: {summary['top5_correct']}/{summary['total_users']} = {summary['top5_accuracy']}%")
    print("=" * 80)


if __name__ == "__main__":
    main()

