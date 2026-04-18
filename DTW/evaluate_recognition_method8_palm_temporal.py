"""
Evaluate Method 8: DTW with L1 cost on angles + relative lengths + palm normal + temporal d1.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import io
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

BASE_DIR    = Path(__file__).resolve().parent
VIDEOS_DIR  = BASE_DIR / "Videos"
RESULTS_DIR = BASE_DIR / "recognition_results"
CACHE_DIR   = BASE_DIR / "cache" / "recognition_method8"

sys.path.append(str(BASE_DIR))
from compare_three_methods import extract_all_frames_and_filter  # noqa: E402
from method8_palm_temporal import (                               # noqa: E402
    features_from_coords_matrix,
    dtw_global_align,
)


def discover_folders() -> Dict[str, Dict[str, List[Path]]]:
    folders: Dict[str, Dict[str, List[Path]]] = {}
    for d in sorted(VIDEOS_DIR.iterdir()):
        if not d.is_dir():
            continue
        trans = sorted(d.glob("translator_*.mp4"))
        users = sorted(d.glob("user_*.mp4"))
        if trans and users:
            folders[d.name] = {"translator": trans, "user": users}
    return folders


def _npz_paths(video_path: Path, thr: float) -> Tuple[Path, Path]:
    key = video_path.parent.name.replace(" ", "_") + "_" + video_path.name.replace(" ", "_")
    base = CACHE_DIR / f"{key}_thr{thr:.3f}"
    return base.with_suffix(".coords.npz"), base.with_suffix(".feat.npz")


def extract_features_with_cache(video_path: Path, thr: float) -> Tuple[np.ndarray, List[int]]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    coords_npz, feat_npz = _npz_paths(video_path, thr)
    if feat_npz.exists():
        data = np.load(feat_npz, allow_pickle=True)
        feats = data["features"].astype(np.float32, copy=False)
        idx = data["indices"].tolist()
        return feats, [int(x) for x in idx]

    coords, kept_idx = extract_all_frames_and_filter(str(video_path), similarity_threshold=thr)
    if coords.size == 0:
        return np.zeros((0, 1), dtype=np.float32), []

    feats = features_from_coords_matrix(coords.astype(np.float32, copy=False))
    np.savez_compressed(
        feat_npz,
        features=feats,
        indices=np.array(kept_idx, dtype=np.int32),
    )
    return feats, kept_idx


def precompute_translators(
    folders: Dict[str, Dict[str, List[Path]]], thr: float
) -> Dict[str, Tuple[np.ndarray, List[int]]]:
    out: Dict[str, Tuple[np.ndarray, List[int]]] = {}
    total = len(folders)
    for i, (folder, info) in enumerate(folders.items(), start=1):
        trans_path = info["translator"][0]
        print(f"[{i}/{total}] Translator: {folder}")
        feats, idx = extract_features_with_cache(trans_path, thr)
        out[folder] = (feats, idx)
    return out


def compare_user_vs_all(
    user_feats: np.ndarray,
    user_idx: List[int],
    translator_data: Dict[str, Tuple[np.ndarray, List[int]]],
    window_ratio: float,
    swap: str,
) -> List[Tuple[str, float]]:
    results: List[Tuple[str, float]] = []
    for folder, (t_feats, t_idx) in translator_data.items():
        if t_feats.size == 0:
            continue
        r = dtw_global_align(
            t_feats,
            user_feats,
            t_idx,
            user_idx,
            window_ratio=window_ratio,
            swap_mode=swap,
        )
        results.append((folder, float(r.score)))
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def run_eval(
    folders: Dict[str, Dict[str, List[Path]]],
    translator_data: Dict[str, Tuple[np.ndarray, List[int]]],
    thr: float,
    window_ratio: float,
    limit_folders: int | None,
    limit_users: int | None,
    swap: str,
) -> Tuple[List[Dict], Dict]:
    all_results: List[Dict] = []
    top1_correct = 0
    top3_correct = 0
    top5_correct = 0
    total_users = 0

    folder_list = list(folders.items())
    total_folders = len(folder_list)

    for f_idx, (folder, info) in enumerate(folder_list, 1):
        user_videos = info["user"]
        if limit_users:
            user_videos = user_videos[:limit_users]

        for user_path in user_videos:
            total_users += 1
            user_basename = user_path.name
            print("=" * 70)
            print(f"[Folder {f_idx}/{total_folders}] [{total_users}] {folder}")
            print(f"  User video: {user_basename}")
            print("  Method: M8 (angles + rel_len + palm normal + d1, L1 DTW)")

            t0 = time.time()
            user_feat, user_idx = extract_features_with_cache(user_path, thr)
            extract_time = time.time() - t0
            if user_feat.size == 0:
                print("  WARNING: No features extracted, skipping")
                continue

            t1 = time.time()
            rankings = compare_user_vs_all(
                user_feat, user_idx, translator_data,
                window_ratio=window_ratio, swap=swap,
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

            if is_top1: top1_correct += 1
            if is_top3: top3_correct += 1
            if is_top5: top5_correct += 1

            result = {
                "method": "method8_palm_temporal",
                "folder": folder,
                "user_video": user_basename,
                "correct_rank": rank,
                "top1": is_top1,
                "top3": is_top3,
                "top5": is_top5,
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
            for i, (rf, rs) in enumerate(result["top5_rankings"]):
                marker = " <<< CORRECT" if rf == folder else ""
                print(f"    {i+1}. {rs:.4f}  {rf}{marker}")
            print(f"  Time: extract={extract_time:.1f}s, compare={compare_time:.1f}s")
            print()

    summary = {
        "method": "method8_palm_temporal",
        "params": {"thr": thr, "window_ratio": window_ratio, "swap": swap},
        "total_users": int(total_users),
        "total_translators": int(len(translator_data)),
        "top1_correct": int(top1_correct),
        "top3_correct": int(top3_correct),
        "top5_correct": int(top5_correct),
        "top1_accuracy": round(top1_correct / total_users * 100, 2) if total_users > 0 else 0,
        "top3_accuracy": round(top3_correct / total_users * 100, 2) if total_users > 0 else 0,
        "top5_accuracy": round(top5_correct / total_users * 100, 2) if total_users > 0 else 0,
    }
    return all_results, summary


def save_results(all_results: List[Dict], summary: Dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = "method8_palm_temporal"

    csv_path = RESULTS_DIR / f"recognition_{tag}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "folder", "user_video", "correct_rank",
                "top1", "top3", "top5",
                "correct_score", "top1_folder", "top1_score",
                "user_frames", "extract_time", "compare_time",
            ],
        )
        writer.writeheader()
        for r in all_results:
            row = {k: v for k, v in r.items() if k not in ("top5_rankings", "method")}
            writer.writerow(row)

    json_path = RESULTS_DIR / f"recognition_{tag}_summary.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "detailed": all_results}, f, ensure_ascii=False, indent=2)

    txt_path = RESULTS_DIR / f"recognition_{tag}_report.txt"
    with txt_path.open("w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("SIGN LANGUAGE RECOGNITION — Method 8 (angles + rel_len + palm + d1, L1 DTW)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total user videos tested:    {summary['total_users']}\n")
        f.write(f"Total translator references: {summary['total_translators']}\n\n")
        f.write(f"Top-1 Accuracy: {summary['top1_correct']}/{summary['total_users']} = {summary['top1_accuracy']}%\n")
        f.write(f"Top-3 Accuracy: {summary['top3_correct']}/{summary['total_users']} = {summary['top3_accuracy']}%\n")
        f.write(f"Top-5 Accuracy: {summary['top5_correct']}/{summary['total_users']} = {summary['top5_accuracy']}%\n\n")
        f.write(f"Feature dim: 852 (426 static + 426 d1)\n")
        f.write(f"  Static: 190 pairwise bone angles + 20 rel lengths + 3 palm normal, per hand\n")
        f.write(f"Params: thr={summary['params']['thr']} window_ratio={summary['params']['window_ratio']} swap={summary['params']['swap']}\n\n")
        f.write("Baseline comparison:\n")
        f.write("  Method 7 (angles + rel_len only):  Top-1=40.00%  Top-3=50.23%  Top-5=56.28%\n")
        f.write(f"  Method 8 (+ palm + d1):            Top-1={summary['top1_accuracy']}%  Top-3={summary['top3_accuracy']}%  Top-5={summary['top5_accuracy']}%\n")

    print("\nResults saved to:")
    print(f"  CSV:    {csv_path}")
    print(f"  JSON:   {json_path}")
    print(f"  Report: {txt_path}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Evaluate Method 8 (angles + rel_len + palm normal + temporal d1, L1 DTW)."
    )
    p.add_argument("--window-ratio", type=float, default=0.25)
    p.add_argument("--similarity-threshold", type=float, default=0.99)
    p.add_argument("--limit-folders", type=int, default=None)
    p.add_argument("--limit-users", type=int, default=None)
    p.add_argument("--swap", choices=["none", "global"], default="global")
    args = p.parse_args()

    os.chdir(str(BASE_DIR))
    folders = discover_folders()
    print(f"Found {len(folders)} folders.")
    if args.limit_folders is not None:
        folders = dict(list(folders.items())[: args.limit_folders])
        print(f"(Limited to first {args.limit_folders} folders.)")

    print("=" * 80)
    print("PHASE 1: Precomputing translator features (Method 8, cached)")
    print("=" * 80)
    translator_data = precompute_translators(folders, args.similarity_threshold)

    print("\n" + "=" * 80)
    print("PHASE 2: Evaluating all user videos")
    print("=" * 80 + "\n")
    all_results, summary = run_eval(
        folders,
        translator_data,
        thr=args.similarity_threshold,
        window_ratio=args.window_ratio,
        limit_folders=args.limit_folders,
        limit_users=args.limit_users,
        swap=args.swap,
    )
    save_results(all_results, summary)

    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"Total users:        {summary['total_users']}")
    print(f"Total translators:  {summary['total_translators']}")
    print(f"Top-1 Accuracy:     {summary['top1_correct']}/{summary['total_users']} = {summary['top1_accuracy']}%")
    print(f"Top-3 Accuracy:     {summary['top3_correct']}/{summary['total_users']} = {summary['top3_accuracy']}%")
    print(f"Top-5 Accuracy:     {summary['top5_correct']}/{summary['total_users']} = {summary['top5_accuracy']}%")
    print(f"\nBaseline (Method 7): Top-1=40.00%  Top-3=50.23%  Top-5=56.28%")


if __name__ == "__main__":
    main()
