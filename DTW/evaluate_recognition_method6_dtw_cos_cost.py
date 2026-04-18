"""
Evaluate recognition accuracy using Method 6:
  DTW alignment with cosine-distance cost (1 - cosine_similarity) and score as mean cosine along the DTW path.

This is intended to reduce "unrelated frame matches" caused by the Method3 mismatch:
  DTW cost = Euclidean, score = cosine.
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
from typing import Dict, List, Sequence, Tuple

import numpy as np

# Fix Unicode output on Windows console
if getattr(sys.stdout, "encoding", "").lower() != "utf-8" and hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if getattr(sys.stderr, "encoding", "").lower() != "utf-8" and hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.append(os.path.dirname(__file__))

from compare_three_methods import extract_all_frames_and_filter  # noqa: E402
from method6_dtw_cos_cost import dtw_align_cosine_cost  # noqa: E402


VIDEOS_DIR = os.path.join(os.path.dirname(__file__), "Videos")
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache", "recognition_method6_dtw_cos_cost")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "recognition_results")


def discover_folders() -> Dict[str, Dict[str, List[str]]]:
    folders: Dict[str, Dict[str, List[str]]] = {}
    for d in sorted(os.listdir(VIDEOS_DIR)):
        full = os.path.join(VIDEOS_DIR, d)
        if not os.path.isdir(full):
            continue
        trans = sorted(glob.glob(os.path.join(full, "translator_*.mp4")))
        users = sorted(glob.glob(os.path.join(full, "user_*.mp4")))
        if trans and users:
            folders[d] = {"translator": trans, "user": users}
    return folders


def _safe_name(folder: str, max_len: int = 60) -> str:
    return folder.replace(" ", "_")[:max_len]


def precompute_translator_features(
    folders: Dict[str, Dict[str, List[str]]],
    similarity_threshold: float,
) -> Dict[str, Tuple[np.ndarray, List[int]]]:
    os.makedirs(CACHE_DIR, exist_ok=True)
    translator_data: Dict[str, Tuple[np.ndarray, List[int]]] = {}
    total = len(folders)

    for idx, (folder, info) in enumerate(folders.items(), 1):
        trans_path = info["translator"][0]
        safe = _safe_name(folder)
        feat_cache = os.path.join(CACHE_DIR, f"trans_{safe}_thr{similarity_threshold:.3f}.npz")

        if os.path.exists(feat_cache):
            data = np.load(feat_cache, allow_pickle=True)
            features = data["features"]
            indices = data["indices"].tolist()
            print(f"[{idx}/{total}] Loaded cached translator: {folder}  ({len(indices)} frames)")
        else:
            print(f"[{idx}/{total}] Extracting translator: {folder}")
            features, indices = extract_all_frames_and_filter(trans_path, similarity_threshold=similarity_threshold)
            if len(features) == 0:
                print("    WARNING: No features extracted, skipping")
                continue
            np.savez_compressed(feat_cache, features=features, indices=np.array(indices))

        translator_data[folder] = (features, indices)

    print(f"\nPre-computed features for {len(translator_data)} translator videos\n")
    return translator_data


def compare_user_vs_all_translators_m6(
    user_features: np.ndarray,
    user_indices: Sequence[int],
    translator_data: Dict[str, Tuple[np.ndarray, List[int]]],
    *,
    window_ratio: float,
) -> List[Tuple[str, float]]:
    results: List[Tuple[str, float]] = []
    for folder, (trans_feat, trans_idx) in translator_data.items():
        if len(trans_feat) == 0:
            continue
        r = dtw_align_cosine_cost(
            trans_feat,
            user_features,
            trans_idx,
            user_indices,
            window_ratio=window_ratio,
        )
        results.append((folder, float(r.score)))
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def run_evaluation(
    folders: Dict[str, Dict[str, List[str]]],
    translator_data: Dict[str, Tuple[np.ndarray, List[int]]],
    *,
    similarity_threshold: float,
    window_ratio: float,
    limit_users: int | None,
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
            user_basename = os.path.basename(user_path)
            print("=" * 70)
            print(f"[Folder {f_idx}/{total_folders}] [{total_users}] {folder}")
            print(f"  User video: {user_basename}")
            print("  Method: M6 (DTW cosine-cost)")

            t0 = time.time()
            user_feat, user_idx = extract_all_frames_and_filter(user_path, similarity_threshold=similarity_threshold)
            extract_time = time.time() - t0
            if len(user_feat) == 0:
                print("  WARNING: No features extracted, skipping")
                continue

            t1 = time.time()
            rankings = compare_user_vs_all_translators_m6(
                user_feat,
                user_idx,
                translator_data,
                window_ratio=window_ratio,
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
                "method": "m6_dtw_cos_cost",
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
        "method": "m6_dtw_cos_cost",
        "similarity_threshold": float(similarity_threshold),
        "window_ratio": float(window_ratio),
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
    os.makedirs(RESULTS_DIR, exist_ok=True)
    tag = "method6_dtw_cos_cost"

    csv_path = os.path.join(RESULTS_DIR, f"recognition_{tag}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
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
            row = {k: v for k, v in r.items() if k not in ("top5_rankings", "method")}
            writer.writerow(row)

    json_path = os.path.join(RESULTS_DIR, f"recognition_{tag}_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "detailed": all_results}, f, ensure_ascii=False, indent=2)

    txt_path = os.path.join(RESULTS_DIR, f"recognition_{tag}_report.txt")
    wrong = [r for r in all_results if not r.get("top1", False)]
    correct = [r for r in all_results if r.get("top1", False)]
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("SIGN LANGUAGE RECOGNITION EVALUATION — Method: M6 (DTW cosine-cost)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total user videos tested:    {summary['total_users']}\n")
        f.write(f"Total translator references: {summary['total_translators']}\n\n")
        f.write(f"Top-1 Accuracy: {summary['top1_correct']}/{summary['total_users']} = {summary['top1_accuracy']}%\n")
        f.write(f"Top-3 Accuracy: {summary['top3_correct']}/{summary['total_users']} = {summary['top3_accuracy']}%\n")
        f.write(f"Top-5 Accuracy: {summary['top5_correct']}/{summary['total_users']} = {summary['top5_accuracy']}%\n\n")

        if wrong:
            f.write("-" * 80 + "\n")
            f.write(f"MISCLASSIFICATIONS ({len(wrong)} cases)\n")
            f.write("-" * 80 + "\n\n")
            for r in wrong:
                f.write(f"  User: {r['user_video']}\n")
                f.write(f"  Correct folder: {r['folder']}\n")
                f.write(f"  Predicted folder: {r['top1_folder']} (score {r['top1_score']:.4f})\n")
                f.write(f"  Correct rank: {r['correct_rank']} (score {r['correct_score']:.4f})\n")
                f.write("  Top-5:\n")
                for i, (rf, rs) in enumerate(r.get("top5_rankings", []) or []):
                    marker = " <<< CORRECT" if rf == r["folder"] else ""
                    f.write(f"    {i+1}. {float(rs):.4f}  {rf}{marker}\n")
                f.write("\n")

        if correct:
            f.write("-" * 80 + "\n")
            f.write(f"CORRECT IDENTIFICATIONS ({len(correct)} cases)\n")
            f.write("-" * 80 + "\n\n")
            for r in correct:
                f.write(f"  {r['folder']} | {r['user_video']} | score={r['correct_score']:.4f}\n")

    print("\nResults saved to:")
    print(f"  CSV:    {csv_path}")
    print(f"  JSON:   {json_path}")
    print(f"  Report: {txt_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate recognition using Method 6 (DTW cosine-cost).")
    parser.add_argument("--limit-folders", type=int, default=None, help="Process only first N folders.")
    parser.add_argument("--limit-users", type=int, default=None, help="Process only first N user videos per folder.")
    parser.add_argument("--similarity-threshold", type=float, default=0.99, help="Drop-similar-frames threshold.")
    parser.add_argument("--window-ratio", type=float, default=0.3, help="DTW Sakoe-Chiba window ratio.")
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    print("Discovering video folders...")
    folders = discover_folders()
    print(f"Found {len(folders)} folders with translator+user videos\n")
    if args.limit_folders:
        folders = dict(list(folders.items())[: args.limit_folders])
        print(f"  (Limited to first {args.limit_folders} folders)\n")

    print("=" * 80)
    print("PHASE 1: Pre-computing translator features")
    print("=" * 80)
    translator_data = precompute_translator_features(folders, similarity_threshold=args.similarity_threshold)

    print("\n" + "=" * 80)
    print("PHASE 2: Evaluating recognition — Method: M6 (DTW cosine-cost)")
    print("=" * 80 + "\n")

    all_results, summary = run_evaluation(
        folders,
        translator_data,
        similarity_threshold=args.similarity_threshold,
        window_ratio=args.window_ratio,
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

