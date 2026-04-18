"""
Evaluate recognition accuracy for Method 4 variants (research script).

This does not edit existing evaluation scripts; it is additive.

Variants:
- feature kinds: angles15, bones_angles, bones_angles_len, bones_angles_len_d1
- swap invariance: none, global (choose best between original and swapped for whole sequence)
- DTW: global end-to-end DTW (same family as Method3 but on variant features)

Outputs written to recognition_results/ as CSV + JSON summary + text report.
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
from typing import Dict, List, Optional, Tuple

import numpy as np

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent
VIDEOS_DIR = BASE_DIR / "Videos"
RESULTS_DIR = BASE_DIR / "recognition_results"
CACHE_DIR = BASE_DIR / "cache" / "recognition_method4_variants"

sys.path.append(str(BASE_DIR))
from compare_three_methods import extract_all_frames_and_filter  # noqa: E402
from method4_variants import (  # noqa: E402
    FeatureKind,
    SwapMode,
    AlignResult,
    features_from_coords_matrix,
    dtw_global_align,
)


def safe_name(text: str, max_len: int = 60) -> str:
    s = unicodedata.normalize("NFKD", str(text)).strip().replace(" ", "_")
    return s[:max_len]


def discover_folders() -> Dict[str, Dict[str, List[Path]]]:
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


def _npz_paths(video_path: Path, feature: str, thr: float) -> Tuple[Path, Path]:
    key = safe_name(video_path.parent.name) + "_" + safe_name(video_path.name)
    base = CACHE_DIR / f"{key}_{feature}_thr{thr:.3f}"
    return base.with_suffix(".coords.npz"), base.with_suffix(".feat.npz")


def extract_variant_features_with_cache(video_path: Path, feature: FeatureKind, thr: float) -> Tuple[np.ndarray, List[int]]:
    """Return (feature_matrix, kept_indices). Cached on disk per video+feature+thr."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    coords_npz, feat_npz = _npz_paths(video_path, str(feature), thr)
    if feat_npz.exists():
        data = np.load(feat_npz, allow_pickle=True)
        feats = data["features"].astype(np.float32, copy=False)
        idx = data["indices"].tolist()
        return feats, [int(x) for x in idx]

    coords, kept_idx = extract_all_frames_and_filter(str(video_path), similarity_threshold=thr)
    if coords.size == 0:
        return np.zeros((0, 1), dtype=np.float32), []

    feats = features_from_coords_matrix(coords.astype(np.float32, copy=False), feature)
    np.savez_compressed(feat_npz, features=feats.astype(np.float32, copy=False), indices=np.array(kept_idx, dtype=np.int32))
    return feats, kept_idx


def precompute_translators(
    folders: Dict[str, Dict[str, List[Path]]],
    feature: FeatureKind,
    thr: float,
) -> Dict[str, Tuple[np.ndarray, List[int]]]:
    out: Dict[str, Tuple[np.ndarray, List[int]]] = {}
    total = len(folders)
    for i, (folder, info) in enumerate(folders.items(), start=1):
        trans_path = info["translator"][0]
        print(f"[{i}/{total}] Translator: {folder}")
        feats, idx = extract_variant_features_with_cache(trans_path, feature, thr)
        out[folder] = (feats, idx)
    return out


def compare_user_vs_all(
    user_feats: np.ndarray,
    user_idx: List[int],
    translator_data: Dict[str, Tuple[np.ndarray, List[int]]],
    feature: FeatureKind,
    swap: SwapMode,
    window_ratio: float,
) -> List[Tuple[str, float]]:
    results: List[Tuple[str, float]] = []
    for folder, (t_feats, t_idx) in translator_data.items():
        if t_feats.size == 0:
            continue
        res: AlignResult = dtw_global_align(t_feats, user_feats, t_idx, user_idx, window_ratio, feature, swap)
        results.append((folder, float(res.score)))
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def run_eval(
    folders: Dict[str, Dict[str, List[Path]]],
    translator_data: Dict[str, Tuple[np.ndarray, List[int]]],
    feature: FeatureKind,
    swap: SwapMode,
    thr: float,
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
        users = info["user"]
        if limit_users is not None:
            users = users[:limit_users]
        for user_path in users:
            total_users += 1
            print("=" * 70)
            print(f"[Folder {f_i}/{len(folder_items)}] [{total_users}] {folder}")
            print(f"  User video: {user_path.name}")
            print(f"  Feature: {feature} | Swap: {swap} | Window: {window_ratio}")

            t0 = time.time()
            u_feats, u_idx = extract_variant_features_with_cache(user_path, feature, thr)
            extract_time = time.time() - t0
            if u_feats.size == 0:
                print("  WARNING: No features, skipping")
                continue

            t1 = time.time()
            rankings = compare_user_vs_all(u_feats, u_idx, translator_data, feature, swap, window_ratio)
            compare_time = time.time() - t1
            if not rankings:
                print("  WARNING: No rankings, skipping")
                continue

            top_folders = [r[0] for r in rankings]
            rank = top_folders.index(folder) + 1 if folder in top_folders else len(top_folders) + 1

            is_top1 = rank == 1
            is_top3 = rank <= 3
            is_top5 = rank <= 5
            top1_correct += int(is_top1)
            top3_correct += int(is_top3)
            top5_correct += int(is_top5)

            result = {
                "folder": folder,
                "user_video": user_path.name,
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
                "user_frames": int(len(u_idx)),
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
        "method": "method4_variants",
        "feature": str(feature),
        "swap": str(swap),
        "total_users": int(total_users),
        "total_translators": int(len(translator_data)),
        "top1_correct": int(top1_correct),
        "top3_correct": int(top3_correct),
        "top5_correct": int(top5_correct),
        "top1_accuracy": round(top1_correct / total_users * 100, 2) if total_users else 0.0,
        "top3_accuracy": round(top3_correct / total_users * 100, 2) if total_users else 0.0,
        "top5_accuracy": round(top5_correct / total_users * 100, 2) if total_users else 0.0,
        "params": {"thr": float(thr), "window_ratio": float(window_ratio)},
    }
    return all_results, summary


def save_results(all_results: List[Dict], summary: Dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{summary['feature']}_{summary['swap']}_win{summary['params']['window_ratio']:.2f}".replace(".", "p")

    csv_path = RESULTS_DIR / f"recognition_method4_{tag}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "folder", "user_video", "correct_rank", "top1", "top3", "top5",
                "correct_score", "top1_folder", "top1_score", "user_frames",
                "extract_time", "compare_time",
            ],
        )
        writer.writeheader()
        for r in all_results:
            row = {k: v for k, v in r.items() if k != "top5_rankings"}
            writer.writerow(row)

    json_path = RESULTS_DIR / f"recognition_method4_{tag}_summary.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "detailed": all_results}, f, ensure_ascii=False, indent=2)

    txt_path = RESULTS_DIR / f"recognition_method4_{tag}_report.txt"
    with txt_path.open("w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"METHOD4 VARIANT EVALUATION — feature={summary['feature']} swap={summary['swap']}\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total user videos tested:    {summary['total_users']}\n")
        f.write(f"Total translator references: {summary['total_translators']}\n\n")
        f.write(f"Top-1 Accuracy: {summary['top1_correct']}/{summary['total_users']} = {summary['top1_accuracy']}%\n")
        f.write(f"Top-3 Accuracy: {summary['top3_correct']}/{summary['total_users']} = {summary['top3_accuracy']}%\n")
        f.write(f"Top-5 Accuracy: {summary['top5_correct']}/{summary['total_users']} = {summary['top5_accuracy']}%\n\n")
        f.write(f"Params: thr={summary['params']['thr']} window_ratio={summary['params']['window_ratio']}\n\n")

    print("\nSaved:")
    print(f"  {csv_path}")
    print(f"  {json_path}")
    print(f"  {txt_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate Method4 variants (angles/bones + swap).")
    p.add_argument("--feature", choices=["angles15", "bones_angles", "bones_angles_len", "bones_angles_len_d1"], default="angles15")
    p.add_argument("--swap", choices=["none", "global"], default="global")
    p.add_argument("--window-ratio", type=float, default=0.3)
    p.add_argument("--similarity-threshold", type=float, default=0.99)
    p.add_argument("--limit-folders", type=int, default=None)
    p.add_argument("--limit-users", type=int, default=None)
    args = p.parse_args()

    os.chdir(str(BASE_DIR))
    folders = discover_folders()
    print(f"Found {len(folders)} folders.")
    if args.limit_folders is not None:
        folders = dict(list(folders.items())[: args.limit_folders])
        print(f"(Limited to first {args.limit_folders} folders.)")

    feature: FeatureKind = args.feature  # type: ignore[assignment]
    swap: SwapMode = args.swap  # type: ignore[assignment]

    print("=" * 80)
    print("PHASE 1: Precomputing translator features (cached)")
    print("=" * 80)
    translator_data = precompute_translators(folders, feature, args.similarity_threshold)

    print("\n" + "=" * 80)
    print("PHASE 2: Evaluating")
    print("=" * 80 + "\n")
    all_results, summary = run_eval(
        folders,
        translator_data,
        feature=feature,
        swap=swap,
        thr=args.similarity_threshold,
        window_ratio=args.window_ratio,
        limit_folders=args.limit_folders,
        limit_users=args.limit_users,
    )
    save_results(all_results, summary)


if __name__ == "__main__":
    main()

