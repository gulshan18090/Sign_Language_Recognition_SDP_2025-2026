"""
Top-3 Accuracy Evaluation

For each subfolder (sentence) with ≥2 videos, pick up to 3 random query videos.
For each query, compare against ALL other videos and check whether any of
the top-3 most-similar results belong to the same subfolder (correct match).

Reports the overall Top-3 accuracy (% of queries with a correct match in top 3).

Usage:
    python top3_accuracy.py
    python top3_accuracy.py --matrices matrices --method cosine --n-queries 3 --seed 42
"""

import argparse
import random
import time
from pathlib import Path

import numpy as np

from inference.similarity import SimilarityEngine


def parse_args():
    parser = argparse.ArgumentParser(description="Top-3 similarity accuracy evaluation")
    parser.add_argument("--matrices", type=str, default="data/processed/matrices",
                        help="Root folder with per-video .npy feature matrices (default: matrices)")
    parser.add_argument("--method", type=str, default="cosine",
                        help="Similarity method: cosine, dtw, frame_wise_cosine, etc. (default: cosine)")
    parser.add_argument("--n-queries", type=int, default=3,
                        help="Max random videos to pick per subfolder as queries (default: 3)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    return parser.parse_args()


def load_all_features(matrices_root: Path):
    """
    Load all .npy feature matrices.

    Returns
    -------
    features : dict[str, np.ndarray]
        {npy_path_relative_to_root: matrix}
    folder_map : dict[str, str]
        {npy_path_relative: subfolder_name}
    folder_to_videos : dict[str, list[str]]
        {subfolder_name: [npy_path_relative, ...]}
    """
    features = {}
    folder_map = {}
    folder_to_videos = {}

    for npy_path in sorted(matrices_root.rglob("*.npy")):
        rel = str(npy_path.relative_to(matrices_root))
        subfolder = npy_path.parent.name  # immediate parent = sentence folder
        if subfolder == matrices_root.name:
            # npy sitting directly in matrices root — skip (e.g. stray files)
            continue
        matrix = np.load(npy_path)
        features[rel] = matrix
        folder_map[rel] = subfolder
        folder_to_videos.setdefault(subfolder, []).append(rel)

    return features, folder_map, folder_to_videos


def main():
    args = parse_args()
    random.seed(args.seed)
    matrices_root = Path(args.matrices)

    if not matrices_root.exists():
        raise FileNotFoundError(f"Matrices folder not found: {matrices_root}")

    print("Loading feature matrices …")
    features, folder_map, folder_to_videos = load_all_features(matrices_root)
    total_videos = len(features)
    total_folders = len(folder_to_videos)
    print(f"  Loaded {total_videos} videos across {total_folders} folders\n")

    engine = SimilarityEngine(default_method=args.method)

    # ── Select query videos ──────────────────────────────
    queries = []  # list of (video_key, true_folder)
    for folder, videos in sorted(folder_to_videos.items()):
        if len(videos) < 2:
            # Need at least 1 other video in the folder for a correct match
            continue
        chosen = random.sample(videos, min(args.n_queries, len(videos)))
        for v in chosen:
            queries.append((v, folder))

    print(f"Selected {len(queries)} query videos from "
          f"{len({f for _, f in queries})} folders (up to {args.n_queries} per folder)\n")
    print("=" * 80)

    # ── Evaluate ─────────────────────────────────────────
    correct = 0
    total = 0
    per_folder_stats = {}  # folder -> [correct_count, total_count]

    t_start = time.time()

    for idx, (query_key, true_folder) in enumerate(queries, 1):
        query_matrix = features[query_key]

        # Compare against ALL other videos
        scores = {}
        for other_key, other_matrix in features.items():
            if other_key == query_key:
                continue  # skip self
            sim = engine.compute_similarity(query_matrix, other_matrix, method=args.method)
            scores[other_key] = sim

        # Sort by similarity descending, take top 3
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top3 = ranked[:3]

        # Check if any top-3 result is from the same folder
        top3_folders = [folder_map[k] for k, _ in top3]
        hit = true_folder in top3_folders

        if hit:
            correct += 1
        total += 1

        # Per-folder tracking
        if true_folder not in per_folder_stats:
            per_folder_stats[true_folder] = [0, 0]
        per_folder_stats[true_folder][1] += 1
        if hit:
            per_folder_stats[true_folder][0] += 1

        # Log
        status = "HIT" if hit else "MISS"
        print(f"[{idx}/{len(queries)}] {status}  query: {query_key}")
        for rank, (k, s) in enumerate(top3, 1):
            match_flag = " <<<" if folder_map[k] == true_folder else ""
            print(f"    Top-{rank}: {k}  (sim={s:.4f}, folder={folder_map[k]}){match_flag}")

    elapsed = time.time() - t_start

    # ── Summary ──────────────────────────────────────────
    acc = correct / total * 100 if total else 0
    print("\n" + "=" * 80)
    print(f"OVERALL TOP-3 ACCURACY: {correct}/{total} = {acc:.2f}%")
    print(f"Time: {elapsed:.1f}s\n")

    # Per-folder breakdown
    print("Per-folder breakdown:")
    print(f"  {'Folder':<60s} {'Acc':>10s}")
    print("  " + "-" * 72)
    for folder in sorted(per_folder_stats):
        c, t = per_folder_stats[folder]
        pct = c / t * 100 if t else 0
        print(f"  {folder:<60s} {c}/{t} = {pct:.0f}%")


if __name__ == "__main__":
    main()
