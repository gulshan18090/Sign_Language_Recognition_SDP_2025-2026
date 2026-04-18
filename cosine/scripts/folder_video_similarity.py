"""
Folder Video Similarity Runner

Steps:
1) Extract 64 frames with hands and convert to feature matrices.
2) Save each video's matrix to an output folder (reuse if already saved).
3) Compute cosine similarity within each subfolder.
4) Compute cosine similarity across different subfolders.

Example:
  python folder_video_similarity.py --videos-root Videos --output matrices
"""

import argparse
import json
from pathlib import Path

import numpy as np

from inference.similarity import VideoMatcher, SimilarityEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract hand feature matrices and compute cosine similarity per folder."
    )
    parser.add_argument(
        "--videos-root",
        type=str,
        default="data/raw/Videos",
        help="Root folder containing subfolders of videos (default: Videos)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/matrices",
        help="Output folder to save matrices and results (default: matrices)",
    )
    parser.add_argument(
        "--n-frames",
        type=int,
        default=64,
        help="Number of frames to extract per video (default: 64)",
    )
    parser.add_argument(
        "--max-hands",
        type=int,
        default=2,
        choices=[1, 2],
        help="Maximum number of hands to track (default: 2)",
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Disable filtering to hand-only frames",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable landmark normalization",
    )
    parser.add_argument(
        "--no-reuse",
        action="store_true",
        help="Force re-extraction even if .npy matrices already exist",
    )
    return parser.parse_args()


def find_video_files(folder: Path, extensions: set) -> list[Path]:
    files: list[Path] = []
    for ext in extensions:
        files.extend(folder.glob(f"*{ext}"))
        files.extend(folder.glob(f"*{ext.upper()}"))
    return sorted(set(files))


def main() -> None:
    args = parse_args()
    videos_root = Path(args.videos_root)
    output_root = Path(args.output)

    if not videos_root.exists():
        raise FileNotFoundError(f"Videos root not found: {videos_root}")

    output_root.mkdir(parents=True, exist_ok=True)

    matcher = VideoMatcher(
        n_frames=args.n_frames,
        max_hands=args.max_hands,
        similarity_method="cosine",
        normalize_landmarks=not args.no_normalize,
        filter_hand_frames=not args.no_filter,
        cache_features=True,
    )
    engine = SimilarityEngine(default_method="cosine")

    results: dict[str, dict] = {}
    all_features: dict[str, np.ndarray] = {}
    video_to_folder: dict[str, str] = {}

    for subfolder in sorted([p for p in videos_root.iterdir() if p.is_dir()]):
        video_files = find_video_files(subfolder, matcher.SUPPORTED_EXTENSIONS)
        if len(video_files) < 2:
            print(f"Skipping {subfolder.name}: need at least 2 videos, found {len(video_files)}")
            continue

        print(f"\nProcessing folder: {subfolder.name} ({len(video_files)} videos)")

        features_by_video: dict[str, np.ndarray] = {}

        # Extract matrices and save them
        folder_output = output_root / subfolder.name
        folder_output.mkdir(parents=True, exist_ok=True)

        for video_path in video_files:
            matrix_path = folder_output / f"{video_path.stem}.npy"
            if matrix_path.exists() and not args.no_reuse:
                matrix = np.load(matrix_path)
                print(f"  Reused matrix: {matrix_path}")
            else:
                video_features = matcher.process_video(str(video_path))
                matrix = video_features.feature_matrix
                np.save(matrix_path, matrix)
                print(f"  Saved matrix: {matrix_path}")

            features_by_video[str(video_path)] = matrix
            all_features[str(video_path)] = matrix
            video_to_folder[str(video_path)] = subfolder.name

        # Compute cosine similarities within the folder
        folder_results: dict[str, dict[str, float]] = {}
        video_paths = list(features_by_video.keys())
        for i, path_a in enumerate(video_paths):
            for j in range(i + 1, len(video_paths)):
                path_b = video_paths[j]
                sim = engine.compute_similarity(
                    features_by_video[path_a],
                    features_by_video[path_b],
                    method="cosine",
                )
                folder_results.setdefault(path_a, {})[path_b] = sim
                folder_results.setdefault(path_b, {})[path_a] = sim

                print(
                    f"  Cosine similarity: {Path(path_a).name} vs {Path(path_b).name} = {sim:.4f}"
                )

        results[subfolder.name] = {
            "videos": video_paths,
            "similarities": folder_results,
        }

    # Compute cross-folder similarities
    cross_folder_results: dict[str, dict[str, float]] = {}
    video_paths = list(all_features.keys())
    for i, path_a in enumerate(video_paths):
        for j in range(i + 1, len(video_paths)):
            path_b = video_paths[j]
            if video_to_folder.get(path_a) == video_to_folder.get(path_b):
                continue
            sim = engine.compute_similarity(
                all_features[path_a],
                all_features[path_b],
                method="cosine",
            )
            cross_folder_results.setdefault(path_a, {})[path_b] = sim
            cross_folder_results.setdefault(path_b, {})[path_a] = sim

    results["cross_folder_similarities"] = cross_folder_results

    results_path = output_root / "similarity_results.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved similarity results: {results_path}")


if __name__ == "__main__":
    main()
