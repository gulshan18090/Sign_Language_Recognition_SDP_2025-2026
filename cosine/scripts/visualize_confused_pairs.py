"""
Visualize Confused Pairs

Finds video pairs from DIFFERENT folders that have high cosine similarity
(false positives) and renders a side-by-side frame strip so you can
visually inspect why the model confuses them.

Usage:
    python visualize_confused_pairs.py
    python visualize_confused_pairs.py --top-k 10 --frames 8 --output confused_pairs
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

from inference.similarity import SimilarityEngine


# ── Helpers ──────────────────────────────────────────────

VIDEO_DIR = Path("data/raw/Videos")
MATRIX_DIR = Path("data/processed/matrices")
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}


def load_all_features(matrices_root: Path):
    """Load every .npy matrix.  Returns (features, folder_map)."""
    features: dict[str, np.ndarray] = {}
    folder_map: dict[str, str] = {}

    for npy_path in sorted(matrices_root.rglob("*.npy")):
        rel = str(npy_path.relative_to(matrices_root))
        subfolder = npy_path.parent.name
        if subfolder == matrices_root.name:
            continue
        features[rel] = np.load(npy_path)
        folder_map[rel] = subfolder

    return features, folder_map


def find_confused_pairs(
    features: dict[str, np.ndarray],
    folder_map: dict[str, str],
    method: str = "cosine",
    top_k: int = 10,
) -> list[tuple[str, str, float]]:
    """
    Return the top-k highest-similarity pairs from DIFFERENT folders.

    Each entry: (video_key_a, video_key_b, similarity).
    """
    engine = SimilarityEngine(default_method=method)
    keys = list(features.keys())
    pairs: list[tuple[str, str, float]] = []

    total = len(keys)
    print(f"Computing pairwise similarities for {total} videos …")
    t0 = time.time()

    for i in range(total):
        for j in range(i + 1, total):
            # Only cross-folder pairs
            if folder_map[keys[i]] == folder_map[keys[j]]:
                continue
            sim = engine.compute_similarity(
                features[keys[i]], features[keys[j]], method=method
            )
            pairs.append((keys[i], keys[j], sim))

    pairs.sort(key=lambda x: x[2], reverse=True)
    elapsed = time.time() - t0
    print(f"  Computed {len(pairs)} cross-folder pairs in {elapsed:.1f}s")
    return pairs[:top_k]


def npy_key_to_video_path(key: str, videos_root: Path = VIDEO_DIR) -> Path | None:
    """Convert an npy relative key (folder/file.npy) to the video path."""
    stem = Path(key).with_suffix("")  # e.g.  folder/file
    for ext in VIDEO_EXTS:
        candidate = videos_root / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def extract_uniform_frames(video_path: Path, n_frames: int = 8) -> list[np.ndarray]:
    """Read *n_frames* uniformly-sampled RGB frames from a video file."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise RuntimeError(f"Empty video {video_path}")

    indices = [int(i * total / n_frames) for i in range(n_frames)]
    indices = [min(idx, total - 1) for idx in indices]

    frames: list[np.ndarray] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            # fallback: black frame
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 320
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 240
            frames.append(np.zeros((h, w, 3), dtype=np.uint8))

    cap.release()
    return frames


def make_comparison_figure(
    frames_a: list[np.ndarray],
    frames_b: list[np.ndarray],
    label_a: str,
    label_b: str,
    similarity: float,
    save_path: Path,
):
    """
    Create a 2-row × N-column figure:
        Row 1: frames from video A
        Row 2: frames from video B
    """
    n = len(frames_a)
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 7))

    if n == 1:
        axes = axes.reshape(2, 1)

    for col in range(n):
        axes[0, col].imshow(frames_a[col])
        axes[0, col].set_xticks([])
        axes[0, col].set_yticks([])

        axes[1, col].imshow(frames_b[col])
        axes[1, col].set_xticks([])
        axes[1, col].set_yticks([])

    # Row labels on the left
    folder_a = Path(label_a).parent.name if "/" in label_a or "\\" in label_a else label_a
    folder_b = Path(label_b).parent.name if "/" in label_b or "\\" in label_b else label_b
    axes[0, 0].set_ylabel(f"{folder_a}\n({Path(label_a).stem})", fontsize=8, rotation=0,
                           labelpad=120, va="center", ha="left")
    axes[1, 0].set_ylabel(f"{folder_b}\n({Path(label_b).stem})", fontsize=8, rotation=0,
                           labelpad=120, va="center", ha="left")

    fig.suptitle(
        f"Confused pair  —  cosine similarity = {similarity:.4f}\n"
        f'Top: "{folder_a}"   |   Bottom: "{folder_b}"',
        fontsize=11,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout(rect=[0.08, 0, 1, 0.94])
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved → {save_path}")


# ── Main ─────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize frame comparisons for high-similarity cross-folder (confused) pairs"
    )
    parser.add_argument("--matrices", type=str, default="data/processed/matrices")
    parser.add_argument("--videos", type=str, default="data/raw/Videos")
    parser.add_argument("--output", type=str, default="outputs/analysis/confused_pairs",
                        help="Output directory for images (default: confused_pairs)")
    parser.add_argument("--method", type=str, default="cosine")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Number of most-confused pairs to visualize (default: 10)")
    parser.add_argument("--frames", "-n", type=int, default=8,
                        help="Number of uniformly-sampled frames per video (default: 8)")
    return parser.parse_args()


def main():
    args = parse_args()
    matrices_root = Path(args.matrices)
    videos_root = Path(args.videos)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load features
    print("Loading feature matrices …")
    features, folder_map = load_all_features(matrices_root)
    print(f"  {len(features)} videos loaded\n")

    # 2. Find confused pairs
    confused = find_confused_pairs(features, folder_map, method=args.method, top_k=args.top_k)

    # 3. Visualize each pair
    print(f"\nVisualizing top-{len(confused)} confused pairs …\n")
    for rank, (key_a, key_b, sim) in enumerate(confused, 1):
        print(f"  [{rank}] sim={sim:.4f}  {folder_map[key_a]}  vs  {folder_map[key_b]}")

        vid_a = npy_key_to_video_path(key_a, videos_root)
        vid_b = npy_key_to_video_path(key_b, videos_root)

        if vid_a is None:
            print(f"    ⚠ Video not found for {key_a}, skipping")
            continue
        if vid_b is None:
            print(f"    ⚠ Video not found for {key_b}, skipping")
            continue

        frames_a = extract_uniform_frames(vid_a, args.frames)
        frames_b = extract_uniform_frames(vid_b, args.frames)

        save_path = output_dir / f"confused_{rank:02d}_sim{sim:.4f}.png"
        make_comparison_figure(frames_a, frames_b, key_a, key_b, sim, save_path)

    # 4. Summary table
    summary_path = output_dir / "summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Top-{len(confused)} most confused cross-folder pairs\n")
        f.write(f"Method: {args.method}\n")
        f.write("=" * 90 + "\n")
        f.write(f"{'Rank':>4}  {'Similarity':>10}  {'Folder A':<35}  {'Folder B':<35}\n")
        f.write("-" * 90 + "\n")
        for rank, (key_a, key_b, sim) in enumerate(confused, 1):
            f.write(f"{rank:>4}  {sim:>10.4f}  {folder_map[key_a]:<35}  {folder_map[key_b]:<35}\n")
    print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()
