"""
Extract feature vectors for ALL videos in the Videos folder.
Saves each video's feature matrix as a .npy file under matrices/.

Features:
- Shows clear progress (video X / N)
- Skips already-extracted videos (resume-friendly)
- Uses --no-filter logic (uniform sampling, no expensive hand pre-filtering)
"""

import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inference.similarity.frame_extractor import FrameExtractor
from inference.similarity.feature_extractor import HandFeatureExtractor

# ── Config ──────────────────────────────────────────────
VIDEO_DIR = Path("data/raw/Videos")
OUTPUT_DIR = Path("data/processed/matrices")
N_FRAMES = 64
MAX_HANDS = 2
EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}
# ────────────────────────────────────────────────────────


def find_videos(root: Path):
    """Recursively find all video files."""
    videos = []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() in EXTENSIONS and p.is_file():
            videos.append(p)
    return videos


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    videos = find_videos(VIDEO_DIR)
    total = len(videos)
    print(f"Found {total} videos in {VIDEO_DIR}/", flush=True)

    # Init extractors once
    frame_ext = FrameExtractor(
        n_frames=N_FRAMES,
        filter_hand_frames=False,  # skip expensive filtering
    )
    feat_ext = HandFeatureExtractor(
        max_hands=MAX_HANDS,
        normalize=True,
    )

    done = 0
    skipped = 0
    errors = 0

    for i, video_path in enumerate(videos, 1):
        # Build output path mirroring folder structure
        rel = video_path.relative_to(VIDEO_DIR)
        npy_path = (OUTPUT_DIR / rel).with_suffix(".npy")

        # Skip if already extracted
        if npy_path.exists():
            skipped += 1
            print(f"[{i}/{total}] SKIP (exists): {rel}", flush=True)
            continue

        t0 = time.time()
        try:
            # Extract frames (uniform sampling, no hand-filter)
            frames, _ = frame_ext.extract_all_frames(str(video_path))
            sampled = frame_ext.sample_to_fixed_count(frames, N_FRAMES)

            # Extract feature matrix
            matrix = feat_ext.frames_to_feature_matrix(sampled)

            # Save
            npy_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(npy_path, matrix)

            elapsed = time.time() - t0
            done += 1
            print(
                f"[{i}/{total}] OK  {rel}  "
                f"shape={matrix.shape}  {elapsed:.1f}s",
                flush=True,
            )

        except Exception as e:
            errors += 1
            print(f"[{i}/{total}] ERR {rel}: {e}", flush=True)

    print(f"\nDone!  extracted={done}  skipped={skipped}  errors={errors}", flush=True)


if __name__ == "__main__":
    main()
