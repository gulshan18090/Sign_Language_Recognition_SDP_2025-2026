"""
Visualize DTW Frame-to-Frame Warping Path Only (No Time Axis)
============================================================
For a given video pair (translator + user), runs DTW alignment and produces
ONLY the frame-index warping path plot (like left side of dtw_warping_path.png).

Usage:
  python visualize_dtw_frame_path.py --folder "51" --user-idx 0
  (or just run with no args to use the first available folder)
"""

import numpy as np
import os
import sys
import cv2
import io
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(__file__))
from compare_three_methods import (
    extract_all_frames_and_filter,
    dtw_align_with_prefiltered_frames,
)

VIDEOS_DIR = os.path.join(os.path.dirname(__file__), 'Videos')
MATRICES_DIR = os.path.join(os.path.dirname(__file__), 'matrices')


def plot_frame_warping_path(path_pairs, title, output_path):
    """
    Plot: Warping path — which translator frame maps to which user frame.
    X-axis: translator frame index, Y-axis: user frame index.
    Shows the diagonal (perfect sync) for reference.
    """
    trans_frames = [p[0] for p in path_pairs]
    user_frames = [p[1] for p in path_pairs]
    sims = [p[2] for p in path_pairs]

    fig, ax = plt.subplots(figsize=(8, 7))
    scatter = ax.scatter(trans_frames, user_frames, c=sims, cmap='RdYlGn',
                         s=14, alpha=0.85, edgecolors='none', vmin=0, vmax=1)
    max_frame = max(max(trans_frames), max(user_frames))
    ax.plot([0, max_frame], [0, max_frame], 'k--', alpha=0.3, label='Perfect sync (diagonal)')
    ax.set_xlabel('Translator Frame Index', fontsize=13)
    ax.set_ylabel('User Frame Index', fontsize=13)
    ax.set_title(title, fontsize=15, fontweight='bold')
    ax.legend(fontsize=10)
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label('Cosine Similarity', fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches='tight')
    plt.close()
    print(f"Saved frame warping path: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize DTW frame-to-frame warping path (no time axis)')
    parser.add_argument('--folder', type=str, default=None,
                        help='Folder name inside Videos/ (default: first folder found)')
    parser.add_argument('--user-idx', type=int, default=0,
                        help='Index of user video to use (0 = first, default: 0)')
    parser.add_argument('--window-ratio', type=float, default=0.3,
                        help='DTW Sakoe-Chiba window ratio (default: 0.3)')
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Find folder
    if args.folder:
        folder_path = os.path.join(VIDEOS_DIR, args.folder)
        if not os.path.isdir(folder_path):
            print(f"ERROR: Folder not found: {folder_path}")
            sys.exit(1)
        folder_name = args.folder
    else:
        # Pick first folder with both translator and user
        for d in sorted(os.listdir(VIDEOS_DIR)):
            full = os.path.join(VIDEOS_DIR, d)
            if not os.path.isdir(full):
                continue
            import glob
            trans = glob.glob(os.path.join(full, 'translator_*.mp4'))
            users = glob.glob(os.path.join(full, 'user_*.mp4'))
            if trans and users:
                folder_name = d
                folder_path = full
                break
        else:
            print("ERROR: No folder found with both translator and user videos")
            sys.exit(1)

    import glob
    trans_videos = sorted(glob.glob(os.path.join(folder_path, 'translator_*.mp4')))
    user_videos = sorted(glob.glob(os.path.join(folder_path, 'user_*.mp4')))

    if not trans_videos:
        print(f"ERROR: No translator videos in {folder_path}")
        sys.exit(1)
    if not user_videos or args.user_idx >= len(user_videos):
        print(f"ERROR: No user video at index {args.user_idx} in {folder_path}")
        sys.exit(1)

    trans_path = trans_videos[0]
    user_path = user_videos[args.user_idx]

    print("Extracting features...")
    trans_feat, trans_idx = extract_all_frames_and_filter(trans_path, similarity_threshold=0.99)
    user_feat, user_idx = extract_all_frames_and_filter(user_path, similarity_threshold=0.99)

    if len(trans_feat) == 0 or len(user_feat) == 0:
        print("ERROR: Could not extract features from one or both videos")
        sys.exit(1)

    print("Running DTW alignment...")
    avg_sim, path_with_indices, frame_sims = dtw_align_with_prefiltered_frames(
        trans_feat, user_feat, trans_idx, user_idx, window_ratio=args.window_ratio
    )

    print(f"DTW Similarity: {avg_sim:.4f}")
    print(f"Aligned pairs:  {len(path_with_indices)}")

    # Save plot
    safe_folder = folder_name.replace(' ', '_')[:60]
    output_dir = os.path.join(MATRICES_DIR, safe_folder)
    os.makedirs(output_dir, exist_ok=True)
    plot_frame_warping_path(
        path_with_indices,
        f'DTW Warping Path — "{folder_name}"\nSimilarity: {avg_sim:.4f}',
        os.path.join(output_dir, 'dtw_frame_warping_path.png')
    )
    print(f"All done. Plot saved to {output_dir}/dtw_frame_warping_path.png")

if __name__ == '__main__':
    main()
