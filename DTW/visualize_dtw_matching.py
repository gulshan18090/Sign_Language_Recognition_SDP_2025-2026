"""
Visualize DTW Frame Matching Between Translator and User Videos
================================================================
For a given video pair (translator + user), runs DTW alignment and produces
graphs showing which translator frame matches which user frame, along with
the similarity score at each matched pair.

Outputs (saved to matrices/<folder>/):
  1. Warping path plot — shows frame-to-frame correspondences
  2. Similarity timeline — per-pair similarity along the alignment
  3. Combined dashboard with matched frame thumbnails

Usage:
  python visualize_dtw_matching.py                         # uses default folder
  python visualize_dtw_matching.py --folder "Bu gün hava çox soyuqdur"
  python visualize_dtw_matching.py --folder "51" --user-idx 0
"""

import numpy as np
import os
import sys
import cv2
import io
import argparse
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from collections import defaultdict

# Fix Unicode output on Windows console
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.append(os.path.dirname(__file__))
from compare_three_methods import (
    extract_all_frames_and_filter,
    dtw_align_with_prefiltered_frames,
)

VIDEOS_DIR = os.path.join(os.path.dirname(__file__), 'Videos')
MATRICES_DIR = os.path.join(os.path.dirname(__file__), 'matrices')


def get_video_fps(video_path):
    """Get the FPS of a video file."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps > 0 else 30.0


def extract_video_frames(video_path):
    """Extract all raw frames from a video."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def plot_warping_path(path_pairs, fps_trans, fps_user, title, output_path):
    """
    Plot 1: Warping path — which translator frame maps to which user frame.
    X-axis: translator frame (or time), Y-axis: user frame (or time).
    Also shows the diagonal (perfect sync) for reference.
    """
    trans_frames = [p[0] for p in path_pairs]
    user_frames = [p[1] for p in path_pairs]
    sims = [p[2] for p in path_pairs]

    trans_times = [f / fps_trans for f in trans_frames]
    user_times = [f / fps_user for f in user_frames]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # --- Left: Frame-to-frame warping path ---
    ax = axes[0]
    scatter = ax.scatter(trans_frames, user_frames, c=sims, cmap='RdYlGn',
                         s=12, alpha=0.8, edgecolors='none', vmin=0, vmax=1)
    # Diagonal reference
    max_frame = max(max(trans_frames), max(user_frames))
    ax.plot([0, max_frame], [0, max_frame], 'k--', alpha=0.3, label='Perfect sync (diagonal)')
    ax.set_xlabel('Translator Frame Index', fontsize=12)
    ax.set_ylabel('User Frame Index', fontsize=12)
    ax.set_title('DTW Warping Path (Frame Indices)', fontsize=13)
    ax.legend(fontsize=9)
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label('Cosine Similarity', fontsize=10)

    # --- Right: Time-to-time warping path ---
    ax2 = axes[1]
    scatter2 = ax2.scatter(trans_times, user_times, c=sims, cmap='RdYlGn',
                           s=12, alpha=0.8, edgecolors='none', vmin=0, vmax=1)
    max_time = max(max(trans_times), max(user_times))
    ax2.plot([0, max_time], [0, max_time], 'k--', alpha=0.3, label='Perfect sync (diagonal)')
    ax2.set_xlabel('Translator Time (seconds)', fontsize=12)
    ax2.set_ylabel('User Time (seconds)', fontsize=12)
    ax2.set_title('DTW Warping Path (Time)', fontsize=13)
    ax2.legend(fontsize=9)
    cbar2 = plt.colorbar(scatter2, ax=ax2, shrink=0.8)
    cbar2.set_label('Cosine Similarity', fontsize=10)

    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved warping path: {output_path}")


def plot_similarity_timeline(path_pairs, fps_trans, title, output_path):
    """
    Plot 2: Similarity score at each alignment step along the sequence.
    Shows how well each matched pair aligns, with average line.
    """
    sims = [p[2] for p in path_pairs]
    steps = list(range(len(sims)))
    avg_sim = np.mean(sims)

    fig, ax = plt.subplots(figsize=(14, 5))

    # Color bars by similarity
    colors = plt.cm.RdYlGn([s for s in sims])
    ax.bar(steps, sims, color=colors, width=1.0, edgecolor='none', alpha=0.8)
    ax.axhline(y=avg_sim, color='blue', linestyle='--', linewidth=1.5,
               label=f'Average similarity: {avg_sim:.4f}')

    ax.set_xlabel('Alignment Step', fontsize=12)
    ax.set_ylabel('Cosine Similarity', fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved similarity timeline: {output_path}")


def plot_matched_frames_dashboard(path_pairs, trans_raw_frames, user_raw_frames,
                                   fps_trans, fps_user, title, output_path,
                                   n_samples=10):
    """
    Plot 3: Dashboard showing actual video frames side by side at key alignment points.
    Selects n_samples evenly spaced alignment steps and shows translator frame
    next to the matched user frame, with similarity and time info.
    """
    n_total = len(path_pairs)
    if n_total <= n_samples:
        sample_indices = list(range(n_total))
    else:
        sample_indices = [int(i * (n_total - 1) / (n_samples - 1)) for i in range(n_samples)]

    n_show = len(sample_indices)
    fig, axes = plt.subplots(n_show, 2, figsize=(12, 3 * n_show))
    if n_show == 1:
        axes = axes.reshape(1, 2)

    for row, step_idx in enumerate(sample_indices):
        trans_frame_idx, user_frame_idx, sim = path_pairs[step_idx]
        trans_time = trans_frame_idx / fps_trans
        user_time = user_frame_idx / fps_user

        # Translator frame
        ax_t = axes[row, 0]
        if trans_frame_idx < len(trans_raw_frames):
            ax_t.imshow(trans_raw_frames[trans_frame_idx])
        else:
            ax_t.text(0.5, 0.5, f'Frame {trans_frame_idx}\n(out of range)',
                      ha='center', va='center', transform=ax_t.transAxes)
        ax_t.set_title(f'Translator  |  Frame {trans_frame_idx}  |  t={trans_time:.2f}s',
                       fontsize=10, color='blue')
        ax_t.axis('off')

        # User frame
        ax_u = axes[row, 1]
        if user_frame_idx < len(user_raw_frames):
            ax_u.imshow(user_raw_frames[user_frame_idx])
        else:
            ax_u.text(0.5, 0.5, f'Frame {user_frame_idx}\n(out of range)',
                      ha='center', va='center', transform=ax_u.transAxes)
        ax_u.set_title(f'User  |  Frame {user_frame_idx}  |  t={user_time:.2f}s',
                       fontsize=10, color='green')
        ax_u.axis('off')

        # Similarity label between
        color = 'green' if sim >= 0.7 else ('orange' if sim >= 0.4 else 'red')
        ax_t.annotate(f'sim={sim:.3f}', xy=(1.02, 0.5), xycoords='axes fraction',
                      fontsize=11, fontweight='bold', color=color, va='center')

    # Column headers
    axes[0, 0].set_title(f'TRANSLATOR  |  Frame {path_pairs[sample_indices[0]][0]}  |  '
                          f't={path_pairs[sample_indices[0]][0]/fps_trans:.2f}s',
                          fontsize=10, color='blue')
    axes[0, 1].set_title(f'USER  |  Frame {path_pairs[sample_indices[0]][1]}  |  '
                          f't={path_pairs[sample_indices[0]][1]/fps_user:.2f}s',
                          fontsize=10, color='green')

    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved matched frames dashboard: {output_path}")


def plot_frame_index_connections(path_pairs, fps_trans, fps_user, title, output_path):
    """
    Plot 4: Two parallel timelines (translator and user) with lines connecting
    matched frames — clearly shows which frame matches which.
    """
    trans_frames = [p[0] for p in path_pairs]
    user_frames = [p[1] for p in path_pairs]
    sims = [p[2] for p in path_pairs]

    # Sample connections to avoid clutter (max 60 lines)
    n_total = len(path_pairs)
    if n_total > 60:
        step = n_total // 60
        sample_idx = list(range(0, n_total, step))
    else:
        sample_idx = list(range(n_total))

    fig, ax = plt.subplots(figsize=(16, 6))

    # Draw two horizontal bars for the timelines
    trans_max = max(trans_frames)
    user_max = max(user_frames)

    ax.barh(1, trans_max, height=0.15, color='#2196F3', alpha=0.3, label='Translator timeline')
    ax.barh(0, user_max, height=0.15, color='#4CAF50', alpha=0.3, label='User timeline')

    # Draw connecting lines
    for idx in sample_idx:
        tf, uf, sim = path_pairs[idx]
        color = plt.cm.RdYlGn(sim)
        ax.plot([tf, uf], [1, 0], color=color, alpha=0.5, linewidth=0.8)

    # Mark frame positions
    trans_sampled = [trans_frames[i] for i in sample_idx]
    user_sampled = [user_frames[i] for i in sample_idx]
    sims_sampled = [sims[i] for i in sample_idx]

    ax.scatter(trans_sampled, [1] * len(trans_sampled), c=[plt.cm.RdYlGn(s) for s in sims_sampled],
               s=20, zorder=5, edgecolors='black', linewidth=0.3)
    ax.scatter(user_sampled, [0] * len(user_sampled), c=[plt.cm.RdYlGn(s) for s in sims_sampled],
               s=20, zorder=5, edgecolors='black', linewidth=0.3)

    ax.set_yticks([0, 1])
    ax.set_yticklabels(['User Frames', 'Translator Frames'], fontsize=12)
    ax.set_xlabel('Frame Index', fontsize=12)
    ax.set_ylim(-0.5, 1.5)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='x', alpha=0.3)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='RdYlGn', norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label('Similarity', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved frame connections: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize DTW frame matching between translator and user videos')
    parser.add_argument('--folder', type=str, default=None,
                        help='Folder name inside Videos/ (default: first folder found)')
    parser.add_argument('--user-idx', type=int, default=0,
                        help='Index of user video to use (0 = first, default: 0)')
    parser.add_argument('--window-ratio', type=float, default=0.3,
                        help='DTW Sakoe-Chiba window ratio (default: 0.3)')
    parser.add_argument('--n-thumbnails', type=int, default=10,
                        help='Number of frame pairs to show in dashboard (default: 10)')
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

    print("=" * 80)
    print("DTW FRAME MATCHING VISUALIZATION")
    print("=" * 80)
    print(f"  Folder:     {folder_name}")
    print(f"  Translator: {os.path.basename(trans_path)}")
    print(f"  User:       {os.path.basename(user_path)}")
    print()

    # Get FPS
    fps_trans = get_video_fps(trans_path)
    fps_user = get_video_fps(user_path)
    print(f"  FPS — Translator: {fps_trans:.1f}, User: {fps_user:.1f}")

    # Step 1: Extract and filter features
    print("\n--- Extracting translator features ---")
    trans_feat, trans_idx = extract_all_frames_and_filter(trans_path, similarity_threshold=0.99)
    print("\n--- Extracting user features ---")
    user_feat, user_idx = extract_all_frames_and_filter(user_path, similarity_threshold=0.99)

    if len(trans_feat) == 0 or len(user_feat) == 0:
        print("ERROR: Could not extract features from one or both videos")
        sys.exit(1)

    print(f"\n  Translator: {len(trans_idx)} filtered frames (from {len(trans_feat)} features)")
    print(f"  User:       {len(user_idx)} filtered frames (from {len(user_feat)} features)")

    # Step 2: Run DTW alignment
    print("\n--- Running DTW alignment ---")
    avg_sim, path_with_indices, frame_sims = dtw_align_with_prefiltered_frames(
        trans_feat, user_feat, trans_idx, user_idx, window_ratio=args.window_ratio
    )

    print(f"\n  DTW Similarity: {avg_sim:.4f}")
    print(f"  Aligned pairs:  {len(path_with_indices)}")

    if len(path_with_indices) == 0:
        print("ERROR: DTW produced no alignment pairs")
        sys.exit(1)

    # Step 3: Extract raw frames for thumbnail visualization
    print("\n--- Extracting raw frames for thumbnails ---")
    trans_raw = extract_video_frames(trans_path)
    user_raw = extract_video_frames(user_path)
    print(f"  Translator: {len(trans_raw)} raw frames")
    print(f"  User:       {len(user_raw)} raw frames")

    # Step 4: Create output directory
    safe_folder = folder_name.replace(' ', '_')[:60]
    output_dir = os.path.join(MATRICES_DIR, safe_folder)
    os.makedirs(output_dir, exist_ok=True)

    short_title = folder_name if len(folder_name) <= 50 else folder_name[:47] + '...'

    # Step 5: Generate plots
    print("\n--- Generating visualizations ---")

    # Plot 1: Warping path
    plot_warping_path(
        path_with_indices, fps_trans, fps_user,
        f'DTW Warping Path — "{short_title}"\nSimilarity: {avg_sim:.4f}',
        os.path.join(output_dir, 'dtw_warping_path.png')
    )

    # Plot 2: Similarity timeline
    plot_similarity_timeline(
        path_with_indices, fps_trans,
        f'Per-Pair Similarity Along Alignment — "{short_title}"',
        os.path.join(output_dir, 'dtw_similarity_timeline.png')
    )

    # Plot 3: Frame connections (two parallel timelines)
    plot_frame_index_connections(
        path_with_indices, fps_trans, fps_user,
        f'Frame-to-Frame Connections — "{short_title}"',
        os.path.join(output_dir, 'dtw_frame_connections.png')
    )

    # Plot 4: Matched frames dashboard with actual video thumbnails
    plot_matched_frames_dashboard(
        path_with_indices, trans_raw, user_raw,
        fps_trans, fps_user,
        f'Matched Frame Pairs — "{short_title}"\nSimilarity: {avg_sim:.4f}',
        os.path.join(output_dir, 'dtw_matched_frames.png'),
        n_samples=args.n_thumbnails
    )

    # Print alignment table
    print("\n" + "=" * 80)
    print("ALIGNMENT TABLE (first 30 pairs)")
    print("=" * 80)
    print(f"{'Step':>5}  {'Trans Frame':>12}  {'Trans Time':>11}  "
          f"{'User Frame':>11}  {'User Time':>10}  {'Similarity':>11}")
    print("-" * 75)

    for i, (tf, uf, sim) in enumerate(path_with_indices[:30]):
        tt = tf / fps_trans
        ut = uf / fps_user
        marker = " ***" if sim < 0.4 else ""
        print(f"{i+1:>5}  {tf:>12}  {tt:>10.2f}s  {uf:>11}  {ut:>9.2f}s  {sim:>10.4f}{marker}")

    if len(path_with_indices) > 30:
        print(f"  ... ({len(path_with_indices) - 30} more pairs)")

    # Summary statistics
    all_sims = [p[2] for p in path_with_indices]
    print(f"\n--- Summary ---")
    print(f"  Total aligned pairs: {len(path_with_indices)}")
    print(f"  Average similarity:  {np.mean(all_sims):.4f}")
    print(f"  Min similarity:      {np.min(all_sims):.4f}")
    print(f"  Max similarity:      {np.max(all_sims):.4f}")
    print(f"  Std deviation:       {np.std(all_sims):.4f}")
    print(f"  Pairs with sim > 0.7: {sum(1 for s in all_sims if s > 0.7)}/{len(all_sims)}")
    print(f"  Pairs with sim < 0.4: {sum(1 for s in all_sims if s < 0.4)}/{len(all_sims)}")

    print(f"\n  All plots saved to: {output_dir}/")
    print("  Done!")


if __name__ == '__main__':
    main()
