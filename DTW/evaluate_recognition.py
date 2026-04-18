"""
Evaluate Sign Language Recognition Accuracy
============================================
For each user video, compare it against ALL translator videos across all folders
using Method 3 (DTW Hybrid). Check if the system correctly identifies which
phrase the user is performing (i.e., the correct folder's translator gets the
highest similarity score).

Outputs:
  - Top-1, Top-3, Top-5 accuracy
  - Per-folder breakdown
  - Confusion cases (where the system got it wrong)
  - Summary CSV for further analysis
"""

import numpy as np
import os
import sys
import glob
import time
import csv
import json
import io
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
    method1_regular_cosine,
    method2_frame_wise_cosine,
)


VIDEOS_DIR = os.path.join(os.path.dirname(__file__), 'Videos')
CACHE_DIR = os.path.join(os.path.dirname(__file__), 'cache', 'recognition')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'recognition_results')


def discover_folders():
    """Find all folders in Videos/ that have both translator and user videos.
    Returns dict: { folder_name: { 'translator': [paths], 'user': [paths] } }
    """
    folders = {}
    for d in sorted(os.listdir(VIDEOS_DIR)):
        full = os.path.join(VIDEOS_DIR, d)
        if not os.path.isdir(full):
            continue
        trans = sorted(glob.glob(os.path.join(full, 'translator_*.mp4')))
        users = sorted(glob.glob(os.path.join(full, 'user_*.mp4')))
        if trans and users:
            folders[d] = {'translator': trans, 'user': users}
    return folders


def precompute_translator_features(folders):
    """Extract and cache features for every translator video.
    Returns dict: { folder_name: (features_matrix, kept_indices) }
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(CACHE_DIR, 'translator_features.npz')

    translator_data = {}
    total = len(folders)

    for idx, (folder, info) in enumerate(folders.items(), 1):
        trans_path = info['translator'][0]
        safe = folder.replace(' ', '_')[:60]
        feat_cache = os.path.join(CACHE_DIR, f'trans_{safe}.npz')

        if os.path.exists(feat_cache):
            data = np.load(feat_cache, allow_pickle=True)
            features = data['features']
            indices = data['indices'].tolist()
            print(f"[{idx}/{total}] Loaded cached translator: {folder}  ({len(indices)} frames)")
        else:
            print(f"[{idx}/{total}] Extracting translator: {folder}")
            features, indices = extract_all_frames_and_filter(trans_path, similarity_threshold=0.99)
            if len(features) == 0:
                print(f"    WARNING: No features extracted, skipping")
                continue
            np.savez_compressed(feat_cache, features=features, indices=np.array(indices))

        translator_data[folder] = (features, indices)

    print(f"\nPre-computed features for {len(translator_data)} translator videos\n")
    return translator_data


def compare_user_vs_all_translators(user_features, user_indices, translator_data, method='dtw'):
    """Compare one user video against all translator videos.

    Args:
        user_features: (N, 126) feature matrix
        user_indices: kept frame indices
        translator_data: dict { folder: (trans_features, trans_indices) }
        method: 'dtw' (Method 3), 'cosine' (Method 1), or 'framewise' (Method 2)

    Returns:
        list of (folder_name, score) sorted by score descending
    """
    results = []

    for folder, (trans_feat, trans_idx) in translator_data.items():
        if len(trans_feat) == 0:
            continue

        if method == 'dtw':
            score, _, _ = dtw_align_with_prefiltered_frames(
                trans_feat, user_features, trans_idx, user_indices, window_ratio=0.3
            )
        elif method == 'cosine':
            score = method1_regular_cosine(trans_feat, user_features)
        elif method == 'framewise':
            score = method2_frame_wise_cosine(trans_feat, user_features)
        else:
            raise ValueError(f"Unknown method: {method}")

        results.append((folder, score))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


def run_evaluation(folders, translator_data, method='dtw', limit_users=None):
    """Run recognition evaluation for all user videos.

    Returns:
        all_results: list of dicts with per-user-video results
        summary: dict with accuracy stats
    """
    all_results = []
    top1_correct = 0
    top3_correct = 0
    top5_correct = 0
    total_users = 0

    folder_list = list(folders.items())
    total_folders = len(folder_list)

    for f_idx, (folder, info) in enumerate(folder_list, 1):
        user_videos = info['user']
        if limit_users:
            user_videos = user_videos[:limit_users]

        for u_idx, user_path in enumerate(user_videos):
            total_users += 1
            user_basename = os.path.basename(user_path)
            print("=" * 70)
            print(f"[Folder {f_idx}/{total_folders}] [{total_users}] {folder}")
            print(f"  User video: {user_basename}")
            print(f"  Method: {method.upper()}")

            # Extract user features
            t0 = time.time()
            user_feat, user_idx = extract_all_frames_and_filter(user_path, similarity_threshold=0.99)
            extract_time = time.time() - t0

            if len(user_feat) == 0:
                print(f"  WARNING: No features extracted, skipping")
                continue

            # Compare against all translators
            t1 = time.time()
            rankings = compare_user_vs_all_translators(
                user_feat, user_idx, translator_data, method=method
            )
            compare_time = time.time() - t1

            if not rankings:
                print(f"  WARNING: No rankings produced, skipping")
                continue

            # Check accuracy
            top_folders = [r[0] for r in rankings]
            rank = top_folders.index(folder) + 1 if folder in top_folders else len(top_folders) + 1

            is_top1 = (rank == 1)
            is_top3 = (rank <= 3)
            is_top5 = (rank <= 5)

            if is_top1:
                top1_correct += 1
            if is_top3:
                top3_correct += 1
            if is_top5:
                top5_correct += 1

            result = {
                'folder': folder,
                'user_video': user_basename,
                'correct_rank': rank,
                'top1': is_top1,
                'top3': is_top3,
                'top5': is_top5,
                'correct_score': rankings[rank - 1][1] if rank <= len(rankings) else 0,
                'top1_folder': rankings[0][0],
                'top1_score': rankings[0][1],
                'top5_rankings': [(r[0], round(r[1], 4)) for r in rankings[:5]],
                'extract_time': round(extract_time, 2),
                'compare_time': round(compare_time, 2),
                'user_frames': len(user_idx),
            }
            all_results.append(result)

            # Print result
            status = "CORRECT" if is_top1 else f"WRONG (rank {rank})"
            print(f"\n  Result: {status}")
            print(f"  Correct score: {result['correct_score']:.4f} | Top-1 score: {result['top1_score']:.4f}")
            print(f"  Top-5 rankings:")
            for i, (rf, rs) in enumerate(result['top5_rankings']):
                marker = " <<< CORRECT" if rf == folder else ""
                print(f"    {i+1}. {rs:.4f}  {rf}{marker}")
            print(f"  Time: extract={extract_time:.1f}s, compare={compare_time:.1f}s")
            print()

    # Summary
    summary = {
        'method': method,
        'total_users': total_users,
        'total_translators': len(translator_data),
        'top1_correct': top1_correct,
        'top3_correct': top3_correct,
        'top5_correct': top5_correct,
        'top1_accuracy': round(top1_correct / total_users * 100, 2) if total_users > 0 else 0,
        'top3_accuracy': round(top3_correct / total_users * 100, 2) if total_users > 0 else 0,
        'top5_accuracy': round(top5_correct / total_users * 100, 2) if total_users > 0 else 0,
    }

    return all_results, summary


def save_results(all_results, summary, method):
    """Save results to CSV and JSON."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Save detailed CSV
    csv_path = os.path.join(RESULTS_DIR, f'recognition_{method}.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'folder', 'user_video', 'correct_rank', 'top1', 'top3', 'top5',
            'correct_score', 'top1_folder', 'top1_score', 'user_frames',
            'extract_time', 'compare_time'
        ])
        writer.writeheader()
        for r in all_results:
            row = {k: v for k, v in r.items() if k != 'top5_rankings'}
            writer.writerow(row)

    # Save summary JSON
    json_path = os.path.join(RESULTS_DIR, f'recognition_{method}_summary.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'summary': summary,
            'detailed': all_results,
        }, f, ensure_ascii=False, indent=2)

    # Save readable text report
    txt_path = os.path.join(RESULTS_DIR, f'recognition_{method}_report.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"SIGN LANGUAGE RECOGNITION EVALUATION — Method: {method.upper()}\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Total user videos tested:    {summary['total_users']}\n")
        f.write(f"Total translator references:  {summary['total_translators']}\n\n")

        f.write(f"Top-1 Accuracy: {summary['top1_correct']}/{summary['total_users']} = {summary['top1_accuracy']}%\n")
        f.write(f"Top-3 Accuracy: {summary['top3_correct']}/{summary['total_users']} = {summary['top3_accuracy']}%\n")
        f.write(f"Top-5 Accuracy: {summary['top5_correct']}/{summary['total_users']} = {summary['top5_accuracy']}%\n\n")

        # Misclassifications
        wrong = [r for r in all_results if not r['top1']]
        if wrong:
            f.write("-" * 80 + "\n")
            f.write(f"MISCLASSIFICATIONS ({len(wrong)} cases)\n")
            f.write("-" * 80 + "\n\n")
            for r in wrong:
                f.write(f"  User: {r['user_video']}\n")
                f.write(f"  Correct folder: {r['folder']}\n")
                f.write(f"  Predicted folder: {r['top1_folder']} (score {r['top1_score']:.4f})\n")
                f.write(f"  Correct rank: {r['correct_rank']} (score {r['correct_score']:.4f})\n")
                f.write(f"  Top-5:\n")
                for i, (rf, rs) in enumerate(r['top5_rankings']):
                    marker = " <<< CORRECT" if rf == r['folder'] else ""
                    f.write(f"    {i+1}. {rs:.4f}  {rf}{marker}\n")
                f.write("\n")

        # Correct cases
        correct = [r for r in all_results if r['top1']]
        if correct:
            f.write("-" * 80 + "\n")
            f.write(f"CORRECT IDENTIFICATIONS ({len(correct)} cases)\n")
            f.write("-" * 80 + "\n\n")
            for r in correct:
                f.write(f"  {r['folder']} | {r['user_video']} | score={r['correct_score']:.4f}\n")

    print(f"\nResults saved to:")
    print(f"  CSV:    {csv_path}")
    print(f"  JSON:   {json_path}")
    print(f"  Report: {txt_path}")


def print_final_summary(summary):
    """Print final summary to console."""
    print("\n" + "=" * 80)
    print("FINAL RECOGNITION RESULTS")
    print("=" * 80)
    print(f"  Method:             {summary['method'].upper()}")
    print(f"  User videos tested: {summary['total_users']}")
    print(f"  Translator refs:    {summary['total_translators']}")
    print()
    print(f"  Top-1 Accuracy: {summary['top1_correct']}/{summary['total_users']} = {summary['top1_accuracy']}%")
    print(f"  Top-3 Accuracy: {summary['top3_correct']}/{summary['total_users']} = {summary['top3_accuracy']}%")
    print(f"  Top-5 Accuracy: {summary['top5_correct']}/{summary['total_users']} = {summary['top5_accuracy']}%")
    print("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate SLR recognition accuracy')
    parser.add_argument('--method', choices=['dtw', 'cosine', 'framewise', 'all'],
                        default='dtw', help='Similarity method to use (default: dtw)')
    parser.add_argument('--limit-folders', type=int, default=None,
                        help='Process only first N folders (for quick testing)')
    parser.add_argument('--limit-users', type=int, default=None,
                        help='Process only first N user videos per folder')
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Step 1: Discover all folders
    print("Discovering video folders...")
    folders = discover_folders()
    print(f"Found {len(folders)} folders with translator+user videos\n")

    if args.limit_folders:
        folders = dict(list(folders.items())[:args.limit_folders])
        print(f"  (Limited to first {args.limit_folders} folders)\n")

    # Step 2: Pre-compute translator features (cached)
    print("=" * 80)
    print("PHASE 1: Pre-computing translator features")
    print("=" * 80)
    translator_data = precompute_translator_features(folders)

    # Step 3: Run evaluation
    methods_to_run = ['dtw', 'cosine', 'framewise'] if args.method == 'all' else [args.method]

    for method in methods_to_run:
        print("\n" + "=" * 80)
        print(f"PHASE 2: Evaluating recognition — Method: {method.upper()}")
        print("=" * 80 + "\n")

        all_results, summary = run_evaluation(
            folders, translator_data, method=method, limit_users=args.limit_users
        )
        save_results(all_results, summary, method)
        print_final_summary(summary)
