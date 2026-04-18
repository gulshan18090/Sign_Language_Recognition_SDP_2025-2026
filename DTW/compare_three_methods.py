"""
Compare three similarity methods:
1. Regular Cosine (flatten all frames)
2. Frame-wise Cosine (average per-frame similarity)
3. DTW-Aligned Filtered Frame-wise Cosine (drop similar frames first + DTW align + frame-wise cos)
"""

import numpy as np
from collections import defaultdict
from scipy.spatial.distance import cosine
from scipy.ndimage import uniform_filter1d
import sys
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.append(os.path.dirname(__file__))
from similarity.similarity_engine import SimilarityEngine
from similarity.feature_extractor import HandFeatureExtractor


def extract_all_frames_and_filter(video_path, similarity_threshold=0.99):
    """
    Extract ALL frames from video, drop similar ones, return filtered features.
    
    Args:
        video_path: Path to video file
        similarity_threshold: Drop frames with similarity >= this value
        
    Returns:
        feature_matrix: Feature matrix of filtered frames (n_filtered, 126)
        kept_indices: Original frame indices that were kept
    """
    # Read all frames from video using OpenCV
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"    ERROR: Cannot open video {video_path}")
        return np.zeros((0, 126)), []
    
    all_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        all_frames.append(frame_rgb)
    cap.release()
    
    if len(all_frames) == 0:
        return np.zeros((0, 126)), []
    
    print(f"    Extracted {len(all_frames)} frames from video")
    
    # Extract features from all frames
    try:
        feature_extractor = HandFeatureExtractor(max_hands=2)
        all_features = feature_extractor.frames_to_feature_matrix(all_frames)
    except AttributeError as e:
        print(f"    ERROR initializing MediaPipe: {e}")
        print(f"    Falling back to loading pre-computed features...")
        return np.zeros((0, 126)), []
    
    # First, keep only frames where hands are detected (non-zero features)
    frames_with_hands = []
    indices_with_hands = []
    for i, features in enumerate(all_features):
        if np.any(features != 0):  # Hand detected
            frames_with_hands.append(features)
            indices_with_hands.append(i)
    
    if len(frames_with_hands) == 0:
        print(f"    No hands detected in any frame")
        return np.zeros((0, 126)), []
    
    frames_with_hands = np.array(frames_with_hands)
    print(f"    Frames with hands: {len(frames_with_hands)} (dropped {len(all_features) - len(frames_with_hands)} without hands)")
    
    # Drop similar consecutive frames
    filtered_features, kept_indices_relative = drop_similar_frames(frames_with_hands, similarity_threshold)
    
    # Map back to original frame indices
    kept_indices = [indices_with_hands[i] for i in kept_indices_relative]
    
    print(f"    After filtering similar: {len(kept_indices)} frames kept")
    
    return filtered_features, kept_indices


def drop_similar_frames(matrix, similarity_threshold=0.99):
    """
    Drop consecutive frames that are nearly identical using normalized Euclidean.
    Normalizes vectors before comparison for better detection of identical poses.
    Returns reduced matrix and indices of kept frames.
    """
    kept_indices = [0]  # Always keep first frame
    
    for i in range(1, matrix.shape[0]):
        prev_frame = matrix[kept_indices[-1]]
        curr_frame = matrix[i]
        
        # Normalize both frames to unit length
        norm_prev = np.linalg.norm(prev_frame)
        norm_curr = np.linalg.norm(curr_frame)
        
        if norm_prev > 0 and norm_curr > 0:
            prev_normalized = prev_frame / norm_prev
            curr_normalized = curr_frame / norm_curr
            
            # Euclidean distance on normalized vectors (equivalent to cosine-based distance)
            # distance = ||a - b|| for unit vectors
            # similarity = 1 - (distance^2 / 2)
            distance = np.linalg.norm(prev_normalized - curr_normalized)
            sim = 1.0 - (distance ** 2) / 2.0
        else:
            sim = 0.0
        
        # Keep frame if different enough
        if sim < similarity_threshold:
            kept_indices.append(i)
    
    reduced_matrix = matrix[kept_indices]
    return reduced_matrix, kept_indices


def dtw_align_with_prefiltered_frames(matrix1, matrix2, indices1, indices2, window_ratio=0.2):
    """
    DTW alignment on pre-filtered frames with 20% window, find best consecutive groups.
    
    Args:
        matrix1, matrix2: Pre-filtered matrices (already reduced)
        indices1, indices2: Original frame indices that were kept
        window_ratio: DTW window constraint (20% of kept frames)
    
    Returns:
        - similarity: Average cosine similarity of best aligned groups
        - path: Alignment path with original frame indices (best groups)
        - frame_similarities: Similarity score for each aligned pair
    """
    from scipy.spatial.distance import euclidean
    
    n1, n2 = matrix1.shape[0], matrix2.shape[0]
    
    # Use 20% window based on kept frame count
    window_size = int(window_ratio * max(n1, n2))
    print(f"    DTW window: ±{window_size} frames (20% of {max(n1, n2)} kept frames)")
    
    # Compute DTW distance matrix with Sakoe-Chiba constraint
    dtw_matrix = np.full((n1 + 1, n2 + 1), np.inf)
    dtw_matrix[0, 0] = 0
    
    for i in range(1, n1 + 1):
        for j in range(max(1, i - window_size), min(n2 + 1, i + window_size + 1)):
            cost = euclidean(matrix1[i-1], matrix2[j-1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i-1, j],    # insertion
                dtw_matrix[i, j-1],    # deletion
                dtw_matrix[i-1, j-1]   # match
            )
    
    # Backtrack to get full alignment path
    path = []
    i, j = n1, n2
    while i > 0 and j > 0:
        path.append((i-1, j-1))
        
        costs = [
            dtw_matrix[i-1, j-1],
            dtw_matrix[i-1, j],
            dtw_matrix[i, j-1]
        ]
        min_idx = np.argmin(costs)
        
        if min_idx == 0:
            i -= 1
            j -= 1
        elif min_idx == 1:
            i -= 1
        else:
            j -= 1
    
    path.reverse()
    
    # Optimize ALL frame matches by searching within DTW window
    # Goal: maximize sum of similarities while maintaining consecutiveness
    MAX_USER_FRAME_REUSE = 3
    frame_similarities = []
    path_with_original_indices = []
    improved_count = 0
    last_chosen_idx2 = 0
    usage_count = defaultdict(int)
    
    for path_idx, (idx1, idx2) in enumerate(path):
        vec1 = matrix1[idx1]
        
        # Determine search window from DTW path neighbors
        if path_idx > 0:
            min_idx2 = path[path_idx - 1][1]
        else:
            min_idx2 = max(0, idx2 - window_size)
        
        if path_idx < len(path) - 1:
            max_idx2 = path[path_idx + 1][1]
        else:
            max_idx2 = min(matrix2.shape[0], idx2 + window_size + 1)
        
        # Never go backwards past last chosen frame
        min_idx2 = max(min_idx2, last_chosen_idx2)
        
        # Search valid range — skip frames that break max-reuse
        best_sim = -1.0
        best_idx2 = -1
        
        for alt_idx2 in range(min_idx2, max_idx2):
            if usage_count[alt_idx2] >= MAX_USER_FRAME_REUSE:
                continue
            
            vec2_alt = matrix2[alt_idx2]
            norm2_alt = np.linalg.norm(vec2_alt)
            
            if norm2_alt > 0:
                alt_sim = 1 - cosine(vec1, vec2_alt)
                if alt_sim > best_sim:
                    best_sim = alt_sim
                    best_idx2 = alt_idx2
        
        # Fallback: scan forward for next available user frame
        if best_idx2 < 0:
            scan = max_idx2
            while scan < matrix2.shape[0] and usage_count[scan] >= MAX_USER_FRAME_REUSE:
                scan += 1
            if scan < matrix2.shape[0]:
                best_idx2 = scan
                vec2_alt = matrix2[scan]
                norm2_alt = np.linalg.norm(vec2_alt)
                best_sim = (1 - cosine(vec1, vec2_alt)) if norm2_alt > 0 else 0.0
            else:
                best_idx2 = last_chosen_idx2
                best_sim = 0.0
        
        if best_idx2 != idx2:
            improved_count += 1
        
        last_chosen_idx2 = best_idx2
        usage_count[best_idx2] += 1
        
        frame_similarities.append(best_sim)
        
        # Map back to original frame indices
        original_idx1 = indices1[idx1]
        original_idx2 = indices2[best_idx2]
        path_with_original_indices.append((original_idx1, original_idx2, best_sim))
    
    if len(frame_similarities) == 0:
        return 0.0, [], []
    
    total_sim_sum = sum(frame_similarities)
    print(f"    Total aligned pairs: {len(frame_similarities)}")
    print(f"    Optimized matches: {improved_count}")
    print(f"    Total similarity sum: {total_sim_sum:.2f}")
    
    # Check quality: count how many pairs have good similarity
    good_matches = sum(1 for s in frame_similarities if s >= 0.45)
    print(f"    Good matches (≥0.45): {good_matches}/{len(frame_similarities)} ({100*good_matches/len(frame_similarities):.1f}%)")
    
    # Use ALL optimized pairs for scoring
    avg_similarity = np.mean(frame_similarities)
    
    return avg_similarity, path_with_original_indices, frame_similarities


def method1_regular_cosine(matrix1, matrix2):
    """Flatten all frames and compute single cosine similarity.
    Truncates to same number of frames first."""
    n_frames = min(matrix1.shape[0], matrix2.shape[0])
    vec1 = matrix1[:n_frames].flatten()
    vec2 = matrix2[:n_frames].flatten()
    
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return 1 - cosine(vec1, vec2)


def method2_frame_wise_cosine(matrix1, matrix2):
    """Average per-frame cosine similarity (no alignment)."""
    n_frames = min(matrix1.shape[0], matrix2.shape[0])
    
    similarities = []
    for i in range(n_frames):
        vec1 = matrix1[i]
        vec2 = matrix2[i]
        
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            sim = 0.0
        else:
            sim = 1 - cosine(vec1, vec2)
        
        similarities.append(sim)
    
    return np.mean(similarities)




def method3_dtw_allframes_filtered(translator_video, user_video):
    """
    1. Extract ALL frames from both videos
    2. Drop similar frames in each video separately using normalized Euclidean
    3. DTW align the reduced frames
    4. Calculate frame-wise cosine similarity on aligned pairs
    """
    print("  Extracting and filtering translator video...")
    reduced1, indices1 = extract_all_frames_and_filter(translator_video, similarity_threshold=0.99)
    
    print("  Extracting and filtering user video...")
    reduced2, indices2 = extract_all_frames_and_filter(user_video, similarity_threshold=0.99)
    
    if len(reduced1) == 0 or len(reduced2) == 0:
        return 0.0, 0, 0, 0, [], []
    
    # Step 2 & 3: DTW align and calculate similarities
    avg_sim, path, frame_sims = dtw_align_with_prefiltered_frames(
        reduced1, reduced2, indices1, indices2, window_ratio=0.3
    )
    
    return avg_sim, len(indices1), len(indices2), len(path), path, frame_sims


def method3_dtw_prefiltered_framewise(matrix1, matrix2):
    """
    OLD METHOD: Works on pre-extracted 64 frames.
    1. Drop similar frames in each video separately using normalized Euclidean
    2. DTW align the reduced frames
    3. Calculate frame-wise cosine similarity on aligned pairs
    """
    # Step 1: Drop similar frames in each video using normalized Euclidean
    reduced1, indices1 = drop_similar_frames(matrix1, similarity_threshold=0.99)
    reduced2, indices2 = drop_similar_frames(matrix2, similarity_threshold=0.99)
    
    # Step 2 & 3: DTW align and calculate similarities
    avg_sim, path, frame_sims = dtw_align_with_prefiltered_frames(
        reduced1, reduced2, indices1, indices2, window_ratio=0.3
    )
    
    return avg_sim, len(indices1), len(indices2), len(path), path, frame_sims


def extract_frame_from_video(video_path, frame_idx, target_size=(160, 120)):
    """Extract a specific frame from video."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        cap.release()
        return None
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    
    if ret and frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, target_size)
        return frame
    
    return None


def create_visualization(folder, path_with_sims, translator_video, user_video, 
                        orig_t_frames, orig_u_frames, kept_t_frames, kept_u_frames):
    """Create side-by-side visualization of DTW-aligned frames."""
    
    print(f"Creating visualization for folder {folder}...")
    print(f"  Filtered frames: T={kept_t_frames}, U={kept_u_frames}")
    print(f"  DTW aligned pairs: {len(path_with_sims)}")
    
    # Determine grid layout
    n_pairs = len(path_with_sims)
    n_cols = 10
    n_rows = (n_pairs + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.3, n_rows * 1.5))
    fig.suptitle(f'Folder {folder}: DTW Frame Alignment (All Frames Filtered)\n'
                 f'Filtered frames: T={kept_t_frames}, U={kept_u_frames} | '
                 f'Aligned: {n_pairs} pairs',
                 fontsize=14, fontweight='bold')
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for pair_idx, (t_idx, u_idx, sim) in enumerate(path_with_sims):
        row = pair_idx // n_cols
        col = pair_idx % n_cols
        ax = axes[row, col]
        
        # Extract frames
        t_frame = extract_frame_from_video(translator_video, t_idx)
        u_frame = extract_frame_from_video(user_video, u_idx)
        
        if t_frame is not None and u_frame is not None:
            # Combine side by side
            combined = np.hstack([t_frame, u_frame])
            ax.imshow(combined)
            
            # Add vertical separator
            ax.axvline(x=t_frame.shape[1] - 0.5, color='white', linewidth=2)
            
            # Color-code border by similarity
            if sim >= 0.8:
                color = 'green'
            elif sim >= 0.5:
                color = 'orange'
            else:
                color = 'red'
            
            rect = patches.Rectangle((0, 0), combined.shape[1]-1, combined.shape[0]-1,
                                     linewidth=3, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            
            ax.set_title(f'T{t_idx}<->U{u_idx}\n{sim:.3f}', 
                        fontsize=8, color=color, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'Frame Error', ha='center', va='center')
            ax.set_title(f'T{t_idx}<->U{u_idx}', fontsize=8)
        
        ax.axis('off')
        
        if (pair_idx + 1) % 10 == 0:
            print(f"  Processed {pair_idx + 1}/{n_pairs} pairs...")
    
    # Hide unused subplots
    for pair_idx in range(n_pairs, n_rows * n_cols):
        row = pair_idx // n_cols
        col = pair_idx % n_cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    output_path = f'matrices/{folder}/dtw_prefiltered_alignment.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved to: {output_path}")
    print()


def compare_all_folders(limit=None):
    """Compare all 3 methods across all folders with visualizations.
    
    Args:
        limit: If set, only process this many folders (for quick testing).
    """
    
    import glob
    
    # Auto-detect all folders in Videos/ that have both translator and user videos
    all_folders = sorted(
        d for d in os.listdir('Videos')
        if os.path.isdir(os.path.join('Videos', d))
        and glob.glob(f'Videos/{d}/translator_*.mp4')
        and glob.glob(f'Videos/{d}/user_*.mp4')
    )
    
    if not all_folders:
        print("ERROR: No folders found with both translator and user videos in Videos/")
        return
    
    print("=" * 80)
    print("COMPARING 3 SIMILARITY METHODS")
    print("=" * 80)
    print()
    print("Method 1: Regular Cosine (flatten all frames)")
    print("Method 2: Frame-wise Cosine (per-frame average, no alignment)")
    print("Method 3: DTW Hybrid (Euclidean drop + DTW align + Cosine score)")
    print("    NOTE: Processing ALL frames from videos")
    print(f"\nFound {len(all_folders)} folders with video pairs")
    if limit:
        all_folders = all_folders[:limit]
        print(f"  (limiting to first {limit} folders)")
    print()
    
    for folder in all_folders:
        print("=" * 80)
        print(f"FOLDER: {folder}")
        print("=" * 80)
        
        # Find video files for all-frames processing
        translator_videos = glob.glob(f'Videos/{folder}/translator_*.mp4')
        user_videos = glob.glob(f'Videos/{folder}/user_*.mp4')
        
        if not translator_videos or not user_videos:
            print(f"ERROR: Could not find video files in folder {folder}")
            print()
            continue
        
        translator_video = translator_videos[0]
        user_video = user_videos[0]
        
        print(f"Translator video: {os.path.basename(translator_video)}")
        print(f"User video: {os.path.basename(user_video)}")
        print()
        
        # Extract ALL frames from both videos (shared across all methods)
        print("  Extracting translator video features...")
        trans_all_features, trans_kept_indices = extract_all_frames_and_filter(
            translator_video, similarity_threshold=0.99
        )
        print("  Extracting user video features...")
        user_all_features, user_kept_indices = extract_all_frames_and_filter(
            user_video, similarity_threshold=0.99
        )
        
        if len(trans_all_features) == 0 or len(user_all_features) == 0:
            print("  ERROR: No features extracted, skipping folder")
            print()
            continue
        
        # Method 1: Regular Cosine (flatten all filtered frames)
        score1 = method1_regular_cosine(trans_all_features, user_all_features)
        print(f"Method 1 (Regular Cosine):           {score1:.4f}")
        
        # Method 2: Frame-wise Cosine (per-frame average, no alignment)
        score2 = method2_frame_wise_cosine(trans_all_features, user_all_features)
        print(f"Method 2 (Frame-wise Cosine):        {score2:.4f}")
        
        # Method 3: DTW Hybrid (reuse already-extracted features)
        print()
        print("Method 3 (DTW Hybrid - All Frames):")
        avg_sim, path, frame_sims = dtw_align_with_prefiltered_frames(
            trans_all_features, user_all_features,
            trans_kept_indices, user_kept_indices,
            window_ratio=0.3
        )
        kept_t = len(trans_kept_indices)
        kept_u = len(user_kept_indices)
        n_aligned = len(path)
        
        print()
        print(f"Method 3 (DTW Hybrid):               {avg_sim:.4f}")
        print(f"  >> Filtered: Translator={kept_t}, User={kept_u}")
        print(f"  >> DTW aligned pairs: {n_aligned}")
        print(f"  >> Drop method: Euclidean | Score method: Cosine")
        
        print()
        print("COMPARISON:")
        print(f"  Method 1 (Regular Cosine):  {score1:.4f}")
        print(f"  Method 2 (Frame-wise):      {score2:.4f}")
        print(f"  Method 3 (DTW Hybrid):      {avg_sim:.4f}")
        if score1 > 0:
            print(f"  M3 vs M1: {avg_sim - score1:+.4f}")
        if score2 > 0:
            print(f"  M3 vs M2: {avg_sim - score2:+.4f}")
        print()
        
        if len(path) > 0:
            # Save visualization to a safe folder name
            safe_name = folder.replace(' ', '_')[:50]
            os.makedirs(f'matrices/{safe_name}', exist_ok=True)
            create_visualization(
                safe_name, path, translator_video, user_video,
                kept_t, kept_u, kept_t, kept_u
            )
        else:
            print(f"  WARNING: Could not create visualization (no aligned frames)")
            print()


if __name__ == "__main__":
    import sys
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    compare_all_folders(limit=limit)

