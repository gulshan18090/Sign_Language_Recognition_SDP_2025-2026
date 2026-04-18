"""
Streaming (Online) DTW for Real-Time Sign Language Similarity Assessment

Based on Subsequence DTW from:
  Müller, Fundamentals of Music Processing (FMP), Springer 2015, Section 7.2
  https://www.audiolabs-erlangen.de/resources/MIR/FMP/C7/C7S2_SubsequenceDTW.html

Architecture:
  1. Pre-computation phase: reference video → extract all frames → filter hands-only
     → drop redundant → store reference feature matrix R (done once)
  2. Online phase: as each user frame arrives →
     - Extract MediaPipe features (skip if no hand)
     - Drop if redundant vs last kept frame (Euclidean, threshold 0.99)
     - Update ONE ROW of accumulated cost matrix D against R
     - Compute running cosine similarity for current best alignment
     - Bidirectional optimization within window for every matched pair

Key difference from offline:
  - Reference R is pre-computed and cached
  - DTW matrix is built incrementally (one row per incoming user frame)
  - Running score available at any time (partial alignment)
  - Complexity: O(|R|) per incoming frame instead of O(|R| * |U|) for full DTW
"""

import numpy as np
from collections import defaultdict

MAX_USER_FRAME_REUSE = 3
import cv2
import os
import sys
import time
import glob
from scipy.spatial.distance import cosine, euclidean
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.append(os.path.dirname(__file__))
from similarity.feature_extractor import HandFeatureExtractor


# =============================================================================
# Shared preprocessing (same as offline)
# =============================================================================

def extract_features_from_video(video_path):
    """Extract ALL frames from video and return MediaPipe features.
    Only keeps frames where hands are detected.

    Returns:
        features: np.ndarray (n_kept, 126)
        kept_indices: list of original frame indices
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ERROR: Cannot open video {video_path}")
        return np.zeros((0, 126)), []

    all_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    if not all_frames:
        return np.zeros((0, 126)), []

    print(f"  Total frames extracted: {len(all_frames)}")

    extractor = HandFeatureExtractor(max_hands=2)
    all_features = extractor.frames_to_feature_matrix(all_frames)

    # Keep only frames with hands detected (non-zero feature vector)
    features_with_hands = []
    indices_with_hands = []
    for i, feat in enumerate(all_features):
        if np.any(feat != 0):
            features_with_hands.append(feat)
            indices_with_hands.append(i)

    if not features_with_hands:
        print("  No hands detected in any frame")
        return np.zeros((0, 126)), []

    features_with_hands = np.array(features_with_hands)
    print(f"  Frames with hands: {len(features_with_hands)} "
          f"(dropped {len(all_features) - len(features_with_hands)} without hands)")
    return features_with_hands, indices_with_hands


def drop_similar_frames(matrix, threshold=0.99):
    """Drop consecutive redundant frames using normalized Euclidean.
    sim = 1 - (d^2 / 2) on unit-normalized vectors.

    Returns:
        reduced: np.ndarray
        kept: list of indices into input matrix
    """
    kept = [0]
    for i in range(1, len(matrix)):
        prev = matrix[kept[-1]]
        curr = matrix[i]
        np_prev = np.linalg.norm(prev)
        np_curr = np.linalg.norm(curr)
        if np_prev > 0 and np_curr > 0:
            d = np.linalg.norm(prev / np_prev - curr / np_curr)
            sim = 1.0 - (d ** 2) / 2.0
        else:
            sim = 0.0
        if sim < threshold:
            kept.append(i)
    return matrix[kept], kept


def precompute_reference(video_path):
    """Full pre-computation for the reference (translator) video.
    Done once, result is cached.

    Returns:
        ref_features: np.ndarray (n_ref, 126) — filtered & deduplicated
        ref_original_indices: list — mapping back to raw frame numbers
    """
    print("PRE-COMPUTING REFERENCE...")
    raw_feat, raw_idx = extract_features_from_video(video_path)
    if len(raw_feat) == 0:
        return np.zeros((0, 126)), []

    filtered, kept_rel = drop_similar_frames(raw_feat, threshold=0.99)
    kept_abs = [raw_idx[i] for i in kept_rel]
    print(f"  After redundancy removal: {len(kept_abs)} reference frames")
    return filtered, kept_abs


# =============================================================================
# Subsequence DTW — accumulated cost matrix (FMP algorithm)
# =============================================================================

def compute_cost_matrix(X, Y, window_ratio=0.25):
    """Euclidean cost matrix between reference X (N, D) and user Y (M, D).
    
    Uses Sakoe-Chiba band to limit warping: frame n can only match
    user frames within ±window_ratio of the diagonal.
    Cells outside the band are set to infinity.
    """
    N, M = len(X), len(Y)
    window_size = max(1, int(window_ratio * max(N, M)))
    C = np.full((N, M), np.inf)
    for n in range(N):
        # Scale n to the user frame space, then apply window
        center_m = int(n * M / N) if N > 0 else 0
        m_lo = max(0, center_m - window_size)
        m_hi = min(M, center_m + window_size + 1)
        for m in range(m_lo, m_hi):
            C[n, m] = euclidean(X[n], Y[m])
    return C


def subsequence_dtw_accumulated(C):
    """Compute accumulated cost matrix D for subsequence DTW.

    Key difference from global DTW:
      - First row D[0, m] = C[0, m]  (can start matching anywhere in Y)
      - First col D[n, 0] = cumsum(C[:n+1, 0])  (must use all of X)

    Respects the Sakoe-Chiba band from the cost matrix:
      cells where C[n,m] = inf stay inf in D (unreachable).

    Step sizes: {(1,0), (0,1), (1,1)}

    Args:
        C: cost matrix (N, M)
    Returns:
        D: accumulated cost matrix (N, M)
    """
    N, M = C.shape
    D = np.full((N, M), np.inf)

    # First row: just C[0, m] — allows free start position in Y
    for m in range(M):
        if np.isfinite(C[0, m]):
            D[0, m] = C[0, m]

    # First column: cumulative sum (must consume all of X from the start)
    for n in range(1, N):
        if np.isfinite(C[n, 0]) and np.isfinite(D[n - 1, 0]):
            D[n, 0] = C[n, 0] + D[n - 1, 0]

    for n in range(1, N):
        for m in range(1, M):
            if np.isfinite(C[n, m]):
                prev = min(D[n - 1, m - 1] if np.isfinite(D[n - 1, m - 1]) else np.inf,
                           D[n - 1, m]     if np.isfinite(D[n - 1, m])     else np.inf,
                           D[n, m - 1]     if np.isfinite(D[n, m - 1])     else np.inf)
                if np.isfinite(prev):
                    D[n, m] = C[n, m] + prev
    return D


def subsequence_dtw_backtrack(D, m_end=-1):
    """Backtrack optimal warping path for subsequence DTW.

    Starts at (N-1, b*) and walks back to row 0.
    Returns path as list of (n, m) index pairs, ordered start→end.

    Args:
        D: accumulated cost matrix (N, M)
        m_end: column to start backtracking (-1 = argmin of last row)
    Returns:
        path: np.ndarray of shape (L, 2)
        a_star: start index in Y
        b_star: end index in Y
    """
    N, M = D.shape
    n = N - 1
    if m_end < 0:
        # Find argmin of last row, ignoring inf values
        last_row = D[N - 1, :]
        finite_mask = np.isfinite(last_row)
        if not np.any(finite_mask):
            # All inf — return empty path
            return np.array([[0, 0]]), 0, 0
        m = int(np.argmin(np.where(finite_mask, last_row, np.inf)))
    else:
        m = m_end
    b_star = m

    path = [(n, m)]
    while n > 0:
        if m == 0:
            cell = (n - 1, 0)
        else:
            candidates = []
            for cn, cm in [(n-1, m-1), (n-1, m), (n, m-1)]:
                if cn >= 0 and cm >= 0 and np.isfinite(D[cn, cm]):
                    candidates.append((D[cn, cm], cn, cm))
            if not candidates:
                cell = (n - 1, m)  # force upward
            else:
                _, cn, cm = min(candidates)
                cell = (cn, cm)
        path.append(cell)
        n, m = cell

    path.reverse()
    a_star = path[0][1]
    return np.array(path), a_star, b_star


# =============================================================================
# Streaming (online) DTW — incremental row-by-row computation
# =============================================================================

class StreamingDTW:
    """Incremental subsequence DTW that processes one user frame at a time.

    The reference feature matrix R (N×D) is fixed after pre-computation.
    For each incoming user frame u_t (D,):
      1. Check if hand is present (non-zero), skip otherwise
      2. Check redundancy vs last kept frame, skip if similar
      3. Compute one new column of cost matrix C[:, t]
      4. Update accumulated cost matrix D[:, t] using previous column
      5. Record matching function value Δ(t) = D[N-1, t] / N
      6. Running score = cosine similarity of current best alignment

    Memory: O(N) — only need current and previous column of D.
    Time per frame: O(N) — one pass over reference.
    """

    def __init__(self, ref_features, ref_indices, redundancy_threshold=0.99):
        """
        Args:
            ref_features: (N, 126) pre-computed reference features
            ref_indices: original frame indices of reference
            redundancy_threshold: drop threshold for user frames
        """
        self.R = ref_features          # (N, D)
        self.ref_indices = ref_indices
        self.N = len(ref_features)
        self.D_dim = ref_features.shape[1]  # 126
        self.threshold = redundancy_threshold

        # Accumulated cost: only keep two columns (prev, curr)
        self.D_prev = np.full(self.N, np.inf)   # previous column of D
        self.D_curr = np.full(self.N, np.inf)   # current column of D

        # Store all kept user features for post-alignment scoring
        self.user_features = []    # list of (126,) vectors
        self.user_indices = []     # original frame numbers
        self.matching_function = []  # Δ(t) for each kept user frame

        # Full D matrix (last row) for matching function visualization
        self.D_last_row_history = []

        self.n_received = 0   # total frames received
        self.n_kept = 0       # frames kept after filtering
        self.last_kept_feat = None  # for redundancy check

        # Timing
        self._frame_times = []

    def _is_redundant(self, feat):
        """Check if feat is too similar to last kept frame."""
        if self.last_kept_feat is None:
            return False
        prev = self.last_kept_feat
        np_prev = np.linalg.norm(prev)
        np_curr = np.linalg.norm(feat)
        if np_prev > 0 and np_curr > 0:
            d = np.linalg.norm(prev / np_prev - feat / np_curr)
            sim = 1.0 - (d ** 2) / 2.0
            return sim >= self.threshold
        return False

    def process_feature_vector(self, feat, frame_idx):
        """Process a pre-extracted feature vector (no MediaPipe call).

        Use this when you extract features once and feed to multiple
        StreamingDTW instances (e.g., camera mode with multiple references).

        Args:
            feat: np.ndarray (126,) — pre-extracted feature vector
            frame_idx: frame number

        Returns:
            dict with 'kept', 'running_score', 'delta', 'elapsed_ms'
        """
        t0 = time.perf_counter()
        self.n_received += 1

        # Skip if no hand (zero vector)
        if not np.any(feat != 0):
            elapsed = (time.perf_counter() - t0) * 1000
            self._frame_times.append(elapsed)
            return {'kept': False, 'running_score': self.get_running_score(),
                    'delta': None, 'elapsed_ms': elapsed}

        # Skip if redundant
        if self._is_redundant(feat):
            elapsed = (time.perf_counter() - t0) * 1000
            self._frame_times.append(elapsed)
            return {'kept': False, 'running_score': self.get_running_score(),
                    'delta': None, 'elapsed_ms': elapsed}

        return self._update_dtw_column(feat, frame_idx, t0)

    def process_frame(self, frame_rgb, frame_idx, extractor):
        """Process a single incoming user frame.

        Args:
            frame_rgb: RGB numpy image
            frame_idx: original frame number in user video
            extractor: HandFeatureExtractor instance

        Returns:
            dict with:
              - 'kept': bool — was this frame used?
              - 'running_score': float — current best matching score
              - 'delta': float — matching function value at this point
              - 'elapsed_ms': float — processing time in ms
        """
        t0 = time.perf_counter()
        self.n_received += 1

        # Extract features
        feat = extractor.extract_features_from_frame(frame_rgb)

        # Skip if no hand detected
        if not np.any(feat != 0):
            elapsed = (time.perf_counter() - t0) * 1000
            self._frame_times.append(elapsed)
            return {'kept': False, 'running_score': self.get_running_score(),
                    'delta': None, 'elapsed_ms': elapsed}

        # Skip if redundant
        if self._is_redundant(feat):
            elapsed = (time.perf_counter() - t0) * 1000
            self._frame_times.append(elapsed)
            return {'kept': False, 'running_score': self.get_running_score(),
                    'delta': None, 'elapsed_ms': elapsed}

        return self._update_dtw_column(feat, frame_idx, t0)

    def _update_dtw_column(self, feat, frame_idx, t0):
        """Shared DTW column update for both process_frame and process_feature_vector."""
        # --- This frame is kept ---
        self.last_kept_feat = feat
        self.user_features.append(feat)
        self.user_indices.append(frame_idx)
        m = self.n_kept  # column index in virtual D matrix
        self.n_kept += 1

        # Compute new column of cost C[:, m]
        cost_col = np.array([euclidean(self.R[n], feat) for n in range(self.N)])

        # Update accumulated cost matrix column D[:, m]
        new_D = np.full(self.N, np.inf)

        if m == 0:
            # First user frame — D[0, 0] = C[0, 0]; D[n, 0] = cumsum
            # But for subsequence DTW: first row allows free start, first col cumulative
            # In our streaming formulation: R is rows (reference), user frames are columns
            # D[n, 0] = cumsum(C[0:n+1, 0])  — cumulative from top
            # BUT for subsequence DTW first ROW = C[0, m], meaning D[0, m] = C[0, m]
            # And first COLUMN = cumsum, meaning D[n, 0] = sum(C[0..n, 0])
            new_D[0] = cost_col[0]
            for n in range(1, self.N):
                new_D[n] = cost_col[n] + new_D[n - 1]
        else:
            # D[0, m] = C[0, m]  (subsequence DTW: free start in user stream)
            new_D[0] = cost_col[0]
            for n in range(1, self.N):
                new_D[n] = cost_col[n] + min(
                    self.D_prev[n - 1],   # diagonal
                    self.D_prev[n],        # horizontal (insertion)
                    new_D[n - 1]           # vertical (deletion)
                )

        self.D_prev = new_D.copy()
        self.D_curr = new_D

        # Matching function: Δ(m) = D[N-1, m] / N
        delta = new_D[self.N - 1] / self.N
        self.matching_function.append(delta)
        self.D_last_row_history.append(new_D[self.N - 1])

        elapsed = (time.perf_counter() - t0) * 1000
        self._frame_times.append(elapsed)

        return {'kept': True, 'running_score': self.get_running_score(),
                'delta': delta, 'elapsed_ms': elapsed}

    def get_running_score(self):
        """Current best cosine similarity based on accumulated cost so far."""
        if self.n_kept == 0:
            return 0.0
        # The best end-point is the column with minimum D[N-1, :] so far
        # We approximate by using the latest column's full D to find best alignment
        # For a true running score, we do a quick cosine on the best-aligned pair
        # at the last row minimum
        best_m = int(np.argmin(
            [self.D_last_row_history[i] for i in range(len(self.D_last_row_history))]
        ))
        # Cosine similarity between last reference frame and best-matched user frame
        ref_last = self.R[-1]
        user_best = self.user_features[best_m]
        n1, n2 = np.linalg.norm(ref_last), np.linalg.norm(user_best)
        if n1 > 0 and n2 > 0:
            return float(1 - cosine(ref_last, user_best))
        return 0.0

    def finalize(self):
        """After all user frames are processed, compute final alignment and score.

        This performs:
          1. Full backtracking on the stored accumulated cost
          2. Bidirectional optimization of each matched pair (same as offline)
          3. Returns final average cosine similarity

        Returns:
            dict with final results
        """
        if self.n_kept == 0:
            return {'score': 0.0, 'path': [], 'frame_sims': [],
                    'n_ref': self.N, 'n_user_kept': 0, 'n_aligned': 0,
                    'optimized': 0, 'total_sim_sum': 0.0}

        # Rebuild full accumulated cost matrix for backtracking
        user_mat = np.array(self.user_features)
        M = len(user_mat)

        C = compute_cost_matrix(self.R, user_mat)
        D = subsequence_dtw_accumulated(C)
        path_arr, a_star, b_star = subsequence_dtw_backtrack(D)

        print(f"  Subsequence DTW: best match in user frames [{a_star}..{b_star}]")
        print(f"  Alignment path length: {len(path_arr)}")

        # Window for bidirectional optimization (20% of kept frames)
        window_size = max(1, int(0.25 * max(self.N, M)))

        # Score each aligned pair with cosine, optimize within window
        # Enforce: monotonicity (only forward) + max reuse (≤5 per user frame)
        frame_sims = []
        path_with_indices = []
        improved = 0
        last_chosen_m = 0          # track last chosen user frame for monotonicity
        usage_count = defaultdict(int)  # track how many ref frames used each user frame

        for pidx in range(len(path_arr)):
            n_idx, m_idx = int(path_arr[pidx, 0]), int(path_arr[pidx, 1])
            vec_r = self.R[n_idx]

            # Determine search window from DTW path neighbors
            if pidx > 0:
                min_m = int(path_arr[pidx - 1, 1])
            else:
                min_m = max(0, m_idx - window_size)

            if pidx < len(path_arr) - 1:
                max_m = int(path_arr[pidx + 1, 1])
            else:
                max_m = min(M, m_idx + window_size + 1)

            # KEY: clamp lower bound to last_chosen_m (never go backwards)
            min_m = max(min_m, last_chosen_m)

            # Search ALL candidates in valid range (including original m_idx)
            # Skip frames that violate max-reuse — don't even compute similarity
            best_sim = -1.0
            best_m = -1

            for alt_m in range(min_m, max_m):
                if usage_count[alt_m] >= MAX_USER_FRAME_REUSE:
                    continue  # at reuse limit, skip entirely
                vec_alt = user_mat[alt_m]
                na = np.linalg.norm(vec_alt)
                if na > 0:
                    alt_sim = float(1 - cosine(vec_r, vec_alt))
                    if alt_sim > best_sim:
                        best_sim = alt_sim
                        best_m = alt_m

            # Fallback: no valid candidate in window — scan forward for next available
            if best_m < 0:
                scan = max_m
                while scan < M and usage_count[scan] >= MAX_USER_FRAME_REUSE:
                    scan += 1
                if scan < M:
                    best_m = scan
                    vec_alt = user_mat[scan]
                    na = np.linalg.norm(vec_alt)
                    best_sim = float(1 - cosine(vec_r, vec_alt)) if na > 0 else 0.0
                else:
                    # Truly exhausted — stay at last chosen with penalty
                    best_m = last_chosen_m
                    best_sim = 0.0

            if best_m != m_idx:
                improved += 1

            last_chosen_m = best_m
            usage_count[best_m] += 1

            frame_sims.append(best_sim)
            orig_r = self.ref_indices[n_idx]
            orig_u = self.user_indices[best_m]
            path_with_indices.append((orig_r, orig_u, best_sim))

        total_sum = sum(frame_sims)
        avg_sim = np.mean(frame_sims) if frame_sims else 0.0
        good = sum(1 for s in frame_sims if s >= 0.45)

        print(f"  Optimized matches: {improved}")
        print(f"  Total similarity sum: {total_sum:.2f}")
        print(f"  Good matches (≥0.45): {good}/{len(frame_sims)} "
              f"({100 * good / len(frame_sims):.1f}%)")

        return {
            'score': avg_sim,
            'path': path_with_indices,
            'frame_sims': frame_sims,
            'n_ref': self.N,
            'n_user_kept': self.n_kept,
            'n_aligned': len(path_arr),
            'optimized': improved,
            'total_sim_sum': total_sum,
            'a_star': a_star,
            'b_star': b_star,
            'matching_function': self.matching_function,
            'avg_frame_ms': np.mean(self._frame_times) if self._frame_times else 0,
            'total_frames_received': self.n_received,
        }


# =============================================================================
# Offline full DTW for comparison (uses same subsequence DTW)
# =============================================================================

def offline_full_dtw(ref_features, ref_indices, user_features, user_indices):
    """Full offline DTW with subsequence alignment and bidirectional optimization.

    Returns same dict structure as StreamingDTW.finalize().
    """
    N = len(ref_features)
    M = len(user_features)

    t0 = time.perf_counter()

    C = compute_cost_matrix(ref_features, user_features)
    D = subsequence_dtw_accumulated(C)
    path_arr, a_star, b_star = subsequence_dtw_backtrack(D)

    print(f"  Subsequence DTW: best match in user frames [{a_star}..{b_star}]")
    print(f"  Alignment path length: {len(path_arr)}")

    window_size = max(1, int(0.25 * max(N, M)))

    frame_sims = []
    path_with_indices = []
    improved = 0
    last_chosen_m = 0
    usage_count = defaultdict(int)

    for pidx in range(len(path_arr)):
        n_idx, m_idx = int(path_arr[pidx, 0]), int(path_arr[pidx, 1])
        vec_r = ref_features[n_idx]

        if pidx > 0:
            min_m = int(path_arr[pidx - 1, 1])
        else:
            min_m = max(0, m_idx - window_size)

        if pidx < len(path_arr) - 1:
            max_m = int(path_arr[pidx + 1, 1])
        else:
            max_m = min(M, m_idx + window_size + 1)

        # Never go backwards past last chosen frame
        min_m = max(min_m, last_chosen_m)

        best_sim = -1.0
        best_m = -1

        for alt_m in range(min_m, max_m):
            if usage_count[alt_m] >= MAX_USER_FRAME_REUSE:
                continue
            vec_alt = user_features[alt_m]
            na = np.linalg.norm(vec_alt)
            if na > 0:
                alt_sim = float(1 - cosine(vec_r, vec_alt))
                if alt_sim > best_sim:
                    best_sim = alt_sim
                    best_m = alt_m

        # Fallback: scan forward for next available user frame
        if best_m < 0:
            scan = max_m
            while scan < M and usage_count[scan] >= MAX_USER_FRAME_REUSE:
                scan += 1
            if scan < M:
                best_m = scan
                vec_alt = user_features[scan]
                na = np.linalg.norm(vec_alt)
                best_sim = float(1 - cosine(vec_r, vec_alt)) if na > 0 else 0.0
            else:
                best_m = last_chosen_m
                best_sim = 0.0

        if best_m != m_idx:
            improved += 1

        last_chosen_m = best_m
        usage_count[best_m] += 1

        frame_sims.append(best_sim)
        orig_r = ref_indices[n_idx]
        orig_u = user_indices[best_m]
        path_with_indices.append((orig_r, orig_u, best_sim))

    elapsed = (time.perf_counter() - t0) * 1000

    total_sum = sum(frame_sims)
    avg_sim = np.mean(frame_sims) if frame_sims else 0.0
    good = sum(1 for s in frame_sims if s >= 0.45)

    print(f"  Optimized matches: {improved}")
    print(f"  Total similarity sum: {total_sum:.2f}")
    print(f"  Good matches (≥0.45): {good}/{len(frame_sims)} "
          f"({100 * good / len(frame_sims):.1f}%)")

    return {
        'score': avg_sim,
        'path': path_with_indices,
        'frame_sims': frame_sims,
        'n_ref': N,
        'n_user_kept': M,
        'n_aligned': len(path_arr),
        'optimized': improved,
        'total_sim_sum': total_sum,
        'a_star': a_star,
        'b_star': b_star,
        'elapsed_ms': elapsed,
    }


# =============================================================================
# Visualization
# =============================================================================

def extract_frame_from_video(video_path, frame_idx, target_size=(160, 120)):
    """Extract a specific frame from video."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if ret and frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return cv2.resize(frame, target_size)
    return None


def visualize_alignment(folder, path_with_sims, translator_video, user_video,
                        n_ref, n_user, title_suffix=""):
    """Side-by-side visualization of aligned frames."""
    n_pairs = len(path_with_sims)
    if n_pairs == 0:
        print("  No pairs to visualize")
        return

    n_cols = 10
    n_rows = (n_pairs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.3, n_rows * 1.5))
    fig.suptitle(f'Folder {folder}: Subsequence DTW Alignment {title_suffix}\n'
                 f'Ref={n_ref} frames, User={n_user} frames | '
                 f'Aligned: {n_pairs} pairs',
                 fontsize=14, fontweight='bold')

    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for idx, (t_idx, u_idx, sim) in enumerate(path_with_sims):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]

        t_frame = extract_frame_from_video(translator_video, t_idx)
        u_frame = extract_frame_from_video(user_video, u_idx)

        if t_frame is not None and u_frame is not None:
            combined = np.hstack([t_frame, u_frame])
            ax.imshow(combined)
            ax.axvline(x=t_frame.shape[1] - 0.5, color='white', linewidth=2)

            color = 'green' if sim >= 0.8 else ('orange' if sim >= 0.5 else 'red')
            rect = mpatches.Rectangle((0, 0), combined.shape[1] - 1,
                                       combined.shape[0] - 1,
                                       linewidth=3, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            ax.set_title(f'T{t_idx}<->U{u_idx}\n{sim:.3f}',
                         fontsize=8, color=color, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'Error', ha='center', va='center')
        ax.axis('off')

    for idx in range(n_pairs, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].axis('off')

    plt.tight_layout()
    output = f'matrices/{folder}/streaming_dtw_alignment{title_suffix}.png'
    plt.savefig(output, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output}")


def visualize_matching_function(folder, matching_fn, a_star, b_star):
    """Plot the matching function Δ(m) showing where best subsequence match is."""
    if not matching_fn:
        return
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(matching_fn, 'k-', linewidth=1)
    ax.axvline(x=a_star, color='green', linestyle='--', alpha=0.7, label=f'a*={a_star}')
    ax.axvline(x=b_star, color='red', linestyle='--', alpha=0.7, label=f'b*={b_star}')
    ax.axvspan(a_star, b_star, alpha=0.15, color='blue', label='Best subsequence')
    ax.set_xlabel('User frame index (kept)')
    ax.set_ylabel('Δ_DTW (normalized cost)')
    ax.set_title(f'Folder {folder}: Matching Function — Best Subsequence [{a_star}..{b_star}]')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    output = f'matrices/{folder}/matching_function.png'
    plt.savefig(output, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output}")


# =============================================================================
# Main: run both modes on all folders
# =============================================================================

def run_comparison():
    folders = ['51', '79', '8']

    print("=" * 80)
    print("STREAMING vs OFFLINE DTW COMPARISON")
    print("=" * 80)
    print()
    print("Shared pipeline: MediaPipe → hand-only → drop redundant → DTW + cosine")
    print("Offline : full subsequence DTW + bidirectional optimization (batch)")
    print("Online  : incremental row-by-row DTW + finalize with optimization")
    print()

    summary = []

    for folder in folders:
        print("=" * 80)
        print(f"FOLDER {folder}")
        print("=" * 80)

        translator_videos = glob.glob(f'Videos/{folder}/translator_*.mp4')
        user_videos = glob.glob(f'Videos/{folder}/user_*.mp4')

        if not translator_videos or not user_videos:
            print(f"  ERROR: Missing video files in folder {folder}")
            continue

        translator_video = translator_videos[0]
        user_video = user_videos[0]
        print(f"  Ref : {os.path.basename(translator_video)}")
        print(f"  User: {os.path.basename(user_video)}")
        print()

        # ------ Pre-compute reference (shared by both modes) ------
        ref_features, ref_indices = precompute_reference(translator_video)
        if len(ref_features) == 0:
            print("  ERROR: No reference features")
            continue
        print()

        # ------ Extract user video features (for offline & simulation) ------
        print("EXTRACTING USER VIDEO...")
        user_raw, user_raw_idx = extract_features_from_video(user_video)
        if len(user_raw) == 0:
            print("  ERROR: No user features")
            continue
        user_filtered, user_kept_rel = drop_similar_frames(user_raw, threshold=0.99)
        user_indices = [user_raw_idx[i] for i in user_kept_rel]
        print(f"  After redundancy removal: {len(user_indices)} user frames")
        print()

        # ==================== OFFLINE MODE ====================
        print("-" * 40)
        print("OFFLINE MODE (full batch)")
        print("-" * 40)
        t_offline_start = time.perf_counter()
        offline_result = offline_full_dtw(ref_features, ref_indices,
                                          user_filtered, user_indices)
        t_offline = (time.perf_counter() - t_offline_start) * 1000
        print(f"\n  OFFLINE SCORE: {offline_result['score']:.4f}")
        print(f"  Aligned pairs: {offline_result['n_aligned']}")
        print(f"  Time: {t_offline:.1f} ms (excl. feature extraction)")
        print()

        # ==================== ONLINE MODE ====================
        print("-" * 40)
        print("ONLINE MODE (streaming simulation)")
        print("-" * 40)

        # Re-read user video frames for streaming simulation
        cap = cv2.VideoCapture(user_video)
        user_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            user_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

        extractor = HandFeatureExtractor(max_hands=2)
        streamer = StreamingDTW(ref_features, ref_indices)

        t_online_start = time.perf_counter()
        for fi, frame_rgb in enumerate(user_frames):
            result = streamer.process_frame(frame_rgb, fi, extractor)
            # Print progress every 50 frames
            if (fi + 1) % 50 == 0:
                status = "KEPT" if result['kept'] else "skip"
                print(f"  Frame {fi + 1}/{len(user_frames)}: {status}, "
                      f"running_score={result['running_score']:.3f}")

        t_online_process = (time.perf_counter() - t_online_start) * 1000

        online_result = streamer.finalize()
        t_online_total = (time.perf_counter() - t_online_start) * 1000

        print(f"\n  ONLINE SCORE: {online_result['score']:.4f}")
        print(f"  Aligned pairs: {online_result['n_aligned']}")
        print(f"  Frames received: {online_result['total_frames_received']}, "
              f"kept: {online_result['n_user_kept']}")
        print(f"  Avg latency/frame: {online_result['avg_frame_ms']:.2f} ms")
        print(f"  Stream processing: {t_online_process:.1f} ms")
        print(f"  Total (incl finalize): {t_online_total:.1f} ms")
        print()

        # ==================== COMPARISON ====================
        print("-" * 40)
        print("COMPARISON")
        print("-" * 40)
        diff = online_result['score'] - offline_result['score']
        print(f"  Offline score:  {offline_result['score']:.4f}")
        print(f"  Online score:   {online_result['score']:.4f}")
        print(f"  Difference:     {diff:+.4f}")
        print(f"  Offline time:   {t_offline:.1f} ms (DTW only)")
        print(f"  Online time:    {t_online_process:.1f} ms (stream + extract)")
        print(f"  Latency/frame:  {online_result['avg_frame_ms']:.2f} ms")
        print()

        summary.append({
            'folder': folder,
            'offline_score': offline_result['score'],
            'online_score': online_result['score'],
            'offline_aligned': offline_result['n_aligned'],
            'online_aligned': online_result['n_aligned'],
            'offline_time_ms': t_offline,
            'online_time_ms': t_online_process,
            'latency_ms': online_result['avg_frame_ms'],
            'ref_frames': len(ref_indices),
            'user_frames': len(user_indices),
        })

        # Visualizations
        visualize_alignment(folder, offline_result['path'],
                            translator_video, user_video,
                            len(ref_indices), len(user_indices),
                            title_suffix="_offline")
        visualize_alignment(folder, online_result['path'],
                            translator_video, user_video,
                            len(ref_indices), online_result['n_user_kept'],
                            title_suffix="_online")

        if 'matching_function' in online_result and online_result['matching_function']:
            visualize_matching_function(folder, online_result['matching_function'],
                                        online_result.get('a_star', 0),
                                        online_result.get('b_star', 0))
        print()

    # ==================== SUMMARY TABLE ====================
    if summary:
        print("=" * 80)
        print("SUMMARY TABLE")
        print("=" * 80)
        print(f"{'Folder':<8} {'Ref':<6} {'User':<6} "
              f"{'Offline':<10} {'Online':<10} {'Diff':<10} "
              f"{'Off ms':<10} {'On ms':<10} {'Lat/f ms':<10}")
        print("-" * 80)
        for s in summary:
            print(f"{s['folder']:<8} {s['ref_frames']:<6} {s['user_frames']:<6} "
                  f"{s['offline_score']:<10.4f} {s['online_score']:<10.4f} "
                  f"{s['online_score'] - s['offline_score']:+<10.4f} "
                  f"{s['offline_time_ms']:<10.1f} {s['online_time_ms']:<10.1f} "
                  f"{s['latency_ms']:<10.2f}")
        print()


if __name__ == "__main__":
    run_comparison()
