"""
Similarity Engine Module

Computes similarity scores between video feature matrices using various methods.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import pearsonr, spearmanr
from enum import Enum


class SimilarityMethod(Enum):
    """Available similarity computation methods."""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    PEARSON = "pearson"
    SPEARMAN = "spearman"
    MANHATTAN = "manhattan"
    FRAME_WISE_COSINE = "frame_wise_cosine"
    TEMPORAL_CORRELATION = "temporal_correlation"
    DTW = "dtw"  # Dynamic Time Warping with Sakoe-Chiba constraint


class SimilarityEngine:
    """
    Computes similarity between video feature matrices.
    
    Supports multiple similarity methods:
    - Cosine Similarity: Best for comparing vector directions
    - Euclidean Distance: Measures absolute distance
    - Pearson Correlation: Linear correlation
    - Frame-wise Cosine: Compares corresponding frames
    - Temporal Correlation: Considers temporal patterns
    - DTW: Dynamic Time Warping with Sakoe-Chiba window constraint
    
    Attributes:
        default_method: Default similarity method to use
        dtw_window_ratio: Window size as ratio of sequence length (0.0-1.0)
    """
    
    def __init__(self, default_method: str = "cosine", dtw_window_ratio: float = 0.3):
        """
        Initialize the SimilarityEngine.
        
        Args:
            default_method: Default similarity computation method
            dtw_window_ratio: Window size for DTW as ratio of sequence length
                              e.g., 0.3 means frame i can only match frame j where |i-j| <= 0.3 * n_frames
        """
        self.default_method = SimilarityMethod(default_method)
        self.dtw_window_ratio = dtw_window_ratio
        
        # Method registry
        self._methods: Dict[SimilarityMethod, Callable] = {
            SimilarityMethod.COSINE: self._cosine_similarity,
            SimilarityMethod.EUCLIDEAN: self._euclidean_similarity,
            SimilarityMethod.PEARSON: self._pearson_similarity,
            SimilarityMethod.SPEARMAN: self._spearman_similarity,
            SimilarityMethod.MANHATTAN: self._manhattan_similarity,
            SimilarityMethod.FRAME_WISE_COSINE: self._frame_wise_cosine_similarity,
            SimilarityMethod.TEMPORAL_CORRELATION: self._temporal_correlation,
            SimilarityMethod.DTW: self._dtw_similarity,
        }
    
    def flatten_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """
        Flatten a 2D feature matrix to a 1D vector.
        
        Args:
            matrix: Feature matrix of shape (n_frames, n_features)
            
        Returns:
            Flattened vector of shape (n_frames * n_features,)
        """
        return matrix.flatten()
    
    def _cosine_similarity(
        self, 
        matrix_a: np.ndarray, 
        matrix_b: np.ndarray
    ) -> float:
        """
        Compute cosine similarity between two flattened matrices.
        
        Cosine similarity = 1 - cosine_distance
        Range: [-1, 1] where 1 means identical direction
        
        Args:
            matrix_a: First feature matrix
            matrix_b: Second feature matrix
            
        Returns:
            Cosine similarity score
        """
        vec_a = self.flatten_matrix(matrix_a)
        vec_b = self.flatten_matrix(matrix_b)
        
        # Handle zero vectors
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        similarity = 1 - cosine(vec_a, vec_b)
        return float(similarity)
    
    def _euclidean_similarity(
        self, 
        matrix_a: np.ndarray, 
        matrix_b: np.ndarray
    ) -> float:
        """
        Compute similarity based on Euclidean distance.
        
        Similarity = 1 / (1 + distance)
        Range: (0, 1] where 1 means identical
        
        Args:
            matrix_a: First feature matrix
            matrix_b: Second feature matrix
            
        Returns:
            Euclidean similarity score
        """
        vec_a = self.flatten_matrix(matrix_a)
        vec_b = self.flatten_matrix(matrix_b)
        
        distance = euclidean(vec_a, vec_b)
        similarity = 1 / (1 + distance)
        return float(similarity)
    
    def _manhattan_similarity(
        self, 
        matrix_a: np.ndarray, 
        matrix_b: np.ndarray
    ) -> float:
        """
        Compute similarity based on Manhattan (L1) distance.
        
        Args:
            matrix_a: First feature matrix
            matrix_b: Second feature matrix
            
        Returns:
            Manhattan similarity score
        """
        vec_a = self.flatten_matrix(matrix_a)
        vec_b = self.flatten_matrix(matrix_b)
        
        distance = np.sum(np.abs(vec_a - vec_b))
        similarity = 1 / (1 + distance)
        return float(similarity)
    
    def _pearson_similarity(
        self, 
        matrix_a: np.ndarray, 
        matrix_b: np.ndarray
    ) -> float:
        """
        Compute Pearson correlation between matrices.
        
        Range: [-1, 1]
        
        Args:
            matrix_a: First feature matrix
            matrix_b: Second feature matrix
            
        Returns:
            Pearson correlation coefficient
        """
        vec_a = self.flatten_matrix(matrix_a)
        vec_b = self.flatten_matrix(matrix_b)
        
        # Handle constant vectors
        if np.std(vec_a) == 0 or np.std(vec_b) == 0:
            return 0.0
        
        correlation, _ = pearsonr(vec_a, vec_b)
        return float(correlation)
    
    def _spearman_similarity(
        self, 
        matrix_a: np.ndarray, 
        matrix_b: np.ndarray
    ) -> float:
        """
        Compute Spearman rank correlation between matrices.
        
        Range: [-1, 1]
        
        Args:
            matrix_a: First feature matrix
            matrix_b: Second feature matrix
            
        Returns:
            Spearman correlation coefficient
        """
        vec_a = self.flatten_matrix(matrix_a)
        vec_b = self.flatten_matrix(matrix_b)
        
        correlation, _ = spearmanr(vec_a, vec_b)
        return float(correlation)
    
    def _frame_wise_cosine_similarity(
        self, 
        matrix_a: np.ndarray, 
        matrix_b: np.ndarray
    ) -> float:
        """
        Compute average cosine similarity between corresponding frames.
        
        This method preserves temporal alignment by comparing
        frame 0 with frame 0, frame 1 with frame 1, etc.
        
        Args:
            matrix_a: First feature matrix (n_frames, n_features)
            matrix_b: Second feature matrix (n_frames, n_features)
            
        Returns:
            Average frame-wise cosine similarity
        """
        if matrix_a.shape != matrix_b.shape:
            raise ValueError("Matrices must have the same shape for frame-wise comparison")
        
        n_frames = matrix_a.shape[0]
        similarities = []
        
        for i in range(n_frames):
            norm_a = np.linalg.norm(matrix_a[i])
            norm_b = np.linalg.norm(matrix_b[i])
            
            if norm_a == 0 or norm_b == 0:
                similarities.append(0.0)
            else:
                sim = 1 - cosine(matrix_a[i], matrix_b[i])
                similarities.append(sim)
        
        return float(np.mean(similarities))
    
    def _temporal_correlation(
        self, 
        matrix_a: np.ndarray, 
        matrix_b: np.ndarray
    ) -> float:
        """
        Compute temporal correlation considering motion patterns.
        
        This method computes the correlation between temporal
        derivatives (frame-to-frame changes) to capture motion patterns.
        
        Args:
            matrix_a: First feature matrix (n_frames, n_features)
            matrix_b: Second feature matrix (n_frames, n_features)
            
        Returns:
            Temporal correlation score
        """
        # Compute temporal derivatives (motion)
        motion_a = np.diff(matrix_a, axis=0)
        motion_b = np.diff(matrix_b, axis=0)
        
        # Flatten and compute correlation
        vec_a = motion_a.flatten()
        vec_b = motion_b.flatten()
        
        if np.std(vec_a) == 0 or np.std(vec_b) == 0:
            return 0.0
        
        correlation, _ = pearsonr(vec_a, vec_b)
        return float(correlation)
    
    def _compute_frame_distance(
        self,
        frame_a: np.ndarray,
        frame_b: np.ndarray
    ) -> float:
        """
        Compute distance between two frame feature vectors.
        
        Uses cosine distance for direction-invariant comparison.
        
        Args:
            frame_a: Feature vector of frame A
            frame_b: Feature vector of frame B
            
        Returns:
            Distance value (0 = identical, higher = more different)
        """
        norm_a = np.linalg.norm(frame_a)
        norm_b = np.linalg.norm(frame_b)
        
        if norm_a == 0 or norm_b == 0:
            return 1.0  # Maximum distance if either is zero
        
        return cosine(frame_a, frame_b)
    
    def _dtw_with_sakoe_chiba(
        self,
        matrix_a: np.ndarray,
        matrix_b: np.ndarray,
        window_ratio: Optional[float] = None
    ) -> Tuple[float, List[Tuple[int, int]]]:
        """
        Compute DTW distance with Sakoe-Chiba band constraint.
        
        The Sakoe-Chiba band restricts the warping path to stay within
        a diagonal band, preventing pathological alignments where
        early frames match late frames.
        
        Constraint: |i - j| <= window
        
        Args:
            matrix_a: First feature matrix (n_frames, n_features)
            matrix_b: Second feature matrix (m_frames, n_features)
            window_ratio: Window size as ratio of sequence length (default: self.dtw_window_ratio)
            
        Returns:
            Tuple of (dtw_distance, alignment_path)
            alignment_path is list of (i, j) pairs showing matched frames
        """
        if window_ratio is None:
            window_ratio = self.dtw_window_ratio
        
        n = matrix_a.shape[0]
        m = matrix_b.shape[0]
        
        # Compute window size based on longer sequence
        window = max(int(window_ratio * max(n, m)), 1)
        
        # Also ensure window accommodates length difference
        window = max(window, abs(n - m))
        
        # Initialize cost matrix with infinity
        dtw_matrix = np.full((n + 1, m + 1), np.inf)
        dtw_matrix[0, 0] = 0
        
        # Fill the DTW matrix with Sakoe-Chiba constraint
        for i in range(1, n + 1):
            # Sakoe-Chiba band: only compute within window
            j_start = max(1, i - window)
            j_end = min(m + 1, i + window + 1)
            
            for j in range(j_start, j_end):
                # Compute distance between frames
                cost = self._compute_frame_distance(
                    matrix_a[i - 1], 
                    matrix_b[j - 1]
                )
                
                # DTW recurrence: min of three possible predecessors
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i - 1, j],      # Insertion (skip frame in B)
                    dtw_matrix[i, j - 1],      # Deletion (skip frame in A)
                    dtw_matrix[i - 1, j - 1]   # Match (align frames)
                )
        
        # Get final distance
        dtw_distance = dtw_matrix[n, m]
        
        # Backtrack to find alignment path
        alignment_path = self._backtrack_dtw(dtw_matrix, n, m, window)
        
        return dtw_distance, alignment_path
    
    def _backtrack_dtw(
        self,
        dtw_matrix: np.ndarray,
        n: int,
        m: int,
        window: int
    ) -> List[Tuple[int, int]]:
        """
        Backtrack through DTW matrix to find optimal alignment path.
        
        Args:
            dtw_matrix: Filled DTW cost matrix
            n: Length of sequence A
            m: Length of sequence B
            window: Sakoe-Chiba window size
            
        Returns:
            List of (i, j) tuples representing aligned frame pairs
        """
        path = []
        i, j = n, m
        
        while i > 0 and j > 0:
            path.append((i - 1, j - 1))  # Convert to 0-indexed
            
            # Check valid predecessors within window constraint
            candidates = []
            
            if i > 0 and j > 0 and abs((i-1) - (j-1)) <= window:
                candidates.append((dtw_matrix[i - 1, j - 1], (i - 1, j - 1)))
            if i > 0 and abs((i-1) - j) <= window:
                candidates.append((dtw_matrix[i - 1, j], (i - 1, j)))
            if j > 0 and abs(i - (j-1)) <= window:
                candidates.append((dtw_matrix[i, j - 1], (i, j - 1)))
            
            if not candidates:
                break
            
            # Move to predecessor with minimum cost
            _, (i, j) = min(candidates, key=lambda x: x[0])
        
        path.reverse()
        return path
    
    def _dtw_similarity(
        self,
        matrix_a: np.ndarray,
        matrix_b: np.ndarray
    ) -> float:
        """
        Compute DTW-based similarity with Sakoe-Chiba constraint.
        
        This method:
        1. Computes DTW distance with window constraint
        2. Converts distance to similarity score in [0, 1]
        
        The window constraint ensures:
        - First frames can only match early frames (not last frames)
        - Last frames can only match late frames (not first frames)
        - Prevents pathological warping paths
        
        Args:
            matrix_a: First feature matrix (n_frames, n_features)
            matrix_b: Second feature matrix (m_frames, n_features)
            
        Returns:
            Similarity score in [0, 1] where 1 = identical
        """
        dtw_distance, alignment_path = self._dtw_with_sakoe_chiba(matrix_a, matrix_b)
        
        # Convert distance to similarity
        # Using exponential decay: similarity = exp(-distance)
        # This maps [0, inf) -> (0, 1]
        if np.isinf(dtw_distance):
            return 0.0
        
        # Normalize by path length to make it comparable across different video lengths
        path_length = len(alignment_path) if alignment_path else 1
        normalized_distance = dtw_distance / path_length
        
        # Convert to similarity using exponential decay
        similarity = np.exp(-normalized_distance)
        
        return float(similarity)
    
    def get_dtw_alignment(
        self,
        matrix_a: np.ndarray,
        matrix_b: np.ndarray,
        window_ratio: Optional[float] = None
    ) -> Tuple[float, List[Tuple[int, int]]]:
        """
        Get DTW alignment path between two videos.
        
        Useful for visualizing which frames are matched.
        
        Args:
            matrix_a: First feature matrix
            matrix_b: Second feature matrix
            window_ratio: Optional custom window ratio
            
        Returns:
            Tuple of (similarity_score, alignment_path)
            alignment_path is list of (frame_a_idx, frame_b_idx) pairs
        """
        dtw_distance, alignment_path = self._dtw_with_sakoe_chiba(
            matrix_a, matrix_b, window_ratio
        )
        
        path_length = len(alignment_path) if alignment_path else 1
        normalized_distance = dtw_distance / path_length
        similarity = np.exp(-normalized_distance)
        
        return float(similarity), alignment_path
    
    def compute_similarity(
        self, 
        matrix_a: np.ndarray, 
        matrix_b: np.ndarray,
        method: Optional[str] = None
    ) -> float:
        """
        Compute similarity between two feature matrices.
        
        Args:
            matrix_a: First feature matrix
            matrix_b: Second feature matrix
            method: Similarity method (uses default if None)
            
        Returns:
            Similarity score
        """
        if method is None:
            method_enum = self.default_method
        else:
            method_enum = SimilarityMethod(method)
        
        similarity_func = self._methods[method_enum]
        return similarity_func(matrix_a, matrix_b)
    
    def compare_to_many(
        self, 
        reference: np.ndarray,
        candidates: Dict[str, np.ndarray],
        method: Optional[str] = None,
        show_progress: bool = False
    ) -> Dict[str, float]:
        """
        Compare a reference matrix to multiple candidate matrices.
        
        Args:
            reference: Reference feature matrix
            candidates: Dictionary mapping IDs to feature matrices
            method: Similarity method to use
            show_progress: Whether to show progress bar
            
        Returns:
            Dictionary mapping IDs to similarity scores
        """
        results = {}
        
        iterator = candidates.items()
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(list(iterator), desc="Computing similarities")
            except ImportError:
                pass
        
        for video_id, matrix in iterator:
            try:
                similarity = self.compute_similarity(reference, matrix, method)
                results[video_id] = similarity
            except Exception as e:
                print(f"Error comparing {video_id}: {e}")
                results[video_id] = 0.0
        
        return results
    
    def rank_by_similarity(
        self, 
        similarities: Dict[str, float],
        top_k: Optional[int] = None,
        descending: bool = True
    ) -> List[Tuple[str, float]]:
        """
        Rank videos by similarity score.
        
        Args:
            similarities: Dictionary of video_id -> similarity_score
            top_k: Return only top K results (None for all)
            descending: If True, highest similarity first
            
        Returns:
            List of (video_id, score) tuples sorted by score
        """
        sorted_results = sorted(
            similarities.items(),
            key=lambda x: x[1],
            reverse=descending
        )
        
        if top_k is not None:
            sorted_results = sorted_results[:top_k]
        
        return sorted_results
    
    def compute_similarity_matrix(
        self, 
        feature_matrices: Dict[str, np.ndarray],
        method: Optional[str] = None,
        show_progress: bool = False
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Compute pairwise similarity matrix for all videos.
        
        Args:
            feature_matrices: Dictionary mapping IDs to feature matrices
            method: Similarity method to use
            show_progress: Whether to show progress bar
            
        Returns:
            Tuple of (similarity_matrix, video_ids)
        """
        video_ids = list(feature_matrices.keys())
        n_videos = len(video_ids)
        
        similarity_matrix = np.zeros((n_videos, n_videos))
        
        total_comparisons = n_videos * (n_videos + 1) // 2
        comparison_count = 0
        
        for i, id_a in enumerate(video_ids):
            for j in range(i, n_videos):
                id_b = video_ids[j]
                
                if i == j:
                    similarity_matrix[i, j] = 1.0  # Self-similarity
                else:
                    similarity = self.compute_similarity(
                        feature_matrices[id_a],
                        feature_matrices[id_b],
                        method
                    )
                    similarity_matrix[i, j] = similarity
                    similarity_matrix[j, i] = similarity  # Symmetric
                
                comparison_count += 1
                
                if show_progress and comparison_count % 100 == 0:
                    print(f"Progress: {comparison_count}/{total_comparisons}")
        
        return similarity_matrix, video_ids
    
    def get_available_methods(self) -> List[str]:
        """Get list of available similarity methods."""
        return [m.value for m in SimilarityMethod]
    
    def weighted_ensemble_similarity(
        self,
        matrix_a: np.ndarray,
        matrix_b: np.ndarray,
        weights: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Compute weighted ensemble of multiple similarity methods.
        
        Args:
            matrix_a: First feature matrix
            matrix_b: Second feature matrix
            weights: Dictionary of method -> weight (defaults to equal weights)
            
        Returns:
            Weighted average similarity score
        """
        if weights is None:
            # Default weights: emphasize cosine and temporal
            weights = {
                'cosine': 0.4,
                'frame_wise_cosine': 0.3,
                'temporal_correlation': 0.3
            }
        
        total_weight = sum(weights.values())
        weighted_sum = 0.0
        
        for method, weight in weights.items():
            similarity = self.compute_similarity(matrix_a, matrix_b, method)
            weighted_sum += weight * similarity
        
        return weighted_sum / total_weight
