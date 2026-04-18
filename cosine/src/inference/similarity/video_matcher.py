"""
Video Matcher Module

Main orchestrator for video similarity matching pipeline.
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime
from dataclasses import dataclass, asdict

from .frame_extractor import FrameExtractor
from .feature_extractor import HandFeatureExtractor
from .similarity_engine import SimilarityEngine


@dataclass
class VideoFeatures:
    """Container for video feature data."""
    video_path: str
    feature_matrix: np.ndarray
    metadata: dict
    extraction_time: float
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'video_path': self.video_path,
            'feature_matrix': self.feature_matrix.tolist(),
            'metadata': self.metadata,
            'extraction_time': self.extraction_time
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'VideoFeatures':
        """Create from dictionary."""
        return cls(
            video_path=data['video_path'],
            feature_matrix=np.array(data['feature_matrix']),
            metadata=data['metadata'],
            extraction_time=data['extraction_time']
        )


@dataclass
class SimilarityResult:
    """Container for similarity matching results."""
    reference_video: str
    rankings: List[Tuple[str, float]]
    method: str
    total_compared: int
    processing_time: float
    timestamp: str


class VideoMatcher:
    """
    Main orchestrator for video similarity matching.
    
    This class coordinates:
    - Frame extraction from videos
    - Feature extraction using MediaPipe
    - Similarity computation between videos
    - Results ranking and export
    
    Typical usage:
        matcher = VideoMatcher(n_frames=64, max_hands=2)
        ref = matcher.process_video("video_A.mp4")
        candidates = matcher.process_video_folder("videos/")
        results = matcher.find_similar(ref, candidates, top_k=5)
    
    Attributes:
        n_frames: Number of frames to extract per video
        max_hands: Maximum hands to track (1 or 2)
        similarity_method: Default similarity computation method
    """
    
    SUPPORTED_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv'}
    
    def __init__(
        self,
        n_frames: int = 64,
        max_hands: int = 2,
        similarity_method: str = 'cosine',
        normalize_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        filter_hand_frames: bool = True,
        cache_features: bool = True,
        dtw_window_ratio: float = 0.3
    ):
        """
        Initialize the VideoMatcher.
        
        Args:
            n_frames: Target number of frames to extract (default: 64)
            max_hands: Maximum number of hands to track (default: 2)
            similarity_method: Similarity computation method
            normalize_landmarks: Whether to normalize hand landmarks
            min_detection_confidence: MediaPipe detection confidence
            filter_hand_frames: Whether to filter only hand-containing frames
            cache_features: Whether to cache extracted features
            dtw_window_ratio: Window ratio for DTW (0.0-1.0), controls how far frames can match
        """
        self.n_frames = n_frames
        self.max_hands = max_hands
        self.similarity_method = similarity_method
        self.cache_features = cache_features
        self.dtw_window_ratio = dtw_window_ratio
        
        # Initialize component extractors
        self.frame_extractor = FrameExtractor(
            n_frames=n_frames,
            min_detection_confidence=min_detection_confidence,
            filter_hand_frames=filter_hand_frames
        )
        
        self.feature_extractor = HandFeatureExtractor(
            max_hands=max_hands,
            normalize=normalize_landmarks,
            min_detection_confidence=min_detection_confidence
        )
        
        self.similarity_engine = SimilarityEngine(
            default_method=similarity_method,
            dtw_window_ratio=dtw_window_ratio
        )
        
        # Feature cache
        self._feature_cache: Dict[str, VideoFeatures] = {}
    
    def process_video(
        self, 
        video_path: str,
        use_cache: bool = True
    ) -> VideoFeatures:
        """
        Process a single video to extract features.
        
        Pipeline:
        1. Extract n_frames from video (with hand filtering)
        2. Extract MediaPipe landmarks from each frame
        3. Create feature matrix (n_frames × n_features)
        
        Args:
            video_path: Path to video file
            use_cache: Whether to use cached features if available
            
        Returns:
            VideoFeatures object containing feature matrix and metadata
        """
        video_path = str(Path(video_path).resolve())
        
        # Check cache
        if use_cache and self.cache_features and video_path in self._feature_cache:
            return self._feature_cache[video_path]
        
        import time
        start_time = time.time()
        
        # Step 1: Extract frames
        frames, frame_metadata = self.frame_extractor.extract_frames(
            video_path, 
            return_metadata=True
        )
        
        # Step 2: Extract features
        feature_matrix = self.feature_extractor.frames_to_feature_matrix(frames)
        
        # Step 3: Compute hand statistics
        hand_stats = self.feature_extractor.compute_hand_statistics(feature_matrix)
        
        extraction_time = time.time() - start_time
        
        # Combine metadata
        metadata = {
            **frame_metadata,
            **hand_stats,
            'feature_shape': list(feature_matrix.shape)
        }
        
        video_features = VideoFeatures(
            video_path=video_path,
            feature_matrix=feature_matrix,
            metadata=metadata,
            extraction_time=extraction_time
        )
        
        # Cache if enabled
        if self.cache_features:
            self._feature_cache[video_path] = video_features
        
        return video_features
    
    def process_video_folder(
        self, 
        folder_path: str,
        recursive: bool = False,
        show_progress: bool = True
    ) -> Dict[str, VideoFeatures]:
        """
        Process all videos in a folder.
        
        Args:
            folder_path: Path to folder containing videos
            recursive: Whether to search subdirectories
            show_progress: Whether to show progress bar
            
        Returns:
            Dictionary mapping video paths to VideoFeatures
        """
        folder_path = Path(folder_path)
        if not folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        
        # Find all video files
        video_files = self._find_video_files(folder_path, recursive)
        
        if len(video_files) == 0:
            print(f"No video files found in {folder_path}")
            return {}
        
        print(f"Found {len(video_files)} video files")
        
        results = {}
        iterator = video_files
        
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(video_files, desc="Processing videos")
            except ImportError:
                pass
        
        for video_path in iterator:
            try:
                features = self.process_video(str(video_path))
                results[str(video_path)] = features
            except Exception as e:
                print(f"\nError processing {video_path}: {e}")
                continue
        
        return results
    
    def _find_video_files(
        self, 
        folder: Path, 
        recursive: bool
    ) -> List[Path]:
        """Find all video files in a folder."""
        pattern = "**/*" if recursive else "*"
        
        video_files = []
        for ext in self.SUPPORTED_EXTENSIONS:
            video_files.extend(folder.glob(f"{pattern}{ext}"))
            video_files.extend(folder.glob(f"{pattern}{ext.upper()}"))
        
        return sorted(set(video_files))
    
    def find_similar(
        self,
        reference: Union[str, VideoFeatures],
        candidates: Union[str, Dict[str, VideoFeatures]],
        top_k: Optional[int] = None,
        method: Optional[str] = None,
        exclude_self: bool = True
    ) -> SimilarityResult:
        """
        Find videos similar to a reference video.
        
        Args:
            reference: Reference video path or VideoFeatures
            candidates: Folder path or dict of VideoFeatures
            top_k: Return only top K results
            method: Similarity method (uses default if None)
            exclude_self: Whether to exclude reference from results
            
        Returns:
            SimilarityResult with ranked videos
        """
        import time
        start_time = time.time()
        
        # Process reference if needed
        if isinstance(reference, str):
            ref_features = self.process_video(reference)
        else:
            ref_features = reference
        
        # Process candidates if needed
        if isinstance(candidates, str):
            candidate_features = self.process_video_folder(candidates)
        else:
            candidate_features = candidates
        
        # Extract feature matrices
        candidate_matrices = {
            path: vf.feature_matrix 
            for path, vf in candidate_features.items()
        }
        
        # Compute similarities
        method = method or self.similarity_method
        similarities = self.similarity_engine.compare_to_many(
            reference=ref_features.feature_matrix,
            candidates=candidate_matrices,
            method=method
        )
        
        # Exclude self if needed
        if exclude_self and ref_features.video_path in similarities:
            del similarities[ref_features.video_path]
        
        # Rank results
        rankings = self.similarity_engine.rank_by_similarity(
            similarities, 
            top_k=top_k
        )
        
        processing_time = time.time() - start_time
        
        return SimilarityResult(
            reference_video=ref_features.video_path,
            rankings=rankings,
            method=method,
            total_compared=len(similarities),
            processing_time=processing_time,
            timestamp=datetime.now().isoformat()
        )
    
    def save_features(
        self, 
        features: Dict[str, VideoFeatures],
        output_path: str
    ):
        """
        Save extracted features to disk.
        
        Args:
            features: Dictionary of video features
            output_path: Path to output file (.json or .npz)
        """
        output_path = Path(output_path)
        
        if output_path.suffix == '.json':
            # Save as JSON (human-readable but larger)
            data = {
                path: vf.to_dict() 
                for path, vf in features.items()
            }
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
        
        elif output_path.suffix == '.npz':
            # Save as compressed numpy (smaller, faster)
            matrices = {path: vf.feature_matrix for path, vf in features.items()}
            metadata = {path: vf.metadata for path, vf in features.items()}
            
            np.savez_compressed(
                output_path,
                matrices=matrices,
                metadata=json.dumps(metadata)
            )
        
        else:
            raise ValueError(f"Unsupported format: {output_path.suffix}")
        
        print(f"Saved features to {output_path}")
    
    def load_features(self, input_path: str) -> Dict[str, VideoFeatures]:
        """
        Load previously saved features.
        
        Args:
            input_path: Path to saved features file
            
        Returns:
            Dictionary of video features
        """
        input_path = Path(input_path)
        
        if input_path.suffix == '.json':
            with open(input_path, 'r') as f:
                data = json.load(f)
            
            features = {
                path: VideoFeatures.from_dict(vf_data)
                for path, vf_data in data.items()
            }
        
        elif input_path.suffix == '.npz':
            loaded = np.load(input_path, allow_pickle=True)
            matrices = loaded['matrices'].item()
            metadata = json.loads(str(loaded['metadata']))
            
            features = {}
            for path, matrix in matrices.items():
                features[path] = VideoFeatures(
                    video_path=path,
                    feature_matrix=matrix,
                    metadata=metadata.get(path, {}),
                    extraction_time=0
                )
        
        else:
            raise ValueError(f"Unsupported format: {input_path.suffix}")
        
        print(f"Loaded {len(features)} video features from {input_path}")
        return features
    
    def get_similarity_matrix(
        self,
        features: Dict[str, VideoFeatures],
        method: Optional[str] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Compute pairwise similarity matrix for all videos.
        
        Args:
            features: Dictionary of video features
            method: Similarity method
            
        Returns:
            Tuple of (similarity_matrix, video_paths)
        """
        matrices = {
            path: vf.feature_matrix 
            for path, vf in features.items()
        }
        
        return self.similarity_engine.compute_similarity_matrix(
            matrices, 
            method=method
        )
    
    def print_results(self, result: SimilarityResult, max_display: int = 10):
        """
        Pretty print similarity results.
        
        Args:
            result: SimilarityResult to display
            max_display: Maximum results to display
        """
        print("\n" + "=" * 60)
        print("VIDEO SIMILARITY RESULTS")
        print("=" * 60)
        print(f"Reference: {Path(result.reference_video).name}")
        print(f"Method: {result.method}")
        print(f"Videos compared: {result.total_compared}")
        print(f"Processing time: {result.processing_time:.2f}s")
        print("-" * 60)
        print(f"{'Rank':<6} {'Similarity':<12} {'Video'}")
        print("-" * 60)
        
        for rank, (video_path, score) in enumerate(result.rankings[:max_display], 1):
            video_name = Path(video_path).name
            print(f"{rank:<6} {score:>10.4f}   {video_name}")
        
        if len(result.rankings) > max_display:
            print(f"... and {len(result.rankings) - max_display} more")
        
        print("=" * 60)
    
    def export_results(
        self, 
        result: SimilarityResult, 
        output_path: str
    ):
        """
        Export results to JSON file.
        
        Args:
            result: SimilarityResult to export
            output_path: Path to output JSON file
        """
        data = {
            'reference_video': result.reference_video,
            'method': result.method,
            'total_compared': result.total_compared,
            'processing_time': result.processing_time,
            'timestamp': result.timestamp,
            'rankings': [
                {'video': path, 'similarity': score}
                for path, score in result.rankings
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Results exported to {output_path}")
    
    def clear_cache(self):
        """Clear the feature cache."""
        self._feature_cache.clear()
        print("Feature cache cleared")
