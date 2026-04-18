"""
Frame Extractor Module

Extracts and samples frames containing hand movements from video files.
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import List, Tuple, Optional
from pathlib import Path


class FrameExtractor:
    """
    Extracts frames containing hand movements from video files.
    
    This class handles:
    - Video loading and frame extraction
    - Hand presence detection for filtering
    - Uniform sampling to fixed frame count
    - Frame preprocessing
    
    Attributes:
        n_frames (int): Target number of frames to extract (default: 64)
        min_detection_confidence (float): MediaPipe detection confidence
        filter_hand_frames (bool): Whether to filter only hand-containing frames
    """
    
    def __init__(
        self,
        n_frames: int = 64,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        filter_hand_frames: bool = True
    ):
        """
        Initialize the FrameExtractor.
        
        Args:
            n_frames: Target number of frames to extract
            min_detection_confidence: Minimum confidence for hand detection
            min_tracking_confidence: Minimum confidence for hand tracking
            filter_hand_frames: If True, only keep frames with hands detected
        """
        self.n_frames = n_frames
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.filter_hand_frames = filter_hand_frames
        
        # Initialize MediaPipe Hands for hand detection
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,  # For individual frame processing
            max_num_hands=2,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
    
    def extract_all_frames(self, video_path: str) -> Tuple[List[np.ndarray], dict]:
        """
        Extract all frames from a video file.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Tuple of (list of frames as numpy arrays, video metadata dict)
            
        Raises:
            FileNotFoundError: If video file doesn't exist
            ValueError: If video cannot be opened
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        # Get video metadata
        metadata = {
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'duration_seconds': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
        }
        
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        
        cap.release()
        
        return frames, metadata
    
    def detect_hand_in_frame(self, frame: np.ndarray) -> bool:
        """
        Check if a hand is detected in the given frame.
        
        Args:
            frame: RGB frame as numpy array
            
        Returns:
            True if at least one hand is detected, False otherwise
        """
        results = self.hands.process(frame)
        return results.multi_hand_landmarks is not None
    
    def filter_frames_with_hands(
        self, 
        frames: List[np.ndarray],
        return_indices: bool = False
    ) -> List[np.ndarray]:
        """
        Filter frames to keep only those containing hands.
        
        Args:
            frames: List of RGB frames
            return_indices: If True, also return indices of kept frames
            
        Returns:
            List of frames containing hands (and optionally their indices)
        """
        hand_frames = []
        indices = []
        
        for i, frame in enumerate(frames):
            if self.detect_hand_in_frame(frame):
                hand_frames.append(frame)
                indices.append(i)
        
        if return_indices:
            return hand_frames, indices
        return hand_frames
    
    def sample_to_fixed_count(
        self, 
        frames: List[np.ndarray], 
        target: Optional[int] = None
    ) -> List[np.ndarray]:
        """
        Sample or pad frames to achieve exactly target number of frames.
        
        Uses uniform sampling strategy:
        - If total < target: Duplicate frames (nearest neighbor interpolation)
        - If total = target: Use all frames
        - If total > target: Uniform sampling with stride
        
        Args:
            frames: List of frames to sample from
            target: Target number of frames (defaults to self.n_frames)
            
        Returns:
            List of exactly target frames
        """
        if target is None:
            target = self.n_frames
            
        total = len(frames)
        
        if total == 0:
            # Return empty frames if no input
            return [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(target)]
        
        if total == target:
            return frames
        
        # Calculate sampling indices
        indices = self._compute_sampling_indices(total, target)
        
        return [frames[i] for i in indices]
    
    def _compute_sampling_indices(self, total: int, target: int) -> List[int]:
        """
        Compute indices for uniform sampling.
        
        Args:
            total: Total number of available frames
            target: Target number of frames needed
            
        Returns:
            List of indices to sample
        """
        if total <= 0:
            return [0] * target
            
        indices = []
        for i in range(target):
            idx = int(i * total / target)
            idx = min(idx, total - 1)  # Ensure we don't exceed bounds
            indices.append(idx)
        
        return indices
    
    def extract_frames(
        self, 
        video_path: str,
        return_metadata: bool = False
    ) -> List[np.ndarray]:
        """
        Main method to extract exactly n_frames from a video.
        
        Pipeline:
        1. Extract all frames from video
        2. Optionally filter to hand-containing frames
        3. Sample to exactly n_frames
        
        Args:
            video_path: Path to video file
            return_metadata: If True, also return video metadata
            
        Returns:
            List of exactly n_frames RGB frames (and optionally metadata)
        """
        # Step 1: Extract all frames
        all_frames, metadata = self.extract_all_frames(video_path)
        
        # Step 2: Filter to hand frames if enabled
        if self.filter_hand_frames:
            filtered_frames = self.filter_frames_with_hands(all_frames)
            # If no hands detected, fall back to all frames
            if len(filtered_frames) == 0:
                filtered_frames = all_frames
                metadata['hand_frames_found'] = 0
            else:
                metadata['hand_frames_found'] = len(filtered_frames)
        else:
            filtered_frames = all_frames
            metadata['hand_frames_found'] = len(all_frames)
        
        # Step 3: Sample to fixed count
        sampled_frames = self.sample_to_fixed_count(filtered_frames)
        
        metadata['sampled_frames'] = len(sampled_frames)
        
        if return_metadata:
            return sampled_frames, metadata
        return sampled_frames
    
    def extract_frames_batch(
        self, 
        video_paths: List[str],
        show_progress: bool = True
    ) -> dict:
        """
        Extract frames from multiple videos.
        
        Args:
            video_paths: List of paths to video files
            show_progress: Whether to show progress bar
            
        Returns:
            Dictionary mapping video paths to their extracted frames
        """
        results = {}
        
        iterator = video_paths
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(video_paths, desc="Extracting frames")
            except ImportError:
                pass
        
        for video_path in iterator:
            try:
                frames = self.extract_frames(video_path)
                results[video_path] = frames
            except Exception as e:
                print(f"Error processing {video_path}: {e}")
                results[video_path] = None
        
        return results
    
    def __del__(self):
        """Cleanup MediaPipe resources."""
        if hasattr(self, 'hands'):
            self.hands.close()
