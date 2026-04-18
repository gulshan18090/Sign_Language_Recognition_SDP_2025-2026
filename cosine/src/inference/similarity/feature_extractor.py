"""
Feature Extractor Module

Extracts MediaPipe hand landmarks as feature vectors from frames.
"""

import numpy as np
import mediapipe as mp
from typing import List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class HandLandmarks:
    """Container for hand landmark data."""
    landmarks: np.ndarray  # Shape: (21, 3) for x, y, z
    handedness: str  # 'Left' or 'Right'
    confidence: float


class HandFeatureExtractor:
    """
    Extracts MediaPipe hand landmarks as feature vectors.
    
    This class handles:
    - MediaPipe landmark extraction
    - Multi-hand processing (up to 2 hands)
    - Feature normalization
    - Matrix generation for video sequences
    
    Landmark Structure:
        - 21 landmarks per hand
        - 3 coordinates per landmark (x, y, z)
        - 126 features total for 2 hands (21 * 3 * 2)
    
    Attributes:
        max_hands (int): Maximum number of hands to track (1 or 2)
        normalize (bool): Whether to normalize landmarks
        features_per_hand (int): Number of features per hand (63 = 21*3)
    """
    
    # MediaPipe hand landmark indices
    WRIST = 0
    THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
    INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
    MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
    RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
    PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20
    
    NUM_LANDMARKS = 21
    COORDS_PER_LANDMARK = 3  # x, y, z
    
    def __init__(
        self,
        max_hands: int = 2,
        normalize: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5
    ):
        """
        Initialize the HandFeatureExtractor.
        
        Args:
            max_hands: Maximum number of hands to detect (1 or 2)
            normalize: Whether to normalize landmarks for scale/position invariance
            min_detection_confidence: Minimum detection confidence
            min_tracking_confidence: Minimum tracking confidence
        """
        self.max_hands = min(max_hands, 2)  # Cap at 2 hands
        self.normalize = normalize
        self.features_per_hand = self.NUM_LANDMARKS * self.COORDS_PER_LANDMARK  # 63
        self.total_features = self.features_per_hand * self.max_hands  # 126 for 2 hands
        
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=self.max_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
    
    def extract_landmarks_from_frame(
        self, 
        frame: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], dict]:
        """
        Extract hand landmarks from a single frame.
        
        Args:
            frame: RGB frame as numpy array (H, W, 3)
            
        Returns:
            Tuple of:
                - Left hand landmarks (21, 3) or None
                - Right hand landmarks (21, 3) or None
                - Metadata dict with detection info
        """
        results = self.hands.process(frame)
        
        left_hand = None
        right_hand = None
        metadata = {
            'hands_detected': 0,
            'left_confidence': 0.0,
            'right_confidence': 0.0
        }
        
        if results.multi_hand_landmarks:
            metadata['hands_detected'] = len(results.multi_hand_landmarks)
            
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Get handedness
                handedness = results.multi_handedness[idx].classification[0]
                hand_label = handedness.label  # 'Left' or 'Right'
                confidence = handedness.score
                
                # Extract landmarks to numpy array
                landmarks = np.array([
                    [lm.x, lm.y, lm.z] 
                    for lm in hand_landmarks.landmark
                ])
                
                if hand_label == 'Left':
                    left_hand = landmarks
                    metadata['left_confidence'] = confidence
                else:
                    right_hand = landmarks
                    metadata['right_confidence'] = confidence
        
        return left_hand, right_hand, metadata
    
    def normalize_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Normalize landmarks for scale and position invariance.
        
        Normalization steps:
        1. Center around wrist (landmark 0)
        2. Scale by maximum distance from wrist
        
        Args:
            landmarks: Raw landmarks array (21, 3)
            
        Returns:
            Normalized landmarks array (21, 3)
        """
        if landmarks is None:
            return np.zeros((self.NUM_LANDMARKS, self.COORDS_PER_LANDMARK))
        
        # Center around wrist
        wrist = landmarks[self.WRIST]
        centered = landmarks - wrist
        
        # Scale by maximum distance from wrist
        distances = np.linalg.norm(centered, axis=1)
        max_dist = np.max(distances)
        
        if max_dist > 0:
            normalized = centered / max_dist
        else:
            normalized = centered
        
        return normalized
    
    def landmarks_to_feature_vector(
        self, 
        left_hand: Optional[np.ndarray],
        right_hand: Optional[np.ndarray]
    ) -> np.ndarray:
        """
        Convert hand landmarks to a single feature vector.
        
        Args:
            left_hand: Left hand landmarks (21, 3) or None
            right_hand: Right hand landmarks (21, 3) or None
            
        Returns:
            Feature vector of shape (total_features,)
            For 2 hands: (126,) = [left_63 | right_63]
        """
        # Process left hand
        if left_hand is not None:
            if self.normalize:
                left_hand = self.normalize_landmarks(left_hand)
            left_features = left_hand.flatten()
        else:
            left_features = np.zeros(self.features_per_hand)
        
        # Process right hand
        if right_hand is not None:
            if self.normalize:
                right_hand = self.normalize_landmarks(right_hand)
            right_features = right_hand.flatten()
        else:
            right_features = np.zeros(self.features_per_hand)
        
        # Concatenate based on max_hands setting
        if self.max_hands == 2:
            return np.concatenate([left_features, right_features])
        else:
            # Use whichever hand is available, prefer right
            if right_hand is not None:
                return right_features
            return left_features
    
    def extract_features_from_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Extract feature vector from a single frame.
        
        Args:
            frame: RGB frame as numpy array
            
        Returns:
            Feature vector of shape (total_features,)
        """
        left_hand, right_hand, _ = self.extract_landmarks_from_frame(frame)
        return self.landmarks_to_feature_vector(left_hand, right_hand)
    
    def frames_to_feature_matrix(
        self, 
        frames: List[np.ndarray],
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Convert a list of frames to a feature matrix.
        
        Args:
            frames: List of RGB frames
            show_progress: Whether to show progress bar
            
        Returns:
            Feature matrix of shape (n_frames, total_features)
            For 64 frames with 2 hands: (64, 126)
        """
        n_frames = len(frames)
        feature_matrix = np.zeros((n_frames, self.total_features))
        
        iterator = range(n_frames)
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, desc="Extracting features")
            except ImportError:
                pass
        
        for i in iterator:
            feature_matrix[i] = self.extract_features_from_frame(frames[i])
        
        return feature_matrix
    
    def get_feature_names(self) -> List[str]:
        """
        Get human-readable names for each feature dimension.
        
        Returns:
            List of feature names like ['left_wrist_x', 'left_wrist_y', ...]
        """
        landmark_names = [
            'wrist', 'thumb_cmc', 'thumb_mcp', 'thumb_ip', 'thumb_tip',
            'index_mcp', 'index_pip', 'index_dip', 'index_tip',
            'middle_mcp', 'middle_pip', 'middle_dip', 'middle_tip',
            'ring_mcp', 'ring_pip', 'ring_dip', 'ring_tip',
            'pinky_mcp', 'pinky_pip', 'pinky_dip', 'pinky_tip'
        ]
        coord_names = ['x', 'y', 'z']
        hand_names = ['left', 'right'] if self.max_hands == 2 else ['hand']
        
        names = []
        for hand in hand_names:
            for landmark in landmark_names:
                for coord in coord_names:
                    names.append(f"{hand}_{landmark}_{coord}")
        
        return names
    
    def compute_hand_statistics(self, feature_matrix: np.ndarray) -> dict:
        """
        Compute statistics about hand presence and features.
        
        Args:
            feature_matrix: Feature matrix (n_frames, n_features)
            
        Returns:
            Dictionary with statistics
        """
        n_frames = feature_matrix.shape[0]
        
        # Check which frames have hands (non-zero features)
        left_present = np.sum(np.abs(feature_matrix[:, :self.features_per_hand]), axis=1) > 0
        
        if self.max_hands == 2:
            right_present = np.sum(np.abs(feature_matrix[:, self.features_per_hand:]), axis=1) > 0
        else:
            right_present = np.zeros(n_frames, dtype=bool)
        
        return {
            'total_frames': n_frames,
            'frames_with_left_hand': int(np.sum(left_present)),
            'frames_with_right_hand': int(np.sum(right_present)),
            'frames_with_both_hands': int(np.sum(left_present & right_present)),
            'frames_with_any_hand': int(np.sum(left_present | right_present)),
            'left_hand_ratio': float(np.mean(left_present)),
            'right_hand_ratio': float(np.mean(right_present)),
            'feature_mean': float(np.mean(feature_matrix)),
            'feature_std': float(np.std(feature_matrix))
        }
    
    def __del__(self):
        """Cleanup MediaPipe resources."""
        if hasattr(self, 'hands'):
            self.hands.close()
