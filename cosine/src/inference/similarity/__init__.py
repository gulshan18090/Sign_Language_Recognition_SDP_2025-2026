"""
Video Similarity Matching Module

A professional implementation for comparing videos based on hand movement
features extracted using MediaPipe.
"""

from .frame_extractor import FrameExtractor
from .feature_extractor import HandFeatureExtractor
from .similarity_engine import SimilarityEngine
from .video_matcher import VideoMatcher
from .dtw_visualizer import DTWVisualizer

__all__ = [
    'FrameExtractor',
    'HandFeatureExtractor', 
    'SimilarityEngine',
    'VideoMatcher',
    'DTWVisualizer'
]

__version__ = '1.0.0'
