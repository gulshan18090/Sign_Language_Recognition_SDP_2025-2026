"""
Test DTW with keyframe extraction
"""
import numpy as np
from pathlib import Path
from similarity import SimilarityEngine

# Load one pair
matrix1 = np.load("matrices/51/translator_video_2_unknown_72.npy")
matrix2 = np.load("matrices/51/user_video_28_OneAK8_25.npy")

print(f"Matrix shapes: {matrix1.shape} vs {matrix2.shape}")
print("\n" + "="*60)

# Test all methods
engine = SimilarityEngine()

print("\n1. Regular DTW-Aligned Cosine (all frames):")
dtw_aligned_sim = engine.compute_similarity(matrix1, matrix2, method="dtw_aligned_cosine")
print(f"   Result: {dtw_aligned_sim:.6f}")

print("\n2. DTW-Aligned Cosine with Keyframes (skip similar frames >95%):")
keyframe_sim = engine._dtw_aligned_cosine_with_keyframes(matrix1, matrix2, keyframe_threshold=0.95)
print(f"   Result: {keyframe_sim:.6f}")

print("\n3. DTW-Aligned Cosine with Keyframes (skip similar frames >90%):")
keyframe_sim_90 = engine._dtw_aligned_cosine_with_keyframes(matrix1, matrix2, keyframe_threshold=0.90)
print(f"   Result: {keyframe_sim_90:.6f}")

print("\n" + "="*60)
print("COMPARISON")
print("="*60)
print(f"All frames:           {dtw_aligned_sim:.6f}")
print(f"Keyframes (>95%):     {keyframe_sim:.6f}  (difference: {abs(dtw_aligned_sim - keyframe_sim):.6f})")
print(f"Keyframes (>90%):     {keyframe_sim_90:.6f}  (difference: {abs(dtw_aligned_sim - keyframe_sim_90):.6f})")

# Show keyframe analysis
print("\n" + "="*60)
print("KEYFRAME ANALYSIS")
print("="*60)

keyframes, indices = engine.extract_keyframes(matrix1, 0.95)
print(f"\nWith 95% threshold:")
print(f"  Original frames: 64")
print(f"  Keyframes: {len(indices)}")
print(f"  Reduction: {(1 - len(indices)/64)*100:.1f}%")
print(f"  Keyframe indices: {indices}")

keyframes90, indices90 = engine.extract_keyframes(matrix1, 0.90)
print(f"\nWith 90% threshold:")
print(f"  Original frames: 64")
print(f"  Keyframes: {len(indices90)}")
print(f"  Reduction: {(1 - len(indices90)/64)*100:.1f}%")
print(f"  Keyframe indices: {indices90}")
