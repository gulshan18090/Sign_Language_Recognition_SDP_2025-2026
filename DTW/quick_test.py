"""Quick test of DTW-aligned cosine"""
import numpy as np
from pathlib import Path
from similarity import SimilarityEngine

# Load one pair
matrix1 = np.load("matrices/51/translator_video_2_unknown_72.npy")
matrix2 = np.load("matrices/51/user_video_28_OneAK8_25.npy")

print(f"Matrix shapes: {matrix1.shape} vs {matrix2.shape}")
print("\nComputing similarities...")

# Test all three methods
engine = SimilarityEngine()

print("\n1. Regular Cosine...")
cosine_sim = engine.compute_similarity(matrix1, matrix2, method="cosine")
print(f"   Result: {cosine_sim:.6f}")

print("\n2. Frame-wise Cosine...")
framewise_sim = engine.compute_similarity(matrix1, matrix2, method="frame_wise_cosine")
print(f"   Result: {framewise_sim:.6f}")

print("\n3. DTW-Aligned Cosine...")
dtw_aligned_sim, alignment_path = engine.get_dtw_alignment(matrix1, matrix2)
print(f"   Result: {dtw_aligned_sim:.6f}")
print(f"   Aligned {len(alignment_path)} frame pairs")

print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)
print(f"Regular Cosine:        {cosine_sim:.6f}")
print(f"Frame-wise Cosine:     {framewise_sim:.6f}")
print(f"DTW-Aligned Cosine:    {dtw_aligned_sim:.6f}")
print("\nBest method:", ["Regular", "Frame-wise", "DTW-Aligned"][np.argmax([cosine_sim, framewise_sim, dtw_aligned_sim])])

print("\n" + "="*60)
print("DTW FRAME ALIGNMENT")
print("="*60)
print("Translator Video Frame  -->  User Video Frame")
print("-"*60)

# Show alignment in groups
for i in range(0, len(alignment_path), 10):
    batch = alignment_path[i:i+10]
    for frame_a, frame_b in batch:
        # Calculate cosine similarity for this frame pair
        from scipy.spatial.distance import cosine
        frame_sim = 1 - cosine(matrix1[frame_a], matrix2[frame_b])
        print(f"Frame {frame_a:2d}              -->  Frame {frame_b:2d}     (sim: {frame_sim:.4f})")
    if i + 10 < len(alignment_path):
        print()

# Show statistics about alignment
frame_a_list = [pair[0] for pair in alignment_path]
frame_b_list = [pair[1] for pair in alignment_path]

print("\n" + "="*60)
print("ALIGNMENT STATISTICS")
print("="*60)
print(f"Total frame pairs aligned: {len(alignment_path)}")
print(f"Translator frames used: {len(set(frame_a_list))} / {matrix1.shape[0]}")
print(f"User frames used:       {len(set(frame_b_list))} / {matrix2.shape[0]}")

# Check for timing differences
avg_offset = np.mean([abs(a - b) for a, b in alignment_path])
max_offset = max([abs(a - b) for a, b in alignment_path])
print(f"\nAverage frame offset: {avg_offset:.1f} frames")
print(f"Maximum frame offset: {max_offset} frames")

if avg_offset < 5:
    print("   -> Videos have good timing alignment")
elif avg_offset < 10:
    print("   -> Some timing differences detected")
else:
    print("   -> Significant timing differences (user faster/slower)")
