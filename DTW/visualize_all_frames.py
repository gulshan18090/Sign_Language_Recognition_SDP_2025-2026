"""
Visualize ALL DTW-aligned frames side by side in a grid
"""
import cv2
import numpy as np
from pathlib import Path
from similarity import SimilarityEngine
import matplotlib.pyplot as plt

def extract_frame(video_path: str, frame_idx: int) -> np.ndarray:
    """Extract a specific frame from video."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None

def visualize_all_frames():
    """Show ALL DTW-aligned frames in a grid."""
    
    # Paths
    folder = "51"
    video1_name = "translator_video_2_unknown_72"
    video2_name = "user_video_28_OneAK8_25"
    
    video1_path = f"Videos/{folder}/{video1_name}.mp4"
    video2_path = f"Videos/{folder}/{video2_name}.mp4"
    
    matrix1_path = Path(f"matrices/{folder}/{video1_name}.npy")
    matrix2_path = Path(f"matrices/{folder}/{video2_name}.npy")
    
    print("Loading matrices...")
    matrix1 = np.load(matrix1_path)
    matrix2 = np.load(matrix2_path)
    
    print("Computing DTW alignment...")
    engine = SimilarityEngine()
    similarity, alignment_path = engine.get_dtw_alignment(matrix1, matrix2)
    
    print(f"DTW Similarity: {similarity:.4f}")
    print(f"Total alignments: {len(alignment_path)}")
    print(f"\nExtracting all {len(alignment_path)} frame pairs from videos...")
    
    # Calculate similarities for each pair
    from scipy.spatial.distance import cosine
    frame_data = []
    for frame_a, frame_b in alignment_path:
        sim = 1 - cosine(matrix1[frame_a], matrix2[frame_b])
        frame_data.append((frame_a, frame_b, sim))
    
    # Create a large grid: 2 columns (translator, user) x 92 rows
    num_pairs = len(alignment_path)
    fig, axes = plt.subplots(num_pairs, 2, figsize=(8, num_pairs * 2))
    
    fig.suptitle(f'Complete DTW Frame Alignment\nOverall Similarity: {similarity:.4f}\nTranslator (Left) vs User (Right)', 
                 fontsize=14, fontweight='bold', y=0.9995)
    
    # Process each frame pair
    for idx, (frame_a, frame_b, sim) in enumerate(frame_data):
        # Extract frames
        img1 = extract_frame(video1_path, frame_a)
        img2 = extract_frame(video2_path, frame_b)
        
        if img1 is None or img2 is None:
            print(f"Warning: Could not extract frames {frame_a}, {frame_b}")
            continue
        
        # Plot translator frame (left)
        axes[idx, 0].imshow(img1)
        axes[idx, 0].set_title(f'T-{frame_a}', fontsize=8)
        axes[idx, 0].axis('off')
        
        # Plot user frame (right)
        axes[idx, 1].imshow(img2)
        axes[idx, 1].set_title(f'U-{frame_b} | {sim:.3f}', fontsize=8)
        axes[idx, 1].axis('off')
        
        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{num_pairs} pairs...")
    
    plt.tight_layout()
    
    # Save figure
    output_path = f"matrices/{folder}/dtw_all_frames_grid.png"
    print(f"\nSaving visualization...")
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"Saved to: {output_path}")
    
    plt.close()
    
    # Also create a summary with key statistics
    print("\n" + "="*60)
    print("ALIGNMENT SUMMARY")
    print("="*60)
    
    # Group consecutive identical mappings
    print("\nRepeated frame mappings (where multiple translator frames map to same user frame):")
    prev_b = -1
    count = 0
    for frame_a, frame_b, sim in frame_data:
        if frame_b == prev_b:
            count += 1
        else:
            if count > 0:
                print(f"  User frame {prev_b} matched {count + 1} times (translator frames)")
            prev_b = frame_b
            count = 0
    
    print(f"\nVisualization complete! Open the image to see all {num_pairs} frame pairs.")

if __name__ == "__main__":
    visualize_all_frames()
