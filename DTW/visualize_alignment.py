"""
Visualize DTW-aligned frames side by side
"""
import cv2
import numpy as np
from pathlib import Path
from similarity import SimilarityEngine
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def load_matrix(matrix_path: Path) -> np.ndarray:
    """Load a .npy matrix file."""
    return np.load(matrix_path)

def extract_frame(video_path: str, frame_idx: int) -> np.ndarray:
    """Extract a specific frame from video."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        # Convert BGR to RGB for matplotlib
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None

def visualize_alignment():
    """Show DTW-aligned frames side by side."""
    
    # Paths
    folder = "51"
    video1_name = "translator_video_2_unknown_72"
    video2_name = "user_video_28_OneAK8_25"
    
    video1_path = f"Videos/{folder}/{video1_name}.mp4"
    video2_path = f"Videos/{folder}/{video2_name}.mp4"
    
    matrix1_path = Path(f"matrices/{folder}/{video1_name}.npy")
    matrix2_path = Path(f"matrices/{folder}/{video2_name}.npy")
    
    print("Loading matrices...")
    matrix1 = load_matrix(matrix1_path)
    matrix2 = load_matrix(matrix2_path)
    
    print("Computing DTW alignment...")
    engine = SimilarityEngine()
    similarity, alignment_path = engine.get_dtw_alignment(matrix1, matrix2)
    
    print(f"DTW Similarity: {similarity:.4f}")
    print(f"Total alignments: {len(alignment_path)}")
    
    # Select interesting frame pairs to visualize
    # Show: best matches, worst matches, and some in between
    from scipy.spatial.distance import cosine
    
    frame_similarities = []
    for frame_a, frame_b in alignment_path:
        sim = 1 - cosine(matrix1[frame_a], matrix2[frame_b])
        frame_similarities.append((frame_a, frame_b, sim))
    
    # Sort by similarity
    frame_similarities.sort(key=lambda x: x[2], reverse=True)
    
    # Select frames to show: 3 best, 3 worst, 3 middle
    best_matches = frame_similarities[:3]
    worst_matches = frame_similarities[-3:]
    middle_idx = len(frame_similarities) // 2
    middle_matches = frame_similarities[middle_idx:middle_idx+3]
    
    selected_frames = {
        "Best Matches": best_matches,
        "Medium Matches": middle_matches,
        "Worst Matches": worst_matches
    }
    
    print("\nExtracting frames from videos...")
    
    # Create visualization
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(f'DTW Frame Alignment Visualization\nOverall Similarity: {similarity:.4f}', 
                 fontsize=16, fontweight='bold')
    
    row = 0
    for category, matches in selected_frames.items():
        for idx, (frame_a, frame_b, sim) in enumerate(matches):
            # Extract frames
            img1 = extract_frame(video1_path, frame_a)
            img2 = extract_frame(video2_path, frame_b)
            
            if img1 is None or img2 is None:
                print(f"Warning: Could not extract frames {frame_a}, {frame_b}")
                continue
            
            # Plot translator frame
            ax1 = plt.subplot(9, 2, row * 2 + 1)
            ax1.imshow(img1)
            ax1.set_title(f'{category}\nTranslator Frame {frame_a}', fontsize=10)
            ax1.axis('off')
            
            # Plot user frame
            ax2 = plt.subplot(9, 2, row * 2 + 2)
            ax2.imshow(img2)
            ax2.set_title(f'User Frame {frame_b}\nSimilarity: {sim:.4f}', fontsize=10)
            ax2.axis('off')
            
            row += 1
    
    plt.tight_layout()
    
    # Save figure
    output_path = f"matrices/{folder}/dtw_alignment_visualization.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    
    # Show the plot
    plt.show()
    
    print("\n" + "="*60)
    print("DETAILED ALIGNMENT INFO")
    print("="*60)
    print("\nBest Matches:")
    for frame_a, frame_b, sim in best_matches:
        print(f"  Translator Frame {frame_a:2d} <--> User Frame {frame_b:2d}  (similarity: {sim:.4f})")
    
    print("\nWorst Matches:")
    for frame_a, frame_b, sim in worst_matches:
        print(f"  Translator Frame {frame_a:2d} <--> User Frame {frame_b:2d}  (similarity: {sim:.4f})")

if __name__ == "__main__":
    visualize_alignment()
