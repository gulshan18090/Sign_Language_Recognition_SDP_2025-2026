"""
Visualize ALL DTW-aligned frames for all folders
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

def visualize_folder(folder: str, video1_name: str, video2_name: str):
    """Show ALL DTW-aligned frames for one folder."""
    
    video1_path = f"Videos/{folder}/{video1_name}.mp4"
    video2_path = f"Videos/{folder}/{video2_name}.mp4"
    
    matrix1_path = Path(f"matrices/{folder}/{video1_name}.npy")
    matrix2_path = Path(f"matrices/{folder}/{video2_name}.npy")
    
    print(f"\n{'='*60}")
    print(f"Processing Folder {folder}")
    print('='*60)
    
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
    
    # Create a large grid: 2 columns (translator, user) x N rows
    num_pairs = len(alignment_path)
    fig, axes = plt.subplots(num_pairs, 2, figsize=(8, num_pairs * 2))
    
    fig.suptitle(f'Folder {folder} - DTW Frame Alignment\nOverall Similarity: {similarity:.4f}\nTranslator (Left) vs User (Right)', 
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
    
    # Print summary
    print("\nALIGNMENT SUMMARY")
    print("-" * 60)
    
    # Find repeated mappings
    from collections import Counter
    user_frame_counts = Counter([frame_b for _, frame_b, _ in frame_data])
    repeated = [(frame, count) for frame, count in user_frame_counts.items() if count > 1]
    
    if repeated:
        print("Repeated user frame mappings:")
        for frame, count in sorted(repeated, key=lambda x: x[1], reverse=True):
            print(f"  User frame {frame} matched {count} times")
    
    # Best and worst matches
    sorted_frames = sorted(frame_data, key=lambda x: x[2], reverse=True)
    print(f"\nBest match:  T-{sorted_frames[0][0]} <--> U-{sorted_frames[0][1]} (sim: {sorted_frames[0][2]:.4f})")
    print(f"Worst match: T-{sorted_frames[-1][0]} <--> U-{sorted_frames[-1][1]} (sim: {sorted_frames[-1][2]:.4f})")

def main():
    """Process all folders."""
    
    folders = [
        ("51", "translator_video_2_unknown_72", "user_video_28_OneAK8_25"),
        ("79", "translator_video_7_jhasanov_15", "user_video_71_Aykanabi_17"),
        ("8", "translator_video_2_unknown_35", "user_video_8_HSTechk_29"),
    ]
    
    print("Starting visualization for all folders...")
    print("This will create 3 large images, one for each folder.\n")
    
    for folder, video1, video2 in folders:
        try:
            visualize_folder(folder, video1, video2)
        except Exception as e:
            print(f"\nError processing folder {folder}: {e}")
            continue
    
    print("\n" + "="*60)
    print("ALL VISUALIZATIONS COMPLETE!")
    print("="*60)
    print("\nCreated files:")
    for folder, _, _ in folders:
        print(f"  - matrices/{folder}/dtw_all_frames_grid.png")

if __name__ == "__main__":
    main()
