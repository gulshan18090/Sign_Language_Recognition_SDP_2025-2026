"""
Visualize keyframe extraction and DTW alignment for ALL folders
"""
import cv2
import numpy as np
from pathlib import Path
from similarity import SimilarityEngine
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

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
    """Show keyframes and their DTW alignment for one folder."""
    
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
    
    print("Extracting keyframes...")
    engine = SimilarityEngine()
    
    # Extract keyframes with 95% threshold
    keyframe_matrix, keyframe_indices = engine.extract_keyframes(matrix1, 0.95)
    
    print(f"Keyframes: {len(keyframe_indices)} / {matrix1.shape[0]}")
    print(f"Reduction: {(1 - len(keyframe_indices)/matrix1.shape[0])*100:.1f}%")
    
    print("\nComputing DTW alignment with keyframes...")
    similarity, alignment_path = engine.get_dtw_alignment(keyframe_matrix, matrix2)
    
    print(f"DTW Similarity: {similarity:.4f}")
    print(f"Total alignments: {len(alignment_path)}")
    
    # Calculate similarities for each pair
    from scipy.spatial.distance import cosine
    frame_data = []
    for keyframe_idx, video2_frame_idx in alignment_path:
        original_frame_idx = keyframe_indices[keyframe_idx]
        sim = 1 - cosine(matrix1[original_frame_idx], matrix2[video2_frame_idx])
        frame_data.append((original_frame_idx, video2_frame_idx, sim))
    
    print(f"\nExtracting {len(alignment_path)} frame pairs from videos...")
    
    # Create visualization
    num_pairs = len(alignment_path)
    fig, axes = plt.subplots(num_pairs, 2, figsize=(8, num_pairs * 2))
    
    fig.suptitle(f'Folder {folder} - Keyframe DTW Alignment (95% threshold)\n{len(keyframe_indices)}/{matrix1.shape[0]} frames kept | Similarity: {similarity:.4f}', 
                 fontsize=14, fontweight='bold', y=0.9995)
    
    # Process each frame pair
    for idx, (frame_a, frame_b, sim) in enumerate(frame_data):
        # Extract frames
        img1 = extract_frame(video1_path, frame_a)
        img2 = extract_frame(video2_path, frame_b)
        
        if img1 is None or img2 is None:
            print(f"Warning: Could not extract frames {frame_a}, {frame_b}")
            continue
        
        # Plot translator keyframe (left)
        axes[idx, 0].imshow(img1)
        axes[idx, 0].set_title(f'T-{frame_a} [KEY]', fontsize=8, fontweight='bold')
        axes[idx, 0].axis('off')
        
        # Add green border to indicate keyframe
        rect = Rectangle((0, 0), img1.shape[1]-1, img1.shape[0]-1, 
                        linewidth=3, edgecolor='green', facecolor='none')
        axes[idx, 0].add_patch(rect)
        
        # Plot user frame (right)
        axes[idx, 1].imshow(img2)
        axes[idx, 1].set_title(f'U-{frame_b} | {sim:.3f}', fontsize=8)
        axes[idx, 1].axis('off')
        
        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{num_pairs} pairs...")
    
    plt.tight_layout()
    
    # Save figure
    output_path = f"matrices/{folder}/dtw_keyframes_alignment.png"
    print(f"\nSaving visualization...")
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"Saved to: {output_path}")
    
    plt.close()
    
    # Print summary
    print("\nKEYFRAME SUMMARY")
    print("-"*60)
    print(f"Original frames: {matrix1.shape[0]}")
    print(f"Keyframes kept: {len(keyframe_indices)}")
    print(f"Frames skipped: {matrix1.shape[0] - len(keyframe_indices)}")
    print(f"Reduction: {(1 - len(keyframe_indices)/matrix1.shape[0])*100:.1f}%")
    
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
    
    print("Starting keyframe visualization for all folders...")
    print("This will create 3 images showing only important keyframes.\n")
    
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
        print(f"  - matrices/{folder}/dtw_keyframes_alignment.png")

if __name__ == "__main__":
    main()
