"""
Visualize frames that pass hand detection from a video
Shows which frames are kept for feature extraction
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
from video.mp_hands import keep_frames_with_hands
from config import config

def visualize_hand_detection(video_path, save_path=None, max_frames_to_show=64):
    """Extract and visualize frames with hand detection"""
    print(f"🎥 Processing video: {video_path}")
    
    # Extract all frames from video
    cap = cv2.VideoCapture(video_path)
    all_frames = []
    frame_indices = []
    frame_count = 0
    
    print("📹 Extracting frames...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(frame)
        frame_indices.append(frame_count)
        frame_count += 1
    cap.release()
    
    print(f"   Total frames in video: {len(all_frames)}")
    
    # Apply hand detection and get frames for display
    print("✋ Detecting hands in frames...")
    import mediapipe as mp
    from torchvision.transforms import CenterCrop
    
    mp_hands = mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.8,
        min_tracking_confidence=0.9
    )
    
    crop_size = 600
    crop = CenterCrop(crop_size)
    frames_for_display = []
    
    for frame in all_frames:
        frame_np = np.ascontiguousarray(frame).astype(np.uint8)
        rgb = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)
        results = mp_hands.process(rgb)
        
        if results.multi_hand_landmarks:
            # Convert to tensor, crop, then back to numpy for display
            t = torch.from_numpy(rgb).permute(2, 0, 1).float()  # (H, W, C) -> (C, H, W)
            t_cropped = crop(t)  # Crop to (C, crop_size, crop_size)
            # Convert back to (H, W, C) for display
            frame_display = t_cropped.permute(1, 2, 0).cpu().numpy()
            # Ensure uint8
            frame_display = np.clip(frame_display, 0, 255).astype(np.uint8)
            frames_for_display.append(frame_display)
    
    num_frames_with_hands = len(frames_for_display)
    print(f"   Frames with hands detected: {num_frames_with_hands}")
    
    if num_frames_with_hands == 0:
        print("⚠️  No frames with hands detected!")
        return None, 0, len(all_frames)
    
    # Limit frames to show
    num_to_show = min(num_frames_with_hands, max_frames_to_show)
    frames_to_show = frames_for_display[:num_to_show]
    
    # Calculate grid size
    rows = int(np.ceil(np.sqrt(num_to_show)))
    cols = int(np.ceil(num_to_show / rows))
    
    # Create visualization
    fig, axes = plt.subplots(rows, cols, figsize=(20, 20))
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    else:
        axes = axes
    
    print(f"🖼️  Visualizing {num_to_show} frames...")
    
    for idx in range(num_to_show):
        row = idx // cols
        col = idx % cols
        
        # Get frame - already in (H, W, C) RGB uint8 format
        frame = frames_to_show[idx]
        
        # Frame should already be uint8 RGB, but double-check
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        
        # Display the frame (already RGB format)
        axes[row, col].imshow(frame)
        axes[row, col].axis('off')
        axes[row, col].set_title(f'Frame {idx+1}/{num_to_show}', fontsize=8)
    
    # Hide unused subplots
    for idx in range(num_to_show, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')
    
    plt.suptitle(
        f'Frames with Hand Detection\n'
        f'Video: {os.path.basename(video_path)}\n'
        f'Total: {len(all_frames)} frames | With hands: {num_frames_with_hands} frames',
        fontsize=14,
        y=0.995
    )
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"💾 Saved visualization to: {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    return frames_for_display, num_frames_with_hands, len(all_frames)

def main():
    parser = argparse.ArgumentParser(description="Visualize frames with hand detection")
    parser.add_argument("video_path", type=str, help="Path to video file")
    parser.add_argument("--save", type=str, default=None, 
                       help="Save visualization to file (e.g., output.png)")
    parser.add_argument("--max-frames", type=int, default=64,
                       help="Maximum number of frames to visualize (default: 64)")
    args = parser.parse_args()
    
    if not os.path.exists(args.video_path):
        print(f"❌ Error: Video file not found: {args.video_path}")
        return
    
    frames, num_with_hands, total_frames = visualize_hand_detection(
        args.video_path, 
        save_path=args.save,
        max_frames_to_show=args.max_frames
    )
    
    print(f"\n📊 Summary:")
    print(f"   Total frames: {total_frames}")
    print(f"   Frames with hands: {num_with_hands}")
    print(f"   Percentage: {100 * num_with_hands / total_frames:.1f}%")

if __name__ == "__main__":
    main()

