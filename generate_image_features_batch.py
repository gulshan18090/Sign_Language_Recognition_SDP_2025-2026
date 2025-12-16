"""
Batch Feature Extraction Script
Extract features from videos in a folder using the same pipeline as extract_features.py
- Extracts frames from videos
- Applies hand detection (keep_frames_with_hands)
- Generates features using bidirectional encoder (512 dims)
- Saves as .pt files
"""
import os
import sys
import argparse
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
import numpy as np
from models.encoder import EncoderRNN
from video.mp_hands import keep_frames_with_hands

# CONFIG
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# SqueezeNet model (same as extract_features.py)
squeezenet = models.squeezenet1_1(pretrained=True).features.to(DEVICE)
squeezenet.eval()

# Encoder LSTM - Bidirectional (same as extract_features.py)
# input_size=86528 is the flattened CNN output size (512*13*13 from SqueezeNet layer12)
# hidden_size=256, bidirectional=TRUE → output dimension = 512 (256*2)
encoder = EncoderRNN(input_size=86528, hidden_size=256, device=DEVICE, biDirectional=True, use_cnn=True).to(DEVICE)
encoder.eval()

# Preprocessing for SqueezeNet (same as extract_features.py)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def extract_frames(video_path):
    """Extract all frames from a video using OpenCV"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def extract_features_from_video(video_path, features_out_path, frames_out_dir=None, save_frames=False):
    """
    Extract features from a single video
    Same pipeline as extract_features.py
    Optionally saves frames to disk for faster retraining
    """
    # Extract all frames from video
    frames = extract_frames(video_path)
    if not frames:
        print(f"  ❌ No frames found in {video_path}")
        return False
    
    # Apply hand detection to keep only frames with hands (same as training pipeline)
    print(f"  Detecting hands in {len(frames)} frames...")
    frames_with_hands = keep_frames_with_hands(frames, crop_size=600)
    
    num_frames = frames_with_hands.shape[0]
    if num_frames == 0:
        print(f"  ⚠️  No frames with hands detected in {video_path}")
        # Create empty features tensor (512 dims for hidden=256, bidirectional)
        empty_features = torch.zeros((1, 512), device=DEVICE)
        torch.save(empty_features.cpu(), features_out_path)
        print(f"  Saved empty features: {features_out_path}")
        return True
    
    print(f"  ✓ Found {num_frames} frames with hands")
    
    # Save frames to disk if requested (for faster retraining)
    if save_frames and frames_out_dir:
        video_id = os.path.splitext(os.path.basename(video_path))[0]
        video_frames_dir = os.path.join(frames_out_dir, video_id)
        os.makedirs(video_frames_dir, exist_ok=True)
        
        for i in range(num_frames):
            frame_tensor = frames_with_hands[i]  # (C, H, W)
            frame_np = frame_tensor.permute(1, 2, 0).cpu().numpy()  # (H, W, C)
            frame_np = (frame_np * 255).astype(np.uint8) if frame_np.max() <= 1.0 else frame_np.astype(np.uint8)
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
            frame_path = os.path.join(video_frames_dir, f"frame_{i:04d}.jpg")
            cv2.imwrite(frame_path, frame_bgr)
        
        print(f"  ✓ Saved {num_frames} frames to: {video_frames_dir}")
    
    # Resize frames to 224x224 for SqueezeNet (frames are already cropped to 600x600)
    frames_list = []
    for i in range(num_frames):
        frame_tensor = frames_with_hands[i]  # (C, H, W)
        # Convert to numpy for PIL transform
        frame_np = frame_tensor.permute(1, 2, 0).cpu().numpy()  # (H, W, C)
        frame_np = (frame_np * 255).astype(np.uint8) if frame_np.max() <= 1.0 else frame_np.astype(np.uint8)
        # Apply transform (resize to 224x224, normalize)
        processed_frame = transform(frame_np).to(DEVICE)
        frames_list.append(processed_frame)
    
    # Stack processed frames
    video_tensor = torch.stack(frames_list, dim=0)  # (T, C, H, W)
    video_tensor = video_tensor.unsqueeze(0)  # (1, T, C, H, W)
    
    # Extract features through encoder
    with torch.no_grad():
        hidden = encoder.initHidden(1)
        encoder_out, _ = encoder(video_tensor, hidden)  # (1, T, 512) for bidir with hidden=256
    encoder_out = encoder_out.squeeze(0)  # (T, 512)
    torch.save(encoder_out.cpu(), features_out_path)
    print(f"  ✓ Saved features ({encoder_out.shape[0]} frames, {encoder_out.shape[1]} dims): {features_out_path}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Batch Feature Extraction for Videos")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory containing videos")
    parser.add_argument("--features_dir", type=str, default="visual_features", help="Output directory for visual feature tensors (.pt)")
    parser.add_argument("--frames_dir", type=str, default="video_frames", help="Output directory for frame images (.jpg)")
    parser.add_argument("--video_ext", type=str, default=".mp4,.avi", help="Video extensions (comma-separated)")
    parser.add_argument("--save_frames", action="store_true", help="Save frames as images (for faster retraining)")
    args = parser.parse_args()
    
    video_dir = args.video_dir
    features_dir = args.features_dir
    frames_dir = args.frames_dir
    save_frames = args.save_frames
    video_extensions = tuple(args.video_ext.split(","))
    
    # Create output directories
    os.makedirs(features_dir, exist_ok=True)
    if save_frames:
        os.makedirs(frames_dir, exist_ok=True)
    
    print(f"🎬 Batch Feature Extraction")
    print(f"Video directory: {video_dir}")
    print(f"Features output: {features_dir}")
    if save_frames:
        print(f"Frames output: {frames_dir}")
    print(f"Device: {DEVICE}")
    print(f"Architecture: Bidirectional LSTM (hidden=256, output=512)")
    print("-" * 60)
    
    # Find all videos
    video_files = []
    for root, dirs, files in os.walk(video_dir):
        for file in files:
            if file.lower().endswith(video_extensions):
                video_files.append(os.path.join(root, file))
    
    print(f"Found {len(video_files)} videos")
    
    if not video_files:
        print("❌ No videos found!")
        return
    
    # Process each video
    success_count = 0
    fail_count = 0
    
    for idx, video_path in enumerate(video_files, 1):
        video_id = os.path.splitext(os.path.basename(video_path))[0]
        features_out_path = os.path.join(features_dir, f"{video_id}.pt")
        
        # Skip if already exists
        if os.path.exists(features_out_path):
            print(f"[{idx}/{len(video_files)}] Features already exist for {video_id}, skipping.")
            success_count += 1
            continue
        
        print(f"\n[{idx}/{len(video_files)}] Processing: {video_id}")
        try:
            if extract_features_from_video(video_path, features_out_path, frames_dir, save_frames):
                success_count += 1
            else:
                fail_count += 1
        except Exception as e:
            print(f"  ❌ Error: {e}")
            fail_count += 1
    
    print("\n" + "=" * 60)
    print(f"✅ Batch extraction complete!")
    print(f"Success: {success_count}/{len(video_files)}")
    print(f"Failed: {fail_count}/{len(video_files)}")
    print(f"Features saved to: {features_dir}")
    if save_frames:
        print(f"Frames saved to: {frames_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()
