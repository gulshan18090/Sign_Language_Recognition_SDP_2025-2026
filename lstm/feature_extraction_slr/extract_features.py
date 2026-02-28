"""
Stage 1: Feature Extraction for SLR (drive/Video/Cam2)
Extracts SqueezeNet + Bidirectional LSTM features from Cam2 videos.

Usage:
    python extract_features.py              # Extract all
    python extract_features.py --limit 50   # Extract first 50 sentence IDs
    python extract_features.py --features   # List existing features
"""
import os
import sys

# Add parent directory to path for shared modules
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

import argparse
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
import numpy as np
from models.encoder import EncoderRNN
from video.mp_hands import keep_frames_with_hands
from feature_extraction_slr.config_slr import (
    VIDEO_DIR, FEATURES_DIR, DEVICE,
    CNN_FEATURE_SIZE, ENCODER_HIDDEN_SIZE, BIDIRECTIONAL,
    CROP_SIZE, RESIZE
)

# Ensure features directory exists
os.makedirs(FEATURES_DIR, exist_ok=True)

# SqueezeNet model
squeezenet = models.squeezenet1_1(pretrained=True).features.to(DEVICE)
squeezenet.eval()

# Encoder LSTM - Bidirectional for better accuracy
encoder = EncoderRNN(
    input_size=CNN_FEATURE_SIZE,
    hidden_size=ENCODER_HIDDEN_SIZE,
    device=DEVICE,
    biDirectional=BIDIRECTIONAL,
    use_cnn=True
).to(DEVICE)
encoder.eval()

# Preprocessing for SqueezeNet
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((RESIZE, RESIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def extract_frames(video_path):
    """Extract all frames from a video file."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def extract_features_from_video(video_path, out_path):
    """Extract CNN-LSTM features from a single video."""
    frames = extract_frames(video_path)
    if not frames:
        print(f"  No frames found in {video_path}")
        return

    # Apply hand detection to keep only frames with hands
    print(f"  Detecting hands in {len(frames)} frames...")
    frames_with_hands = keep_frames_with_hands(frames, crop_size=CROP_SIZE)

    num_frames = frames_with_hands.shape[0]
    if num_frames == 0:
        print(f"  ⚠️  No frames with hands detected in {video_path}")
        feature_dim = ENCODER_HIDDEN_SIZE * (2 if BIDIRECTIONAL else 1)
        empty_features = torch.zeros((1, feature_dim), device=DEVICE)
        torch.save(empty_features.cpu(), out_path)
        print(f"  Saved empty features: {out_path}")
        return

    print(f"  ✓ Found {num_frames} frames with hands")

    # Preprocess frames
    frames_list = []
    for i in range(num_frames):
        frame_tensor = frames_with_hands[i]
        frame_np = frame_tensor.permute(1, 2, 0).cpu().numpy()
        frame_np = (frame_np * 255).astype(np.uint8) if frame_np.max() <= 1.0 else frame_np.astype(np.uint8)
        processed_frame = transform(frame_np).to(DEVICE)
        frames_list.append(processed_frame)

    video_tensor = torch.stack(frames_list, dim=0).unsqueeze(0)  # (1, T, C, H, W)

    # Extract features through encoder
    with torch.no_grad():
        hidden = encoder.initHidden(1)
        encoder_out, _ = encoder(video_tensor, hidden)
    encoder_out = encoder_out.squeeze(0)  # (T, feature_dim)
    torch.save(encoder_out.cpu(), out_path)
    print(f"  ✓ Saved features ({encoder_out.shape[0]} frames, {encoder_out.shape[1]} dims): {out_path}")


def main():
    parser = argparse.ArgumentParser(description="SLR Feature Extraction (Cam2)")
    parser.add_argument("--features", action="store_true",
                        help="List existing features only")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of sentence folders to process")
    parser.add_argument("--force", action="store_true",
                        help="Re-extract even if features already exist")
    args = parser.parse_args()

    if args.features:
        available = [f for f in os.listdir(FEATURES_DIR) if f.endswith(".pt")]
        print(f"Found {len(available)} feature files in {FEATURES_DIR}")
        for f in sorted(available):
            print(f"  {f}")
        return

    if not os.path.exists(VIDEO_DIR):
        print(f"❌ Video directory not found: {VIDEO_DIR}")
        return

    extracted = 0
    skipped = 0
    errors = 0

    for root, dirs, files in os.walk(VIDEO_DIR):
        video_files = [f for f in files if f.endswith((".mp4", ".avi"))]
        for file in video_files:
            if args.limit and extracted >= args.limit:
                break
            video_path = os.path.join(root, file)
            video_id = os.path.splitext(file)[0]
            out_path = os.path.join(FEATURES_DIR, f"{video_id}.pt")

            if os.path.exists(out_path) and not args.force:
                skipped += 1
                continue

            print(f"\n📹 Processing: {video_path}")
            try:
                extract_features_from_video(video_path, out_path)
                extracted += 1
            except Exception as e:
                print(f"  ❌ Error: {e}")
                errors += 1

    print(f"\n{'='*60}")
    print(f"SLR Feature Extraction Complete")
    print(f"  Extracted: {extracted}")
    print(f"  Skipped (already exist): {skipped}")
    print(f"  Errors: {errors}")
    print(f"  Features directory: {FEATURES_DIR}")


if __name__ == "__main__":
    main()
