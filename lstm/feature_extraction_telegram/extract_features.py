"""
Stage 1: Feature Extraction for Telegram Videos (Videos/ folder)
Extracts SqueezeNet + Bidirectional LSTM features from telegram-collected videos.

The Videos/ folder structure:
    Videos/
        sentence_name_1/
            video1.mp4
            video2.mp4
        sentence_name_2/
            video1.mp4
        ...

Usage:
    python extract_features.py              # Extract all
    python extract_features.py --limit 50   # Only first 50 sentence folders
    python extract_features.py --features   # List existing features
"""
import os
import sys

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
from feature_extraction_telegram.config_telegram import (
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
        return False

    # Apply hand detection to keep only frames with hands
    print(f"  Detecting hands in {len(frames)} frames...")
    frames_with_hands = keep_frames_with_hands(frames, crop_size=CROP_SIZE)

    num_frames = frames_with_hands.shape[0]
    if num_frames == 0:
        print(f"  ⚠️  No frames with hands detected")
        feature_dim = ENCODER_HIDDEN_SIZE * (2 if BIDIRECTIONAL else 1)
        empty_features = torch.zeros((1, feature_dim), device=DEVICE)
        torch.save(empty_features.cpu(), out_path)
        print(f"  Saved empty features: {out_path}")
        return True

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
    return True


def sanitize_filename(name):
    """Convert a sentence folder name to a safe filename."""
    # Replace problematic characters
    safe = name.replace(' ', '_').replace('/', '_').replace('\\', '_')
    safe = safe.replace('?', '_').replace('!', '_').replace(';', '_')
    safe = safe.replace(',', '_').replace('(', '_').replace(')', '_')
    safe = safe.replace("'", '_').replace('"', '_')
    # Remove consecutive underscores
    while '__' in safe:
        safe = safe.replace('__', '_')
    return safe.strip('_')


def main():
    parser = argparse.ArgumentParser(description="Telegram Feature Extraction (Videos/)")
    parser.add_argument("--features", action="store_true",
                        help="List existing features only")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of sentence folders to process")
    parser.add_argument("--force", action="store_true",
                        help="Re-extract even if features already exist")
    args = parser.parse_args()

    if args.features:
        if os.path.exists(FEATURES_DIR):
            available = [f for f in os.listdir(FEATURES_DIR) if f.endswith(".pt")]
            print(f"Found {len(available)} feature files in {FEATURES_DIR}")
            for f in sorted(available):
                print(f"  {f}")
        else:
            print(f"Features directory not found: {FEATURES_DIR}")
        return

    if not os.path.exists(VIDEO_DIR):
        print(f"❌ Video directory not found: {VIDEO_DIR}")
        return

    # Get all sentence folders
    sentence_folders = sorted([
        d for d in os.listdir(VIDEO_DIR)
        if os.path.isdir(os.path.join(VIDEO_DIR, d))
    ])

    if args.limit:
        sentence_folders = sentence_folders[:args.limit]

    print(f"Found {len(sentence_folders)} sentence folders in {VIDEO_DIR}")

    total_extracted = 0
    total_skipped = 0
    total_errors = 0
    total_videos = 0

    for folder_idx, sentence_folder in enumerate(sentence_folders, 1):
        folder_path = os.path.join(VIDEO_DIR, sentence_folder)
        video_files = sorted([
            f for f in os.listdir(folder_path)
            if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))
        ])

        if not video_files:
            continue

        print(f"\n{'='*60}")
        print(f"[{folder_idx}/{len(sentence_folders)}] Sentence: {sentence_folder}")
        print(f"  Videos: {len(video_files)}")

        safe_sentence = sanitize_filename(sentence_folder)

        for video_file in video_files:
            total_videos += 1
            video_path = os.path.join(folder_path, video_file)
            video_name = os.path.splitext(video_file)[0]
            # Feature file name: sanitized_sentence__video_name.pt
            feature_name = f"{safe_sentence}__{sanitize_filename(video_name)}.pt"
            out_path = os.path.join(FEATURES_DIR, feature_name)

            if os.path.exists(out_path) and not args.force:
                total_skipped += 1
                continue

            print(f"\n  📹 {video_file}")
            try:
                success = extract_features_from_video(video_path, out_path)
                if success:
                    total_extracted += 1
                else:
                    total_errors += 1
            except Exception as e:
                print(f"  ❌ Error: {e}")
                total_errors += 1

    print(f"\n{'='*60}")
    print(f"Telegram Feature Extraction Complete")
    print(f"  Sentence folders: {len(sentence_folders)}")
    print(f"  Total videos: {total_videos}")
    print(f"  Extracted: {total_extracted}")
    print(f"  Skipped (already exist): {total_skipped}")
    print(f"  Errors: {total_errors}")
    print(f"  Features directory: {FEATURES_DIR}")


if __name__ == "__main__":
    main()
