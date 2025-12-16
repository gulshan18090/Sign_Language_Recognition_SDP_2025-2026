"""
Stage 1: Offline Feature Extraction for SLR
Extract features from all videos using SqueezeNet + Encoder, save as .pt tensors.
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
VIDEO_DIR = "drive/Video/Cam2"  # Updated to correct location
FEATURES_DIR = "features"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ensure features directory exists
os.makedirs(FEATURES_DIR, exist_ok=True)

# SqueezeNet model
squeezenet = models.squeezenet1_1(pretrained=True).features.to(DEVICE)
squeezenet.eval()

# Encoder LSTM - Bidirectional for better accuracy
# input_size=86528 is the flattened CNN output size (512*13*13 from SqueezeNet layer12)
# hidden_size=256, bidirectional=TRUE → output dimension = 512 (256*2)
# use_cnn=True to enable SqueezeNet feature extraction
encoder = EncoderRNN(input_size=86528, hidden_size=256, device=DEVICE, biDirectional=True, use_cnn=True).to(DEVICE)
encoder.eval()

# Preprocessing for SqueezeNet
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def extract_frames(video_path):
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
    # Extract all frames from video
    frames = extract_frames(video_path)
    if not frames:
        print(f"No frames found in {video_path}")
        return
    
    # Apply hand detection to keep only frames with hands (same as training pipeline)
    print(f"  Detecting hands in {len(frames)} frames...")
    frames_with_hands = keep_frames_with_hands(frames, crop_size=600)
    
    num_frames = frames_with_hands.shape[0]
    if num_frames == 0:
        print(f"  ⚠️  No frames with hands detected in {video_path}")
        # Create empty features tensor (512 dims for hidden=256, bidirectional)
        empty_features = torch.zeros((1, 512), device=DEVICE)
        torch.save(empty_features.cpu(), out_path)
        print(f"  Saved empty features: {out_path}")
        return
    
    print(f"  ✓ Found {num_frames} frames with hands")
    
    # Resize frames to 224x224 for SqueezeNet (frames are already cropped to 600x600)
    # Convert from (T, C, H, W) to list for transform
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
    torch.save(encoder_out.cpu(), out_path)
    print(f"  ✓ Saved features ({encoder_out.shape[0]} frames, {encoder_out.shape[1]} dims): {out_path}")

def main():
    parser = argparse.ArgumentParser(description="SLR Feature Extraction/Training Pipeline")
    parser.add_argument("--features", action="store_true", help="Use only precomputed features from the features folder")
    parser.add_argument("--only", action="store_true", help="Train only on available features, ignore missing videos")
    args = parser.parse_args()

    if args.features:
        print("[INFO] Using only precomputed features from the features folder.")
        available_features = set(f for f in os.listdir(FEATURES_DIR) if f.endswith(".pt"))
        print(f"Found {len(available_features)} feature files.")
        for f in available_features:
            print(f"Feature: {f}")
        return

    for root, dirs, files in os.walk(VIDEO_DIR):
        for file in files:
            if file.endswith(".mp4") or file.endswith(".avi"):
                video_path = os.path.join(root, file)
                video_id = os.path.splitext(file)[0]
                out_path = os.path.join(FEATURES_DIR, f"{video_id}.pt")
                if os.path.exists(out_path):
                    print(f"Features already exist for {video_id}, skipping.")
                    continue
                extract_features_from_video(video_path, out_path)

if __name__ == "__main__":
    main()
