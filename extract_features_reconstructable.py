"""
Feature Extraction with Reconstruction Support
Saves both CNN features (for reconstruction) and LSTM features (for translation)
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
VIDEO_DIR = "drive/Video/Cam2"
FEATURES_DIR = "features"
CNN_FEATURES_DIR = "features_cnn"  # Raw CNN features for reconstruction
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ensure features directories exist
os.makedirs(FEATURES_DIR, exist_ok=True)
os.makedirs(CNN_FEATURES_DIR, exist_ok=True)

# SqueezeNet model
squeezenet = models.squeezenet1_1(pretrained=True).features.to(DEVICE)
squeezenet.eval()

# Encoder LSTM - Bidirectional
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

def extract_features_from_video(video_path, lstm_out_path, cnn_out_path=None):
    """
    Extract both LSTM features (for translation) and CNN features (for reconstruction)
    
    Args:
        video_path: Input video file
        lstm_out_path: Output path for LSTM features (512 dims) - for translation
        cnn_out_path: Output path for CNN features (86528 dims) - for reconstruction
    """
    # Extract all frames from video
    frames = extract_frames(video_path)
    if not frames:
        print(f"No frames found in {video_path}")
        return
    
    # Apply hand detection to keep only frames with hands
    print(f"  Detecting hands in {len(frames)} frames...")
    frames_with_hands = keep_frames_with_hands(frames, crop_size=600)
    
    num_frames = frames_with_hands.shape[0]
    if num_frames == 0:
        print(f"  ⚠️  No frames with hands detected in {video_path}")
        # Create empty features
        empty_lstm_features = torch.zeros((1, 512), device=DEVICE)
        torch.save(empty_lstm_features.cpu(), lstm_out_path)
        if cnn_out_path:
            empty_cnn_features = torch.zeros((1, 86528), device=DEVICE)
            torch.save(empty_cnn_features.cpu(), cnn_out_path)
        print(f"  Saved empty features")
        return
    
    print(f"  ✓ Found {num_frames} frames with hands")
    
    # Preprocess frames for SqueezeNet
    frames_list = []
    for i in range(num_frames):
        frame_tensor = frames_with_hands[i]  # (C, H, W)
        frame_np = frame_tensor.permute(1, 2, 0).cpu().numpy()  # (H, W, C)
        frame_np = (frame_np * 255).astype(np.uint8) if frame_np.max() <= 1.0 else frame_np.astype(np.uint8)
        processed_frame = transform(frame_np).to(DEVICE)
        frames_list.append(processed_frame)
    
    video_tensor = torch.stack(frames_list, dim=0)  # (T, C, H, W)
    
    # Extract CNN features (before LSTM)
    with torch.no_grad():
        # Process through SqueezeNet
        cnn_features_list = []
        for frame in video_tensor:
            frame_batch = frame.unsqueeze(0)  # (1, C, H, W)
            cnn_out = squeezenet(frame_batch)  # (1, 512, 13, 13)
            cnn_out_flat = cnn_out.view(1, -1)  # (1, 86528)
            cnn_features_list.append(cnn_out_flat)
        
        cnn_features = torch.cat(cnn_features_list, dim=0)  # (T, 86528)
        
        # Save CNN features for reconstruction
        if cnn_out_path:
            torch.save(cnn_features.cpu(), cnn_out_path)
            print(f"  ✓ Saved CNN features ({cnn_features.shape[0]} frames, {cnn_features.shape[1]} dims): {cnn_out_path}")
        
        # Extract LSTM features for translation
        video_tensor_batch = video_tensor.unsqueeze(0)  # (1, T, C, H, W)
        hidden = encoder.initHidden(1)
        encoder_out, _ = encoder(video_tensor_batch, hidden)  # (1, T, 512)
        encoder_out = encoder_out.squeeze(0)  # (T, 512)
        
        # Save LSTM features
        torch.save(encoder_out.cpu(), lstm_out_path)
        print(f"  ✓ Saved LSTM features ({encoder_out.shape[0]} frames, {encoder_out.shape[1]} dims): {lstm_out_path}")

def main():
    parser = argparse.ArgumentParser(description="Feature Extraction with Reconstruction Support")
    parser.add_argument("--cnn", action="store_true", help="Also save raw CNN features for reconstruction")
    parser.add_argument("--cnn-only", action="store_true", help="Save only CNN features (skip LSTM)")
    args = parser.parse_args()

    print("=" * 60)
    print("Feature Extraction Pipeline")
    print(f"Device: {DEVICE}")
    if args.cnn or args.cnn_only:
        print("Mode: Saving CNN features (86528 dims) for reconstruction")
    if not args.cnn_only:
        print("Mode: Saving LSTM features (512 dims) for translation")
    print("=" * 60)

    for root, dirs, files in os.walk(VIDEO_DIR):
        for file in files:
            if file.endswith(".mp4") or file.endswith(".avi"):
                video_path = os.path.join(root, file)
                video_id = os.path.splitext(file)[0]
                
                lstm_out_path = os.path.join(FEATURES_DIR, f"{video_id}.pt")
                cnn_out_path = os.path.join(CNN_FEATURES_DIR, f"{video_id}.pt") if (args.cnn or args.cnn_only) else None
                
                # Check if already processed
                skip = False
                if not args.cnn_only and os.path.exists(lstm_out_path):
                    print(f"LSTM features exist for {video_id}, skipping.")
                    skip = True
                if (args.cnn or args.cnn_only) and os.path.exists(cnn_out_path):
                    print(f"CNN features exist for {video_id}, skipping.")
                    skip = True
                
                if skip:
                    continue
                
                print(f"\nProcessing: {video_id}")
                extract_features_from_video(video_path, lstm_out_path, cnn_out_path)

    print("\n" + "=" * 60)
    print("✅ Feature extraction complete!")
    print(f"LSTM features (512 dims): {FEATURES_DIR}/")
    if args.cnn or args.cnn_only:
        print(f"CNN features (86528 dims): {CNN_FEATURES_DIR}/")
    print("=" * 60)

if __name__ == "__main__":
    main()
