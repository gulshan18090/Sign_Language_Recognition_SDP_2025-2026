"""
Validate Feature Quality
Check if features properly capture video information by:
1. Loading original frames from video_frames folder
2. Loading corresponding features
3. Checking if features are diverse (not all the same)
4. Verifying feature dimensions match architecture
5. Comparing features between different frames
"""
import os
import torch
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def load_feature(feature_path):
    """Load a .pt feature file"""
    features = torch.load(feature_path, map_location='cpu', weights_only=False)
    return features.numpy() if isinstance(features, torch.Tensor) else features

def load_frames_from_folder(frames_dir):
    """Load all frames from a video frames folder"""
    if not os.path.exists(frames_dir):
        return None
    
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
    frames = []
    for frame_file in frame_files:
        frame_path = os.path.join(frames_dir, frame_file)
        frame = cv2.imread(frame_path)
        if frame is not None:
            frames.append(frame)
    return frames

def validate_features(video_name, features_dir="features", frames_dir="video_frames"):
    """Validate features for a single video"""
    print(f"\n{'='*70}")
    print(f"VALIDATING: {video_name}")
    print(f"{'='*70}")
    
    # Load features
    feature_path = os.path.join(features_dir, f"{video_name}.pt")
    if not os.path.exists(feature_path):
        print(f"❌ Feature file not found: {feature_path}")
        return False
    
    features = load_feature(feature_path)
    print(f"✓ Loaded features: {features.shape}")
    
    # Load frames
    video_frames_dir = os.path.join(frames_dir, video_name)
    frames = load_frames_from_folder(video_frames_dir)
    
    if frames is None:
        print(f"⚠️  No frames folder found (this is OK if using old extraction method)")
    else:
        print(f"✓ Loaded frames: {len(frames)} frames")
        if len(frames) != features.shape[0]:
            print(f"⚠️  WARNING: Frame count mismatch! Frames={len(frames)}, Features={features.shape[0]}")
    
    # Validation checks
    print(f"\n{'─'*70}")
    print("VALIDATION CHECKS:")
    print(f"{'─'*70}")
    
    # Check 1: Feature dimensions
    expected_dim = 512  # hidden=256, bidirectional=True -> 512
    if features.shape[1] == expected_dim:
        print(f"✅ Feature dimension: {features.shape[1]} (matches expected {expected_dim})")
    else:
        print(f"❌ Feature dimension: {features.shape[1]} (expected {expected_dim})")
        return False
    
    # Check 2: Features are not all zeros
    if np.all(features == 0):
        print(f"❌ All features are ZERO! No information captured!")
        return False
    else:
        print(f"✅ Features contain non-zero values")
    
    # Check 3: Features have diversity (not all the same)
    feature_std = features.std(axis=0).mean()
    if feature_std < 0.01:
        print(f"❌ Features have very low diversity (std={feature_std:.6f})")
        print(f"   This suggests features are too similar - possible issue!")
        return False
    else:
        print(f"✅ Features have good diversity (avg std={feature_std:.4f})")
    
    # Check 4: Frame-to-frame variation
    if features.shape[0] > 1:
        frame_diffs = np.diff(features, axis=0)
        avg_diff = np.abs(frame_diffs).mean()
        if avg_diff < 0.01:
            print(f"⚠️  Low frame-to-frame variation (avg diff={avg_diff:.6f})")
            print(f"   Frames might be too similar or model not capturing motion")
        else:
            print(f"✅ Good frame-to-frame variation (avg diff={avg_diff:.4f})")
    
    # Check 5: Feature statistics
    print(f"\n{'─'*70}")
    print("FEATURE STATISTICS:")
    print(f"{'─'*70}")
    print(f"Mean:     {features.mean():.4f}")
    print(f"Std:      {features.std():.4f}")
    print(f"Min:      {features.min():.4f}")
    print(f"Max:      {features.max():.4f}")
    print(f"Range:    {features.max() - features.min():.4f}")
    
    # Check 6: Per-frame statistics
    frame_means = features.mean(axis=1)
    frame_stds = features.std(axis=1)
    
    mean_variance = frame_means.std()
    std_variance = frame_stds.std()
    
    print(f"\nPer-frame statistics:")
    print(f"Mean variance across frames: {mean_variance:.4f}")
    print(f"Std variance across frames:  {std_variance:.4f}")
    
    if mean_variance < 0.01:
        print(f"⚠️  WARNING: Very low variance between frames!")
        print(f"   All frames produce similar features - possible issue!")
    else:
        print(f"✅ Good variance between frames")
    
    # Check 7: Visualize feature similarity matrix
    if features.shape[0] > 1:
        # Compute cosine similarity between all frame pairs
        from sklearn.metrics.pairwise import cosine_similarity
        similarity_matrix = cosine_similarity(features)
        
        # Check if all frames are too similar (>0.95 similarity)
        off_diagonal = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
        avg_similarity = off_diagonal.mean()
        
        print(f"\nFrame similarity (cosine):")
        print(f"Average similarity: {avg_similarity:.4f}")
        
        if avg_similarity > 0.95:
            print(f"⚠️  WARNING: Frames are very similar (>{avg_similarity:.2f})")
            print(f"   This might indicate all frames look the same to the model")
        else:
            print(f"✅ Frames have good diversity (similarity={avg_similarity:.2f})")
    
    print(f"\n{'='*70}")
    print(f"✅ VALIDATION PASSED for {video_name}")
    print(f"{'='*70}\n")
    
    return True

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Validate Feature Quality")
    parser.add_argument("--video_name", type=str, help="Video name (without .pt extension)")
    parser.add_argument("--features_dir", type=str, default="features", help="Features directory")
    parser.add_argument("--frames_dir", type=str, default="video_frames", help="Frames directory")
    parser.add_argument("--test_random", type=int, default=5, help="Test N random videos")
    args = parser.parse_args()
    
    if args.video_name:
        # Test single video
        validate_features(args.video_name, args.features_dir, args.frames_dir)
    else:
        # Test random videos
        print(f"Testing {args.test_random} random videos...\n")
        
        # Get all feature files
        feature_files = [f[:-3] for f in os.listdir(args.features_dir) if f.endswith('.pt')]
        
        if len(feature_files) == 0:
            print(f"❌ No feature files found in {args.features_dir}")
            return
        
        # Test random sample
        import random
        sample_videos = random.sample(feature_files, min(args.test_random, len(feature_files)))
        
        passed = 0
        failed = 0
        
        for video_name in sample_videos:
            if validate_features(video_name, args.features_dir, args.frames_dir):
                passed += 1
            else:
                failed += 1
        
        print(f"\n{'='*70}")
        print(f"SUMMARY: {passed}/{len(sample_videos)} videos passed validation")
        if failed > 0:
            print(f"⚠️  {failed} videos failed - check feature extraction!")
        else:
            print(f"✅ All videos passed - features look good!")
        print(f"{'='*70}\n")

if __name__ == "__main__":
    main()

