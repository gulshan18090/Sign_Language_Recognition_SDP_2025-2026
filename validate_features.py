"""
Feature Validation Script
Checks if extracted features are correct and vary across videos
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_features():
    """Analyze all extracted features"""
    features_dir = "features"
    feature_files = list(Path(features_dir).glob("*.pt"))
    
    if len(feature_files) == 0:
        print("❌ No feature files found!")
        return
    
    print(f"📊 Analyzing {len(feature_files)} feature files...\n")
    
    # Statistics
    shapes = []
    means = []
    stds = []
    norms = []
    all_zero = []
    
    # Sample 20 random features for detailed analysis
    import random
    sample_files = random.sample(feature_files, min(20, len(feature_files)))
    
    print("=" * 70)
    print("FEATURE FILE ANALYSIS")
    print("=" * 70)
    
    for i, feat_file in enumerate(sample_files, 1):
        try:
            features = torch.load(feat_file, weights_only=True)
            
            # Basic stats
            shape = features.shape
            mean = features.mean().item()
            std = features.std().item()
            norm = torch.norm(features).item()
            is_zero = (features.abs().sum().item() == 0)
            
            shapes.append(shape)
            means.append(mean)
            stds.append(std)
            norms.append(norm)
            all_zero.append(is_zero)
            
            status = "❌ ALL ZEROS" if is_zero else "✓"
            
            print(f"{i:2d}. {feat_file.name[:40]:40s} | Shape: {str(shape):15s} | "
                  f"Mean: {mean:7.4f} | Std: {std:7.4f} | Norm: {norm:8.2f} | {status}")
            
        except Exception as e:
            print(f"{i:2d}. {feat_file.name[:40]:40s} | ❌ Error: {e}")
    
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    
    # Overall statistics
    print(f"Total feature files: {len(feature_files)}")
    print(f"Files with all zeros: {sum(all_zero)} ({100*sum(all_zero)/len(all_zero):.1f}%)")
    print(f"\nShape distribution:")
    from collections import Counter
    shape_counts = Counter([str(s) for s in shapes])
    for shape, count in shape_counts.most_common():
        print(f"  {shape}: {count} files ({100*count/len(shapes):.1f}%)")
    
    print(f"\nValue statistics across all features:")
    print(f"  Mean: {np.mean(means):.4f} ± {np.std(means):.4f}")
    print(f"  Std:  {np.mean(stds):.4f} ± {np.std(stds):.4f}")
    print(f"  Norm: {np.mean(norms):.2f} ± {np.std(norms):.2f}")
    
    # Feature diversity check
    print(f"\n" + "=" * 70)
    print("FEATURE DIVERSITY CHECK")
    print("=" * 70)
    
    # Load 5 random features and compare
    test_features = []
    test_names = []
    for feat_file in random.sample(feature_files, min(5, len(feature_files))):
        feat = torch.load(feat_file, weights_only=True)
        test_features.append(feat)
        test_names.append(feat_file.name)
    
    print("\nPairwise cosine similarity (should be LOW for diverse features):")
    from torch.nn.functional import cosine_similarity
    
    for i in range(len(test_features)):
        for j in range(i+1, len(test_features)):
            # Average cosine similarity across time dimension
            feat_i = test_features[i]
            feat_j = test_features[j]
            
            # Pad to same length
            max_len = max(feat_i.shape[0], feat_j.shape[0])
            if feat_i.shape[0] < max_len:
                pad = torch.zeros(max_len - feat_i.shape[0], feat_i.shape[1])
                feat_i = torch.cat([feat_i, pad], dim=0)
            if feat_j.shape[0] < max_len:
                pad = torch.zeros(max_len - feat_j.shape[0], feat_j.shape[1])
                feat_j = torch.cat([feat_j, pad], dim=0)
            
            sim = cosine_similarity(feat_i, feat_j, dim=1).mean().item()
            print(f"  {test_names[i][:30]:30s} <-> {test_names[j][:30]:30s}: {sim:.4f}")
    
    print(f"\n💡 Good features should have LOW similarity (< 0.5) between different videos")
    print(f"💡 High similarity (> 0.9) suggests features don't capture video differences")
    
    return shapes, means, stds, norms, all_zero

def visualize_features():
    """Visualize feature tensors"""
    features_dir = "features"
    feature_files = list(Path(features_dir).glob("*.pt"))
    
    if len(feature_files) == 0:
        print("❌ No feature files found!")
        return
    
    # Select 4 random features to visualize
    import random
    sample_files = random.sample(feature_files, min(4, len(feature_files)))
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, feat_file in enumerate(sample_files):
        features = torch.load(feat_file, weights_only=True).numpy()
        
        # Plot as heatmap
        ax = axes[idx]
        im = ax.imshow(features.T, aspect='auto', cmap='viridis', interpolation='nearest')
        ax.set_title(f'{feat_file.name}\nShape: {features.shape}')
        ax.set_xlabel('Time (frames)')
        ax.set_ylabel('Feature dimension')
        plt.colorbar(im, ax=ax)
        
        # Add statistics
        mean = features.mean()
        std = features.std()
        ax.text(0.02, 0.98, f'Mean: {mean:.4f}\nStd: {std:.4f}', 
                transform=ax.transAxes, va='top', bbox=dict(boxstyle='round', 
                facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('feature_visualization.png', dpi=150, bbox_inches='tight')
    print(f"💾 Saved feature visualization to: feature_visualization.png")
    plt.close()

def main():
    print("\n" + "=" * 70)
    print("FEATURE VALIDATION TOOL")
    print("=" * 70 + "\n")
    
    shapes, means, stds, norms, all_zero = analyze_features()
    print()
    visualize_features()
    
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    
    if sum(all_zero) > 0:
        print("⚠️  WARNING: Some features are all zeros!")
        print("   → Re-extract features with hand detection")
    
    if len(set([str(s) for s in shapes])) > 3:
        print("⚠️  WARNING: High variability in feature shapes!")
        print("   → Check if padding/truncation is working correctly")
    
    if np.mean(stds) < 0.01:
        print("⚠️  WARNING: Very low standard deviation in features!")
        print("   → Features may not be diverse enough")
        print("   → Check if feature extraction is working properly")
    
    print("\n✅ Feature validation complete!")

if __name__ == "__main__":
    main()

