"""
Visualize Visual Features (CNN+LSTM outputs)
Shows:
- Feature tensor heatmap
- Feature statistics
- Frame-by-frame feature evolution
- PCA/t-SNE dimensionality reduction
"""
import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_feature(feature_path):
    """Load a .pt feature file"""
    features = torch.load(feature_path, map_location='cpu')
    return features.numpy() if isinstance(features, torch.Tensor) else features

def visualize_feature_heatmap(features, title, save_path=None):
    """
    Visualize feature tensor as a heatmap
    features: (num_frames, feature_dim) e.g., (24, 512)
    """
    plt.figure(figsize=(14, 8))
    
    # Plot heatmap
    sns.heatmap(features.T, cmap='viridis', cbar=True, 
                xticklabels=False, yticklabels=False)
    plt.title(f'{title}\nShape: {features.shape}', fontsize=14, fontweight='bold')
    plt.xlabel('Frame Index', fontsize=12)
    plt.ylabel('Feature Dimension (512)', fontsize=12)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved heatmap: {save_path}")
    else:
        plt.show()
    plt.close()

def visualize_feature_statistics(features, title, save_path=None):
    """
    Show statistics: mean, std, min, max per frame
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{title} - Statistics', fontsize=14, fontweight='bold')
    
    num_frames = features.shape[0]
    frame_indices = np.arange(num_frames)
    
    # Mean per frame
    frame_means = features.mean(axis=1)
    axes[0, 0].plot(frame_indices, frame_means, marker='o', linewidth=2)
    axes[0, 0].set_title('Mean Feature Value per Frame')
    axes[0, 0].set_xlabel('Frame Index')
    axes[0, 0].set_ylabel('Mean Value')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Std per frame
    frame_stds = features.std(axis=1)
    axes[0, 1].plot(frame_indices, frame_stds, marker='s', color='orange', linewidth=2)
    axes[0, 1].set_title('Std Dev per Frame')
    axes[0, 1].set_xlabel('Frame Index')
    axes[0, 1].set_ylabel('Std Dev')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Min/Max per frame
    frame_mins = features.min(axis=1)
    frame_maxs = features.max(axis=1)
    axes[1, 0].plot(frame_indices, frame_mins, marker='v', label='Min', linewidth=2)
    axes[1, 0].plot(frame_indices, frame_maxs, marker='^', label='Max', linewidth=2)
    axes[1, 0].set_title('Min/Max per Frame')
    axes[1, 0].set_xlabel('Frame Index')
    axes[1, 0].set_ylabel('Value')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Overall distribution histogram
    axes[1, 1].hist(features.flatten(), bins=50, color='purple', alpha=0.7, edgecolor='black')
    axes[1, 1].set_title('Overall Feature Distribution')
    axes[1, 1].set_xlabel('Feature Value')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved statistics: {save_path}")
    else:
        plt.show()
    plt.close()

def visualize_feature_evolution(features, title, save_path=None, num_dims=10):
    """
    Show how first N feature dimensions evolve over frames
    """
    plt.figure(figsize=(14, 6))
    
    num_frames = features.shape[0]
    frame_indices = np.arange(num_frames)
    
    # Plot first num_dims dimensions
    for dim in range(min(num_dims, features.shape[1])):
        plt.plot(frame_indices, features[:, dim], label=f'Dim {dim}', alpha=0.7, linewidth=1.5)
    
    plt.title(f'{title} - Feature Evolution (First {num_dims} Dims)', fontsize=14, fontweight='bold')
    plt.xlabel('Frame Index', fontsize=12)
    plt.ylabel('Feature Value', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved evolution: {save_path}")
    else:
        plt.show()
    plt.close()

def visualize_pca_2d(features, title, save_path=None):
    """
    PCA dimensionality reduction to 2D
    """
    try:
        from sklearn.decomposition import PCA
        
        if features.shape[0] < 2:
            print("  ⚠️  Not enough frames for PCA (need at least 2)")
            return
        
        pca = PCA(n_components=min(2, features.shape[0]))
        features_2d = pca.fit_transform(features)
        
        plt.figure(figsize=(10, 8))
        
        # Color by frame index (temporal evolution)
        scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                             c=np.arange(len(features_2d)), 
                             cmap='viridis', s=100, alpha=0.7, edgecolors='black')
        
        # Add frame numbers
        for i, (x, y) in enumerate(features_2d):
            plt.annotate(str(i), (x, y), fontsize=8, ha='center', va='center')
        
        plt.colorbar(scatter, label='Frame Index')
        plt.title(f'{title} - PCA 2D Projection\nVariance Explained: {pca.explained_variance_ratio_.sum():.2%}', 
                  fontsize=14, fontweight='bold')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})', fontsize=12)
        if features_2d.shape[1] > 1:
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  ✓ Saved PCA: {save_path}")
        else:
            plt.show()
        plt.close()
    except ImportError:
        print("  ⚠️  scikit-learn not installed. Skip PCA visualization.")

def print_feature_summary(features, video_name):
    """Print text summary of features"""
    print(f"\n{'='*60}")
    print(f"Feature Summary: {video_name}")
    print(f"{'='*60}")
    print(f"Shape:          {features.shape}")
    print(f"Num Frames:     {features.shape[0]}")
    print(f"Feature Dim:    {features.shape[1]}")
    print(f"Mean:           {features.mean():.4f}")
    print(f"Std:            {features.std():.4f}")
    print(f"Min:            {features.min():.4f}")
    print(f"Max:            {features.max():.4f}")
    print(f"{'='*60}\n")

def main():
    parser = argparse.ArgumentParser(description="Visualize Visual Features")
    parser.add_argument("--feature_file", type=str, required=True, help="Path to .pt feature file")
    parser.add_argument("--output_dir", type=str, default="feature_visualizations", help="Output directory for plots")
    parser.add_argument("--show", action="store_true", help="Show plots instead of saving")
    args = parser.parse_args()
    
    feature_path = args.feature_file
    output_dir = args.output_dir
    show_plots = args.show
    
    # Load features
    if not os.path.exists(feature_path):
        print(f"❌ Feature file not found: {feature_path}")
        return
    
    print(f"Loading features from: {feature_path}")
    features = load_feature(feature_path)
    
    # Get video name
    video_name = Path(feature_path).stem
    
    # Print summary
    print_feature_summary(features, video_name)
    
    # Create output directory
    if not show_plots:
        os.makedirs(output_dir, exist_ok=True)
        video_output_dir = os.path.join(output_dir, video_name)
        os.makedirs(video_output_dir, exist_ok=True)
        print(f"Saving visualizations to: {video_output_dir}\n")
    
    # Generate visualizations
    print("Generating visualizations...")
    
    # 1. Heatmap
    heatmap_path = None if show_plots else os.path.join(video_output_dir, "heatmap.png")
    visualize_feature_heatmap(features, f"Feature Heatmap - {video_name}", heatmap_path)
    
    # 2. Statistics
    stats_path = None if show_plots else os.path.join(video_output_dir, "statistics.png")
    visualize_feature_statistics(features, video_name, stats_path)
    
    # 3. Evolution
    evolution_path = None if show_plots else os.path.join(video_output_dir, "evolution.png")
    visualize_feature_evolution(features, video_name, evolution_path, num_dims=20)
    
    # 4. PCA
    pca_path = None if show_plots else os.path.join(video_output_dir, "pca_2d.png")
    visualize_pca_2d(features, video_name, pca_path)
    
    if not show_plots:
        print(f"\n✅ All visualizations saved to: {video_output_dir}")
    else:
        print("\n✅ All visualizations displayed!")

if __name__ == "__main__":
    main()

