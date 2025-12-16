"""
Visualize Features as Images (Reverse Engineering)
Takes 512-dim features and creates visual representations:
- Reshape features into 2D images
- Show feature activation maps
- Display features as reconstructed frames
"""
import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2

def load_feature(feature_path):
    """Load a .pt feature file"""
    features = torch.load(feature_path, map_location='cpu')
    return features.numpy() if isinstance(features, torch.Tensor) else features

def features_to_image_grid(features, method='reshape'):
    """
    Convert 512-dim features to visual representations
    
    Methods:
    - 'reshape': Reshape 512 dims to closest square (32x16)
    - 'blocks': Show as colored blocks grid
    - 'heatmap': Show as activation heatmap
    """
    num_frames, feat_dim = features.shape
    
    if method == 'reshape':
        # Reshape 512 -> 32x16 image
        h, w = 32, 16
        images = []
        for i in range(num_frames):
            feat = features[i].reshape(h, w)
            # Normalize to [0, 255]
            feat_norm = ((feat - feat.min()) / (feat.max() - feat.min() + 1e-8) * 255).astype(np.uint8)
            # Apply colormap
            img_colored = cv2.applyColorMap(feat_norm, cv2.COLORMAP_JET)
            images.append(img_colored)
        return images
    
    elif method == 'blocks':
        # Show as 16x32 blocks (each block = 1 feature dim)
        block_h, block_w = 16, 32
        images = []
        for i in range(num_frames):
            feat = features[i].reshape(block_h, block_w)
            # Normalize to [0, 255]
            feat_norm = ((feat - feat.min()) / (feat.max() - feat.min() + 1e-8) * 255).astype(np.uint8)
            # Resize for better visibility
            img_large = cv2.resize(feat_norm, (512, 512), interpolation=cv2.INTER_NEAREST)
            # Apply colormap
            img_colored = cv2.applyColorMap(img_large, cv2.COLORMAP_VIRIDIS)
            images.append(img_colored)
        return images
    
    elif method == 'heatmap':
        # Show first 256 dims as 16x16 heatmap
        images = []
        for i in range(num_frames):
            feat = features[i, :256].reshape(16, 16)
            # Normalize to [0, 255]
            feat_norm = ((feat - feat.min()) / (feat.max() - feat.min() + 1e-8) * 255).astype(np.uint8)
            # Resize for better visibility
            img_large = cv2.resize(feat_norm, (512, 512), interpolation=cv2.INTER_LINEAR)
            # Apply colormap
            img_colored = cv2.applyColorMap(img_large, cv2.COLORMAP_HOT)
            images.append(img_colored)
        return images

def create_grid_visualization(images, grid_cols=5, title="Feature Visualization"):
    """
    Create a grid visualization like the example image
    """
    num_images = len(images)
    grid_rows = (num_images + grid_cols - 1) // grid_cols
    
    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(grid_cols*3, grid_rows*3))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Flatten axes for easy iteration
    if grid_rows == 1 and grid_cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if grid_rows > 1 else [axes] if grid_cols == 1 else axes
    
    for idx in range(grid_rows * grid_cols):
        ax = axes[idx] if isinstance(axes, (list, np.ndarray)) else axes
        
        if idx < num_images:
            # Convert BGR to RGB for display
            img_rgb = cv2.cvtColor(images[idx], cv2.COLOR_BGR2RGB)
            ax.imshow(img_rgb)
            ax.set_title(f'Frame {idx}', fontsize=10)
        else:
            # Empty subplot
            ax.axis('off')
        
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    return fig

def visualize_all_methods(features, video_name, save_dir):
    """Create visualizations for all methods"""
    methods = {
        'reshape': 'Reshaped Features (32x16)',
        'blocks': 'Block Features (16x32)',
        'heatmap': 'Heatmap Features (16x16)'
    }
    
    for method, method_title in methods.items():
        print(f"  Generating {method} visualization...")
        images = features_to_image_grid(features, method=method)
        
        # Create grid
        fig = create_grid_visualization(
            images, 
            grid_cols=5, 
            title=f"{video_name} - {method_title}"
        )
        
        # Save
        save_path = os.path.join(save_dir, f"{method}_grid.png")
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"    ✓ Saved: {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Visualize Features as Images")
    parser.add_argument("--feature_file", type=str, required=True, help="Path to .pt feature file")
    parser.add_argument("--output_dir", type=str, default="feature_images", help="Output directory")
    parser.add_argument("--method", type=str, default="all", 
                       choices=['reshape', 'blocks', 'heatmap', 'all'],
                       help="Visualization method")
    args = parser.parse_args()
    
    feature_path = args.feature_file
    output_dir = args.output_dir
    method = args.method
    
    # Load features
    if not os.path.exists(feature_path):
        print(f"❌ Feature file not found: {feature_path}")
        return
    
    print(f"Loading features from: {feature_path}")
    features = load_feature(feature_path)
    
    video_name = Path(feature_path).stem
    print(f"Video: {video_name}")
    print(f"Features shape: {features.shape} (frames={features.shape[0]}, dims={features.shape[1]})")
    
    # Create output directory
    video_output_dir = os.path.join(output_dir, video_name)
    os.makedirs(video_output_dir, exist_ok=True)
    
    print(f"\nGenerating visualizations...")
    
    if method == 'all':
        visualize_all_methods(features, video_name, video_output_dir)
    else:
        print(f"  Generating {method} visualization...")
        images = features_to_image_grid(features, method=method)
        fig = create_grid_visualization(
            images, 
            grid_cols=5, 
            title=f"{video_name} - {method}"
        )
        save_path = os.path.join(video_output_dir, f"{method}_grid.png")
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"    ✓ Saved: {save_path}")
    
    print(f"\n✅ Visualizations saved to: {video_output_dir}")

if __name__ == "__main__":
    main()

