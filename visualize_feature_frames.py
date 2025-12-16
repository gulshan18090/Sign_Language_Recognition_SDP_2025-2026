import torch
import matplotlib.pyplot as plt
import numpy as np
import sys

# Usage: python visualize_feature_frames.py <feature_file.pt>

def main(feature_path):
    features = torch.load(feature_path)
    # Features shape: [T, D] (e.g., [num_frames, 512])
    if isinstance(features, dict):
        # Some .pt files may store features under a key
        features = features[list(features.keys())[0]]
    features = features.cpu().numpy() if hasattr(features, 'cpu') else np.array(features)
    print(f"Loaded features shape: {features.shape}")

    # Visualize as an image: each row is a frame, each column is a feature dimension
    plt.figure(figsize=(12, 6))
    plt.imshow(features, aspect='auto', cmap='viridis')
    plt.colorbar(label='Feature Value')
    plt.xlabel('Feature Dimension')
    plt.ylabel('Frame Index')
    plt.title(f'Feature Frames Visualization\n{feature_path}')
    plt.tight_layout()
    # Save the image
    out_path = feature_path.replace('.pt', '_features.png')
    plt.savefig(out_path)
    print(f"Saved feature visualization to {out_path}")
    # Optionally, also show the image (uncomment if needed)
    # plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python visualize_feature_frames.py <feature_file.pt>")
        sys.exit(1)
    main(sys.argv[1])
