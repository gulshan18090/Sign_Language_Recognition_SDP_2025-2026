"""
Frame Reconstruction from CNN Features
Reconstructs frames from saved CNN features using a trained decoder
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CNNDecoder(nn.Module):
    """
    Decoder to reconstruct frames from SqueezeNet features
    Input: (86528,) flattened CNN features from SqueezeNet layer 12 (512×13×13)
    Output: (3, 224, 224) reconstructed image
    """
    def __init__(self):
        super(CNNDecoder, self).__init__()
        
        # Reshape 86528 -> (512, 13, 13)
        # Then progressively upsample to (3, 224, 224)
        
        # 512×13×13 -> 256×26×26
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        
        # 256×26×26 -> 128×52×52
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        # 128×52×52 -> 64×104×104
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        # 64×104×104 -> 32×208×208
        self.up4 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(32)
        
        # 32×208×208 -> 3×224×224 (add padding to reach exact size)
        self.up5 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1, output_padding=1)
        
    def forward(self, x):
        # x: (batch, 86528)
        batch_size = x.size(0)
        
        # Reshape to feature maps: (batch, 512, 13, 13)
        x = x.view(batch_size, 512, 13, 13)
        
        # Progressive upsampling
        x = F.relu(self.bn1(self.up1(x)))  # -> (batch, 256, 26, 26)
        x = F.relu(self.bn2(self.up2(x)))  # -> (batch, 128, 52, 52)
        x = F.relu(self.bn3(self.up3(x)))  # -> (batch, 64, 104, 104)
        x = F.relu(self.bn4(self.up4(x)))  # -> (batch, 32, 208, 208)
        x = torch.tanh(self.up5(x))         # -> (batch, 3, 224, 224)
        
        return x

def denormalize_image(tensor):
    """
    Denormalize image from SqueezeNet normalization
    Input: (3, H, W) tensor normalized with mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    Output: (H, W, 3) numpy array in [0, 255]
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    # Denormalize: x_original = x_normalized * std + mean
    img = tensor * std + mean
    
    # Clip to [0, 1] and convert to [0, 255]
    img = torch.clamp(img, 0, 1)
    img = img.permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
    img = (img * 255).astype(np.uint8)
    
    return img

def reconstruct_from_cnn_features(cnn_features_path, decoder_model_path, output_dir):
    """
    Reconstruct frames from saved CNN features
    
    Args:
        cnn_features_path: Path to .pt file with CNN features (T, 86528)
        decoder_model_path: Path to trained decoder weights
        output_dir: Directory to save reconstructed frames
    """
    # Load CNN features
    cnn_features = torch.load(cnn_features_path, map_location=DEVICE)  # (T, 86528)
    print(f"Loaded CNN features: {cnn_features.shape}")
    
    # Load decoder
    decoder = CNNDecoder().to(DEVICE)
    if os.path.exists(decoder_model_path):
        decoder.load_state_dict(torch.load(decoder_model_path, map_location=DEVICE))
        print(f"Loaded decoder from: {decoder_model_path}")
    else:
        print(f"⚠️  Decoder not trained! Using random initialization.")
        print(f"   To get good reconstruction, train the decoder first.")
    
    decoder.eval()
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Reconstruct each frame
    with torch.no_grad():
        for i, features in enumerate(cnn_features):
            features_batch = features.unsqueeze(0).to(DEVICE)  # (1, 86528)
            
            # Decode
            reconstructed = decoder(features_batch)  # (1, 3, 224, 224)
            reconstructed = reconstructed.squeeze(0)  # (3, 224, 224)
            
            # Denormalize
            img = denormalize_image(reconstructed)
            
            # Save
            output_path = os.path.join(output_dir, f"frame_{i:04d}.jpg")
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, img_bgr)
            
            if i % 10 == 0:
                print(f"  Reconstructed frame {i+1}/{len(cnn_features)}")
    
    print(f"✅ Saved {len(cnn_features)} reconstructed frames to: {output_dir}")

def train_decoder(cnn_features_dir, original_frames_dir, decoder_save_path, epochs=50):
    """
    Train decoder to reconstruct frames from CNN features
    
    Args:
        cnn_features_dir: Directory with CNN feature .pt files
        original_frames_dir: Directory with original video frame images
        decoder_save_path: Where to save trained decoder
        epochs: Number of training epochs
    """
    print("=" * 60)
    print("Training CNN Decoder for Frame Reconstruction")
    print("=" * 60)
    
    # TODO: Implement training loop
    # This requires:
    # 1. Loading paired CNN features + original frames
    # 2. Decoder forward pass
    # 3. MSE loss between reconstructed and original
    # 4. Backpropagation
    
    print("⚠️  Training not implemented yet!")
    print("This requires original frame images paired with CNN features.")
    print("Use generate_image_features_batch.py with --save_frames to get frame images.")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Reconstruct frames from CNN features")
    parser.add_argument("--cnn_features", type=str, required=True, help="Path to CNN features .pt file")
    parser.add_argument("--decoder", type=str, default="decoder.pth", help="Path to decoder model")
    parser.add_argument("--output", type=str, default="reconstructed_frames", help="Output directory")
    parser.add_argument("--train", action="store_true", help="Train decoder instead of reconstructing")
    args = parser.parse_args()
    
    if args.train:
        print("Decoder training not implemented yet.")
        print("For now, you can:")
        print("1. Use extract_features_reconstructable.py --cnn to save CNN features")
        print("2. Use this script without --train to see random decoder output")
        print("3. Implement training loop with paired data")
    else:
        reconstruct_from_cnn_features(args.cnn_features, args.decoder, args.output)

if __name__ == "__main__":
    main()
