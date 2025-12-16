import torch
from torchvision.models import squeezenet1_1, SqueezeNet1_1_Weights
from torchvision.models.feature_extraction import create_feature_extractor
import torchvision.transforms as T
from PIL import Image
import numpy as np

def get_cnn_feature_extractor(device='cpu'):
    model = squeezenet1_1(weights=SqueezeNet1_1_Weights.DEFAULT).to(device)
    extractor = create_feature_extractor(model, return_nodes={'features.12.cat': 'layer12'})
    extractor.eval()
    return extractor

def image_to_feature(image, extractor, device='cpu'):
    # image: PIL Image or np.ndarray (H,W,C)
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = extractor(img_tensor)['layer12']
    return features.flatten().cpu().numpy()

def generate_image_features(image_list, device='cpu'):
    extractor = get_cnn_feature_extractor(device)
    features = [image_to_feature(img, extractor, device) for img in image_list]
    return np.stack(features)

# Example usage:
# imgs = [Image.open('frame1.jpg'), Image.open('frame2.jpg')]
# feats = generate_image_features(imgs, device='cuda')
# print(feats.shape)  # (num_images, feature_dim)
