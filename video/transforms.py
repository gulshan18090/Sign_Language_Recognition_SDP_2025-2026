import torch
from torchvision.transforms import Compose, Lambda, Resize

def apply_video_transforms(resize_size=224):
    return Compose([
        Lambda(lambda x: torch.stack([
            Resize((resize_size, resize_size))(frame) for frame in x
        ]))
    ])
