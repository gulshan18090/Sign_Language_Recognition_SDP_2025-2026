import os
import random
import torch
import numpy as np
from PIL import Image

# Path to the features directory
features_dir = "./features"

# Output directory for images
output_dir = "./output_images"
os.makedirs(output_dir, exist_ok=True)

# Function to visualize a tensor as an image
def tensor_to_image(tensor, output_path):
    # Normalize the tensor to [0, 255] for visualization
    tensor = tensor - tensor.min()
    tensor = tensor / tensor.max()
    tensor = (tensor * 255).byte()

    # Convert to numpy array and save as image
    array = tensor.numpy()
    if len(array.shape) == 3 and array.shape[0] in [1, 3]:  # Handle channel-first tensors
        if array.shape[0] == 1:  # Grayscale image
            array = array[0]  # Remove channel dimension
        else:  # RGB image
            array = np.transpose(array, (1, 2, 0))
    elif len(array.shape) == 3:  # If more than 3 channels, take the first 3
        array = array[:, :, :3]
    elif len(array.shape) == 2:  # Grayscale
        pass
    else:
        raise ValueError("Unsupported tensor shape for visualization.")

    # Save the image
    image = Image.fromarray(array)
    image.save(output_path)

# Modify the script to process N random feature files

# Number of feature files to process
n = 5  # Change this value to process a different number of files

# Randomly select N .pt files
pt_files = [f for f in os.listdir(features_dir) if f.endswith(".pt")]
if not pt_files:
    print("No .pt files found in the features directory.")
    exit()

if len(pt_files) < n:
    print(f"Only {len(pt_files)} files available, processing all of them.")
    selected_files = pt_files
else:
    selected_files = random.sample(pt_files, n)

for random_file in selected_files:
    print(f"Processing file: {random_file}")

    # Load the tensor
    tensor_path = os.path.join(features_dir, random_file)
    tensor = torch.load(tensor_path)

    # Handle empty features (1, 256)
    if tensor.shape == (1, 256):
        side = int(tensor.shape[1] ** 0.5)
        tensor = tensor.reshape(1, side, side)  # Reshape to square

    # Handle extracted features (T, 256)
    elif len(tensor.shape) == 2 and tensor.shape[1] == 256:
        # Visualize the first row as an image
        tensor = tensor[0].reshape(1, 16, 16)  # Reshape to square

    # Save the tensor as an image
    output_path = os.path.join(output_dir, f"{random_file}.png")
    tensor_to_image(tensor, output_path)
    print(f"Image saved to {output_path}")