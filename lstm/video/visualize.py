import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)  # Only show ERROR logs, hide INFO/DEBUG
import matplotlib.pyplot as plt
import math
import torch
from config import config

rows = int(math.sqrt(config.max_frames))
cols = config.max_frames // rows

def visualize_frames(frames):
    if frames.ndim == 5:
        frames = frames.squeeze(0)

    fig, ax = plt.subplots(rows, cols, figsize=(20,20))
    idx = 0

    for i in range(rows):
        for j in range(cols):
            if idx < frames.shape[0]:
                frame = frames[idx].permute(1,2,0).cpu()

                # normalize
                mn, mx = frame.min(), frame.max()
                frame = (frame - mn) / (mx - mn) if mx > mn else frame
                ax[i][j].imshow(frame)
                ax[i][j].axis('off')

            idx += 1

    plt.tight_layout()
    plt.show()
