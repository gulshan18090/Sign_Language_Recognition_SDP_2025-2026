import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

VIDEO_PATH = r"c:\Users\Gulsh\Desktop\inference\Videos\Bu gün hava çox soyuqdur\translator_video_2_unknown_16.mp4"
OUTPUT_DIR = r"c:\Users\Gulsh\Desktop\inference\extracted_frames"
N_EXTRACT = 64

def read_all_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    cap.release()
    return frames

def visualize_all_frames(frames):
    total = len(frames)
    cols = 10
    rows = (total + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = axes.flatten()
    for i, frame in enumerate(frames):
        axes[i].imshow(frame)
        axes[i].set_title(f"#{i}", fontsize=6)
        axes[i].axis("off")
    for j in range(total, len(axes)):
        axes[j].axis("off")
    plt.suptitle(f"All {total} Frames - {os.path.basename(VIDEO_PATH)}", fontsize=12)
    plt.tight_layout()
    all_frames_path = r"c:\Users\Gulsh\Desktop\inference\all_frames_grid.png"
    plt.savefig(all_frames_path, dpi=80, bbox_inches="tight")
    print(f"All frames grid saved -> {all_frames_path}")
    plt.show()

def extract_64_frames(frames, output_dir, n=64):
    os.makedirs(output_dir, exist_ok=True)
    total = len(frames)
    if total <= n:
        indices = list(range(total))
    else:
        indices = [int(round(i * (total - 1) / (n - 1))) for i in range(n)]
    extracted = []
    for rank, idx in enumerate(indices):
        frame = frames[idx]
        extracted.append(frame)
        out_path = os.path.join(output_dir, f"frame_{rank:03d}_orig{idx:04d}.png")
        cv2.imwrite(out_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    print(f"Extracted {len(extracted)} frames -> {output_dir}")
    return extracted, indices

def visualize_64_frames(extracted, indices):
    cols = 8
    rows = 8  # 8x8 = 64
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = axes.flatten()
    for i, (frame, orig_idx) in enumerate(zip(extracted, indices)):
        axes[i].imshow(frame)
        axes[i].set_title(f"#{orig_idx}", fontsize=6)
        axes[i].axis("off")
    plt.suptitle(f"64 Extracted Frames - {os.path.basename(VIDEO_PATH)}", fontsize=12)
    plt.tight_layout()
    grid_path = r"c:\Users\Gulsh\Desktop\inference\extracted_64_grid.png"
    plt.savefig(grid_path, dpi=80, bbox_inches="tight")
    print(f"64-frame grid saved -> {grid_path}")
    plt.show()

if __name__ == "__main__":
    print("Reading video frames...")
    frames = read_all_frames(VIDEO_PATH)
    print(f"Total frames: {len(frames)}")

    print("\nVisualizing all frames...")
    visualize_all_frames(frames)

    print("\nExtracting 64 evenly-spaced frames...")
    extracted, indices = extract_64_frames(frames, OUTPUT_DIR, n=N_EXTRACT)

    print("\nVisualizing 64 extracted frames...")
    visualize_64_frames(extracted, indices)

    print("\nDone!")
