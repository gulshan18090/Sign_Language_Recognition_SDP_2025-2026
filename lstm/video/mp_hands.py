import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)  # Only show ERROR logs, hide INFO/DEBUG

import torch
import numpy as np
import cv2
import logging
logging.getLogger("mediapipe").setLevel(logging.ERROR)  # Set MediaPipe logs to ERROR only

import os
import mediapipe as mp
from torchvision.transforms import CenterCrop
from config import config
import warnings
warnings.filterwarnings("ignore")
# Configure absl logging to suppress logs
import absl.logging
# Suppress Abseil logs
absl.logging.get_absl_handler().python_handler.stream = open(os.devnull, 'w')
absl.logging.set_verbosity(absl.logging.FATAL)
absl.logging.set_stderrthreshold(absl.logging.FATAL)
from torchvision.transforms import Compose, CenterCrop


def keep_frames_with_hands(video_data, crop_size=None):
    mp_hands = mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.8,
        min_tracking_confidence=0.9
    )

    # prepare output tensor
    if crop_size:
        out = torch.zeros((0,3,crop_size,crop_size), device=config.device)
        crop = CenterCrop(crop_size)
        h, w = crop_size, crop_size
    else:
        h, w = 960, 1280
        out = torch.zeros((0,3,h,w), device=config.device)

    for frame in video_data:
        # torch → numpy
        if torch.is_tensor(frame):
            if frame.ndim == 3 and frame.shape[0] == 3:
                frame_np = frame.permute(1,2,0).cpu().numpy()
            else:
                frame_np = frame.cpu().numpy()
        else:
            frame_np = frame

        frame_np = np.ascontiguousarray(frame_np).astype(np.uint8)

        # convert BGR to RGB if needed
        if config.video_processing_tool in ["OpenCV", "VidGear"]:
            if frame_np.shape[-1] == 3:
                rgb = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)
            else:
                continue
        else:
            rgb = frame_np if frame_np.shape[-1] == 3 else frame_np.transpose(1,2,0)

        # detect
        results = mp_hands.process(rgb)
        if results.multi_hand_landmarks:
            t = torch.from_numpy(rgb).permute(2,0,1).to(config.device)
            if crop_size:
                t = crop(t)
            out = torch.cat((out, t.unsqueeze(0)), dim=0)

    if out.shape[0] == 0:
        out = torch.zeros((config.max_frames, 3, h, w), device=config.device)

    return out
