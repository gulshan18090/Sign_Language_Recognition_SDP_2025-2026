import cv2
import math
import torch
import mediapipe as mp
import matplotlib.pyplot as plt
from inference.EncoderRNN import *
from inference.AttnDecoderRNN import *
from inference.lstm_attention_inference import *
import time
from torchvision.models import squeezenet1_1
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.transforms import (
    Compose,
    Normalize,
    CenterCrop,
    Resize
)

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False, max_num_hands=2,
                        min_detection_confidence=0.5, min_tracking_confidence=0.8)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

frames = torch.zeros((0,3,480,640)).to(device)

max_frames = 64
img_size = 224
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
color = (255, 0, 0)
thickness = 2
recognize = False
sentence = " - "
text_coord = (10, 450)

def show_frames(list1):
    global frames
    print(list1)
    frames.clear()

def begin_threading(starter):
    global start_thread
    if starter:
        t1.start()
    start_thread=False

def apply_video_transforms(resize_size: int = img_size):
    return Compose([
        CenterCrop(500),
        Resize(size=(resize_size, resize_size))
    ])

def visualize_frames(frames, output_words, top_attn_mask):
    rows = int(math.sqrt(max_frames))
    cols = max_frames//rows
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(13,13))

    out_split = output_words.split()

    idx = 0
    for i in range(rows):
        for j in range(cols):
            img = frames[idx,:,:,:].permute(1,2,0).cpu()
            axes[i, j].imshow(img)

            attn_words = ''
            for widx in range(len(out_split)-1):
                if top_attn_mask[widx,i*cols+j]:
                    attn_words += out_split[widx] + ' '

            axes[i, j].set_title(attn_words)
            axes[i, j].axis('off')
            idx += 1

    plt.subplots_adjust(left=0.05,
                    bottom=0.05,
                    right=0.95,
                    top=0.95,
                    wspace=0.4,
                    hspace=0.4)
    plt.show()

model = squeezenet1_1(pretrained=True).to(device)
return_nodes = {
      'features.12.cat': 'layer12'
      }
pretrained_model = create_feature_extractor(model, return_nodes=return_nodes).to(device)
pretrained_model.eval()

def frame_to_feats(pretrained_model, frames):
  features = pretrained_model(frames.squeeze())['layer12'].to(device=device)
  feat_shape = features.shape
  feat_flat =  torch.reshape(features,(1,feat_shape[0],feat_shape[1]*feat_shape[2]*feat_shape[3])).to(device=device)
  return feat_flat

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    img = cv2.flip(frame, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        if recognize == False:
            recognize = True

        cv2.circle(img, (300, 30), 30, (0, 0, 255), cv2.FILLED)

        frames = torch.cat((frames, torch.unsqueeze(torch.from_numpy(imgRGB).permute(2, 0, 1),dim=0)/255.0),0)
    else:
        if recognize == True:
            # Recognize the collected frames
            measure = time.time()

            video_frames = frames
            print('List to tensor conversion:',time.time() - measure)
            measure = time.time()

            n,l,w,h = video_frames.shape

            video_frames = apply_video_transforms()(video_frames)
            print('Transformation:',time.time() - measure)
            measure = time.time()
            print('Shape:',video_frames.shape)

            compliment_arr = torch.zeros(max(0,max_frames-n),l,img_size,img_size).to(device)
            video_frames = torch.cat((video_frames,compliment_arr),0)

            feats = frame_to_feats(pretrained_model, video_frames)
            print('Feature shape:',feats.shape)

            output_words, attentions = evaluate(feats)
            top_attn_cnt = 8
            top_attn_mask =  attentions > torch.topk(attentions,top_attn_cnt)[0].min(dim=1)[0].unsqueeze(1) # find top 'top_attn_cnt' values, get the minimum for each row and return bitmask for top 3 attn
            # attn_max_val, attn_max_val_idx = torch.topk(attentions,3)
            visualize_frames(video_frames,output_words, top_attn_mask)

            sentence = output_words
            frames = torch.zeros((0,3,480,640)).to(device)

            recognize = False

        cv2.circle(img, (300, 30), 30, (0, 255, 0), cv2.FILLED)

        cv2.putText(img, sentence, text_coord, font, fontScale,
                        color, thickness, cv2.LINE_4, False)

    cv2.imshow('frame', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        break

# After the loop release the cap object
cap.release()

# Destroy all the windows
cv2.destroyAllWindows()