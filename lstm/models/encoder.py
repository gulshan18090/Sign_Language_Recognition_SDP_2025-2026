
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)  # Only show ERROR logs, hide INFO/DEBUG
import torch
import torch.nn as nn
from config import config
from torchvision.models import squeezenet1_1, SqueezeNet1_1_Weights
from torchvision.models.feature_extraction import create_feature_extractor


class EncoderRNN(nn.Module):
    """Encoder for processing pre-extracted features or raw video frames"""
    def __init__(self, input_size, hidden_size, device, biDirectional=False, use_cnn=False):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device
        self.D = 2 if biDirectional else 1
        self.use_cnn = use_cnn

        # Only load CNN if explicitly requested (for feature extraction)
        if use_cnn:
            model = squeezenet1_1(weights=SqueezeNet1_1_Weights.DEFAULT).to(device)
            self.pretrained_model = create_feature_extractor(
                model, return_nodes={'features.12.cat': 'layer12'}
            )
            self.pretrained_model.eval()
        else:
            self.pretrained_model = None

        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            bidirectional=biDirectional,
            batch_first=True
        ).to(device)

    def forward(self, x, hidden):
        # Check if input is raw video frames (5D) or pre-extracted features (3D)
        if len(x.shape) == 5:
            # Raw video frames: (B, T, C, H, W)
            if self.pretrained_model is None:
                raise ValueError("Encoder was initialized without CNN but received raw video frames. "
                               "Set use_cnn=True when creating the encoder.")
            B, T, C, H, W = x.shape

            x = x.view(B*T, C, H, W)

            with torch.no_grad():
                features = self.pretrained_model(x)['layer12']

            _, c, h, w = features.shape
            features = features.view(B, T, c*h*w)
        elif len(x.shape) == 3:
            # Pre-extracted features: (B, T, H)
            features = x
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}. Expected 3D (B, T, H) or 5D (B, T, C, H, W)")

        output, hidden = self.rnn(features, hidden)
        return output, hidden

    def initHidden(self, batch_size):
        return (
            torch.zeros(self.D, batch_size, self.hidden_size, device=self.device),
            torch.zeros(self.D, batch_size, self.hidden_size, device=self.device)
        )
