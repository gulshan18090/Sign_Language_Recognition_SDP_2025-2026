import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import squeezenet1_1
from torchvision.models.feature_extraction import create_feature_extractor

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, device, biDirectional = False):
        super(EncoderRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device
        self.D = 2 if biDirectional else 1

        self.rnn = nn.LSTM(
                input_size = self.input_size,
                hidden_size = self.hidden_size*self.D,
                num_layers = 1,
                dropout = 0,
                bidirectional = biDirectional,
                batch_first = True).to(device)

    def forward(self, input, hidden):
        output, hidden = self.rnn(input, hidden)
        return output, hidden

    def initHidden(self):
        return (torch.zeros(self.D, 1, self.hidden_size*self.D, device=self.device),
                torch.zeros(self.D, 1, self.hidden_size*self.D, device=self.device))