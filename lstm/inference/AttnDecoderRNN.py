import torch
import torch.nn as nn
import torch.nn.functional as F

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, device, dropout_p=0.1, max_length=64, biDirectional = False, debug=False): #max_length=config.max_words_in_sentence
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.debug = debug
        self.device = device

        self.D = 2 if biDirectional else 1

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 3, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.rnn = nn.LSTM(
                input_size = self.hidden_size,
                hidden_size = self.hidden_size*self.D,
                num_layers = 1,
                dropout = 0,
                bidirectional = biDirectional,
                batch_first = True)
        self.out = nn.Linear(self.hidden_size*self.D, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(input.shape[0],input.shape[1], self.hidden_size)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1).to(device=self.device)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),encoder_outputs).to(device=self.device)

        output = torch.cat((embedded[0], attn_applied[0]), 1).to(device=self.device)
        output = self.attn_combine(output).unsqueeze(0).to(device=self.device)

        output = F.relu(output)
        output, hidden = self.rnn(output, (hidden[0].unsqueeze(0),hidden[0].unsqueeze(0)))

        output = F.log_softmax(self.out(output[0]), dim=1).to(device=self.device)
        return output, hidden, attn_weights

    def initHidden(self):
        return (torch.zeros(self.D, 1, self.hidden_size*self.D, device=self.device),
                torch.zeros(self.D, 1, self.hidden_size*self.D, device=self.device))