import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)  # Only show ERROR logs, hide INFO/DEBUG
import torch
import torch.nn.functional as F
from config import config


def train_step(input_tensor, target_tensor,
               encoder, decoder,
               enc_opt, dec_opt,
               criterion, encodings, max_grad_norm=5.0):

    enc_opt.zero_grad()
    dec_opt.zero_grad()

    B = input_tensor.size(0)
    hidden = encoder.initHidden(B)

    encoder_output, encoder_hidden = encoder(input_tensor, hidden)

    # If bidirectional, merge
    if encoder.D == 2:
        h = torch.cat((encoder_hidden[0][0:1], encoder_hidden[0][1:2]), dim=2)
        c = torch.cat((encoder_hidden[1][0:1], encoder_hidden[1][1:2]), dim=2)
        decoder_hidden = (h, c)
    else:
        decoder_hidden = encoder_hidden

    max_len = target_tensor.shape[1]
    # Fix: Remove extra dimension - decoder expects (B, T) not (B, T, 1)
    decoder_input = target_tensor[:, :max_len-2, :].squeeze(-1)
    decoder_target = target_tensor[:, 1:max_len-1, :].squeeze(-1)

    loss = 0
    for t in range(decoder_target.size(1)):
        output, decoder_hidden, _ = decoder(
            decoder_input[:, t].unsqueeze(1), decoder_hidden, encoder_output
        )

        # Fix: Use class indices directly for CrossEntropyLoss (no one-hot encoding)
        tgt = decoder_target[:, t]
        
        # CrossEntropyLoss expects (N, C) for output and (N,) for target
        loss += criterion(output, tgt)

    # Backward pass
    loss.backward()
    
    # Fix: Add gradient clipping to prevent exploding gradients
    torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_grad_norm)
    torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_grad_norm)
    
    enc_opt.step()
    dec_opt.step()

    # Fix: Correct loss normalization - divide by sequence length only
    # Loss is already averaged over batch by criterion (reduction='mean' by default)
    return loss.item() / decoder_target.size(1)
