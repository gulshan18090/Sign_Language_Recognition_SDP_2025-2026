import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)  # Only show ERROR logs, hide INFO/DEBUG
import torch
import torch.nn.functional as F
import random
from config import config


def train_step(input_tensor, target_tensor,
               encoder, decoder,
               enc_opt, dec_opt,
               criterion, encodings, max_grad_norm=5.0,
               teacher_forcing_ratio=0.5, epoch=0, total_epochs=20):
    """
    Training step with scheduled sampling.
    
    Args:
        teacher_forcing_ratio: Base probability of using teacher forcing (0.0-1.0)
                              Set to 1.0 for pure teacher forcing (old behavior)
                              Set to 0.5 for 50% scheduled sampling
        epoch: Current epoch number (used for curriculum learning)
        total_epochs: Total epochs (used for curriculum learning)
    
    Scheduled Sampling: Gradually reduces teacher forcing as training progresses.
    This forces the model to learn from its own predictions, preventing it from
    ignoring video features.
    """
    
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
    
    # Scheduled sampling: reduce teacher forcing as training progresses
    # Start with high teacher forcing, gradually decrease
    scheduled_ratio = teacher_forcing_ratio * (1 - epoch / total_epochs)
    
    # Get SOS token for starting
    current_input = decoder_input[:, 0].unsqueeze(1)  # Start with SOS
    
    for t in range(decoder_target.size(1)):
        output, decoder_hidden, attn_weights = decoder(
            current_input, decoder_hidden, encoder_output
        )

        # Fix: Use class indices directly for CrossEntropyLoss (no one-hot encoding)
        tgt = decoder_target[:, t]
        
        # CrossEntropyLoss expects (N, C) for output and (N,) for target
        loss += criterion(output, tgt)
        
        # Scheduled sampling: decide whether to use teacher forcing or model prediction
        use_teacher_forcing = random.random() < scheduled_ratio
        
        if use_teacher_forcing and t + 1 < decoder_target.size(1):
            # Teacher forcing: use ground truth as next input
            current_input = decoder_input[:, t + 1].unsqueeze(1)
        else:
            # Use model's prediction as next input
            # This forces the model to actually learn from video features
            predicted = output.argmax(dim=1)
            current_input = predicted.unsqueeze(1)

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
