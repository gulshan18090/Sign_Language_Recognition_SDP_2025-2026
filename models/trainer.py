import os
import time
import gc
import torch
import torch.optim as optim
import numpy as np
from config import config
from models.train_step import train_step
from data.dataloader import get_feature_dataloader

# ---------------------------------------------------------
# Clean & Correct Training Loop with Validation & Early Stopping
# ---------------------------------------------------------

def validate(encoder, decoder, val_loader, criterion, encodings, device):
    """Run validation and return average loss"""
    encoder.eval()
    decoder.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for x, y, fname in val_loader:
            try:
                x = x.to(device)
                y = y.to(device)
                
                B = x.size(0)
                hidden = encoder.initHidden(B)
                encoder_output, encoder_hidden = encoder(x, hidden)
                
                # Prepare decoder hidden
                if encoder.D == 2:
                    h = torch.cat((encoder_hidden[0][0:1], encoder_hidden[0][1:2]), dim=2)
                    c = torch.cat((encoder_hidden[1][0:1], encoder_hidden[1][1:2]), dim=2)
                    decoder_hidden = (h, c)
                else:
                    decoder_hidden = encoder_hidden
                
                max_len = y.shape[1]
                # Fix: Remove extra dimension - decoder expects (B, T) not (B, T, 1)
                decoder_input = y[:, :max_len-2, :].squeeze(-1)
                decoder_target = y[:, 1:max_len-1, :].squeeze(-1)
                
                loss = 0
                for t in range(decoder_target.size(1)):
                    output, decoder_hidden, _ = decoder(
                        decoder_input[:, t].unsqueeze(1), decoder_hidden, encoder_output
                    )
                    # Fix: Use class indices directly for CrossEntropyLoss (no one-hot encoding)
                    tgt = decoder_target[:, t]
                    loss += criterion(output, tgt)
                
                # Fix: Correct loss normalization - divide by sequence length only
                total_loss += loss.item() / decoder_target.size(1)
                num_batches += 1
            except Exception as e:
                continue
    
    encoder.train()
    decoder.train()
    
    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    return avg_loss

def trainIters(df, encoder, decoder, encodings,
               print_every=100, lr=0.01, epochs=15, 
               patience=5, min_delta=0.001):

    # Optimizers with learning rate scheduling
    enc_opt = optim.Adam(encoder.parameters(), lr=lr)
    dec_opt = optim.Adam(decoder.parameters(), lr=lr)
    
    # Learning rate scheduler (reduce on plateau) - applied to encoder, decoder follows
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        enc_opt, mode='min', factor=0.5, patience=3, verbose=True, min_lr=1e-6
    )
    
    criterion = torch.nn.CrossEntropyLoss()

    # Get train and validation loaders
    train_loader = get_feature_dataloader(df, "train", config.BATCH_SIZE)
    val_loader = get_feature_dataloader(df, "val", config.BATCH_SIZE)

    print(f"[INFO] Training batches: {len(train_loader)}")
    print(f"[INFO] Validation batches: {len(val_loader)}")

    total_batches = len(train_loader)
    iter_count = 1
    
    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0

    for epoch in range(epochs):

        print(f"\n=== Epoch {epoch+1}/{epochs} ===")
        epoch_start = time.time()
        epoch_train_losses = []

        for i, (x, y, fname) in enumerate(train_loader, start=1):

            # Timer: Data loading time
            data_load_start = time.time()
            x = x.to(config.device)                  # (B, T, H)
            y = y.to(config.device)                  # labels
            torch.cuda.synchronize()
            data_load_time = time.time() - data_load_start

            # Timer: Training time
            train_start = time.time()
            try:
                loss = train_step(
                    x, y,
                    encoder, decoder,
                    enc_opt, dec_opt,
                    criterion, encodings,
                    teacher_forcing_ratio=0.5,  # Start with 50% teacher forcing
                    epoch=epoch,
                    total_epochs=epochs
                )
                epoch_train_losses.append(loss)
            except Exception as e:
                print(f"⚠️ Error with sample {fname}: {e}")
                continue
            torch.cuda.synchronize()
            train_time = time.time() - train_start

            # Logging
            if i % 10 == 0 or i == total_batches:  # Print every 10 batches
                print(f"[{i}/{total_batches}] "
                      f"Loss: {loss:.4f} | "
                      f"DataLoad: {data_load_time:.3f}s | "
                      f"Train: {train_time:.3f}s")

            # Save every N iterations
            if iter_count % print_every == 0:
                enc_path = f"{config.drive_folder}/jamal/encoder.model"
                dec_path = f"{config.drive_folder}/jamal/decoder.model"
                torch.save(encoder.state_dict(), enc_path)
                torch.save(decoder.state_dict(), dec_path)
                print(f"💾 Saved models to:\n  {enc_path}\n  {dec_path}")

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            iter_count += 1

        # Calculate average training loss
        avg_train_loss = np.mean(epoch_train_losses) if epoch_train_losses else float('inf')
        
        # Validation
        print(f"\n🔍 Running validation...")
        val_loss = validate(encoder, decoder, val_loader, criterion, encodings, config.device)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        # Sync decoder learning rate with encoder
        for param_group in dec_opt.param_groups:
            param_group['lr'] = enc_opt.param_groups[0]['lr']
        current_lr = enc_opt.param_groups[0]['lr']
        
        print(f"📊 Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}")
        
        # Early stopping check
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            
            # Save best model
            enc_path = f"{config.drive_folder}/jamal/encoder_best.model"
            dec_path = f"{config.drive_folder}/jamal/decoder_best.model"
            torch.save(encoder.state_dict(), enc_path)
            torch.save(decoder.state_dict(), dec_path)
            print(f"✅ New best model! Val Loss: {val_loss:.4f} | Saved to:\n  {enc_path}\n  {dec_path}")
        else:
            patience_counter += 1
            print(f"⏳ No improvement. Patience: {patience_counter}/{patience}")
            
            if patience_counter >= patience:
                print(f"\n🛑 Early stopping triggered! Best epoch: {best_epoch}, Best Val Loss: {best_val_loss:.4f}")
                print(f"💾 Loading best model from epoch {best_epoch}...")
                # Load best model
                encoder.load_state_dict(torch.load(f"{config.drive_folder}/jamal/encoder_best.model"))
                decoder.load_state_dict(torch.load(f"{config.drive_folder}/jamal/decoder_best.model"))
                # Also save as main model
                torch.save(encoder.state_dict(), f"{config.drive_folder}/jamal/encoder.model")
                torch.save(decoder.state_dict(), f"{config.drive_folder}/jamal/decoder.model")
                break

        print(f"⏱️  Epoch time: {time.time() - epoch_start:.1f}s")
    
    print(f"\n✅ Training completed! Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")
