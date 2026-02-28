"""
Training Pipeline Validation
Checks if model training is working correctly
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import torch
import numpy as np
from config import Config as config
from data.vocab import build_word_dict
from data.dataset_builder import load_sentences, build_video_table
from data.dataloader import get_feature_dataloader
from models.encoder import EncoderRNN
from models.decoder import AttnDecoderRNN
from utils.seed import seed_everything

def test_dataloader():
    """Test if dataloader works correctly"""
    print("\n" + "=" * 70)
    print("1. DATALOADER VALIDATION")
    print("=" * 70)
    
    # Load data
    df_sent = load_sentences(config.train_csv_path, limit=50)
    encodings, word_idx = build_word_dict(df_sent["sign_language"])
    df = build_video_table(df_sent, encodings)
    
    # Filter to available features
    features_dir = "features"
    available_features = set(f[:-3] for f in os.listdir(features_dir) if f.endswith(".pt"))
    df = df[df["video_file"].apply(lambda x: os.path.splitext(os.path.basename(x))[0] in available_features)].reset_index(drop=True)
    
    print(f"✓ Dataset size: {len(df)} samples")
    print(f"✓ Vocabulary size: {len(encodings)}")
    
    # Create dataloader
    loader = get_feature_dataloader(df, "train", config.BATCH_SIZE)
    
    print(f"✓ Batches: {len(loader)}")
    
    # Get one batch
    for batch_idx, (x, y, fname) in enumerate(loader):
        print(f"\n📦 Batch {batch_idx + 1}:")
        print(f"   Input shape: {x.shape} (should be [B={config.BATCH_SIZE}, T=64, H=256])")
        print(f"   Label shape: {y.shape}")
        print(f"   Filenames: {len(fname)}")
        
        # Check if features are diverse
        print(f"\n   Feature statistics:")
        print(f"     Mean: {x.mean().item():.4f}")
        print(f"     Std:  {x.std().item():.4f}")
        print(f"     Min:  {x.min().item():.4f}")
        print(f"     Max:  {x.max().item():.4f}")
        
        # Check if samples are different
        if x.shape[0] > 1:
            sim = torch.nn.functional.cosine_similarity(x[0].flatten(), x[1].flatten(), dim=0)
            print(f"     Similarity between sample 0 and 1: {sim.item():.4f}")
            print(f"     → Should be LOW (< 0.5) for diverse samples")
        
        if batch_idx >= 2:  # Check 3 batches
            break
    
    return df, encodings

def test_model_forward():
    """Test if models forward pass works correctly"""
    print("\n" + "=" * 70)
    print("2. MODEL FORWARD PASS VALIDATION")
    print("=" * 70)
    
    df, encodings = test_dataloader()
    
    # Initialize models
    input_size = 256
    hidden_size = 128
    
    print(f"\n🤖 Initializing models...")
    print(f"   Encoder: input_size={input_size}, hidden_size={hidden_size}, bidirectional=True")
    print(f"   Decoder: hidden_size={hidden_size*2}, output_size={len(encodings)}")
    
    encoder = EncoderRNN(input_size, hidden_size, device=config.device, biDirectional=True).to(config.device)
    decoder = AttnDecoderRNN(hidden_size * 2, len(encodings), device=config.device, encoder_hidden_size=hidden_size).to(config.device)
    
    # Get one batch
    loader = get_feature_dataloader(df, "train", config.BATCH_SIZE)
    x, y, fname = next(iter(loader))
    x = x.to(config.device)
    y = y.to(config.device)
    
    print(f"\n🔄 Testing forward pass...")
    
    # Encoder forward
    B = x.size(0)
    hidden = encoder.initHidden(B)
    encoder_output, encoder_hidden = encoder(x, hidden)
    
    print(f"   ✓ Encoder output: {encoder_output.shape}")
    print(f"   ✓ Encoder hidden: {encoder_hidden[0].shape}, {encoder_hidden[1].shape}")
    
    # Merge for decoder
    if encoder.D == 2:
        h = torch.cat((encoder_hidden[0][0:1], encoder_hidden[0][1:2]), dim=2)
        c = torch.cat((encoder_hidden[1][0:1], encoder_hidden[1][1:2]), dim=2)
        decoder_hidden = (h, c)
    else:
        decoder_hidden = encoder_hidden
    
    print(f"   ✓ Decoder hidden: {decoder_hidden[0].shape}, {decoder_hidden[1].shape}")
    
    # Decoder forward
    max_len = y.shape[1]
    decoder_input = y[:, 0:1, :]
    
    output, decoder_hidden, attn_weights = decoder(decoder_input, decoder_hidden, encoder_output)
    
    print(f"   ✓ Decoder output: {output.shape} (should be [B={B}, {len(encodings)}])")
    print(f"   ✓ Attention weights: {attn_weights.shape} (should be [B={B}, 64])")
    
    # Check if output is diverse
    print(f"\n   Output statistics:")
    print(f"     Mean: {output.mean().item():.4f}")
    print(f"     Std:  {output.std().item():.4f}")
    print(f"     Min:  {output.min().item():.4f}")
    print(f"     Max:  {output.max().item():.4f}")
    
    # Check predicted tokens
    predicted_tokens = output.argmax(dim=1)
    print(f"\n   Predicted tokens for batch: {predicted_tokens.tolist()}")
    print(f"     Unique predictions: {len(torch.unique(predicted_tokens))}/{B}")
    
    if len(torch.unique(predicted_tokens)) == 1:
        print(f"     ⚠️  WARNING: All predictions are the same!")
        print(f"     → Model may not be learning properly")
    else:
        print(f"     ✓ Good: Predictions are diverse")
    
    return encoder, decoder, encodings

def test_loaded_model():
    """Test if loaded model predictions are diverse"""
    print("\n" + "=" * 70)
    print("3. LOADED MODEL VALIDATION")
    print("=" * 70)
    
    df_sent = load_sentences(config.train_csv_path, limit=50)
    encodings, word_idx = build_word_dict(df_sent["sign_language"])
    
    # Check if model exists
    encoder_path = f"{config.drive_folder}/jamal/encoder.model"
    decoder_path = f"{config.drive_folder}/jamal/decoder.model"
    
    if not os.path.exists(encoder_path):
        print(f"❌ Encoder model not found: {encoder_path}")
        return
    
    print(f"✓ Loading models from:\n  {encoder_path}\n  {decoder_path}")
    
    # Initialize and load
    input_size = 256
    hidden_size = 128
    
    encoder = EncoderRNN(input_size, hidden_size, device=config.device, biDirectional=True).to(config.device)
    decoder = AttnDecoderRNN(hidden_size * 2, len(encodings), device=config.device, encoder_hidden_size=hidden_size).to(config.device)
    
    try:
        encoder.load_state_dict(torch.load(encoder_path, map_location=config.device))
        decoder.load_state_dict(torch.load(decoder_path, map_location=config.device))
        print(f"✓ Models loaded successfully")
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return
    
    encoder.eval()
    decoder.eval()
    
    # Test on multiple random features
    import random
    feature_files = [f for f in os.listdir("features") if f.endswith(".pt")]
    test_files = random.sample(feature_files, min(10, len(feature_files)))
    
    print(f"\n🔍 Testing predictions on {len(test_files)} random features...")
    
    predictions = []
    for feat_file in test_files:
        features = torch.load(f"features/{feat_file}", weights_only=True).to(config.device)
        
        # Pad/truncate to 64 frames
        T = features.shape[0]
        if T < 64:
            pad = torch.zeros((64 - T, features.shape[1]), device=config.device)
            features = torch.cat([features, pad], dim=0)
        else:
            features = features[:64]
        
        features = features.unsqueeze(0)  # (1, 64, 256)
        
        with torch.no_grad():
            hidden = encoder.initHidden(1)
            encoder_output, encoder_hidden = encoder(features, hidden)
            
            if encoder.D == 2:
                h = torch.cat((encoder_hidden[0][0:1], encoder_hidden[0][1:2]), dim=2)
                c = torch.cat((encoder_hidden[1][0:1], encoder_hidden[1][1:2]), dim=2)
                decoder_hidden = (h, c)
            else:
                decoder_hidden = encoder_hidden
            
            # Decode first token
            sos_token = encodings.get('SOS', 0)
            decoder_input = torch.tensor([[sos_token]], device=config.device)
            
            tokens = []
            for _ in range(10):  # Decode up to 10 tokens
                output, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_output)
                token_id = output.argmax(dim=1).item()
                
                if token_id == encodings.get('EOS', 1):
                    break
                
                tokens.append(token_id)
                decoder_input = torch.tensor([[token_id]], device=config.device)
            
            predictions.append(tokens)
            
            # Convert to words
            words = [word_idx.get(t, '<UNK>') for t in tokens]
            print(f"  {feat_file[:30]:30s} → {' '.join(words[:5])}")
    
    # Check diversity
    print(f"\n📊 Prediction diversity:")
    unique_predictions = len(set([tuple(p) for p in predictions]))
    print(f"   Unique predictions: {unique_predictions}/{len(predictions)}")
    
    if unique_predictions == 1:
        print(f"   ❌ CRITICAL: All predictions are IDENTICAL!")
        print(f"   → Model is not learning - produces same output for all inputs")
        print(f"   → Recommendations:")
        print(f"      1. Increase model capacity (hidden_size)")
        print(f"      2. Train for more epochs")
        print(f"      3. Check if loss is decreasing during training")
        print(f"      4. Verify features are diverse (done above)")
    elif unique_predictions < len(predictions) * 0.5:
        print(f"   ⚠️  WARNING: Low prediction diversity")
        print(f"   → Model may be underfitting")
    else:
        print(f"   ✓ Good: Predictions are diverse")

def main():
    seed_everything(config.seed)
    
    print("\n" + "=" * 70)
    print("TRAINING PIPELINE VALIDATION")
    print("=" * 70)
    
    # Run tests
    encoder, decoder, encodings = test_model_forward()
    test_loaded_model()
    
    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()

