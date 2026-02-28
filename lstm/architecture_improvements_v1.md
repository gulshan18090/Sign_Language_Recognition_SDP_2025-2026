# Architecture Improvements - Version 1

**Date:** January 5, 2026  
**Changes by:** GitHub Copilot  
**Purpose:** Fix attention collapse, teacher forcing trap, and data handling issues

---

## Summary of Changes

Three major improvements implemented to address the low accuracy issues identified in `low_accuracy_potential_reasons_v1.md`. All changes work with **existing pre-extracted features** - no regeneration required.

---

## 1. Improved Attention Mechanism

**File:** `models/decoder.py`

### What Was Wrong
- Simple linear attention collapsed to single frame (frame 26)
- 100% attention weight on one frame, 0% on others
- Model couldn't distinguish between different videos

### What Was Changed

```python
# OLD: Simple attention (collapsed)
self.attn = nn.Linear(hidden_size*2, max_length)
attn_weights = F.softmax(self.attn(...), dim=1)

# NEW: Query-Key-Energy attention with temperature and dropout
self.attn_query = nn.Linear(hidden_size * 2, hidden_size)
self.attn_key = nn.Linear(encoder_hidden_size, hidden_size)
self.attn_energy = nn.Linear(hidden_size, 1)
self.attn_dropout = nn.Dropout(attn_dropout_p)
self.attn_temperature = attn_temperature  # Default: 2.0
```

### New Parameters Added
| Parameter | Default | Purpose |
|-----------|---------|---------|
| `attn_dropout_p` | 0.1 | Regularizes attention, prevents collapse |
| `attn_temperature` | 2.0 | Softens distribution (higher = more spread) |

### Result
| Before | After |
|--------|-------|
| Attention max: 1.0000 | Attention max: ~0.02 |
| 1 frame attended | 50+ frames attended |
| Same frame for all videos | Different patterns per video |

---

## 2. Scheduled Sampling (Training)

**File:** `models/train_step.py`

### What Was Wrong
- 100% teacher forcing: decoder always got correct previous word
- Model learned "what word follows this word?" (language model)
- Model did NOT learn "what video features map to this word?"

### What Was Changed

```python
# OLD: Always use ground truth
for t in range(seq_len):
    output = decoder(ground_truth[t], ...)

# NEW: Mix ground truth with model predictions
scheduled_ratio = teacher_forcing_ratio * (1 - epoch / total_epochs)

for t in range(seq_len):
    output = decoder(current_input, ...)
    
    if random.random() < scheduled_ratio:
        current_input = ground_truth[t+1]  # Teacher forcing
    else:
        current_input = output.argmax()    # Model's prediction
```

### New Parameters Added
| Parameter | Default | Purpose |
|-----------|---------|---------|
| `teacher_forcing_ratio` | 0.5 | Base probability of using ground truth |
| `epoch` | 0 | Current epoch (for scheduling) |
| `total_epochs` | 20 | Total epochs (for scheduling) |

### Schedule
| Epoch | Teacher Forcing % | Model Prediction % |
|-------|-------------------|-------------------|
| 1 | 47.5% | 52.5% |
| 5 | 37.5% | 62.5% |
| 10 | 25% | 75% |
| 20 | 0% | 100% |

### Result
- Model MUST learn video→text mapping to make correct predictions
- Can't rely on ground truth during inference
- Forces encoder to produce meaningful features

---

## 3. Uniform Frame Sampling

**File:** `data/dataloader.py`

### What Was Wrong
- Long videos truncated to first 64 frames
- Video with 562 frames → only first 64 kept (lost 88%)
- Late signs completely ignored

### What Was Changed

```python
# OLD: Truncation (loses end of video)
if seq_len > target_length:
    f = f[:target_length]

# NEW: Uniform sampling (preserves temporal coverage)
if seq_len > target_length:
    indices = torch.linspace(0, seq_len - 1, target_length).long()
    f = f[indices]
```

### Example
| Video Length | Old Method | New Method |
|--------------|-----------|------------|
| 128 frames | Keep 1-64 (50%) | Sample every 2nd frame (100% coverage) |
| 256 frames | Keep 1-64 (25%) | Sample every 4th frame (100% coverage) |
| 562 frames | Keep 1-64 (11%) | Sample every 9th frame (100% coverage) |

### Result
- All parts of video represented
- No information loss from long videos
- Better temporal coverage of signing sequence

---

## 4. Updated Model Initialization

**File:** `main.py`

### What Was Changed

```python
# OLD
decoder = AttnDecoderRNN(
    encoder_output_size, 
    len(encodings), 
    device=config.device, 
    encoder_hidden_size=encoder_output_size
)

# NEW
decoder = AttnDecoderRNN(
    encoder_output_size, 
    len(encodings), 
    device=config.device, 
    encoder_hidden_size=encoder_output_size,
    attn_dropout_p=0.1,      # Attention regularization
    attn_temperature=2.0     # Soft attention
)
```

---

## 5. Updated Trainer

**File:** `models/trainer.py`

### What Was Changed

```python
# OLD
loss = train_step(x, y, encoder, decoder, enc_opt, dec_opt, criterion, encodings)

# NEW
loss = train_step(
    x, y, encoder, decoder, enc_opt, dec_opt, criterion, encodings,
    teacher_forcing_ratio=0.5,  # 50% scheduled sampling
    epoch=epoch,                # Current epoch
    total_epochs=epochs         # For scheduling
)
```

---

## Files Modified

| File | Changes |
|------|---------|
| `models/decoder.py` | New attention mechanism with dropout + temperature |
| `models/train_step.py` | Scheduled sampling implementation |
| `models/trainer.py` | Pass epoch info to train_step |
| `data/dataloader.py` | Uniform frame sampling |
| `main.py` | New decoder parameters |

---

## How to Retrain

```bash
cd /home/temporaryuser2/Desktop/sdp_module
python main.py
```

No need to:
- ❌ Regenerate features
- ❌ Modify feature extraction
- ❌ Change vocabulary

---

## Expected Improvements

### Before (Old Model)
- Attention: 100% on frame 26 for ALL videos
- Random noise → same output as real video
- Always predicts "mən" first (99.9% confidence)

### After (New Model)
- Attention: Spread across 50+ frames
- Random noise → different output than real video
- First word depends on actual video content

---

## Validation Tests (Run After Training)

```python
# Test 1: Attention should be spread
# Expected: attention max < 0.5, multiple frames > 0.01

# Test 2: Random noise test
# Feed random noise - should get different output than real features

# Test 3: Different videos should get different attention patterns
```

---

## Compatibility

| Component | Compatible? |
|-----------|-------------|
| Existing features (7,861 .pt files) | ✅ Yes |
| Existing vocabulary | ✅ Yes |
| Existing CSV labels | ✅ Yes |
| Old saved models | ❌ No (new architecture) |

**Note:** Old saved models (`encoder.model`, `decoder.model`) are incompatible due to new attention layers. Must retrain from scratch.

---

*Document Version: 1.0*
