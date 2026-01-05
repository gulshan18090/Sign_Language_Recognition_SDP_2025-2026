# Low Accuracy Potential Reasons - Version 1

**Date:** January 5, 2026  
**Analysis by:** GitHub Copilot  
**Model:** CNN-LSTM with Attention for Azerbaijani Sign Language Recognition

---

## Executive Summary

The model produces seemingly random predictions that don't match expected outputs. Investigation revealed the model has **not learned to map video features to text** - instead, it generates grammatically plausible sentences by memorizing word patterns while ignoring video input entirely.

---

## Key Findings

### 1. Attention Mechanism Collapse

**Observation:**
```
Video: 2022-04-26 12-58-07
  Expected: Mən Bakı yaşamaq
  Attention max: 1.0000
  Attention argmax: frame 26

Video: 2022-04-21 17-27-53
  Expected: Mən ana iş yox
  Attention max: 1.0000
  Attention argmax: frame 26  (SAME FRAME!)
```

**Problem:** Attention puts ~100% weight on frame 26, ignoring 63 out of 64 frames for ALL videos.

**Impact:** Model cannot distinguish between different sign sequences.

---

### 2. Model Ignores Video Features

**Test:** Fed random noise vs real video features to the model.

| Input | First Word | Confidence |
|-------|-----------|------------|
| Real video features | "mən" | 99.88% |
| Random noise | "mən" | 99.99% |

**Conclusion:** Model produces nearly identical outputs regardless of input - it has not learned the video→text mapping.

---

### 3. Training Data Imbalance

#### First Word Distribution
| Word | Count | Percentage |
|------|-------|------------|
| mən | 188 | 47.0% |
| mənim | 41 | 10.2% |
| siz | 33 | 8.2% |
| bu | 20 | 5.0% |
| o | 18 | 4.5% |
| others | 100 | 25.1% |

**Problem:** 47% of training sentences start with "mən" → model always predicts "mən" first with 99%+ confidence.

#### Class Distribution
| Metric | Value |
|--------|-------|
| Total samples | 3,440 |
| Unique sentences (classes) | 312 |
| Avg samples per class | 11 |
| Classes with ≤2 samples | 160 (51%) |
| Max samples per class | 44 |
| Min samples per class | 1 |

**Problem:** 51% of classes have only 1-2 training samples - impossible to learn these patterns reliably.

---

### 4. Feature Extraction Inconsistencies

#### Frame Count Variation
| Metric | Value |
|--------|-------|
| Min frames | 1 |
| Max frames | 562 |
| Mean | 58.4 |
| Std deviation | 53.5 |
| Target (max_frames) | 64 |

#### Padding/Truncation Issues
| Scenario | Count | Problem |
|----------|-------|---------|
| Videos > 64 frames (truncated) | 2,578 (33%) | Loses late signing information |
| Videos < 64 frames (padded) | 4,627 (59%) | Zero padding dilutes signal |
| Videos < 10 frames | 1,285 (16%) | Mostly padding, little signal |
| Videos > 100 frames | 1,262 (16%) | Loses >36% of content |

**Problem:** Truncation takes only first 64 frames, potentially missing important signs. A video with 562 frames loses 88% of its content.

---

### 5. Teacher Forcing Trap

**During Training:**
- Decoder receives **correct previous word** as input
- Model learns: "what word typically follows this word?"
- Model does NOT learn: "what word matches this video frame?"

**Result:** Model becomes a **language model** instead of a **video-to-text translator**.

---

## Root Cause Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     LOW ACCURACY                            │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│   ATTENTION   │    │    TEACHER    │    │     DATA      │
│   COLLAPSE    │    │   FORCING     │    │  IMBALANCE    │
└───────────────┘    └───────────────┘    └───────────────┘
        │                     │                     │
        ▼                     ▼                     ▼
  Model looks at        Model learns         47% start with
  only 1 frame          word patterns,       "mən", 51% of
  out of 64             not video→text       classes have ≤2
                        mapping              samples
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              ▼
                 ┌─────────────────────────┐
                 │  Model ignores video    │
                 │  features completely    │
                 └─────────────────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │  Outputs "grammatically │
                 │  plausible" but random  │
                 │  sentences              │
                 └─────────────────────────┘
```

---

## Recommended Fixes

### Priority 1: Fix Training Method (High Impact)

| Fix | Description | Effort |
|-----|-------------|--------|
| **Scheduled Sampling** | Gradually replace teacher forcing with model's own predictions | Medium |
| **Curriculum Learning** | Start with easy examples, increase difficulty | Medium |

### Priority 2: Fix Attention (High Impact)

| Fix | Description | Effort |
|-----|-------------|--------|
| **Multi-Head Attention** | Multiple attention heads prevent collapse | Medium |
| **Attention Dropout** | Regularize attention weights | Low |
| **Temperature Scaling** | Soften attention distribution | Low |

### Priority 3: Fix Data Issues (Medium Impact)

| Fix | Description | Effort |
|-----|-------------|--------|
| **Uniform Frame Sampling** | Sample evenly across video instead of truncating | Low |
| **Class Balancing** | Oversample rare classes, undersample common | Low |
| **Data Augmentation** | Augment underrepresented sentences | Medium |
| **Filter Short Videos** | Remove videos with <5 frames | Low |

### Priority 4: Architecture Changes (Optional)

| Fix | Description | Effort |
|-----|-------------|--------|
| **Auxiliary Encoder Loss** | Force encoder to learn meaningful features | High |
| **Transformer Architecture** | Replace LSTM with Transformer | High |
| **CTC Loss** | Alternative loss function for sequence tasks | High |

---

## Validation Tests

After implementing fixes, verify:

1. **Attention Distribution:** Should be spread across multiple frames, not collapsed to one
2. **Random Noise Test:** Random input should produce different output than real features
3. **Feature Sensitivity:** Small changes in video should produce small changes in output
4. **Class Accuracy:** Check accuracy per class, not just overall

---

## Files Analyzed

| File | Purpose |
|------|---------|
| `models/decoder.py` | Attention mechanism |
| `models/encoder.py` | Feature encoding |
| `models/trainer.py` | Training loop with teacher forcing |
| `data/dataloader.py` | Padding/truncation logic |
| `data/vocab.py` | Vocabulary building |
| `extract_features.py` | Feature extraction |
| `video/mp_hands.py` | Hand detection filtering |

---

## Next Steps

1. Implement scheduled sampling in `models/trainer.py`
2. Add attention dropout in `models/decoder.py`
3. Change truncation to uniform sampling in `data/dataloader.py`
4. Retrain model and re-evaluate

---

*Document Version: 1.0*
