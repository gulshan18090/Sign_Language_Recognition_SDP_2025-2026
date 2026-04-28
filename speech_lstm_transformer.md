# Speech — LSTM and Transformer Pipelines

> 🖥️ **SHOW** = open file on screen · ▶️ **RUN** = run command live · 🎞️ **PLAY** = play video

---

## 3. Development and Technical Work (≈4 min)

Good morning. I'll walk you through our two neural pipelines — the LSTM and the Transformer — for Azerbaijani Sign Language recognition.

### Data

> 🖥️ **SHOW** [lstm/drive/sentences_all.csv](lstm/drive/sentences_all.csv)

Both pipelines use the same dataset: about **750 videos** covering **123 phrases**. Each phrase has one reference video from a translator and a few videos from learners. Labels are simple word sequences, like "axşam xeyir" meaning "good evening."

### Preprocessing

The two pipelines handle video very differently.

**LSTM — two stages.** First, we extract features *once* and save them. Each video goes through MediaPipe to find hand frames, gets cropped around the hand, resized to 224×224, and passed through a pretrained SqueezeNet. A small BiLSTM then compresses each frame into a **512-dimensional vector**. We save the whole sequence as a `.pt` file.

> 🖥️ **SHOW** [lstm/feature_extraction_slr/extract_features.py](lstm/feature_extraction_slr/extract_features.py)
> ▶️ **RUN** `ls lstm/features_slr/ | head -5`

We do this heavy work only once. Training then loads the cached features and runs fast.

**Transformer — no preprocessing tricks.** We just sample 16 frames per video, resize to 224×224, and normalize. The Vision Transformer learns features directly from raw RGB frames.

### Feature Engineering — and Augmentation

For the LSTM, the feature pipeline *is* the engineering: SqueezeNet + BiLSTM before training even starts.

For the Transformer, we rely on **heavy augmentation** to fight overfitting on a small dataset.

> 🖥️ **SHOW** [Transformer/data/augmentation.py](Transformer/data/augmentation.py)

Three layers:

1. **Spatial** (same transform on every frame in a clip): random crop, horizontal flip, color jitter, grayscale, Gaussian blur.
2. **Temporal** (on the frame sequence): frame dropout (10%), temporal reverse (10%), sampling jitter.
3. **Regularization**: CutOut (zero a random patch, 30%), label smoothing (0.1).

> 🎞️ **PLAY** [Transformer/aug_preview.mp4](Transformer/aug_preview.mp4) — original vs augmented, side by side.
> 🖥️ **SHOW** [Transformer/aug_preview.png](Transformer/aug_preview.png) — all transforms in one grid.

### Models

**LSTM:** a bidirectional LSTM encoder plus an attention-based LSTM decoder. Vocabulary is ~500 word tokens. Inference uses beam search.

> 🖥️ **SHOW** [lstm/models/decoder.py](lstm/models/decoder.py)

**Transformer:** ViT-Small for each frame (384-dim output), a 4-layer temporal Transformer encoder, and a 4-layer Transformer decoder.

> 🖥️ **SHOW** [Transformer/model/seq2seq.py](Transformer/model/seq2seq.py)

### Training

Both train for 200 epochs with early stopping. The Transformer uses mixed precision, two learning rates (smaller for the ViT backbone), warm-up plus cosine decay, and freezes the ViT for the first 5 epochs.

> 🖥️ **SHOW** [Transformer/config/config.py](Transformer/config/config.py)

We evaluate both with Top-1, Top-3, Top-5 accuracy, and Word Error Rate.

---

## 4. Results and Achievements (≈2–3 min)

### LSTM Results

> 🖥️ **SHOW** [results.md](results.md) — LSTM section

| Metric | Value |
|---|---|
| Top-1 | **16.0 %** |
| Top-3 | **25.6 %** |
| Top-5 | **26.5 %** |
| Inference | **~18 ms / video (GPU)** |

Training loss fell from 3.7 to 0.3, but validation loss stayed around 1.8 — that's **overfitting**. Short, common phrases work well: "axşam xeyir" 81%, "siz ad nə?" 78%. Rare phrases fail. The model learns; it just lacks examples.

> ▶️ **RUN** (optional) `python lstm/inference/lstm_attention_inference.py` on one clip — show the ~18 ms speed.

### Transformer Results

> 🖥️ **SHOW** [Transformer/artifacts/checkpoints/](Transformer/artifacts/checkpoints/)

Training went smoothly — best checkpoint at **epoch 27, val loss 6.32**. On a 30-sample random test, exact-match accuracy was **6.7%**, mean WER 0.86. Honest caveat: this is a partial evaluation; we didn't run the full test set.

> 🖥️ **SHOW** [Transformer/random_eval_results_recent30.csv](Transformer/random_eval_results_recent30.csv) — a few rows of reference, prediction, WER.

### Comparison Table

| Method | Top-1 | Top-3 | Top-5 | Latency |
|---|---|---|---|---|
| Cosine (baseline) | 30.7 % | 45.0 % | 49.4 % | ~2.7 s CPU |
| **DTW (best)** | **40.0 %** | **50.2 %** | **56.3 %** | ~7.8 s CPU |
| LSTM | 16.0 % | 25.6 % | 26.5 % | **~18 ms GPU** |
| Transformer | 6.7 %* | — | — | ~100–500 ms GPU |

The big finding: on only 750 samples, **classical DTW beats both neural methods by a wide margin**.

### What We Still Achieved

- Both pipelines fully work and are reproducible.
- LSTM gives the **fastest inference in the whole system** — 18 ms per video.
- The Transformer checkpoint and training setup are ready to scale once we have more data.

### Takeaway

**Small data beats small models.** With only 750 samples, changing the *representation* (hand-engineered geometric features + DTW) wins over scaling up the *model* (LSTM, Transformer). The neural infrastructure is ready — data is the bottleneck, not architecture.

Thank you.

---

## Pre-Talk Checklist

**Open in tabs:**
1. [lstm/drive/sentences_all.csv](lstm/drive/sentences_all.csv)
2. [lstm/feature_extraction_slr/extract_features.py](lstm/feature_extraction_slr/extract_features.py)
3. [Transformer/data/augmentation.py](Transformer/data/augmentation.py)
4. [Transformer/aug_preview.png](Transformer/aug_preview.png)
5. [lstm/models/decoder.py](lstm/models/decoder.py)
6. [Transformer/model/seq2seq.py](Transformer/model/seq2seq.py)
7. [Transformer/config/config.py](Transformer/config/config.py)
8. [results.md](results.md)
9. [Transformer/random_eval_results_recent30.csv](Transformer/random_eval_results_recent30.csv)

**Pre-load video:** [Transformer/aug_preview.mp4](Transformer/aug_preview.mp4) (loop, muted)

**Dry-run these commands beforehand:**
- `ls lstm/features_slr/ | head -5`
- `ls Transformer/artifacts/checkpoints/`
- `python lstm/inference/lstm_attention_inference.py <clip>` (optional)
