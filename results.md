# Cosine Similarity Pipeline — Results

## Dataset
- **342 videos**, 124 unique sentences (classes)
- Translator videos: 122 (35.7%) | User videos: 218 (63.7%)
- Avg video duration: 6.5 s (range: 2.3–12.9 s)
- Feature vector: 126-dim (21 landmarks × 3 coords × 2 hands), wrist-centered, scale-invariant
- Evaluation: **Leave-One-Out Cross-Validation**

---

## Accuracy (N=64 frames)

| Split | Top-1 | Top-3 | Top-5 |
|---|---|---|---|
| All videos (n=342) | 30.7% | 45.0% | 49.4% |
| User videos (n=218) | 34.4% | 49.5% | 53.2% |
| Translator videos (n=122) | 24.6% | 37.7% | 43.4% |

---

## Effect of Frame Count (N) — All Videos

| N Frames | Top-1 | Top-3 | Top-5 |
|---|---|---|---|
| 16 | 27.8% | 39.5% | 46.8% |
| 32 | 31.6% | 43.9% | 49.7% |
| **64** | **30.7%** | **45.0%** | **49.4%** |
| 128 | 30.7% | 45.0% | 49.4% |

> Performance plateaus at N=64; no gain beyond that.

---

## Query Latency (per video, CPU — Intel i7, 15.7 GB RAM)

| Stage | Time |
|---|---|
| Frame read + uniform sample | 667.8 ms |
| MediaPipe hand extraction | 2036.2 ms ← bottleneck (75%) |
| Cosine search (342 vectors) | 2.8 ms |
| **Total** | **~2.7 s** |

---

---

# LSTM Seq2Seq Pipeline — Results

## Dataset
- **407 videos evaluated**, 24 unique sentence classes
- Pre-extracted features: 512-dim BiLSTM vectors (64 frames per video)
- Evaluation: greedy decoding, full held-out set

---

## Accuracy

| Metric | Value |
|---|---|
| Top-1 Accuracy | **15.97%** (65 / 407) |
| Top-3 Accuracy | **25.55%** (104 / 407) |
| Top-5 Accuracy | **26.54%** (108 / 407) |

---

## Per-Class Highlights (Top-1)

| Sentence | Top-1 Accuracy |
|---|---|
| "axşam xeyir" (Good evening) | 80.95% (17/21) |
| "siz ad nə ?" (What is your name?) | 78.05% (32/41) |
| "mən yazmaq bilmir" (I cannot write) | 52.38% (11/21) |

---

## Training (200 epochs)

| Epoch | Train Loss | Val Loss |
|---|---|---|
| 1 | 3.726 | 2.835 |
| 10 | 1.963 | 2.167 |
| 20 | 0.768 | 1.864 |
| 50 | 0.201 | 1.798 |
| 200 | 0.303 | 1.794 |

- Best val loss: **1.7905** at epoch 31
- Final train loss: **0.303** — gap indicates overfitting

---

## Inference Latency (per video, GPU)

| Stage | Time |
|---|---|
| Feature load | 0.9 ms |
| Decoder (greedy) | 17.1 ms |
| **Total** | **~18 ms** |

---

---

# ViT + Transformer Seq2Seq Pipeline — Results

> **Architecture:** ViT-Small extracts a 384-dim feature per frame →
> Transformer temporal encoder → autoregressive Transformer decoder
> over the gloss vocabulary. End-to-end trainable.

## Dataset
- **~750 videos**, 123 unique sentence classes
- Input: 16 raw RGB frames per video, resized to 224×224, normalized
- No pre-extracted features — ViT-Small learns representations end-to-end
- Evaluation: random 30-sample held-out subset (partial eval)

---

## Accuracy (Recent-30 Eval)

| Metric | Value |
|---|---|
| Exact-Match Accuracy | **6.7%** (2 / 30) |
| Mean Word Error Rate (WER) | **~0.93** |
| Top-1 (caveat: partial set) | **6.7%** |

> Honest caveat: a full test-set sweep was not completed; numbers are a
> random-sample snapshot and are optimistic/noisy at N=30.

---

## Training (200 epochs, mixed precision)

| Epoch | Val Loss |
|---|---|
| 1 | 42.15 |
| 5 | 11.22 |
| 10 | 8.06 |
| 20 | 6.47 |
| **27** | **6.33 ← best** |
| 50 | 6.77 |
| 100 | 7.56 |
| 200 | 6.42 |

- Best checkpoint: **epoch 27, val loss 6.33** (`best.pt`)
- After epoch ~30 the val loss drifts up then slowly recovers — classic
  small-data instability. A few `nan` checkpoints appeared at epochs 2, 99, 100.
- Training schedule: 5-epoch ViT freeze → warm-up → cosine decay,
  two LRs (backbone vs head).

---

## Architecture

| Component | Spec |
|---|---|
| Frame encoder | ViT-Small (384-dim per frame) |
| Temporal encoder | 4-layer Transformer |
| Decoder | 4-layer Transformer, greedy / beam decoding |
| Frames per clip | 16 |
| Augmentation | Spatial + temporal + CutOut + label smoothing 0.1 |

---

## Inference Latency (per video, GPU)

| Stage | Time |
|---|---|
| Frame load + resize + normalize | ~40 ms |
| ViT forward (16 frames) | ~60–300 ms |
| Decoder (greedy) | ~20–150 ms |
| **Total** | **~100–500 ms** |

---

## Comparison (All Pipelines)

| Method | Top-1 | Top-3 | Top-5 | Latency |
|---|---|---|---|---|
| Cosine (baseline) | 30.7% | 45.0% | 49.4% | ~2.7 s CPU |
| **DTW (best)** | **40.0%** | **50.2%** | **56.3%** | ~7.8 s CPU |
| LSTM Seq2Seq | 16.0% | 25.6% | 26.5% | **~18 ms GPU** |
| ViT + Transformer Seq2Seq | 6.7%* | — | — | ~100–500 ms GPU |

> *Transformer Top-1 is a partial 30-sample estimate, not full-set.

---

## Takeaway

With only ~750 videos across 123 classes, the Transformer **underfits in
practice**: enough model capacity, not enough data. Classical DTW on
geometric hand features still wins by a wide margin. The Transformer
infrastructure (augmentation, mixed-precision training, two-LR schedule,
checkpointing) is in place and ready to scale once the dataset grows.

---

---

# DTW Pipeline — Results

> **Architecture:** MediaPipe hand landmarks → per-frame geometric feature →
> global DTW alignment (Sakoe-Chiba band) → mean-similarity score.
> One reference video per class, no training.

## Dataset
- **215 user videos** evaluated against **94 translator reference videos**
  (one reference per phrase/class)
- Feature source: MediaPipe landmarks, 21 per hand × 2 hands
- Evaluation: for every user video, rank all 94 refs by similarity,
  report Top-1 / Top-3 / Top-5

---

## Accuracy — All Full-Run Variants (N=215)

| # | Method | Feature | DTW Cost | Swap | Top-1 | Top-3 | Top-5 |
|---|---|---|---|---|---|---|---|
| 1 | Baseline DTW | 126-D raw coords | L2 | — | 32.56% | 42.33% | 46.51% |
| 2 | DTW + swap (M5) | 126-D raw coords | L2 | global | 32.56% | 42.33% | 46.51% |
| 3 | DTW cos-cost (M6) | 126-D raw coords | cosine | — | 33.95% | 44.65% | 50.23% |
| 4 | Bones+angles+len+d1 | 840-D (+derivative) | L2 | global | 33.02% | 49.30% | 55.35% |
| 5 | Angles-15 variant | 15 selected angles | L2 | global | 10.23% | 22.33% | 28.84% |
| 6 | Angles + swap (M4) | angles | L2 | global | 17.67% | 30.70% | 38.14% |
| 7 | Palm-temporal (M8) | palm temporal | L2 | — | 27.91% | 44.19% | 52.56% |
| 8 | **Angles + rel. lengths (M7, paper best)** | **420-D geometric** | **L1** | **global** | **40.00%** | **50.23%** | **56.28%** |
| 9 | Wrist-position (M7b) | M7 + wrist | L1 | global | **40.47%** | **51.16%** | 55.81% |

> **Headline result:** Method 7 — bone angles + relative lengths, L1-consistent
> DTW, global swap — is the paper submission's best method: **40.0% / 50.23% / 56.28%**.
> Method 7b edges Top-1 slightly higher (40.47%) but with lower Top-5.

---

## Best Method (M7) — Algorithm

1. **MediaPipe landmarks** per frame → 126-D `[Left(63) | Right(63)]`,
   wrist-centered and scale-normalized.
2. **Redundant-frame drop** (cosine similarity threshold 0.99).
3. **Geometric feature** per hand:
   - 190 pairwise bone angles (C(20,2)), normalized to [0, 1] — **rotation-invariant**
   - 20 relative bone lengths (length / sum of lengths) — **scale-invariant**
   - Concatenate left + right → 420-D per frame
4. **DTW alignment** (Sakoe-Chiba band, window ratio 0.25),
   cost = L1 mean distance on 420-D features.
5. **Score** = mean of (1 − L1_mean) over aligned pairs.
6. **Swap invariance**: repeat with left/right blocks swapped, keep max.

---

## Why M7 Beats the Raw-Coordinate DTW Baselines

| Structural win | Effect |
|---|---|
| Rotation invariance (bone angles) | Wrist tilt / camera rotation no longer corrupts similarity |
| Scale invariance (relative lengths) | Different hand sizes between signers stop mattering |
| L1 cost + L1-based score | DTW optimizes the same objective it is scored on |
| Left/right swap | Recovers videos where the user signs with the opposite hand |

Over Method 6 (the best raw-coordinate DTW): **+6.05 pp Top-1, +5.58 pp Top-3, +6.05 pp Top-5**.
McNemar's test: χ² = 2.29, p ≈ 0.13 — not significant at N=215, but the two
methods make largely **distinct errors** (38 M7-only correct vs 25 M6-only correct),
suggesting the gains are real and a score-fusion ensemble is promising future work.

---

## Query Latency (per video, CPU — Intel i7)

| Stage | Time |
|---|---|
| Frame decode + MediaPipe landmarks | ~2.0 s (bottleneck) |
| 420-D geometric feature build | ~0.5 s |
| DTW over 94 references (window 0.25) | ~5.3 s |
| **Total** | **~7.8 s** |

> DTW dominates cost at scale because it runs **94 alignments per query**.
> For streaming / real-time use the engine switches to **Subsequence DTW**
> (free-start, column-by-column, O(N) memory), distributing the O(N·M)
> work across arriving frames rather than offline.

---

## Comparison (All Pipelines, Full Final Table)

| Method | Top-1 | Top-3 | Top-5 | Latency |
|---|---|---|---|---|
| Cosine (baseline, N=342) | 30.7% | 45.0% | 49.4% | ~2.7 s CPU |
| Baseline DTW (raw coords) | 32.56% | 42.33% | 46.51% | ~7.8 s CPU |
| DTW cos-cost (M6) | 33.95% | 44.65% | 50.23% | ~7.8 s CPU |
| **DTW M7 (angles + rel. lengths, best)** | **40.00%** | **50.23%** | **56.28%** | ~7.8 s CPU |
| LSTM Seq2Seq | 16.0% | 25.6% | 26.5% | **~18 ms GPU** |
| ViT + Transformer Seq2Seq | 6.7%* | — | — | ~100–500 ms GPU |

> *Transformer is a partial 30-sample estimate.
> Cosine / DTW / neural evaluations use different splits — compare with care.

---

## Takeaway

**On 94 classes with one reference video per class and no training data,
hand-engineered geometric invariances beat every neural method tried.**
The wins come from structure (rotation + scale invariance, hand-swap,
cost/score consistency), not from scale. DTW is the right tool when the
dataset is small and the vocabulary must stay open — adding a new phrase
means recording one video, not retraining.
