# Slide 1 — Title
## Sign Language Sentence Recognition (AzSL)
- Instructor format: Recognition slide + one slide per method
- Methods: Cosine, DTW, LSTM, Transformer
- Goal: recognize/assess signed sentences from video

## Defense Message
We start with deployable similarity baselines (cosine/DTW), then test deep models (LSTM and Transformer) under the same data constraints.

---

# Slide 2 — Recognition (What the end user sees)
## End-to-End Recognition UX
- Input: a learner records a sentence video
- Output (system): predicted sentence or Top-K candidates + confidence/similarity
- Feedback: “closest match” + score helps grading (not only correct/incorrect)

## Shared Pipeline (Architecture)
1. Video → frames (sampling / padding)
2. Hand-centric representation
   - MediaPipe landmarks (baseline) OR frame embeddings (deep models)
3. Temporal reasoning
   - alignment/retrieval OR seq2seq decoding
4. Prediction → sentence text

## Augmentation / Robustness (Approach)
- Temporal variance: fixed-length sampling (e.g., N=64 for retrieval; N=16 for ViT)
- Frame quality: keep only frames with detected hands (baseline + LSTM features)
- Scale/position: center-crop around hands (LSTM feature extraction)

---

# Slide 3 — Method 1: Similarity Baseline (Cosine)
## Architecture
- Video → MediaPipe hands (up to 2 hands)
- Feature per frame: 21 landmarks × (x,y,z) × 2 hands = 126D → stacked over time
- Normalization: center at wrist + scale by hand spread (invariant to scale/shift)
- Matching: cosine similarity on flattened landmark matrix

## Augmentation / Approach
- Filter frames to “hand-present” frames to reduce noise
- Fixed-length representation: N=64 sampled frames per clip

## Results (Qualitative — end user view)
- Feels stable: returns Top-K most similar sentences even if not perfect
- Interpretable: you can show the similarity score and nearest examples
- Main failure: different sentences with similar motion rank high (confusions)

## Results (Quantitative)
- Leave-one-out (342 videos, 124 sentences, N=64):
  - Top-1: 30.70%  | Top-3: 45.03% | Top-5: 49.42%
- Latency benchmark (single query):
  - MediaPipe extraction dominates (~2036 ms)
  - Search step is fast (~2.8 ms over 342 vectors)
  - Total per query ~2707 ms

---

# Slide 4 — Method 2: Similarity Baseline (DTW Alignment)
## Architecture
- Same input representation as cosine baseline (normalized landmark matrices)
- Matching: Dynamic Time Warping (DTW) with Sakoe–Chiba constraint (window ratio = 0.3)
- DTW distance → similarity via exponential decay: similarity = exp(-normalized_distance)

## Augmentation / Approach
- Solves temporal mismatch: aligns frames when sign speed differs
- Optional visualization: shows the warping path + matched frame pairs

## Results (Qualitative — end user view)
- More forgiving when the learner performs the same sign faster/slower
- Produces a more “fair” similarity score when timing shifts happen
- Tradeoff: noticeably slower matching than cosine (DTW is more expensive per candidate)

## Results (Quantitative)
- Sampled DTW evaluation on saved landmark matrices (342 total videos; 30 random queries; window_ratio=0.3):
  - Top-1: 36.67% | Top-3: 36.67% | Top-5: 46.67%
  - DTW matching time (search only; excludes feature extraction): mean 7.783 s/query (median 7.799; min/max 7.643/7.947)

---

# Slide 5 — Method 3: LSTM Seq2Seq on Cached Features
## Architecture
Stage A — offline feature extraction (per video):
- Keep only frames where hands are detected (MediaPipe-based filter)
- Hand-centric crop (crop_size=600) → resize to 224×224 → ImageNet normalization
- Frame encoder: SqueezeNet1.1 conv features (layer12 = 512×13×13, flattened to 86528D)
- Temporal feature encoder: 1-layer BiLSTM (hidden=256 per direction) → cached features (T×512)

Stage B — sentence decoding (seq2seq):
- Encoder: 1-layer BiLSTM over cached features (input 512D, hidden 512) → encoder outputs (T×1024)
- Decoder: attention-based LSTM (hidden 1024) generating tokens autoregressively (uses <sos>/<eos> vocabulary)
- Inference: beam search for Top-K hypotheses

## Augmentation / Approach
- Robustness via preprocessing (hand-frame filtering + hand-centric cropping + ImageNet normalization)
- Temporal handling: pad/truncate features to max_frames=64 (uniform downsample if long)
- Practical approach: cache (T×512) features to make training/inference fast and stable on limited data
- Training stabilization: attention dropout (0.1) + temperature scaling (2.0) + early stopping

## Results (Qualitative — end user view)
- Outputs a full sentence prediction (not just nearest neighbor)
- Often gets high-frequency / easier phrases right, but can “jump” to a different learned phrase
- Confusions happen when multiple classes share similar motion patterns

## Results (Quantitative — research evaluation)
- Seq2Seq evaluation (407 videos evaluated; 93 skipped due missing feature cache):
  - Top-1: 15.97% | Top-3: 25.55% | Top-5: 26.54%
- Latency (cached features): mean total ≈ 0.018 s per video

---

# Slide 6 — Method 4: ViT + Temporal Transformer + Transformer Decoder
## Architecture
- Input: sample 16 RGB frames per clip → (224×224)
- Frame encoder: ViT-Small (timm: vit_small_patch16_224) → 384D per frame (CLS token)
- Temporal encoder: TransformerEncoder over frame sequence (d_model=256, 8 heads, 4 layers)
- Decoder: TransformerDecoder (d_model=256, 8 heads, 4 layers) generating tokens autoregressively
- Training: label smoothing (0.1), freeze ViT for first 5 epochs, two LR groups (backbone smaller LR), warmup (300 steps), AMP, grad clipping (1.0), checkpoint top-k

## Augmentation / Approach (Training-only)
- Spatial (consistent per clip): random resized crop (scale 0.7–1.0), horizontal flip (p=0.5), color jitter (p=0.8), grayscale (p=0.1), gaussian blur (p=0.3)
- Temporal: sampling-grid jitter + frame dropout (p=0.1) + temporal reverse (p=0.1)
- Regularization: CutOut (p=0.3, size=0.15 of frame) ; always ImageNet normalization (train/val/test)

## Results (Qualitative — end user view)
- Sometimes produces partially related words, but exact sentence is usually wrong
- Common error mode: repeats frequent tokens / produces generic phrases
- Indicates the need for more data + stronger alignment between labels and visual evidence

## Results (Quantitative)
- Best checkpoint evidence: epoch 27, validation loss 6.3176
- Random evaluation (100 samples): Seq-Acc 0.00, mean WER 1.0044
- Random evaluation (30 samples): Seq-Acc 6.67% (2/30), mean WER 0.8551
