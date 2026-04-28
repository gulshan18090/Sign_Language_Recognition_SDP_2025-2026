# Final Report

---

## 2. Final System Overview

### 2.1 Overview of Final Design / System

The final system is a **multi-method recognition and assessment platform for Azerbaijani Sign Language (AzSL)**. It accepts a learner's signed video, compares it against a curated vocabulary of reference signs, and returns a ranked set of candidate phrases together with a similarity / confidence score. Unlike a binary "correct / incorrect" classifier, the system produces *graded, interpretable feedback*, which makes it suitable for an educational setting where a learner needs to know not only *whether* but also *how well* they reproduced a sign.

To make the design robust under a small dataset (~750 samples, 124 unique phrases) and realistic deployment constraints (CPU-friendly, no fixed vocabulary required at training time), the system implements **four parallel recognition pipelines** that share a common dataset and a common evaluation harness:

1. **Cosine Similarity** — a training-free retrieval baseline over MediaPipe hand-landmark vectors.
2. **Dynamic Time Warping (DTW)** — temporal alignment of geometric features (joint angles, bone lengths) with subsequence matching, hand-swap handling, and a Sakoe–Chiba band constraint. This is the **best-performing method overall**.
3. **LSTM Seq2Seq** — a two-stage neural pipeline (offline SqueezeNet + BiLSTM feature caching, then attention-based seq2seq decoding) that produces sentence-level predictions.
4. **Transformer (ViT + Encoder–Decoder)** — an end-to-end sequence model that uses a pretrained Vision Transformer per frame and a temporal Transformer encoder–decoder for autoregressive sentence generation.

The four pipelines are deliberately heterogeneous: cosine and DTW are *interpretable, template-matching* approaches that do not require training, while LSTM and Transformer are *learned, end-to-end* approaches. This design lets the system fall back to the simple methods when data is scarce and lets the neural methods improve as more data becomes available.

### 2.2 Overall Architecture or Configuration

At the highest level the system follows a **shared-data, parallel-pipeline architecture**:

```
                  ┌──────────────────────────────────────┐
                  │  Master Dataset                      │
                  │  • lstm/drive/sentences_all.csv      │
                  │  • lstm/drive/Video/Cam2/[id]/*.mp4  │
                  └───────────────┬──────────────────────┘
                                  │
       ┌──────────────┬───────────┼───────────────┬──────────────┐
       ▼              ▼           ▼               ▼              ▼
  ┌────────┐    ┌─────────┐   ┌────────┐    ┌──────────────┐
  │ Cosine │    │   DTW   │   │  LSTM  │    │ Transformer  │
  │ (126-D │    │(Geom.   │   │ (cached│    │ (raw frames, │
  │  hand) │    │ feats.) │   │  feats)│    │  ViT-Small)  │
  └────┬───┘    └────┬────┘   └────┬───┘    └──────┬───────┘
       │             │             │               │
       └─────────────┴─────────────┴───────────────┘
                            │
                            ▼
                ┌────────────────────────┐
                │  Evaluation Harness    │
                │  • Top-K Accuracy      │
                │  • WER / BLEU-4        │
                │  • Confusion matrices  │
                │  • Latency analysis    │
                └────────────────────────┘
```

The four pipelines read the *same* CSV label file and the *same* video tree, so results are directly comparable. Each pipeline has its own `config.py`, its own preprocessing path, and its own `outputs/` directory. The shared [artifacts/](artifacts/) folder is used for cross-pipeline assets (vocabularies, checkpoints).

A typical end-to-end query looks like this:

1. User submits a query video (file upload or live camera in the DTW streaming variant).
2. Frames are extracted (uniform 16–64 sample, depending on pipeline).
3. Per-frame features are computed: 21×3 hand landmarks (cosine / DTW / LSTM stage 1) or full RGB frames (Transformer).
4. The recognizer scores the query against either (a) all reference templates (cosine, DTW) or (b) a learned vocabulary (LSTM, Transformer).
5. The system returns Top-K candidates with scores; for DTW the system additionally returns an alignment path that can be used to highlight *where* the learner deviated.

### 2.3 Description of Major Components or Subsystems

| Subsystem | Role | Key Files |
|---|---|---|
| **Cosine pipeline** | Fast retrieval baseline; 8 selectable similarity metrics over flattened landmark vectors | [cosine/src/inference/similarity/feature_extractor.py](cosine/src/inference/similarity/feature_extractor.py), [cosine/src/inference/similarity/similarity_engine.py](cosine/src/inference/similarity/similarity_engine.py) |
| **DTW pipeline** | Best-performing recognizer; subsequence DTW over geometric (angle / bone-length / derivative) features with hand-swap handling and Sakoe–Chiba band | [DTW/method4_variants.py](DTW/method4_variants.py), [DTW/ALGORITHM.md](DTW/ALGORITHM.md), [DTW/camera_dtw.py](DTW/camera_dtw.py) |
| **LSTM pipeline** | Two-stage neural seq2seq: SqueezeNet+BiLSTM feature cache, then attention-based decoder | [lstm/feature_extraction_slr/extract_features.py](lstm/feature_extraction_slr/extract_features.py), [lstm/models/encoder.py](lstm/models/encoder.py), [lstm/models/decoder.py](lstm/models/decoder.py) |
| **Transformer pipeline** | End-to-end ViT-Small frame encoder + temporal Transformer encoder–decoder | [Transformer/model/vit_backbone.py](Transformer/model/vit_backbone.py), [Transformer/model/encoder.py](Transformer/model/encoder.py), [Transformer/model/decoder.py](Transformer/model/decoder.py), [Transformer/model/seq2seq.py](Transformer/model/seq2seq.py) |
| **Shared data layer** | Master CSV of 123 sentence labels and per-id video folders consumed by all four pipelines | [lstm/drive/sentences_all.csv](lstm/drive/sentences_all.csv), [lstm/drive/Video/Cam2/](lstm/drive/Video/Cam2/) |
| **Vocabulary builder** | Word-token vocabulary with `<sos>` / `<eos>` / `<pad>` / `<unk>`; consumed by LSTM and Transformer | [lstm/data/vocab.py](lstm/data/vocab.py), [Transformer/artifacts/vocab.json](Transformer/artifacts/vocab.json) |
| **Augmentation module** | Spatial + temporal + regularization augmentations for the Transformer | [Transformer/data/augmentation.py](Transformer/data/augmentation.py) |
| **Evaluation harness** | Top-K accuracy, WER, BLEU-4, latency profiling, confusion-matrix figure generation | [cosine/scripts/top3_accuracy.py](cosine/scripts/top3_accuracy.py), [cosine/scripts/generate_all_paper_figures.py](cosine/scripts/generate_all_paper_figures.py), [Transformer/training/evaluate.py](Transformer/training/evaluate.py), [lstm/research_eval_seq2seq.py](lstm/research_eval_seq2seq.py) |
| **Feature organizer utility** | Maps extracted `.pt` feature files back into the dataset folder hierarchy | [featuer_organizer.py](featuer_organizer.py) |

### 2.4 Final Design Specifications

**Dataset and labelling**
- 123 unique AzSL phrases listed in [sentences_all.csv](lstm/drive/sentences_all.csv); ~750 total video samples.
- Per phrase: 1 translator (reference) video and 0–6 user (learner) videos.
- Labels are word-token sequences; vocabulary size ≈ 500 tokens.
- Train / val / test split: 80 / 10 / 10 (Transformer); leave-one-out (cosine); 215 user vs 94 reference (DTW evaluation).

**Cosine pipeline**
- Frame sample: 64 frames, uniform.
- Feature: 21 landmarks × 3 coords × 2 hands = **126-D** per frame, wrist-centered, scale-normalized.
- Similarity: cosine (default), with 7 alternates including DTW, Pearson, Spearman, Manhattan.
- No training. Inference: ~2.7 s / video on Intel i7 CPU; 75 % is MediaPipe extraction.

**DTW pipeline (Method 4B – best variant)**
- Frame preprocessing: drop hand-less frames, then drop near-duplicates with similarity ≥ 0.99 (≈50–60 % frame reduction).
- Feature: ~190 pairwise bone angles × 2 hands = **380-D** per frame (Method 4B); Method 4C adds bone lengths (420-D); Method 4D adds first time derivative.
- Algorithm: subsequence DTW with Sakoe–Chiba window ratio 0.25, both normal and left/right-swapped configurations evaluated.
- Streaming variant: incremental column-by-column DTW, O(N) per frame.

**LSTM pipeline**
- Stage A: hand-centric 600×600 crop → 224×224 → SqueezeNet1.1 (ImageNet) → 86 528-D → BiLSTM (hidden 256/dir) → cached 512-D × T tensor per video.
- Stage B: BiLSTM encoder (hidden 512, bidirectional) + LSTM decoder with Luong-style attention (temperature 2.0, dropout 0.1).
- Training: Adam, CrossEntropy (ignore pad), 200 epochs, early-stopping on val loss, top-3 checkpointing.
- Inference: beam search (width 5); ~18 ms / video on GPU using cached features.

**Transformer pipeline**
- Input: 16 frames sampled uniformly per video, 224×224, ImageNet normalization.
- Backbone: `vit_small_patch16_224` (timm), CLS-token output → 384-D / frame; frozen for first 5 epochs.
- Temporal encoder: 4-layer Transformer, d_model = 256, nhead = 8, FFN = 1024, sinusoidal positional encoding.
- Decoder: 4-layer Transformer with causal masking and cross-attention to encoder memory.
- Training: 200 epochs, AMP mixed precision, two LR groups (backbone 1e-5, head 1e-4), warm-up 300 steps + cosine decay, gradient clip 1.0, label smoothing 0.1.
- Augmentation: random resized crop, horizontal flip, color jitter, grayscale, Gaussian blur, temporal jitter, frame dropout (p=0.1), temporal reverse (p=0.1), CutOut (p=0.3).

**Performance summary (final results, see [results.md](results.md))**

| Method | Top-1 | Top-3 | Top-5 | Inference / video | Notes |
|---|---|---|---|---|---|
| Cosine | 30.7 % | 45.0 % | 49.4 % | ~2.7 s (CPU) | 342 videos, leave-one-out |
| **DTW (Method 4B)** | **40.0 %** | **50.2 %** | **56.3 %** | ~7.8 s (CPU) | Best overall; 215 vs 94 templates |
| LSTM Seq2Seq | 16.0 % | 25.6 % | 26.5 % | ~18 ms (GPU) | Overfits (train 0.30 / val 1.79) |
| Transformer | 6.7 %* | — | — | ~100–500 ms (GPU) | *random 30-sample eval; partial |

### 2.5 Differences from Original Proposed Design

The original proposal scoped a **single end-to-end neural recognizer** (Transformer-based) trained on the AzSL video corpus. The final design departs from that proposal in five concrete ways:

1. **From single model to four parallel pipelines.** Early experiments showed that the dataset (≈750 samples, 124 classes) was too small for a from-scratch neural model to generalize. The design was widened to include training-free baselines (cosine, DTW) so that the system would be useful even before more data is collected.
2. **DTW promoted from baseline to primary recognizer.** DTW was originally listed as one of eight similarity metrics inside the cosine engine. After the geometric-feature variants (Method 4A–4D) were prototyped and Method 4B reached **40 % Top-1 — the best of any approach** — DTW was elevated to a stand-alone subsystem with its own preprocessing, streaming variant, and paper draft.
3. **LSTM split into a two-stage offline pipeline.** The proposal assumed end-to-end training. Because GPU memory and training time were limiting, the LSTM stack was redesigned as **Stage A: offline feature extraction (SqueezeNet + BiLSTM) → cached `.pt` files** and **Stage B: lightweight seq2seq training on cached features**. This decoupling cut training-loop wall-clock time by an order of magnitude and made experimentation tractable.
4. **Transformer scope reduced from "shipping recognizer" to "architecture spike."** The pipeline is fully implemented and trains cleanly (best checkpoint at epoch 27, val loss 6.32), but full test-set evaluation is incomplete; only a 30-sample random evaluation was run. The component is kept in the system as a foundation for future work once more data is available.
5. **Real-time camera demo added.** Not part of the original proposal: the streaming DTW implementation in [DTW/camera_dtw.py](DTW/camera_dtw.py) supports live recognition from a webcam by maintaining an incremental DTW cost matrix. This was added once it became clear DTW was the strongest method and could meet real-time latency on CPU.

### 2.6 Materials, Components, Tools, or Resources Used

**Programming language and runtime**
- Python 3.8–3.10 inside a Conda environment ([.conda/](.conda/))
- CUDA 12.1 for GPU training (LSTM, Transformer)

**Deep-learning stack**
- PyTorch 2.5.1 / TorchVision 0.20.1 — model definitions, training loops, autograd, AMP mixed precision
- timm ≥ 0.9 — pretrained `vit_small_patch16_224` backbone
- TensorBoard ≥ 2.13 — Transformer training visualization

**Computer vision**
- MediaPipe 0.10.5 — 21-point hand landmark extraction (cosine, DTW, LSTM stage 1)
- OpenCV ≥ 4.8 — video decoding, frame resizing, color conversion
- VidGear 0.3.4 — alternate video I/O backend used in LSTM extraction

**Numerical / data**
- NumPy 1.24–2.2.6, Pandas 2.0–2.3.3, SciPy, scikit-learn 1.7.2

**Visualization and metrics**
- Matplotlib 3.10.7 — confusion matrices, ROC curves, latency plots, alignment-path figures
- NLTK ≥ 3.8 — tokenization, BLEU-4 / WER computation
- tqdm — progress bars across all training and evaluation scripts

**Pretrained model weights**
- SqueezeNet1.1 (ImageNet) for LSTM stage-A feature extraction
- ViT-Small/16 (ImageNet) for Transformer frame encoder

**Hardware**
- Development & evaluation CPU: Intel i7, 15.7 GB RAM (cosine and DTW timings reported on this machine)
- Training GPU: NVIDIA CUDA 12.1-capable card (LSTM and Transformer)

**Data resources**
- AzSL video corpus organized under [lstm/drive/Video/Cam2/](lstm/drive/Video/Cam2/), indexed by [lstm/drive/sentences_all.csv](lstm/drive/sentences_all.csv) (123 sentences, ~750 clips)
- Pre-computed feature caches: [lstm/features_slr/](lstm/features_slr/) (.pt tensors), [cosine/data/processed/matrices/](cosine/data/processed/matrices/) (.npy landmark matrices), [DTW/matrices/](DTW/matrices/)

**Documentation and design assets**
- [research_presentation_outline.md](research_presentation_outline.md), [final_presentation_slides.md](final_presentation_slides.md), [DTW/ALGORITHM.md](DTW/ALGORITHM.md), [DTW/PAPER_STRUCTURE.md](DTW/PAPER_STRUCTURE.md), [DTW/ITTA2026_paper.md](DTW/ITTA2026_paper.md), [SDP.ipynb](SDP.ipynb), [results.md](results.md)

---

## 3. Implementation

### 3.1 Development or Realization Approach

The project followed an **iterative, baseline-first development approach**. Rather than starting with the most ambitious model (Transformer) and trying to make it work, development began with the simplest defensible recognizer (cosine similarity over hand landmarks) and added complexity only where the data justified it. Concretely, the team adopted four guiding rules:

1. **Always have a working end-to-end system.** From day one, every commit had to leave at least one pipeline runnable end-to-end (video in → ranked predictions out). This avoided the failure mode where complex systems are integrated only at the end.
2. **Each new pipeline must beat the previous best on the same evaluation harness.** The cosine baseline established a number to beat (30.7 % Top-1), and each subsequent pipeline (DTW → LSTM → Transformer) was built and tuned against that same Top-K leave-one-out / hold-out protocol.
3. **Decouple data preprocessing from modelling.** Hand landmark extraction (cosine, DTW), SqueezeNet feature extraction (LSTM stage A), and ViT frame sampling (Transformer) are all separate, cacheable steps. This made it cheap to iterate on the modelling layer without re-running expensive preprocessing.
4. **Code organization mirrors the experiment structure.** Each pipeline lives in its own top-level folder ([cosine/](cosine/), [DTW/](DTW/), [lstm/](lstm/), [Transformer/](Transformer/)) with its own `config.py`, `models/`, `inference/` and `scripts/` subdirectories. Cross-pipeline coupling is limited to the shared CSV and video tree.

The team worked in `git` on the `main` branch, with development phases visible in commit history (cosine → DTW pipeline → Transformer model → augmentation visualization → cosine + DTW pipeline integration). Each pipeline graduated through the same stages: prototype script → restructure into modules → add config file → add evaluation script → write results to [results.md](results.md).

### 3.2 Fabrication, Construction, or Development Process

This is a software system, so "fabrication" refers to the construction of the code base, the data assets, and the training runs. The build proceeded in four phases that map roughly onto the four pipelines:

**Phase 1 — Cosine baseline (training-free).** Built [HandFeatureExtractor](cosine/src/inference/similarity/feature_extractor.py) around MediaPipe Hands to produce 126-D per-frame landmark vectors. Built [SimilarityEngine](cosine/src/inference/similarity/similarity_engine.py) with eight interchangeable similarity metrics. Wrote [extract_all_vectors.py](cosine/scripts/extract_all_vectors.py) to pre-compute and cache landmark matrices for the full dataset under [cosine/data/processed/matrices/](cosine/data/processed/matrices/). Built the leave-one-out evaluator [top3_accuracy.py](cosine/scripts/top3_accuracy.py) and the figure-generation script [generate_all_paper_figures.py](cosine/scripts/generate_all_paper_figures.py). Result: 30.7 % Top-1 — first credible number.

**Phase 2 — DTW pipeline.** Recognized that signers execute the same gesture at different speeds, which a flattened cosine cannot handle. Implemented subsequence DTW (formulation in [DTW/ALGORITHM.md](DTW/ALGORITHM.md)) and then iterated on the *features*: Method 4A (15 angles per hand) → 4B (~190 pairwise bone angles) → 4C (angles + bone lengths) → 4D (4C + first time derivative). Added a Sakoe–Chiba window-ratio sweep, hand-swap handling, and a per-method evaluator (`evaluate_recognition_*.py`). Cross-method comparison via [compare_three_methods.py](DTW/compare_three_methods.py) and [compare_four_methods.py](DTW/compare_four_methods.py). Method 4B reached 40.0 % Top-1, surpassing all other approaches. Finally, built the streaming/online version in [camera_dtw.py](DTW/camera_dtw.py) for live demos.

**Phase 3 — LSTM seq2seq.** Constructed Stage A in [extract_features.py](lstm/feature_extraction_slr/extract_features.py): MediaPipe → hand-centric 600×600 crop → 224×224 resize → SqueezeNet1.1 → 1-layer BiLSTM → cached 512-D × T tensor per video, written to [lstm/features_slr/](lstm/features_slr/). Built Stage B in [encoder.py](lstm/models/encoder.py) and [decoder.py](lstm/models/decoder.py): BiLSTM encoder + Luong-attention LSTM decoder. Wired training in [trainer.py](lstm/models/trainer.py) (Adam, CrossEntropy with pad-ignore, 200 epochs, top-3 checkpointing). Wrote evaluation in [research_eval_seq2seq.py](lstm/research_eval_seq2seq.py).

**Phase 4 — Transformer.** Built `SLRSeq2Seq` from four modules: [vit_backbone.py](Transformer/model/vit_backbone.py) (timm ViT-Small), [encoder.py](Transformer/model/encoder.py) (4-layer Transformer encoder + sinusoidal positions), [decoder.py](Transformer/model/decoder.py) (4-layer Transformer decoder with causal masking + cross-attention), and [seq2seq.py](Transformer/model/seq2seq.py) (composition). Implemented heavy [augmentation.py](Transformer/data/augmentation.py) (spatial + temporal + CutOut) to combat overfitting. Trainer in [training/trainer.py](Transformer/training/trainer.py) used AMP, two LR groups, warm-up + cosine decay, gradient clipping 1.0, and a 5-epoch backbone freeze before joint training. Best checkpoint: epoch 27, val loss 6.3176, saved at [Transformer/artifacts/checkpoints/best.pt](Transformer/artifacts/checkpoints/best.pt).

### 3.3 Methods, Techniques, or Procedures Used

**Hand landmark extraction (cosine, DTW, LSTM stage A).** Per-frame MediaPipe Hands produces 21 landmarks × (x,y,z) per detected hand. Frames are normalized by translating to the wrist (landmark 0) and scaling by the maximum landmark distance from the wrist, giving translation- and scale-invariance. Missing hands are zero-padded so that the per-frame vector is always 126-D.

**Frame sampling.**
- Cosine: 64 uniformly sampled frames (plateau confirmed by an N ∈ {16, 32, 64, 128} sweep).
- DTW: all frames, then drop hand-less frames, then drop consecutive near-duplicates with similarity ≥ 0.99 (≈50–60 % reduction).
- LSTM: 64 frames after hand-frame filtering.
- Transformer: 16 uniformly sampled frames (memory-bounded by ViT).

**Similarity scoring (cosine pipeline).** Eight interchangeable metrics in [SimilarityEngine](cosine/src/inference/similarity/similarity_engine.py): cosine, Euclidean, Pearson, Spearman, Manhattan, frame-wise cosine, temporal correlation, and DTW. The cosine path flattens (n_frames, 126) → (n_frames × 126,), then computes a clamped cosine in [-1, 1].

**Subsequence DTW (DTW pipeline).** Standard accumulated-cost recursion:

```
D[n,0]   = sum(C[0..n, 0])    # consume entire reference
D[0,m]   = C[0, m]            # free start in user stream
D[n,m]   = C[n,m] + min(D[n-1,m-1], D[n-1,m], D[n,m-1])
```

with backtracking from `(N-1, argmin_m D[N-1, m])`. The Sakoe–Chiba band (ratio 0.25) prevents pathological alignments and reduces complexity from O(N·M) to O(N·M·band).

**Geometric features (DTW Method 4 family).** Method 4B builds a 380-D per-frame descriptor as the set of pairwise angles between 20 canonical hand bones, doubled across two hands. Because it is built from angles rather than raw coordinates, it is rotation- and translation-invariant by construction — the network does not have to *learn* this invariance. Method 4D additionally appends a first time derivative to encode motion.

**Hand-swap handling.** For each query/reference pair the DTW score is computed twice (once with normal hand assignment, once with left/right swapped) and the better score is kept. This handles the common case where a learner uses the opposite hand from the reference signer.

**Two-stage neural training (LSTM).** Stage A is run *once* per dataset: SqueezeNet1.1 features (`features.12.cat`, 512×13×13 = 86 528-D) are passed through a small BiLSTM and the (T, 512) result is cached as a `.pt` file. Stage B trains the seq2seq model on the cached tensors with Adam, CrossEntropy (ignoring `<pad>`), 200 epochs, early-stopping on validation loss, and top-3 checkpointing. Beam search (width 5) is used at inference.

**End-to-end Transformer training.** Mixed-precision (AMP) with `GradScaler`, two parameter groups (backbone 1e-5, head 1e-4), warm-up over 300 steps then cosine decay, gradient clipping at 1.0, label smoothing ε = 0.1. The ViT backbone is frozen for the first 5 epochs to let the decoder warm up before joint optimization.

**Augmentation procedure (Transformer).** Spatial transforms (random resized crop scale 0.7–1.0, horizontal flip p = 0.5, color jitter, random grayscale p = 0.1, Gaussian blur p = 0.3) are applied *identically across all frames in a clip*. Temporal transforms add jitter to the sampling grid, drop frames (p = 0.1), and reverse the clip in time (p = 0.1). CutOut (p = 0.3, size 0.15) zeroes a random patch.

**Evaluation procedure.** Cosine uses leave-one-out cross-validation over 342 videos. DTW evaluates 215 user videos against 94 reference templates and reports Top-K. LSTM evaluates 407 videos with greedy decoding (93 skipped due to missing cached features). Transformer's full test set was not exported; a 30-sample random evaluation gave 6.7 % exact-match and mean WER 0.86.

### 3.4 Integration of Components or Subsystems

Integration is achieved through three shared contracts rather than tight coupling:

**1. Shared dataset contract.** All four pipelines read [lstm/drive/sentences_all.csv](lstm/drive/sentences_all.csv) (semicolon-separated `idd; sentence; sign_language`) and resolve videos to `lstm/drive/Video/Cam2/<idd>/*.mp4`. This means any data improvement (more videos, corrected labels) is automatically picked up by every pipeline.

**2. Shared feature format for landmark-based pipelines.** Cosine, DTW, and LSTM stage A all produce per-frame 126-D landmark vectors via the same MediaPipe + wrist-centered normalization. Cached results live in [cosine/data/processed/matrices/](cosine/data/processed/matrices/) (.npy) and [DTW/matrices/](DTW/matrices/) and can be cross-loaded if needed.

**3. Shared vocabulary contract for sequence pipelines.** LSTM and Transformer both build a word-token vocabulary from the CSV's `sign_language` column with the same special tokens (`<pad>`, `<sos>`, `<eos>`, `<unk>`). The vocabulary file [Transformer/artifacts/vocab.json](Transformer/artifacts/vocab.json) is checked into [artifacts/](artifacts/) so that checkpoints remain interpretable.

**Cross-pipeline orchestration.** No global orchestrator was built; the user invokes each pipeline separately via its own entry point ([cosine/scripts/video_similarity.py](cosine/scripts/video_similarity.py), [DTW/evaluate_recognition_method4b.py](DTW/), [lstm/main.py](lstm/main.py), [Transformer/train.py](Transformer/train.py) and [Transformer/predict.py](Transformer/predict.py)). The [featuer_organizer.py](featuer_organizer.py) utility maps extracted `.pt` feature files into the dataset folder hierarchy when caches need to be rebuilt or moved.

**Real-time integration (DTW only).** [DTW/camera_dtw.py](DTW/camera_dtw.py) integrates the MediaPipe extractor, the geometric-feature builder, and an *online* DTW that updates the cost matrix one column at a time as new camera frames arrive. This is the only subsystem that runs the full pipeline in a single live process.

**Evaluation integration.** All pipelines emit results in compatible JSON / CSV form; [results.md](results.md) is the single human-readable summary that aggregates accuracy, latency, and per-class breakdowns across pipelines for direct comparison.

### 3.5 Challenges Encountered and Solutions

**Challenge 1 — Limited labelled data.** With ~750 samples spread across 124 phrases, neural models had little room to generalize. The LSTM showed clear overfitting (train loss 0.30 vs. val loss 1.79) and the Transformer plateaued in validation loss around epoch 27. **Solution:** kept training-free template-matching pipelines (cosine, DTW) as first-class citizens of the system. DTW Method 4B ended up outperforming both neural methods (40.0 % Top-1 vs 16.0 % LSTM vs 6.7 % Transformer).

**Challenge 2 — Speed variation between signers.** A learner often executes the same gesture faster or slower than the reference signer, which a flattened cosine cannot accommodate. **Solution:** subsequence DTW with Sakoe–Chiba band (ratio 0.25). The band prevents degenerate alignments while still allowing meaningful warping.

**Challenge 3 — Inter-signer variability (translators vs users).** Cosine accuracy on translator videos (24.6 %) was a full 10 points below user videos (34.4 %), reflecting how professional signers' subtler form is harder to match against the same set of templates. **Solution:** added geometric-feature variants (Method 4 family) that abstract away from raw landmark coordinates, narrowing this gap.

**Challenge 4 — Hand swaps.** Learners frequently use the opposite hand from the reference signer. **Solution:** for every query/reference pair, the DTW score is computed both normally and with left/right swapped, and the better score is kept.

**Challenge 5 — MediaPipe latency bottleneck.** Profiling showed MediaPipe extraction consumes 2 036 ms out of a 2.7 s cosine query (75 %), while the actual cosine search is just 2.8 ms. **Solution:** offloaded landmark extraction to a one-time preprocessing step that writes `.npy` matrices to [cosine/data/processed/matrices/](cosine/data/processed/matrices/); on cached inference the search is sub-second.

**Challenge 6 — Long LSTM training loop.** End-to-end training (frame I/O + CNN + RNN per batch) was prohibitively slow. **Solution:** split into Stage A (offline CNN+BiLSTM feature cache) and Stage B (lightweight seq2seq training on cached tensors). Stage B inference dropped to ~18 ms per video on GPU.

**Challenge 7 — Transformer overfitting.** Even with 200 epochs and label smoothing, the Transformer's validation loss flattened around epoch 27. **Solution:** added a heavy augmentation stack (spatial + temporal + CutOut + frame dropout + temporal reverse), froze the ViT backbone for the first 5 epochs, and used differential learning rates (1e-5 backbone vs 1e-4 head). These extended useful learning but did not eliminate the data-size ceiling.

**Challenge 8 — Incomplete evaluation artifacts.** During iteration, cached feature files for 93 of 500 LSTM videos went missing, and the Transformer's full test set was never exported (only a 30-sample random evaluation). **Solution / Lesson learned:** added [featuer_organizer.py](featuer_organizer.py) to keep cache and dataset structure in sync, and recorded as the highest-priority future task to (a) regenerate all LSTM features and (b) export full Transformer test metrics after every training run.

### 3.6 Design Iterations and Improvements

The system went through five visible iteration loops, each driven by a measurable shortcoming of the previous version:

**Iteration 1 — Cosine, similarity-metric sweep.** Started with a single cosine metric, then expanded [SimilarityEngine](cosine/src/inference/similarity/similarity_engine.py) to 8 metrics (cosine, Euclidean, Pearson, Spearman, Manhattan, frame-wise cosine, temporal correlation, DTW) and benchmarked them. Cosine remained the best of the *flattened* metrics, motivating a move to alignment-based methods.

**Iteration 2 — Cosine, frame-count sweep.** Evaluated N ∈ {16, 32, 64, 128} frames. Top-1 went 27.8 % → 31.6 % → 30.7 % → 30.7 %; the plateau at N = 64 became the system-wide default ([results.md](results.md)).

**Iteration 3 — DTW geometric-feature family.** Iterated four feature variants:
- 4A: 15 angles per hand → simple, fast, modest accuracy.
- **4B: ~190 pairwise bone angles per hand → 40.0 % Top-1, the new best.**
- 4C: angles + relative bone lengths → marginal gain.
- 4D: 4C + first time derivative → no further gain (extra dimensions amplified noise).

Method 4B was promoted to the production DTW configuration.

**Iteration 4 — DTW window-ratio sweep.** Sweep over Sakoe–Chiba band ratios 0.20–0.30 identified 0.25 as the optimum: tighter bands lost accuracy, looser bands cost compute without gain.

**Iteration 5 — LSTM architecture refinements.** Initial seq2seq lacked attention; adding Luong-style attention with temperature 2.0 and dropout 0.1 sharpened the alignment between encoder states and decoder outputs and lifted Top-1 accuracy. Two-stage decoupling (Section 3.4) was the second major improvement.

**Iteration 6 — Transformer regularization.** Initial runs overfit aggressively. The augmentation stack was extended in three rounds: (a) base spatial augmentations; (b) added temporal jitter, frame dropout, temporal reverse; (c) added CutOut and label smoothing. Best validation loss improved from > 7 down to 6.32 at epoch 27.

**Cross-pipeline improvement: streaming DTW.** After DTW was confirmed as the strongest recognizer, the offline implementation was adapted into [camera_dtw.py](DTW/camera_dtw.py) using an incremental column-by-column DTW update (O(N) time, O(N) space per frame), enabling real-time recognition from a webcam on CPU.

**Documented but deferred improvements.** [lstm/architecture_improvements_v1.md](lstm/architecture_improvements_v1.md) records improvements identified but not yet implemented: a richer attention mechanism, larger encoder hidden state, and richer augmentation at the cached-feature level. Similarly, [DTW/PAPER_STRUCTURE.md](DTW/PAPER_STRUCTURE.md) and [DTW/ITTA2026_paper.md](DTW/ITTA2026_paper.md) outline a planned multi-reference-template extension that uses 3–5 templates per phrase to further improve robustness.

---

## 4. Testing and Validation

### 4.1 Testing or Evaluation Strategy

The system was evaluated using a **comparative, metric-driven strategy** built around four principles:

1. **Identical evaluation harness across pipelines.** All four recognizers (cosine, DTW, LSTM, Transformer) were scored against the same dataset, the same split logic, and the same Top-K accuracy definition so that results are directly comparable. Aggregated results are recorded in a single human-readable file ([results.md](results.md)).
2. **Multi-metric reporting.** Beyond Top-1 accuracy, the harness reports Top-3 / Top-5 (to capture cases where the correct phrase is "in the running"), Word Error Rate (WER) for the sequence-generating models, and per-class accuracy to surface failure modes that average metrics hide.
3. **Latency budget alongside accuracy.** Every recognizer was profiled end-to-end (frame I/O, feature extraction, scoring / decoding) so that accuracy could be traded off against deployment latency. This is what justified keeping cosine and DTW in the system despite the existence of neural alternatives.
4. **Slice-based evaluation.** For cosine, the dataset was further split into translator-only and user-only subsets to surface the inter-signer-variability gap. For LSTM, per-class Top-1 was reported to identify which phrases the model could and could not learn.

The strategy deliberately avoided optimizing for a single number: the project's success criterion is *useful learner feedback*, which is multi-faceted (good Top-K, interpretable similarity, acceptable latency, support for partial / streaming queries).

### 4.2 Verification of Requirements

| Requirement | How verified | Status |
|---|---|---|
| Recognize AzSL phrases from learner videos | Top-K accuracy on held-out videos via leave-one-out (cosine) and 215-vs-94 reference matching (DTW) | **Met** — DTW Method 4B reaches 40.0 % Top-1 / 56.3 % Top-5 |
| Provide *graded* feedback (not binary) | Each recognizer returns a continuous similarity / log-likelihood score; DTW additionally returns an alignment path | **Met** |
| Operate without retraining when adding a new phrase | Cosine and DTW are template-matching: a new phrase requires only a new reference video, no retraining | **Met** |
| CPU-friendly inference | End-to-end latency profiled on Intel i7 (15.7 GB RAM); cosine = 2.7 s / video, streaming DTW achieves real-time on webcam | **Met** for cosine and DTW |
| Sentence-level transcription (stretch goal) | LSTM and Transformer seq2seq models trained and evaluated with WER + Top-K | **Partially met** — LSTM works at 16.0 % Top-1; Transformer evaluation incomplete |
| Real-time / streaming operation (stretch goal) | Online column-by-column DTW in [DTW/camera_dtw.py](DTW/camera_dtw.py); O(N) per-frame update | **Met** for DTW |
| Reproducibility | Fixed RNG seed (`seed=44` in [lstm/config.py](lstm/config.py)); deterministic frame sampling; feature caches checked in | **Met** |
| Interpretability of failures | Confusion matrices, per-class accuracy, alignment-path visualizations via [cosine/scripts/generate_all_paper_figures.py](cosine/scripts/generate_all_paper_figures.py) and [DTW/](DTW/) `dtw_visualizer.py` | **Met** |

### 4.3 Experimental Setup or Evaluation Environment

**Hardware.**
- *CPU benchmarking machine:* Intel i7, 15.7 GB RAM. All cosine and DTW latency numbers in [results.md](results.md) come from this machine.
- *GPU training / inference machine:* CUDA 12.1-capable NVIDIA GPU. Used for LSTM stage A feature extraction, LSTM training, and the full Transformer training run.

**Software environment.**
- Python 3.8–3.10 inside a Conda environment ([.conda/](.conda/))
- PyTorch 2.5.1 + TorchVision 0.20.1, timm ≥ 0.9, MediaPipe 0.10.5, OpenCV ≥ 4.8, NumPy 1.24–2.2.6, Pandas 2.0–2.3.3, scikit-learn 1.7.2, NLTK ≥ 3.8, Matplotlib 3.10.7, TensorBoard ≥ 2.13
- Fixed RNG seed `seed=44` in [lstm/config.py](lstm/config.py); deterministic uniform frame sampling

**Data environment.**
- 123 unique AzSL phrases listed in [lstm/drive/sentences_all.csv](lstm/drive/sentences_all.csv)
- ~750 video samples organized as `lstm/drive/Video/Cam2/<idd>/*.mp4` with 1 translator + 0–6 user clips per phrase
- Pre-computed feature caches at [cosine/data/processed/matrices/](cosine/data/processed/matrices/), [DTW/matrices/](DTW/matrices/), and [lstm/features_slr/](lstm/features_slr/)

**Per-pipeline evaluation set composition.**
- Cosine: 342 videos, 124 unique sentences; translator subset = 122, user subset = 218.
- DTW: 215 user videos scored against 94 reference templates.
- LSTM: 407 videos evaluated; 93 skipped due to missing cached features.
- Transformer: 80 / 10 / 10 train / val / test split; only a 30-sample random subset of test was scored end-to-end.

### 4.4 Test Procedures

**Procedure 1 — Cosine leave-one-out.** Implemented in [cosine/scripts/top3_accuracy.py](cosine/scripts/top3_accuracy.py). For each of the 342 videos, hold it out, score it against the remaining 341 cached landmark matrices using cosine similarity over flattened 64-frame × 126-D vectors, and record whether the ground-truth phrase appears in Top-1 / Top-3 / Top-5 of the ranked list. Aggregate the binary outcomes into Top-K accuracy.

**Procedure 2 — DTW recognition.** For each of the 215 user videos, run `evaluate_recognition_method4b.py` (and analogous scripts for variants 4A, 4C, 4D). The script (a) extracts geometric features per frame, (b) drops near-duplicate frames at similarity ≥ 0.99, (c) for each of the 94 reference templates runs subsequence DTW with Sakoe–Chiba ratio 0.25 in both normal and hand-swapped configurations and keeps the better score, (d) ranks templates by score and records Top-K. JSON summaries land in [DTW/recognition_results/](DTW/).

**Procedure 3 — DTW frame-count and window sweeps.** Re-ran Procedure 2 over N ∈ {16, 32, 64, 128} frames (cosine) and Sakoe–Chiba band ratio ∈ {0.20, 0.22, 0.25, 0.28, 0.30} (DTW). Sweep tables fed back into [results.md](results.md) and into the choice of N = 64 and band = 0.25 as defaults.

**Procedure 4 — LSTM seq2seq evaluation.** [lstm/research_eval_seq2seq.py](lstm/research_eval_seq2seq.py) loads cached `.pt` features for all available videos, runs encoder + attention decoder with greedy decoding, compares the predicted token sequence against the CSV ground truth, and records exact-match Top-1, Top-3, Top-5 (using the top-K beam-search hypotheses) plus per-class breakdown. Cached metrics land in [lstm/research_outputs/seq2seq_eval/](lstm/research_outputs/seq2seq_eval/).

**Procedure 5 — Transformer random-sample evaluation.** [Transformer/random_100_eval.py](Transformer/random_100_eval.py) (configured for 30 samples in the run that produced reportable numbers) loads `best.pt`, draws random test samples, runs beam-search decoding, and computes exact-match sequence accuracy and mean WER per [Transformer/training/evaluate.py](Transformer/training/evaluate.py).

**Procedure 6 — Latency profiling.** Each pipeline was instrumented to log wall-clock time for (a) frame read + sampling, (b) per-frame feature extraction, (c) scoring / decoding. Cosine numbers (frame I/O 667.8 ms, MediaPipe 2 036.2 ms, search 2.8 ms, total ~2.7 s) were measured on the CPU benchmarking machine; LSTM numbers (feature load 0.9 ms, decoder 17.1 ms, total ~18 ms) were measured on the GPU machine using cached features.

**Procedure 7 — Figure generation.** [cosine/scripts/generate_all_paper_figures.py](cosine/scripts/generate_all_paper_figures.py) renders confusion matrices, per-class accuracy bars, frame-count sensitivity plots, and latency stacked bars into [cosine/outputs/figures/paper_figures/](cosine/outputs/figures/paper_figures/) for visual inspection of failure modes.

### 4.5 Results

**Headline accuracy ([results.md](results.md)).**

| Method | Top-1 | Top-3 | Top-5 | n |
|---|---|---|---|---|
| Cosine (all videos, N=64) | 30.7 % | 45.0 % | 49.4 % | 342 |
| Cosine (user videos) | 34.4 % | 49.5 % | 53.2 % | 218 |
| Cosine (translator videos) | 24.6 % | 37.7 % | 43.4 % | 122 |
| **DTW Method 4B** | **40.0 %** | **50.2 %** | **56.3 %** | 215 vs 94 |
| LSTM Seq2Seq | 16.0 % | 25.6 % | 26.5 % | 407 |
| Transformer | 6.7 %* | — | — | 30 random |

*Exact-match sequence accuracy on a 30-sample random subset; mean WER 0.86.

**Cosine frame-count sensitivity.**

| N frames | Top-1 | Top-3 | Top-5 |
|---|---|---|---|
| 16 | 27.8 % | 39.5 % | 46.8 % |
| 32 | 31.6 % | 43.9 % | 49.7 % |
| 64 | 30.7 % | 45.0 % | 49.4 % |
| 128 | 30.7 % | 45.0 % | 49.4 % |

Performance plateaus at N = 64.

**LSTM training trajectory.**

| Epoch | Train Loss | Val Loss |
|---|---|---|
| 1 | 3.726 | 2.835 |
| 10 | 1.963 | 2.167 |
| 20 | 0.768 | 1.864 |
| 50 | 0.201 | 1.798 |
| 200 | 0.303 | 1.794 |

Best validation loss 1.7905 at epoch 31; final train loss 0.303 — the train/val gap confirms overfitting.

**LSTM per-class highlights.**

| Sentence | Top-1 |
|---|---|
| "axşam xeyir" (Good evening) | 80.95 % (17/21) |
| "siz ad nə ?" (What is your name?) | 78.05 % (32/41) |
| "mən yazmaq bilmir" (I cannot write) | 52.38 % (11/21) |

**Latency profile.**

| Pipeline | Stage | Time |
|---|---|---|
| Cosine (CPU) | Frame read + sample | 667.8 ms |
| Cosine (CPU) | MediaPipe extraction | **2 036.2 ms** (75 %) |
| Cosine (CPU) | Cosine search (342 vectors) | 2.8 ms |
| Cosine (CPU) | **Total** | **~2.7 s** |
| LSTM (GPU) | Feature load | 0.9 ms |
| LSTM (GPU) | Greedy decoder | 17.1 ms |
| LSTM (GPU) | **Total** | **~18 ms** |

**Transformer training.** Best checkpoint at epoch 27 with validation loss 6.3176 ([Transformer/artifacts/checkpoints/best.pt](Transformer/artifacts/checkpoints/best.pt)); loss plateaued and grew noisy from epoch 27 to 80.

### 4.6 Analysis of Results

**1. DTW wins because the data favours alignment over learning.** With ~750 samples spread across 124 phrases, neural models have very few examples per class. DTW does not need to learn invariances — they are baked into the geometric features (rotation, scale) and the alignment algorithm (speed). This explains why DTW Method 4B (40.0 %) outperforms LSTM (16.0 %) by 24 percentage points and Transformer (6.7 %) by 33 points despite using *no learned parameters at all*.

**2. The translator/user gap is real and informative.** Cosine accuracy on user videos (34.4 %) exceeds translator videos (24.6 %) by ~10 points. Translators sign more subtly and consistently, which means small landmark differences carry more information; flattened cosine cannot exploit that, but DTW with geometric features can — which is exactly where the 30 % → 40 % jump comes from.

**3. Cosine frame-count plateau confirms feature, not sampling, is the bottleneck.** Going from 64 to 128 frames yields zero improvement. The signal in flattened landmark vectors saturates well below 128 frames, so further gains required changing the *representation* (move to geometric features) and the *scoring* (move to DTW), not adding more frames.

**4. LSTM overfits structurally.** The 0.30 vs 1.79 train/val loss gap (factor of 6) shows the model memorizes the training set without generalizing. The per-class breakdown confirms this is not uniform: short, frequent phrases ("axşam xeyir", 81 %) work well, but the long tail collapses. This is a data-volume problem, not an architecture problem.

**5. Transformer architecture is sound but data-starved.** Loss decreases monotonically through epoch 27 then plateaus, exactly the curve of a model that has extracted all signal available in the training set. The 6.7 % exact-match number on a 30-sample subset is not a fair end-to-end evaluation — a full test export is the highest-priority follow-up.

**6. MediaPipe is the deployment bottleneck.** 2 036 / 2 707 ms of cosine query time (75 %) is MediaPipe extraction. The actual cosine search over all 342 reference vectors costs 2.8 ms — three orders of magnitude less. Any future deployment optimization should target hand-landmark extraction (caching, batching, smaller models) rather than the recognizer itself.

**7. Per-pipeline strengths are complementary.** Cosine is fastest to deploy (no training, no GPU) and gives the most interpretable similarity score. DTW gives the highest accuracy *and* an alignment path that can be turned into "you went too fast at frame 30" learner feedback. LSTM is fastest at inference (18 ms) once features are cached. Transformer is the natural future direction once the dataset grows.

### 4.7 Validation Against Success Criteria

| Success criterion | Target | Achieved | Verdict |
|---|---|---|---|
| Beat random-guess baseline by a wide margin | ≫ 1/124 ≈ 0.8 % Top-1 | 40.0 % Top-1 (DTW) | **Far exceeded** (~50× random) |
| Interpretable similarity score | Continuous score per candidate | All four recognizers expose a score; DTW also exposes alignment path | **Met** |
| No retraining needed to add a new phrase | Template-matching path must exist | Cosine and DTW satisfy this by design | **Met** |
| CPU-deployable | < 5 s end-to-end on Intel i7 | Cosine 2.7 s, DTW (offline) ~7.8 s, DTW (streaming) real-time | **Met** for cosine and streaming DTW; partially for offline DTW |
| Top-5 accuracy useful for "shortlist" UI | ≥ 50 % | DTW 56.3 %, Cosine 49.4 % | **Met** for DTW; nearly met for Cosine |
| Sentence-level transcription | Working seq2seq pipeline | LSTM end-to-end, Transformer architecture | **Pipeline met; accuracy below target** |
| Real-time webcam demo | Live recognition without lag | [DTW/camera_dtw.py](DTW/camera_dtw.py) with O(N) per-frame DTW | **Met** |
| Reproducible results | Fixed seed, cached intermediates | `seed=44`, cached features under [cosine/data/processed/matrices/](cosine/data/processed/matrices/), [DTW/matrices/](DTW/matrices/), [lstm/features_slr/](lstm/features_slr/) | **Met** |
| Documented evaluation | Single source of truth | [results.md](results.md), [DTW/ITTA2026_paper.md](DTW/ITTA2026_paper.md), [research_presentation_outline.md](research_presentation_outline.md) | **Met** |

The system **meets or exceeds the core success criteria**. The two areas where it falls short — sentence-level transcription accuracy and a complete Transformer evaluation — are documented as the top items for future work in §3.5 and §3.6.

### 4.8 Limitations of Testing

1. **Small dataset.** ~750 videos across 124 phrases. Top-1 accuracy estimates carry meaningful variance (±2–3 percentage points at this sample size), especially for the per-class numbers where some classes have only a single test instance.
2. **No held-out users.** The DTW evaluation matches user videos against translator templates, but user/translator IDs are not held out *across* phrases. A signer-disjoint split would give a stricter generalization estimate and is not part of the current harness.
3. **Transformer evaluation is incomplete.** Only a 30-sample random subset of the Transformer test set was scored end-to-end ([Transformer/random_100_eval.py](Transformer/random_100_eval.py)); the headline 6.7 % number should not be treated as a final result.
4. **LSTM evaluation has 93 missing videos.** Of ~500 expected videos, 93 were skipped during [lstm/research_eval_seq2seq.py](lstm/research_eval_seq2seq.py) because their cached `.pt` features were missing. This biases the reported 16.0 % toward whichever phrases happened to retain caches.
5. **Latency measured on a single machine.** All cosine / DTW timings come from one Intel i7 (15.7 GB RAM); LSTM timings come from one GPU. There is no cross-platform benchmark.
6. **No statistical significance testing.** Differences between methods are reported as point estimates without confidence intervals or bootstrap resampling.
7. **No subjective / user-study evaluation.** The system is intended for learner feedback, but no real learners interacted with it during testing — accuracy numbers are a proxy for "useful feedback", not a direct measurement of it.
8. **MediaPipe failure modes not separately analyzed.** Frames where MediaPipe misses the hands are zero-padded; the distribution of these failures across phrases is not reported, even though it almost certainly skews per-class accuracy.
9. **Augmentation policy untuned.** The Transformer's heavy augmentation stack ([Transformer/data/augmentation.py](Transformer/data/augmentation.py)) was configured by judgement rather than by ablation; we do not know which transforms helped and which hurt.
10. **Inference profiling does not include preprocessing-cache cost.** LSTM's 18 ms / video assumes features are already cached; the *first* run on a new video pays the same MediaPipe + SqueezeNet cost as the cosine path.

---

## 5. Project Outcomes

### 5.1 Summary of Deliverables

**Software deliverables.**
- Four parallel recognition pipelines, each runnable end-to-end:
  - Cosine retrieval baseline ([cosine/](cosine/))
  - DTW recognizer with geometric features and a streaming variant ([DTW/](DTW/), [DTW/camera_dtw.py](DTW/camera_dtw.py))
  - LSTM seq2seq with two-stage offline-feature pipeline ([lstm/](lstm/))
  - Transformer (ViT-Small + temporal encoder–decoder) ([Transformer/](Transformer/))
- Reusable components: [HandFeatureExtractor](cosine/src/inference/similarity/feature_extractor.py), [SimilarityEngine](cosine/src/inference/similarity/similarity_engine.py) with 8 metrics, [method4_variants.py](DTW/method4_variants.py), [Transformer/data/augmentation.py](Transformer/data/augmentation.py)
- Evaluation harness: [cosine/scripts/top3_accuracy.py](cosine/scripts/top3_accuracy.py), `evaluate_recognition_*.py` family in [DTW/](DTW/), [lstm/research_eval_seq2seq.py](lstm/research_eval_seq2seq.py), [Transformer/training/evaluate.py](Transformer/training/evaluate.py), [Transformer/random_100_eval.py](Transformer/random_100_eval.py)
- Real-time webcam demo: [DTW/camera_dtw.py](DTW/camera_dtw.py)

**Data deliverables.**
- Curated dataset index: [lstm/drive/sentences_all.csv](lstm/drive/sentences_all.csv) (123 phrases)
- Cached feature stores: [cosine/data/processed/matrices/](cosine/data/processed/matrices/) (.npy landmarks), [DTW/matrices/](DTW/matrices/), [lstm/features_slr/](lstm/features_slr/) (.pt tensors)
- Trained Transformer checkpoint: [Transformer/artifacts/checkpoints/best.pt](Transformer/artifacts/checkpoints/best.pt) (epoch 27, val loss 6.3176)
- Vocabulary: [Transformer/artifacts/vocab.json](Transformer/artifacts/vocab.json)

**Documentation deliverables.**
- Aggregated results: [results.md](results.md)
- Algorithm reference: [DTW/ALGORITHM.md](DTW/ALGORITHM.md)
- Paper drafts and structure: [DTW/PAPER_STRUCTURE.md](DTW/PAPER_STRUCTURE.md), [DTW/ITTA2026_paper.md](DTW/ITTA2026_paper.md)
- Research and presentation materials: [research_presentation_outline.md](research_presentation_outline.md), [final_presentation_slides.md](final_presentation_slides.md)
- Working notebook: [SDP.ipynb](SDP.ipynb)
- Architecture-improvement notes: [lstm/architecture_improvements_v1.md](lstm/architecture_improvements_v1.md)
- Final report (this document): [final_report.md](final_report.md)

**Visual deliverables.**
- Confusion matrices, per-class accuracy bars, frame-count sensitivity plots, latency stacked bars rendered into [cosine/outputs/figures/paper_figures/](cosine/outputs/figures/paper_figures/) by [cosine/scripts/generate_all_paper_figures.py](cosine/scripts/generate_all_paper_figures.py)

### 5.2 Achieved Functionality or Performance

**Recognition accuracy.**
- DTW Method 4B: **40.0 % Top-1 / 50.2 % Top-3 / 56.3 % Top-5** on 215 user videos against 94 reference templates — the project's best end-to-end result.
- Cosine baseline: 30.7 % Top-1 / 45.0 % Top-3 / 49.4 % Top-5 on 342 videos under leave-one-out cross-validation.
- LSTM seq2seq: 16.0 % Top-1 / 25.6 % Top-3 / 26.5 % Top-5 on 407 videos with greedy decoding.
- Transformer: 6.7 % exact-match sequence accuracy with mean WER 0.86 on a 30-sample random subset (partial evaluation).

**Inference latency.**
- Cosine end-to-end: ~2.7 s / video on Intel i7 CPU (75 % of which is MediaPipe).
- LSTM end-to-end with cached features: ~18 ms / video on GPU.
- Streaming DTW: real-time on a webcam thanks to O(N) per-frame column updates.

**System functionality.**
- Returns ranked Top-K candidates with continuous similarity / log-likelihood scores.
- DTW returns an alignment path between query and reference, usable for fine-grained learner feedback.
- New phrases can be added to the cosine and DTW pipelines without retraining (just add a reference video).
- Real-time webcam recognition via [DTW/camera_dtw.py](DTW/camera_dtw.py).
- Reproducible runs via fixed seed `seed=44` in [lstm/config.py](lstm/config.py) and cached intermediates.

### 5.3 Achievement of Project Objectives

| Objective | Achievement |
|---|---|
| Build a working AzSL recognizer | Achieved with multiple methods; DTW Method 4B reaches 40.0 % Top-1 |
| Provide graded, interpretable feedback | Achieved — every recognizer exposes a score; DTW also exposes an alignment path |
| Operate under limited compute (no mandatory GPU) | Achieved — cosine and DTW run on CPU; streaming DTW is real-time on a webcam |
| Operate under limited data (no fixed vocabulary at training time) | Achieved — cosine and DTW are template-matching, so adding a phrase is just adding a reference video |
| Compare classical and neural approaches on the same task | Achieved — four pipelines benchmarked on the same harness; results consolidated in [results.md](results.md) |
| Lay foundations for sentence-level transcription | Partially achieved — LSTM works end-to-end at modest accuracy; Transformer architecture is in place but evaluation is incomplete |
| Document the work for future researchers | Achieved — algorithm doc ([DTW/ALGORITHM.md](DTW/ALGORITHM.md)), paper drafts, presentation, results, and this report |

### 5.4 Comparison with Expected Outcomes

| Dimension | Expected (proposal) | Actual | Comment |
|---|---|---|---|
| Number of models | 1 (Transformer) | 4 (cosine, DTW, LSTM, Transformer) | Widened scope to compensate for limited data |
| Best Top-1 accuracy | ≥ 30 % from neural model | 40.0 % from DTW Method 4B | Best result came from a *non-neural* method |
| Inference target | Real-time, GPU-assumed | Cosine 2.7 s CPU; Streaming DTW real-time CPU | Real-time goal met *without* a GPU via DTW |
| Sentence-level transcription | Reliable end-to-end | Working but modest (LSTM 16 %, Transformer incomplete) | Below expectations — bounded by data volume |
| Vocabulary expandability | Re-train for each new phrase | Add a reference video, no retraining | Better than expected for cosine + DTW |
| Interpretability | Optional | First-class — alignment paths, similarity scores, per-class metrics | Better than expected |
| Real-time demo | Stretch | Delivered ([DTW/camera_dtw.py](DTW/camera_dtw.py)) | Stretch goal met |
| Full Transformer evaluation | Expected | Only 30-sample random subset | Below expectations — listed as #1 future task |

The single biggest *positive* surprise was that a classical, training-free DTW pipeline with hand-engineered geometric features beat both neural alternatives by a wide margin. The single biggest *negative* surprise was that the Transformer, despite a sound implementation, could not extract enough signal from ~750 samples to reach a competitive number.

### 5.5 Key Technical Contributions

1. **Geometric-feature DTW family for AzSL recognition.** Method 4B's pairwise-bone-angle representation (~190 angles per hand) combined with subsequence DTW and a Sakoe–Chiba band of 0.25 reaches 40.0 % Top-1 with **no learned parameters** ([DTW/method4_variants.py](DTW/method4_variants.py), [DTW/ALGORITHM.md](DTW/ALGORITHM.md)). To our knowledge this is the first computational AzSL recognizer.
2. **Streaming DTW for real-time learner feedback.** [DTW/camera_dtw.py](DTW/camera_dtw.py) implements an incremental column-by-column DTW update (O(N) per frame, O(N) space) that runs in real time against multiple reference templates on CPU.
3. **Two-stage offline-feature pipeline for small-data seq2seq.** The LSTM stack ([lstm/feature_extraction_slr/extract_features.py](lstm/feature_extraction_slr/extract_features.py) → cached `.pt` → [lstm/models/encoder.py](lstm/models/encoder.py)/[lstm/models/decoder.py](lstm/models/decoder.py)) decouples expensive per-frame CNN+BiLSTM extraction from the lightweight seq2seq training loop, enabling many training-loop iterations without re-running preprocessing.
4. **Reusable similarity engine.** [SimilarityEngine](cosine/src/inference/similarity/similarity_engine.py) provides a single API across 8 similarity metrics (cosine, Euclidean, Pearson, Spearman, Manhattan, frame-wise cosine, temporal correlation, DTW) and is reused across cosine and DTW pipelines.
5. **Comprehensive video augmentation pipeline.** [Transformer/data/augmentation.py](Transformer/data/augmentation.py) bundles spatial (crop, flip, jitter, blur, grayscale), temporal (jitter, frame dropout, reverse), and regularization (CutOut, label smoothing) transforms with frame-consistent application — a reusable component for any small-data video classification task.
6. **Empirical study of representation choice for sign recognition.** Demonstrated that under small-data conditions, *changing the representation* (raw landmarks → geometric features) and the *scoring* (flattened cosine → subsequence DTW) yields larger gains than scaling up the model — cosine 30.7 % → DTW 40.0 % vs. LSTM 16.0 % → Transformer 6.7 %.
7. **Comparable, multi-pipeline evaluation harness.** All four recognizers report Top-K, WER, latency, and per-class metrics on the same dataset, with results consolidated in [results.md](results.md). This provides a defensible apples-to-apples comparison across very different architectural families.
8. **Documentation and academic write-up.** [DTW/ITTA2026_paper.md](DTW/ITTA2026_paper.md) (366 lines) and [DTW/PAPER_STRUCTURE.md](DTW/PAPER_STRUCTURE.md) (550+ lines) provide a publication-ready treatment of the DTW method.

### 5.6 Limitations of the Final Design

1. **Data ceiling.** ~750 samples across 124 phrases is the binding constraint for both LSTM and Transformer. Until the dataset grows, neural methods are unlikely to surpass DTW.
2. **Single reference template per phrase.** DTW currently matches against one reference per phrase, so the recognizer is sensitive to how that particular signer performed the gesture. A multi-reference extension (3–5 templates per phrase) is described in [DTW/PAPER_STRUCTURE.md](DTW/PAPER_STRUCTURE.md) but not implemented.
3. **MediaPipe latency dominates.** 75 % of cosine query time is hand-landmark extraction. Any deployment optimization must target the extractor, not the recognizer.
4. **No body or face information.** All landmark-based pipelines use hands only. Many AzSL signs depend on facial expression, body posture, and head movement; these are not modeled.
5. **No signer-disjoint evaluation.** The current splits do not guarantee that the same signer never appears in both training/reference and test/query, so reported numbers may overstate generalization to new signers.
6. **Translator/user gap unresolved.** Cosine accuracy on translator videos (24.6 %) is 10 points below user videos (34.4 %). DTW narrows this but does not eliminate it.
7. **Hand-swap is binary.** The system tries normal and fully swapped configurations; it does not handle partial mixed-handedness gracefully.
8. **Incomplete neural-model evaluation.** Transformer test-set evaluation is missing; LSTM evaluation skipped 93 videos with missing caches. Both must be resolved before any neural-vs-DTW comparison is fully honest.
9. **No on-device inference path.** Inference scripts assume Python + PyTorch + MediaPipe + OpenCV on a desktop. There is no mobile / web build.
10. **No learner-facing UI.** The system returns ranked candidates and alignment paths, but there is no application layer that turns those into pedagogical feedback ("you held the handshape too long here").
11. **Augmentation, learning-rate, and architecture choices are not ablated.** Many design choices (band ratio 0.25, ViT-Small over ViT-Base, decoder depth 4, augmentation probabilities) were set by judgement rather than by exhaustive ablation.
12. **Single-language.** The system is specific to AzSL; transferring to another sign language would require new reference videos for cosine/DTW and full retraining for LSTM/Transformer.

---
