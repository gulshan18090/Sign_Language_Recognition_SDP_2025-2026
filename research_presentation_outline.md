# Recognition of Complex Sign Language Gestures with High-Variety Data

## Research Presentation Content

---

## 1. Goals and Sub-goals

### Main Problem

Azerbaijani Sign Language (AzSL) has no prior computational recognition or assessment system. Existing deep learning sign language recognition systems usually require large labeled corpora, GPU hardware, and fixed vocabularies, which are difficult to guarantee in the current AzSL setting. In practice, learners also need graded and interpretable feedback rather than only binary correct/incorrect outputs.

### Main Goal

Build a training-free or low-dependency sign language recognition and assessment system for AzSL that can compare learner videos against reference examples, provide continuous similarity scoring, and run on CPU hardware.

### Sub-goals

- SG1: Design a feature extraction pipeline from raw video to hand-pose representation.
- SG2: Build temporal modeling or alignment methods that can handle speed, duration, and execution differences between signers.
- SG3: Improve invariance to scale, pose, and rotation where possible.
- SG4: Support real-time or near-real-time assessment without GPU dependence.
- SG5: Establish quantitative recognition baselines for AzSL.
- SG6: Analyze failure patterns and method tradeoffs to guide future improvement.

---

## 2. Tasks and Sub-tasks

### Data Collection and Preparation

- Organize labeled sentence/gloss data from CSV files.
- Link labels to either raw video folders or cached feature files.
- Build vocabularies for sequence-generation pipelines.
- Collect or process translator and user recordings for phrase-level recognition experiments.

### Shared Preprocessing Work

- Detect hand landmarks or hand-containing frames.
- Remove frames without useful hand information.
- Normalize or resample videos into fixed-length or aligned temporal sequences.
- Cache extracted features for faster repeated evaluation when needed.

### Method Development

- Implement a recurrent sequence-generation pipeline using offline video features.
- Implement a transformer-based sequence-generation pipeline using sampled raw frames.
- Implement a similarity-based template matching pipeline using landmark matrices and temporal alignment/similarity scoring.
- Compare alternative feature sets, matching metrics, and evaluation settings.

### Evaluation

- Measure Top-1, Top-3, and Top-5 accuracy where ranking-based retrieval is used.
- Measure WER, sequence accuracy, and BLEU-4 where sequence generation is used.
- Export confusion summaries, latency benchmarks, and per-class behavior where available.
- Compare methods using only saved outputs and repository evidence.

---

## 3. Core Idea of the Solution

### Central Concept

The project explores three different ways to recognize complex sign language gestures from high-variation video data:

- retrieval-based similarity matching,
- recurrent sequence generation on cached features,
- transformer-based sequence generation directly from video frames.

Instead of relying on a single modeling philosophy, the system compares lightweight matching and richer neural sequence models to determine which strategy works best under current AzSL data conditions.

### Key Design Principles

- Use compact hand-centered representations where possible.
- Preserve temporal structure rather than treating videos as unordered images.
- Compare methods that trade off simplicity, accuracy, and computational cost differently.
- Keep the pipeline modular so feature extraction, modeling, and evaluation can be changed independently.

### How the Four Pipelines Differ

| Aspect | LSTM Pipeline | Transformer Pipeline | Cosine Similarity Pipeline | Method 4: DTW Template Matching |
|---|---|---|---|---|
| Core idea | Seq2seq generation from cached temporal features | Seq2seq generation from sampled raw frames | Nearest-neighbor retrieval by feature similarity | DTW-aligned nearest-neighbor retrieval with invariant descriptors |
| Input form | Pre-extracted `.pt` feature sequences | Raw video frames | Landmark matrices (`.npy`) | Landmark sequences (raw coords or geometric features) |
| Temporal modeling | Bidirectional LSTM encoder + attention decoder | Transformer encoder-decoder | None (or optional simple DTW in engine) | Global DTW + optional streaming subsequence DTW |
| Output type | Predicted sentence/gloss sequence | Predicted gloss sequence | Ranked matching phrase labels | Ranked matching phrase labels + continuous similarity score |
| Training needed | Yes | Yes | No supervised training for matching | No supervised training for matching |

---

## 4. Method Analysis

### Method 1: LSTM Pipeline

#### Pipeline Overview

- Stage 1 extracts offline features from videos in `drive/Video/Cam2`.
- Frames with detected hands are kept and resized to `224x224`.
- SqueezeNet features are passed through a bidirectional LSTM encoder and saved as `.pt` tensors.
- Training then uses an attention-based encoder-decoder seq2seq model on those cached features.
- Inference supports greedy decoding and beam search.

#### Key Implementation Details

- Configured maximum sequence length: `64` frames.
- Batch size: `64`.
- Seed: `44`.
- Feature extraction output dimension: `512` per frame.
- Decoder includes attention dropout (`0.1`) and temperature scaling (`2.0`).
- Long cached feature sequences are uniformly downsampled to `64` frames.
- The research evaluation uses beam search and Top-K reporting.
- Evaluation can skip videos whose cached feature file is missing.

#### Results

From `lstm/research_outputs/seq2seq_eval/seq2seq_eval_metrics.json`:

- Evaluated videos: `407`
- Missing cached features skipped: `93`
- Top-1 accuracy: `15.97%`
- Top-3 accuracy: `25.55%`
- Top-5 accuracy: `26.54%`

Latency statistics:

- Feature stage mean: `0.0009 s`
- Decode stage mean: `0.0171 s`
- Total mean: `0.0180 s`

Additional saved observations:

- Training history improves from validation loss `2.8352` at epoch 1 to best saved value `1.7905`.
- The saved per-class file contains `22` evaluated labels.
- Frequent saved confusion pairs include:
- `mənim anam yazmaq-oxumaq bilmir -> mən soyuq` (`19`)
- `salam -> mən soyuq` (`18`)
- `siz futbol oynamaq ? -> axşam xeyir` (`16`)

#### Strengths

- Fast inference after feature caching.
- Modular two-stage design.
- Attention and beam search allow sentence-level decoding rather than pure label retrieval.
- Strong evaluation export support in the repository.

#### Weaknesses

- Lowest saved accuracy among the three implemented approaches with usable result artifacts.
- Depends on cached feature availability; missing caches affected evaluation coverage.
- Representation learning is not end-to-end because feature extraction is separated from decoder training.

---

### Method 2: Transformer Pipeline

#### Pipeline Overview

- Reads labels from `lstm/drive/sentences_all.csv`.
- Builds a word-level vocabulary with `<pad>`, `<sos>`, `<eos>`, and `<unk>`.
- Loads raw videos from `lstm/drive/Video/Cam2`.
- Uniformly samples `16` frames and applies video augmentation during training.
- Uses a ViT frame encoder, a temporal transformer encoder, and a transformer decoder for autoregressive gloss generation.

#### Key Implementation Details

- Train/validation/test split: `80% / 10% / 10%`
- Seed: `42`
- Frame size: `224x224`
- Number of sampled frames: `16`
- ViT backbone: `vit_small_patch16_224`
- ViT output dimension: `384`
- Temporal encoder: `d_model=256`, `8` heads, `4` layers
- Decoder: `d_model=256`, `8` heads, `4` layers
- Training includes:
- separate learning rates for backbone and decoder/head,
- warmup + cosine-style scheduling,
- optional backbone freezing,
- AMP mixed precision,
- gradient clipping,
- checkpointing and TensorBoard logging.

#### Results

Saved checkpoint evidence:

- Best checkpoint file exists at `artifacts/checkpoints/best.pt`
- Saved checkpoint epoch: `27`
- Saved validation loss: `6.3176`

Saved evaluation evidence:

- Full held-out summary file was not found.
- Available saved result files are:
- `Transformer/random_eval_results_recent10.csv`
- `Transformer/random_eval_results_recent30.csv`

From the saved `recent30` random-sample CSV:

- Sample size: `30`
- Exact matches: `2`
- Sequence accuracy on sample: `6.67%`
- Mean WER on sample: `0.8551`

The evaluation code itself supports:

- WER
- Sequence accuracy
- BLEU-4

But no full exported test-set summary was found in the repository.

#### Strengths

- Most modern architecture in the repository.
- End-to-end frame-based modeling.
- Includes augmentation, transformer temporal reasoning, beam search, and standard text-generation metrics.
- Checkpoint history shows major validation-loss reduction before the best checkpoint.

#### Weaknesses

- Result artifacts are incomplete compared with the cosine and LSTM pipelines.
- The saved random-sample outputs indicate weak performance in the available run.
- The project note and implementation are not fully aligned: one describes `.pt` feature inputs, while the actual code uses raw video frames.

---

### Method 3: Cosine Similarity Pipeline

#### Pipeline Overview

- Read raw videos.
- Extract frames and optionally keep only frames with detected hands.
- Convert each frame into MediaPipe hand landmarks.
- Build a per-video feature matrix and compare query videos against reference matrices.
- Rank candidate phrases by similarity score.

#### Key Implementation Details

- Up to `2` hands tracked per frame.
- `21` landmarks x `3` coordinates x `2` hands = `126`-D raw landmark feature vector.
- Landmark normalization centers coordinates around the wrist and scales by maximum hand spread.
- Default saved evaluation uses cosine similarity on flattened matrices.
- The similarity engine also supports Euclidean, Manhattan, Pearson, Spearman, frame-wise cosine, temporal correlation, and DTW.
- Leave-one-out evaluation ranks label matches after excluding the query sample itself.

#### Results

From `cosine/outputs/figures/paper_figures/accuracy_results.txt`:

- Videos evaluated: `342`
- Unique sentences: `124`
- Top-1 accuracy: `30.70%` (`105/342`)
- Top-3 accuracy: `45.03%` (`154/342`)
- Top-5 accuracy: `49.42%` (`169/342`)

From `cosine/outputs/figures/paper_figures/section4_stats.txt`:

- Translator videos: `122`
- User videos: `218`
- Total duration: `2222.6 s` (`37.0 min`)
- Average duration/video: `6.50 s`
- Average frames/video: `199.6`

Accuracy split:

- Translator Top-1 / Top-3 / Top-5: `24.6% / 37.7% / 43.4%`
- User Top-1 / Top-3 / Top-5: `34.4% / 49.5% / 53.2%`

Latency benchmark:

- Frame read + sample: `667.8 ms`
- MediaPipe extraction: `2036.2 ms`
- Cosine search across `342` vectors: `2.8 ms`
- Total per query: `2706.8 ms`

Accuracy vs sampled frame count:

- `N=16`: Top-1 `27.78%`, Top-3 `39.47%`, Top-5 `46.78%`
- `N=32`: Top-1 `31.58%`, Top-3 `43.86%`, Top-5 `49.71%`
- `N=64`: Top-1 `30.70%`, Top-3 `45.03%`, Top-5 `49.42%`
- `N=128`: Top-1 `30.70%`, Top-3 `45.03%`, Top-5 `49.42%`

#### Strengths

- Best saved Top-1, Top-3, and Top-5 accuracy among the implemented methods.
- Transparent and training-free recognition logic.
- Very fast retrieval stage after features are extracted.
- Includes strong supporting analysis: split by video type, latency, confusion analysis, and frame-count study.

#### Weaknesses

- End-to-end latency is dominated by preprocessing rather than similarity search.
- Retrieval quality depends strongly on dataset coverage and near-neighbor quality.
- Some processed folder names appear duplicated or inconsistent in the saved confusion summary.
- Retrieval does not generate novel sequences; it matches against known stored examples.

---

### Method 4: DTW-Based Template Matching (Alignment + Invariance)

#### Motivation

The cosine baseline compares sequences with limited temporal alignment and limited invariance. Method 4 upgrades the retrieval pipeline by explicitly aligning sequences with Dynamic Time Warping (DTW) and by using feature representations designed to be rotation/scale invariant.

#### Shared Preprocessing (Same as Cosine Pipeline)

- MediaPipe extraction per frame: `21` landmarks x `3` coordinates x `2` hands = `126`-D raw feature vector.
- Hand-only filtering: discard frames with no detected hands.
- Redundancy removal: drop consecutive near-duplicate frames to reduce DTW path distortion and latency.

#### Method 4A: Cosine-Consistent DTW on Coordinates

- Input: per-frame `126`-D landmark vectors (optionally normalized to unit length).
- Alignment: global DTW with Sakoe–Chiba band constraint (window ratio ~`0.30`).
- Cost: cosine distance `1 - dot(ref_i, user_j)`.
- Score: mean cosine similarity over aligned frame pairs.

#### Method 4B: Rotation-Invariant Geometric DTW (Angles + Relative Lengths)

- Input: geometric descriptor per hand:
	- `190` pairwise bone angles (normalized)
	- `20` relative bone lengths (length / sum of lengths)
	- Total: `210` per hand x `2` hands = `420`-D per frame
- Alignment: global DTW with Sakoe–Chiba band constraint (window ratio ~`0.25`).
- Cost: mean L1 distance `mean(|ref_i - user_j|)`.
- Score: `1 - L1_mean` averaged over aligned pairs.
- Handedness handling: global hand swap (evaluate original + swapped, keep best score).

#### Method 4C: Real-Time Streaming DTW (Subsequence DTW)

- Precompute and cache reference sequences for all phrases.
- For each incoming frame, update one DTW column per reference (O(reference_length) per reference).
- Provides per-frame similarity and final ranked results at session end.
- Current practical path: start with Method 4A features for real-time, then integrate Method 4B features.

---

## 5. Challenges and Learnings

### Main Difficulties

- High signer variability across user and translator videos.
- Sentence-level gestures create strong inter-class overlap.
- Speed, timing, and pause patterns differ across signers.
- Hand detection quality affects all methods because each pipeline depends on usable hand-centric input.
- Feature availability and artifact completeness differ by method.

### Unexpected Findings

- The simplest retrieval-based method outperformed both neural sequence-generation pipelines in saved repository metrics.
- For the cosine pipeline, increasing sampled frames beyond `32-64` did not materially improve Top-1 accuracy.
- The cosine split shows better performance on user videos than translator videos in the saved run.
- The transformer code is architecturally strong, but the saved repository evidence is weaker because a full held-out metrics export was not found.

### Key Insights

- Under current data conditions, robust similarity-based matching is a stronger practical baseline than sentence-generation models.
- Compact hand-centered features remain highly competitive when labeled data is limited.
- Strong evaluation artifacts matter: the cosine and LSTM pipelines are easier to analyze because they export clearer metrics.
- The transformer pipeline likely needs more disciplined experiment tracking before it can be judged fairly against the others.

---

## 6. Results and Conclusions

### Results Summary Across the Four Implemented Methods

| Method | Input Representation | Evaluation Evidence Found | Main Result |
|---|---|---|---|
| LSTM Pipeline | Cached `.pt` temporal features | Full saved Top-K metrics + latency | Top-1 `15.97%`, Top-3 `25.55%`, Top-5 `26.54%` |
| Transformer Pipeline | Raw sampled frames | Random-sample CSVs + checkpoint metadata | `6.67%` exact-match rate on saved 30-sample CSV, mean WER `0.8551` |
| Cosine Similarity Pipeline | Landmark matrices | Full saved Top-K metrics + latency + splits | Top-1 `30.70%`, Top-3 `45.03%`, Top-5 `49.42%` |
| Method 4 (DTW Template Matching) | Aligned landmark sequences (coords or geometric) | Presentation/evaluation results (215 user vs 94 references) | Best variant (Geometric DTW): Top-1 `40.00%`, Top-3 `50.23%`, Top-5 `56.28%` |

### Performance Comparison

- Best overall method when including Method 4: Geometric DTW (Method 4B).
- LSTM pipeline is faster at decode time than cosine once cached features exist, but its saved recognition accuracy is much lower.
- Transformer pipeline cannot be compared as rigorously because a full exported held-out summary was not found.

### Why Method 4 Beats Basic Cosine

- DTW alignment explicitly handles signer speed and duration variation instead of forcing frame-to-frame matching.
- Geometric descriptors (angles + relative lengths) add rotation + scale invariance.
- Hand swap handling reduces handedness/mirroring errors.
- Metric consistency (cosine-cosine or L1-L1) aligns DTW path optimization with the final similarity score.

### Key Conclusions

- The project establishes an initial computational baseline for AzSL recognition under data variability.
- With DTW alignment and geometric invariance (Method 4), retrieval-based similarity becomes a stronger recognizer than basic cosine matching.
- Sequence-generation approaches are promising architecturally but do not yet outperform the simpler similarity baseline in saved outputs.
- Better experiment logging and fuller evaluation exports are essential for future neural-model comparison.

---

## 7. Future Improvements

### Most Promising Current Direction

Method 4 (DTW template matching), especially the geometric DTW variant, should be treated as the reference system for future improvements.

### Immediate Improvements

- Clean duplicated or inconsistent folder/label names in the cosine processed data.
- Cache preprocessing outputs more aggressively to reduce end-to-end latency.
- Integrate geometric DTW features into the real-time streaming DTW engine.
- Add simple score fusion between Method 4A (coordinate cosine-DTW) and Method 4B (geometric DTW) since they can make complementary errors.
- Regenerate missing LSTM cached features so evaluation covers more videos.
- Save full transformer test-set metrics after each training run.

### Medium-Term Improvements

- Expand multi-reference template sets (3–5 translator references per phrase) to reduce score compression and improve ranking stability.
- Normalize / reweight feature components (e.g., angles vs lengths) to improve DTW alignment quality.
- Improve LSTM and transformer training data coverage and class balance.
- Use multi-reference matching where more than one example per phrase is available.

### Long-Term Directions

- Expand the AzSL dataset with more phrases and more signer diversity.
- Add body and facial information where phrase meaning depends on more than hand shape.
- Build more informative learner feedback using alignment paths, not just final labels.
- Revisit transformer and seq2seq models after stronger data curation and evaluation discipline.

---

## Appendix: Project Technical Details

### Repository-Backed Technology Stack

- Python
- PyTorch
- MediaPipe
- OpenCV
- NumPy / SciPy
- TensorBoard logging for transformer training

### Repository-Backed Dataset Signals

- Cosine saved evaluation: `342` videos, `124` unique sentences
- Cosine split: `122` translator videos, `218` user videos, `2` unknown-type videos
- LSTM saved evaluation: `407` evaluated videos, `93` skipped for missing cached features
- Transformer project note mentions `750` CSV rows and `16,803` samples, but the repository does not include a full exported evaluation summary using those counts

### Feature Representations Summary

| Pipeline | Feature Form | Dimensionality / Structure |
|---|---|---|
| LSTM | Cached temporal feature vectors | `512` per frame after offline extraction |
| Transformer | Raw RGB video frames | `16` sampled frames, ViT feature output `384` |
| Cosine | Hand landmark matrices | `126` values per frame (`21 x 3 x 2`) |
