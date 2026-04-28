# Supporting Design Documentation

This document consolidates the design references that govern the four AzSL recognition pipelines (cosine, DTW, LSTM Seq2Seq, Transformer): the **DTW algorithm reference** (the formal definition of the alignment used by the project's best recognizer) and a snapshot of every **pipeline configuration file**. Source files are linked directly so the in-repository version remains authoritative.

---

## 1. DTW Algorithm Reference

The complete reference is [DTW/ALGORITHM.md](../DTW/ALGORITHM.md) (~390 lines). The essential mathematical definitions, taken from that document, are summarised below; section numbers match the source.

### 1.1 Feature extraction and preprocessing (§2–§3)

Every frame is passed through MediaPipe Hands to produce a 126-D vector (21 landmarks × 3 coords × 2 hands). For Method 4B this is replaced by the ~190-D pairwise-bone-angle descriptor per hand (380-D total); see [DTW/method4_variants.py](../DTW/method4_variants.py). Frames without any detected hands are dropped. Consecutive near-duplicate frames are then removed using the normalized-Euclidean similarity

```
sim(a, b) = 1 − ‖â − b̂‖² / 2,   â = a / ‖a‖
```

with threshold 0.99. This step removes ~50–60 % of frames while preserving all meaningful motion.

### 1.2 Cost matrix (§4.1)

Given reference frames `R[0..N)` and user frames `U[0..M)`,

```
C[n, m] = euclidean(R[n], U[m])
```

### 1.3 Accumulated cost matrix — subsequence DTW (§4.2)

The initialization differs from global DTW: the first column consumes the entire reference but the first row is free in the user stream.

```
D[n, 0] = Σ_{k=0..n} C[k, 0]    # consume entire reference
D[0, m] = C[0, m]               # free start in user stream
D[n, m] = C[n, m] + min( D[n−1, m−1],  # diagonal — match
                         D[n−1, m],    # vertical — reference skip
                         D[n,   m−1] ) # horizontal — user skip
```

### 1.4 Optimal end point and backtracking (§4.3–§4.4)

The optimal end column is `b* = argmin_m D[N−1, m]`; backtracking follows the minimum-predecessor rule from `(N−1, b*)` to row 0, yielding the alignment path used to highlight where the learner deviated.

### 1.5 Sakoe–Chiba band

Cells more than `floor(band_ratio × max(N, M))` off the main diagonal are forbidden. The project default is `band_ratio = 0.25`, identified by a 0.20–0.30 sweep. The band reduces complexity from O(N·M) to O(N·M·band) and prevents pathological alignments.

### 1.6 Streaming DTW (§5)

The accumulated cost matrix is computed one column at a time, so only two columns (`D_prev`, `D_curr`) need to be kept in memory.

```
For each incoming user frame u_m:
    1. feat = MediaPipe(u_m)        (skip empty / redundant frames)
    2. C[n, m] = euclidean(R[n], feat)  for all n
    3. D_curr[0] = C[0, m]                              # free start
       D_curr[n] = C[n, m] + min(D_prev[n−1], D_prev[n], D_curr[n−1])
    4. running score = D_curr[N−1] / N
    5. D_prev ← D_curr
```

Per-frame cost is O(N), per-frame memory is O(N), and the offline and streaming modes produce identical alignment scores when the same frames are seen.

### 1.7 Hand-swap handling

For every (query, reference) pair the alignment is computed twice — once with the natural left/right hand assignment, once with the assignment swapped — and the lower-cost result is kept. This is the single largest source of robustness gain reported in [results.md](../results.md) for learner-recorded videos.

### 1.8 Scoring (§9)

The final per-pair score combines the alignment cost with a cosine similarity computed along the optimised path; the per-phrase ranking is then `min` over reference templates. The full derivation, including the bidirectional local-window refinement from §6 of the source document, is preserved in [DTW/ALGORITHM.md](../DTW/ALGORITHM.md).

---

## 2. Pipeline Configuration Files

Each of the four pipelines has a single `config` source of truth. The snapshots below are taken from those files; any future change should be made *in the file* and re-summarised here, not the other way around.

### 2.1 Cosine pipeline — [cosine/scripts/config.py](../cosine/scripts/config.py)

| Group | Key | Value |
|---|---|---|
| Environment | `env` | `'CeDAR'` |
| Environment | `seed` | `44` |
| Environment | `device` | `cuda:0` if available, else `cpu` |
| Data | `max_frames` | `64` |
| Data | `BATCH_SIZE` | `8` |
| Data | `video_processing_tool` | `'TorchVision'` |
| Data | `camera_source` | `'Cam2'` |
| Paths | `train_csv_path` | `<drive>/sentences_all.csv` |
| Paths | `video_folder` | `<drive>/Video` |
| Similarity | `n_frames` | `64` |
| Similarity | `max_hands` | `2` |
| Similarity | `min_detection_confidence` | `0.5` |
| Similarity | `min_tracking_confidence` | `0.5` |
| Similarity | `normalize_landmarks` | `True` |
| Similarity | `similarity_method` | `'cosine'` (default; 7 alternates available) |
| Similarity | `filter_hand_frames` | `True` |
| Similarity | `feature_dim` | `126` (21 × 3 × 2) |

### 2.2 DTW pipeline — [DTW/config.py](../DTW/config.py)

| Group | Key | Value |
|---|---|---|
| Environment | `env` | `'CeDAR'` |
| Environment | `seed` | `44` |
| Environment | `device` | `cuda:0` if available, else `cpu` |
| Data | `max_frames` | `64` |
| Data | `BATCH_SIZE` | `8` |
| Data | `video_processing_tool` | `'TorchVision'` |
| Similarity | `n_frames` | `64` |
| Similarity | `max_hands` | `2` |
| Similarity | `min_detection_confidence` | `0.5` |
| Similarity | `normalize_landmarks` | `True` |
| Similarity | `similarity_method` | `'cosine'` (used as the per-frame cost inside DTW) |
| Similarity | `feature_dim` | `126` (per-frame landmark vector) |
| Method 4B | feature dimension | `380-D` (≈190 pairwise bone angles × 2 hands; see [DTW/method4_variants.py](../DTW/method4_variants.py)) |
| Method 4B | Sakoe–Chiba `band_ratio` | `0.25` |
| Method 4B | hand-swap | enabled (normal + swapped, take min cost) |
| Method 4B | redundant-frame threshold | `0.99` |

### 2.3 LSTM pipeline — [lstm/config.py](../lstm/config.py)

| Group | Key | Value |
|---|---|---|
| Environment | `env` | `'CeDAR'` |
| Environment | `seed` | `44` |
| Environment | `device` | `cuda:0` if available, else `cpu` |
| Data | `max_frames` | `64` |
| Data | `max_words_in_sentence` | `10` |
| Data | `BATCH_SIZE` | `64` |
| Data | `video_processing_tool` | `'TorchVision'` |
| Paths | `train_csv_path` | `<drive>/sentences_all.csv` |
| Paths | `encoder_model_path` | `<drive>/jamal/encoder.model` |
| Paths | `decoder_model_path` | `<drive>/jamal/decoder.model` |
| Stage A | backbone | SqueezeNet 1.1 (ImageNet, frozen) |
| Stage A | per-frame feature | `86 528-D` (`features.12.cat` flattened) |
| Stage A | BiLSTM hidden | `256` per direction → `(T, 512)` cached `.pt` |
| Stage B | encoder | BiLSTM, hidden `512`, bidirectional |
| Stage B | decoder | LSTM, hidden `512`, Bahdanau attention (τ = 2.0, dropout 0.1) |
| Training | optimiser | Adam |
| Training | loss | CrossEntropy (ignore `<pad>`) |
| Training | epochs / patience | 200 / early stop |
| Inference | beam width | `5` |

### 2.4 Transformer pipeline — [Transformer/config/config.py](../Transformer/config/config.py)

| Group | Key | Value |
|---|---|---|
| Paths | `csv_path` | `lstm/drive/sentences_all.csv` |
| Paths | `videos_root` | `lstm/drive/Video/Cam2` |
| Paths | `vocab_path` | `artifacts/vocab.json` |
| Paths | `checkpoint_dir` | `artifacts/checkpoints` |
| Video | `num_frames` | `16` |
| Video | `frame_size` | `(224, 224)` |
| Video | `mean` / `std` | ImageNet stats |
| Augment (spatial) | `random_crop` / `random_crop_scale` | `True` / `(0.7, 1.0)` |
| Augment (spatial) | `horizontal_flip_p` | `0.5` |
| Augment (spatial) | color jitter (brightness/contrast/sat/hue) | `0.3 / 0.3 / 0.2 / 0.1`, `p = 0.8` |
| Augment (spatial) | `grayscale_p` / `gaussian_blur_p` | `0.1` / `0.3` |
| Augment (temporal) | `temporal_jitter` | `True` |
| Augment (temporal) | `frame_drop_p` | `0.1` |
| Augment (temporal) | `temporal_reverse_p` | `0.1` |
| Augment (regularisation) | `cutout_p` / `cutout_size` | `0.3` / `0.15` |
| Data split | train / val / test | `0.80 / 0.10 / 0.10`, seed `42` |
| Data | `batch_size` / `num_workers` | `8` / `4` |
| Vocab | special tokens | `<pad>`, `<sos>`, `<eos>`, `<unk>` |
| Vocab | `min_freq` | `1` |
| ViT backbone | `model_name` | `vit_small_patch16_224` (timm, pretrained) |
| ViT backbone | `freeze_backbone` / `freeze_epochs` | `False` / `5` |
| ViT backbone | `output_dim` | `384` |
| Encoder | `d_model` / `nhead` / `layers` / `FFN` / `dropout` | `256` / `8` / `4` / `1024` / `0.1` |
| Decoder | `d_model` / `nhead` / `layers` / `FFN` / `dropout` | `256` / `8` / `4` / `1024` / `0.1` |
| Decoder | `max_target_len` / `label_smoothing` | `30` / `0.1` |
| Training | `epochs` | `200` |
| Training | `lr_backbone` / `lr_head` | `1e-5` / `1e-4` |
| Training | `weight_decay` / `warmup_steps` / `clip_grad_norm` | `1e-4` / `300` / `1.0` |
| Training | `save_top_k` / `mixed_precision` | `3` / `True` |

---

## 3. Other Supporting Design Documents

| Document | Purpose |
|---|---|
| [DTW/ALGORITHM.md](../DTW/ALGORITHM.md) | Full DTW algorithm specification (subsequence + streaming, complexity, scoring) |
| [DTW/PAPER_STRUCTURE.md](../DTW/PAPER_STRUCTURE.md) | DTW paper outline, including the planned multi-reference-template extension |
| [DTW/ITTA2026_paper.md](../DTW/ITTA2026_paper.md) | Publication-ready DTW write-up |
| [confrence.md](../confrence.md) | Springer LNCS conference paper (cosine + Seq2Seq comparative study) |
| [cosine/README.md](../cosine/README.md) | Cosine pipeline operating manual |
| [lstm/architecture_improvements_v1.md](../lstm/architecture_improvements_v1.md) | LSTM design notes and deferred improvements |
| [lstm/low_accuracy_potential_reasons_v1.md](../lstm/low_accuracy_potential_reasons_v1.md) | LSTM error analysis |
| [results.md](../results.md) | Cross-pipeline results consolidation |
| [final_presentation_slides.md](../final_presentation_slides.md) | Final presentation deck source |
| [research_presentation_outline.md](../research_presentation_outline.md) | Research presentation outline |
| [SDP.ipynb](../SDP.ipynb) | Working notebook with end-to-end runs |
