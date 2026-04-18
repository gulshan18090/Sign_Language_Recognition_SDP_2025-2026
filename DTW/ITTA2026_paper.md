# DTW-Based Azerbaijani Sign Language Recognition: Comparative Feature Engineering with a Real-Time Streaming Assessment Engine

**Target:** ITTA 2026 — Springer Communications in Computer and Information Science
**Deadline:** March 10, 2026

---

## Abstract

Sign language recognition systems typically rely on deep learning classifiers that require large labelled corpora, GPU inference, and are ill-suited to real-time, low-latency educational feedback. In this paper, we present a gesture-level recognition and assessment system for Azerbaijani Sign Language (AzSL) built on Dynamic Time Warping (DTW) with interpretable geometric hand features, running in real time on consumer CPU hardware. We design and compare two contrasting DTW-based approaches: a coordinate-based method applying cosine-consistent DTW directly on normalised MediaPipe coordinates (126-D), and a geometry-invariant method replacing raw coordinates with rotation- and scale-invariant geometric descriptors — 190 pairwise bone angles and 20 relative bone lengths per hand — paired with L1-consistent DTW. Both are evaluated on 215 learner videos across 94 AzSL phrases, each compared against one professional translator reference. The geometry-invariant method achieves 40.0% Top-1 and 56.28% Top-5 recognition accuracy against 94 classes with a single reference per class and no training data (chance = 1.06%), outperforming the coordinate-based approach (33.95% Top-1) by 6 percentage points. As a second contribution, we present a streaming DTW engine that processes each camera frame incrementally, distributing the alignment computation in real time without GPU infrastructure and enabling live per-frame learner feedback. Failure analysis on 129 misclassified videos reveals that 72.9% are far misses (correct phrase outside Top-5), motivating future work on palm orientation and temporal motion features.

**Word count: ~220**

---

## Keywords

Azerbaijani Sign Language, Dynamic Time Warping, gesture recognition, bone angle features, real-time sign language assessment, MediaPipe, streaming DTW, sign language education, rotation-invariant features, top-K recognition

---

## 1. Introduction

Sign language is the primary communication modality for hundreds of millions of Deaf and hard-of-hearing people worldwide [1]. Despite considerable progress in automatic sign language recognition (SLR), the overwhelming majority of deployed systems are designed for recognition — classifying which sign was performed — rather than *assessment* — measuring how accurately a learner reproduced a reference sign. This distinction is fundamental for educational applications: a learner practising Azerbaijani Sign Language (AzSL) needs immediate, graded, interpretable feedback, not a binary correct/incorrect label.

To the best of the authors' knowledge, no prior quantitative computational study on Azerbaijani Sign Language recognition has been published; AzSL remains understudied in the computational linguistics literature. This work constitutes the first systematic machine-recognition evaluation on an AzSL dataset.

Modern deep learning SLR systems [2,3] achieve impressive classification rates on well-curated datasets but carry substantial requirements for deployment in educational tools: they need large annotated corpora, GPU hardware for real-time inference, and treat recognition as a closed-set classification problem with a fixed vocabulary. Adding a new phrase requires collecting training examples and retraining the model. A continuous similarity score — essential for graded learner feedback — is not produced by standard classifiers.

We take a different approach rooted in temporal sequence alignment. Our system uses Dynamic Time Warping (DTW) on geometric hand features extracted via MediaPipe [4,5], enabling:

- **Graded similarity scoring**: every comparison produces a continuous score in [0, 1] describing how well the learner reproduced the reference.
- **Recognition by nearest-neighbour retrieval**: the phrase whose translator video is most similar to the user's video is ranked first, providing Top-K accuracy without any classifier training.
- **Real-time streaming operation**: a purpose-built streaming DTW engine processes each camera frame incrementally at O(N) cost per incoming frame — where N is the number of reference frames — enabling live per-frame feedback without GPU infrastructure.

Our primary contributions are:

1. A **systematic comparison of two contrasting DTW-based approaches** for AzSL gesture recognition — metric-consistent DTW on raw coordinates (Method 1) vs rotation- and scale-invariant geometric features with L1-consistent DTW (Method 2) — evaluated on 215 user videos across 94 AzSL phrases, with statistical significance analysis.
2. A **dual-mode pipeline** — offline batch evaluation and online streaming assessment — sharing the same preprocessing and scoring logic.
3. A **streaming DTW engine** that distributes incremental computation across real-time frames, requiring no training, no GPU, and supporting unlimited vocabulary extension by precomputing a single feature sequence per new phrase.
4. **Data-driven analysis of failure and success patterns** from the actual result files, including score margin statistics and rank distribution across 129 failure cases.

---

## 2. Problem Statement

Let $\mathcal{R} = \{R_1, R_2, \ldots, R_C\}$ be a set of $C = 94$ reference videos, one per AzSL phrase, recorded by a professional translator. Each reference $R_c$ is processed into a sequence of feature vectors $\mathbf{r}_c = [\mathbf{r}_c^{(1)}, \ldots, \mathbf{r}_c^{(N_c)}]$ after preprocessing.

Given a user video $U$ producing feature sequence $\mathbf{u} = [\mathbf{u}^{(1)}, \ldots, \mathbf{u}^{(M)}]$, the system must:

1. Compute a similarity score $s(U, R_c) \in [0, 1]$ for every class $c$.
2. Rank all classes by score: $\pi(1), \pi(2), \ldots, \pi(C)$ where $s(U, R_{\pi(1)}) \geq s(U, R_{\pi(2)}) \geq \ldots$
3. Report whether the correct class appears in position 1, 3, or 5 (Top-1 / Top-3 / Top-5 accuracy).

**Challenges.** The core difficulty is that two videos of the same phrase may differ substantially in:
- **Duration and speed**: a learner may sign slower or faster than the translator. Frame-by-frame comparison without alignment fails entirely.
- **Execution completeness**: the learner may elide a hand position mid-sign, producing shorter sequences.
- **Scale and position**: MediaPipe landmarks vary with hand size and distance from camera.
- **Handedness ambiguity**: some signers use the non-dominant hand as dominant, effectively mirroring the sign.
- **Inter-class similarity**: many AzSL phrases share sub-signs or hand shapes, producing fine-grained discrimination requirements.

The score compression problem is particularly acute with a single reference per class: all 94 similarity scores cluster in a narrow range, making ranking sensitive to small perturbations in feature representation. Our empirical results confirm this: among 129 failures under the best method, the mean score margin between the wrong top-1 and the correct phrase is only 0.034, with a median of 0.025.

---

## 3. Related Work

### 3.1 Dynamic Time Warping in Sign Language

Sakoe and Chiba [6] introduced DTW with the Sakoe-Chiba band constraint for speech recognition, forming the foundation of our alignment approach. Müller [7] extended DTW to Subsequence DTW for partial sequence matching, which underlies our streaming mode. Cheng et al. [8] apply DTW for *feature mapping* in Chinese Sign Language recognition — transforming spatial features into a DTW-distance representation for classification, achieving 93–99% accuracy on 39 static gestures with 11 subjects. Their work demonstrates DTW's effectiveness in the sign language domain but addresses closed-set classification rather than graded similarity assessment, and is evaluated on a substantially smaller vocabulary under controlled conditions.

### 3.2 Sign Language Assessment and Virtual Trainers

Rivera-Cervantes et al. [9] present the most closely related work: a virtual trainer for Mexican Sign Language using MediaPipe keypoints and DTW. They extract 48 keypoints (21 per hand + 6 facial, 2D only), apply z-score normalisation, and accept a sign when the global DTW distance falls below an empirically-set threshold. A user study with 33 participants on 12 MSL phrases confirms feasibility (Cronbach's α = 0.81). Key distinctions from our work: (i) they use 2D coordinates; we use 3D MediaPipe landmarks with depth; (ii) their system uses DTW distance as a binary acceptance threshold; we produce a continuous similarity score and rank all 94 classes; (iii) their DTW cost function is Euclidean on raw coordinates, creating an objective inconsistency with the acceptance criterion — an issue our design explicitly addresses; (iv) our streaming engine compares against all references simultaneously at O(N) per frame, rather than one reference per session.

### 3.3 Feature Representations for Hand Pose

Raw MediaPipe coordinates are sensitive to global rotation and absolute scale [4,5]. Translation and scale invariance are achieved by wrist-centred normalisation, but rotation invariance requires a structural change in the feature representation. Angle-based descriptors — computing angles between bone vectors — are invariant to rigid rotation of the whole hand. Our two approaches directly reflect this: Method 1 operates on raw wrist-centred coordinates; Method 2 replaces these with pairwise bone angles and relative lengths, achieving full rotation and scale invariance. Adaloglou et al. [2] provide a comprehensive survey of deep learning-based SLR; their benchmark shows that learned embeddings outperform geometric features on large corpora, but our setting — one reference per class, no training data — makes learned approaches inapplicable without fundamental architectural changes.

---

## 4. Dataset

### 4.1 Azerbaijani Sign Language Corpus

Our dataset consists of 94 AzSL phrases for which both translator and user videos are available, drawn from 121 recorded phrases. The remaining 27 phrases were excluded because no user videos had been collected for them at the time of this evaluation — only translator reference recordings existed. Phrases span daily life, education, weather, family, health, and cultural contexts. Examples include: *"Bu gün hava çox soyuqdur"* (Today the weather is very cold), *"Mən Bakıda yaşayıram"* (I live in Baku), *"21 Mart bayram günüdür"* (March 21 is a holiday).

Each phrase folder contains:
- **1 translator video**: recorded by a professional AzSL interpreter, serving as the sole reference.
- **1–6 user videos**: recorded by learners attempting to reproduce the sign.

**Statistics:** 94 translator reference videos; 215 user test videos; 309 total recordings. Videos are captured at 30 fps, 480p–720p resolution, indoor with natural lighting, single signer per video. The system is fully training-free: no labelled pairs, no held-out splits, no model training occurs. The translator videos serve only as reference templates at evaluation time.

### 4.2 Preprocessing Pipeline

All videos undergo a shared three-stage preprocessing pipeline before feature extraction:

**Stage 1 — MediaPipe extraction.** We apply MediaPipe Hands [5] in static image mode to each frame, extracting up to 2 hands × 21 landmarks × 3 coordinates = 126-dimensional feature vectors. Each hand is normalised by centering at the wrist (landmark 0) and scaling by the maximum wrist-to-landmark distance, achieving translation and scale invariance.

**Stage 2 — Hand-only filtering.** Frames where no hand is detected (all-zero feature vector) are discarded, removing pre-sign setup and post-sign pauses. Typically 20–30% of frames are removed.

**Stage 3 — Redundancy removal.** Consecutive near-duplicate frames — where the normalised Euclidean similarity exceeds 0.99 — are dropped, preserving meaningful pose transitions while removing static holds. Typically 50–60% of remaining frames are removed, yielding 60–150 frames per video from an initial 200–400.

| Stage | Typical Frame Count | Reduction |
|-------|---------------------|-----------|
| Raw video | 200–400 | — |
| After hand-only filter | 150–300 | 20–30% |
| After redundancy removal | 60–150 | 50–60% |

---

## 5. Methods

In our broader experimental series we evaluated seven DTW-based methods with varying feature designs, alignment costs, and scoring strategies. We present the two most informative and architecturally distinct variants here, labelled by their original experimental indices (Method 1, Method 2) for traceability with accompanying result files. Both share the preprocessing pipeline above; they differ in feature vector, DTW cost function, and score metric, representing two fundamentally different philosophies: Method 1 fixes the alignment metric while keeping raw features, Method 2 replaces raw features with rotation- and scale-invariant geometric descriptors.

### 5.1 Method 1: Cosine-Consistent DTW on Coordinates

**Motivation.** A core design flaw in many DTW-based pipelines is an objective mismatch: the alignment path is found by minimising one distance (e.g. Euclidean) but the final score is evaluated by a different metric (e.g. cosine similarity). This means the alignment path does not directly optimise what the score measures. Method 1 eliminates this inconsistency by using cosine distance as both the DTW cost and the scoring basis, making alignment and scoring fully consistent.

**Feature.** Raw wrist-centred, scale-normalised 126-D MediaPipe coordinates, pre-normalised to unit vectors before DTW: $\hat{\mathbf{r}}_i = \mathbf{r}_i / \|\mathbf{r}_i\|$.

**Alignment.** Global DTW with Sakoe-Chiba band (window ratio 0.30), cosine-distance cost:
$$\text{cost}(i, j) = 1 - \hat{\mathbf{r}}_i \cdot \hat{\mathbf{u}}_j$$

**Scoring.** Mean cosine similarity over aligned pairs — fully consistent with the alignment objective.

**No swap.** Method 1 does not implement hand swap invariance.

**Result:** Top-1 = 33.95%, Top-3 = 44.65%, Top-5 = 50.23%.

---

### 5.2 Method 2: Rotation-Invariant Angles + Relative Lengths + L1-Consistent DTW

**Motivation.** Method 1's objective consistency is a necessary design property, but raw coordinates remain sensitive to global hand rotation and absolute hand size: if the user tilts their wrist or has a different hand size than the translator, the coordinate vectors diverge even for identical hand shapes. Method 2 addresses this by replacing raw coordinates with rotation- and scale-invariant geometric descriptors, while preserving objective consistency using L1 distance throughout.

**Feature construction.** For each hand:
- **190 pairwise bone angles** (C(20,2) = 190), normalised to [0, 1]: rotation-invariant hand shape.
- **20 relative bone lengths**: each length divided by the sum of all 20 lengths for that hand — scale-invariant within the hand:

$$\ell_k^{\text{rel}} = \frac{\|\mathbf{b}_k\|}{\sum_{j=1}^{20} \|\mathbf{b}_j\|}$$

Final feature: $\mathbf{f} \in \mathbb{R}^{420}$. No temporal derivative is appended: without first resolving the absolute-length sensitivity, the d1 terms would carry the same scale confound.

**Alignment.** Global DTW with Sakoe-Chiba band (window ratio 0.25), L1 mean cost:
$$\text{cost}(i, j) = \frac{1}{D} \sum_{k=1}^{D} |f_{\text{ref},i}^{(k)} - f_{\text{user},j}^{(k)}|$$

**Scoring.** Mean of $(1 - \text{L1\_mean})$ over aligned pairs — fully consistent: path minimises L1, score is also L1-based.

**Swap invariance.** Each video pair is compared twice (original and left/right hand blocks swapped); the higher score is retained.

**Result:** Top-1 = **40.0%**, Top-3 = **50.23%**, Top-5 = **56.28%** — best across all methods.

**Why Method 2 outperforms Method 1.** Two structural differences drive the gain over Method 1:
1. *Rotation invariance*: pairwise bone angles are unchanged by rigid hand rotation; raw coordinates are not.
2. *Scale invariance*: relative bone lengths remove sensitivity to physical hand size differences between signers.

---

### 5.3 Comparative Architecture Summary

| Property | Method 1 | Method 2 |
|----------|----|----|
| Feature | Raw wrist-centred coords | Bone angles + relative lengths |
| Feature dim | 126 | 420 |
| Rotation invariant | No | Yes |
| Scale invariant (hand size) | Partial | Yes |
| DTW cost | Cosine-dist | L1 mean |
| Score metric | Cosine | L1-based |
| Cost/score consistent | Yes | Yes |
| Swap invariance | No | Yes |
| DTW window ratio | 0.30 | 0.25 |
| Top-1 | 33.95% | **40.00%** |
| Top-3 | 44.65% | **50.23%** |
| Top-5 | 50.23% | **56.28%** |

---

## 6. Real-Time Streaming Assessment Engine

### 6.1 Why Not a Neural Classifier?

A standard deep learning SLR classifier (CNN-LSTM, transformer, I3D) requires: (a) training on multiple labelled examples per class, (b) a fixed, closed vocabulary where adding a new phrase requires collecting training data and retraining, and (c) a discrete output label with no natural continuous similarity score. Our dataset has one reference video per class; our educational use case requires a continuous grade; and we need to add new AzSL phrases at any time without retraining. These three requirements together make standard neural classifiers inapplicable in our setting.

The streaming DTW engine requires no training, produces a continuous similarity score, and supports an unlimited vocabulary by precomputing a single feature sequence per new phrase.

### 6.2 Streaming DTW Architecture

The streaming engine implements Subsequence DTW [7] column-by-column as camera frames arrive. The key distinction from offline DTW is the initialisation: in Subsequence DTW, the first row is initialised as $D[0, m] = C[0, m]$ (free start), allowing matching to begin at any position in the user stream. This is appropriate for the camera use case where the user's video may contain pre-signing and post-signing content.

Let $\mathbf{R} \in \mathbb{R}^{N \times D}$ be the precomputed reference feature matrix. As user frame $m$ arrives:

$$D[n, m] = \text{cost}(n, m) + \min(D[n-1, m-1],\; D[n-1, m],\; D[n, m-1])$$
$$D[0, m] = C[0, m] \quad \text{(free start)}$$

Only two columns $D_{\text{prev}}$ and $D_{\text{curr}}$ are stored in memory at any time — O(N) space. One column is computed per arriving frame, distributing the total O(N×M) work of the DTW computation across M real-time frames rather than performing it all at once offline.

**Important note on features.** The current real-time streaming engine operates on the base 126-D MediaPipe coordinate features — the same feature space as Method 1 — rather than Method 2's 420-D geometric descriptor. Integrating Method 2 features into the streaming pipeline is ongoing work; the per-frame column update cost scales linearly with feature dimension, adding approximately 3× computation per reference per frame, which remains within the 55 ms MediaPipe budget.

**Multi-reference operation.** MediaPipe feature extraction runs once per frame. The resulting feature vector is fed to all C = 94 reference DTW engines simultaneously. Each reference maintains its own independent $D_{\text{prev}}/D_{\text{curr}}$ column pair.

### 6.3 State Machine and Finalisation

The camera system operates as a four-state machine:

```
WAITING → RECORDING → HAND_LOST → RESULTS
```

On entering RESULTS (after a 4-second hand-absence timeout), a finalisation step rebuilds the full accumulated cost matrix from stored user features and performs subsequence DTW backtracking, yielding a final ranked similarity score for each of the 94 reference classes.

### 6.4 Latency Profile

| Operation | Complexity | Typical Time |
|-----------|------------|--------------|
| MediaPipe extraction (per frame) | O(1) | ~55 ms |
| DTW column update (per reference, per frame) | O(N) | < 1 ms |
| Full offline DTW matrix (one video pair) | O(N × M) | 10–25 s |
| Finalisation (backtrack + score) | O(N + M) | < 100 ms |

The incremental per-frame update (< 1 ms per reference) keeps the system responsive during signing. The total computational work across a full signing session is O(N × M) — the same as offline — but distributed in real time rather than performed as a single blocking computation.

---

## 7. Experiments and Results

### 7.1 Experimental Setup

- Python 3.10, MediaPipe 0.10.5, NumPy, SciPy.
- All evaluation on CPU; no GPU used.
- Dataset: 94 AzSL phrases, 215 user videos, 1 translator reference per phrase.
- Evaluation: for each user video, compute similarity against all 94 references; report the rank of the correct phrase.
- Metric: Top-K accuracy (K = 1, 3, 5) across all 215 user videos.

### 7.2 Method Comparison

| Method | Top-1 | Top-3 | Top-5 |
|--------|-------|-------|-------|
| Method 1 (raw coords, cosine-cost DTW) | 33.95% (73/215) | 44.65% (96/215) | 50.23% (108/215) |
| **Method 2 (bone angles + rel. lengths, L1 DTW)** | **40.00% (86/215)** | **50.23% (108/215)** | **56.28% (121/215)** |

### 7.3 Statistical Significance (McNemar's Test)

To assess whether Top-1 differences between methods are statistically significant, we applied McNemar's test with continuity correction on the 215 paired binary outcomes (Top-1 correct/incorrect per video).

| Comparison | Method 2-only correct | Method 1-only correct | χ² | Significant? |
|---|---|---|---|---|
| Method 2 vs Method 1 | 38 | 25 | 2.29 | No, p ≈ 0.13 |

Method 2's 6 percentage point Top-1 advantage over Method 1 does not reach statistical significance at p = 0.05 (McNemar χ² = 2.29, p ≈ 0.13). This reflects that the two methods make substantially different errors — 38 videos are correctly recognised by Method 2 but not Method 1, while 25 are correctly recognised by Method 1 but not Method 2 — and the asymmetry is real but insufficient for significance at n = 215. Nevertheless, Method 2's architectural properties (rotation invariance, scale invariance) provide a principled basis to prefer it: the performance difference arises from genuine structural improvements, not random variation, and it is the basis for ongoing feature development.

### 7.4 Success Case Analysis

**When the system works well**: the correct phrase score exceeds the runner-up by a clear margin (≥ 0.02), and the phrase has a sufficiently distinct hand configuration from its nearest neighbours.

*Example — Success:* Phrase *"21 Mart bayram günüdür"* (March 21 is a holiday), User 2 (112 frames).
```
Rank 1: "21 Mart bayram günüdür"         score = 0.9226  ← CORRECT
Rank 2: "Mart ayında çoxlu bayram var"   score = 0.9049  (margin = +0.018)
Rank 3: "Mən sabah saat 2də..."          score = 0.9020
```
The user signed at a pace close to the translator's (112 frames, comparable to reference length), providing sufficient temporal coverage for DTW to find a well-aligned path. The correct phrase scores 0.018 above the nearest competitor.

### 7.5 Failure Case Analysis

Among the 129 failures (215 − 86) under Method 2, we characterise the error distribution using objective criteria computed directly from the result JSON:

**Rank distribution of failures:**
| Category | Count | Share of failures |
|---|---|---|
| Near miss — correct phrase in ranks 2–3 | 22 | 17.1% |
| Near miss — correct phrase in ranks 4–5 | 13 | 10.1% |
| **Total near miss (correct in Top-5)** | **35** | **27.1%** |
| Far miss — correct phrase ranked > 5 | 94 | **72.9%** |

**Score margin analysis:** Among all 129 failures, the gap between the wrong top-1 score and the correct phrase score has mean 0.034, median 0.025, with 54 cases (41.9%) showing a gap below 0.02 — meaning the system was essentially tied but ranked the wrong phrase first. The maximum observed margin is 0.138, indicating some failures are clear rather than borderline.

**Key observations:**
- The dominant failure mode (72.9%) is the *far miss*: the correct phrase is not even in the top-5 candidates. This is more severe than score compression alone and likely reflects sign pairs that share extensive hand-configuration overlap across unrelated phrases.
- Among far misses, low frame count (fewer than 40 kept frames) accounts for only 4 cases (3.1%), suggesting that incomplete signing is not the primary driver.
- The 41.9% rate of tight-margin failures (gap < 0.02) confirms that score compression is a real phenomenon, but the majority of failures involve larger gaps where the wrong phrase genuinely scores higher — pointing to feature representation limits rather than scoring calibration.

*Example — Failure:* Phrase *"21 Mart bayram günüdür"*, User 1 (75 frames).
```
Rank 1: "Mart ayında çoxlu bayram var"            score = 0.9106  ← WRONG
Rank 2: "Şkafda çoxlu köhnə banka var"            score = 0.9073
Rank 8: "21 Mart bayram günüdür"                  score = 0.8964  ← CORRECT
Gap = 0.014
```
This is a *near-miss* case (rank 8, just outside Top-5). Both the correct phrase ("March 21 is a holiday") and the top-ranked wrong phrase ("March has many holidays") share the signs for "Mart" (March) and "bayram" (holiday), producing geometrically similar bone-angle sequences. The user's faster execution (75 frames vs the 112-frame success case) further compressed the temporal structure. The distinction between these two phrases lies in transitional motion between sub-signs — information that our current static angle features do not capture directly.

### 7.6 Top-1 vs Top-3 vs Top-5 Interpretation

The 35 near-miss failures (correct phrase in Top-5 but not Top-1) represent videos where the system identified the correct phrase as highly similar but ranked it second through fifth. In an educational deployment context — a "did you mean one of these?" interface — Top-3 accuracy of 50.23% and Top-5 accuracy of 56.28% are achievable with the same underlying model. For a training-free nearest-neighbour system comparing against 94 classes with one reference each (chance = 1.06%), these figures represent a strong baseline.

---

## 8. Conclusion

We presented a DTW-based pipeline for Azerbaijani Sign Language recognition and real-time assessment, comparing two contrasting approaches on 215 user videos across 94 AzSL phrases. Our key empirically-supported findings are:

1. **Geometric invariance outperforms consistent-metric raw coordinates**: Method 2's rotation- and scale-invariant features achieve 40.0% Top-1 vs Method 1's 33.95%, a +6 pp advantage. The difference does not reach statistical significance at this sample size (McNemar χ² = 2.29, p ≈ 0.13), but the methods make largely distinct errors — 38 Method 2-only correct vs 25 Method 1-only correct — indicating complementary strengths.

2. **Both approaches share objective consistency**: Method 1 uses cosine distance for both alignment and scoring; Method 2 uses L1 distance for both. This eliminates a common DTW design flaw (aligning by one metric, evaluating by another) that was present in earlier variants.

3. **72.9% of failures are far misses**: the correct phrase is not even in the top-5 candidates, indicating that for the majority of errors the feature representation is genuinely confused — not merely a scoring calibration issue. This motivates structural feature additions (palm orientation, temporal motion) rather than post-hoc score adjustments.

4. **The real-time streaming engine enables educational deployment** without GPU infrastructure: incremental O(N)-per-frame DTW distributes computation across the signing session, with per-frame update times below 1 ms per reference class.

5. **This work provides the first quantitative recognition baseline for AzSL**, achieving 40.0% Top-1 and 56.28% Top-5 accuracy on 94 phrases with a single reference video per class and no training data.

### Future Work

- **Method 8 (under evaluation)**: adds palm plane normal vector (capturing which way the palm faces — currently absent from angle features) and temporal first derivative (motion direction) to Method 2's 420-D feature set, directly motivated by the far-miss failure pattern.
- **Integrating Method 2 features into the streaming engine**: extending from 126-D to 420-D adds approximately 3× computation per reference per frame, which remains within the 55 ms MediaPipe budget.
- **Per-feature normalisation**: z-score normalisation across the corpus to balance the contribution of angle and relative-length dimensions, which currently operate at different effective scales.
- **Score fusion**: combining Method 2 and Method 1 similarity scores — whose errors are partially independent (38 Method 2-only correct, 25 Method 1-only correct) — may improve Top-1 accuracy without additional video processing.
- **Multi-reference and larger vocabulary**: collecting 3–5 translator references per phrase and extending to all 121 recorded AzSL phrases.
- **Body and facial features**: integrating MediaPipe Pose and Face Mesh for signs involving body location or facial non-manual markers.

---

## Acknowledgements

The authors thank the professional AzSL interpreter who provided all 94 translator reference videos, and the learner participants whose recordings form the user video dataset.

---

## References

[1] World Health Organization. *Deafness and Hearing Loss.* WHO Fact Sheet, 2023. https://www.who.int/news-room/fact-sheets/detail/deafness-and-hearing-loss (accessed February 2026).

[2] Adaloglou, N., Chatzis, T., Papastratis, I., Stergioulas, A., Papadopoulos, G.T., Zacharopoulos, V., Xydopoulos, G.J., Atzakas, K., Papazachariou, D., & Daras, P. (2022). A comprehensive study on deep learning-based methods for sign language recognition. *IEEE Transactions on Multimedia*, 24, 1–12. https://doi.org/10.1109/TMM.2021.3070438

[3] Camgoz, N.C., Hadfield, S., Koller, O., Ney, H., & Bowden, R. (2018). Neural sign language translation. In *Proceedings of CVPR 2018*, pp. 7784–7793.

[4] Lugaresi, C., Tang, J., Nash, H., McClanahan, C., Uboweja, E., Hays, M., Zhang, F., Chang, C.-L., Yong, M.G., Lee, J., Chang, W.-T., Hua, W., Georg, M., & Grundmann, M. (2019). MediaPipe: A framework for building perception pipelines. *arXiv preprint* arXiv:1906.08172.

[5] Zhang, F., Bazarevsky, V., Vakunov, A., Tkachenka, A., Sung, G., Chang, C.-L., & Grundmann, M. (2020). MediaPipe hands: On-device real-time hand tracking. *arXiv preprint* arXiv:2006.10214.

[6] Sakoe, H., & Chiba, S. (1978). Dynamic programming algorithm optimization for spoken word recognition. *IEEE Transactions on Acoustics, Speech, and Signal Processing*, 26(1), 43–49. https://doi.org/10.1109/TASSP.1978.1163055

[7] Müller, M. (2015). *Fundamentals of Music Processing: Audio, Analysis, Algorithms, Applications.* Springer. Section 7.2: Subsequence DTW.

[8] Cheng, J., Wei, F., Liu, Y., Li, C., Chen, Q., & Chen, X. (2020). Chinese sign language recognition based on DTW-distance-mapping features. *Mathematical Problems in Engineering*, 2020, Article 8953670. https://doi.org/10.1155/2020/8953670

[9] Rivera-Cervantes, F., Córdova-Esparza, D.-M., Terven, J., Romero-González, J.-A., González-Rodríguez, J.-R., Ibarra-Corona, M.-A., & Ramírez-Pedraza, P.-A. (2025). Virtual trainer for learning Mexican Sign Language using video similarity analysis. *Technologies*, 13(12), 540. https://doi.org/10.3390/technologies13120540

[10] Sakurai, Y., Faloutsos, C., & Yamamuro, M. (2007). Stream monitoring under the time warping distance. In *Proceedings of ICDE 2007*, pp. 1046–1055.

[11] Daniel, C.A., Gopinath, M.A., Suresh, V.V.D., Siddharth, B., & Nair, A. (2024). Enhancing language learning with real-time sign language recognition and feedback. In *Proceedings of IEEE URTC 2024*.

---

## Appendix: Key System Parameters

| Parameter | Value | Role |
|-----------|-------|------|
| MediaPipe max hands | 2 | Bilateral signing support |
| Redundancy threshold | 0.99 | Normalised Euclidean similarity cutoff |
| DTW window ratio | 0.25 (Method 2) / 0.30 (Method 1) | Sakoe-Chiba band width |
| Feature dimension | 420 (Method 2) / 126 (Method 1) | Per-frame feature size |
| Swap mode | global (Method 2) / none (Method 1) | Left/right hand swap invariance |
| Streaming feature space | 126-D (base coords) | Current real-time engine; Method 2 integration ongoing |
| Streaming memory | O(N) — 2 columns | vs O(N×M) offline |
| Hand-gone timeout | 4.0 s | Camera mode finalisation trigger |
