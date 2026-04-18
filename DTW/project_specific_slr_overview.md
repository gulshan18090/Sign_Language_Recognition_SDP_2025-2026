# Project-Specific SLR Overview: Azerbaijani Sign Language Gesture Similarity Assessment

## 1. The Landscape of Sign Language Recognition (SLR)
Sign Language Recognition (SLR) is a vital area in AI and Human-Computer Interaction, enabling communication for millions with hearing or speech impairments. Our work addresses the need for objective, quantitative assessment of sign language gesture reproduction, focusing on Azerbaijani Sign Language. Unlike generic SLR systems, which often focus on recognition/classification, our system provides similarity assessment—measuring how accurately a learner reproduces a reference sign.

### SLR Taxonomy in This Work
- **Dataset:** 121 Azerbaijani Sign Language phrases, 119 reference (translator) videos, 215 user (learner) videos (vision-based, RGB only).
- **Features:** Manual features only (hand posture, location, orientation) extracted via MediaPipe Hands (21 landmarks × 3D × 2 hands = 126-D vector per frame).
- **Modalities:** Vision-based (webcam/video), no sensor fusion or wearable hardware.
- **System Barriers:** Environmental factors (lighting, background) are mitigated by wrist-centered normalization and robust preprocessing (hand-only filtering, redundancy removal).

## 2. Strategic System Architecture: Dual-Mode Integration
Our system implements a dual-mode pipeline:
- **Offline Mode:** Batch evaluation of recorded videos using global DTW with Sakoe-Chiba band and bidirectional optimization.
- **Online Mode:** Real-time camera-based assessment using streaming subsequence DTW, enabling immediate feedback for learners.

**Pipeline Steps:**
1. Data Acquisition: Video input (RGB, 30 fps, 480p–720p).
2. Preprocessing: Hand-only frame filtering (discard frames with no hands), redundant frame removal (normalized Euclidean similarity ≥ 0.99).
3. Feature Extraction: MediaPipe Hands, wrist-centered and scale-normalized 126-D vectors.
4. Alignment/Scoring:
   - Offline: Global DTW, bidirectional optimization, cosine similarity scoring.
   - Online: Streaming DTW, incremental updates, real-time scoring.
5. User Interface: Scores and feedback visualized for learners.

| Feature | Our Method |
|---------|------------|
| Capturing Device | Webcam/Video |
| Efficiency | Real-time (online) and batch (offline) |
| Cost | Low |
| Key Advantage | Immediate feedback, robust to speed/length variations |
| Limitation | Sensitive to hand occlusion, lighting |

## 3. Robust Data Preprocessing and Frame Optimization
Preprocessing is critical for throughput and accuracy. Our pipeline:
- Removes frames with no hands (using MediaPipe detection).
- Drops redundant frames (normalized Euclidean similarity ≥ 0.99), typically reducing frame count by 50–60%.
- Ensures only meaningful gesture transitions are retained, reducing computational load and improving DTW alignment.

| Stage | Frames (typical) | Reduction |
|-------|------------------|-----------|
| Raw video frames | 200–400 | — |
| After hand-only filtering | 150–300 | ~20–30% removed |
| After redundancy removal | 60–150 | ~50–60% removed |

## 4. High-Dimensional Feature Extraction and Normalization
- MediaPipe Hands extracts 21 landmarks per hand (x, y, z), concatenated for both hands (126-D vector).
- Wrist-centered normalization: All landmarks are centered and scaled by the wrist, making features invariant to hand position and scale.
- No sensor fusion, PCA/LDA, or deep learning classifiers are used—our focus is on interpretable, robust geometric features.

## 5. Temporal Alignment Methodologies: The DTW Framework
- Global DTW (Offline): Aligns full reference and user sequences, with Sakoe-Chiba band (window ratio 0.3) and bidirectional optimization (max frame reuse = 3).
- Streaming DTW (Online): Subsequence DTW with free-start boundary, incremental O(N) per-frame updates, enabling real-time feedback.
- Scoring: Cosine similarity on bidirectionally-optimized aligned pairs.

| Folder | Method 1 | Method 2 | Method 3 | M3 vs M1 | Aligned Pairs |
|--------|----------|----------|----------|----------|---------------|
| 51     | 0.4254   | 0.4412   | 0.6249   | +46.9%   | 75            |
| 79     | 0.7785   | 0.7706   | 0.8430   | +8.3%    | 73            |
| 8      | 0.4768   | 0.4817   | 0.6292   | +32.0%   | 71            |

- Average improvement of Method 3 over Method 1: +29.1%
- Average improvement of Method 3 over Method 2: +27.2%

## 6. Bidirectional Optimization and Real-Time Feedback
- Bidirectional optimization: After DTW, each aligned pair is locally optimized to maximize cosine similarity, enforcing monotonicity and max reuse.
- Real-time feedback: In online mode, the system provides running similarity scores, with the intended phrase appearing in the top-3 ranked results in over 90% of cases.

## 7. Future Directions and Strategic Implications
- Facial/body pose integration: Extend to non-manual features (facial expressions, body pose) using MediaPipe Face/Pose.
- Larger datasets: Expand to more users and phrases for cross-lingual evaluation.
- Adaptive thresholds: Dynamically adjust DTW window and deduplication thresholds based on video quality.
- Mobile deployment: Port online mode to smartphones for accessible at-home practice.
- Spatial feedback: Provide per-landmark error visualization for targeted learner feedback.

**Summary:**
This project delivers a robust, interpretable, and real-time SLR similarity assessment pipeline for Azerbaijani Sign Language, validated on a real dataset with significant improvements over baseline methods. All methods, results, and figures are based on the actual implementation and experiments described in this paper.
