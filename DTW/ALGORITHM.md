# Hybrid Online-Offline Sign Language Similarity Assessment Using Streaming DTW and Cosine Similarity

## Table of Contents

1. [Overview](#1-overview)
2. [Feature Extraction](#2-feature-extraction)
3. [Preprocessing Pipeline](#3-preprocessing-pipeline)
4. [Subsequence DTW (Offline Mode)](#4-subsequence-dtw-offline-mode)
5. [Streaming DTW (Online Mode)](#5-streaming-dtw-online-mode)
6. [Bidirectional Optimization](#6-bidirectional-optimization)
7. [Dual Distance Metric Strategy](#7-dual-distance-metric-strategy)
8. [Real-Time Camera Mode](#8-real-time-camera-mode)
9. [Scoring](#9-scoring)
10. [Complexity Analysis](#10-complexity-analysis)
11. [References](#11-references)

---

## 1. Overview

The system compares a **user's** sign language video (or live camera stream) against one or more **reference (translator)** videos. It outputs a similarity score in `[0, 1]` indicating how closely the user replicated the reference sign.

Two modes are supported:

| Mode | Input | DTW Construction | Use Case |
|------|-------|-----------------|----------|
| **Offline** | Two complete videos | Full cost matrix at once | Batch evaluation |
| **Online (Streaming)** | Live camera / frame-by-frame | Incremental, one column per frame | Real-time feedback |

Both modes produce **identical** alignment scores — they differ only in *when* the computation happens.

---

## 2. Feature Extraction

Each video frame is processed by **MediaPipe Hands** (v0.10.5) to extract hand landmarks.

```
Frame (RGB) → MediaPipe Hands → 21 landmarks × 3 coords × 2 hands = 126-D vector
```

- **21 landmarks** per hand: wrist, thumb (4), index (4), middle (4), ring (4), pinky (4)
- **3 coordinates** per landmark: (x, y, z) normalized to [0, 1]
- **2 hands**: left and right → concatenated into a single 126-dimensional feature vector
- If only one hand is detected, the other 63 dimensions are zero
- If **no hands** detected, the frame is **discarded** (zero vector)

**Implementation:** `similarity/feature_extractor.py` → `HandFeatureExtractor` class

---

## 3. Preprocessing Pipeline

Both videos go through the same pipeline before DTW:

```
Raw Video → Extract ALL Frames → MediaPipe → Keep Hands-Only → Drop Redundant
```

### 3.1 Hand-Only Filtering

Frames where MediaPipe detects no hands are removed. This eliminates irrelevant frames (blank backgrounds, transitions, etc.).

### 3.2 Redundancy Removal

Consecutive similar frames are dropped using **Normalized Euclidean Distance**:

$$
\text{sim}(a, b) = 1 - \frac{\| \hat{a} - \hat{b} \|^2}{2}
$$

where $\hat{a} = \frac{a}{\|a\|}$ is the unit-normalized vector.

- **Threshold:** 0.99 — frames with similarity ≥ 0.99 to the previous kept frame are dropped
- This removes near-duplicate frames (e.g., when the hand is stationary)
- Reduces frame count by ~50-60% while preserving all meaningful motion

### 3.3 Result

| Video | Raw Frames | With Hands | After Dedup | Reduction |
|-------|-----------|------------|-------------|-----------|
| Reference (example) | 262 | 253 | 120 | 54% |
| User (example) | 344 | 276 | 142 | 59% |

---

## 4. Subsequence DTW (Offline Mode)

We use **Subsequence DTW** rather than global DTW. This is critical because:

- The user video may be **longer** than the reference (extra frames before/after the sign)
- We want to find the **best-matching subsequence** in the user video that corresponds to the entire reference
- The reference must be fully consumed; the user video can match partially

### 4.1 Cost Matrix

Given reference features $R$ (N frames) and user features $U$ (M frames):

$$
C[n, m] = \text{euclidean}(R[n], U[m]) \quad \forall \, n \in [0, N), \; m \in [0, M)
$$

### 4.2 Accumulated Cost Matrix

The key difference from global DTW is the **initialization**:

$$
D[n, 0] = \sum_{k=0}^{n} C[k, 0] \quad \text{(must use ALL of reference)}
$$

$$
D[0, m] = C[0, m] \quad \text{(can START anywhere in user stream)}
$$

Recursion (same as global DTW):

$$
D[n, m] = C[n, m] + \min \begin{cases} D[n-1, m-1] & \text{(diagonal — match)} \\ D[n-1, m] & \text{(vertical — reference skip)} \\ D[n, m-1] & \text{(horizontal — user skip)} \end{cases}
$$

### 4.3 Optimal End Point

The best subsequence ends at:

$$
b^* = \arg\min_{m} D[N-1, m]
$$

This finds the column in the last row with minimum accumulated cost — the optimal endpoint of the user's matching subsequence.

### 4.4 Backtracking

Starting from $(N-1, b^*)$, trace back to row 0 by following the minimum-predecessor rule:

```
path = [(N-1, b*)]
while n > 0:
    (n, m) = argmin of {D[n-1,m-1], D[n-1,m], D[n,m-1]}
    path.prepend((n, m))
a* = path[0].m   # start point in user stream
```

The path maps each reference frame to a user frame: `path[i] = (ref_idx, user_idx)`.

### 4.5 Matching Function

The matching function $\Delta(m)$ shows the normalized cost of the best alignment ending at each user frame:

$$
\Delta(m) = \frac{D[N-1, m]}{N}
$$

The minimum of $\Delta$ corresponds to $b^*$. This is visualized as a plot to show where the best matching subsequence is located.

---

## 5. Streaming DTW (Online Mode)

### 5.1 Key Insight

The accumulated cost matrix $D$ can be computed **column by column**. Each column $D[:, m]$ depends only on:
- The previous column $D[:, m-1]$
- The cost column $C[:, m]$

This means we only need to store **two columns** at any time: `D_prev` and `D_curr`.

### 5.2 Algorithm

```
Pre-compute: R = reference features (N × 126)    ← done once
Initialize:  D_prev = [∞, ∞, ..., ∞]            ← N values

For each incoming user frame u_m:
    1. Extract features: feat = MediaPipe(u_m)
    2. Skip if no hand (zero vector)
    3. Skip if redundant vs last kept frame
    4. Compute cost column: C[n, m] = euclidean(R[n], feat) for all n
    5. Update accumulated cost:
         D_curr[0] = C[0, m]                     ← free start
         D_curr[n] = C[n,m] + min(D_prev[n-1], D_prev[n], D_curr[n-1])
    6. Running score: Δ(m) = D_curr[N-1] / N
    7. D_prev ← D_curr
```

### 5.3 Running Score

At any point during streaming, we can estimate the current similarity by looking at the best alignment so far:

$$
\text{running\_score} = \cos(\text{R}[\text{last}], \; \text{U}[\text{best\_m}])
$$

where $\text{best\_m} = \arg\min_t \Delta(t)$ across all columns processed so far.

### 5.4 Finalization

When the user stops signing (hand disappears), we **rebuild** the full accumulated cost matrix from stored user features and perform:
1. Full backtracking to get the optimal alignment path
2. Bidirectional optimization (Section 6) for each matched pair
3. Cosine similarity scoring

This finalization step ensures the streaming mode produces **identical** results to offline mode.

### 5.5 Memory

| | Offline | Streaming |
|---|---|---|
| Cost matrix | N × M (full) | N × 1 (two columns) |
| Memory | O(N × M) | O(N) |
| Per-frame cost | N/A | O(N) |

---

## 6. Bidirectional Optimization

After DTW alignment, each matched pair `(ref[n], user[m])` is **optimized** by searching for a better match within a local window.

### 6.1 Window Size

$$
W = \max(1, \lfloor 0.2 \times \max(N, M) \rfloor)
$$

This is 20% of the longer sequence length.

### 6.2 Search Strategy

For each aligned pair at position `pidx` in the path:

```
Original match: (ref[n], user[m]) with similarity s

Search range: [min_m, max_m)
  where min_m = previous path point's user index (preserves monotonicity)
        max_m = next path point's user index

For each alt_m in [min_m, max_m):
    alt_s = cosine_similarity(ref[n], user[alt_m])
    if alt_s > best_s:
        best_s = alt_s
        best_m = alt_m
```

**Monotonicity constraint:** The search range is bounded by neighboring path points, ensuring the alignment order is preserved.

### 6.3 Effect

This optimization typically improves 5-15% of matched pairs, increasing total similarity by capturing small timing differences that DTW's discrete steps miss.

---

## 7. Dual Distance Metric Strategy

Two different distance metrics are used for different purposes:

| Purpose | Metric | Formula | Why |
|---------|--------|---------|-----|
| **DTW alignment** (path finding) | Euclidean | $d(a,b) = \|a - b\|_2$ | Better for geometric distance in DTW cost matrix |
| **Redundancy removal** | Normalized Euclidean | $\text{sim} = 1 - \frac{d^2}{2}$ on unit vectors | Scale-invariant duplicate detection |
| **Final scoring** | Cosine Similarity | $\text{sim} = \frac{a \cdot b}{\|a\| \|b\|}$ | Direction-based, captures pose similarity regardless of hand size |

**Rationale:** Euclidean distance is appropriate for DTW because it penalizes magnitude differences, which helps the path-finding algorithm avoid degenerate alignments. Cosine similarity is used for final scoring because sign language similarity depends more on the *direction* (hand shape/pose) than the *magnitude* (hand size/distance from camera).

---

## 8. Real-Time Camera Mode

The camera system (`camera_dtw.py`) combines streaming DTW with a state machine:

```
                    ┌─────────────┐
                    │   WAITING   │ ← "Show your hand..."
                    │  (no hand)  │
                    └──────┬──────┘
                           │ hand detected
                    ┌──────▼──────┐
              ┌────►│  RECORDING  │ ← REC indicator + live scores
              │     │ (streaming) │
              │     └──────┬──────┘
              │            │ hand disappears
              │     ┌──────▼──────┐
              │     │  HAND LOST  │ ← 4-second countdown
              │     │ (countdown) │
              │     └──────┬──────┘
     hand     │            │ timeout (4s)
     returns  │     ┌──────▼──────┐
              └─────┤   RESULTS   │ ← Ranked scores
                    │  (finalize) │
                    └─────────────┘
                     [A]pprove [R]etry [Q]uit
```

### 8.1 Multi-Reference Comparison

Features are extracted **once** per camera frame via MediaPipe, then the 126-D vector is fed to **all** StreamingDTW instances simultaneously:

```
Camera Frame → MediaPipe (1×) → feat (126-D)
                                   ├── StreamingDTW(Ref 51).process_feature_vector(feat)
                                   ├── StreamingDTW(Ref 79).process_feature_vector(feat)
                                   └── StreamingDTW(Ref 8).process_feature_vector(feat)
```

This avoids running MediaPipe 3× per frame — the bottleneck (~55ms) is paid only once.

### 8.2 Live Score Display

Each reference's running score is displayed as a colored bar on the camera feed:
- 🟢 Green: score ≥ 0.70
- 🟡 Yellow: score ≥ 0.45
- 🔴 Red: score < 0.45

---

## 9. Scoring

### 9.1 Per-Frame Similarity

After alignment and optimization, each matched pair is scored with cosine similarity:

$$
s_i = \frac{R[n_i] \cdot U[m_i]}{\|R[n_i]\| \cdot \|U[m_i]\|}
$$

### 9.2 Final Score

The final similarity score is the **average** of all per-frame similarities:

$$
\text{Score} = \frac{1}{L} \sum_{i=1}^{L} s_i
$$

where $L$ is the alignment path length (= number of reference frames, since each must be matched).

### 9.3 Additional Metrics

| Metric | Description |
|--------|-------------|
| **Total similarity sum** | $\sum s_i$ — absolute measure of match quality |
| **Good matches** | Fraction of pairs with $s_i \geq 0.45$ |
| **Alignment range** | $[a^*, b^*]$ — where in the user video the best match was found |

---

## 10. Complexity Analysis

### 10.1 Offline Mode

| Step | Complexity |
|------|-----------|
| Feature extraction | O((N + M) × T_mediapipe) |
| Cost matrix | O(N × M × D) |
| Accumulated cost | O(N × M) |
| Backtracking | O(N + M) |
| Optimization | O(L × W) where W = window size |
| **Total** | **O(N × M × D)** |

### 10.2 Online (Streaming) Mode

| Step | Per frame | Total |
|------|-----------|-------|
| MediaPipe extraction | O(T_mediapipe) ≈ 55ms | O(M × T_mediapipe) |
| DTW column update | O(N × D) | O(M × N × D) |
| Finalization | — | O(N × M) (one-time) |
| **Total** | **O(N)** per frame | **O(N × M × D)** |

Where:
- N = reference frames (after filtering), typically 80–120
- M = user frames (after filtering), typically 50–150
- D = feature dimension = 126
- T_mediapipe ≈ 55ms per frame

### 10.3 Practical Timing

| Operation | Typical Time |
|-----------|-------------|
| MediaPipe per frame | ~55 ms |
| DTW column update per frame | ~0.1 ms |
| Full offline DTW (matrix only) | 36–141 ms |
| Finalization (backtrack + optimize) | ~50 ms |
| Total for 150-frame video | ~10 s (dominated by MediaPipe) |

---

## 11. References

1. **Müller, M.** *Fundamentals of Music Processing* (FMP), Springer, 2015, Section 7.2 — Subsequence DTW algorithm
   - Online companion: https://www.audiolabs-erlangen.de/resources/MIR/FMP/C7/C7S2_SubsequenceDTW.html

2. **MediaPipe Hands** — Google, 2020. Real-time hand landmark detection.
   - https://google.github.io/mediapipe/solutions/hands.html

3. **Sakoe, H. & Chiba, S.** *Dynamic Programming Algorithm Optimization for Spoken Word Recognition.* IEEE T-ASSP, 1978 — Classic DTW formulation.
