# Top-2 Methods (Full Dataset) + Visual Debug

Dataset (full run):
- Translator reference videos: **94** (one per phrase/folder)
- User test videos: **215**
- Evaluation: for each user video, compare against **all 94** translator videos, rank by similarity, report Top-1/Top-3/Top-5 accuracy.

## Best Full-Run Accuracies

1) **Method 4 Variant (bones_angles_len_d1, swap=global, win=0.25)**  
Source: `inference/recognition_results/recognition_method4_bones_angles_len_d1_global_win0p25_summary.json`
- Top-1: **33.02%**
- Top-3: **49.30%**
- Top-5: **55.35%**

2) **Method 6 (DTW with cosine-distance cost, win=0.30)**  
Source: `inference/recognition_results/recognition_method6_dtw_cos_cost_summary.json`
- Top-1: **33.95%**
- Top-3: **44.65%**
- Top-5: **50.23%**

## Method 4 Variant (bones_angles_len_d1) — Algorithm (Concise)

Goal: be more robust to arm/camera rotation by using **angles/lengths** instead of raw coordinates, plus optional motion (d1), and make left/right hand usage comparable.

Steps:
1. Decode video frames with OpenCV.
2. Extract MediaPipe hand landmarks per frame (up to 2 hands) as a 126-D vector: `[Left(63) | Right(63)]`.
3. Normalize per hand (already in our pipeline): subtract wrist landmark (translation invariance); divide by max wrist-to-landmark distance (scale invariance).
4. Drop redundant consecutive frames (similarity threshold `thr=0.99`) to keep only informative frames.
5. Convert each kept frame from coordinates into a higher-level feature vector: build 20 bone vectors per hand (MediaPipe connections); compute all pairwise bone angles per hand `C(20,2)=190`; add 20 bone lengths per hand; concatenate left+right `210 + 210 = 420`; append first derivative `d1` (frame-to-frame difference) to get `420 + 420 = 840` dims per frame.
6. **Global swap invariance** (left/right mismatch handling): score once with user features as-is; score again with user left/right blocks swapped; keep the higher final score (single global decision per video pair).
7. DTW alignment (Sakoe-Chiba band with `window_ratio=0.25`): DTW accumulated cost uses **Euclidean distance** on the per-frame 840-D features; backtrack one end-to-end alignment path.
8. Final similarity score = **mean cosine similarity** over the aligned frame pairs (higher is better).

## Method 6 (Cosine-Cost DTW on Coordinates) — Algorithm (Concise)

Goal: make DTW alignment optimize the *same* objective as the displayed similarity, reducing “unrelated” DTW matches caused by metric mismatch.

Steps:
1. Decode video frames with OpenCV.
2. Extract MediaPipe hand landmarks per frame as 126-D coordinates `[Left(63) | Right(63)]` with wrist+scale normalization.
3. Drop redundant frames with threshold `thr=0.99`.
4. DTW alignment (Sakoe-Chiba band with `window_ratio=0.30`) using **cosine-distance** as the DP cost: `cost(i,j) = 1 - cosine_similarity(frame_i, frame_j)`; dynamic programming finds the globally best monotonic path for this cost (within the band).
5. Final similarity score = **mean cosine similarity** along the DTW path.

## Visual Debug (3-Panel Viewer)

Viewer script: `inference/debug_top5_misses_flow.py`  
Panels: **Correct translator | User | Wrong(top-1) translator**, with MediaPipe landmarks overlay.

To debug misses for each method, point the viewer at that method’s summary JSON and use match mode:

```powershell
cd inference

# Method 4 Variant (bones_angles_len_d1, swap=global, win=0.25) viewer
.\venv310\Scripts\python.exe debug_top5_misses_flow.py `
  --results-json .\recognition_results\recognition_method4_bones_angles_len_d1_global_win0p25_summary.json `
  --mode match --method-view m4

# Method 6 viewer
.\venv310\Scripts\python.exe debug_top5_misses_flow.py `
  --results-json .\recognition_results\recognition_method6_dtw_cos_cost_summary.json `
  --mode match --method-view m6
```

Match-mode keys:
- `4` = Method 4 variant view (M4V)
- `6` = Method 6 view (CosCostDTW)
- `Space` toggle autoplay, `A/D` step, `H` help, `N` next case, `Q` quit
