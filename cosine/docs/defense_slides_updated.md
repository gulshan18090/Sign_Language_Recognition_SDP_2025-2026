# Slide 1 - Problem, Objective, and Contribution

## Problem
Manual sign-gesture evaluation is subjective and inconsistent, especially when signer speed and hand position differ.

## Objective
Develop a reproducible pipeline to compare a user gesture video against a reference (translator) gesture using hand landmarks.

## What We Contributed
- Built a full landmark extraction pipeline with MediaPipe (21 landmarks x 3 coordinates).
- Produced both raw and normalized feature representations.
- Prepared analysis and temporal matching workflow in one notebook.
- Continued the system with a new similarity stage using cosine similarity on video feature vectors.

## Defense Message
This work establishes a measurable baseline for sign gesture similarity, moving from visual judgment to quantitative evidence.

# Slide 2 - Method and System Pipeline

## Pipeline
1. Read input videos with OpenCV.
2. Detect hand and extract 3D landmarks per frame with MediaPipe.
3. Save raw vectors to CSV.
4. Normalize landmarks (wrist-centered + scale normalization).
5. Build frame sequences for user and translator videos.
6. New step: convert videos to fixed-size feature matrices and compare with cosine similarity.
7. Return Top-k ranked matches with similarity scores.

## Why These Choices
- Normalization removes translation and scale bias.
- Cosine similarity is fast, explainable, and suitable for high-dimensional landmark features.
- Top-k ranking is practical because some gesture classes are visually close.

## Defense Message
The method is simple, explainable, and technically justified for variable-speed gesture data.

# Slide 3 - Evidence from EDA

## Analyses Performed
- Variance analysis (raw vs normalized features)
- Correlation heatmaps
- PCA projection and cumulative explained variance
- Low-variance feature filtering
- Additional retrieval analysis with cosine-based ranking outputs

## Main Findings
- Raw coordinates are highly affected by hand position and scale.
- Normalized coordinates are more stable and compact.
- Feature redundancy exists; some coordinates carry limited information.
- Most normalized features remain informative for modeling and similarity ranking.

## Interpretation
Normalized landmarks provide a stronger feature space for comparison and model training than raw landmarks.

## Defense Message
Our EDA validates the preprocessing decisions with statistical and visual evidence.

# Slide 4 - Cosine Results, Limitations, and Next Work

## Cosine Evaluation
- Computed cosine similarity between query and reference landmark feature vectors.
- Visualized ranked outputs and inspected confusion cases.
- Tested realtime Top-3 matching on user-performed sentences.

## Current Limitations
- Threshold is manually selected.
- Similar hand-motion patterns can produce semantically wrong matches.
- Single-hand-centric features and limited dataset coverage reduce robustness.
- No class-wise quantitative benchmark yet.

## Next Work
- Learn class-specific thresholds from labeled data.
- Add robustness to detection noise and missing-hand frames.
- Combine cosine ranking with temporal alignment for ambiguous cases.
- Train and evaluate gesture classifiers using normalized selected features.

## Defense Message
The current workflow demonstrates a working, explainable baseline, and the newly applied cosine stage improves practical retrieval while keeping a clear path to a deployable evaluation system.
