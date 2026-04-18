"""
Cosine Retrieval Evaluation on Telegram Dataset
Extracts MediaPipe hand landmarks from each video, builds 64×126 feature
matrices, and runs leave-one-out cosine similarity evaluation.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger("mediapipe").setLevel(logging.ERROR)
import absl.logging
absl.logging.set_verbosity(absl.logging.FATAL)

import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial.distance import cosine
import csv
import time


NUM_FRAMES = 64
NUM_LANDMARKS = 21
COORDS_PER_LANDMARK = 3
NUM_HANDS = 2
FEATURES_PER_FRAME = NUM_LANDMARKS * COORDS_PER_LANDMARK * NUM_HANDS  # 126
EMBEDDING_DIM = NUM_FRAMES * FEATURES_PER_FRAME  # 8064


def uniform_sample_frames(video_path, n_frames=NUM_FRAMES):
    """Extract n_frames uniformly from a video."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    indices = [int(i * total / n_frames) for i in range(n_frames)]
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            frames.append(np.zeros((480, 640, 3), dtype=np.uint8))
    cap.release()

    # If we got fewer frames, pad with duplicates
    while len(frames) < n_frames:
        frames.append(frames[-1] if frames else np.zeros((480, 640, 3), dtype=np.uint8))

    return frames[:n_frames]


def extract_landmarks(frames, mp_hands):
    """Extract wrist-centred normalised landmarks from frames."""
    feature_matrix = np.zeros((NUM_FRAMES, FEATURES_PER_FRAME), dtype=np.float32)

    for i, frame in enumerate(frames):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_hands.process(rgb)

        frame_features = np.zeros(FEATURES_PER_FRAME, dtype=np.float32)

        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                if hand_idx >= NUM_HANDS:
                    break

                # Determine handedness
                if results.multi_handedness:
                    label = results.multi_handedness[hand_idx].classification[0].label
                    offset = 0 if label == "Left" else NUM_LANDMARKS * COORDS_PER_LANDMARK
                else:
                    offset = hand_idx * NUM_LANDMARKS * COORDS_PER_LANDMARK

                # Extract raw coordinates
                coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
                # Wrist-centred normalisation
                wrist = coords[0]
                centred = coords - wrist
                max_dist = np.max(np.linalg.norm(centred, axis=1))
                if max_dist > 0:
                    normalised = centred / max_dist
                else:
                    normalised = centred

                frame_features[offset:offset + NUM_LANDMARKS * COORDS_PER_LANDMARK] = normalised.flatten()

        feature_matrix[i] = frame_features

    return feature_matrix


def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def main():
    videos_dir = "Videos"
    results_csv = "cosine_eval_telegram_results.csv"

    print("=" * 70)
    print("COSINE RETRIEVAL EVALUATION ON TELEGRAM DATASET")
    print("=" * 70)

    # Collect all video paths and their sentence labels
    entries = []
    for sentence_folder in sorted(os.listdir(videos_dir)):
        sentence_path = os.path.join(videos_dir, sentence_folder)
        if not os.path.isdir(sentence_path):
            continue
        for video_file in sorted(os.listdir(sentence_path)):
            if video_file.lower().endswith(('.mp4', '.avi', '.mov')):
                entries.append({
                    'sentence': sentence_folder,
                    'video_id': video_file,
                    'video_path': os.path.join(sentence_path, video_file),
                    'signer_type': 'translator' if 'translator' in video_file.lower() else 'user'
                })

    print(f"Found {len(entries)} videos across {len(set(e['sentence'] for e in entries))} sentences")
    n_translator = sum(1 for e in entries if e['signer_type'] == 'translator')
    n_user = sum(1 for e in entries if e['signer_type'] == 'user')
    print(f"  Translator: {n_translator}, User: {n_user}")

    # Extract features for all videos
    print("\n--- Extracting MediaPipe landmarks ---")
    mp_hands = mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5,
    )

    embeddings = []
    t0 = time.time()
    for idx, entry in enumerate(entries):
        frames = uniform_sample_frames(entry['video_path'])
        feat_matrix = extract_landmarks(frames, mp_hands)
        embedding = feat_matrix.flatten()
        embeddings.append(embedding)
        if (idx + 1) % 50 == 0 or idx == 0:
            elapsed = time.time() - t0
            print(f"  [{idx+1}/{len(entries)}] {elapsed:.1f}s elapsed")

    mp_hands.close()
    elapsed = time.time() - t0
    print(f"Feature extraction complete: {elapsed:.1f}s total")

    embeddings = np.array(embeddings)

    # Leave-one-out cosine evaluation
    print("\n--- Leave-one-out cosine evaluation ---")
    # Build sentence-to-indices map
    sentence_to_indices = {}
    for i, e in enumerate(entries):
        sentence_to_indices.setdefault(e['sentence'], []).append(i)

    all_sentences = sorted(sentence_to_indices.keys())
    top1_correct = 0
    top3_correct = 0
    top5_correct = 0
    total = len(entries)

    # Per-signer counters
    translator_correct_1 = 0
    translator_total = 0
    user_correct_1 = 0
    user_total = 0
    translator_correct_3 = 0
    user_correct_3 = 0

    detailed_results = []

    for q_idx in range(total):
        query_emb = embeddings[q_idx]
        query_sentence = entries[q_idx]['sentence']
        query_signer = entries[q_idx]['signer_type']

        # Compute per-sentence max similarity (excluding query)
        sentence_scores = {}
        for sent, indices in sentence_to_indices.items():
            ref_indices = [i for i in indices if i != q_idx]
            if not ref_indices:
                continue
            max_sim = max(cosine_similarity(query_emb, embeddings[ri]) for ri in ref_indices)
            sentence_scores[sent] = max_sim

        # Rank sentences
        ranked = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)
        top_sentences = [s for s, _ in ranked]

        is_top1 = query_sentence in top_sentences[:1]
        is_top3 = query_sentence in top_sentences[:3]
        is_top5 = query_sentence in top_sentences[:5]

        if is_top1:
            top1_correct += 1
        if is_top3:
            top3_correct += 1
        if is_top5:
            top5_correct += 1

        if query_signer == 'translator':
            translator_total += 1
            if is_top1: translator_correct_1 += 1
            if is_top3: translator_correct_3 += 1
        else:
            user_total += 1
            if is_top1: user_correct_1 += 1
            if is_top3: user_correct_3 += 1

        detailed_results.append({
            'sentence': query_sentence,
            'video_id': entries[q_idx]['video_id'],
            'signer_type': query_signer,
            'top1_pred': top_sentences[0] if top_sentences else '',
            'top1_match': is_top1,
            'top3_match': is_top3,
            'top5_match': is_top5,
            'top1_sim': ranked[0][1] if ranked else 0,
        })

    # Print results
    print(f"\n{'='*70}")
    print(f"RESULTS: Cosine Retrieval on Telegram ({total} videos)")
    print(f"{'='*70}")
    print(f"Overall Top-1: {top1_correct}/{total} = {100*top1_correct/total:.1f}%")
    print(f"Overall Top-3: {top3_correct}/{total} = {100*top3_correct/total:.1f}%")
    print(f"Overall Top-5: {top5_correct}/{total} = {100*top5_correct/total:.1f}%")
    print(f"\nTranslator Top-1: {translator_correct_1}/{translator_total} = {100*translator_correct_1/translator_total:.1f}%")
    print(f"Translator Top-3: {translator_correct_3}/{translator_total} = {100*translator_correct_3/translator_total:.1f}%")
    print(f"User Top-1: {user_correct_1}/{user_total} = {100*user_correct_1/user_total:.1f}%")
    print(f"User Top-3: {user_correct_3}/{user_total} = {100*user_correct_3/user_total:.1f}%")

    # Frame count sensitivity (resample from 64-frame matrices)
    print(f"\n{'='*70}")
    print(f"FRAME COUNT SENSITIVITY")
    print(f"{'='*70}")
    for n_frames in [16, 32, 64]:
        if n_frames == 64:
            sub_embeddings = embeddings
        else:
            # Resample: pick n_frames out of 64
            indices = [int(i * 64 / n_frames) for i in range(n_frames)]
            sub_matrices = embeddings.reshape(total, NUM_FRAMES, FEATURES_PER_FRAME)[:, indices, :]
            sub_embeddings = sub_matrices.reshape(total, n_frames * FEATURES_PER_FRAME)

        fc_top1 = 0
        fc_top3 = 0
        for q_idx in range(total):
            query_emb = sub_embeddings[q_idx]
            query_sentence = entries[q_idx]['sentence']
            sentence_scores = {}
            for sent, s_indices in sentence_to_indices.items():
                ref_indices = [i for i in s_indices if i != q_idx]
                if not ref_indices:
                    continue
                max_sim = max(cosine_similarity(query_emb, sub_embeddings[ri]) for ri in ref_indices)
                sentence_scores[sent] = max_sim
            ranked = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)
            top_sentences = [s for s, _ in ranked]
            if query_sentence in top_sentences[:1]: fc_top1 += 1
            if query_sentence in top_sentences[:3]: fc_top3 += 1
        dim = n_frames * FEATURES_PER_FRAME
        print(f"N={n_frames:3d}: Top-1={100*fc_top1/total:.1f}%  Top-3={100*fc_top3/total:.1f}%  Embedding dim={dim:,}")

    # Save detailed results
    with open(results_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=detailed_results[0].keys())
        writer.writeheader()
        writer.writerows(detailed_results)
    print(f"\nDetailed results saved to {results_csv}")


if __name__ == '__main__':
    main()
