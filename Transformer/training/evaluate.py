"""
Evaluation metrics:
  - Word Error Rate  (WER)  – primary metric for SLR
  - Sequence Accuracy       – exact match
  - BLEU-4                  – secondary
"""
from typing import List, Tuple
import torch
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction


def word_error_rate(hypothesis: List[str], reference: List[str]) -> float:
    """
    Standard WER: (S + D + I) / N
    Both inputs are lists of tokens (already split).
    """
    N = len(reference)
    if N == 0:
        return 0.0
    # Dynamic programming edit distance
    d = [[0] * (len(hypothesis) + 1) for _ in range(N + 1)]
    for i in range(N + 1):
        d[i][0] = i
    for j in range(len(hypothesis) + 1):
        d[0][j] = j
    for i in range(1, N + 1):
        for j in range(1, len(hypothesis) + 1):
            cost = 0 if reference[i - 1] == hypothesis[j - 1] else 1
            d[i][j] = min(d[i-1][j] + 1, d[i][j-1] + 1, d[i-1][j-1] + cost)
    return d[N][len(hypothesis)] / N


def evaluate_batch(
    predictions: List[str],
    references: List[str],
) -> dict:
    """
    predictions, references: lists of decoded gloss strings.
    Returns dict with wer, seq_acc, bleu4.
    """
    wer_total, exact_total = 0.0, 0
    hyp_corpus, ref_corpus = [], []

    for pred, ref in zip(predictions, references):
        pred_toks = pred.lower().split()
        ref_toks  = ref.lower().split()

        wer_total   += word_error_rate(pred_toks, ref_toks)
        exact_total += int(pred_toks == ref_toks)

        hyp_corpus.append(pred_toks)
        ref_corpus.append([ref_toks])

    n = len(predictions)
    smooth = SmoothingFunction().method1
    bleu = corpus_bleu(ref_corpus, hyp_corpus, smoothing_function=smooth)

    return {
        "wer":      wer_total / max(n, 1),
        "seq_acc":  exact_total / max(n, 1),
        "bleu4":    bleu,
        "n_samples": n,
    }
