# Research Workflow (Seq2Seq vs Cosine)

This runbook produces the key artifacts for your comparative paper.

## 1) Unified Seq2Seq Evaluation (Top-1/3/5 + Confusion + Latency)

```bash
python3 research_eval_seq2seq.py \
  --beam-width 5 \
  --top-k 5 \
  --require-feature-cache \
  --output-dir research_outputs/seq2seq_full
```

Outputs:
- `research_outputs/seq2seq_full/seq2seq_eval_metrics.json`
- `research_outputs/seq2seq_full/seq2seq_eval_results.csv`
- `research_outputs/seq2seq_full/seq2seq_per_class_metrics.csv`
- `research_outputs/seq2seq_full/confusion_pairs.csv`
- `research_outputs/seq2seq_full/confusion_matrix_top_classes.csv`

## 2) Data Efficiency Study (25/50/75/100)

```bash
python3 run_data_efficiency_study.py \
  --fractions 0.25,0.50,0.75,1.00 \
  --epochs 20 \
  --beam-width 5 \
  --top-k 5 \
  --require-feature-cache \
  --output-dir research_outputs/data_efficiency
```

Outputs:
- Per fraction folder:
  - `training_history.csv`
  - `seq2seq_eval_metrics.json`
  - `seq2seq_eval_results.csv`
  - model snapshots (`encoder.model`, `decoder.model`, best variants)
- Summary:
  - `research_outputs/data_efficiency/data_efficiency_summary.csv`

## 3) Learning Curve + Confusion Figure

```bash
python3 plot_research_figures.py \
  --history-csv research_outputs/data_efficiency/f100/training_history.csv \
  --confusion-csv research_outputs/seq2seq_full/confusion_matrix_top_classes.csv \
  --output-dir research_outputs/figures
```

## 4) Final Comparison Table (Cosine vs Seq2Seq)

```bash
python3 build_comparison_table.py \
  --seq2seq-metrics research_outputs/seq2seq_full/seq2seq_eval_metrics.json \
  --cosine-top1 30.7 \
  --cosine-top3 45.0 \
  --cosine-top5 49.4 \
  --cosine-inference-ms 3.0 \
  --output-csv research_outputs/comparison_table.csv
```

## Notes
- `research_eval_seq2seq.py` uses beam search for Top-K.
- Use `--max-videos N` for quick dry runs before full evaluation.
- Keep the same `limit` and preprocessing setup across experiments for fair comparison.
