# Inference Project

Cleaned project layout for video/sign-language inference experiments.

## Structure

```
inference/
  src/
    inference/
      __init__.py
      similarity/
        __init__.py
        feature_extractor.py
        frame_extractor.py
        similarity_engine.py
        video_matcher.py
        dtw_visualizer.py

  scripts/
    extract_all_vectors.py
    video_similarity.py
    realtime_similarity.py
    folder_video_similarity.py
    top3_accuracy.py
    visualize_confused_pairs.py
    visualize_frames.py
    generate_paper_figures.py
    generate_all_paper_figures.py
    compute_paper_stats.py
    config.py

  data/
    raw/
      Videos/
      Videos.zip
    interim/
      extracted_frames/
    processed/
      matrices/

  outputs/
    figures/
      paper_figures/
    analysis/
      confused_pairs/
    reports/

  docs/
    paper.tex
    research_paper_structure.txt

  tmp/
    tmpclaude-*-cwd/
```

## Typical Commands

Run commands from the repository root (`inference/`):

- Extract vectors: `python scripts/extract_all_vectors.py`
- Compare videos: `python scripts/video_similarity.py --help`
- Realtime matching: `python scripts/realtime_similarity.py`
- Generate paper figures: `python scripts/generate_all_paper_figures.py`
- Compute stats: `python scripts/compute_paper_stats.py`

## Notes

- Scripts are now path-configured to use:
  - videos: `data/raw/Videos`
  - matrices: `data/processed/matrices`
  - figures: `outputs/figures/paper_figures`
  - analysis exports: `outputs/analysis/confused_pairs`
- Core library code lives under `src/inference/similarity`.
