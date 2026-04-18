"""
Generate DTW warping-path figure from actual computation.

This wrapper calls the unified paper figure generator implementation so we
don't accidentally use older screenshot-cropping logic.
"""
from generate_all_paper_figures import fig_dtw_warping_path


if __name__ == "__main__":
    print("Generating fig_dtw_warping_path from actual computation...")
    fig_dtw_warping_path()
    print("Done.")
