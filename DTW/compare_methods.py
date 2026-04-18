"""
Compare cosine vs frame-wise cosine similarity methods
"""

import numpy as np
from pathlib import Path
from similarity import SimilarityEngine
import json


def load_matrix(matrix_path: Path) -> np.ndarray:
    """Load a .npy matrix file."""
    return np.load(matrix_path)


def compare_methods():
    """Compare regular cosine vs frame-wise cosine vs DTW-aligned cosine on existing video pairs."""
    
    matrices_root = Path("matrices")
    
    # Initialize engines with different methods
    engine_cosine = SimilarityEngine(default_method="cosine")
    engine_framewise = SimilarityEngine(default_method="frame_wise_cosine")
    engine_dtw_aligned = SimilarityEngine(default_method="dtw_aligned_cosine")
    
    results = {
        "method_comparison": {},
        "summary": {}
    }
    
    # Define the pairs based on the directory structure
    pairs = [
        {
            "folder": "51",
            "video1": "translator_video_2_unknown_72",
            "video2": "user_video_28_OneAK8_25",
        },
        {
            "folder": "79",
            "video1": "translator_video_7_jhasanov_15",
            "video2": "user_video_71_Aykanabi_17",
        },
        {
            "folder": "8",
            "video1": "translator_video_2_unknown_35",
            "video2": "user_video_8_HSTechk_29",
        },
    ]
    
    print("\n" + "="*80)
    print("COMPARISON: Regular Cosine vs Frame-wise Cosine vs DTW-Aligned Cosine")
    print("="*80 + "\n")
    
    differences_fw = []  # frame-wise vs regular
    differences_dtw = []  # dtw-aligned vs regular
    
    for pair in pairs:
        folder = pair["folder"]
        video1 = pair["video1"]
        video2 = pair["video2"]
        
        # Load matrices
        matrix1_path = matrices_root / folder / f"{video1}.npy"
        matrix2_path = matrices_root / folder / f"{video2}.npy"
        
        if not matrix1_path.exists() or not matrix2_path.exists():
            print(f"WARNING: Skipping {folder}: Matrix files not found")
            continue
        
        matrix1 = load_matrix(matrix1_path)
        matrix2 = load_matrix(matrix2_path)
        
        # Compute similarities with all three methods
        cosine_sim = engine_cosine.compute_similarity(matrix1, matrix2, method="cosine")
        framewise_sim = engine_framewise.compute_similarity(matrix1, matrix2, method="frame_wise_cosine")
        dtw_aligned_sim = engine_dtw_aligned.compute_similarity(matrix1, matrix2, method="dtw_aligned_cosine")
        
        diff_fw = abs(cosine_sim - framewise_sim)
        diff_dtw = abs(cosine_sim - dtw_aligned_sim)
        differences_fw.append(diff_fw)
        differences_dtw.append(diff_dtw)
        
        # Store results
        pair_key = f"Folder {folder}: {video1} vs {video2}"
        results["method_comparison"][pair_key] = {
            "regular_cosine": float(cosine_sim),
            "frame_wise_cosine": float(framewise_sim),
            "dtw_aligned_cosine": float(dtw_aligned_sim),
            "diff_framewise_vs_regular": float(diff_fw),
            "diff_dtw_vs_regular": float(diff_dtw)
        }
        
        # Print results
        print(f"[Folder {folder}]")
        print(f"   Video 1: {video1}")
        print(f"   Video 2: {video2}")
        print(f"   Matrix shapes: {matrix1.shape} vs {matrix2.shape}")
        print(f"   ")
        print(f"   Regular Cosine:        {cosine_sim:.6f}")
        print(f"   Frame-wise Cosine:     {framewise_sim:.6f}")
        print(f"   DTW-Aligned Cosine:    {dtw_aligned_sim:.6f}")
        print(f"   ")
        
        # Find the best method
        best_sim = max(cosine_sim, framewise_sim, dtw_aligned_sim)
        if best_sim == dtw_aligned_sim:
            print(f"   >>> DTW-Aligned is HIGHEST (best frame matching)")
        elif best_sim == framewise_sim:
            print(f"   >>> Frame-wise is HIGHEST (good temporal alignment)")
        else:
            print(f"   >>> Regular cosine is HIGHEST (overall similarity)")
        print()
    
    # Summary statistics
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Frame-wise vs Regular:")
    print(f"  Average difference: {np.mean(differences_fw):.6f} ({np.mean(differences_fw)*100:.2f}%)")
    print(f"  Max difference:     {np.max(differences_fw):.6f} ({np.max(differences_fw)*100:.2f}%)")
    print(f"  Min difference:     {np.min(differences_fw):.6f} ({np.min(differences_fw)*100:.2f}%)")
    print()
    print(f"DTW-Aligned vs Regular:")
    print(f"  Average difference: {np.mean(differences_dtw):.6f} ({np.mean(differences_dtw)*100:.2f}%)")
    print(f"  Max difference:     {np.max(differences_dtw):.6f} ({np.max(differences_dtw)*100:.2f}%)")
    print(f"  Min difference:     {np.min(differences_dtw):.6f} ({np.min(differences_dtw)*100:.2f}%)")
    
    results["summary"] = {
        "framewise_vs_regular": {
            "avg_difference": float(np.mean(differences_fw)),
            "max_difference": float(np.max(differences_fw)),
            "min_difference": float(np.min(differences_fw))
        },
        "dtw_aligned_vs_regular": {
            "avg_difference": float(np.mean(differences_dtw)),
            "max_difference": float(np.max(differences_dtw)),
            "min_difference": float(np.min(differences_dtw))
        }
    }
    
    # Save results
    output_file = matrices_root / "method_comparison.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[SUCCESS] Results saved to: {output_file}")
    print("\n" + "="*80)


if __name__ == "__main__":
    compare_methods()
