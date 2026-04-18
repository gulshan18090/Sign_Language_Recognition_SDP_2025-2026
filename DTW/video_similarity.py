"""
Video Similarity Matching - Entry Point Script

This script provides a CLI interface for comparing videos based on
hand movement features extracted using MediaPipe.

Usage:
    # Compare a reference video to all videos in a folder
    python video_similarity.py --reference video_A.mp4 --folder videos/
    
    # Compare with custom settings
    python video_similarity.py --reference video_A.mp4 --folder videos/ --top-k 10 --method cosine
    
    # Process and save features for later use
    python video_similarity.py --folder videos/ --save-features features.npz
    
    # Load pre-computed features and compare
    python video_similarity.py --reference video_A.mp4 --load-features features.npz
"""

import argparse
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from similarity import VideoMatcher


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Video Similarity Matching based on Hand Movements",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    python video_similarity.py --reference video_A.mp4 --folder videos/
    
    # Find top 5 similar videos
    python video_similarity.py --reference video_A.mp4 --folder videos/ --top-k 5
    
    # Use different similarity method
    python video_similarity.py --reference video_A.mp4 --folder videos/ --method frame_wise_cosine
    
    # Save extracted features for later use
    python video_similarity.py --folder videos/ --save-features features.json
        """
    )
    
    # Input arguments
    parser.add_argument(
        '--reference', '-r',
        type=str,
        help='Path to reference video (Video A)'
    )
    
    parser.add_argument(
        '--folder', '-f',
        type=str,
        help='Path to folder containing candidate videos'
    )
    
    parser.add_argument(
        '--videos', '-v',
        nargs='+',
        type=str,
        help='List of specific video paths to compare'
    )
    
    # Feature caching
    parser.add_argument(
        '--save-features',
        type=str,
        help='Save extracted features to file (.json or .npz)'
    )
    
    parser.add_argument(
        '--load-features',
        type=str,
        help='Load pre-computed features from file'
    )
    
    # Processing parameters
    parser.add_argument(
        '--n-frames',
        type=int,
        default=64,
        help='Number of frames to extract per video (default: 64)'
    )
    
    parser.add_argument(
        '--max-hands',
        type=int,
        default=2,
        choices=[1, 2],
        help='Maximum number of hands to track (default: 2)'
    )
    
    parser.add_argument(
        '--method',
        type=str,
        default='cosine',
        choices=['cosine', 'euclidean', 'pearson', 'frame_wise_cosine', 'temporal_correlation', 'dtw'],
        help='Similarity computation method (default: cosine). Use "dtw" for Dynamic Time Warping with temporal alignment.'
    )
    
    parser.add_argument(
        '--dtw-window',
        type=float,
        default=0.3,
        help='DTW window ratio (0.0-1.0). Frame i can only match frame j where |i-j| <= window_ratio * n_frames. Default: 0.3 (30%%)'
    )
    
    parser.add_argument(
        '--top-k',
        type=int,
        default=None,
        help='Return only top K similar videos'
    )
    
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.5,
        help='Minimum hand detection confidence (default: 0.5)'
    )
    
    parser.add_argument(
        '--no-filter',
        action='store_true',
        help='Disable filtering to hand-only frames'
    )
    
    parser.add_argument(
        '--no-normalize',
        action='store_true',
        help='Disable landmark normalization'
    )
    
    # Output options
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Export results to JSON file'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress progress output'
    )
    
    parser.add_argument(
        '--recursive',
        action='store_true',
        help='Search for videos recursively in subdirectories'
    )
    
    # Visualization options
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Visualize DTW alignment for top matches (requires --method dtw)'
    )
    
    parser.add_argument(
        '--visualize-top',
        type=int,
        default=1,
        help='Number of top matches to visualize (default: 1)'
    )
    
    parser.add_argument(
        '--save-viz',
        type=str,
        help='Directory to save visualization images'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Validate arguments
    if not args.reference and not args.save_features:
        print("Error: --reference is required unless just saving features")
        sys.exit(1)
    
    if not args.folder and not args.videos and not args.load_features:
        print("Error: --folder, --videos, or --load-features is required")
        sys.exit(1)
    
    # Initialize matcher
    print("\n" + "=" * 60)
    print("VIDEO SIMILARITY MATCHING")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  - Frames per video: {args.n_frames}")
    print(f"  - Max hands: {args.max_hands}")
    print(f"  - Similarity method: {args.method}")
    if args.method == 'dtw':
        print(f"  - DTW window ratio: {args.dtw_window} ({int(args.dtw_window * 100)}% of frames)")
    print(f"  - Detection confidence: {args.confidence}")
    print(f"  - Filter hand frames: {not args.no_filter}")
    print(f"  - Normalize landmarks: {not args.no_normalize}")
    print("-" * 60)
    
    matcher = VideoMatcher(
        n_frames=args.n_frames,
        max_hands=args.max_hands,
        similarity_method=args.method,
        normalize_landmarks=not args.no_normalize,
        min_detection_confidence=args.confidence,
        filter_hand_frames=not args.no_filter,
        cache_features=True,
        dtw_window_ratio=args.dtw_window
    )
    
    # Load or extract features
    if args.load_features:
        print(f"\nLoading features from {args.load_features}...")
        candidate_features = matcher.load_features(args.load_features)
    else:
        if args.folder:
            print(f"\nProcessing videos in {args.folder}...")
            candidate_features = matcher.process_video_folder(
                args.folder,
                recursive=args.recursive,
                show_progress=not args.quiet
            )
        elif args.videos:
            print(f"\nProcessing {len(args.videos)} videos...")
            candidate_features = {}
            for video_path in args.videos:
                try:
                    features = matcher.process_video(video_path)
                    candidate_features[video_path] = features
                except Exception as e:
                    print(f"Error processing {video_path}: {e}")
    
    # Save features if requested
    if args.save_features:
        matcher.save_features(candidate_features, args.save_features)
    
    # Find similar videos if reference provided
    if args.reference:
        print(f"\nProcessing reference video: {args.reference}")
        
        # Process reference
        ref_features = matcher.process_video(args.reference)
        
        result = matcher.find_similar(
            reference=ref_features,
            candidates=candidate_features,
            top_k=args.top_k,
            method=args.method,
            exclude_self=True
        )
        
        # Display results
        matcher.print_results(result, max_display=args.top_k or 10)
        
        # Export if requested
        if args.output:
            matcher.export_results(result, args.output)
        
        # Visualize DTW alignment if requested
        if args.visualize and args.method == 'dtw':
            from similarity import DTWVisualizer
            
            visualizer = DTWVisualizer()
            
            # Filter out self-matches and get top matches to visualize
            ref_path_resolved = str(Path(args.reference).resolve())
            non_self_rankings = [
                (vp, score) for vp, score in result.rankings
                if str(Path(vp).resolve()) != ref_path_resolved
            ]
            
            n_viz = min(args.visualize_top, len(non_self_rankings))
            viz_count = 0
            
            for video_path, similarity_score in non_self_rankings[:n_viz]:
                viz_count += 1
                
                print(f"\n{'='*60}")
                print(f"Visualizing match #{viz_count}: {Path(video_path).name}")
                print(f"{'='*60}")
                
                # Get feature matrices
                candidate_matrix = candidate_features[video_path].feature_matrix
                
                # Get DTW alignment path
                similarity, alignment_path = matcher.similarity_engine.get_dtw_alignment(
                    ref_features.feature_matrix,
                    candidate_matrix
                )
                
                # Create visualizations
                visualizer.create_full_visualization(
                    video_a_path=args.reference,
                    video_b_path=video_path,
                    matrix_a=ref_features.feature_matrix,
                    matrix_b=candidate_matrix,
                    alignment_path=alignment_path,
                    similarity_score=similarity,
                    save_dir=args.save_viz,
                    show=True
                )
        elif args.visualize and args.method != 'dtw':
            print("\n⚠️  Visualization is only supported for --method dtw")
    
    print("\nDone!")


def demo():
    """
    Demo function showing basic usage.
    
    Run this to see an example of how to use the VideoMatcher.
    """
    print("=" * 60)
    print("VIDEO SIMILARITY DEMO")
    print("=" * 60)
    
    # Create sample usage code
    demo_code = '''
# Initialize the matcher
from similarity import VideoMatcher

matcher = VideoMatcher(
    n_frames=64,        # Extract 64 frames per video
    max_hands=2,        # Track both hands
    similarity_method='cosine'
)

# Process reference video
reference = matcher.process_video("path/to/video_A.mp4")
print(f"Reference features shape: {reference.feature_matrix.shape}")

# Process candidate videos
candidates = matcher.process_video_folder("path/to/videos/")
print(f"Processed {len(candidates)} candidate videos")

# Find top 5 similar videos
result = matcher.find_similar(
    reference=reference,
    candidates=candidates,
    top_k=5
)

# Display results
matcher.print_results(result)

# Export results
matcher.export_results(result, "similarity_results.json")
'''
    
    print(demo_code)
    print("=" * 60)
    print("\nTo run on your videos, use:")
    print("  python video_similarity.py --reference video_A.mp4 --folder videos/")


if __name__ == '__main__':
    if len(sys.argv) == 1:
        # No arguments, show demo
        demo()
    else:
        main()
