"""
DTW Visualization Module

Visualizes Dynamic Time Warping alignments between videos.
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from typing import List, Tuple, Optional
import os


class DTWVisualizer:
    """
    Visualizes DTW alignment between two videos.
    
    Features:
    - Warping path plot
    - Side-by-side matched frames
    - Alignment matrix heatmap
    """
    
    def __init__(self, output_dir: str = "dtw_visualizations"):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def plot_warping_path(
        self,
        alignment_path: List[Tuple[int, int]],
        n_frames_a: int,
        n_frames_b: int,
        video_a_name: str = "Video A",
        video_b_name: str = "Video B",
        save_path: Optional[str] = None,
        show: bool = True
    ):
        """
        Plot the DTW warping path.
        
        Shows which frames from Video A are matched to Video B.
        Diagonal line = perfect 1:1 alignment.
        Deviations = temporal warping.
        
        Args:
            alignment_path: List of (frame_a_idx, frame_b_idx) tuples
            n_frames_a: Number of frames in Video A
            n_frames_b: Number of frames in Video B
            video_a_name: Name for Video A label
            video_b_name: Name for Video B label
            save_path: Optional path to save figure
            show: Whether to display the plot
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Extract path coordinates
        path_a = [p[0] for p in alignment_path]
        path_b = [p[1] for p in alignment_path]
        
        # Plot the warping path
        ax.plot(path_b, path_a, 'b-', linewidth=2, label='DTW Path')
        ax.plot(path_b, path_a, 'bo', markersize=4)
        
        # Plot diagonal (perfect alignment reference)
        diag_max = max(n_frames_a, n_frames_b)
        ax.plot([0, diag_max], [0, diag_max], 'r--', linewidth=1, alpha=0.5, label='Perfect Alignment')
        
        # Highlight start and end points
        ax.plot(path_b[0], path_a[0], 'go', markersize=12, label='Start', zorder=5)
        ax.plot(path_b[-1], path_a[-1], 'r*', markersize=15, label='End', zorder=5)
        
        ax.set_xlabel(f'{video_b_name} Frame Index', fontsize=12)
        ax.set_ylabel(f'{video_a_name} Frame Index', fontsize=12)
        ax.set_title('DTW Warping Path\n(Deviation from diagonal = temporal warping)', fontsize=14)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-1, n_frames_b)
        ax.set_ylim(-1, n_frames_a)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved warping path plot to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def plot_alignment_matrix(
        self,
        matrix_a: np.ndarray,
        matrix_b: np.ndarray,
        alignment_path: List[Tuple[int, int]],
        video_a_name: str = "Video A",
        video_b_name: str = "Video B",
        save_path: Optional[str] = None,
        show: bool = True
    ):
        """
        Plot the distance matrix with alignment path overlaid.
        
        Args:
            matrix_a: Feature matrix of Video A (n_frames, n_features)
            matrix_b: Feature matrix of Video B (m_frames, n_features)
            alignment_path: DTW alignment path
            video_a_name: Name for Video A
            video_b_name: Name for Video B
            save_path: Optional path to save figure
            show: Whether to display
        """
        from scipy.spatial.distance import cdist
        
        # Compute distance matrix
        distance_matrix = cdist(matrix_a, matrix_b, metric='cosine')
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot heatmap
        im = ax.imshow(distance_matrix, cmap='viridis', aspect='auto', origin='lower')
        plt.colorbar(im, ax=ax, label='Cosine Distance')
        
        # Overlay alignment path
        path_a = [p[0] for p in alignment_path]
        path_b = [p[1] for p in alignment_path]
        ax.plot(path_b, path_a, 'r-', linewidth=2, label='DTW Path')
        ax.plot(path_b, path_a, 'w.', markersize=3)
        
        ax.set_xlabel(f'{video_b_name} Frame Index', fontsize=12)
        ax.set_ylabel(f'{video_a_name} Frame Index', fontsize=12)
        ax.set_title('Distance Matrix with DTW Alignment Path\n(Darker = More Similar)', fontsize=14)
        ax.legend(loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved alignment matrix to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def visualize_all_64_frames(
        self,
        video_a_path: str,
        video_b_path: str,
        n_frames: int = 64,
        video_a_name: str = "Video A",
        video_b_name: str = "Video B",
        save_path: Optional[str] = None,
        show: bool = True
    ):
        """
        Visualize all 64 frames from both videos in a grid comparison.
        
        Creates two 8x8 grids side by side showing all extracted frames.
        
        Args:
            video_a_path: Path to reference video
            video_b_path: Path to candidate video
            n_frames: Number of frames (default: 64)
            video_a_name: Name for Video A
            video_b_name: Name for Video B
            save_path: Optional path to save figure
            show: Whether to display
        """
        # Extract all frames
        indices = list(range(n_frames))
        frames_a = self.extract_frames_at_indices(video_a_path, indices)
        frames_b = self.extract_frames_at_indices(video_b_path, indices)
        
        # Determine grid size (8x8 for 64 frames)
        grid_size = int(np.ceil(np.sqrt(n_frames)))
        
        # Create figure with two grids side by side
        fig = plt.figure(figsize=(24, 12))
        
        # Add main title
        fig.suptitle(f'64-Frame Comparison\n{video_a_name} (Left) vs {video_b_name} (Right)', 
                     fontsize=16, fontweight='bold')
        
        # Left grid for Video A
        for i in range(n_frames):
            row = i // grid_size
            col = i % grid_size
            
            ax = fig.add_subplot(grid_size, grid_size * 2, row * grid_size * 2 + col + 1)
            
            if i < len(frames_a):
                # Resize frame for display
                frame = cv2.resize(frames_a[i], (80, 60))
                ax.imshow(frame)
            
            ax.axis('off')
            if i < 8:  # Only show frame numbers on top row
                ax.set_title(f'{i}', fontsize=6)
        
        # Right grid for Video B
        for i in range(n_frames):
            row = i // grid_size
            col = i % grid_size
            
            ax = fig.add_subplot(grid_size, grid_size * 2, row * grid_size * 2 + grid_size + col + 1)
            
            if i < len(frames_b):
                # Resize frame for display
                frame = cv2.resize(frames_b[i], (80, 60))
                ax.imshow(frame)
            
            ax.axis('off')
            if i < 8:  # Only show frame numbers on top row
                ax.set_title(f'{i}', fontsize=6)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved 64-frame comparison to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def visualize_frames_with_dtw_lines(
        self,
        video_a_path: str,
        video_b_path: str,
        alignment_path: List[Tuple[int, int]],
        n_frames: int = 64,
        video_a_name: str = "Video A",
        video_b_name: str = "Video B",
        save_path: Optional[str] = None,
        show: bool = True
    ):
        """
        Visualize frames with DTW alignment connections.
        
        Shows sampled frames from both videos with lines connecting
        matched frame pairs according to DTW alignment.
        
        Args:
            video_a_path: Path to reference video
            video_b_path: Path to candidate video
            alignment_path: DTW alignment path
            n_frames: Number of frames
            video_a_name: Name for Video A
            video_b_name: Name for Video B
            save_path: Optional path to save figure
            show: Whether to display
        """
        # Sample 16 evenly spaced alignment pairs for visualization
        n_display = 16
        step = max(1, len(alignment_path) // n_display)
        sampled_pairs = alignment_path[::step][:n_display]
        
        # Get unique frame indices
        indices_a = sorted(set([p[0] for p in sampled_pairs]))
        indices_b = sorted(set([p[1] for p in sampled_pairs]))
        
        # Extract frames
        frames_a = self.extract_frames_at_indices(video_a_path, indices_a)
        frames_b = self.extract_frames_at_indices(video_b_path, indices_b)
        
        # Create mapping
        idx_to_pos_a = {idx: pos for pos, idx in enumerate(indices_a)}
        idx_to_pos_b = {idx: pos for pos, idx in enumerate(indices_b)}
        
        # Create figure
        fig, axes = plt.subplots(3, 1, figsize=(20, 12), 
                                  gridspec_kw={'height_ratios': [1, 0.3, 1]})
        
        # Top row: Video A frames
        ax_a = axes[0]
        ax_a.set_xlim(0, len(indices_a))
        ax_a.set_ylim(0, 1)
        ax_a.set_title(f'{video_a_name} Frames', fontsize=14)
        ax_a.axis('off')
        
        frame_width = 0.8 / len(indices_a)
        for i, (idx, frame) in enumerate(zip(indices_a, frames_a)):
            frame_resized = cv2.resize(frame, (60, 45))
            x_pos = (i + 0.1) / len(indices_a)
            ax_a.imshow(frame_resized, extent=[i + 0.1, i + 0.9, 0.1, 0.9], aspect='auto')
            ax_a.text(i + 0.5, 0.02, f'{idx}', ha='center', fontsize=8)
        
        # Middle: Connection lines
        ax_mid = axes[1]
        ax_mid.set_xlim(0, max(len(indices_a), len(indices_b)))
        ax_mid.set_ylim(0, 1)
        ax_mid.axis('off')
        
        for idx_a, idx_b in sampled_pairs:
            if idx_a in idx_to_pos_a and idx_b in idx_to_pos_b:
                pos_a = idx_to_pos_a[idx_a] + 0.5
                pos_b = idx_to_pos_b[idx_b] + 0.5
                ax_mid.plot([pos_a, pos_b], [0.9, 0.1], 'b-', alpha=0.5, linewidth=1)
        
        ax_mid.text(0.5, 0.5, 'DTW Alignment', ha='center', va='center', 
                   fontsize=12, transform=ax_mid.transAxes)
        
        # Bottom row: Video B frames
        ax_b = axes[2]
        ax_b.set_xlim(0, len(indices_b))
        ax_b.set_ylim(0, 1)
        ax_b.set_title(f'{video_b_name} Frames', fontsize=14)
        ax_b.axis('off')
        
        for i, (idx, frame) in enumerate(zip(indices_b, frames_b)):
            frame_resized = cv2.resize(frame, (60, 45))
            ax_b.imshow(frame_resized, extent=[i + 0.1, i + 0.9, 0.1, 0.9], aspect='auto')
            ax_b.text(i + 0.5, 0.02, f'{idx}', ha='center', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved DTW alignment visualization to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def extract_frames_at_indices(
        self,
        video_path: str,
        indices: List[int]
    ) -> List[np.ndarray]:
        """
        Extract specific frames from a video.
        
        Args:
            video_path: Path to video file
            indices: List of frame indices to extract
            
        Returns:
            List of frames as numpy arrays (RGB)
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frames = []
        for idx in indices:
            if idx >= total_frames:
                idx = total_frames - 1
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
            else:
                frames.append(np.zeros((480, 640, 3), dtype=np.uint8))
        
        cap.release()
        return frames
    
    def visualize_matched_frames(
        self,
        video_a_path: str,
        video_b_path: str,
        alignment_path: List[Tuple[int, int]],
        n_pairs: int = 8,
        save_path: Optional[str] = None,
        show: bool = True
    ):
        """
        Visualize matched frame pairs side by side.
        
        Args:
            video_a_path: Path to reference video
            video_b_path: Path to candidate video
            alignment_path: DTW alignment path
            n_pairs: Number of frame pairs to show
            save_path: Optional path to save figure
            show: Whether to display
        """
        # Sample evenly spaced pairs from alignment path
        total_pairs = len(alignment_path)
        if total_pairs <= n_pairs:
            selected_indices = list(range(total_pairs))
        else:
            step = total_pairs // n_pairs
            selected_indices = [i * step for i in range(n_pairs)]
        
        selected_pairs = [alignment_path[i] for i in selected_indices]
        
        # Extract frame indices
        frames_a_indices = [p[0] for p in selected_pairs]
        frames_b_indices = [p[1] for p in selected_pairs]
        
        # Get actual frames from videos
        frames_a = self.extract_frames_at_indices(video_a_path, frames_a_indices)
        frames_b = self.extract_frames_at_indices(video_b_path, frames_b_indices)
        
        # Create visualization
        n_cols = min(4, n_pairs)
        n_rows = (n_pairs + n_cols - 1) // n_cols * 2  # *2 for pairs
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
        if n_rows == 2 and n_cols == 1:
            axes = axes.reshape(2, 1)
        elif n_rows == 2:
            axes = axes.reshape(2, -1)
        
        video_a_name = Path(video_a_path).stem
        video_b_name = Path(video_b_path).stem
        
        fig.suptitle(f'DTW Matched Frames\n{video_a_name} (top) ↔ {video_b_name} (bottom)', fontsize=14)
        
        for i, (frame_a, frame_b, (idx_a, idx_b)) in enumerate(zip(frames_a, frames_b, selected_pairs)):
            col = i % n_cols
            row_a = (i // n_cols) * 2
            row_b = row_a + 1
            
            if n_rows == 2:
                ax_a = axes[0, col] if n_cols > 1 else axes[0, 0]
                ax_b = axes[1, col] if n_cols > 1 else axes[1, 0]
            else:
                ax_a = axes[row_a, col]
                ax_b = axes[row_b, col]
            
            # Show frames
            ax_a.imshow(frame_a)
            ax_a.set_title(f'A[{idx_a}]', fontsize=10)
            ax_a.axis('off')
            
            ax_b.imshow(frame_b)
            ax_b.set_title(f'B[{idx_b}]', fontsize=10)
            ax_b.axis('off')
        
        # Hide unused axes
        for i in range(len(selected_pairs), n_cols * (n_rows // 2)):
            col = i % n_cols
            row_a = (i // n_cols) * 2
            row_b = row_a + 1
            if row_a < n_rows and col < n_cols:
                if n_rows == 2:
                    axes[0, col].axis('off')
                    axes[1, col].axis('off')
                else:
                    axes[row_a, col].axis('off')
                    axes[row_b, col].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved matched frames to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def create_full_visualization(
        self,
        video_a_path: str,
        video_b_path: str,
        matrix_a: np.ndarray,
        matrix_b: np.ndarray,
        alignment_path: List[Tuple[int, int]],
        similarity_score: float,
        save_dir: Optional[str] = None,
        show: bool = True
    ):
        """
        Create complete DTW visualization suite.
        
        Args:
            video_a_path: Path to reference video
            video_b_path: Path to candidate video
            matrix_a: Feature matrix of Video A
            matrix_b: Feature matrix of Video B
            alignment_path: DTW alignment path
            similarity_score: DTW similarity score
            save_dir: Directory to save all visualizations
            show: Whether to display plots
        """
        video_a_name = Path(video_a_path).stem
        video_b_name = Path(video_b_path).stem
        
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"DTW VISUALIZATION")
        print(f"{'='*60}")
        print(f"Reference: {video_a_name}")
        print(f"Candidate: {video_b_name}")
        print(f"Similarity: {similarity_score:.4f}")
        print(f"Alignment path length: {len(alignment_path)} pairs")
        print(f"{'='*60}")
        
        # 1. All 64 frames comparison
        print("\n[1/4] Generating 64-frame comparison grid...")
        self.visualize_all_64_frames(
            video_a_path,
            video_b_path,
            n_frames=matrix_a.shape[0],
            video_a_name=video_a_name,
            video_b_name=video_b_name,
            save_path=str(save_dir / f"all_frames_{video_a_name}_vs_{video_b_name}.png") if save_dir else None,
            show=show
        )
        
        # 2. Warping path plot
        print("[2/4] Generating warping path plot...")
        self.plot_warping_path(
            alignment_path,
            matrix_a.shape[0],
            matrix_b.shape[0],
            video_a_name,
            video_b_name,
            save_path=str(save_dir / f"warping_path_{video_a_name}_vs_{video_b_name}.png") if save_dir else None,
            show=show
        )
        
        # 3. Distance matrix with path
        print("[3/4] Generating distance matrix heatmap...")
        self.plot_alignment_matrix(
            matrix_a,
            matrix_b,
            alignment_path,
            video_a_name,
            video_b_name,
            save_path=str(save_dir / f"distance_matrix_{video_a_name}_vs_{video_b_name}.png") if save_dir else None,
            show=show
        )
        
        # 4. Matched frames
        print("[4/4] Generating matched frames visualization...")
        self.visualize_matched_frames(
            video_a_path,
            video_b_path,
            alignment_path,
            n_pairs=8,
            save_path=str(save_dir / f"matched_frames_{video_a_name}_vs_{video_b_name}.png") if save_dir else None,
            show=show
        )
        
        print(f"\n✅ Visualization complete!")
        if save_dir:
            print(f"📁 Saved to: {save_dir}")
