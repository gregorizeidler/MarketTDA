"""
Advanced Topological Metrics for Persistence Diagrams
Wasserstein distances, Bottleneck distances, Persistence Landscapes, Persistence Images
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class AdvancedTopologicalMetrics:
    """Advanced metrics for comparing and analyzing persistence diagrams"""
    
    def __init__(self):
        """Initialize advanced metrics calculator"""
        self.colors = {
            0: '#1f77b4',  # Blue for H0
            1: '#ff7f0e',  # Orange for H1
            2: '#2ca02c',  # Green for H2
        }
    
    # ==================== DISTANCE METRICS ====================
    
    def bottleneck_distance(
        self,
        dgm1: np.ndarray,
        dgm2: np.ndarray,
        dim: int = 1
    ) -> float:
        """
        Compute bottleneck distance between two persistence diagrams
        (simplified implementation - for production use persim library)
        
        Args:
            dgm1: First persistence diagram
            dgm2: Second persistence diagram
            dim: Homology dimension to compare
            
        Returns:
            Bottleneck distance
        """
        # Filter by dimension
        mask1 = dgm1[:, 2] == dim
        mask2 = dgm2[:, 2] == dim
        
        d1 = dgm1[mask1, :2]  # (birth, death)
        d2 = dgm2[mask2, :2]
        
        if len(d1) == 0 and len(d2) == 0:
            return 0.0
        if len(d1) == 0 or len(d2) == 0:
            # Distance to diagonal
            non_empty = d1 if len(d1) > 0 else d2
            persistences = non_empty[:, 1] - non_empty[:, 0]
            return np.max(persistences) / 2
        
        # Compute pairwise distances (simplified)
        # Include projection to diagonal
        n1, n2 = len(d1), len(d2)
        
        # Distance matrix
        dist_matrix = np.zeros((n1, n2))
        for i in range(n1):
            for j in range(n2):
                dist_matrix[i, j] = self._point_to_point_distance(d1[i], d2[j])
        
        # Add diagonal projections
        diag_dist1 = np.array([(d1[i, 1] - d1[i, 0]) / 2 for i in range(n1)])
        diag_dist2 = np.array([(d2[j, 1] - d2[j, 0]) / 2 for j in range(n2)])
        
        # Simplified bottleneck (max of min distances)
        max_dist = 0.0
        for i in range(n1):
            min_dist = min(np.min(dist_matrix[i, :]), diag_dist1[i])
            max_dist = max(max_dist, min_dist)
        for j in range(n2):
            min_dist = min(np.min(dist_matrix[:, j]), diag_dist2[j])
            max_dist = max(max_dist, min_dist)
        
        return max_dist
    
    def wasserstein_distance(
        self,
        dgm1: np.ndarray,
        dgm2: np.ndarray,
        dim: int = 1,
        p: int = 2
    ) -> float:
        """
        Compute p-Wasserstein distance between persistence diagrams
        
        Args:
            dgm1: First persistence diagram
            dgm2: Second persistence diagram
            dim: Homology dimension
            p: Wasserstein exponent (default: 2)
            
        Returns:
            p-Wasserstein distance
        """
        # Filter by dimension
        mask1 = dgm1[:, 2] == dim
        mask2 = dgm2[:, 2] == dim
        
        d1 = dgm1[mask1, :2]
        d2 = dgm2[mask2, :2]
        
        if len(d1) == 0 and len(d2) == 0:
            return 0.0
        if len(d1) == 0 or len(d2) == 0:
            non_empty = d1 if len(d1) > 0 else d2
            persistences = non_empty[:, 1] - non_empty[:, 0]
            return (np.sum(persistences**p) / 2**p) ** (1/p)
        
        # Build cost matrix (including diagonal projections)
        n1, n2 = len(d1), len(d2)
        max_size = max(n1, n2)
        
        # Pad with diagonal points if needed
        d1_padded = np.vstack([d1, np.zeros((max_size - n1, 2))])
        d2_padded = np.vstack([d2, np.zeros((max_size - n2, 2))])
        
        # Add diagonal projections
        if n1 < max_size:
            for i in range(n1, max_size):
                # Use diagonal points from d2
                mid = (d2[i - n1, 0] + d2[i - n1, 1]) / 2
                d1_padded[i] = [mid, mid]
        
        if n2 < max_size:
            for j in range(n2, max_size):
                mid = (d1[j - n2, 0] + d1[j - n2, 1]) / 2
                d2_padded[j] = [mid, mid]
        
        # Cost matrix
        cost_matrix = np.zeros((max_size, max_size))
        for i in range(max_size):
            for j in range(max_size):
                cost_matrix[i, j] = self._point_to_point_distance(
                    d1_padded[i], d2_padded[j]
                ) ** p
        
        # Optimal matching (Hungarian algorithm)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        total_cost = cost_matrix[row_ind, col_ind].sum()
        
        return total_cost ** (1/p)
    
    def _point_to_point_distance(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """L-infinity distance between points in persistence diagram"""
        return max(abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
    
    def compute_distance_matrix(
        self,
        diagrams: List[np.ndarray],
        metric: str = 'wasserstein',
        dim: int = 1
    ) -> np.ndarray:
        """
        Compute pairwise distance matrix between all persistence diagrams
        
        Args:
            diagrams: List of persistence diagrams
            metric: 'wasserstein' or 'bottleneck'
            dim: Homology dimension
            
        Returns:
            Distance matrix (n_diagrams x n_diagrams)
        """
        n = len(diagrams)
        dist_matrix = np.zeros((n, n))
        
        distance_func = (self.wasserstein_distance if metric == 'wasserstein'
                        else self.bottleneck_distance)
        
        for i in range(n):
            for j in range(i+1, n):
                dist = distance_func(diagrams[i], diagrams[j], dim=dim)
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
        
        return dist_matrix
    
    # ==================== PERSISTENCE LANDSCAPES ====================
    
    def persistence_landscape(
        self,
        diagram: np.ndarray,
        dim: int = 1,
        k: int = 5,
        num_points: int = 100
    ) -> np.ndarray:
        """
        Compute k-th persistence landscape function
        
        Args:
            diagram: Persistence diagram
            dim: Homology dimension
            k: Landscape level (1 = max, 2 = second max, etc.)
            num_points: Resolution of landscape
            
        Returns:
            Landscape function values (num_points,)
        """
        # Filter by dimension
        mask = diagram[:, 2] == dim
        dgm = diagram[mask, :2]
        
        if len(dgm) == 0:
            return np.zeros(num_points)
        
        # Create grid
        t_min = np.min(dgm[:, 0])
        t_max = np.max(dgm[:, 1])
        t_grid = np.linspace(t_min, t_max, num_points)
        
        # Compute landscape values
        landscape = np.zeros(num_points)
        
        for i, t in enumerate(t_grid):
            # For each point, compute tent function values
            tent_values = []
            for birth, death in dgm:
                if birth <= t <= death:
                    # Tent function: min(t - birth, death - t)
                    tent_val = min(t - birth, death - t)
                    tent_values.append(tent_val)
            
            # k-th largest value (0-indexed)
            if len(tent_values) >= k:
                tent_values.sort(reverse=True)
                landscape[i] = tent_values[k-1]
        
        return landscape
    
    def compute_all_landscapes(
        self,
        diagram: np.ndarray,
        dim: int = 1,
        k_max: int = 5,
        num_points: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute multiple landscape levels
        
        Returns:
            landscapes: Array of shape (k_max, num_points)
            t_grid: Time grid
        """
        mask = diagram[:, 2] == dim
        dgm = diagram[mask, :2]
        
        if len(dgm) == 0:
            t_grid = np.linspace(0, 1, num_points)
            return np.zeros((k_max, num_points)), t_grid
        
        t_min = np.min(dgm[:, 0])
        t_max = np.max(dgm[:, 1])
        t_grid = np.linspace(t_min, t_max, num_points)
        
        landscapes = np.zeros((k_max, num_points))
        for k in range(1, k_max + 1):
            landscapes[k-1] = self.persistence_landscape(
                diagram, dim=dim, k=k, num_points=num_points
            )
        
        return landscapes, t_grid
    
    # ==================== PERSISTENCE IMAGES ====================
    
    def persistence_image(
        self,
        diagram: np.ndarray,
        dim: int = 1,
        resolution: Tuple[int, int] = (20, 20),
        sigma: float = 0.1
    ) -> np.ndarray:
        """
        Convert persistence diagram to persistence image (for ML)
        
        Args:
            diagram: Persistence diagram
            dim: Homology dimension
            resolution: Image resolution (height, width)
            sigma: Gaussian smoothing parameter
            
        Returns:
            Persistence image (2D array)
        """
        # Filter by dimension
        mask = diagram[:, 2] == dim
        dgm = diagram[mask, :2]
        
        if len(dgm) == 0:
            return np.zeros(resolution)
        
        # Compute persistence and midpoint
        births = dgm[:, 0]
        deaths = dgm[:, 1]
        persistences = deaths - births
        midpoints = (births + deaths) / 2
        
        # Create grid
        birth_min, birth_max = births.min(), deaths.max()
        pers_min, pers_max = 0, persistences.max()
        
        # Add padding
        padding = 0.1
        birth_range = birth_max - birth_min
        pers_range = pers_max - pers_min
        
        birth_grid = np.linspace(
            birth_min - padding * birth_range,
            birth_max + padding * birth_range,
            resolution[1]
        )
        pers_grid = np.linspace(
            pers_min - padding * pers_range,
            pers_max + padding * pers_range,
            resolution[0]
        )
        
        # Initialize image
        image = np.zeros(resolution)
        
        # Weight function (persistence weighting)
        weights = persistences
        
        # Gaussian weighting
        for i in range(len(dgm)):
            birth = midpoints[i]
            pers = persistences[i]
            weight = weights[i]
            
            # Add Gaussian to image
            for xi, x in enumerate(birth_grid):
                for yi, y in enumerate(pers_grid):
                    dist_sq = (x - birth)**2 + (y - pers)**2
                    image[yi, xi] += weight * np.exp(-dist_sq / (2 * sigma**2))
        
        return image
    
    # ==================== VISUALIZATIONS ====================
    
    def plot_distance_matrix(
        self,
        distance_matrix: np.ndarray,
        dates: List[pd.Timestamp],
        metric: str = 'Wasserstein',
        dim: int = 1,
        save_path: Optional[str] = None
    ):
        """Plot heatmap of distance matrix"""
        plt.figure(figsize=(12, 10))
        
        # Format dates
        date_labels = [d.strftime('%Y-%m-%d') for d in dates]
        
        # Plot heatmap
        sns.heatmap(
            distance_matrix,
            cmap='viridis',
            xticklabels=date_labels,
            yticklabels=date_labels,
            cbar_kws={'label': f'{metric} Distance'},
            square=True,
            linewidths=0.5,
            linecolor='gray'
        )
        
        plt.title(
            f'{metric} Distance Matrix - H{dim}\n'
            f'Temporal Evolution of Topological Similarity',
            fontsize=14, fontweight='bold', pad=20
        )
        plt.xlabel('Date', fontsize=12, fontweight='bold')
        plt.ylabel('Date', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Saved to {save_path}")
        
        plt.show()
    
    def plot_persistence_landscapes(
        self,
        diagram: np.ndarray,
        dim: int = 1,
        k_max: int = 5,
        save_path: Optional[str] = None
    ):
        """Plot persistence landscape functions"""
        landscapes, t_grid = self.compute_all_landscapes(
            diagram, dim=dim, k_max=k_max
        )
        
        plt.figure(figsize=(14, 8))
        
        colors = plt.cm.viridis(np.linspace(0, 0.9, k_max))
        
        for k in range(k_max):
            plt.plot(
                t_grid, landscapes[k],
                label=f'λ_{k+1}',
                linewidth=2,
                color=colors[k],
                alpha=0.8
            )
            plt.fill_between(
                t_grid, landscapes[k],
                alpha=0.2,
                color=colors[k]
            )
        
        plt.xlabel('Filtration Value', fontsize=12, fontweight='bold')
        plt.ylabel('Landscape Value', fontsize=12, fontweight='bold')
        plt.title(
            f'Persistence Landscapes - H{dim}\n'
            f'Functional Representation of Topology',
            fontsize=14, fontweight='bold'
        )
        plt.legend(fontsize=10, loc='upper right')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Saved to {save_path}")
        
        plt.show()
    
    def plot_persistence_images(
        self,
        diagrams: List[np.ndarray],
        dates: List[pd.Timestamp],
        dim: int = 1,
        resolution: Tuple[int, int] = (20, 20),
        save_path: Optional[str] = None
    ):
        """Plot grid of persistence images over time"""
        n_diagrams = len(diagrams)
        n_cols = min(5, n_diagrams)
        n_rows = (n_diagrams + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        for i, (diagram, date) in enumerate(zip(diagrams, dates)):
            if i >= len(axes):
                break
            
            img = self.persistence_image(diagram, dim=dim, resolution=resolution)
            
            im = axes[i].imshow(img, cmap='hot', interpolation='bilinear', origin='lower')
            axes[i].set_title(date.strftime('%Y-%m-%d'), fontsize=10)
            axes[i].axis('off')
            plt.colorbar(im, ax=axes[i], fraction=0.046)
        
        # Hide unused subplots
        for i in range(n_diagrams, len(axes)):
            axes[i].axis('off')
        
        fig.suptitle(
            f'Persistence Images - H{dim}\n'
            f'Vectorized Topology for Machine Learning',
            fontsize=14, fontweight='bold'
        )
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Saved to {save_path}")
        
        plt.show()
    
    def plot_3d_persistence_surface(
        self,
        diagram: np.ndarray,
        dim: int = 1,
        save_path: Optional[str] = None
    ):
        """Plot 3D surface of persistence (birth, death, persistence)"""
        from mpl_toolkits.mplot3d import Axes3D
        
        # Filter by dimension
        mask = diagram[:, 2] == dim
        dgm = diagram[mask, :2]
        
        if len(dgm) == 0:
            print("⚠️ No features to plot")
            return
        
        births = dgm[:, 0]
        deaths = dgm[:, 1]
        persistences = deaths - births
        
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create surface
        scatter = ax.scatter(
            births, deaths, persistences,
            c=persistences,
            cmap='viridis',
            s=100,
            alpha=0.6,
            edgecolors='black',
            linewidth=1
        )
        
        # Plot diagonal plane (birth = death)
        b_range = np.linspace(births.min(), deaths.max(), 20)
        d_range = b_range
        B, D = np.meshgrid(b_range, d_range)
        Z = np.zeros_like(B)
        ax.plot_surface(B, D, Z, alpha=0.2, color='red')
        
        ax.set_xlabel('Birth', fontsize=12, fontweight='bold')
        ax.set_ylabel('Death', fontsize=12, fontweight='bold')
        ax.set_zlabel('Persistence', fontsize=12, fontweight='bold')
        ax.set_title(
            f'3D Persistence Surface - H{dim}\n'
            f'Birth × Death × Persistence',
            fontsize=14, fontweight='bold'
        )
        
        plt.colorbar(scatter, ax=ax, label='Persistence', shrink=0.5)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Saved to {save_path}")
        
        plt.show()


if __name__ == "__main__":
    print("="*60)
    print("ADVANCED TOPOLOGICAL METRICS DEMO")
    print("="*60)
    
    # Create sample diagrams
    np.random.seed(42)
    
    dgm1 = np.array([
        [0.1, 0.5, 1],
        [0.2, 0.8, 1],
        [0.3, 0.6, 1],
    ])
    
    dgm2 = np.array([
        [0.15, 0.55, 1],
        [0.25, 0.75, 1],
    ])
    
    metrics = AdvancedTopologicalMetrics()
    
    # Test distances
    print("\n📏 Distance Metrics:")
    w_dist = metrics.wasserstein_distance(dgm1, dgm2, dim=1)
    b_dist = metrics.bottleneck_distance(dgm1, dgm2, dim=1)
    print(f"   Wasserstein Distance: {w_dist:.4f}")
    print(f"   Bottleneck Distance: {b_dist:.4f}")
    
    # Test landscapes
    print("\n🏔️ Persistence Landscapes:")
    landscapes, t_grid = metrics.compute_all_landscapes(dgm1, dim=1, k_max=3)
    print(f"   Generated {len(landscapes)} landscape levels")
    print(f"   Resolution: {len(t_grid)} points")
    
    # Test persistence image
    print("\n🖼️ Persistence Image:")
    img = metrics.persistence_image(dgm1, dim=1, resolution=(20, 20))
    print(f"   Image shape: {img.shape}")
    print(f"   Max intensity: {img.max():.4f}")
    
    print("\n✅ All metrics computed successfully!")

