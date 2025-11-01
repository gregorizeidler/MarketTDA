"""
Point Cloud Construction for Topological Data Analysis
Transforms time series returns into high-dimensional point clouds
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class MarketPointCloud:
    """Transform market returns into point clouds for TDA"""
    
    def __init__(self, returns: pd.DataFrame):
        """
        Initialize point cloud generator
        
        Args:
            returns: DataFrame of returns (rows=time, columns=assets)
        """
        self.returns = returns
        self.point_clouds = []
        self.windows = []
        self.dates = []
        
    def create_point_cloud(
        self, 
        normalize: bool = True,
        window_size: Optional[int] = None
    ) -> np.ndarray:
        """
        Create a single point cloud from all returns data
        Each point = vector of N asset returns on a given day
        
        Args:
            normalize: Whether to standardize returns
            window_size: If specified, only use last window_size days
            
        Returns:
            Point cloud array (n_days × n_assets)
        """
        data = self.returns.copy()
        
        if window_size:
            data = data.iloc[-window_size:]
        
        # Each row is a point in N-dimensional space
        point_cloud = data.values
        
        if normalize:
            scaler = StandardScaler()
            point_cloud = scaler.fit_transform(point_cloud)
        
        return point_cloud
    
    def create_sliding_windows(
        self,
        window_size: int = 60,
        step_size: int = 10,
        normalize: bool = True
    ) -> List[Tuple[np.ndarray, pd.Timestamp]]:
        """
        Create sliding window point clouds over time
        This enables temporal analysis of market topology
        
        Args:
            window_size: Number of days in each window
            step_size: Days to slide forward
            normalize: Whether to standardize within each window
            
        Returns:
            List of (point_cloud, end_date) tuples
        """
        print(f"\n🪟 Creating sliding windows...")
        print(f"   Window size: {window_size} days")
        print(f"   Step size: {step_size} days")
        
        n_periods = len(self.returns)
        windows = []
        
        for i in range(0, n_periods - window_size + 1, step_size):
            window_data = self.returns.iloc[i:i+window_size]
            end_date = window_data.index[-1]
            
            # Create point cloud for this window
            point_cloud = window_data.values
            
            if normalize:
                scaler = StandardScaler()
                point_cloud = scaler.fit_transform(point_cloud)
            
            windows.append((point_cloud, end_date))
        
        self.point_clouds = [w[0] for w in windows]
        self.dates = [w[1] for w in windows]
        self.windows = windows
        
        print(f"✅ Created {len(windows)} windows")
        print(f"   Date range: {self.dates[0]} to {self.dates[-1]}")
        print(f"   Points per window: {window_size}")
        print(f"   Dimensions: {self.returns.shape[1]}")
        
        return windows
    
    def visualize_point_cloud_3d(
        self,
        window_idx: int = -1,
        method: str = "pca",
        save_path: Optional[str] = None
    ):
        """
        Visualize a point cloud in 3D using dimensionality reduction
        
        Args:
            window_idx: Which window to visualize (-1 for last)
            method: Dimensionality reduction method ('pca', 'tsne')
            save_path: Path to save figure
        """
        if not self.point_clouds:
            print("⚠️ No point clouds available. Run create_sliding_windows() first.")
            return
        
        point_cloud = self.point_clouds[window_idx]
        date = self.dates[window_idx]
        
        # Reduce to 3D
        if method == "pca":
            reducer = PCA(n_components=3)
            points_3d = reducer.fit_transform(point_cloud)
            variance_explained = reducer.explained_variance_ratio_.sum()
            title_suffix = f"(PCA - {variance_explained:.1%} variance)"
        else:
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=3, random_state=42)
            points_3d = reducer.fit_transform(point_cloud)
            title_suffix = "(t-SNE)"
        
        # Create 3D plot
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        # Color by time (early to late in window)
        colors = np.arange(len(points_3d))
        scatter = ax.scatter(
            points_3d[:, 0], 
            points_3d[:, 1], 
            points_3d[:, 2],
            c=colors,
            cmap='viridis',
            s=50,
            alpha=0.6,
            edgecolors='k',
            linewidth=0.5
        )
        
        ax.set_xlabel('Component 1', fontsize=12)
        ax.set_ylabel('Component 2', fontsize=12)
        ax.set_zlabel('Component 3', fontsize=12)
        ax.set_title(f'Market Point Cloud - {date.date()} {title_suffix}', 
                    fontsize=14, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
        cbar.set_label('Time in Window (Days)', fontsize=10)
        
        # Add info text
        info_text = (f"Dimensions: {point_cloud.shape[1]}\n"
                    f"Points: {point_cloud.shape[0]}\n"
                    f"Window: {window_idx}")
        ax.text2D(0.02, 0.98, info_text, transform=ax.transAxes,
                 fontsize=9, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Saved to {save_path}")
        
        plt.show()
    
    def compute_pairwise_distances(
        self, 
        window_idx: int = -1,
        metric: str = "euclidean"
    ) -> np.ndarray:
        """
        Compute pairwise distance matrix for a point cloud
        This is used as input to persistent homology algorithms
        
        Args:
            window_idx: Which window to analyze
            metric: Distance metric ('euclidean', 'correlation', etc.)
            
        Returns:
            Distance matrix (n_points × n_points)
        """
        point_cloud = self.point_clouds[window_idx]
        
        if metric == "euclidean":
            from scipy.spatial.distance import pdist, squareform
            distances = squareform(pdist(point_cloud, metric='euclidean'))
        elif metric == "correlation":
            # Correlation-based distance
            corr_matrix = np.corrcoef(point_cloud)
            distances = np.sqrt(2 * (1 - corr_matrix))
        else:
            from scipy.spatial.distance import pdist, squareform
            distances = squareform(pdist(point_cloud, metric=metric))
        
        return distances
    
    def get_cloud_statistics(self, window_idx: int = -1) -> dict:
        """
        Compute basic statistics about a point cloud
        
        Args:
            window_idx: Which window to analyze
            
        Returns:
            Dictionary of statistics
        """
        point_cloud = self.point_clouds[window_idx]
        distances = self.compute_pairwise_distances(window_idx)
        
        # Compute statistics
        stats = {
            'n_points': point_cloud.shape[0],
            'n_dimensions': point_cloud.shape[1],
            'mean_distance': distances[np.triu_indices_from(distances, k=1)].mean(),
            'max_distance': distances.max(),
            'min_distance': distances[np.triu_indices_from(distances, k=1)].min(),
            'std_distance': distances[np.triu_indices_from(distances, k=1)].std(),
            'diameter': distances.max(),  # Maximum pairwise distance
        }
        
        # PCA analysis
        pca = PCA()
        pca.fit(point_cloud)
        stats['intrinsic_dim_90'] = np.argmax(
            np.cumsum(pca.explained_variance_ratio_) >= 0.90
        ) + 1
        stats['intrinsic_dim_95'] = np.argmax(
            np.cumsum(pca.explained_variance_ratio_) >= 0.95
        ) + 1
        
        return stats


if __name__ == "__main__":
    # Example usage
    from data_fetcher import MarketDataFetcher
    
    print("="*60)
    print("POINT CLOUD CONSTRUCTION DEMO")
    print("="*60)
    
    # Fetch data
    fetcher = MarketDataFetcher()
    data = fetcher.fetch_data(period="1y", max_tickers=50)
    returns = fetcher.compute_returns()
    
    # Create point clouds
    pc = MarketPointCloud(returns)
    windows = pc.create_sliding_windows(window_size=60, step_size=20)
    
    # Analyze latest window
    print("\n" + "="*60)
    print("POINT CLOUD STATISTICS (Latest Window)")
    print("="*60)
    stats = pc.get_cloud_statistics(-1)
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Visualize
    print("\n📊 Generating 3D visualization...")
    pc.visualize_point_cloud_3d()

