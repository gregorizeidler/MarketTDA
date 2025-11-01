"""
Ultra-Advanced Topological Metrics
Persistent Entropy 2D, Silhouette, Landscape Norms, Kernel Methods
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import rbf_kernel
import warnings
warnings.filterwarnings('ignore')


class UltraAdvancedMetrics:
    """Ultra-advanced topological metrics for cutting-edge research"""
    
    def __init__(self):
        """Initialize ultra-advanced metrics calculator"""
        self.colors = {0: '#3498DB', 1: '#E74C3C', 2: '#2ECC71'}
    
    # ==================== PERSISTENT ENTROPY 2D ====================
    
    def persistent_entropy_2d(
        self,
        diagrams: List[np.ndarray],
        dates: List[pd.Timestamp]
    ) -> pd.DataFrame:
        """
        Compute 2D persistent entropy (entropy per dimension over time)
        
        Args:
            diagrams: List of persistence diagrams
            dates: Corresponding dates
            
        Returns:
            DataFrame with entropy values per dimension over time
        """
        print("\n🔬 Computing 2D Persistent Entropy...")
        
        results = []
        for date, diagram in zip(dates, diagrams):
            row = {'date': date}
            
            for dim in [0, 1, 2]:
                mask = diagram[:, 2] == dim
                dgm_dim = diagram[mask, :2]
                
                if len(dgm_dim) > 0:
                    # Compute persistence values
                    persistences = dgm_dim[:, 1] - dgm_dim[:, 0]
                    
                    # Normalize to probabilities
                    if persistences.sum() > 0:
                        probs = persistences / persistences.sum()
                        # Shannon entropy
                        ent = -np.sum(probs * np.log(probs + 1e-10))
                    else:
                        ent = 0.0
                else:
                    ent = 0.0
                
                row[f'H{dim}_entropy'] = ent
            
            results.append(row)
        
        df = pd.DataFrame(results)
        df.set_index('date', inplace=True)
        
        print(f"✅ Computed entropy for {len(df)} time points")
        return df
    
    def plot_persistent_entropy_2d(
        self,
        entropy_df: pd.DataFrame,
        save_path: Optional[str] = None
    ):
        """Plot 2D heatmap of persistent entropy evolution"""
        fig, ax = plt.subplots(figsize=(16, 6))
        
        # Prepare data for heatmap
        data = entropy_df[['H0_entropy', 'H1_entropy', 'H2_entropy']].T
        
        # Create heatmap
        sns.heatmap(
            data,
            cmap='YlOrRd',
            cbar_kws={'label': 'Persistent Entropy'},
            xticklabels=[d.strftime('%Y-%m-%d') for d in entropy_df.index[::max(1, len(entropy_df)//10)]],
            yticklabels=['H₀ (Clusters)', 'H₁ (Loops)', 'H₂ (Voids)'],
            linewidths=0.5,
            ax=ax
        )
        
        ax.set_title(
            '2D Persistent Entropy Evolution\n'
            'Information Content of Topological Features Over Time',
            fontsize=14, fontweight='bold', pad=20
        )
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Homology Dimension', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Saved to {save_path}")
        
        plt.show()
    
    # ==================== SILHOUETTE OF PERSISTENCE ====================
    
    def silhouette_of_persistence(
        self,
        diagram: np.ndarray,
        dim: int = 1,
        resolution: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute silhouette of persistence diagram
        
        The silhouette is the mean of all landscape functions
        
        Args:
            diagram: Persistence diagram
            dim: Homology dimension
            resolution: Number of points
            
        Returns:
            t_values, silhouette_values
        """
        mask = diagram[:, 2] == dim
        dgm = diagram[mask, :2]
        
        if len(dgm) == 0:
            return np.array([]), np.array([])
        
        # Get time range
        t_min = dgm[:, 0].min()
        t_max = dgm[:, 1].max()
        t_values = np.linspace(t_min, t_max, resolution)
        
        # For each time point, compute average tent function
        silhouette = np.zeros(resolution)
        
        for i, t in enumerate(t_values):
            tent_values = []
            for birth, death in dgm:
                if birth <= t <= death:
                    # Tent function height
                    tent_val = min(t - birth, death - t)
                    tent_values.append(tent_val)
            
            if tent_values:
                silhouette[i] = np.mean(tent_values)
        
        return t_values, silhouette
    
    def plot_silhouette(
        self,
        diagram: np.ndarray,
        dim: int = 1,
        save_path: Optional[str] = None
    ):
        """Plot silhouette of persistence"""
        t_values, silhouette = self.silhouette_of_persistence(diagram, dim)
        
        if len(t_values) == 0:
            print("⚠️ No features to plot")
            return
        
        plt.figure(figsize=(12, 6))
        
        plt.fill_between(t_values, silhouette, alpha=0.6, color=self.colors[dim])
        plt.plot(t_values, silhouette, linewidth=2, color=self.colors[dim], label=f'H{dim} Silhouette')
        
        plt.xlabel('Filtration Value', fontsize=12, fontweight='bold')
        plt.ylabel('Average Persistence', fontsize=12, fontweight='bold')
        plt.title(
            f'Silhouette of Persistence - H{dim}\n'
            'Mean of All Persistence Landscapes',
            fontsize=14, fontweight='bold'
        )
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Saved to {save_path}")
        
        plt.show()
    
    # ==================== LANDSCAPE NORMS ====================
    
    def landscape_norms(
        self,
        diagram: np.ndarray,
        dim: int = 1,
        k: int = 1,
        resolution: int = 100
    ) -> Dict[str, float]:
        """
        Compute L1, L2, and L∞ norms of persistence landscape
        
        Args:
            diagram: Persistence diagram
            dim: Homology dimension
            k: Landscape level
            resolution: Number of discretization points
            
        Returns:
            Dictionary with L1, L2, Linf norms
        """
        mask = diagram[:, 2] == dim
        dgm = diagram[mask, :2]
        
        if len(dgm) == 0:
            return {'L1': 0.0, 'L2': 0.0, 'Linf': 0.0}
        
        # Compute landscape
        t_min = dgm[:, 0].min()
        t_max = dgm[:, 1].max()
        t_grid = np.linspace(t_min, t_max, resolution)
        dt = (t_max - t_min) / resolution
        
        landscape = np.zeros(resolution)
        
        for i, t in enumerate(t_grid):
            tent_values = []
            for birth, death in dgm:
                if birth <= t <= death:
                    tent_val = min(t - birth, death - t)
                    tent_values.append(tent_val)
            
            if len(tent_values) >= k:
                tent_values.sort(reverse=True)
                landscape[i] = tent_values[k-1]
        
        # Compute norms
        L1 = np.sum(np.abs(landscape)) * dt
        L2 = np.sqrt(np.sum(landscape**2) * dt)
        Linf = np.max(np.abs(landscape))
        
        return {
            'L1': L1,
            'L2': L2,
            'Linf': Linf
        }
    
    def compute_all_landscape_norms(
        self,
        diagrams: List[np.ndarray],
        dates: List[pd.Timestamp],
        dim: int = 1
    ) -> pd.DataFrame:
        """Compute landscape norms for all diagrams"""
        print(f"\n📏 Computing Landscape Norms for H{dim}...")
        
        results = []
        for date, diagram in zip(dates, diagrams):
            norms = self.landscape_norms(diagram, dim=dim)
            results.append({
                'date': date,
                f'H{dim}_L1': norms['L1'],
                f'H{dim}_L2': norms['L2'],
                f'H{dim}_Linf': norms['Linf']
            })
        
        df = pd.DataFrame(results)
        df.set_index('date', inplace=True)
        
        print(f"✅ Computed norms for {len(df)} diagrams")
        return df
    
    # ==================== PERSISTENCE FISHER KERNEL ====================
    
    def persistence_fisher_kernel(
        self,
        diagram1: np.ndarray,
        diagram2: np.ndarray,
        dim: int = 1,
        sigma: float = 1.0
    ) -> float:
        """
        Compute Persistence Fisher Kernel between two diagrams
        
        This is a kernel method that measures similarity
        using RBF kernel on persistence images
        
        Args:
            diagram1: First persistence diagram
            diagram2: Second persistence diagram
            dim: Homology dimension
            sigma: RBF kernel bandwidth
            
        Returns:
            Kernel similarity value
        """
        from advanced_metrics import AdvancedTopologicalMetrics
        
        metrics = AdvancedTopologicalMetrics()
        
        # Convert to persistence images
        img1 = metrics.persistence_image(diagram1, dim=dim, resolution=(20, 20))
        img2 = metrics.persistence_image(diagram2, dim=dim, resolution=(20, 20))
        
        # Flatten
        vec1 = img1.flatten().reshape(1, -1)
        vec2 = img2.flatten().reshape(1, -1)
        
        # Compute RBF kernel
        kernel_val = rbf_kernel(vec1, vec2, gamma=1.0/(2*sigma**2))[0, 0]
        
        return kernel_val
    
    def compute_kernel_matrix(
        self,
        diagrams: List[np.ndarray],
        dim: int = 1,
        sigma: float = 1.0
    ) -> np.ndarray:
        """Compute full kernel matrix for all diagrams"""
        print(f"\n🔬 Computing Persistence Fisher Kernel Matrix...")
        
        n = len(diagrams)
        kernel_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i, n):
                k_val = self.persistence_fisher_kernel(
                    diagrams[i], diagrams[j], dim=dim, sigma=sigma
                )
                kernel_matrix[i, j] = k_val
                kernel_matrix[j, i] = k_val
        
        print(f"✅ Computed {n}x{n} kernel matrix")
        return kernel_matrix
    
    def plot_kernel_matrix(
        self,
        kernel_matrix: np.ndarray,
        dates: List[pd.Timestamp],
        save_path: Optional[str] = None
    ):
        """Plot kernel matrix heatmap"""
        fig, ax = plt.subplots(figsize=(12, 10))
        
        date_labels = [d.strftime('%Y-%m-%d') for d in dates]
        
        sns.heatmap(
            kernel_matrix,
            cmap='RdYlGn',
            xticklabels=date_labels,
            yticklabels=date_labels,
            cbar_kws={'label': 'Kernel Similarity'},
            square=True,
            linewidths=0.5,
            vmin=0,
            vmax=1,
            ax=ax
        )
        
        ax.set_title(
            'Persistence Fisher Kernel Matrix\n'
            'Topological Similarity via RBF Kernel',
            fontsize=14, fontweight='bold', pad=20
        )
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Saved to {save_path}")
        
        plt.show()


if __name__ == "__main__":
    print("="*60)
    print("ULTRA-ADVANCED TOPOLOGICAL METRICS DEMO")
    print("="*60)
    
    # Create sample data
    np.random.seed(42)
    
    # Sample diagram
    diagram = np.array([
        [0.1, 0.5, 1],
        [0.2, 0.8, 1],
        [0.3, 0.6, 1],
        [0.15, 0.45, 1],
    ])
    
    metrics = UltraAdvancedMetrics()
    
    # Test silhouette
    print("\n🔬 Computing Silhouette...")
    t_vals, silh = metrics.silhouette_of_persistence(diagram, dim=1)
    print(f"✅ Silhouette computed: {len(t_vals)} points")
    
    # Test landscape norms
    print("\n📏 Computing Landscape Norms...")
    norms = metrics.landscape_norms(diagram, dim=1)
    print(f"✅ L1={norms['L1']:.3f}, L2={norms['L2']:.3f}, L∞={norms['Linf']:.3f}")
    
    # Test kernel
    print("\n🔬 Computing Fisher Kernel...")
    kernel_val = metrics.persistence_fisher_kernel(diagram, diagram, dim=1)
    print(f"✅ Self-kernel: {kernel_val:.3f}")
    
    print("\n✅ All ultra-advanced metrics working!")

