"""
Visualization Tools for Topological Data Analysis
Persistence diagrams, barcodes, and topology evolution
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class TopologyVisualizer:
    """Visualize persistent homology results"""
    
    def __init__(self):
        """Initialize visualizer with style settings"""
        sns.set_style("whitegrid")
        self.colors = {
            0: '#1f77b4',  # Blue for H0
            1: '#ff7f0e',  # Orange for H1
            2: '#2ca02c',  # Green for H2
        }
        self.labels = {
            0: 'H₀ (Clusters)',
            1: 'H₁ (Loops)',
            2: 'H₂ (Voids)',
        }
    
    def plot_persistence_diagram(
        self,
        diagram: np.ndarray,
        title: str = "Persistence Diagram",
        save_path: Optional[str] = None
    ):
        """
        Plot persistence diagram (birth vs death times)
        
        Args:
            diagram: Persistence diagram (birth, death, dimension)
            title: Plot title
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot diagonal (birth = death line)
        max_val = diagram[:, 1].max() if len(diagram) > 0 else 1
        ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, linewidth=2, 
                label='Birth = Death')
        
        # Plot points by dimension
        for dim in [0, 1, 2]:
            mask = diagram[:, 2] == dim
            dgm_dim = diagram[mask]
            
            if len(dgm_dim) > 0:
                # Size by persistence
                persistence = dgm_dim[:, 1] - dgm_dim[:, 0]
                sizes = 100 * (persistence / persistence.max()) if len(persistence) > 0 else 50
                
                ax.scatter(
                    dgm_dim[:, 0], 
                    dgm_dim[:, 1],
                    s=sizes,
                    c=self.colors[dim],
                    alpha=0.6,
                    edgecolors='black',
                    linewidth=1,
                    label=f'{self.labels[dim]} (n={len(dgm_dim)})'
                )
        
        ax.set_xlabel('Birth', fontsize=14, fontweight='bold')
        ax.set_ylabel('Death', fontsize=14, fontweight='bold')
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.legend(fontsize=12, loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Saved to {save_path}")
        
        plt.show()
    
    def plot_barcode(
        self,
        diagram: np.ndarray,
        title: str = "Persistence Barcode",
        save_path: Optional[str] = None
    ):
        """
        Plot persistence barcode (birth to death intervals)
        
        Args:
            diagram: Persistence diagram (birth, death, dimension)
            title: Plot title
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
        for dim, ax in enumerate(axes):
            mask = diagram[:, 2] == dim
            dgm_dim = diagram[mask]
            
            if len(dgm_dim) > 0:
                # Sort by birth time
                idx_sort = np.argsort(dgm_dim[:, 0])
                dgm_sorted = dgm_dim[idx_sort]
                
                # Plot bars
                for i, (birth, death, _) in enumerate(dgm_sorted):
                    persistence = death - birth
                    ax.barh(i, persistence, left=birth, height=0.8,
                           color=self.colors[dim], alpha=0.7, 
                           edgecolor='black', linewidth=0.5)
                
                ax.set_ylabel('Feature Index', fontsize=12)
                ax.set_title(f'{self.labels[dim]} (n={len(dgm_dim)})', 
                           fontsize=13, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='x')
            else:
                ax.text(0.5, 0.5, 'No features detected', 
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=12, style='italic')
                ax.set_title(f'{self.labels[dim]} (n=0)', 
                           fontsize=13, fontweight='bold')
            
            if dim == 2:
                ax.set_xlabel('Filtration Value', fontsize=12, fontweight='bold')
        
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Saved to {save_path}")
        
        plt.show()
    
    def plot_topology_evolution(
        self,
        features: pd.DataFrame,
        metrics: List[str] = None,
        save_path: Optional[str] = None
    ):
        """
        Plot evolution of topological features over time
        
        Args:
            features: DataFrame with topological features and dates
            metrics: List of metrics to plot (default: key metrics)
            save_path: Path to save figure
        """
        if metrics is None:
            metrics = [
                'H0_count', 'H1_count', 'H2_count',
                'H1_max_persistence', 'H2_max_persistence'
            ]
        
        # Filter available metrics
        metrics = [m for m in metrics if m in features.columns]
        
        fig, axes = plt.subplots(len(metrics), 1, figsize=(14, 3*len(metrics)))
        if len(metrics) == 1:
            axes = [axes]
        
        for ax, metric in zip(axes, metrics):
            # Determine dimension from metric name
            if 'H0' in metric:
                color = self.colors[0]
            elif 'H1' in metric:
                color = self.colors[1]
            elif 'H2' in metric:
                color = self.colors[2]
            else:
                color = 'gray'
            
            ax.plot(features.index, features[metric], 
                   linewidth=2, color=color, marker='o', markersize=4)
            ax.fill_between(features.index, features[metric], 
                           alpha=0.3, color=color)
            ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Highlight significant changes
            if 'persistence' in metric:
                mean = features[metric].mean()
                std = features[metric].std()
                threshold = mean + 2*std
                significant = features[metric] > threshold
                if significant.any():
                    ax.scatter(features.index[significant], 
                             features[metric][significant],
                             color='red', s=100, marker='*', 
                             zorder=5, label='Anomaly')
                    ax.axhline(threshold, color='red', linestyle='--', 
                             alpha=0.5, linewidth=1)
                    ax.legend()
        
        axes[-1].set_xlabel('Date', fontsize=12, fontweight='bold')
        fig.suptitle('Topological Features Evolution', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Saved to {save_path}")
        
        plt.show()
    
    def plot_betti_curves(
        self,
        diagrams: List[np.ndarray],
        dates: List[pd.Timestamp],
        resolution: int = 100,
        save_path: Optional[str] = None
    ):
        """
        Plot Betti curves over time (topological complexity)
        
        Args:
            diagrams: List of persistence diagrams
            dates: Corresponding dates
            resolution: Number of filtration steps
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
        # Compute Betti curves for each window
        for dim, ax in enumerate(axes):
            betti_curves = []
            
            for diagram in diagrams:
                mask = diagram[:, 2] == dim
                dgm_dim = diagram[mask]
                
                if len(dgm_dim) > 0:
                    # Create filtration range
                    max_val = diagram[:, 1].max()
                    filtration = np.linspace(0, max_val, resolution)
                    
                    # Count features alive at each filtration value
                    betti = np.zeros(resolution)
                    for i, f in enumerate(filtration):
                        alive = np.sum((dgm_dim[:, 0] <= f) & (dgm_dim[:, 1] > f))
                        betti[i] = alive
                    
                    betti_curves.append(betti)
                else:
                    betti_curves.append(np.zeros(resolution))
            
            # Plot as heatmap
            if betti_curves:
                betti_matrix = np.array(betti_curves).T
                
                im = ax.imshow(betti_matrix, aspect='auto', cmap='YlOrRd',
                             interpolation='bilinear')
                
                ax.set_ylabel('Filtration Value', fontsize=12)
                ax.set_title(f'{self.labels[dim]} - Betti Curves Over Time', 
                           fontsize=13, fontweight='bold')
                
                # Set x-axis to dates
                n_ticks = min(10, len(dates))
                tick_indices = np.linspace(0, len(dates)-1, n_ticks, dtype=int)
                ax.set_xticks(tick_indices)
                ax.set_xticklabels([dates[i].strftime('%Y-%m-%d') 
                                   for i in tick_indices], rotation=45)
                
                # Colorbar
                plt.colorbar(im, ax=ax, label='Betti Number')
        
        axes[-1].set_xlabel('Date', fontsize=12, fontweight='bold')
        fig.suptitle('Topological Complexity Evolution (Betti Curves)', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Saved to {save_path}")
        
        plt.show()
    
    def create_interactive_dashboard(
        self,
        features: pd.DataFrame,
        save_path: Optional[str] = None
    ):
        """
        Create interactive Plotly dashboard
        
        Args:
            features: DataFrame with topological features
            save_path: Path to save HTML
        """
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'H₀ Features (Clusters)',
                'H₁ Features (Loops)',
                'H₂ Features (Voids)',
                'Persistence Comparison',
                'Entropy Evolution',
                'Feature Counts'
            ),
            specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
                   [{'type': 'scatter'}, {'type': 'scatter'}],
                   [{'type': 'scatter'}, {'type': 'bar'}]]
        )
        
        # H0 metrics
        fig.add_trace(
            go.Scatter(x=features.index, y=features['H0_count'],
                      mode='lines+markers', name='H₀ Count',
                      line=dict(color=self.colors[0], width=2)),
            row=1, col=1
        )
        
        # H1 metrics
        fig.add_trace(
            go.Scatter(x=features.index, y=features['H1_max_persistence'],
                      mode='lines+markers', name='H₁ Max Persistence',
                      line=dict(color=self.colors[1], width=2)),
            row=1, col=2
        )
        
        # H2 metrics
        fig.add_trace(
            go.Scatter(x=features.index, y=features['H2_max_persistence'],
                      mode='lines+markers', name='H₂ Max Persistence',
                      line=dict(color=self.colors[2], width=2)),
            row=2, col=1
        )
        
        # Persistence comparison
        for dim in [0, 1, 2]:
            fig.add_trace(
                go.Scatter(x=features.index, 
                          y=features[f'H{dim}_sum_persistence'],
                          mode='lines', name=f'H{dim} Total',
                          line=dict(color=self.colors[dim], width=2)),
                row=2, col=2
            )
        
        # Entropy
        for dim in [0, 1, 2]:
            fig.add_trace(
                go.Scatter(x=features.index, 
                          y=features[f'H{dim}_entropy'],
                          mode='lines', name=f'H{dim} Entropy',
                          line=dict(color=self.colors[dim], width=2)),
                row=3, col=1
            )
        
        # Feature counts (latest window)
        latest = features.iloc[-1]
        fig.add_trace(
            go.Bar(x=['H₀', 'H₁', 'H₂'],
                  y=[latest['H0_count'], latest['H1_count'], latest['H2_count']],
                  marker_color=[self.colors[0], self.colors[1], self.colors[2]],
                  name='Feature Counts'),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=1200,
            title_text="Market Topology Dashboard",
            showlegend=True,
            template='plotly_white'
        )
        
        fig.update_xaxes(title_text="Date", row=3)
        fig.update_yaxes(title_text="Count")
        
        if save_path:
            fig.write_html(save_path)
            print(f"💾 Interactive dashboard saved to {save_path}")
        
        fig.show()
    
    def plot_3d_persistence_surface(
        self,
        diagram: np.ndarray,
        dim: int = 1,
        save_path: Optional[str] = None
    ):
        """
        Plot 3D surface of persistence (birth, death, persistence)
        
        Args:
            diagram: Persistence diagram
            dim: Homology dimension
            save_path: Path to save figure
        """
        from mpl_toolkits.mplot3d import Axes3D
        
        # Filter by dimension
        mask = diagram[:, 2] == dim
        dgm = diagram[mask, :2]
        
        if len(dgm) == 0:
            print("⚠️ No features to plot for 3D surface")
            return
        
        births = dgm[:, 0]
        deaths = dgm[:, 1]
        persistences = deaths - births
        
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create scatter plot
        scatter = ax.scatter(
            births, deaths, persistences,
            c=persistences,
            cmap='viridis',
            s=100,
            alpha=0.7,
            edgecolors='black',
            linewidth=1
        )
        
        # Plot diagonal plane (birth = death, persistence = 0)
        b_range = np.linspace(births.min(), deaths.max(), 20)
        d_range = b_range
        B, D = np.meshgrid(b_range, d_range)
        Z = np.zeros_like(B)
        ax.plot_surface(B, D, Z, alpha=0.2, color='red')
        
        # Plot persistence surfaces
        for birth, death, pers in zip(births, deaths, persistences):
            # Draw line from diagonal to point
            ax.plot([birth, birth], [death, death], [0, pers], 
                   'k--', alpha=0.3, linewidth=0.5)
        
        ax.set_xlabel('Birth', fontsize=12, fontweight='bold')
        ax.set_ylabel('Death', fontsize=12, fontweight='bold')
        ax.set_zlabel('Persistence', fontsize=12, fontweight='bold')
        ax.set_title(
            f'3D Persistence Surface - {self.labels[dim]}\n'
            f'Birth × Death × Persistence',
            fontsize=14, fontweight='bold', pad=20
        )
        
        plt.colorbar(scatter, ax=ax, label='Persistence', shrink=0.6)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Saved to {save_path}")
        
        plt.show()
    
    def plot_mapper_graph(
        self,
        point_cloud: np.ndarray,
        filter_func: Optional[np.ndarray] = None,
        n_intervals: int = 10,
        overlap: float = 0.3,
        save_path: Optional[str] = None
    ):
        """
        Simplified Mapper algorithm visualization
        
        Args:
            point_cloud: Market returns point cloud (n_points, n_dimensions)
            filter_func: Optional filter function values (if None, use PCA)
            n_intervals: Number of intervals for covering
            overlap: Overlap between intervals
            save_path: Path to save figure
        """
        from sklearn.decomposition import PCA
        from sklearn.cluster import DBSCAN
        import networkx as nx
        
        print(f"\n🗺️ Computing Mapper Graph...")
        
        n_points = len(point_cloud)
        
        # Compute filter function (use first PCA component if not provided)
        if filter_func is None:
            pca = PCA(n_components=1)
            filter_func = pca.fit_transform(point_cloud).flatten()
        
        # Create overlapping intervals
        f_min, f_max = filter_func.min(), filter_func.max()
        f_range = f_max - f_min
        step = f_range / n_intervals
        overlap_size = step * overlap
        
        # Build mapper graph
        G = nx.Graph()
        node_id = 0
        node_info = {}
        
        for i in range(n_intervals):
            # Define interval
            interval_min = f_min + i * step - overlap_size
            interval_max = f_min + (i + 1) * step + overlap_size
            
            # Get points in interval
            mask = (filter_func >= interval_min) & (filter_func <= interval_max)
            interval_points = point_cloud[mask]
            
            if len(interval_points) < 3:
                continue
            
            # Cluster points in interval
            clustering = DBSCAN(eps=1.0, min_samples=2).fit(interval_points)
            labels = clustering.labels_
            
            # Create nodes for each cluster
            for cluster_id in set(labels):
                if cluster_id == -1:  # Skip noise
                    continue
                
                cluster_mask = labels == cluster_id
                cluster_points = interval_points[cluster_mask]
                
                # Add node
                G.add_node(node_id, 
                          interval=i,
                          cluster=cluster_id,
                          size=len(cluster_points),
                          mean_filter=filter_func[mask][cluster_mask].mean())
                
                node_info[node_id] = {
                    'interval': i,
                    'cluster': cluster_id,
                    'size': len(cluster_points)
                }
                
                node_id += 1
        
        # Add edges between overlapping clusters
        for n1 in G.nodes():
            for n2 in G.nodes():
                if n1 >= n2:
                    continue
                
                # Check if intervals overlap
                i1 = G.nodes[n1]['interval']
                i2 = G.nodes[n2]['interval']
                
                if abs(i1 - i2) <= 1:  # Adjacent or same interval
                    G.add_edge(n1, n2)
        
        print(f"✅ Mapper graph: {len(G.nodes)} nodes, {len(G.edges)} edges")
        
        # Visualize
        if len(G.nodes) == 0:
            print("⚠️ No nodes in Mapper graph")
            return
        
        plt.figure(figsize=(14, 10))
        
        # Layout
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        
        # Node colors by interval
        node_colors = [G.nodes[n]['interval'] for n in G.nodes()]
        
        # Node sizes by cluster size
        node_sizes = [G.nodes[n]['size'] * 20 for n in G.nodes()]
        
        # Draw
        nx.draw_networkx_nodes(
            G, pos,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.8,
            cmap=plt.cm.viridis,
            edgecolors='black',
            linewidths=2
        )
        
        nx.draw_networkx_edges(
            G, pos,
            width=2,
            alpha=0.5,
            edge_color='gray'
        )
        
        nx.draw_networkx_labels(
            G, pos,
            font_size=8,
            font_weight='bold'
        )
        
        plt.title(
            'Mapper Graph - Market Shape\n'
            f'Filter: PCA Component 1 | {n_intervals} intervals | {overlap*100:.0f}% overlap',
            fontsize=14, fontweight='bold', pad=20
        )
        plt.colorbar(
            plt.cm.ScalarMappable(cmap=plt.cm.viridis, 
                                 norm=plt.Normalize(vmin=0, vmax=n_intervals)),
            label='Interval',
            ax=plt.gca(),
            fraction=0.03
        )
        plt.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Saved to {save_path}")
        
        plt.show()
        
        return G


if __name__ == "__main__":
    # Example usage
    from data_fetcher import MarketDataFetcher
    from point_cloud import MarketPointCloud
    from persistent_homology import PersistentHomologyAnalyzer
    
    print("="*60)
    print("VISUALIZATION DEMO")
    print("="*60)
    
    # Fetch and prepare data
    fetcher = MarketDataFetcher()
    data = fetcher.fetch_data(period="6mo", max_tickers=30)
    returns = fetcher.compute_returns()
    
    # Create point clouds
    pc = MarketPointCloud(returns)
    windows = pc.create_sliding_windows(window_size=30, step_size=10)
    point_clouds = [w[0] for w in windows]
    dates = [w[1] for w in windows]
    
    # Compute persistent homology
    analyzer = PersistentHomologyAnalyzer(library="ripser")
    features = analyzer.analyze_windows(point_clouds, dates)
    
    # Visualize
    visualizer = TopologyVisualizer()
    
    print("\n📊 Generating visualizations...")
    
    # 1. Persistence diagram for latest window
    visualizer.plot_persistence_diagram(
        analyzer.diagrams[-1],
        title=f"Persistence Diagram - {dates[-1].date()}"
    )
    
    # 2. Barcode for latest window
    visualizer.plot_barcode(
        analyzer.diagrams[-1],
        title=f"Persistence Barcode - {dates[-1].date()}"
    )
    
    # 3. Evolution over time
    visualizer.plot_topology_evolution(features)
    
    # 4. Interactive dashboard
    visualizer.create_interactive_dashboard(features)

