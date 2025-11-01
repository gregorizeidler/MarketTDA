"""
Advanced Visualizations - Animations, 3D Interactive, Heatmaps
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation, PillowWriter
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')


class AdvancedVisualizations:
    """Advanced visualization techniques including animations and interactive plots"""
    
    def __init__(self):
        """Initialize advanced visualizer"""
        self.colors = {0: '#3498DB', 1: '#E74C3C', 2: '#2ECC71'}
    
    # ==================== PERSISTENCE LANDSCAPE ANIMATION ====================
    
    def create_landscape_animation(
        self,
        diagrams: List[np.ndarray],
        dates: List[pd.Timestamp],
        dim: int = 1,
        save_path: Optional[str] = None,
        fps: int = 5
    ):
        """
        Create animated GIF of persistence landscape evolution
        
        Args:
            diagrams: List of persistence diagrams
            dates: Corresponding dates
            dim: Homology dimension
            save_path: Path to save GIF
            fps: Frames per second
        """
        print(f"\n🎬 Creating Persistence Landscape Animation...")
        
        # Compute all landscapes first
        landscapes = []
        t_range = None
        
        for diagram in diagrams:
            mask = diagram[:, 2] == dim
            dgm = diagram[mask, :2]
            
            if len(dgm) > 0:
                if t_range is None:
                    t_min = dgm[:, 0].min()
                    t_max = dgm[:, 1].max()
                    t_range = np.linspace(t_min, t_max, 100)
                
                landscape = np.zeros(len(t_range))
                for birth, death in dgm:
                    for i, t in enumerate(t_range):
                        if birth <= t <= death:
                            landscape[i] = max(landscape[i], min(t - birth, death - t))
                
                landscapes.append(landscape)
            else:
                landscapes.append(np.zeros(100))
        
        # Create animation
        fig, ax = plt.subplots(figsize=(12, 6))
        
        def update(frame):
            ax.clear()
            ax.fill_between(t_range, landscapes[frame], alpha=0.6, color=self.colors[dim])
            ax.plot(t_range, landscapes[frame], linewidth=2, color=self.colors[dim])
            ax.set_xlabel('Filtration Value', fontsize=12, fontweight='bold')
            ax.set_ylabel('Landscape Function', fontsize=12, fontweight='bold')
            ax.set_title(
                f'Persistence Landscape H{dim}\n'
                f'{dates[frame].strftime("%Y-%m-%d")}',
                fontsize=14, fontweight='bold'
            )
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, max([l.max() for l in landscapes]) * 1.1])
        
        ani = FuncAnimation(fig, update, frames=len(diagrams), repeat=True)
        
        if save_path:
            writer = PillowWriter(fps=fps)
            ani.save(save_path, writer=writer)
            print(f"💾 Saved animation to {save_path}")
        
        plt.close()
        print(f"✅ Created animation with {len(diagrams)} frames")
    
    # ==================== 3D INTERACTIVE PERSISTENCE DIAGRAM ====================
    
    def plot_3d_persistence_diagram(
        self,
        diagram: np.ndarray,
        save_path: Optional[str] = None
    ):
        """
        Create 3D interactive persistence diagram with Plotly
        
        Args:
            diagram: Persistence diagram (birth, death, dimension)
            save_path: Path to save HTML file
        """
        print("\n📊 Creating 3D Interactive Persistence Diagram...")
        
        fig = make_subplots(
            rows=1, cols=1,
            specs=[[{'type': 'scatter3d'}]]
        )
        
        for dim in [0, 1, 2]:
            mask = diagram[:, 2] == dim
            dgm_dim = diagram[mask, :2]
            
            if len(dgm_dim) == 0:
                continue
            
            births = dgm_dim[:, 0]
            deaths = dgm_dim[:, 1]
            persistences = deaths - births
            
            fig.add_trace(go.Scatter3d(
                x=births,
                y=deaths,
                z=persistences,
                mode='markers',
                name=f'H{dim}',
                marker=dict(
                    size=6,
                    color=self.colors[dim],
                    opacity=0.8
                ),
                hovertemplate=
                    f'<b>H{dim}</b><br>' +
                    'Birth: %{x:.3f}<br>' +
                    'Death: %{y:.3f}<br>' +
                    'Persistence: %{z:.3f}<br>' +
                    '<extra></extra>'
            ))
        
        # Add diagonal plane
        max_val = diagram[:, :2].max()
        grid = np.linspace(0, max_val, 10)
        X, Y = np.meshgrid(grid, grid)
        Z = np.zeros_like(X)
        
        fig.add_trace(go.Surface(
            x=X, y=Y, z=Z,
            colorscale='Greys',
            opacity=0.2,
            showscale=False,
            name='Diagonal'
        ))
        
        fig.update_layout(
            title={
                'text': '3D Interactive Persistence Diagram<br><sub>Birth × Death × Persistence</sub>',
                'font': {'size': 18, 'family': 'Arial Black'}
            },
            scene=dict(
                xaxis_title='Birth',
                yaxis_title='Death',
                zaxis_title='Persistence',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.3)
                )
            ),
            width=1000,
            height=800,
            hovermode='closest'
        )
        
        if save_path:
            if save_path.endswith('.html'):
                fig.write_html(save_path)
                print(f"💾 Saved interactive plot to {save_path}")
            else:
                fig.write_image(save_path)
                print(f"💾 Saved static image to {save_path}")
        
        fig.show()
    
    # ==================== WASSERSTEIN DISTANCE HEATMAP OVER TIME ====================
    
    def plot_wasserstein_heatmap_temporal(
        self,
        wasserstein_matrix: np.ndarray,
        dates: List[pd.Timestamp],
        save_path: Optional[str] = None
    ):
        """
        Plot Wasserstein distance matrix with temporal annotations
        
        Args:
            wasserstein_matrix: Distance matrix
            dates: Date labels
            save_path: Save path
        """
        print("\n🔥 Creating Temporal Wasserstein Heatmap...")
        
        fig, ax = plt.subplots(figsize=(14, 12))
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(wasserstein_matrix, dtype=bool), k=1)
        
        # Plot heatmap
        sns.heatmap(
            wasserstein_matrix,
            mask=mask,
            cmap='YlOrRd',
            square=True,
            linewidths=0.5,
            cbar_kws={'label': 'Wasserstein Distance'},
            ax=ax,
            vmin=0,
            annot=wasserstein_matrix < 0.1,  # Annotate very similar diagrams
            fmt='.2f'
        )
        
        # Set date labels
        date_labels = [d.strftime('%Y-%m-%d') for d in dates]
        ax.set_xticklabels(date_labels, rotation=45, ha='right')
        ax.set_yticklabels(date_labels, rotation=0)
        
        ax.set_title(
            'Temporal Wasserstein Distance Matrix\n'
            'Topological Similarity Across Time',
            fontsize=14, fontweight='bold', pad=20
        )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Saved to {save_path}")
        
        plt.show()
    
    # ==================== NETWORK EVOLUTION ANIMATION ====================
    
    def create_network_evolution_animation(
        self,
        msts: List[Dict],
        dates: List[pd.Timestamp],
        save_path: Optional[str] = None,
        fps: int = 3
    ):
        """
        Create animated GIF of MST network evolution
        
        Args:
            msts: List of MST dictionaries with 'edges' and 'weights'
            dates: Corresponding dates
            save_path: Path to save GIF
            fps: Frames per second
        """
        print(f"\n🌐 Creating Network Evolution Animation...")
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        def update(frame):
            ax.clear()
            
            mst = msts[frame]
            edges = mst['edges']
            weights = mst['weights']
            
            # Create adjacency matrix for layout
            nodes = list(set([e[0] for e in edges] + [e[1] for e in edges]))
            n_nodes = len(nodes)
            
            # Simple circular layout
            angles = np.linspace(0, 2*np.pi, n_nodes, endpoint=False)
            pos = {node: (np.cos(angles[i]), np.sin(angles[i])) for i, node in enumerate(nodes)}
            
            # Draw edges
            for (u, v), weight in zip(edges, weights):
                x = [pos[u][0], pos[v][0]]
                y = [pos[u][1], pos[v][1]]
                ax.plot(x, y, 'gray', linewidth=weight*5, alpha=0.5)
            
            # Draw nodes
            x_coords = [pos[node][0] for node in nodes]
            y_coords = [pos[node][1] for node in nodes]
            ax.scatter(x_coords, y_coords, s=500, c='steelblue', alpha=0.8, edgecolors='black', linewidths=2)
            
            # Labels
            for node in nodes:
                ax.text(pos[node][0], pos[node][1], str(node), 
                       ha='center', va='center', fontsize=8, fontweight='bold', color='white')
            
            ax.set_title(
                f'Market Structure Network (MST)\n'
                f'{dates[frame].strftime("%Y-%m-%d")} - {len(edges)} edges',
                fontsize=14, fontweight='bold'
            )
            ax.axis('off')
            ax.set_xlim([-1.3, 1.3])
            ax.set_ylim([-1.3, 1.3])
        
        ani = FuncAnimation(fig, update, frames=min(len(msts), 50), repeat=True)  # Limit frames
        
        if save_path:
            writer = PillowWriter(fps=fps)
            ani.save(save_path, writer=writer)
            print(f"💾 Saved animation to {save_path}")
        
        plt.close()
        print(f"✅ Created animation with {min(len(msts), 50)} frames")
    
    # ==================== REGIME PROBABILITY HEATMAP ====================
    
    def plot_regime_probability_heatmap(
        self,
        regime_probs: pd.DataFrame,
        save_path: Optional[str] = None
    ):
        """
        Plot regime probability heatmap over time
        
        Args:
            regime_probs: DataFrame with regime probabilities (columns) over time (index)
            save_path: Save path
        """
        print("\n📊 Creating Regime Probability Heatmap...")
        
        fig, ax = plt.subplots(figsize=(16, 8))
        
        # Transpose for better visualization
        data = regime_probs.T
        
        sns.heatmap(
            data,
            cmap='RdYlGn_r',
            cbar_kws={'label': 'Probability'},
            linewidths=0.5,
            vmin=0,
            vmax=1,
            ax=ax
        )
        
        # Set labels
        date_labels = [d.strftime('%Y-%m-%d') for d in regime_probs.index]
        ax.set_xticklabels(date_labels, rotation=45, ha='right')
        ax.set_yticklabels(data.index, rotation=0)
        
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Regime', fontsize=12, fontweight='bold')
        ax.set_title(
            'Market Regime Probability Evolution\n'
            'Probabilistic State Assignment Over Time',
            fontsize=14, fontweight='bold', pad=20
        )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Saved to {save_path}")
        
        plt.show()
    
    # ==================== 3D PERSISTENCE SURFACE ====================
    
    def plot_3d_persistence_surface_interactive(
        self,
        diagrams: List[np.ndarray],
        dates: List[pd.Timestamp],
        dim: int = 1,
        save_path: Optional[str] = None
    ):
        """
        Create interactive 3D surface of persistence evolution over time
        
        Args:
            diagrams: List of persistence diagrams
            dates: Corresponding dates
            dim: Homology dimension
            save_path: Path to save HTML
        """
        print(f"\n📈 Creating 3D Persistence Surface...")
        
        # Compute persistence images for all diagrams
        from advanced_metrics import AdvancedTopologicalMetrics
        metrics = AdvancedTopologicalMetrics()
        
        images = []
        for diagram in diagrams:
            img = metrics.persistence_image(diagram, dim=dim, resolution=(20, 20))
            images.append(img)
        
        images = np.array(images)
        
        # Create surface plot
        fig = go.Figure()
        
        # Time axis
        time_indices = np.arange(len(dates))
        
        # Add surface for each time slice
        for i in range(0, len(dates), max(1, len(dates)//20)):  # Sample every Nth diagram
            fig.add_trace(go.Surface(
                z=images[i],
                name=dates[i].strftime('%Y-%m-%d'),
                colorscale='Viridis',
                opacity=0.7,
                showscale=(i==0)
            ))
        
        fig.update_layout(
            title=f'3D Persistence Surface Evolution - H{dim}<br><sub>Persistence Images Over Time</sub>',
            scene=dict(
                xaxis_title='Birth Coordinate',
                yaxis_title='Death Coordinate',
                zaxis_title='Persistence Weight',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2)
                )
            ),
            width=1000,
            height=800
        )
        
        if save_path:
            if save_path.endswith('.html'):
                fig.write_html(save_path)
                print(f"💾 Saved interactive plot to {save_path}")
            else:
                fig.write_image(save_path)
                print(f"💾 Saved static image to {save_path}")
        
        fig.show()


if __name__ == "__main__":
    print("="*60)
    print("ADVANCED VISUALIZATIONS DEMO")
    print("="*60)
    
    # Create sample data
    np.random.seed(42)
    
    # Sample diagrams
    diagrams = []
    for i in range(10):
        dgm = np.array([
            [0.1 + i*0.01, 0.5 + i*0.01, 1],
            [0.2, 0.8, 1],
            [0.3, 0.6, 1],
        ])
        diagrams.append(dgm)
    
    dates = pd.date_range('2024-01-01', periods=10, freq='D')
    
    viz = AdvancedVisualizations()
    
    # Test 3D persistence diagram
    print("\n🔬 Testing 3D Persistence Diagram...")
    # viz.plot_3d_persistence_diagram(diagrams[0])  # Uncomment to show
    
    print("\n✅ Advanced visualizations ready!")

