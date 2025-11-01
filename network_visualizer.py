"""
Network Visualization for Market Topology
Correlation networks, MST, Force-directed graphs, Sankey diagrams
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')


class NetworkVisualizer:
    """Visualize market structure as networks"""
    
    def __init__(self):
        """Initialize network visualizer"""
        self.colors = plt.cm.Set3(np.linspace(0, 1, 12))
    
    # ==================== MINIMUM SPANNING TREE ====================
    
    def compute_mst(
        self,
        correlation_matrix: np.ndarray,
        tickers: List[str] = None
    ) -> nx.Graph:
        """
        Compute Minimum Spanning Tree from correlation matrix
        
        Args:
            correlation_matrix: Asset correlation matrix
            tickers: List of ticker symbols
            
        Returns:
            NetworkX graph of MST
        """
        print(f"\n🌳 Computing Minimum Spanning Tree...")
        
        # Convert correlation to distance
        # distance = sqrt(2 * (1 - correlation))
        distance_matrix = np.sqrt(2 * (1 - correlation_matrix))
        np.fill_diagonal(distance_matrix, 0)
        
        # Create complete graph
        n = len(correlation_matrix)
        G = nx.Graph()
        
        if tickers is None:
            tickers = [f"Asset_{i}" for i in range(n)]
        
        # Add edges with weights (distances)
        for i in range(n):
            for j in range(i+1, n):
                G.add_edge(tickers[i], tickers[j], weight=distance_matrix[i, j])
        
        # Compute MST
        mst = nx.minimum_spanning_tree(G)
        
        print(f"✅ MST computed: {len(mst.nodes)} nodes, {len(mst.edges)} edges")
        
        return mst
    
    def plot_mst(
        self,
        mst: nx.Graph,
        regimes: Optional[Dict[str, int]] = None,
        save_path: Optional[str] = None
    ):
        """
        Plot Minimum Spanning Tree
        
        Args:
            mst: MST graph
            regimes: Optional dict mapping node -> regime for coloring
            save_path: Path to save figure
        """
        plt.figure(figsize=(16, 12))
        
        # Layout
        pos = nx.spring_layout(mst, k=2, iterations=50, seed=42)
        
        # Node colors by regime
        if regimes:
            node_colors = [regimes.get(node, 0) for node in mst.nodes()]
            cmap = plt.cm.Set3
        else:
            node_colors = 'lightblue'
            cmap = None
        
        # Draw network
        nx.draw_networkx_nodes(
            mst, pos,
            node_color=node_colors,
            node_size=800,
            alpha=0.9,
            cmap=cmap,
            edgecolors='black',
            linewidths=2
        )
        
        # Draw edges with weights
        edges = mst.edges()
        weights = [mst[u][v]['weight'] for u, v in edges]
        
        nx.draw_networkx_edges(
            mst, pos,
            width=[5 * (1 - w) for w in weights],  # Thicker = stronger correlation
            alpha=0.6,
            edge_color='gray'
        )
        
        # Draw labels
        nx.draw_networkx_labels(
            mst, pos,
            font_size=8,
            font_weight='bold'
        )
        
        plt.title(
            'Minimum Spanning Tree (MST)\n'
            'Market Structure - Strongest Correlations Only',
            fontsize=16, fontweight='bold', pad=20
        )
        plt.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Saved to {save_path}")
        
        plt.show()
    
    def plot_mst_evolution(
        self,
        correlation_matrices: List[np.ndarray],
        dates: List[pd.Timestamp],
        tickers: List[str],
        save_path: Optional[str] = None
    ):
        """
        Plot evolution of MST over time
        
        Args:
            correlation_matrices: List of correlation matrices over time
            dates: Corresponding dates
            tickers: Asset tickers
            save_path: Path to save animation/grid
        """
        n_windows = min(6, len(correlation_matrices))
        indices = np.linspace(0, len(correlation_matrices)-1, n_windows, dtype=int)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, ax in zip(indices, axes):
            corr = correlation_matrices[idx]
            date = dates[idx]
            
            # Compute MST
            mst = self.compute_mst(corr, tickers)
            
            # Layout
            pos = nx.spring_layout(mst, k=2, iterations=30, seed=42)
            
            # Draw
            nx.draw_networkx_nodes(
                mst, pos,
                node_color='lightblue',
                node_size=300,
                alpha=0.8,
                edgecolors='black',
                ax=ax
            )
            
            weights = [mst[u][v]['weight'] for u, v in mst.edges()]
            nx.draw_networkx_edges(
                mst, pos,
                width=[3 * (1 - w) for w in weights],
                alpha=0.5,
                edge_color='gray',
                ax=ax
            )
            
            nx.draw_networkx_labels(
                mst, pos,
                font_size=6,
                ax=ax
            )
            
            ax.set_title(date.strftime('%Y-%m-%d'), fontsize=12, fontweight='bold')
            ax.axis('off')
        
        fig.suptitle(
            'MST Evolution Over Time\n'
            'Market Structure Dynamics',
            fontsize=16, fontweight='bold'
        )
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Saved to {save_path}")
        
        plt.show()
    
    # ==================== CORRELATION NETWORK ====================
    
    def plot_correlation_network(
        self,
        correlation_matrix: np.ndarray,
        tickers: List[str],
        threshold: float = 0.5,
        regimes: Optional[pd.Series] = None,
        save_path: Optional[str] = None
    ):
        """
        Plot full correlation network (filtered by threshold)
        
        Args:
            correlation_matrix: Asset correlations
            tickers: Asset names
            threshold: Minimum correlation to show edge
            regimes: Optional regime labels for coloring
            save_path: Path to save
        """
        print(f"\n🕸️ Creating Correlation Network (threshold={threshold})...")
        
        n = len(correlation_matrix)
        
        # Create graph
        G = nx.Graph()
        
        for i in range(n):
            for j in range(i+1, n):
                corr = correlation_matrix[i, j]
                if abs(corr) >= threshold:
                    G.add_edge(tickers[i], tickers[j], weight=corr, abs_weight=abs(corr))
        
        print(f"✅ Network: {len(G.nodes)} nodes, {len(G.edges)} edges")
        
        if len(G.edges) == 0:
            print("⚠️ No edges above threshold")
            return
        
        # Layout
        pos = nx.spring_layout(G, k=1, iterations=50, seed=42)
        
        plt.figure(figsize=(16, 12))
        
        # Node colors by regime or degree
        if regimes is not None:
            node_colors = [regimes.get(node, 0) if node in regimes.index else 0 
                          for node in G.nodes()]
            cmap = plt.cm.Set3
        else:
            node_colors = [G.degree(node) for node in G.nodes()]
            cmap = plt.cm.viridis
        
        # Node sizes by degree
        node_sizes = [100 * G.degree(node) for node in G.nodes()]
        
        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.8,
            cmap=cmap,
            edgecolors='black',
            linewidths=2
        )
        
        # Draw edges (color by positive/negative correlation)
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        edge_colors = ['green' if w > 0 else 'red' for w in weights]
        
        nx.draw_networkx_edges(
            G, pos,
            width=[5 * abs(w) for w in weights],
            alpha=0.4,
            edge_color=edge_colors
        )
        
        # Labels
        nx.draw_networkx_labels(
            G, pos,
            font_size=8,
            font_weight='bold'
        )
        
        plt.title(
            f'Correlation Network (|r| > {threshold})\n'
            f'Green = Positive, Red = Negative',
            fontsize=16, fontweight='bold', pad=20
        )
        plt.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Saved to {save_path}")
        
        plt.show()
    
    # ==================== FORCE-DIRECTED GRAPH ====================
    
    def plot_force_directed_graph_interactive(
        self,
        correlation_matrix: np.ndarray,
        tickers: List[str],
        threshold: float = 0.3,
        regimes: Optional[pd.Series] = None,
        save_path: Optional[str] = None
    ):
        """
        Create interactive force-directed graph using Plotly
        
        Args:
            correlation_matrix: Asset correlations
            tickers: Asset names
            threshold: Minimum correlation threshold
            regimes: Optional regime labels
            save_path: Path to save HTML
        """
        print(f"\n⚡ Creating Interactive Force-Directed Graph...")
        
        n = len(correlation_matrix)
        
        # Create graph
        G = nx.Graph()
        
        for i in range(n):
            for j in range(i+1, n):
                corr = correlation_matrix[i, j]
                if abs(corr) >= threshold:
                    G.add_edge(tickers[i], tickers[j], weight=corr)
        
        if len(G.edges) == 0:
            print("⚠️ No edges above threshold")
            return
        
        # Layout
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        
        # Create edges
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Create nodes
        node_x = []
        node_y = []
        node_text = []
        node_colors = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Hover text
            degree = G.degree(node)
            node_text.append(f"{node}<br>Connections: {degree}")
            
            # Color by regime or degree
            if regimes is not None and node in regimes.index:
                node_colors.append(regimes[node])
            else:
                node_colors.append(degree)
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=list(G.nodes()),
            textposition='top center',
            textfont=dict(size=8),
            hovertext=node_text,
            marker=dict(
                showscale=True,
                colorscale='Viridis',
                color=node_colors,
                size=10,
                colorbar=dict(
                    thickness=15,
                    title='Regime' if regimes is not None else 'Degree',
                    xanchor='left',
                    titleside='right'
                ),
                line=dict(width=2, color='white')
            )
        )
        
        # Create figure
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=dict(
                    text=f'Interactive Correlation Network<br>(|r| > {threshold})',
                    x=0.5,
                    xanchor='center',
                    font=dict(size=20)
                ),
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=80),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                width=1200,
                height=800
            )
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"💾 Saved interactive graph to {save_path}")
        
        fig.show()
    
    # ==================== SANKEY DIAGRAM ====================
    
    def plot_regime_transitions_sankey(
        self,
        regimes: pd.Series,
        regime_names: Dict[int, str] = None,
        save_path: Optional[str] = None
    ):
        """
        Create Sankey diagram of regime transitions
        
        Args:
            regimes: Time series of regime labels
            regime_names: Optional mapping of regime IDs to names
            save_path: Path to save HTML
        """
        print(f"\n🌊 Creating Sankey Diagram of Regime Transitions...")
        
        # Compute transitions
        transitions = pd.DataFrame({
            'from': regimes[:-1].values,
            'to': regimes[1:].values
        })
        
        # Count transitions
        transition_counts = transitions.groupby(['from', 'to']).size().reset_index(name='count')
        
        # Get unique regimes
        unique_regimes = sorted(regimes.unique())
        
        if regime_names is None:
            regime_names = {r: f"Regime {r}" for r in unique_regimes}
        
        # Create labels (duplicate for source and target)
        labels = []
        for regime in unique_regimes:
            labels.append(f"{regime_names[regime]} (t)")
        for regime in unique_regimes:
            labels.append(f"{regime_names[regime]} (t+1)")
        
        n_regimes = len(unique_regimes)
        
        # Map regime IDs to indices
        regime_to_idx = {r: i for i, r in enumerate(unique_regimes)}
        
        # Create source, target, value lists
        source = []
        target = []
        value = []
        colors = []
        
        for _, row in transition_counts.iterrows():
            from_regime = int(row['from'])
            to_regime = int(row['to'])
            count = row['count']
            
            source.append(regime_to_idx[from_regime])
            target.append(n_regimes + regime_to_idx[to_regime])
            value.append(count)
            
            # Color: green if same regime, orange if different
            if from_regime == to_regime:
                colors.append('rgba(0, 255, 0, 0.4)')
            else:
                colors.append('rgba(255, 165, 0, 0.4)')
        
        # Create Sankey
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color='black', width=0.5),
                label=labels,
                color=['lightblue'] * n_regimes + ['lightcoral'] * n_regimes
            ),
            link=dict(
                source=source,
                target=target,
                value=value,
                color=colors
            )
        )])
        
        fig.update_layout(
            title=dict(
                text='Market Regime Transitions<br>Sankey Diagram',
                x=0.5,
                xanchor='center',
                font=dict(size=20)
            ),
            font=dict(size=12),
            width=1200,
            height=600
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"💾 Saved Sankey diagram to {save_path}")
        
        fig.show()
        
        # Print transition matrix
        print("\n   Transition Probabilities:")
        trans_matrix = pd.crosstab(
            transitions['from'], 
            transitions['to'], 
            normalize='index'
        ).round(3)
        print(trans_matrix)
    
    # ==================== DENDROGRAM ====================
    
    def plot_hierarchical_clustering(
        self,
        correlation_matrix: np.ndarray,
        tickers: List[str],
        save_path: Optional[str] = None
    ):
        """
        Plot hierarchical clustering dendrogram
        
        Args:
            correlation_matrix: Asset correlations
            tickers: Asset names
            save_path: Path to save
        """
        print(f"\n🌲 Creating Hierarchical Clustering Dendrogram...")
        
        # Convert correlation to distance
        distance_matrix = np.sqrt(2 * (1 - correlation_matrix))
        
        # Convert to condensed distance matrix
        condensed_dist = squareform(distance_matrix)
        
        # Perform hierarchical clustering
        linkage_matrix = linkage(condensed_dist, method='average')
        
        # Plot
        plt.figure(figsize=(16, 8))
        
        dendrogram(
            linkage_matrix,
            labels=tickers,
            leaf_rotation=90,
            leaf_font_size=10,
            color_threshold=0.7 * max(linkage_matrix[:, 2])
        )
        
        plt.title(
            'Hierarchical Clustering of Assets\n'
            'Based on Correlation Distance',
            fontsize=16, fontweight='bold', pad=20
        )
        plt.xlabel('Asset', fontsize=12, fontweight='bold')
        plt.ylabel('Distance', fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Saved to {save_path}")
        
        plt.show()


if __name__ == "__main__":
    print("="*60)
    print("NETWORK VISUALIZATION DEMO")
    print("="*60)
    
    # Create sample data
    np.random.seed(42)
    n_assets = 20
    tickers = [f"TICK{i}" for i in range(n_assets)]
    
    # Generate random correlation matrix
    random_matrix = np.random.randn(n_assets, n_assets)
    correlation_matrix = np.corrcoef(random_matrix)
    
    # Create regimes
    regimes = pd.Series(
        np.random.choice([0, 1, 2], size=50),
        index=pd.date_range('2024-01-01', periods=50)
    )
    
    visualizer = NetworkVisualizer()
    
    # Test MST
    mst = visualizer.compute_mst(correlation_matrix, tickers)
    print(f"✅ MST has {len(mst.edges)} edges")
    
    # Test Sankey
    visualizer.plot_regime_transitions_sankey(regimes)
    
    print("\n✅ Network visualization completed successfully!")

