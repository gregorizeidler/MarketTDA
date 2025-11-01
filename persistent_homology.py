"""
Persistent Homology Engine for Market Topology Analysis
Computes H0 (clusters), H1 (loops), H2 (voids) using TDA
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from ripser import ripser
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')


class PersistentHomologyAnalyzer:
    """Analyze market topology using persistent homology"""
    
    def __init__(self, library: str = "ripser"):
        """
        Initialize persistent homology analyzer
        
        Args:
            library: 'ripser' for Ripser (default and recommended)
        """
        self.library = library
        self.diagrams = []
        self.features = []
        self.dates = []
        
    def compute_persistence(
        self,
        point_cloud: np.ndarray,
        max_edge_length: Optional[float] = None
    ) -> np.ndarray:
        """
        Compute persistent homology for a single point cloud
        
        Args:
            point_cloud: Array of shape (n_points, n_dimensions)
            max_edge_length: Maximum edge length for Rips complex
            
        Returns:
            Persistence diagram (birth, death, dimension)
        """
        if self.library == "ripser":
            # Ripser works with distance matrices or point clouds
            result = ripser(
                point_cloud,
                maxdim=2,  # Compute up to H2
                thresh=max_edge_length if max_edge_length else np.inf
            )
            
            # Convert to standard format (birth, death, dimension)
            diagrams = []
            for dim in range(3):  # H0, H1, H2
                if dim < len(result['dgms']):
                    dgm = result['dgms'][dim]
                    # Remove infinite death times (for H0 connected components)
                    dgm = dgm[np.isfinite(dgm).all(axis=1)]
                    
                    # Add dimension column
                    dim_col = np.full((len(dgm), 1), dim)
                    dgm_with_dim = np.hstack([dgm, dim_col])
                    diagrams.append(dgm_with_dim)
            
            # Combine all dimensions
            if diagrams:
                diagram = np.vstack(diagrams)
            else:
                diagram = np.empty((0, 3))
            
            return diagram
        
        else:
            raise ValueError(f"Unknown library: {self.library}")
    
    def analyze_windows(
        self,
        point_clouds: List[np.ndarray],
        dates: List[pd.Timestamp],
        max_edge_length: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Analyze persistent homology for multiple time windows
        
        Args:
            point_clouds: List of point cloud arrays
            dates: Corresponding dates
            max_edge_length: Maximum edge length for filtration
            
        Returns:
            DataFrame with topological features over time
        """
        print(f"\n🔬 Computing Persistent Homology...")
        print(f"   Library: {self.library}")
        print(f"   Windows: {len(point_clouds)}")
        print(f"   Computing H0, H1, H2...")
        
        results = []
        self.diagrams = []
        self.dates = dates
        
        from tqdm import tqdm
        for i, (pc, date) in enumerate(tqdm(zip(point_clouds, dates), 
                                            total=len(point_clouds),
                                            desc="Processing windows")):
            try:
                # Compute persistence diagram
                diagram = self.compute_persistence(pc, max_edge_length)
                self.diagrams.append(diagram)
                
                # Extract features from diagram
                features = self._extract_features(diagram, date)
                results.append(features)
                
            except Exception as e:
                print(f"⚠️ Error processing window {i} ({date}): {e}")
                # Add empty features
                results.append(self._empty_features(date))
        
        # Convert to DataFrame
        self.features = pd.DataFrame(results)
        self.features.set_index('date', inplace=True)
        
        print(f"✅ Persistent homology computed")
        print(f"   Features extracted: {len(self.features.columns)}")
        
        return self.features
    
    def _extract_features(self, diagram: np.ndarray, date: pd.Timestamp) -> dict:
        """
        Extract topological features from persistence diagram
        
        Args:
            diagram: Persistence diagram (birth, death, dimension)
            date: Date of this window
            
        Returns:
            Dictionary of features
        """
        features = {'date': date}
        
        # Separate by dimension
        for dim in [0, 1, 2]:
            mask = diagram[:, 2] == dim
            dgm_dim = diagram[mask]
            
            # Persistence = death - birth
            if len(dgm_dim) > 0:
                persistence = dgm_dim[:, 1] - dgm_dim[:, 0]
                
                # Basic statistics
                features[f'H{dim}_count'] = len(dgm_dim)
                features[f'H{dim}_max_persistence'] = persistence.max()
                features[f'H{dim}_mean_persistence'] = persistence.mean()
                features[f'H{dim}_sum_persistence'] = persistence.sum()
                features[f'H{dim}_std_persistence'] = persistence.std()
                
                # Betti numbers (at various thresholds)
                # Estimate the scale at which features exist
                if len(persistence) > 0:
                    # Features alive at mean birth time
                    mean_birth = dgm_dim[:, 0].mean()
                    alive_at_mean = np.sum((dgm_dim[:, 0] <= mean_birth) & 
                                          (dgm_dim[:, 1] >= mean_birth))
                    features[f'H{dim}_betti_mean'] = alive_at_mean
                else:
                    features[f'H{dim}_betti_mean'] = 0
                
                # Persistent entropy (measure of complexity)
                if persistence.sum() > 0:
                    p = persistence / persistence.sum()
                    entropy = -np.sum(p * np.log(p + 1e-10))
                    features[f'H{dim}_entropy'] = entropy
                else:
                    features[f'H{dim}_entropy'] = 0
                
            else:
                # No features in this dimension
                features[f'H{dim}_count'] = 0
                features[f'H{dim}_max_persistence'] = 0
                features[f'H{dim}_mean_persistence'] = 0
                features[f'H{dim}_sum_persistence'] = 0
                features[f'H{dim}_std_persistence'] = 0
                features[f'H{dim}_betti_mean'] = 0
                features[f'H{dim}_entropy'] = 0
        
        return features
    
    def _empty_features(self, date: pd.Timestamp) -> dict:
        """Create empty feature dictionary for failed analysis"""
        features = {'date': date}
        for dim in [0, 1, 2]:
            features[f'H{dim}_count'] = 0
            features[f'H{dim}_max_persistence'] = 0
            features[f'H{dim}_mean_persistence'] = 0
            features[f'H{dim}_sum_persistence'] = 0
            features[f'H{dim}_std_persistence'] = 0
            features[f'H{dim}_betti_mean'] = 0
            features[f'H{dim}_entropy'] = 0
        return features
    
    def get_diagram_summary(self, window_idx: int = -1) -> dict:
        """
        Get summary of persistence diagram for a specific window
        
        Args:
            window_idx: Which window to summarize
            
        Returns:
            Dictionary with summary statistics
        """
        if not self.diagrams:
            return {}
        
        diagram = self.diagrams[window_idx]
        date = self.dates[window_idx]
        
        summary = {
            'date': date,
            'total_features': len(diagram)
        }
        
        for dim in [0, 1, 2]:
            mask = diagram[:, 2] == dim
            dgm_dim = diagram[mask]
            persistence = dgm_dim[:, 1] - dgm_dim[:, 0] if len(dgm_dim) > 0 else np.array([])
            
            summary[f'H{dim}'] = {
                'count': len(dgm_dim),
                'max_persistence': persistence.max() if len(persistence) > 0 else 0,
                'mean_birth': dgm_dim[:, 0].mean() if len(dgm_dim) > 0 else 0,
                'mean_death': dgm_dim[:, 1].mean() if len(dgm_dim) > 0 else 0,
            }
        
        return summary
    
    def interpret_topology(self, window_idx: int = -1) -> str:
        """
        Generate human-readable interpretation of topology
        
        Args:
            window_idx: Which window to interpret
            
        Returns:
            String interpretation
        """
        summary = self.get_diagram_summary(window_idx)
        date = summary.get('date', 'Unknown')
        
        interpretation = [
            f"\n{'='*60}",
            f"TOPOLOGICAL INTERPRETATION - {date.date() if hasattr(date, 'date') else date}",
            f"{'='*60}\n"
        ]
        
        # H0 - Connected Components / Clusters
        h0 = summary.get('H0', {})
        h0_count = h0.get('count', 0)
        h0_persistence = h0.get('max_persistence', 0)
        
        interpretation.append("🔵 H₀ (Clusters / Connected Components):")
        interpretation.append(f"   Count: {h0_count}")
        interpretation.append(f"   Max Persistence: {h0_persistence:.4f}")
        if h0_count > 1:
            interpretation.append("   → Market shows distinct asset clusters")
            interpretation.append("   → Fragmented correlation structure")
        else:
            interpretation.append("   → Highly connected market (single cluster)")
            interpretation.append("   → Strong inter-asset correlations")
        
        # H1 - Loops / Cyclic Dependencies
        h1 = summary.get('H1', {})
        h1_count = h1.get('count', 0)
        h1_persistence = h1.get('max_persistence', 0)
        
        interpretation.append("\n🔴 H₁ (Loops / Cyclic Interdependencies):")
        interpretation.append(f"   Count: {h1_count}")
        interpretation.append(f"   Max Persistence: {h1_persistence:.4f}")
        if h1_count > 0 and h1_persistence > 0.1:
            interpretation.append("   → SIGNIFICANT cyclic dependencies detected!")
            interpretation.append("   → Assets form feedback loops (not just correlations)")
            interpretation.append("   → Potential for contagion or cascade effects")
        elif h1_count > 0:
            interpretation.append("   → Weak cyclic structures present")
            interpretation.append("   → Some interdependencies beyond pairwise correlations")
        else:
            interpretation.append("   → No significant loops detected")
            interpretation.append("   → Market structure is tree-like or linear")
        
        # H2 - Voids / Structural Fragility
        h2 = summary.get('H2', {})
        h2_count = h2.get('count', 0)
        h2_persistence = h2.get('max_persistence', 0)
        
        interpretation.append("\n⚪ H₂ (Voids / Structural Fragility):")
        interpretation.append(f"   Count: {h2_count}")
        interpretation.append(f"   Max Persistence: {h2_persistence:.4f}")
        if h2_count > 0 and h2_persistence > 0.05:
            interpretation.append("   → ⚠️ VOIDS DETECTED - STRUCTURAL FRAGILITY!")
            interpretation.append("   → Empty regions in return space")
            interpretation.append("   → Possible arbitrage opportunities")
            interpretation.append("   → Warning: System instability or market anomaly")
        elif h2_count > 0:
            interpretation.append("   → Small voids present")
            interpretation.append("   → Minor structural inefficiencies")
        else:
            interpretation.append("   → No voids detected")
            interpretation.append("   → Dense, stable market structure")
        
        interpretation.append(f"\n{'='*60}\n")
        
        return "\n".join(interpretation)
    
    def compute_topological_distance(self, idx1: int, idx2: int) -> float:
        """
        Compute distance between two persistence diagrams
        Useful for detecting regime changes
        
        Args:
            idx1: First window index
            idx2: Second window index
            
        Returns:
            Simple L2 distance between feature vectors
        """
        dgm1 = self.diagrams[idx1]
        dgm2 = self.diagrams[idx2]
        
        # Simple distance based on feature counts
        h1_diff = abs(len(dgm1[dgm1[:, 2] == 1]) - len(dgm2[dgm2[:, 2] == 1]))
        h2_diff = abs(len(dgm1[dgm1[:, 2] == 2]) - len(dgm2[dgm2[:, 2] == 2]))
        
        return np.sqrt(h1_diff**2 + h2_diff**2)


if __name__ == "__main__":
    # Example usage
    from data_fetcher import MarketDataFetcher
    from point_cloud import MarketPointCloud
    
    print("="*60)
    print("PERSISTENT HOMOLOGY DEMO")
    print("="*60)
    
    # Fetch and prepare data
    fetcher = MarketDataFetcher()
    data = fetcher.fetch_data(period="1y", max_tickers=30)
    returns = fetcher.compute_returns()
    
    # Create point clouds
    pc = MarketPointCloud(returns)
    windows = pc.create_sliding_windows(window_size=40, step_size=15)
    point_clouds = [w[0] for w in windows]
    dates = [w[1] for w in windows]
    
    # Compute persistent homology
    analyzer = PersistentHomologyAnalyzer(library="ripser")
    features = analyzer.analyze_windows(point_clouds, dates)
    
    print("\n" + "="*60)
    print("TOPOLOGICAL FEATURES (Latest 5 Windows)")
    print("="*60)
    print(features[['H0_count', 'H1_count', 'H2_count', 
                    'H1_max_persistence', 'H2_max_persistence']].tail())
    
    # Interpret latest window
    print(analyzer.interpret_topology(-1))

