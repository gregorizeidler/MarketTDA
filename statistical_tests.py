"""
Statistical Testing and Bootstrap Methods for Topological Features
Confidence intervals, hypothesis testing, cross-correlation analysis
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Callable, Dict
from scipy import stats
from scipy.signal import correlate
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class TopologicalStatisticalTests:
    """Statistical tests and bootstrap methods for TDA"""
    
    def __init__(self, random_state: int = 42):
        """Initialize statistical testing framework"""
        self.random_state = random_state
        np.random.seed(random_state)
    
    # ==================== BOOTSTRAP METHODS ====================
    
    def bootstrap_betti_numbers(
        self,
        point_cloud: np.ndarray,
        compute_persistence_func: Callable,
        n_bootstrap: int = 100,
        sample_fraction: float = 0.8,
        confidence_level: float = 0.95
    ) -> Dict[str, Dict[str, float]]:
        """
        Bootstrap confidence intervals for Betti numbers
        
        Args:
            point_cloud: Market point cloud (n_points, n_dimensions)
            compute_persistence_func: Function that computes persistence from point cloud
            n_bootstrap: Number of bootstrap samples
            sample_fraction: Fraction of data to sample
            confidence_level: Confidence level (e.g., 0.95 for 95% CI)
            
        Returns:
            Dictionary with CI for H0, H1, H2 counts
        """
        print(f"\n🔁 Running Bootstrap (n={n_bootstrap})...")
        
        n_points = len(point_cloud)
        sample_size = int(n_points * sample_fraction)
        
        # Store bootstrap results
        h0_counts = []
        h1_counts = []
        h2_counts = []
        
        for i in tqdm(range(n_bootstrap), desc="Bootstrap samples"):
            # Resample with replacement
            indices = np.random.choice(n_points, size=sample_size, replace=True)
            bootstrap_cloud = point_cloud[indices]
            
            # Compute persistence
            try:
                diagrams = compute_persistence_func(bootstrap_cloud)
                
                # Count Betti numbers
                if isinstance(diagrams, list):
                    # Ripser format
                    h0_counts.append(len(diagrams[0]))
                    h1_counts.append(len(diagrams[1]) if len(diagrams) > 1 else 0)
                    h2_counts.append(len(diagrams[2]) if len(diagrams) > 2 else 0)
                else:
                    # Array format
                    h0_counts.append(np.sum(diagrams[:, 2] == 0))
                    h1_counts.append(np.sum(diagrams[:, 2] == 1))
                    h2_counts.append(np.sum(diagrams[:, 2] == 2))
            except Exception as e:
                # Skip failed samples
                continue
        
        # Compute confidence intervals
        alpha = 1 - confidence_level
        
        results = {}
        for dim, counts in [('H0', h0_counts), ('H1', h1_counts), ('H2', h2_counts)]:
            if len(counts) > 0:
                results[dim] = {
                    'mean': np.mean(counts),
                    'std': np.std(counts),
                    'median': np.median(counts),
                    'ci_lower': np.percentile(counts, 100 * alpha/2),
                    'ci_upper': np.percentile(counts, 100 * (1 - alpha/2)),
                    'samples': counts
                }
        
        print("✅ Bootstrap complete")
        return results
    
    # ==================== HYPOTHESIS TESTING ====================
    
    def permutation_test(
        self,
        group1_features: pd.DataFrame,
        group2_features: pd.DataFrame,
        metric: str = 'H1_count',
        n_permutations: int = 1000
    ) -> Tuple[float, float]:
        """
        Permutation test for difference in topological features between two groups
        (e.g., stable vs crisis regimes)
        
        Args:
            group1_features: Features from first group
            group2_features: Features from second group
            metric: Feature to test
            n_permutations: Number of permutations
            
        Returns:
            test_statistic: Observed difference in means
            p_value: Permutation p-value
        """
        print(f"\n🧪 Permutation Test: {metric}")
        
        # Observed statistic
        obs_stat = group1_features[metric].mean() - group2_features[metric].mean()
        
        # Combine data
        combined = np.concatenate([
            group1_features[metric].values,
            group2_features[metric].values
        ])
        
        n1 = len(group1_features)
        n2 = len(group2_features)
        
        # Permutation distribution
        perm_stats = []
        for _ in tqdm(range(n_permutations), desc="Permutations"):
            # Shuffle
            np.random.shuffle(combined)
            
            # Split
            perm_group1 = combined[:n1]
            perm_group2 = combined[n1:]
            
            # Compute statistic
            perm_stat = perm_group1.mean() - perm_group2.mean()
            perm_stats.append(perm_stat)
        
        # Compute p-value
        perm_stats = np.array(perm_stats)
        p_value = np.mean(np.abs(perm_stats) >= np.abs(obs_stat))
        
        print(f"   Observed difference: {obs_stat:.4f}")
        print(f"   P-value: {p_value:.4f}")
        print(f"   Significant: {'Yes' if p_value < 0.05 else 'No'} (α=0.05)")
        
        return obs_stat, p_value
    
    def test_regime_differences(
        self,
        features: pd.DataFrame,
        regimes: pd.Series,
        metrics: List[str] = None
    ) -> pd.DataFrame:
        """
        Test if topological features differ significantly between regimes
        
        Returns:
            DataFrame with test results for each metric
        """
        if metrics is None:
            metrics = ['H0_count', 'H1_count', 'H2_count',
                      'H1_max_persistence', 'H2_max_persistence']
        
        print(f"\n📊 Testing {len(metrics)} metrics across regimes...")
        
        # Get unique regimes
        unique_regimes = regimes.unique()
        
        if len(unique_regimes) < 2:
            print("⚠️ Need at least 2 regimes for comparison")
            return pd.DataFrame()
        
        # Compare first two regimes
        regime1_mask = regimes == unique_regimes[0]
        regime2_mask = regimes == unique_regimes[1]
        
        results = []
        for metric in metrics:
            if metric not in features.columns:
                continue
            
            group1 = features[regime1_mask]
            group2 = features[regime2_mask]
            
            if len(group1) < 2 or len(group2) < 2:
                continue
            
            # Permutation test
            stat, pval = self.permutation_test(
                group1, group2, metric=metric, n_permutations=1000
            )
            
            results.append({
                'metric': metric,
                'regime1_mean': group1[metric].mean(),
                'regime2_mean': group2[metric].mean(),
                'difference': stat,
                'p_value': pval,
                'significant': pval < 0.05
            })
        
        return pd.DataFrame(results)
    
    # ==================== ENTROPY RATE ====================
    
    def persistent_entropy_rate(
        self,
        features: pd.DataFrame,
        window_size: int = 5
    ) -> pd.Series:
        """
        Compute rate of change of persistent entropy
        
        Args:
            features: DataFrame with entropy features
            window_size: Window for computing rate
            
        Returns:
            Series with entropy rate for each dimension
        """
        print(f"\n📈 Computing Persistent Entropy Rate...")
        
        rates = {}
        for dim in [0, 1, 2]:
            entropy_col = f'H{dim}_entropy'
            if entropy_col in features.columns:
                entropy = features[entropy_col].values
                
                # Compute rolling gradient
                rate = np.gradient(entropy)
                
                # Smooth with moving average
                if len(rate) >= window_size:
                    rate = pd.Series(rate).rolling(window_size, center=True).mean().values
                
                rates[f'H{dim}_entropy_rate'] = rate
        
        return pd.DataFrame(rates, index=features.index)
    
    # ==================== CROSS-CORRELATION ====================
    
    def cross_correlation_analysis(
        self,
        features: pd.DataFrame,
        max_lag: int = 10
    ) -> Dict[str, np.ndarray]:
        """
        Compute cross-correlation between H0, H1, H2 with time lags
        
        Args:
            features: DataFrame with topological features
            max_lag: Maximum lag to consider
            
        Returns:
            Dictionary with cross-correlation results
        """
        print(f"\n🔗 Cross-Correlation Analysis (max_lag={max_lag})...")
        
        # Extract time series
        h0 = features['H0_count'].values
        h1 = features['H1_count'].values
        h2 = features['H2_count'].values if 'H2_count' in features.columns else None
        
        # Standardize
        h0 = (h0 - h0.mean()) / h0.std()
        h1 = (h1 - h1.mean()) / h1.std()
        if h2 is not None and len(h2) > 0:
            h2 = (h2 - h2.mean()) / (h2.std() + 1e-10)
        
        results = {}
        
        # H0 vs H1
        ccf_h0_h1 = self._compute_ccf(h0, h1, max_lag)
        results['H0_H1'] = ccf_h0_h1
        
        # H1 vs H2
        if h2 is not None:
            ccf_h1_h2 = self._compute_ccf(h1, h2, max_lag)
            results['H1_H2'] = ccf_h1_h2
        
        # H0 vs H2
        if h2 is not None:
            ccf_h0_h2 = self._compute_ccf(h0, h2, max_lag)
            results['H0_H2'] = ccf_h0_h2
        
        # Print significant lags
        print("\n   Significant lagged correlations (|r| > 0.3):")
        for pair, ccf in results.items():
            lags = np.arange(-max_lag, max_lag + 1)
            significant = np.abs(ccf) > 0.3
            if significant.any():
                for lag, corr in zip(lags[significant], ccf[significant]):
                    print(f"      {pair} at lag {lag:+d}: {corr:.3f}")
        
        return results
    
    def _compute_ccf(
        self,
        x: np.ndarray,
        y: np.ndarray,
        max_lag: int
    ) -> np.ndarray:
        """Compute cross-correlation function"""
        n = len(x)
        ccf = correlate(x, y, mode='full') / n
        
        # Extract relevant lags
        center = len(ccf) // 2
        ccf = ccf[center - max_lag : center + max_lag + 1]
        
        return ccf
    
    # ==================== TOPOLOGICAL LOSS ====================
    
    def topological_loss(
        self,
        features: pd.DataFrame,
        weights: Dict[str, float] = None
    ) -> pd.Series:
        """
        Compute topological instability/loss function
        
        High loss = unstable topology (high H1, H2, low persistence)
        
        Args:
            features: DataFrame with topological features
            weights: Custom weights for each component
            
        Returns:
            Series with loss values
        """
        if weights is None:
            weights = {
                'H1_count': 0.3,
                'H2_count': 0.5,
                'H1_persistence': -0.2,  # Negative = lower persistence = higher loss
            }
        
        loss = np.zeros(len(features))
        
        for metric, weight in weights.items():
            if metric in features.columns:
                values = features[metric].values
                # Normalize
                values_norm = (values - values.mean()) / (values.std() + 1e-10)
                loss += weight * values_norm
        
        return pd.Series(loss, index=features.index, name='topological_loss')
    
    # ==================== VISUALIZATIONS ====================
    
    def plot_bootstrap_distributions(
        self,
        bootstrap_results: Dict[str, Dict[str, float]],
        save_path: Optional[str] = None
    ):
        """Plot bootstrap distributions with confidence intervals"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, (dim, results) in enumerate(bootstrap_results.items()):
            ax = axes[i]
            samples = results['samples']
            
            # Histogram
            ax.hist(samples, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
            
            # Mean and CI
            ax.axvline(results['mean'], color='red', linestyle='--', 
                      linewidth=2, label=f"Mean: {results['mean']:.1f}")
            ax.axvline(results['ci_lower'], color='orange', linestyle=':', 
                      linewidth=2, label=f"95% CI: [{results['ci_lower']:.1f}, {results['ci_upper']:.1f}]")
            ax.axvline(results['ci_upper'], color='orange', linestyle=':', linewidth=2)
            
            ax.set_xlabel(f'{dim} Count', fontsize=12, fontweight='bold')
            ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
            ax.set_title(f'{dim} Bootstrap Distribution', fontsize=13, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Saved to {save_path}")
        
        plt.show()
    
    def plot_cross_correlations(
        self,
        ccf_results: Dict[str, np.ndarray],
        max_lag: int,
        save_path: Optional[str] = None
    ):
        """Plot cross-correlation functions"""
        n_pairs = len(ccf_results)
        fig, axes = plt.subplots(1, n_pairs, figsize=(5*n_pairs, 4))
        
        if n_pairs == 1:
            axes = [axes]
        
        lags = np.arange(-max_lag, max_lag + 1)
        
        for ax, (pair, ccf) in zip(axes, ccf_results.items()):
            ax.stem(lags, ccf, basefmt=' ')
            ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
            ax.axhline(0.3, color='red', linestyle='--', alpha=0.5, label='Threshold')
            ax.axhline(-0.3, color='red', linestyle='--', alpha=0.5)
            
            ax.set_xlabel('Lag', fontsize=12, fontweight='bold')
            ax.set_ylabel('Cross-Correlation', fontsize=12, fontweight='bold')
            ax.set_title(f'{pair} Cross-Correlation', fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Saved to {save_path}")
        
        plt.show()
    
    def plot_entropy_rate(
        self,
        features: pd.DataFrame,
        entropy_rates: pd.DataFrame,
        save_path: Optional[str] = None
    ):
        """Plot persistent entropy and its rate of change"""
        fig, axes = plt.subplots(3, 2, figsize=(14, 10))
        
        colors = {0: '#1f77b4', 1: '#ff7f0e', 2: '#2ca02c'}
        
        for dim in [0, 1, 2]:
            entropy_col = f'H{dim}_entropy'
            rate_col = f'H{dim}_entropy_rate'
            
            if entropy_col in features.columns:
                # Entropy
                axes[dim, 0].plot(features.index, features[entropy_col],
                                 linewidth=2, color=colors[dim], marker='o', markersize=4)
                axes[dim, 0].fill_between(features.index, features[entropy_col],
                                         alpha=0.3, color=colors[dim])
                axes[dim, 0].set_ylabel(f'H{dim} Entropy', fontsize=11, fontweight='bold')
                axes[dim, 0].grid(True, alpha=0.3)
                
                # Rate
                if rate_col in entropy_rates.columns:
                    axes[dim, 1].plot(entropy_rates.index, entropy_rates[rate_col],
                                     linewidth=2, color=colors[dim], marker='o', markersize=4)
                    axes[dim, 1].fill_between(entropy_rates.index, entropy_rates[rate_col],
                                             alpha=0.3, color=colors[dim])
                    axes[dim, 1].axhline(0, color='red', linestyle='--', alpha=0.5)
                    axes[dim, 1].set_ylabel(f'H{dim} Entropy Rate', fontsize=11, fontweight='bold')
                    axes[dim, 1].grid(True, alpha=0.3)
        
        axes[2, 0].set_xlabel('Date', fontsize=12, fontweight='bold')
        axes[2, 1].set_xlabel('Date', fontsize=12, fontweight='bold')
        
        fig.suptitle('Persistent Entropy and Rate of Change', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Saved to {save_path}")
        
        plt.show()


if __name__ == "__main__":
    print("="*60)
    print("STATISTICAL TESTS DEMO")
    print("="*60)
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=50, freq='D')
    
    features = pd.DataFrame({
        'H0_count': 30 + np.random.randn(50) * 2,
        'H1_count': 10 + np.random.randn(50) * 3 + np.linspace(0, 5, 50),
        'H2_count': np.random.poisson(0.5, 50),
        'H0_entropy': 2 + np.random.randn(50) * 0.5,
        'H1_entropy': 1.5 + np.random.randn(50) * 0.3,
        'H2_entropy': 0.5 + np.random.randn(50) * 0.2,
    }, index=dates)
    
    tester = TopologicalStatisticalTests()
    
    # Test entropy rate
    rates = tester.persistent_entropy_rate(features)
    print(f"\n✅ Computed entropy rates: {rates.columns.tolist()}")
    
    # Test cross-correlation
    ccf_results = tester.cross_correlation_analysis(features, max_lag=10)
    print(f"\n✅ Computed cross-correlations for {len(ccf_results)} pairs")
    
    # Test topological loss
    loss = tester.topological_loss(features)
    print(f"\n✅ Computed topological loss (mean: {loss.mean():.3f})")
    
    print("\n✅ All tests completed successfully!")

