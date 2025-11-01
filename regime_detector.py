"""
Market Regime Detection using Topological Features
Identifies structural changes, crises, and regime shifts
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.signal import find_peaks
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns


class MarketRegimeDetector:
    """Detect market regimes using topological features"""
    
    def __init__(self, features: pd.DataFrame):
        """
        Initialize regime detector
        
        Args:
            features: DataFrame with topological features from persistent homology
        """
        self.features = features
        self.regimes = None
        self.regime_labels = None
        self.anomalies = None
        self.transitions = None
        
    def detect_regimes(
        self,
        n_regimes: int = 4,
        features_to_use: Optional[List[str]] = None
    ) -> pd.Series:
        """
        Cluster windows into distinct market regimes
        
        Args:
            n_regimes: Number of regimes to identify
            features_to_use: List of feature columns to use (default: key features)
            
        Returns:
            Series with regime labels for each window
        """
        print(f"\n🎯 Detecting Market Regimes...")
        print(f"   Number of regimes: {n_regimes}")
        
        # Select features
        if features_to_use is None:
            features_to_use = [
                'H0_count', 'H1_count', 'H2_count',
                'H0_max_persistence', 'H1_max_persistence', 'H2_max_persistence',
                'H1_entropy', 'H2_entropy'
            ]
        
        # Filter available features
        features_to_use = [f for f in features_to_use if f in self.features.columns]
        
        print(f"   Using {len(features_to_use)} features")
        
        # Prepare data
        X = self.features[features_to_use].values
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Cluster
        kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init=20)
        labels = kmeans.fit_predict(X_scaled)
        
        # Create regime series
        self.regime_labels = pd.Series(labels, index=self.features.index, name='regime')
        
        # Assign meaningful names based on topology
        self.regimes = self._interpret_regimes(labels, X_scaled, features_to_use)
        
        print(f"✅ Regimes detected")
        print(f"\nRegime Distribution:")
        for regime_name, count in self.regimes['regime_name'].value_counts().items():
            print(f"   {regime_name}: {count} windows")
        
        return self.regime_labels
    
    def _interpret_regimes(
        self, 
        labels: np.ndarray, 
        X_scaled: np.ndarray,
        feature_names: List[str]
    ) -> pd.DataFrame:
        """
        Interpret regime clusters based on topological characteristics
        
        Args:
            labels: Cluster labels
            X_scaled: Scaled feature matrix
            feature_names: Names of features
            
        Returns:
            DataFrame with regime interpretations
        """
        n_regimes = len(np.unique(labels))
        regimes = []
        
        for i in range(n_regimes):
            mask = labels == i
            cluster_features = X_scaled[mask].mean(axis=0)
            
            # Create feature dict
            feature_dict = dict(zip(feature_names, cluster_features))
            
            # Interpret based on dominant features
            h0_level = feature_dict.get('H0_count', 0)
            h1_level = feature_dict.get('H1_max_persistence', 0)
            h2_level = feature_dict.get('H2_max_persistence', 0)
            
            # Assign names based on topology
            if h2_level > 0.5:
                name = "🔴 CRISIS / FRAGILE"
                description = "High H₂ voids - structural fragility"
            elif h1_level > 0.5:
                name = "🟠 COMPLEX / CYCLICAL"
                description = "Strong H₁ loops - cyclic dependencies"
            elif h0_level > 0.5:
                name = "🟡 FRAGMENTED"
                description = "Many H₀ clusters - disconnected markets"
            elif h1_level < -0.5 and h2_level < -0.5:
                name = "🟢 STABLE / NORMAL"
                description = "Low complexity - stable structure"
            else:
                name = f"🔵 REGIME {i}"
                description = "Mixed characteristics"
            
            regimes.append({
                'regime_id': i,
                'regime_name': name,
                'description': description,
                'count': mask.sum(),
                **feature_dict
            })
        
        return pd.DataFrame(regimes)
    
    def detect_anomalies(
        self,
        features_to_monitor: Optional[List[str]] = None,
        threshold: float = 2.5
    ) -> pd.DataFrame:
        """
        Detect anomalous windows based on topological features
        
        Args:
            features_to_monitor: Features to check for anomalies
            threshold: Z-score threshold for anomaly detection
            
        Returns:
            DataFrame with anomaly flags and scores
        """
        print(f"\n🚨 Detecting Topological Anomalies...")
        print(f"   Threshold: {threshold} standard deviations")
        
        if features_to_monitor is None:
            # Focus on H1 and H2 (loops and voids are most important)
            features_to_monitor = [
                'H1_max_persistence', 'H2_max_persistence',
                'H1_count', 'H2_count'
            ]
        
        # Filter available features
        features_to_monitor = [f for f in features_to_monitor 
                              if f in self.features.columns]
        
        # Compute z-scores
        anomaly_scores = pd.DataFrame(index=self.features.index)
        
        for feature in features_to_monitor:
            z_scores = zscore(self.features[feature])
            anomaly_scores[f'{feature}_zscore'] = z_scores
            anomaly_scores[f'{feature}_anomaly'] = np.abs(z_scores) > threshold
        
        # Overall anomaly flag (any feature anomalous)
        anomaly_cols = [col for col in anomaly_scores.columns if col.endswith('_anomaly')]
        anomaly_scores['is_anomaly'] = anomaly_scores[anomaly_cols].any(axis=1)
        
        # Max absolute z-score
        zscore_cols = [col for col in anomaly_scores.columns if col.endswith('_zscore')]
        anomaly_scores['max_zscore'] = anomaly_scores[zscore_cols].abs().max(axis=1)
        
        self.anomalies = anomaly_scores
        
        n_anomalies = anomaly_scores['is_anomaly'].sum()
        print(f"✅ Found {n_anomalies} anomalous windows")
        
        if n_anomalies > 0:
            print(f"\n⚠️ Anomalous Periods:")
            anomalous_dates = anomaly_scores[anomaly_scores['is_anomaly']].index
            for date in anomalous_dates[:5]:  # Show first 5
                score = anomaly_scores.loc[date, 'max_zscore']
                print(f"   {date.date()}: score = {score:.2f}")
            if n_anomalies > 5:
                print(f"   ... and {n_anomalies - 5} more")
        
        return anomaly_scores
    
    def detect_regime_transitions(self) -> pd.DataFrame:
        """
        Identify regime transition points
        
        Returns:
            DataFrame with transition information
        """
        if self.regime_labels is None:
            print("⚠️ Run detect_regimes() first")
            return pd.DataFrame()
        
        print(f"\n🔄 Detecting Regime Transitions...")
        
        # Find where regime changes
        regime_changes = self.regime_labels != self.regime_labels.shift(1)
        transition_dates = self.regime_labels[regime_changes].index[1:]  # Skip first
        
        transitions = []
        for date in transition_dates:
            idx = self.regime_labels.index.get_loc(date)
            prev_regime = self.regime_labels.iloc[idx - 1]
            curr_regime = self.regime_labels.iloc[idx]
            
            # Get regime names if available
            if self.regimes is not None:
                prev_name = self.regimes.loc[
                    self.regimes['regime_id'] == prev_regime, 'regime_name'
                ].values[0]
                curr_name = self.regimes.loc[
                    self.regimes['regime_id'] == curr_regime, 'regime_name'
                ].values[0]
            else:
                prev_name = f"Regime {prev_regime}"
                curr_name = f"Regime {curr_regime}"
            
            transitions.append({
                'date': date,
                'from_regime': prev_regime,
                'to_regime': curr_regime,
                'from_name': prev_name,
                'to_name': curr_name
            })
        
        self.transitions = pd.DataFrame(transitions)
        
        print(f"✅ Found {len(transitions)} regime transitions")
        
        if len(transitions) > 0:
            print(f"\nRecent Transitions:")
            for _, row in self.transitions.tail(3).iterrows():
                print(f"   {row['date'].date()}: {row['from_name']} → {row['to_name']}")
        
        return self.transitions
    
    def plot_regime_timeline(self, save_path: Optional[str] = None):
        """
        Visualize regimes over time
        
        Args:
            save_path: Path to save figure
        """
        if self.regime_labels is None:
            print("⚠️ Run detect_regimes() first")
            return
        
        fig, axes = plt.subplots(3, 1, figsize=(16, 10))
        
        # Plot 1: Regime timeline
        ax = axes[0]
        
        # Create colored segments
        dates = self.regime_labels.index
        regimes = self.regime_labels.values
        
        # Get regime colors
        unique_regimes = sorted(set(regimes))
        colors = sns.color_palette("husl", len(unique_regimes))
        regime_colors = {r: colors[i] for i, r in enumerate(unique_regimes)}
        
        # Plot as colored background
        for i in range(len(dates) - 1):
            ax.axvspan(dates[i], dates[i+1], 
                      facecolor=regime_colors[regimes[i]], 
                      alpha=0.6)
        
        # Add regime labels
        if self.regimes is not None:
            for regime_id in unique_regimes:
                name = self.regimes.loc[
                    self.regimes['regime_id'] == regime_id, 'regime_name'
                ].values[0]
                ax.scatter([], [], c=[regime_colors[regime_id]], 
                          label=name, s=100)
        
        ax.set_ylabel('Market Regime', fontsize=12, fontweight='bold')
        ax.set_title('Market Regime Timeline', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=9)
        ax.set_yticks([])
        ax.grid(True, alpha=0.3, axis='x')
        
        # Plot 2: Key topological features
        ax = axes[1]
        ax2 = ax.twinx()
        
        ax.plot(self.features.index, self.features['H1_max_persistence'],
               color='orange', linewidth=2, label='H₁ Max Persistence', alpha=0.8)
        ax2.plot(self.features.index, self.features['H2_max_persistence'],
                color='green', linewidth=2, label='H₂ Max Persistence', alpha=0.8)
        
        ax.set_ylabel('H₁ Persistence', color='orange', fontsize=11, fontweight='bold')
        ax2.set_ylabel('H₂ Persistence', color='green', fontsize=11, fontweight='bold')
        ax.tick_params(axis='y', labelcolor='orange')
        ax2.tick_params(axis='y', labelcolor='green')
        ax.grid(True, alpha=0.3)
        ax.set_title('Topological Complexity (Loops & Voids)', 
                    fontsize=13, fontweight='bold')
        
        # Mark anomalies if detected
        if self.anomalies is not None:
            anomalous = self.anomalies['is_anomaly']
            if anomalous.any():
                anomaly_dates = anomalous[anomalous].index
                ax.scatter(anomaly_dates, 
                          self.features.loc[anomaly_dates, 'H1_max_persistence'],
                          color='red', s=100, marker='*', zorder=5, 
                          label='Anomaly', edgecolors='black', linewidth=1)
        
        ax.legend(loc='upper left', fontsize=9)
        ax2.legend(loc='upper right', fontsize=9)
        
        # Plot 3: Feature counts
        ax = axes[2]
        ax.plot(self.features.index, self.features['H0_count'],
               label='H₀ (Clusters)', linewidth=2, alpha=0.7)
        ax.plot(self.features.index, self.features['H1_count'],
               label='H₁ (Loops)', linewidth=2, alpha=0.7)
        ax.plot(self.features.index, self.features['H2_count'],
               label='H₂ (Voids)', linewidth=2, alpha=0.7)
        
        ax.set_ylabel('Feature Count', fontsize=11, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_title('Topological Feature Counts', fontsize=13, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Mark transitions
        if self.transitions is not None and len(self.transitions) > 0:
            for _, trans in self.transitions.iterrows():
                for ax in axes:
                    ax.axvline(trans['date'], color='red', linestyle='--', 
                             alpha=0.5, linewidth=1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Saved to {save_path}")
        
        plt.show()
    
    def generate_report(self) -> str:
        """
        Generate comprehensive regime analysis report
        
        Returns:
            Formatted report string
        """
        report = [
            "\n" + "="*70,
            "MARKET REGIME ANALYSIS REPORT",
            "="*70 + "\n"
        ]
        
        # Regime summary
        if self.regimes is not None:
            report.append("📊 IDENTIFIED REGIMES")
            report.append("-" * 70)
            for _, regime in self.regimes.iterrows():
                report.append(f"\n{regime['regime_name']}")
                report.append(f"   {regime['description']}")
                report.append(f"   Occurrences: {regime['count']} windows")
                report.append(f"   H₀: {regime.get('H0_count', 0):.2f} | "
                            f"H₁: {regime.get('H1_max_persistence', 0):.2f} | "
                            f"H₂: {regime.get('H2_max_persistence', 0):.2f}")
        
        # Current regime
        if self.regime_labels is not None:
            current_regime_id = self.regime_labels.iloc[-1]
            current_date = self.regime_labels.index[-1]
            
            if self.regimes is not None:
                current_name = self.regimes.loc[
                    self.regimes['regime_id'] == current_regime_id, 'regime_name'
                ].values[0]
            else:
                current_name = f"Regime {current_regime_id}"
            
            report.append(f"\n\n🎯 CURRENT REGIME (as of {current_date.date()})")
            report.append("-" * 70)
            report.append(f"   {current_name}")
        
        # Anomalies
        if self.anomalies is not None:
            n_anomalies = self.anomalies['is_anomaly'].sum()
            report.append(f"\n\n🚨 ANOMALIES DETECTED: {n_anomalies}")
            report.append("-" * 70)
            
            if n_anomalies > 0:
                recent_anomalies = self.anomalies[self.anomalies['is_anomaly']].tail(5)
                report.append("   Recent anomalous periods:")
                for date, row in recent_anomalies.iterrows():
                    score = row['max_zscore']
                    report.append(f"   • {date.date()}: z-score = {score:.2f}")
        
        # Transitions
        if self.transitions is not None and len(self.transitions) > 0:
            report.append(f"\n\n🔄 REGIME TRANSITIONS: {len(self.transitions)}")
            report.append("-" * 70)
            report.append("   Recent transitions:")
            for _, trans in self.transitions.tail(5).iterrows():
                report.append(f"   • {trans['date'].date()}: "
                            f"{trans['from_name']} → {trans['to_name']}")
        
        report.append("\n" + "="*70 + "\n")
        
        return "\n".join(report)


if __name__ == "__main__":
    # Example usage
    from data_fetcher import MarketDataFetcher
    from point_cloud import MarketPointCloud
    from persistent_homology import PersistentHomologyAnalyzer
    
    print("="*70)
    print("REGIME DETECTION DEMO")
    print("="*70)
    
    # Fetch and prepare data
    fetcher = MarketDataFetcher()
    data = fetcher.fetch_data(period="1y", max_tickers=40)
    returns = fetcher.compute_returns()
    
    # Create point clouds
    pc = MarketPointCloud(returns)
    windows = pc.create_sliding_windows(window_size=40, step_size=10)
    point_clouds = [w[0] for w in windows]
    dates = [w[1] for w in windows]
    
    # Compute persistent homology
    analyzer = PersistentHomologyAnalyzer(library="ripser")
    features = analyzer.analyze_windows(point_clouds, dates)
    
    # Detect regimes
    detector = MarketRegimeDetector(features)
    regimes = detector.detect_regimes(n_regimes=4)
    anomalies = detector.detect_anomalies()
    transitions = detector.detect_regime_transitions()
    
    # Generate report
    print(detector.generate_report())
    
    # Visualize
    detector.plot_regime_timeline()

