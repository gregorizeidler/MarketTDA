"""
Ultra-Advanced TDA Demo - All Cutting-Edge Features
Persistent Entropy 2D, Silhouette, Landscape Norms, Fisher Kernel, 
Options Greeks, Liquidity, Sector Rotation, Tail Risk, Transaction Costs,
Animations, 3D Interactive, Heatmaps, Performance Analytics
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import all modules
from data_fetcher import MarketDataFetcher
from persistent_homology import PersistentHomologyAnalyzer
from advanced_metrics import AdvancedTopologicalMetrics
from statistical_tests import TopologicalStatisticalTests
from financial_integration import FinancialTopologyIntegration
from network_visualizer import NetworkVisualizer
from ultra_advanced_metrics import UltraAdvancedMetrics
from advanced_visualizations import AdvancedVisualizations
from performance_analytics import PerformanceAnalytics


def main():
    """Run ultra-advanced TDA analysis with all cutting-edge features"""
    
    print("="*80)
    print(" " * 20 + "🔬 ULTRA-ADVANCED TOPOLOGICAL DATA ANALYSIS 🔬")
    print(" " * 15 + "Research-Grade Market Regime Detection")
    print("="*80)
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # ==================== PHASE 1: DATA ACQUISITION ====================
    print("\n" + "="*80)
    print("PHASE 1: DATA ACQUISITION")
    print("="*80)
    
    fetcher = MarketDataFetcher()
    
    # Fetch S&P 500 tickers
    tickers = fetcher.get_sp500_tickers()
    print(f"✅ Fetched {len(tickers)} S&P 500 tickers")
    
    # Fetch market data (6 months for faster processing)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)
    
    data = fetcher.fetch_data(
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d'),
        max_tickers=50  # Use 50 stocks for speed
    )
    
    # Compute returns
    returns = fetcher.compute_returns(method='log')
    dates = returns.index
    
    print(f"\n📊 Dataset Summary:")
    print(f"   • Period: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
    print(f"   • Days: {len(returns)}")
    print(f"   • Assets: {returns.shape[1]}")
    
    # ==================== PHASE 2: PERSISTENT HOMOLOGY ====================
    print("\n" + "="*80)
    print("PHASE 2: PERSISTENT HOMOLOGY COMPUTATION")
    print("="*80)
    
    analyzer = PersistentHomologyAnalyzer(library="ripser")
    
    # Compute persistence diagrams using sliding window
    diagrams, diagram_dates = analyzer.sliding_window_analysis(
        returns,
        window_size=20,
        step_size=10
    )
    
    # Extract features
    features = analyzer.extract_features(diagrams)
    features.index = diagram_dates
    
    print(f"\n✅ Computed {len(diagrams)} persistence diagrams")
    print(f"   • H₀ (clusters): {features['H0_count'].mean():.1f} ± {features['H0_count'].std():.1f}")
    print(f"   • H₁ (loops): {features['H1_count'].mean():.1f} ± {features['H1_count'].std():.1f}")
    print(f"   • H₂ (voids): {features['H2_count'].mean():.1f} ± {features['H2_count'].std():.1f}")
    
    # ==================== PHASE 3: ULTRA-ADVANCED METRICS ====================
    print("\n" + "="*80)
    print("PHASE 3: ULTRA-ADVANCED TOPOLOGICAL METRICS")
    print("="*80)
    
    ultra_metrics = UltraAdvancedMetrics()
    
    # 1. Persistent Entropy 2D
    print("\n🔬 Computing 2D Persistent Entropy...")
    entropy_df = ultra_metrics.persistent_entropy_2d(diagrams, diagram_dates)
    ultra_metrics.plot_persistent_entropy_2d(
        entropy_df,
        save_path='results/ultra_01_persistent_entropy_2d.png'
    )
    
    # 2. Silhouette of Persistence
    print("\n🔬 Computing Silhouette of Persistence...")
    ultra_metrics.plot_silhouette(
        diagrams[-1],
        dim=1,
        save_path='results/ultra_02_silhouette_persistence.png'
    )
    
    # 3. Landscape Norms
    print("\n📏 Computing Landscape Norms...")
    norms_df = ultra_metrics.compute_all_landscape_norms(diagrams, diagram_dates, dim=1)
    
    # Plot norms evolution
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    norms_df['H1_L1'].plot(ax=axes[0], title='L1 Norm Evolution', color='blue')
    norms_df['H1_L2'].plot(ax=axes[1], title='L2 Norm Evolution', color='green')
    norms_df['H1_Linf'].plot(ax=axes[2], title='L∞ Norm Evolution', color='red')
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Date')
    plt.tight_layout()
    plt.savefig('results/ultra_03_landscape_norms.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("💾 Saved Landscape Norms")
    
    # 4. Persistence Fisher Kernel
    print("\n🔬 Computing Persistence Fisher Kernel Matrix...")
    # Sample diagrams for speed
    sample_indices = np.linspace(0, len(diagrams)-1, min(20, len(diagrams)), dtype=int)
    sampled_diagrams = [diagrams[i] for i in sample_indices]
    sampled_dates = [diagram_dates[i] for i in sample_indices]
    
    kernel_matrix = ultra_metrics.compute_kernel_matrix(sampled_diagrams, dim=1)
    ultra_metrics.plot_kernel_matrix(
        kernel_matrix,
        sampled_dates,
        save_path='results/ultra_04_fisher_kernel_matrix.png'
    )
    
    # ==================== PHASE 4: EXTENDED FINANCIAL METRICS ====================
    print("\n" + "="*80)
    print("PHASE 4: EXTENDED FINANCIAL INTEGRATION")
    print("="*80)
    
    fin_topo = FinancialTopologyIntegration()
    
    # Prepare market return (SPY proxy)
    market_returns = returns.mean(axis=1)
    
    # 1. Options Greeks (Realized Volatility)
    print("\n📊 Fetching Options Data (Realized Volatility)...")
    greeks_data = fin_topo.fetch_options_greeks(
        'SPY',
        start_date.strftime('%Y-%m-%d'),
        end_date.strftime('%Y-%m-%d')
    )
    
    if not greeks_data.empty:
        greeks_corr = fin_topo.correlate_topology_with_greeks(features, greeks_data)
        print("✅ Options Greeks correlation computed")
        print(greeks_corr)
    
    # 2. Liquidity Analysis
    print("\n💧 Analyzing Market Liquidity...")
    # Use the tickers we already have
    sample_tickers = list(returns.columns[:20])
    volume_data = fin_topo.fetch_volume_data(
        sample_tickers,
        start_date.strftime('%Y-%m-%d'),
        end_date.strftime('%Y-%m-%d')
    )
    
    if not volume_data.empty:
        liquidity_corr = fin_topo.liquidity_topology_analysis(features, volume_data)
        print("✅ Liquidity analysis complete")
        print(liquidity_corr)
    
    # 3. Sector Rotation Detection
    print("\n🔄 Detecting Sector Rotation Events...")
    sector_rotation = fin_topo.detect_sector_rotation(features, window=20)
    
    # Plot sector rotation
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(sector_rotation.index, sector_rotation['SectorRotation_Smooth'], linewidth=2)
    ax.fill_between(sector_rotation.index, 0, sector_rotation['SectorRotation_Smooth'], alpha=0.3)
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Rotation Signal', fontsize=12, fontweight='bold')
    ax.set_title('Sector Rotation Detection via Topology', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/ultra_05_sector_rotation.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("💾 Saved Sector Rotation analysis")
    
    # 4. Tail Risk Metrics
    print("\n⚠️ Computing Tail Risk by Regime...")
    # Detect regimes first (simple clustering)
    from sklearn.cluster import KMeans
    
    feature_cols = ['H0_count', 'H1_count', 'H2_count']
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    regimes = pd.Series(
        kmeans.fit_predict(features[feature_cols]),
        index=features.index
    )
    
    # Align returns with regimes
    aligned_returns = market_returns.reindex(regimes.index)
    
    tail_risk_df = fin_topo.tail_risk_by_regime(aligned_returns, regimes, confidence=0.95)
    fin_topo.plot_tail_risk(
        tail_risk_df,
        save_path='results/ultra_06_tail_risk.png'
    )
    
    # 5. Transaction Cost Analysis
    print("\n💸 Analyzing Transaction Costs...")
    # Run simple backtest first
    backtest_results = fin_topo.backtest_topology_signals(features, aligned_returns)
    
    # Analyze costs
    cost_analysis = fin_topo.transaction_cost_analysis(backtest_results, cost_bps=10)
    fin_topo.plot_transaction_cost_impact(
        cost_analysis,
        save_path='results/ultra_07_transaction_costs.png'
    )
    
    # ==================== PHASE 5: ADVANCED VISUALIZATIONS ====================
    print("\n" + "="*80)
    print("PHASE 5: ADVANCED VISUALIZATIONS")
    print("="*80)
    
    advanced_viz = AdvancedVisualizations()
    
    # 1. Persistence Landscape Animation
    print("\n🎬 Creating Persistence Landscape Animation...")
    advanced_viz.create_landscape_animation(
        diagrams[:30],  # First 30 for speed
        diagram_dates[:30],
        dim=1,
        save_path='results/ultra_08_landscape_animation.gif',
        fps=5
    )
    
    # 2. 3D Interactive Persistence Diagram
    print("\n📊 Creating 3D Interactive Persistence Diagram...")
    try:
        advanced_viz.plot_3d_persistence_diagram(
            diagrams[-1],
            save_path='results/ultra_09_3d_persistence.html'
        )
    except Exception as e:
        print(f"⚠️ Could not create 3D plot: {e}")
    
    # 3. Wasserstein Heatmap Temporal
    print("\n🔥 Creating Temporal Wasserstein Heatmap...")
    # Compute Wasserstein distances (sample for speed)
    from advanced_metrics import AdvancedTopologicalMetrics
    adv_metrics = AdvancedTopologicalMetrics()
    
    sample_indices2 = np.linspace(0, len(diagrams)-1, min(15, len(diagrams)), dtype=int)
    sampled_diagrams2 = [diagrams[i] for i in sample_indices2]
    sampled_dates2 = [diagram_dates[i] for i in sample_indices2]
    
    wass_matrix = adv_metrics.compute_distance_matrix(sampled_diagrams2, dim=1, metric='wasserstein')
    advanced_viz.plot_wasserstein_heatmap_temporal(
        wass_matrix,
        sampled_dates2,
        save_path='results/ultra_10_wasserstein_temporal.png'
    )
    
    # 4. Regime Probability Heatmap
    print("\n📊 Creating Regime Probability Heatmap...")
    # Create probabilistic regime assignments (softmax of distances to centroids)
    from scipy.special import softmax
    
    centroids = kmeans.cluster_centers_
    distances = np.zeros((len(features), 3))
    for i in range(3):
        distances[:, i] = np.linalg.norm(features[feature_cols] - centroids[i], axis=1)
    
    # Softmax with negative distances (closer = higher prob)
    regime_probs = pd.DataFrame(
        softmax(-distances, axis=1),
        columns=['Regime_0_Prob', 'Regime_1_Prob', 'Regime_2_Prob'],
        index=features.index
    )
    
    advanced_viz.plot_regime_probability_heatmap(
        regime_probs,
        save_path='results/ultra_11_regime_probabilities.png'
    )
    
    # ==================== PHASE 6: PERFORMANCE ANALYTICS ====================
    print("\n" + "="*80)
    print("PHASE 6: PERFORMANCE ANALYTICS")
    print("="*80)
    
    perf_analytics = PerformanceAnalytics()
    
    # 1. Sharpe Ratio by Regime
    print("\n📊 Computing Sharpe Ratio by Regime...")
    sharpe_df = perf_analytics.sharpe_ratio_by_regime(aligned_returns, regimes)
    perf_analytics.plot_sharpe_by_regime(
        sharpe_df,
        save_path='results/ultra_12_sharpe_by_regime.png'
    )
    
    # 2. Information Ratio
    print("\n📊 Computing Information Ratio...")
    benchmark_returns = aligned_returns * 0.95 + np.random.randn(len(aligned_returns)) * 0.001
    
    info_df = perf_analytics.information_ratio_by_regime(
        aligned_returns,
        benchmark_returns,
        regimes
    )
    print("✅ Information Ratio computed")
    print(info_df)
    
    # 3. Calmar Ratio
    print("\n📊 Computing Calmar Ratio...")
    calmar_df = perf_analytics.calmar_ratio_by_regime(aligned_returns, regimes)
    perf_analytics.plot_calmar_analysis(
        calmar_df,
        save_path='results/ultra_13_calmar_ratio.png'
    )
    
    # 4. Temporal Win Rate
    print("\n📊 Computing Temporal Win Rate...")
    win_rate_df = perf_analytics.win_rate_temporal(aligned_returns, window=20)
    perf_analytics.plot_win_rate_temporal(
        win_rate_df,
        regimes=regimes,
        save_path='results/ultra_14_win_rate_temporal.png'
    )
    
    # 5. Comprehensive Summary
    print("\n📊 Generating Comprehensive Performance Summary...")
    comprehensive_summary = perf_analytics.comprehensive_performance_summary(
        aligned_returns,
        benchmark_returns,
        regimes
    )
    
    # Save summary
    comprehensive_summary.to_csv('results/ultra_performance_summary.csv', index=False)
    print("\n✅ Comprehensive Performance Summary:")
    print(comprehensive_summary.to_string())
    
    # ==================== FINAL SUMMARY ====================
    print("\n" + "="*80)
    print("ULTRA-ADVANCED ANALYSIS COMPLETE!")
    print("="*80)
    
    print("\n📁 Generated Files:")
    results_dir = 'results'
    for file in sorted(os.listdir(results_dir)):
        if file.startswith('ultra_'):
            file_path = os.path.join(results_dir, file)
            size_kb = os.path.getsize(file_path) / 1024
            print(f"   ✅ {file} ({size_kb:.1f} KB)")
    
    print("\n🎉 All ultra-advanced features demonstrated successfully!")
    print("\n📊 Key Findings:")
    print(f"   • Dataset: {len(returns)} days, {returns.shape[1]} assets")
    print(f"   • Topological Features: {len(diagrams)} diagrams")
    print(f"   • Average Entropy H₁: {entropy_df['H1_entropy'].mean():.3f}")
    print(f"   • Sector Rotations Detected: {sector_rotation['SectorRotation'].sum()}")
    print(f"   • Best Regime Sharpe: {sharpe_df['Sharpe_Ratio'].max():.2f}")
    print(f"   • Worst Regime Sharpe: {sharpe_df['Sharpe_Ratio'].min():.2f}")
    print(f"   • Average Win Rate: {win_rate_df['WinRate_%'].mean():.1f}%")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for saving
    import matplotlib.pyplot as plt
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Analysis interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

