"""
COMPREHENSIVE DEMO: Advanced Topological Market Analysis
Showcases all new advanced features
"""

import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import all modules
from data_fetcher import MarketDataFetcher
from point_cloud import MarketPointCloud
from persistent_homology import PersistentHomologyAnalyzer
from visualizer import TopologyVisualizer
from regime_detector import MarketRegimeDetector
from advanced_metrics import AdvancedTopologicalMetrics
from statistical_tests import TopologicalStatisticalTests
from financial_integration import FinancialTopologyIntegration
from network_visualizer import NetworkVisualizer

print("="*80)
print("🌀 ADVANCED TOPOLOGICAL MARKET ANALYSIS - COMPREHENSIVE DEMO 🌀")
print("="*80)
print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# ==================== PHASE 1: DATA COLLECTION ====================
print("\n" + "="*80)
print("PHASE 1: DATA COLLECTION & PREPROCESSING")
print("="*80)

fetcher = MarketDataFetcher()
data = fetcher.fetch_data(start_date="2024-01-01", period="1y", max_tickers=30)
returns = fetcher.compute_returns(method="log")

print(f"\n📊 Dataset Summary:")
print(f"   Tickers: {len(fetcher.tickers)}")
print(f"   Date Range: {returns.index.min()} to {returns.index.max()}")
print(f"   Trading Days: {len(returns)}")

# ==================== PHASE 2: TOPOLOGICAL ANALYSIS ====================
print("\n" + "="*80)
print("PHASE 2: PERSISTENT HOMOLOGY COMPUTATION")
print("="*80)

# Create point clouds
pc = MarketPointCloud(returns)
windows = pc.create_sliding_windows(window_size=30, step_size=10)
point_clouds = [w[0] for w in windows]
dates = [w[1] for w in windows]

print(f"\n🔬 Created {len(windows)} sliding windows")

# Compute persistent homology
analyzer = PersistentHomologyAnalyzer(library="ripser")
features = analyzer.analyze_windows(point_clouds, dates)
diagrams = analyzer.diagrams

print(f"\n✅ Computed persistence for {len(diagrams)} windows")
print(f"\nTopological Features Summary:")
print(features[['H0_count', 'H1_count', 'H2_count', 
              'H1_max_persistence', 'H2_max_persistence']].describe())

# ==================== PHASE 3: ADVANCED METRICS ====================
print("\n" + "="*80)
print("PHASE 3: ADVANCED TOPOLOGICAL METRICS")
print("="*80)

advanced_metrics = AdvancedTopologicalMetrics()

# 3.1 Wasserstein Distance Matrix
print("\n📏 Computing Wasserstein Distance Matrix...")
try:
    wasserstein_matrix = advanced_metrics.compute_distance_matrix(
        diagrams, metric='wasserstein', dim=1
    )
    print(f"✅ Distance matrix shape: {wasserstein_matrix.shape}")
    
    # Visualize
    advanced_metrics.plot_distance_matrix(
        wasserstein_matrix, dates, metric='Wasserstein', dim=1,
        save_path='results/wasserstein_matrix.png'
    )
except Exception as e:
    print(f"⚠️ Error: {e}")

# 3.2 Persistence Landscapes
print("\n🏔️ Computing Persistence Landscapes...")
try:
    latest_diagram = diagrams[-1]
    advanced_metrics.plot_persistence_landscapes(
        latest_diagram, dim=1, k_max=5,
        save_path='results/persistence_landscapes.png'
    )
except Exception as e:
    print(f"⚠️ Error: {e}")

# 3.3 Persistence Images
print("\n🖼️ Generating Persistence Images...")
try:
    # Select subset of diagrams
    indices = np.linspace(0, len(diagrams)-1, min(6, len(diagrams)), dtype=int)
    selected_diagrams = [diagrams[i] for i in indices]
    selected_dates = [dates[i] for i in indices]
    
    advanced_metrics.plot_persistence_images(
        selected_diagrams, selected_dates, dim=1, resolution=(25, 25),
        save_path='results/persistence_images.png'
    )
except Exception as e:
    print(f"⚠️ Error: {e}")

# ==================== PHASE 4: STATISTICAL TESTS ====================
print("\n" + "="*80)
print("PHASE 4: STATISTICAL HYPOTHESIS TESTING")
print("="*80)

stat_tester = TopologicalStatisticalTests(random_state=42)

# 4.1 Cross-Correlation Analysis
print("\n🔗 Cross-Correlation Analysis...")
try:
    ccf_results = stat_tester.cross_correlation_analysis(features, max_lag=5)
    stat_tester.plot_cross_correlations(
        ccf_results, max_lag=5,
        save_path='results/cross_correlations.png'
    )
except Exception as e:
    print(f"⚠️ Error: {e}")

# 4.2 Entropy Rate
print("\n📈 Persistent Entropy Rate...")
try:
    entropy_rates = stat_tester.persistent_entropy_rate(features, window_size=5)
    stat_tester.plot_entropy_rate(
        features, entropy_rates,
        save_path='results/entropy_rate.png'
    )
except Exception as e:
    print(f"⚠️ Error: {e}")

# 4.3 Topological Loss Function
print("\n⚠️ Topological Instability Loss...")
try:
    loss = stat_tester.topological_loss(features)
    print(f"   Mean Loss: {loss.mean():.3f}")
    print(f"   Max Loss: {loss.max():.3f} (Date: {loss.idxmax()})")
    print(f"   Min Loss: {loss.min():.3f} (Date: {loss.idxmin()})")
except Exception as e:
    print(f"⚠️ Error: {e}")

# ==================== PHASE 5: REGIME DETECTION ====================
print("\n" + "="*80)
print("PHASE 5: MARKET REGIME DETECTION")
print("="*80)

detector = MarketRegimeDetector(features)
regimes = detector.detect_regimes(n_regimes=3)
regime_info = detector.regimes

print(f"\n📊 Regime Distribution:")
print(regime_info['regime_name'].value_counts())

# Detect anomalies
anomalies = detector.detect_anomalies(threshold=2.5)
print(f"\n🚨 Detected {anomalies['is_anomaly'].sum()} anomalies")

# ==================== PHASE 6: FINANCIAL INTEGRATION ====================
print("\n" + "="*80)
print("PHASE 6: FINANCIAL METRICS INTEGRATION")
print("="*80)

fin_topo = FinancialTopologyIntegration()

# 6.1 Fetch VIX and correlate
print("\n📊 Fetching VIX data...")
try:
    vix_data = fin_topo.fetch_vix_data(
        start_date=returns.index.min().strftime('%Y-%m-%d'),
        end_date=returns.index.max().strftime('%Y-%m-%d')
    )
    
    if len(vix_data) > 0:
        vix_corr = fin_topo.correlate_topology_with_vix(features, vix_data)
        print(f"\n✅ Computed {len(vix_corr)} correlations with VIX")
        
        # Plot
        fin_topo.plot_topology_vix_correlation(
            features, vix_data, feature='H1_count',
            save_path='results/topology_vix_correlation.png'
        )
except Exception as e:
    print(f"⚠️ VIX analysis skipped: {e}")

# 6.2 Backtesting
print("\n💰 Backtesting Topological Trading Signals...")
try:
    # Create synthetic market returns (SPY proxy)
    market_returns = pd.Series(
        np.random.randn(len(features)) * 0.01,
        index=features.index,
        name='returns'
    )
    
    backtest_results = fin_topo.backtest_topology_signals(
        features, market_returns
    )
    
    fin_topo.plot_backtest_results(
        backtest_results,
        save_path='results/backtest_results.png'
    )
except Exception as e:
    print(f"⚠️ Error: {e}")

# 6.3 Topological Sharpe Ratio
print("\n📊 Topological Sharpe Ratio...")
try:
    topo_sharpe = fin_topo.topological_sharpe_ratio(features, window=10)
    print(f"   Mean Topo-Sharpe: {topo_sharpe.mean():.3f}")
    print(f"   Std Topo-Sharpe: {topo_sharpe.std():.3f}")
except Exception as e:
    print(f"⚠️ Error: {e}")

# ==================== PHASE 7: NETWORK VISUALIZATION ====================
print("\n" + "="*80)
print("PHASE 7: NETWORK ANALYSIS & VISUALIZATION")
print("="*80)

network_viz = NetworkVisualizer()

# 7.1 Minimum Spanning Tree
print("\n🌳 Computing Minimum Spanning Tree...")
try:
    # Compute correlation matrix from latest window
    latest_returns = returns.iloc[-30:]  # Last 30 days
    corr_matrix = latest_returns.corr().values
    tickers = list(latest_returns.columns)
    
    mst = network_viz.compute_mst(corr_matrix, tickers)
    network_viz.plot_mst(
        mst,
        save_path='results/mst_latest.png'
    )
except Exception as e:
    print(f"⚠️ Error: {e}")

# 7.2 Correlation Network
print("\n🕸️ Creating Correlation Network...")
try:
    network_viz.plot_correlation_network(
        corr_matrix, tickers, threshold=0.5,
        save_path='results/correlation_network.png'
    )
except Exception as e:
    print(f"⚠️ Error: {e}")

# 7.3 Sankey Diagram of Regime Transitions
print("\n🌊 Creating Regime Transition Sankey Diagram...")
try:
    regime_names = {
        0: "Stable",
        1: "Complex",
        2: "Crisis"
    }
    network_viz.plot_regime_transitions_sankey(
        regimes, regime_names=regime_names,
        save_path='results/regime_sankey.html'
    )
except Exception as e:
    print(f"⚠️ Error: {e}")

# 7.4 Hierarchical Clustering
print("\n🌲 Creating Hierarchical Clustering Dendrogram...")
try:
    network_viz.plot_hierarchical_clustering(
        corr_matrix, tickers,
        save_path='results/dendrogram.png'
    )
except Exception as e:
    print(f"⚠️ Error: {e}")

# ==================== PHASE 8: ADVANCED VISUALIZATIONS ====================
print("\n" + "="*80)
print("PHASE 8: ADVANCED 3D & MAPPER VISUALIZATIONS")
print("="*80)

visualizer = TopologyVisualizer()

# 8.1 3D Persistence Surface
print("\n🎨 Creating 3D Persistence Surface...")
try:
    visualizer.plot_3d_persistence_surface(
        diagrams[-1], dim=1,
        save_path='results/3d_persistence_surface.png'
    )
except Exception as e:
    print(f"⚠️ Error: {e}")

# 8.2 Mapper Graph
print("\n🗺️ Creating Mapper Graph...")
try:
    latest_cloud = point_clouds[-1]
    mapper_graph = visualizer.plot_mapper_graph(
        latest_cloud, n_intervals=8, overlap=0.4,
        save_path='results/mapper_graph.png'
    )
except Exception as e:
    print(f"⚠️ Error: {e}")

# ==================== PHASE 9: COMPREHENSIVE SUMMARY ====================
print("\n" + "="*80)
print("PHASE 9: COMPREHENSIVE ANALYSIS SUMMARY")
print("="*80)

print("\n📊 KEY FINDINGS:")
print("-" * 80)

# Topological summary
print("\n1️⃣ TOPOLOGICAL FEATURES:")
print(f"   • H₀ (Clusters): {features['H0_count'].mean():.1f} ± {features['H0_count'].std():.1f}")
print(f"   • H₁ (Loops): {features['H1_count'].mean():.1f} ± {features['H1_count'].std():.1f}")
print(f"   • H₂ (Voids): {features['H2_count'].mean():.2f} ± {features['H2_count'].std():.2f}")
print(f"   • Max H₁ Persistence: {features['H1_max_persistence'].max():.3f}")

# Regime summary
print("\n2️⃣ MARKET REGIMES:")
for regime_name, count in regime_info['regime_name'].value_counts().items():
    pct = count / len(regime_info) * 100
    print(f"   • {regime_name}: {count} windows ({pct:.1f}%)")

# Anomalies
print("\n3️⃣ ANOMALIES DETECTED:")
if anomalies['is_anomaly'].sum() > 0:
    anomaly_dates = anomalies[anomalies['is_anomaly']].index
    for date in anomaly_dates:
        h1_count = features.loc[date, 'H1_count']
        h2_count = features.loc[date, 'H2_count']
        print(f"   • {date.strftime('%Y-%m-%d')}: H₁={h1_count:.0f}, H₂={h2_count:.0f}")
else:
    print("   • No significant anomalies detected")

# Wasserstein insights
print("\n4️⃣ TOPOLOGICAL STABILITY:")
if 'wasserstein_matrix' in locals():
    mean_dist = wasserstein_matrix[wasserstein_matrix > 0].mean()
    max_dist = wasserstein_matrix.max()
    print(f"   • Mean Wasserstein Distance: {mean_dist:.3f}")
    print(f"   • Max Wasserstein Distance: {max_dist:.3f}")
    print(f"   • Interpretation: {'HIGH' if mean_dist > 0.5 else 'MODERATE' if mean_dist > 0.3 else 'LOW'} topology variance")

# Cross-correlation insights
print("\n5️⃣ CROSS-CORRELATIONS:")
if 'ccf_results' in locals():
    for pair, ccf in ccf_results.items():
        max_corr = ccf[np.abs(ccf).argmax()]
        lag = np.arange(-5, 6)[np.abs(ccf).argmax()]
        if abs(max_corr) > 0.3:
            print(f"   • {pair}: r={max_corr:.3f} at lag={lag:+d}")

print("\n" + "="*80)
print("✅ COMPREHENSIVE DEMO COMPLETED SUCCESSFULLY!")
print("="*80)
print(f"\nAll visualizations saved to: results/")
print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Generate summary report
summary_report = f"""
TOPOLOGICAL MARKET ANALYSIS - SUMMARY REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}

DATASET:
  - Tickers: {len(fetcher.tickers)}
  - Period: {returns.index.min()} to {returns.index.max()}
  - Trading Days: {len(returns)}
  - Windows: {len(windows)}

TOPOLOGICAL FEATURES:
  - H₀ (Clusters): {features['H0_count'].mean():.1f} ± {features['H0_count'].std():.1f}
  - H₁ (Loops): {features['H1_count'].mean():.1f} ± {features['H1_count'].std():.1f}
  - H₂ (Voids): {features['H2_count'].mean():.2f} ± {features['H2_count'].std():.2f}
  
MARKET REGIMES:
{regime_info['regime_name'].value_counts().to_string()}

ANOMALIES: {anomalies['is_anomaly'].sum()} detected

STATUS: Analysis completed successfully
{'='*80}
"""

# Save report
with open('results/analysis_summary.txt', 'w') as f:
    f.write(summary_report)

print("📄 Summary report saved to: results/analysis_summary.txt")
print("\n🎉 All done! Check the results/ folder for visualizations.\n")

