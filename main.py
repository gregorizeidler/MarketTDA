"""
MarketTDA - Main Analysis Pipeline
Topological Data Analysis of Market Regimes

This script demonstrates the complete workflow:
1. Fetch S&P 500 data from Yahoo Finance
2. Build high-dimensional point clouds
3. Compute persistent homology (H0, H1, H2)
4. Detect market regimes and anomalies
5. Visualize and interpret results
"""

import argparse
import warnings
warnings.filterwarnings('ignore')

from data_fetcher import MarketDataFetcher
from point_cloud import MarketPointCloud
from persistent_homology import PersistentHomologyAnalyzer
from visualizer import TopologyVisualizer
from regime_detector import MarketRegimeDetector


def print_header(text: str):
    """Print formatted section header"""
    print("\n" + "="*70)
    print(text.center(70))
    print("="*70 + "\n")


def main(
    period: str = "2y",
    max_tickers: int = 100,
    window_size: int = 60,
    step_size: int = 15,
    n_regimes: int = 4,
    library: str = "ripser",
    save_results: bool = False
):
    """
    Run complete MarketTDA analysis
    
    Args:
        period: Time period to analyze (1y, 2y, 5y, etc.)
        max_tickers: Maximum number of tickers to analyze
        window_size: Size of sliding window in days
        step_size: Step size for sliding window
        n_regimes: Number of market regimes to detect
        library: TDA library to use ('giotto' or 'ripser')
        save_results: Whether to save plots and data
    """
    
    print_header("🌀 MarketTDA: Topological Data Analysis of Market Regimes")
    
    print("Configuration:")
    print(f"   Period: {period}")
    print(f"   Max Tickers: {max_tickers}")
    print(f"   Window Size: {window_size} days")
    print(f"   Step Size: {step_size} days")
    print(f"   Number of Regimes: {n_regimes}")
    print(f"   TDA Library: {library}")
    
    # ========================================================================
    # STEP 1: FETCH DATA
    # ========================================================================
    print_header("STEP 1: Data Collection")
    
    fetcher = MarketDataFetcher(index="SP500")
    
    # Get S&P 500 tickers
    tickers = fetcher.get_sp500_tickers()
    
    # Fetch historical data
    data = fetcher.fetch_data(
        period=period,
        interval="1d",
        max_tickers=max_tickers
    )
    
    # Compute returns
    returns = fetcher.compute_returns(method="log")
    
    # Print summary
    summary = fetcher.get_data_summary()
    print("\n📊 Data Summary:")
    print(f"   Assets: {summary['n_assets']}")
    print(f"   Time Periods: {summary['n_periods']}")
    print(f"   Date Range: {summary['date_range'][0].date()} to {summary['date_range'][1].date()}")
    print(f"   Mean Return: {summary['mean_return']:.4%}")
    print(f"   Mean Volatility: {summary['mean_volatility']:.4%}")
    print(f"   Mean Correlation: {summary['correlation_mean']:.4f}")
    
    # ========================================================================
    # STEP 2: BUILD POINT CLOUDS
    # ========================================================================
    print_header("STEP 2: Point Cloud Construction")
    
    pc = MarketPointCloud(returns)
    
    # Create sliding windows
    windows = pc.create_sliding_windows(
        window_size=window_size,
        step_size=step_size,
        normalize=True
    )
    
    point_clouds = [w[0] for w in windows]
    dates = [w[1] for w in windows]
    
    # Analyze a sample window
    print("\n📊 Sample Window Statistics:")
    stats = pc.get_cloud_statistics(-1)
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # ========================================================================
    # STEP 3: PERSISTENT HOMOLOGY
    # ========================================================================
    print_header("STEP 3: Persistent Homology Analysis")
    
    analyzer = PersistentHomologyAnalyzer(library=library)
    
    # Compute persistent homology for all windows
    features = analyzer.analyze_windows(point_clouds, dates)
    
    # Show latest window interpretation
    print(analyzer.interpret_topology(-1))
    
    # Save features if requested
    if save_results:
        features_path = "topology_features.csv"
        features.to_csv(features_path)
        print(f"💾 Features saved to {features_path}")
    
    # ========================================================================
    # STEP 4: REGIME DETECTION
    # ========================================================================
    print_header("STEP 4: Market Regime Detection")
    
    detector = MarketRegimeDetector(features)
    
    # Detect regimes
    regimes = detector.detect_regimes(n_regimes=n_regimes)
    
    # Detect anomalies
    anomalies = detector.detect_anomalies(threshold=2.5)
    
    # Detect transitions
    transitions = detector.detect_regime_transitions()
    
    # Generate report
    report = detector.generate_report()
    print(report)
    
    if save_results:
        report_path = "regime_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"💾 Report saved to {report_path}")
    
    # ========================================================================
    # STEP 5: VISUALIZATION
    # ========================================================================
    print_header("STEP 5: Visualization")
    
    visualizer = TopologyVisualizer()
    
    print("📊 Generating visualizations...")
    
    # 1. Persistence diagram (latest window)
    print("\n1. Persistence Diagram...")
    visualizer.plot_persistence_diagram(
        analyzer.diagrams[-1],
        title=f"Persistence Diagram - {dates[-1].date()}",
        save_path="persistence_diagram.png" if save_results else None
    )
    
    # 2. Barcode (latest window)
    print("2. Persistence Barcode...")
    visualizer.plot_barcode(
        analyzer.diagrams[-1],
        title=f"Persistence Barcode - {dates[-1].date()}",
        save_path="persistence_barcode.png" if save_results else None
    )
    
    # 3. Topology evolution
    print("3. Topology Evolution...")
    visualizer.plot_topology_evolution(
        features,
        save_path="topology_evolution.png" if save_results else None
    )
    
    # 4. Regime timeline
    print("4. Regime Timeline...")
    detector.plot_regime_timeline(
        save_path="regime_timeline.png" if save_results else None
    )
    
    # 5. Betti curves
    print("5. Betti Curves...")
    visualizer.plot_betti_curves(
        analyzer.diagrams,
        dates,
        save_path="betti_curves.png" if save_results else None
    )
    
    # 6. Interactive dashboard
    print("6. Interactive Dashboard...")
    visualizer.create_interactive_dashboard(
        features,
        save_path="topology_dashboard.html" if save_results else None
    )
    
    # 7. 3D Point Cloud
    print("7. 3D Point Cloud Visualization...")
    pc.visualize_point_cloud_3d(
        window_idx=-1,
        save_path="point_cloud_3d.png" if save_results else None
    )
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print_header("✅ Analysis Complete")
    
    print("Summary of Results:")
    print(f"   • Analyzed {len(point_clouds)} time windows")
    print(f"   • Assets: {returns.shape[1]}")
    print(f"   • Dimensions: {returns.shape[1]}")
    print(f"   • Regimes detected: {n_regimes}")
    print(f"   • Anomalies found: {anomalies['is_anomaly'].sum()}")
    print(f"   • Regime transitions: {len(transitions)}")
    
    # Current state
    current_date = dates[-1]
    current_regime_id = regimes.iloc[-1]
    current_regime_name = detector.regimes.loc[
        detector.regimes['regime_id'] == current_regime_id, 'regime_name'
    ].values[0]
    
    print(f"\n🎯 Current Market State (as of {current_date.date()}):")
    print(f"   Regime: {current_regime_name}")
    
    latest_features = features.iloc[-1]
    print(f"   H₀ (Clusters): {latest_features['H0_count']:.0f}")
    print(f"   H₁ (Loops): {latest_features['H1_count']:.0f} "
          f"(max persistence: {latest_features['H1_max_persistence']:.4f})")
    print(f"   H₂ (Voids): {latest_features['H2_count']:.0f} "
          f"(max persistence: {latest_features['H2_max_persistence']:.4f})")
    
    # Check if current window is anomalous
    if anomalies.iloc[-1]['is_anomaly']:
        print("\n   ⚠️ WARNING: Current window is ANOMALOUS!")
        print(f"   Anomaly score: {anomalies.iloc[-1]['max_zscore']:.2f}")
    
    print("\n" + "="*70)
    print("🌀 MarketTDA analysis complete!")
    print("="*70 + "\n")
    
    return {
        'features': features,
        'regimes': regimes,
        'anomalies': anomalies,
        'transitions': transitions,
        'analyzer': analyzer,
        'detector': detector
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MarketTDA - Topological Data Analysis of Market Regimes"
    )
    
    parser.add_argument(
        '--period',
        type=str,
        default='2y',
        help='Time period to analyze (e.g., 1y, 2y, 5y, 10y)'
    )
    
    parser.add_argument(
        '--tickers',
        type=int,
        default=100,
        help='Maximum number of tickers to analyze'
    )
    
    parser.add_argument(
        '--window',
        type=int,
        default=60,
        help='Window size in days'
    )
    
    parser.add_argument(
        '--step',
        type=int,
        default=15,
        help='Step size for sliding window'
    )
    
    parser.add_argument(
        '--regimes',
        type=int,
        default=4,
        help='Number of market regimes to detect'
    )
    
    parser.add_argument(
        '--library',
        type=str,
        default='ripser',
        choices=['ripser', 'giotto'],
        help='TDA library to use'
    )
    
    parser.add_argument(
        '--save',
        action='store_true',
        help='Save results and plots'
    )
    
    args = parser.parse_args()
    
    # Run analysis
    results = main(
        period=args.period,
        max_tickers=args.tickers,
        window_size=args.window,
        step_size=args.step,
        n_regimes=args.regimes,
        library=args.library,
        save_results=args.save
    )

