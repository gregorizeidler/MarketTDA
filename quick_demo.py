"""
Quick Demo - MarketTDA
Run a fast demonstration with limited data
"""

from data_fetcher import MarketDataFetcher
from point_cloud import MarketPointCloud
from persistent_homology import PersistentHomologyAnalyzer
from visualizer import TopologyVisualizer
from regime_detector import MarketRegimeDetector

def quick_demo():
    """
    Run a quick demo with limited data for fast testing
    """
    print("\n" + "="*70)
    print("🌀 MarketTDA - Quick Demo".center(70))
    print("="*70 + "\n")
    
    print("📊 This demo uses a small dataset for fast testing")
    print("   For full analysis, run: python main.py\n")
    
    # Step 1: Fetch limited data
    print("Step 1: Fetching data (30 tickers, 6 months)...")
    fetcher = MarketDataFetcher()
    data = fetcher.fetch_data(period="6mo", max_tickers=30)
    returns = fetcher.compute_returns()
    
    # Step 2: Build point clouds
    print("\nStep 2: Building point clouds...")
    pc = MarketPointCloud(returns)
    windows = pc.create_sliding_windows(window_size=30, step_size=10)
    point_clouds = [w[0] for w in windows]
    dates = [w[1] for w in windows]
    
    # Step 3: Compute persistent homology
    print("\nStep 3: Computing persistent homology...")
    analyzer = PersistentHomologyAnalyzer(library="ripser")
    features = analyzer.analyze_windows(point_clouds, dates)
    
    # Step 4: Interpret latest window
    print(analyzer.interpret_topology(-1))
    
    # Step 5: Detect regimes
    print("Step 4: Detecting market regimes...")
    detector = MarketRegimeDetector(features)
    regimes = detector.detect_regimes(n_regimes=3)
    anomalies = detector.detect_anomalies()
    
    # Step 6: Visualize key results
    print("\nStep 5: Generating visualizations...\n")
    
    visualizer = TopologyVisualizer()
    
    # Show persistence diagram for latest window
    visualizer.plot_persistence_diagram(
        analyzer.diagrams[-1],
        title=f"Persistence Diagram - {dates[-1].date()}"
    )
    
    # Show topology evolution
    visualizer.plot_topology_evolution(features)
    
    # Show regime timeline
    detector.plot_regime_timeline()
    
    print("\n" + "="*70)
    print("✅ Quick demo complete!".center(70))
    print("For full analysis with more data, run: python main.py".center(70))
    print("="*70 + "\n")
    
    # Return results for further exploration
    return {
        'features': features,
        'regimes': regimes,
        'anomalies': anomalies,
        'analyzer': analyzer,
        'detector': detector,
        'visualizer': visualizer
    }


if __name__ == "__main__":
    results = quick_demo()
    
    # Optional: Print summary statistics
    print("\n📊 Summary Statistics:")
    print(f"   Windows analyzed: {len(results['features'])}")
    print(f"   Regimes found: {results['regimes'].nunique()}")
    print(f"   Anomalies detected: {results['anomalies']['is_anomaly'].sum()}")
    
    print("\n💡 Next steps:")
    print("   • Run 'python main.py --help' for full options")
    print("   • Increase dataset: python main.py --tickers 200 --period 2y")
    print("   • Save results: python main.py --save")
    print("   • Adjust parameters: python main.py --window 90 --step 20")

