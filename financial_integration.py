"""
Financial Integration for Topological Analysis
VIX correlation, Backtesting, Risk metrics, Portfolio optimization
"""

import numpy as np
import pandas as pd
import yfinance as yf
from typing import List, Dict, Tuple, Optional
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class FinancialTopologyIntegration:
    """Integrate topological features with financial metrics and trading strategies"""
    
    def __init__(self):
        """Initialize financial integration module"""
        self.vix_data = None
        self.backtest_results = None
        
    # ==================== MARKET DATA INTEGRATION ====================
    
    def fetch_vix_data(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Fetch VIX (volatility index) data
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with VIX data
        """
        print(f"\n📊 Fetching VIX data ({start_date} to {end_date})...")
        
        try:
            vix = yf.download('^VIX', start=start_date, end=end_date, progress=False)
            self.vix_data = vix[['Close']].rename(columns={'Close': 'VIX'})
            print(f"✅ Fetched {len(self.vix_data)} days of VIX data")
            return self.vix_data
        except Exception as e:
            print(f"⚠️ Error fetching VIX: {e}")
            return pd.DataFrame()
    
    def fetch_volume_data(
        self,
        ticker: str = 'SPY',
        start_date: str = None,
        end_date: str = None
    ) -> pd.DataFrame:
        """Fetch volume data for market proxy (SPY)"""
        print(f"\n📈 Fetching volume data for {ticker}...")
        
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            volume_data = data[['Volume', 'Close']].copy()
            volume_data['Returns'] = volume_data['Close'].pct_change()
            volume_data['Volatility'] = volume_data['Returns'].rolling(20).std()
            print(f"✅ Fetched {len(volume_data)} days of volume data")
            return volume_data
        except Exception as e:
            print(f"⚠️ Error fetching volume: {e}")
            return pd.DataFrame()
    
    # ==================== CORRELATION ANALYSIS ====================
    
    def correlate_topology_with_vix(
        self,
        features: pd.DataFrame,
        vix_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Compute correlation between topological features and VIX
        
        Args:
            features: DataFrame with topological features
            vix_data: VIX data (if None, will fetch automatically)
            
        Returns:
            DataFrame with correlation results
        """
        print(f"\n🔗 Correlating Topology with VIX...")
        
        if vix_data is None:
            if self.vix_data is None:
                # Fetch VIX for the date range
                start = features.index.min().strftime('%Y-%m-%d')
                end = features.index.max().strftime('%Y-%m-%d')
                vix_data = self.fetch_vix_data(start, end)
            else:
                vix_data = self.vix_data
        
        if len(vix_data) == 0:
            print("⚠️ No VIX data available")
            return pd.DataFrame()
        
        # Merge datasets
        merged = features.join(vix_data, how='inner')
        
        if len(merged) < 10:
            print(f"⚠️ Insufficient overlapping data ({len(merged)} points)")
            return pd.DataFrame()
        
        # Compute correlations
        topology_cols = [col for col in features.columns 
                        if any(x in col for x in ['H0', 'H1', 'H2'])]
        
        results = []
        for col in topology_cols:
            if col in merged.columns and 'VIX' in merged.columns:
                # Pearson correlation
                corr, p_value = stats.pearsonr(
                    merged[col].dropna(),
                    merged['VIX'].dropna()
                )
                
                # Spearman correlation (rank-based)
                spearman, sp_pval = stats.spearmanr(
                    merged[col].dropna(),
                    merged['VIX'].dropna()
                )
                
                results.append({
                    'feature': col,
                    'pearson_corr': corr,
                    'pearson_pval': p_value,
                    'spearman_corr': spearman,
                    'spearman_pval': sp_pval,
                    'significant': p_value < 0.05
                })
        
        results_df = pd.DataFrame(results)
        
        # Print significant correlations
        print("\n   Significant correlations (p < 0.05):")
        sig = results_df[results_df['significant']].sort_values('pearson_corr', key=abs, ascending=False)
        for _, row in sig.iterrows():
            print(f"      {row['feature']:25s}: r={row['pearson_corr']:+.3f} (p={row['pearson_pval']:.4f})")
        
        return results_df
    
    def correlate_topology_with_market_metrics(
        self,
        features: pd.DataFrame,
        market_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Correlate topology with volume, volatility, returns
        
        Args:
            features: Topological features
            market_data: DataFrame with Volume, Volatility, Returns columns
            
        Returns:
            Correlation matrix
        """
        print(f"\n🔗 Correlating Topology with Market Metrics...")
        
        # Merge
        merged = features.join(market_data, how='inner')
        
        # Select relevant columns
        topo_cols = [c for c in features.columns if any(x in c for x in ['H0', 'H1', 'H2'])]
        market_cols = ['Volume', 'Volatility', 'Returns']
        market_cols = [c for c in market_cols if c in merged.columns]
        
        # Correlation matrix
        corr_matrix = merged[topo_cols + market_cols].corr()
        
        return corr_matrix
    
    # ==================== BACKTESTING ====================
    
    def backtest_topology_signals(
        self,
        features: pd.DataFrame,
        returns: pd.Series,
        signal_rules: Dict[str, Tuple[str, float, str]] = None
    ) -> Dict[str, any]:
        """
        Backtest trading strategy based on topological signals
        
        Args:
            features: Topological features
            returns: Market returns (e.g., SPY returns)
            signal_rules: Dictionary of rules:
                {
                    'rule_name': ('feature', threshold, 'above'/'below')
                }
                
        Returns:
            Dictionary with backtest results
        """
        print(f"\n💰 Backtesting Topological Trading Signals...")
        
        if signal_rules is None:
            # Default rules
            signal_rules = {
                'Crisis_H1': ('H1_count', features['H1_count'].mean() + 2*features['H1_count'].std(), 'above'),
                'Crisis_H2': ('H2_count', 0.5, 'above'),
                'Stable': ('H1_count', features['H1_count'].mean() - features['H1_count'].std(), 'below'),
            }
        
        print(f"   Testing {len(signal_rules)} signal rules")
        
        # Merge data
        merged = features.join(returns.to_frame('returns'), how='inner')
        
        results = {}
        
        for rule_name, (feature, threshold, direction) in signal_rules.items():
            if feature not in merged.columns:
                continue
            
            # Generate signals
            if direction == 'above':
                signal = (merged[feature] > threshold).astype(int)
            else:
                signal = (merged[feature] < threshold).astype(int)
            
            # Shift signal (trade next day)
            signal = signal.shift(1).fillna(0)
            
            # Compute strategy returns
            strategy_returns = signal * merged['returns']
            
            # Performance metrics
            cumulative_returns = (1 + strategy_returns).cumprod()
            total_return = cumulative_returns.iloc[-1] - 1
            
            # Sharpe ratio (annualized)
            sharpe = np.sqrt(252) * strategy_returns.mean() / (strategy_returns.std() + 1e-10)
            
            # Max drawdown
            cummax = cumulative_returns.cummax()
            drawdown = (cumulative_returns - cummax) / cummax
            max_drawdown = drawdown.min()
            
            # Win rate
            trades = strategy_returns[signal == 1]
            win_rate = (trades > 0).sum() / len(trades) if len(trades) > 0 else 0
            
            # Number of trades
            n_trades = signal.diff().abs().sum() / 2
            
            results[rule_name] = {
                'total_return': total_return,
                'sharpe_ratio': sharpe,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'n_trades': int(n_trades),
                'cumulative_returns': cumulative_returns,
                'strategy_returns': strategy_returns,
                'signal': signal
            }
            
            print(f"\n   {rule_name}:")
            print(f"      Total Return: {total_return*100:+.2f}%")
            print(f"      Sharpe Ratio: {sharpe:.3f}")
            print(f"      Max Drawdown: {max_drawdown*100:.2f}%")
            print(f"      Win Rate: {win_rate*100:.1f}%")
            print(f"      # Trades: {int(n_trades)}")
        
        # Buy-and-hold benchmark
        bh_returns = merged['returns']
        bh_cumulative = (1 + bh_returns).cumprod()
        bh_total_return = bh_cumulative.iloc[-1] - 1
        bh_sharpe = np.sqrt(252) * bh_returns.mean() / (bh_returns.std() + 1e-10)
        
        results['Buy_and_Hold'] = {
            'total_return': bh_total_return,
            'sharpe_ratio': bh_sharpe,
            'cumulative_returns': bh_cumulative
        }
        
        print(f"\n   Buy-and-Hold Benchmark:")
        print(f"      Total Return: {bh_total_return*100:+.2f}%")
        print(f"      Sharpe Ratio: {bh_sharpe:.3f}")
        
        self.backtest_results = results
        return results
    
    # ==================== RISK METRICS ====================
    
    def topological_sharpe_ratio(
        self,
        features: pd.DataFrame,
        window: int = 20
    ) -> pd.Series:
        """
        Compute rolling Sharpe-like metric for topological stability
        
        High topology Sharpe = stable features with low variance
        
        Args:
            features: Topological features
            window: Rolling window size
            
        Returns:
            Series with topological Sharpe ratios
        """
        print(f"\n📊 Computing Topological Sharpe Ratio (window={window})...")
        
        # Use H1_count as primary metric
        if 'H1_count' not in features.columns:
            print("⚠️ H1_count not found")
            return pd.Series()
        
        h1 = features['H1_count']
        
        # Rolling mean and std
        rolling_mean = h1.rolling(window).mean()
        rolling_std = h1.rolling(window).std()
        
        # Inverse Sharpe (lower H1 = better, lower variance = better)
        # Negative sign because we want low H1
        topo_sharpe = -rolling_mean / (rolling_std + 1e-10)
        
        return topo_sharpe
    
    def drawdown_by_regime(
        self,
        returns: pd.Series,
        regimes: pd.Series
    ) -> pd.DataFrame:
        """
        Compute drawdown statistics for each topological regime
        
        Args:
            returns: Market returns
            regimes: Regime labels
            
        Returns:
            DataFrame with drawdown stats by regime
        """
        print(f"\n📉 Computing Drawdown by Regime...")
        
        # Merge
        merged = pd.DataFrame({'returns': returns, 'regime': regimes}).dropna()
        
        # Cumulative returns
        merged['cumulative'] = (1 + merged['returns']).cumprod()
        
        results = []
        for regime in merged['regime'].unique():
            regime_data = merged[merged['regime'] == regime]
            
            # Max drawdown
            cummax = regime_data['cumulative'].cummax()
            drawdown = (regime_data['cumulative'] - cummax) / cummax
            max_dd = drawdown.min()
            
            # Average return
            avg_return = regime_data['returns'].mean()
            
            # Volatility
            volatility = regime_data['returns'].std()
            
            results.append({
                'regime': regime,
                'max_drawdown': max_dd,
                'avg_return': avg_return,
                'volatility': volatility,
                'sharpe': avg_return / (volatility + 1e-10) * np.sqrt(252),
                'n_periods': len(regime_data)
            })
        
        results_df = pd.DataFrame(results)
        
        print("\n   Regime Statistics:")
        for _, row in results_df.iterrows():
            print(f"      Regime {row['regime']}:")
            print(f"         Max DD: {row['max_drawdown']*100:.2f}%")
            print(f"         Avg Return: {row['avg_return']*100:.3f}%")
            print(f"         Volatility: {row['volatility']*100:.2f}%")
            print(f"         Sharpe: {row['sharpe']:.3f}")
        
        return results_df
    
    # ==================== VISUALIZATIONS ====================
    
    def plot_topology_vix_correlation(
        self,
        features: pd.DataFrame,
        vix_data: pd.DataFrame,
        feature: str = 'H1_count',
        save_path: Optional[str] = None
    ):
        """Plot topological feature vs VIX"""
        # Merge
        merged = features.join(vix_data, how='inner')
        
        if feature not in merged.columns or 'VIX' not in merged.columns:
            print("⚠️ Data not available for plotting")
            return
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        
        # Time series
        ax1 = axes[0]
        ax1_twin = ax1.twinx()
        
        ax1.plot(merged.index, merged[feature], 
                color='blue', linewidth=2, label=feature, marker='o', markersize=4)
        ax1_twin.plot(merged.index, merged['VIX'], 
                     color='red', linewidth=2, label='VIX', marker='s', markersize=4)
        
        ax1.set_ylabel(feature, fontsize=12, fontweight='bold', color='blue')
        ax1_twin.set_ylabel('VIX', fontsize=12, fontweight='bold', color='red')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1_twin.tick_params(axis='y', labelcolor='red')
        ax1.set_title('Topological Feature vs VIX Over Time', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Scatter plot
        ax2 = axes[1]
        ax2.scatter(merged[feature], merged['VIX'], alpha=0.6, s=100, edgecolors='black')
        
        # Fit line
        z = np.polyfit(merged[feature].dropna(), merged['VIX'].dropna(), 1)
        p = np.poly1d(z)
        x_line = np.linspace(merged[feature].min(), merged[feature].max(), 100)
        ax2.plot(x_line, p(x_line), 'r--', linewidth=2, label=f'Fit: y={z[0]:.2f}x+{z[1]:.2f}')
        
        # Correlation
        corr, pval = stats.pearsonr(merged[feature].dropna(), merged['VIX'].dropna())
        ax2.text(0.05, 0.95, f'r = {corr:.3f}\np = {pval:.4f}',
                transform=ax2.transAxes, fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax2.set_xlabel(feature, fontsize=12, fontweight='bold')
        ax2.set_ylabel('VIX', fontsize=12, fontweight='bold')
        ax2.set_title('Correlation Analysis', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Saved to {save_path}")
        
        plt.show()
    
    def plot_backtest_results(
        self,
        backtest_results: Dict[str, any] = None,
        save_path: Optional[str] = None
    ):
        """Plot backtest cumulative returns"""
        if backtest_results is None:
            backtest_results = self.backtest_results
        
        if backtest_results is None:
            print("⚠️ No backtest results available")
            return
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Cumulative returns
        ax1 = axes[0]
        for strategy, results in backtest_results.items():
            if 'cumulative_returns' in results:
                cum_ret = results['cumulative_returns']
                ax1.plot(cum_ret.index, (cum_ret - 1) * 100,
                        linewidth=2, label=strategy, marker='o', markersize=3)
        
        ax1.set_ylabel('Cumulative Return (%)', fontsize=12, fontweight='bold')
        ax1.set_title('Strategy Performance Comparison', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(0, color='black', linestyle='--', linewidth=1)
        
        # Performance metrics bar chart
        ax2 = axes[1]
        strategies = [k for k in backtest_results.keys() if k != 'Buy_and_Hold']
        sharpes = [backtest_results[k]['sharpe_ratio'] for k in strategies]
        returns = [backtest_results[k]['total_return'] * 100 for k in strategies]
        
        x = np.arange(len(strategies))
        width = 0.35
        
        ax2.bar(x - width/2, returns, width, label='Total Return (%)', alpha=0.7)
        ax2.bar(x + width/2, sharpes, width, label='Sharpe Ratio', alpha=0.7)
        
        ax2.set_xticks(x)
        ax2.set_xticklabels(strategies, rotation=45, ha='right')
        ax2.set_ylabel('Value', fontsize=12, fontweight='bold')
        ax2.set_title('Performance Metrics', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.axhline(0, color='black', linestyle='--', linewidth=1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Saved to {save_path}")
        
        plt.show()


if __name__ == "__main__":
    print("="*60)
    print("FINANCIAL INTEGRATION DEMO")
    print("="*60)
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    
    features = pd.DataFrame({
        'H0_count': 30 + np.random.randn(100) * 2,
        'H1_count': 10 + np.random.randn(100) * 3,
        'H2_count': np.random.poisson(0.5, 100),
    }, index=dates)
    
    returns = pd.Series(np.random.randn(100) * 0.01, index=dates, name='returns')
    
    fin_topo = FinancialTopologyIntegration()
    
    # Test backtesting
    backtest_results = fin_topo.backtest_topology_signals(features, returns)
    
    # Test topological Sharpe
    topo_sharpe = fin_topo.topological_sharpe_ratio(features, window=20)
    print(f"\n✅ Topological Sharpe computed (mean: {topo_sharpe.mean():.3f})")
    
    print("\n✅ Financial integration completed successfully!")

