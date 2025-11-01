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
        self.options_data = None
        self.volume_data = None
        self.sector_data = None
        
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
    
    # ==================== OPTIONS GREEKS CORRELATION ====================
    
    def fetch_options_greeks(
        self,
        ticker: str = 'SPY',
        start_date: str = None,
        end_date: str = None
    ) -> pd.DataFrame:
        """
        Fetch implied volatility as proxy for options Greeks
        (Full Greeks require Bloomberg/paid API)
        
        Args:
            ticker: Ticker symbol
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with IV data
        """
        print(f"\n📊 Fetching Options IV for {ticker}...")
        
        try:
            # Fetch historical volatility as proxy
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            # Compute realized volatility (21-day rolling)
            returns = np.log(data['Close'] / data['Close'].shift(1))
            realized_vol = returns.rolling(21).std() * np.sqrt(252) * 100
            
            self.options_data = pd.DataFrame({
                'RealizedVol': realized_vol
            })
            
            print(f"✅ Fetched {len(self.options_data)} days of volatility data")
            return self.options_data
            
        except Exception as e:
            print(f"⚠️ Error fetching options data: {e}")
            return pd.DataFrame()
    
    def correlate_topology_with_greeks(
        self,
        features: pd.DataFrame,
        greeks_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Correlate topological features with options Greeks
        
        Args:
            features: Topological features DataFrame
            greeks_data: Options Greeks DataFrame
            
        Returns:
            Correlation DataFrame
        """
        print("\n🔗 Correlating Topology with Options Greeks...")
        
        # Merge on date
        merged = features.join(greeks_data, how='inner')
        
        # Compute correlations
        topo_cols = [c for c in merged.columns if c.startswith('H')]
        greeks_cols = [c for c in merged.columns if c not in topo_cols]
        
        corr_matrix = merged[topo_cols + greeks_cols].corr().loc[topo_cols, greeks_cols]
        
        print(f"✅ Computed correlations for {len(topo_cols)} features")
        return corr_matrix
    
    # ==================== LIQUIDITY ANALYSIS ====================
    
    def fetch_volume_data(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Fetch volume data for liquidity analysis"""
        print(f"\n📊 Fetching Volume data for {len(tickers)} tickers...")
        
        try:
            volumes = []
            for ticker in tickers[:20]:  # Limit to 20 for speed
                data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                if not data.empty:
                    volumes.append(data['Volume'])
            
            self.volume_data = pd.concat(volumes, axis=1)
            self.volume_data.columns = tickers[:len(self.volume_data.columns)]
            
            print(f"✅ Fetched volume for {len(self.volume_data.columns)} tickers")
            return self.volume_data
            
        except Exception as e:
            print(f"⚠️ Error fetching volume: {e}")
            return pd.DataFrame()
    
    def liquidity_topology_analysis(
        self,
        features: pd.DataFrame,
        volume_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Analyze relationship between market liquidity and topology
        
        Args:
            features: Topological features
            volume_data: Volume data
            
        Returns:
            Analysis results
        """
        print("\n💧 Analyzing Liquidity vs Topology...")
        
        # Compute aggregate liquidity metrics
        avg_volume = volume_data.mean(axis=1)
        volume_std = volume_data.std(axis=1)
        
        # Merge with topology
        merged = features.copy()
        merged['AvgVolume'] = avg_volume
        merged['VolumeStd'] = volume_std
        merged['LiquidityRatio'] = avg_volume / (volume_std + 1)
        
        # Compute correlations
        liquidity_cols = ['AvgVolume', 'VolumeStd', 'LiquidityRatio']
        topo_cols = [c for c in features.columns if c.startswith('H')]
        
        correlations = merged[topo_cols + liquidity_cols].corr().loc[topo_cols, liquidity_cols]
        
        print(f"✅ Liquidity analysis complete")
        return correlations
    
    # ==================== SECTOR ROTATION ====================
    
    def detect_sector_rotation(
        self,
        features: pd.DataFrame,
        window: int = 20
    ) -> pd.DataFrame:
        """
        Detect sector rotation events using topological changes
        
        A sector rotation is indicated by:
        - Sharp increase in H1 (new cycles forming)
        - Moderate increase in H0 (fragmentation)
        - Stable or decreasing H2
        
        Args:
            features: Topological features
            window: Rolling window
            
        Returns:
            Sector rotation signals
        """
        print(f"\n🔄 Detecting Sector Rotation (window={window})...")
        
        results = features.copy()
        
        # Compute changes
        for col in ['H0_count', 'H1_count', 'H2_count']:
            if col in results.columns:
                results[f'{col}_change'] = results[col].diff()
                results[f'{col}_pct_change'] = results[col].pct_change()
        
        # Rotation signal: H1 up, H0 moderate up, H2 stable/down
        rotation_signal = (
            (results['H1_count_pct_change'] > 0.1) &  # H1 increasing
            (results['H0_count_pct_change'] > 0) &     # H0 increasing
            (results['H2_count_pct_change'] <= 0.05)   # H2 stable
        ).astype(int)
        
        results['SectorRotation'] = rotation_signal
        
        # Smooth signal
        results['SectorRotation_Smooth'] = results['SectorRotation'].rolling(5).mean()
        
        rotation_count = results['SectorRotation'].sum()
        print(f"✅ Detected {rotation_count} rotation events ({rotation_count/len(results)*100:.1f}%)")
        
        return results
    
    # ==================== TAIL RISK METRICS ====================
    
    def tail_risk_by_regime(
        self,
        returns: pd.Series,
        regimes: pd.Series,
        confidence: float = 0.95
    ) -> pd.DataFrame:
        """
        Compute tail risk metrics (CVaR, Expected Shortfall) by regime
        
        Args:
            returns: Returns series
            regimes: Regime labels
            confidence: Confidence level for VaR
            
        Returns:
            DataFrame with tail risk metrics
        """
        print(f"\n⚠️ Computing Tail Risk Metrics (confidence={confidence})...")
        
        results = []
        
        for regime in sorted(regimes.unique()):
            mask = regimes == regime
            regime_returns = returns[mask]
            
            if len(regime_returns) < 10:
                continue
            
            # Value at Risk (VaR)
            var = np.percentile(regime_returns, (1 - confidence) * 100)
            
            # Conditional Value at Risk (CVaR / Expected Shortfall)
            cvar = regime_returns[regime_returns <= var].mean()
            
            # Maximum drawdown
            cum_returns = (1 + regime_returns).cumprod()
            running_max = cum_returns.expanding().max()
            drawdown = (cum_returns - running_max) / running_max
            max_drawdown = drawdown.min()
            
            # Skewness and Kurtosis
            skew = stats.skew(regime_returns)
            kurt = stats.kurtosis(regime_returns)
            
            results.append({
                'Regime': regime,
                'Count': len(regime_returns),
                f'VaR_{int(confidence*100)}%': var * 100,
                f'CVaR_{int(confidence*100)}%': cvar * 100,
                'MaxDrawdown_%': max_drawdown * 100,
                'Skewness': skew,
                'Kurtosis': kurt
            })
        
        df = pd.DataFrame(results)
        print(f"✅ Computed tail risk for {len(df)} regimes")
        
        return df
    
    def plot_tail_risk(
        self,
        tail_risk_df: pd.DataFrame,
        save_path: Optional[str] = None
    ):
        """Plot tail risk metrics by regime"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        regimes = tail_risk_df['Regime']
        colors = plt.cm.Set3(np.linspace(0, 1, len(regimes)))
        
        # VaR and CVaR
        ax1 = axes[0, 0]
        var_col = [c for c in tail_risk_df.columns if 'VaR_' in c and 'CVaR' not in c][0]
        cvar_col = [c for c in tail_risk_df.columns if 'CVaR_' in c][0]
        
        x = np.arange(len(regimes))
        width = 0.35
        
        ax1.bar(x - width/2, tail_risk_df[var_col], width, label='VaR', alpha=0.8)
        ax1.bar(x + width/2, tail_risk_df[cvar_col], width, label='CVaR', alpha=0.8)
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'Regime {r}' for r in regimes])
        ax1.set_ylabel('Loss (%)', fontsize=12, fontweight='bold')
        ax1.set_title('Value at Risk by Regime', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Max Drawdown
        ax2 = axes[0, 1]
        ax2.bar(regimes, tail_risk_df['MaxDrawdown_%'], color=colors, alpha=0.8)
        ax2.set_xlabel('Regime', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Max Drawdown (%)', fontsize=12, fontweight='bold')
        ax2.set_title('Maximum Drawdown by Regime', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Skewness
        ax3 = axes[1, 0]
        ax3.bar(regimes, tail_risk_df['Skewness'], color=colors, alpha=0.8)
        ax3.set_xlabel('Regime', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Skewness', fontsize=12, fontweight='bold')
        ax3.set_title('Return Distribution Skewness', fontsize=14, fontweight='bold')
        ax3.axhline(0, color='black', linestyle='--', linewidth=1)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Kurtosis
        ax4 = axes[1, 1]
        ax4.bar(regimes, tail_risk_df['Kurtosis'], color=colors, alpha=0.8)
        ax4.set_xlabel('Regime', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Excess Kurtosis', fontsize=12, fontweight='bold')
        ax4.set_title('Return Distribution Kurtosis (Fat Tails)', fontsize=14, fontweight='bold')
        ax4.axhline(0, color='black', linestyle='--', linewidth=1)
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(
            'Tail Risk Analysis by Market Regime\n'
            'CVaR, Drawdowns, and Distribution Properties',
            fontsize=16, fontweight='bold', y=1.00
        )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Saved to {save_path}")
        
        plt.show()
    
    # ==================== TRANSACTION COST ANALYSIS ====================
    
    def transaction_cost_analysis(
        self,
        backtest_results: Dict,
        cost_bps: int = 10
    ) -> Dict:
        """
        Analyze impact of transaction costs on topological strategies
        
        Args:
            backtest_results: Results from backtesting
            cost_bps: Transaction cost in basis points (default 10 bps)
            
        Returns:
            Adjusted results with costs
        """
        print(f"\n💸 Analyzing Transaction Costs ({cost_bps} bps)...")
        
        adjusted_results = {}
        
        for strategy_name, metrics in backtest_results.items():
            if 'trades' not in metrics:
                continue
            
            trades = metrics['trades']
            returns = metrics.get('returns', pd.Series())
            
            # Count trades
            num_trades = trades.sum()
            
            # Transaction cost
            total_cost_pct = (num_trades * cost_bps) / 10000  # Convert bps to percentage
            
            # Adjust returns
            adjusted_total_return = metrics['total_return'] - total_cost_pct
            
            # Adjust Sharpe (approximate)
            adjusted_sharpe = metrics['sharpe_ratio'] * (1 - total_cost_pct / 100)
            
            adjusted_results[strategy_name] = {
                'original_return': metrics['total_return'],
                'adjusted_return': adjusted_total_return,
                'transaction_cost': total_cost_pct,
                'num_trades': num_trades,
                'original_sharpe': metrics['sharpe_ratio'],
                'adjusted_sharpe': adjusted_sharpe,
                'cost_impact_pct': (total_cost_pct / metrics['total_return'] * 100) if metrics['total_return'] != 0 else 0
            }
        
        print(f"✅ Transaction cost analysis complete for {len(adjusted_results)} strategies")
        return adjusted_results
    
    def plot_transaction_cost_impact(
        self,
        cost_analysis: Dict,
        save_path: Optional[str] = None
    ):
        """Plot impact of transaction costs"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        strategies = list(cost_analysis.keys())
        original_returns = [cost_analysis[s]['original_return'] for s in strategies]
        adjusted_returns = [cost_analysis[s]['adjusted_return'] for s in strategies]
        cost_impacts = [cost_analysis[s]['cost_impact_pct'] for s in strategies]
        
        # Returns comparison
        ax1 = axes[0]
        x = np.arange(len(strategies))
        width = 0.35
        
        ax1.bar(x - width/2, original_returns, width, label='Pre-Cost', alpha=0.8, color='green')
        ax1.bar(x + width/2, adjusted_returns, width, label='Post-Cost', alpha=0.8, color='orange')
        ax1.set_xticks(x)
        ax1.set_xticklabels(strategies, rotation=45, ha='right')
        ax1.set_ylabel('Total Return (%)', fontsize=12, fontweight='bold')
        ax1.set_title('Returns: Pre vs Post Transaction Costs', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.axhline(0, color='black', linestyle='--', linewidth=1)
        
        # Cost impact
        ax2 = axes[1]
        colors = ['red' if ci > 50 else 'orange' if ci > 25 else 'green' for ci in cost_impacts]
        ax2.bar(strategies, cost_impacts, color=colors, alpha=0.8)
        ax2.set_ylabel('Cost Impact (%)', fontsize=12, fontweight='bold')
        ax2.set_title('Transaction Costs as % of Returns', fontsize=14, fontweight='bold')
        ax2.set_xticklabels(strategies, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.axhline(50, color='red', linestyle='--', linewidth=1, label='High Impact (50%)')
        ax2.legend()
        
        plt.suptitle(
            'Transaction Cost Analysis\n'
            'Impact on Strategy Performance',
            fontsize=16, fontweight='bold', y=1.00
        )
        
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

