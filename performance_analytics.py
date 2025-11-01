"""
Performance Analytics - Sharpe/Info/Calmar Ratios, Win Rate Analysis
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class PerformanceAnalytics:
    """Detailed performance analysis by regime and over time"""
    
    def __init__(self):
        """Initialize performance analytics"""
        self.colors = plt.cm.Set3(np.linspace(0, 1, 10))
    
    # ==================== SHARPE RATIO BY REGIME ====================
    
    def sharpe_ratio_by_regime(
        self,
        returns: pd.Series,
        regimes: pd.Series,
        risk_free_rate: float = 0.02
    ) -> pd.DataFrame:
        """
        Compute detailed Sharpe ratio for each regime
        
        Args:
            returns: Returns series
            regimes: Regime labels
            risk_free_rate: Annual risk-free rate
            
        Returns:
            DataFrame with Sharpe ratios and components
        """
        print("\n📊 Computing Sharpe Ratio by Regime...")
        
        daily_rf = (1 + risk_free_rate) ** (1/252) - 1
        
        results = []
        
        for regime in sorted(regimes.unique()):
            mask = regimes == regime
            regime_returns = returns[mask]
            
            if len(regime_returns) < 2:
                continue
            
            # Excess returns
            excess_returns = regime_returns - daily_rf
            
            # Sharpe ratio
            mean_excess = excess_returns.mean()
            std_excess = excess_returns.std()
            sharpe = (mean_excess / std_excess) * np.sqrt(252) if std_excess > 0 else 0
            
            # Annualized metrics
            annual_return = (1 + regime_returns.mean()) ** 252 - 1
            annual_vol = regime_returns.std() * np.sqrt(252)
            
            results.append({
                'Regime': regime,
                'Count': len(regime_returns),
                'Sharpe_Ratio': sharpe,
                'Annual_Return_%': annual_return * 100,
                'Annual_Vol_%': annual_vol * 100,
                'Mean_Daily_Return_%': regime_returns.mean() * 100,
                'Std_Daily_Return_%': regime_returns.std() * 100
            })
        
        df = pd.DataFrame(results)
        print(f"✅ Computed Sharpe for {len(df)} regimes")
        
        return df
    
    def plot_sharpe_by_regime(
        self,
        sharpe_df: pd.DataFrame,
        save_path: Optional[str] = None
    ):
        """Plot Sharpe ratio analysis by regime"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        regimes = sharpe_df['Regime']
        colors = plt.cm.Set3(np.linspace(0, 1, len(regimes)))
        
        # Sharpe Ratio
        ax1 = axes[0, 0]
        bars = ax1.bar(regimes, sharpe_df['Sharpe_Ratio'], color=colors, alpha=0.8)
        ax1.set_xlabel('Regime', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Sharpe Ratio', fontsize=12, fontweight='bold')
        ax1.set_title('Sharpe Ratio by Regime', fontsize=14, fontweight='bold')
        ax1.axhline(0, color='black', linestyle='--', linewidth=1)
        ax1.axhline(1, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Sharpe=1')
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.legend()
        
        # Return vs Volatility
        ax2 = axes[0, 1]
        scatter = ax2.scatter(
            sharpe_df['Annual_Vol_%'],
            sharpe_df['Annual_Return_%'],
            s=200,
            c=sharpe_df['Sharpe_Ratio'],
            cmap='RdYlGn',
            alpha=0.8,
            edgecolors='black',
            linewidths=2
        )
        
        for i, regime in enumerate(regimes):
            ax2.annotate(
                f'R{regime}',
                (sharpe_df.iloc[i]['Annual_Vol_%'], sharpe_df.iloc[i]['Annual_Return_%']),
                ha='center', va='center', fontweight='bold', fontsize=10
            )
        
        ax2.set_xlabel('Annual Volatility (%)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Annual Return (%)', fontsize=12, fontweight='bold')
        ax2.set_title('Risk-Return Profile by Regime', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax2, label='Sharpe Ratio')
        
        # Daily Return Distribution
        ax3 = axes[1, 0]
        x = np.arange(len(regimes))
        ax3.bar(x, sharpe_df['Mean_Daily_Return_%'], color=colors, alpha=0.8)
        ax3.set_xticks(x)
        ax3.set_xticklabels([f'Regime {r}' for r in regimes])
        ax3.set_ylabel('Mean Daily Return (%)', fontsize=12, fontweight='bold')
        ax3.set_title('Average Daily Returns', fontsize=14, fontweight='bold')
        ax3.axhline(0, color='black', linestyle='--', linewidth=1)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Volatility Comparison
        ax4 = axes[1, 1]
        ax4.bar(regimes, sharpe_df['Std_Daily_Return_%'], color=colors, alpha=0.8)
        ax4.set_xlabel('Regime', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Daily Volatility (%)', fontsize=12, fontweight='bold')
        ax4.set_title('Risk Profile by Regime', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(
            'Sharpe Ratio Analysis by Market Regime\n'
            'Risk-Adjusted Performance Metrics',
            fontsize=16, fontweight='bold', y=1.00
        )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Saved to {save_path}")
        
        plt.show()
    
    # ==================== INFORMATION RATIO ====================
    
    def information_ratio_by_regime(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series,
        regimes: pd.Series
    ) -> pd.DataFrame:
        """
        Compute Information Ratio (active return / tracking error) by regime
        
        Args:
            returns: Strategy returns
            benchmark_returns: Benchmark returns
            regimes: Regime labels
            
        Returns:
            DataFrame with Information Ratios
        """
        print("\n📊 Computing Information Ratio by Regime...")
        
        results = []
        
        for regime in sorted(regimes.unique()):
            mask = regimes == regime
            regime_returns = returns[mask]
            regime_benchmark = benchmark_returns[mask]
            
            if len(regime_returns) < 2:
                continue
            
            # Active returns
            active_returns = regime_returns - regime_benchmark
            
            # Information Ratio
            mean_active = active_returns.mean()
            std_active = active_returns.std()
            info_ratio = (mean_active / std_active) * np.sqrt(252) if std_active > 0 else 0
            
            # Annualized metrics
            annual_alpha = (1 + mean_active) ** 252 - 1
            tracking_error = std_active * np.sqrt(252)
            
            results.append({
                'Regime': regime,
                'Count': len(regime_returns),
                'Information_Ratio': info_ratio,
                'Annual_Alpha_%': annual_alpha * 100,
                'Tracking_Error_%': tracking_error * 100,
                'Correlation_with_Benchmark': regime_returns.corr(regime_benchmark)
            })
        
        df = pd.DataFrame(results)
        print(f"✅ Computed IR for {len(df)} regimes")
        
        return df
    
    # ==================== CALMAR RATIO ====================
    
    def calmar_ratio_by_regime(
        self,
        returns: pd.Series,
        regimes: pd.Series
    ) -> pd.DataFrame:
        """
        Compute Calmar Ratio (return / max drawdown) by regime
        
        Args:
            returns: Returns series
            regimes: Regime labels
            
        Returns:
            DataFrame with Calmar ratios
        """
        print("\n📊 Computing Calmar Ratio by Regime...")
        
        results = []
        
        for regime in sorted(regimes.unique()):
            mask = regimes == regime
            regime_returns = returns[mask]
            
            if len(regime_returns) < 2:
                continue
            
            # Cumulative returns
            cum_returns = (1 + regime_returns).cumprod()
            
            # Running maximum
            running_max = cum_returns.expanding().max()
            
            # Drawdown series
            drawdown = (cum_returns - running_max) / running_max
            max_drawdown = abs(drawdown.min())
            
            # Calmar ratio
            annual_return = (1 + regime_returns.mean()) ** 252 - 1
            calmar = annual_return / max_drawdown if max_drawdown > 0 else 0
            
            # Average drawdown
            avg_drawdown = abs(drawdown[drawdown < 0].mean()) if (drawdown < 0).any() else 0
            
            results.append({
                'Regime': regime,
                'Count': len(regime_returns),
                'Calmar_Ratio': calmar,
                'Annual_Return_%': annual_return * 100,
                'Max_Drawdown_%': max_drawdown * 100,
                'Avg_Drawdown_%': avg_drawdown * 100
            })
        
        df = pd.DataFrame(results)
        print(f"✅ Computed Calmar for {len(df)} regimes")
        
        return df
    
    def plot_calmar_analysis(
        self,
        calmar_df: pd.DataFrame,
        save_path: Optional[str] = None
    ):
        """Plot Calmar ratio analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        regimes = calmar_df['Regime']
        colors = plt.cm.Set3(np.linspace(0, 1, len(regimes)))
        
        # Calmar Ratio
        ax1 = axes[0, 0]
        ax1.bar(regimes, calmar_df['Calmar_Ratio'], color=colors, alpha=0.8)
        ax1.set_xlabel('Regime', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Calmar Ratio', fontsize=12, fontweight='bold')
        ax1.set_title('Calmar Ratio by Regime\n(Return / Max Drawdown)', fontsize=14, fontweight='bold')
        ax1.axhline(0, color='black', linestyle='--', linewidth=1)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Return vs Max Drawdown
        ax2 = axes[0, 1]
        x = np.arange(len(regimes))
        width = 0.35
        
        ax2.bar(x - width/2, calmar_df['Annual_Return_%'], width, label='Annual Return', alpha=0.8, color='green')
        ax2.bar(x + width/2, -calmar_df['Max_Drawdown_%'], width, label='Max Drawdown (neg)', alpha=0.8, color='red')
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'R{r}' for r in regimes])
        ax2.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
        ax2.set_title('Returns vs Drawdowns', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.axhline(0, color='black', linestyle='--', linewidth=1)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Max vs Avg Drawdown
        ax3 = axes[1, 0]
        ax3.bar(x - width/2, calmar_df['Max_Drawdown_%'], width, label='Max Drawdown', alpha=0.8)
        ax3.bar(x + width/2, calmar_df['Avg_Drawdown_%'], width, label='Avg Drawdown', alpha=0.8)
        ax3.set_xticks(x)
        ax3.set_xticklabels([f'R{r}' for r in regimes])
        ax3.set_ylabel('Drawdown (%)', fontsize=12, fontweight='bold')
        ax3.set_title('Drawdown Severity', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Efficiency: Calmar vs Sharpe proxy
        ax4 = axes[1, 1]
        efficiency = calmar_df['Annual_Return_%'] / (calmar_df['Max_Drawdown_%'] + 1)
        ax4.bar(regimes, efficiency, color=colors, alpha=0.8)
        ax4.set_xlabel('Regime', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Return Efficiency', fontsize=12, fontweight='bold')
        ax4.set_title('Risk-Adjusted Return Efficiency', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(
            'Calmar Ratio Analysis\n'
            'Drawdown-Adjusted Performance',
            fontsize=16, fontweight='bold', y=1.00
        )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Saved to {save_path}")
        
        plt.show()
    
    # ==================== WIN RATE TEMPORAL ANALYSIS ====================
    
    def win_rate_temporal(
        self,
        returns: pd.Series,
        window: int = 20
    ) -> pd.DataFrame:
        """
        Compute rolling win rate over time
        
        Args:
            returns: Returns series
            window: Rolling window size
            
        Returns:
            DataFrame with temporal win rate metrics
        """
        print(f"\n📊 Computing Temporal Win Rate (window={window})...")
        
        # Binary wins
        wins = (returns > 0).astype(int)
        
        # Rolling win rate
        win_rate = wins.rolling(window).mean() * 100
        
        # Rolling average gain/loss
        gains = returns[returns > 0].rolling(window).mean()
        losses = returns[returns < 0].rolling(window).mean()
        
        # Profit factor (sum gains / sum losses)
        rolling_gains_sum = returns[returns > 0].rolling(window).sum()
        rolling_losses_sum = abs(returns[returns < 0].rolling(window).sum())
        profit_factor = rolling_gains_sum / (rolling_losses_sum + 1e-10)
        
        results = pd.DataFrame({
            'WinRate_%': win_rate,
            'AvgGain_%': gains * 100,
            'AvgLoss_%': losses * 100,
            'ProfitFactor': profit_factor
        }, index=returns.index)
        
        print(f"✅ Computed temporal win rate")
        return results
    
    def plot_win_rate_temporal(
        self,
        win_rate_df: pd.DataFrame,
        regimes: Optional[pd.Series] = None,
        save_path: Optional[str] = None
    ):
        """Plot temporal win rate analysis"""
        fig, axes = plt.subplots(3, 1, figsize=(16, 12))
        
        dates = win_rate_df.index
        
        # Win Rate
        ax1 = axes[0]
        ax1.plot(dates, win_rate_df['WinRate_%'], linewidth=2, color='steelblue', label='Win Rate')
        ax1.axhline(50, color='red', linestyle='--', linewidth=1, label='50% (Random)')
        ax1.fill_between(dates, win_rate_df['WinRate_%'], 50, alpha=0.3, color='steelblue')
        
        if regimes is not None:
            # Color background by regime
            for regime in regimes.unique():
                mask = regimes == regime
                regime_dates = regimes[mask].index
                if len(regime_dates) > 0:
                    ax1.axvspan(regime_dates[0], regime_dates[-1], alpha=0.1, color=f'C{regime}')
        
        ax1.set_ylabel('Win Rate (%)', fontsize=12, fontweight='bold')
        ax1.set_title('Temporal Win Rate Evolution', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Avg Gain vs Avg Loss
        ax2 = axes[1]
        ax2.plot(dates, win_rate_df['AvgGain_%'], linewidth=2, color='green', label='Avg Gain', alpha=0.8)
        ax2.plot(dates, win_rate_df['AvgLoss_%'], linewidth=2, color='red', label='Avg Loss', alpha=0.8)
        ax2.axhline(0, color='black', linestyle='--', linewidth=1)
        ax2.fill_between(dates, win_rate_df['AvgGain_%'], 0, alpha=0.3, color='green')
        ax2.fill_between(dates, win_rate_df['AvgLoss_%'], 0, alpha=0.3, color='red')
        ax2.set_ylabel('Avg Return (%)', fontsize=12, fontweight='bold')
        ax2.set_title('Average Gain vs Loss Over Time', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Profit Factor
        ax3 = axes[2]
        ax3.plot(dates, win_rate_df['ProfitFactor'], linewidth=2, color='purple', label='Profit Factor')
        ax3.axhline(1, color='red', linestyle='--', linewidth=1, label='Break-even (PF=1)')
        ax3.axhline(2, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Good (PF=2)')
        ax3.fill_between(dates, win_rate_df['ProfitFactor'], 1, alpha=0.3, color='purple')
        ax3.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Profit Factor', fontsize=12, fontweight='bold')
        ax3.set_title('Profit Factor Evolution (Gains/Losses Ratio)', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Set ylim safely
        max_pf = win_rate_df['ProfitFactor'].replace([np.inf, -np.inf], np.nan).quantile(0.95)
        if not np.isnan(max_pf) and max_pf > 0:
            ax3.set_ylim([0, max_pf * 1.2])
        else:
            ax3.set_ylim([0, 3])
        
        plt.suptitle(
            'Temporal Win Rate Analysis\n'
            'Strategy Performance Evolution',
            fontsize=16, fontweight='bold', y=0.995
        )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Saved to {save_path}")
        
        plt.show()
    
    # ==================== COMPREHENSIVE PERFORMANCE SUMMARY ====================
    
    def comprehensive_performance_summary(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series,
        regimes: pd.Series,
        risk_free_rate: float = 0.02
    ) -> pd.DataFrame:
        """
        Generate comprehensive performance summary combining all metrics
        
        Args:
            returns: Strategy returns
            benchmark_returns: Benchmark returns
            regimes: Regime labels
            risk_free_rate: Annual risk-free rate
            
        Returns:
            Comprehensive performance DataFrame
        """
        print("\n📊 Computing Comprehensive Performance Summary...")
        
        sharpe_df = self.sharpe_ratio_by_regime(returns, regimes, risk_free_rate)
        info_df = self.information_ratio_by_regime(returns, benchmark_returns, regimes)
        calmar_df = self.calmar_ratio_by_regime(returns, regimes)
        
        # Merge all metrics
        summary = sharpe_df.merge(info_df, on='Regime', suffixes=('', '_info'))
        summary = summary.merge(calmar_df, on='Regime', suffixes=('', '_calmar'))
        
        # Clean up duplicate columns
        summary = summary.loc[:, ~summary.columns.str.endswith('_info')]
        summary = summary.loc[:, ~summary.columns.str.endswith('_calmar')]
        summary = summary.drop(columns=['Count_info', 'Count_calmar'], errors='ignore')
        
        print(f"✅ Generated summary for {len(summary)} regimes")
        return summary


if __name__ == "__main__":
    print("="*60)
    print("PERFORMANCE ANALYTICS DEMO")
    print("="*60)
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    
    returns = pd.Series(np.random.randn(100) * 0.01 + 0.0005, index=dates)
    benchmark = pd.Series(np.random.randn(100) * 0.008 + 0.0003, index=dates)
    regimes = pd.Series(np.random.choice([0, 1, 2], 100), index=dates)
    
    perf = PerformanceAnalytics()
    
    # Test Sharpe
    sharpe_df = perf.sharpe_ratio_by_regime(returns, regimes)
    print(f"\n✅ Sharpe computed (mean: {sharpe_df['Sharpe_Ratio'].mean():.2f})")
    
    # Test Calmar
    calmar_df = perf.calmar_ratio_by_regime(returns, regimes)
    print(f"✅ Calmar computed (mean: {calmar_df['Calmar_Ratio'].mean():.2f})")
    
    # Test Win Rate
    win_rate_df = perf.win_rate_temporal(returns, window=20)
    print(f"✅ Win rate computed (mean: {win_rate_df['WinRate_%'].mean():.1f}%)")
    
    print("\n✅ All performance analytics working!")

