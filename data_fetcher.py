"""
Data Fetcher for S&P 500 constituents from Yahoo Finance
Handles data collection, cleaning, and preparation for TDA analysis
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple, Optional
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


class MarketDataFetcher:
    """Fetch and prepare market data for topological analysis"""
    
    def __init__(self, index: str = "SP500"):
        """
        Initialize data fetcher
        
        Args:
            index: Market index to analyze (SP500, NASDAQ100, etc.)
        """
        self.index = index
        self.tickers = []
        self.data = None
        self.returns = None
        
    def get_sp500_tickers(self) -> List[str]:
        """
        Scrape S&P 500 constituent tickers from Wikipedia
        
        Returns:
            List of ticker symbols
        """
        print("📊 Fetching S&P 500 constituents...")
        
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        try:
            response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Try multiple methods to find the table
            table = soup.find('table', {'id': 'constituents'})
            if table is None:
                # Try finding by class
                table = soup.find('table', {'class': 'wikitable'})
            if table is None:
                # Try finding first table
                table = soup.find('table')
            
            if table is None:
                raise ValueError("Could not find S&P 500 table on Wikipedia")
            
            tickers = []
            rows = table.find_all('tr')[1:]  # Skip header
            
            for row in rows:
                cells = row.find_all('td')
                if len(cells) > 0:
                    ticker = cells[0].text.strip()
                    # Clean ticker symbols (remove newlines, fix special characters)
                    ticker = ticker.replace('\n', '').replace('.', '-')
                    if ticker:  # Only add non-empty tickers
                        tickers.append(ticker)
            
            if len(tickers) == 0:
                raise ValueError("No tickers found in table")
            
            print(f"✅ Found {len(tickers)} S&P 500 tickers")
            self.tickers = tickers
            return tickers
            
        except Exception as e:
            print(f"⚠️ Error fetching S&P 500 tickers: {e}")
            print("📌 Using fallback: top 100 tickers")
            # Fallback to a subset if scraping fails
            self.tickers = self._get_fallback_tickers()
            return self.tickers
    
    def _get_fallback_tickers(self) -> List[str]:
        """Fallback list of major S&P 500 tickers"""
        return [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 
            'UNH', 'XOM', 'JNJ', 'JPM', 'V', 'PG', 'MA', 'HD', 'CVX', 'MRK',
            'ABBV', 'PEP', 'COST', 'AVGO', 'KO', 'ADBE', 'WMT', 'MCD', 'CSCO',
            'CRM', 'ACN', 'TMO', 'ABT', 'LIN', 'NFLX', 'NKE', 'DHR', 'VZ',
            'CMCSA', 'TXN', 'ORCL', 'NEE', 'PM', 'INTC', 'UPS', 'RTX', 'HON',
            'COP', 'QCOM', 'INTU', 'AMGN', 'LOW', 'AMD', 'BA', 'SPGI', 'UNP',
            'SBUX', 'ELV', 'GE', 'PFE', 'DE', 'AMAT', 'LMT', 'GILD', 'CAT',
            'MDT', 'BLK', 'ADI', 'ADP', 'C', 'BKNG', 'CI', 'SYK', 'MMC',
            'MDLZ', 'VRTX', 'CVS', 'TJX', 'ZTS', 'REGN', 'MO', 'SO', 'PLD',
            'ISRG', 'CB', 'DUK', 'BDX', 'NOC', 'SCHW', 'ITW', 'EOG', 'BMY',
            'USB', 'HCA', 'GD', 'TGT', 'MS', 'PNC', 'MMM', 'SLB', 'LRCX'
        ]
    
    def fetch_data(
        self, 
        start_date: str = None,
        end_date: str = None,
        period: str = "2y",
        interval: str = "1d",
        max_tickers: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Fetch historical data for all tickers
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            period: Period to fetch if dates not specified (1y, 2y, 5y, etc.)
            interval: Data interval (1d, 1h, 5m, etc.)
            max_tickers: Limit number of tickers (for testing)
            
        Returns:
            DataFrame with adjusted close prices
        """
        if not self.tickers:
            self.get_sp500_tickers()
        
        tickers_to_fetch = self.tickers[:max_tickers] if max_tickers else self.tickers
        
        print(f"\n📥 Fetching {len(tickers_to_fetch)} tickers from Yahoo Finance...")
        print(f"   Period: {period if not start_date else f'{start_date} to {end_date}'}")
        print(f"   Interval: {interval}")
        
        # Fetch data
        all_data = []
        failed_tickers = []
        
        for ticker in tqdm(tickers_to_fetch, desc="Downloading"):
            try:
                if start_date and end_date:
                    df = yf.download(ticker, start=start_date, end=end_date, 
                                   interval=interval, progress=False)
                else:
                    df = yf.download(ticker, period=period, interval=interval, 
                                   progress=False)
                
                if not df.empty:
                    # Use Adjusted Close if available, else Close
                    price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
                    df = df[[price_col]].rename(columns={price_col: ticker})
                    all_data.append(df)
                else:
                    failed_tickers.append(ticker)
                    
            except Exception as e:
                failed_tickers.append(ticker)
                continue
        
        if failed_tickers:
            print(f"⚠️ Failed to fetch {len(failed_tickers)} tickers: {failed_tickers[:5]}...")
        
        # Merge all data
        if all_data:
            self.data = pd.concat(all_data, axis=1)
            
            # Forward fill missing values (up to 5 days)
            self.data = self.data.ffill(limit=5)
            
            # Drop any remaining NaN columns or rows
            self.data = self.data.dropna(axis=1, how='any')
            
            print(f"✅ Successfully fetched {self.data.shape[1]} assets with {self.data.shape[0]} time periods")
            return self.data
        else:
            raise ValueError("Failed to fetch any data")
    
    def compute_returns(self, method: str = "log") -> pd.DataFrame:
        """
        Compute returns from price data
        
        Args:
            method: 'log' for log returns, 'simple' for simple returns
            
        Returns:
            DataFrame of returns
        """
        if self.data is None:
            raise ValueError("No data available. Call fetch_data() first.")
        
        print(f"\n📈 Computing {method} returns...")
        
        if method == "log":
            self.returns = np.log(self.data / self.data.shift(1))
        elif method == "simple":
            self.returns = self.data.pct_change()
        else:
            raise ValueError("method must be 'log' or 'simple'")
        
        # Drop first row (NaN from differencing)
        self.returns = self.returns.dropna()
        
        # Remove any extreme outliers (> 5 standard deviations)
        # This prevents data errors from corrupting the topology
        for col in self.returns.columns:
            mean = self.returns[col].mean()
            std = self.returns[col].std()
            self.returns[col] = self.returns[col].clip(
                lower=mean - 5*std, 
                upper=mean + 5*std
            )
        
        print(f"✅ Returns computed: {self.returns.shape}")
        print(f"   Date range: {self.returns.index[0]} to {self.returns.index[-1]}")
        print(f"   Mean daily return: {self.returns.mean().mean():.4%}")
        print(f"   Mean daily volatility: {self.returns.std().mean():.4%}")
        
        return self.returns
    
    def get_data_summary(self) -> dict:
        """Get summary statistics of the data"""
        if self.returns is None:
            return {}
        
        return {
            'n_assets': self.returns.shape[1],
            'n_periods': self.returns.shape[0],
            'date_range': (self.returns.index[0], self.returns.index[-1]),
            'mean_return': self.returns.mean().mean(),
            'mean_volatility': self.returns.std().mean(),
            'correlation_mean': self.returns.corr().values[np.triu_indices_from(
                self.returns.corr().values, k=1)].mean(),
        }


if __name__ == "__main__":
    # Example usage
    fetcher = MarketDataFetcher()
    
    # Fetch data for testing (limited to 50 tickers)
    data = fetcher.fetch_data(period="1y", max_tickers=50)
    returns = fetcher.compute_returns(method="log")
    
    print("\n" + "="*60)
    print("DATA SUMMARY")
    print("="*60)
    summary = fetcher.get_data_summary()
    for key, value in summary.items():
        print(f"{key}: {value}")

