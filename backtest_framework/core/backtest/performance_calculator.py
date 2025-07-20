"""
Performance metrics calculation module for backtesting results.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any


class PerformanceCalculator:
    """
    Calculates various performance metrics for backtest results.
    """
    
    def __init__(self, initial_capital: float):
        """
        Initialize performance calculator.
        
        Args:
            initial_capital: Initial capital for the backtest
        """
        self.initial_capital = initial_capital
    
    def calculate_returns_and_drawdown(self, results: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate returns and drawdown series.
        
        Args:
            results: DataFrame with equity curve
            
        Returns:
            DataFrame with added return and drawdown columns
        """
        results = results.copy()
        
        # Calculate returns
        results['returns'] = results['equity'] / self.initial_capital - 1
        results['daily_return'] = results['equity'].pct_change()
        
        # Calculate drawdown
        results['peak_equity'] = results['equity'].cummax()
        results['drawdown'] = (results['equity'] - results['peak_equity']) / results['peak_equity']
        
        return results
    
    def calculate_performance_metrics(self, results: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            results: DataFrame with backtest results
            
        Returns:
            DataFrame with added performance metrics
        """
        results = results.copy()
        
        # Basic return metrics
        total_return = results['returns'].iloc[-1]
        
        # Handle insufficient data
        if len(results) <= 1:
            self._add_nan_metrics(results)
            return results
        
        # Time-based calculations
        days = (results.index[-1] - results.index[0]).days
        years = max(days / 365.25, 0.01)
        
        # CAGR
        results['cagr'] = ((1 + total_return) ** (1 / years)) - 1
        
        # Risk metrics
        daily_returns = results['daily_return'].dropna()
        
        if len(daily_returns) > 0:
            self._calculate_risk_metrics(results, daily_returns)
        else:
            self._add_nan_risk_metrics(results)
        
        # Drawdown metrics
        results['max_drawdown'] = results['drawdown'].min()
        
        # Trading metrics
        self._calculate_trading_metrics(results)
        
        return results
    
    def _calculate_risk_metrics(self, results: pd.DataFrame, daily_returns: pd.Series):
        """Calculate risk-based performance metrics."""
        # Annualized volatility
        annual_vol = daily_returns.std() * np.sqrt(252)
        
        # Sharpe Ratio
        if annual_vol > 0:
            results['sharpe_ratio'] = results['cagr'] / annual_vol
        else:
            results['sharpe_ratio'] = np.nan
        
        # Sortino Ratio
        downside_returns = daily_returns[daily_returns < 0]
        if len(downside_returns) > 0:
            downside_vol = downside_returns.std() * np.sqrt(252)
            if downside_vol > 0:
                results['sortino_ratio'] = results['cagr'] / downside_vol
            else:
                results['sortino_ratio'] = np.nan
        else:
            results['sortino_ratio'] = np.nan
    
    def _calculate_trading_metrics(self, results: pd.DataFrame):
        """Calculate trading-specific metrics."""
        if 'trade_count' in results.columns and 'win_count' in results.columns:
            total_trades = results['trade_count'].iloc[-1]
            if total_trades > 0:
                results['win_rate'] = results['win_count'].iloc[-1] / total_trades
            else:
                results['win_rate'] = np.nan
        else:
            results['win_rate'] = np.nan
    
    def _add_nan_metrics(self, results: pd.DataFrame):
        """Add NaN values for all metrics when insufficient data."""
        metrics = ['sharpe_ratio', 'sortino_ratio', 'max_drawdown', 'win_rate', 'cagr']
        for metric in metrics:
            results[metric] = np.nan
    
    def _add_nan_risk_metrics(self, results: pd.DataFrame):
        """Add NaN values for risk metrics."""
        results['sharpe_ratio'] = np.nan
        results['sortino_ratio'] = np.nan
    
    def calculate_benchmark_performance(self, results: pd.DataFrame, 
                                     include_dividends: bool = True) -> pd.DataFrame:
        """
        Calculate buy & hold benchmark performance.
        
        Args:
            results: DataFrame with backtest results
            include_dividends: Whether to include dividend reinvestment
            
        Returns:
            DataFrame with benchmark columns added
        """
        results = results.copy()
        
        # Find first signal date
        first_signal_idx = self._find_first_signal_index(results)
        
        # Get price series from first signal onwards
        price_series = results['Close'].iloc[first_signal_idx:]
        initial_price = price_series.iloc[0]
        
        # Calculate benchmark with or without dividends
        if include_dividends and 'Dividends' in results.columns:
            benchmark_equity = self._calculate_dividend_reinvestment_benchmark(
                results, price_series, initial_price, first_signal_idx)
        else:
            benchmark_equity = price_series / initial_price * self.initial_capital
        
        # Add benchmark columns
        self._add_benchmark_columns(results, benchmark_equity)
        
        return results
    
    def _find_first_signal_index(self, results: pd.DataFrame) -> int:
        """Find the index of the first signal (buy or sell) - matches capital allocation logic."""
        first_signal_idx = 0
        first_signal_date = None
        
        # Check for buy signals
        if 'buy_signal' in results.columns:
            buy_signals = results[results['buy_signal'] == 1]
            if not buy_signals.empty:
                first_signal_date = buy_signals.index[0]
        
        # Check for sell signals
        if 'sell_signal' in results.columns:
            sell_signals = results[results['sell_signal'] == 1]
            if not sell_signals.empty:
                first_sell_date = sell_signals.index[0]
                if first_signal_date is None or first_sell_date < first_signal_date:
                    first_signal_date = first_sell_date
        
        # Convert date to index if found
        if first_signal_date is not None:
            first_signal_idx = results.index.get_indexer(
                [first_signal_date], method='nearest')[0]
        
        return first_signal_idx
    
    def _calculate_dividend_reinvestment_benchmark(self, results: pd.DataFrame,
                                                 price_series: pd.Series,
                                                 initial_price: float,
                                                 first_signal_idx: int) -> pd.Series:
        """Calculate benchmark with dividend reinvestment."""
        dividend_series = results['Dividends'].iloc[first_signal_idx:]
        benchmark_shares = self.initial_capital / initial_price
        benchmark_values = []
        
        for i, (date, price) in enumerate(price_series.items()):
            if i > 0 and dividend_series.iloc[i] > 0:
                # Reinvest dividends
                dividend_payment = benchmark_shares * dividend_series.iloc[i]
                additional_shares = dividend_payment / price
                benchmark_shares += additional_shares
            
            portfolio_value = benchmark_shares * price
            benchmark_values.append(portfolio_value)
        
        return pd.Series(benchmark_values, index=price_series.index)
    
    def _add_benchmark_columns(self, results: pd.DataFrame, benchmark_equity: pd.Series):
        """Add benchmark performance columns to results."""
        # Initialize benchmark columns
        results['benchmark_equity'] = self.initial_capital
        results['benchmark_returns'] = 0.0
        results['benchmark_drawdown'] = 0.0
        
        # Fill benchmark data
        results.loc[benchmark_equity.index, 'benchmark_equity'] = benchmark_equity
        results.loc[benchmark_equity.index, 'benchmark_returns'] = \
            benchmark_equity / self.initial_capital - 1
        
        # Calculate benchmark drawdown
        benchmark_peak = results['benchmark_equity'].cummax()
        results['benchmark_drawdown'] = \
            (results['benchmark_equity'] - benchmark_peak) / benchmark_peak
        
        # Calculate benchmark metrics
        self._calculate_benchmark_metrics(results)
    
    def _calculate_benchmark_metrics(self, results: pd.DataFrame):
        """Calculate benchmark performance metrics."""
        final_benchmark_return = results['benchmark_returns'].iloc[-1]
        
        # CAGR
        days = (results.index[-1] - results.index[0]).days
        years = max(days / 365.25, 0.01)
        results['benchmark_cagr'] = ((1 + final_benchmark_return) ** (1 / years)) - 1
        
        # Max drawdown
        results['benchmark_max_drawdown'] = results['benchmark_drawdown'].min()
        
        # Volatility and Sharpe ratio
        benchmark_daily_returns = results['benchmark_equity'].pct_change().dropna()
        if len(benchmark_daily_returns) > 0:
            benchmark_annual_vol = benchmark_daily_returns.std() * np.sqrt(252)
            if benchmark_annual_vol > 0:
                results['benchmark_sharpe_ratio'] = \
                    results['benchmark_cagr'] / benchmark_annual_vol
            else:
                results['benchmark_sharpe_ratio'] = np.nan
        else:
            results['benchmark_sharpe_ratio'] = np.nan
    
    def get_summary_stats(self, results: pd.DataFrame) -> Dict[str, Any]:
        """
        Get summary statistics from backtest results.
        
        Args:
            results: DataFrame with backtest results
            
        Returns:
            Dictionary with key performance metrics
        """
        if len(results) == 0:
            return {}
        
        summary = {
            'total_return': results['returns'].iloc[-1],
            'cagr': results['cagr'].iloc[-1] if 'cagr' in results.columns else np.nan,
            'max_drawdown': results['max_drawdown'].iloc[-1] if 'max_drawdown' in results.columns else np.nan,
            'sharpe_ratio': results['sharpe_ratio'].iloc[-1] if 'sharpe_ratio' in results.columns else np.nan,
            'sortino_ratio': results['sortino_ratio'].iloc[-1] if 'sortino_ratio' in results.columns else np.nan,
            'win_rate': results['win_rate'].iloc[-1] if 'win_rate' in results.columns else np.nan,
            'total_trades': results['trade_count'].iloc[-1] if 'trade_count' in results.columns else 0,
            'final_equity': results['equity'].iloc[-1],
        }
        
        # Add benchmark comparison if available
        if 'benchmark_returns' in results.columns:
            summary['benchmark_return'] = results['benchmark_returns'].iloc[-1]
            summary['excess_return'] = summary['total_return'] - summary['benchmark_return']
        
        return summary
