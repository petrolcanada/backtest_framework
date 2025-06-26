"""
Utility functions for creating strategy vs benchmark charts.
"""
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from backtest_framework.core.visualization.plotter import Plotter

def plot_strategy_vs_benchmark(data: pd.DataFrame, 
                              ticker: str = '', 
                              save_path: str = None) -> go.Figure:
    """
    Create a dark-themed chart comparing strategy performance against buy-and-hold benchmark.
    
    Args:
        data: DataFrame with OHLC price data, indicators, signals, and equity curve
        ticker: Ticker symbol for chart title
        save_path: If provided, save chart to this path as HTML file
        
    Returns:
        Plotly Figure object
    """
    # Create plotter and chart
    plotter = Plotter(data, data)
    fig = plotter.plot_chart_with_benchmark(ticker=ticker)
    
    # Save chart if path is provided
    if save_path:
        # Create directory if it doesn't exist
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            
        # Save chart
        fig.write_html(save_path)
        print(f"Chart saved to: {save_path}")
    
    return fig

def calculate_cum_returns(df: pd.DataFrame, buy_col: str = 'BUY_SIGNAL', exit_col: str = 'BUY_EXIT_SIGNAL', 
                          initial_capital: float = 10000) -> pd.DataFrame:
    """
    Calculate equity curve and cumulative returns for a strategy.
    
    Args:
        df: DataFrame with price data and buy/exit signals
        buy_col: Column name for buy signals
        exit_col: Column name for exit signals
        initial_capital: Initial investment amount
        
    Returns:
        DataFrame with added equity and position columns
    """
    # Create copy to avoid modifying original
    data = df.copy()
    
    # Initialize new columns
    data['equity'] = initial_capital
    data['position'] = 0
    
    # Implement strategy
    for i in range(1, len(data)):
        # Carry forward previous position and equity
        data.loc[data.index[i], 'position'] = data['position'].iloc[i-1]
        
        # If buy signal, go long
        if data[buy_col].iloc[i] == 1:
            # Invest all capital at current price
            shares = data['equity'].iloc[i-1] / data['Close'].iloc[i]
            data.loc[data.index[i], 'position'] = shares
        
        # If exit signal, close position
        elif data[exit_col].iloc[i] == 1:
            # Sell all shares at current price
            data.loc[data.index[i], 'position'] = 0
        
        # Calculate equity
        if data['position'].iloc[i] > 0:
            # If we have a position, equity is position * current price
            data.loc[data.index[i], 'equity'] = data['position'].iloc[i] * data['Close'].iloc[i]
        else:
            # If no position, equity stays the same
            data.loc[data.index[i], 'equity'] = data['equity'].iloc[i-1]
    
    # Calculate additional performance metrics
    data['returns'] = data['equity'] / initial_capital - 1
    
    # Calculate drawdown - the percentage decline from the running max
    data['peak_equity'] = data['equity'].cummax()
    data['drawdown'] = (data['equity'] / data['peak_equity']) - 1
    
    # Calculate buy-and-hold benchmark
    first_close = data['Close'].iloc[0]
    data['benchmark_equity'] = initial_capital * (data['Close'] / first_close)
    data['benchmark_returns'] = data['benchmark_equity'] / initial_capital - 1
    
    # Calculate alpha (outperformance over benchmark)
    data['alpha'] = data['returns'] - data['benchmark_returns']
    
    return data

def calculate_performance_metrics(data: pd.DataFrame, ann_factor: float = 252) -> dict:
    """
    Calculate key performance metrics for a strategy.
    
    Args:
        data: DataFrame with 'equity' and 'benchmark_equity' columns
        ann_factor: Annualization factor (252 for daily, 52 for weekly, 12 for monthly)
        
    Returns:
        Dictionary with performance metrics
    """
    # Calculate returns and metrics
    daily_returns = data['equity'].pct_change().dropna()
    benchmark_returns = data['benchmark_equity'].pct_change().dropna()
    
    # Calculate metrics
    start_date = data.index[0]
    end_date = data.index[-1]
    years = (end_date - start_date).days / 365.25
    
    # Strategy metrics
    total_return = data['equity'].iloc[-1] / data['equity'].iloc[0] - 1
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    max_drawdown = data['drawdown'].min()
    
    # Benchmark metrics
    benchmark_total_return = data['benchmark_equity'].iloc[-1] / data['benchmark_equity'].iloc[0] - 1
    benchmark_cagr = (1 + benchmark_total_return) ** (1 / years) - 1 if years > 0 else 0
    
    # Risk-adjusted metrics
    if len(daily_returns) > 1:
        volatility = daily_returns.std() * np.sqrt(ann_factor)
        sharpe_ratio = (cagr - 0.02) / volatility if volatility > 0 else 0
        
        # Sortino ratio (downside risk)
        negative_returns = daily_returns[daily_returns < 0]
        downside_dev = negative_returns.std() * np.sqrt(ann_factor) if len(negative_returns) > 1 else volatility
        sortino_ratio = (cagr - 0.02) / downside_dev if downside_dev > 0 else 0
    else:
        volatility = 0
        sharpe_ratio = 0
        sortino_ratio = 0
    
    # Alpha (outperformance)
    alpha = cagr - benchmark_cagr
    
    # Create metrics dictionary
    metrics = {
        'start_date': start_date,
        'end_date': end_date,
        'total_return': total_return,
        'cagr': cagr,
        'volatility': volatility,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'benchmark_return': benchmark_total_return,
        'benchmark_cagr': benchmark_cagr,
        'alpha': alpha
    }
    
    return metrics
