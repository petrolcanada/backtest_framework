"""
Metrics cards component for displaying key performance indicators.
"""
import streamlit as st
import pandas as pd
from typing import Optional, Any
from utils.formatters import format_percentage, format_currency, format_ratio


class MetricCards:
    """Display key performance metrics in card format."""
    
    def __init__(self, results: pd.DataFrame, engine: Optional[Any] = None):
        self.results = results
        self.engine = engine
    
    def render(self):
        """Render the metrics cards."""
        st.subheader("📊 Key Performance Metrics")
        
        # Calculate metrics
        metrics = self._calculate_metrics()
        
        # Display in columns
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            st.metric(
                label="Total Return",
                value=format_percentage(metrics['total_return']),
                delta=format_percentage(metrics['vs_benchmark']) if metrics['vs_benchmark'] is not None else None,
                delta_color="normal" if metrics['vs_benchmark'] is None else "normal"
            )
        
        with col2:
            st.metric(
                label="CAGR",
                value=format_percentage(metrics['cagr']),
                delta=None
            )
        
        with col3:
            st.metric(
                label="Sharpe Ratio",
                value=format_ratio(metrics['sharpe']),
                delta=None
            )
        
        with col4:
            st.metric(
                label="Max Drawdown",
                value=format_percentage(metrics['max_drawdown']),
                delta=None,
                delta_color="inverse"  # Red is bad for drawdown
            )
        
        with col5:
            st.metric(
                label="Win Rate",
                value=format_percentage(metrics['win_rate']),
                delta=None
            )
        
        with col6:
            st.metric(
                label="Total Trades",
                value=f"{metrics['total_trades']:,}",
                delta=None
            )
        
        # Second row of metrics
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            st.metric(
                label="Initial Capital",
                value=format_currency(metrics['initial_capital'], 0),
                delta=None
            )
        
        with col2:
            st.metric(
                label="Final Equity",
                value=format_currency(metrics['final_equity'], 0),
                delta=None
            )
        
        with col3:
            st.metric(
                label="Profit Factor",
                value=format_ratio(metrics['profit_factor']),
                delta=None
            )
        
        with col4:
            st.metric(
                label="Avg Win/Loss",
                value=format_ratio(metrics['avg_win_loss']),
                delta=None
            )
        
        with col5:
            st.metric(
                label="Volatility",
                value=format_percentage(metrics['volatility']),
                delta=None
            )
        
        with col6:
            if self.engine:
                st.metric(
                    label="Commission",
                    value=format_percentage(self.engine.commission * 100),
                    delta=None
                )
    
    def _calculate_metrics(self) -> dict:
        """Calculate all metrics from results DataFrame."""
        metrics = {}
        
        # Basic returns
        if 'returns' in self.results.columns:
            metrics['total_return'] = self.results['returns'].iloc[-1] * 100
        else:
            metrics['total_return'] = 0
        
        # CAGR
        if 'cagr' in self.results.columns:
            metrics['cagr'] = self.results['cagr'].iloc[-1] * 100
        else:
            metrics['cagr'] = 0
        
        # Sharpe Ratio
        if 'sharpe_ratio' in self.results.columns:
            metrics['sharpe'] = self.results['sharpe_ratio'].iloc[-1]
        else:
            metrics['sharpe'] = 0
        
        # Max Drawdown
        if 'max_drawdown' in self.results.columns:
            metrics['max_drawdown'] = self.results['max_drawdown'].iloc[-1] * 100
        else:
            metrics['max_drawdown'] = 0
        
        # Win Rate
        if 'win_rate' in self.results.columns:
            metrics['win_rate'] = self.results['win_rate'].iloc[-1] * 100
        else:
            metrics['win_rate'] = 0
        
        # Trade Count
        if 'trade_count' in self.results.columns:
            metrics['total_trades'] = int(self.results['trade_count'].iloc[-1])
        else:
            metrics['total_trades'] = 0
        
        # Capital metrics
        if self.engine:
            metrics['initial_capital'] = self.engine.initial_capital
        else:
            metrics['initial_capital'] = 10000  # Default
        
        if 'equity' in self.results.columns:
            metrics['final_equity'] = self.results['equity'].iloc[-1]
        else:
            metrics['final_equity'] = metrics['initial_capital']
        
        # Benchmark comparison
        if 'benchmark_returns' in self.results.columns:
            benchmark_return = self.results['benchmark_returns'].iloc[-1] * 100
            metrics['vs_benchmark'] = metrics['total_return'] - benchmark_return
        else:
            metrics['vs_benchmark'] = None
        
        # Additional metrics
        metrics['profit_factor'] = self._calculate_profit_factor()
        metrics['avg_win_loss'] = self._calculate_avg_win_loss()
        metrics['volatility'] = self._calculate_volatility()
        
        return metrics
    
    def _calculate_profit_factor(self) -> float:
        """Calculate profit factor from trades."""
        if 'trade_pnl' not in self.results.columns:
            return 0
        
        trades = self.results[self.results['trade_pnl'] != 0]['trade_pnl']
        if len(trades) == 0:
            return 0
        
        wins = trades[trades > 0].sum()
        losses = abs(trades[trades < 0].sum())
        
        return wins / losses if losses > 0 else 0
    
    def _calculate_avg_win_loss(self) -> float:
        """Calculate average win/loss ratio."""
        if 'trade_pnl' not in self.results.columns:
            return 0
        
        trades = self.results[self.results['trade_pnl'] != 0]['trade_pnl']
        if len(trades) == 0:
            return 0
        
        wins = trades[trades > 0]
        losses = trades[trades < 0]
        
        if len(wins) == 0 or len(losses) == 0:
            return 0
        
        avg_win = wins.mean()
        avg_loss = abs(losses.mean())
        
        return avg_win / avg_loss if avg_loss > 0 else 0
    
    def _calculate_volatility(self) -> float:
        """Calculate annualized volatility."""
        if 'equity' not in self.results.columns:
            return 0
        
        returns = self.results['equity'].pct_change().dropna()
        if len(returns) == 0:
            return 0
        
        # Annualize assuming daily data
        return returns.std() * (252 ** 0.5) * 100
