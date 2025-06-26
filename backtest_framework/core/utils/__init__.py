"""
Utility modules for the backtesting framework.
"""

from backtest_framework.core.utils.chart_utils import (
    plot_strategy_vs_benchmark,
    calculate_cum_returns,
    calculate_performance_metrics
)

__all__ = [
    'plot_strategy_vs_benchmark',
    'calculate_cum_returns',
    'calculate_performance_metrics'
]
