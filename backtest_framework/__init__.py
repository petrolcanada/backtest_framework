"""
Backtesting Framework - A modular, dependency-aware backtesting framework for multi-ticker trading strategies.

Zero-maintenance module system - use direct imports:
    from backtest_framework.core.data.loader import DataLoader
    from backtest_framework.core.backtest.engine import BacktestEngine
    from backtest_framework.core.indicators.registry import IndicatorRegistry
    from backtest_framework.core.signals.kdj_cross import GoldenDeadCrossStrategyMonthly
    from backtest_framework.core.visualization.plotter import Plotter
"""

# Version info
__version__ = "1.1.0"
__author__ = "Your Name"
__description__ = "A modular backtesting framework with decorator-based indicator registration and zero-maintenance modules"

# Import indicators to trigger registration (required for decorator system)
import backtest_framework.core.indicators.kdj
import backtest_framework.core.indicators.sma
import backtest_framework.core.indicators.adx
import backtest_framework.core.indicators.mfi
import backtest_framework.core.indicators.rsi
import backtest_framework.core.indicators.kdj_derived_factors
import backtest_framework.core.indicators.adx_derived_factors
import backtest_framework.core.indicators.mfi_derived_factors
import backtest_framework.core.indicators.rsi_derived_factors

# Note: All other imports should be direct imports as needed
# No more manual maintenance of import lists!
