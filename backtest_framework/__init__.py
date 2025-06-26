"""
Backtesting Framework - A modular, dependency-aware backtesting framework for multi-ticker trading strategies.
"""
# Import core components
from backtest_framework.core.data.loader import DataLoader
from backtest_framework.core.indicators.calculator import IndicatorCalculator
from backtest_framework.core.indicators.registry import IndicatorRegistry
from backtest_framework.core.strategies.base import BaseStrategy
from backtest_framework.core.strategies.kdj_cross import (
    GoldenDeadCrossStrategyMonthly,
    GoldenDeadCrossStrategyDaily
)
from backtest_framework.core.backtest.engine import BacktestEngine
from backtest_framework.core.backtest.risk_management import DrawdownProtection, TrailingStop
from backtest_framework.core.visualization.plotter import Plotter
from backtest_framework.core.utils.helpers import (
    read_index_constituents,
    suppress_warnings,
    Timer,
    format_number,
    format_percentage
)

# Import indicator implementations to trigger registration
from backtest_framework.core.indicators import kdj, sma

# Version info
__version__ = "1.1.0"
__author__ = "Your Name"
__description__ = "A modular backtesting framework with decorator-based indicator registration and dividend support"

# All registered indicators are now managed via decorators - no config files needed!
