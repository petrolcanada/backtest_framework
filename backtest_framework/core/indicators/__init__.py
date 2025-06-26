"""Import all indicator implementations to ensure they're registered correctly."""
# Import indicator modules to ensure decorators are processed
from backtest_framework.core.indicators import kdj
from backtest_framework.core.indicators import sma

# Re-export key classes for convenience
from backtest_framework.core.indicators.registry import IndicatorRegistry
from backtest_framework.core.indicators.calculator import IndicatorCalculator
from backtest_framework.core.indicators.resolver import DependencyResolver
