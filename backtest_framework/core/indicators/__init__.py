"""Import all indicator implementations to ensure they're registered correctly."""
# Import indicator modules to ensure decorators are processed
from backtest_framework.core.indicators import kdj
from backtest_framework.core.indicators import sma
from backtest_framework.core.indicators import adx
from backtest_framework.core.indicators import mfi
from backtest_framework.core.indicators import rsi

# Import derived factor modules
from backtest_framework.core.indicators import kdj_derived_factors
from backtest_framework.core.indicators import adx_derived_factors
from backtest_framework.core.indicators import mfi_derived_factors
from backtest_framework.core.indicators import rsi_derived_factors

# Re-export key classes for convenience
from backtest_framework.core.indicators.registry import IndicatorRegistry
from backtest_framework.core.indicators.calculator import IndicatorCalculator
from backtest_framework.core.indicators.resolver import DependencyResolver
