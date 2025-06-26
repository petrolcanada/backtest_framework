"""Core strategy modules for generating trading signals."""
from backtest_framework.core.strategies.base import BaseStrategy
from backtest_framework.core.strategies.kdj_cross import (
    GoldenDeadCrossStrategyBase,
    GoldenDeadCrossStrategyMonthly,
    GoldenDeadCrossStrategyDaily
)

__all__ = [
    'BaseStrategy', 
    'GoldenDeadCrossStrategyBase',
    'GoldenDeadCrossStrategyMonthly',
    'GoldenDeadCrossStrategyDaily'
]
