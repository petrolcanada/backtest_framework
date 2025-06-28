"""Core backtesting engine for simulating trading strategies."""
from .engine import BacktestEngine
from .portfolio_manager import PortfolioManager
from .cost_calculator import CostCalculator
from .trade_executor import TradeExecutor
from .trade_tracker import TradeTracker
from .performance_calculator import PerformanceCalculator
from .data_validator import DataValidator
from .risk_management import DrawdownProtection, TrailingStop

__all__ = [
    'BacktestEngine',
    'PortfolioManager',
    'CostCalculator', 
    'TradeExecutor',
    'TradeTracker',
    'PerformanceCalculator',
    'DataValidator',
    'DrawdownProtection',
    'TrailingStop'
]
