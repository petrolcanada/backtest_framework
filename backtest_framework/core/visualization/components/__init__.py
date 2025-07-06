"""
Visualization components for the backtesting framework.

This module contains focused components for different aspects of chart creation:
- titles: Title and subtitle generation
- styling: Theme and layout utilities  
- chart_elements: Basic chart components (candlesticks, volume, etc.)
- indicators: Technical indicator visualizations
- performance: Performance and benchmark charts
- allocation: Capital allocation visualizations
"""

from .titles import TitleGenerator
from .styling import ChartStyler
from .chart_elements import ChartElements
from .performance import PerformancePlots
from .allocation import AllocationPlots
from .dynamic_indicators import DynamicIndicatorCoordinator

# Note: indicators.py (old static system) has been replaced by dynamic_indicators.py
# and the modular indicator_components/ directory

__all__ = [
    'TitleGenerator',
    'ChartStyler', 
    'ChartElements',
    'PerformancePlots',
    'AllocationPlots',
    'DynamicIndicatorCoordinator'
]
