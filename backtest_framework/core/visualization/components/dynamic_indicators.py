"""
Dynamic indicator visualization coordinator.

This module provides functionality to dynamically determine and apply
indicator visualizations based on computed indicators and their registrations.
"""
from typing import Dict, List, Tuple, Type
import pandas as pd
import plotly.graph_objects as go
from backtest_framework.core.indicators.registry import IndicatorRegistry
from .indicator_components.base import BaseIndicatorVisualization


class DynamicIndicatorCoordinator:
    """
    Coordinates dynamic indicator visualization based on computed indicators.
    
    This class automatically determines which indicator visualization components
    to use based on what indicators have been computed and registered.
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the coordinator.
        
        Args:
            data: DataFrame containing computed indicator data
        """
        self.data = data
        self._visualization_registry = self._build_visualization_registry()
    
    def _build_visualization_registry(self) -> Dict[str, Type[BaseIndicatorVisualization]]:
        """
        Build a registry of available visualization classes.
        
        Returns:
            Dictionary mapping visualization class names to their classes
        """
        # Import available visualization classes
        from .indicator_components.monthly_kdj import MonthlyKDJ
        from .indicator_components.sma import SMA
        
        # Build registry
        registry = {
            'MonthlyKDJ': MonthlyKDJ,
            'SMA': SMA,
            # Add more visualization classes here as they're created
        }
        
        return registry
    
    def get_computed_indicators(self) -> List[str]:
        """
        Get list of indicators that have been computed based on data columns.
        
        Returns:
            List of computed indicator names
        """
        # Get all registered indicators
        all_indicators = IndicatorRegistry.list_all()
        
        # Check which indicators have their outputs present in the data
        computed_indicators = []
        for indicator_name in all_indicators:
            indicator_info = IndicatorRegistry.get(indicator_name)
            outputs = indicator_info['outputs']
            
            # Check if all outputs are present in data
            if all(output in self.data.columns for output in outputs):
                computed_indicators.append(indicator_name)
        
        return computed_indicators
    
    def get_available_visualizations(self) -> List[str]:
        """
        Get list of visualization classes that can be applied to current data.
        
        Returns:
            List of visualization class names that have data available
        """
        computed_indicators = self.get_computed_indicators()
        
        # Get visualization classes from registry
        viz_classes = IndicatorRegistry.get_visualization_classes(computed_indicators)
        
        # Filter to only those that are actually available and have data
        available_viz = []
        for viz_class_name in viz_classes:
            if viz_class_name in self._visualization_registry:
                viz_class = self._visualization_registry[viz_class_name]
                viz_instance = viz_class(self.data)
                if viz_instance.check_data_availability():
                    available_viz.append(viz_class_name)
        
        return available_viz
    
    def apply_indicators_to_chart(self, fig: go.Figure, row: int = 1, col: int = 1) -> Tuple[go.Figure, bool]:
        """
        Dynamically apply all available indicator visualizations to a chart.
        
        Args:
            fig: Plotly figure to add indicators to
            row: Row index for the subplot
            col: Column index for the subplot
            
        Returns:
            Tuple of (updated figure, success boolean)
        """
        available_viz = self.get_available_visualizations()
        
        if not available_viz:
            return fig, False
        
        success = False
        for viz_class_name in available_viz:
            viz_class = self._visualization_registry[viz_class_name]
            viz_instance = viz_class(self.data)
            
            # Apply the visualization
            fig, indicator_success = viz_instance.add_to_chart(fig, row, col)
            if indicator_success:
                success = True
        
        return fig, success
    
    def apply_price_panel_indicators(self, fig: go.Figure, row: int = 1, col: int = 1) -> Tuple[go.Figure, bool]:
        """
        Apply indicators that belong on the price panel (like SMA).
        
        Args:
            fig: Plotly figure to add indicators to
            row: Row index for the subplot
            col: Column index for the subplot
            
        Returns:
            Tuple of (updated figure, success boolean)
        """
        # Define which indicators belong on the price panel
        price_panel_indicators = ['SMA']  # Add more as needed
        
        available_viz = self.get_available_visualizations()
        price_viz = [viz for viz in available_viz if viz in price_panel_indicators]
        
        if not price_viz:
            return fig, False
        
        success = False
        for viz_class_name in price_viz:
            viz_class = self._visualization_registry[viz_class_name]
            viz_instance = viz_class(self.data)
            
            # Apply the visualization
            fig, indicator_success = viz_instance.add_to_chart(fig, row, col)
            if indicator_success:
                success = True
        
        return fig, success
    
    def apply_indicator_panel_indicators(self, fig: go.Figure, row: int = 1, col: int = 1) -> Tuple[go.Figure, bool]:
        """
        Apply indicators that belong on the indicator panel (like KDJ).
        
        Args:
            fig: Plotly figure to add indicators to
            row: Row index for the subplot
            col: Column index for the subplot
            
        Returns:
            Tuple of (updated figure, success boolean)
        """
        # Define which indicators belong on the indicator panel
        indicator_panel_indicators = ['MonthlyKDJ']  # Add more as needed
        
        available_viz = self.get_available_visualizations()
        indicator_viz = [viz for viz in available_viz if viz in indicator_panel_indicators]
        
        if not indicator_viz:
            return fig, False
        
        success = False
        for viz_class_name in indicator_viz:
            viz_class = self._visualization_registry[viz_class_name]
            viz_instance = viz_class(self.data)
            
            # Apply the visualization
            fig, indicator_success = viz_instance.add_to_chart(fig, row, col)
            if indicator_success:
                success = True
        
        return fig, success
    
    def get_debug_info(self) -> Dict:
        """
        Get debug information about computed indicators and available visualizations.
        
        Returns:
            Dictionary with debug information
        """
        computed_indicators = self.get_computed_indicators()
        available_viz = self.get_available_visualizations()
        
        debug_info = {
            'computed_indicators': computed_indicators,
            'available_visualizations': available_viz,
            'data_columns': list(self.data.columns),
            'registered_indicators': IndicatorRegistry.list_all(),
            'visualization_registry': list(self._visualization_registry.keys())
        }
        
        return debug_info
