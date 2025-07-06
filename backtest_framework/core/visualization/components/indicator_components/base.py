"""
Base class for indicator visualization components.
"""
from abc import ABC, abstractmethod
from typing import Tuple
import pandas as pd
import plotly.graph_objects as go


class BaseIndicatorVisualization(ABC):
    """
    Abstract base class for indicator visualization components.
    
    Each indicator visualization class should inherit from this and implement
    the required methods for adding indicator plots to charts.
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the indicator visualization.
        
        Args:
            data: DataFrame containing the indicator data
        """
        self.data = data
    
    @abstractmethod
    def add_to_chart(self, fig: go.Figure, row: int = 1, col: int = 1) -> Tuple[go.Figure, bool]:
        """
        Add indicator visualization to the chart.
        
        Args:
            fig: Plotly figure to add traces to
            row: Row index for the subplot
            col: Column index for the subplot
            
        Returns:
            Tuple of (updated figure, success boolean)
        """
        pass
    
    @abstractmethod
    def check_data_availability(self) -> bool:
        """
        Check if required data columns are available.
        
        Returns:
            True if all required columns are present, False otherwise
        """
        pass
    
    def get_required_columns(self) -> list:
        """
        Get the list of required column names for this indicator.
        
        Returns:
            List of required column names
        """
        # This should be overridden by subclasses if needed
        return []
