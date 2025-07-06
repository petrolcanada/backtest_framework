"""
SMA (Simple Moving Average) indicator visualization component.
"""
from typing import Tuple
import pandas as pd
import plotly.graph_objects as go
from .base import BaseIndicatorVisualization
from ..styling import ChartStyler


class SMA(BaseIndicatorVisualization):
    """Visualization component for Simple Moving Average indicator."""
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize SMA visualization.
        
        Args:
            data: DataFrame containing SMA data
        """
        super().__init__(data)
        self.styler = ChartStyler()
        self.required_columns = ['SMA']
    
    def check_data_availability(self) -> bool:
        """
        Check if SMA data is available.
        
        Returns:
            True if all required columns are present, False otherwise
        """
        return all(col in self.data.columns for col in self.required_columns)
    
    def get_required_columns(self) -> list:
        """
        Get the list of required column names for SMA.
        
        Returns:
            List of required column names
        """
        return self.required_columns
    
    def add_to_chart(self, fig: go.Figure, row: int = 1, col: int = 1) -> Tuple[go.Figure, bool]:
        """
        Add SMA line to the specified subplot.
        
        Args:
            fig: Plotly figure to add trace to
            row: Row index for the subplot
            col: Column index for the subplot
            
        Returns:
            Tuple of (updated figure, boolean indicating if indicator was added)
        """
        if not self.check_data_availability():
            return fig, False
            
        # Add SMA line
        fig.add_trace(
            go.Scatter(
                x=self.data.index,
                y=self.data['SMA'],
                mode='lines',
                line=self.styler.get_line_style(color='blue'),
                name="SMA"
            ),
            row=row, col=col
        )
        
        return fig, True
