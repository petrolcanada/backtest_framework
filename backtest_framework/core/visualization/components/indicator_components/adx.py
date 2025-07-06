"""
ADX (Average Directional Index) indicator visualization component.
"""
from typing import Tuple
import pandas as pd
import plotly.graph_objects as go
from .base import BaseIndicatorVisualization
from ..styling import ChartStyler


class ADX(BaseIndicatorVisualization):
    """Visualization component for ADX (Average Directional Index) indicator."""
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize ADX visualization.
        
        Args:
            data: DataFrame containing ADX data
        """
        super().__init__(data)
        self.styler = ChartStyler()
        self.required_columns = ['ADX']
        self.optional_columns = ['ADX_SMA']
    
    def check_data_availability(self) -> bool:
        """
        Check if ADX data is available.
        
        Returns:
            True if required columns are present, False otherwise
        """
        return all(col in self.data.columns for col in self.required_columns)
    
    def get_required_columns(self) -> list:
        """
        Get the list of required column names for ADX.
        
        Returns:
            List of required column names
        """
        return self.required_columns
    
    def add_to_chart(self, fig: go.Figure, row: int = 1, col: int = 1) -> Tuple[go.Figure, bool]:
        """
        Add ADX indicator lines to the specified subplot.
        
        Args:
            fig: Plotly figure to add trace to
            row: Row index for the subplot
            col: Column index for the subplot
            
        Returns:
            Tuple of (updated figure, boolean indicating if indicators were found)
        """
        if not self.check_data_availability():
            return fig, False
            
        # Add ADX line
        fig.add_trace(
            go.Scatter(
                x=self.data.index,
                y=self.data['ADX'],
                mode='lines',
                line=self.styler.get_line_style(color='orange', width=2),
                name="ADX"
            ),
            row=row, col=col
        )
        
        # Add ADX SMA if available
        if 'ADX_SMA' in self.data.columns:
            fig.add_trace(
                go.Scatter(
                    x=self.data.index,
                    y=self.data['ADX_SMA'],
                    mode='lines',
                    line=self.styler.get_line_style(color='red', width=1),
                    name="ADX SMA"
                ),
                row=row, col=col
            )
        
        # Add reference lines for ADX interpretation
        fig.add_hline(y=25, line_width=1, line_dash="dash", line_color="gray", 
                    annotation_text="Strong Trend", row=row, col=col)
        fig.add_hline(y=50, line_width=1, line_dash="dash", line_color="lightgray", 
                    annotation_text="Very Strong", row=row, col=col)
        
        # Add ending value annotations
        self._add_adx_ending_annotations(fig, row, col)
        
        return fig, True
    
    def _add_adx_ending_annotations(self, fig: go.Figure, row: int, col: int) -> None:
        """
        Add ending value annotations for ADX indicators.
        
        Args:
            fig: Plotly figure to add annotations to
            row: Row index for the subplot
            col: Column index for the subplot
        """
        # Get final ADX value
        final_adx = self.data['ADX'].iloc[-1]
        
        # Generate correct xref and yref for subplot
        if row == 1:
            xref = "x domain"
            yref = "y"
        else:
            xref = f"x{row} domain"
            yref = f"y{row}"
        
        # Format value and determine trend strength
        formatted_value = f"{final_adx:.1f}"
        if final_adx >= 50:
            trend_label = "Very Strong"
            bg_color = "rgba(255, 0, 0, 0.8)"  # Red
        elif final_adx >= 25:
            trend_label = "Strong"
            bg_color = "rgba(255, 165, 0, 0.8)"  # Orange
        else:
            trend_label = "Weak"
            bg_color = "rgba(128, 128, 128, 0.8)"  # Gray
        
        # Add annotation
        fig.add_annotation(
            x=1.002,  # Just outside the plot area
            y=final_adx,
            xref=xref,
            yref=yref,
            text=f"{formatted_value}<br>{trend_label}",
            showarrow=False,
            font=dict(color="white", size=9, family="Arial", weight="bold"),
            bgcolor=bg_color,
            bordercolor="white",
            borderwidth=1,
            borderpad=2,
            xanchor="left",
            yanchor="middle"
        )
