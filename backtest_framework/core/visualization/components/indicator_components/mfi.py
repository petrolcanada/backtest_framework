"""
MFI (Money Flow Index) indicator visualization component.
"""
from typing import Tuple
import pandas as pd
import plotly.graph_objects as go
from .base import BaseIndicatorVisualization
from ..styling import ChartStyler


class MFI(BaseIndicatorVisualization):
    """Visualization component for MFI (Money Flow Index) indicator."""
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize MFI visualization.
        
        Args:
            data: DataFrame containing MFI data
        """
        super().__init__(data)
        self.styler = ChartStyler()
        self.required_columns = ['MFI']
    
    def check_data_availability(self) -> bool:
        """
        Check if MFI data is available.
        
        Returns:
            True if all required columns are present, False otherwise
        """
        return all(col in self.data.columns for col in self.required_columns)
    
    def get_required_columns(self) -> list:
        """
        Get the list of required column names for MFI.
        
        Returns:
            List of required column names
        """
        return self.required_columns
    
    def add_to_chart(self, fig: go.Figure, row: int = 1, col: int = 1) -> Tuple[go.Figure, bool]:
        """
        Add MFI indicator line to the specified subplot.
        
        Args:
            fig: Plotly figure to add trace to
            row: Row index for the subplot
            col: Column index for the subplot
            
        Returns:
            Tuple of (updated figure, boolean indicating if indicators were found)
        """
        if not self.check_data_availability():
            return fig, False
            
        # Add MFI line with area fill for better visualization
        fig.add_trace(
            go.Scatter(
                x=self.data.index,
                y=self.data['MFI'],
                mode='lines',
                line=self.styler.get_line_style(color='purple', width=2),
                name="MFI",
                fill='tonexty' if row > 1 else None,
                fillcolor='rgba(128, 0, 128, 0.1)'
            ),
            row=row, col=col
        )
        
        # Add reference lines for MFI interpretation
        fig.add_hline(y=80, line_width=1, line_dash="dash", line_color="red", 
                    annotation_text="Overbought", row=row, col=col)
        fig.add_hline(y=50, line_width=1, line_dash="dot", line_color="gray", 
                    annotation_text="Midline", row=row, col=col)
        fig.add_hline(y=20, line_width=1, line_dash="dash", line_color="green", 
                    annotation_text="Oversold", row=row, col=col)
        
        # Add ending value annotations
        self._add_mfi_ending_annotations(fig, row, col)
        
        # Add overbought/oversold zones
        self._add_mfi_zones(fig, row, col)
        
        return fig, True
    
    def _add_mfi_ending_annotations(self, fig: go.Figure, row: int, col: int) -> None:
        """
        Add ending value annotations for MFI indicator.
        
        Args:
            fig: Plotly figure to add annotations to
            row: Row index for the subplot
            col: Column index for the subplot
        """
        # Get final MFI value
        final_mfi = self.data['MFI'].iloc[-1]
        
        # Generate correct xref and yref for subplot
        if row == 1:
            xref = "x domain"
            yref = "y"
        else:
            xref = f"x{row} domain"
            yref = f"y{row}"
        
        # Format value and determine market condition
        formatted_value = f"{final_mfi:.1f}"
        if final_mfi >= 80:
            condition = "Overbought"
            bg_color = "rgba(255, 0, 0, 0.8)"  # Red
        elif final_mfi <= 20:
            condition = "Oversold"
            bg_color = "rgba(0, 255, 0, 0.8)"  # Green
        elif final_mfi >= 60:
            condition = "Bullish"
            bg_color = "rgba(255, 165, 0, 0.8)"  # Orange
        elif final_mfi <= 40:
            condition = "Bearish"
            bg_color = "rgba(255, 255, 0, 0.8)"  # Yellow
        else:
            condition = "Neutral"
            bg_color = "rgba(128, 128, 128, 0.8)"  # Gray
        
        # Add annotation
        fig.add_annotation(
            x=1.002,  # Just outside the plot area
            y=final_mfi,
            xref=xref,
            yref=yref,
            text=f"{formatted_value}<br>{condition}",
            showarrow=False,
            font=dict(color="white", size=9, family="Arial", weight="bold"),
            bgcolor=bg_color,
            bordercolor="white",
            borderwidth=1,
            borderpad=2,
            xanchor="left",
            yanchor="middle"
        )
    
    def _add_mfi_zones(self, fig: go.Figure, row: int, col: int) -> None:
        """
        Add colored zones for overbought/oversold regions.
        
        Args:
            fig: Plotly figure to add zones to
            row: Row index for the subplot
            col: Column index for the subplot
        """
        # Add overbought zone (80-100)
        fig.add_hrect(
            y0=80, y1=100,
            fillcolor="rgba(255, 0, 0, 0.1)",
            layer="below",
            line_width=0,
            row=row, col=col
        )
        
        # Add oversold zone (0-20)
        fig.add_hrect(
            y0=0, y1=20,
            fillcolor="rgba(0, 255, 0, 0.1)",
            layer="below",
            line_width=0,
            row=row, col=col
        )
