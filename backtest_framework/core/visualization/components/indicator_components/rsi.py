"""
RSI (Relative Strength Index) indicator visualization component.
"""
from typing import Tuple
import pandas as pd
import plotly.graph_objects as go
from .base import BaseIndicatorVisualization
from ..styling import ChartStyler


class RSI(BaseIndicatorVisualization):
    """Visualization component for RSI (Relative Strength Index) indicator."""
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize RSI visualization.
        
        Args:
            data: DataFrame containing RSI data
        """
        super().__init__(data)
        self.styler = ChartStyler()
        self.required_columns = ['RSI']
    
    def check_data_availability(self) -> bool:
        """
        Check if RSI data is available.
        
        Returns:
            True if all required columns are present, False otherwise
        """
        return all(col in self.data.columns for col in self.required_columns)
    
    def get_required_columns(self) -> list:
        """
        Get the list of required column names for RSI.
        
        Returns:
            List of required column names
        """
        return self.required_columns
    
    def add_to_chart(self, fig: go.Figure, row: int = 1, col: int = 1) -> Tuple[go.Figure, bool]:
        """
        Add RSI indicator line to the specified subplot.
        
        Args:
            fig: Plotly figure to add trace to
            row: Row index for the subplot
            col: Column index for the subplot
            
        Returns:
            Tuple of (updated figure, boolean indicating if indicators were found)
        """
        if not self.check_data_availability():
            return fig, False
            
        # Add RSI line
        fig.add_trace(
            go.Scatter(
                x=self.data.index,
                y=self.data['RSI'],
                mode='lines',
                line=self.styler.get_line_style(color='cyan', width=2),
                name="RSI"
            ),
            row=row, col=col
        )
        
        # Add reference lines for RSI interpretation
        fig.add_hline(y=70, line_width=1, line_dash="dash", line_color="red", 
                    annotation_text="Overbought", row=row, col=col)
        fig.add_hline(y=50, line_width=1, line_dash="dot", line_color="gray", 
                    annotation_text="Midline", row=row, col=col)
        fig.add_hline(y=30, line_width=1, line_dash="dash", line_color="green", 
                    annotation_text="Oversold", row=row, col=col)
        
        # Add ending value annotations
        self._add_rsi_ending_annotations(fig, row, col)
        
        # Add overbought/oversold zones
        self._add_rsi_zones(fig, row, col)
        
        # Add divergence detection if available
        self._add_rsi_divergences(fig, row, col)
        
        return fig, True
    
    def _add_rsi_ending_annotations(self, fig: go.Figure, row: int, col: int) -> None:
        """
        Add ending value annotations for RSI indicator.
        
        Args:
            fig: Plotly figure to add annotations to
            row: Row index for the subplot
            col: Column index for the subplot
        """
        # Get final RSI value
        final_rsi = self.data['RSI'].iloc[-1]
        
        # Generate correct xref and yref for subplot
        if row == 1:
            xref = "x domain"
            yref = "y"
        else:
            xref = f"x{row} domain"
            yref = f"y{row}"
        
        # Format value and determine market condition
        formatted_value = f"{final_rsi:.1f}"
        if final_rsi >= 70:
            condition = "Overbought"
            bg_color = "rgba(255, 0, 0, 0.8)"  # Red
        elif final_rsi <= 30:
            condition = "Oversold"
            bg_color = "rgba(0, 255, 0, 0.8)"  # Green
        elif final_rsi >= 55:
            condition = "Bullish"
            bg_color = "rgba(255, 165, 0, 0.8)"  # Orange
        elif final_rsi <= 45:
            condition = "Bearish"
            bg_color = "rgba(255, 255, 0, 0.8)"  # Yellow
        else:
            condition = "Neutral"
            bg_color = "rgba(128, 128, 128, 0.8)"  # Gray
        
        # Add annotation
        fig.add_annotation(
            x=1.002,  # Just outside the plot area
            y=final_rsi,
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
    
    def _add_rsi_zones(self, fig: go.Figure, row: int, col: int) -> None:
        """
        Add colored zones for overbought/oversold regions.
        
        Args:
            fig: Plotly figure to add zones to
            row: Row index for the subplot
            col: Column index for the subplot
        """
        # Add overbought zone (70-100)
        fig.add_hrect(
            y0=70, y1=100,
            fillcolor="rgba(255, 0, 0, 0.1)",
            layer="below",
            line_width=0,
            row=row, col=col
        )
        
        # Add oversold zone (0-30)
        fig.add_hrect(
            y0=0, y1=30,
            fillcolor="rgba(0, 255, 0, 0.1)",
            layer="below",
            line_width=0,
            row=row, col=col
        )
    
    def _add_rsi_divergences(self, fig: go.Figure, row: int, col: int) -> None:
        """
        Add RSI divergence detection markers.
        
        Args:
            fig: Plotly figure to add divergences to
            row: Row index for the subplot
            col: Column index for the subplot
        """
        # Simple divergence detection - look for RSI extremes
        rsi_data = self.data['RSI'].dropna()
        
        # Find potential bullish divergences (RSI oversold and rising)
        bullish_signals = []
        for i in range(5, len(rsi_data)):
            # Look for RSI below 30 and forming higher lows
            if (rsi_data.iloc[i] <= 35 and 
                rsi_data.iloc[i] > rsi_data.iloc[i-2] and 
                rsi_data.iloc[i-2] > rsi_data.iloc[i-4]):
                bullish_signals.append(rsi_data.index[i])
        
        # Find potential bearish divergences (RSI overbought and falling)
        bearish_signals = []
        for i in range(5, len(rsi_data)):
            # Look for RSI above 65 and forming lower highs
            if (rsi_data.iloc[i] >= 65 and 
                rsi_data.iloc[i] < rsi_data.iloc[i-2] and 
                rsi_data.iloc[i-2] < rsi_data.iloc[i-4]):
                bearish_signals.append(rsi_data.index[i])
        
        # Add bullish divergence markers
        if bullish_signals:
            rsi_values = [self.data.loc[date, 'RSI'] for date in bullish_signals]
            fig.add_trace(
                go.Scatter(
                    x=bullish_signals,
                    y=rsi_values,
                    mode='markers',
                    marker=self.styler.get_marker_style(
                        symbol='triangle-up',
                        size=8,
                        color='green'
                    ),
                    name="RSI Bullish Div"
                ),
                row=row, col=col
            )
        
        # Add bearish divergence markers
        if bearish_signals:
            rsi_values = [self.data.loc[date, 'RSI'] for date in bearish_signals]
            fig.add_trace(
                go.Scatter(
                    x=bearish_signals,
                    y=rsi_values,
                    mode='markers',
                    marker=self.styler.get_marker_style(
                        symbol='triangle-down',
                        size=8,
                        color='red'
                    ),
                    name="RSI Bearish Div"
                ),
                row=row, col=col
            )
