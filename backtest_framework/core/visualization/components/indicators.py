"""
Technical indicator visualization components.
"""
from typing import Tuple
import pandas as pd
import plotly.graph_objects as go
from .styling import ChartStyler


class IndicatorPlots:
    """Handles technical indicator visualizations."""
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize indicator plots.
        
        Args:
            data: DataFrame with OHLCV data and indicators
        """
        self.data = data
        self.styler = ChartStyler()
    
    def add_kdj_indicators(self, fig: go.Figure, row: int = 1, col: int = 1, 
                          is_monthly: bool = False) -> Tuple[go.Figure, bool]:
        """
        Add KDJ indicator lines to the specified subplot.
        
        Args:
            fig: Plotly figure to add trace to
            row: Row index for the subplot
            col: Column index for the subplot
            is_monthly: Whether to use monthly KDJ indicators
            
        Returns:
            Tuple of (updated figure, boolean indicating if indicators were found)
        """
        # Only using snake_case naming convention going forward
        prefix = "monthly_" if is_monthly else ""
        
        # Check if all KDJ indicators are present
        kdj_cols = [f"{prefix}k", f"{prefix}d", f"{prefix}j"]
        if not all(col in self.data.columns for col in kdj_cols):
            return fig, False
            
        # Add K line
        fig.add_trace(
            go.Scatter(
                x=self.data.index,
                y=self.data[f"{prefix}k"],
                mode='lines',
                line=self.styler.get_line_style(color=self.styler.COLORS['kdj_k']),
                name=f"{prefix.capitalize()}K"
            ),
            row=row, col=col
        )
        
        # Add D line
        fig.add_trace(
            go.Scatter(
                x=self.data.index,
                y=self.data[f"{prefix}d"],
                mode='lines',
                line=self.styler.get_line_style(color=self.styler.COLORS['kdj_d']),
                name=f"{prefix.capitalize()}D"
            ),
            row=row, col=col
        )
        
        # Add J line
        fig.add_trace(
            go.Scatter(
                x=self.data.index,
                y=self.data[f"{prefix}j"],
                mode='lines',
                line=self.styler.get_line_style(color=self.styler.COLORS['kdj_j'], width=2),
                name=f"{prefix.capitalize()}J"
            ),
            row=row, col=col
        )
        
        # Add reference lines
        fig.add_hline(y=80, line_width=1, line_dash="dash", line_color="gray", 
                    annotation_text="Overbought", row=row, col=col)
        fig.add_hline(y=20, line_width=1, line_dash="dash", line_color="gray", 
                    annotation_text="Oversold", row=row, col=col)
        
        # Add ending value annotations for KDJ values
        self._add_kdj_ending_annotations(fig, row, col, prefix)
        
        # Add golden/dead crosses for monthly KDJ
        if is_monthly:
            self._add_kdj_crosses(fig, prefix, row, col)
        
        return fig, True
    
    def add_sma(self, fig: go.Figure, row: int = 1, col: int = 1) -> go.Figure:
        """
        Add Simple Moving Average to the price chart.
        
        Args:
            fig: Plotly figure to add trace to
            row: Row index for the subplot
            col: Column index for the subplot
            
        Returns:
            Updated figure
        """
        if 'SMA' in self.data.columns:
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
    
    def _add_kdj_ending_annotations(self, fig: go.Figure, row: int, col: int, prefix: str) -> None:
        """
        Add ending value annotations for KDJ indicators at their actual value positions.
        
        Args:
            fig: Plotly figure to add annotations to
            row: Row index for the subplot
            col: Column index for the subplot
            prefix: Prefix for column names (e.g., "monthly_")
        """
        # Get final values for each KDJ component
        final_k = self.data[f"{prefix}k"].iloc[-1]
        final_d = self.data[f"{prefix}d"].iloc[-1]
        final_j = self.data[f"{prefix}j"].iloc[-1]
        
        # Generate correct xref and yref for subplot
        if row == 1:
            xref = "x domain"
            yref = "y"
        else:
            xref = f"x{row} domain"
            yref = f"y{row}"
        
        # Define KDJ values with their colors
        kdj_values = [
            ('K', final_k, self.styler.COLORS['kdj_k']),
            ('D', final_d, self.styler.COLORS['kdj_d']),
            ('J', final_j, self.styler.COLORS['kdj_j'])
        ]
        
        # Sort by value for better positioning
        kdj_values.sort(key=lambda x: x[1], reverse=True)
        
        for i, (label, value, color) in enumerate(kdj_values):
            # Format value
            formatted_value = f"{value:.1f}"
            
            # Apply small vertical offset if values are too close
            if len(kdj_values) > 1:
                max_val = kdj_values[0][1]
                min_val = kdj_values[-1][1]
                y_offset = i * 3 if abs(max_val - min_val) < 15 else 0
            else:
                y_offset = 0
            
            # Add annotation at actual value position
            fig.add_annotation(
                x=1.002,  # Just outside the plot area
                y=value + y_offset,  # Actual value position with offset if needed
                xref=xref,
                yref=yref,
                text=formatted_value,
                showarrow=False,
                font=dict(color=color, size=9, family="Arial", weight="bold"),
                bgcolor="rgba(0, 0, 0, 0.7)",
                bordercolor=color,
                borderwidth=1,
                borderpad=2,
                xanchor="left",
                yanchor="middle"
            )
        return fig
    
    def _add_kdj_crosses(self, fig: go.Figure, prefix: str, row: int, col: int) -> None:
        """
        Add golden and dead cross markers for KDJ indicators.
        
        Args:
            fig: Plotly figure to add trace to
            prefix: Prefix for column names (e.g., "monthly_")
            row: Row index for the subplot
            col: Column index for the subplot
        """
        # Find golden crosses (J crosses above K)
        golden_crosses = []
        for i in range(1, len(self.data)):
            if (self.data[f'{prefix}j'].iloc[i-1] <= self.data[f'{prefix}k'].iloc[i-1] and 
                self.data[f'{prefix}j'].iloc[i] >= self.data[f'{prefix}k'].iloc[i]):
                golden_crosses.append(self.data.index[i])
        
        # Find dead crosses (J crosses below K)
        dead_crosses = []
        for i in range(1, len(self.data)):
            if (self.data[f'{prefix}j'].iloc[i-1] >= self.data[f'{prefix}k'].iloc[i-1] and 
                self.data[f'{prefix}j'].iloc[i] < self.data[f'{prefix}k'].iloc[i]):
                dead_crosses.append(self.data.index[i])
        
        # Add markers for golden crosses
        if golden_crosses:
            j_values = [self.data.loc[date, f'{prefix}j'] for date in golden_crosses]
            fig.add_trace(
                go.Scatter(
                    x=golden_crosses,
                    y=j_values,
                    mode='markers',
                    marker=self.styler.get_marker_style(
                        symbol='star',
                        size=10,
                        color='gold'
                    ),
                    name="Golden Cross"
                ),
                row=row, col=col
            )
        
        # Add markers for dead crosses
        if dead_crosses:
            j_values = [self.data.loc[date, f'{prefix}j'] for date in dead_crosses]
            fig.add_trace(
                go.Scatter(
                    x=dead_crosses,
                    y=j_values,
                    mode='markers',
                    marker=self.styler.get_marker_style(
                        symbol='x',
                        size=10,
                        color='black'
                    ),
                    name="Dead Cross"
                ),
                row=row, col=col
            )
