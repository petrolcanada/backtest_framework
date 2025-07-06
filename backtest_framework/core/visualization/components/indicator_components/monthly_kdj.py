"""
Monthly KDJ indicator visualization component.
"""
from typing import Tuple
import pandas as pd
import plotly.graph_objects as go
from .base import BaseIndicatorVisualization
from ..styling import ChartStyler


class MonthlyKDJ(BaseIndicatorVisualization):
    """Visualization component for Monthly KDJ indicator."""
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize Monthly KDJ visualization.
        
        Args:
            data: DataFrame containing monthly KDJ data
        """
        super().__init__(data)
        self.styler = ChartStyler()
        self.required_columns = ['monthly_k', 'monthly_d', 'monthly_j']
    
    def check_data_availability(self) -> bool:
        """
        Check if Monthly KDJ data is available.
        
        Returns:
            True if all required columns are present, False otherwise
        """
        return all(col in self.data.columns for col in self.required_columns)
    
    def get_required_columns(self) -> list:
        """
        Get the list of required column names for Monthly KDJ.
        
        Returns:
            List of required column names
        """
        return self.required_columns
    
    def add_to_chart(self, fig: go.Figure, row: int = 1, col: int = 1) -> Tuple[go.Figure, bool]:
        """
        Add Monthly KDJ indicator lines to the specified subplot.
        
        Args:
            fig: Plotly figure to add trace to
            row: Row index for the subplot
            col: Column index for the subplot
            
        Returns:
            Tuple of (updated figure, boolean indicating if indicators were found)
        """
        if not self.check_data_availability():
            return fig, False
            
        # Add K line
        fig.add_trace(
            go.Scatter(
                x=self.data.index,
                y=self.data['monthly_k'],
                mode='lines',
                line=self.styler.get_line_style(color=self.styler.COLORS['kdj_k']),
                name="Monthly K"
            ),
            row=row, col=col
        )
        
        # Add D line
        fig.add_trace(
            go.Scatter(
                x=self.data.index,
                y=self.data['monthly_d'],
                mode='lines',
                line=self.styler.get_line_style(color=self.styler.COLORS['kdj_d']),
                name="Monthly D"
            ),
            row=row, col=col
        )
        
        # Add J line
        fig.add_trace(
            go.Scatter(
                x=self.data.index,
                y=self.data['monthly_j'],
                mode='lines',
                line=self.styler.get_line_style(color=self.styler.COLORS['kdj_j'], width=2),
                name="Monthly J"
            ),
            row=row, col=col
        )
        
        # Add reference lines
        fig.add_hline(y=80, line_width=1, line_dash="dash", line_color="gray", 
                    annotation_text="Overbought", row=row, col=col)
        fig.add_hline(y=20, line_width=1, line_dash="dash", line_color="gray", 
                    annotation_text="Oversold", row=row, col=col)
        
        # Add ending value annotations for KDJ values
        self._add_kdj_ending_annotations(fig, row, col)
        
        # Add golden/dead crosses
        self._add_kdj_crosses(fig, row, col)
        
        return fig, True
    
    def _add_kdj_ending_annotations(self, fig: go.Figure, row: int, col: int) -> None:
        """
        Add ending value annotations for Monthly KDJ indicators at their actual value positions.
        
        Args:
            fig: Plotly figure to add annotations to
            row: Row index for the subplot
            col: Column index for the subplot
        """
        # Get final values for each KDJ component
        final_k = self.data['monthly_k'].iloc[-1]
        final_d = self.data['monthly_d'].iloc[-1]
        final_j = self.data['monthly_j'].iloc[-1]
        
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
    
    def _add_kdj_crosses(self, fig: go.Figure, row: int, col: int) -> None:
        """
        Add golden and dead cross markers for Monthly KDJ indicators.
        
        Args:
            fig: Plotly figure to add trace to
            row: Row index for the subplot
            col: Column index for the subplot
        """
        # Find golden crosses (J crosses above K)
        golden_crosses = []
        for i in range(1, len(self.data)):
            if (self.data['monthly_j'].iloc[i-1] <= self.data['monthly_k'].iloc[i-1] and 
                self.data['monthly_j'].iloc[i] >= self.data['monthly_k'].iloc[i]):
                golden_crosses.append(self.data.index[i])
        
        # Find dead crosses (J crosses below K)
        dead_crosses = []
        for i in range(1, len(self.data)):
            if (self.data['monthly_j'].iloc[i-1] >= self.data['monthly_k'].iloc[i-1] and 
                self.data['monthly_j'].iloc[i] < self.data['monthly_k'].iloc[i]):
                dead_crosses.append(self.data.index[i])
        
        # Add markers for golden crosses
        if golden_crosses:
            j_values = [self.data.loc[date, 'monthly_j'] for date in golden_crosses]
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
            j_values = [self.data.loc[date, 'monthly_j'] for date in dead_crosses]
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
