"""
Capital allocation visualization components.
"""
from typing import Tuple, Dict, List
import pandas as pd
import plotly.graph_objects as go
from .styling import ChartStyler


class AllocationPlots:
    """Handles capital allocation and portfolio composition visualizations."""
    
    def __init__(self, results: pd.DataFrame):
        """
        Initialize allocation plots.
        
        Args:
            results: DataFrame with backtest results
        """
        self.results = results
        self.styler = ChartStyler()
    
    def add_capital_allocation(self, fig: go.Figure, row: int = 1, col: int = 1) -> Tuple[go.Figure, bool]:
        """
        Add capital allocation stacked area chart to the specified subplot.
        Shows cash, long positions, short positions, and margin usage over time.
        
        Args:
            fig: Plotly figure to add trace to
            row: Row index for the subplot
            col: Column index for the subplot
            
        Returns:
            Tuple of (updated figure, boolean indicating if allocation data was found)
        """
        # Define expected column names for capital allocation
        allocation_cols = {
            'cash': ['cash', 'Cash', 'available_cash'],
            'original_cash': ['original_cash', 'Original_Cash'],
            'short_proceeds': ['short_proceeds', 'Short_Proceeds'],
            'long_positions': ['long_positions', 'Long_Positions', 'long_value', 'stock_value'],
            'short_positions': ['short_positions', 'Short_Positions', 'short_value'],
            'margin_cash': ['margin_cash', 'Margin_Cash'],  # Use margin_cash (negative values)
            'position': ['position'],  # Need position to detect short periods
        }
        
        # Find available columns
        found_cols = self._find_available_columns(allocation_cols)
        
        # Check if we have at least cash and long positions
        if 'cash' not in found_cols and 'long_positions' not in found_cols:
            return fig, False
        
        # Get the data for each component
        x_data = self.results.index
        allocation_data = self._prepare_allocation_data(found_cols, x_data)
        
        # Add allocation traces
        self._add_allocation_traces(fig, allocation_data, x_data, row, col)
        
        # Add summary annotation
        self._add_allocation_summary(fig, allocation_data, row, col)
        
        return fig, True
    
    def _find_available_columns(self, allocation_cols: Dict[str, List[str]]) -> Dict[str, str]:
        """Find available allocation columns in the results DataFrame."""
        found_cols = {}
        for category, possible_names in allocation_cols.items():
            for name in possible_names:
                if name in self.results.columns:
                    found_cols[category] = name
                    break
        return found_cols
    
    def _prepare_allocation_data(self, found_cols: Dict[str, str], x_data: pd.DatetimeIndex) -> Dict[str, pd.Series]:
        """Prepare allocation data series for plotting."""
        allocation_data = {
            'original_cash': pd.Series(0, index=x_data),
            'short_proceeds': pd.Series(0, index=x_data),
            'long_positions': self.results[found_cols['long_positions']] if 'long_positions' in found_cols else pd.Series(0, index=x_data),
            'short_positions': self._prepare_short_positions(found_cols, x_data),
            'margin_cash': self.results[found_cols['margin_cash']] if 'margin_cash' in found_cols else pd.Series(0, index=x_data)
        }
        
        # Use the new separate cash tracking if available
        if 'original_cash' in self.results.columns and 'short_proceeds' in self.results.columns:
            allocation_data['original_cash'] = self.results['original_cash']
            allocation_data['short_proceeds'] = self.results['short_proceeds']
        elif 'cash' in found_cols and 'position' in found_cols:
            # Fallback: estimate cash breakdown from total cash and position
            total_cash = self.results[found_cols['cash']]
            positions = self.results[found_cols['position']]
            
            # Estimate short sale proceeds when position is negative
            initial_capital = total_cash.iloc[0]  # Assume first value is initial capital
            
            for i in range(len(total_cash)):
                current_cash = total_cash.iloc[i]
                current_position = positions.iloc[i]
                
                if current_position < 0:
                    # During short position: cash includes proceeds
                    # Estimate proceeds as the excess cash above initial capital
                    estimated_proceeds = max(0, current_cash - initial_capital)
                    allocation_data['short_proceeds'].iloc[i] = estimated_proceeds
                    allocation_data['original_cash'].iloc[i] = current_cash - estimated_proceeds
                else:
                    # No short position: all cash is original
                    allocation_data['original_cash'].iloc[i] = current_cash
                    allocation_data['short_proceeds'].iloc[i] = 0
        else:
            # Fallback: treat all cash as original cash
            allocation_data['original_cash'] = self.results[found_cols['cash']] if 'cash' in found_cols else pd.Series(0, index=x_data)
        
        return allocation_data
    
    def _prepare_short_positions(self, found_cols: Dict[str, str], x_data: pd.DatetimeIndex) -> pd.Series:
        """Prepare short positions data (ensure negative values for proper display)."""
        if 'short_positions' in found_cols:
            short_data = self.results[found_cols['short_positions']]
            return -abs(short_data)  # Make sure short positions are negative
        else:
            return pd.Series(0, index=x_data)
    
    def _add_allocation_traces(self, fig: go.Figure, allocation_data: Dict[str, pd.Series], 
                             x_data: pd.DatetimeIndex, row: int, col: int) -> None:
        """Add allocation traces to the figure."""
        # 1. Margin cash (below zero, orange area) - shows as negative debt
        if not allocation_data['margin_cash'].eq(0).all():
            margin_debt_values = allocation_data['margin_cash'].clip(upper=0)  # Only negative values
            if not margin_debt_values.eq(0).all():
                fig.add_trace(
                    go.Scatter(
                        x=x_data,
                        y=margin_debt_values,
                        mode='lines',
                        line=dict(width=0),
                        fill='tozeroy',
                        fillcolor=self.styler.COLORS['margin'],
                        name="Margin Cash",
                        hovertemplate="<b>Margin Cash (Debt)</b><br>Date: %{x}<br>Value: $%{y:,.0f}<extra></extra>"
                    ),
                    row=row, col=col
                )
        
        # 2. Short positions (below zero or below margin, red area)
        if not allocation_data['short_positions'].eq(0).all():
            # Stack short positions below margin debt if it exists
            base_level = allocation_data['margin_cash'].clip(upper=0) if not allocation_data['margin_cash'].eq(0).all() else pd.Series(0, index=x_data)
            fig.add_trace(
                go.Scatter(
                    x=x_data,
                    y=base_level + allocation_data['short_positions'],
                    mode='lines',
                    line=dict(width=0),
                    fill='tonexty' if not allocation_data['margin_cash'].eq(0).all() else 'tozeroy',
                    fillcolor=self.styler.COLORS['short_pos'],
                    name="Short Positions",
                    hovertemplate="<b>Short Positions</b><br>Date: %{x}<br>Value: $%{customdata:,.0f}<extra></extra>",
                    customdata=allocation_data['short_positions']
                ),
                row=row, col=col
            )
        
        # 3. Original cash (base layer above zero, forest green area)
        if not allocation_data['original_cash'].eq(0).all():
            fig.add_trace(
                go.Scatter(
                    x=x_data,
                    y=allocation_data['original_cash'],
                    mode='lines',
                    line=dict(width=0),
                    fill='tozeroy',
                    fillcolor=self.styler.COLORS['cash'],
                    name="Original Cash",
                    hovertemplate="<b>Original Cash</b><br>Date: %{x}<br>Value: $%{y:,.0f}<extra></extra>"
                ),
                row=row, col=col
            )
        
        # 4. Short sale proceeds (stacked on original cash, bright green area)
        if not allocation_data['short_proceeds'].eq(0).all():
            stack_base = allocation_data['original_cash']
            fig.add_trace(
                go.Scatter(
                    x=x_data,
                    y=stack_base + allocation_data['short_proceeds'],
                    mode='lines',
                    line=dict(width=0),
                    fill='tonexty',  # Fill to previous trace (original cash)
                    fillcolor=self.styler.COLORS['short_proceeds'],
                    name="Short Sale Proceeds",
                    hovertemplate="<b>Short Sale Proceeds</b><br>Date: %{x}<br>Value: $%{customdata:,.0f}<extra></extra>",
                    customdata=allocation_data['short_proceeds']
                ),
                row=row, col=col
            )
        
        # 5. Long stock positions (stacked on total cash, blue area)
        if not allocation_data['long_positions'].eq(0).all():
            total_cash = allocation_data['original_cash'] + allocation_data['short_proceeds']
            fig.add_trace(
                go.Scatter(
                    x=x_data,
                    y=total_cash + allocation_data['long_positions'],
                    mode='lines',
                    line=dict(width=0),
                    fill='tonexty',  # Fill to previous trace (total cash)
                    fillcolor=self.styler.COLORS['long_pos'],
                    name="Long Positions",
                    hovertemplate="<b>Long Positions</b><br>Date: %{x}<br>Value: $%{customdata:,.0f}<extra></extra>",
                    customdata=allocation_data['long_positions']
                ),
                row=row, col=col
            )
        
        # Add zero reference line
        fig.add_hline(y=0, line_width=1, line_dash="solid", line_color="white", 
                    opacity=0.8, row=row, col=col)
    
    def _add_allocation_summary(self, fig: go.Figure, allocation_data: Dict[str, pd.Series], 
                               row: int, col: int) -> None:
        """Add ending value annotations for allocation components at their actual y-axis positions."""
        # Get latest values
        latest_original_cash = allocation_data['original_cash'].iloc[-1]
        latest_short_proceeds = allocation_data['short_proceeds'].iloc[-1]
        latest_total_cash = latest_original_cash + latest_short_proceeds
        latest_long = allocation_data['long_positions'].iloc[-1]
        latest_short = allocation_data['short_positions'].iloc[-1]  # Keep negative
        latest_margin_debt = allocation_data['margin_cash'].iloc[-1]  # Keep negative
        
        # Calculate y-positions for each component
        # These match how the traces are stacked in _add_allocation_traces
        y_positions = {}
        
        # Margin debt (if negative)
        if latest_margin_debt < 0:
            y_positions['Margin'] = latest_margin_debt
        
        # Short positions (stacked below margin if exists)
        if latest_short < 0:
            base = latest_margin_debt if latest_margin_debt < 0 else 0
            y_positions['Short'] = base + latest_short
        
        # Original cash (starts at 0)
        if latest_original_cash > 0:
            y_positions['Cash'] = latest_original_cash
        
        # Short proceeds (stacked on original cash)
        if latest_short_proceeds > 0:
            y_positions['Proceeds'] = latest_original_cash + latest_short_proceeds
        
        # Long positions (stacked on total cash)
        if latest_long > 0:
            y_positions['Long'] = latest_total_cash + latest_long
        
        # Add y-axis annotations using the same style as other panels
        self._add_ending_value_annotations(fig, row, col, y_positions)
    
    def _add_ending_value_annotations(self, fig: go.Figure, row: int, col: int, values: dict):
        """
        Add ending value annotations at the actual value positions on y-axis.
        
        Args:
            fig: Plotly figure to add annotations to
            row: Row index for the subplot
            col: Column index for the subplot
            values: Dictionary with label: value pairs
        """
        # Generate correct xref and yref for subplot
        if row == 1:
            xref = "x domain"
            yref = "y"
        else:
            xref = f"x{row} domain"
            yref = f"y{row}"
        
        # Define colors for each component
        colors = {
            'Cash': self.styler.COLORS['cash'],
            'Proceeds': self.styler.COLORS['short_proceeds'],
            'Long': self.styler.COLORS['long_pos'],
            'Short': self.styler.COLORS['short_pos'],
            'Margin': self.styler.COLORS['margin']
        }
        
        # Sort values by magnitude for better positioning
        sorted_values = sorted(values.items(), key=lambda x: abs(x[1]), reverse=True)
        
        for i, (label, value) in enumerate(sorted_values):
            # Get color for this component
            color = colors.get(label, '#FFFFFF')
            
            # Format value as currency
            if abs(value) >= 1000000:
                formatted_value = f"${value/1000000:.1f}M"
            elif abs(value) >= 1000:
                formatted_value = f"${value/1000:.0f}K"
            else:
                formatted_value = f"${value:.0f}"
            
            # Add annotation on the right side at actual value position
            fig.add_annotation(
                x=1.002,  # Just outside the plot area
                y=value,  # Actual value position
                xref=xref,
                yref=yref,
                text=formatted_value,
                showarrow=False,
                font=dict(color=color, size=9, family="Arial"),  # Match other panels
                bgcolor="rgba(0, 0, 0, 0.7)",
                bordercolor=color,
                borderwidth=1,
                borderpad=2,
                xanchor="left",
                yanchor="middle"
            )
