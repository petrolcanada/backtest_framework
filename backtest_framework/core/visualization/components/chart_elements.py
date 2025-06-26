"""
Basic chart elements like candlesticks, volume, signals, etc.
"""
from typing import Optional
import pandas as pd
import plotly.graph_objects as go
from .styling import ChartStyler


class ChartElements:
    """Handles basic chart elements like candlesticks, volume, signals."""
    
    def __init__(self, data: pd.DataFrame, results: Optional[pd.DataFrame] = None):
        """
        Initialize chart elements.
        
        Args:
            data: DataFrame with OHLCV data
            results: DataFrame with backtest results
        """
        self.data = data
        self.results = results if results is not None else data
        self.styler = ChartStyler()
    
    def add_candlestick(self, fig: go.Figure, row: int = 1, col: int = 1, add_ending_values: bool = True) -> go.Figure:
        """
        Add price candlesticks to the specified subplot.
        
        Args:
            fig: Plotly figure to add trace to
            row: Row index for the subplot
            col: Column index for the subplot
            add_ending_values: Whether to add ending value annotations
            
        Returns:
            Updated figure
        """
        fig.add_trace(
            go.Candlestick(
                x=self.data.index,
                open=self.data['Open'],
                high=self.data['High'],
                low=self.data['Low'],
                close=self.data['Close'],
                name="Price"
            ),
            row=row, col=col
        )
        
        # Add ending value annotation for closing price
        if add_ending_values:
            final_close = self.data['Close'].iloc[-1]
            # Try adding as a visible scatter trace instead of annotation
            self._add_price_as_trace(fig, row, col, final_close)
        
        return fig
    
    def add_volume(self, fig: go.Figure, row: int = 1, col: int = 1) -> go.Figure:
        """
        Add volume bars to the specified subplot.
        
        Args:
            fig: Plotly figure to add trace to
            row: Row index for the subplot
            col: Column index for the subplot
            
        Returns:
            Updated figure
        """
        if 'Volume' in self.data.columns:
            fig.add_trace(
                go.Bar(
                    x=self.data.index,
                    y=self.data['Volume'],
                    name="Volume",
                    marker=dict(color=self.styler.COLORS['volume'])
                ),
                row=row, col=col
            )
        return fig
    
    def add_dividend_markers(self, fig: go.Figure, row: int = 1, col: int = 1) -> go.Figure:
        """
        Add dividend event markers to the specified subplot.
        
        Args:
            fig: Plotly figure to add trace to
            row: Row index for the subplot
            col: Column index for the subplot
            
        Returns:
            Updated figure
        """
        if 'Dividends' not in self.data.columns:
            return fig
        
        # Find dates with dividend payments
        dividend_dates = self.data[self.data['Dividends'] > 0]
        
        if dividend_dates.empty:
            return fig
        
        # Filter to show only significant dividends (avoid tiny rounding amounts)
        significant_dividends = dividend_dates[dividend_dates['Dividends'] >= 0.01]
        
        if not significant_dividends.empty:
            # Add dividend markers as small diamonds at the bottom of the price chart
            fig.add_trace(
                go.Scatter(
                    x=significant_dividends.index,
                    y=significant_dividends['Low'] * 0.96,  # Position below the low prices
                    mode='markers',
                    marker=self.styler.get_marker_style(
                        symbol='diamond',
                        size=8,
                        color=self.styler.COLORS['dividend'],
                        line_color='#B8860B'  # Dark gold border
                    ),
                    name="Dividend",
                    hovertemplate="<b>Dividend</b><br>Date: %{x}<br>Amount: $%{customdata:.3f}<br>Price: $%{text:.2f}<extra></extra>",
                    customdata=significant_dividends['Dividends'],
                    text=significant_dividends['Close']
                ),
                row=row, col=col
            )
        
        return fig
    
    def add_trade_signals(self, fig: go.Figure, row: int = 1, col: int = 1) -> go.Figure:
        """
        Add buy and sell signals to the specified subplot with T+1 execution timing.
        
        Args:
            fig: Plotly figure to add trace to
            row: Row index for the subplot
            col: Column index for the subplot
            
        Returns:
            Updated figure
        """
        # Use simplified signal structure
        buy_signal_col = 'buy_signal' if 'buy_signal' in self.results.columns else None
        sell_signal_col = 'sell_signal' if 'sell_signal' in self.results.columns else None
        
        # Check for execution tracking columns (new T+1 implementation)
        execution_date_col = 'execution_date' if 'execution_date' in self.results.columns else None
        signal_date_col = 'signal_date' if 'signal_date' in self.results.columns else None
        
        # Add buy signals
        if buy_signal_col:
            self._add_buy_signals(fig, buy_signal_col, execution_date_col, signal_date_col, row, col)
        
        # Add sell signals
        if sell_signal_col:
            self._add_sell_signals(fig, sell_signal_col, execution_date_col, signal_date_col, row, col)
        
        # Add trailing stop level visualization if available
        trailing_stop_col = 'trailing_stop_level'
        if trailing_stop_col in self.results.columns:
            self._add_trailing_stop(fig, trailing_stop_col, row, col)
        
        return fig
    
    def _add_buy_signals(self, fig: go.Figure, buy_signal_col: str, 
                        execution_date_col: Optional[str], signal_date_col: Optional[str],
                        row: int, col: int) -> None:
        """Add buy signals to the chart."""
        if execution_date_col and signal_date_col:  # New T+1 implementation
            buy_executions = self._get_signal_executions(buy_signal_col, execution_date_col, signal_date_col)
            
            if buy_executions:
                execution_dates = [exec_info['execution_date'] for exec_info in buy_executions]
                execution_prices = [
                    self.data.loc[exec_date, 'Open'] if 'Open' in self.data.columns 
                    else self.data.loc[exec_date, 'Close']
                    for exec_date in execution_dates
                ]
                signal_dates = [exec_info['signal_date'] for exec_info in buy_executions]
                
                fig.add_trace(
                    go.Scatter(
                        x=execution_dates,
                        y=[price * 0.98 for price in execution_prices],
                        mode='markers',
                        marker=self.styler.get_marker_style(
                            symbol='triangle-up',
                            size=16,
                            color=self.styler.COLORS['buy_signal']
                        ),
                        name="Buy Execution (T+1)",
                        hovertemplate="<b>Buy Executed</b><br>Signal Date: %{customdata}<br>Execution Date: %{x}<br>Entry Price: $%{text:.2f}<extra></extra>",
                        customdata=signal_dates,
                        text=execution_prices
                    ),
                    row=row, col=col
                )
        else:  # Legacy implementation
            buy_points = self.results[self.results[buy_signal_col] == 1].copy()
            
            if not buy_points.empty:
                actual_entry_prices = self._calculate_actual_entry_prices(buy_points)
                
                fig.add_trace(
                    go.Scatter(
                        x=buy_points.index,
                        y=buy_points['Low'] * 0.98,
                        mode='markers',
                        marker=self.styler.get_marker_style(
                            symbol='triangle-up',
                            size=16,
                            color=self.styler.COLORS['buy_signal']
                        ),
                        name="Buy Signal",
                        hovertemplate="<b>Buy Signal</b><br>Date: %{x}<br>Entry Price: $%{customdata:.2f}<extra></extra>",
                        customdata=actual_entry_prices
                    ),
                    row=row, col=col
                )
    
    def _add_sell_signals(self, fig: go.Figure, sell_signal_col: str,
                         execution_date_col: Optional[str], signal_date_col: Optional[str],
                         row: int, col: int) -> None:
        """Add sell signals to the chart."""
        if execution_date_col and signal_date_col:  # New T+1 implementation
            sell_executions = self._get_signal_executions(sell_signal_col, execution_date_col, signal_date_col)
            
            if sell_executions:
                execution_dates = [exec_info['execution_date'] for exec_info in sell_executions]
                execution_prices = [
                    self.data.loc[exec_date, 'Open'] if 'Open' in self.data.columns 
                    else self.data.loc[exec_date, 'Close']
                    for exec_date in execution_dates
                ]
                signal_dates = [exec_info['signal_date'] for exec_info in sell_executions]
                
                fig.add_trace(
                    go.Scatter(
                        x=execution_dates,
                        y=[price * 1.02 for price in execution_prices],
                        mode='markers',
                        marker=self.styler.get_marker_style(
                            symbol='triangle-down',
                            size=16,
                            color=self.styler.COLORS['sell_signal']
                        ),
                        name="Sell Execution (T+1)",
                        hovertemplate="<b>Sell Executed</b><br>Signal Date: %{customdata}<br>Execution Date: %{x}<br>Entry Price: $%{text:.2f}<extra></extra>",
                        customdata=signal_dates,
                        text=execution_prices
                    ),
                    row=row, col=col
                )
        else:  # Legacy implementation
            sell_points = self.results[self.results[sell_signal_col] == 1].copy()
            
            if not sell_points.empty:
                actual_entry_prices = self._calculate_actual_entry_prices(sell_points)
                
                fig.add_trace(
                    go.Scatter(
                        x=sell_points.index,
                        y=sell_points['High'] * 1.02,
                        mode='markers',
                        marker=self.styler.get_marker_style(
                            symbol='triangle-down',
                            size=16,
                            color=self.styler.COLORS['sell_signal']
                        ),
                        name="Sell Signal",
                        hovertemplate="<b>Sell Signal</b><br>Date: %{x}<br>Entry Price: $%{customdata:.2f}<extra></extra>",
                        customdata=actual_entry_prices
                    ),
                    row=row, col=col
                )
    
    def _add_trailing_stop(self, fig: go.Figure, trailing_stop_col: str, row: int, col: int) -> None:
        """Add trailing stop level visualization."""
        trailing_data = self.results[self.results[trailing_stop_col].notna()]
        if not trailing_data.empty:
            fig.add_trace(
                go.Scatter(
                    x=trailing_data.index,
                    y=trailing_data[trailing_stop_col],
                    mode='lines',
                    line=self.styler.get_line_style(color='#FFA500', width=1, dash='dash'),
                    name="Trailing Stop",
                    opacity=0.7,
                    hovertemplate="<b>Trailing Stop</b><br>Date: %{x}<br>Level: $%{y:.2f}<extra></extra>"
                ),
                row=row, col=col
            )
    
    def _get_signal_executions(self, signal_col: str, execution_date_col: str, signal_date_col: str) -> list:
        """Get signal execution information for T+1 implementation."""
        signal_rows = self.results[
            (self.results[signal_col] == 1) & 
            (self.results[signal_date_col] != '') & 
            (self.results[signal_date_col].notna())
        ].copy()
        
        executions = []
        for signal_idx, signal_row in signal_rows.iterrows():
            signal_date_str = signal_row[signal_date_col]
            execution_rows = self.results[
                (self.results[execution_date_col] != '') & 
                (self.results[signal_date_col] == signal_date_str) &
                (self.results[signal_col] == 1)
            ]
            
            if not execution_rows.empty:
                execution_idx = execution_rows.index[0]
                executions.append({
                    'signal_date': signal_date_str,
                    'execution_date': execution_idx,
                    'signal_idx': signal_idx
                })
        
        return executions
    
    def _calculate_actual_entry_prices(self, signal_points: pd.DataFrame) -> list:
        """Calculate actual entry prices for legacy implementation."""
        actual_entry_prices = []
        for idx in signal_points.index:
            i = self.results.index.get_loc(idx)
            if i < len(self.results) - 1:
                # Next day's open (as used in backtesting)
                next_date = self.results.index[i + 1]
                if next_date in self.data.index:
                    actual_price = self.data.loc[next_date, 'Open']
                else:
                    actual_price = signal_points.loc[idx, 'Close']  # Fallback
            else:
                actual_price = signal_points.loc[idx, 'Close']  # End of data fallback
            actual_entry_prices.append(actual_price)
        
        return actual_entry_prices
    
    def _add_price_ending_annotation(self, fig: go.Figure, row: int, col: int, price: float):
        """
        Add ending price value annotation at the actual price level.
        
        Args:
            fig: Plotly figure to add annotation to
            row: Row index for the subplot
            col: Column index for the subplot
            price: Final closing price to display
        """
        # Format price based on magnitude
        if price >= 1000:
            formatted_price = f"${price:,.0f}"
        elif price >= 100:
            formatted_price = f"${price:.2f}"
        else:
            formatted_price = f"${price:.3f}"
        
        # Use annotation positioned at the actual price level
        # This is the approach that was working before
        fig.add_annotation(
            x=1.01,  # Just outside the plot area
            y=price,  # Actual price value
            xref="x domain" if row == 1 else f"x{row} domain",
            yref="y" if row == 1 else f"y{row}",
            text=formatted_price,
            showarrow=False,
            font=dict(color="#FFFFFF", size=11, family="Arial", weight="bold"),
            bgcolor="rgba(0, 0, 0, 0.8)",
            bordercolor="#FFFFFF",
            borderwidth=1,
            borderpad=3,
            xanchor="left",
            yanchor="middle"
        )
        
        print(f"Added price annotation: {formatted_price} at y={price:.2f}")
    
    def _add_price_as_trace(self, fig: go.Figure, row: int, col: int, price: float):
        """
        Add price as a scatter trace positioned within visible chart area.
        
        Args:
            fig: Plotly figure to add trace to
            row: Row index for the subplot
            col: Column index for the subplot
            price: Final closing price to display
        """
        # Format price
        if price >= 1000:
            formatted_price = f"${price:,.0f}"
        elif price >= 100:
            formatted_price = f"${price:.2f}"
        else:
            formatted_price = f"${price:.3f}"
        
        # Position the text within the visible chart area
        # Get the date range and position text at ~95% of the way across
        date_range = self.data.index[-1] - self.data.index[0]
        position_date = self.data.index[-1] - (date_range * 0.05)  # 5% from the right edge
        
        # Add as scatter trace with text positioned inside chart
        fig.add_trace(
            go.Scatter(
                x=[position_date],
                y=[price],
                mode='text',  # Only text, no marker
                text=[formatted_price],
                textposition='middle center',
                textfont=dict(
                    color='#FFFFFF',
                    size=12,
                    family='Arial',
                    weight='bold'
                ),
                name='Price Label',
                showlegend=False,
                hoverinfo='skip'
            ),
            row=row, col=col
        )
        
        # Also add a subtle line from the text to the actual price level at the end
        fig.add_trace(
            go.Scatter(
                x=[position_date, self.data.index[-1]],
                y=[price, price],
                mode='lines',
                line=dict(
                    color='#FFFFFF',
                    width=1,
                    dash='dot'
                ),
                name='Price Level',
                showlegend=False,
                hoverinfo='skip'
            ),
            row=row, col=col
        )
        
        print(f"Added price label: {formatted_price} at y={price:.2f} positioned at {position_date}")
    
    def _add_ending_value_annotations(self, fig: go.Figure, row: int, col: int, values: dict):
        """
        Add ending value annotations at the actual value positions.
        
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
        
        # Sort values by magnitude for better positioning
        sorted_values = sorted(values.items(), key=lambda x: x[1], reverse=True)
        
        for i, (label, value) in enumerate(sorted_values):
            # Choose color and formatting based on label
            if 'Price' in label:
                color = '#FFFFFF'
                # Format as price
                if value >= 1000:
                    formatted_value = f"${value:,.0f}"
                elif value >= 100:
                    formatted_value = f"${value:.2f}"
                else:
                    formatted_value = f"${value:.3f}"
            else:
                color = '#FFFFFF'
                # Format as number
                if value >= 1000:
                    formatted_value = f"{value:.0f}"
                elif value >= 100:
                    formatted_value = f"{value:.1f}"
                else:
                    formatted_value = f"{value:.2f}"
            
            # Apply small vertical offset if values are too close
            y_offset = i * 2 if len(sorted_values) > 1 and abs(sorted_values[0][1] - sorted_values[-1][1]) < 10 else 0
            
            print(f"Adding price annotation: {formatted_value} at y={value + y_offset}, xref={xref}, yref={yref}")
            
            # Add annotation on the right side at actual value position
            fig.add_annotation(
                x=1.002,  # Just outside the plot area
                y=value + y_offset,  # Actual value position with offset if needed
                xref=xref,
                yref=yref,
                text=formatted_value,
                showarrow=False,
                font=dict(color=color, size=12, family="Arial", weight="bold"),
                bgcolor="rgba(0, 0, 0, 0.8)",
                bordercolor=color,
                borderwidth=1,
                borderpad=3,
                xanchor="left",
                yanchor="middle"
            )
