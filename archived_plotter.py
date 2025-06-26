"""
Plotting module for visualizing prices, indicators, and backtest results.
"""
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import the new modular components
from .components import (
    TitleGenerator,
    ChartStyler,
    ChartElements,
    IndicatorPlots,
    PerformancePlots,
    AllocationPlots
)

class Plotter:
    """
    Creates interactive visualizations for backtest results and technical indicators.
    Module is organized into small, reusable components that are combined for different visualizations.
    """
    
    def __init__(self, data: pd.DataFrame, results: Optional[pd.DataFrame] = None, engine: Optional = None):
        """
        Initialize the plotter with data and optional backtest results.
        
        Args:
            data: DataFrame with OHLCV data and indicators
            results: DataFrame with backtest results, if None, assumes data contains results
            engine: BacktestEngine instance to read configuration for dynamic titles
        """
        self.data = data
        self.results = results if results is not None else data
        self.engine = engine
        self.fig = None
        
        # Initialize component modules
        self.title_gen = TitleGenerator(self.data, self.results, self.engine)
        self.styler = ChartStyler()
        self.chart_elements = ChartElements(self.data, self.results)
        self.indicators = IndicatorPlots(self.data)
        self.performance = PerformancePlots(self.data, self.results, self.engine)
        self.allocation = AllocationPlots(self.results)
    
    def _generate_dynamic_title(self, ticker: str, base_strategy_name: str = "Strategy") -> str:
        """Generate dynamic chart title based on engine configuration (delegates to TitleGenerator)."""
        return self.title_gen.generate_main_title(ticker, base_strategy_name)
    
    def _generate_costs_subtitle(self) -> str:
        """Generate subtitle with commission, slippage, and dividend information (delegates to TitleGenerator)."""
        return self.title_gen.generate_costs_subtitle()
    
    def _generate_performance_subtitle(self) -> str:
        """Generate subtitle with performance metrics summary (delegates to TitleGenerator)."""
        return self.title_gen.generate_performance_subtitle()
        
    def _get_column_names(self, column_patterns: Dict[str, List[str]]) -> Dict[str, str]:
        """
        Helper method to find the actual column names that match the expected patterns.
        Handles both snake_case and legacy naming conventions.
        
        Args:
            column_patterns: Dictionary mapping logical names to potential column name patterns
            
        Returns:
            Dictionary mapping logical names to actual column names found in data/results
        """
        found_columns = {}
        
        for logical_name, possible_names in column_patterns.items():
            for name in possible_names:
                if name in self.data.columns:
                    found_columns[logical_name] = name
                    break
                elif self.results is not None and name in self.results.columns:
                    found_columns[logical_name] = name
                    break
                    
        return found_columns
    
    def _add_candlestick(self, fig, row=1, col=1):
        """Add price candlesticks (delegates to ChartElements)."""
        return self.chart_elements.add_candlestick(fig, row, col)
    
    def _add_dividend_markers(self, fig, row=1, col=1):
        """Add dividend event markers (delegates to ChartElements)."""
        return self.chart_elements.add_dividend_markers(fig, row, col)
    
    def _add_volume(self, fig, row=1, col=1):
        """Add volume bars (delegates to ChartElements)."""
        return self.chart_elements.add_volume(fig, row, col)
    
    def _add_trade_signals(self, fig, row=1, col=1):
        """Add buy and sell signals (delegates to ChartElements)."""
        return self.chart_elements.add_trade_signals(fig, row, col)
    
    def _add_kdj_indicators(self, fig, row=1, col=1, is_monthly=False):
        """Add KDJ indicator lines (delegates to IndicatorPlots)."""
        return self.indicators.add_kdj_indicators(fig, row, col, is_monthly)
    
    def _add_capital_allocation(self, fig, row=1, col=1):
        """Add capital allocation stacked area chart (delegates to AllocationPlots)."""
        return self.allocation.add_capital_allocation(fig, row, col)
    

    

    

    
    def _add_capital_allocation(self, fig, row=1, col=1):
        """
        Add capital allocation stacked area chart to the specified subplot.
        Shows cash, long positions, short positions, and margin usage over time.
        
        Args:
            fig: Plotly figure to add trace to
            row: Row index for the subplot
            col: Column index for the subplot
            
        Returns:
            Updated figure, and boolean indicating if allocation data was found
        """
        # Define expected column names for capital allocation
        allocation_cols = {
            'cash': ['cash', 'Cash', 'available_cash'],
            'long_positions': ['long_positions', 'Long_Positions', 'long_value', 'stock_value'],
            'short_positions': ['short_positions', 'Short_Positions', 'short_value'],
            'margin_used': ['margin_used', 'Margin_Used', 'borrowed_amount']
        }
        
        # Find available columns
        found_cols = {}
        for category, possible_names in allocation_cols.items():
            for name in possible_names:
                if name in self.results.columns:
                    found_cols[category] = name
                    break
        
        # Check if we have at least cash and long positions
        if 'cash' not in found_cols and 'long_positions' not in found_cols:
            return fig, False
        
        # Get the data for each component
        x_data = self.results.index
        
        # Cash (base layer, always positive)
        cash_data = self.results[found_cols['cash']] if 'cash' in found_cols else pd.Series(0, index=x_data)
        
        # Long positions (stacked on top of cash)
        long_data = self.results[found_cols['long_positions']] if 'long_positions' in found_cols else pd.Series(0, index=x_data)
        
        # Short positions (below zero line, negative values)
        short_data = self.results[found_cols['short_positions']] if 'short_positions' in found_cols else pd.Series(0, index=x_data)
        # Make sure short positions are negative for proper display
        short_data = -abs(short_data)
        
        # Margin used (top layer, if applicable)
        margin_data = self.results[found_cols['margin_used']] if 'margin_used' in found_cols else pd.Series(0, index=x_data)
        
        # Create stacked areas
        
        # 1. Short positions (below zero, red area)
        if not short_data.eq(0).all():
            fig.add_trace(
                go.Scatter(
                    x=x_data,
                    y=short_data,
                    mode='lines',
                    line=dict(width=0),
                    fill='tozeroy',
                    fillcolor='rgba(220, 20, 60, 0.6)',  # Crimson with transparency
                    name="Short Positions",
                    hovertemplate="<b>Short Positions</b><br>Date: %{x}<br>Value: $%{y:,.0f}<extra></extra>"
                ),
                row=row, col=col
            )
        
        # 2. Cash (base layer above zero, green area)
        if not cash_data.eq(0).all():
            fig.add_trace(
                go.Scatter(
                    x=x_data,
                    y=cash_data,
                    mode='lines',
                    line=dict(width=0),
                    fill='tozeroy',
                    fillcolor='rgba(34, 139, 34, 0.6)',  # Forest green with transparency
                    name="Cash",
                    hovertemplate="<b>Cash</b><br>Date: %{x}<br>Value: $%{y:,.0f}<extra></extra>"
                ),
                row=row, col=col
            )
        
        # 3. Long stock positions (stacked on cash, blue area)
        if not long_data.eq(0).all():
            stack_base = cash_data
            fig.add_trace(
                go.Scatter(
                    x=x_data,
                    y=stack_base + long_data,
                    mode='lines',
                    line=dict(width=0),
                    fill='tonexty',  # Fill to previous trace (cash)
                    fillcolor='rgba(70, 130, 180, 0.6)',  # Steel blue with transparency
                    name="Long Positions",
                    hovertemplate="<b>Long Positions</b><br>Date: %{x}<br>Value: $%{customdata:,.0f}<extra></extra>",
                    customdata=long_data
                ),
                row=row, col=col
            )
            
            # 4. Margin used (top layer, orange/red area)
            if not margin_data.eq(0).all():
                stack_base = cash_data + long_data
                fig.add_trace(
                    go.Scatter(
                        x=x_data,
                        y=stack_base + margin_data,
                        mode='lines',
                        line=dict(width=0),
                        fill='tonexty',  # Fill to previous trace (long positions)
                        fillcolor='rgba(255, 165, 0, 0.6)',  # Orange with transparency
                        name="Margin Used",
                        hovertemplate="<b>Margin Used</b><br>Date: %{x}<br>Value: $%{customdata:,.0f}<extra></extra>",
                        customdata=margin_data
                    ),
                    row=row, col=col
                )
        else:
            # If no long positions, margin goes directly on cash
            if not margin_data.eq(0).all():
                stack_base = cash_data
                fig.add_trace(
                    go.Scatter(
                        x=x_data,
                        y=stack_base + margin_data,
                        mode='lines',
                        line=dict(width=0),
                        fill='tonexty',  # Fill to previous trace (cash)
                        fillcolor='rgba(255, 165, 0, 0.6)',  # Orange with transparency
                        name="Margin Used",
                        hovertemplate="<b>Margin Used</b><br>Date: %{x}<br>Value: $%{customdata:,.0f}<extra></extra>",
                        customdata=margin_data
                    ),
                    row=row, col=col
                )
        
        # Add zero reference line
        fig.add_hline(y=0, line_width=1, line_dash="solid", line_color="white", 
                    opacity=0.8, row=row, col=col)
        
        # Add annotations for key metrics if we have the data
        if 'cash' in found_cols and 'long_positions' in found_cols:
            latest_cash = cash_data.iloc[-1]
            latest_long = long_data.iloc[-1]
            latest_short = abs(short_data.iloc[-1]) if not short_data.eq(0).all() else 0
            latest_margin = margin_data.iloc[-1] if not margin_data.eq(0).all() else 0
            
            total_assets = latest_cash + latest_long + latest_margin
            total_exposure = latest_long + latest_short
            
            # Calculate key ratios
            cash_pct = (latest_cash / total_assets * 100) if total_assets > 0 else 0
            leverage_ratio = (total_exposure / (total_assets - latest_margin)) if (total_assets - latest_margin) > 0 else 0
            
            # Add summary annotation
            annotation_text = f"Cash: {cash_pct:.1f}%<br>"
            if latest_long > 0:
                long_pct = (latest_long / total_assets * 100) if total_assets > 0 else 0
                annotation_text += f"Long: {long_pct:.1f}%<br>"
            if latest_short > 0:
                short_pct = (latest_short / total_assets * 100) if total_assets > 0 else 0
                annotation_text += f"Short: {short_pct:.1f}%<br>"
            if latest_margin > 0:
                margin_pct = (latest_margin / total_assets * 100) if total_assets > 0 else 0
                annotation_text += f"Margin: {margin_pct:.1f}%<br>"
            annotation_text += f"Leverage: {leverage_ratio:.2f}x"
            
            # Calculate correct axis reference for subplot
            # In Plotly subplots: row 1 = axis1, row 2 = axis2, etc.
            axis_num = row if row > 1 else ""
            xref = f"x{axis_num} domain" if axis_num else "x domain"
            yref = f"y{axis_num} domain" if axis_num else "y domain"
            
            fig.add_annotation(
                x=0.02,
                y=0.98,
                xref=xref,
                yref=yref,
                text=annotation_text,
                showarrow=False,
                font=dict(family="Arial", size=11, color="white"),
                align="left",
                bgcolor="rgba(50, 50, 50, 0.8)",
                bordercolor="#999999",
                borderwidth=1,
                borderpad=4
            )
        
        return fig, True
    
    def _add_trade_signals(self, fig, row=1, col=1):
        """
        Add buy and sell signals to the specified subplot with T+1 execution timing.
        Shows signals on execution day (T+1) rather than signal generation day (T+0).
        
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
        
        # Check for risk management signals
        drawdown_exit_col = 'drawdown_since_entry' if 'drawdown_since_entry' in self.results.columns else None
        trailing_stop_col = 'trailing_stop_level' if 'trailing_stop_level' in self.results.columns else None
        
        # Add buy signals - show on execution dates for T+1 implementation
        if buy_signal_col:
            if execution_date_col and signal_date_col:  # New T+1 implementation
                # FIXED: Find signal generation dates, then find their corresponding execution dates
                signal_rows = self.results[
                    (self.results[buy_signal_col] == 1) & 
                    (self.results[signal_date_col] != '') & 
                    (self.results[signal_date_col].notna())
                ].copy()
                
                buy_executions = []
                for signal_idx, signal_row in signal_rows.iterrows():
                    signal_date_str = signal_row[signal_date_col]
                    # Find the execution row for this signal
                    execution_rows = self.results[
                        (self.results[execution_date_col] != '') & 
                        (self.results[signal_date_col] == signal_date_str) &
                        (self.results[buy_signal_col] == 1)
                    ]
                    
                    if not execution_rows.empty:
                        # Take the first execution (should only be one)
                        execution_idx = execution_rows.index[0]
                        buy_executions.append({
                            'signal_date': signal_date_str,
                            'execution_date': execution_idx,
                            'signal_idx': signal_idx
                        })
                
                if buy_executions:
                    print(f"DEBUG: Found {len(buy_executions)} buy executions for visualization")
                    
                    execution_dates = []
                    execution_prices = []
                    signal_dates = []
                    
                    for exec_info in buy_executions:
                        exec_date = exec_info['execution_date']
                        execution_dates.append(exec_date)
                        execution_prices.append(self.data.loc[exec_date, 'Open'] if 'Open' in self.data.columns else self.data.loc[exec_date, 'Close'])
                        signal_dates.append(exec_info['signal_date'])
                    
                    fig.add_trace(
                        go.Scatter(
                            x=execution_dates,
                            y=[price * 0.98 for price in execution_prices],  # Position below price
                            mode='markers',
                            marker=dict(
                                symbol='triangle-up', 
                                size=16,
                                color='#00FF7F',  # SpringGreen
                                line=dict(width=2, color='#FFFFFF'),
                                opacity=1.0
                            ),
                            name="Buy Execution (T+1)",
                            hovertemplate="<b>Buy Executed</b><br>Signal Date: %{customdata}<br>Execution Date: %{x}<br>Entry Price: $%{text:.2f}<extra></extra>",
                            customdata=signal_dates,
                            text=execution_prices
                        ),
                        row=row, col=col
                    )
                else:
                    print("DEBUG: No buy executions found for visualization")
            else:  # Legacy implementation - fallback
                buy_points = self.results[self.results[buy_signal_col] == 1].copy()
                
                if not buy_points.empty:
                    # Calculate actual entry prices used in backtesting
                    actual_entry_prices = []
                    for idx in buy_points.index:
                        i = self.results.index.get_loc(idx)
                        if i < len(self.results) - 1:
                            # Next day's open (as used in backtesting)
                            next_date = self.results.index[i + 1]
                            if next_date in self.data.index:
                                actual_price = self.data.loc[next_date, 'Open']
                            else:
                                actual_price = buy_points.loc[idx, 'Close']  # Fallback
                        else:
                            actual_price = buy_points.loc[idx, 'Close']  # End of data fallback
                        actual_entry_prices.append(actual_price)
                    
                    fig.add_trace(
                        go.Scatter(
                            x=buy_points.index,
                            y=buy_points['Low'] * 0.98,
                            mode='markers',
                            marker=dict(
                                symbol='triangle-up', 
                                size=16,
                                color='#00FF7F',  # SpringGreen
                                line=dict(width=2, color='#FFFFFF'),
                                opacity=1.0
                            ),
                            name="Buy Signal",
                            hovertemplate="<b>Buy Signal</b><br>Date: %{x}<br>Entry Price: $%{customdata:.2f}<extra></extra>",
                            customdata=actual_entry_prices
                        ),
                        row=row, col=col
                    )
        
        # Add sell signals - show on execution dates for T+1 implementation
        if sell_signal_col:
            if execution_date_col and signal_date_col:  # New T+1 implementation
                # FIXED: Find signal generation dates, then find their corresponding execution dates
                signal_rows = self.results[
                    (self.results[sell_signal_col] == 1) & 
                    (self.results[signal_date_col] != '') & 
                    (self.results[signal_date_col].notna())
                ].copy()
                
                sell_executions = []
                for signal_idx, signal_row in signal_rows.iterrows():
                    signal_date_str = signal_row[signal_date_col]
                    # Find the execution row for this signal
                    execution_rows = self.results[
                        (self.results[execution_date_col] != '') & 
                        (self.results[signal_date_col] == signal_date_str) &
                        (self.results[sell_signal_col] == 1)
                    ]
                    
                    if not execution_rows.empty:
                        # Take the first execution (should only be one)
                        execution_idx = execution_rows.index[0]
                        sell_executions.append({
                            'signal_date': signal_date_str,
                            'execution_date': execution_idx,
                            'signal_idx': signal_idx
                        })
                
                if sell_executions:
                    print(f"DEBUG: Found {len(sell_executions)} sell executions for visualization")
                    
                    execution_dates = []
                    execution_prices = []
                    signal_dates = []
                    
                    for exec_info in sell_executions:
                        exec_date = exec_info['execution_date']
                        execution_dates.append(exec_date)
                        execution_prices.append(self.data.loc[exec_date, 'Open'] if 'Open' in self.data.columns else self.data.loc[exec_date, 'Close'])
                        signal_dates.append(exec_info['signal_date'])
                    
                    fig.add_trace(
                        go.Scatter(
                            x=execution_dates,
                            y=[price * 1.02 for price in execution_prices],  # Position above price
                            mode='markers',
                            marker=dict(
                                symbol='triangle-down', 
                                size=16,
                                color='#FF3030',  # Firebrick red
                                line=dict(width=2, color='#FFFFFF'),
                                opacity=1.0
                            ),
                            name="Sell Execution (T+1)",
                            hovertemplate="<b>Sell Executed</b><br>Signal Date: %{customdata}<br>Execution Date: %{x}<br>Entry Price: $%{text:.2f}<extra></extra>",
                            customdata=signal_dates,
                            text=execution_prices
                        ),
                        row=row, col=col
                    )
                else:
                    print("DEBUG: No sell executions found for visualization")
            else:  # Legacy implementation - fallback
                sell_points = self.results[self.results[sell_signal_col] == 1].copy()
                
                if not sell_points.empty:
                    # Calculate actual entry prices used in backtesting (next day open)
                    actual_entry_prices = []
                    for idx in sell_points.index:
                        i = self.results.index.get_loc(idx)
                        if i < len(self.results) - 1:
                            # Next day's open (as used in backtesting)
                            next_date = self.results.index[i + 1]
                            if next_date in self.data.index:
                                actual_price = self.data.loc[next_date, 'Open']
                            else:
                                actual_price = sell_points.loc[idx, 'Close']  # Fallback
                        else:
                            actual_price = sell_points.loc[idx, 'Close']  # End of data fallback
                        actual_entry_prices.append(actual_price)
                    
                    fig.add_trace(
                        go.Scatter(
                            x=sell_points.index,
                            y=sell_points['High'] * 1.02,
                            mode='markers',
                            marker=dict(
                                symbol='triangle-down', 
                                size=16,
                                color='#FF3030',
                                line=dict(width=2, color='#FFFFFF'),
                                opacity=1.0
                            ),
                            name="Sell Signal",
                            hovertemplate="<b>Sell Signal</b><br>Date: %{x}<br>Entry Price: $%{customdata:.2f}<extra></extra>",
                            customdata=actual_entry_prices
                        ),
                        row=row, col=col
                    )
        
        # Add trailing stop level visualization if available
        if trailing_stop_col and trailing_stop_col in self.results.columns:
            trailing_data = self.results[self.results[trailing_stop_col].notna()]
            if not trailing_data.empty:
                fig.add_trace(
                    go.Scatter(
                        x=trailing_data.index,
                        y=trailing_data[trailing_stop_col],
                        mode='lines',
                        line=dict(width=1, color='#FFA500', dash='dash'),  # Orange dashed line
                        name="Trailing Stop",
                        opacity=0.7,
                        hovertemplate="<b>Trailing Stop</b><br>Date: %{x}<br>Level: $%{y:.2f}<extra></extra>"
                    ),
                    row=row, col=col
                )
        
        return fig
    
    def _calculate_performance_metrics(self):
        """
        Calculate key performance metrics from the results DataFrame.
        
        Returns:
            Dictionary with performance metrics
        """
        metrics = {}
        
        # Get column names (using only snake_case going forward)
        equity_col = 'equity' if 'equity' in self.results.columns else None
        
        if equity_col:
            # Calculate main performance metrics
            initial_equity = self.results[equity_col].iloc[0]
            final_equity = self.results[equity_col].iloc[-1]
            total_return = (final_equity / initial_equity) - 1
            
            # Store metrics
            metrics['total_return'] = total_return
            metrics['initial_equity'] = initial_equity
            metrics['final_equity'] = final_equity
            
            # Get other metrics if available
            for metric in ['cagr', 'sharpe_ratio', 'sortino_ratio', 'max_drawdown', 'win_rate']:
                if metric in self.results.columns:
                    metrics[metric] = self.results[metric].iloc[-1]
        
        return metrics
    
    def plot_price_and_signals(self, ticker: str = '', panel_heights: List[float] = None, 
                            show_volume: bool = True) -> go.Figure:
        """
        Create a plot with price, indicators, and signals.
        
        Args:
            ticker: Ticker symbol for chart title
            panel_heights: List of relative heights for each panel
            show_volume: Whether to include volume in the plot
            
        Returns:
            Plotly Figure object
        """
        # Determine which panels to include (only using snake_case going forward)
        has_daily_kdj = all(col in self.data.columns for col in ['k', 'd', 'j'])
        has_monthly_kdj = all(col in self.data.columns for col in ['monthly_k', 'monthly_d', 'monthly_j'])
        
        # Calculate number of rows
        num_rows = 1  # Price panel
        if has_daily_kdj:
            num_rows += 1
        if has_monthly_kdj:
            num_rows += 1
        if show_volume and 'Volume' in self.data.columns:
            num_rows += 1
            
        # Default panel heights
        if panel_heights is None:
            # Default to price: 40%, indicators: 20% each, volume: 20%
            panel_heights = [0.4] + [0.2] * (num_rows - 1)
        
        # Create subplot titles
        subplot_titles = [f"{ticker} Price"]
        if has_daily_kdj:
            subplot_titles.append("Daily KDJ")
        if has_monthly_kdj:
            subplot_titles.append("Monthly KDJ")
        if show_volume and 'Volume' in self.data.columns:
            subplot_titles.append("Volume")
        
        # Create subplots
        fig = make_subplots(
            rows=num_rows, 
            cols=1, 
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=panel_heights,
            subplot_titles=subplot_titles
        )
        
        # Add price candlesticks to the first panel
        fig.add_trace(
            go.Candlestick(
                x=self.data.index,
                open=self.data['Open'],
                high=self.data['High'],
                low=self.data['Low'],
                close=self.data['Close'],
                name="Price"
            ),
            row=1, col=1
        )
        
        # Add SMA if available
        if 'SMA' in self.data.columns:
            fig.add_trace(
                go.Scatter(
                    x=self.data.index,
                    y=self.data['SMA'],
                    mode='lines',
                    line=dict(width=1.5, color='blue'),
                    name="SMA"
                ),
                row=1, col=1
            )
        
        # Check for signal column names (simplified structure)
        buy_signal_col = 'buy_signal' if 'buy_signal' in self.results.columns else None
        sell_signal_col = 'sell_signal' if 'sell_signal' in self.results.columns else None
        
        # Add signals if they exist
        if buy_signal_col:
            buy_points = self.results[self.results[buy_signal_col] == 1]
            if not buy_points.empty:
                fig.add_trace(
                    go.Scatter(
                        x=buy_points.index,
                        y=buy_points['Low'] * 0.99,  # Slight offset for visibility
                        mode='markers',
                        marker=dict(symbol='triangle-up', size=10, color='green'),
                        name="Buy Signal"
                    ),
                    row=1, col=1
                )
        
        if sell_signal_col:
            sell_points = self.results[self.results[sell_signal_col] == 1]
            if not sell_points.empty:
                fig.add_trace(
                    go.Scatter(
                        x=sell_points.index,
                        y=sell_points['High'] * 1.01,  # Slight offset for visibility
                        mode='markers',
                        marker=dict(symbol='triangle-down', size=10, color='red'),
                        name="Sell Signal"
                    ),
                    row=1, col=1
                )
        
        # Add equity curve overlay (only using snake_case going forward)
        equity_col = 'equity' if 'equity' in self.results.columns else None
            
        if equity_col:
            # Normalize equity to match price scale for visibility
            scale_factor = self.data['Close'].iloc[0] / self.results[equity_col].iloc[0]
            normalized_equity = self.results[equity_col] * scale_factor
            
            fig.add_trace(
                go.Scatter(
                    x=self.results.index,
                    y=normalized_equity,
                    mode='lines',
                    line=dict(width=1.5, color='purple'),
                    name="Equity (Scaled)",
                    opacity=0.7
                ),
                row=1, col=1
            )
        
        # Current row tracker
        current_row = 2
        
        # Add Daily KDJ indicators if available
        if has_daily_kdj:
            # Add K line
            fig.add_trace(
                go.Scatter(
                    x=self.data.index,
                    y=self.data['k'],
                    mode='lines',
                    line=dict(width=1.5, color='blue'),
                    name="K Line"
                ),
                row=current_row, col=1
            )
            
            # Add D line
            fig.add_trace(
                go.Scatter(
                    x=self.data.index,
                    y=self.data['d'],
                    mode='lines',
                    line=dict(width=1.5, color='red'),
                    name="D Line"
                ),
                row=current_row, col=1
            )
            
            # Add J line
            fig.add_trace(
                go.Scatter(
                    x=self.data.index,
                    y=self.data['j'],
                    mode='lines',
                    line=dict(width=1.5, color='green'),
                    name="J Line"
                ),
                row=current_row, col=1
            )
            
            # Add horizontal reference lines
            fig.add_hline(y=80, line_width=1, line_dash="dash", line_color="gray", 
                        annotation_text="Overbought", row=current_row, col=1)
            fig.add_hline(y=20, line_width=1, line_dash="dash", line_color="gray", 
                        annotation_text="Oversold", row=current_row, col=1)
            
            # Move to next row
            current_row += 1
        
        # Add Monthly KDJ indicators if available
        if has_monthly_kdj:
            # Add Monthly K line
            fig.add_trace(
                go.Scatter(
                    x=self.data.index,
                    y=self.data['monthly_k'],
                    mode='lines',
                    line=dict(width=1.5, color='blue'),
                    name="Monthly K"
                ),
                row=current_row, col=1
            )
            
            # Add Monthly D line
            fig.add_trace(
                go.Scatter(
                    x=self.data.index,
                    y=self.data['monthly_d'],
                    mode='lines',
                    line=dict(width=1.5, color='red'),
                    name="Monthly D"
                ),
                row=current_row, col=1
            )
            
            # Add Monthly J line
            fig.add_trace(
                go.Scatter(
                    x=self.data.index,
                    y=self.data['monthly_j'],
                    mode='lines',
                    line=dict(width=2, color='green'),
                    name="Monthly J"
                ),
                row=current_row, col=1
            )
            
            # Add horizontal reference lines
            fig.add_hline(y=80, line_width=1, line_dash="dash", line_color="gray", 
                        annotation_text="Overbought", row=current_row, col=1)
            fig.add_hline(y=20, line_width=1, line_dash="dash", line_color="gray", 
                        annotation_text="Oversold", row=current_row, col=1)
            
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
                        marker=dict(symbol='star', size=10, color='gold'),
                        name="Golden Cross"
                    ),
                    row=current_row, col=1
                )
            
            # Add markers for dead crosses
            if dead_crosses:
                j_values = [self.data.loc[date, 'monthly_j'] for date in dead_crosses]
                fig.add_trace(
                    go.Scatter(
                        x=dead_crosses,
                        y=j_values,
                        mode='markers',
                        marker=dict(symbol='x', size=10, color='black'),
                        name="Dead Cross"
                    ),
                    row=current_row, col=1
                )
            
            # Move to next row
            current_row += 1
        
        # Add volume if requested and available
        if show_volume and 'Volume' in self.data.columns:
            fig.add_trace(
                go.Bar(
                    x=self.data.index,
                    y=self.data['Volume'],
                    name="Volume",
                    marker=dict(color='rgba(0, 0, 255, 0.5)')
                ),
                row=current_row, col=1
            )
        
        # Update layout
        fig.update_layout(
            title=f"{ticker} Trading Strategy Backtest" if ticker else "Trading Strategy Backtest",
            xaxis_title="Date",
            yaxis_title="Price",
            height=200 * num_rows,  # Dynamic height based on number of panels
            width=1200,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis_rangeslider_visible=False,
        )
        
        # Update y-axis labels for KDJ panels
        if has_daily_kdj:
            fig.update_yaxes(title_text="KDJ Value", row=2, col=1)
        if has_monthly_kdj:
            row_index = 3 if has_daily_kdj else 2
            fig.update_yaxes(title_text="Monthly KDJ Value", row=row_index, col=1)
        
        self.fig = fig
        return fig
    
    def plot_equity_curve(self, title: str = "Portfolio Equity Curve") -> go.Figure:
        """
        Create a standalone equity curve plot.
        
        Args:
            title: Plot title
            
        Returns:
            Plotly Figure object
        """
        # Check for snake_case or old naming
        if 'equity' in self.results.columns:
            equity_col = 'equity'
            returns_col = 'returns'
            drawdown_col = 'drawdown'
        else:
            equity_col = 'Equity'
            returns_col = 'Returns'
            drawdown_col = 'Drawdown'
        
        if equity_col not in self.results.columns:
            raise ValueError(f"{equity_col} column not found in results DataFrame")
        
        # Create subplots if drawdown is available
        if drawdown_col in self.results.columns:
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                row_heights=[0.7, 0.3],
                subplot_titles=["Equity Curve", "Drawdown"]
            )
            
            # Add equity curve to first subplot
            fig.add_trace(
                go.Scatter(
                    x=self.results.index,
                    y=self.results[equity_col],
                    mode='lines',
                    line=dict(width=2, color='blue'),
                    name="Equity"
                ),
                row=1, col=1
            )
            
            # Add drawdown to second subplot
            fig.add_trace(
                go.Scatter(
                    x=self.results.index,
                    y=self.results[drawdown_col] * 100,  # Convert to percentage
                    mode='lines',
                    line=dict(width=1.5, color='red'),
                    fill='tozeroy',
                    name="Drawdown %"
                ),
                row=2, col=1
            )
            
            # Update y-axes
            fig.update_yaxes(title_text="Equity ($)", row=1, col=1)
            fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
            fig.update_xaxes(title_text="Date", row=2, col=1)
            
        else:
            # Single plot without drawdown
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=self.results.index,
                    y=self.results[equity_col],
                    mode='lines',
                    line=dict(width=2, color='blue'),
                    name="Equity"
                )
            )
            fig.update_xaxis(title_text="Date")
            fig.update_yaxis(title_text="Equity ($)")
        
        # Update layout
        fig.update_layout(
            title=title,
            height=600,
            width=1000,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        return fig
        
    def plot_chart_with_benchmark(self, ticker: str = '', base_strategy_name: str = "Monthly KDJ", log_scale: bool = True) -> go.Figure:
        """
        Create a dark-themed plot with price, strategy equity, and benchmark (long-hold) performance.
        
        Args:
            ticker: Ticker symbol for chart title
            base_strategy_name: Base strategy name for dynamic title generation
            log_scale: Whether to use logarithmic scale for price and performance charts
            
        Returns:
            Plotly Figure object
        """
        # Get column names (using only snake_case going forward)
        equity_col = 'equity' if 'equity' in self.results.columns else None
        
        # Get performance metrics for display
        metrics = {}
        if equity_col:
            # Basic metrics
            initial_equity = self.results[equity_col].iloc[0]
            final_equity = self.results[equity_col].iloc[-1]
            total_return = (final_equity / initial_equity) - 1
            metrics['initial_capital'] = initial_equity
            metrics['final_equity'] = final_equity
            metrics['total_return'] = total_return
            
            # Additional metrics from the results dataframe
            for key in ['cagr', 'sharpe_ratio', 'max_drawdown', 'win_rate', 'trade_count']:
                if key in self.results.columns:
                    metrics[key] = self.results[key].iloc[-1]
        
        # First create a figure for the main charts with separate panels for returns, drawdowns, indicators, and capital allocation
        fig = make_subplots(
            rows=5, 
            cols=1, 
            shared_xaxes=True,
            vertical_spacing=0.04,  # Increased spacing to provide room for titles and legend
            row_heights=[0.35, 0.15, 0.15, 0.15, 0.2],  # Allocate space for each panel (price, returns, drawdowns, indicators, capital)
            subplot_titles=[
                "",  # Empty title for price chart (will add title after summary)
                "Strategy vs Benchmark Performance", 
                "Drawdowns",
                "Technical Indicators",
                "Capital Allocation"
            ],
            specs=[
                [{"type": "candlestick"}],
                [{"type": "xy"}],  # Returns panel
                [{"type": "xy"}],  # Separate drawdowns panel
                [{"type": "xy"}],   # Indicators panel
                [{"type": "xy"}]    # Capital allocation panel
            ]
        )
        
        # Update subplot title font size to make them more prominent
        for i in fig['layout']['annotations']:
            i['font'] = dict(size=14, color='white', family='Arial')
        
        # Create a complete title with performance metrics integrated
        if metrics:
            # Calculate Buy & Hold metrics
            first_signal_idx = 0  # Default to start of data
            buy_signal_col = 'buy_signal' if 'buy_signal' in self.results.columns else None
            if buy_signal_col:
                buy_signals = self.results[self.results[buy_signal_col] == 1]
                if not buy_signals.empty:
                    first_signal_idx = self.data.index.get_indexer([buy_signals.index[0]], method='nearest')[0]
            
            # Calculate buy-and-hold (benchmark) performance from engine results
            if 'benchmark_equity' in self.results.columns:
                benchmark_equity = self.results['benchmark_equity'].iloc[-1]
            else:
                # Fallback calculation if benchmark not available
                initial_price = self.data['Close'].iloc[first_signal_idx]
                final_price = self.data['Close'].iloc[-1]
                benchmark_equity = initial_equity * (final_price / initial_price)
            
            # Main title - more concise
            dynamic_title = self._generate_dynamic_title(ticker, base_strategy_name)
            subtitle_costs = self._generate_costs_subtitle()
            subtitle_performance = self._generate_performance_subtitle()
            
            # Add main title at the top
            fig.add_annotation(
                text=f"<b>{dynamic_title}</b>",
                xref="paper", yref="paper",
                x=0.5, y=1.13,  # Centered at top
                showarrow=False,
                font=dict(size=18, color="white", family="Arial"),
                align="center",
                bgcolor="rgba(70, 70, 70, 0.8)",
                bordercolor="#CCCCCC",
                borderwidth=1,
                borderpad=6
            )
            
            # Add costs and settings subtitle in the middle between title and performance
            if subtitle_costs:
                fig.add_annotation(
                    text=f"<i>{subtitle_costs}</i>",
                    xref="paper", yref="paper",
                    x=0.5, y=1.105,  # Positioned in middle between title and performance
                    showarrow=False,
                    font=dict(size=10, color="#CCCCCC", family="Arial"),
                    align="center",
                    bgcolor="rgba(40, 40, 40, 0.7)",
                    bordercolor="#666666",
                    borderwidth=1,
                    borderpad=4
                )
            
            # Add performance summary at the bottom
            if subtitle_performance:
                fig.add_annotation(
                    text=subtitle_performance,
                    xref="paper", yref="paper",
                    x=0.5, y=1.08,  # Below the costs subtitle
                    showarrow=False,
                    font=dict(size=11, color="white", family="Arial"),
                    align="center",
                    bgcolor="rgba(50, 50, 50, 0.7)",
                    bordercolor="#999999",
                    borderwidth=1,
                    borderpad=5
                )
            
            # Set title to empty to ensure clean layout
            fig.update_layout(title="")
        else:
            # Default title with increased spacing
            dynamic_title = self._generate_dynamic_title(ticker, base_strategy_name)
            fig.add_annotation(
                text=f"<b>{dynamic_title}</b>",
                xref="paper", yref="paper",
                x=0.5, y=1.12,  # Moved higher to avoid legend overlap
                showarrow=False,
                font=dict(size=18, color="white", family="Arial"),
                align="center"
            )
            fig.update_layout(title="")
        
        # Add price candlesticks to first row
        self._add_candlestick(fig, row=1, col=1)
        
        # Add trade signals to price chart
        self._add_trade_signals(fig, row=1, col=1)
        
        # Add dividend markers to price chart
        self._add_dividend_markers(fig, row=1, col=1)
        
        # Second row: Performance comparison
        if equity_col:
            # Find first signal date
            first_signal_date = None
            buy_signal_col = 'buy_signal' if 'buy_signal' in self.results.columns else None
            if buy_signal_col:
                buy_signals = self.results[self.results[buy_signal_col] == 1]
                if not buy_signals.empty:
                    first_signal_date = buy_signals.index[0]
            
            # If no signals found, use the first date
            if first_signal_date is None:
                first_signal_date = self.data.index[0]
            
            # Calculate buy-and-hold (benchmark) performance starting from first signal
            first_signal_idx = self.data.index.get_indexer([first_signal_date], method='nearest')[0]
            initial_price = self.data['Close'].iloc[first_signal_idx]
            
            # Use benchmark data from engine if available, otherwise calculate
            if 'benchmark_equity' in self.results.columns:
                # Use engine-calculated benchmark
                benchmark_equity_series = self.results['benchmark_equity'].iloc[first_signal_idx:]
                benchmark = benchmark_equity_series / benchmark_equity_series.iloc[0] * 100  # Normalized to 100
            else:
                # Fallback: Calculate benchmark with dividend reinvestment to match strategy treatment
                price_series = self.data['Close'].iloc[first_signal_idx:]
                
                # Include dividends in benchmark calculation if engine includes dividends
                include_divs = hasattr(self.engine, 'include_dividends') and self.engine.include_dividends
                
                if include_divs and 'Dividends' in self.data.columns:
                    # Calculate dividend-adjusted benchmark (total return)
                    dividend_series = self.data['Dividends'].iloc[first_signal_idx:]
                    
                    # Simulate dividend reinvestment: shares * (1 + dividend_yield) each period
                    benchmark_shares = 1.0  # Start with 1 share equivalent
                    benchmark_values = []
                    
                    for i, (date, price) in enumerate(price_series.items()):
                        if i > 0 and dividend_series.iloc[i] > 0:
                            # Reinvest dividends into more shares
                            dividend_yield = dividend_series.iloc[i] / price_series.iloc[i-1]  # Use previous price
                            benchmark_shares *= (1 + dividend_yield)
                        
                        # Calculate current portfolio value
                        portfolio_value = benchmark_shares * price
                        benchmark_values.append(portfolio_value)
                    
                    # Convert to series and normalize to 100
                    benchmark = pd.Series(benchmark_values, index=price_series.index)
                    benchmark = benchmark / benchmark.iloc[0] * 100  # Normalized to 100
                else:
                    # Price-only benchmark (no dividends)
                    initial_price = price_series.iloc[0]
                    benchmark = price_series / initial_price * 100  # Normalized to 100
            
            # Calculate benchmark drawdown from engine if available
            if 'benchmark_drawdown' in self.results.columns:
                benchmark_drawdown = self.results['benchmark_drawdown'].iloc[first_signal_idx:] * 100  # As percentage
            else:
                # Calculate benchmark drawdown
                benchmark_peak = benchmark.cummax()
                benchmark_drawdown = (benchmark - benchmark_peak) / benchmark_peak * 100  # As percentage
            
            # Calculate strategy performance starting from first signal
            initial_equity = self.results[equity_col].iloc[first_signal_idx]
            equity_series = self.results[equity_col].iloc[first_signal_idx:]
            normalized_equity = equity_series / initial_equity * 100  # Normalized to 100
            
            # Plot date range
            date_range = self.data.index[first_signal_idx:]
            
            # Define consistent color scheme for strategy and benchmark
            strategy_color = '#FF00FF'  # Magenta for strategy
            benchmark_color = '#00FFFF'  # Cyan for benchmark
            
            # PANEL 2: RETURNS - Add benchmark and strategy returns
            # Convert to cumulative index for log scale (starting from 100)
            benchmark_index = benchmark  # Already normalized to 100
            strategy_index = normalized_equity  # Already normalized to 100
            
            # Determine benchmark name based on engine configuration and dividend data
            has_engine_benchmark = 'benchmark_equity' in self.results.columns
            include_divs = hasattr(self.engine, 'include_dividends') and self.engine.include_dividends
            benchmark_name = "Buy & Hold Total Return" if (has_engine_benchmark and include_divs and 'Dividends' in self.data.columns) else "Buy & Hold Return"
            
            fig.add_trace(
                go.Scatter(
                    x=date_range,
                    y=benchmark_index,
                    mode='lines',
                    line=dict(width=1.5, color=benchmark_color),
                    name=benchmark_name
                ),
                row=2, col=1
            )
            
            # Add strategy equity curve
            fig.add_trace(
                go.Scatter(
                    x=date_range,
                    y=strategy_index,
                    mode='lines',
                    line=dict(width=3.0, color=strategy_color),
                    name="Strategy Return"
                ),
                row=2, col=1
            )
            
            # Add horizontal reference line at 100 (starting point)
            fig.add_hline(y=100, line_width=1, line_dash="dash", line_color="#999999", 
                        row=2, col=1)
            
            # Calculate metrics
            strategy_perf = (normalized_equity.iloc[-1] - 100) / 100  # Final value - initial value
            benchmark_perf = (benchmark.iloc[-1] - 100) / 100
            outperformance = strategy_perf - benchmark_perf
            
            # Add an annotation with performance metrics in the returns panel - REMOVED TO AVOID DUPLICATION
            # This was causing the duplicate performance block under the price area
            
            # PANEL 3: DRAWDOWNS - Add strategy and benchmark drawdowns
            if 'drawdown' in self.results.columns:
                # Strategy drawdown (as area chart with same color as strategy line)
                drawdown_series = self.results['drawdown'].iloc[first_signal_idx:] * 100  # As percentage
                
                # Make sure the series is properly sorted by index
                drawdown_series = drawdown_series.sort_index()
                
                # Convert RGB hex to rgba with transparency
                # Strategy color: convert hex to RGB components
                r = int(strategy_color[1:3], 16)
                g = int(strategy_color[3:5], 16)
                b = int(strategy_color[5:7], 16)
                strategy_fillcolor = f'rgba({r}, {g}, {b}, 0.3)'
                
                # Add strategy drawdown as area chart in the dedicated drawdown panel
                fig.add_trace(
                    go.Scatter(
                        x=date_range,
                        y=drawdown_series,
                        mode='lines',  # Use lines for better visibility
                        line=dict(width=0.5, color=strategy_color),
                        fill='tozeroy',  # Fill to zero
                        fillcolor=strategy_fillcolor,
                        name="Strategy Drawdown"
                    ),
                    row=3, col=1  # Dedicated drawdown panel
                )
                
                # Benchmark drawdown (as area chart with same color as benchmark line)
                # Make sure the benchmark drawdown series is properly sorted
                benchmark_drawdown = benchmark_drawdown.sort_index()
                
                # Benchmark color: convert hex to RGB components
                r = int(benchmark_color[1:3], 16)
                g = int(benchmark_color[3:5], 16)
                b = int(benchmark_color[5:7], 16)
                benchmark_fillcolor = f'rgba({r}, {g}, {b}, 0.3)'
                
                # Add benchmark drawdown to the dedicated drawdown panel
                has_engine_benchmark = 'benchmark_equity' in self.results.columns
                include_divs = hasattr(self.engine, 'include_dividends') and self.engine.include_dividends
                benchmark_drawdown_name = "Buy & Hold Total Return Drawdown" if (has_engine_benchmark and include_divs and 'Dividends' in self.data.columns) else "Buy & Hold Drawdown"
                
                fig.add_trace(
                    go.Scatter(
                        x=date_range,
                        y=benchmark_drawdown,
                        mode='lines',  # Use lines for better visibility
                        line=dict(width=0.5, color=benchmark_color),
                        fill='tozeroy',  # Fill to zero
                        fillcolor=benchmark_fillcolor,
                        name=benchmark_drawdown_name
                    ),
                    row=3, col=1  # Dedicated drawdown panel
                )
                
                # Configure y-axis for drawdowns panel
                min_drawdown = min(drawdown_series.min(), benchmark_drawdown.min()) if not drawdown_series.empty and not benchmark_drawdown.empty else -30
                fig.update_yaxes(
                    title_text="Drawdown (%)",
                    range=[min_drawdown*1.1, 5],  # Ensure all drawdowns are visible with some buffer at top
                    row=3, col=1  # Drawdown panel
                )
                
                # Add zero line for reference
                fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="#999999", 
                          row=3, col=1)
        
        # Add KDJ indicators to fourth row
        self._add_kdj_indicators(fig, row=4, col=1, is_monthly=True)
        
        # Add capital allocation stacked area chart to fifth row
        self._add_capital_allocation(fig, row=5, col=1)
        
        # Set base height for the chart
        base_height = 1200
        
        # Determine scale types
        scale_type = "log" if log_scale else "linear"
        
        # CRITICAL FIX: Set explicit x-axis ranges for all subplots to prevent compression
        start_date = self.results.index.min()
        end_date = self.results.index.max()
        
        # Update layout for dark theme with FIXED x-axis configuration
        fig.update_layout(
            title="",  # Empty string to ensure no default title is displayed
            
            # FIXED: Chart dimensions optimized for time series
            height=1200,
            width=1600,   # Increased width for better time series display
            
            # FIXED: Explicit x-axis configuration for ALL subplots
            xaxis=dict(
                type='date',
                range=[start_date, end_date],  # CRITICAL: Explicit range
                title="",
                gridcolor='#333333', 
                showgrid=True, 
                showspikes=True, 
                spikecolor="white", 
                spikethickness=1,
                matches='x'  # Ensure all axes are linked
            ),
            xaxis2=dict(
                type='date',
                range=[start_date, end_date],  # Same range for consistency
                title="",
                gridcolor='#333333', 
                showgrid=True, 
                showspikes=True, 
                spikecolor="white", 
                spikethickness=1,
                matches='x'
            ),
            xaxis3=dict(
                type='date',
                range=[start_date, end_date],  # Same range for consistency
                title="",
                gridcolor='#333333', 
                showgrid=True, 
                showspikes=True, 
                spikecolor="white", 
                spikethickness=1,
                matches='x'
            ),
            xaxis4=dict(
                type='date',
                range=[start_date, end_date],  # Same range for consistency
                title="",
                gridcolor='#333333', 
                showgrid=True, 
                showspikes=True, 
                spikecolor="white", 
                spikethickness=1,
                matches='x'
            ),
            xaxis5=dict(
                type='date',
                range=[start_date, end_date],  # Same range for consistency
                title="Date",  # Only show title on bottom panel
                gridcolor='#333333', 
                showgrid=True, 
                showspikes=True, 
                spikecolor="white", 
                spikethickness=1,
                matches='x'
            ),
            
            # Y-axis configurations with proper labeling - FIXED: Only one y-axis per subplot
            yaxis=dict(
                title="Price",
                type=scale_type,
                range=[None, None] if log_scale else None,
                gridcolor='#333333', 
                showgrid=True, 
                showspikes=True, 
                spikecolor="white", 
                spikethickness=1,
                side='right'  # Put y-axis on right side for price
            ),
            yaxis2=dict(
                title="Performance (%)",
                type="linear",  # Always use linear for performance to show full range
                gridcolor='#333333', 
                showgrid=True, 
                showspikes=True, 
                spikecolor="white", 
                spikethickness=1,
                autorange=True,
                side='right'
            ),
            yaxis3=dict(
                title="Drawdown (%)",
                gridcolor='#333333', 
                showgrid=True, 
                showspikes=True, 
                spikecolor="white", 
                spikethickness=1,
                side='right'
            ),
            yaxis4=dict(
                title="KDJ Values",
                gridcolor='#333333', 
                showgrid=True, 
                showspikes=True, 
                spikecolor="white", 
                spikethickness=1,
                side='right'
            ),
            yaxis5=dict(
                title="Capital ($)",
                gridcolor='#333333', 
                showgrid=True, 
                showspikes=True, 
                spikecolor="white", 
                spikethickness=1,
                side='right'
            ),
            
            # Theme and interaction
            template="plotly_dark",
            showlegend=True,
            legend=dict(
                orientation="h", 
                yanchor="bottom", 
                y=1.01,  # Increased margin (1.3x) below performance summary to avoid overlap
                xanchor="right", 
                x=1
            ),
            xaxis_rangeslider_visible=False,
            plot_bgcolor='#1E1E1E',
            paper_bgcolor='#1E1E1E',
            font=dict(color='white'),
            hovermode='x unified',
            
            # FIXED: Increased top margin to accommodate higher title position
            margin=dict(t=200, r=60, l=80, b=60)  # Increased top margin for higher title position
        )
        
        self.fig = fig
        return fig
    

    
    def show(self):
        """Display the current figure."""
        if self.fig is None:
            raise ValueError("No figure has been created yet. Call a plotting method first.")
        self.fig.show()
    
    def save(self, filename: str):
        """
        Save the current figure to a file.
        
        Args:
            filename: Output filename (can be HTML, PNG, JPEG, etc.)
        """
        if self.fig is None:
            raise ValueError("No figure has been created yet. Call a plotting method first.")
        self.fig.write_html(filename) if filename.endswith('.html') else self.fig.write_image(filename)
