"""
Performance and benchmark visualization components.
"""
from typing import Tuple
import pandas as pd
import plotly.graph_objects as go
from .styling import ChartStyler


class PerformancePlots:
    """Handles performance, equity curves, and benchmark visualizations."""
    
    def __init__(self, data: pd.DataFrame, results: pd.DataFrame, engine=None):
        """
        Initialize performance plots.
        
        Args:
            data: DataFrame with OHLCV data
            results: DataFrame with backtest results
            engine: BacktestEngine instance for configuration access
        """
        self.data = data
        self.results = results
        self.engine = engine
        self.styler = ChartStyler()
    
    def add_performance_comparison(self, fig: go.Figure, row: int = 1, col: int = 1) -> go.Figure:
        """
        Add strategy vs benchmark performance comparison to the specified subplot.
        
        Args:
            fig: Plotly figure to add trace to
            row: Row index for the subplot
            col: Column index for the subplot
            
        Returns:
            Updated figure
        """
        equity_col = 'equity' if 'equity' in self.results.columns else None
        if not equity_col:
            return fig
        
        # Find first signal date
        first_signal_date = self._get_first_signal_date()
        first_signal_idx = self.data.index.get_indexer([first_signal_date], method='nearest')[0]
        
        # Calculate benchmark performance
        benchmark_data = self._calculate_benchmark_performance(first_signal_idx)
        
        # Calculate strategy performance
        strategy_data = self._calculate_strategy_performance(first_signal_idx)
        
        # Plot date range
        date_range = self.data.index[first_signal_idx:]
        
        # Add benchmark performance
        benchmark_name = self._get_benchmark_name()
        fig.add_trace(
            go.Scatter(
                x=date_range,
                y=benchmark_data,
                mode='lines',
                line=self.styler.get_line_style(color=self.styler.COLORS['benchmark']),
                name=benchmark_name
            ),
            row=row, col=col
        )
        
        # Add strategy performance
        fig.add_trace(
            go.Scatter(
                x=date_range,
                y=strategy_data,
                mode='lines',
                line=self.styler.get_line_style(color=self.styler.COLORS['strategy'], width=3.0),
                name="Strategy Return"
            ),
            row=row, col=col
        )
        
        # Add horizontal reference line at 0 (starting point)
        fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="#999999", row=row, col=col)
        
        # Add ending value annotations
        self._add_ending_value_annotations(fig, row, col, {
            'Strategy': strategy_data.iloc[-1],
            'Benchmark': benchmark_data.iloc[-1]
        })
        
        return fig
    
    def add_drawdown_comparison(self, fig: go.Figure, row: int = 1, col: int = 1) -> go.Figure:
        """
        Add strategy and benchmark drawdown comparison.
        
        Args:
            fig: Plotly figure to add trace to
            row: Row index for the subplot
            col: Column index for the subplot
            
        Returns:
            Updated figure
        """
        if 'drawdown' not in self.results.columns:
            return fig
        
        # Find first signal date
        first_signal_date = self._get_first_signal_date()
        first_signal_idx = self.data.index.get_indexer([first_signal_date], method='nearest')[0]
        date_range = self.data.index[first_signal_idx:]
        
        # Strategy drawdown
        drawdown_series = self.results['drawdown'].iloc[first_signal_idx:] * 100  # As percentage
        drawdown_series = drawdown_series.sort_index()
        
        # Strategy color with transparency
        # Using DodgerBlue RGB (30, 144, 255)
        r, g, b = 30, 144, 255  # DodgerBlue RGB
        strategy_fillcolor = f'rgba({r}, {g}, {b}, 0.3)'
        
        fig.add_trace(
            go.Scatter(
                x=date_range,
                y=drawdown_series,
                mode='lines',
                line=self.styler.get_line_style(color=self.styler.COLORS['strategy'], width=3.0),
                fill='tozeroy',
                fillcolor=strategy_fillcolor,
                name="Strategy Drawdown"
            ),
            row=row, col=col
        )
        
        # Benchmark drawdown
        if 'benchmark_drawdown' in self.results.columns:
            benchmark_drawdown = self.results['benchmark_drawdown'].iloc[first_signal_idx:] * 100
        else:
            # Calculate benchmark drawdown from performance data starting at 0%
            benchmark_data = self._calculate_benchmark_performance(first_signal_idx)  # This is now 0% to X%
            # Convert back to cumulative values (1 + return) to calculate drawdown properly
            benchmark_values = (benchmark_data / 100) + 1  # Convert from percentage back to cumulative values
            benchmark_peak = benchmark_values.cummax()
            benchmark_drawdown = (benchmark_values - benchmark_peak) / benchmark_peak * 100
        
        benchmark_drawdown = benchmark_drawdown.sort_index()
        
        # Benchmark color with transparency
        # Using Grey RGB (128, 128, 128)
        r, g, b = 128, 128, 128  # Grey RGB
        benchmark_fillcolor = f'rgba({r}, {g}, {b}, 0.3)'
        
        benchmark_drawdown_name = self._get_benchmark_drawdown_name()
        fig.add_trace(
            go.Scatter(
                x=date_range,
                y=benchmark_drawdown,
                mode='lines',
                line=self.styler.get_line_style(color=self.styler.COLORS['benchmark'], width=2.0),
                fill='tozeroy',
                fillcolor=benchmark_fillcolor,
                name=benchmark_drawdown_name
            ),
            row=row, col=col
        )
        
        # Configure y-axis for drawdowns panel
        min_drawdown = min(drawdown_series.min(), benchmark_drawdown.min()) if not drawdown_series.empty and not benchmark_drawdown.empty else -30
        fig.update_yaxes(
            title_text="Drawdown (%)",
            range=[min_drawdown*1.1, 5],
            row=row, col=col
        )
        
        # Add zero line for reference
        fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="#999999", row=row, col=col)
        
        # Add ending value annotations for drawdowns
        self._add_ending_value_annotations(fig, row, col, {
            'Strategy DD': drawdown_series.iloc[-1],
            'Benchmark DD': benchmark_drawdown.iloc[-1]
        })
        
        return fig
    
    def add_equity_curve(self, fig: go.Figure, row: int = 1, col: int = 1) -> go.Figure:
        """
        Add equity curve overlay to price chart.
        
        Args:
            fig: Plotly figure to add trace to
            row: Row index for the subplot
            col: Column index for the subplot
            
        Returns:
            Updated figure
        """
        equity_col = 'equity' if 'equity' in self.results.columns else None
        if not equity_col:
            return fig
        
        # Normalize equity to match price scale for visibility
        scale_factor = self.data['Close'].iloc[0] / self.results[equity_col].iloc[0]
        normalized_equity = self.results[equity_col] * scale_factor
        
        fig.add_trace(
            go.Scatter(
                x=self.results.index,
                y=normalized_equity,
                mode='lines',
                line=self.styler.get_line_style(color='purple', width=1.5),
                name="Equity (Scaled)",
                opacity=0.7
            ),
            row=row, col=col
        )
        
        return fig
    
    def _get_first_signal_date(self):
        """Get the first signal date (buy or sell) from the results - matches capital allocation logic."""
        first_signal_date = None
        
        # Check for buy signals
        if 'buy_signal' in self.results.columns:
            buy_signals = self.results[self.results['buy_signal'] == 1]
            if not buy_signals.empty:
                first_signal_date = buy_signals.index[0]
        
        # Check for sell signals
        if 'sell_signal' in self.results.columns:
            sell_signals = self.results[self.results['sell_signal'] == 1]
            if not sell_signals.empty:
                first_sell_date = sell_signals.index[0]
                if first_signal_date is None or first_sell_date < first_signal_date:
                    first_signal_date = first_sell_date
        
        # If no signals found, use the first date
        if first_signal_date is None:
            first_signal_date = self.data.index[0]
        
        return first_signal_date
    
    def _calculate_benchmark_performance(self, first_signal_idx: int) -> pd.Series:
        """Calculate benchmark performance starting from first signal."""
        if 'benchmark_equity' in self.results.columns:
            # Use engine-calculated benchmark
            benchmark_equity_series = self.results['benchmark_equity'].iloc[first_signal_idx:]
            return (benchmark_equity_series / benchmark_equity_series.iloc[0] - 1) * 100  # Normalized to start at 0%
        else:
            # Fallback: Calculate benchmark with dividend reinvestment
            price_series = self.data['Close'].iloc[first_signal_idx:]
            include_divs = hasattr(self.engine, 'include_dividends') and self.engine.include_dividends
            
            if include_divs and 'Dividends' in self.data.columns:
                # Calculate dividend-adjusted benchmark (total return)
                dividend_series = self.data['Dividends'].iloc[first_signal_idx:]
                
                # Simulate dividend reinvestment
                benchmark_shares = 1.0  # Start with 1 share equivalent
                benchmark_values = []
                
                for i, (date, price) in enumerate(price_series.items()):
                    if i > 0 and dividend_series.iloc[i] > 0:
                        # Reinvest dividends into more shares
                        dividend_yield = dividend_series.iloc[i] / price_series.iloc[i-1]
                        benchmark_shares *= (1 + dividend_yield)
                    
                    # Calculate current portfolio value
                    portfolio_value = benchmark_shares * price
                    benchmark_values.append(portfolio_value)
                
                # Convert to series and normalize to 100
                benchmark = pd.Series(benchmark_values, index=price_series.index)
                return (benchmark / benchmark.iloc[0] - 1) * 100
            else:
                # Price-only benchmark (no dividends)
                initial_price = price_series.iloc[0]
                return (price_series / initial_price - 1) * 100
    
    def _calculate_strategy_performance(self, first_signal_idx: int) -> pd.Series:
        """Calculate strategy performance starting from first signal."""
        equity_col = 'equity' if 'equity' in self.results.columns else None
        initial_equity = self.results[equity_col].iloc[first_signal_idx]
        equity_series = self.results[equity_col].iloc[first_signal_idx:]
        return (equity_series / initial_equity - 1) * 100  # Normalized to start at 0%
    
    def _get_benchmark_name(self) -> str:
        """Get appropriate benchmark name based on configuration."""
        has_engine_benchmark = 'benchmark_equity' in self.results.columns
        include_divs = hasattr(self.engine, 'include_dividends') and self.engine.include_dividends
        return "Buy & Hold Total Return" if (has_engine_benchmark and include_divs and 'Dividends' in self.data.columns) else "Buy & Hold Return"
    
    def _get_benchmark_drawdown_name(self) -> str:
        """Get appropriate benchmark drawdown name based on configuration."""
        has_engine_benchmark = 'benchmark_equity' in self.results.columns
        include_divs = hasattr(self.engine, 'include_dividends') and self.engine.include_dividends
        return "Buy & Hold Total Return Drawdown" if (has_engine_benchmark and include_divs and 'Dividends' in self.data.columns) else "Buy & Hold Drawdown"
    
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
            # Choose color based on label
            if 'Strategy' in label:
                color = self.styler.COLORS['strategy']
            elif 'Benchmark' in label:
                color = self.styler.COLORS['benchmark']
            elif 'DD' in label:  # Drawdown values
                if 'Strategy' in label:
                    color = self.styler.COLORS['strategy']
                else:
                    color = self.styler.COLORS['benchmark']
            else:
                color = '#FFFFFF'
            
            # Format value based on magnitude and type
            if 'DD' in label:  # Drawdown formatting (negative percentages)
                formatted_value = f"{value:.1f}%"
            elif value >= 1000:
                formatted_value = f"{value:.0f}"
            elif value >= 100:
                formatted_value = f"{value:.1f}"
            else:
                formatted_value = f"{value:.2f}"
            
            # Apply small vertical offset if values are too close
            y_offset = i * 2 if len(sorted_values) > 1 and abs(sorted_values[0][1] - sorted_values[-1][1]) < 10 else 0
            
            # Add annotation on the right side at actual value position
            fig.add_annotation(
                x=1.002,  # Just outside the plot area
                y=value + y_offset,  # Actual value position with offset if needed
                xref=xref,
                yref=yref,
                text=formatted_value,
                showarrow=False,
                font=dict(color=color, size=10, family="Arial"),
                bgcolor="rgba(0, 0, 0, 0.7)",
                bordercolor=color,
                borderwidth=1,
                borderpad=2,
                xanchor="left",
                yanchor="middle"
            )
