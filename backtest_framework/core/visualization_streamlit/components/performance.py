"""
Performance visualization component.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import Optional, Any
from config import COLORS, CHART_THEME, CHART_HEIGHTS, DEFAULT_XAXIS, DEFAULT_YAXIS


class PerformanceChart:
    """Display performance comparison between strategy and benchmark."""
    
    def __init__(self, results: pd.DataFrame, engine: Optional = None):
        self.results = results
        self.engine = engine
    
    def render(self):
        """Render the performance chart."""
        st.subheader("📊 Strategy vs Benchmark Performance")
        
        # Create the chart
        fig = self._create_chart()
        
        # Display the chart
        st.plotly_chart(fig, use_container_width=True)
    
    def _create_chart(self) -> go.Figure:
        """Create the performance comparison chart."""
        fig = go.Figure()
        
        # Find first signal date for alignment
        first_signal_date = self._get_first_signal_date()
        
        # Add strategy performance
        if 'equity' in self.results.columns:
            # Calculate returns from first signal
            mask = self.results.index >= first_signal_date
            equity = self.results.loc[mask, 'equity']
            
            if len(equity) > 0:
                initial_equity = equity.iloc[0]
                strategy_returns = ((equity / initial_equity) - 1) * 100
                
                fig.add_trace(go.Scatter(
                    x=strategy_returns.index,
                    y=strategy_returns,
                    mode='lines',
                    name='Strategy',
                    line=dict(color=COLORS['strategy'], width=2),
                    hovertemplate='%{x|%Y-%m-%d}<br>Strategy: %{y:.2f}%<extra></extra>'
                ))
        
        # Add benchmark performance
        if 'benchmark_returns' in self.results.columns:
            # Align benchmark to same starting point
            mask = self.results.index >= first_signal_date
            benchmark_data = self.results.loc[mask, 'benchmark_returns']
            
            if len(benchmark_data) > 0:
                # Convert to percentage
                benchmark_returns = benchmark_data * 100
                
                fig.add_trace(go.Scatter(
                    x=benchmark_returns.index,
                    y=benchmark_returns,
                    mode='lines',
                    name='Benchmark (Buy & Hold)',
                    line=dict(color=COLORS['benchmark'], width=2),
                    hovertemplate='%{x|%Y-%m-%d}<br>Benchmark: %{y:.2f}%<extra></extra>'
                ))
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Break Even")
        
        # Update layout - avoid legend conflict by creating custom layout
        layout_settings = {
            'template': 'plotly_dark',
            'paper_bgcolor': COLORS['paper'],
            'plot_bgcolor': COLORS['background'],
            'font': {
                'family': 'Arial, sans-serif',
                'size': 12,
                'color': COLORS['text']
            },
            'hoverlabel': {
                'bgcolor': COLORS['paper'],
                'font_size': 12,
                'font_family': 'Arial, sans-serif'
            },
            'showlegend': True,
            'legend': {
                'bgcolor': 'rgba(0,0,0,0)',
                'bordercolor': COLORS['grid'],
                'borderwidth': 1,
                'font': {'size': 10},
                'yanchor': "top",
                'y': 0.99,
                'xanchor': "left",
                'x': 0.01
            },
            'margin': dict(l=10, r=80, t=30, b=30),
            'height': CHART_HEIGHTS['performance'],
            'xaxis_title': "Date",
            'yaxis_title': "Cumulative Returns (%)",
            'hovermode': 'x unified'
        }
        
        fig.update_layout(**layout_settings)
        
        return fig
    
    def _get_first_signal_date(self):
        """Get the date of the first trade signal."""
        if 'buy_signal' in self.results.columns:
            buy_signals = self.results[self.results['buy_signal'] == 1]
            if len(buy_signals) > 0:
                return buy_signals.index[0]
        
        if 'sell_signal' in self.results.columns:
            sell_signals = self.results[self.results['sell_signal'] == 1]
            if len(sell_signals) > 0:
                return sell_signals.index[0]
        
        # Default to first date if no signals
        return self.results.index[0]


class DrawdownChart:
    """Display drawdown visualization."""
    
    def __init__(self, results: pd.DataFrame):
        self.results = results
    
    def render(self):
        """Render the drawdown chart."""
        st.subheader("📉 Drawdown Analysis")
        
        # Create the chart
        fig = self._create_chart()
        
        # Display the chart
        st.plotly_chart(fig, use_container_width=True)
    
    def _create_chart(self) -> go.Figure:
        """Create the drawdown chart."""
        fig = go.Figure()
        
        # Add strategy drawdown
        if 'drawdown' in self.results.columns:
            drawdown_pct = self.results['drawdown'] * 100
            
            fig.add_trace(go.Scatter(
                x=self.results.index,
                y=drawdown_pct,
                mode='lines',
                name='Strategy Drawdown',
                line=dict(color=COLORS['sell_signal'], width=2),
                fill='tozeroy',
                fillcolor='rgba(255, 48, 48, 0.2)',
                hovertemplate='%{x|%Y-%m-%d}<br>Drawdown: %{y:.2f}%<extra></extra>'
            ))
        
        # Add benchmark drawdown if available
        if 'benchmark_drawdown' in self.results.columns:
            benchmark_dd_pct = self.results['benchmark_drawdown'] * 100
            
            fig.add_trace(go.Scatter(
                x=self.results.index,
                y=benchmark_dd_pct,
                mode='lines',
                name='Benchmark Drawdown',
                line=dict(color=COLORS['benchmark'], width=2),
                hovertemplate='%{x|%Y-%m-%d}<br>Benchmark DD: %{y:.2f}%<extra></extra>'
            ))
        
        # Update layout - avoid conflicts by creating custom layout
        layout_settings = {
            'template': 'plotly_dark',
            'paper_bgcolor': COLORS['paper'],
            'plot_bgcolor': COLORS['background'],
            'font': {
                'family': 'Arial, sans-serif',
                'size': 12,
                'color': COLORS['text']
            },
            'hoverlabel': {
                'bgcolor': COLORS['paper'],
                'font_size': 12,
                'font_family': 'Arial, sans-serif'
            },
            'showlegend': True,
            'legend': {
                'bgcolor': 'rgba(0,0,0,0)',
                'bordercolor': COLORS['grid'],
                'borderwidth': 1,
                'font': {'size': 10}
            },
            'margin': dict(l=10, r=80, t=30, b=30),
            'height': CHART_HEIGHTS['drawdown'],
            'xaxis_title': "Date",
            'yaxis_title': "Drawdown (%)",
            'hovermode': 'x unified',
            'yaxis': dict(
                gridcolor=COLORS['grid'],
                showgrid=True,
                zeroline=False,
                side='right',
                autorange='reversed'  # Invert y-axis so drawdowns go down
            )
        }
        
        fig.update_layout(**layout_settings)
        
        return fig
