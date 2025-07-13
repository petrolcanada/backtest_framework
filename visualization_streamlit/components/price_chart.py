"""
Price chart component with trade signals.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import Optional
from config import COLORS, CHART_THEME, CHART_HEIGHTS, DEFAULT_XAXIS, DEFAULT_YAXIS


class PriceChart:
    """Display price chart with candlesticks, indicators, and trade signals."""
    
    def __init__(self, data: pd.DataFrame, results: pd.DataFrame):
        self.data = data
        self.results = results
    
    def render(self):
        """Render the price chart."""
        st.subheader("📈 Price Action & Trade Signals")
        
        # Create price chart
        price_fig = self._create_price_chart()
        st.plotly_chart(price_fig, use_container_width=True)
        
        # Create volume chart separately
        if 'Volume' in self.data.columns:
            volume_fig = self._create_volume_chart()
            st.plotly_chart(volume_fig, use_container_width=True)
    
    def _create_price_chart(self) -> go.Figure:
        """Create the price chart with candlesticks and signals."""
        fig = go.Figure()
        
        # Add candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=self.data.index,
                open=self.data['Open'],
                high=self.data['High'],
                low=self.data['Low'],
                close=self.data['Close'],
                name='Price',
                increasing_line_color=COLORS['buy_signal'],
                decreasing_line_color=COLORS['sell_signal'],
                showlegend=True
            )
        )
        
        # Add SMA if available
        self._add_sma(fig)
        
        # Add trade signals
        self._add_trade_signals(fig)
        
        # Add dividend markers if available
        self._add_dividends(fig)
        
        # Update layout
        fig.update_layout(
            **CHART_THEME,
            height=CHART_HEIGHTS['price'] - 100,  # Slightly smaller since volume is separate
            xaxis_rangeslider_visible=False,
            hovermode='x unified',
        )
        
        # Update axes separately to avoid conflicts
        fig.update_xaxes(**DEFAULT_XAXIS, title="")
        fig.update_yaxes(**DEFAULT_YAXIS, title="Price")
        
        return fig
    
    def _create_volume_chart(self) -> go.Figure:
        """Create a separate volume chart."""
        fig = go.Figure()
        
        # Determine colors based on price change
        colors = []
        for i in range(len(self.data)):
            if i == 0:
                colors.append(COLORS['volume'])
            else:
                if self.data['Close'].iloc[i] >= self.data['Close'].iloc[i-1]:
                    colors.append(COLORS['buy_signal'])
                else:
                    colors.append(COLORS['sell_signal'])
        
        # Add volume bars
        fig.add_trace(
            go.Bar(
                x=self.data.index,
                y=self.data['Volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.5,
                showlegend=False,
                hovertemplate='%{x|%Y-%m-%d}<br>Volume: %{y:,.0f}<extra></extra>'
            )
        )
        
        # Update layout - avoid margin conflict by creating custom layout
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
            'margin': dict(l=10, r=80, t=10, b=30),  # Custom margin for volume chart
            'height': 150,  # Smaller height for volume
            'hovermode': 'x unified'
        }
        
        fig.update_layout(**layout_settings)
        
        # Update axes separately
        fig.update_xaxes(**DEFAULT_XAXIS, title="Date")
        fig.update_yaxes(**DEFAULT_YAXIS, title="Volume")
        
        return fig
    
    def _add_sma(self, fig: go.Figure):
        """Add SMA lines if available."""
        # Check for different SMA periods
        sma_columns = [col for col in self.data.columns if col.startswith('SMA_')]
        
        for sma_col in sma_columns:
            period = sma_col.split('_')[1]
            fig.add_trace(
                go.Scatter(
                    x=self.data.index,
                    y=self.data[sma_col],
                    mode='lines',
                    name=f'SMA {period}',
                    line=dict(width=2, color='orange'),
                    showlegend=True
                )
            )
    
    def _add_trade_signals(self, fig: go.Figure):
        """Add buy and sell signal markers."""
        # Buy signals
        if 'buy_signal' in self.results.columns:
            buy_signals = self.results[self.results['buy_signal'] == 1]
            if len(buy_signals) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=buy_signals.index,
                        y=self.data.loc[buy_signals.index, 'Low'] * 0.99,
                        mode='markers',
                        name='Buy Signal',
                        marker=dict(
                            symbol='triangle-up',
                            size=12,
                            color=COLORS['buy_signal'],
                            line=dict(width=1, color='white')
                        ),
                        showlegend=True
                    )
                )
        
        # Sell signals
        if 'sell_signal' in self.results.columns:
            sell_signals = self.results[self.results['sell_signal'] == 1]
            if len(sell_signals) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=sell_signals.index,
                        y=self.data.loc[sell_signals.index, 'High'] * 1.01,
                        mode='markers',
                        name='Sell Signal',
                        marker=dict(
                            symbol='triangle-down',
                            size=12,
                            color=COLORS['sell_signal'],
                            line=dict(width=1, color='white')
                        ),
                        showlegend=True
                    )
                )
    
    def _add_dividends(self, fig: go.Figure):
        """Add dividend markers if available."""
        if 'Dividends' in self.data.columns:
            dividends = self.data[self.data['Dividends'] > 0]
            if len(dividends) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=dividends.index,
                        y=self.data.loc[dividends.index, 'High'] * 1.02,
                        mode='markers+text',
                        name='Dividend',
                        marker=dict(
                            symbol='diamond',
                            size=10,
                            color=COLORS['dividend'],
                            line=dict(width=1, color='white')
                        ),
                        text=[f"${d:.2f}" for d in dividends['Dividends']],
                        textposition="top center",
                        showlegend=True
                    )
                )
