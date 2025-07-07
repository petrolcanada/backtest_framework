"""
Capital allocation visualization component.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import Optional
from config import COLORS, CHART_THEME, CHART_HEIGHTS


class AllocationChart:
    """Display capital allocation over time."""
    
    def __init__(self, results: pd.DataFrame):
        self.results = results
    
    def render(self):
        """Render the allocation chart."""
        st.subheader("💰 Capital Allocation")
        
        # Check if we have allocation data
        allocation_cols = ['cash', 'long_position_value', 'short_position_value', 
                          'short_proceeds', 'margin_used']
        
        if not any(col in self.results.columns for col in allocation_cols):
            st.info("No capital allocation data available")
            return
        
        # Create the chart
        fig = self._create_chart()
        
        # Display the chart
        st.plotly_chart(fig, use_container_width=True)
    
    def _create_chart(self) -> go.Figure:
        """Create the stacked area chart for capital allocation."""
        fig = go.Figure()
        
        # Prepare data for stacked area chart
        data_to_plot = []
        
        # Add components in order (bottom to top)
        if 'cash' in self.results.columns:
            data_to_plot.append(('Cash', self.results['cash'], COLORS['cash']))
        
        if 'short_proceeds' in self.results.columns:
            data_to_plot.append(('Short Proceeds', self.results['short_proceeds'], COLORS['short_proceeds']))
        
        if 'long_position_value' in self.results.columns:
            data_to_plot.append(('Long Positions', self.results['long_position_value'], COLORS['long_pos']))
        
        if 'short_position_value' in self.results.columns:
            # Short positions are typically negative, so we take absolute value
            short_vals = self.results['short_position_value'].abs()
            data_to_plot.append(('Short Positions', short_vals, COLORS['short_pos']))
        
        if 'margin_used' in self.results.columns:
            data_to_plot.append(('Margin Used', self.results['margin_used'], COLORS['margin']))
        
        # Add traces in reverse order for proper stacking
        for name, values, color in reversed(data_to_plot):
            fig.add_trace(go.Scatter(
                x=self.results.index,
                y=values,
                mode='lines',
                name=name,
                line=dict(width=0),
                fillcolor=color,
                stackgroup='one',
                hovertemplate=f'%{{x|%Y-%m-%d}}<br>{name}: $%{{y:,.0f}}<extra></extra>'
            ))
        
        # Add total equity line
        if 'equity' in self.results.columns:
            fig.add_trace(go.Scatter(
                x=self.results.index,
                y=self.results['equity'],
                mode='lines',
                name='Total Equity',
                line=dict(color='white', width=2, dash='dash'),
                hovertemplate='%{x|%Y-%m-%d}<br>Total Equity: $%{y:,.0f}<extra></extra>'
            ))
        
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
            'height': CHART_HEIGHTS['allocation'],
            'xaxis_title': "Date",
            'yaxis_title': "Capital ($)",
            'hovermode': 'x unified'
        }
        
        fig.update_layout(**layout_settings)
        
        return fig
