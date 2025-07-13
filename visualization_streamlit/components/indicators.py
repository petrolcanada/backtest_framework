"""
Technical indicators visualization component.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Any
from config import COLORS, CHART_THEME, CHART_HEIGHTS


class IndicatorPanels:
    """Display technical indicators in separate panels."""
    
    def __init__(self, data: pd.DataFrame, results: pd.DataFrame):
        self.data = data
        self.results = results
        self.indicators = self._detect_indicators()
    
    def render(self):
        """Render all available indicator panels."""
        if not self.indicators:
            return
        
        st.subheader("📈 Technical Indicators")
        
        # Create a chart for each indicator
        for indicator_name, indicator_config in self.indicators.items():
            self._render_indicator(indicator_name, indicator_config)
    
    def _detect_indicators(self) -> Dict[str, Dict[str, Any]]:
        """Detect which indicators are available in the data."""
        indicators = {}
        
        # KDJ Indicator
        if all(col in self.results.columns for col in ['kdj_k', 'kdj_d', 'kdj_j']):
            indicators['KDJ'] = {
                'traces': [
                    {'column': 'kdj_k', 'name': 'K', 'color': COLORS['kdj_k']},
                    {'column': 'kdj_d', 'name': 'D', 'color': COLORS['kdj_d']},
                    {'column': 'kdj_j', 'name': 'J', 'color': COLORS['kdj_j']}
                ],
                'y_range': [-20, 120],
                'reference_lines': [0, 20, 50, 80, 100]
            }
        
        # ADX Indicator
        if all(col in self.results.columns for col in ['adx', 'adx_plus_di', 'adx_minus_di']):
            indicators['ADX'] = {
                'traces': [
                    {'column': 'adx', 'name': 'ADX', 'color': COLORS['adx']},
                    {'column': 'adx_plus_di', 'name': '+DI', 'color': COLORS['adx_plus']},
                    {'column': 'adx_minus_di', 'name': '-DI', 'color': COLORS['adx_minus']}
                ],
                'y_range': [0, 100],
                'reference_lines': [20, 25, 40]
            }
        
        # MFI Indicator
        if 'mfi' in self.results.columns:
            indicators['MFI'] = {
                'traces': [
                    {'column': 'mfi', 'name': 'MFI', 'color': COLORS['mfi']}
                ],
                'y_range': [0, 100],
                'reference_lines': [20, 50, 80]
            }
        
        # RSI Indicator
        if 'rsi' in self.results.columns:
            indicators['RSI'] = {
                'traces': [
                    {'column': 'rsi', 'name': 'RSI', 'color': COLORS['rsi']}
                ],
                'y_range': [0, 100],
                'reference_lines': [30, 50, 70]
            }
        
        return indicators
    
    def _render_indicator(self, name: str, config: Dict[str, Any]):
        """Render a single indicator panel."""
        # Create figure
        fig = go.Figure()
        
        # Add traces
        for trace_config in config['traces']:
            fig.add_trace(go.Scatter(
                x=self.results.index,
                y=self.results[trace_config['column']],
                mode='lines',
                name=trace_config['name'],
                line=dict(color=trace_config['color'], width=2),
                hovertemplate=f'%{{x|%Y-%m-%d}}<br>{trace_config["name"]}: %{{y:.2f}}<extra></extra>'
            ))
        
        # Add reference lines
        for ref_line in config.get('reference_lines', []):
            fig.add_hline(
                y=ref_line, 
                line_dash="dot", 
                line_color="gray", 
                opacity=0.5,
                annotation_text=str(ref_line),
                annotation_position="right"
            )
        
        # Add shaded regions for overbought/oversold
        if name in ['RSI', 'MFI']:
            # Oversold region
            fig.add_hrect(
                y0=0, y1=config['reference_lines'][0],
                fillcolor="rgba(0, 255, 0, 0.1)",
                line_width=0
            )
            # Overbought region
            fig.add_hrect(
                y0=config['reference_lines'][2], y1=100,
                fillcolor="rgba(255, 0, 0, 0.1)",
                line_width=0
            )
        elif name == 'KDJ':
            # Oversold region
            fig.add_hrect(
                y0=0, y1=20,
                fillcolor="rgba(0, 255, 0, 0.1)",
                line_width=0
            )
            # Overbought region
            fig.add_hrect(
                y0=80, y1=100,
                fillcolor="rgba(255, 0, 0, 0.1)",
                line_width=0
            )
        
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
                'orientation': "h",
                'yanchor': "bottom",
                'y': 1.02,
                'xanchor': "right",
                'x': 1
            },
            'margin': dict(l=10, r=80, t=30, b=30),
            'height': CHART_HEIGHTS['indicator'],
            'title': f"{name} Indicator",
            'xaxis_title': "Date",
            'yaxis_title': name,
            'yaxis': dict(
                range=config.get('y_range'),
                gridcolor=COLORS['grid'],
                showgrid=True,
                zeroline=False,
                side='right'
            ),
            'hovermode': 'x unified'
        }
        
        fig.update_layout(**layout_settings)
        
        # Display the chart
        st.plotly_chart(fig, use_container_width=True)
