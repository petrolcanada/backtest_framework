"""
Configuration and theme settings for Streamlit visualization.
"""
import plotly.graph_objects as go

# Color scheme matching the original dark theme
COLORS = {
    'background': '#0e1117',  # Streamlit's dark background
    'paper': '#262730',       # Streamlit's container background
    'text': '#fafafa',
    'grid': '#333333',
    'strategy': '#1E90FF',    # DodgerBlue for strategy
    'benchmark': '#808080',   # Grey for benchmark
    'buy_signal': '#00FF7F',  # SpringGreen
    'sell_signal': '#FF3030', # Firebrick red
    'dividend': '#FFD700',    # Gold
    'volume': 'rgba(0, 0, 255, 0.5)',
    'kdj_k': '#4169E1',       # Royal Blue
    'kdj_d': '#FF8C00',       # Dark Orange  
    'kdj_j': '#32CD32',       # Lime Green
    'cash': 'rgba(34, 139, 34, 0.6)',       # Forest green
    'short_proceeds': 'rgba(0, 255, 127, 0.6)',  # Spring green
    'long_pos': 'rgba(70, 130, 180, 0.6)',  # Steel blue
    'short_pos': 'rgba(220, 20, 60, 0.6)',  # Crimson
    'margin': 'rgba(255, 165, 0, 0.6)',     # Orange
    'adx': '#9370DB',         # Medium Purple
    'adx_plus': '#00CED1',    # Dark Turquoise
    'adx_minus': '#DC143C',   # Crimson
    'mfi': '#FF69B4',         # Hot Pink
    'rsi': '#FFD700',         # Gold
}

# Plotly chart theme configuration - without axis defaults
CHART_THEME = {
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
}

# Default axis configuration
DEFAULT_XAXIS = {
    'gridcolor': COLORS['grid'],
    'showgrid': True,
    'zeroline': False,
    'showspikes': True,
    'spikecolor': COLORS['text'],
    'spikethickness': 1,
    'spikemode': 'across',
}

DEFAULT_YAXIS = {
    'gridcolor': COLORS['grid'],
    'showgrid': True,
    'zeroline': False,
    'side': 'right',
}

# Standard chart heights
CHART_HEIGHTS = {
    'price': 500,
    'performance': 350,
    'drawdown': 300,
    'allocation': 300,
    'indicator': 250,
}

# Metric card styling
METRIC_CARD_STYLE = """
    <style>
    [data-testid="metric-container"] {
        background-color: rgba(38, 39, 48, 0.5);
        border: 1px solid rgba(250, 250, 250, 0.2);
        padding: 15px;
        border-radius: 5px;
        margin: 5px;
    }
    </style>
"""
