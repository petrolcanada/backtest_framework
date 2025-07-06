"""
Modern, component-based plotting module for visualizing prices, indicators, and backtest results.
"""
from typing import Optional, List
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import the modular components
from .components import (
    TitleGenerator,
    ChartStyler,
    ChartElements,
    PerformancePlots,
    AllocationPlots,
    DynamicIndicatorCoordinator
)


class Plotter:
    """
    Modern visualization engine for backtest results.
    
    This class coordinates between specialized component modules to create
    comprehensive, interactive charts for trading strategy analysis.
    """
    
    def __init__(self, data: pd.DataFrame, results: Optional[pd.DataFrame] = None, engine: Optional = None):
        """
        Initialize the plotter with data and components.
        
        Args:
            data: DataFrame with OHLCV data and indicators
            results: DataFrame with backtest results, if None, assumes data contains results
            engine: BacktestEngine instance for configuration access
        """
        self.data = data
        self.results = results if results is not None else data
        self.engine = engine
        self.fig = None
        
        # Initialize specialized components
        self.title_gen = TitleGenerator(self.data, self.results, self.engine)
        self.styler = ChartStyler()
        self.chart_elements = ChartElements(self.data, self.results)
        self.performance = PerformancePlots(self.data, self.results, self.engine)
        self.allocation = AllocationPlots(self.results)
        
        # Initialize dynamic indicator coordinator
        self.dynamic_indicators = DynamicIndicatorCoordinator(self.results)
    
    def create_comprehensive_chart(self, ticker: str = '', base_strategy_name: str = "Strategy", 
                                 log_scale: bool = True) -> go.Figure:
        """
        Create a comprehensive chart with all visualization panels.
        
        Args:
            ticker: Ticker symbol for chart title
            base_strategy_name: Base strategy name for dynamic title generation
            log_scale: Whether to use logarithmic scale for price charts
            
        Returns:
            Plotly Figure object with comprehensive visualization
        """
        # Configure subplot layout with all panels reduced by 40%
        # Original heights: [0.28, 0.16, 0.16, 0.16, 0.24]
        # Price panel reduced by 40%: 0.28 * 0.6 = 0.168
        # All panels reduced by 40%: [0.168, 0.096, 0.096, 0.096, 0.144]
        # Normalized to maintain proportions: [0.22, 0.195, 0.195, 0.195, 0.195]
        subplot_config = self.styler.get_subplot_config(
            rows=5,
            vertical_spacing=0.03,  # Slightly increased spacing for better separation
            row_heights=[0.22, 0.195, 0.195, 0.195, 0.195]  # All panels reduced by ~40%
        )
        
        # Create subplot structure with adjusted spacing for legend
        fig = make_subplots(
            **subplot_config,
            subplot_titles=[
                "",  # Price chart (title added separately)
                "Strategy vs Benchmark Performance",
                "Drawdowns",
                "Capital Allocation",  # Moved next to drawdowns
                "Technical Indicators"
            ],
            specs=[
                [{"type": "xy"}],      # Price panel
                [{"type": "xy"}],      # Performance panel
                [{"type": "xy"}],      # Drawdowns panel
                [{"type": "xy"}],      # Capital allocation panel (moved up)
                [{"type": "xy"}]       # Indicators panel (moved down)
            ]
        )
        
        # Adjust subplot positions with perfectly equal heights for all sub-panels
        # Price panel: 21% height, All sub-panels: exactly 15% height each with 3% spacing
        fig.update_layout(
            yaxis=dict(domain=[0.73, 0.94]),     # Price panel - 21% height
            yaxis2=dict(domain=[0.58, 0.73]),    # Performance panel - 15% height
            yaxis3=dict(domain=[0.43, 0.58]),    # Drawdowns panel - 15% height  
            yaxis4=dict(domain=[0.28, 0.43]),    # Capital allocation panel - 15% height
            yaxis5=dict(domain=[0.13, 0.28])     # Indicators panel - 15% height (same as others)
        )
        
        # Style subplot titles
        self._style_subplot_titles(fig)
        
        # Add chart content with reordered panels
        self._build_price_panel(fig, row=1, log_scale=log_scale)
        self._build_performance_panel(fig, row=2)
        self._build_drawdown_panel(fig, row=3)
        self._build_allocation_panel(fig, row=4)  # Moved up next to drawdowns
        self._build_indicators_panel(fig, row=5)  # Moved down
        
        # Add titles and annotations
        self._add_chart_titles(fig, ticker, base_strategy_name)
        
        # Apply layout and styling
        self._apply_layout_styling(fig, log_scale)
        
        self.fig = fig
        return fig
    
    def _build_price_panel(self, fig: go.Figure, row: int, log_scale: bool = False) -> None:
        """Build the price panel with candlesticks, signals, and overlays."""
        # Add core price data
        self.chart_elements.add_candlestick(fig, row=row, col=1, log_scale=log_scale)
        
        # Add trade signals
        self.chart_elements.add_trade_signals(fig, row=row, col=1)
        
        # Add dividend markers
        self.chart_elements.add_dividend_markers(fig, row=row, col=1)
        
        # Add price panel indicators (like SMA) using dynamic system
        self.dynamic_indicators.apply_price_panel_indicators(fig, row=row, col=1)
    
    def _build_performance_panel(self, fig: go.Figure, row: int) -> None:
        """Build the performance comparison panel."""
        self.performance.add_performance_comparison(fig, row=row, col=1)
    
    def _build_drawdown_panel(self, fig: go.Figure, row: int) -> None:
        """Build the drawdown comparison panel."""
        self.performance.add_drawdown_comparison(fig, row=row, col=1)
    
    def _build_indicators_panel(self, fig: go.Figure, row: int) -> None:
        """Build the technical indicators panel using dynamic indicator system."""
        # Use the dynamic indicator coordinator to automatically detect and apply indicator panel indicators
        fig, success = self.dynamic_indicators.apply_indicator_panel_indicators(fig, row=row, col=1)
        
        if not success:
            # Get debug info for troubleshooting
            debug_info = self.dynamic_indicators.get_debug_info()
            
            # Add placeholder with debug information
            fig.add_annotation(
                x=0.5, y=0.5,
                xref=f"x{row} domain", yref=f"y{row} domain",
                text=f"No indicators available<br>Computed: {debug_info['computed_indicators']}<br>Available viz: {debug_info['available_visualizations']}",
                showarrow=False,
                font=dict(size=12, color="gray")
            )
    
    def _build_allocation_panel(self, fig: go.Figure, row: int) -> None:
        """Build the capital allocation panel."""
        _, success = self.allocation.add_capital_allocation(fig, row=row, col=1)
        
        if not success:
            # Add placeholder if no allocation data available
            fig.add_annotation(
                x=0.5, y=0.5,
                xref=f"x{row} domain", yref=f"y{row} domain",
                text="No capital allocation data available",
                showarrow=False,
                font=dict(size=14, color="gray")
            )
    
    def _add_chart_titles(self, fig: go.Figure, ticker: str, base_strategy_name: str) -> None:
        """Add main title and subtitles to the chart with proper spacing."""
        # Generate titles
        main_title = self.title_gen.generate_main_title(ticker, base_strategy_name)
        costs_subtitle = self.title_gen.generate_costs_subtitle()
        performance_subtitle = self.title_gen.generate_performance_subtitle()
        
        # Position titles with proper spacing to avoid overlap
        # Using paper coordinates for consistent positioning
        main_title_y = 0.995    # Even closer to top edge
        costs_subtitle_y = 0.970 # Increased spacing from main title (was 0.975)
        performance_subtitle_y = 0.950  # Adjusted to maintain even spacing
        
        # Add main title
        fig.add_annotation(
            **self.styler.create_annotation(
                text=f"<b>{main_title}</b>",
                x=0.5, y=main_title_y,
                xref="paper", yref="paper",  # Use paper coordinates for consistent positioning
                font_size=14,  # Smaller to reduce vertical space
                bgcolor="rgba(70, 70, 70, 0.8)",
                bordercolor="#CCCCCC"
            )
        )
        
        # Add costs subtitle
        if costs_subtitle:
            fig.add_annotation(
                **self.styler.create_annotation(
                    text=f"<i>{costs_subtitle}</i>",
                    x=0.5, y=costs_subtitle_y,
                    xref="paper", yref="paper",
                    font_size=8,  # Even smaller font for subtitle
                    bgcolor="rgba(40, 40, 40, 0.7)",
                    bordercolor="#666666"
                )
            )
        
        # Add performance subtitle
        if performance_subtitle:
            fig.add_annotation(
                **self.styler.create_annotation(
                    text=performance_subtitle,
                    x=0.5, y=performance_subtitle_y,
                    xref="paper", yref="paper",
                    font_size=9,  # Smaller to save space
                    bgcolor="rgba(50, 50, 50, 0.7)",
                    bordercolor="#999999"
                )
            )
    
    def _apply_layout_styling(self, fig: go.Figure, log_scale: bool) -> None:
        """Apply comprehensive layout and styling to the figure."""
        # Get base layout configuration with minimal top margin for titles
        layout_config = self.styler.get_dark_theme_layout(
            height=1200,
            width=1600,
            top_margin=30  # Minimal margin to reduce wasted space
        )
        
        # Configure x-axes for all subplots
        start_date = self.results.index.min()
        end_date = self.results.index.max()
        
        x_axis_config = {
            'type': 'date',
            'range': [start_date, end_date],
            'gridcolor': '#333333',
            'showgrid': True,
            'showspikes': True,
            'spikecolor': "white",
            'spikethickness': 1
        }
        
        # Configure y-axes for each panel
        scale_type = "log" if log_scale else "linear"
        
        axis_configs = {
            'xaxis': {**x_axis_config, 'title': "", 'matches': 'x'},
            'xaxis2': {**x_axis_config, 'title': "", 'matches': 'x'},
            'xaxis3': {**x_axis_config, 'title': "", 'matches': 'x'},
            'xaxis4': {**x_axis_config, 'title': "", 'matches': 'x'},
            'xaxis5': {**x_axis_config, 'title': "Date", 'matches': 'x'},
            
            'yaxis': self.styler.get_axis_config("Price", scale_type, "right"),
            'yaxis2': self.styler.get_axis_config("Performance (%)", "linear", "right"),
            'yaxis3': self.styler.get_axis_config("Drawdown (%)", "linear", "right"),
            'yaxis4': self.styler.get_axis_config("Capital ($)", "linear", "right"),  # Capital allocation moved to row 4
            'yaxis5': self.styler.get_axis_config("KDJ Values", "linear", "right")   # Indicators moved to row 5
        }
        
        # Apply all configurations
        fig.update_layout(**layout_config, **axis_configs, title="")
    
    def _style_subplot_titles(self, fig: go.Figure) -> None:
        """Apply styling to subplot titles."""
        for annotation in fig['layout']['annotations']:
            annotation['font'] = dict(size=14, color='white', family='Arial')
    
    # Convenience methods for backward compatibility
    def plot_chart_with_benchmark(self, ticker: str = '', base_strategy_name: str = "Monthly KDJ", 
                                 log_scale: bool = True) -> go.Figure:
        """Create comprehensive chart (delegates to create_comprehensive_chart)."""
        return self.create_comprehensive_chart(ticker, base_strategy_name, log_scale)
    
    def show(self) -> None:
        """Display the current figure."""
        if self.fig is None:
            raise ValueError("No figure has been created yet. Call a plotting method first.")
        self.fig.show()
    
    def open_in_browser(self, filename: str) -> None:
        """
        Open the saved HTML file directly in the default browser.
        
        Args:
            filename: Path to the HTML file to open
        """
        import os
        import webbrowser
        import platform
        
        # Ensure the file exists
        if not os.path.exists(filename):
            raise FileNotFoundError(f"HTML file not found: {filename}")
        
        # Get the absolute path
        abs_path = os.path.abspath(filename)
        
        # Use file:// protocol for local files
        if platform.system() == 'Windows':
            # Windows requires three slashes after file:
            file_url = f"file:///{abs_path.replace(os.sep, '/')}"
        else:
            # Unix-like systems
            file_url = f"file://{abs_path}"
        
        # Open in the default browser
        try:
            webbrowser.open(file_url, new=2)  # new=2 opens in a new tab
            print(f"Opened {filename} in browser")
        except Exception as e:
            print(f"Failed to open browser: {e}")
            # Fallback: use OS-specific command
            if platform.system() == 'Windows':
                os.startfile(abs_path)
            elif platform.system() == 'Darwin':  # macOS
                os.system(f'open "{abs_path}"')
            else:  # Linux
                os.system(f'xdg-open "{abs_path}"')
    
    def save(self, filename: str) -> None:
        """
        Save the current figure to a file.
        
        Args:
            filename: Output filename (HTML, PNG, JPEG, etc.)
        """
        if self.fig is None:
            raise ValueError("No figure has been created yet. Call a plotting method first.")
        
        if filename.endswith('.html'):
            self.fig.write_html(filename)
        else:
            self.fig.write_image(filename)
