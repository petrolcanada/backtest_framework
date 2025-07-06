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
        # Get available indicators to determine subplot structure
        available_indicators = self.dynamic_indicators.get_available_visualizations()
        indicator_panel_indicators = [viz for viz in available_indicators if viz in ['MonthlyKDJ', 'ADX', 'MFI', 'RSI']]
        
        # Calculate dynamic subplot structure
        # Base panels: Price, Performance, Drawdowns, Allocation
        base_panels = 4
        indicator_panels = len(indicator_panel_indicators)
        total_panels = base_panels + indicator_panels
        
        # Calculate heights - give indicators more space if there are many
        if indicator_panels <= 2:
            # Standard layout: Price gets more space
            price_height = 0.25
            other_height = 0.15
            indicator_height = 0.12
        elif indicator_panels <= 4:
            # Balanced layout
            price_height = 0.20
            other_height = 0.12
            indicator_height = 0.10
        else:
            # Indicator-heavy layout
            price_height = 0.15
            other_height = 0.10
            indicator_height = 0.08
        
        # Build row heights list
        row_heights = [price_height, other_height, other_height, other_height]  # Base panels
        row_heights.extend([indicator_height] * indicator_panels)  # Indicator panels
        
        # Build subplot titles
        subplot_titles = [
            "",  # Price chart (title added separately)
            "Strategy vs Benchmark Performance",
            "Drawdowns",
            "Capital Allocation"
        ]
        subplot_titles.extend([f"{indicator} Indicator" for indicator in indicator_panel_indicators])
        
        # Build specs
        specs = [[{"type": "xy"}]] * total_panels
        
        # Configure subplot layout with better spacing for readability
        subplot_config = self.styler.get_subplot_config(
            rows=total_panels,
            vertical_spacing=0.03,  # Increased spacing for better separation
            row_heights=row_heights
        )
        
        # Create subplot structure WITHOUT automatic titles (we'll add custom ones)
        fig = make_subplots(
            **subplot_config,
            subplot_titles=None,  # Don't use automatic titles
            specs=specs
        )
        
        # Calculate domain positions dynamically
        self._set_dynamic_subplot_domains(fig, row_heights)
        
        # Add custom subplot titles and panel separators
        self._add_custom_subplot_titles_and_separators(fig, subplot_titles, row_heights)
        
        # Add chart content
        self._build_price_panel(fig, row=1, log_scale=log_scale)
        self._build_performance_panel(fig, row=2)
        self._build_drawdown_panel(fig, row=3)
        self._build_allocation_panel(fig, row=4)
        self._build_individual_indicators_panels(fig, indicator_panel_indicators, start_row=5)
        
        # Add titles and annotations
        self._add_chart_titles(fig, ticker, base_strategy_name)
        
        # Apply layout and styling
        self._apply_layout_styling(fig, log_scale, total_panels)
        
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
    
    def _set_dynamic_subplot_domains(self, fig: go.Figure, row_heights: list) -> None:
        """
        Set subplot domains dynamically based on row heights.
        
        Args:
            fig: Plotly figure to update
            row_heights: List of relative heights for each row
        """
        # Calculate cumulative positions from bottom up
        total_height = sum(row_heights)
        normalized_heights = [h / total_height for h in row_heights]
        
        # Calculate domains (bottom to top)
        current_bottom = 0.05  # Start from 5% to leave margin
        current_top = 0.95     # End at 95% to leave margin
        
        available_height = current_top - current_bottom
        
        # Calculate gap size based on number of panels (matching vertical_spacing)
        num_panels = len(row_heights)
        gap_size = 0.015 if num_panels > 4 else 0.02  # Increased gaps for better visual separation
        total_gap_height = gap_size * (num_panels - 1)
        
        # Reduce available height by total gap space
        usable_height = available_height - total_gap_height
        
        domains = []
        current_pos = current_bottom
        
        for i, norm_height in enumerate(normalized_heights):
            height = norm_height * usable_height
            domain_bottom = current_pos
            domain_top = current_pos + height
            
            # Ensure domain_top doesn't exceed 1.0
            domain_top = min(domain_top, 0.99)
            domain_bottom = min(domain_bottom, domain_top - 0.01)  # Ensure minimum height
            
            domains.append([domain_bottom, domain_top])
            current_pos = domain_top + gap_size  # Add gap for next panel
        
        # Reverse domains since we want them from top to bottom
        domains.reverse()
        
        # Final validation: ensure all domains are within [0, 1]
        validated_domains = []
        for domain in domains:
            bottom = max(0.0, min(domain[0], 0.99))
            top = max(bottom + 0.01, min(domain[1], 1.0))
            validated_domains.append([bottom, top])
        
        # Apply domains to axes
        axis_updates = {}
        for i, domain in enumerate(validated_domains):
            axis_num = i + 1
            if axis_num == 1:
                axis_updates['yaxis'] = dict(domain=domain)
            else:
                axis_updates[f'yaxis{axis_num}'] = dict(domain=domain)
        
        fig.update_layout(**axis_updates)
    
    def _add_custom_subplot_titles_and_separators(self, fig: go.Figure, subplot_titles: list, row_heights: list) -> None:
        """
        Add custom subplot titles positioned correctly and visual panel separators.
        
        Args:
            fig: Plotly figure to update
            subplot_titles: List of subplot titles
            row_heights: List of relative heights for each row
        """
        # Calculate the same domain positions we used for axes
        total_height = sum(row_heights)
        normalized_heights = [h / total_height for h in row_heights]
        
        current_bottom = 0.05
        current_top = 0.95
        available_height = current_top - current_bottom
        
        num_panels = len(row_heights)
        gap_size = 0.015 if num_panels > 4 else 0.02  # Match the spacing from domain calculation
        total_gap_height = gap_size * (num_panels - 1)
        usable_height = available_height - total_gap_height
        
        domains = []
        current_pos = current_bottom
        
        for i, norm_height in enumerate(normalized_heights):
            height = norm_height * usable_height
            domain_bottom = current_pos
            domain_top = current_pos + height
            
            domain_top = min(domain_top, 0.99)
            domain_bottom = min(domain_bottom, domain_top - 0.01)
            
            domains.append([domain_bottom, domain_top])
            current_pos = domain_top + gap_size
        
        domains.reverse()  # Reverse since we want them from top to bottom
        
        # Final validation
        validated_domains = []
        for domain in domains:
            bottom = max(0.0, min(domain[0], 0.99))
            top = max(bottom + 0.01, min(domain[1], 1.0))
            validated_domains.append([bottom, top])
        
        # Define subtle background colors for different panel types
        panel_colors = [
            "rgba(25, 25, 35, 0.3)",    # Price panel - slightly blue tint
            "rgba(35, 25, 25, 0.3)",    # Performance panel - slightly red tint  
            "rgba(35, 30, 25, 0.3)",    # Drawdown panel - slightly orange tint
            "rgba(25, 35, 25, 0.3)",    # Allocation panel - slightly green tint
        ]
        
        # Add indicator panel colors
        indicator_colors = ["rgba(30, 25, 35, 0.3)", "rgba(35, 35, 25, 0.3)", "rgba(25, 35, 35, 0.3)", "rgba(35, 25, 30, 0.3)"]
        panel_colors.extend(indicator_colors * ((len(validated_domains) - 4) // len(indicator_colors) + 1))
        
        # Add subplot titles positioned correctly within each domain
        for i, (title, domain) in enumerate(zip(subplot_titles, validated_domains)):
            domain_bottom, domain_top = domain
            
            # Add subtle panel background for distinction
            if i < len(panel_colors):
                fig.add_shape(
                    type="rect",
                    x0=0, x1=1,
                    y0=domain_bottom, y1=domain_top,
                    xref="paper", yref="paper",
                    fillcolor=panel_colors[i],
                    line=dict(color="rgba(60, 60, 60, 0.4)", width=0.5),
                    layer="below"
                )
            
            if title:  # Skip empty titles
                # Position title at the top-left of each panel
                title_y = domain_top - 0.015  # Slightly closer to top since no background
                
                fig.add_annotation(
                    text=f"<b>{title}</b>",
                    x=0.02, y=title_y,
                    xref="paper", yref="paper",
                    showarrow=False,
                    font=dict(size=11, color="white", family="Arial"),
                    align="left",
                    bgcolor="rgba(0, 0, 0, 0)",  # Fully transparent background
                )
        
        # Add horizontal separators between panels
        for i in range(len(validated_domains) - 1):
            separator_y = validated_domains[i][0] - gap_size/2  # Middle of the gap
            fig.add_shape(
                type="line",
                x0=0, x1=1,
                y0=separator_y, y1=separator_y,
                xref="paper", yref="paper",
                line=dict(color="rgba(100, 100, 100, 0.6)", width=1, dash="dot")
            )
    
    def _build_individual_indicators_panels(self, fig: go.Figure, indicators: list, start_row: int) -> None:
        """
        Build individual indicator panels, one per indicator.
        
        Args:
            fig: Plotly figure to add indicators to
            indicators: List of indicator names to add
            start_row: Starting row number for indicators
        """
        # Map indicators to their visualization classes
        indicator_map = {
            'MonthlyKDJ': 'MonthlyKDJ',
            'ADX': 'ADX',
            'MFI': 'MFI', 
            'RSI': 'RSI'
        }
        
        for i, indicator in enumerate(indicators):
            row = start_row + i
            
            # Get the specific visualization class
            if indicator in indicator_map:
                viz_class_name = indicator_map[indicator]
                
                # Get the visualization class and apply it
                viz_registry = self.dynamic_indicators._visualization_registry
                if viz_class_name in viz_registry:
                    viz_class = viz_registry[viz_class_name]
                    viz_instance = viz_class(self.dynamic_indicators.data)
                    
                    if viz_instance.check_data_availability():
                        fig, success = viz_instance.add_to_chart(fig, row=row, col=1)
                        
                        if not success:
                            # Add placeholder if indicator failed to render
                            fig.add_annotation(
                                x=0.5, y=0.5,
                                xref=f"x{row} domain", yref=f"y{row} domain",
                                text=f"{indicator} data not available",
                                showarrow=False,
                                font=dict(size=12, color="gray")
                            )
                    else:
                        # Add placeholder for missing data
                        fig.add_annotation(
                            x=0.5, y=0.5,
                            xref=f"x{row} domain", yref=f"y{row} domain",
                            text=f"{indicator} data not available",
                            showarrow=False,
                            font=dict(size=12, color="gray")
                        )
                else:
                    # Add placeholder for missing visualization class
                    fig.add_annotation(
                        x=0.5, y=0.5,
                        xref=f"x{row} domain", yref=f"y{row} domain",
                        text=f"{indicator} visualization not implemented",
                        showarrow=False,
                        font=dict(size=12, color="gray")
                    )
    
    def _build_performance_panel(self, fig: go.Figure, row: int) -> None:
        """Build the performance comparison panel."""
        self.performance.add_performance_comparison(fig, row=row, col=1)
    
    def _build_drawdown_panel(self, fig: go.Figure, row: int) -> None:
        """Build the drawdown comparison panel."""
        self.performance.add_drawdown_comparison(fig, row=row, col=1)
    
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
    
    def _apply_layout_styling(self, fig: go.Figure, log_scale: bool, total_panels: int) -> None:
        """Apply comprehensive layout and styling to the figure."""
        # Get base layout configuration with increased height for better spacing
        layout_config = self.styler.get_dark_theme_layout(
            height=1400 + (total_panels - 5) * 120,  # Increased height for better panel separation
            width=1600,
            top_margin=40  # Slightly more margin for titles
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
        
        # Build axis configurations dynamically
        axis_configs = {}
        
        # X-axes configuration
        for i in range(1, total_panels + 1):
            if i == 1:
                axis_configs['xaxis'] = {**x_axis_config, 'title': "", 'matches': 'x'}
            elif i == total_panels:  # Last panel gets the date label
                axis_configs[f'xaxis{i}'] = {**x_axis_config, 'title': "Date", 'matches': 'x'}
            else:
                axis_configs[f'xaxis{i}'] = {**x_axis_config, 'title': "", 'matches': 'x'}
        
        # Y-axes configuration
        y_axis_labels = [
            ("Price", scale_type),
            ("Performance (%)", "linear"),
            ("Drawdown (%)", "linear"),
            ("Capital ($)", "linear")
        ]
        
        # Add indicator labels for remaining panels
        for i in range(4, total_panels):
            y_axis_labels.append(("Indicator Values", "linear"))
        
        # Apply y-axis configurations
        for i, (label, scale) in enumerate(y_axis_labels):
            axis_num = i + 1
            if axis_num == 1:
                axis_configs['yaxis'] = self.styler.get_axis_config(label, scale, "right")
            else:
                axis_configs[f'yaxis{axis_num}'] = self.styler.get_axis_config(label, scale, "right")
        
        # Apply all configurations
        fig.update_layout(**layout_config, **axis_configs, title="")
    
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
