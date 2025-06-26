"""
Chart styling and theme utilities.
"""
from typing import Dict, Any, Optional
import plotly.graph_objects as go


class ChartStyler:
    """Handles chart styling, themes, and layout configurations."""
    
    # Color scheme constants
    COLORS = {
        'strategy': '#1E90FF',      # DodgerBlue for strategy (was Magenta)
        'benchmark': '#808080',     # Grey for benchmark (was Cyan)
        'buy_signal': '#00FF7F',    # SpringGreen
        'sell_signal': '#FF3030',   # Firebrick red
        'dividend': '#FFD700',      # Gold
        'volume': 'rgba(0, 0, 255, 0.5)',  # Blue with transparency
        'kdj_k': '#4169E1',         # Royal Blue
        'kdj_d': '#FF8C00',         # Dark Orange  
        'kdj_j': '#32CD32',         # Lime Green
        'cash': 'rgba(34, 139, 34, 0.6)',      # Forest green
        'short_proceeds': 'rgba(0, 255, 127, 0.6)', # Spring green (brighter green for proceeds)
        'long_pos': 'rgba(70, 130, 180, 0.6)', # Steel blue
        'short_pos': 'rgba(220, 20, 60, 0.6)', # Crimson
        'margin': 'rgba(255, 165, 0, 0.6)',    # Orange
    }
    
    @staticmethod
    def get_dark_theme_layout(height: int = 1200, width: int = 1600, 
                             top_margin: int = 200) -> Dict[str, Any]:
        """
        Get dark theme layout configuration with improved legend positioning.
        
        Args:
            height: Chart height in pixels
            width: Chart width in pixels  
            top_margin: Top margin for titles and annotations
            
        Returns:
            Dictionary with layout configuration
        """
        return {
            'height': height,
            'width': width,
            'template': "plotly_dark",
            'showlegend': True,
            'legend': dict(
                orientation="h",
                yanchor="bottom", 
                y=-0.055,  # Moved further down to ensure no overlap with Date label
                xanchor="center", 
                x=0.5,  # Center the legend horizontally
                bgcolor="rgba(0, 0, 0, 0.8)",  # Semi-transparent background
                bordercolor="#666666",
                borderwidth=1,
                font=dict(size=9, color="white"),  # Smaller font to reduce space
                itemsizing="constant",  # Consistent item sizing
                itemwidth=30,  # Minimum allowed value
                tracegroupgap=3,  # Reduced space between trace groups
                itemclick="toggleothers",  # Better legend interaction
                itemdoubleclick="toggle"  # Double-click to isolate trace
            ),
            'xaxis_rangeslider_visible': False,
            'plot_bgcolor': '#1E1E1E',
            'paper_bgcolor': '#1E1E1E',
            'font': dict(color='white'),
            'hovermode': 'x unified',
            'margin': dict(t=top_margin, r=60, l=80, b=70)  # Reduced bottom margin from 120px
        }
    
    @staticmethod
    def get_axis_config(title: str = "", axis_type: str = "linear", 
                       side: str = "right") -> Dict[str, Any]:
        """
        Get axis configuration.
        
        Args:
            title: Axis title
            axis_type: Type of axis ('linear', 'log', 'date')
            side: Side to place axis ('left', 'right')
            
        Returns:
            Dictionary with axis configuration
        """
        return {
            'title': title,
            'type': axis_type,
            'gridcolor': '#333333',
            'showgrid': True,
            'showspikes': True,
            'spikecolor': "white",
            'spikethickness': 1,
            'side': side
        }
    
    @staticmethod
    def get_subplot_config(rows: int, vertical_spacing: float = 0.04,
                          row_heights: Optional[list] = None) -> Dict[str, Any]:
        """
        Get subplot configuration.
        
        Args:
            rows: Number of subplot rows
            vertical_spacing: Spacing between rows
            row_heights: List of relative row heights
            
        Returns:
            Dictionary with subplot configuration
        """
        if row_heights is None:
            # Default heights: price gets 35%, others split remaining
            row_heights = [0.35] + [0.65 / (rows - 1)] * (rows - 1) if rows > 1 else [1.0]
        
        return {
            'rows': rows,
            'cols': 1,
            'shared_xaxes': True,
            'vertical_spacing': vertical_spacing,
            'row_heights': row_heights
        }
    
    @classmethod
    def create_annotation(cls, text: str, x: float, y: float, 
                         font_size: int = 12, bgcolor: str = "rgba(50, 50, 50, 0.7)",
                         bordercolor: str = "#999999", xref: str = "paper", 
                         yref: str = "paper") -> Dict[str, Any]:
        """
        Create a styled annotation with flexible coordinate references.
        
        Args:
            text: Annotation text (can include HTML)
            x: X position 
            y: Y position 
            font_size: Font size
            bgcolor: Background color
            bordercolor: Border color
            xref: X coordinate reference ('paper', 'x', etc.)
            yref: Y coordinate reference ('paper', 'y', etc.)
            
        Returns:
            Dictionary with annotation configuration
        """
        return {
            'text': text,
            'xref': xref,
            'yref': yref, 
            'x': x,
            'y': y,
            'showarrow': False,
            'font': dict(size=font_size, color="white", family="Arial"),
            'align': "center",
            'bgcolor': bgcolor,
            'bordercolor': bordercolor,
            'borderwidth': 1,
            'borderpad': 3
        }
    
    @classmethod
    def get_line_style(cls, color: str, width: float = 1.5, 
                      dash: Optional[str] = None) -> Dict[str, Any]:
        """
        Get line style configuration.
        
        Args:
            color: Line color (hex or named)
            width: Line width
            dash: Dash pattern ('solid', 'dash', 'dot', etc.)
            
        Returns:
            Dictionary with line style configuration
        """
        style = {'color': color, 'width': width}
        if dash:
            style['dash'] = dash
        return style
    
    @classmethod 
    def get_marker_style(cls, symbol: str, size: int = 10, color: str = 'blue',
                        line_width: int = 1, line_color: str = 'white') -> Dict[str, Any]:
        """
        Get marker style configuration.
        
        Args:
            symbol: Marker symbol
            size: Marker size
            color: Marker color
            line_width: Border line width
            line_color: Border line color
            
        Returns:
            Dictionary with marker style configuration
        """
        return {
            'symbol': symbol,
            'size': size,
            'color': color,
            'line': dict(width=line_width, color=line_color),
            'opacity': 1.0
        }
