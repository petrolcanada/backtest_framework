"""
Header component for displaying title and basic information.
"""
import streamlit as st
from typing import Dict, Any
from utils.formatters import get_period_label


class Header:
    """Display header with title and basic information."""
    
    def __init__(self, strategy_info: Dict[str, Any], data, results):
        self.strategy_info = strategy_info
        self.data = data
        self.results = results
    
    def render(self):
        """Render the header component."""
        # Main title
        st.title(f"📈 Backtest Results: {self.strategy_info.get('name', 'Strategy')}")
        
        # Subtitle with basic info
        ticker = self.strategy_info.get('ticker', 'Unknown')
        start_date = self.data.index[0]
        end_date = self.data.index[-1]
        period_label = get_period_label(start_date, end_date)
        
        st.markdown(
            f"**Ticker:** {ticker} | **Period:** {period_label}",
            unsafe_allow_html=True
        )
        
        # Add a subtle separator
        st.markdown("---")
