"""
Main dashboard layout for Streamlit visualization.
"""
import streamlit as st
import pandas as pd
from typing import Dict, Any, Optional
from components import (
    Header, MetricCards, PriceChart, 
    PerformanceChart, DrawdownChart,
    AllocationChart, IndicatorPanels
)


class Dashboard:
    """Main dashboard that coordinates all visualization components."""
    
    def __init__(self, data: pd.DataFrame, results: pd.DataFrame, 
                 engine: Optional[Any] = None, strategy_info: Dict[str, Any] = None):
        """
        Initialize the dashboard.
        
        Args:
            data: DataFrame with OHLCV data and indicators
            results: DataFrame with backtest results
            engine: BacktestEngine instance
            strategy_info: Dictionary with strategy information
        """
        self.data = data
        self.results = results
        self.engine = engine
        self.strategy_info = strategy_info or {}
    
    def render(self):
        """Render the complete dashboard."""
        # Header section
        Header(self.strategy_info, self.data, self.results).render()
        
        # Key metrics section
        MetricCards(self.results, self.engine).render()
        
        # Add spacing
        st.markdown("---")
        
        # Price chart section
        PriceChart(self.data, self.results).render()
        
        # Add spacing
        st.markdown("---")
        
        # Performance analysis section - using Streamlit columns
        st.subheader("📊 Performance Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            PerformanceChart(self.results, self.engine).render()
        
        with col2:
            DrawdownChart(self.results).render()
        
        # Add spacing
        st.markdown("---")
        
        # Capital allocation section
        AllocationChart(self.results).render()
        
        # Add spacing
        st.markdown("---")
        
        # Technical indicators section
        IndicatorPanels(self.data, self.results).render()
        
        # Add spacing
        st.markdown("---")
        
        # Trade statistics section
        self._render_trade_statistics()
        
        # Footer
        self._render_footer()
    
    def _render_trade_statistics(self):
        """Render trade statistics table."""
        st.subheader("📊 Trade Statistics")
        
        # Check if we have trade data
        if 'trade_pnl' not in self.results.columns:
            st.info("No individual trade data available")
            return
        
        # Get trades
        trades = self.results[self.results['trade_pnl'] != 0].copy()
        
        if len(trades) == 0:
            st.info("No completed trades")
            return
        
        # Create summary statistics using Streamlit columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Winning Trades**")
            winning_trades = trades[trades['trade_pnl'] > 0]
            st.metric("Count", len(winning_trades))
            if len(winning_trades) > 0:
                st.metric("Average Win", f"${winning_trades['trade_pnl'].mean():,.2f}")
                st.metric("Largest Win", f"${winning_trades['trade_pnl'].max():,.2f}")
        
        with col2:
            st.markdown("**Losing Trades**")
            losing_trades = trades[trades['trade_pnl'] < 0]
            st.metric("Count", len(losing_trades))
            if len(losing_trades) > 0:
                st.metric("Average Loss", f"${losing_trades['trade_pnl'].mean():,.2f}")
                st.metric("Largest Loss", f"${losing_trades['trade_pnl'].min():,.2f}")
        
        with col3:
            st.markdown("**Overall Statistics**")
            st.metric("Total P&L", f"${trades['trade_pnl'].sum():,.2f}")
            st.metric("Average Trade", f"${trades['trade_pnl'].mean():,.2f}")
            
            # Win streak info
            if len(trades) > 0:
                win_rate = len(winning_trades) / len(trades) * 100
                st.metric("Win Rate", f"{win_rate:.1f}%")
        
        # Add some spacing
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Show recent trades
        st.markdown("**Recent Trades (Last 10)**")
        
        # Prepare trade data for display
        recent_trades = trades.tail(10).copy()
        recent_trades['Date'] = recent_trades.index.strftime('%Y-%m-%d')
        recent_trades['P&L'] = recent_trades['trade_pnl'].apply(lambda x: f"${x:,.2f}")
        
        # Add position type if available
        display_cols = ['Date', 'P&L']
        if 'position_type' in recent_trades.columns:
            recent_trades['Type'] = recent_trades['position_type'].str.title()
            display_cols.insert(1, 'Type')
        
        # Style the dataframe
        def style_pnl(val):
            """Style P&L values with color."""
            if '$' in str(val):
                amount = float(str(val).replace('$', '').replace(',', ''))
                if amount > 0:
                    return 'color: #00FF7F'
                elif amount < 0:
                    return 'color: #FF3030'
            return ''
        
        # Display table with styling
        styled_df = recent_trades[display_cols].style.applymap(
            style_pnl, subset=['P&L']
        )
        
        st.dataframe(
            styled_df,
            use_container_width=True,
            hide_index=True
        )
    
    def _render_footer(self):
        """Render footer with additional information."""
        st.markdown("---")
        
        # Footer with metadata
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.caption("📅 Generated with Backtest Framework")
        
        with col2:
            if self.engine:
                commission_pct = self.engine.commission * 100
                st.caption(f"💰 Commission: {commission_pct:.2f}%")
        
        with col3:
            st.caption("🔧 Powered by Streamlit")
