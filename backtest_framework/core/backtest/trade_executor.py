"""
Trade execution module for handling buy/sell signals and position management.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from .portfolio_manager import PortfolioManager
from .cost_calculator import CostCalculator


class TradeExecutor:
    """
    Handles trade execution logic including signal processing and position management.
    """
    
    def __init__(self, portfolio_manager: PortfolioManager, cost_calculator: CostCalculator,
                 enable_short_selling: bool = False):
        """
        Initialize trade executor.
        
        Args:
            portfolio_manager: Portfolio manager instance
            cost_calculator: Cost calculator instance
            enable_short_selling: Whether short selling is enabled
        """
        self.portfolio_manager = portfolio_manager
        self.cost_calculator = cost_calculator
        self.enable_short_selling = enable_short_selling
        
        # Trade tracking
        self.trade_count = 0
        self.win_count = 0
        self.loss_count = 0
        
        # Dividend and cost tracking
        self.cumulative_dividends = 0.0
        self.cumulative_borrowing_costs = 0.0
    
    def prepare_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare T+1 execution signals.
        
        Args:
            data: DataFrame with buy_signal and sell_signal columns
            
        Returns:
            DataFrame with T+1 shifted signals
        """
        data = data.copy()
        
        # Validate signal columns
        if 'buy_signal' not in data.columns or 'sell_signal' not in data.columns:
            raise ValueError("Data must contain 'buy_signal' and 'sell_signal' columns")
        
        # Shift signals for T+1 execution
        data['buy_signal_t1'] = data['buy_signal'].shift(1).fillna(0).astype(int)
        data['sell_signal_t1'] = data['sell_signal'].shift(1).fillna(0).astype(int)
        
        return data
    
    def execute_buy_signal(self, current_price: float, execution_price: float) -> Dict[str, Any]:
        """
        Execute buy signal.
        
        Args:
            current_price: Current market price
            execution_price: Price for execution (usually Open price)
            
        Returns:
            Dictionary with execution results
        """
        result = {
            'executed': False,
            'trade_occurred': False,
            'pnl': 0.0,
            'trade_type': ''
        }
        
        effective_price = self.cost_calculator.calculate_execution_price(execution_price, True)
        
        # Close short position first if exists
        if self.portfolio_manager.position < 0:
            pnl = self.portfolio_manager._close_short_position(effective_price)
            self._update_trade_stats(pnl)
            result['trade_occurred'] = True
            result['pnl'] = pnl
            result['trade_type'] = 'close_short'
        
        # Enter long position
        if self.portfolio_manager.position <= 0:
            success = self.portfolio_manager.enter_long_position(effective_price)
            if success:
                self.trade_count += 1
                result['executed'] = True
                if not result['trade_occurred']:
                    result['trade_type'] = 'open_long'
        
        return result
    
    def execute_sell_signal(self, current_price: float, execution_price: float) -> Dict[str, Any]:
        """
        Execute sell signal.
        
        Args:
            current_price: Current market price
            execution_price: Price for execution (usually Open price)
            
        Returns:
            Dictionary with execution results
        """
        result = {
            'executed': False,
            'trade_occurred': False,
            'pnl': 0.0,
            'trade_type': ''
        }
        
        effective_price = self.cost_calculator.calculate_execution_price(execution_price, False)
        
        # Close long position first if exists
        if self.portfolio_manager.position > 0:
            pnl = self.portfolio_manager._close_long_position(effective_price)
            self._update_trade_stats(pnl)
            result['trade_occurred'] = True
            result['pnl'] = pnl
            result['trade_type'] = 'close_long'
        
        # Enter short position if enabled
        if (self.portfolio_manager.position >= 0 and self.enable_short_selling):
            success = self.portfolio_manager.enter_short_position(effective_price)
            if success:
                self.trade_count += 1
                result['executed'] = True
                if not result['trade_occurred']:
                    result['trade_type'] = 'open_short'
        
        return result
    
    def process_daily_costs_and_dividends(self, current_price: float, current_date: pd.Timestamp,
                                         current_dividend: float = 0.0) -> Dict[str, float]:
        """
        Process daily costs and dividend payments.
        
        Args:
            current_price: Current price of the asset
            current_date: Current date
            current_dividend: Dividend per share for current date
            
        Returns:
            Dictionary with cost and dividend information
        """
        result = {
            'dividend_impact': 0.0,
            'margin_cost': 0.0,
            'short_borrow_cost': 0.0,
            'mmf_interest': 0.0
        }
        
        # Process dividends
        if current_dividend > 0:
            dividend_impact = self.portfolio_manager.process_dividend(current_dividend)
            result['dividend_impact'] = dividend_impact
            self.cumulative_dividends += dividend_impact
        
        # Apply MMF interest to cash components
        daily_mmf_rate = self.cost_calculator.get_daily_mmf_rate(current_date)
        mmf_interest = self.portfolio_manager.apply_mmf_interest(daily_mmf_rate)
        result['mmf_interest'] = mmf_interest
        
        # Calculate and apply margin costs
        if self.portfolio_manager.margin_cash < 0:
            daily_margin_rate = self.cost_calculator.get_daily_margin_rate(current_date)
            margin_cost = self.portfolio_manager.apply_margin_cost(daily_margin_rate)
            result['margin_cost'] = margin_cost
            self.cumulative_borrowing_costs += margin_cost
        
        # Calculate and apply short borrowing costs
        if self.portfolio_manager.position < 0:
            daily_borrow_rate = self.cost_calculator.get_daily_short_borrow_rate(current_date)
            short_cost = self.portfolio_manager.apply_short_borrow_cost(
                current_price, daily_borrow_rate)
            result['short_borrow_cost'] = short_cost
            self.cumulative_borrowing_costs += short_cost
        
        return result
    
    def _update_trade_stats(self, pnl: float):
        """
        Update trade statistics based on P&L.
        
        Args:
            pnl: Profit/Loss from the trade
        """
        self.trade_count += 1
        if pnl > 0:
            self.win_count += 1
        else:
            self.loss_count += 1
    
    def get_trade_stats(self) -> Dict[str, Any]:
        """
        Get current trade statistics.
        
        Returns:
            Dictionary with trade statistics
        """
        return {
            'trade_count': self.trade_count,
            'win_count': self.win_count,
            'loss_count': self.loss_count,
            'cumulative_dividends': self.cumulative_dividends,
            'cumulative_borrowing_costs': self.cumulative_borrowing_costs
        }
    
    def reset(self):
        """Reset executor state."""
        self.trade_count = 0
        self.win_count = 0
        self.loss_count = 0
        self.cumulative_dividends = 0.0
        self.cumulative_borrowing_costs = 0.0
        self.portfolio_manager.reset()
