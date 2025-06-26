"""
Portfolio management module for handling positions, cash, and equity calculations.
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional


class PortfolioManager:
    """
    Manages portfolio state including positions, cash, margin, and equity calculations.
    """
    
    def __init__(self, initial_capital: float, position_sizing: float = 1.0,
                 long_leverage: float = 1.0, short_leverage: float = 1.0):
        """
        Initialize portfolio manager.
        
        Args:
            initial_capital: Starting capital
            position_sizing: Fraction of capital to use per trade
            long_leverage: Leverage for long positions
            short_leverage: Leverage for short positions
        """
        self.initial_capital = initial_capital
        self.position_sizing = position_sizing
        self.long_leverage = long_leverage
        self.short_leverage = short_leverage
        
        # Portfolio state
        self.cash = float(initial_capital)  # Original cash + short proceeds
        self.original_cash = float(initial_capital)  # Track original funds separately
        self.short_proceeds = 0.0  # Track short sale proceeds separately
        self.margin_cash = 0.0  # Cumulative margin debt (negative for long leverage)
        self.position = 0.0
        self.entry_price = 0.0
        self.position_type = ''
        
    def get_current_equity(self, current_price: float) -> float:
        """
        Calculate current portfolio equity.
        
        Args:
            current_price: Current price of the asset
            
        Returns:
            Current equity value
        """
        if self.position > 0:
            # Long position: equity = original_cash + position_value + margin_debt(negative)
            position_value = self.position * current_price
            return self.original_cash + position_value + self.margin_cash
        elif self.position < 0:
            # Short position: Total available cash minus current obligation to buy back shares
            # This properly accounts for the short liability
            current_short_liability = abs(self.position) * current_price
            total_cash_available = self.original_cash + self.short_proceeds
            return total_cash_available - current_short_liability
        else:
            # No position: equity = original_cash + short_proceeds + margin_debt
            return self.original_cash + self.short_proceeds + self.margin_cash
    
    def get_position_value(self, current_price: float) -> Dict[str, float]:
        """
        Get breakdown of position values.
        
        Args:
            current_price: Current price of the asset
            
        Returns:
            Dictionary with position value breakdown
        """
        if self.position > 0:
            return {
                'long_value': self.position * current_price,
                'short_value': 0.0
            }
        elif self.position < 0:
            return {
                'long_value': 0.0,
                'short_value': abs(self.position) * current_price
            }
        else:
            return {
                'long_value': 0.0,
                'short_value': 0.0
            }
    
    def enter_long_position(self, execution_price: float) -> bool:
        """
        Enter a long position.
        
        Args:
            execution_price: Price at which to execute the trade
            
        Returns:
            True if position was entered successfully
        """
        # Close any existing short position first
        if self.position < 0:
            self._close_short_position(execution_price)
        
        if self.position <= 0:
            # Use total portfolio value for leverage calculation
            current_portfolio_value = self.original_cash + self.short_proceeds + self.margin_cash
            desired_position_value = current_portfolio_value * self.position_sizing * self.long_leverage
            
            # Calculate how much cash we have available vs need to borrow
            available_cash = self.original_cash + self.short_proceeds
            margin_to_borrow = max(0, desired_position_value - available_cash)
            
            # Actual position value
            position_value = min(desired_position_value, available_cash + margin_to_borrow)
            shares = position_value / execution_price
            
            # Update state - use cash first, then margin
            cash_used = min(available_cash, position_value)
            margin_borrowed = position_value - cash_used
            
            self.position = shares
            
            # Reduce cash proportionally from original_cash and short_proceeds
            if available_cash > 0:
                original_ratio = self.original_cash / available_cash
                short_ratio = self.short_proceeds / available_cash
                
                self.original_cash -= cash_used * original_ratio
                self.short_proceeds -= cash_used * short_ratio
            
            self.margin_cash -= margin_borrowed  # Add margin debt (negative)
            self.cash = self.original_cash + self.short_proceeds
            self.entry_price = execution_price
            self.position_type = 'long'
            
            return True
        return False
    
    def enter_short_position(self, execution_price: float) -> bool:
        """
        Enter a short position with proper MMF interest handling.
        
        When shorting with 2x leverage on $100 fund:
        - Short $200 worth of shares
        - Original $100 earns MMF interest
        - $200 short proceeds earn MMF interest
        - $200 short position bears borrowing costs
        
        Args:
            execution_price: Price at which to execute the trade
            
        Returns:
            True if position was entered successfully
        """
        # Close any existing long position first
        if self.position > 0:
            self._close_long_position(execution_price)
        
        if self.position >= 0:
            # Calculate short position value based on current portfolio value and leverage
            current_portfolio_value = self.original_cash + self.short_proceeds + self.margin_cash
            desired_short_value = current_portfolio_value * self.position_sizing * self.short_leverage
            
            # Calculate shares to short
            shares_to_short = desired_short_value / execution_price
            
            # Update state
            self.position = -shares_to_short
            self.short_proceeds += desired_short_value  # Add short proceeds
            self.cash = self.original_cash + self.short_proceeds  # Update total cash
            self.entry_price = execution_price
            self.position_type = 'short'
            
            return True
        return False
    
    def _close_long_position(self, execution_price: float) -> float:
        """
        Close long position and return P&L.
        
        Args:
            execution_price: Price at which to execute the trade
            
        Returns:
            P&L from the trade
        """
        if self.position <= 0:
            return 0.0
        
        # Calculate sale proceeds
        sale_proceeds = self.position * execution_price
        
        # Pay back margin debt first
        if self.margin_cash < 0:
            margin_to_repay = min(-self.margin_cash, sale_proceeds)
            self.margin_cash += margin_to_repay
            sale_proceeds -= margin_to_repay
        
        # Add remaining to original cash
        self.original_cash += sale_proceeds
        self.cash = self.original_cash + self.short_proceeds
        
        # Calculate P&L
        total_cost = self.position * self.entry_price
        pnl = sale_proceeds - (total_cost - abs(min(self.margin_cash, 0)))
        
        self.position = 0.0
        return pnl
    
    def _close_short_position(self, execution_price: float) -> float:
        """
        Close short position and return P&L.
        
        Args:
            execution_price: Price at which to execute the trade
            
        Returns:
            P&L from the trade
        """
        if self.position >= 0:
            return 0.0
        
        # Calculate cost to cover short position
        cover_cost = abs(self.position) * execution_price
        
        # Calculate P&L on short position  
        pnl = abs(self.position) * (self.entry_price - execution_price)
        
        # Remove the short proceeds that were used to cover
        short_value = abs(self.position) * self.entry_price
        self.short_proceeds -= short_value
        
        # Net P&L goes to original cash
        self.original_cash += pnl
        self.cash = self.original_cash + self.short_proceeds
        
        self.position = 0.0
        return pnl
    
    def process_dividend(self, dividend_per_share: float) -> float:
        """
        Process dividend payment.
        
        Args:
            dividend_per_share: Dividend amount per share
            
        Returns:
            Net dividend impact (positive for long, negative for short)
        """
        if self.position == 0 or dividend_per_share <= 0:
            return 0.0
        
        if self.position > 0:
            # Long position receives dividends - add to original cash
            dividend_payment = self.position * dividend_per_share
            self.original_cash += dividend_payment
            self.cash = self.original_cash + self.short_proceeds
            return dividend_payment
        else:
            # Short position pays dividends - subtract from short proceeds
            dividend_payment = abs(self.position) * dividend_per_share
            self.short_proceeds -= dividend_payment
            self.cash = self.original_cash + self.short_proceeds
            return -dividend_payment
    
    def apply_margin_cost(self, daily_margin_rate: float) -> float:
        """
        Apply daily margin interest cost.
        
        Args:
            daily_margin_rate: Daily margin interest rate
            
        Returns:
            Daily margin cost
        """
        if self.margin_cash >= 0:
            return 0.0
        
        daily_cost = -self.margin_cash * daily_margin_rate
        self.margin_cash -= daily_cost
        return daily_cost
    
    def apply_short_borrow_cost(self, current_price: float, daily_borrow_rate: float) -> float:
        """
        Apply daily short borrowing cost (reduces short proceeds).
        
        Args:
            current_price: Current price of the asset
            daily_borrow_rate: Daily borrowing rate
            
        Returns:
            Daily borrowing cost
        """
        if self.position >= 0:
            return 0.0
        
        borrowed_value = abs(self.position) * current_price
        daily_cost = borrowed_value * daily_borrow_rate
        
        # Borrowing cost reduces short proceeds
        self.short_proceeds -= daily_cost
        self.cash = self.original_cash + self.short_proceeds
        
        return daily_cost
    
    def apply_mmf_interest(self, daily_mmf_rate: float) -> float:
        """
        Apply MMF interest to cash components.
        
        Args:
            daily_mmf_rate: Daily MMF interest rate
            
        Returns:
            Total daily MMF interest earned
        """
        # Apply MMF interest to original cash
        original_cash_interest = self.original_cash * daily_mmf_rate
        self.original_cash += original_cash_interest
        
        # Apply MMF interest to short proceeds (if any)
        short_proceeds_interest = self.short_proceeds * daily_mmf_rate
        self.short_proceeds += short_proceeds_interest
        
        # Update total cash
        old_total_cash = self.cash
        self.cash = self.original_cash + self.short_proceeds
        
        total_interest = original_cash_interest + short_proceeds_interest
        
        # Debug: Print first few applications to verify
        if hasattr(self, '_debug_mmf_counter'):
            self._debug_mmf_counter += 1
        else:
            self._debug_mmf_counter = 1
            
        if self._debug_mmf_counter <= 3:
            print(f"DEBUG MMF Applied Day {self._debug_mmf_counter}: Rate={daily_mmf_rate:.8f}, Original=${self.original_cash-original_cash_interest:.2f}→${self.original_cash:.2f} (+${original_cash_interest:.4f}), Total Interest=${total_interest:.4f}")
        
        return total_interest
    
    def reset(self):
        """Reset portfolio to initial state."""
        self.cash = float(self.initial_capital)
        self.original_cash = float(self.initial_capital)
        self.short_proceeds = 0.0
        self.margin_cash = 0.0
        self.position = 0.0
        self.entry_price = 0.0
        self.position_type = ''
