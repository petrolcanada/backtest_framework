"""
Cost calculation module for trading costs, margin interest, and borrowing fees.
"""
import pandas as pd
import numpy as np
from typing import Optional


class CostCalculator:
    """
    Calculates various trading costs including commission, slippage, margin interest, and borrowing fees.
    """
    
    def __init__(self, commission: float = 0.001, slippage: float = 0.0,
                 short_borrow_rate: float = 0.02, use_margin_costs: bool = True):
        """
        Initialize cost calculator.
        
        Args:
            commission: Commission rate as decimal (e.g., 0.001 for 0.1%)
            slippage: Slippage rate as decimal (e.g., 0.001 for 0.1%)
            short_borrow_rate: Annual short borrowing rate as decimal (e.g., 0.02 for 2%)
            use_margin_costs: Whether to include margin interest costs
        """
        self.commission = commission
        self.slippage = slippage
        # Ensure short_borrow_rate is in absolute form (convert if in percentage form)
        self.short_borrow_rate = short_borrow_rate / 100.0 if short_borrow_rate > 1 else short_borrow_rate
        self.use_margin_costs = use_margin_costs
        self.ffr_data = None
    
    def load_ffr_data(self, ffr_data: Optional[pd.Series]):
        """
        Load Federal Funds Rate data for margin cost calculations.
        
        Args:
            ffr_data: Series with FFR data indexed by date
        """
        self.ffr_data = ffr_data
    
    def calculate_execution_price(self, market_price: float, is_buy: bool) -> float:
        """
        Calculate execution price including commission and slippage.
        
        Args:
            market_price: Market price of the asset
            is_buy: True for buy orders, False for sell orders
            
        Returns:
            Effective execution price
        """
        if is_buy:
            return market_price * (1 + self.commission + self.slippage)
        else:
            return market_price * (1 - self.commission - self.slippage)
    
    def get_daily_margin_rate(self, date: pd.Timestamp) -> float:
        """
        Get daily margin interest rate for a given date.
        
        Args:
            date: Date to get rate for
            
        Returns:
            Daily margin rate (annual rate / 252 trading days)
        """
        if not self.use_margin_costs:
            return 0.0
        
        annual_rate = 0.02  # Default 2.0% in absolute form (0.02)
        
        if self.ffr_data is not None and not self.ffr_data.empty:
            if date in self.ffr_data.index:
                ffr_rate = self.ffr_data.loc[date]  # FFR is in absolute form (e.g., 0.04 for 4%)
                annual_rate = ffr_rate + 0.015  # FFR + 1.5% spread (0.015)
            else:
                # Use most recent available rate
                available_dates = self.ffr_data.index[self.ffr_data.index <= date]
                if not available_dates.empty:
                    recent_date = available_dates.max()
                    ffr_rate = self.ffr_data.loc[recent_date]
                    annual_rate = ffr_rate + 0.015  # FFR + 1.5% spread (0.015)
        
        return annual_rate / 252  # Already in decimal form, just convert to daily
    
    def get_daily_short_borrow_rate(self, date: pd.Timestamp) -> float:
        """
        Get daily short borrowing rate based on FFR + spread.
        
        Args:
            date: Date to get rate for
            
        Returns:
            Daily short borrow rate (FFR + spread / 252 trading days)
        """
        # Default to static rate if no FFR data available
        annual_rate = self.short_borrow_rate  # Default fallback to original rate
        
        if self.ffr_data is not None and not self.ffr_data.empty:
            if date in self.ffr_data.index:
                ffr_rate = self.ffr_data.loc[date]  # FFR is in absolute form (e.g., 0.04 for 4%)
                # Convert static short_borrow_rate to FFR + spread
                # If short_borrow_rate is 0.02 (2%), treat as FFR + 2.0% spread
                spread = self.short_borrow_rate  # Treat original rate as the spread over FFR
                annual_rate = ffr_rate + spread
            else:
                # Use most recent available rate
                available_dates = self.ffr_data.index[self.ffr_data.index <= date]
                if not available_dates.empty:
                    recent_date = available_dates.max()
                    ffr_rate = self.ffr_data.loc[recent_date]
                    spread = self.short_borrow_rate  # Treat original rate as the spread over FFR
                    annual_rate = ffr_rate + spread
        
        return annual_rate / 252  # Convert to daily rate
    
    def get_daily_mmf_rate(self, date: pd.Timestamp) -> float:
        """
        Get daily Money Market Fund (MMF) interest rate.
        
        Args:
            date: Date to get rate for
            
        Returns:
            Daily MMF rate (typically FFR - small spread / 252 trading days)
        """
        annual_rate = 0.04  # Default 4.0% in absolute form (0.04)
        
        if self.ffr_data is not None and not self.ffr_data.empty:
            if date in self.ffr_data.index:
                ffr_rate = self.ffr_data.loc[date]  # FFR is already in absolute form (0.04 for 4%)
                annual_rate = max(0.001, ffr_rate - 0.0025)  # FFR - 0.25% spread for MMF, minimum 0.1%
            else:
                # Use most recent available rate
                available_dates = self.ffr_data.index[self.ffr_data.index <= date]
                if not available_dates.empty:
                    recent_date = available_dates.max()
                    ffr_rate = self.ffr_data.loc[recent_date]  # Already in absolute form
                    annual_rate = max(0.001, ffr_rate - 0.0025)  # FFR - 0.25% spread, minimum 0.1%
        
        daily_rate = annual_rate / 252  # Already in decimal form, just convert to daily
        
        return daily_rate
    
    def calculate_margin_cost(self, margin_cash_debt: float, date: pd.Timestamp) -> float:
        """
        Calculate daily margin interest cost.
        
        Args:
            margin_cash_debt: Amount of margin debt (positive value from -margin_cash)
            date: Current date
            
        Returns:
            Daily margin cost
        """
        if margin_cash_debt <= 0:
            return 0.0
        
        daily_rate = self.get_daily_margin_rate(date)
        return margin_cash_debt * daily_rate
    
    def calculate_short_borrow_cost(self, borrowed_value: float) -> float:
        """
        Calculate daily short borrowing cost.
        
        Args:
            borrowed_value: Value of borrowed shares
            
        Returns:
            Daily borrowing cost
        """
        if borrowed_value <= 0:
            return 0.0
        
        daily_rate = self.get_daily_short_borrow_rate()
        return borrowed_value * daily_rate
