"""
Risk management modules for protecting capital during backtests.
"""
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np

class DrawdownProtection:
    """
    Provides drawdown protection for backtesting, allowing for automatic 
    exits when a position experiences a specified drawdown.
    """
    
    def __init__(self, threshold: float = 0.2):
        """
        Initialize drawdown protection with specified threshold.
        
        Args:
            threshold: Maximum drawdown allowed before exiting (as decimal, 0.2 = 20%)
        """
        self.threshold = threshold
    
    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply drawdown protection by modifying sell signals when drawdown threshold is hit.
        Works for both long and short positions.
        
        Args:
            data: DataFrame with OHLCV data and buy/sell signals
            
        Returns:
            DataFrame with modified sell signals for drawdown protection
        """
        # Create a copy to avoid modifying the original
        result = data.copy()
        
        # Initialize tracking columns
        result['drawdown_since_entry'] = 0.0
        result['peak_price_since_entry'] = np.nan
        result['entry_price'] = np.nan
        
        # Variables for tracking
        position_type = None  # 'long' or 'short'
        entry_price = None
        peak_price = None  # For long: highest price, for short: lowest price
        
        # Process each row
        for i in range(len(result)):
            current_price = result['Close'].iloc[i]
            
            # Check for buy signal (enter long or cover short)
            if result['buy_signal'].iloc[i] == 1:
                if position_type == 'short':
                    # Covering short position - no drawdown tracking needed here
                    pass
                # Enter long position
                position_type = 'long'
                entry_price = current_price
                peak_price = current_price
                result.at[result.index[i], 'entry_price'] = entry_price
            
            # Check for sell signal (enter short or exit long)
            elif result['sell_signal'].iloc[i] == 1:
                if position_type == 'long':
                    # Exiting long position - no drawdown tracking needed here
                    pass
                # Enter short position  
                position_type = 'short'
                entry_price = current_price
                peak_price = current_price  # For shorts, this will be the lowest price
                result.at[result.index[i], 'entry_price'] = entry_price
            
            # Update tracking for active positions
            if position_type == 'long' and entry_price is not None:
                # For long positions: track drawdown from peak
                peak_price = max(peak_price, current_price)
                drawdown = (current_price - peak_price) / peak_price
                
                # Store tracking data
                result.at[result.index[i], 'drawdown_since_entry'] = drawdown
                result.at[result.index[i], 'peak_price_since_entry'] = peak_price
                
                # Apply drawdown protection for long positions
                if drawdown < -self.threshold:
                    result.at[result.index[i], 'sell_signal'] = 1
                    position_type = None
                    entry_price = None
                    peak_price = None
                    
            elif position_type == 'short' and entry_price is not None:
                # For short positions: track drawdown from lowest point
                peak_price = min(peak_price, current_price)  # Lowest price for shorts
                drawdown = (peak_price - current_price) / entry_price  # Profit when price goes down
                
                # For shorts, drawdown is when price goes UP from entry
                actual_drawdown = (current_price - entry_price) / entry_price
                
                # Store tracking data
                result.at[result.index[i], 'drawdown_since_entry'] = actual_drawdown
                result.at[result.index[i], 'peak_price_since_entry'] = peak_price
                
                # Apply drawdown protection for short positions
                if actual_drawdown > self.threshold:  # Note: positive drawdown for shorts
                    result.at[result.index[i], 'buy_signal'] = 1  # Cover short with buy signal
                    position_type = None
                    entry_price = None
                    peak_price = None
                    
            else:
                result.at[result.index[i], 'drawdown_since_entry'] = 0.0
        
        return result
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get configuration parameters for this protection module.
        
        Returns:
            Dictionary with configuration parameters
        """
        return {
            'name': 'DrawdownProtection',
            'threshold': self.threshold
        }


class TrailingStop:
    """
    Implements a trailing stop for protecting profits.
    """
    
    def __init__(self, initial_threshold: float = 0.1, trail_percent: float = 0.5):
        """
        Initialize trailing stop with specified parameters.
        
        Args:
            initial_threshold: Initial stop distance from entry (as decimal)
            trail_percent: Percentage of profit to protect (0.5 = 50% of gains)
        """
        self.initial_threshold = initial_threshold
        self.trail_percent = trail_percent
    
    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply trailing stop by modifying sell signals when stop level is hit.
        
        Args:
            data: DataFrame with OHLCV data and buy/sell signals
            
        Returns:
            DataFrame with modified sell signals for trailing stop protection
        """
        # Create a copy to avoid modifying the original
        result = data.copy()
        
        # Initialize tracking columns
        result['trailing_stop_level'] = np.nan
        
        # Variables for tracking
        in_position = False
        entry_price = None
        highest_price = None
        
        # Process each row
        for i in range(len(result)):
            current_price = result['Close'].iloc[i]
            
            # Check for buy signal
            if result['buy_signal'].iloc[i] == 1:
                in_position = True
                entry_price = current_price
                highest_price = current_price
            
            # Check for existing sell signal
            if result['sell_signal'].iloc[i] == 1:
                in_position = False
                entry_price = None
                highest_price = None
            
            # Update trailing stop for active position
            if in_position and entry_price is not None:
                # Update highest price
                highest_price = max(highest_price, current_price)
                
                # Calculate initial stop level
                initial_stop = entry_price * (1 - self.initial_threshold)
                
                # Calculate trailing stop level based on profit
                profit = highest_price - entry_price
                if profit > 0:
                    trailing_stop = entry_price + (profit * self.trail_percent)
                    # Use the higher of initial stop or trailing stop
                    stop_level = max(initial_stop, trailing_stop)
                else:
                    stop_level = initial_stop
                
                result.at[result.index[i], 'trailing_stop_level'] = stop_level
                
                # Check if stop is hit
                if current_price < stop_level:
                    result.at[result.index[i], 'sell_signal'] = 1
                    in_position = False
                    entry_price = None
                    highest_price = None
        
        return result
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get configuration parameters for this protection module.
        
        Returns:
            Dictionary with configuration parameters
        """
        return {
            'name': 'TrailingStop',
            'initial_threshold': self.initial_threshold,
            'trail_percent': self.trail_percent
        }


class PositionSizing:
    """
    Manages position sizing based on volatility or fixed percentage rules.
    """
    
    def __init__(self, method: str = 'fixed', allocation: float = 1.0, lookback: int = 20):
        """
        Initialize position sizing module.
        
        Args:
            method: 'fixed' for fixed allocation, 'volatility' for volatility-based sizing
            allocation: For fixed method, percentage of capital to use (1.0 = 100%)
            lookback: Lookback period for volatility calculation
        """
        self.method = method
        self.allocation = allocation
        self.lookback = lookback
    
    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply position sizing rules by adding position_size column.
        
        Args:
            data: DataFrame with OHLCV data and signals
            
        Returns:
            DataFrame with added position_size column
        """
        result = data.copy()
        
        if self.method == 'fixed':
            # Fixed allocation
            result['position_size'] = self.allocation
        elif self.method == 'volatility':
            # Volatility-based sizing (inverse volatility)
            result['volatility'] = result['Close'].pct_change().rolling(self.lookback).std()
            # Higher volatility = smaller position size
            result['position_size'] = self.allocation / (1 + result['volatility'].fillna(0.1) * 10)
        else:
            raise ValueError(f"Unknown position sizing method: {self.method}")
        
        return result
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get configuration parameters for this sizing module.
        
        Returns:
            Dictionary with configuration parameters
        """
        return {
            'name': 'PositionSizing',
            'method': self.method,
            'allocation': self.allocation,
            'lookback': self.lookback
        }
