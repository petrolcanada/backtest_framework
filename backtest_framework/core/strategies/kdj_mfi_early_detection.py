"""
KDJ MFI Early Detection strategy implementation.
"""
from typing import List
import pandas as pd
import numpy as np
from backtest_framework.core.strategies.base import BaseStrategy

class KDJMFIEarlyDetectionStrategy(BaseStrategy):
    """
    KDJ MFI Early Detection strategy that generates buy/sell entry signals only.
    
    SIGNAL LOGIC:
    ============
    
    BUY SIGNALS (Golden Cross Period Entry):
    - Must be in golden cross period (cross_status == 1)
    - J line increasing for N consecutive days (j_consecutive_up_days >= required_up_days)
    - D slope is positive (monthly_d_slope > 0) 
    - Haven't exceeded max buy signals per cross (can_generate_buy == 1)
    
    SELL SIGNALS (Death Cross Exit):
    - Death cross occurs (death_cross == 1)
    
    ALTERNATIVE BUY (Death Cross Period - Optional):
    - In death cross period (cross_status == 0)
    - J line flips from negative to positive (j_up_flip == 1)
    - ADX declining for required days (adx_down_condition_met == 1)
    """
    
    def __init__(self, 
                 required_up_days: int = 7,           # J must increase for 7 days
                 required_down_days: int = 4,         # For flip detection
                 required_adx_down_days: int = 5,     # ADX down for alt buy
                 max_buy_signals_per_cross: int = 2,  # Limit signals per golden cross
                 enable_death_cross_buys: bool = False): # Alt buy during death cross
        
        self.required_up_days = required_up_days
        self.required_down_days = required_down_days  
        self.required_adx_down_days = required_adx_down_days
        self.max_buy_signals_per_cross = max_buy_signals_per_cross
        self.enable_death_cross_buys = enable_death_cross_buys
    
    @property
    def required_indicators(self) -> List[str]:
        """All indicators needed, including derived factors."""
        return [
            # Core indicators (Phase 1)
            "MONTHLY_KDJ",
            "MONTHLY_KDJ_SLOPES", 
            "ADX",
            "MFI", 
            "RSI",
            
            # Derived factors (Phase 2)
            "KDJ_CONSECUTIVE_DAYS",      # j_consecutive_up_days, j_up_flip
            "GOLDEN_DEATH_CROSS",        # cross_status, death_cross
            "BUY_SIGNAL_COUNTER",        # can_generate_buy
            "ADX_CONSECUTIVE_DOWN"       # adx_down_condition_met
        ]
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate buy and sell entry signals only.
        
        The framework handles:
        - Position management (when to exit positions)
        - Risk management (stop losses, drawdown protection)
        - Order execution (T+1 delays, etc.)
        """
        # Ensure all required indicators are computed
        data = self.prepare_data(data)
        
        # Initialize signal columns
        data['buy_signal'] = 0
        data['sell_signal'] = 0
        
        # Track buy signal usage per golden cross
        buy_signals_used = {}
        
        # Generate signals row by row
        for i in range(len(data)):
            # === PRIMARY BUY LOGIC (Golden Cross Period) ===
            if self._check_golden_cross_buy_conditions(data, i, buy_signals_used):
                data.iloc[i, data.columns.get_loc('buy_signal')] = 1
                buy_signals_used = self._increment_buy_counter(buy_signals_used, data, i)
                
            # === ALTERNATIVE BUY LOGIC (Death Cross Period) ===
            elif self.enable_death_cross_buys and self._check_death_cross_buy_conditions(data, i):
                data.iloc[i, data.columns.get_loc('buy_signal')] = 1
                
            # === SELL LOGIC (Death Cross Occurrence) ===
            if self._check_sell_conditions(data, i):
                data.iloc[i, data.columns.get_loc('sell_signal')] = 1
                buy_signals_used = self._reset_buy_counter_on_sell(buy_signals_used, data, i)
        
        return data
    
    def _check_golden_cross_buy_conditions(self, data: pd.DataFrame, i: int, buy_signals_used: dict) -> bool:
        """
        Check if golden cross buy conditions are met.
        
        CONDITIONS:
        1. In golden cross period
        2. J increasing for required consecutive days  
        3. D slope positive
        4. Haven't exceeded max signals for this cross
        """
        try:
            # Condition 1: Must be in golden cross period
            if data['cross_status'].iloc[i] != 1:
                return False
                
            # Condition 2: J increasing for required days
            if data['j_consecutive_up_days'].iloc[i] < self.required_up_days:
                return False
                
            # Condition 3: D slope must be positive
            if data['monthly_d_slope'].iloc[i] <= 0:
                return False
                
            # Condition 4: Check buy signal limit for current cross
            current_cross_id = self._get_current_cross_id(data, i)
            signals_used = buy_signals_used.get(current_cross_id, 0)
            if signals_used >= self.max_buy_signals_per_cross:
                return False
                
            return True
            
        except (KeyError, IndexError):
            return False
    
    def _check_death_cross_buy_conditions(self, data: pd.DataFrame, i: int) -> bool:
        """
        Check if death cross period buy conditions are met.
        
        CONDITIONS:
        1. In death cross period  
        2. J line flips from negative to positive
        3. ADX declining for required days
        """
        try:
            # Condition 1: Must be in death cross period
            if data['cross_status'].iloc[i] != 0:
                return False
                
            # Condition 2: J up flip occurred
            if data['j_up_flip'].iloc[i] != 1:
                return False
                
            # Condition 3: ADX down condition met
            if data['adx_down_condition_met'].iloc[i] != 1:
                return False
                
            return True
            
        except (KeyError, IndexError):
            return False
    
    def _check_sell_conditions(self, data: pd.DataFrame, i: int) -> bool:
        """
        Check if sell conditions are met.
        
        CONDITIONS:
        1. Death cross occurs
        """
        try:
            return data['death_cross'].iloc[i] == 1
        except (KeyError, IndexError):
            return False
    
    def _get_current_cross_id(self, data: pd.DataFrame, i: int) -> str:
        """Generate unique ID for current golden cross period."""
        # Find the most recent golden cross
        golden_crosses = data['golden_cross'][:i+1]
        golden_indices = golden_crosses[golden_crosses == 1].index
        
        if len(golden_indices) > 0:
            return f"golden_{golden_indices[-1]}"
        return "no_cross"
    
    def _increment_buy_counter(self, buy_signals_used: dict, data: pd.DataFrame, i: int) -> dict:
        """Increment buy signal counter for current cross."""
        current_cross_id = self._get_current_cross_id(data, i)
        buy_signals_used[current_cross_id] = buy_signals_used.get(current_cross_id, 0) + 1
        return buy_signals_used
    
    def _reset_buy_counter_on_sell(self, buy_signals_used: dict, data: pd.DataFrame, i: int) -> dict:
        """Reset buy counter when sell signal occurs (optional behavior)."""
        # Could reset counters here if desired
        return buy_signals_used
