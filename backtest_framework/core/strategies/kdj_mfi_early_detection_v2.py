"""
Enhanced KDJ MFI Early Detection strategy with indicator parameter support.

CLEANED VERSION: Removed BUY_SIGNAL_COUNTER logic to show all signals without limiting.
"""
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from backtest_framework.core.strategies.base import BaseStrategy
from backtest_framework.core.utils.helpers import clean_params, filter_empty_dicts

class KDJMFIEarlyDetectionStrategy(BaseStrategy):
    """
    Enhanced KDJ MFI Early Detection strategy with configurable indicator parameters.
    
    SIGNAL LOGIC:
    ============
    
    BUY SIGNALS (Golden Cross Period Entry):
    - Must be in golden cross period (cross_status == 1)
    - J line increasing for N consecutive days (j_consecutive_up_days >= required_up_days)
    - D slope is positive (monthly_d_slope > 0) 
    
    SELL SIGNALS (Death Cross Exit):
    - Death cross occurs (death_cross == 1)
    
    ALTERNATIVE BUY (Death Cross Period - Optional):
    - In death cross period (cross_status == 0)
    - J line flips from negative to positive (j_up_flip == 1)
    - ADX declining for required days (adx_down_condition_met == 1)
    
    NOTE: Signal limiting has been removed to show all qualifying signals.
    """
    
    def __init__(self, 
                 # Strategy parameters
                 required_up_days: int = 7,           # J must increase for 7 days
                 required_down_days: int = 4,         # For flip detection
                 required_adx_down_days: int = 5,     # ADX down for alt buy
                 enable_death_cross_buys: bool = False, # Alt buy during death cross
                 
                 # Indicator parameter overrides (fully standardized to use 'period')
                 adx_period: int = 14*22,                   # ADX calculation period (default: ~14 months)
                 adx_sma_period: int = 5,                 # ADX SMA smoothing period
                 kdj_period: int = 198,                   # KDJ lookback period (default: 9 months * 22 days)
                 kdj_signal: int = 66,                    # KDJ signal period (default: 3 months * 22 days)
                 mfi_period: int = 48,                    # MFI calculation period (default: ~2.2 months)
                 rsi_period: int = 14):                   # RSI calculation period
        
        # Initialize strategy parameters
        self.required_up_days = required_up_days
        self.required_down_days = required_down_days  
        self.required_adx_down_days = required_adx_down_days
        self.enable_death_cross_buys = enable_death_cross_buys
        
        # Build indicator parameter overrides in a single dictionary
        indicator_overrides = {
            # Core indicator parameters (fully standardized to use 'period')
            'ADX': clean_params(period=adx_period, sma_period=adx_sma_period),
            'MONTHLY_KDJ': clean_params(period=kdj_period, signal=kdj_signal),
            'MFI': clean_params(period=mfi_period),
            'RSI': clean_params(period=rsi_period),
            
            # Strategy-derived indicator parameters (removed BUY_SIGNAL_COUNTER)
            'ADX_CONSECUTIVE_DOWN': {'required_days': self.required_adx_down_days},
            'KDJ_CONSECUTIVE_DAYS': {'up_days': self.required_up_days, 'down_days': self.required_down_days}
        }
        
        # Filter out any empty parameter dictionaries
        indicator_overrides = filter_empty_dicts(indicator_overrides)
        
        # Initialize base strategy with indicator parameters
        super().__init__(indicator_params=indicator_overrides)
    
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
            
            # Derived factors (Phase 2) - removed BUY_SIGNAL_COUNTER
            "KDJ_CONSECUTIVE_DAYS",      # j_consecutive_up_days, j_up_flip
            "GOLDEN_DEATH_CROSS",        # cross_status, death_cross
            "ADX_CONSECUTIVE_DOWN"       # adx_down_condition_met
        ]
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate buy and sell entry signals without any limiting.
        
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
        
        # Generate signals row by row
        for i in range(len(data)):
            current_date = data.index[i].strftime('%Y-%m-%d')
            
            # === PRIMARY BUY LOGIC (Golden Cross Period) ===
            if self._check_golden_cross_buy_conditions(data, i):
                data.iloc[i, data.columns.get_loc('buy_signal')] = 1
                
            # === ALTERNATIVE BUY LOGIC (Death Cross Period) ===
            elif self.enable_death_cross_buys and self._check_death_cross_buy_conditions(data, i):
                data.iloc[i, data.columns.get_loc('buy_signal')] = 1
                
            # === SELL LOGIC (Death Cross Occurrence) ===
            if self._check_sell_conditions(data, i):
                data.iloc[i, data.columns.get_loc('sell_signal')] = 1
        
        return data
    
    def _check_golden_cross_buy_conditions(self, data: pd.DataFrame, i: int) -> bool:
        """
        Check if golden cross buy conditions are met.
        
        CONDITIONS:
        1. In golden cross period
        2. J increasing for required consecutive days  
        3. D slope positive
        
        REMOVED: Signal limiting logic
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
            
            # Condition 3: D slope must be positive
            if data['monthly_d'].iloc[i] > 30 or data['monthly_j'].iloc[i] > 30:
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
                

            # Condition 3: J up less than 50
            if data['monthly_j'].iloc[i] > 50:
                return False
            
            # # Condition 3: ADX down condition met
            # if data['adx_down_condition_met'].iloc[i] != 1:
            #     return False
                
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
            # return data['death_cross'].iloc[i] == 1
            return data['monthly_j'].iloc[i] >= 100 or data['death_cross'].iloc[i] == 1
        except (KeyError, IndexError):
            return False
