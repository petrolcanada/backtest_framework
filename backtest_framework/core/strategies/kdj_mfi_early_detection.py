"""
Enhanced KDJ MFI Early Detection strategy with indicator parameter support.
"""
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from backtest_framework.core.strategies.base import BaseStrategy

class KDJMFIEarlyDetectionStrategy(BaseStrategy):
    """
    Enhanced KDJ MFI Early Detection strategy with configurable indicator parameters.
    
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
                 # Strategy parameters
                 required_up_days: int = 7,           # J must increase for 7 days
                 required_down_days: int = 4,         # For flip detection
                 required_adx_down_days: int = 5,     # ADX down for alt buy
                 max_buy_signals_per_cross: int = 2,  # Limit signals per golden cross
                 enable_death_cross_buys: bool = False, # Alt buy during death cross
                 
                 # Indicator parameter overrides
                 adx_window: Optional[int] = None,        # ADX calculation window
                 adx_sma_length: Optional[int] = None,    # ADX SMA smoothing length
                 kdj_window: Optional[int] = None,        # KDJ calculation window
                 kdj_k_period: Optional[int] = None,      # KDJ K period
                 kdj_d_period: Optional[int] = None,      # KDJ D period
                 mfi_window: Optional[int] = None,        # MFI calculation window
                 rsi_window: Optional[int] = None,        # RSI calculation window
                 
                 # Custom indicator parameters (full override)
                 indicator_params: Optional[Dict[str, Dict[str, Any]]] = None):
        
        # Initialize strategy parameters
        self.required_up_days = required_up_days
        self.required_down_days = required_down_days  
        self.required_adx_down_days = required_adx_down_days
        self.max_buy_signals_per_cross = max_buy_signals_per_cross
        self.enable_death_cross_buys = enable_death_cross_buys
        
        # Build indicator parameter overrides
        indicator_overrides = {}
        
        # ADX parameters
        if adx_window is not None or adx_sma_length is not None:
            adx_params = {}
            if adx_window is not None:
                adx_params['window'] = adx_window
            if adx_sma_length is not None:
                adx_params['sma_length'] = adx_sma_length
            indicator_overrides['ADX'] = adx_params
        
        # ADX consecutive down parameters (sync with strategy parameter)
        indicator_overrides['ADX_CONSECUTIVE_DOWN'] = {
            'required_days': self.required_adx_down_days
        }
        
        # KDJ parameters
        if kdj_window is not None or kdj_k_period is not None or kdj_d_period is not None:
            kdj_params = {}
            if kdj_window is not None:
                kdj_params['window'] = kdj_window
            if kdj_k_period is not None:
                kdj_params['k_period'] = kdj_k_period
            if kdj_d_period is not None:
                kdj_params['d_period'] = kdj_d_period
            indicator_overrides['MONTHLY_KDJ'] = kdj_params
            indicator_overrides['MONTHLY_KDJ_SLOPES'] = kdj_params
        
        # KDJ consecutive days parameters (sync with strategy parameters)
        indicator_overrides['KDJ_CONSECUTIVE_DAYS'] = {
            'up_days': self.required_up_days,
            'down_days': self.required_down_days
        }
        
        # Buy signal counter parameters (sync with strategy parameter)
        indicator_overrides['BUY_SIGNAL_COUNTER'] = {
            'max_signals_per_cross': self.max_buy_signals_per_cross
        }
        
        # MFI parameters
        if mfi_window is not None:
            indicator_overrides['MFI'] = {'window': mfi_window}
        
        # RSI parameters
        if rsi_window is not None:
            indicator_overrides['RSI'] = {'window': rsi_window}
        
        # Merge with custom indicator parameters if provided
        if indicator_params:
            for indicator, params in indicator_params.items():
                if indicator in indicator_overrides:
                    indicator_overrides[indicator].update(params)
                else:
                    indicator_overrides[indicator] = params
        
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
            
            # Derived factors (Phase 2)
            "KDJ_CONSECUTIVE_DAYS",      # j_consecutive_up_days, j_up_flip
            "GOLDEN_DEATH_CROSS",        # cross_status, death_cross
            "BUY_SIGNAL_COUNTER",        # can_generate_buy
            "ADX_CONSECUTIVE_DOWN"       # adx_down_condition_met
        ]
    
    def set_adx_params(self, window: Optional[int] = None, sma_length: Optional[int] = None, 
                       consecutive_down_days: Optional[int] = None) -> None:
        """
        Convenience method to set ADX-related parameters.
        
        Args:
            window: ADX calculation window
            sma_length: ADX SMA smoothing length  
            consecutive_down_days: Required consecutive down days for ADX condition
        """
        if window is not None or sma_length is not None:
            adx_params = self.get_indicator_params('ADX')
            if window is not None:
                adx_params['window'] = window
            if sma_length is not None:
                adx_params['sma_length'] = sma_length
            self.set_indicator_params('ADX', **adx_params)
        
        if consecutive_down_days is not None:
            self.required_adx_down_days = consecutive_down_days
            self.set_indicator_params('ADX_CONSECUTIVE_DOWN', required_days=consecutive_down_days)
    
    def set_kdj_params(self, window: Optional[int] = None, k_period: Optional[int] = None, 
                       d_period: Optional[int] = None, required_up_days: Optional[int] = None,
                       required_down_days: Optional[int] = None) -> None:
        """
        Convenience method to set KDJ-related parameters.
        
        Args:
            window: KDJ calculation window
            k_period: KDJ K smoothing period
            d_period: KDJ D smoothing period
            required_up_days: Required consecutive up days for J momentum
            required_down_days: Required down days for flip detection
        """
        if window is not None or k_period is not None or d_period is not None:
            kdj_params = {}
            if window is not None:
                kdj_params['window'] = window
            if k_period is not None:
                kdj_params['k_period'] = k_period
            if d_period is not None:
                kdj_params['d_period'] = d_period
            
            # Apply to both KDJ indicators
            self.set_indicator_params('MONTHLY_KDJ', **kdj_params)
            self.set_indicator_params('MONTHLY_KDJ_SLOPES', **kdj_params)
        
        if required_up_days is not None or required_down_days is not None:
            consecutive_params = self.get_indicator_params('KDJ_CONSECUTIVE_DAYS')
            if required_up_days is not None:
                self.required_up_days = required_up_days
                consecutive_params['up_days'] = required_up_days
            if required_down_days is not None:
                self.required_down_days = required_down_days
                consecutive_params['down_days'] = required_down_days
            self.set_indicator_params('KDJ_CONSECUTIVE_DAYS', **consecutive_params)
    
    def print_indicator_config(self) -> None:
        """Print current indicator configuration for debugging."""
        info = self.get_strategy_info()
        
        print("Strategy Indicator Configuration:")
        print("=" * 50)
        print(f"Strategy: {info['name']}")
        print(f"Required Indicators: {len(info['required_indicators'])}")
        
        for indicator in info['required_indicators']:
            if indicator in info['indicator_details']:
                details = info['indicator_details'][indicator]
                print(f"\n{indicator}:")
                if 'error' in details:
                    print(f"  Error: {details['error']}")
                else:
                    print(f"  Default params: {details['default_params']}")
                    if details['custom_params']:
                        print(f"  Custom params:  {details['custom_params']}")
                    print(f"  Final params:   {details['final_params']}")
                    print(f"  Outputs: {details['outputs']}")
    
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
