"""
KDJ Cross Depression strategy with modern framework structure.

This strategy focuses on buy signals generated when golden cross happens
in the depression zone (all K, D, J values < 50), which typically indicates
oversold conditions with potential for strong reversal.
"""
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from backtest_framework.core.signals.base import BaseStrategy
from backtest_framework.core.utils.helpers import clean_params, filter_empty_dicts

class KDJCrossDepressionStrategy(BaseStrategy):
    """
    KDJ Cross Depression strategy with configurable indicator parameters.
    
    SIGNAL LOGIC:
    ============
    
    BUY SIGNALS (Golden Cross in Depression Zone):
    - Golden cross occurs (golden_cross == 1)
    - All K, D, J values are below 50 (depression zone)
    - K and D slopes are positive (upward momentum)
    - J value not extremely high (≤ 100)
    
    SELL SIGNALS:
    - Death cross occurs (death_cross == 1)
    - OR J reaches extreme high (≥ 100)
    
    The depression zone entry strategy aims to capture strong reversals
    from oversold conditions with reduced risk.
    """
    
    def __init__(self, 
                 # Strategy parameters
                 depression_threshold: float = 50.0,      # Depression zone threshold
                 j_extreme_threshold: float = 200.0,      # J extreme level for sell
                 use_monthly_kdj: bool = True,            # Use monthly vs daily KDJ
                 
                 # Indicator parameter overrides
                 kdj_period: int = 198,                   # KDJ lookback period (default: 9 months * 22 days)
                 kdj_signal: int = 66,                    # KDJ signal period (default: 3 months * 22 days)
                 daily_kdj_period: int = 9,               # Daily KDJ period if used
                 daily_kdj_signal: int = 3):              # Daily KDJ signal if used
        
        # Initialize strategy parameters
        self.depression_threshold = depression_threshold
        self.j_extreme_threshold = j_extreme_threshold
        self.use_monthly_kdj = use_monthly_kdj
        
        # Build indicator parameter overrides
        indicator_overrides = {}
        
        if use_monthly_kdj:
            indicator_overrides.update({
                'MONTHLY_KDJ': clean_params(period=kdj_period, signal=kdj_signal)
            })
        else:
            indicator_overrides.update({
                'KDJ': clean_params(period=daily_kdj_period, signal=daily_kdj_signal)
            })
        
        # Filter out any empty parameter dictionaries
        indicator_overrides = filter_empty_dicts(indicator_overrides)
        
        # Initialize base strategy with indicator parameters
        super().__init__(indicator_params=indicator_overrides)
    
    @property
    def required_indicators(self) -> List[str]:
        """All indicators needed for the strategy."""
        if self.use_monthly_kdj:
            return [
                "MONTHLY_KDJ",           # monthly_k, monthly_d, monthly_j
                "MONTHLY_KDJ_SLOPES",    # monthly_k_slope, monthly_d_slope, monthly_j_slope
                "GOLDEN_DEATH_CROSS"     # golden_cross, death_cross, cross_status
            ]
        else:
            return [
                "KDJ",                   # k, d, j
                "KDJ_SLOPES",           # k_slope, d_slope, j_slope
                "GOLDEN_DEATH_CROSS"    # golden_cross, death_cross, cross_status (uses monthly by default)
            ]
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate buy and sell signals based on KDJ depression zone logic.
        
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
        
        # Track position state to avoid duplicate signals
        in_position = False
        
        # Get column names based on KDJ type
        if self.use_monthly_kdj:
            k_col, d_col, j_col = 'monthly_k', 'monthly_d', 'monthly_j'
            k_slope_col, d_slope_col = 'monthly_k_slope', 'monthly_d_slope'
        else:
            k_col, d_col, j_col = 'k', 'd', 'j'
            k_slope_col, d_slope_col = 'k_slope', 'd_slope'
        
        # Generate signals row by row
        for i in range(len(data)):
            current_date = data.index[i].strftime('%Y-%m-%d')
            
            # === BUY LOGIC (Golden Cross in Depression Zone) ===
            if not in_position and self._check_depression_buy_conditions(data, i, k_col, d_col, j_col, k_slope_col, d_slope_col):
                data.iloc[i, data.columns.get_loc('buy_signal')] = 1
                in_position = True
                
            # === SELL LOGIC ===
            elif in_position and self._check_sell_conditions(data, i, j_col):
                data.iloc[i, data.columns.get_loc('sell_signal')] = 1
                in_position = False
        
        return data
    
    def _check_depression_buy_conditions(self, data: pd.DataFrame, i: int, 
                                       k_col: str, d_col: str, j_col: str,
                                       k_slope_col: str, d_slope_col: str) -> bool:
        """
        Check if depression zone buy conditions are met.
        
        CONDITIONS:
        1. Golden cross occurs (golden_cross == 1)
        2. All K, D, J values are below depression threshold (default: 50)
        3. K and D slopes are positive (upward momentum)
        4. J value not extremely high (≤ j_extreme_threshold)
        """
        try:
            # Condition 1: Golden Cross using derived indicator
            if data['golden_cross'].iloc[i] != 1:
                return False
            
            # Condition 2: All K, D, J values in depression zone
            k_val = data[k_col].iloc[i]
            d_val = data[d_col].iloc[i]
            j_val = data[j_col].iloc[i]
            
            # Skip if any values are NaN
            if pd.isna(k_val) or pd.isna(d_val) or pd.isna(j_val):
                return False
            
            all_in_depression = (k_val < self.depression_threshold and 
                               d_val < self.depression_threshold and 
                               j_val < self.depression_threshold)
            if not all_in_depression:
                return False
            
            # Condition 3: Positive momentum (K and D slopes > 0)
            k_slope = data[k_slope_col].iloc[i]
            d_slope = data[d_slope_col].iloc[i]
            
            # Skip if slopes are NaN
            if pd.isna(k_slope) or pd.isna(d_slope):
                return False
            
            positive_momentum = (k_slope > 0 and d_slope > 0)
            if not positive_momentum:
                return False
            
            # Condition 4: J not extremely high
            j_not_extreme = j_val <= self.j_extreme_threshold
            if not j_not_extreme:
                return False
            
            return True
            
        except (KeyError, IndexError) as e:
            return False
    
    def _check_sell_conditions(self, data: pd.DataFrame, i: int, j_col: str) -> bool:
        """
        Check if sell conditions are met.
        
        CONDITIONS:
        1. Death cross occurs (death_cross == 1)
        2. OR J reaches extreme high level
        """
        try:
            # Condition 1: Death Cross using derived indicator
            death_cross = data['death_cross'].iloc[i] == 1
            
            # Condition 2: J reaches extreme high
            j_val = data[j_col].iloc[i]
            j_extreme = False if pd.isna(j_val) else j_val >= self.j_extreme_threshold
            
            # return death_cross or j_extreme
            return j_extreme
            
        except (KeyError, IndexError):
            return False


class KDJCrossDepressionStrategyMonthly(KDJCrossDepressionStrategy):
    """
    Monthly KDJ Cross Depression strategy - convenience class.
    
    This strategy is ideal for:
    - Long-term position trading
    - Identifying major reversal points from oversold conditions
    - Lower frequency trading with potentially higher accuracy
    """
    
    def __init__(self, **kwargs):
        # Force monthly KDJ usage
        kwargs['use_monthly_kdj'] = True
        super().__init__(**kwargs)


class KDJCrossDepressionStrategyDaily(KDJCrossDepressionStrategy):
    """
    Daily KDJ Cross Depression strategy - convenience class.
    
    This strategy is ideal for:
    - Short to medium-term trading
    - More frequent trading opportunities
    - Quick reversal identification from oversold conditions
    
    Note: GOLDEN_DEATH_CROSS indicator uses monthly KDJ by default,
    so this will use daily KDJ for K,D,J values but monthly for cross detection.
    """
    
    def __init__(self, **kwargs):
        # Force daily KDJ usage
        kwargs['use_monthly_kdj'] = False
        super().__init__(**kwargs)
