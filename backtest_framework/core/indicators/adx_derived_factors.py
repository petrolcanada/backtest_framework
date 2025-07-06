"""
ADX Derived Factors module - Derived indicators for ADX-based trading strategies.
"""
import pandas as pd
import numpy as np
from backtest_framework.core.indicators.registry import IndicatorRegistry

@IndicatorRegistry.register(
    name="ADX_CONSECUTIVE_DOWN",
    inputs=["ADX"],
    params={"required_days": 5},
    outputs=["adx_consecutive_down_days", "adx_down_condition_met"]
)
def calculate_adx_consecutive_down(data: pd.DataFrame, required_days: int = 5) -> pd.DataFrame:
    """
    Track consecutive ADX down days and condition fulfillment.
    
    This identifies periods where ADX has been declining for a specified
    number of consecutive days, which can indicate weakening trend strength.
    
    Args:
        data: DataFrame with adx column
        required_days: Minimum consecutive down days required (default: 5)
        
    Returns:
        DataFrame with adx_consecutive_down_days, adx_down_condition_met columns
    """
    # Ensure ADX column exists
    if 'ADX' not in data.columns:
        raise ValueError(f"Required column 'ADX' not found in data. Available columns: {list(data.columns)}")
    
    # Create result DataFrame
    result = pd.DataFrame(index=data.index)
    
    # Calculate ADX changes
    adx_diff = data['ADX'].diff()
    
    # Initialize tracking arrays
    consecutive_down = np.zeros(len(data))
    condition_met = np.zeros(len(data))
    
    # Track consecutive down days
    current_down_streak = 0
    
    for i in range(1, len(data)):
        if pd.isna(adx_diff.iloc[i]):
            consecutive_down[i] = 0
            current_down_streak = 0
            continue
            
        if adx_diff.iloc[i] < 0:  # ADX is decreasing
            current_down_streak += 1
            consecutive_down[i] = current_down_streak
            
            # Check if condition is met
            if current_down_streak >= required_days:
                condition_met[i] = 1
                
        else:  # ADX is increasing or flat
            current_down_streak = 0
            consecutive_down[i] = 0
            condition_met[i] = 0
    
    # Assign to result DataFrame
    result['adx_consecutive_down_days'] = consecutive_down
    result['adx_down_condition_met'] = condition_met
    
    return result

@IndicatorRegistry.register(
    name="ADX_TREND_STRENGTH",
    inputs=["ADX", "ADX_SMA"],
    params={"weak_threshold": 20, "strong_threshold": 40},
    outputs=["adx_trend_strength", "adx_above_sma", "adx_strength_signal"]
)
def calculate_adx_trend_strength(data: pd.DataFrame, weak_threshold: int = 20, strong_threshold: int = 40) -> pd.DataFrame:
    """
    Analyze ADX trend strength and relationship with its SMA.
    
    This provides categorical trend strength analysis and signals based on
    ADX levels and its relationship to the smoothed average.
    
    Args:
        data: DataFrame with adx, adx_sma columns
        weak_threshold: Threshold below which trend is considered weak (default: 20)
        strong_threshold: Threshold above which trend is considered strong (default: 40)
        
    Returns:
        DataFrame with adx_trend_strength, adx_above_sma, adx_strength_signal columns
    """
    # Ensure required columns exist
    required_cols = ['ADX', 'ADX_SMA']
    for col in required_cols:
        if col not in data.columns:
            raise ValueError(f"Required column '{col}' not found in data. Available columns: {list(data.columns)}")
    
    # Create result DataFrame
    result = pd.DataFrame(index=data.index)
    
    # Categorize trend strength
    # 0 = weak, 1 = moderate, 2 = strong
    trend_strength = np.zeros(len(data))
    
    for i in range(len(data)):
        adx_val = data['ADX'].iloc[i]
        
        if pd.isna(adx_val):
            trend_strength[i] = 0
        elif adx_val < weak_threshold:
            trend_strength[i] = 0  # Weak trend
        elif adx_val < strong_threshold:
            trend_strength[i] = 1  # Moderate trend
        else:
            trend_strength[i] = 2  # Strong trend
    
    # Check if ADX is above its SMA
    adx_above_sma = (data['ADX'] > data['ADX_SMA']).astype(int)
    
    # Generate strength signals
    # 1 = strengthening trend, -1 = weakening trend, 0 = neutral
    strength_signal = np.zeros(len(data))
    
    for i in range(1, len(data)):
        if pd.isna(data['ADX'].iloc[i]) or pd.isna(data['ADX_SMA'].iloc[i]):
            continue
            
        # Strengthening: ADX rising and above SMA
        if (data['ADX'].iloc[i] > data['ADX'].iloc[i-1] and 
            data['ADX'].iloc[i] > data['ADX_SMA'].iloc[i]):
            strength_signal[i] = 1
            
        # Weakening: ADX falling and below SMA
        elif (data['ADX'].iloc[i] < data['ADX'].iloc[i-1] and 
              data['ADX'].iloc[i] < data['ADX_SMA'].iloc[i]):
            strength_signal[i] = -1
    
    # Assign to result DataFrame
    result['adx_trend_strength'] = trend_strength
    result['adx_above_sma'] = adx_above_sma
    result['adx_strength_signal'] = strength_signal
    
    return result
