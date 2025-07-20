"""
KDJ Derived Factors module - Derived indicators for KDJ-based trading strategies.
"""
import pandas as pd
import numpy as np
from backtest_framework.core.indicators.registry import IndicatorRegistry

@IndicatorRegistry.register(
    name="KDJ_CONSECUTIVE_DAYS",
    inputs=["monthly_j"],
    params={"up_days": 7, "down_days": 4},
    outputs=["j_consecutive_up_days", "j_consecutive_down_days", "j_up_flip", "j_down_flip"]
)
def calculate_kdj_consecutive_days(data: pd.DataFrame, up_days: int = 7, down_days: int = 4) -> pd.DataFrame:
    """
    Calculate consecutive days and flip points for Monthly J line.
    
    This tracks how many consecutive days the J line has been moving up or down,
    and identifies flip points where the direction changes.
    
    Args:
        data: DataFrame with monthly_j column
        up_days: Minimum consecutive up days to track (default: 7)
        down_days: Minimum consecutive down days to track (default: 4)
        
    Returns:
        DataFrame with j_consecutive_up_days, j_consecutive_down_days, j_up_flip, j_down_flip columns
    """
    # Ensure monthly_j column exists
    if 'monthly_j' not in data.columns:
        raise ValueError(f"Required column 'monthly_j' not found in data. Available columns: {list(data.columns)}")
    
    # Create result DataFrame
    result = pd.DataFrame(index=data.index)
    
    # Calculate J line changes
    j_diff = data['monthly_j'].diff()
    
    # Initialize tracking arrays
    consecutive_up = np.zeros(len(data))
    consecutive_down = np.zeros(len(data))
    up_flip = np.zeros(len(data))
    down_flip = np.zeros(len(data))
    
    # Track consecutive days
    current_up_streak = 0
    current_down_streak = 0
    
    for i in range(1, len(data)):
        if pd.isna(j_diff.iloc[i]):
            consecutive_up[i] = 0
            consecutive_down[i] = 0
            current_up_streak = 0
            current_down_streak = 0
            continue
            
        if j_diff.iloc[i] > 0:  # J is increasing
            current_up_streak += 1
            current_down_streak = 0
            consecutive_up[i] = current_up_streak
            consecutive_down[i] = 0
            
            # Check for up flip (from down to up)
            if i > 0 and consecutive_down[i-1] > 0:
                up_flip[i] = 1
                
        elif j_diff.iloc[i] < 0:  # J is decreasing
            current_down_streak += 1
            current_up_streak = 0
            consecutive_down[i] = current_down_streak
            consecutive_up[i] = 0
            
            # Check for down flip (from up to down)
            if i > 0 and consecutive_up[i-1] > 0:
                down_flip[i] = 1
                
        else:  # No change
            consecutive_up[i] = current_up_streak
            consecutive_down[i] = current_down_streak
    
    # Assign to result DataFrame
    result['j_consecutive_up_days'] = consecutive_up
    result['j_consecutive_down_days'] = consecutive_down
    result['j_up_flip'] = up_flip
    result['j_down_flip'] = down_flip
    
    return result

@IndicatorRegistry.register(
    name="GOLDEN_DEATH_CROSS",
    inputs=["monthly_j", "monthly_k"],
    outputs=["golden_cross", "death_cross", "cross_status", "bars_since_golden_cross"]
)
def calculate_golden_death_cross(data: pd.DataFrame) -> pd.DataFrame:
    """
    Identify golden/death crosses and track cross status for Monthly KDJ.
    
    Golden Cross: J line crosses above K line
    Death Cross: J line crosses below K line
    
    Args:
        data: DataFrame with monthly_j, monthly_k columns
        
    Returns:
        DataFrame with golden_cross, death_cross, cross_status, bars_since_golden_cross columns
    """
    # Ensure required columns exist
    required_cols = ['monthly_j', 'monthly_k']
    for col in required_cols:
        if col not in data.columns:
            raise ValueError(f"Required column '{col}' not found in data. Available columns: {list(data.columns)}")
    
    # Create result DataFrame
    result = pd.DataFrame(index=data.index)
    
    # Initialize output arrays
    golden_cross = np.zeros(len(data))
    death_cross = np.zeros(len(data))
    cross_status = np.zeros(len(data))  # 1 = golden period, 0 = death period
    bars_since_golden = np.zeros(len(data))
    
    # Track current status
    current_status = 0  # Start neutral
    bars_since_last_golden = 0
    
    for i in range(1, len(data)):
        j_curr = data['monthly_j'].iloc[i]
        k_curr = data['monthly_k'].iloc[i]
        j_prev = data['monthly_j'].iloc[i-1]
        k_prev = data['monthly_k'].iloc[i-1]
        
        # Skip if any values are NaN
        if pd.isna(j_curr) or pd.isna(k_curr) or pd.isna(j_prev) or pd.isna(k_prev):
            cross_status[i] = current_status
            bars_since_golden[i] = bars_since_last_golden
            if current_status == 1:
                bars_since_last_golden += 1
            continue
        
        # Check for Golden Cross (J crosses above K)
        if j_prev <= k_prev and j_curr > k_curr:
            golden_cross[i] = 1
            current_status = 1
            bars_since_last_golden = 0
            
        # Check for Death Cross (J crosses below K)
        elif j_prev >= k_prev and j_curr < k_curr:
            death_cross[i] = 1
            current_status = 0
            
        # Update status and counters
        cross_status[i] = current_status
        bars_since_golden[i] = bars_since_last_golden
        
        if current_status == 1:
            bars_since_last_golden += 1
    
    # Assign to result DataFrame
    result['golden_cross'] = golden_cross
    result['death_cross'] = death_cross
    result['cross_status'] = cross_status
    result['bars_since_golden_cross'] = bars_since_golden
    
    return result

# BUY_SIGNAL_COUNTER removed - signal limiting should be handled in strategy logic, not indicators
