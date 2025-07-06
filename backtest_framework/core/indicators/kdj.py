"""
KDJ Indicator module implementing KDJ and related derivatives.
"""
import pandas as pd
import pandas_ta as ta
from backtest_framework.core.indicators.registry import IndicatorRegistry

@IndicatorRegistry.register(
    name="KDJ",
    inputs=["High", "Low", "Close"],
    params={"period": 9, "signal": 3},
    outputs=["k", "d", "j"]
)
def calculate_kdj(data: pd.DataFrame, period: int = 9, signal: int = 3) -> pd.DataFrame:
    """
    Calculate KDJ indicator.
    
    Args:
        data: DataFrame with High, Low, Close columns
        period: Lookback period
        signal: Signal period
        
    Returns:
        DataFrame with k, d, j columns (snake_case)
    """
    kdj = ta.kdj(data['High'], data['Low'], data['Close'], length=period, signal=signal)
    
    # Rename columns to simpler names with snake_case
    result = pd.DataFrame(index=data.index)
    result['k'] = kdj[f'K_{period}_{signal}']
    result['d'] = kdj[f'D_{period}_{signal}']
    result['j'] = kdj[f'J_{period}_{signal}']
    
    return result

@IndicatorRegistry.register(
    name="KDJ_SLOPES",
    inputs=["k", "d", "j"],
    outputs=["k_slope", "d_slope", "j_slope"]
)
def calculate_kdj_slopes(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate slopes (first derivatives) of k, d, j lines.
    
    Args:
        data: DataFrame with k, d, j columns
        
    Returns:
        DataFrame with k_slope, d_slope, j_slope columns
    """
    result = pd.DataFrame(index=data.index)
    result['k_slope'] = data['k'].diff()
    result['d_slope'] = data['d'].diff()
    result['j_slope'] = data['j'].diff()
    
    return result

@IndicatorRegistry.register(
    name="MONTHLY_KDJ",
    inputs=["High", "Low", "Close"],
    params={"period": 198, "signal": 66},  # 9 months * 22 trading days, 3 months * 22 trading days
    outputs=["monthly_k", "monthly_d", "monthly_j"],
    visualization_class="MonthlyKDJ"
)
def calculate_monthly_kdj(data: pd.DataFrame, period: int = 198, signal: int = 66) -> pd.DataFrame:
    """
    Calculate Monthly KDJ using daily data with monthly-equivalent periods.
    
    Args:
        data: DataFrame with High, Low, Close columns
        period: Lookback period (default: 198 = 9 months * 22 trading days)
        signal: Signal period (default: 66 = 3 months * 22 trading days)
        
    Returns:
        DataFrame with monthly_k, monthly_d, monthly_j columns (snake_case)
    """
    kdj = ta.kdj(data['High'], data['Low'], data['Close'], length=period, signal=signal)
    
    # Create result DataFrame
    result = pd.DataFrame(index=data.index)
    
    if kdj is None or f'K_{period}_{signal}' not in kdj.columns:
        # Not enough data or calculation failed, return empty columns
        result['monthly_k'] = float('nan')
        result['monthly_d'] = float('nan')
        result['monthly_j'] = float('nan')
    else:
        # Rename columns to match monthly KDJ with snake_case
        result['monthly_k'] = kdj[f'K_{period}_{signal}']
        result['monthly_d'] = kdj[f'D_{period}_{signal}']
        result['monthly_j'] = kdj[f'J_{period}_{signal}']
    
    return result

@IndicatorRegistry.register(
    name="MONTHLY_KDJ_SLOPES",
    inputs=["monthly_k", "monthly_d", "monthly_j"],  # Snake_case inputs
    outputs=["monthly_k_slope", "monthly_d_slope", "monthly_j_slope"]  # Snake_case outputs
)
def calculate_monthly_kdj_slopes(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate slopes (first derivatives) of Monthly k, d, j lines.
    
    Args:
        data: DataFrame with monthly_k, monthly_d, monthly_j columns
        
    Returns:
        DataFrame with monthly_k_slope, monthly_d_slope, monthly_j_slope columns
    """
    # Check if all required columns exist
    for col in ["monthly_k", "monthly_d", "monthly_j"]:
        if col not in data.columns:
            raise ValueError(f"Required column {col} missing from data. Available columns: {data.columns}")
    
    # Create result DataFrame
    result = pd.DataFrame(index=data.index)
    
    # Calculate slopes
    result['monthly_k_slope'] = data['monthly_k'].diff()
    result['monthly_d_slope'] = data['monthly_d'].diff()
    result['monthly_j_slope'] = data['monthly_j'].diff()
    
    return result
