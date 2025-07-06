"""
SMA Indicator module implementing Simple Moving Average and related derivatives.
"""
import pandas as pd
import pandas_ta as ta
from backtest_framework.core.indicators.registry import IndicatorRegistry

@IndicatorRegistry.register(
    name="SMA",
    inputs=["Close"],
    params={"window": 30},
    outputs=["SMA"],
    visualization_class="SMA"
)
def calculate_sma(data: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    """
    Calculate Simple Moving Average.
    
    Args:
        data: DataFrame with Close column
        window: Lookback period
        
    Returns:
        DataFrame with SMA column
    """
    result = pd.DataFrame(index=data.index)
    result['SMA'] = ta.sma(data['Close'], length=window)
    
    return result

@IndicatorRegistry.register(
    name="CLOSE_MINUS_SMA",
    inputs=["Close", "SMA"],
    outputs=["close_minus_sma"]
)
def calculate_close_minus_sma(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate difference between Close and SMA.
    
    Args:
        data: DataFrame with Close and SMA columns
        
    Returns:
        DataFrame with Close_Minus_SMA column
    """
    result = pd.DataFrame(index=data.index)
    result['close_minus_sma'] = data['Close'] - data['SMA']
    
    return result

@IndicatorRegistry.register(
    name="SMA_SLOPE",
    inputs=["SMA"],
    outputs=["sma_slope"]
)
def calculate_sma_slope(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate slope (first derivative) of SMA.
    
    Args:
        data: DataFrame with SMA column
        
    Returns:
        DataFrame with SMA_Slope column
    """
    result = pd.DataFrame(index=data.index)
    result['sma_slope'] = ta.percent_return(data['SMA'])
    
    return result
