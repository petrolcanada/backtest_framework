"""
MFI (Money Flow Index) Indicator module.
"""
import pandas as pd
import pandas_ta as ta
from backtest_framework.core.indicators.registry import IndicatorRegistry

@IndicatorRegistry.register(
    name="MFI",
    inputs=["High", "Low", "Close", "Volume"],
    params={"period": 48},
    outputs=["MFI"],
    visualization_class="MFI"
)
def calculate_mfi(data: pd.DataFrame, period: int = 48) -> pd.DataFrame:
    """
    Calculate MFI (Money Flow Index) indicator.
    
    The Money Flow Index is a momentum oscillator that uses both price and volume
    to measure buying and selling pressure. It ranges from 0 to 100.
    - Values above 80 typically indicate overbought conditions
    - Values below 20 typically indicate oversold conditions
    
    Args:
        data: DataFrame with High, Low, Close, Volume columns
        period: Lookback period for MFI calculation (default: 48 ≈ 2.2 months)
        
    Returns:
        DataFrame with mfi column
    """
    # Ensure required columns exist
    required_cols = ['High', 'Low', 'Close', 'Volume']
    for col in required_cols:
        if col not in data.columns:
            raise ValueError(f"Required column '{col}' not found in data. Available columns: {list(data.columns)}")
    
    # Calculate MFI using pandas_ta
    mfi_result = ta.mfi(
        high=data['High'], 
        low=data['Low'], 
        close=data['Close'], 
        volume=data['Volume'], 
        length=period
    )
    
    # Create result DataFrame
    result = pd.DataFrame(index=data.index)
    
    if mfi_result is None:
        # Not enough data or calculation failed, return empty column
        result['MFI'] = float('nan')
    else:
        result['MFI'] = mfi_result
    
    return result

@IndicatorRegistry.register(
    name="MFI_MULTI_TIMEFRAME",
    inputs=["High", "Low", "Close", "Volume"],
    params={"short_period": 14, "long_period": 48},
    outputs=["mfi_short", "mfi_long", "mfi_ratio"]
)
def calculate_mfi_multi_timeframe(data: pd.DataFrame, short_period: int = 14, long_period: int = 48) -> pd.DataFrame:
    """
    Calculate MFI for multiple timeframes and their ratio.
    
    This provides both short-term and long-term MFI readings, plus their ratio
    which can help identify divergences and momentum shifts.
    
    Args:
        data: DataFrame with High, Low, Close, Volume columns
        short_period: Short-term MFI period (default: 14)
        long_period: Long-term MFI period (default: 48)
        
    Returns:
        DataFrame with mfi_short, mfi_long, mfi_ratio columns
    """
    # Ensure required columns exist
    required_cols = ['High', 'Low', 'Close', 'Volume']
    for col in required_cols:
        if col not in data.columns:
            raise ValueError(f"Required column '{col}' not found in data. Available columns: {list(data.columns)}")
    
    # Calculate short-term MFI
    mfi_short = ta.mfi(
        high=data['High'], 
        low=data['Low'], 
        close=data['Close'], 
        volume=data['Volume'], 
        length=short_period
    )
    
    # Calculate long-term MFI
    mfi_long = ta.mfi(
        high=data['High'], 
        low=data['Low'], 
        close=data['Close'], 
        volume=data['Volume'], 
        length=long_period
    )
    
    # Create result DataFrame
    result = pd.DataFrame(index=data.index)
    
    if mfi_short is None or mfi_long is None:
        # Not enough data or calculation failed
        result['mfi_short'] = float('nan')
        result['mfi_long'] = float('nan')
        result['mfi_ratio'] = float('nan')
    else:
        result['mfi_short'] = mfi_short
        result['mfi_long'] = mfi_long
        
        # Calculate ratio (short/long) - values > 1 indicate short-term strength
        result['mfi_ratio'] = result['mfi_short'] / result['mfi_long']
    
    return result
