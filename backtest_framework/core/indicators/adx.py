"""
ADX (Average Directional Index) Indicator module.
"""
import pandas as pd
import pandas_ta as ta
from backtest_framework.core.indicators.registry import IndicatorRegistry

@IndicatorRegistry.register(
    name="ADX",
    inputs=["High", "Low", "Close"],
    params={"window": 308, "sma_length": 5},
    outputs=["ADX", "ADX_SMA"],
    visualization_class="ADX"
)
def calculate_adx(data: pd.DataFrame, window: int = 308, sma_length: int = 5) -> pd.DataFrame:
    """
    Calculate ADX (Average Directional Index) indicator with SMA smoothing.
    
    The ADX measures the strength of a trend, regardless of direction.
    Values above 25 typically indicate a strong trend.
    
    Args:
        data: DataFrame with High, Low, Close columns
        window: Lookback period for ADX calculation (default: 308 ≈ 14 months)
        sma_length: Simple moving average length for smoothing (default: 5)
        
    Returns:
        DataFrame with adx, adx_sma columns
    """
    # Calculate ADX using pandas_ta
    adx_result = ta.adx(data['High'], data['Low'], data['Close'], length=window)
    
    # Create result DataFrame
    result = pd.DataFrame(index=data.index)
    
    if adx_result is None or f'ADX_{window}' not in adx_result.columns:
        # Not enough data or calculation failed, return empty columns
        result['ADX'] = float('nan')
        result['ADX_SMA'] = float('nan')
    else:
        # Extract ADX values
        result['ADX'] = adx_result[f'ADX_{window}']
        
        # Calculate SMA smoothing of ADX
        result['ADX_SMA'] = result['ADX'].rolling(window=sma_length, min_periods=1).mean()
    
    return result

@IndicatorRegistry.register(
    name="ADX_DIRECTIONAL",
    inputs=["High", "Low", "Close"],
    params={"window": 14},
    outputs=["adx_14", "dmp_14", "dmn_14"]
)
def calculate_adx_directional(data: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Calculate ADX with Directional Movement indicators (DM+ and DM-).
    
    This provides the standard ADX with directional components for
    more detailed trend analysis.
    
    Args:
        data: DataFrame with High, Low, Close columns
        window: Lookback period for ADX calculation (default: 14)
        
    Returns:
        DataFrame with adx_14, dmp_14, dmn_14 columns
    """
    # Calculate ADX with directional movement
    adx_result = ta.adx(data['High'], data['Low'], data['Close'], length=window)
    
    # Create result DataFrame
    result = pd.DataFrame(index=data.index)
    
    if adx_result is None:
        # Not enough data or calculation failed
        result['adx_14'] = float('nan')
        result['dmp_14'] = float('nan')
        result['dmn_14'] = float('nan')
    else:
        # Extract all ADX components
        result['adx_14'] = adx_result.get(f'ADX_{window}', float('nan'))
        result['dmp_14'] = adx_result.get(f'DMP_{window}', float('nan'))  # Directional Movement Positive
        result['dmn_14'] = adx_result.get(f'DMN_{window}', float('nan'))  # Directional Movement Negative
    
    return result
