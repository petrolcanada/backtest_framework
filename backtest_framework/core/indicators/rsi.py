"""
RSI (Relative Strength Index) Indicator module.
"""
import pandas as pd
import pandas_ta as ta
from backtest_framework.core.indicators.registry import IndicatorRegistry

@IndicatorRegistry.register(
    name="RSI",
    inputs=["Close"],
    params={"period": 14},
    outputs=["rsi"]
)
def calculate_rsi(data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Calculate RSI (Relative Strength Index) indicator.
    
    The RSI is a momentum oscillator that measures the speed and magnitude
    of price changes. It ranges from 0 to 100.
    - Values above 70 typically indicate overbought conditions
    - Values below 30 typically indicate oversold conditions
    
    Args:
        data: DataFrame with Close column
        period: Lookback period for RSI calculation (default: 14)
        
    Returns:
        DataFrame with rsi column
    """
    # Ensure Close column exists
    if 'Close' not in data.columns:
        raise ValueError(f"Required column 'Close' not found in data. Available columns: {list(data.columns)}")
    
    # Calculate RSI using pandas_ta
    rsi_result = ta.rsi(data['Close'], length=period)
    
    # Create result DataFrame
    result = pd.DataFrame(index=data.index)
    
    if rsi_result is None:
        # Not enough data or calculation failed, return empty column
        result['rsi'] = float('nan')
    else:
        result['rsi'] = rsi_result
    
    return result

@IndicatorRegistry.register(
    name="RSI_MULTI_TIMEFRAME",
    inputs=["Close"],
    params={"short_period": 14, "long_period": 21},
    outputs=["rsi_short", "rsi_long", "rsi_divergence"]
)
def calculate_rsi_multi_timeframe(data: pd.DataFrame, short_period: int = 14, long_period: int = 21) -> pd.DataFrame:
    """
    Calculate RSI for multiple timeframes and divergence analysis.
    
    This provides both short-term and long-term RSI readings, plus a divergence
    indicator that can help identify momentum shifts.
    
    Args:
        data: DataFrame with Close column
        short_period: Short-term RSI period (default: 14)
        long_period: Long-term RSI period (default: 21)
        
    Returns:
        DataFrame with rsi_short, rsi_long, rsi_divergence columns
    """
    # Ensure Close column exists
    if 'Close' not in data.columns:
        raise ValueError(f"Required column 'Close' not found in data. Available columns: {list(data.columns)}")
    
    # Calculate short-term RSI
    rsi_short = ta.rsi(data['Close'], length=short_period)
    
    # Calculate long-term RSI
    rsi_long = ta.rsi(data['Close'], length=long_period)
    
    # Create result DataFrame
    result = pd.DataFrame(index=data.index)
    
    if rsi_short is None or rsi_long is None:
        # Not enough data or calculation failed
        result['rsi_short'] = float('nan')
        result['rsi_long'] = float('nan')
        result['rsi_divergence'] = float('nan')
    else:
        result['rsi_short'] = rsi_short
        result['rsi_long'] = rsi_long
        
        # Calculate divergence (short - long)
        # Positive values indicate short-term momentum > long-term
        # Negative values indicate short-term momentum < long-term
        result['rsi_divergence'] = result['rsi_short'] - result['rsi_long']
    
    return result

@IndicatorRegistry.register(
    name="RSI_STOCHASTIC",
    inputs=["Close"],
    params={"rsi_period": 14, "stoch_period": 14, "smooth_k": 3, "smooth_d": 3},
    outputs=["stoch_rsi", "stoch_rsi_k", "stoch_rsi_d"]
)
def calculate_rsi_stochastic(data: pd.DataFrame, rsi_period: int = 14, stoch_period: int = 14, 
                           smooth_k: int = 3, smooth_d: int = 3) -> pd.DataFrame:
    """
    Calculate Stochastic RSI indicator.
    
    The Stochastic RSI applies the stochastic oscillator formula to RSI values
    instead of price values. This creates a more sensitive oscillator.
    
    Args:
        data: DataFrame with Close column
        rsi_period: Period for RSI calculation (default: 14)
        stoch_period: Period for stochastic calculation (default: 14)
        smooth_k: Smoothing period for %K line (default: 3)
        smooth_d: Smoothing period for %D line (default: 3)
        
    Returns:
        DataFrame with stoch_rsi, stoch_rsi_k, stoch_rsi_d columns
    """
    # Ensure Close column exists
    if 'Close' not in data.columns:
        raise ValueError(f"Required column 'Close' not found in data. Available columns: {list(data.columns)}")
    
    # Calculate Stochastic RSI using pandas_ta
    stoch_rsi_result = ta.stochrsi(
        data['Close'], 
        length=rsi_period, 
        rsi_length=rsi_period,
        k=smooth_k, 
        d=smooth_d
    )
    
    # Create result DataFrame
    result = pd.DataFrame(index=data.index)
    
    if stoch_rsi_result is None:
        # Not enough data or calculation failed
        result['stoch_rsi'] = float('nan')
        result['stoch_rsi_k'] = float('nan')
        result['stoch_rsi_d'] = float('nan')
    else:
        # Extract StochRSI components
        result['stoch_rsi'] = stoch_rsi_result.get(f'STOCHRSIk_{rsi_period}_{stoch_period}_{smooth_k}_{smooth_d}', float('nan'))
        result['stoch_rsi_k'] = stoch_rsi_result.get(f'STOCHRSIk_{rsi_period}_{stoch_period}_{smooth_k}_{smooth_d}', float('nan'))
        result['stoch_rsi_d'] = stoch_rsi_result.get(f'STOCHRSId_{rsi_period}_{stoch_period}_{smooth_k}_{smooth_d}', float('nan'))
    
    return result
