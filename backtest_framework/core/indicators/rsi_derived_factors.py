"""
RSI Derived Factors module - Derived indicators for RSI-based trading strategies.
"""
import pandas as pd
import numpy as np
from backtest_framework.core.indicators.registry import IndicatorRegistry

@IndicatorRegistry.register(
    name="RSI_OVERBOUGHT_OVERSOLD",
    inputs=["rsi"],
    params={"overbought_level": 70, "oversold_level": 30, "extreme_overbought": 80, "extreme_oversold": 20},
    outputs=["rsi_zone", "rsi_overbought", "rsi_oversold", "rsi_extreme_overbought", "rsi_extreme_oversold"]
)
def calculate_rsi_overbought_oversold(data: pd.DataFrame, overbought_level: int = 70, oversold_level: int = 30,
                                    extreme_overbought: int = 80, extreme_oversold: int = 20) -> pd.DataFrame:
    """
    Categorize RSI levels into zones and identify key conditions.
    
    This creates categorical zones for RSI and identifies specific
    overbought/oversold conditions for trading signals.
    
    Args:
        data: DataFrame with rsi column
        overbought_level: RSI level considered overbought (default: 70)
        oversold_level: RSI level considered oversold (default: 30)
        extreme_overbought: RSI level considered extremely overbought (default: 80)
        extreme_oversold: RSI level considered extremely oversold (default: 20)
        
    Returns:
        DataFrame with rsi_zone, rsi_overbought, rsi_oversold, rsi_extreme_overbought, rsi_extreme_oversold columns
    """
    # Ensure rsi column exists
    if 'rsi' not in data.columns:
        raise ValueError(f"Required column 'rsi' not found in data. Available columns: {list(data.columns)}")
    
    # Create result DataFrame
    result = pd.DataFrame(index=data.index)
    
    # Categorize RSI zones
    # 0 = oversold, 1 = neutral, 2 = overbought
    rsi_zone = np.zeros(len(data))
    
    for i in range(len(data)):
        rsi_val = data['rsi'].iloc[i]
        
        if pd.isna(rsi_val):
            rsi_zone[i] = 1  # Default to neutral
        elif rsi_val <= oversold_level:
            rsi_zone[i] = 0  # Oversold
        elif rsi_val >= overbought_level:
            rsi_zone[i] = 2  # Overbought
        else:
            rsi_zone[i] = 1  # Neutral
    
    # Create binary indicators
    result['rsi_zone'] = rsi_zone
    result['rsi_overbought'] = (data['rsi'] >= overbought_level).astype(int)
    result['rsi_oversold'] = (data['rsi'] <= oversold_level).astype(int)
    result['rsi_extreme_overbought'] = (data['rsi'] >= extreme_overbought).astype(int)
    result['rsi_extreme_oversold'] = (data['rsi'] <= extreme_oversold).astype(int)
    
    return result

@IndicatorRegistry.register(
    name="RSI_REVERSAL_SIGNALS",
    inputs=["rsi"],
    params={"lookback_period": 14, "reversal_threshold": 5},
    outputs=["rsi_bullish_reversal", "rsi_bearish_reversal", "rsi_reversal_strength"]
)
def calculate_rsi_reversal_signals(data: pd.DataFrame, lookback_period: int = 14, reversal_threshold: int = 5) -> pd.DataFrame:
    """
    Detect RSI reversal signals and patterns.
    
    This identifies potential reversal points based on RSI behavior
    in overbought/oversold zones and momentum shifts.
    
    Args:
        data: DataFrame with rsi column
        lookback_period: Period to analyze for reversal patterns (default: 14)
        reversal_threshold: Minimum RSI change required for reversal signal (default: 5)
        
    Returns:
        DataFrame with rsi_bullish_reversal, rsi_bearish_reversal, rsi_reversal_strength columns
    """
    # Ensure rsi column exists
    if 'rsi' not in data.columns:
        raise ValueError(f"Required column 'rsi' not found in data. Available columns: {list(data.columns)}")
    
    # Create result DataFrame
    result = pd.DataFrame(index=data.index)
    
    # Initialize reversal indicators
    bullish_reversal = np.zeros(len(data))
    bearish_reversal = np.zeros(len(data))
    reversal_strength = np.zeros(len(data))
    
    for i in range(lookback_period, len(data)):
        current_rsi = data['rsi'].iloc[i]
        
        if pd.isna(current_rsi):
            continue
            
        # Get recent RSI values
        recent_rsi = data['rsi'].iloc[i-lookback_period:i+1]
        min_recent_rsi = recent_rsi.min()
        max_recent_rsi = recent_rsi.max()
        
        # Bullish reversal: RSI was oversold and is now rising
        if (min_recent_rsi <= 30 and  # Was in oversold territory
            current_rsi > min_recent_rsi + reversal_threshold and  # Significant rise
            current_rsi > data['rsi'].iloc[i-1]):  # Currently rising
            
            bullish_reversal[i] = 1
            reversal_strength[i] = current_rsi - min_recent_rsi
            
        # Bearish reversal: RSI was overbought and is now falling
        elif (max_recent_rsi >= 70 and  # Was in overbought territory
              current_rsi < max_recent_rsi - reversal_threshold and  # Significant fall
              current_rsi < data['rsi'].iloc[i-1]):  # Currently falling
            
            bearish_reversal[i] = 1
            reversal_strength[i] = max_recent_rsi - current_rsi
    
    # Assign to result DataFrame
    result['rsi_bullish_reversal'] = bullish_reversal
    result['rsi_bearish_reversal'] = bearish_reversal
    result['rsi_reversal_strength'] = reversal_strength
    
    return result

@IndicatorRegistry.register(
    name="RSI_TREND_CONFIRMATION",
    inputs=["rsi"],
    params={"trend_period": 20, "confirmation_threshold": 50},
    outputs=["rsi_trend_direction", "rsi_trend_strength", "rsi_trend_confirmed"]
)
def calculate_rsi_trend_confirmation(data: pd.DataFrame, trend_period: int = 20, confirmation_threshold: int = 50) -> pd.DataFrame:
    """
    Analyze RSI for trend confirmation and direction.
    
    This uses RSI behavior to confirm trend direction and strength,
    providing additional confirmation for trading decisions.
    
    Args:
        data: DataFrame with rsi column
        trend_period: Period to analyze for trend confirmation (default: 20)
        confirmation_threshold: RSI level that separates bullish/bearish bias (default: 50)
        
    Returns:
        DataFrame with rsi_trend_direction, rsi_trend_strength, rsi_trend_confirmed columns
    """
    # Ensure rsi column exists
    if 'rsi' not in data.columns:
        raise ValueError(f"Required column 'rsi' not found in data. Available columns: {list(data.columns)}")
    
    # Create result DataFrame
    result = pd.DataFrame(index=data.index)
    
    # Calculate RSI moving average for trend analysis
    rsi_ma = data['rsi'].rolling(window=trend_period).mean()
    
    # Initialize trend indicators
    trend_direction = np.zeros(len(data))  # 1 = bullish, -1 = bearish, 0 = neutral
    trend_strength = np.zeros(len(data))
    trend_confirmed = np.zeros(len(data))
    
    for i in range(trend_period, len(data)):
        current_rsi = data['rsi'].iloc[i]
        current_rsi_ma = rsi_ma.iloc[i]
        
        if pd.isna(current_rsi) or pd.isna(current_rsi_ma):
            continue
        
        # Determine trend direction based on RSI vs threshold and MA
        if current_rsi > confirmation_threshold and current_rsi > current_rsi_ma:
            trend_direction[i] = 1  # Bullish
            trend_strength[i] = (current_rsi - confirmation_threshold) / 50  # Normalize to 0-1
        elif current_rsi < confirmation_threshold and current_rsi < current_rsi_ma:
            trend_direction[i] = -1  # Bearish
            trend_strength[i] = (confirmation_threshold - current_rsi) / 50  # Normalize to 0-1
        else:
            trend_direction[i] = 0  # Neutral
            trend_strength[i] = 0
        
        # Confirm trend if RSI has been consistently on one side
        recent_rsi = data['rsi'].iloc[i-10:i+1]  # Last 10 periods
        
        if trend_direction[i] == 1:  # Bullish trend
            bullish_consistency = (recent_rsi > confirmation_threshold).mean()
            if bullish_consistency >= 0.7:  # 70% of recent periods above threshold
                trend_confirmed[i] = 1
        elif trend_direction[i] == -1:  # Bearish trend
            bearish_consistency = (recent_rsi < confirmation_threshold).mean()
            if bearish_consistency >= 0.7:  # 70% of recent periods below threshold
                trend_confirmed[i] = 1
    
    # Assign to result DataFrame
    result['rsi_trend_direction'] = trend_direction
    result['rsi_trend_strength'] = trend_strength
    result['rsi_trend_confirmed'] = trend_confirmed
    
    return result
