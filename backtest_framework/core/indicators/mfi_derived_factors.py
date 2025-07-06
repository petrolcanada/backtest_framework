"""
MFI Derived Factors module - Derived indicators for MFI-based trading strategies.
"""
import pandas as pd
import numpy as np
from backtest_framework.core.indicators.registry import IndicatorRegistry

@IndicatorRegistry.register(
    name="MFI_OVERBOUGHT_OVERSOLD",
    inputs=["mfi"],
    params={"overbought_level": 80, "oversold_level": 20, "neutral_upper": 60, "neutral_lower": 40},
    outputs=["mfi_zone", "mfi_overbought", "mfi_oversold", "mfi_neutral"]
)
def calculate_mfi_overbought_oversold(data: pd.DataFrame, overbought_level: int = 80, oversold_level: int = 20,
                                    neutral_upper: int = 60, neutral_lower: int = 40) -> pd.DataFrame:
    """
    Categorize MFI levels into zones and identify key conditions.
    
    This creates categorical zones for MFI and identifies specific
    overbought/oversold conditions for trading signals.
    
    Args:
        data: DataFrame with mfi column
        overbought_level: MFI level considered overbought (default: 80)
        oversold_level: MFI level considered oversold (default: 20)
        neutral_upper: Upper bound of neutral zone (default: 60)
        neutral_lower: Lower bound of neutral zone (default: 40)
        
    Returns:
        DataFrame with mfi_zone, mfi_overbought, mfi_oversold, mfi_neutral columns
    """
    # Ensure mfi column exists
    if 'mfi' not in data.columns:
        raise ValueError(f"Required column 'mfi' not found in data. Available columns: {list(data.columns)}")
    
    # Create result DataFrame
    result = pd.DataFrame(index=data.index)
    
    # Categorize MFI zones
    # 0 = oversold, 1 = neutral, 2 = overbought
    mfi_zone = np.zeros(len(data))
    
    for i in range(len(data)):
        mfi_val = data['mfi'].iloc[i]
        
        if pd.isna(mfi_val):
            mfi_zone[i] = 1  # Default to neutral
        elif mfi_val <= oversold_level:
            mfi_zone[i] = 0  # Oversold
        elif mfi_val >= overbought_level:
            mfi_zone[i] = 2  # Overbought
        else:
            mfi_zone[i] = 1  # Neutral
    
    # Create binary indicators
    result['mfi_zone'] = mfi_zone
    result['mfi_overbought'] = (data['mfi'] >= overbought_level).astype(int)
    result['mfi_oversold'] = (data['mfi'] <= oversold_level).astype(int)
    result['mfi_neutral'] = ((data['mfi'] >= neutral_lower) & (data['mfi'] <= neutral_upper)).astype(int)
    
    return result

@IndicatorRegistry.register(
    name="MFI_DIVERGENCE",
    inputs=["mfi", "Close"],
    params={"lookback_period": 20},
    outputs=["mfi_price_divergence", "mfi_bullish_divergence", "mfi_bearish_divergence"]
)
def calculate_mfi_divergence(data: pd.DataFrame, lookback_period: int = 20) -> pd.DataFrame:
    """
    Detect MFI divergences with price action.
    
    Divergences occur when price and MFI move in opposite directions,
    potentially signaling trend reversals.
    
    Args:
        data: DataFrame with mfi, Close columns
        lookback_period: Period to look back for divergence analysis (default: 20)
        
    Returns:
        DataFrame with mfi_price_divergence, mfi_bullish_divergence, mfi_bearish_divergence columns
    """
    # Ensure required columns exist
    required_cols = ['mfi', 'Close']
    for col in required_cols:
        if col not in data.columns:
            raise ValueError(f"Required column '{col}' not found in data. Available columns: {list(data.columns)}")
    
    # Create result DataFrame
    result = pd.DataFrame(index=data.index)
    
    # Calculate rolling correlations between MFI and price
    mfi_change = data['mfi'].pct_change()
    price_change = data['Close'].pct_change()
    
    # Rolling correlation
    divergence = mfi_change.rolling(window=lookback_period).corr(price_change)
    
    # Initialize divergence indicators
    bullish_divergence = np.zeros(len(data))
    bearish_divergence = np.zeros(len(data))
    
    for i in range(lookback_period, len(data)):
        # Look for significant negative correlation (divergence)
        if pd.notna(divergence.iloc[i]) and divergence.iloc[i] < -0.3:
            
            # Check recent price and MFI trends
            recent_price_trend = data['Close'].iloc[i-5:i+1].pct_change().mean()
            recent_mfi_trend = data['mfi'].iloc[i-5:i+1].pct_change().mean()
            
            # Bullish divergence: price falling, MFI rising
            if recent_price_trend < 0 and recent_mfi_trend > 0:
                bullish_divergence[i] = 1
                
            # Bearish divergence: price rising, MFI falling
            elif recent_price_trend > 0 and recent_mfi_trend < 0:
                bearish_divergence[i] = 1
    
    # Assign to result DataFrame
    result['mfi_price_divergence'] = divergence
    result['mfi_bullish_divergence'] = bullish_divergence
    result['mfi_bearish_divergence'] = bearish_divergence
    
    return result

@IndicatorRegistry.register(
    name="MFI_MOMENTUM",
    inputs=["mfi"],
    params={"short_period": 5, "long_period": 14},
    outputs=["mfi_momentum", "mfi_acceleration", "mfi_momentum_signal"]
)
def calculate_mfi_momentum(data: pd.DataFrame, short_period: int = 5, long_period: int = 14) -> pd.DataFrame:
    """
    Calculate MFI momentum and acceleration indicators.
    
    This tracks the rate of change in MFI and its acceleration,
    providing early signals of momentum shifts.
    
    Args:
        data: DataFrame with mfi column
        short_period: Short-term momentum period (default: 5)
        long_period: Long-term momentum period (default: 14)
        
    Returns:
        DataFrame with mfi_momentum, mfi_acceleration, mfi_momentum_signal columns
    """
    # Ensure mfi column exists
    if 'mfi' not in data.columns:
        raise ValueError(f"Required column 'mfi' not found in data. Available columns: {list(data.columns)}")
    
    # Create result DataFrame
    result = pd.DataFrame(index=data.index)
    
    # Calculate momentum (rate of change)
    mfi_momentum = data['mfi'].pct_change(periods=short_period)
    
    # Calculate acceleration (change in momentum)
    mfi_acceleration = mfi_momentum.diff()
    
    # Generate momentum signals
    # 1 = positive momentum, -1 = negative momentum, 0 = neutral
    momentum_signal = np.zeros(len(data))
    
    # Use long period SMA for signal smoothing
    momentum_sma = mfi_momentum.rolling(window=long_period).mean()
    
    for i in range(long_period, len(data)):
        if pd.notna(momentum_sma.iloc[i]):
            if momentum_sma.iloc[i] > 0.01:  # Positive momentum threshold
                momentum_signal[i] = 1
            elif momentum_sma.iloc[i] < -0.01:  # Negative momentum threshold
                momentum_signal[i] = -1
    
    # Assign to result DataFrame
    result['mfi_momentum'] = mfi_momentum
    result['mfi_acceleration'] = mfi_acceleration
    result['mfi_momentum_signal'] = momentum_signal
    
    return result
