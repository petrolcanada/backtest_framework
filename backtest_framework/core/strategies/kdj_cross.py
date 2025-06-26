"""
KDJ Golden/Dead Cross strategy implementations.
"""
from typing import List
import pandas as pd
import numpy as np
from backtest_framework.core.strategies.base import BaseStrategy

class GoldenDeadCrossStrategyBase(BaseStrategy):
    """
    Base class for strategies based on KDJ Golden Cross (J line crosses above K line) and 
    Dead Cross (J line crosses below K line) signals.
    
    This strategy generates clean buy/sell signals. All risk management (drawdown protection,
    trailing stops, etc.) is handled by separate risk manager modules.
    """
    
    def __init__(self):
        """
        Initialize the strategy.
        
        Note: Risk management is now handled by separate risk manager modules,
        not within the strategy itself.
        """
        pass
    
    def _generate_signals_from_kdj(self, data: pd.DataFrame, 
                                 k_col: str, j_col: str, 
                                 k_slope_col: str, d_slope_col: str) -> pd.DataFrame:
        """
        Generate clean buy/sell signals based on golden cross and dead cross patterns.
        
        Args:
            data: DataFrame with OHLCV data and computed indicators
            k_col: Name of the K column
            j_col: Name of the J column
            k_slope_col: Name of the K slope column
            d_slope_col: Name of the D slope column
            
        Returns:
            DataFrame with added signal columns (buy_signal, sell_signal)
        """
        # Initialize signal columns
        data['buy_signal'] = 0
        data['sell_signal'] = 0
        
        # Track position state to avoid duplicate signals
        in_position = False
        
        # Iterate through the DataFrame to generate signals
        for i in range(1, len(data)):
            # Generate Buy Signal when J crosses above K (Golden Cross)
            if not in_position:
                if (data[j_col].iloc[i-1] <= data[k_col].iloc[i-1] and 
                    data[j_col].iloc[i] >= data[k_col].iloc[i] and 
                    data[k_slope_col].iloc[i] > 0 and 
                    data[d_slope_col].iloc[i] > 0 and 
                    data[j_col].iloc[i] <= 100):
                    
                    data.at[data.index[i], 'buy_signal'] = 1
                    in_position = True
            
            # Generate Sell Signal when J crosses below K (Dead Cross)
            elif in_position:
                if (data[j_col].iloc[i-1] > data[k_col].iloc[i-1] and 
                    data[j_col].iloc[i] < data[k_col].iloc[i]):
                    
                    data.at[data.index[i], 'sell_signal'] = 1
                    in_position = False
        
        return data
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract method to be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement generate_signals method")


class GoldenDeadCrossStrategyMonthly(GoldenDeadCrossStrategyBase):
    """
    Trading strategy based on Monthly KDJ Golden Cross and Dead Cross signals.
    Uses longer-term KDJ calculation with monthly-equivalent periods.
    """
    
    @property
    def required_indicators(self) -> List[str]:
        """
        List indicators required by this strategy.
        
        Returns:
            List of required indicator names
        """
        return ["MONTHLY_KDJ", "MONTHLY_KDJ_SLOPES"]
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate buy/sell signals based on Monthly KDJ golden cross and dead cross patterns.
        
        Args:
            data: DataFrame with OHLCV data and computed indicators
            
        Returns:
            DataFrame with added signal columns (buy_signal, sell_signal)
        """
        # Ensure all required indicators are computed
        data = self.prepare_data(data)
        
        # Generate signals using monthly KDJ
        return self._generate_signals_from_kdj(
            data,
            k_col='monthly_k',
            j_col='monthly_j',
            k_slope_col='monthly_k_slope',
            d_slope_col='monthly_d_slope'
        )


class GoldenDeadCrossStrategyDaily(GoldenDeadCrossStrategyBase):
    """
    Trading strategy based on Daily KDJ Golden Cross and Dead Cross signals.
    Uses standard KDJ calculation with daily periods.
    """
    
    @property
    def required_indicators(self) -> List[str]:
        """
        List indicators required by this strategy.
        
        Returns:
            List of required indicator names
        """
        return ["KDJ", "KDJ_SLOPES"]
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate buy/sell signals based on Daily KDJ golden cross and dead cross patterns.
        
        Args:
            data: DataFrame with OHLCV data and computed indicators
            
        Returns:
            DataFrame with added signal columns (buy_signal, sell_signal)
        """
        # Ensure all required indicators are computed
        data = self.prepare_data(data)
        
        # Generate signals using daily KDJ
        return self._generate_signals_from_kdj(
            data,
            k_col='k',
            j_col='j',
            k_slope_col='k_slope',
            d_slope_col='d_slope'
        )
