"""
Base Strategy module defining the abstract strategy interface.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import pandas as pd

from backtest_framework.core.indicators.calculator import IndicatorCalculator

class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    """
    
    @property
    @abstractmethod
    def required_indicators(self) -> List[str]:
        """
        List of indicator names required by this strategy.
        
        Returns:
            List of indicator names
        """
        pass
    
    @property
    def name(self) -> str:
        """
        Strategy name, defaults to class name.
        
        Returns:
            Strategy name
        """
        return self.__class__.__name__
    
    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for the strategy by computing required indicators.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with computed indicators
        """
        # Check if indicators are already computed
        all_indicators_present = True
        required_outputs = []
        
        # Get all required output columns
        for indicator in self.required_indicators:
            outputs = IndicatorCalculator.get_indicator_outputs(indicator)
            if outputs:
                required_outputs.extend(outputs)
        
        # Check if any required outputs are missing
        for output in required_outputs:
            if output not in data.columns:
                all_indicators_present = False
                break
                
        # Only compute if needed
        if not all_indicators_present:
            return IndicatorCalculator.compute(data, self.required_indicators)
        else:
            return data
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on the data and computed indicators.
        
        Args:
            data: DataFrame with OHLCV data and computed indicators
            
        Returns:
            DataFrame with added signal columns
        """
        pass
    
    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Run the strategy on the provided data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with computed indicators and signals
        """
        # Ensure all required indicators are computed
        data = self.prepare_data(data)
        
        # Generate signals
        return self.generate_signals(data)
