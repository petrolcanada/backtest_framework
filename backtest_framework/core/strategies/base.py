"""
Base Strategy module defining the abstract strategy interface.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import pandas as pd

from backtest_framework.core.indicators.calculator import IndicatorCalculator

class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies with indicator parameter support.
    """
    
    def __init__(self, indicator_params: Optional[Dict[str, Dict[str, Any]]] = None):
        """
        Initialize the strategy with optional indicator parameter overrides.
        
        Args:
            indicator_params: Dictionary mapping indicator names to their parameter overrides
                            Example: {
                                "ADX": {"window": 14, "sma_length": 3},
                                "ADX_CONSECUTIVE_DOWN": {"required_days": 7}
                            }
        """
        self.indicator_params = indicator_params or {}
    
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
    
    def set_indicator_params(self, indicator_name: str, **params) -> None:
        """
        Set parameters for a specific indicator.
        
        Args:
            indicator_name: Name of the indicator
            **params: Parameter key-value pairs to set
        """
        if indicator_name not in self.indicator_params:
            self.indicator_params[indicator_name] = {}
        self.indicator_params[indicator_name].update(params)
    
    def get_indicator_params(self, indicator_name: str) -> Dict[str, Any]:
        """
        Get parameter overrides for a specific indicator.
        
        Args:
            indicator_name: Name of the indicator
            
        Returns:
            Dictionary of parameter overrides for the indicator
        """
        return self.indicator_params.get(indicator_name, {})
    
    def update_indicator_params(self, params_dict: Dict[str, Dict[str, Any]]) -> None:
        """
        Update indicator parameters with a dictionary.
        
        Args:
            params_dict: Dictionary mapping indicator names to parameter dictionaries
        """
        for indicator_name, params in params_dict.items():
            self.set_indicator_params(indicator_name, **params)
    
    def print_indicator_config(self) -> None:
        """
        Universal method to print current indicator configuration for debugging.
        """
        info = self.get_strategy_info()
        
        print("Strategy Indicator Configuration:")
        print("=" * 50)
        print(f"Strategy: {info['name']}")
        print(f"Required Indicators: {len(info['required_indicators'])}")
        
        for indicator in info['required_indicators']:
            if indicator in info['indicator_details']:
                details = info['indicator_details'][indicator]
                print(f"\n{indicator}:")
                if 'error' in details:
                    print(f"  Error: {details['error']}")
                else:
                    print(f"  Default params: {details['default_params']}")
                    if details['custom_params']:
                        print(f"  Custom params:  {details['custom_params']}")
                    print(f"  Final params:   {details['final_params']}")
                    print(f"  Outputs: {details['outputs']}")
    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for the strategy by computing required indicators with custom parameters.
        
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
                
        # Only compute if needed, passing custom parameters
        if not all_indicators_present:
            return IndicatorCalculator.compute(
                data, 
                self.required_indicators, 
                override_params=self.indicator_params
            )
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
        # Ensure all required indicators are computed with custom parameters
        data = self.prepare_data(data)
        
        # Generate signals
        return self.generate_signals(data)
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the strategy including indicator parameters.
        
        Returns:
            Dictionary with strategy information
        """
        from backtest_framework.core.indicators.registry import IndicatorRegistry
        
        info = {
            "name": self.name,
            "required_indicators": self.required_indicators,
            "indicator_params": self.indicator_params,
            "indicator_details": {}
        }
        
        # Add details about each required indicator
        for indicator in self.required_indicators:
            try:
                indicator_info = IndicatorRegistry.get(indicator)
                default_params = indicator_info.get('params', {})
                custom_params = self.get_indicator_params(indicator)
                final_params = {**default_params, **custom_params}
                
                info["indicator_details"][indicator] = {
                    "default_params": default_params,
                    "custom_params": custom_params,
                    "final_params": final_params,
                    "inputs": indicator_info.get('inputs', []),
                    "outputs": indicator_info.get('outputs', [])
                }
            except KeyError:
                info["indicator_details"][indicator] = {
                    "error": "Indicator not found in registry"
                }
        
        return info
