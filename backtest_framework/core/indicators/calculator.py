"""
Indicator Calculator module for computing technical indicators.
"""
from typing import Dict, List, Any, Optional, Union
import pandas as pd

from backtest_framework.core.indicators.registry import IndicatorRegistry
from backtest_framework.core.indicators.resolver import DependencyResolver

class IndicatorCalculator:
    """
    Computes technical indicators with automatic dependency resolution.
    """
    
    @classmethod
    def get_indicator_outputs(cls, indicator: str) -> List[str]:
        """
        Get the output column names that an indicator produces.
        
        Args:
            indicator: Name of the indicator
            
        Returns:
            List of output column names
        """
        ind_data = IndicatorRegistry.get(indicator)
        if not ind_data:
            return []
            
        # Return the outputs if defined, otherwise return the indicator name itself
        return ind_data.get('outputs', [indicator])
    
    @classmethod
    def compute(cls, data: pd.DataFrame, indicators: Union[str, List[str]], 
               override_params: Optional[Dict[str, Dict[str, Any]]] = None) -> pd.DataFrame:
        """
        Compute indicators with automatic dependency resolution.
        
        Args:
            data: DataFrame with OHLCV data
            indicators: Single indicator name or list of indicator names to compute
            override_params: Dictionary mapping indicator names to parameter overrides
            
        Returns:
            DataFrame with computed indicators
            
        Raises:
            ValueError: If indicators cannot be computed due to missing data or circular dependencies
        """
        if isinstance(indicators, str):
            indicators = [indicators]
            
        if override_params is None:
            override_params = {}
            
        # Create a copy of the data to avoid modifying the original
        result_data = data.copy()
        
        # Build a list of all required indicators including dependencies
        all_indicators = []
        for indicator in indicators:
            # Add the main indicator
            if indicator not in all_indicators:
                all_indicators.append(indicator)
                
            # Add its dependencies if they're registered indicators
            ind_data = IndicatorRegistry.get(indicator)
            for input_col in ind_data['inputs']:
                if input_col in IndicatorRegistry._indicators and input_col not in all_indicators:
                    all_indicators.append(input_col)
        
        # Sort indicators by dependency (basic implementation)
        # This is a simplified topological sort that doesn't handle all edge cases
        sorted_indicators = []
        remaining = all_indicators.copy()
        
        # Keep iterating until all indicators are processed
        while remaining:
            for indicator in list(remaining):
                ind_data = IndicatorRegistry.get(indicator)
                
                # Check if all dependencies are already in sorted_indicators or are base columns
                deps_satisfied = True
                for input_col in ind_data['inputs']:
                    if input_col in IndicatorRegistry._indicators and input_col not in sorted_indicators:
                        deps_satisfied = False
                        break
                
                if deps_satisfied:
                    # All dependencies are satisfied, add to sorted list
                    sorted_indicators.append(indicator)
                    remaining.remove(indicator)
            
            # If we can't make progress, there might be a circular dependency
            if not sorted_indicators and remaining:
                raise ValueError(f"Possible circular dependency detected among indicators: {remaining}")
        
        # Compute each indicator in the sorted order
        for indicator in sorted_indicators:
            # Get parameters for this indicator
            params = override_params.get(indicator, {})
            
            # Debug output
            print(f"Computing {indicator} with inputs {IndicatorRegistry.get(indicator)['inputs']}")
            
            # Compute the indicator
            ind_data = IndicatorRegistry.get(indicator)
            
            # Check if all required inputs exist in the data
            missing_inputs = [inp for inp in ind_data['inputs'] if inp not in result_data.columns]
            if missing_inputs:
                raise ValueError(f"Missing required inputs for indicator '{indicator}': {missing_inputs}")
                
            # Compute the indicator
            try:
                result = ind_data['function'](result_data, **params)
                
                # If the result is a Series, convert to DataFrame with the indicator name as column
                if isinstance(result, pd.Series):
                    result_data[indicator] = result
                # If the result is a DataFrame, merge it with the input data
                elif isinstance(result, pd.DataFrame):
                    # Add all output columns to result_data
                    for col in result.columns:
                        result_data[col] = result[col]
            except Exception as e:
                print(f"Error computing {indicator}: {str(e)}")
                raise
        
        return result_data
    
    @classmethod
    def compute_all(cls, data: pd.DataFrame, 
                  override_params: Optional[Dict[str, Dict[str, Any]]] = None) -> pd.DataFrame:
        """
        Compute all registered indicators.
        
        Args:
            data: DataFrame with OHLCV data
            override_params: Dictionary mapping indicator names to parameter overrides
            
        Returns:
            DataFrame with all indicators computed
        """
        return cls.compute(data, IndicatorRegistry.list_all(), override_params)
