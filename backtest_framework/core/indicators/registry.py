"""
Indicator Registry module for managing technical indicators and their dependencies.
"""
from typing import Callable, Dict, List, Any, Set, Optional
import pandas as pd
import inspect

class IndicatorRegistry:
    """
    Central registry for all technical indicators in the system.
    Handles indicator registration, dependency tracking, and computation.
    """
    _indicators: Dict[str, Dict[str, Any]] = {}
    
    @classmethod
    def register(cls, name: str, inputs: List[str], params: Dict[str, Any] = None, outputs: Optional[List[str]] = None):
        """
        Decorator to register an indicator function with its metadata.
        
        Args:
            name: Unique identifier for the indicator
            inputs: List of input columns/indicators required
            params: Default parameters for the indicator
            outputs: List of output column names generated (defaults to [name] if None)
        
        Returns:
            Decorator function that registers the indicator
        """
        if params is None:
            params = {}
        if outputs is None:
            outputs = [name]
            
        def decorator(func: Callable):
            cls._indicators[name] = {
                'function': func,
                'inputs': inputs,
                'params': params,
                'outputs': outputs,
                'sig': inspect.signature(func)
            }
            return func
        return decorator
    
    @classmethod
    def get(cls, name: str) -> Dict[str, Any]:
        """
        Get indicator metadata by name.
        
        Args:
            name: Indicator name to retrieve
            
        Returns:
            Dictionary with indicator metadata
            
        Raises:
            KeyError: If the indicator is not registered
        """
        if name not in cls._indicators:
            raise KeyError(f"Indicator '{name}' not registered. Available indicators: {list(cls._indicators.keys())}")
        return cls._indicators[name]
    
    @classmethod
    def compute(cls, name: str, data: pd.DataFrame, **override_params) -> pd.DataFrame:
        """
        Compute an indicator and add its result to the DataFrame.
        
        Args:
            name: Name of the indicator to compute
            data: DataFrame containing required input data
            **override_params: Parameters to override default values
            
        Returns:
            DataFrame with the indicator results added
            
        Raises:
            KeyError: If the indicator is not registered
            ValueError: If required inputs are missing from the data
        """
        if name not in cls._indicators:
            raise KeyError(f"Indicator '{name}' not registered")
            
        indicator = cls._indicators[name]
        
        # Check that all required inputs exist in the data
        missing_inputs = [inp for inp in indicator['inputs'] if inp not in data.columns]
        if missing_inputs:
            raise ValueError(f"Missing required inputs for indicator '{name}': {missing_inputs}")
        
        # Merge default params with overrides
        params = {**indicator['params'], **override_params}
        
        # Check that all required parameters are provided
        sig = indicator['sig']
        required_params = {
            name: param for name, param in sig.parameters.items()
            if param.default == inspect.Parameter.empty and name != 'data'
        }
        
        missing_params = [name for name in required_params if name not in params]
        if missing_params:
            raise ValueError(f"Missing required parameters for indicator '{name}': {missing_params}")
        
        # Compute the indicator
        result = indicator['function'](data, **params)
        
        # If the result is a Series, convert to DataFrame with the indicator name as column
        if isinstance(result, pd.Series):
            data[name] = result
        # If the result is a DataFrame, merge it with the input data
        elif isinstance(result, pd.DataFrame):
            # Ensure we don't have duplicate columns
            result_columns = [col for col in result.columns if col not in data.columns or col in indicator['outputs']]
            data = pd.concat([data, result[result_columns]], axis=1)
        
        return data
    
    @classmethod
    def list_all(cls) -> List[str]:
        """List all registered indicators."""
        return list(cls._indicators.keys())
    
    @classmethod
    def get_dependencies(cls, indicators: List[str]) -> Set[str]:
        """
        Get all dependencies for a list of indicators.
        
        Args:
            indicators: List of indicator names
            
        Returns:
            Set of all dependencies required by the indicators
        """
        dependencies = set()
        for indicator in indicators:
            indicator_deps = cls.get(indicator)['inputs']
            dependencies.update(indicator_deps)
            
            # Recursively check if any dependencies are also indicators
            for dep in indicator_deps:
                if dep in cls._indicators:
                    dependencies.update(cls.get_dependencies([dep]))
        
        return dependencies
