"""Utility functions for the backtesting framework."""
import os
import pandas as pd
import warnings
from typing import Dict, List, Optional, Union, Any

def read_index_constituents(filepath: str = None, sheet_name: str = 'sp500') -> pd.DataFrame:
    """
    Read index constituents from an Excel file.
    
    Args:
        filepath: Path to the Excel file (defaults to index_constituent.xlsx in current directory)
        sheet_name: Sheet name in the Excel file
        
    Returns:
        DataFrame with ticker information
    """
    if filepath is None:
        # First check in current directory
        if os.path.exists('./index_constituent.xlsx'):
            filepath = './index_constituent.xlsx'
        # Then check in data directory
        else:
            data_dir = os.path.join(os.path.expanduser("~"), "local_script", 
                                   "Local Technical Indicator Data")
            filepath = os.path.join(data_dir, 'index_constituent.xlsx')
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Index constituent file not found at {filepath}")
    
    return pd.read_excel(filepath, sheet_name=sheet_name)

def suppress_warnings():
    """Suppress all warnings (useful for cleaner notebook output)."""
    warnings.filterwarnings('ignore')
    
    # Suppress pandas future warnings
    pd.options.mode.chained_assignment = None

class Timer:
    """Simple timer for measuring execution time."""
    
    def __init__(self):
        """Initialize the timer."""
        import time
        self.start_time = time.time()
    
    def elapsed(self) -> float:
        """
        Get elapsed time in seconds.
        
        Returns:
            Elapsed time in seconds
        """
        import time
        return time.time() - self.start_time
    
    def reset(self):
        """Reset the timer."""
        import time
        self.start_time = time.time()
    
    def elapsed_str(self) -> str:
        """
        Get formatted elapsed time string.
        
        Returns:
            Formatted elapsed time (e.g., "2.5 seconds" or "1 minute 30 seconds")
        """
        elapsed = self.elapsed()
        if elapsed < 60:
            return f"{elapsed:.2f} seconds"
        else:
            minutes = int(elapsed / 60)
            seconds = elapsed % 60
            return f"{minutes} minute{'s' if minutes != 1 else ''} {seconds:.2f} seconds"

def format_number(value: float, precision: int = 2, include_sign: bool = False) -> str:
    """
    Format a number with specified precision and optional sign.
    
    Args:
        value: Number to format
        precision: Number of decimal places
        include_sign: Whether to include + sign for positive numbers
        
    Returns:
        Formatted number string
    """
    if pd.isna(value):
        return "N/A"
    
    format_str = f"{{:+.{precision}f}}" if include_sign else f"{{:.{precision}f}}"
    return format_str.format(value)

def format_percentage(value: float, precision: int = 2) -> str:
    """
    Format a number as a percentage with specified precision.
    
    Args:
        value: Number to format (e.g., 0.1234 for 12.34%)
        precision: Number of decimal places
        
    Returns:
        Formatted percentage string (e.g., "12.34%")
    """
    if pd.isna(value):
        return "N/A"
    
    return f"{value * 100:.{precision}f}%"

# Parameter handling utilities
def clean_params(**kwargs) -> Dict[str, Any]:
    """
    Build parameter dictionary, filtering out None values.
    
    This utility function is commonly used in strategy constructors to build
    clean parameter dictionaries for indicator configuration, removing any
    parameters that are None.
    
    Args:
        **kwargs: Parameter key-value pairs
        
    Returns:
        Dictionary with non-None values only
        
    Example:
        >>> clean_params(window=14, period=None, threshold=0.5)
        {'window': 14, 'threshold': 0.5}
    """
    return {k: v for k, v in kwargs.items() if v is not None}

def filter_empty_dicts(params_dict: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Filter out empty dictionaries from a nested parameter dictionary.
    
    This is useful for removing indicators that have no parameter overrides,
    keeping the configuration clean.
    
    Args:
        params_dict: Dictionary mapping names to parameter dictionaries
        
    Returns:
        Dictionary with empty parameter dictionaries removed
        
    Example:
        >>> filter_empty_dicts({'ADX': {'window': 14}, 'RSI': {}, 'MFI': {'period': 20}})
        {'ADX': {'window': 14}, 'MFI': {'period': 20}}
    """
    return {k: v for k, v in params_dict.items() if v}
