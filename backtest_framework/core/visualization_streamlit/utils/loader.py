"""
Utilities for loading backtest results.
"""
import pickle
from typing import Tuple, Dict, Any
import pandas as pd
from pathlib import Path


def load_results(results_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, Any, Dict[str, Any]]:
    """
    Load backtest results from a pickle file.
    
    Args:
        results_path: Path to the pickled results file
        
    Returns:
        Tuple of (data, results, engine, strategy_info)
    """
    path = Path(results_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")
    
    with open(path, 'rb') as f:
        results_data = pickle.load(f)
    
    # Handle different pickle formats
    if isinstance(results_data, dict):
        # New format with all components
        data = results_data.get('data')
        results = results_data.get('results')
        engine = results_data.get('engine')
        strategy_info = results_data.get('strategy_info', {})
    else:
        # Legacy format - just results DataFrame
        data = results_data
        results = results_data
        engine = None
        strategy_info = {}
    
    # Ensure we have required data
    if data is None or results is None:
        raise ValueError("Invalid results file format")
    
    # Fill in default strategy info if missing
    if not strategy_info:
        strategy_info = {
            'name': 'Unknown Strategy',
            'ticker': 'Unknown',
            'params': {}
        }
    
    return data, results, engine, strategy_info
