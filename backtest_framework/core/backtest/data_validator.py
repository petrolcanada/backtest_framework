"""
Data validation and preparation module for backtesting.
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, Union


class DataValidator:
    """
    Validates and prepares data for backtesting.
    """
    
    def __init__(self, include_dividends: bool = True):
        """
        Initialize data validator.
        
        Args:
            include_dividends: Whether to include dividend data
        """
        self.include_dividends = include_dividends
    
    def validate_and_prepare_data(self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
                                start_date: Optional[str] = None,
                                end_date: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Validate and prepare data for backtesting.
        
        Args:
            data: Single DataFrame or dict of DataFrames
            start_date: Start date for filtering
            end_date: End date for filtering
            
        Returns:
            Dictionary mapping tickers to prepared DataFrames
        """
        # Convert single DataFrame to dict format
        if isinstance(data, pd.DataFrame):
            data_dict = {'SINGLE': data}
        else:
            data_dict = data
        
        # Validate each DataFrame
        validated_dict = {}
        for ticker, df in data_dict.items():
            validated_df = self._validate_single_dataframe(df, ticker)
            validated_df = self._prepare_dividend_data(validated_df)
            validated_dict[ticker] = validated_df
        
        # Filter by date range if specified
        if start_date or end_date:
            validated_dict = self._filter_data_by_date(validated_dict, start_date, end_date)
        
        return validated_dict
    
    def _validate_single_dataframe(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Validate a single DataFrame.
        
        Args:
            df: DataFrame to validate
            ticker: Ticker symbol for error messages
            
        Returns:
            Validated DataFrame
        """
        if df.empty:
            raise ValueError(f"DataFrame for {ticker} is empty")
        
        # Check for required columns
        required_columns = ['Close']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns for {ticker}: {missing_columns}")
        
        # Check for NaN values in critical columns
        critical_columns = ['Close', 'Open', 'High', 'Low', 'Volume']
        available_critical = [col for col in critical_columns if col in df.columns]
        
        for col in available_critical:
            if df[col].isna().any():
                print(f"Warning: NaN values found in {col} for {ticker}. Forward filling...")
                df[col] = df[col].fillna(method='ffill')
        
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
            except Exception as e:
                raise ValueError(f"Cannot convert index to datetime for {ticker}: {e}")
        
        # Sort by date
        df = df.sort_index()
        
        # Check for duplicate dates
        if df.index.duplicated().any():
            print(f"Warning: Duplicate dates found for {ticker}. Keeping last occurrence...")
            df = df[~df.index.duplicated(keep='last')]
        
        return df
    
    def _prepare_dividend_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare dividend data for backtesting.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with Dividends column
        """
        df = df.copy()
        
        # If dividends column already exists, use it
        if 'Dividends' in df.columns:
            # Clean dividend data
            df['Dividends'] = df['Dividends'].fillna(0.0)
            df['Dividends'] = df['Dividends'].clip(lower=0.0)  # Ensure non-negative
            return df
        
        # If include_dividends is False, create empty dividend column
        if not self.include_dividends:
            df['Dividends'] = 0.0
            return df
        
        # Try to create dividend column or warn user
        df['Dividends'] = 0.0
        print("Warning: No dividend data available. Dividends set to 0.")
        print("Available columns:", list(df.columns))
        
        return df
    
    def _filter_data_by_date(self, data_dict: Dict[str, pd.DataFrame],
                           start_date: Optional[str],
                           end_date: Optional[str]) -> Dict[str, pd.DataFrame]:
        """
        Filter data by date range.
        
        Args:
            data_dict: Dictionary of DataFrames
            start_date: Start date string
            end_date: End date string
            
        Returns:
            Filtered data dictionary
        """
        filtered_dict = {}
        
        for ticker, df in data_dict.items():
            filtered_df = df.copy()
            
            if start_date:
                start_date_dt = pd.to_datetime(start_date).normalize()
                filtered_df = filtered_df[filtered_df.index >= start_date_dt]
            
            if end_date:
                end_date_dt = pd.to_datetime(end_date).normalize()
                filtered_df = filtered_df[filtered_df.index <= end_date_dt]
            
            if filtered_df.empty:
                print(f"Warning: No data for {ticker} in specified date range")
            
            filtered_dict[ticker] = filtered_df
        
        return filtered_dict
    
    def validate_signals(self, df: pd.DataFrame, ticker: str = "SINGLE") -> pd.DataFrame:
        """
        Validate trading signals in DataFrame.
        
        Args:
            df: DataFrame with signals
            ticker: Ticker name for error messages
            
        Returns:
            DataFrame with validated signals
        """
        df = df.copy()
        
        # Check for required signal columns
        if 'buy_signal' not in df.columns or 'sell_signal' not in df.columns:
            raise ValueError(f"Data for {ticker} must contain 'buy_signal' and 'sell_signal' columns")
        
        # Ensure signals are binary (0 or 1)
        df['buy_signal'] = df['buy_signal'].fillna(0).astype(int)
        df['sell_signal'] = df['sell_signal'].fillna(0).astype(int)
        
        # Clip to valid range
        df['buy_signal'] = df['buy_signal'].clip(0, 1)
        df['sell_signal'] = df['sell_signal'].clip(0, 1)
        
        # Check for simultaneous buy and sell signals
        simultaneous = (df['buy_signal'] == 1) & (df['sell_signal'] == 1)
        if simultaneous.any():
            print(f"Warning: Simultaneous buy and sell signals found for {ticker}. "
                  f"Setting both to 0 for {simultaneous.sum()} dates.")
            df.loc[simultaneous, ['buy_signal', 'sell_signal']] = 0
        
        return df
    
    def get_date_range(self, data_dict: Dict[str, pd.DataFrame]) -> tuple:
        """
        Get the overall date range from data.
        
        Args:
            data_dict: Dictionary of DataFrames
            
        Returns:
            Tuple of (start_date, end_date) as strings
        """
        all_dates = []
        for df in data_dict.values():
            if not df.empty:
                all_dates.extend([df.index.min(), df.index.max()])
        
        if not all_dates:
            return None, None
        
        start_date = min(all_dates).strftime('%Y-%m-%d')
        end_date = max(all_dates).strftime('%Y-%m-%d')
        
        return start_date, end_date
    
    def check_data_alignment(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, any]:
        """
        Check alignment of multiple DataFrames.
        
        Args:
            data_dict: Dictionary of DataFrames
            
        Returns:
            Dictionary with alignment information
        """
        if len(data_dict) <= 1:
            return {'aligned': True, 'common_dates': len(list(data_dict.values())[0]) if data_dict else 0}
        
        # Find common dates
        date_sets = [set(df.index) for df in data_dict.values() if not df.empty]
        if not date_sets:
            return {'aligned': True, 'common_dates': 0}
        
        common_dates = set.intersection(*date_sets)
        
        return {
            'aligned': len(common_dates) > 0,
            'common_dates': len(common_dates),
            'total_unique_dates': len(set.union(*date_sets)),
            'tickers': list(data_dict.keys()),
            'date_counts': {ticker: len(df) for ticker, df in data_dict.items()}
        }
