"""
Data Loader module for fetching and preprocessing price data.
"""
import os
from typing import List, Optional, Union, Dict, Any
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
try:
    from pandas_datareader import data as pdr
    DATAREADER_AVAILABLE = True
except ImportError:
    DATAREADER_AVAILABLE = False
    print("Warning: pandas-datareader not available. FFR functionality will be disabled.")

class DataLoader:
    """
    Handles loading OHLCV data from various sources including yfinance, CSV files, etc.
    """
    
    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize the data loader with a specified data directory.
        
        Args:
            data_dir: Directory for storing/retrieving cached data files
        """
        # Set default data directory if none provided
        if data_dir is None:
            self.data_dir = os.path.join(os.path.expanduser("~"), "local_script", 
                                        "Local Technical Indicator Data", "security_data")
        else:
            self.data_dir = data_dir
            
        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
        
    def load(self, tickers: Union[str, List[str]], 
           period: str = '1y', 
           resample_period: str = 'D',
           start_date: Optional[str] = None,
           end_date: Optional[str] = None,
           mode: str = 'incremental') -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Load data for one or more tickers with three simple modes.
        
        Args:
            tickers: Single ticker or list of tickers to load
            period: Time period to load ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'max')
            resample_period: Frequency to resample data to ('D', 'W', 'M', 'Q', etc.)
            start_date: Specific start date (format: 'YYYY-MM-DD'), overrides period if provided
            end_date: Specific end date (format: 'YYYY-MM-DD')
            mode: Data loading mode:
                  - 'full_reload': Download max period from yfinance, overwrite CSV
                  - 'incremental': Update last 10 days from CSV file to current date
                  - 'no_reload': Use existing CSV file as-is, no API calls
            
        Returns:
            DataFrame for single ticker or Dictionary mapping tickers to DataFrames
        """
        if isinstance(tickers, str):
            tickers = [tickers]
            
        result = {}
        
        for ticker in tickers:
            print(f"\n[DataLoader] Processing {ticker} in '{mode}' mode...")
            if mode == 'full_reload':
                print(f"[DataLoader] Full reload: Downloading max period from yfinance for {ticker}")
                df = self._load_from_yfinance(ticker, 'max', None, None)
                if not df.empty:
                    self._save_to_csv(ticker, df)
                    print(f"[DataLoader] Saved {len(df)} records to CSV for {ticker} ({df.index[0]} to {df.index[-1]})")
                else:
                    print(f"[DataLoader] Warning: No data retrieved for {ticker}")
                
            elif mode == 'incremental':
                ticker_file = os.path.join(self.data_dir, f"{ticker}_daily.csv")
                if os.path.exists(ticker_file):
                    print(f"[DataLoader] Found existing CSV for {ticker}, loading...")
                    existing_df = self._load_from_csv(ticker)
                    print(f"[DataLoader] Existing data: {len(existing_df)} records ({existing_df.index[0]} to {existing_df.index[-1]})")
                    
                    if len(existing_df) >= 10:
                        last_10th_date = existing_df.index[-10].strftime('%Y-%m-%d')
                        print(f"[DataLoader] Fetching updates from {last_10th_date} to current for {ticker}")
                        
                        new_df = self._load_from_yfinance(ticker, None, last_10th_date, None)
                        
                        if not new_df.empty:
                            print(f"[DataLoader] Downloaded {len(new_df)} new records")
                            # Combine datasets, dropping duplicates
                            df = pd.concat([existing_df.iloc[:-10], new_df])
                            df = df[~df.index.duplicated(keep='last')]
                            self._save_to_csv(ticker, df)
                            print(f"[DataLoader] Updated CSV: {len(df)} total records ({df.index[0]} to {df.index[-1]})")
                        else:
                            print(f"[DataLoader] No new data available, using existing data")
                            df = existing_df
                    else:
                        print(f"[DataLoader] Insufficient existing data ({len(existing_df)} < 10 records), doing full reload")
                        df = self._load_from_yfinance(ticker, 'max', None, None)
                        if not df.empty:
                            self._save_to_csv(ticker, df)
                            print(f"[DataLoader] Saved {len(df)} records to CSV for {ticker}")
                else:
                    print(f"[DataLoader] No CSV file found for {ticker}, doing full reload")
                    df = self._load_from_yfinance(ticker, 'max', None, None)
                    if not df.empty:
                        self._save_to_csv(ticker, df)
                        print(f"[DataLoader] Created new CSV with {len(df)} records for {ticker}")
                    
            elif mode == 'no_reload':
                ticker_file = os.path.join(self.data_dir, f"{ticker}_daily.csv")
                if os.path.exists(ticker_file):
                    print(f"[DataLoader] Loading existing CSV file for {ticker}")
                    df = self._load_from_csv(ticker)
                    print(f"[DataLoader] Loaded {len(df)} records from CSV ({df.index[0]} to {df.index[-1]})")
                else:
                    raise FileNotFoundError(f"No CSV file found for {ticker}. Use 'full_reload' or 'incremental' mode first.")
                    
            else:
                raise ValueError(f"Invalid mode: {mode}. Use 'full_reload', 'incremental', or 'no_reload'")
            
            # Filter by date range if specified
            if start_date or end_date or period != 'max':
                original_len = len(df)
                df = self._filter_by_date_range(df, period, start_date, end_date)
                if len(df) != original_len:
                    print(f"[DataLoader] Filtered to {len(df)} records for period/date range ({df.index[0]} to {df.index[-1]})")
            
            # Apply resampling if needed
            if resample_period != 'D':
                original_len = len(df)
                df = self._resample_data(df, resample_period)
                print(f"[DataLoader] Resampled from {original_len} daily records to {len(df)} {resample_period} records")
                
            result[ticker] = df
            
        print(f"\n[DataLoader] Data loading completed for {len(tickers)} ticker(s)")
        return result if len(tickers) > 1 else result[tickers[0]]
    
    def load_fed_funds_rate(self, start_date: str = '1990-01-01', 
                          force_update: bool = False) -> pd.Series:
        """
        Load Federal Funds Rate data from FRED via pandas-datareader.
        Updates automatically if data is more than 3 months behind current date.
        Auto-backfills latest rate to current date if data is behind.
        
        Args:
            start_date: Start date for FFR data (default: '1990-01-01')
            force_update: Force update even if data is recent
            
        Returns:
            Series with Federal Funds Rate data (daily frequency)
        """
        if not DATAREADER_AVAILABLE:
            print("pandas-datareader not available. Cannot load FFR data.")
            return pd.Series(dtype=float)
        
        ffr_file = os.path.join(self.data_dir, "fed_funds_rate.csv")
        current_date = datetime.now()
        three_months_ago = current_date - timedelta(days=90)
        
        # Check if we need to update the data
        needs_update = force_update
        existing_data = None
        
        if os.path.exists(ffr_file) and not force_update:
            try:
                # Try reading with different possible date column names
                existing_data = None
                try:
                    # First try with 'Date' column name
                    existing_data = pd.read_csv(ffr_file, index_col='Date', parse_dates=True)
                except (ValueError, KeyError):
                    try:
                        # Try with 'DATE' column name (all caps)
                        existing_data = pd.read_csv(ffr_file, index_col='DATE', parse_dates=True)
                    except (ValueError, KeyError):
                        # Try reading without specifying index column and fix it
                        temp_data = pd.read_csv(ffr_file)
                        # Find the date column (could be 'Date', 'DATE', or first column)
                        date_col = None
                        if 'Date' in temp_data.columns:
                            date_col = 'Date'
                        elif 'DATE' in temp_data.columns:
                            date_col = 'DATE'
                        elif len(temp_data.columns) >= 2:
                            # Assume first column is date if we can't find a proper date column
                            date_col = temp_data.columns[0]
                        
                        if date_col:
                            temp_data[date_col] = pd.to_datetime(temp_data[date_col])
                            existing_data = temp_data.set_index(date_col)
                        else:
                            raise ValueError("Could not identify date column")
                
                if existing_data is not None:
                    last_data_date = existing_data.index.max()
                    
                    # Update if data is more than 3 months old
                    if last_data_date < three_months_ago:
                        print(f"FFR data is {(current_date - last_data_date).days} days old. Updating...")
                        needs_update = True
                    else:
                        print(f"Using cached FFR data (last updated: {last_data_date.strftime('%Y-%m-%d')})")
                        # Auto-backfill to current date if needed
                        return self._backfill_ffr_data(existing_data)
            except Exception as e:
                print(f"Error reading cached FFR data: {e}. Downloading fresh data...")
                needs_update = True
        else:
            print("No cached FFR data found. Downloading...")
            needs_update = True
        
        if needs_update:
            ffr_data = self._download_ffr_data(start_date, existing_data)
            if not ffr_data.empty:
                # Save to CSV
                ffr_data.to_csv(ffr_file)
                print(f"FFR data saved to {ffr_file}")
                return self._process_ffr_data(ffr_data)
            else:
                print("Failed to download FFR data")
                return pd.Series(dtype=float)
        
        return pd.Series(dtype=float)
    
    def _download_ffr_data(self, start_date: str, existing_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Download Federal Funds Rate data from FRED via pandas-datareader.
        Uses FEDFUNDS (monthly effective rate) which has historical data back to 1954.
        Automatically backfills to daily frequency for consistency with stock data.
        
        Args:
            start_date: Start date for download
            existing_data: Existing data to merge with (for incremental updates)
            
        Returns:
            DataFrame with FFR data (daily frequency)
        """
        if not DATAREADER_AVAILABLE:
            print("pandas-datareader not available. Cannot download FFR data.")
            return pd.DataFrame()
        
        print("Downloading Federal Funds Rate data from FRED (FEDFUNDS - monthly effective rate)...")
        
        try:
            # Download FFR from FRED (FEDFUNDS = Monthly Effective Federal Funds Rate)
            # This series goes back to 1954, providing full historical data
            ffr_data = pdr.DataReader('FEDFUNDS', 'fred', start=start_date)
            
            if not ffr_data.empty:
                # Rename column to standard name
                ffr_data.columns = ['FFR']
                
                # Convert percentage to decimal (FRED data is in percentage)
                ffr_data['FFR'] = ffr_data['FFR'] / 100
                
                # Remove timezone if present
                if isinstance(ffr_data.index, pd.DatetimeIndex) and ffr_data.index.tz is not None:
                    ffr_data.index = ffr_data.index.tz_localize(None)
                
                # Backfill from monthly to daily frequency using forward fill
                # This ensures we have daily data consistent with stock prices
                print(f"Backfilling monthly FFR data to daily frequency...")
                daily_ffr = ffr_data.resample('D').ffill()
                
                # Remove any NaN values at the beginning
                daily_ffr = daily_ffr.dropna()
                
                print(f"Successfully processed FFR data:")
                print(f"  - Original monthly records: {len(ffr_data)}")
                print(f"  - Daily records after backfill: {len(daily_ffr)}")
                print(f"  - Date range: {daily_ffr.index.min()} to {daily_ffr.index.max()}")
                
                # Merge with existing data if available
                if existing_data is not None and not existing_data.empty:
                    # Combine datasets, keeping the most recent data for overlapping dates
                    combined_data = pd.concat([existing_data, daily_ffr])
                    combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
                    combined_data = combined_data.sort_index()
                    print(f"Successfully updated FFR data ({len(combined_data)} total daily records)")
                    return combined_data
                else:
                    return daily_ffr
                    
        except Exception as e:
            print(f"FRED FEDFUNDS download failed: {e}")
            # Try the old DFF series as a fallback for recent data
            try:
                print("Trying DFF (daily) series as fallback for recent data...")
                ffr_data = pdr.DataReader('DFF', 'fred', start=start_date)
                if not ffr_data.empty:
                    ffr_data.columns = ['FFR']
                    ffr_data['FFR'] = ffr_data['FFR'] / 100
                    if isinstance(ffr_data.index, pd.DatetimeIndex) and ffr_data.index.tz is not None:
                        ffr_data.index = ffr_data.index.tz_localize(None)
                    print(f"Successfully downloaded DFF data ({len(ffr_data)} records)")
                    return ffr_data
            except Exception as e2:
                print(f"DFF fallback also failed: {e2}")
        
        # Final fallback: create a historical approximation with realistic rates
        print("Creating historical FFR approximation with realistic rates...")
        date_range = pd.date_range(start=start_date, end=datetime.now(), freq='D')
        ffr_approx = pd.DataFrame(index=date_range)
        
        # Use historically realistic rates based on periods
        rates = []
        for date in date_range:
            year = date.year
            if year < 2000:
                rate = 0.055  # ~5.5% average for 1990s
            elif year < 2008:
                rate = 0.035  # ~3.5% average for 2000-2007
            elif year < 2016:
                rate = 0.005  # ~0.5% average for 2008-2015 (crisis + recovery)
            elif year < 2022:
                rate = 0.015  # ~1.5% average for 2016-2021
            else:
                rate = 0.045  # ~4.5% for recent period
            rates.append(rate)
        
        ffr_approx['FFR'] = rates
        print(f"Created approximation with {len(ffr_approx)} daily records")
        
        # Merge with existing if available
        if existing_data is not None and not existing_data.empty:
            combined_data = pd.concat([existing_data, ffr_approx])
            combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
            return combined_data.sort_index()
        
        return ffr_approx
    
    def _backfill_ffr_data(self, data: pd.DataFrame) -> pd.Series:
        """
        Auto-backfill FFR data to current date using the latest available rate.
        
        Args:
            data: Existing FFR DataFrame
            
        Returns:
            FFR Series extended to current date
        """
        current_date = datetime.now().date()
        last_data_date = data.index.max().date()
        
        if last_data_date < current_date:
            # Get the latest rate for backfilling
            latest_rate = data.iloc[-1]['FFR'] if 'FFR' in data.columns else data.iloc[-1, 0]
            
            # Create date range from last data date to current date
            backfill_dates = pd.date_range(start=last_data_date + timedelta(days=1), 
                                         end=current_date, freq='D')
            
            if len(backfill_dates) > 0:
                # Create backfill data
                backfill_data = pd.DataFrame(index=backfill_dates)
                backfill_data['FFR'] = latest_rate
                
                # Combine with existing data
                extended_data = pd.concat([data, backfill_data])
                print(f"Auto-backfilled FFR ({latest_rate*100:.2f}%) for {len(backfill_dates)} days")
                
                return self._process_ffr_data(extended_data)
        
        return self._process_ffr_data(data)
    
    def _process_ffr_data(self, ffr_data: pd.DataFrame) -> pd.Series:
        """
        Process and clean FFR data for use in backtesting.
        
        Args:
            ffr_data: Raw FFR DataFrame
            
        Returns:
            Cleaned FFR Series with daily frequency
        """
        # Ensure we have the FFR column
        if 'FFR' not in ffr_data.columns:
            if len(ffr_data.columns) == 1:
                ffr_data.columns = ['FFR']
            else:
                raise ValueError("Unable to identify FFR column in data")
        
        # Remove timezone information if present
        if isinstance(ffr_data.index, pd.DatetimeIndex) and ffr_data.index.tz is not None:
            ffr_data.index = ffr_data.index.tz_localize(None)
        
        # Convert to daily frequency and forward fill missing values
        ffr_series = ffr_data['FFR'].copy()
        
        # Forward fill NaN values (FFR doesn't change daily)
        ffr_series = ffr_series.fillna(method='ffill')
        
        # Ensure values are reasonable (between 0% and 25%)
        ffr_series = ffr_series.clip(0, 0.25)
        
        return ffr_series
    
    def _load_from_yfinance(self, ticker: str, period: str, 
                          start_date: Optional[str], 
                          end_date: Optional[str]) -> pd.DataFrame:
        """
        Load data from Yahoo Finance.
        
        Args:
            ticker: Ticker symbol
            period: Time period to load
            start_date: Specific start date
            end_date: Specific end date
            
        Returns:
            DataFrame with OHLCV data
        """
        stock = yf.Ticker(ticker)
        
        try:
            if start_date:
                # If end_date not provided, use current date
                if not end_date:
                    end_date = datetime.now().strftime('%Y-%m-%d')
                df = stock.history(start=start_date, end=end_date, auto_adjust=True, actions=True)
            else:
                df = stock.history(period=period, auto_adjust=True, actions=True)
                
            if df.empty:
                # If no data for specified period, try using 'max'
                print(f"No data available for {ticker} with period '{period}'. Trying 'max' period.")
                df = stock.history(period='max', auto_adjust=True, actions=True)
                if df.empty:
                    print(f"No data available for {ticker} even with 'max' period.")
                    return pd.DataFrame()  # Return empty DataFrame
        except Exception as e:
            print(f"Error fetching data for {ticker}: {str(e)}")
            # Try with 'max' period as fallback
            try:
                df = stock.history(period='max', auto_adjust=True, actions=True)
            except Exception as e2:
                print(f"Failed again with 'max' period: {str(e2)}")
                return pd.DataFrame()  # Return empty DataFrame
        
        # Remove timezone information to avoid comparison issues
        if not df.empty and isinstance(df.index, pd.DatetimeIndex):
            df.index = df.index.tz_localize(None).normalize()
        
        return df
    
    def _filter_by_date_range(self, df: pd.DataFrame, period: str, 
                           start_date: Optional[str], 
                           end_date: Optional[str]) -> pd.DataFrame:
        """
        Filter DataFrame by date range.
        
        Args:
            df: DataFrame to filter
            period: Time period string
            start_date: Specific start date
            end_date: Specific end date
            
        Returns:
            Filtered DataFrame
        """
        if df.empty:
            return df
        
        # Remove timezone information if it exists
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
            df.index = df.index.tz_localize(None).normalize()
        
        # Filter by specific start/end dates if provided
        if start_date:
            start_dt = pd.to_datetime(start_date).normalize()
            df = df[df.index >= start_dt]
        elif period and period != 'max':
            # Calculate start date based on period
            end_dt = df.index.max()
            
            if period.endswith('y') or period.endswith('Y'):
                years = int(period[:-1])
                start_dt = end_dt - pd.DateOffset(years=years)
            elif period.endswith('mo') or period.endswith('MO'):
                months = int(period[:-2])
                start_dt = end_dt - pd.DateOffset(months=months)
            elif period.endswith('d') or period.endswith('D'):
                days = int(period[:-1])
                start_dt = end_dt - pd.DateOffset(days=days)
            else:
                # Handle other period formats (like '1wk', '1m', etc.)
                # For now, default to 1 year
                start_dt = end_dt - pd.DateOffset(years=1)
                
            df = df[df.index >= start_dt]
        
        if end_date:
            end_dt = pd.to_datetime(end_date).normalize()
            df = df[df.index <= end_dt]
        
        return df
    
    def _save_to_csv(self, ticker: str, df: pd.DataFrame) -> None:
        """
        Save DataFrame to CSV file.
        
        Args:
            ticker: Ticker symbol
            df: DataFrame to save
        """
        # Remove timezone information if it exists
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
            
        filename = os.path.join(self.data_dir, f"{ticker}_daily.csv")
        df.to_csv(filename)
        
    def _load_from_csv(self, ticker: str) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Args:
            ticker: Ticker symbol
            
        Returns:
            DataFrame with OHLCV data
        """
        filename = os.path.join(self.data_dir, f"{ticker}_daily.csv")
        
        try:
            # Try reading with automatic date parsing first
            df = pd.read_csv(filename, index_col='Date', parse_dates=True)
            
        except ValueError as e:
            if "Tz-aware" in str(e):
                # Handle timezone-aware datetime issue gracefully
                print(f"Timezone issue detected in {ticker} CSV file. Converting timezone data...")
                # Read without parsing dates first
                df = pd.read_csv(filename, index_col='Date')
                # Convert to datetime with UTC handling, then remove timezone
                df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
            else:
                # For other parsing errors, try reading without date parsing
                print(f"Date parsing issue for {ticker}. Attempting manual conversion...")
                df = pd.read_csv(filename, index_col='Date')
                df.index = pd.to_datetime(df.index, errors='coerce')
                # Remove any rows with invalid dates
                df = df[df.index.notna()]
        
        # Ensure index is DatetimeIndex and handle any remaining timezone issues
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
            except ValueError as e:
                if "Tz-aware" in str(e):
                    # Handle timezone conversion issue
                    print(f"Converting timezone-aware index for {ticker}...")
                    df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
                else:
                    # Try with error handling
                    df.index = pd.to_datetime(df.index, errors='coerce')
                    df = df[df.index.notna()]  # Remove invalid dates
        
        # Remove timezone information if it exists
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
            
        df.index = df.index.normalize()
        
        return df
    
    def _resample_data(self, df: pd.DataFrame, resample_period: str) -> pd.DataFrame:
        """
        Resample OHLCV data to a different frequency.
        
        Args:
            df: DataFrame with OHLCV data
            resample_period: Target frequency ('D', 'W', 'M', 'Q', etc.)
            
        Returns:
            Resampled DataFrame
        """
        # Handle special bi-weekly case
        if resample_period == 'BW':
            rule = '2W-MON'
        elif resample_period == 'W':
            rule = 'W-MON'
        else:
            rule = resample_period
            
        # Resample with appropriate aggregation functions
        agg_dict = {
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }
        
        # Add dividends aggregation if column exists
        if 'Dividends' in df.columns:
            agg_dict['Dividends'] = 'sum'  # Sum dividends over the resampling period
        
        # Add stock splits aggregation if column exists
        if 'Stock Splits' in df.columns:
            agg_dict['Stock Splits'] = 'prod'  # Product of stock splits (multiply splits)
            
        resampled = df.resample(rule).agg(agg_dict)
        
        return resampled
    
    def update_data(self, tickers: Union[str, List[str]]) -> Dict[str, pd.DataFrame]:
        """
        Update existing data by fetching only the latest points (equivalent to incremental mode).
        
        Args:
            tickers: Single ticker or list of tickers to update
            
        Returns:
            Dictionary mapping tickers to updated DataFrames
        """
        return self.load(tickers, mode='incremental')
