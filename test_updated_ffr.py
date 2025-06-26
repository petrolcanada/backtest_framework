"""
Test script to download fresh FFR data using the updated loader
Run this to get historical data from 1990
"""

import sys
import os

# Add the backtest framework to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backtest_framework'))

from core.data.loader import DataLoader

def test_ffr_update():
    """Test the updated FFR data loading"""
    
    print("=== Testing Updated FFR Data Loading ===")
    print()
    
    # Initialize the data loader
    loader = DataLoader()
    
    # Force download fresh FFR data from 1990
    print("Downloading fresh FFR data from 1990...")
    ffr_data = loader.load_fed_funds_rate(start_date='1990-01-01', force_update=True)
    
    if not ffr_data.empty:
        print("SUCCESS! FFR data loaded successfully.")
        print()
        print(f"Data summary:")
        print(f"  - Total records: {len(ffr_data):,}")
        print(f"  - Date range: {ffr_data.index.min()} to {ffr_data.index.max()}")
        print(f"  - Data frequency: Daily")
        print()
        print("Sample FFR values:")
        
        # Show some key historical points
        sample_dates = ['1990-01-01', '1995-01-01', '2000-01-01', '2005-01-01', 
                       '2010-01-01', '2015-01-01', '2020-01-01', '2023-01-01']
        
        for date_str in sample_dates:
            try:
                if date_str in ffr_data.index:
                    rate = ffr_data.loc[date_str]
                    print(f"  {date_str}: {rate:.4f} ({rate*100:.2f}%)")
                elif pd.to_datetime(date_str) <= ffr_data.index.max():
                    # Find the nearest date
                    nearest_date = ffr_data.index[ffr_data.index >= pd.to_datetime(date_str)][0]
                    rate = ffr_data.loc[nearest_date]
                    print(f"  {nearest_date.strftime('%Y-%m-%d')}: {rate:.4f} ({rate*100:.2f}%)")
            except:
                pass
        
        print()
        print("FFR data is now ready for your backtest framework!")
        print("You should see data going back to 1990 instead of just 2020.")
        
        return True
    else:
        print("FAILED: No FFR data was loaded.")
        print("Check your internet connection and pandas-datareader installation.")
        return False

if __name__ == "__main__":
    import pandas as pd
    
    success = test_ffr_update()
    
    if success:
        print()
        print("Next steps:")
        print("1. Your FFR data file has been updated with historical data")
        print("2. You can now run your backtests with data from 1990 onwards")
        print("3. The FFR data will be consistent daily frequency like your stock data")
    else:
        print()
        print("Troubleshooting:")
        print("1. Make sure you have pandas-datareader installed: pip install pandas-datareader")
        print("2. Check your internet connection")
        print("3. Try running the script again")
