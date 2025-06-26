import os
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

import datetime
import os
import pandas_ta as ta


import concurrent.futures
from functools import partial
from concurrent.futures import ProcessPoolExecutor



import warnings



# Suppress all warnings
warnings.filterwarnings('ignore')

# Change the working directory
new_directory = r"C:\Users\peli\local_script\Local Technical Indicator Data"
# os.chdir(new_directory)


# Create the subfolder if it doesn't exist
subfolder = os.path.join(new_directory, 'security_data')
if not os.path.exists(subfolder):
    os.makedirs(subfolder)

print("Local Data Directory:", subfolder)



# Function to read index constituents
# Function to read index constituents
def read_index_constituents(filepath='./index_constituent.xlsx', sheet_name='sp500'):
    return pd.read_excel(filepath, sheet_name=sheet_name)

# Step 1: Initial Data Pull from Vendor, pointing to subfolder (already includes new_directory)
def pull_initial_df(ticker, period='1y'):
    stock = yf.Ticker(ticker)
    try:
        df = stock.history(period=period)
    except Exception as e:
        print(f"An error occurred with period '{period}' for {ticker}: {str(e)}")
        print("Attempting to use 'max' period instead.")
        try:
            df = stock.history(period='max')
        except Exception as e:
            print(f"Error occurred with 'max' period as well: {str(e)}")
            return None
    
    if df.empty:
        print(f"No data available for {ticker} with period '{period}'. Trying 'max' period.")
        df = stock.history(period='max')
        if df.empty:
            print(f"No data available for {ticker} even with 'max' period.")
            return None

    filename = os.path.join(subfolder, f'{ticker}_daily.csv')  # Use subfolder
    df.to_csv(filename)
    return df


# Step 2: Marginal Pull and Update from Vendor, pointing to subfolder (already includes new_directory)
def pull_update_df(ticker):
    stock = yf.Ticker(ticker)
    filename = os.path.join(subfolder, f'{ticker}_daily.csv')  # Use subfolder
    existing_df = pd.read_csv(filename, index_col='Date', parse_dates=True)
    last_10th_date = existing_df.index[-10]
    new_df = stock.history(start=last_10th_date)
    updated_df = pd.concat([existing_df.iloc[:-10], new_df])
    updated_df.to_csv(filename)
    return updated_df

# Step 3: Pull Data from Local CSV and Resample if Necessary, pointing to subfolder (already includes new_directory)
def read_local_df(ticker, resample_period='D', period='1y'):
    filename = os.path.join(subfolder, f'{ticker}_daily.csv')  # Use subfolder
    df = pd.read_csv(filename, index_col='Date', parse_dates=True)
    
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)

    # Truncate the time information while keeping the index as a DatetimeIndex
    df.index = df.index.normalize()
    
    # Filter data based on period
    if period:
        end_date = df.index.max()
        if period.endswith('y') or period.endswith('Y'):
            years = int(period[:-1])
            start_date = end_date - pd.DateOffset(years=years)
        elif period.endswith('m') or period.endswith('M'):
            months = int(period[:-1])
            start_date = end_date - pd.DateOffset(months=months)
        elif period.endswith('d') or period.endswith('D'):
            days = int(period[:-1])
            start_date = end_date - pd.DateOffset(days=days)
        df = df[df.index >= start_date]
    
    if resample_period == 'W':
        df = df.resample('W-MON', label='left').agg({'Open': 'first', 
                                                     'High': 'max', 
                                                     'Low': 'min', 
                                                     'Close': 'last', 
                                                     'Volume': 'sum'})
    elif resample_period == 'BW':  # Bi-weekly frequency
        df = df.resample('2W-MON', label='left').agg({'Open': 'first', 
                                                      'High': 'max', 
                                                      'Low': 'min', 
                                                      'Close': 'last', 
                                                      'Volume': 'sum'})
    elif resample_period == 'M':
        df = df.resample('M', label='left').agg({'Open': 'first', 
                                                 'High': 'max', 
                                                 'Low': 'min', 
                                                 'Close': 'last', 
                                                 'Volume': 'sum'})
    elif resample_period == 'Q':
        df = df.resample('Q', label='left').agg({'Open': 'first', 
                                                 'High': 'max', 
                                                 'Low': 'min', 
                                                 'Close': 'last', 
                                                 'Volume': 'sum'})
    return df



def calculate_sma(df, window=30):
    df['SMA_30'] = ta.ema(df['Close'], length=window)
    df['Close_minus_SMA_30'] = df['Close'] - df['SMA_30']
    return df

def calculate_sma_slope(df):
    df['SMA_30_slope'] = ta.percent_return(df['SMA_30'])
    return df

def calculate_donchian_channel(df, window=20):
    donchian = ta.donchian(df['High'], df['Low'], lower_length=window, upper_length=window)
    df['Donchian_High'] = donchian[f'DCU_{window}_{window}'].shift(1)
    df['Donchian_Low'] = donchian[f'DCL_{window}_{window}'].shift(1)
    return df

def calculate_daily_returns(df):
    df['Daily_Returns'] = ta.percent_return(df['Close'])
    # df['Daily_Returns'] = (df['Close'] / df['Close'].shift(1) - 1) * 100

    return df

def calculate_volume_signal(df, days_short=2, days_long=5, multiplier=2):
    df['Avg_Volume_Short'] = ta.sma(df['Volume'], length=days_short)
    df['Avg_Volume_Long'] = ta.sma(df['Volume'], length=days_long)
    df['Volume_Signal'] = df['Avg_Volume_Short'] / df['Avg_Volume_Long'] > multiplier
    return df

# def calculate_adx(df, window=14):
def calculate_adx_orig(df, window=100):
    adx_column_name = f'ADX_{window}'
    df[adx_column_name] = ta.adx(df['High'], df['Low'], df['Close'], length=window)[adx_column_name]
    
    # Rename the column to 'ADX'
    df['ADX'] = df[adx_column_name]
    
    # Drop the original ADX column with the specific window name (optional)
    df.drop(columns=[adx_column_name], inplace=True)
    
    df['ADX_Slope'] = ta.percent_return(df['ADX'])

    # Calculate SMA_14 of the ADX
    df['ADX_SMA_14'] = ta.sma(df['ADX'], length=1)

    # Initialize ADX_FLIP with zeros
    df['ADX_FLIP'] = 0

    # Calculate ADX_FLIP
    for i in range(1, len(df)):
        if df['ADX_Slope'].iloc[i-1] < 0 and df['ADX_Slope'].iloc[i] > 0:
            df.at[df.index[i], 'ADX_FLIP'] = 1
        elif df['ADX_Slope'].iloc[i-1] > 0 and df['ADX_Slope'].iloc[i] < 0:
            df.at[df.index[i], 'ADX_FLIP'] = -1

    return df


    
# def calculate_adx(df, window=14):
def calculate_adx(df, window=14 * 22, sma_length=5):
    adx_column_name = f'ADX_{window}'
    df[adx_column_name] = ta.adx(df['High'], df['Low'], df['Close'], length=window)[adx_column_name]
    
    # Rename the column to 'ADX'
    df['ADX'] = df[adx_column_name]
    
    # Drop the original ADX column with the specific window name (optional)
    df.drop(columns=[adx_column_name], inplace=True)

    # Calculate SMA of the ADX
    df['ADX_SMA'] = ta.sma(df['ADX'], length=sma_length)
    
    # Calculate ADX slope
    # df['ADX_Slope'] = ta.percent_return(df['ADX_SMA'])
    df['ADX_Slope'] = ta.percent_return(df['ADX'])

    # Initialize ADX_FLIP with zeros
    df['ADX_Flip'] = 0

    # Calculate ADX_FLIP
    for i in range(1, len(df)):
        if df['ADX_Slope'].iloc[i-1] < 0 and df['ADX_Slope'].iloc[i] > 0:
            df.at[df.index[i], 'ADX_Flip'] = 1
        elif df['ADX_Slope'].iloc[i-1] > 0 and df['ADX_Slope'].iloc[i] < 0:
            df.at[df.index[i], 'ADX_Flip'] = -1

    return df

def calculate_awesome_oscillator(df):
    df['AO'] = ta.ao(df['High'], df['Low'])
    df['AO_Slope'] = df['AO'].diff()  # Calculate the difference between the current and previous AO value
    return df


# Function to calculate MACD using pandas_ta built-in function
def calculate_macd(df, fast_period=12, slow_period=26, signal_period=9):
    macd = ta.macd(df['Close'], fast=fast_period, slow=slow_period, signal=signal_period)
    df['MACD'] = macd['MACD_12_26_9']
    df['MACD_Signal'] = macd['MACDs_12_26_9']
    df['MACD_Hist'] = macd['MACDh_12_26_9']
    
    # Calculate the slope (difference) of MACD
    df['MACD_Slope'] = df['MACD_Hist'].diff()
    
    return df


# Function to calculate KDJ using pandas_ta built-in function
def calculate_kdj(df, period=9, signal=3):
    # Original KDJ calculation
    kdj = ta.kdj(df['High'], df['Low'], df['Close'], length=period, signal=signal)
    df['K'] = kdj[f'K_{period}_{signal}']
    df['D'] = kdj[f'D_{period}_{signal}']
    df['J'] = kdj[f'J_{period}_{signal}']
    df['J_Slope'] = df['J'].diff()
    df['K_Slope'] = df['K'].diff()
    df['D_Slope'] = df['D'].diff()

    # Monthly KDJ calculation
    monthly_period = period * 22
    monthly_signal = signal * 22
    # monthly_signal = signal * 11
    monthly_kdj = ta.kdj(df['High'], df['Low'], df['Close'], length=monthly_period, signal=monthly_signal)
    
    if monthly_kdj is not None:
        # Use snake_case for monthly KDJ variables
        df['monthly_k'] = monthly_kdj[f'K_{monthly_period}_{monthly_signal}']
        df['monthly_d'] = monthly_kdj[f'D_{monthly_period}_{monthly_signal}']
        df['monthly_j'] = monthly_kdj[f'J_{monthly_period}_{monthly_signal}']
        df['monthly_j_slope'] = df['monthly_j'].diff()
        df['monthly_k_slope'] = df['monthly_k'].diff()
        df['monthly_d_slope'] = df['monthly_d'].diff()
        
        # For backward compatibility, keep the old names too
        df['Monthly_K'] = df['monthly_k']
        df['Monthly_D'] = df['monthly_d']
        df['Monthly_J'] = df['monthly_j']
        df['Monthly_J_Slope'] = df['monthly_j_slope']
        df['Monthly_K_Slope'] = df['monthly_k_slope']
        df['Monthly_D_Slope'] = df['monthly_d_slope']
    else:
        # Initialize all variables as NaN
        df['monthly_k'] = float('nan')
        df['monthly_d'] = float('nan')
        df['monthly_j'] = float('nan')
        df['monthly_j_slope'] = float('nan')
        df['monthly_k_slope'] = float('nan')
        df['monthly_d_slope'] = float('nan')
        
        # For backward compatibility, keep the old names too
        df['Monthly_K'] = float('nan')
        df['Monthly_D'] = float('nan')
        df['Monthly_J'] = float('nan')
        df['Monthly_J_Slope'] = float('nan')
        df['Monthly_K_Slope'] = float('nan')
        df['Monthly_D_Slope'] = float('nan')

    return df

def calculate_rsi(df, period=14 * 1):
    df['RSI'] = ta.rsi(df['Close'], length=period)
    return df


def calculate_mfi(df, length=48):
    """
    Calculate Money Flow Index using pandas_ta
    
    Args:
        df (pd.DataFrame): DataFrame with High, Low, Close, Volume columns
        length (int): lookback period, default 48
        
    Returns:
        pd.DataFrame: Original dataframe with MFI column added
    """
    df['MFI'] = ta.mfi(df['High'], df['Low'], df['Close'], df['Volume'], length=length)
    return df




# Step 7: Generate Flipping Points (Both Buy and Sell)
def generate_flipping_points(df):
    df['SMA_30_slope_flipping_buy'] = ((df['SMA_30_slope'].shift(1) <= 0) & 
                                       (df['SMA_30_slope'].shift(2) <= 0) & 
                                       (df['SMA_30_slope'].shift(3) <= 0) & 
                                       (df['SMA_30_slope'] > 0))
    df['SMA_30_slope_flipping_sell'] = ((df['SMA_30_slope'].shift(1) >= 0) & 
                                        (df['SMA_30_slope'].shift(2) >= 0) & 
                                        (df['SMA_30_slope'].shift(3) >= 0) & 
                                        (df['SMA_30_slope'] < 0))
    return df



# Step 8: Check within Lookback Window
def within_lookback_window(df, lookback_window=10):
    last_flipping_point_buy = None
    last_flipping_point_sell = None
    within_window_buy = []
    within_window_sell = []

    for i in range(len(df)):
        if df['SMA_30_slope_flipping_buy'][i]:
            last_flipping_point_buy = i
        if df['SMA_30_slope_flipping_sell'][i]:
            last_flipping_point_sell = i
        
        if last_flipping_point_buy is not None and (i - last_flipping_point_buy) <= lookback_window:
            within_window_buy.append(True)
        else:
            within_window_buy.append(False)
        
        if last_flipping_point_sell is not None and (i - last_flipping_point_sell) <= lookback_window:
            within_window_sell.append(True)
        else:
            within_window_sell.append(False)
    
    df['Within_Lookback_Window_Buy'] = within_window_buy
    df['Within_Lookback_Window_Sell'] = within_window_sell
    return df





# Step 9: Generate Individual Condition Columns (Both Buy and Sell)
def generate_individual_conditions(df):
    # Buy conditions
    df['Close_Greater_Than_Open'] = df['Close'] > df['Open']
    df['SMA_30_Positive'] = df['SMA_30_slope'] > 0
    df['Close_Greater_Than_Donchian_High'] = df['Close'] > df['Donchian_High']
    
    # Sell conditions
    df['Close_Less_Than_Open'] = df['Close'] < df['Open']
    df['SMA_30_Negative'] = df['SMA_30_slope'] < 0
    df['Close_Less_Than_Donchian_Low'] = df['Close'] < df['Donchian_Low']
    return df



# Function to generate stop loss price
def generate_stop_loss_price(df):
    df['Buy_Price'] = np.nan
    df['Stop_Loss_Price'] = np.nan
    df['Stop_Loss_Executed'] = False
    df['active_position']=False
    active_position = False

    for i in range(1, len(df)-1):  # Adjust the range to avoid index out of range error
        df['active_position'].iloc[i] = active_position
        if df['BUY_SIGNAL'].iloc[i] == 1:
            df['Buy_Price'].iloc[i+1] = df['Open'].iloc[i+1]  # Buy at the open price of the next day
            df['Stop_Loss_Price'].iloc[i+1] = df['SMA_30'].iloc[i]  # Set the stop loss price for the next day
            active_position = True
        elif active_position:
            df['Buy_Price'].iloc[i+1] = df['Buy_Price'].iloc[i]
            df['Stop_Loss_Price'].iloc[i+1] = df['SMA_30'].iloc[i]

        if active_position and pd.notna(df['Stop_Loss_Price'].iloc[i]):
            if df['Stop_Loss_Price'].iloc[i] > df['Low'].iloc[i]:
                df['Stop_Loss_Executed'].iloc[i] = True
                active_position = False
            # else:
            #     df['Stop_Loss_Price'].iloc[i] = df['SMA_30'].iloc[i]

    return df



# Step 10: Generate Buy and Sell Signals
def generate_buy_sell_signals(df):
    # Buy signals
    df['BUY_SIGNAL'] = (df['Close_Greater_Than_Open'] & 
                        df['SMA_30_Positive'] & 
                        df['Close_Greater_Than_Donchian_High'] & 
                        df['Volume_Signal'] &
                        df['Within_Lookback_Window_Buy']).astype(int)
    
    # Sell signals
    df['SELL_SIGNAL'] = (df['Close_Less_Than_Open'] & 
                         df['SMA_30_Negative'] & 
                         df['Close_Less_Than_Donchian_Low'] & 
                         df['Volume_Signal'] &
                         df['Within_Lookback_Window_Sell']).astype(int)
    
    # Generate session buy and sell signals
    buy_sessions = []
    sell_sessions = []
    session_start_buy = None
    session_start_sell = None
    
    for i in range(len(df)):
        if df['SMA_30_slope_flipping_buy'].iloc[i]:
            session_start_buy = i
        if df['SMA_30_slope_flipping_sell'].iloc[i]:
            session_start_sell = i
        
        if session_start_buy is not None and (i == len(df) - 1 or not df['Within_Lookback_Window_Buy'].iloc[i] or (i < len(df) - 1 and df['SMA_30_slope_flipping_buy'].iloc[i + 1])):
            session_end_buy = i
            buy_sessions.append((session_start_buy, session_end_buy))
            session_start_buy = None
        
        if session_start_sell is not None and (i == len(df) - 1 or not df['Within_Lookback_Window_Sell'].iloc[i] or (i < len(df) - 1 and df['SMA_30_slope_flipping_sell'].iloc[i + 1])):
            session_end_sell = i
            sell_sessions.append((session_start_sell, session_end_sell))
            session_start_sell = None
    
    buy_signals = []
    sell_signals = []
    
    for start, end in buy_sessions:
        session_df = df.iloc[start:end + 1]
        buy_signal = session_df[session_df['BUY_SIGNAL'] == 1].index
        if not buy_signal.empty:
            buy_signals.append(buy_signal[0])
    
    for start, end in sell_sessions:
        session_df = df.iloc[start:end + 1]
        sell_signal = session_df[session_df['SELL_SIGNAL'] == 1].index
        if not sell_signal.empty:
            sell_signals.append(sell_signal[0])
    
    df['BUY_SIGNAL'] = df.index.isin(buy_signals).astype(int)
    df['SELL_SIGNAL'] = df.index.isin(sell_signals).astype(int)

    df = generate_stop_loss_price(df)
    
    return df


def generate_buy_sell_signals_by_ADX(df, adx_percent_threshold=0.3):
    df['BUY_SIGNAL'] = 0
    df['SELL_SIGNAL'] = 0
    df['BUY_EXIT_SIGNAL'] = 0
    df['SELL_EXIT_SIGNAL'] = 0
    df['ADX_Slope_SUM'] = 0.0  # Initialize the column for storing ADX slope sums
    active_trade = False
    buy_index = None
    sell_index = None

    # Calculate ADX range and dynamic threshold
    adx_min = df['ADX'].min()
    adx_max = df['ADX'].max()
    adx_range = adx_max - adx_min
    adx_threshold = adx_min + adx_range * adx_percent_threshold

    window_size = 10

    for i in range(window_size, len(df)):  # Start from the 5th index to allow lookback
        # Extract ADX, ADX_FLIP, and ADX_Slope for the window i-5 to i
        adx_window = df['ADX'].iloc[i-window_size:i+1].tolist()
        adx_flip_window = df['ADX_FLIP'].iloc[i-window_size:i+1].tolist()
        adx_slope_window = df['ADX_Slope'].iloc[i-window_size:i+1].tolist()

        adx_slope_sum = 0.0  # Default value in case no flip is detected

        # Check if there is an ADX_FLIP of 1 in the window
        if 1 in adx_flip_window:
            flip_index = adx_flip_window.index(1)  # Position in the window
            adx_slope_sum = sum(adx_slope_window[flip_index:])  # Sum of ADX_Slope from flip position to the last value

        # Store the calculated ADX slope sum in the DataFrame
        df.at[df.index[i], 'ADX_Slope_SUM'] = adx_slope_sum

        # # Generate Buy Signal
        # if not active_trade and df['ADX'].iloc[i] < adx_threshold and adx_slope_sum > 0.0001 and df['SMA_30_slope'].iloc[i] > 0:
        #     df.at[df.index[i], 'BUY_SIGNAL'] = 1
        #     active_trade = True
        #     buy_index = i

        # # Generate Sell Signal
        # elif not active_trade and df['ADX'].iloc[i] < adx_threshold and adx_slope_sum > 0.0001 and df['SMA_30_slope'].iloc[i] < 0:
        #     df.at[df.index[i], 'SELL_SIGNAL'] = 1
        #     active_trade = True
        #     sell_index = i


        # Generate Buy Signal with the updated logic
        if buy_index is None and df['SMA_30_slope'].iloc[i] > 0 and df['Close_Greater_Than_Donchian_High'].iloc[i] and 1 in adx_flip_window and adx_slope_sum > 0.0001 and df['ADX'].iloc[i] < adx_threshold :
            df.at[df.index[i], 'BUY_SIGNAL'] = 1
            active_trade = True
            buy_index = i

        # Generate Sell Signal with the updated logic
        elif sell_index is None and df['SMA_30_slope'].iloc[i] < 0 and df['Close_Less_Than_Donchian_Low'].iloc[i] and 1 in adx_flip_window and adx_slope_sum > 0.0001 and df['ADX'].iloc[i] < adx_threshold :
            df.at[df.index[i], 'SELL_SIGNAL'] = 1
            active_trade = True
            sell_index = i

        
        # Generate Buy Exit Signal
        if active_trade and buy_index is not None and df['ADX_FLIP'].iloc[i] < 0:
            df.at[df.index[i], 'BUY_EXIT_SIGNAL'] = 1
            active_trade = False
            buy_index = None

        # Generate Sell Exit Signal
        elif active_trade and sell_index is not None and df['ADX_FLIP'].iloc[i] < 0:
            df.at[df.index[i], 'SELL_EXIT_SIGNAL'] = 1
            active_trade = False
            sell_index = None

    return df




def generate_buy_sell_signals_by_ADX_v2(df, adx_percent_threshold=0.3, stop_loss_threshold_principal=0.1, stop_loss_threshold_profit=0.3):
    df['BUY_SIGNAL'] = 0
    df['BUY_EXIT_SIGNAL'] = 0
    df['SELL_SIGNAL'] = 0
    df['SELL_EXIT_SIGNAL'] = 0
    df['Buy_Stop_Loss_Price'] = np.nan
    df['Cum_Profit_Since_Buy'] = np.nan
    df['Cum_Max_Profit_Since_Buy'] = np.nan
    df['Buy_Stop_Loss_Signal'] = 0
    buy_index = None  # Track the index of the last buy signal

    # Calculate ADX range and dynamic threshold
    adx_min = df['ADX'].min()
    adx_max = df['ADX'].max()
    adx_range = adx_max - adx_min
    adx_threshold = adx_min + adx_range * adx_percent_threshold

    for i in range(1, len(df)):
        # Generate Buy Signal
        if buy_index is None and df['AO_Slope'].iloc[i] > 0 and df['ADX_Slope'].iloc[i] > 0 and df['ADX'].iloc[i] < adx_threshold:
            df.at[df.index[i], 'BUY_SIGNAL'] = 1
            buy_index = i  # Update buy_index to the current index
            df.at[df.index[i], 'Buy_Stop_Loss_Price'] = -df['Close'].iloc[i]
            df.at[df.index[i], 'Cum_Profit_Since_Buy'] = 0
            df.at[df.index[i], 'Cum_Max_Profit_Since_Buy'] = 0  # Initialize the max profit at 0

            continue

        # Calculate Cumulative Profit Since Buy
        if buy_index is not None and i > buy_index:
            cum_profit = df['Close'].iloc[i] - df['Close'].iloc[buy_index]
            df.at[df.index[i], 'Cum_Profit_Since_Buy'] = cum_profit

            # Calculate Cumulative Max Profit Since Buy
            cum_max_profit = max(cum_profit, df['Cum_Max_Profit_Since_Buy'].iloc[i-1])
            df.at[df.index[i], 'Cum_Max_Profit_Since_Buy'] = cum_max_profit

            # Update the stop loss price based on the maximum cumulative profit
            stop_loss_price = df['Close'].iloc[i] - cum_max_profit * stop_loss_threshold_profit
            
            if stop_loss_price >= df['Buy_Stop_Loss_Price'].iloc[i-1]:
                df.at[df.index[i], 'Buy_Stop_Loss_Price'] = df['Buy_Stop_Loss_Price'].iloc[i-1]
            else:
                df.at[df.index[i], 'Buy_Stop_Loss_Price'] = df['Buy_Stop_Loss_Price'].iloc[i-1]

            # Check if the current low is less than the stop loss price from the last row
            if df['Low'].iloc[i] < df['Buy_Stop_Loss_Price'].iloc[i - 1]:
                df.at[df.index[i], 'Buy_Stop_Loss_Signal'] = 1
                buy_index = None  # Reset buy_index after stop loss is triggered
                continue

            # Generate Buy Exit Signal
            if df['AO_Slope'].iloc[i] < 0 or df['ADX_Slope'].iloc[i] < 0:
                df.at[df.index[i], 'BUY_EXIT_SIGNAL'] = 1
                buy_index = None  # Reset buy_index after exiting the trade
                continue

    return df


def generate_buy_sell_signals_by_ADX_v3(df, adx_percent_threshold=0.3, stop_loss_threshold_principal=0.1, stop_loss_threshold_profit=0.3):
    df['BUY_SIGNAL'] = 0
    df['BUY_EXIT_SIGNAL'] = 0
    df['SELL_SIGNAL'] = 0
    df['SELL_EXIT_SIGNAL'] = 0
    df['Buy_Stop_Loss_Price'] = np.nan
    df['Cum_Profit_Since_Buy'] = np.nan
    df['Cum_Max_Profit_Since_Buy'] = np.nan
    df['Buy_Stop_Loss_Signal'] = 0
    buy_index = None  # Track the index of the last buy signal

    # Calculate ADX range and dynamic threshold
    adx_min = df['ADX'].min()
    adx_max = df['ADX'].max()
    adx_range = adx_max - adx_min
    adx_threshold = adx_min + adx_range * adx_percent_threshold

    for i in range(1, len(df)):
        # Generate Buy Signal
        if buy_index is None and df['AO'].iloc[i] > 0 and df['ADX_Slope'].iloc[i] > 0 and df['ADX'].iloc[i] < adx_threshold:
            df.at[df.index[i], 'BUY_SIGNAL'] = 1
            buy_index = i  # Update buy_index to the current index
            df.at[df.index[i], 'Buy_Stop_Loss_Price'] = df['Close'].iloc[i] * (1 - stop_loss_threshold_principal)
            df.at[df.index[i], 'Cum_Profit_Since_Buy'] = 0
            df.at[df.index[i], 'Cum_Max_Profit_Since_Buy'] = 0  # Initialize the max profit at 0

            continue

        # Calculate Cumulative Profit Since Buy
        if buy_index is not None and i > buy_index:
            cum_profit = df['Close'].iloc[i] - df['Close'].iloc[buy_index]
            df.at[df.index[i], 'Cum_Profit_Since_Buy'] = cum_profit

            # Calculate Cumulative Max Profit Since Buy
            cum_max_profit = max(cum_profit, df['Cum_Max_Profit_Since_Buy'].iloc[i-1])
            df.at[df.index[i], 'Cum_Max_Profit_Since_Buy'] = cum_max_profit

            # Update the stop loss price based on the maximum cumulative profit
            stop_loss_price = df['Close'].iloc[i] - cum_max_profit * stop_loss_threshold_profit
            
            if stop_loss_price >= df['Buy_Stop_Loss_Price'].iloc[i-1]:
                df.at[df.index[i], 'Buy_Stop_Loss_Price'] = stop_loss_price
            else:
                df.at[df.index[i], 'Buy_Stop_Loss_Price'] = df['Buy_Stop_Loss_Price'].iloc[i-1]

            # Check if the current low is less than the stop loss price from the last row
            if df['Low'].iloc[i] < df['Buy_Stop_Loss_Price'].iloc[i - 1]:
                df.at[df.index[i], 'Buy_Stop_Loss_Signal'] = 1
                buy_index = None  # Reset buy_index after stop loss is triggered
                continue

            # Generate Buy Exit Signal
            if df['AO'].iloc[i] < 0 or df['ADX_Slope'].iloc[i] < 0:
                df.at[df.index[i], 'BUY_EXIT_SIGNAL'] = 1
                buy_index = None  # Reset buy_index after exiting the trade
                continue

    return df



def generate_buy_sell_signals_by_ADX_v4(df, adx_percent_threshold=0.3, stop_loss_threshold_principal=0.1, stop_loss_threshold_profit=0.3):
    df['BUY_SIGNAL'] = 0
    df['BUY_EXIT_SIGNAL'] = 0
    df['SELL_SIGNAL'] = 0
    df['SELL_EXIT_SIGNAL'] = 0
    df['Buy_Stop_Loss_Price'] = np.nan
    df['Cum_Profit_Since_Buy'] = np.nan
    df['Cum_Max_Profit_Since_Buy'] = np.nan
    df['Buy_Stop_Loss_Signal'] = 0
    buy_index = None  # Track the index of the last buy signal

    # Calculate ADX range and dynamic threshold
    adx_min = df['ADX'].min()
    adx_max = df['ADX'].max()
    adx_range = adx_max - adx_min
    adx_threshold = adx_min + adx_range * adx_percent_threshold

    for i in range(1, len(df)):
        # Generate Buy Signal
        if buy_index is None and df['AO'].iloc[i] > 0 and df['ADX_Slope'].iloc[i] > 0 and df['ADX'].iloc[i] < adx_threshold:
            df.at[df.index[i], 'BUY_SIGNAL'] = 1
            buy_index = i  # Update buy_index to the current index
            df.at[df.index[i], 'Buy_Stop_Loss_Price'] = -df['Close'].iloc[i]
            df.at[df.index[i], 'Cum_Profit_Since_Buy'] = 0
            df.at[df.index[i], 'Cum_Max_Profit_Since_Buy'] = 0  # Initialize the max profit at 0

            continue

        # Calculate Cumulative Profit Since Buy
        if buy_index is not None and i > buy_index:
            cum_profit = df['Close'].iloc[i] - df['Close'].iloc[buy_index]
            df.at[df.index[i], 'Cum_Profit_Since_Buy'] = cum_profit

            # Calculate Cumulative Max Profit Since Buy
            cum_max_profit = max(cum_profit, df['Cum_Max_Profit_Since_Buy'].iloc[i-1])
            df.at[df.index[i], 'Cum_Max_Profit_Since_Buy'] = cum_max_profit

            # Update the stop loss price based on the maximum cumulative profit
            stop_loss_price = df['Close'].iloc[i] - cum_max_profit * stop_loss_threshold_profit
            
            if stop_loss_price >= df['Buy_Stop_Loss_Price'].iloc[i-1]:
                df.at[df.index[i], 'Buy_Stop_Loss_Price'] = df['Buy_Stop_Loss_Price'].iloc[i-1]
            else:
                df.at[df.index[i], 'Buy_Stop_Loss_Price'] = df['Buy_Stop_Loss_Price'].iloc[i-1]

            # Check if the current low is less than the stop loss price from the last row
            if df['Low'].iloc[i] < df['Buy_Stop_Loss_Price'].iloc[i - 1]:
                df.at[df.index[i], 'Buy_Stop_Loss_Signal'] = 1
                buy_index = None  # Reset buy_index after stop loss is triggered
                continue

            # Generate Buy Exit Signal
            if df['AO'].iloc[i] < 0 or df['ADX_Slope'].iloc[i] < 0:
                df.at[df.index[i], 'BUY_EXIT_SIGNAL'] = 1
                buy_index = None  # Reset buy_index after exiting the trade
                continue

    return df


def generate_buy_sell_signals_by_KDJ(df):
    # Initialize columns with zero values
    df['BUY_SIGNAL'] = 0
    df['BUY_EXIT_SIGNAL'] = 0
    df['SELL_SIGNAL'] = 0
    df['SELL_EXIT_SIGNAL'] = 0
    df['Buy_Stop_Loss_Price'] = np.nan
    df['Cum_Profit_Since_Buy'] = np.nan
    df['Cum_Max_Profit_Since_Buy'] = np.nan
    df['Buy_Stop_Loss_Signal'] = 0
    
    # Loop through the DataFrame to generate buy/sell signals based on the J line slope
    for i in range(1, len(df)):
        # Generate Buy Signal: J-line slope turns positive from negative
        if df['J_Slope'].iloc[i-1] < 0 and df['J_Slope'].iloc[i] > 0 and df['K_Slope'].iloc[i] > 0:
            df.at[df.index[i], 'BUY_SIGNAL'] = 1
        
        # Generate Sell Signal: J-line slope turns negative from positive
        elif df['J_Slope'].iloc[i-1] > 0 and df['J_Slope'].iloc[i] < 0:
            df.at[df.index[i], 'SELL_SIGNAL'] = 1
    
    return df


def generate_buy_sell_signals_by_gold_dead_cross(df):
    # Initialize columns with zero values
    df['BUY_SIGNAL'] = 0
    df['BUY_EXIT_SIGNAL'] = 0
    df['SELL_SIGNAL'] = 0
    df['Sell_Price'] = np.nan

    df['SELL_EXIT_SIGNAL'] = 0
    df['Buy_Stop_Loss_Price'] = np.nan
    df['Cum_Profit_Since_Buy'] = np.nan
    df['Cum_Max_Profit_Since_Buy'] = np.nan
    df['Buy_Stop_Loss_Signal'] = 0
    df['Buy_Exit_Price'] = np.nan



    df['HOLD_SIGNAL'] = 0
    df['HOLD_EXIT_SIGNAL'] = 0
    df['Hold_Price'] = np.nan
    df['Hold_Exit_Price'] = np.nan
    
    latest_buy_index = None
    
    # Loop through the DataFrame to generate buy/sell signals based on the J and K line crosses
    for i in range(1, len(df)):
        # Generate Buy Signal when J crosses above K (Golden Cross) and there's no active buy position
        if latest_buy_index is None:
            if df['J'].iloc[i-1] <= df['K'].iloc[i-1] and df['J'].iloc[i] >= df['K'].iloc[i] and df['K_Slope'].iloc[i] > 0 and df['D_Slope'].iloc[i] > 0 and df['J'].iloc[i]<=100:
            # if df['J'].iloc[i-1] <= df['K'].iloc[i-1] and df['J'].iloc[i] >= df['K'].iloc[i] and df['K_Slope'].iloc[i] > 0 and df['D_Slope'].iloc[i] > 0 and df['J'].iloc[i]<=100 and df['RSI'].iloc[i] >= 50:
            # if df['J'].iloc[i] <= df['K'].iloc[i] and df['J_Slope'].iloc[i] >= 0:
                df.at[df.index[i], 'BUY_SIGNAL'] = 1
                latest_buy_index = i
        
        # Generate Sell Signal when J crosses below K (Dead Cross) or J_Slope < 0
        if latest_buy_index is not None:
            # if (df['J'].iloc[i-1] > df['K'].iloc[i-1] and df['J'].iloc[i] < df['K'].iloc[i]) or (df['J'].iloc[i] >= df['K'].iloc[i] and df['J_Slope'].iloc[i] < 0):
            if (df['J'].iloc[i-1] > df['K'].iloc[i-1] and df['J'].iloc[i] < df['K'].iloc[i]) :
                df.at[df.index[i], 'BUY_EXIT_SIGNAL'] = 1
                df.at[df.index[i], 'Buy_Exit_Price'] = df['Close'].iloc[i]
                latest_buy_index = None


        if latest_buy_index is not None:
            # if (df['J'].iloc[i-1] > df['K'].iloc[i-1] and df['J'].iloc[i] < df['K'].iloc[i]) or (df['J'].iloc[i] >= df['K'].iloc[i] and df['J_Slope'].iloc[i] < 0):
            if (df['J'].iloc[i-1] > df['K'].iloc[i-1] and df['J'].iloc[i] < df['K'].iloc[i]) :
                df.at[df.index[i], 'SELL_SIGNAL'] = 1
                df.at[df.index[i], 'Sell_Price'] = df['Close'].iloc[i]
                latest_buy_index = None
    
    return df

def generate_buy_sell_signals_by_gold_dead_cross_only_monthly_from_daily_return(df):
    # Initialize columns with zero values
    df['BUY_SIGNAL'] = 0
    df['BUY_EXIT_SIGNAL'] = 0
    df['SELL_SIGNAL'] = 0
    df['SELL_EXIT_SIGNAL'] = 0
    df['Buy_Stop_Loss_Price'] = np.nan
    df['Cum_Profit_Since_Buy'] = np.nan
    df['Cum_Max_Profit_Since_Buy'] = np.nan
    df['Buy_Stop_Loss_Signal'] = 0
    df['DrawDown_Since_Latest_Buy'] = 0
    df['DrawDown_Life_Time'] = 0
    df['Buy_Exit_Price'] = np.nan

    drawdown_threshold = 1.0
    
    monthly_kdj_condition = 0
    latest_buy_index = None
    peak_price_since_buy = None
    peak_price_lifetime = df['Close'].iloc[0]
    
    # Loop through the DataFrame to generate buy/sell signals based on the J and K line crosses
    for i in range(1, len(df)):
        current_price = df['Close'].iloc[i]
        
        # Calculate DrawDown_Since_Latest_Buy
        if latest_buy_index is not None:
            peak_price_since_buy = max(peak_price_since_buy, current_price)
            drawdown = (current_price - peak_price_since_buy) / peak_price_since_buy
            price_just_meet_the_threshold = peak_price_since_buy * (1 - drawdown_threshold)  # 20% drawdown
            df.at[df.index[i], 'DrawDown_Since_Latest_Buy'] = drawdown
        else:
            df.at[df.index[i], 'DrawDown_Since_Latest_Buy'] = 0
        
        # Calculate DrawDown_Life_Time
        peak_price_lifetime = max(peak_price_lifetime, current_price)
        drawdown_lifetime = (current_price - peak_price_lifetime) / peak_price_lifetime
        df.at[df.index[i], 'DrawDown_Life_Time'] = drawdown_lifetime
        
        # Generate Buy Signal when J crosses above K (Golden Cross) on monthly KDJ, then check daily KDJ
        # if latest_buy_index is None:
        #     if (df['Monthly_J'].iloc[i-1] <= df['Monthly_K'].iloc[i-1] and 
        #         df['Monthly_J'].iloc[i] >= df['Monthly_K'].iloc[i]
        #     #  ): 
        #      and 
        #     #     df['Monthly_K_Slope'].iloc[i] > 0 and 
        #     #     df['Monthly_D_Slope'].iloc[i] > 0 and 
        #         # df['Monthly_J'].iloc[i] <= 100):

        #         df['Monthly_J'].iloc[i] <= 100 and
        #         df['RSI'].iloc[i] >= 45):

        #     #     df['Monthly_J'].iloc[i] <= 100 and
        #     #     df['SPY_Close_minus_SMA_30'].iloc[i] >= -20):

        #         df.at[df.index[i], 'BUY_SIGNAL'] = 1
        #         latest_buy_index = i
        #         peak_price_since_buy = current_price
        #     elif df['SPY_Close_minus_SMA_30'] >=0:
        #         df.at[df.index[i], 'BUY_SIGNAL'] = 1
        #         latest_buy_index = i
        #         peak_price_since_buy = current_price


        if latest_buy_index is None:
            if (df['Monthly_J'].iloc[i-1] <= df['Monthly_K'].iloc[i-1] and 
                df['Monthly_J'].iloc[i] >= df['Monthly_K'].iloc[i] and 
                df['Monthly_J'].iloc[i] <= 100 and
                df['RSI'].iloc[i] >= 45):
                # Optional conditions you had commented out:
                # df['Monthly_K_Slope'].iloc[i] > 0 and 
                # df['Monthly_D_Slope'].iloc[i] > 0 and 
                # df['SPY_Close_minus_SMA_30'].iloc[i] >= -20:

                df.at[df.index[i], 'BUY_SIGNAL'] = 1
                latest_buy_index = i
                peak_price_since_buy = current_price
            elif  df['Monthly_J'].iloc[i] >= df['Monthly_K'].iloc[i]  and df['SPY_Close_minus_SMA_30'].iloc[i] >= 0:
            # elif df['SPY_Close_minus_SMA_30'].iloc[i] >= 0:
                df.at[df.index[i], 'BUY_SIGNAL'] = 1
                latest_buy_index = i
                peak_price_since_buy = current_price
            
        # Generate Buy Exit Signal when J crosses below K (Dead Cross) on daily KDJ or DrawDown_Since_Latest_Buy < -20%
        if latest_buy_index is not None:
            if df.at[df.index[i], 'DrawDown_Since_Latest_Buy'] < -1 * drawdown_threshold:
                df.at[df.index[i], 'BUY_EXIT_SIGNAL'] = 1
                
                df.at[df.index[i], 'Buy_Exit_Price'] = price_just_meet_the_threshold
                latest_buy_index = None
            elif df['SPY_Close_minus_SMA_30'].iloc[i] <= -20:
                df.at[df.index[i], 'BUY_EXIT_SIGNAL'] = 1
                df.at[df.index[i], 'Buy_Exit_Price'] = current_price
                latest_buy_index = None
            elif df['Monthly_J'].iloc[i-1] > df['Monthly_K'].iloc[i-1] and df['Monthly_J'].iloc[i] < df['Monthly_K'].iloc[i]:
                df.at[df.index[i], 'BUY_EXIT_SIGNAL'] = 1
                df.at[df.index[i], 'Buy_Exit_Price'] = current_price
                latest_buy_index = None
            # elif df['MACD'].iloc[i] <= -20:
            #     df.at[df.index[i], 'BUY_EXIT_SIGNAL'] = 1
            #     df.at[df.index[i], 'Buy_Exit_Price'] = current_price
            #     latest_buy_index = None
    return df


def generate_buy_sell_signals_early_detection(df):
    """
    Generate buy and buy exit signals with early detection based on specified conditions.
    
    Arguments:
    df -- DataFrame containing columns 'Monthly_J', 'Monthly_K', 'Monthly_D', 'Close', etc.
    
    Returns:
    DataFrame with additional columns for buy/sell signals and performance tracking.
    """

    # Initialize columns with zero values
    df['BUY_SIGNAL'] = 0
    df['BUY_EXIT_SIGNAL'] = 0
    df['SELL_SIGNAL'] = 0
    df['SELL_EXIT_SIGNAL'] = 0
    df['Buy_Stop_Loss_Price'] = np.nan
    df['Cum_Profit_Since_Buy'] = np.nan
    df['Cum_Max_Profit_Since_Buy'] = np.nan
    df['Buy_Stop_Loss_Signal'] = 0
    df['DrawDown_Since_Latest_Buy'] = 0
    df['DrawDown_Life_Time'] = 0
    df['Buy_Exit_Price'] = np.nan

    drawdown_threshold = 1.0
    latest_buy_index = None
    peak_price_since_buy = None
    peak_price_lifetime = df['Close'].iloc[0]

    # Iterate over each row to generate signals
    for i in range(1, len(df)):
        current_price = df['Close'].iloc[i]
        
        # Track the peak price for lifetime and since the latest buy
        peak_price_lifetime = max(peak_price_lifetime, current_price)
        if latest_buy_index is not None:
            peak_price_since_buy = max(peak_price_since_buy, current_price)
        
        # Calculate DrawDown_Since_Latest_Buy
        if latest_buy_index is not None:
            drawdown = (current_price - peak_price_since_buy) / peak_price_since_buy
            price_just_meet_the_threshold = peak_price_since_buy * (1 - drawdown_threshold)  # e.g., 20% drawdown
            df.at[df.index[i], 'DrawDown_Since_Latest_Buy'] = drawdown
        else:
            df.at[df.index[i], 'DrawDown_Since_Latest_Buy'] = 0

        # Calculate DrawDown_Life_Time
        drawdown_lifetime = (current_price - peak_price_lifetime) / peak_price_lifetime
        df.at[df.index[i], 'DrawDown_Life_Time'] = drawdown_lifetime

        # Generate Buy Signal based on updated Monthly J, K, D conditions with 5-day uptrend check
        if latest_buy_index is None:
            # Check if Monthly_J has been increasing for the last 5 days
            n_days = 3
            j_increasing_n_days = all(df['Monthly_J'].iloc[i - j] > df['Monthly_J'].iloc[i - j - 1] for j in range(1, n_days))
            
            if (j_increasing_n_days and
                df['Monthly_J'].iloc[i] < df['Monthly_K'].iloc[i] and
                df['Monthly_J'].iloc[i] <= 70 and
                df['Monthly_J'].iloc[i] < df['Monthly_D'].iloc[i]):
                df.at[df.index[i], 'BUY_SIGNAL'] = 1
                latest_buy_index = i
                peak_price_since_buy = current_price
                df.at[df.index[i], 'Buy_Stop_Loss_Price'] = current_price * (1 - drawdown_threshold / 100)
                
                
        # Generate Buy Exit Signal based on updated Monthly J, K, D conditions
        elif latest_buy_index is not None:
            if df.at[df.index[i], 'DrawDown_Since_Latest_Buy'] < -drawdown_threshold:
                df.at[df.index[i], 'BUY_EXIT_SIGNAL'] = 1
                df.at[df.index[i], 'Buy_Exit_Price'] = price_just_meet_the_threshold
                latest_buy_index = None
            elif (df['Monthly_J'].iloc[i] < df['Monthly_J'].iloc[i - 1] and
                  df['Monthly_J'].iloc[i] > df['Monthly_K'].iloc[i] and
                  df['Monthly_J'].iloc[i] > df['Monthly_D'].iloc[i]):
                df.at[df.index[i], 'BUY_EXIT_SIGNAL'] = 1
                df.at[df.index[i], 'Buy_Exit_Price'] = current_price
                latest_buy_index = None

    return df



def generate_buy_sell_signals_early_detection_3_stages(df, n_days=10, drawdown_threshold=1.0):
    """
    Generate buy, hold, and sell signals with early detection based on specified conditions.
    """
    # Initialize columns
    df['BUY_SIGNAL'] = 0
    df['BUY_EXIT_SIGNAL'] = 0
    df['HOLD_SIGNAL'] = 0
    df['HOLD_EXIT_SIGNAL'] = 0
    df['SELL_SIGNAL'] = 0
    df['SELL_EXIT_SIGNAL'] = 0
    df['Buy_Price'] = np.nan
    df['Buy_Exit_Price'] = np.nan
    df['Hold_Price'] = np.nan
    df['Hold_Exit_Price'] = np.nan
    df['Sell_Price'] = np.nan
    df['Sell_Exit_Price'] = np.nan
    df['DrawDown_Since_Latest_Buy'] = 0
    df['DrawDown_Life_Time'] = 0
    df['Buy_Stop_Loss_Price'] = np.nan

    # Trade state tracking
    latest_buy_index = None
    latest_hold_index = None
    latest_sell_index = None
    peak_price_since_buy = None
    peak_price_lifetime = df['Close'].iloc[0]

    def end_existing_trade(i, current_price, trade_type):
        """End a specific type of existing trade."""
        nonlocal latest_buy_index, latest_hold_index, latest_sell_index
        
        if trade_type == 'buy' and latest_buy_index is not None:
            df.at[df.index[i], 'BUY_EXIT_SIGNAL'] = 1
            df.at[df.index[i], 'Buy_Exit_Price'] = current_price
            latest_buy_index = None
            
        elif trade_type == 'hold' and latest_hold_index is not None:
            df.at[df.index[i], 'HOLD_EXIT_SIGNAL'] = 1
            df.at[df.index[i], 'Hold_Exit_Price'] = current_price
            latest_hold_index = None
            
        elif trade_type == 'sell' and latest_sell_index is not None:
            df.at[df.index[i], 'SELL_EXIT_SIGNAL'] = 1
            df.at[df.index[i], 'Sell_Exit_Price'] = current_price
            latest_sell_index = None

    def start_new_trade(i, trade_type, current_price):
        """Start a new trade of the specified type."""
        nonlocal latest_buy_index, latest_hold_index, latest_sell_index, peak_price_since_buy
        
        if trade_type == 'buy' and latest_buy_index is None:
            df.at[df.index[i], 'BUY_SIGNAL'] = 1
            df.at[df.index[i], 'Buy_Price'] = current_price
            latest_buy_index = i
            peak_price_since_buy = current_price
            df.at[df.index[i], 'Buy_Stop_Loss_Price'] = current_price * (1 - drawdown_threshold / 100)
            return True
            
        elif trade_type == 'hold' and latest_hold_index is None:
            df.at[df.index[i], 'HOLD_SIGNAL'] = 1
            df.at[df.index[i], 'Hold_Price'] = current_price
            latest_hold_index = i
            return True
            
        elif trade_type == 'sell' and latest_sell_index is None:
            df.at[df.index[i], 'SELL_SIGNAL'] = 1
            df.at[df.index[i], 'Sell_Price'] = current_price
            latest_sell_index = i
            return True
            
        return False

    # Iterate through the DataFrame
    for i in range(n_days, len(df)):
        current_price = df['Close'].iloc[i]
        
        # Update peak prices and drawdowns
        peak_price_lifetime = max(peak_price_lifetime, current_price)
        if latest_buy_index is not None:
            peak_price_since_buy = max(peak_price_since_buy, current_price)
            drawdown = (current_price - peak_price_since_buy) / peak_price_since_buy
            df.at[df.index[i], 'DrawDown_Since_Latest_Buy'] = drawdown
        
        drawdown_lifetime = (current_price - peak_price_lifetime) / peak_price_lifetime
        df.at[df.index[i], 'DrawDown_Life_Time'] = drawdown_lifetime
        # Check for buy conditions
        j_increasing_n_days = all(df['Monthly_J'].iloc[i - j] > df['Monthly_J'].iloc[i - j - 1] for j in range(1, n_days))
        if j_increasing_n_days and df['Monthly_J'].iloc[i] < df['Monthly_K'].iloc[i] and df['Monthly_J'].iloc[i] < df['Monthly_D'].iloc[i] and (df['Monthly_K'].iloc[i] - df['Monthly_J'].iloc[i] >= 10):
        # if j_increasing_n_days and df['Monthly_J'].iloc[i] < df['Monthly_K'].iloc[i] and df['Monthly_J'].iloc[i] < df['Monthly_D'].iloc[i] and (df['Monthly_D_Slope']>=-0.5):
            end_existing_trade(i, current_price, 'sell')
            end_existing_trade(i, current_price, 'hold')
            start_new_trade(i, 'buy', current_price)
            continue

        # Check for hold conditions
        if df['Monthly_J'].iloc[i] < df['Monthly_J'].iloc[i - 1] and ( df['Monthly_J'].iloc[i] > df['Monthly_K'].iloc[i] and df['Monthly_J'].iloc[i] > df['Monthly_D'].iloc[i]):
        # if (df['Monthly_J'].iloc[i] < df['Monthly_J'].iloc[i - 1] and latest_buy_index) or (df['Monthly_J'].iloc[i] > df['Monthly_J'].iloc[i - 1] and latest_sell_index):
        # if (df['Monthly_J'].iloc[i] < df['Monthly_J'].iloc[i - 1] and latest_buy_index):
            end_existing_trade(i, current_price, 'buy')
            end_existing_trade(i, current_price, 'sell')
            start_new_trade(i, 'hold', current_price)
            continue

        # Check for sell conditions
        if df['Monthly_J'].iloc[i - 1] > df['Monthly_K'].iloc[i - 1] and df['Monthly_J'].iloc[i] < df['Monthly_K'].iloc[i]:
            end_existing_trade(i, current_price, 'buy')
            end_existing_trade(i, current_price, 'hold')
            start_new_trade(i, 'sell', current_price)
            continue

        # Check for exit conditions
        if latest_buy_index is not None and df.at[df.index[i], 'DrawDown_Since_Latest_Buy'] < -drawdown_threshold:
            price_just_meet_the_threshold = peak_price_since_buy * (1 - drawdown_threshold)
            end_existing_trade(i, price_just_meet_the_threshold, 'buy')

    return df


def generate_buy_sell_signals_golden_dead_cross_MFI(df, n_days=10, drawdown_threshold=1.0, required_down_days=4, required_up_days=7, required_ADX_down_days=5):
    """
    Generate buy, hold, and sell signals with early detection based on specified conditions.
    """
    # Initialize columns
    df['BUY_SIGNAL'] = 0
    df['BUY_EXIT_SIGNAL'] = 0
    df['HOLD_SIGNAL'] = 0
    df['HOLD_EXIT_SIGNAL'] = 0
    df['SELL_SIGNAL'] = 0
    df['SELL_EXIT_SIGNAL'] = 0
    df['Buy_Price'] = np.nan
    df['Buy_Exit_Price'] = np.nan
    df['Hold_Price'] = np.nan
    df['Hold_Exit_Price'] = np.nan
    df['Sell_Price'] = np.nan
    df['Sell_Exit_Price'] = np.nan
    df['DrawDown_Since_Latest_Buy'] = 0
    df['DrawDown_Life_Time'] = 0
    df['Buy_Stop_Loss_Price'] = np.nan
    
    # Add debug and state columns
    df['Golden_Cross'] = 0  # Marks when golden cross occurs
    df['Death_Cross'] = 0   # Marks when death cross occurs
    df['Cross_Status'] = 0  # 1 for golden cross period, 0 for death cross period
    df['Buy_Signal_Count'] = 0

    # Trade state tracking
    latest_buy_index = None
    latest_hold_index = None
    latest_sell_index = None
    peak_price_since_buy = None
    peak_price_lifetime = df['Close'].iloc[0]

    # Cross tracking
    last_golden_cross_index = None
    number_buy_signal_since_golden_cross = None  # None means no golden cross yet observed
    is_in_golden_cross = False  # Tracks whether we're in golden cross period

    # Initialize cross status based on initial J/K relationship
    if df['Monthly_J'].iloc[0] > df['Monthly_K'].iloc[0]:
        is_in_golden_cross = True
        df.at[df.index[0], 'Cross_Status'] = is_in_golden_cross

    def end_existing_trade(i, current_price, trade_type):
        """End a specific type of existing trade."""
        nonlocal latest_buy_index, latest_hold_index, latest_sell_index
        
        if trade_type == 'buy' and latest_buy_index is not None:
            df.at[df.index[i], 'BUY_EXIT_SIGNAL'] = 1
            df.at[df.index[i], 'Buy_Exit_Price'] = current_price
            latest_buy_index = None
            
        elif trade_type == 'hold' and latest_hold_index is not None:
            df.at[df.index[i], 'HOLD_EXIT_SIGNAL'] = 1
            df.at[df.index[i], 'Hold_Exit_Price'] = current_price
            latest_hold_index = None
            
        elif trade_type == 'sell' and latest_sell_index is not None:
            df.at[df.index[i], 'SELL_EXIT_SIGNAL'] = 1
            df.at[df.index[i], 'Sell_Exit_Price'] = current_price
            latest_sell_index = None

    def start_new_trade(i, trade_type, current_price):
        """Start a new trade of the specified type."""
        nonlocal latest_buy_index, latest_hold_index, latest_sell_index, peak_price_since_buy, number_buy_signal_since_golden_cross
        
        if trade_type == 'buy' and latest_buy_index is None:
            df.at[df.index[i], 'BUY_SIGNAL'] = 1
            df.at[df.index[i], 'Buy_Price'] = current_price
            latest_buy_index = i
            peak_price_since_buy = current_price
            df.at[df.index[i], 'Buy_Stop_Loss_Price'] = current_price * (1 - drawdown_threshold / 100)
            if number_buy_signal_since_golden_cross is not None:
                number_buy_signal_since_golden_cross += 1
                df.at[df.index[i], 'Buy_Signal_Count'] = number_buy_signal_since_golden_cross
            return True
            
        elif trade_type == 'hold' and latest_hold_index is None:
            df.at[df.index[i], 'HOLD_SIGNAL'] = 1
            df.at[df.index[i], 'Hold_Price'] = current_price
            latest_hold_index = i
            return True
            
        elif trade_type == 'sell' and latest_sell_index is None:
            df.at[df.index[i], 'SELL_SIGNAL'] = 1
            df.at[df.index[i], 'Sell_Price'] = current_price
            latest_sell_index = i
            return True
            
        return False

    def can_generate_buy_signal(i):
        """Check if we can generate a buy signal based on golden cross conditions."""
        nonlocal number_buy_signal_since_golden_cross, last_golden_cross_index, is_in_golden_cross
        
        # Can only generate signal if we're in golden cross period and haven't generated a signal since last golden cross
        return (is_in_golden_cross and
                last_golden_cross_index is not None and 
                i > last_golden_cross_index and 
                number_buy_signal_since_golden_cross <= 1)



    def days_ADX_goes_down(i):
        """
        Check if ADX has been consistently decreasing for the required number of days.
        Only checks for consecutive down days.
        
        Parameters:
        i (int): Current index in the dataframe
        
        Returns:
        bool: True if ADX has been going down for required_ADX_down_days, False otherwise
        """
        if i < required_ADX_down_days:
            return False
            
        # Count consecutive negative days up to current position
        consecutive_down_days = 0
        
        # Start from current day and look backward
        for j in range(i, i - required_ADX_down_days - 1, -1):
            if j > 0 and df['ADX'].iloc[j] < df['ADX'].iloc[j-1]:
                consecutive_down_days += 1
            else:
                break
                
        return consecutive_down_days >= required_ADX_down_days
    

    def days_J_goes_up(i):
        """
        Check if Monthly_J has been consistently increasing for the required number of days.
        Only checks for consecutive up days without looking for flip points.
        
        Parameters:
        i (int): Current index in the dataframe
        
        Returns:
        bool: True if J has been going up for required_up_days, False otherwise
        """
        if i < required_up_days:
            return False
            
        # Count consecutive positive days up to current position
        consecutive_up_days = 0
        
        # Start from current day and look backward
        for j in range(i, i - required_up_days - 1, -1):
            if j > 0 and df['Monthly_J'].iloc[j] > df['Monthly_J'].iloc[j-1]:
                consecutive_up_days += 1
            else:
                break
                
        return consecutive_up_days >= required_up_days


    def days_J_goes_up_since_flip(i):
        """
        Check for upward flipping point and count consecutive positive days.
        Returns True if finds a flip point and enough consecutive positive days after it.
        """
        if i < required_up_days + 2:
            return False
            
        # Look for flip point from i-(required_up_days+1) to i
        flip_idx = None
        for j in range(i - (required_up_days + 1), i + 1):
            if j > 0 and j < i:  # Ensure we have data for j-1 and j
                # Check for flip from negative to positive growth
                if (df['Monthly_J'].iloc[j-1] < df['Monthly_J'].iloc[j-2] and  # Previous growth was negative
                    df['Monthly_J'].iloc[j] > df['Monthly_J'].iloc[j-1]):      # Current growth is positive
                    flip_idx = j
                    break
                    
        if flip_idx is None:  # No flip point found
            return False
            
        # Count consecutive positive days since flip
        positive_days = 0
        for k in range(flip_idx, i + 1):
            if k > 0 and df['Monthly_J'].iloc[k] > df['Monthly_J'].iloc[k-1]:
                positive_days += 1
            else:
                break
                
        return positive_days >= required_up_days

    def days_J_goes_down_since_flip(i):
        """
        Check for flipping point in lookback period and count consecutive negative days.
        Returns True if finds a flip point and enough consecutive negative days after it.
        """
        if i < required_down_days + 2:
            return False
            
        # Look for flip point from i-(required_down_days+1) to i
        flip_idx = None
        for j in range(i - (required_down_days + 1), i + 1):
            if j > 0 and j < i:  # Ensure we have data for j-1 and j
                # Check for flip from positive to negative growth
                if (df['Monthly_J'].iloc[j-1] > df['Monthly_J'].iloc[j-2] and  # Previous growth was positive
                    df['Monthly_J'].iloc[j] < df['Monthly_J'].iloc[j-1]):      # Current growth is negative
                    flip_idx = j
                    break
                    
        if flip_idx is None:  # No flip point found
            return False
            
        # Count consecutive negative days since flip
        negative_days = 0
        for k in range(flip_idx, i + 1):
            if k > 0 and df['Monthly_J'].iloc[k] < df['Monthly_J'].iloc[k-1]:
                negative_days += 1
            else:
                break
                
        return negative_days >= required_down_days

    # Iterate through the DataFrame
    for i in range(n_days, len(df)):
        current_price = df['Close'].iloc[i]
        
        # Update peak prices and drawdowns
        peak_price_lifetime = max(peak_price_lifetime, current_price)
        if latest_buy_index is not None:
            peak_price_since_buy = max(peak_price_since_buy, current_price)
            drawdown = (current_price - peak_price_since_buy) / peak_price_since_buy
            df.at[df.index[i], 'DrawDown_Since_Latest_Buy'] = drawdown
        
        drawdown_lifetime = (current_price - peak_price_lifetime) / peak_price_lifetime
        df.at[df.index[i], 'DrawDown_Life_Time'] = drawdown_lifetime
        
        # Check for golden cross and death cross
        if i > 1:
            # Check for cross changes
            # if df['Monthly_J'].iloc[i] >= df['Monthly_K'].iloc[i] and df['Monthly_J'].iloc[i-1] <= df['Monthly_K'].iloc[i-1] :
            if df['Monthly_J'].iloc[i] > df['Monthly_K'].iloc[i]:
                # Currently in golden cross state
                if not is_in_golden_cross:  # Just crossed over
                    is_in_golden_cross = True
                    last_golden_cross_index = i
                    number_buy_signal_since_golden_cross = 0
                    df.at[df.index[i], 'Golden_Cross'] = 1
                df.at[df.index[i], 'Cross_Status'] = is_in_golden_cross

            # if df['Monthly_J'].iloc[i] <= df['Monthly_K'].iloc[i] and df['Monthly_J'].iloc[i-1] >= df['Monthly_K'].iloc[i-1] :
            if df['Monthly_J'].iloc[i] < df['Monthly_K'].iloc[i] :
                # Currently in death cross state
                if is_in_golden_cross:  # Just crossed under
                    is_in_golden_cross = False
                    number_buy_signal_since_golden_cross = None
                    df.at[df.index[i], 'Death_Cross'] = 1
                df.at[df.index[i], 'Cross_Status'] = is_in_golden_cross

        # Check for buy conditions with different rules for golden/death cross
        if is_in_golden_cross:
            # During golden cross period - check all conditions including buy signal count
            # if (days_J_goes_up_since_flip(i) and 
            #     df['Monthly_D'].iloc[i] > df['Monthly_D'].iloc[i-1] and 
            #     can_generate_buy_signal(i)):
                

            if (days_J_goes_up(i) and 
                df['Monthly_D'].iloc[i] > df['Monthly_D'].iloc[i-1] and 
                can_generate_buy_signal(i)):
                end_existing_trade(i, current_price, 'sell')
                end_existing_trade(i, current_price, 'hold')
                start_new_trade(i, 'buy', current_price)
                continue
        # else:
        #     # During death cross period - only check basic conditions
        #     # if (days_J_goes_up_since_flip(i) and 
        #     #     df['Monthly_D'].iloc[i] > df['Monthly_D'].iloc[i-1]):
                
        #     if (days_J_goes_up_since_flip(i) and
        #         days_ADX_goes_down(i)):
        #         # df['ADX'].iloc[i] <= df['ADX'].iloc[i-1] and
        #         # df['Monthly_J'].iloc[i] <= df['Monthly_K'].iloc[i] - 10):

                
        #         end_existing_trade(i, current_price, 'sell')
        #         end_existing_trade(i, current_price, 'hold')
        #         start_new_trade(i, 'buy', current_price)
        #         continue


        # During golden cross period - check for death cross happening
        if (df['Monthly_J'].iloc[i] <= df['Monthly_K'].iloc[i] and 
            df['Monthly_J'].iloc[i-1] >= df['Monthly_K'].iloc[i-1]):
            
            end_existing_trade(i, current_price, 'buy')
            end_existing_trade(i, current_price, 'hold')
            start_new_trade(i, 'sell', current_price)
            continue

        # if number_buy_signal_since_golden_cross == 1:
        #     # First buy signal - check for J going down
        #     if days_J_goes_down_since_flip(i):
        #         end_existing_trade(i, current_price, 'buy')
        #         end_existing_trade(i, current_price, 'hold')
        #         start_new_trade(i, 'sell', current_price)
        #         continue

        # # Check for sell conditions based on cross state
        # if is_in_golden_cross:
        #     # During golden cross period - check for death cross happening
        #     if (df['Monthly_J'].iloc[i] <= df['Monthly_K'].iloc[i] and 
        #         df['Monthly_J'].iloc[i-1] >= df['Monthly_K'].iloc[i-1]):
                
        #         end_existing_trade(i, current_price, 'buy')
        #         end_existing_trade(i, current_price, 'hold')
        #         start_new_trade(i, 'sell', current_price)
        #         continue
        # else:
        #     # During death cross period - check for J going down
        #     if days_J_goes_down_since_flip(i):
        #         end_existing_trade(i, current_price, 'buy')
        #         end_existing_trade(i, current_price, 'hold')
        #         start_new_trade(i, 'sell', current_price)
        #         continue

        # Check for exit conditions
        if latest_buy_index is not None and df.at[df.index[i], 'DrawDown_Since_Latest_Buy'] < -drawdown_threshold:
            price_just_meet_the_threshold = peak_price_since_buy * (1 - drawdown_threshold)
            end_existing_trade(i, price_just_meet_the_threshold, 'buy')

    return df


def generate_buy_sell_signals_early_detection_3_stages_MFI(df, n_days=10, drawdown_threshold=1.0, required_down_days=4, required_up_days=7, required_ADX_down_days=5):
    """
    Generate buy, hold, and sell signals with early detection based on specified conditions.
    """
    # Initialize columns
    df['BUY_SIGNAL'] = 0
    df['BUY_EXIT_SIGNAL'] = 0
    df['HOLD_SIGNAL'] = 0
    df['HOLD_EXIT_SIGNAL'] = 0
    df['SELL_SIGNAL'] = 0
    df['SELL_EXIT_SIGNAL'] = 0
    df['Buy_Price'] = np.nan
    df['Buy_Exit_Price'] = np.nan
    df['Hold_Price'] = np.nan
    df['Hold_Exit_Price'] = np.nan
    df['Sell_Price'] = np.nan
    df['Sell_Exit_Price'] = np.nan
    df['DrawDown_Since_Latest_Buy'] = 0
    df['DrawDown_Life_Time'] = 0
    df['Buy_Stop_Loss_Price'] = np.nan
    
    # Add debug and state columns
    df['Golden_Cross'] = 0  # Marks when golden cross occurs
    df['Death_Cross'] = 0   # Marks when death cross occurs
    df['Cross_Status'] = 0  # 1 for golden cross period, 0 for death cross period
    df['Buy_Signal_Count'] = 0

    # Trade state tracking
    latest_buy_index = None
    latest_hold_index = None
    latest_sell_index = None
    peak_price_since_buy = None
    peak_price_lifetime = df['Close'].iloc[0]

    # Cross tracking
    last_golden_cross_index = None
    number_buy_signal_since_golden_cross = None  # None means no golden cross yet observed
    is_in_golden_cross = False  # Tracks whether we're in golden cross period

    # Initialize cross status based on initial J/K relationship
    if df['Monthly_J'].iloc[0] > df['Monthly_K'].iloc[0]:
        is_in_golden_cross = True
        df.at[df.index[0], 'Cross_Status'] = is_in_golden_cross

    def end_existing_trade(i, current_price, trade_type):
        """End a specific type of existing trade."""
        nonlocal latest_buy_index, latest_hold_index, latest_sell_index
        
        if trade_type == 'buy' and latest_buy_index is not None:
            df.at[df.index[i], 'BUY_EXIT_SIGNAL'] = 1
            df.at[df.index[i], 'Buy_Exit_Price'] = current_price
            latest_buy_index = None
            
        elif trade_type == 'hold' and latest_hold_index is not None:
            df.at[df.index[i], 'HOLD_EXIT_SIGNAL'] = 1
            df.at[df.index[i], 'Hold_Exit_Price'] = current_price
            latest_hold_index = None
            
        elif trade_type == 'sell' and latest_sell_index is not None:
            df.at[df.index[i], 'SELL_EXIT_SIGNAL'] = 1
            df.at[df.index[i], 'Sell_Exit_Price'] = current_price
            latest_sell_index = None

    def start_new_trade(i, trade_type, current_price):
        """Start a new trade of the specified type."""
        nonlocal latest_buy_index, latest_hold_index, latest_sell_index, peak_price_since_buy, number_buy_signal_since_golden_cross
        
        if trade_type == 'buy' and latest_buy_index is None:
            df.at[df.index[i], 'BUY_SIGNAL'] = 1
            df.at[df.index[i], 'Buy_Price'] = current_price
            latest_buy_index = i
            peak_price_since_buy = current_price
            df.at[df.index[i], 'Buy_Stop_Loss_Price'] = current_price * (1 - drawdown_threshold / 100)
            if number_buy_signal_since_golden_cross is not None:
                number_buy_signal_since_golden_cross += 1
                df.at[df.index[i], 'Buy_Signal_Count'] = number_buy_signal_since_golden_cross
            return True
            
        elif trade_type == 'hold' and latest_hold_index is None:
            df.at[df.index[i], 'HOLD_SIGNAL'] = 1
            df.at[df.index[i], 'Hold_Price'] = current_price
            latest_hold_index = i
            return True
            
        elif trade_type == 'sell' and latest_sell_index is None:
            df.at[df.index[i], 'SELL_SIGNAL'] = 1
            df.at[df.index[i], 'Sell_Price'] = current_price
            latest_sell_index = i
            return True
            
        return False

    def can_generate_buy_signal(i):
        """Check if we can generate a buy signal based on golden cross conditions."""
        nonlocal number_buy_signal_since_golden_cross, last_golden_cross_index, is_in_golden_cross
        
        # Can only generate signal if we're in golden cross period and haven't generated a signal since last golden cross
        return (is_in_golden_cross and
                last_golden_cross_index is not None and 
                i > last_golden_cross_index and 
                number_buy_signal_since_golden_cross <= 1)



    def days_ADX_goes_down(i):
        """
        Check if ADX has been consistently decreasing for the required number of days.
        Only checks for consecutive down days.
        
        Parameters:
        i (int): Current index in the dataframe
        
        Returns:
        bool: True if ADX has been going down for required_ADX_down_days, False otherwise
        """
        if i < required_ADX_down_days:
            return False
            
        # Count consecutive negative days up to current position
        consecutive_down_days = 0
        
        # Start from current day and look backward
        for j in range(i, i - required_ADX_down_days - 1, -1):
            if j > 0 and df['ADX'].iloc[j] < df['ADX'].iloc[j-1]:
                consecutive_down_days += 1
            else:
                break
                
        return consecutive_down_days >= required_ADX_down_days
    

    def days_J_goes_up(i):
        """
        Check if Monthly_J has been consistently increasing for the required number of days.
        Only checks for consecutive up days without looking for flip points.
        
        Parameters:
        i (int): Current index in the dataframe
        
        Returns:
        bool: True if J has been going up for required_up_days, False otherwise
        """
        if i < required_up_days:
            return False
            
        # Count consecutive positive days up to current position
        consecutive_up_days = 0
        
        # Start from current day and look backward
        for j in range(i, i - required_up_days - 1, -1):
            if j > 0 and df['Monthly_J'].iloc[j] > df['Monthly_J'].iloc[j-1]:
                consecutive_up_days += 1
            else:
                break
                
        return consecutive_up_days >= required_up_days


    def days_J_goes_up_since_flip(i):
        """
        Check for upward flipping point and count consecutive positive days.
        Returns True if finds a flip point and enough consecutive positive days after it.
        """
        if i < required_up_days + 2:
            return False
            
        # Look for flip point from i-(required_up_days+1) to i
        flip_idx = None
        for j in range(i - (required_up_days + 1), i + 1):
            if j > 0 and j < i:  # Ensure we have data for j-1 and j
                # Check for flip from negative to positive growth
                if (df['Monthly_J'].iloc[j-1] < df['Monthly_J'].iloc[j-2] and  # Previous growth was negative
                    df['Monthly_J'].iloc[j] > df['Monthly_J'].iloc[j-1]):      # Current growth is positive
                    flip_idx = j
                    break
                    
        if flip_idx is None:  # No flip point found
            return False
            
        # Count consecutive positive days since flip
        positive_days = 0
        for k in range(flip_idx, i + 1):
            if k > 0 and df['Monthly_J'].iloc[k] > df['Monthly_J'].iloc[k-1]:
                positive_days += 1
            else:
                break
                
        return positive_days >= required_up_days

    def days_J_goes_down_since_flip(i):
        """
        Check for flipping point in lookback period and count consecutive negative days.
        Returns True if finds a flip point and enough consecutive negative days after it.
        """
        if i < required_down_days + 2:
            return False
            
        # Look for flip point from i-(required_down_days+1) to i
        flip_idx = None
        for j in range(i - (required_down_days + 1), i + 1):
            if j > 0 and j < i:  # Ensure we have data for j-1 and j
                # Check for flip from positive to negative growth
                if (df['Monthly_J'].iloc[j-1] > df['Monthly_J'].iloc[j-2] and  # Previous growth was positive
                    df['Monthly_J'].iloc[j] < df['Monthly_J'].iloc[j-1]):      # Current growth is negative
                    flip_idx = j
                    break
                    
        if flip_idx is None:  # No flip point found
            return False
            
        # Count consecutive negative days since flip
        negative_days = 0
        for k in range(flip_idx, i + 1):
            if k > 0 and df['Monthly_J'].iloc[k] < df['Monthly_J'].iloc[k-1]:
                negative_days += 1
            else:
                break
                
        return negative_days >= required_down_days

    # Iterate through the DataFrame
    for i in range(n_days, len(df)):
        current_price = df['Close'].iloc[i]
        
        # Update peak prices and drawdowns
        peak_price_lifetime = max(peak_price_lifetime, current_price)
        if latest_buy_index is not None:
            peak_price_since_buy = max(peak_price_since_buy, current_price)
            drawdown = (current_price - peak_price_since_buy) / peak_price_since_buy
            df.at[df.index[i], 'DrawDown_Since_Latest_Buy'] = drawdown
        
        drawdown_lifetime = (current_price - peak_price_lifetime) / peak_price_lifetime
        df.at[df.index[i], 'DrawDown_Life_Time'] = drawdown_lifetime
        
        # Check for golden cross and death cross
        if i > 1:
            # Check for cross changes
            # if df['Monthly_J'].iloc[i] >= df['Monthly_K'].iloc[i] and df['Monthly_J'].iloc[i-1] <= df['Monthly_K'].iloc[i-1] :
            if df['Monthly_J'].iloc[i] > df['Monthly_K'].iloc[i]:
                # Currently in golden cross state
                if not is_in_golden_cross:  # Just crossed over
                    is_in_golden_cross = True
                    last_golden_cross_index = i
                    number_buy_signal_since_golden_cross = 0
                    df.at[df.index[i], 'Golden_Cross'] = 1
                df.at[df.index[i], 'Cross_Status'] = is_in_golden_cross

            # if df['Monthly_J'].iloc[i] <= df['Monthly_K'].iloc[i] and df['Monthly_J'].iloc[i-1] >= df['Monthly_K'].iloc[i-1] :
            if df['Monthly_J'].iloc[i] < df['Monthly_K'].iloc[i] :
                # Currently in death cross state
                if is_in_golden_cross:  # Just crossed under
                    is_in_golden_cross = False
                    number_buy_signal_since_golden_cross = None
                    df.at[df.index[i], 'Death_Cross'] = 1
                df.at[df.index[i], 'Cross_Status'] = is_in_golden_cross

        # Check for buy conditions with different rules for golden/death cross
        if is_in_golden_cross:
            # During golden cross period - check all conditions including buy signal count
            # if (days_J_goes_up_since_flip(i) and 
            #     df['Monthly_D'].iloc[i] > df['Monthly_D'].iloc[i-1] and 
            #     can_generate_buy_signal(i)):
                

            if (days_J_goes_up(i) and 
                df['Monthly_D'].iloc[i] > df['Monthly_D'].iloc[i-1] and 
                can_generate_buy_signal(i)):
                end_existing_trade(i, current_price, 'sell')
                end_existing_trade(i, current_price, 'hold')
                start_new_trade(i, 'buy', current_price)
                continue
        else:
            # During death cross period - only check basic conditions
            # if (days_J_goes_up_since_flip(i) and 
            #     df['Monthly_D'].iloc[i] > df['Monthly_D'].iloc[i-1]):
                
            if (days_J_goes_up_since_flip(i) and
                days_ADX_goes_down(i)):
                # df['ADX'].iloc[i] <= df['ADX'].iloc[i-1] and
                # df['Monthly_J'].iloc[i] <= df['Monthly_K'].iloc[i] - 10):

                
                end_existing_trade(i, current_price, 'sell')
                end_existing_trade(i, current_price, 'hold')
                start_new_trade(i, 'buy', current_price)
                continue


        # During golden cross period - check for death cross happening
        if (df['Monthly_J'].iloc[i] <= df['Monthly_K'].iloc[i] and 
            df['Monthly_J'].iloc[i-1] >= df['Monthly_K'].iloc[i-1]):
            
            end_existing_trade(i, current_price, 'buy')
            end_existing_trade(i, current_price, 'hold')
            start_new_trade(i, 'sell', current_price)
            continue

        # if number_buy_signal_since_golden_cross == 1:
        #     # First buy signal - check for J going down
        #     if days_J_goes_down_since_flip(i):
        #         end_existing_trade(i, current_price, 'buy')
        #         end_existing_trade(i, current_price, 'hold')
        #         start_new_trade(i, 'sell', current_price)
        #         continue

        # # Check for sell conditions based on cross state
        # if is_in_golden_cross:
        #     # During golden cross period - check for death cross happening
        #     if (df['Monthly_J'].iloc[i] <= df['Monthly_K'].iloc[i] and 
        #         df['Monthly_J'].iloc[i-1] >= df['Monthly_K'].iloc[i-1]):
                
        #         end_existing_trade(i, current_price, 'buy')
        #         end_existing_trade(i, current_price, 'hold')
        #         start_new_trade(i, 'sell', current_price)
        #         continue
        # else:
        #     # During death cross period - check for J going down
        #     if days_J_goes_down_since_flip(i):
        #         end_existing_trade(i, current_price, 'buy')
        #         end_existing_trade(i, current_price, 'hold')
        #         start_new_trade(i, 'sell', current_price)
        #         continue

        # Check for exit conditions
        if latest_buy_index is not None and df.at[df.index[i], 'DrawDown_Since_Latest_Buy'] < -drawdown_threshold:
            price_just_meet_the_threshold = peak_price_since_buy * (1 - drawdown_threshold)
            end_existing_trade(i, price_just_meet_the_threshold, 'buy')

    return df

def generate_buy_sell_signals_early_detection_late_exit(df):
    """
    Generate buy and buy exit signals with early detection based on specified conditions.
    
    Arguments:
    df -- DataFrame containing columns 'Monthly_J', 'Monthly_K', 'Monthly_D', 'Close', etc.
    
    Returns:
    DataFrame with additional columns for buy/sell signals and performance tracking.
    """

    # Initialize columns with zero values
    df['BUY_SIGNAL'] = 0
    df['BUY_EXIT_SIGNAL'] = 0
    df['SELL_SIGNAL'] = 0
    df['SELL_EXIT_SIGNAL'] = 0
    df['Buy_Stop_Loss_Price'] = np.nan
    df['Cum_Profit_Since_Buy'] = np.nan
    df['Cum_Max_Profit_Since_Buy'] = np.nan
    df['Buy_Stop_Loss_Signal'] = 0
    df['DrawDown_Since_Latest_Buy'] = 0
    df['DrawDown_Life_Time'] = 0
    df['Buy_Exit_Price'] = np.nan

    drawdown_threshold = 1.0
    latest_buy_index = None
    peak_price_since_buy = None
    peak_price_lifetime = df['Close'].iloc[0]

    # Iterate over each row to generate signals
    for i in range(1, len(df)):
        current_price = df['Close'].iloc[i]
        
        # Track the peak price for lifetime and since the latest buy
        peak_price_lifetime = max(peak_price_lifetime, current_price)
        if latest_buy_index is not None:
            peak_price_since_buy = max(peak_price_since_buy, current_price)
        
        # Calculate DrawDown_Since_Latest_Buy
        if latest_buy_index is not None:
            drawdown = (current_price - peak_price_since_buy) / peak_price_since_buy
            price_just_meet_the_threshold = peak_price_since_buy * (1 - drawdown_threshold)  # e.g., 20% drawdown
            df.at[df.index[i], 'DrawDown_Since_Latest_Buy'] = drawdown
        else:
            df.at[df.index[i], 'DrawDown_Since_Latest_Buy'] = 0

        # Calculate DrawDown_Life_Time
        drawdown_lifetime = (current_price - peak_price_lifetime) / peak_price_lifetime
        df.at[df.index[i], 'DrawDown_Life_Time'] = drawdown_lifetime

        # Generate Buy Signal based on updated Monthly J, K, D conditions
        if latest_buy_index is None:
            if (df['Monthly_J'].iloc[i] > df['Monthly_J'].iloc[i - 1] and
                df['Monthly_J'].iloc[i] < df['Monthly_K'].iloc[i] and
                df['Monthly_J'].iloc[i]  <= 100 and
                df['Monthly_J'].iloc[i] < df['Monthly_D'].iloc[i]):
                df.at[df.index[i], 'BUY_SIGNAL'] = 1
                latest_buy_index = i
                peak_price_since_buy = current_price
                df.at[df.index[i], 'Buy_Stop_Loss_Price'] = current_price * (1 - drawdown_threshold / 100)
                
        # Generate Buy Exit Signal based on updated Monthly J, K, D conditions
        elif latest_buy_index is not None:
            if df.at[df.index[i], 'DrawDown_Since_Latest_Buy'] < -drawdown_threshold:
                df.at[df.index[i], 'BUY_EXIT_SIGNAL'] = 1
                df.at[df.index[i], 'Buy_Exit_Price'] = price_just_meet_the_threshold
                latest_buy_index = None
            elif df['Monthly_J'].iloc[i-1] > df['Monthly_K'].iloc[i-1] and df['Monthly_J'].iloc[i] < df['Monthly_K'].iloc[i]:
                df.at[df.index[i], 'BUY_EXIT_SIGNAL'] = 1
                df.at[df.index[i], 'Buy_Exit_Price'] = current_price
                latest_buy_index = None

    return df



def generate_buy_sell_signals_daily_kdj_detection(df):
    """
    Generate buy and buy exit signals with early detection based on daily KDJ conditions.
    
    Arguments:
    df -- DataFrame containing columns 'J', 'K', 'D', 'Close', etc.
    
    Returns:
    DataFrame with additional columns for buy/sell signals and performance tracking.
    """

    # Initialize columns with zero values
    df['BUY_SIGNAL'] = 0
    df['BUY_EXIT_SIGNAL'] = 0
    df['SELL_SIGNAL'] = 0
    df['SELL_EXIT_SIGNAL'] = 0
    df['Buy_Stop_Loss_Price'] = np.nan
    df['Cum_Profit_Since_Buy'] = np.nan
    df['Cum_Max_Profit_Since_Buy'] = np.nan
    df['Buy_Stop_Loss_Signal'] = 0
    df['DrawDown_Since_Latest_Buy'] = 0
    df['DrawDown_Life_Time'] = 0
    df['Buy_Exit_Price'] = np.nan

    drawdown_threshold = 1.0
    latest_buy_index = None
    peak_price_since_buy = None
    peak_price_lifetime = df['Close'].iloc[0]

    # Iterate over each row to generate signals
    for i in range(1, len(df)):
        current_price = df['Close'].iloc[i]
        
        # Track the peak price for lifetime and since the latest buy
        peak_price_lifetime = max(peak_price_lifetime, current_price)
        if latest_buy_index is not None:
            peak_price_since_buy = max(peak_price_since_buy, current_price)
        
        # Calculate DrawDown_Since_Latest_Buy
        if latest_buy_index is not None:
            drawdown = (current_price - peak_price_since_buy) / peak_price_since_buy
            price_just_meet_the_threshold = peak_price_since_buy * (1 - drawdown_threshold)  # e.g., 20% drawdown
            df.at[df.index[i], 'DrawDown_Since_Latest_Buy'] = drawdown
        else:
            df.at[df.index[i], 'DrawDown_Since_Latest_Buy'] = 0

        # Calculate DrawDown_Life_Time
        drawdown_lifetime = (current_price - peak_price_lifetime) / peak_price_lifetime
        df.at[df.index[i], 'DrawDown_Life_Time'] = drawdown_lifetime

        # Generate Buy Signal based on daily KDJ conditions
        if latest_buy_index is None:
            if (df['J'].iloc[i] > df['J'].iloc[i - 1] and
                df['J'].iloc[i] < df['K'].iloc[i] and
                df['J'].iloc[i] < df['D'].iloc[i] and
                df['J'].iloc[i] <= 30):
                df.at[df.index[i], 'BUY_SIGNAL'] = 1
                latest_buy_index = i
                peak_price_since_buy = current_price
                df.at[df.index[i], 'Buy_Stop_Loss_Price'] = current_price * (1 - drawdown_threshold / 100)
                
        # Generate Buy Exit Signal based on daily KDJ conditions
        elif latest_buy_index is not None:
            if df.at[df.index[i], 'DrawDown_Since_Latest_Buy'] < -drawdown_threshold:
                df.at[df.index[i], 'BUY_EXIT_SIGNAL'] = 1
                df.at[df.index[i], 'Buy_Exit_Price'] = price_just_meet_the_threshold
                latest_buy_index = None
            elif (df['J'].iloc[i] < df['J'].iloc[i - 1] and
                  df['J'].iloc[i] > df['K'].iloc[i] and
                  df['J'].iloc[i] > df['D'].iloc[i]):
                df.at[df.index[i], 'BUY_EXIT_SIGNAL'] = 1
                df.at[df.index[i], 'Buy_Exit_Price'] = current_price
                latest_buy_index = None

    return df



# Function to calculate ticker to SPY ratio with date alignment
def calculate_ratio(df, spy_df):
    # Align the data frames on the index (dates)
    df, spy_df = df.align(spy_df, join='left', axis=0)
    
    # Calculate the ratio
    df['SPY_Close'] = spy_df['Close']
    df['Price_Ratio'] = df['Close'] / df['SPY_Close']
    df['SPY_Close_minus_SMA_30'] = spy_df['Close_minus_SMA_30']
    
    return df




def generate_buy_sell_signals_by_gold_dead_cross_only_monthly_from_daily_return_And_RSI(df):
    # Initialize columns with zero values
    df['BUY_SIGNAL'] = 0
    df['BUY_EXIT_SIGNAL'] = 0
    df['SELL_SIGNAL'] = 0
    df['SELL_EXIT_SIGNAL'] = 0
    df['Buy_Stop_Loss_Price'] = np.nan
    df['Cum_Profit_Since_Buy'] = np.nan
    df['Cum_Max_Profit_Since_Buy'] = np.nan
    df['Buy_Stop_Loss_Signal'] = 0
    df['DrawDown_Since_Latest_Buy'] = 0
    df['DrawDown_Life_Time'] = 0
    df['Buy_Exit_Price'] = np.nan

    drawdown_threshold = 1.0
    
    monthly_kdj_condition = 0
    latest_buy_index = None
    peak_price_since_buy = None
    peak_price_lifetime = df['Close'].iloc[0]
    
    # Loop through the DataFrame to generate buy/sell signals based on the J and K line crosses
    for i in range(1, len(df)):
        current_price = df['Close'].iloc[i]
        
        # Calculate DrawDown_Since_Latest_Buy
        if latest_buy_index is not None:
            peak_price_since_buy = max(peak_price_since_buy, current_price)
            drawdown = (current_price - peak_price_since_buy) / peak_price_since_buy
            price_just_meet_the_threshold = peak_price_since_buy * (1 - drawdown_threshold)  # 20% drawdown
            df.at[df.index[i], 'DrawDown_Since_Latest_Buy'] = drawdown
        else:
            df.at[df.index[i], 'DrawDown_Since_Latest_Buy'] = 0
        
        # Calculate DrawDown_Life_Time
        peak_price_lifetime = max(peak_price_lifetime, current_price)
        drawdown_lifetime = (current_price - peak_price_lifetime) / peak_price_lifetime
        df.at[df.index[i], 'DrawDown_Life_Time'] = drawdown_lifetime
        
        # Generate Buy Signal when J crosses above K (Golden Cross) on monthly KDJ, then check daily KDJ
        if (df['Monthly_J'].iloc[i-1] <= df['Monthly_K'].iloc[i-1] and 
            df['Monthly_J'].iloc[i] >= df['Monthly_K'].iloc[i] and 
            df['Monthly_K_Slope'].iloc[i] > 0 and 
            df['Monthly_D_Slope'].iloc[i] > 0 and 
            df['Monthly_J'].iloc[i] <= 100 and
            df['SPY_Close_minus_SMA_30'].iloc[i] >= -10):

            df.at[df.index[i], 'BUY_SIGNAL'] = 1
            latest_buy_index = i
            peak_price_since_buy = current_price
            
        # Generate Buy Exit Signal when J crosses below K (Dead Cross) on daily KDJ or DrawDown_Since_Latest_Buy < -20% or RSI < 30
        if latest_buy_index is not None:
            if df.at[df.index[i], 'DrawDown_Since_Latest_Buy'] < -1 * drawdown_threshold:
                df.at[df.index[i], 'BUY_EXIT_SIGNAL'] = 1
                df.at[df.index[i], 'Buy_Exit_Price'] = price_just_meet_the_threshold
                latest_buy_index = None
            elif (df['Monthly_J'].iloc[i-1] > df['Monthly_K'].iloc[i-1] and 
                  df['Monthly_J'].iloc[i] < df['Monthly_K'].iloc[i]) or df['RSI'].iloc[i] < 30:
                df.at[df.index[i], 'BUY_EXIT_SIGNAL'] = 1
                df.at[df.index[i], 'Buy_Exit_Price'] = current_price
                latest_buy_index = None
    return df

def generate_buy_sell_signals_by_gold_dead_cross_only_monthly_from_daily_return_And_RSI_Forfront(df):
    # Initialize columns with zero values
    df['BUY_SIGNAL'] = 0
    df['BUY_EXIT_SIGNAL'] = 0
    df['SELL_SIGNAL'] = 0
    df['SELL_EXIT_SIGNAL'] = 0
    df['Buy_Stop_Loss_Price'] = np.nan
    df['Cum_Profit_Since_Buy'] = np.nan
    df['Cum_Max_Profit_Since_Buy'] = np.nan
    df['Buy_Stop_Loss_Signal'] = 0
    df['DrawDown_Since_Latest_Buy'] = 0
    df['DrawDown_Life_Time'] = 0
    df['Buy_Exit_Price'] = np.nan

    drawdown_threshold = 1.0
    
    monthly_kdj_condition = 0
    latest_buy_index = None
    peak_price_since_buy = None
    peak_price_lifetime = df['Close'].iloc[0]
    
    # Loop through the DataFrame to generate buy/sell signals based on the J and K line crosses
    for i in range(1, len(df)):
        current_price = df['Close'].iloc[i]
        
        # Calculate DrawDown_Since_Latest_Buy
        if latest_buy_index is not None:
            peak_price_since_buy = max(peak_price_since_buy, current_price)
            drawdown = (current_price - peak_price_since_buy) / peak_price_since_buy
            price_just_meet_the_threshold = peak_price_since_buy * (1 - drawdown_threshold)  # 20% drawdown
            df.at[df.index[i], 'DrawDown_Since_Latest_Buy'] = drawdown
        else:
            df.at[df.index[i], 'DrawDown_Since_Latest_Buy'] = 0
        
        # Calculate DrawDown_Life_Time
        peak_price_lifetime = max(peak_price_lifetime, current_price)
        drawdown_lifetime = (current_price - peak_price_lifetime) / peak_price_lifetime
        df.at[df.index[i], 'DrawDown_Life_Time'] = drawdown_lifetime
        
        # Generate Buy Signal with updated logic
        if latest_buy_index is None:
            if ((df['Monthly_J'].iloc[i-1] <= df['Monthly_K'].iloc[i-1] and 
                 df['Monthly_J'].iloc[i] >= df['Monthly_K'].iloc[i] and 
                 df['Monthly_K_Slope'].iloc[i] > 0 and 
                 df['Monthly_D_Slope'].iloc[i] > 0 and 
                 df['Monthly_J'].iloc[i] <= 100) or
                (df['Monthly_J'].iloc[i] < df['Monthly_K'].iloc[i] and 
                 df['Monthly_J_Slope'].iloc[i] > 0 and 
                 df['RSI'].iloc[i] >= 70)):

                df.at[df.index[i], 'BUY_SIGNAL'] = 1
                latest_buy_index = i
                peak_price_since_buy = current_price
            
        # Generate Buy Exit Signal when J crosses below K (Dead Cross) on daily KDJ or DrawDown_Since_Latest_Buy < -20% or RSI < 30
        if latest_buy_index is not None:
            if df.at[df.index[i], 'DrawDown_Since_Latest_Buy'] < -1 * drawdown_threshold:
                df.at[df.index[i], 'BUY_EXIT_SIGNAL'] = 1
                df.at[df.index[i], 'Buy_Exit_Price'] = price_just_meet_the_threshold
                latest_buy_index = None
            # elif (df['Monthly_J'].iloc[i-1] > df['Monthly_K'].iloc[i-1] and 
            #       df['Monthly_J'].iloc[i] < df['Monthly_K'].iloc[i]) or df['RSI'].iloc[i] < 30:
                
            elif (df['Monthly_J'].iloc[i-1] > df['Monthly_K'].iloc[i-1] and 
                  df['Monthly_J'].iloc[i] < df['Monthly_K'].iloc[i]):
                df.at[df.index[i], 'BUY_EXIT_SIGNAL'] = 1
                df.at[df.index[i], 'Buy_Exit_Price'] = current_price
                latest_buy_index = None
    return df


def generate_buy_sell_signals_by_KDJ_only_monthly_from_daily_return_J_over_KD_more_than_10_drawdown_protection_profit_taking(df):
    # Initialize columns with zero values
    df['BUY_SIGNAL'] = 0
    df['BUY_EXIT_SIGNAL'] = 0
    df['SELL_SIGNAL'] = 0
    df['SELL_EXIT_SIGNAL'] = 0
    df['Buy_Stop_Loss_Price'] = np.nan
    df['Cum_Profit_Since_Buy'] = np.nan
    df['Cum_Max_Profit_Since_Buy'] = np.nan
    df['Buy_Stop_Loss_Signal'] = 0
    df['DrawDown_Since_Latest_Buy'] = 0
    df['DrawDown_Life_Time'] = 0
    df['Buy_Exit_Price'] = np.nan
    df['Max_Cum_Return_Since_Latest_Buy'] = 0  # New column

    drawdown_threshold = 0.8
    principal_protection_threshold = 0.2
    profit_taking_threshold = 0.01  # New threshold for profit taking
    
    latest_buy_index = None
    peak_price_since_buy = None
    peak_price_lifetime = df['Close'].iloc[0]
    buy_price = None
    principal_drawdown = None
    max_cum_return = 0  # Initialize max cumulative return
    
    for i in range(1, len(df)):
        current_price = df['Close'].iloc[i]
        
        if latest_buy_index is not None:
            peak_price_since_buy = max(peak_price_since_buy, current_price)
            drawdown = (current_price - peak_price_since_buy) / peak_price_since_buy
            price_just_meet_the_threshold = peak_price_since_buy * (1 - drawdown_threshold)
            df.at[df.index[i], 'DrawDown_Since_Latest_Buy'] = drawdown
            
            principal_drawdown = (current_price - buy_price) / buy_price
            cum_return = (current_price - buy_price) / buy_price
            max_cum_return = max(max_cum_return, cum_return)
            df.at[df.index[i], 'Max_Cum_Return_Since_Latest_Buy'] = max_cum_return
        else:
            df.at[df.index[i], 'DrawDown_Since_Latest_Buy'] = 0
            principal_drawdown = 0
            max_cum_return = 0
            cum_return = 0  # Initialize cum_return when there's no active buy position
        
        peak_price_lifetime = max(peak_price_lifetime, current_price)
        drawdown_lifetime = (current_price - peak_price_lifetime) / peak_price_lifetime
        df.at[df.index[i], 'DrawDown_Life_Time'] = drawdown_lifetime
        
        current_j_k_diff = df['Monthly_J'].iloc[i] - df['Monthly_K'].iloc[i]
        previous_j_k_diff = df['Monthly_J'].iloc[i-1] - df['Monthly_K'].iloc[i-1]
        
        if latest_buy_index is not None:
            if df.at[df.index[i], 'DrawDown_Since_Latest_Buy'] < -1 * drawdown_threshold:
                df.at[df.index[i], 'BUY_EXIT_SIGNAL'] = 1
                df.at[df.index[i], 'Buy_Exit_Price'] = price_just_meet_the_threshold
                latest_buy_index = None
            elif principal_drawdown < -1 * principal_protection_threshold:
                df.at[df.index[i], 'BUY_EXIT_SIGNAL'] = 1
                df.at[df.index[i], 'Buy_Exit_Price'] = buy_price * (1 - principal_protection_threshold)
                latest_buy_index = None
            elif df['Monthly_J'].iloc[i-1] > df['Monthly_K'].iloc[i-1] and df['Monthly_J'].iloc[i] < df['Monthly_K'].iloc[i]:
                df.at[df.index[i], 'BUY_EXIT_SIGNAL'] = 1
                df.at[df.index[i], 'Buy_Exit_Price'] = current_price
                latest_buy_index = None
            elif cum_return <= profit_taking_threshold * max_cum_return:
                df.at[df.index[i], 'BUY_EXIT_SIGNAL'] = 1
                df.at[df.index[i], 'Buy_Exit_Price'] = buy_price * (1 + profit_taking_threshold * max_cum_return)
                latest_buy_index = None

        if latest_buy_index is None:
            if current_j_k_diff > 10 and previous_j_k_diff <= 10 and df['Monthly_J_Slope'].iloc[i] > 0:
                df.at[df.index[i], 'BUY_SIGNAL'] = 1
                latest_buy_index = i
                buy_price = current_price
                peak_price_since_buy = current_price
                max_cum_return = 0
                cum_return = 0  # Reset cum_return when a new buy position is opened
    return df

def generate_buy_sell_signals_by_KDJ_only_monthly_from_daily_return_J_over_KD_more_than_10(df):
    # Initialize columns with zero values
    df['BUY_SIGNAL'] = 0
    df['BUY_EXIT_SIGNAL'] = 0
    df['Buy_Exit_Price'] = np.nan
    df['Buy_Stop_Loss_Signal'] = 0
    df['Buy_Stop_Loss_Price'] = np.nan

    
    df['SELL_SIGNAL'] = 0
    df['SELL_EXIT_SIGNAL'] = 0

    latest_buy_index = None
    
    for i in range(1, len(df)):
        current_price = df['Close'].iloc[i]
        
        current_j_k_diff = df['Monthly_J'].iloc[i] - df['Monthly_K'].iloc[i]
        previous_j_k_diff = df['Monthly_J'].iloc[i-1] - df['Monthly_K'].iloc[i-1]
        
        # Generate Buy Signal
        if latest_buy_index is None:
            if current_j_k_diff > 10 and previous_j_k_diff <= 10 and df['Monthly_J_Slope'].iloc[i] > 0:
                df.at[df.index[i], 'BUY_SIGNAL'] = 1
                latest_buy_index = i

        # Generate Buy Exit Signal
        if latest_buy_index is not None:
            if df['Monthly_J'].iloc[i-1] > df['Monthly_K'].iloc[i-1] and df['Monthly_J'].iloc[i] < df['Monthly_K'].iloc[i]:
                df.at[df.index[i], 'BUY_EXIT_SIGNAL'] = 1
                df.at[df.index[i], 'Buy_Exit_Price'] = current_price
                latest_buy_index = None

    return df



def generate_buy_sell_signals_by_gold_dead_cross_both_daily_monthly(df):
    # Initialize columns with zero values
    df['BUY_SIGNAL'] = 0
    df['BUY_EXIT_SIGNAL'] = 0
    df['Buy_Exit_Price'] = 0
    df['SELL_SIGNAL'] = 0
    df['SELL_EXIT_SIGNAL'] = 0
    df['Buy_Stop_Loss_Price'] = np.nan
    df['Cum_Profit_Since_Buy'] = np.nan
    df['Cum_Max_Profit_Since_Buy'] = np.nan
    df['Buy_Stop_Loss_Signal'] = 0
    
    monthly_kdj_condition = 0
    
    # Loop through the DataFrame to generate buy/sell signals based on the J and K line crosses
    for i in range(1, len(df)):
        # Check for Monthly Golden Cross
        if (df['Monthly_J'].iloc[i-1] <= df['Monthly_K'].iloc[i-1] and 
            df['Monthly_J'].iloc[i] >= df['Monthly_K'].iloc[i] and 
            df['Monthly_K_Slope'].iloc[i] > 0 and 
            df['Monthly_D_Slope'].iloc[i] > 0 and 
            df['Monthly_J'].iloc[i] <= 100):
            monthly_kdj_condition = 1
        
        # Check for Monthly Dead Cross
        elif df['Monthly_J'].iloc[i-1] > df['Monthly_K'].iloc[i-1] and df['Monthly_J'].iloc[i] < df['Monthly_K'].iloc[i]:
            monthly_kdj_condition = 0
            df.at[df.index[i], 'BUY_EXIT_SIGNAL'] = 1
            df.at[df.index[i], 'Buy_Exit_Price'] = df['Close'].iloc[i]

        
        # If monthly_kdj_condition is 1, check for daily signals
        if monthly_kdj_condition == 1:
            # Generate Buy Signal when J crosses above K (Golden Cross) on daily KDJ
            if (df['J'].iloc[i-1] <= df['K'].iloc[i-1] and 
                df['J'].iloc[i] >= df['K'].iloc[i] and 
                df['K_Slope'].iloc[i] > 0 and 
                df['D_Slope'].iloc[i] > 0 and 
                df['J'].iloc[i] <= 100):
                df.at[df.index[i], 'BUY_SIGNAL'] = 1
            
            # Generate Buy Exit Signal when J crosses below K (Dead Cross) on daily KDJ
            elif df['J'].iloc[i-1] > df['K'].iloc[i-1] and df['J'].iloc[i] < df['K'].iloc[i]:
                df.at[df.index[i], 'BUY_EXIT_SIGNAL'] = 1
                df.at[df.index[i], 'Buy_Exit_Price'] = df['Close'].iloc[i]
    
    return df



def calculate_cumulative_returns_by_ADX(df, initial_capital=100, strategy='LS'):
    df['Portfolio_Value'] = initial_capital
    df['Cash'] = initial_capital
    df['Position'] = 0  # Tracks the number of shares held (positive for long, negative for short)
    df['Cumulative_Returns'] = 0
    df['Strategy_Daily_Return'] = np.nan  # Initialize the new column

    for i in range(1, len(df)):
        # Carry forward the position, cash, and portfolio value from the previous day
        previous_position = df['Position'].iloc[i-1]
        previous_cash = df['Cash'].iloc[i-1]
        previous_close = df['Close'].iloc[i-1]
        previous_portfolio_value = df['Portfolio_Value'].iloc[i-1]
        current_close = df['Close'].iloc[i]

        df.at[df.index[i], 'Position'] = previous_position
        df.at[df.index[i], 'Cash'] = previous_cash
        df.at[df.index[i], 'Portfolio_Value'] = previous_portfolio_value

        # Calculate Strategy_Daily_Return
        if previous_position > 0:
            df.at[df.index[i], 'Strategy_Daily_Return'] = (current_close - previous_close) / previous_close
        elif previous_position < 0:
            df.at[df.index[i], 'Strategy_Daily_Return'] = (previous_close - current_close) / previous_close

        # Long Only Strategy
        if strategy == 'LO':
            # If a buy signal is seen
            if df['BUY_SIGNAL'].iloc[i] == 1 and previous_position <= 0:
                shares_to_buy = previous_portfolio_value / current_close
                df.at[df.index[i], 'Position'] = shares_to_buy
                df.at[df.index[i], 'Cash'] = 0

            # If a buy exit signal is seen and previous position is positive
            elif df['BUY_EXIT_SIGNAL'].iloc[i] == 1 and previous_position > 0:
                df.at[df.index[i], 'Cash'] = previous_cash + previous_position * df['Close'].iloc[i]
                df.at[df.index[i], 'Position'] = 0

            # If Buy Stop Loss Signal is seen and previous position is positive
            elif df['Buy_Stop_Loss_Signal'].iloc[i] == 1 and previous_position > 0:
                df.at[df.index[i], 'Cash'] = previous_cash + previous_position * df['Buy_Stop_Loss_Price'].iloc[i-1]
                df.at[df.index[i], 'Position'] = 0

        # Short Only Strategy
        elif strategy == 'SO':
            # If a sell signal is seen
            if df['SELL_SIGNAL'].iloc[i] == 1 and previous_position >= 0:
                shares_to_sell = -previous_portfolio_value / current_close
                df.at[df.index[i], 'Position'] = shares_to_sell
                df.at[df.index[i], 'Cash'] = 2 * previous_portfolio_value

            # If a sell exit signal is seen and previous position is negative
            elif df['SELL_EXIT_SIGNAL'].iloc[i] == 1 and previous_position < 0:
                df.at[df.index[i], 'Cash'] = previous_cash + previous_position * df['Close'].iloc[i]
                df.at[df.index[i], 'Position'] = 0

        # Long Short Strategy (Default)
        elif strategy == 'LS':
            # If a buy signal is seen
            if df['BUY_SIGNAL'].iloc[i] == 1 and previous_position <= 0:
                shares_to_buy = previous_portfolio_value / current_close
                df.at[df.index[i], 'Position'] = shares_to_buy
                df.at[df.index[i], 'Cash'] = 0

            # If a sell signal is seen
            elif df['SELL_SIGNAL'].iloc[i] == 1 and previous_position >= 0:
                shares_to_sell = -previous_portfolio_value / current_close
                df.at[df.index[i], 'Position'] = shares_to_sell
                df.at[df.index[i], 'Cash'] = 2 * previous_portfolio_value

            # If a buy exit signal is seen and previous position is positive
            elif df['BUY_EXIT_SIGNAL'].iloc[i] == 1 and previous_position > 0:
                df.at[df.index[i], 'Cash'] = previous_cash + previous_position * df['Close'].iloc[i]
                df.at[df.index[i], 'Position'] = 0

            # If Buy Stop Loss Signal is seen and previous position is positive
            elif df['Buy_Stop_Loss_Signal'].iloc[i] == 1 and previous_position > 0:
                df.at[df.index[i], 'Cash'] = previous_cash + previous_position * df['Buy_Stop_Loss_Price'].iloc[i-1]
                df.at[df.index[i], 'Position'] = 0

            # If a sell exit signal is seen and previous position is negative
            elif df['SELL_EXIT_SIGNAL'].iloc[i] == 1 and previous_position < 0:
                df.at[df.index[i], 'Cash'] = previous_cash + previous_position * df['Close'].iloc[i]
                df.at[df.index[i], 'Position'] = 0

        # Update the portfolio value
        df.at[df.index[i], 'Portfolio_Value'] = df['Cash'].iloc[i] + df['Position'].iloc[i] * df['Close'].iloc[i]

    # Calculate cumulative returns
    df['Cumulative_Returns'] = df['Portfolio_Value'] / initial_capital - 1

    return df

def plot_chart_by_ADX(df, trades_df, ticker='AAPL'):
    if df.empty:
        print(f"No data available to plot for {ticker}. The DataFrame is empty.")
        return

    buy_signals = df[df['BUY_SIGNAL'] == 1].index
    hold_signals = df[df['HOLD_SIGNAL'] == 1].index
    sell_signals = df[df['SELL_SIGNAL'] == 1].index

    # Calculate padding
    date_range = df.index[-1] - df.index[0]
    padding = pd.Timedelta(days=date_range.days * 0.02)  # 2% padding on each side

    stage_colors = {'Buy': '#00ff00', 'Hold': '#00ffff', 'Sell': '#ff4444'}

    fig = make_subplots(rows=14, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.01,
                        subplot_titles=(
                            f'{ticker} Price Chart', 'Volume Chart', 'Cumulative Returns', 'Drawdowns',
                            'Monthly KDJ Chart', 'MFI Chart', 'ADX Chart', 'MACD Chart', 'KDJ Chart', 'RSI Chart',
                            'Awesome Oscillator', 'SPY Close - SMA 30', 'Price Ratio Chart', 'Trade Scatter Plot'))

    # Price Chart (row 1) - with logarithmic scale
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'], name='Candlesticks',
        increasing_line_color='#00ff00', decreasing_line_color='#ff4444'
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=df['SMA_30'],
        mode='lines', name='SMA 30',
        line=dict(color='#00ff88')
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=df['Donchian_High'],
        mode='lines', name='Donchian High',
        line=dict(color='#88cccc', width=1, dash='dot')
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=df['Donchian_Low'],
        mode='lines', name='Donchian Low',
        line=dict(color='#cc8888', width=1, dash='dot')
    ), row=1, col=1)

    # Buy Signals
    fig.add_trace(go.Scatter(
        x=buy_signals,
        y=df.loc[buy_signals]['Close'],
        mode='markers+text',
        name='Buy Signals',
        text=['BUY' for _ in buy_signals],
        textposition=['bottom center' for _ in buy_signals],
        textfont=dict(
            color='#333333',  # Darker grey
            size=8
        ),
        marker=dict(
            color='rgb(0, 255, 0)',  # Bright green
            size=20,
            symbol='arrow-up',
            line=dict(width=0)
        ),
        showlegend=True
    ), row=1, col=1)

    # Sell Signals
    fig.add_trace(go.Scatter(
        x=sell_signals,
        y=df.loc[sell_signals]['Close'],
        mode='markers+text',
        name='Sell Signals',
        text=['SELL' for _ in sell_signals],
        textposition=['top center' for _ in sell_signals],
        textfont=dict(
            color='white',
            size=8
        ),
        marker=dict(
            color='rgb(255, 0, 0)',  # Red
            size=20,
            symbol='arrow-down',
            line=dict(width=0)
        ),
        showlegend=True
    ), row=1, col=1)

    # Hold Signals - now using arrow-up like Buy signals
    fig.add_trace(go.Scatter(
        x=hold_signals,
        y=df.loc[hold_signals]['Low'] - (df.loc[hold_signals]['Low'] * 0.002),  # Increased offset
        mode='markers+text',
        name='Hold Signals',
        text=['HOLD' for _ in hold_signals],
        textposition=['bottom center' for _ in hold_signals],
        textfont=dict(
            color='white',
            size=8
        ),
        marker=dict(
            color='rgb(255, 165, 0)',  # Orange
            size=20,
            symbol='arrow-up',
            line=dict(width=0)
        ),
        showlegend=True
    ), row=1, col=1)

    # Volume Chart (row 2)
    colors = ['#00ff00' if df['Close'][i] > df['Open'][i] else '#ff4444' for i in range(len(df))]
    fig.add_trace(go.Bar(
        x=df.index, y=df['Volume'],
        name='Volume', marker_color=colors
    ), row=2, col=1)

    # Cumulative Returns (row 3) - with logarithmic scale
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Strategy_Cumulative_Returns'],
        mode='lines', name='Strategy Cumulative Returns',
        line=dict(color='#0088ff')
    ), row=3, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=df['Security_Cumulative_Returns'],
        mode='lines', name='Security Cumulative Returns',
        line=dict(color='#ff8800')
    ), row=3, col=1)

    # Drawdowns (row 4)
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Strategy_Drawdown'],
        mode='lines',
        name='Strategy Drawdown',
        line=dict(color='#0088ff', width=2)
    ), row=4, col=1)

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Security_Drawdown'],
        fill='tozeroy',
        name='Security Drawdown',
        line=dict(color='rgba(255,136,0,0.3)')
    ), row=4, col=1)

    # Monthly KDJ (row 5)
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Monthly_K'],
        mode='lines', name='Monthly K',
        line=dict(color='#00ffff')
    ), row=5, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=df['Monthly_D'],
        mode='lines', name='Monthly D',
        line=dict(color='#ffaa00')
    ), row=5, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=df['Monthly_J'],
        mode='lines', name='Monthly J',
        line=dict(color='#00ff88')
    ), row=5, col=1)

    # MFI Chart (row 6)
    fig.add_trace(go.Scatter(
        x=df.index, y=df['MFI'],
        mode='lines', name='MFI',
        line=dict(color='#ff00ff')
    ), row=6, col=1)

    # Add MFI zones with adjusted colors
    for level in [20, 40, 60, 80]:
        fig.add_shape(
            type="line", x0=df.index[0], x1=df.index[-1],
            y0=level, y1=level,
            line=dict(
                color="#888888" if level in [40, 60] else "#ff4444" if level == 80 else "#00ff00",
                width=1, dash="dash"
            ),
            row=6, col=1
        )

    fig.add_hrect(y0=80, y1=100, fillcolor="rgba(255,68,68,0.1)", row=6, col=1)
    fig.add_hrect(y0=0, y1=20, fillcolor="rgba(0,255,0,0.1)", row=6, col=1)
    fig.add_hrect(y0=40, y1=60, fillcolor="rgba(136,136,136,0.1)", row=6, col=1)

    # ADX Chart (row 7)
    fig.add_trace(go.Scatter(
        x=df.index, y=df['ADX'],
        mode='lines', name='ADX',
        line=dict(color='#ffaa00')
    ), row=7, col=1)

    # MACD Chart (row 8)
    fig.add_trace(go.Scatter(
        x=df.index, y=df['MACD'],
        mode='lines', name='MACD',
        line=dict(color='#00ffff')
    ), row=8, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=df['MACD_Signal'],
        mode='lines', name='MACD Signal',
        line=dict(color='#ffaa00')
    ), row=8, col=1)

    # KDJ Chart (row 9)
    fig.add_trace(go.Scatter(
        x=df.index, y=df['K'],
        mode='lines', name='K',
        line=dict(color='#00ffff')
    ), row=9, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=df['D'],
        mode='lines', name='D',
        line=dict(color='#ffaa00')
    ), row=9, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=df['J'],
        mode='lines', name='J',
        line=dict(color='#00ff88')
    ), row=9, col=1)

    # RSI Chart (row 10)
    fig.add_trace(go.Scatter(
        x=df.index, y=df['RSI'],
        mode='lines', name='RSI',
        line=dict(color='#ff00ff')
    ), row=10, col=1)

    # Awesome Oscillator (row 11)
    ao_colors = ['#00ff00' if slope > 0 else '#ff4444' for slope in df['AO_Slope']]
    fig.add_trace(go.Bar(
        x=df.index, y=df['AO'],
        name='Awesome Oscillator',
        marker_color=ao_colors
    ), row=11, col=1)

    # SPY Close - SMA 30 (row 12)
    fig.add_trace(go.Scatter(
        x=df.index, y=df['SPY_Close_minus_SMA_30'],
        mode='lines', name='SPY Close - SPY SMA30',
        line=dict(color='#00ffff')
    ), row=12, col=1)

    # Price Ratio Chart (row 13)
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Price_Ratio'],
        mode='lines', name='Price Ratio',
        line=dict(color='#ff00ff')
    ), row=13, col=1)

    # Trade Scatter Plot (row 14)
    if not trades_df.empty:
        for stage in trades_df['Stage'].unique():
            stage_data = trades_df[trades_df['Stage'] == stage]
            fig.add_trace(go.Scatter(
                x=stage_data['Max Drawdown'],
                y=stage_data['Return'],
                mode='markers',
                name=f'{stage} Trades',
                marker=dict(size=10, color=stage_colors.get(stage, '#888888')),
                text=[f"Entry: {row['Entry Date']}, Exit: {row['Exit Date']}" 
                      for _, row in stage_data.iterrows()]
            ), row=14, col=1)

    # Update x-axes ranges with padding for time series plots
    for i in range(1, 14):
        fig.update_xaxes(
            range=[df.index[0] - padding, df.index[-1] + padding],
            row=i, col=1,
            gridcolor='#333333',
            showgrid=True
        )

    # Remove padding for scatter plot
    if not trades_df.empty:
        fig.update_xaxes(range=None, row=14, col=1)

    # Set logarithmic scale for price chart (row 1)
    fig.update_yaxes(
        type="log",
        title_text="Price (log scale)",
        row=1, col=1
    )

    # Set logarithmic scale for cumulative returns (row 3)
    fig.update_yaxes(
        type="log",
        title_text="Cumulative Returns (log scale)",
        row=3, col=1
    )

    # Update layout with dark theme
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#111111',
        plot_bgcolor='#111111',
        title={
            'text': f'{ticker} Buy and Sell Signals',
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(color='#ffffff')
        },
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="right",
            x=1.2,
            font=dict(color='#ffffff'),
            bgcolor='rgba(0,0,0,0.5)'
        ),
        xaxis_rangeslider_visible=False,
        height=2000,
        margin=dict(t=100, b=50),
        uniformtext=dict(minsize=8, mode='hide'),
    )

    # Update y-axes with dark theme styling
    for i in range(1, 15):
        if i not in [1, 3]:  # Skip rows 1 and 3 which already have log scale set
            fig.update_yaxes(
                gridcolor='#333333',
                showgrid=True,
                title_standoff=5,
                title_font=dict(color='#ffffff'),
                tickfont=dict(color='#ffffff'),
                row=i,
                col=1
            )

    # Update subplot titles color
    for annotation in fig.layout.annotations:
        annotation.font.color = '#ffffff'

    fig.show()


def plot_aggregated_portfolio_value(strategy_daily_returns, df_strategy_cumulative_returns, df_security_cumulative_returns, active_metrics, sheet_name='Portfolio', security_daily_returns=None, combined_trades_df=None):
    # Find the first non-zero strategy return
    first_active_date = strategy_daily_returns[
        strategy_daily_returns['Index_Strategy_Daily_Returns'] != 0
    ].index.min()

    fig = make_subplots(
        rows=7, cols=1,
        shared_xaxes=False,
        vertical_spacing=0.08,
        subplot_titles=(
            f'Strategy & Security Returns (since {first_active_date.strftime("%Y-%m-%d")}) - {sheet_name}',
            'Strategy & Security Drawdowns',
            'Active Max Drawdown vs Active Cumulative Returns',
            'Strategy Cumulative Returns Distribution Histogram', 
            'Security Cumulative Returns Distribution Histogram',
            'Trades Return Distribution',
            'Trade Return vs Max Drawdown'
        ),
        specs=[
            [{"secondary_y": True}],  # First subplot with secondary y-axis
            [{}],  # Second subplot for drawdowns
            [{}], 
            [{}], 
            [{}],
            [{}],
            [{}]
        ]
    )

    # Add trace for the Index Strategy Cumulative Returns (blue line)
    fig.add_trace(go.Scatter(
        x=strategy_daily_returns.index,
        y=strategy_daily_returns['Index_Strategy_Cum_Returns'],
        mode='lines',
        name='Strategy Cumulative Returns',
        line=dict(color='blue', width=2)
    ), row=1, col=1, secondary_y=False)

    # Add trace for Security Cumulative Returns (green line)
    if security_daily_returns is not None:
        fig.add_trace(go.Scatter(
            x=security_daily_returns.index,
            y=security_daily_returns['Index_Security_Cum_Returns'],
            mode='lines',
            name='Security Cumulative Returns',
            line=dict(color='green', width=2)
        ), row=1, col=1, secondary_y=False)

        # Add active returns as filled area
        fig.add_trace(go.Scatter(
            x=strategy_daily_returns.index,
            y=strategy_daily_returns['Strategy_Active_Cum_Returns'],
            mode='lines',
            name='Active Returns',
            line=dict(color='purple', width=0),
            fill='tozeroy',
            fillcolor='rgba(230, 230, 250, 0.5)' if strategy_daily_returns['Strategy_Active_Cum_Returns'].iloc[-1] >= 0 else 'rgba(147, 112, 219, 0.3)'
        ), row=1, col=1, secondary_y=False)

    # Add Number of Active Stocks trace on secondary y-axis
    fig.add_trace(go.Scatter(
        x=strategy_daily_returns.index,
        y=strategy_daily_returns['Number_Active_Stocks'],
        mode='lines',
        name='Number Active Stocks',
        line=dict(color='gray', width=1),
        yaxis='y2'
    ), row=1, col=1, secondary_y=True)

    # Add drawdown traces on second subplot
    fig.add_trace(go.Scatter(
        x=strategy_daily_returns.index,
        y=strategy_daily_returns['Strategy_Drawdown'],
        mode='lines',
        name='Strategy Drawdown',
        line=dict(color='blue', width=1.5),
        fill='tozeroy',
        fillcolor='rgba(0,0,255,0.1)'
    ), row=2, col=1)

    if security_daily_returns is not None:
        fig.add_trace(go.Scatter(
            x=security_daily_returns.index,
            y=security_daily_returns['Security_Drawdown'],
            mode='lines',
            name='Security Drawdown',
            line=dict(color='green', width=1.5),
            fill='tozeroy',
            fillcolor='rgba(0,255,0,0.1)'
        ), row=2, col=1)

    # Scatter plot of Active Max Drawdown vs. Active Cumulative Returns
    fig.add_trace(go.Scatter(
        x=active_metrics['Active_Max_Drawdown'],
        y=active_metrics['Active_Cumulative_Returns'],
        mode='markers+text',
        text=active_metrics['Ticker'],
        textposition='top center',
        marker=dict(
            color='blue',
            size=10,
            line=dict(width=2, color='DarkSlateGrey')
        ),
        name='Active Metrics'
    ), row=3, col=1)

    # Extract the last row values from df_strategy_cumulative_returns
    final_strategy_cumulative_returns = df_strategy_cumulative_returns.iloc[-1].dropna()
    strategy_bin_edges = list(np.arange(-1, 5.2, 0.2))
    strategy_bin_edges += [final_strategy_cumulative_returns.max()]

    # Plot the histogram of the final strategy cumulative returns
    fig.add_trace(go.Histogram(
        x=final_strategy_cumulative_returns,
        name='Strategy Cumulative Returns Distribution',
        marker=dict(color='orange'),
        xbins=dict(
            start=strategy_bin_edges[0],
            end=strategy_bin_edges[-1],
            size=0.2
        ),
        autobinx=False
    ), row=4, col=1)

    # Extract the last row values from df_security_cumulative_returns
    final_security_cumulative_returns = df_security_cumulative_returns.iloc[-1].dropna()
    security_bin_edges = list(np.arange(-1, 5.2, 0.2))
    security_bin_edges += [final_security_cumulative_returns.max()]

    # Plot the histogram of the final security cumulative returns
    fig.add_trace(go.Histogram(
        x=final_security_cumulative_returns,
        name='Security Cumulative Returns Distribution',
        marker=dict(color='purple'),
        xbins=dict(
            start=security_bin_edges[0],
            end=security_bin_edges[-1],
            size=0.2
        ),
        autobinx=False
    ), row=5, col=1)

    # Plot trades distribution and scatter
    if combined_trades_df is not None:
        # Add trade returns histogram
        fig.add_trace(go.Histogram(
            x=combined_trades_df['Return'],
            name='Trade Return Distribution',
            marker=dict(color='green'),
            nbinsx=50
        ), row=6, col=1)

        # Add density heatmap for trade concentration
        fig.add_trace(go.Histogram2dContour(
            x=combined_trades_df['Max Drawdown'],
            y=combined_trades_df['Return'],
            colorscale='Viridis',
            showscale=False,
            opacity=0.3,
            name='Trade Density'
        ), row=7, col=1)

        # Add scatter plots with improved visibility
        stage_colors = {'Buy': 'green', 'Hold': 'blue', 'Sell': 'red'}
        for stage in combined_trades_df['Stage'].unique():
            stage_data = combined_trades_df[combined_trades_df['Stage'] == stage]
            
            # Add scatter points
            fig.add_trace(go.Scatter(
                x=stage_data['Max Drawdown'],
                y=stage_data['Return'],
                mode='markers',
                name=f'{stage} Trades',
                marker=dict(
                    size=6,
                    opacity=0.5,
                    color=stage_colors.get(stage, 'gray'),
                    symbol='circle'
                ),
                text=[f"Ticker: {row['Ticker']}<br>" +
                      f"Entry: {row['Entry Date'].strftime('%Y-%m-%d')}<br>" +
                      f"Exit: {row['Exit Date'].strftime('%Y-%m-%d')}<br>" +
                      f"Duration: {row['Duration']} days<br>" +
                      f"Return: {row['Return']:.1%}<br>" +
                      f"Max Drawdown: {row['Max Drawdown']:.1%}"
                      for _, row in stage_data.iterrows()],
                hoverinfo='text'
            ), row=7, col=1)

            # Add mean and std dev indicators
            mean_dd = stage_data['Max Drawdown'].mean()
            mean_ret = stage_data['Return'].mean()
            std_dd = stage_data['Max Drawdown'].std()
            std_ret = stage_data['Return'].std()

            # Add crosshair for mean values
            fig.add_shape(
                type="line",
                x0=mean_dd - std_dd,
                x1=mean_dd + std_dd,
                y0=mean_ret,
                y1=mean_ret,
                line=dict(color=stage_colors.get(stage, 'gray'), width=2),
                row=7, col=1
            )
            fig.add_shape(
                type="line",
                x0=mean_dd,
                x1=mean_dd,
                y0=mean_ret - std_ret,
                y1=mean_ret + std_ret,
                line=dict(color=stage_colors.get(stage, 'gray'), width=2),
                row=7, col=1
            )

    # Update layout
    fig.update_layout(
        title_text=f'Cumulative Returns Analysis - {sheet_name}',
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="right",
            x=1.2
        ),
        height=2800,
        margin=dict(t=250),
        template='plotly_dark',
        hovermode='closest'
    )

    # Update axes labels and formatting
    fig.update_yaxes(title_text="Cumulative Returns", row=1, col=1, secondary_y=False)

    # fig.update_yaxes(
    #     title_text="Cumulative Returns (log)", 
    #     row=1, 
    #     col=1, 
    #     secondary_y=False,
    #     type="log"  # Set y-axis to logarithmic scale
    # )
    fig.update_yaxes(title_text="Number of Active Stocks", row=1, col=1, secondary_y=True)
    
    fig.update_yaxes(
        title_text="Drawdown",
        row=2, col=1,
        range=[-1, 0],  # Reverse the axis to show 0 at top
        tickformat='.0%'  # Format as percentage
    )

    # Add a horizontal line at y=0 for drawdown reference
    fig.add_hline(
        y=0,
        line_dash="solid",
        line_color="gray",
        line_width=1,
        row=2, col=1
    )

    # Update other axes labels and formatting
    fig.update_yaxes(title_text="Active Cumulative Returns", row=3, col=1)
    fig.update_yaxes(title_text="Frequency", row=4, col=1)
    fig.update_yaxes(title_text="Frequency", row=5, col=1)
    fig.update_yaxes(title_text="Frequency", row=6, col=1)
    fig.update_yaxes(title_text="Trade Return", row=7, col=1)

    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_xaxes(title_text="Active Max Drawdown", row=3, col=1)
    fig.update_xaxes(title_text="Strategy Cumulative Returns", row=4, col=1)
    fig.update_xaxes(title_text="Security Cumulative Returns", row=5, col=1)
    fig.update_xaxes(title_text="Trade Returns", row=6, col=1)
    fig.update_xaxes(title_text="Max Drawdown", row=7, col=1)

    # Add grid lines for better readability in scatter plot
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        row=7, col=1,
        dtick=0.1
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        row=7, col=1,
        dtick=0.1
    )

    # Add performance quadrant annotation
    fig.add_annotation(
        x=0.02,
        y=0.95,
        text="← Better",
        showarrow=False,
        xref="paper",
        yref="paper",
        row=7, col=1
    )

    fig.show()
    return fig

# Function to collect buy/sell signals
def collect_buy_sell_signals(df, constituents, ticker):
    signals_df = df[(df['BUY_SIGNAL'] == 1) | (df['SELL_SIGNAL'] == 1)].copy()
    signals_df['Signal'] = np.where(signals_df['BUY_SIGNAL'] == 1, 'Buy', 'Sell')
    signals_df = signals_df[['Signal']]
    signals_df = signals_df.reset_index()

    # Add additional columns from constituents
    const_info = constituents[constituents['bb_ticker'] == ticker]
    
    if const_info.empty:
        gics_sector = 'N/A'
        gics_industry = 'N/A'
        instr_name = 'N/A'
    else:
        const_info = const_info.iloc[0]
        gics_sector = const_info['gics_sector']
        gics_industry = const_info['gics_industry']
        instr_name = const_info['instr_name']
    
    signals_df['gics_sector'] = gics_sector
    signals_df['gics_industry'] = gics_industry
    signals_df['instr_name'] = instr_name
    signals_df['bb_ticker'] = ticker

    return signals_df


def calculate_IC(df, ticker):
    # Identify buy and buy_exit signals
    buy_signals = df[df['BUY_SIGNAL'] == 1].copy()
    buy_exit_signals = df[df['BUY_EXIT_SIGNAL'] == 1].copy()

    # Ensure that buy and exit signals are aligned properly
    buy_exit_signals = buy_exit_signals[buy_exit_signals.index > buy_signals.index[0]]

    # Initialize lists to store trade details
    buy_dates = []
    buy_exit_dates = []
    durations = []
    buy_prices = []
    buy_exit_prices = []
    returns = []
    tickers = []  # List to store the ticker information

    # Iterate over buy signals to match with corresponding exit signals
    for i, buy_row in buy_signals.iterrows():
        buy_price = buy_row['Close']
        buy_date = buy_row.name  # Since 'Date' is the index, use 'name' to get the index value

        # Find the next corresponding exit signal
        exit_signal = buy_exit_signals[buy_exit_signals.index > i].head(1)
        
        if not exit_signal.empty:
            exit_row = exit_signal.iloc[0]
            buy_exit_price = exit_row['Close']
            buy_exit_date = exit_row.name  # Use 'name' to get the index value (Date)

            # Calculate return and duration
            trade_return = (buy_exit_price - buy_price) / buy_price
            duration = (buy_exit_date - buy_date).days

            # Record the details, including ticker
            buy_dates.append(buy_date)
            buy_exit_dates.append(buy_exit_date)
            durations.append(duration)
            buy_prices.append(buy_price)
            buy_exit_prices.append(buy_exit_price)
            returns.append(trade_return)
            tickers.append(ticker)  # Add the ticker for each trade

    # Create DataFrame to store trade details, including ticker
    trades_df = pd.DataFrame({
        'Ticker': tickers,
        'Buy Date': buy_dates,
        'Buy Exit Date': buy_exit_dates,
        'Duration': durations,
        'Buy Price': buy_prices,
        'Buy Exit Price': buy_exit_prices,
        'Return': returns
    })

    # Calculate success rate (Proportion of correct trades)
    proportion_correct = np.mean(trades_df['Return'] > 0)

    # Calculate IC (Information Coefficient)
    IC = (2 * proportion_correct) - 1

    return trades_df, proportion_correct, IC


def identify_trades(df, ticker):
    """
    Identify and compile all trades from price data and signals.
    
    Args:
        df (pd.DataFrame): DataFrame with price and signal data
        ticker (str): Ticker symbol
        
    Returns:
        pd.DataFrame: DataFrame containing all trade details
    """
    # Identify signal pairs for each stage
    buy_signals = df[df['BUY_SIGNAL'] == 1].copy()
    buy_exit_signals = df[df['BUY_EXIT_SIGNAL'] == 1].copy()
    hold_signals = df[df['HOLD_SIGNAL'] == 1].copy()
    hold_exit_signals = df[df['HOLD_EXIT_SIGNAL'] == 1].copy()
    sell_signals = df[df['SELL_SIGNAL'] == 1].copy()
    sell_exit_signals = df[df['SELL_EXIT_SIGNAL'] == 1].copy()

    # Initialize lists to store trade details
    stages = []
    entry_dates = []
    exit_dates = []
    durations = []
    entry_prices = []
    exit_prices = []
    returns = []
    max_drawdowns = []
    tickers = []

    def calculate_max_drawdown(prices):
        peak = prices[0]
        max_drawdown = 0
        for price in prices:
            peak = max(peak, price)
            drawdown = (price - peak) / peak
            max_drawdown = min(max_drawdown, drawdown)
        return abs(max_drawdown)

    def process_trade_signals(entry_df, exit_df, stage_name):
        for i, entry_row in entry_df.iterrows():
            entry_price = entry_row['Close']
            entry_date = entry_row.name
            
            # Find the next corresponding exit signal
            exit_signal = exit_df[exit_df.index > i].head(1)
            
            if not exit_signal.empty:
                exit_row = exit_signal.iloc[0]
                exit_price = exit_row['Close']
                exit_date = exit_row.name
                
                # Calculate trade metrics
                trade_return = (exit_price - entry_price) / entry_price
                duration = (exit_date - entry_date).days
                max_drawdown = calculate_max_drawdown(df['Close'].loc[entry_date:exit_date])
                
                # Store trade details
                stages.append(stage_name)
                entry_dates.append(entry_date)
                exit_dates.append(exit_date)
                durations.append(duration)
                entry_prices.append(entry_price)
                exit_prices.append(exit_price)
                returns.append(trade_return)
                max_drawdowns.append(max_drawdown)
                tickers.append(ticker)

    # Process each stage's trades
    process_trade_signals(buy_signals, buy_exit_signals, 'Buy')
    process_trade_signals(hold_signals, hold_exit_signals, 'Hold')
    process_trade_signals(sell_signals, sell_exit_signals, 'Sell')

    # Create and return DataFrame with trade details
    trades_df = pd.DataFrame({
        'Ticker': tickers,
        'Stage': stages,
        'Entry Date': entry_dates,
        'Exit Date': exit_dates,
        'Duration': durations,
        'Entry Price': entry_prices,
        'Exit Price': exit_prices,
        'Return': returns,
        'Max Drawdown': max_drawdowns
    })

    return trades_df

def calculate_IC_3_stages(trades_df):
    """
    Calculate performance metrics for each trading stage from trades data.
    
    Args:
        trades_df (pd.DataFrame): DataFrame containing trade details
        
    Returns:
        dict: Dictionary containing performance metrics for each stage
    """
    stage_metrics = {}
    total_trades = len(trades_df)
    
    for stage, group in trades_df.groupby('Stage'):
        trade_count = len(group)
        win_rate = np.mean(group['Return'] > 0)
        avg_return = group['Return'].mean()
        avg_duration = group['Duration'].mean()
        avg_drawdown = group['Max Drawdown'].mean()
        
        stage_metrics[stage] = {
            'Trade Count': trade_count,
            'Trade Count %': trade_count / total_trades if total_trades > 0 else 0,
            'Win Rate': win_rate,
            'IC': (2 * win_rate) - 1,
            'Avg Return': avg_return,
            'Avg Duration': avg_duration,
            'Avg Drawdown': avg_drawdown
        }

    return stage_metrics

# Updated display function for the main processing functions
def print_stage_metrics(stage_metrics):
    """Helper function to print stage metrics in a formatted way."""
    print("\nDetailed Stage Metrics:")
    print("-" * 80)
    
    for stage, metrics in stage_metrics.items():
        print(f"\n{stage} Stage:")
        print(f"  Trade Count: {metrics['Trade Count']} ({metrics['Trade Count %']:.1%} of total)")
        print(f"  Win Rate: {metrics['Win Rate']:.1%}")
        print(f"  IC: {metrics['IC']:.3f}")
        print(f"  Average Return: {metrics['Avg Return']:.1%}")
        print(f"  Average Duration: {metrics['Avg Duration']:.1f} days")
        print(f"  Average Drawdown: {metrics['Avg Drawdown']:.1%}")



# Function to save aggregated buy/sell signals to CSV with timestamp
def save_aggregated_signals(signals_df, index_name):
    # Generate a timestamp for the filename
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'./buy_sell_signals_{index_name}_{timestamp}.csv'

    # Save to CSV
    signals_df.to_csv(os.path.join( subfolder, filename), index=False, mode='w', header=True)


def save_aggregated_trades(trades_df, ticker):
    # Generate a timestamp for the filename
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'./aggregated_trades_{ticker}_{timestamp}.csv'
    
    # Save to CSV
    trades_df.to_csv(os.path.join(subfolder, filename), index=False, mode='w', header=True)



# Function to update SPY data
def update_spy_data(yf_data_strategy='full_reload', period='1y'):
    ticker = 'SPY'
    
    if yf_data_strategy == 'full_reload':
        pull_initial_df(ticker, period=period)
    elif yf_data_strategy == 'latest_10':
        if os.path.exists(os.path.join(subfolder, f'{ticker}_daily.csv')):
            pull_update_df(ticker)
        else:
            print(f"File for {ticker} does not exist. Reloading data.")
            pull_initial_df(ticker, period=period)
    elif yf_data_strategy == 'no_load' and not os.path.exists(os.path.join(subfolder, f'{ticker}_daily.csv')):
        print(f"File for {ticker} does not exist. Reloading data.")
        pull_initial_df(ticker, period=period)
    elif yf_data_strategy == 'no_load' and os.path.exists(os.path.join(subfolder, f'{ticker}_daily.csv')):
        print(f"Using existing file for {ticker} without reloading.")
    else:
        print("Invalid yf_data_strategy for SPY. Defaulting to full reload.")
        pull_initial_df(ticker, period=period)



def calculate_cumulative_returns(df, initial_capital=100):
    df['Portfolio_Value'] = initial_capital
    df['Cash'] = initial_capital
    df['Position'] = 0  # This will track the number of shares held
    df['Cumulative_Returns'] = 0

    for i in range(1, len(df)):
        # Carry forward the position and cash from the previous day
        df.at[df.index[i], 'Position'] = df['Position'].iloc[i-1]
        df.at[df.index[i], 'Cash'] = df['Cash'].iloc[i-1]

        # Check for buy signal
        if df['BUY_SIGNAL'].iloc[i] == 1 and df['Cash'].iloc[i] > 0:
            # Buy as many shares as possible with the available cash
            df.at[df.index[i], 'Position'] = df['Cash'].iloc[i] / df['Close'].iloc[i]
            df.at[df.index[i], 'Cash'] = 0

        # Check for stop loss execution
        if df['Stop_Loss_Executed'].iloc[i] == True:
            # Sell all shares at the low price of the day
            df.at[df.index[i], 'Cash'] = df['Position'].iloc[i] * df['Low'].iloc[i]
            df.at[df.index[i], 'Position'] = 0

        # Calculate the portfolio value
        df.at[df.index[i], 'Portfolio_Value'] = df['Cash'].iloc[i] + df['Position'].iloc[i] * df['Close'].iloc[i]

    # Calculate cumulative returns
    df['Cumulative_Returns'] = df['Portfolio_Value'] / initial_capital - 1

    return df


def calculate_cumulative_returns_leverage(df, initial_capital=100, leverage_ratio=1.2):
    if leverage_ratio < 1:
        raise ValueError("leverage_ratio must be no less than 1.")
    
    df['Portfolio_Value'] = initial_capital
    df['Cash'] = initial_capital
    df['Position'] = 0  # This will track the number of shares held
    df['Cumulative_Returns'] = 0
    df['Strategy_Daily_Return'] = np.nan  # Initialize the new column for daily returns
    df['First_Buy'] = False  # Flag to track the first buy order

    for i in range(1, len(df)):
        # Carry forward the position, cash, and portfolio value from the previous day
        previous_position = df['Position'].iloc[i-1]
        previous_cash = df['Cash'].iloc[i-1]
        previous_close = df['Close'].iloc[i-1]
        previous_portfolio_value = df['Portfolio_Value'].iloc[i-1]
        current_close = df['Close'].iloc[i]

        # Carry forward the position and cash from the previous day
        df.at[df.index[i], 'Position'] = previous_position
        df.at[df.index[i], 'Cash'] = previous_cash
        df.at[df.index[i], 'Portfolio_Value'] = previous_portfolio_value

        # Calculate Strategy_Daily_Return based on position
        if previous_position > 0:
            df.at[df.index[i], 'Strategy_Daily_Return'] = ((current_close / previous_close) - 1) * leverage_ratio

        # Check for buy signal
        if df['BUY_SIGNAL'].iloc[i] == 1 and df['BUY_EXIT_SIGNAL'].iloc[i] == 0 and previous_cash > 0:
            shares_to_buy = previous_portfolio_value / current_close * leverage_ratio
            df.at[df.index[i], 'Position'] = shares_to_buy
            # Adjust cash to ensure the sum of cash and position value equals previous_portfolio_value
            df.at[df.index[i], 'Cash'] = previous_portfolio_value - (shares_to_buy * current_close)
            if not df['First_Buy'].any():  # If this is the first buy
                df.at[df.index[i], 'First_Buy'] = True

        # Check for buy exit signal
        elif df['BUY_EXIT_SIGNAL'].iloc[i] == 1 and df['BUY_SIGNAL'].iloc[i] == 0 and previous_position > 0:
            buy_exit_price = df['Buy_Exit_Price'].iloc[i]
            df.at[df.index[i], 'Cash'] = previous_position * buy_exit_price + previous_cash
            df.at[df.index[i], 'Position'] = 0

        # Calculate the portfolio value
        df.at[df.index[i], 'Portfolio_Value'] = df['Cash'].iloc[i] + df['Position'].iloc[i] * df['Close'].iloc[i]

    # Calculate strategy cumulative returns with leverage
    df['Strategy_Cumulative_Returns'] = df['Portfolio_Value'] / initial_capital

    # Calculate strategy drawdown
    df['Strategy_Peak'] = df['Strategy_Cumulative_Returns'].cummax()
    df['Strategy_Drawdown'] = (df['Strategy_Cumulative_Returns'] - df['Strategy_Peak']) / df['Strategy_Peak']

    # Calculate security cumulative returns from the first buy order
    first_buy_index = df[df['First_Buy']].index[0] if df['First_Buy'].any() else df.index[-1]
    df['Security_Cumulative_Returns'] = 1  # Initialize all rows to 1
    df.loc[first_buy_index:, 'Security_Cumulative_Returns'] = df.loc[first_buy_index:, 'Close'] / df.loc[first_buy_index, 'Close']

    # Calculate security drawdown
    df['Security_Peak'] = df['Security_Cumulative_Returns'].cummax()
    df['Security_Drawdown'] = (df['Security_Cumulative_Returns'] - df['Security_Peak']) / df['Security_Peak']

    # Calculate Active Cumulative Returns as the difference between Strategy and Security Cumulative Returns (symmetric ratio)
    df['Active_Cumulative_Returns'] = (df['Strategy_Cumulative_Returns'] - df['Security_Cumulative_Returns']) / (abs(df['Strategy_Cumulative_Returns']) + abs(df['Security_Cumulative_Returns']))

    # Calculate Max Drawdowns
    df['Strategy_Max_Drawdown'] = df['Strategy_Drawdown'].min()  # Max drawdown is the minimum (most negative) drawdown
    df['Security_Max_Drawdown'] = df['Security_Drawdown'].min()

    # Calculate Active Max Drawdown as the difference between Strategy Max Drawdown and Security Max Drawdown
    df['Active_Max_Drawdown'] = 1- df['Strategy_Max_Drawdown'] / df['Security_Max_Drawdown']

    return df



def calculate_cumulative_returns_early_detection_3_stages(df, initial_capital=100, leverage_ratio_buy=1.2, leverage_ratio_hold=1.0, leverage_ratio_sell=0.8):
    df['Portfolio_Value'] = initial_capital
    df['Cash'] = initial_capital
    df['Position'] = 0
    df['Current_Leverage'] = 0
    df['Cumulative_Returns'] = 0
    df['Strategy_Daily_Return'] = np.nan
    df['First_Trade'] = False
    df['Trade_Type'] = 'NONE'

    for i in range(1, len(df)):
        current_close = df['Close'].iloc[i]
        previous_close = df['Close'].iloc[i-1]
        previous_portfolio_value = df['Portfolio_Value'].iloc[i-1]

        # Calculate the daily return first based on previous position
        if df['Position'].iloc[i-1] != 0:
            base_return = (current_close / previous_close) - 1
            if df['Trade_Type'].iloc[i-1] == 'BUY':
                df.at[df.index[i], 'Strategy_Daily_Return'] = base_return * leverage_ratio_buy
            elif df['Trade_Type'].iloc[i-1] == 'HOLD':
                df.at[df.index[i], 'Strategy_Daily_Return'] = base_return * leverage_ratio_hold
            elif df['Trade_Type'].iloc[i-1] == 'SELL':
                df.at[df.index[i], 'Strategy_Daily_Return'] = base_return * leverage_ratio_sell

        # Update portfolio value based on the daily return
        if pd.notna(df['Strategy_Daily_Return'].iloc[i]):
            current_portfolio_value = previous_portfolio_value * (1 + df['Strategy_Daily_Return'].iloc[i])
        else:
            current_portfolio_value = previous_portfolio_value

        # Now handle the signals using the current portfolio value
        if df['BUY_SIGNAL'].iloc[i] == 1:
            shares_to_trade = current_portfolio_value / current_close * abs(leverage_ratio_buy)
            df.at[df.index[i], 'Position'] = shares_to_trade
            df.at[df.index[i], 'Cash'] = current_portfolio_value - (shares_to_trade * current_close)
            df.at[df.index[i], 'Trade_Type'] = 'BUY'
            if not df['First_Trade'].any():
                df.at[df.index[i], 'First_Trade'] = True

        elif df['HOLD_SIGNAL'].iloc[i] == 1:
            shares_to_trade = current_portfolio_value / current_close * abs(leverage_ratio_hold)
            df.at[df.index[i], 'Position'] = shares_to_trade
            df.at[df.index[i], 'Cash'] = current_portfolio_value - (shares_to_trade * current_close)
            df.at[df.index[i], 'Trade_Type'] = 'HOLD'
            if not df['First_Trade'].any():
                df.at[df.index[i], 'First_Trade'] = True

        elif df['SELL_SIGNAL'].iloc[i] == 1:
            shares_to_trade = current_portfolio_value / current_close * abs(leverage_ratio_sell)
            df.at[df.index[i], 'Position'] = shares_to_trade
            df.at[df.index[i], 'Cash'] = current_portfolio_value - (shares_to_trade * current_close)
            df.at[df.index[i], 'Trade_Type'] = 'SELL'
            if not df['First_Trade'].any():
                df.at[df.index[i], 'First_Trade'] = True

        # Handle exits
        elif (df['BUY_EXIT_SIGNAL'].iloc[i] == 1 and df['Trade_Type'].iloc[i-1] == 'BUY') or \
             (df['HOLD_EXIT_SIGNAL'].iloc[i] == 1 and df['Trade_Type'].iloc[i-1] == 'HOLD') or \
             (df['SELL_EXIT_SIGNAL'].iloc[i] == 1 and df['Trade_Type'].iloc[i-1] == 'SELL'):
            df.at[df.index[i], 'Cash'] = current_portfolio_value
            df.at[df.index[i], 'Position'] = 0
            df.at[df.index[i], 'Trade_Type'] = 'NONE'
        else:
            # Carry forward position and trade type if no new signals
            df.at[df.index[i], 'Position'] = df['Position'].iloc[i-1]
            df.at[df.index[i], 'Cash'] = df['Cash'].iloc[i-1]
            df.at[df.index[i], 'Trade_Type'] = df['Trade_Type'].iloc[i-1]

        # Update portfolio value
        df.at[df.index[i], 'Portfolio_Value'] = current_portfolio_value

    # Calculate performance metrics
    df['Strategy_Cumulative_Returns'] = df['Portfolio_Value'] / initial_capital

    # Calculate strategy drawdown
    df['Strategy_Peak'] = df['Strategy_Cumulative_Returns'].cummax()
    df['Strategy_Drawdown'] = (df['Strategy_Cumulative_Returns'] - df['Strategy_Peak']) / df['Strategy_Peak']

    # Calculate security cumulative returns from the first trade
    first_trade_index = df[df['First_Trade']].index[0] if df['First_Trade'].any() else df.index[-1]
    df['Security_Cumulative_Returns'] = 1
    df.loc[first_trade_index:, 'Security_Cumulative_Returns'] = df.loc[first_trade_index:, 'Close'] / df.loc[first_trade_index, 'Close']

    # Calculate security drawdown
    df['Security_Peak'] = df['Security_Cumulative_Returns'].cummax()
    df['Security_Drawdown'] = (df['Security_Cumulative_Returns'] - df['Security_Peak']) / df['Security_Peak']

    # Calculate Active Cumulative Returns
    df['Active_Cumulative_Returns'] = (df['Strategy_Cumulative_Returns'] - df['Security_Cumulative_Returns']) / \
                                    (abs(df['Strategy_Cumulative_Returns']) + abs(df['Security_Cumulative_Returns']))

    # Calculate Max Drawdowns
    df['Strategy_Max_Drawdown'] = df['Strategy_Drawdown'].min()
    df['Security_Max_Drawdown'] = df['Security_Drawdown'].min()

    # Calculate Active Max Drawdown
    df['Active_Max_Drawdown'] = 1 - df['Strategy_Max_Drawdown'] / df['Security_Max_Drawdown']

    return df

# Function to process a single ticker
def process_single_ticker(ticker, constituents, yf_data_strategy='full_reload', period='1y', option='single', resample_period='W'):
    print(f'Processing {ticker}...')
    filename = os.path.join( subfolder, f'{ticker}_daily.csv')

    # Use the new yf_data_strategy logic
    if yf_data_strategy == 'full_reload':
        pull_initial_df(ticker, period=period)
    elif yf_data_strategy == 'latest_10':
        if os.path.exists(filename):
            pull_update_df(ticker)
        else:
            print(f"File for {ticker} does not exist. Reloading data.")
            pull_initial_df(ticker, period=period)
    elif yf_data_strategy == 'no_load' and not os.path.exists(filename):
        print(f"File for {ticker} does not exist. Reloading data.")
        pull_initial_df(ticker, period=period)
    elif yf_data_strategy == 'no_load' and os.path.exists(filename):
        print(f"Using existing file for {ticker} without reloading.")
    else:
        print("Invalid yf_data_strategy. Defaulting to full reload.")
        pull_initial_df(ticker, period=period)

    # Read from local CSV with resampling option
    local_df = read_local_df(ticker, resample_period=resample_period, period=period)

    # Calculate indicators
    df = calculate_sma(local_df, window=20)
    df = calculate_sma_slope(df)
    df = calculate_donchian_channel(df, window=10)
    df = calculate_adx(df, window=14 * 22)  # Add ADX calculation here
    # df = calculate_adx(df, window=14)  # Add ADX calculation here
    df = calculate_awesome_oscillator(df)  # Add this line to calculate AO

    # Calculate MACD and KDJ
    df = calculate_macd(df)
    df = calculate_kdj(df)
    df = calculate_rsi(df, period=14 *22) 
    df = calculate_mfi(df, length=48)  # Add this line


    df = calculate_daily_returns(df)

    # Generate flipping points and check within lookback window
    df = generate_flipping_points(df)
    df = within_lookback_window(df, lookback_window=5) 
    df = calculate_volume_signal(df, days_short=2, days_long=5, multiplier=0.5)

    # Calculate ratio against SPY
    local_spy_df = read_local_df('SPY', resample_period=resample_period, period=period)
    local_spy_df = calculate_sma(local_spy_df)  # This will now calculate SMA_30 and Close-SMA_30 for SPY
    df = calculate_ratio(df, local_spy_df)

    # Generate individual condition columns
    df = generate_individual_conditions(df)

    # Generate buy and sell signals BY ADX
    # df = generate_buy_sell_signals_by_ADX_v2(df, adx_percent_threshold=0.99, stop_loss_threshold_principal=8, stop_loss_threshold_profit=10)
    # df = generate_buy_sell_signals_by_gold_dead_cross(df)
    # df = generate_buy_sell_signals_by_gold_dead_cross_both_daily_monthly(df)
    # df = generate_buy_sell_signals_by_gold_dead_cross_only_monthly_from_daily_return(df)
    # df = generate_buy_sell_signals_early_detection(df)
    df = generate_buy_sell_signals_early_detection_3_stages_MFI(df) # current model
    # df = generate_buy_sell_signals_golden_dead_cross_MFI(df) # current model

    
    # df = generate_buy_sell_signals_early_detection_late_exit(df)
    # df = generate_buy_sell_signals_daily_kdj_detection(df)
    # df = generate_buy_sell_signals_by_gold_dead_cross_only_monthly_from_daily_return_And_RSI(df)
    # df = generate_buy_sell_signals_by_gold_dead_cross_only_monthly_from_daily_return_And_RSI_Forfront(df)

    
    # df = generate_buy_sell_signals_by_KDJ_only_monthly_from_daily_return_J_over_KD_more_than_10_drawdown_protection_profit_taking(df)

    # df = generate_buy_sell_signals_by_KDJ_only_monthly_from_daily_return_J_over_KD_more_than_10(df)






    

    
    
    # Calculate cumulative returns
    # df = calculate_cumulative_returns_by_ADX(df, initial_capital=100, strategy='LO')
    # df = calculate_cumulative_returns_leverage(df, initial_capital=100, leverage_ratio=1.0)

    df = calculate_cumulative_returns_early_detection_3_stages(
        df, 
        initial_capital=100,
        leverage_ratio_buy=1.2,    # 150% exposure for buy signals
        leverage_ratio_hold=0,   # 100% exposure for hold signals
        leverage_ratio_sell=0.8   # -50% exposure (short) for sell signals
    )
    

    # # Call the calculate_IC function after calculating cumulative returns
    # trades_df, proportion_correct, IC = calculate_IC(df, ticker)

    # Call the function
    trades_df = identify_trades(df, ticker)

    stage_metrics = calculate_IC_3_stages(trades_df)

    # Display stage metrics for each stage
    # print("\nStage Metrics:")
    # for stage, metrics in stage_metrics.items():
    #     print(f"Stage: {stage}")
    #     print(f"  Proportion Correct: {metrics['Proportion Correct']}")
    #     print(f"  IC: {metrics['IC']}")

    # Collect buy/sell signals
    signals_df = collect_buy_sell_signals(df, constituents, ticker)

    if option == 'single':
        df.to_csv(os.path.join(new_directory, subfolder, f'{ticker}_processed.csv'))
        print(f"\nAnalysis Results for {ticker}:")
        print_stage_metrics(stage_metrics)
        plot_chart_by_ADX(df, trades_df, ticker)
        save_aggregated_signals(signals_df, ticker)
        save_aggregated_trades(trades_df, ticker)

    return signals_df, df, trades_df, stage_metrics



def process_ticker(args):
    """Process a single ticker with the given arguments."""
    ticker, constituents, yf_data_strategy, period, option, resample_period = args
    try:
        return process_single_ticker(ticker, constituents, yf_data_strategy, period, option, resample_period)
    except Exception as e:
        print(f"Error processing {ticker}: {e}")
        return None

def process_all_index_tickers(sheet_name, yf_data_strategy='full_reload', period='1y', top_n=None, resample_period='D', sector=None, if_run_multi_process=True):
    start_time = datetime.datetime.now()
    print(f"\nStarting full processing at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    constituents = read_index_constituents(sheet_name=sheet_name)
    
    if sector is not None:
        constituents = constituents[constituents['gics_sector'] == sector]

    tickers = constituents['bb_ticker']
    if top_n is not None:
        tickers = tickers.head(top_n)

    total_tickers = len(tickers)
    print(f"Total tickers to process: {total_tickers}")
    results_dict = {}
    completed_count = 0

    if if_run_multi_process:
        try:
            tickers_list = list(tickers)
            max_workers = max(1, (os.cpu_count() or 1) - 1)
            print(f"Running with {max_workers} processes")
            
            # Prepare arguments for each process
            process_args = [
                (ticker, constituents, yf_data_strategy, period, 'all', resample_period)
                for ticker in tickers_list
            ]
            
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks and get futures
                futures = {
                    executor.submit(process_ticker, args): args[0]  # args[0] is ticker
                    for args in process_args
                }
                
                # Process completed futures as they complete
                for future in concurrent.futures.as_completed(futures):
                    ticker = futures[future]
                    completed_count += 1
                    current_time = datetime.datetime.now()
                    
                    print(f"\nProcessing {completed_count}/{total_tickers} ({(completed_count/total_tickers)*100:.1f}%)")
                    print(f"Started processing {ticker} at {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
                    
                    try:
                        result = future.result()
                        if result is not None:
                            signals_df, df, trades_df, stage_metrics = result
                            results_dict[ticker] = {
                                'signals_df': signals_df,
                                'df': df,
                                'trades_df': trades_df,
                                'stage_metrics': stage_metrics
                            }
                            elapsed_time = datetime.datetime.now() - current_time
                            print(f"Completed {ticker} in {elapsed_time.total_seconds():.1f} seconds")
                        else:
                            print(f"Failed to process {ticker}")
                    except Exception as exc:
                        print(f'{ticker} generated an exception: {exc}')
        
        except Exception as e:
            print(f"Error in concurrent processing: {str(e)}")
            print("Falling back to sequential processing")
            if_run_multi_process = False

    if not if_run_multi_process:
        print("Running with sequential processing")
        for idx, ticker in enumerate(tickers, 1):
            try:
                current_time = datetime.datetime.now()
                print(f"\nProcessing {idx}/{total_tickers} ({(idx/total_tickers)*100:.1f}%)")
                print(f"Started processing {ticker} at {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
                
                signals_df, df, trades_df, stage_metrics = process_single_ticker(
                    ticker, 
                    constituents, 
                    yf_data_strategy=yf_data_strategy, 
                    period=period, 
                    option='all', 
                    resample_period=resample_period
                )
                
                elapsed_time = datetime.datetime.now() - current_time
                print(f"Completed {ticker} in {elapsed_time.total_seconds():.1f} seconds")
                
                results_dict[ticker] = {
                    'signals_df': signals_df,
                    'df': df,
                    'trades_df': trades_df,
                    'stage_metrics': stage_metrics
                }
            except Exception as e:
                print(f"Error processing {ticker}: {str(e)}")

    end_time = datetime.datetime.now()
    duration = end_time - start_time
    print(f"\nTotal processing completed at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total duration: {duration}")
    print(f"Successfully processed {len(results_dict)} out of {total_tickers} tickers")

    # Now process all results together
    all_signals = []
    all_trades_df = []
    strategy_daily_returns = pd.DataFrame()
    security_daily_returns = pd.DataFrame()
    df_strategy_cumulative_returns = pd.DataFrame()
    df_security_cumulative_returns = pd.DataFrame()
    df_active_metrics = pd.DataFrame(columns=['Ticker', 'Active_Max_Drawdown', 'Active_Cumulative_Returns'])

    # Process results from dictionary
    for ticker, result in results_dict.items():
        signals_df = result['signals_df']
        df = result['df']
        trades_df = result['trades_df']
        
        if not trades_df.empty:
            all_trades_df.append(trades_df)

        if not signals_df.empty:
            all_signals.append(signals_df)

            # Process returns data
            if 'Strategy_Daily_Return' in df.columns:
                strategy_daily_returns = strategy_daily_returns.join(
                    pd.Series(df['Strategy_Daily_Return'], name=ticker), 
                    how='outer'
                )

            if 'Daily_Returns' in df.columns:
                security_daily_returns = security_daily_returns.join(
                    pd.Series(df['Daily_Returns'], name=ticker), 
                    how='outer'
                )

            if 'Strategy_Cumulative_Returns' in df.columns:
                df_strategy_cumulative_returns[ticker] = df['Strategy_Cumulative_Returns']

            if 'Security_Cumulative_Returns' in df.columns:
                df_security_cumulative_returns[ticker] = df['Security_Cumulative_Returns']

            # Add active metrics
            if all(col in df.columns for col in ['Active_Max_Drawdown', 'Active_Cumulative_Returns']):
                new_row = pd.DataFrame({
                    'Ticker': [ticker],
                    'Active_Max_Drawdown': [df['Active_Max_Drawdown'].iloc[-1]],
                    'Active_Cumulative_Returns': [df['Active_Cumulative_Returns'].iloc[-1]]
                })
                df_active_metrics = pd.concat([df_active_metrics, new_row], ignore_index=True)

    # Process combined trades metrics if we have any trades
    if all_trades_df:
        combined_trades_df = pd.concat(all_trades_df, ignore_index=True)
        combined_trades_metrics = calculate_IC_3_stages(combined_trades_df)
        print("\nOverall Stage Metrics:")
        print_stage_metrics(combined_trades_metrics)
    else:
        combined_trades_df = pd.DataFrame()  # Empty DataFrame if no trades



    if not strategy_daily_returns.empty:
        numeric_cols = strategy_daily_returns.select_dtypes(include=['float64', 'int64']).columns
        
        first_active_date = strategy_daily_returns[
            strategy_daily_returns[numeric_cols].apply(lambda x: x.notna().any(), axis=1)
        ].index.min()
        

        if first_active_date is not None:
            # Truncate both series to start from the first active date
            strategy_daily_returns = strategy_daily_returns[strategy_daily_returns.index >= first_active_date]
            if security_daily_returns is not None:
                security_daily_returns = security_daily_returns[security_daily_returns.index >= first_active_date]

        strategy_daily_returns['Index_Strategy_Daily_Returns'] = strategy_daily_returns[numeric_cols].apply(
            lambda row: row.mean(skipna=True) if row.notna().sum() > 0 else 0, 
            axis=1
        )

        if not security_daily_returns.empty:
            security_daily_returns['Index_Security_Daily_Returns'] = security_daily_returns[numeric_cols].apply(
                lambda row: row.mean(skipna=True) if row.notna().sum() > 0 else 0, 
                axis=1
            )

        # Calculate number of active stocks using numeric columns
        strategy_daily_returns['Number_Active_Stocks'] = strategy_daily_returns[numeric_cols].apply(
            lambda row: row.notna().sum(), 
            axis=1
        )


        # Normalize both series to start at 1 from the first active date
        # strategy_daily_returns['Index_Strategy_Cum_Returns'] = (1 + strategy_daily_returns['Index_Strategy_Daily_Returns']).cumprod() - 1
        strategy_daily_returns['Index_Strategy_Cum_Returns'] = (1 + strategy_daily_returns['Index_Strategy_Daily_Returns']).cumprod()
        
        if security_daily_returns is not None:
            # security_daily_returns['Index_Security_Cum_Returns'] = (1 + security_daily_returns['Index_Security_Daily_Returns']).cumprod() - 1
            security_daily_returns['Index_Security_Cum_Returns'] = (1 + security_daily_returns['Index_Security_Daily_Returns']).cumprod()
            
            # Calculate active returns
            strategy_daily_returns['Strategy_Active_Cum_Returns'] = (
                strategy_daily_returns['Index_Strategy_Cum_Returns'] - 
                security_daily_returns['Index_Security_Cum_Returns']
            )

        # Calculate drawdowns for strategy
        strategy_daily_returns['Strategy_Peak'] = strategy_daily_returns['Index_Strategy_Cum_Returns'].cummax()
        strategy_daily_returns['Strategy_Drawdown'] = (
            (strategy_daily_returns['Index_Strategy_Cum_Returns'] - strategy_daily_returns['Strategy_Peak']) / 
            strategy_daily_returns['Strategy_Peak']
        )
        
        # Calculate drawdowns for security in its own dataframe
        if security_daily_returns is not None:
            security_daily_returns['Security_Peak'] = security_daily_returns['Index_Security_Cum_Returns'].cummax()
            security_daily_returns['Security_Drawdown'] = (
                (security_daily_returns['Index_Security_Cum_Returns'] - security_daily_returns['Security_Peak']) / 
                security_daily_returns['Security_Peak'] 
            )

        # Plot aggregated portfolio value
        try:
            plot_aggregated_portfolio_value(
                strategy_daily_returns, 
                df_strategy_cumulative_returns, 
                df_security_cumulative_returns, 
                sheet_name=sheet_name, 
                security_daily_returns=security_daily_returns,
                active_metrics=df_active_metrics,
                combined_trades_df=combined_trades_df
            )
        except Exception as e:
            print(f"Error in plotting: {str(e)}")

    # Save results to files
    try:
        # Add sector to filename if provided
        sector_suffix = f"_{sector}" if sector is not None else ""
        
        # Save signals and trades if we have any
        if all_signals:
            aggregated_signals_df = pd.concat(all_signals, ignore_index=True)
            save_aggregated_signals(aggregated_signals_df, f"{sheet_name}{sector_suffix}")
            
        if not combined_trades_df.empty:
            save_aggregated_trades(combined_trades_df, f"{sheet_name}{sector_suffix}")

        # Save returns and metrics data
        strategy_daily_returns.to_csv(os.path.join(subfolder, f'{sheet_name}{sector_suffix}_portfolio_strategy_cum_returns.csv'))
        security_daily_returns.to_csv(os.path.join(subfolder, f'{sheet_name}{sector_suffix}_portfolio_security_cum_returns.csv'))
        df_strategy_cumulative_returns.to_csv(os.path.join(subfolder, f'{sheet_name}{sector_suffix}_signal_constituent_strategy_cum_returns.csv'))
        df_security_cumulative_returns.to_csv(os.path.join(subfolder, f'{sheet_name}{sector_suffix}_singal_constituent_security_cum_returns.csv'))
        df_active_metrics.to_csv(os.path.join(subfolder, f'{sheet_name}{sector_suffix}_active_metrics.csv'), index=False)
    
    except Exception as e:
        print(f"Error saving files: {str(e)}")

    return (
        strategy_daily_returns, 
        security_daily_returns, 
        df_strategy_cumulative_returns, 
        df_security_cumulative_returns, 
        df_active_metrics
    )


