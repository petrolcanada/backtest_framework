"""
Backtest Engine module for simulating trading strategies.
"""
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from backtest_framework.core.strategies.base import BaseStrategy

class BacktestEngine:
    """
    Simulates and evaluates trading strategies on historical data.
    """
    
    def __init__(self, initial_capital: float = 10000.0, commission: float = 0.001,
               slippage: float = 0.0, leverage: Union[float, Dict[str, float]] = 1.0, 
               enable_short_selling: bool = False, position_sizing: float = 1.0,
               include_dividends: bool = True, short_borrow_rate: float = 0.02,
               use_margin_costs: bool = True):
        """
        Initialize the backtest engine with trading parameters.
        
        Args:
            initial_capital: Starting capital for the backtest
            commission: Commission rate as a decimal (e.g., 0.001 = 0.1%)
            slippage: Slippage as a decimal (e.g., 0.001 = 0.1%)
            leverage: Leverage multiplier. Can be:
                     - float: Same leverage for long and short (e.g., 2.0)
                     - dict: Different leverage by direction {'long': 2.0, 'short': 1.0}
            enable_short_selling: Whether to allow short selling on sell signals
            position_sizing: Fraction of capital to use per trade (0.5 = 50% of capital)
            include_dividends: Whether to include dividend payments in backtest calculations
            short_borrow_rate: Annual borrowing rate for short positions (default: 2%)
            use_margin_costs: Whether to include margin interest costs based on Fed Funds Rate
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        
        # Handle leverage configuration
        if isinstance(leverage, dict):
            self.long_leverage = leverage.get('long', 1.0)
            self.short_leverage = leverage.get('short', 1.0)
        else:
            self.long_leverage = leverage
            self.short_leverage = leverage
            
        self.enable_short_selling = enable_short_selling
        self.position_sizing = position_sizing
        self.include_dividends = include_dividends
        self.short_borrow_rate = short_borrow_rate
        self.use_margin_costs = use_margin_costs
        self.risk_managers = []
        
        # Federal Funds Rate data (will be loaded when needed)
        self.ffr_data = None
    
    def add_risk_manager(self, risk_manager):
        """
        Add a risk management module to the backtest engine.
        
        Args:
            risk_manager: Risk management module to add
        """
        self.risk_managers.append(risk_manager)
    
    def _load_ffr_data(self, start_date: str, end_date: str):
        """
        Load Federal Funds Rate data for the backtest period.
        
        Args:
            start_date: Start date for FFR data
            end_date: End date for FFR data
        """
        if not self.use_margin_costs:
            return
        
        try:
            from backtest_framework.core.data.loader import DataLoader
            loader = DataLoader()
            self.ffr_data = loader.load_fed_funds_rate(start_date=start_date)
            
            # Filter to backtest period
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            self.ffr_data = self.ffr_data[(self.ffr_data.index >= start_dt) & 
                                        (self.ffr_data.index <= end_dt)]
            
            print(f"Loaded FFR data from {self.ffr_data.index.min()} to {self.ffr_data.index.max()}")
        except Exception as e:
            print(f"Warning: Could not load FFR data: {e}. Margin costs will use 2% approximation.")
            self.ffr_data = None
    
    def _get_daily_margin_rate(self, date: pd.Timestamp) -> float:
        """
        Get the daily margin interest rate for a given date.
        
        Args:
            date: Date to get rate for
            
        Returns:
            Daily margin rate (annual rate / 252 trading days)
        """
        if not self.use_margin_costs:
            return 0.0
        
        annual_rate = 2.0  # Default 2.0% (absolute rate)
        
        if self.ffr_data is not None and date in self.ffr_data.index:
            # Use FFR + spread (typically 1-3% above FFR for margin)
            ffr_rate = self.ffr_data.loc[date]  # FFR is already in absolute form (e.g., 4.0 for 4%)
            annual_rate = ffr_rate + 1.5  # FFR + 1.5% spread
        elif self.ffr_data is not None and not self.ffr_data.empty:
            # Use most recent available rate
            available_dates = self.ffr_data.index[self.ffr_data.index <= date]
            if not available_dates.empty:
                recent_date = available_dates.max()
                ffr_rate = self.ffr_data.loc[recent_date]
                annual_rate = ffr_rate + 1.5
        
        return (annual_rate / 100.0) / 252  # Convert percentage to decimal and use 252 trading days
    
    def _get_daily_short_borrow_rate(self) -> float:
        """
        Get the daily short borrowing rate.
        
        Returns:
            Daily short borrow rate (annual rate / 252 trading days)
        """
        return (self.short_borrow_rate / 100.0) / 252 if self.short_borrow_rate > 1 else self.short_borrow_rate / 252
    
    def run(self, strategy: BaseStrategy, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
          portfolio_rebalance: str = 'buy_sell', start_date: Optional[str] = None,
          end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Run a backtest for the given strategy and data.
        
        Args:
            strategy: Strategy to backtest
            data: DataFrame with OHLCV data or dict mapping tickers to DataFrames
            portfolio_rebalance: Portfolio rebalancing method ('buy_sell', 'equal_weight', 'buy_hold')
            start_date: Start date for the backtest (format: 'YYYY-MM-DD')
            end_date: End date for the backtest (format: 'YYYY-MM-DD')
            
        Returns:
            DataFrame with backtest results including equity curve and performance metrics
        """
        # Handle single or multiple tickers
        if isinstance(data, pd.DataFrame):
            data_dict = {'SINGLE': data}
        else:
            data_dict = data
        
        # Filter data by date range if specified
        if start_date or end_date:
            data_dict = self._filter_data_by_date(data_dict, start_date, end_date)
        
        # Determine actual start and end dates for FFR loading
        actual_start_date = start_date
        actual_end_date = end_date
        
        if not actual_start_date or not actual_end_date:
            # Get date range from data
            all_dates = []
            for df in data_dict.values():
                if not df.empty:
                    all_dates.extend([df.index.min(), df.index.max()])
            
            if all_dates:
                if not actual_start_date:
                    actual_start_date = min(all_dates).strftime('%Y-%m-%d')
                if not actual_end_date:
                    actual_end_date = max(all_dates).strftime('%Y-%m-%d')
        
        # Load FFR data for margin cost calculations
        if actual_start_date and actual_end_date:
            self._load_ffr_data(actual_start_date, actual_end_date)
        
        # Generate signals for each ticker
        signal_dfs = {}
        for ticker, df in data_dict.items():
            signal_df = strategy.run(df)
            
            # Apply risk management
            for risk_manager in self.risk_managers:
                signal_df = risk_manager.apply(signal_df)
            
            signal_dfs[ticker] = signal_df
        
        # If multiple tickers, run portfolio backtest
        if len(signal_dfs) > 1:
            return self._run_portfolio_backtest(signal_dfs, portfolio_rebalance)
        else:
            # Single ticker backtest
            ticker = list(signal_dfs.keys())[0]
            return self._run_single_backtest(signal_dfs[ticker])
    
    def _filter_data_by_date(self, data_dict: Dict[str, pd.DataFrame], 
                          start_date: Optional[str], 
                          end_date: Optional[str]) -> Dict[str, pd.DataFrame]:
        """
        Filter data by date range.
        
        Args:
            data_dict: Dictionary mapping tickers to DataFrames
            start_date: Start date string (format: 'YYYY-MM-DD')
            end_date: End date string (format: 'YYYY-MM-DD')
            
        Returns:
            Dictionary with filtered DataFrames
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
            
            filtered_dict[ticker] = filtered_df
        
        return filtered_dict
    
    def _prepare_dividend_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare dividend data for backtesting. If 'Dividends' column exists,
        use it; otherwise, try to fetch dividend data or create empty column.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with 'Dividends' column added
        """
        result = data.copy()
        
        # If dividends column already exists, use it
        if 'Dividends' in result.columns:
            return result
        
        # If include_dividends is False, create empty dividend column
        if not self.include_dividends:
            result['Dividends'] = 0.0
            return result
        
        # Try to create dividend column (yfinance data should include this)
        # If not available, create empty column and warn user
        if 'Dividends' not in result.columns:
            result['Dividends'] = 0.0
            print("Warning: No dividend data available. Dividends set to 0.")
            print("Available columns:", list(result.columns))
        
        return result
    
    def _run_single_backtest(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Run backtest for a single ticker with T+1 execution.
        Simplified logic: Signal detection on day T → Generate signal and execute trade on day T+1.
        
        Args:
            data: DataFrame with OHLCV data and signals
            
        Returns:
            DataFrame with backtest results
        """
        # Create a copy and prepare dividend data
        results = self._prepare_dividend_data(data)
        
        # Validate required signal columns
        if 'buy_signal' not in results.columns or 'sell_signal' not in results.columns:
            raise ValueError("Data must contain 'buy_signal' and 'sell_signal' columns")
        
        # Initialize result columns
        results['position'] = 0.0
        results['cash'] = float(self.initial_capital)
        results['equity'] = float(self.initial_capital)
        results['long_positions'] = 0.0
        results['short_positions'] = 0.0
        results['margin_used'] = 0.0
        results['margin_cash'] = 0.0  # Cumulative margin debt (always negative)
        results['returns'] = 0.0
        results['trade_count'] = 0
        results['win_count'] = 0
        results['loss_count'] = 0
        results['position_type'] = ''
        results['dividends_received'] = 0.0
        results['cumulative_dividends'] = 0.0
        results['daily_margin_cost'] = 0.0
        results['daily_short_borrow_cost'] = 0.0
        results['cumulative_borrowing_costs'] = 0.0
        
        # Simplified T+1: Remove execution tracking, just shift signals forward
        results['buy_signal_t1'] = results['buy_signal'].shift(1).fillna(0).astype(int)
        results['sell_signal_t1'] = results['sell_signal'].shift(1).fillna(0).astype(int)
        
        # Trading state variables
        position = 0.0
        cash = float(self.initial_capital)
        margin_cash = 0.0  # Cumulative margin debt (always negative)
        trade_count = 0
        win_count = 0
        loss_count = 0
        entry_price = 0.0
        position_type = 'long'
        cumulative_dividends = 0.0
        cumulative_borrowing_costs = 0.0
        
        # SUPER SIMPLE T+1 EXECUTION LOGIC
        for i in range(len(results)):
            current_price = results['Close'].iloc[i]
            current_date = results.index[i]
            current_dividend = results['Dividends'].iloc[i] if self.include_dividends else 0.0
            
            # Use open price for execution (T+1 logic)
            execution_price = results['Open'].iloc[i] if 'Open' in results.columns else current_price
            
            # STEP 1: Execute T+1 buy signals (signal was generated yesterday)
            if results['buy_signal_t1'].iloc[i] == 1:
                # Close any short position first
                if position < 0:
                    effective_price = execution_price * (1 + self.commission + self.slippage)
                    
                    # Calculate P&L on short position
                    short_pnl = abs(position) * (entry_price - effective_price)
                    
                    # Return margin collateral plus P&L
                    margin_collateral = abs(position) * entry_price / self.short_leverage
                    cash += margin_collateral + short_pnl
                    
                    # Track win/loss
                    if short_pnl > 0:
                        win_count += 1
                    else:
                        loss_count += 1
                    
                    position = 0.0
                    trade_count += 1
                
                # Enter long position
                if position <= 0:
                    effective_price = execution_price * (1 + self.commission + self.slippage)
                    current_portfolio_value = cash - margin_cash  # Net equity
                    
                    # Calculate desired position value based on leverage
                    # If leverage is 2.0, we want 2x portfolio value in stocks
                    desired_position_value = current_portfolio_value * self.position_sizing * self.long_leverage
                    
                    # Calculate how much we need to borrow
                    available_cash = cash
                    margin_to_borrow = max(0, desired_position_value - available_cash)
                    
                    # Actual position value (limited by available cash + margin)
                    position_value = min(desired_position_value, available_cash + margin_to_borrow)
                    shares = position_value / effective_price
                    
                    # Update cash and margin
                    cash_used = min(available_cash, position_value)
                    margin_borrowed = position_value - cash_used
                    
                    position = shares
                    cash -= cash_used
                    margin_cash -= margin_borrowed  # Add to margin debt (negative)
                    entry_price = effective_price
                    position_type = 'long'
                    trade_count += 1
            
            # STEP 2: Execute T+1 sell signals (signal was generated yesterday)
            elif results['sell_signal_t1'].iloc[i] == 1:
                # Close any long position first
                if position > 0:
                    effective_price = execution_price * (1 - self.commission - self.slippage)
                    
                    # Calculate total sale proceeds
                    sale_proceeds = position * effective_price
                    
                    # Pay back margin debt first if any
                    if margin_cash < 0:
                        margin_to_repay = min(-margin_cash, sale_proceeds)
                        margin_cash += margin_to_repay
                        sale_proceeds -= margin_to_repay
                    
                    # Remaining goes to cash
                    cash += sale_proceeds
                    
                    # Calculate P&L for win/loss tracking
                    total_cost = position * entry_price
                    total_proceeds = position * effective_price
                    if total_proceeds > total_cost:
                        win_count += 1
                    else:
                        loss_count += 1
                    
                    position = 0.0
                    trade_count += 1
                
                # Enter short position (if enabled)
                if position >= 0 and self.enable_short_selling:
                    effective_price = execution_price * (1 - self.commission - self.slippage)
                    current_portfolio_value = cash - margin_cash  # Net equity
                    
                    # Calculate desired short position value based on leverage
                    desired_position_value = current_portfolio_value * self.position_sizing * self.short_leverage
                    
                    # For short positions, we need to post margin collateral
                    margin_required = desired_position_value / self.short_leverage
                    
                    # Can only short if we have enough cash for margin
                    if cash >= margin_required:
                        shares = desired_position_value / effective_price
                        position = -shares
                        cash -= margin_required  # Post margin collateral
                        entry_price = execution_price
                        position_type = 'short'
                        trade_count += 1
            
            # STEP 3: Calculate daily costs and dividends
            daily_margin_cost = 0.0
            daily_short_borrow_cost = 0.0
            
            # Calculate margin interest on any outstanding margin debt
            if margin_cash < 0:
                daily_margin_rate = self._get_daily_margin_rate(current_date)
                daily_margin_cost = -margin_cash * daily_margin_rate
                margin_cash -= daily_margin_cost  # Add interest to margin debt
                cumulative_borrowing_costs += daily_margin_cost
            
            # Calculate short borrowing costs
            if position < 0:
                borrowed_value = abs(position) * current_price
                daily_short_borrow_rate = self._get_daily_short_borrow_rate()
                daily_short_borrow_cost = borrowed_value * daily_short_borrow_rate
                cash -= daily_short_borrow_cost
                cumulative_borrowing_costs += daily_short_borrow_cost
            
            # Process dividends
            dividend_impact = 0.0
            if position != 0 and current_dividend > 0:
                if position > 0:
                    dividend_impact = position * current_dividend
                    cash += dividend_impact
                else:
                    dividend_impact = abs(position) * current_dividend
                    cash -= dividend_impact
                    dividend_impact = -dividend_impact
                cumulative_dividends += dividend_impact
            
            # STEP 4: Calculate portfolio state
            long_position_value = 0.0
            short_position_value = 0.0
            
            if position > 0:
                long_position_value = position * current_price
                # Equity = cash + stock value + margin debt (negative)
                equity = cash + long_position_value + margin_cash
            elif position < 0:
                unrealized_pnl = abs(position) * (entry_price - current_price)
                margin_used = abs(position) * entry_price / self.short_leverage
                short_position_value = abs(position) * current_price
                equity = cash + margin_used + unrealized_pnl
            else:
                # No position: equity = cash + margin debt
                equity = cash + margin_cash
            
            # STEP 5: Update all result columns
            results.at[current_date, 'position'] = position
            results.at[current_date, 'cash'] = cash
            results.at[current_date, 'equity'] = equity
            results.at[current_date, 'returns'] = equity / self.initial_capital - 1
            results.at[current_date, 'trade_count'] = trade_count
            results.at[current_date, 'win_count'] = win_count
            results.at[current_date, 'loss_count'] = loss_count
            results.at[current_date, 'position_type'] = position_type
            results.at[current_date, 'dividends_received'] = dividend_impact
            results.at[current_date, 'cumulative_dividends'] = cumulative_dividends
            results.at[current_date, 'daily_margin_cost'] = daily_margin_cost
            results.at[current_date, 'daily_short_borrow_cost'] = daily_short_borrow_cost
            results.at[current_date, 'cumulative_borrowing_costs'] = cumulative_borrowing_costs
            results.at[current_date, 'long_positions'] = long_position_value
            results.at[current_date, 'short_positions'] = short_position_value
            results.at[current_date, 'margin_used'] = -margin_cash if margin_cash < 0 else 0.0  # Show as positive for visualization
            results.at[current_date, 'margin_cash'] = margin_cash  # Keep actual negative value
        
        # Calculate performance metrics
        results = self._calculate_performance_metrics(results)
        
        # Calculate buy & hold benchmark
        results = self._calculate_benchmark(results)
        
        return results
    
    def _calculate_benchmark(self, results: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate buy & hold benchmark performance with proper dividend treatment.
        
        Args:
            results: DataFrame with backtest results
            
        Returns:
            DataFrame with added benchmark columns
        """
        # Find the first signal date to align benchmark with strategy start
        first_signal_idx = 0
        buy_signal_col = 'buy_signal'
        if buy_signal_col in results.columns:
            buy_signals = results[results[buy_signal_col] == 1]
            if not buy_signals.empty:
                first_signal_idx = results.index.get_indexer([buy_signals.index[0]], method='nearest')[0]
        
        # Get price and dividend data from first signal onwards
        price_series = results['Close'].iloc[first_signal_idx:]
        initial_price = price_series.iloc[0]
        
        # Calculate benchmark with dividend reinvestment if dividends are enabled
        if self.include_dividends and 'Dividends' in results.columns:
            dividend_series = results['Dividends'].iloc[first_signal_idx:]
            
            # Simulate dividend reinvestment starting with initial_capital worth of shares
            benchmark_shares = self.initial_capital / initial_price
            benchmark_values = []
            
            for i, (date, price) in enumerate(price_series.items()):
                if i > 0 and dividend_series.iloc[i] > 0:
                    # Reinvest dividends into more shares
                    dividend_payment = benchmark_shares * dividend_series.iloc[i]
                    additional_shares = dividend_payment / price
                    benchmark_shares += additional_shares
                
                # Calculate current portfolio value
                portfolio_value = benchmark_shares * price
                benchmark_values.append(portfolio_value)
            
            # Create benchmark series aligned with strategy dates
            benchmark_equity = pd.Series(benchmark_values, index=price_series.index)
        else:
            # Price-only benchmark (no dividend reinvestment)
            benchmark_equity = price_series / initial_price * self.initial_capital
        
        # Initialize benchmark columns for the full results DataFrame
        results['benchmark_equity'] = self.initial_capital
        results['benchmark_returns'] = 0.0
        results['benchmark_drawdown'] = 0.0
        
        # Fill in the benchmark data from first signal onwards
        results.loc[benchmark_equity.index, 'benchmark_equity'] = benchmark_equity
        results.loc[benchmark_equity.index, 'benchmark_returns'] = benchmark_equity / self.initial_capital - 1
        
        # Calculate benchmark drawdown
        benchmark_peak = results['benchmark_equity'].cummax()
        results['benchmark_drawdown'] = (results['benchmark_equity'] - benchmark_peak) / benchmark_peak
        
        # Calculate benchmark performance metrics
        final_benchmark_return = results['benchmark_returns'].iloc[-1]
        
        # Calculate benchmark CAGR
        days = (results.index[-1] - results.index[0]).days
        years = max(days / 365.25, 0.01)
        results['benchmark_cagr'] = ((1 + final_benchmark_return) ** (1 / years)) - 1
        
        # Calculate benchmark max drawdown
        results['benchmark_max_drawdown'] = results['benchmark_drawdown'].min()
        
        # Calculate benchmark volatility and Sharpe ratio
        benchmark_daily_returns = results['benchmark_equity'].pct_change().dropna()
        if len(benchmark_daily_returns) > 0:
            benchmark_annual_vol = benchmark_daily_returns.std() * np.sqrt(252)
            if benchmark_annual_vol > 0:
                results['benchmark_sharpe_ratio'] = results['benchmark_cagr'] / benchmark_annual_vol
            else:
                results['benchmark_sharpe_ratio'] = np.nan
        else:
            results['benchmark_sharpe_ratio'] = np.nan
        
        return results
    
    def _run_portfolio_backtest(self, data_dict: Dict[str, pd.DataFrame], 
                             rebalance_method: str) -> pd.DataFrame:
        """
        Run backtest for a portfolio of tickers.
        
        Args:
            data_dict: Dictionary mapping tickers to DataFrames with signals
            rebalance_method: Portfolio rebalancing method
            
        Returns:
            DataFrame with portfolio backtest results
        """
        # Align all dataframes on the same index
        common_dates = pd.DatetimeIndex(sorted(set.intersection(
            *[set(df.index) for df in data_dict.values()])))
        
        # Create a portfolio results dataframe
        portfolio = pd.DataFrame(index=common_dates)
        portfolio['cash'] = float(self.initial_capital)
        portfolio['equity'] = float(self.initial_capital)
        portfolio['returns'] = 0.0
        
        # Initialize position tracking for each ticker
        positions = {ticker: 0.0 for ticker in data_dict}
        cash = float(self.initial_capital)
        
        # Portfolio implementation based on rebalance method
        if rebalance_method == 'buy_sell':
            # Each ticker gets signals independently
            for ticker, df in data_dict.items():
                # Prepare dividend data for this ticker
                df_with_dividends = self._prepare_dividend_data(df)
                
                # Create a column for this ticker's position
                portfolio[f'{ticker}_position'] = 0.0
                
                # Align dataframe to portfolio index
                aligned_df = df_with_dividends.reindex(portfolio.index)
                
                for i in range(len(portfolio)):
                    current_date = portfolio.index[i]
                    
                    if current_date not in aligned_df.index:
                        continue
                    
                    current_price = aligned_df.loc[current_date, 'Close']
                    current_dividend = aligned_df.loc[current_date, 'Dividends'] if 'Dividends' in aligned_df.columns and self.include_dividends else 0.0
                    
                    # Process dividends for existing position
                    if positions[ticker] > 0 and current_dividend > 0:
                        dividend_payment = positions[ticker] * current_dividend
                        cash += dividend_payment
                    
                    # Check for buy signals
                    if (aligned_df.loc[current_date, 'buy_signal'] == 1 and 
                        positions[ticker] == 0):
                        
                        # Allocate a portion of cash to this ticker
                        ticker_cash = cash / len(data_dict)
                        effective_price = current_price * (1 + self.commission + self.slippage)
                        positions[ticker] = ticker_cash / effective_price
                        cash -= ticker_cash
                    
                    # Check for sell signals
                    elif (aligned_df.loc[current_date, 'sell_signal'] == 1 and 
                          positions[ticker] > 0):
                        
                        effective_price = current_price * (1 - self.commission - self.slippage)
                        cash += positions[ticker] * effective_price
                        positions[ticker] = 0.0
                    
                    # Update position in portfolio
                    portfolio.at[current_date, f'{ticker}_position'] = positions[ticker]
            
            # Calculate portfolio equity and returns for each date
            for i in range(len(portfolio)):
                position_value = sum(
                    positions[ticker] * data_dict[ticker].loc[portfolio.index[i], 'Close'] 
                    for ticker in data_dict 
                    if portfolio.index[i] in data_dict[ticker].index and positions[ticker] > 0
                )
                
                portfolio.at[portfolio.index[i], 'cash'] = cash
                portfolio.at[portfolio.index[i], 'equity'] = cash + position_value
                portfolio.at[portfolio.index[i], 'returns'] = portfolio['equity'].iloc[i] / self.initial_capital - 1
        
        # Calculate portfolio performance metrics
        portfolio = self._calculate_performance_metrics(portfolio)
        
        # Calculate benchmark for portfolio (using first ticker's data as proxy)
        if data_dict:
            first_ticker = list(data_dict.keys())[0]
            first_ticker_data = data_dict[first_ticker]
            
            # Create a temporary results DataFrame with the first ticker's price data
            temp_results = portfolio.copy()
            temp_results['Close'] = first_ticker_data.reindex(portfolio.index)['Close']
            if 'Dividends' in first_ticker_data.columns:
                temp_results['Dividends'] = first_ticker_data.reindex(portfolio.index)['Dividends']
            else:
                temp_results['Dividends'] = 0.0
                
            # Add dummy signals for benchmark calculation
            temp_results['buy_signal'] = 0
            temp_results.at[temp_results.index[0], 'buy_signal'] = 1  # Buy at start
            
            # Calculate benchmark
            portfolio = self._calculate_benchmark(temp_results)
        
        return portfolio
    
    def _calculate_performance_metrics(self, results: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate performance metrics for backtest results.
        
        Args:
            results: DataFrame with equity curve data
            
        Returns:
            DataFrame with added performance metrics
        """
        # Calculate daily returns
        results['daily_return'] = results['equity'].pct_change()
        
        # Calculate drawdown series
        results['peak_equity'] = results['equity'].cummax()
        results['drawdown'] = (results['equity'] - results['peak_equity']) / results['peak_equity']
        
        # Calculate key metrics
        total_return = results['returns'].iloc[-1]
        
        # Handle case where we don't have enough data
        if len(results) <= 1:
            results['sharpe_ratio'] = np.nan
            results['sortino_ratio'] = np.nan
            results['max_drawdown'] = np.nan
            results['win_rate'] = np.nan
            results['cagr'] = np.nan
            return results
        
        # Calculate annualized metrics
        days = (results.index[-1] - results.index[0]).days
        years = max(days / 365.25, 0.01)  # Avoid division by zero
        
        # Compound Annual Growth Rate
        results['cagr'] = ((1 + total_return) ** (1 / years)) - 1
        
        # Risk metrics
        daily_returns = results['daily_return'].dropna()
        
        if len(daily_returns) > 0:
            # Annualized volatility
            annual_vol = daily_returns.std() * np.sqrt(252)
            
            # Sharpe Ratio (assuming risk-free rate of 0 for simplicity)
            if annual_vol > 0:
                results['sharpe_ratio'] = results['cagr'] / annual_vol
            else:
                results['sharpe_ratio'] = np.nan
            
            # Sortino Ratio (downside deviation only)
            downside_returns = daily_returns[daily_returns < 0]
            if len(downside_returns) > 0:
                downside_vol = downside_returns.std() * np.sqrt(252)
                if downside_vol > 0:
                    results['sortino_ratio'] = results['cagr'] / downside_vol
                else:
                    results['sortino_ratio'] = np.nan
            else:
                results['sortino_ratio'] = np.nan
        else:
            results['sharpe_ratio'] = np.nan
            results['sortino_ratio'] = np.nan
        
        # Maximum drawdown
        results['max_drawdown'] = results['drawdown'].min()
        
        # Win rate (if trade information is available)
        if 'trade_count' in results.columns and 'win_count' in results.columns:
            total_trades = results['trade_count'].iloc[-1]
            if total_trades > 0:
                results['win_rate'] = results['win_count'].iloc[-1] / total_trades
            else:
                results['win_rate'] = np.nan
        
        return results