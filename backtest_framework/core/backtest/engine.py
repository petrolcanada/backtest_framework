"""
Modular Backtest Engine - Main orchestrator for backtesting strategies.
"""
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from backtest_framework.core.strategies.base import BaseStrategy
from .portfolio_manager import PortfolioManager
from .cost_calculator import CostCalculator
from .trade_executor import TradeExecutor
from .performance_calculator import PerformanceCalculator
from .data_validator import DataValidator


class BacktestEngine:
    """
    Modular backtest engine that orchestrates portfolio management, trade execution,
    cost calculation, and performance analysis.
    """
    
    def __init__(self, initial_capital: float = 10000.0, commission: float = 0.001,
                 slippage: float = 0.0, leverage: Union[float, Dict[str, float]] = 1.0, 
                 enable_short_selling: bool = False, position_sizing: float = 1.0,
                 include_dividends: bool = True, short_borrow_rate: float = 0.02,
                 use_margin_costs: bool = True):
        """
        Initialize the modular backtest engine.
        
        Args:
            initial_capital: Starting capital for the backtest
            commission: Commission rate as a decimal (e.g., 0.001 = 0.1%)
            slippage: Slippage as a decimal (e.g., 0.001 = 0.1%)
            leverage: Leverage multiplier. Can be float or dict with 'long'/'short' keys
            enable_short_selling: Whether to allow short selling on sell signals
            position_sizing: Fraction of capital to use per trade (0.5 = 50% of capital)
            include_dividends: Whether to include dividend payments in calculations
            short_borrow_rate: Annual borrowing rate for short positions as decimal (default: 0.02 for 2%)
            use_margin_costs: Whether to include margin interest costs
        """
        self.initial_capital = initial_capital
        self.include_dividends = include_dividends
        
        # Parse leverage configuration
        if isinstance(leverage, dict):
            long_leverage = leverage.get('long', 1.0)
            short_leverage = leverage.get('short', 1.0)
        else:
            long_leverage = leverage
            short_leverage = leverage
        
        # Initialize modular components
        self.portfolio_manager = PortfolioManager(
            initial_capital=initial_capital,
            position_sizing=position_sizing,
            long_leverage=long_leverage,
            short_leverage=short_leverage
        )
        
        self.cost_calculator = CostCalculator(
            commission=commission,
            slippage=slippage,
            short_borrow_rate=short_borrow_rate,
            use_margin_costs=use_margin_costs
        )
        
        self.trade_executor = TradeExecutor(
            portfolio_manager=self.portfolio_manager,
            cost_calculator=self.cost_calculator,
            enable_short_selling=enable_short_selling
        )
        
        self.performance_calculator = PerformanceCalculator(initial_capital)
        
        self.data_validator = DataValidator(include_dividends=include_dividends)
        
        # Store configuration for access by visualization components
        self.commission = commission
        self.slippage = slippage
        self.long_leverage = long_leverage
        self.short_leverage = short_leverage
        self.enable_short_selling = enable_short_selling
        self.position_sizing = position_sizing
        self.short_borrow_rate = short_borrow_rate
        self.use_margin_costs = use_margin_costs
        
        # Risk managers
        self.risk_managers = []
    
    def add_risk_manager(self, risk_manager):
        """
        Add a risk management module to the backtest engine.
        
        Args:
            risk_manager: Risk management module to add
        """
        self.risk_managers.append(risk_manager)
    
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
        # Validate and prepare data
        data_dict = self.data_validator.validate_and_prepare_data(data, start_date, end_date)
        
        # Load Federal Funds Rate data for cost calculations
        actual_start_date, actual_end_date = self.data_validator.get_date_range(data_dict)
        if actual_start_date and actual_end_date:
            self._load_ffr_data(actual_start_date, actual_end_date)
        
        # Generate signals for each ticker
        signal_dfs = self._generate_signals(strategy, data_dict)
        
        # Run appropriate backtest based on number of tickers
        if len(signal_dfs) > 1:
            return self._run_portfolio_backtest(signal_dfs, portfolio_rebalance)
        else:
            ticker = list(signal_dfs.keys())[0]
            return self._run_single_backtest(signal_dfs[ticker])
    
    def _load_ffr_data(self, start_date: str, end_date: str):
        """
        Load Federal Funds Rate data for margin cost calculations.
        
        Args:
            start_date: Start date for FFR data
            end_date: End date for FFR data
        """
        try:
            from backtest_framework.core.data.loader import DataLoader
            loader = DataLoader()
            ffr_data = loader.load_fed_funds_rate(start_date=start_date)
            
            # Filter to backtest period
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            ffr_data = ffr_data[(ffr_data.index >= start_dt) & (ffr_data.index <= end_dt)]
            
            self.cost_calculator.load_ffr_data(ffr_data)
            print(f"Loaded FFR data from {ffr_data.index.min()} to {ffr_data.index.max()}")
        except Exception as e:
            print(f"Warning: Could not load FFR data: {e}. Using default margin costs.")
    
    def _generate_signals(self, strategy: BaseStrategy, 
                         data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Generate trading signals for each ticker.
        
        Args:
            strategy: Trading strategy
            data_dict: Dictionary of price data
            
        Returns:
            Dictionary of DataFrames with signals
        """
        signal_dfs = {}
        
        for ticker, df in data_dict.items():
            # Generate signals using strategy
            signal_df = strategy.run(df)
            
            # Validate signals
            signal_df = self.data_validator.validate_signals(signal_df, ticker)
            
            # Apply risk management
            for risk_manager in self.risk_managers:
                signal_df = risk_manager.apply(signal_df)
            
            signal_dfs[ticker] = signal_df
        
        return signal_dfs
    
    def _run_single_backtest(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Run backtest for a single ticker.
        
        Args:
            data: DataFrame with OHLCV data and signals
            
        Returns:
            DataFrame with backtest results
        """
        # Prepare signals for T+1 execution
        results = self.trade_executor.prepare_signals(data)
        
        # Initialize result columns
        self._initialize_result_columns(results)
        
        # Reset components for fresh backtest
        self.trade_executor.reset()
        
        # Find first signal date to start applying costs/interest
        first_signal_date = self._find_first_signal_date(results)
        strategy_active = False
        
        print(f"Strategy will become active after first signal: {first_signal_date.strftime('%Y-%m-%d') if first_signal_date else 'No signals found'}")
        
        # Main backtest loop
        for i in range(len(results)):
            current_date = results.index[i]
            current_price = results['Close'].iloc[i]
            current_dividend = results['Dividends'].iloc[i] if self.include_dividends else 0.0
            
            # Use Open price for execution (T+1 logic)
            execution_price = results['Open'].iloc[i] if 'Open' in results.columns else current_price
            
            # Execute T+1 buy signals
            if results['buy_signal_t1'].iloc[i] == 1:
                self.trade_executor.execute_buy_signal(current_price, execution_price)
            
            # Execute T+1 sell signals
            elif results['sell_signal_t1'].iloc[i] == 1:
                self.trade_executor.execute_sell_signal(current_price, execution_price)
            
            # Check if strategy becomes active AFTER processing signals
            # This ensures costs start applying the day after first signal
            if not strategy_active and first_signal_date and current_date > first_signal_date:
                strategy_active = True
                print(f"Strategy activated on {current_date.strftime('%Y-%m-%d')} - costs/interest will now apply (day after first signal)")
            
            # Process daily costs and dividends ONLY if strategy is active
            if strategy_active:
                daily_costs = self.trade_executor.process_daily_costs_and_dividends(
                    current_price, current_date, current_dividend)
            else:
                # Before strategy activation: no costs, no interest
                daily_costs = {
                    'dividend_impact': 0.0,
                    'margin_cost': 0.0,
                    'short_borrow_cost': 0.0,
                    'mmf_interest': 0.0
                }
            
            # Update result columns
            self._update_result_row(results, i, current_date, current_price, daily_costs)
        
        # Calculate performance metrics
        results = self.performance_calculator.calculate_returns_and_drawdown(results)
        results = self.performance_calculator.calculate_performance_metrics(results)
        results = self.performance_calculator.calculate_benchmark_performance(
            results, self.include_dividends)
        
        return results
    
    def _find_first_signal_date(self, results: pd.DataFrame) -> Optional[pd.Timestamp]:
        """
        Find the first signal date (buy or sell) in the results.
        
        Args:
            results: DataFrame with backtest results
            
        Returns:
            First signal date or None if no signals found
        """
        first_signal_date = None
        
        # Check for buy signals
        if 'buy_signal' in results.columns:
            buy_signals = results[results['buy_signal'] == 1]
            if not buy_signals.empty:
                first_signal_date = buy_signals.index[0]
        
        # Check for sell signals
        if 'sell_signal' in results.columns:
            sell_signals = results[results['sell_signal'] == 1]
            if not sell_signals.empty:
                first_sell_date = sell_signals.index[0]
                if first_signal_date is None or first_sell_date < first_signal_date:
                    first_signal_date = first_sell_date
        
        return first_signal_date
    
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
        # For now, implement basic equal-weight portfolio
        # This can be extended with more sophisticated portfolio methods
        
        # Align all dataframes on common dates
        alignment_info = self.data_validator.check_data_alignment(data_dict)
        if not alignment_info['aligned']:
            raise ValueError("Cannot create portfolio: no common dates between tickers")
        
        common_dates = pd.DatetimeIndex(sorted(set.intersection(
            *[set(df.index) for df in data_dict.values()])))
        
        # Create portfolio results dataframe
        portfolio = pd.DataFrame(index=common_dates)
        self._initialize_portfolio_columns(portfolio)
        
        # Simple equal-weight portfolio implementation
        if rebalance_method == 'buy_sell':
            portfolio = self._run_equal_weight_portfolio(data_dict, portfolio)
        
        # Calculate portfolio performance metrics
        portfolio = self.performance_calculator.calculate_returns_and_drawdown(portfolio)
        portfolio = self.performance_calculator.calculate_performance_metrics(portfolio)
        
        # Calculate benchmark using first ticker as proxy
        if data_dict:
            first_ticker = list(data_dict.keys())[0]
            portfolio = self._add_portfolio_benchmark(portfolio, data_dict[first_ticker])
        
        return portfolio
    
    def _initialize_result_columns(self, results: pd.DataFrame):
        """Initialize all result columns for single ticker backtest."""
        results['position'] = 0.0
        results['cash'] = float(self.initial_capital)
        results['original_cash'] = float(self.initial_capital)
        results['short_proceeds'] = 0.0
        results['equity'] = float(self.initial_capital)
        results['long_positions'] = 0.0
        results['short_positions'] = 0.0
        results['margin_cash'] = 0.0  # Always negative for margin debt
        results['returns'] = 0.0
        results['trade_count'] = 0
        results['win_count'] = 0
        results['loss_count'] = 0
        results['position_type'] = ''
        results['dividends_received'] = 0.0
        results['cumulative_dividends'] = 0.0
        results['daily_margin_cost'] = 0.0
        results['daily_short_borrow_cost'] = 0.0
        results['daily_mmf_interest'] = 0.0
        results['cumulative_borrowing_costs'] = 0.0
        results['cumulative_mmf_interest'] = 0.0
    
    def _initialize_portfolio_columns(self, portfolio: pd.DataFrame):
        """Initialize columns for portfolio backtest."""
        portfolio['cash'] = float(self.initial_capital)
        portfolio['equity'] = float(self.initial_capital)
        portfolio['returns'] = 0.0
    
    def _update_result_row(self, results: pd.DataFrame, i: int, current_date: pd.Timestamp,
                          current_price: float, daily_costs: Dict[str, float]):
        """Update a single row of results with current portfolio state."""
        # Get current portfolio state
        equity = self.portfolio_manager.get_current_equity(current_price)
        position_values = self.portfolio_manager.get_position_value(current_price)
        trade_stats = self.trade_executor.get_trade_stats()
        
        # Track cumulative MMF interest
        cumulative_mmf = results['cumulative_mmf_interest'].iloc[i-1] if i > 0 else 0.0
        cumulative_mmf += daily_costs.get('mmf_interest', 0.0)
        
        # Update all columns
        results.at[current_date, 'position'] = self.portfolio_manager.position
        results.at[current_date, 'cash'] = self.portfolio_manager.cash
        results.at[current_date, 'original_cash'] = self.portfolio_manager.original_cash
        results.at[current_date, 'short_proceeds'] = self.portfolio_manager.short_proceeds
        results.at[current_date, 'equity'] = equity
        results.at[current_date, 'long_positions'] = position_values['long_value']
        results.at[current_date, 'short_positions'] = position_values['short_value']
        results.at[current_date, 'margin_cash'] = self.portfolio_manager.margin_cash  # Always negative for debt
        results.at[current_date, 'trade_count'] = trade_stats['trade_count']
        results.at[current_date, 'win_count'] = trade_stats['win_count']
        results.at[current_date, 'loss_count'] = trade_stats['loss_count']
        results.at[current_date, 'position_type'] = self.portfolio_manager.position_type
        results.at[current_date, 'dividends_received'] = daily_costs['dividend_impact']
        results.at[current_date, 'cumulative_dividends'] = trade_stats['cumulative_dividends']
        results.at[current_date, 'daily_margin_cost'] = daily_costs['margin_cost']
        results.at[current_date, 'daily_short_borrow_cost'] = daily_costs['short_borrow_cost']
        results.at[current_date, 'daily_mmf_interest'] = daily_costs.get('mmf_interest', 0.0)
        results.at[current_date, 'cumulative_borrowing_costs'] = trade_stats['cumulative_borrowing_costs']
        results.at[current_date, 'cumulative_mmf_interest'] = cumulative_mmf
    
    def _run_equal_weight_portfolio(self, data_dict: Dict[str, pd.DataFrame],
                                   portfolio: pd.DataFrame) -> pd.DataFrame:
        """Run equal-weight portfolio backtest."""
        # This is a simplified implementation
        # In practice, you'd want more sophisticated portfolio management
        
        positions = {ticker: 0.0 for ticker in data_dict}
        cash = float(self.initial_capital)
        
        for i in range(len(portfolio)):
            current_date = portfolio.index[i]
            total_position_value = 0.0
            
            # Process each ticker
            for ticker, df in data_dict.items():
                if current_date not in df.index:
                    continue
                
                current_price = df.loc[current_date, 'Close']
                current_dividend = df.loc[current_date, 'Dividends'] if 'Dividends' in df.columns else 0.0
                
                # Process dividends
                if positions[ticker] > 0 and current_dividend > 0:
                    dividend_payment = positions[ticker] * current_dividend
                    cash += dividend_payment
                
                # Check for signals (simplified)
                if df.loc[current_date, 'buy_signal'] == 1 and positions[ticker] == 0:
                    # Allocate equal portion of cash to this ticker
                    ticker_allocation = cash / len(data_dict)
                    effective_price = self.cost_calculator.calculate_execution_price(current_price, True)
                    positions[ticker] = ticker_allocation / effective_price
                    cash -= ticker_allocation
                
                elif df.loc[current_date, 'sell_signal'] == 1 and positions[ticker] > 0:
                    effective_price = self.cost_calculator.calculate_execution_price(current_price, False)
                    cash += positions[ticker] * effective_price
                    positions[ticker] = 0.0
                
                # Add to total position value
                total_position_value += positions[ticker] * current_price
            
            # Update portfolio state
            equity = cash + total_position_value
            portfolio.at[current_date, 'cash'] = cash
            portfolio.at[current_date, 'equity'] = equity
        
        return portfolio
    
    def _add_portfolio_benchmark(self, portfolio: pd.DataFrame,
                                first_ticker_data: pd.DataFrame) -> pd.DataFrame:
        """Add benchmark data to portfolio results using first ticker as proxy."""
        # Create temporary results with first ticker's price data
        temp_results = portfolio.copy()
        temp_results['Close'] = first_ticker_data.reindex(portfolio.index)['Close']
        
        if 'Dividends' in first_ticker_data.columns:
            temp_results['Dividends'] = first_ticker_data.reindex(portfolio.index)['Dividends']
        else:
            temp_results['Dividends'] = 0.0
        
        # Add dummy signals for benchmark calculation
        temp_results['buy_signal'] = 0
        if len(temp_results) > 0:
            temp_results.at[temp_results.index[0], 'buy_signal'] = 1  # Buy at start
        
        # Calculate benchmark
        portfolio_with_benchmark = self.performance_calculator.calculate_benchmark_performance(
            temp_results, self.include_dividends)
        
        # Copy benchmark columns back to portfolio
        benchmark_columns = [col for col in portfolio_with_benchmark.columns 
                           if col.startswith('benchmark_')]
        for col in benchmark_columns:
            portfolio[col] = portfolio_with_benchmark[col]
        
        return portfolio
    
    def get_summary_stats(self, results: pd.DataFrame) -> Dict[str, Any]:
        """Get summary statistics from backtest results."""
        return self.performance_calculator.get_summary_stats(results)
    
    def reset(self):
        """Reset the engine for a new backtest."""
        self.portfolio_manager.reset()
        self.trade_executor.reset()
        self.risk_managers = []
