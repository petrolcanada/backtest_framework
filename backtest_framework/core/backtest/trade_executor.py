"""
Trade execution module for handling buy/sell signals and position management.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from .portfolio_manager import PortfolioManager
from .cost_calculator import CostCalculator
from .trade_tracker import TradeTracker


class TradeExecutor:
    """
    Handles trade execution logic including signal processing and position management.
    """
    
    def __init__(self, portfolio_manager: PortfolioManager, cost_calculator: CostCalculator,
                 enable_short_selling: bool = False):
        """
        Initialize trade executor.
        
        Args:
            portfolio_manager: Portfolio manager instance
            cost_calculator: Cost calculator instance
            enable_short_selling: Whether short selling is enabled
        """
        self.portfolio_manager = portfolio_manager
        self.cost_calculator = cost_calculator
        self.enable_short_selling = enable_short_selling
        
        # Enhanced trade tracking (consolidated system)
        self.trade_tracker = TradeTracker()
        
        # Dividend and cost tracking
        self.cumulative_dividends = 0.0
        self.cumulative_borrowing_costs = 0.0
    
    def prepare_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare T+1 execution signals.
        
        Args:
            data: DataFrame with buy_signal and sell_signal columns
            
        Returns:
            DataFrame with T+1 shifted signals
        """
        data = data.copy()
        
        # Validate signal columns
        if 'buy_signal' not in data.columns or 'sell_signal' not in data.columns:
            raise ValueError("Data must contain 'buy_signal' and 'sell_signal' columns")
        
        # Shift signals for T+1 execution
        data['buy_signal_t1'] = data['buy_signal'].shift(1).fillna(0).astype(int)
        data['sell_signal_t1'] = data['sell_signal'].shift(1).fillna(0).astype(int)
        
        return data
    
    def execute_buy_signal(self, current_price: float, execution_price: float, current_date: pd.Timestamp = None, current_equity: float = None) -> Dict[str, Any]:
        """
        Execute buy signal.
        
        Args:
            current_price: Current market price
            execution_price: Price for execution (usually Open price)
            current_date: Current date for tracking
            current_equity: Current portfolio equity
            
        Returns:
            Dictionary with execution results
        """
        result = {
            'executed': False,
            'trade_occurred': False,
            'pnl': 0.0,
            'trade_type': ''
        }
        
        effective_price = self.cost_calculator.calculate_execution_price(execution_price, True)
        
        # Close short position first if exists
        if self.portfolio_manager.position < 0:
            # Calculate exit cost for the short position
            shares_to_cover = abs(self.portfolio_manager.position)
            cover_cost = shares_to_cover * effective_price
            
            # Record trade completion using enhanced tracking
            if current_date and self.trade_tracker.current_trade:
                self.trade_tracker.close_trade(
                    date=current_date,
                    price=execution_price,
                    exit_proceeds=cover_cost
                )
            
            pnl = self.portfolio_manager._close_short_position(effective_price)
            result['trade_occurred'] = True
            result['pnl'] = pnl
            result['trade_type'] = 'close_short'
        
        # Enter long position
        if self.portfolio_manager.position <= 0:
            # Calculate position details before entering
            current_portfolio_value = self.portfolio_manager.get_current_equity(current_price)
            position_value = current_portfolio_value * self.portfolio_manager.position_sizing * self.portfolio_manager.long_leverage
            shares = position_value / effective_price
            entry_cost = shares * effective_price
            
            success = self.portfolio_manager.enter_long_position(effective_price)
            if success:
                # Open new trade using enhanced tracking
                if current_date:
                    self.trade_tracker.open_trade(
                        date=current_date,
                        price=execution_price,
                        position_size=shares,
                        trade_type='long',
                        entry_cost=entry_cost,
                        entry_equity=current_equity
                    )
                
                result['executed'] = True
                if not result['trade_occurred']:
                    result['trade_type'] = 'open_long'
        
        return result
    
    def update_trade_tracking(self, current_date: pd.Timestamp, current_price: float, current_equity: float) -> None:
        """
        Update trade tracking with current market data.
        
        Args:
            current_date: Current date
            current_price: Current market price
            current_equity: Current portfolio equity
        """
        # Update MAE/MFE for current trade
        self.trade_tracker.update_trade_excursions(current_price)
        
        # Update equity tracking and drawdown metrics
        self.trade_tracker.update_equity_tracking(current_date, current_equity)
    
    def execute_sell_signal(self, current_price: float, execution_price: float, current_date: pd.Timestamp = None, current_equity: float = None) -> Dict[str, Any]:
        """
        Execute sell signal.
        
        Args:
            current_price: Current market price
            execution_price: Price for execution (usually Open price)
            current_date: Current date for tracking
            current_equity: Current portfolio equity
            
        Returns:
            Dictionary with execution results
        """
        result = {
            'executed': False,
            'trade_occurred': False,
            'pnl': 0.0,
            'trade_type': ''
        }
        
        effective_price = self.cost_calculator.calculate_execution_price(execution_price, False)
        
        # Close long position first if exists
        if self.portfolio_manager.position > 0:
            # Calculate exit proceeds
            shares = self.portfolio_manager.position
            exit_proceeds = shares * effective_price
            
            # Record trade completion using enhanced tracking
            if current_date and self.trade_tracker.current_trade:
                self.trade_tracker.close_trade(
                    date=current_date,
                    price=execution_price,
                    exit_proceeds=exit_proceeds,
                    exit_equity=current_equity
                )
            
            pnl = self.portfolio_manager._close_long_position(effective_price)
            result['trade_occurred'] = True
            result['pnl'] = pnl
            result['trade_type'] = 'close_long'
        
        # Enter short position if enabled
        if (self.portfolio_manager.position >= 0 and self.enable_short_selling):
            # Calculate position details before entering
            current_portfolio_value = self.portfolio_manager.get_current_equity(current_price)
            short_value = current_portfolio_value * self.portfolio_manager.position_sizing * self.portfolio_manager.short_leverage
            shares_to_short = short_value / effective_price
            
            success = self.portfolio_manager.enter_short_position(effective_price)
            if success:
                # For shorts, the "entry cost" is actually proceeds received
                entry_proceeds = shares_to_short * effective_price
                
                # Open new trade using enhanced tracking
                if current_date:
                    self.trade_tracker.open_trade(
                        date=current_date,
                        price=execution_price,
                        position_size=shares_to_short,
                        trade_type='short',
                        entry_cost=entry_proceeds,  # For shorts, this represents proceeds
                        entry_equity=current_equity
                    )
                
                result['executed'] = True
                if not result['trade_occurred']:
                    result['trade_type'] = 'open_short'
        
        return result
    
    def process_daily_costs_and_dividends(self, current_price: float, current_date: pd.Timestamp,
                                         current_dividend: float = 0.0) -> Dict[str, float]:
        """
        Process daily costs and dividend payments.
        
        Args:
            current_price: Current price of the asset
            current_date: Current date
            current_dividend: Dividend per share for current date
            
        Returns:
            Dictionary with cost and dividend information
        """
        result = {
            'dividend_impact': 0.0,
            'margin_cost': 0.0,
            'short_borrow_cost': 0.0,
            'mmf_interest': 0.0
        }
        
        # Process dividends
        if current_dividend > 0:
            dividend_impact = self.portfolio_manager.process_dividend(current_dividend)
            result['dividend_impact'] = dividend_impact
            self.cumulative_dividends += dividend_impact
            
            # Track dividends in current trade
            self.trade_tracker.add_dividend(dividend_impact)
        
        # Apply MMF interest to cash components
        daily_mmf_rate = self.cost_calculator.get_daily_mmf_rate(current_date)
        mmf_interest = self.portfolio_manager.apply_mmf_interest(daily_mmf_rate)
        result['mmf_interest'] = mmf_interest
        
        # Calculate and apply margin costs
        if self.portfolio_manager.margin_cash < 0:
            daily_margin_rate = self.cost_calculator.get_daily_margin_rate(current_date)
            margin_cost = self.portfolio_manager.apply_margin_cost(daily_margin_rate)
            result['margin_cost'] = margin_cost
            self.cumulative_borrowing_costs += margin_cost
            
            # Track borrowing costs in current trade
            self.trade_tracker.add_borrowing_cost(margin_cost)
        
        # Calculate and apply short borrowing costs
        if self.portfolio_manager.position < 0:
            daily_borrow_rate = self.cost_calculator.get_daily_short_borrow_rate(current_date)
            short_cost = self.portfolio_manager.apply_short_borrow_cost(
                current_price, daily_borrow_rate)
            result['short_borrow_cost'] = short_cost
            self.cumulative_borrowing_costs += short_cost
            
            # Track borrowing costs in current trade
            self.trade_tracker.add_borrowing_cost(short_cost)
        
        return result
    

    
    def get_trade_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive trade statistics from enhanced tracker.
        
        Returns:
            Dictionary with all trade statistics
        """
        # Get all stats from enhanced trade tracker
        enhanced_stats = self.trade_tracker.get_stats()
        
        return {
            # Core trade statistics (from enhanced tracker)
            'trade_count': enhanced_stats['total_trades'],
            'win_count': enhanced_stats['winning_trades'],
            'loss_count': enhanced_stats['losing_trades'],
            'win_rate': enhanced_stats['win_rate'],
            
            # Cost tracking (maintained at executor level)
            'cumulative_dividends': self.cumulative_dividends,
            'cumulative_borrowing_costs': self.cumulative_borrowing_costs,
            
            # Enhanced P&L metrics
            'avg_win': enhanced_stats['avg_win'],
            'avg_loss': enhanced_stats['avg_loss'],
            'profit_factor': enhanced_stats['profit_factor'],
            'best_trade': enhanced_stats['best_trade'],
            'worst_trade': enhanced_stats['worst_trade'],
            'total_pnl': enhanced_stats['total_pnl'],
            'avg_pnl': enhanced_stats['avg_pnl'],
            'avg_pnl_percent': enhanced_stats['avg_pnl_percent'],
            
            # Drawdown metrics
            'max_drawdown': enhanced_stats['max_drawdown'],
            'current_drawdown': enhanced_stats['current_drawdown'],
            'max_drawdown_duration': enhanced_stats['max_drawdown_duration'],
            
            # Trade execution metrics
            'avg_mae': enhanced_stats['avg_mae'],
            'avg_mfe': enhanced_stats['avg_mfe'],
            'max_consecutive_wins': enhanced_stats['max_consecutive_wins'],
            'max_consecutive_losses': enhanced_stats['max_consecutive_losses'],
            'equity_peak': enhanced_stats['equity_peak'],
            'avg_trade_duration': enhanced_stats['avg_trade_duration']
        }
    
    def get_trade_log(self) -> pd.DataFrame:
        """
        Get detailed trade log.
        
        Returns:
            DataFrame with all trade details
        """
        return self.trade_tracker.get_trade_log()
    
    def get_equity_curve(self) -> pd.DataFrame:
        """
        Get equity curve with drawdown information.
        
        Returns:
            DataFrame with equity and drawdown data
        """
        return self.trade_tracker.get_equity_curve()
    
    def get_drawdown_periods(self, min_duration_days: int = 5) -> pd.DataFrame:
        """
        Get detailed information about drawdown periods.
        
        Args:
            min_duration_days: Minimum duration to consider a significant drawdown
            
        Returns:
            DataFrame with drawdown period details
        """
        equity_curve = self.get_equity_curve()
        if equity_curve.empty:
            return pd.DataFrame()
            
        drawdown_periods = []
        in_drawdown = False
        start_date = None
        start_equity = None
        min_equity = None
        
        for date, row in equity_curve.iterrows():
            if not in_drawdown and row['Drawdown'] < -0.001:  # Start drawdown (> 0.1%)
                in_drawdown = True
                start_date = date
                start_equity = row['Peak']
                min_equity = row['Equity']
            elif in_drawdown:
                if row['Equity'] < min_equity:
                    min_equity = row['Equity']
                
                if abs(row['Drawdown']) < 0.001:  # End drawdown
                    duration = (date - start_date).days
                    if duration >= min_duration_days:
                        max_dd = (min_equity - start_equity) / start_equity
                        drawdown_periods.append({
                            'Start Date': start_date,
                            'End Date': date,
                            'Duration (days)': duration,
                            'Peak Equity': start_equity,
                            'Trough Equity': min_equity,
                            'Max Drawdown': max_dd,
                            'Max Drawdown %': max_dd * 100
                        })
                    in_drawdown = False
        
        return pd.DataFrame(drawdown_periods)
    
    def reset(self):
        """Reset executor state."""
        # Reset enhanced tracker (handles all trade statistics)
        self.trade_tracker.reset()
        
        # Reset cost tracking
        self.cumulative_dividends = 0.0
        self.cumulative_borrowing_costs = 0.0
        
        # Reset portfolio state
        self.portfolio_manager.reset()
    
    def get_enhanced_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary including all enhanced metrics.
        
        Returns:
            Dictionary with comprehensive trade and performance metrics
        """
        stats = self.get_trade_stats()
        equity_curve = self.get_equity_curve()
        
        summary = {
            # Trade Statistics
            'total_trades': stats['trade_count'],
            'winning_trades': stats['win_count'],
            'losing_trades': stats['loss_count'],
            'win_rate_pct': stats['win_rate'] * 100,
            'profit_factor': stats['profit_factor'],
            
            # P&L Metrics
            'total_pnl': stats['total_pnl'],
            'avg_pnl': stats['avg_pnl'],
            'avg_pnl_pct': stats['avg_pnl_percent'] * 100,
            'avg_win_pct': stats['avg_win'] * 100 if stats['avg_win'] else 0,
            'avg_loss_pct': stats['avg_loss'] * 100 if stats['avg_loss'] else 0,
            'best_trade_pct': stats['best_trade'] * 100,
            'worst_trade_pct': stats['worst_trade'] * 100,
            
            # Drawdown Metrics
            'max_drawdown_pct': stats['max_drawdown'] * 100,
            'current_drawdown_pct': stats['current_drawdown'] * 100,
            'max_drawdown_duration_days': stats['max_drawdown_duration'],
            'equity_peak': stats['equity_peak'],
            
            # Execution Metrics
            'avg_mae': stats['avg_mae'],
            'avg_mfe': stats['avg_mfe'],
            'max_consecutive_wins': stats['max_consecutive_wins'],
            'max_consecutive_losses': stats['max_consecutive_losses'],
            'avg_trade_duration_days': stats['avg_trade_duration'],
            
            # Cost Metrics
            'cumulative_dividends': stats['cumulative_dividends'],
            'cumulative_borrowing_costs': stats['cumulative_borrowing_costs'],
        }
        
        # Add equity curve statistics if available
        if not equity_curve.empty:
            summary.update({
                'total_equity_points': len(equity_curve),
                'final_equity': equity_curve['Equity'].iloc[-1],
                'equity_volatility': equity_curve['Equity'].pct_change().std() * np.sqrt(252),
                'drawdown_periods': len(self.get_drawdown_periods()),
            })
        
        return summary
