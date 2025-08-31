"""
Enhanced trade tracking module for accurate P&L and win rate calculation.
"""
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Trade:
    """Represents a single trade from entry to exit."""
    entry_date: pd.Timestamp
    exit_date: Optional[pd.Timestamp]
    entry_price: float
    exit_price: Optional[float]
    position_size: float
    trade_type: str  # 'long' or 'short'
    entry_cost: float  # Total cost including commission/slippage
    exit_proceeds: Optional[float]  # Total proceeds after commission/slippage
    pnl: Optional[float]
    pnl_percent: Optional[float]
    is_closed: bool = False
    dividends_received: float = 0.0
    borrowing_costs: float = 0.0  # For margin interest or short borrowing
    max_adverse_excursion: Optional[float] = None  # MAE - maximum loss during trade
    max_favorable_excursion: Optional[float] = None  # MFE - maximum gain during trade
    entry_equity: Optional[float] = None  # Portfolio equity at entry
    exit_equity: Optional[float] = None  # Portfolio equity at exit
    # Drawdown metrics during trade
    max_trade_drawdown: Optional[float] = None  # Max portfolio drawdown experienced during trade
    equity_peak_during_trade: Optional[float] = None  # Highest equity during this trade


class TradeTracker:
    """
    Enhanced trade tracking for accurate P&L and win rate calculation.
    """
    
    def __init__(self):
        """Initialize trade tracker."""
        self.trades: List[Trade] = []
        self.current_trade: Optional[Trade] = None
        
        # Drawdown tracking
        self.equity_peak: float = 0.0
        self.current_drawdown: float = 0.0
        self.max_drawdown: float = 0.0
        self.max_drawdown_duration: int = 0
        self.current_drawdown_duration: int = 0
        self.drawdown_start_date: Optional[pd.Timestamp] = None
        self.equity_history: List[Tuple[pd.Timestamp, float]] = []
        
    def open_trade(self, date: pd.Timestamp, price: float, position_size: float, 
                   trade_type: str, entry_cost: float, entry_equity: float = None) -> None:
        """
        Record a new trade entry.
        
        Args:
            date: Entry date
            price: Entry price
            position_size: Number of shares (positive for both long and short)
            trade_type: 'long' or 'short'
            entry_cost: Total cost including commission/slippage
            entry_equity: Portfolio equity at entry
        """
        self.current_trade = Trade(
            entry_date=date,
            exit_date=None,
            entry_price=price,
            exit_price=None,
            position_size=abs(position_size),
            trade_type=trade_type,
            entry_cost=entry_cost,
            exit_proceeds=None,
            pnl=None,
            pnl_percent=None,
            is_closed=False,
            entry_equity=entry_equity,
            equity_peak_during_trade=entry_equity,  # Start with entry equity as peak
            max_trade_drawdown=0.0  # Initialize at 0% since we start at the peak
        )
        
    def close_trade(self, date: pd.Timestamp, price: float, exit_proceeds: float, 
                   exit_equity: float = None) -> Optional[Trade]:
        """
        Close the current trade and calculate P&L.
        
        Args:
            date: Exit date
            price: Exit price
            exit_proceeds: Total proceeds after commission/slippage
            exit_equity: Portfolio equity at exit
            
        Returns:
            Completed trade object with P&L calculated
        """
        if not self.current_trade:
            return None
            
        self.current_trade.exit_date = date
        self.current_trade.exit_price = price
        self.current_trade.exit_proceeds = exit_proceeds
        self.current_trade.is_closed = True
        self.current_trade.exit_equity = exit_equity
        
        # Update equity peak during trade if we reached a new high at exit
        if exit_equity and (self.current_trade.equity_peak_during_trade is None or 
                           exit_equity > self.current_trade.equity_peak_during_trade):
            self.current_trade.equity_peak_during_trade = exit_equity
        
        # Calculate P&L based on trade type
        if self.current_trade.trade_type == 'long':
            # Long P&L = exit proceeds - entry cost + dividends - borrowing costs
            self.current_trade.pnl = (
                exit_proceeds - 
                self.current_trade.entry_cost + 
                self.current_trade.dividends_received - 
                self.current_trade.borrowing_costs
            )
        else:  # short
            # Short P&L = entry proceeds - exit cost - dividends paid - borrowing costs
            # Note: For shorts, entry_cost is actually proceeds received
            self.current_trade.pnl = (
                self.current_trade.entry_cost - 
                exit_proceeds - 
                self.current_trade.dividends_received -  # Negative for shorts
                self.current_trade.borrowing_costs
            )
        
        # Calculate percentage return
        if self.current_trade.entry_cost != 0:
            if self.current_trade.trade_type == 'long':
                self.current_trade.pnl_percent = self.current_trade.pnl / self.current_trade.entry_cost
            else:  # short
                # For shorts, use the exit cost as the base
                self.current_trade.pnl_percent = self.current_trade.pnl / exit_proceeds
        else:
            self.current_trade.pnl_percent = 0.0
        
        # Add to completed trades
        self.trades.append(self.current_trade)
        completed_trade = self.current_trade
        self.current_trade = None
        
        return completed_trade
    
    def add_dividend(self, amount: float) -> None:
        """
        Add dividend to current trade.
        
        Args:
            amount: Dividend amount (positive for long, negative for short)
        """
        if self.current_trade:
            self.current_trade.dividends_received += amount
    
    def add_borrowing_cost(self, amount: float) -> None:
        """
        Add borrowing cost to current trade.
        
        Args:
            amount: Borrowing cost (always positive)
        """
        if self.current_trade:
            self.current_trade.borrowing_costs += amount
    
    def update_trade_excursions(self, current_price: float) -> None:
        """
        Update MAE and MFE for current trade.
        
        Args:
            current_price: Current market price
        """
        if not self.current_trade:
            return
            
        # Calculate unrealized P&L
        if self.current_trade.trade_type == 'long':
            unrealized_pnl = (current_price - self.current_trade.entry_price) * self.current_trade.position_size
        else:  # short
            unrealized_pnl = (self.current_trade.entry_price - current_price) * self.current_trade.position_size
            
        # Update MAE (most negative excursion)
        if self.current_trade.max_adverse_excursion is None or unrealized_pnl < self.current_trade.max_adverse_excursion:
            self.current_trade.max_adverse_excursion = unrealized_pnl
            
        # Update MFE (most positive excursion)
        if self.current_trade.max_favorable_excursion is None or unrealized_pnl > self.current_trade.max_favorable_excursion:
            self.current_trade.max_favorable_excursion = unrealized_pnl
    
    def update_equity_tracking(self, date: pd.Timestamp, equity: float) -> None:
        """
        Update drawdown tracking with current equity.
        
        Args:
            date: Current date
            equity: Current portfolio equity
        """
        # Store equity history
        self.equity_history.append((date, equity))
        
        # Update peak equity
        if equity > self.equity_peak:
            self.equity_peak = equity
            # Reset drawdown duration if new peak
            if self.current_drawdown_duration > 0:
                self.current_drawdown_duration = 0
                self.drawdown_start_date = None
        
        # Calculate current drawdown
        if self.equity_peak > 0:
            self.current_drawdown = (equity - self.equity_peak) / self.equity_peak
        else:
            self.current_drawdown = 0.0
            
        # Update drawdown duration
        if self.current_drawdown < 0:
            self.current_drawdown_duration += 1
            if self.drawdown_start_date is None:
                self.drawdown_start_date = date
        else:
            self.current_drawdown_duration = 0
            self.drawdown_start_date = None
            
        # Update max drawdown
        if self.current_drawdown < self.max_drawdown:
            self.max_drawdown = self.current_drawdown
            
        # Update max drawdown duration
        if self.current_drawdown_duration > self.max_drawdown_duration:
            self.max_drawdown_duration = self.current_drawdown_duration
            
        # Update current trade's drawdown tracking if a trade is active
        if self.current_trade and self.current_trade.equity_peak_during_trade:
            # Update equity peak during trade if current equity is higher
            if equity > self.current_trade.equity_peak_during_trade:
                self.current_trade.equity_peak_during_trade = equity
                
            # Calculate trade-specific drawdown from trade's own peak
            trade_drawdown = (equity - self.current_trade.equity_peak_during_trade) / self.current_trade.equity_peak_during_trade
            
            # Update max trade drawdown if current is worse
            if trade_drawdown < self.current_trade.max_trade_drawdown:
                self.current_trade.max_trade_drawdown = trade_drawdown
    
    def get_stats(self) -> Dict[str, float]:
        """
        Calculate comprehensive trade statistics.
        
        Returns:
            Dictionary with trade statistics
        """
        closed_trades = [t for t in self.trades if t.is_closed]
        
        if not closed_trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
                'total_pnl': 0.0,
                'avg_pnl': 0.0,
                'avg_pnl_percent': 0.0,
                'best_trade': 0.0,
                'worst_trade': 0.0,
                'avg_trade_duration': 0.0,
                # Enhanced metrics (same as main return)
                'max_drawdown': self.max_drawdown,
                'current_drawdown': self.current_drawdown,
                'max_drawdown_duration': self.max_drawdown_duration,
                'avg_mae': 0.0,
                'avg_mfe': 0.0,
                'max_consecutive_wins': 0,
                'max_consecutive_losses': 0,
                'equity_peak': self.equity_peak
            }
        
        # Calculate basic stats
        winning_trades = [t for t in closed_trades if t.pnl > 0]
        losing_trades = [t for t in closed_trades if t.pnl <= 0]
        
        total_trades = len(closed_trades)
        num_winners = len(winning_trades)
        num_losers = len(losing_trades)
        
        # Calculate P&L stats
        total_pnl = sum(t.pnl for t in closed_trades)
        avg_pnl = total_pnl / total_trades if total_trades > 0 else 0.0
        
        # Calculate win/loss averages
        avg_win = sum(t.pnl for t in winning_trades) / num_winners if num_winners > 0 else 0.0
        avg_loss = sum(t.pnl for t in losing_trades) / num_losers if num_losers > 0 else 0.0
        
        # Calculate profit factor
        gross_profit = sum(t.pnl for t in winning_trades) if winning_trades else 0.0
        gross_loss = abs(sum(t.pnl for t in losing_trades)) if losing_trades else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Calculate percentage returns
        avg_pnl_percent = sum(t.pnl_percent for t in closed_trades) / total_trades
        
        # Find best and worst trades
        best_trade = max(t.pnl_percent for t in closed_trades) if closed_trades else 0.0
        worst_trade = min(t.pnl_percent for t in closed_trades) if closed_trades else 0.0
        
        # Calculate average trade duration
        durations = [(t.exit_date - t.entry_date).days for t in closed_trades if t.exit_date]
        avg_duration = sum(durations) / len(durations) if durations else 0.0
        
        # Calculate MAE/MFE statistics
        mae_values = [t.max_adverse_excursion for t in closed_trades if t.max_adverse_excursion is not None]
        mfe_values = [t.max_favorable_excursion for t in closed_trades if t.max_favorable_excursion is not None]
        
        avg_mae = np.mean(mae_values) if mae_values else 0.0
        avg_mfe = np.mean(mfe_values) if mfe_values else 0.0
        
        # Calculate consecutive wins/losses
        consecutive_wins, consecutive_losses = self._calculate_consecutive_streaks(closed_trades)
        
        return {
            'total_trades': total_trades,
            'winning_trades': num_winners,
            'losing_trades': num_losers,
            'win_rate': num_winners / total_trades if total_trades > 0 else 0.0,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'avg_pnl_percent': avg_pnl_percent,
            'best_trade': best_trade,
            'worst_trade': worst_trade,
            'avg_trade_duration': avg_duration,
            # Enhanced metrics
            'max_drawdown': self.max_drawdown,
            'current_drawdown': self.current_drawdown,
            'max_drawdown_duration': self.max_drawdown_duration,
            'avg_mae': avg_mae,
            'avg_mfe': avg_mfe,
            'max_consecutive_wins': consecutive_wins,
            'max_consecutive_losses': consecutive_losses,
            'equity_peak': self.equity_peak
        }
    
    def _calculate_consecutive_streaks(self, trades: List[Trade]) -> Tuple[int, int]:
        """
        Calculate maximum consecutive wins and losses.
        
        Args:
            trades: List of completed trades
            
        Returns:
            Tuple of (max_consecutive_wins, max_consecutive_losses)
        """
        if not trades:
            return 0, 0
            
        max_wins = 0
        max_losses = 0
        current_wins = 0
        current_losses = 0
        
        for trade in trades:
            if trade.pnl > 0:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)
                
        return max_wins, max_losses
    
    def get_trade_log(self) -> pd.DataFrame:
        """
        Get detailed trade log as DataFrame.
        
        Returns:
            DataFrame with all trade details
        """
        if not self.trades:
            return pd.DataFrame()
        
        trade_data = []
        for trade in self.trades:
            if trade.is_closed:
                # Calculate MAE and MFE percentages
                mae_percent = 0.0
                mfe_percent = 0.0
                max_drawdown_dollar = 0.0
                
                if trade.entry_cost and trade.entry_cost != 0:
                    if trade.max_adverse_excursion is not None:
                        mae_percent = (trade.max_adverse_excursion / trade.entry_cost) * 100
                    if trade.max_favorable_excursion is not None:
                        mfe_percent = (trade.max_favorable_excursion / trade.entry_cost) * 100
                
                # Calculate max drawdown dollar value
                if (trade.max_trade_drawdown is not None and 
                    trade.equity_peak_during_trade is not None):
                    max_drawdown_dollar = trade.max_trade_drawdown * trade.equity_peak_during_trade
                
                trade_data.append({
                    'Entry Date': trade.entry_date,
                    'Exit Date': trade.exit_date,
                    'Type': trade.trade_type.upper(),
                    'Entry Price': trade.entry_price,
                    'Exit Price': trade.exit_price,
                    'Position Size': trade.position_size,
                    'Entry Cost': trade.entry_cost,
                    'Exit Proceeds': trade.exit_proceeds,
                    'Dividends': trade.dividends_received,
                    'Borrowing Costs': trade.borrowing_costs,
                    'P&L': trade.pnl,
                    'P&L %': trade.pnl_percent * 100,
                    'MAE': trade.max_adverse_excursion or 0.0,
                    'MAE %': mae_percent,
                    'MFE': trade.max_favorable_excursion or 0.0,
                    'MFE %': mfe_percent,
                    'Max Drawdown %': (trade.max_trade_drawdown or 0.0) * 100,
                    'Max Drawdown $': max_drawdown_dollar,
                    'Duration (days)': (trade.exit_date - trade.entry_date).days,
                    'Equity Peak During Trade': trade.equity_peak_during_trade
                })
        
        return pd.DataFrame(trade_data)
    
    def reset(self):
        """Reset the tracker."""
        self.trades = []
        self.current_trade = None
        
        # Reset drawdown tracking
        self.equity_peak = 0.0
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0
        self.max_drawdown_duration = 0
        self.current_drawdown_duration = 0
        self.drawdown_start_date = None
        self.equity_history = []
    
    def get_equity_curve(self) -> pd.DataFrame:
        """
        Get equity curve with drawdown information.
        
        Returns:
            DataFrame with equity and drawdown data
        """
        if not self.equity_history:
            return pd.DataFrame()
            
        df = pd.DataFrame(self.equity_history, columns=['Date', 'Equity'])
        df = df.set_index('Date')
        
        # Calculate drawdown series
        df['Peak'] = df['Equity'].cummax()
        df['Drawdown'] = (df['Equity'] - df['Peak']) / df['Peak']
        df['Drawdown_Pct'] = df['Drawdown'] * 100
        
        return df
