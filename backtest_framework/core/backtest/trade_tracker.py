"""
Enhanced trade tracking module for accurate P&L and win rate calculation.
"""
from typing import Dict, List, Optional, Tuple
import pandas as pd
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


class TradeTracker:
    """
    Enhanced trade tracking for accurate P&L and win rate calculation.
    """
    
    def __init__(self):
        """Initialize trade tracker."""
        self.trades: List[Trade] = []
        self.current_trade: Optional[Trade] = None
        
    def open_trade(self, date: pd.Timestamp, price: float, position_size: float, 
                   trade_type: str, entry_cost: float) -> None:
        """
        Record a new trade entry.
        
        Args:
            date: Entry date
            price: Entry price
            position_size: Number of shares (positive for both long and short)
            trade_type: 'long' or 'short'
            entry_cost: Total cost including commission/slippage
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
            is_closed=False
        )
        
    def close_trade(self, date: pd.Timestamp, price: float, exit_proceeds: float) -> Optional[Trade]:
        """
        Close the current trade and calculate P&L.
        
        Args:
            date: Exit date
            price: Exit price
            exit_proceeds: Total proceeds after commission/slippage
            
        Returns:
            Completed trade object with P&L calculated
        """
        if not self.current_trade:
            return None
            
        self.current_trade.exit_date = date
        self.current_trade.exit_price = price
        self.current_trade.exit_proceeds = exit_proceeds
        self.current_trade.is_closed = True
        
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
                'avg_trade_duration': 0.0
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
            'avg_trade_duration': avg_duration
        }
    
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
                    'Duration (days)': (trade.exit_date - trade.entry_date).days
                })
        
        return pd.DataFrame(trade_data)
    
    def reset(self):
        """Reset the tracker."""
        self.trades = []
        self.current_trade = None
