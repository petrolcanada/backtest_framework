"""
Title and subtitle generation utilities for charts.
"""
from typing import Optional, Dict, Any
import pandas as pd


class TitleGenerator:
    """Generates dynamic titles and subtitles for backtest charts."""
    
    def __init__(self, data: pd.DataFrame, results: pd.DataFrame, engine: Optional = None):
        """
        Initialize title generator.
        
        Args:
            data: DataFrame with OHLCV data and indicators
            results: DataFrame with backtest results
            engine: BacktestEngine instance for configuration access
        """
        self.data = data
        self.results = results
        self.engine = engine
    
    def generate_main_title(self, ticker: str, base_strategy_name: str = "Strategy") -> str:
        """
        Generate dynamic chart title based on engine configuration.
        
        Args:
            ticker: Ticker symbol
            base_strategy_name: Base name of the strategy (e.g., "Monthly KDJ")
            
        Returns:
            Formatted title string
        """
        if not self.engine:
            return f"{ticker} {base_strategy_name}"
        
        title_parts = [ticker]
        
        # Add strategy name
        title_parts.append(base_strategy_name)
        
        # Determine trading mode based on short selling
        if hasattr(self.engine, 'enable_short_selling') and self.engine.enable_short_selling:
            # Check if we have asymmetric leverage
            if hasattr(self.engine, 'long_leverage') and hasattr(self.engine, 'short_leverage'):
                long_lev = self.engine.long_leverage
                short_lev = self.engine.short_leverage
                
                if long_lev == short_lev:
                    # Symmetric leverage
                    if long_lev == 1.0:
                        title_parts.append("Long/Short")
                    else:
                        title_parts.append(f"Long/Short ({long_lev:.1f}x)")
                else:
                    # Asymmetric leverage
                    title_parts.append(f"Long/Short ({long_lev:.1f}x/{short_lev:.1f}x)")
            else:
                title_parts.append("Long/Short")
        else:
            # Long only mode
            if hasattr(self.engine, 'long_leverage'):
                long_lev = self.engine.long_leverage
                if long_lev == 1.0:
                    title_parts.append("Long Only")
                else:
                    title_parts.append(f"Long Only ({long_lev:.1f}x)")
            else:
                title_parts.append("Long Only")
        
        # Add position sizing info if not 100%
        if hasattr(self.engine, 'position_sizing') and self.engine.position_sizing < 1.0:
            title_parts.append(f"({self.engine.position_sizing*100:.0f}% Sizing)")
        
        return " - ".join(title_parts)
    
    def generate_costs_subtitle(self) -> str:
        """
        Generate dynamic subtitle with costs based on actual backtest configuration.
        Only shows relevant costs based on strategy setup:
        - Long only: Commission + Dividends + (Margin if leveraged)
        - Long/Short: + Short borrow rate + Short proceeds interest
        
        Returns:
            Formatted subtitle string with relevant trading costs and settings
        """
        if not self.engine:
            return ""
        
        subtitle_parts = []
        
        # Always show commission
        if hasattr(self.engine, 'commission'):
            comm_pct = self.engine.commission * 100
            subtitle_parts.append(f"Comm: {comm_pct:.2f}%")
        
        # Show slippage if configured
        if hasattr(self.engine, 'slippage') and self.engine.slippage > 0:
            slip_pct = self.engine.slippage * 100
            subtitle_parts.append(f"Slip: {slip_pct:.3f}%")
        
        # Always show dividend setting
        if hasattr(self.engine, 'include_dividends'):
            div_status = "Divs: ✓" if self.engine.include_dividends else "Divs: ✗"
            subtitle_parts.append(div_status)
        
        # Check if we're using any leverage (long or short)
        has_leverage = False
        if hasattr(self.engine, 'long_leverage') and self.engine.long_leverage > 1.0:
            has_leverage = True
        if hasattr(self.engine, 'short_leverage') and self.engine.short_leverage > 1.0:
            has_leverage = True
        
        # Show margin costs if leverage is used
        if has_leverage and hasattr(self.engine, 'use_margin_costs') and self.engine.use_margin_costs:
            subtitle_parts.append("Margin: FFR+1.5%")
        
        # Show short-specific costs only if short selling is enabled
        if hasattr(self.engine, 'enable_short_selling') and self.engine.enable_short_selling:
            # Short borrow rate (stock borrowing cost) - show as FFR + spread
            if hasattr(self.engine, 'short_borrow_rate'):
                borrow_rate = self.engine.short_borrow_rate
                # Convert to FFR + spread format (assuming FFR baseline of ~0%)
                # Most short borrow rates are quoted as spreads over FFR
                spread_pct = borrow_rate * 100
                subtitle_parts.append(f"Short Borrow: FFR+{spread_pct:.1f}%")
            
            # Short proceeds interest (interest earned on short sale proceeds)
            # This is typically a positive return, so we show it as a benefit
            subtitle_parts.append("Short Proceeds: FFR-0.25%")
        
        return " | ".join(subtitle_parts) if subtitle_parts else ""
    
    def generate_performance_subtitle(self) -> str:
        """
        Generate subtitle with performance metrics summary.
        
        Returns:
            Formatted subtitle string with key performance metrics
        """
        metrics = self._calculate_performance_metrics()
        if not metrics:
            return ""
        
        # Calculate Buy & Hold metrics if available
        benchmark_equity = None
        if 'benchmark_equity' in self.results.columns:
            benchmark_equity = self.results['benchmark_equity'].iloc[-1]
        else:
            # Fallback calculation if benchmark not available
            first_signal_idx = 0
            buy_signal_col = 'buy_signal' if 'buy_signal' in self.results.columns else None
            if buy_signal_col:
                buy_signals = self.results[self.results[buy_signal_col] == 1]
                if not buy_signals.empty:
                    first_signal_idx = self.data.index.get_indexer([buy_signals.index[0]], method='nearest')[0]
            
            initial_price = self.data['Close'].iloc[first_signal_idx]
            final_price = self.data['Close'].iloc[-1]
            benchmark_equity = metrics['initial_capital'] * (final_price / initial_price)
        
        # Performance summary line 1 - Capital and returns
        performance_line1 = f"Capital: ${metrics['initial_capital']:,.0f} → ${metrics['final_equity']:,.0f}  |  "
        performance_line1 += f"Strategy: {metrics['total_return']*100:+.1f}%  |  "
        
        # Add appropriate benchmark label
        has_engine_benchmark = 'benchmark_equity' in self.results.columns
        include_divs = hasattr(self.engine, 'include_dividends') and self.engine.include_dividends
        benchmark_label = "Buy & Hold Total Return" if (has_engine_benchmark and include_divs and 'Dividends' in self.data.columns) else "Buy & Hold"
        performance_line1 += f"{benchmark_label}: {((benchmark_equity/metrics['initial_capital'])-1)*100:+.1f}%"
        
        # Performance summary line 2 - Risk metrics
        performance_line2 = f"Max DD: {metrics.get('max_drawdown', 0)*100:.1f}%  |  "
        if 'cagr' in metrics:
            performance_line2 += f"CAGR: {metrics['cagr']*100:.1f}%  |  "
        if 'sharpe_ratio' in metrics:
            performance_line2 += f"Sharpe: {metrics['sharpe_ratio']:.2f}  |  "
        if 'win_rate' in metrics:
            performance_line2 += f"Win Rate: {metrics['win_rate']*100:.0f}%  |  "
        if 'trade_count' in metrics:
            performance_line2 += f"Trades: {int(metrics['trade_count'])}"
        
        return f"{performance_line1}<br>{performance_line2}"
    
    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """
        Calculate key performance metrics from the results DataFrame.
        
        Returns:
            Dictionary with performance metrics
        """
        metrics = {}
        
        # Get column names
        equity_col = 'equity' if 'equity' in self.results.columns else None
        
        if equity_col:
            # Calculate main performance metrics
            initial_equity = self.results[equity_col].iloc[0]
            final_equity = self.results[equity_col].iloc[-1]
            total_return = (final_equity / initial_equity) - 1
            
            # Store metrics
            metrics['total_return'] = total_return
            metrics['initial_capital'] = initial_equity
            metrics['final_equity'] = final_equity
            
            # Get other metrics if available
            for metric in ['cagr', 'sharpe_ratio', 'sortino_ratio', 'max_drawdown', 'win_rate', 'trade_count']:
                if metric in self.results.columns:
                    metrics[metric] = self.results[metric].iloc[-1]
        
        return metrics
