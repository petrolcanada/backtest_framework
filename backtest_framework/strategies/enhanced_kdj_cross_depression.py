"""
Enhanced KDJ Cross Depression Strategy with comprehensive trade analysis and max drawdown tracking.

This enhanced version includes:
- Max drawdown tracking in real-time
- MAE/MFE (Maximum Adverse/Favorable Excursion) analysis
- Detailed trade logs with all execution metrics
- Consecutive wins/losses tracking
- Enhanced performance summary with drawdown periods
- Equity curve analysis
"""
import os
import sys

# Add current directory and parent to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
backtest_framework_dir = os.path.dirname(current_dir)
scripts_dir = os.path.dirname(backtest_framework_dir)
sys.path.append(scripts_dir)

# Import from the backtesting framework
from backtest_framework.core.data.loader import DataLoader
from backtest_framework.core.signals.kdj_cross_depression import KDJCrossDepressionStrategy
from backtest_framework.core.backtest.engine import BacktestEngine
from backtest_framework.core.backtest.risk_management import DrawdownProtection
from backtest_framework.core.visualization.plotter import Plotter
from backtest_framework.core.utils.helpers import Timer, suppress_warnings

# Import indicator modules to ensure they're registered
from backtest_framework.core.indicators import kdj, sma, adx, mfi, rsi
from backtest_framework.core.indicators import kdj_derived_factors, adx_derived_factors, mfi_derived_factors, rsi_derived_factors

import pandas as pd

# Suppress warnings
suppress_warnings()


def print_enhanced_performance_summary(results: pd.DataFrame, trade_executor, ticker: str, initial_capital: float):
    """
    Print comprehensive performance summary with enhanced trade analysis.
    
    Args:
        results: Backtest results DataFrame
        trade_executor: TradeExecutor instance with enhanced tracking
        ticker: Stock ticker symbol
        initial_capital: Initial capital amount
    """
    print(f"\n{ticker} Enhanced KDJ Cross Depression Strategy Results:")
    print("=" * 80)
    
    # Basic Performance Metrics
    print(f"\nBASIC PERFORMANCE:")
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print(f"Final Equity: ${results['equity'].iloc[-1]:,.2f}")
    print(f"Total Return: {results['returns'].iloc[-1] * 100:.2f}%")
    print(f"CAGR: {results['cagr'].iloc[-1] * 100:.2f}%")
    print(f"Sharpe Ratio: {results['sharpe_ratio'].iloc[-1]:.2f}")
    print(f"Max Drawdown: {results['max_drawdown'].iloc[-1] * 100:.2f}%")
    
    # Enhanced Trade Statistics from TradeExecutor
    enhanced_stats = trade_executor.get_enhanced_summary()
    
    print(f"\nTRADE STATISTICS:")
    print(f"Total Trades: {enhanced_stats['total_trades']}")
    print(f"Winning Trades: {enhanced_stats['winning_trades']}")
    print(f"Losing Trades: {enhanced_stats['losing_trades']}")
    print(f"Win Rate: {enhanced_stats['win_rate_pct']:.1f}%")
    print(f"Profit Factor: {enhanced_stats['profit_factor']:.2f}")
    
    print(f"\nPERFORMANCE METRICS:")
    print(f"Average Win: {enhanced_stats['avg_win_pct']:.2f}%")
    print(f"Average Loss: {enhanced_stats['avg_loss_pct']:.2f}%")
    print(f"Best Trade: {enhanced_stats['best_trade_pct']:.2f}%")
    print(f"Worst Trade: {enhanced_stats['worst_trade_pct']:.2f}%")
    
    print(f"\nENHANCED DRAWDOWN ANALYSIS:")
    print(f"Maximum Drawdown: {enhanced_stats['max_drawdown_pct']:.2f}%")
    print(f"Current Drawdown: {enhanced_stats['current_drawdown_pct']:.2f}%")
    print(f"Max Drawdown Duration: {enhanced_stats['max_drawdown_duration_days']} days")
    print(f"Equity Peak: ${enhanced_stats['equity_peak']:,.2f}")
    
    print(f"\nTRADE EXECUTION METRICS:")
    print(f"Average MAE (Max Adverse Excursion): ${enhanced_stats['avg_mae']:,.2f}")
    print(f"Average MFE (Max Favorable Excursion): ${enhanced_stats['avg_mfe']:,.2f}")
    print(f"Max Consecutive Wins: {enhanced_stats['max_consecutive_wins']}")
    print(f"Max Consecutive Losses: {enhanced_stats['max_consecutive_losses']}")
    
    print(f"\nCOST ANALYSIS:")
    print(f"Cumulative Dividends: ${enhanced_stats['cumulative_dividends']:,.2f}")
    print(f"Cumulative Borrowing Costs: ${enhanced_stats['cumulative_borrowing_costs']:,.2f}")
    
    # Signal Statistics
    buy_signals = (results['buy_signal'] == 1).sum()
    sell_signals = (results['sell_signal'] == 1).sum()
    print(f"\nSIGNAL STATISTICS:")
    print(f"Buy Signals Generated: {buy_signals}")
    print(f"Sell Signals Generated: {sell_signals}")
    
    # Benchmark comparison if available
    if 'benchmark_returns' in results.columns:
        benchmark_return = results['benchmark_returns'].iloc[-1] * 100
        outperformance = (results['returns'].iloc[-1] - results['benchmark_returns'].iloc[-1]) * 100
        print(f"\nBENCHMARK COMPARISON:")
        print(f"Benchmark Return: {benchmark_return:.2f}%")
        print(f"Strategy Outperformance: {outperformance:+.2f}%")
    
    print("=" * 80)


def print_detailed_trade_analysis(trade_executor):
    """
    Print detailed trade-by-trade analysis.
    
    Args:
        trade_executor: TradeExecutor instance with enhanced tracking
    """
    print("\nDETAILED TRADE ANALYSIS:")
    print("=" * 120)
    
    # Get detailed trade log
    trade_log = trade_executor.get_trade_log()
    
    if not trade_log.empty:
        print(f"\nTRADE LOG (Showing all {len(trade_log)} trades):")
        print("-" * 120)
        
        # Display trade log with formatted output
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', 12)
        
        # Format the trade log for better readability
        display_log = trade_log.copy()
        
        # Format currency columns
        currency_cols = ['Entry Price', 'Exit Price', 'Entry Cost', 'Exit Proceeds', 
                        'Dividends', 'Borrowing Costs', 'P&L', 'MAE', 'MFE', 
                        'Equity Peak During Trade']
        
        for col in currency_cols:
            if col in display_log.columns:
                display_log[col] = display_log[col].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "N/A")
        
        # Format percentage columns
        percentage_cols = ['P&L %', 'MAE %', 'MFE %', 'Max Drawdown %']
        for col in percentage_cols:
            if col in display_log.columns:
                display_log[col] = display_log[col].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")
        
        print(display_log.to_string(index=False))
        
        # Reset pandas display options
        pd.reset_option('display.max_columns')
        pd.reset_option('display.width')
        pd.reset_option('display.max_colwidth')
    else:
        print("No trades completed during the backtest period.")
    


def run_enhanced_backtest(ticker: str, strategy_params: dict, backtest_params: dict):
    """
    Run backtest with enhanced trade tracking and analysis.
    
    Args:
        ticker: Stock ticker symbol
        strategy_params: Strategy configuration parameters
        backtest_params: Backtest configuration parameters
        
    Returns:
        Tuple of (results, trade_executor, data, engine)
    """
    # Load data
    data_dir = os.path.join(os.path.expanduser("~"), "local_script", 
                           "Local Technical Indicator Data", "security_data")
    loader = DataLoader(data_dir=data_dir)
    
    data = loader.load(ticker, 
                      period=backtest_params.get('period', '20y'), 
                      resample_period=backtest_params.get('resample_period', 'D'), 
                      mode=backtest_params.get('mode', 'no_reload'))
    
    # Initialize strategy
    strategy = KDJCrossDepressionStrategy(**strategy_params)
    
    # Setup backtest engine
    engine = BacktestEngine(
        initial_capital=backtest_params['initial_capital'], 
        commission=backtest_params.get('commission', 0.001),
        slippage=backtest_params.get('slippage', 0.001),
        leverage=backtest_params.get('leverage', {"long": 1.0, "short": 1.0}),
        position_sizing=backtest_params.get('position_sizing', 1.0),
        enable_short_selling=backtest_params.get('enable_short_selling', False)
    )
    
    # Add risk management if specified
    if 'drawdown_threshold' in backtest_params:
        engine.add_risk_manager(DrawdownProtection(threshold=backtest_params['drawdown_threshold']))
    
    # Run backtest
    results = engine.run(strategy, data)
    
    # Get the trade executor for enhanced analysis
    trade_executor = engine.trade_executor
    
    return results, trade_executor, data, engine


def main():
    """Run enhanced KDJ Cross Depression Strategy backtest with comprehensive analysis."""
    # Start timer
    timer = Timer()
    
    # Configuration
    ticker = "NVDA"
    
    # Strategy parameters
    strategy_params = {
        'depression_threshold': 40,              # Depression zone threshold
        'j_extreme_threshold': 100.0,            # J extreme level for sell
        'use_monthly_kdj': True,                 # Use monthly vs daily KDJ
        'kdj_period': 9 * 5,                     # KDJ lookback period (45 days ~ 9 weeks)
        'kdj_signal': 3 * 5,                     # KDJ signal period (15 days ~ 3 weeks)
        'daily_kdj_period': 9,                   # Daily KDJ period if used
        'daily_kdj_signal': 3,                   # Daily KDJ signal if used
    }
    
    # Backtest parameters
    backtest_params = {
        'initial_capital': 10000,
        'commission': 0.001,                     # 0.1% commission
        'slippage': 0.001,                       # 0.1% slippage
        'leverage': {"long": 1.1, "short": 1.0}, # 1x leverage
        'position_sizing': 1.0,                  # Use 100% of capital per trade
        'enable_short_selling': False,           # Long-only strategy
        'period': '10y',                         # Data period
        'resample_period': 'D',                  # Daily data
        'mode': 'no_reload'                      # Use existing CSV data
    }
    
    # Export options
    export_options = {
        'export_backtest_results': False,        # Set to True to export main results dataframe
    }
    
    try:
        print(f"Running Enhanced KDJ Cross Depression Strategy for {ticker}...")
        print("Loading data and initializing strategy...")
        
        # Run enhanced backtest
        results, trade_executor, data, engine = run_enhanced_backtest(
            ticker=ticker,
            strategy_params=strategy_params,
            backtest_params=backtest_params
        )
        
        # Print enhanced performance summary
        print_enhanced_performance_summary(
            results=results,
            trade_executor=trade_executor,
            ticker=ticker,
            initial_capital=backtest_params['initial_capital']
        )
        
        # Print detailed trade analysis
        print_detailed_trade_analysis(trade_executor)
        
        # Create visualization
        print(f"\nGenerating visualization...")
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
        os.makedirs(output_dir, exist_ok=True)
        
        # Create plotter with enhanced results
        plotter = Plotter(data=data, results=results, engine=engine)
        
        # Create comprehensive chart
        fig = plotter.create_comprehensive_chart(
            ticker=ticker, 
            base_strategy_name="Enhanced KDJ Cross Depression", 
            log_scale=True,
            sync_y_axis_cum_return=False
        )
        
        # Save chart with enhanced name
        output_file = os.path.join(output_dir, f"{ticker}_enhanced_kdj_cross_depression_strategy.html")
        plotter.save(output_file)
        
        print(f"Chart saved to: {output_file}")
        
        # Open chart in browser
        plotter.open_in_browser(output_file)
        
        # Optional: Save detailed trade log to CSV
        trade_log = trade_executor.get_trade_log()
        if not trade_log.empty:
            csv_file = os.path.join(output_dir, f"{ticker}_enhanced_trade_log.csv")
            trade_log.to_csv(csv_file, index=False)
            print(f"Detailed trade log saved to: {csv_file}")
        
        # Optional: Save main backtest results dataframe to CSV
        if export_options.get('export_backtest_results', False):
            results_file = os.path.join(output_dir, f"{ticker}_backtest_results.csv")
            results.to_csv(results_file)
            print(f"Backtest results dataframe saved to: {results_file}")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print(f"\nCompleted in {timer.elapsed_str()}")


if __name__ == "__main__":
    main()
