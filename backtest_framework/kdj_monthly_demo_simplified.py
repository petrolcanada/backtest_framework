"""
Simplified demo script for the KDJ Golden/Dead Cross strategy using the modular backtesting framework.
"""
import os
import sys

# Add current directory and parent to path for imports
# We're inside the backtest_framework directory, so add scripts directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
scripts_dir = os.path.dirname(current_dir)
sys.path.append(scripts_dir)

# Import from the backtesting framework
from backtest_framework.core.data.loader import DataLoader
from backtest_framework.core.strategies.kdj_cross import GoldenDeadCrossStrategyMonthly
from backtest_framework.core.backtest.engine import BacktestEngine
from backtest_framework.core.backtest.risk_management import DrawdownProtection
from backtest_framework.core.visualization.plotter import Plotter
from backtest_framework.core.utils.helpers import Timer, suppress_warnings

# Import indicator modules to ensure they're registered
from backtest_framework.core.indicators import kdj, sma

# Suppress warnings
suppress_warnings()


def main():
    """Run a simplified demo backtest for the Monthly KDJ Golden/Dead Cross strategy."""
    print("Monthly KDJ Golden/Dead Cross Strategy Demo (Simplified)")
    print("======================================================")
    
    # Start timer
    timer = Timer()
    
    # Configuration
    ticker = "SPY"
    initial_capital = 10000
    commission = 0.001
    drawdown_threshold = 0.2
    
    try:
        # 1. Load data
        print(f"\nLoading data for {ticker}...")
        data_dir = os.path.join(os.path.expanduser("~"), "local_script", 
                               "Local Technical Indicator Data", "security_data")
        loader = DataLoader(data_dir=data_dir)
        data = loader.load(ticker, period="3y", resample_period="D")
        print(f"Loaded {len(data)} rows of data from {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
        
        # 2. Initialize strategy (indicators will be computed automatically)
        print("\nInitializing Monthly KDJ strategy...")
        strategy = GoldenDeadCrossStrategyMonthly()
        
        # 3. Setup backtest engine with configuration
        print("Setting up backtest engine...")
        engine = BacktestEngine(
            initial_capital=initial_capital, 
            commission=commission,
            leverage={"long": 2.0, "short": 3.0},  # 2x long leverage, 3x short leverage
            position_sizing=1.0,                     # Use 100% of capital per trade
            enable_short_selling=True                # Enable short selling for long/short strategy
        )
        # engine.add_risk_manager(DrawdownProtection(threshold=drawdown_threshold))
        
        # 4. Run backtest (strategy will automatically compute required indicators)
        print("Running backtest...")
        results = engine.run(strategy, data)
        
        # 5. Display performance metrics
        print(f"\n{ticker} Monthly KDJ Strategy Results:")
        print("=" * 60)
        print("📊 Configuration:")
        print(f"   • Long Leverage: {engine.long_leverage:.1f}x")
        print(f"   • Short Leverage: {engine.short_leverage:.1f}x")
        print(f"   • Position Sizing: {engine.position_sizing*100:.0f}%")
        print(f"   • Short Selling: {'Enabled' if engine.enable_short_selling else 'Disabled (Long-only)'}")
        print(f"   • Commission: {commission*100:.2f}%")
        print("\n📈 Performance Metrics:")
        print(f"Initial Capital: ${initial_capital:,.2f}")
        print(f"Final Equity: ${results['equity'].iloc[-1]:,.2f}")
        print(f"Total Return: {results['returns'].iloc[-1] * 100:.2f}%")
        print(f"CAGR: {results['cagr'].iloc[-1] * 100:.2f}%")
        print(f"Sharpe Ratio: {results['sharpe_ratio'].iloc[-1]:.2f}")
        print(f"Max Drawdown: {results['max_drawdown'].iloc[-1] * 100:.2f}%")
        if 'win_rate' in results.columns:
            print(f"Win Rate: {results['win_rate'].iloc[-1] * 100:.2f}%")
        print(f"Total Trades: {int(results['trade_count'].iloc[-1])}")
        
        # Benchmark Performance (calculated in engine)
        if 'benchmark_returns' in results.columns:
            print("\n📊 Benchmark Performance (Buy & Hold):")
            benchmark_label = "Total Return" if engine.include_dividends and 'Dividends' in results.columns else "Price Return"
            print(f"Benchmark Type: {benchmark_label}")
            print(f"Benchmark Return: {results['benchmark_returns'].iloc[-1] * 100:.2f}%")
            if 'benchmark_cagr' in results.columns:
                print(f"Benchmark CAGR: {results['benchmark_cagr'].iloc[-1] * 100:.2f}%")
            if 'benchmark_max_drawdown' in results.columns:
                print(f"Benchmark Max DD: {results['benchmark_max_drawdown'].iloc[-1] * 100:.2f}%")
            
            # Outperformance
            outperformance = (results['returns'].iloc[-1] - results['benchmark_returns'].iloc[-1]) * 100
            print(f"Outperformance: {outperformance:+.2f}%")
        
        # Dividend analysis (new feature)
        if 'cumulative_dividends' in results.columns:
            total_dividends = results['cumulative_dividends'].iloc[-1]
            if abs(total_dividends) > 0.01:  # Only show if meaningful dividend amount
                print(f"Total Dividends: ${total_dividends:,.2f}")
                dividend_contribution = (total_dividends / initial_capital) * 100
                print(f"Dividend Contribution: {dividend_contribution:.2f}%")
        
        # Borrowing cost analysis (new feature)
        if 'cumulative_borrowing_costs' in results.columns:
            total_borrowing_costs = results['cumulative_borrowing_costs'].iloc[-1]
            if abs(total_borrowing_costs) > 0.01:
                print(f"Total Borrowing Costs: ${total_borrowing_costs:,.2f}")
                cost_impact = (total_borrowing_costs / initial_capital) * 100
                print(f"Cost Impact: -{cost_impact:.2f}%")
        
        # Position analysis
        long_periods = (results['position_type'] == 'long').sum()
        short_periods = (results['position_type'] == 'short').sum()
        total_periods = len(results)
        
        print("\n📊 Position Analysis:")
        print(f"Long Periods: {long_periods} days ({long_periods/total_periods*100:.1f}%)")
        print(f"Short Periods: {short_periods} days ({short_periods/total_periods*100:.1f}%)")
        
        # Signal analysis
        buy_signals = (results['buy_signal'] == 1).sum()
        sell_signals = (results['sell_signal'] == 1).sum()
        print(f"\n📡 Signal Analysis:")
        print(f"Buy Signals: {buy_signals}")
        print(f"Sell Signals: {sell_signals}")
        
        # DEBUG: Check T+1 execution tracking
        if 'execution_date' in results.columns:
            executed_buys = ((results['execution_date'] != '') & (results['buy_signal'] == 1)).sum()
            executed_sells = ((results['execution_date'] != '') & (results['sell_signal'] == 1)).sum()
            print(f"\n🔍 T+1 Execution Debug:")
            print(f"Executed Buy Signals: {executed_buys}")
            print(f"Executed Sell Signals: {executed_sells}")
            
            # DEBUG: Check why signals aren't being executed
            signal_dates_with_signals = results[(results['buy_signal'] == 1) | (results['sell_signal'] == 1)][['buy_signal', 'sell_signal', 'signal_date', 'execution_date']]
            print(f"\n🔍 All Signal Generation Dates:")
            for idx, row in signal_dates_with_signals.iterrows():
                signal_type = 'BUY' if row['buy_signal'] == 1 else 'SELL'
                signal_date = row['signal_date'] if row['signal_date'] != '' else 'NOT_SET'
                execution_date = row['execution_date'] if row['execution_date'] != '' else 'NOT_EXECUTED'
                print(f"  {signal_type} on {idx.strftime('%Y-%m-%d')}: Signal_Date='{signal_date}', Execution_Date='{execution_date}'")
            
            # Show details of executed signals
            executed_signals = results[
                (results['execution_date'] != '') & 
                ((results['buy_signal'] == 1) | (results['sell_signal'] == 1))
            ][['buy_signal', 'sell_signal', 'signal_date', 'execution_date']]
            
            if not executed_signals.empty:
                print(f"\n✅ Successful Executions:")
                for idx, row in executed_signals.iterrows():
                    signal_type = 'BUY' if row['buy_signal'] == 1 else 'SELL'
                    print(f"  {signal_type}: Signal on {row['signal_date']}, Executed on {row['execution_date']} (Date: {idx.strftime('%Y-%m-%d')})")
            else:
                print(f"\n❌ No successful executions found!")
                
            # DEBUG: Check for pending trades that weren't executed
            signals_without_execution = results[
                ((results['buy_signal'] == 1) | (results['sell_signal'] == 1)) &
                (results['signal_date'] != '') &
                (results['execution_date'] == '')
            ]
            if not signals_without_execution.empty:
                print(f"\n⚠️  Signals that were NOT executed ({len(signals_without_execution)}):")
                for idx, row in signals_without_execution.iterrows():
                    signal_type = 'BUY' if row['buy_signal'] == 1 else 'SELL'
                    print(f"  {signal_type} on {idx.strftime('%Y-%m-%d')}: Signal generated but never executed")
        
        # 6. Create visualization
        print("\nGenerating visualization...")
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
        os.makedirs(output_dir, exist_ok=True)
        
        # Create plotter with the data and results
        # Note: Pass original data for OHLC columns and results for strategy data
        plotter = Plotter(data=data, results=results, engine=engine)
        
        # Create comprehensive chart using the new component-based system
        print("Creating comprehensive strategy analysis chart...")
        fig = plotter.create_comprehensive_chart(
            ticker=ticker, 
            base_strategy_name="Monthly KDJ", 
            log_scale=False
        )
        
        # Save chart
        output_file = os.path.join(output_dir, f"{ticker}_monthly_kdj_strategy.html")
        plotter.save(output_file)
        print(f"Chart saved to: {output_file}")
        
        # Open chart in browser
        plotter.open_in_browser(output_file)
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print(f"\nCompleted in {timer.elapsed_str()}")


if __name__ == "__main__":
    main()
