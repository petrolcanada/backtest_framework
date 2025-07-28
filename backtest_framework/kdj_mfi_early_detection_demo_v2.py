"""
Demo script for the KDJ MFI Early Detection Strategy using the modular backtesting framework.
"""
import os
import sys

# Add current directory and parent to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
scripts_dir = os.path.dirname(current_dir)
sys.path.append(scripts_dir)

# Import from the backtesting framework
from backtest_framework.core.data.loader import DataLoader
from backtest_framework.core.strategies.kdj_mfi_early_detection_v2 import KDJMFIEarlyDetectionStrategy
from backtest_framework.core.backtest.engine import BacktestEngine
from backtest_framework.core.backtest.risk_management import DrawdownProtection
from backtest_framework.core.visualization.plotter import Plotter
from backtest_framework.core.utils.helpers import Timer, suppress_warnings

# Import indicator modules to ensure they're registered
from backtest_framework.core.indicators import kdj, sma, adx, mfi, rsi
from backtest_framework.core.indicators import kdj_derived_factors, adx_derived_factors, mfi_derived_factors, rsi_derived_factors

# Suppress warnings
suppress_warnings()


def main():
    """Run a demo backtest for the KDJ MFI Early Detection Strategy."""
    # Start timer
    timer = Timer()
    
    # Configuration
    ticker = "SPY"
    initial_capital = 10000
    commission = 0.001
    slippage = 0.001                        # Add 0.1% slippage for realistic execution costs
    drawdown_threshold = 0.2
    
    try:
        # 1. Load data with mode selection
        data_dir = os.path.join(os.path.expanduser("~"), "local_script", 
                               "Local Technical Indicator Data", "security_data")
        loader = DataLoader(data_dir=data_dir)
        
        # Data loading modes:
        # 'full_reload': Download max period from yfinance, overwrite CSV
        # 'incremental': Update last 10 days from CSV file to current date  
        # 'no_reload': Use existing CSV file as-is, no API calls
        data = loader.load(ticker, period="10y", resample_period="D", mode="no_reload")
    
        # 2. Initialize strategy with parameters (including custom indicator parameters)
        strategy = KDJMFIEarlyDetectionStrategy(
            # Strategy parameters (REMOVED: max_buy_signals_per_cross - no longer limiting signals)
            required_up_days=7,              # J must rise for 7 days
            required_down_days=4,            # For flip detection
            required_adx_down_days=5,        # ADX down requirement  
            enable_death_cross_buys=True,   # Disable alternative buys initially
            
            # Custom indicator parameters - showing explicit overrides of defaults
            adx_period=14*22,                   # Override default (308 -> 14) - fully standardized
            adx_sma_period=3,                # Override default (5 -> 3) - now uses 'period' too
            # kdj_period=198,                # Using default (9 months * 22 days)
            # kdj_signal=66,                 # Using default (3 months * 22 days)
            # mfi_period=48,                 # Using default (~2.2 months)
            # rsi_period=14,                 # Using default
        )
        
        # 3. Setup backtest engine with configuration
        engine = BacktestEngine(
            initial_capital=initial_capital, 
            commission=commission,
            slippage=slippage,                       # Add slippage parameter
            leverage={"long": 1.0, "short": 1.0},    # 2x long leverage, 1x short leverage
            position_sizing=1.0,                     # Use 100% of capital per trade
            enable_short_selling=False                # Enable short selling for long/short strategy
        )
        # Add drawdown protection risk manager
        # engine.add_risk_manager(DrawdownProtection(threshold=drawdown_threshold))
        
        # 4. Run backtest (strategy will automatically compute required indicators)
        results = engine.run(strategy, data)
        
        # 5. Print performance summary
        print(f"\n{ticker} KDJ MFI Early Detection Strategy Results:")
        print("=" * 50)
        
        # Core performance metrics
        print(f"Initial Capital: ${initial_capital:,.2f}")
        print(f"Final Equity: ${results['equity'].iloc[-1]:,.2f}")
        print(f"Total Return: {results['returns'].iloc[-1] * 100:.2f}%")
        print(f"CAGR: {results['cagr'].iloc[-1] * 100:.2f}%")
        print(f"Sharpe Ratio: {results['sharpe_ratio'].iloc[-1]:.2f}")
        print(f"Max Drawdown: {results['max_drawdown'].iloc[-1] * 100:.2f}%")
        
        # Win rate if available
        if 'win_rate' in results.columns:
            print(f"Win Rate: {results['win_rate'].iloc[-1] * 100:.2f}%")
        
        # Trade statistics
        print(f"Total Trades: {int(results['trade_count'].iloc[-1])}")
        
        # Signal counts
        buy_signals = (results['buy_signal'] == 1).sum()
        sell_signals = (results['sell_signal'] == 1).sum()
        print(f"Buy Signals: {buy_signals}")
        print(f"Sell Signals: {sell_signals}")
        
        # Benchmark comparison if available
        if 'benchmark_returns' in results.columns:
            benchmark_return = results['benchmark_returns'].iloc[-1] * 100
            outperformance = (results['returns'].iloc[-1] - results['benchmark_returns'].iloc[-1]) * 100
            print(f"Benchmark Return: {benchmark_return:.2f}%")
            print(f"Outperformance: {outperformance:+.2f}%")
        
        print("=" * 50)
        
        # 6. Create visualization
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
        os.makedirs(output_dir, exist_ok=True)
        
        # Create plotter with the data and results
        plotter = Plotter(data=data, results=results, engine=engine)
        
        # Create comprehensive chart
        fig = plotter.create_comprehensive_chart(
            ticker=ticker, 
            base_strategy_name="KDJ MFI Early Detection", 
            log_scale=False
        )
        
        # Save chart
        output_file = os.path.join(output_dir, f"{ticker}_kdj_mfi_early_detection_strategy_v2.html")
        plotter.save(output_file)
        
        # Open chart in browser
        plotter.open_in_browser(output_file)
    
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print(f"Completed in {timer.elapsed_str()}")


if __name__ == "__main__":
    main()
