"""
Demo script for the KDJ Cross Depression Strategy using the modular backtesting framework.
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

# Suppress warnings
suppress_warnings()


def main():
    """Run a demo backtest for the KDJ Cross Depression Strategy."""
    # Start timer
    timer = Timer()
    
    # Configuration
    ticker = "AAPL"
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
        data = loader.load(ticker, period="20y", resample_period="D", mode="no_reload")
    
        # 2. Initialize strategy with parameters
        strategy = KDJCrossDepressionStrategy(
            # Strategy parameters
            depression_threshold=50.0,           # Depression zone threshold
            j_extreme_threshold=100.0,           # J extreme level for sell
            use_monthly_kdj=True,                # Use monthly vs daily KDJ
            
            # Custom indicator parameters - showing explicit overrides of defaults
            # kdj_period=9 * 5,                      # KDJ lookback period (198 = 9 months * 22 days)
            # kdj_signal=3 * 5,                       # KDJ signal period (66 = 3 months * 22 days)
            
            kdj_period=9,                      # KDJ lookback period (198 = 9 months * 22 days)
            kdj_signal=3,                       # KDJ signal period (66 = 3 months * 22 days)
            daily_kdj_period=9,                  # Daily KDJ period if used
            daily_kdj_signal=3,                  # Daily KDJ signal if used
        )
        
        # 3. Setup backtest engine with configuration
        engine = BacktestEngine(
            initial_capital=initial_capital, 
            commission=commission,
            slippage=slippage,                       # Add slippage parameter
            leverage={"long": 1.0, "short": 1.0},    # 1x long leverage, 1x short leverage
            position_sizing=1.0,                     # Use 100% of capital per trade
            enable_short_selling=False                # Disable short selling for long-only strategy
        )
        # Add drawdown protection risk manager (commented out for initial testing)
        # engine.add_risk_manager(DrawdownProtection(threshold=drawdown_threshold))
        
        # 4. Run backtest (strategy will automatically compute required indicators)
        results = engine.run(strategy, data)
        
        # 5. Print performance summary
        print(f"\n{ticker} KDJ Cross Depression Strategy Results:")
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
        
        # Create comprehensive chart with configurable benchmark visibility
        # Set show_benchmark=False to hide benchmark and better scale for strategy performance
        # Set show_benchmark=True (default) to show both strategy and benchmark for comparison
        fig = plotter.create_comprehensive_chart(
            ticker=ticker, 
            base_strategy_name="KDJ Cross Depression", 
            log_scale=False,
            show_benchmark=True  # Default: ON - shows benchmark for comparison
        )
        
        # Save chart
        output_file = os.path.join(output_dir, f"{ticker}_kdj_cross_depression_strategy.html")
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
