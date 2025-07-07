"""
Demo script for the KDJ MFI Early Detection Strategy using the modular backtesting framework.
This version uses Streamlit for visualization instead of static HTML.
"""
import os
import sys
import pickle
import subprocess

# Add current directory and parent to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
scripts_dir = os.path.dirname(current_dir)
sys.path.append(scripts_dir)

# Import from the backtesting framework
from backtest_framework.core.data.loader import DataLoader
from backtest_framework.core.strategies.kdj_mfi_early_detection import KDJMFIEarlyDetectionStrategy
from backtest_framework.core.backtest.engine import BacktestEngine
from backtest_framework.core.backtest.risk_management import DrawdownProtection
from backtest_framework.core.utils.helpers import Timer, suppress_warnings

# Import indicator modules to ensure they're registered
from backtest_framework.core.indicators import kdj, sma, adx, mfi, rsi
from backtest_framework.core.indicators import kdj_derived_factors, adx_derived_factors, mfi_derived_factors, rsi_derived_factors

# Suppress warnings
suppress_warnings()


def main():
    """Run a demo backtest for the KDJ MFI Early Detection Strategy with Streamlit visualization."""
    print("KDJ MFI Early Detection Strategy Demo (Streamlit Version)")
    print("=" * 60)
    
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
        
        # 2. Initialize strategy with parameters
        print("\nInitializing KDJ MFI Early Detection strategy...")
        strategy = KDJMFIEarlyDetectionStrategy(
            required_up_days=7,              # J must rise for 7 days
            required_down_days=4,            # For flip detection
            required_adx_down_days=5,        # ADX down requirement  
            max_buy_signals_per_cross=2,     # Max 2 buys per golden cross
            enable_death_cross_buys=False    # Disable alternative buys initially
        )
        
        print(f"Strategy Configuration:")
        print(f"  • Required J up days: {strategy.required_up_days}")
        print(f"  • Max signals per cross: {strategy.max_buy_signals_per_cross}")
        print(f"  • ADX down days: {strategy.required_adx_down_days}")
        print(f"  • Death cross buys: {'Enabled' if strategy.enable_death_cross_buys else 'Disabled'}")
        print(f"  • Required indicators: {len(strategy.required_indicators)}")
        
        # 3. Setup backtest engine with configuration
        print("Setting up backtest engine...")
        engine = BacktestEngine(
            initial_capital=initial_capital, 
            commission=commission,
            leverage={"long": 1.0, "short": 1.0},    # 1x leverage for both long and short
            position_sizing=1.0,                     # Use 100% of capital per trade
            enable_short_selling=True                # Enable short selling for long/short strategy
        )
        # Optionally add risk manager
        # engine.add_risk_manager(DrawdownProtection(threshold=drawdown_threshold))
        
        # 4. Run backtest (strategy will automatically compute required indicators)
        print("Running backtest...")
        print("Computing indicators and derived factors...")
        results = engine.run(strategy, data)
        
        # 5. Analyze strategy-specific metrics
        print(f"\n{ticker} KDJ MFI Early Detection Strategy Results:")
        print("=" * 65)
        print("📊 Strategy Configuration:")
        print(f"   • Algorithm: KDJ MFI Early Detection")
        print(f"   • Entry Logic: Golden Cross + J momentum + D slope")
        print(f"   • Exit Logic: Death Cross occurrence")
        print(f"   • Signal Limiting: Max {strategy.max_buy_signals_per_cross} buys per golden cross")
        print(f"   • Long Leverage: {engine.long_leverage:.1f}x")
        print(f"   • Short Leverage: {engine.short_leverage:.1f}x")
        print(f"   • Position Sizing: {engine.position_sizing*100:.0f}%")
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
        
        # Strategy-specific signal analysis
        buy_signals = (results['buy_signal'] == 1).sum()
        sell_signals = (results['sell_signal'] == 1).sum()
        
        print(f"\n📡 KDJ MFI Signal Analysis:")
        print(f"Buy Signals Generated: {buy_signals}")
        print(f"Sell Signals Generated: {sell_signals}")
        
        # Golden/Death Cross Analysis
        if 'golden_cross' in results.columns and 'death_cross' in results.columns:
            golden_crosses = (results['golden_cross'] == 1).sum()
            death_crosses = (results['death_cross'] == 1).sum()
            print(f"Golden Crosses: {golden_crosses}")
            print(f"Death Crosses: {death_crosses}")
            
            if golden_crosses > 0:
                avg_signals_per_cross = buy_signals / golden_crosses
                print(f"Avg Buy Signals per Golden Cross: {avg_signals_per_cross:.1f}")
        
        # Benchmark Performance
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
            print(f"Strategy Outperformance: {outperformance:+.2f}%")
        
        # 6. Save results for Streamlit visualization
        print("\n💾 Saving results for Streamlit visualization...")
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
        os.makedirs(output_dir, exist_ok=True)
        
        # Create results data package
        results_data = {
            'data': data,
            'results': results,
            'engine': engine,
            'strategy_info': {
                'ticker': ticker,
                'name': 'KDJ MFI Early Detection',
                'params': {
                    'required_up_days': strategy.required_up_days,
                    'required_down_days': strategy.required_down_days,
                    'required_adx_down_days': strategy.required_adx_down_days,
                    'max_buy_signals_per_cross': strategy.max_buy_signals_per_cross,
                    'enable_death_cross_buys': strategy.enable_death_cross_buys
                }
            }
        }
        
        # Save as pickle file
        results_file = os.path.join(output_dir, f"{ticker}_kdj_mfi_streamlit_results.pkl")
        with open(results_file, 'wb') as f:
            pickle.dump(results_data, f)
        print(f"Results saved to: {results_file}")
        
        # 7. Launch Streamlit visualization
        print("\n🚀 Launching Streamlit visualization...")
        streamlit_app = os.path.join(
            current_dir,
            "core", 
            "visualization_streamlit", 
            "app.py"
        )
        
        # Check if streamlit app exists
        if os.path.exists(streamlit_app):
            print(f"Starting Streamlit app at: {streamlit_app}")
            print("Press Ctrl+C to stop the server when done.\n")
            
            try:
                # Launch Streamlit with the results file
                # Using sys.executable ensures we use the same Python interpreter
                process = subprocess.Popen([
                    sys.executable, '-m', 'streamlit', 'run',
                    streamlit_app,
                    '--server.port', '8501',
                    '--server.headless', 'true',
                    '--browser.gatherUsageStats', 'false',
                    '--',
                    results_file
                ])
                
                # Wait for the process to complete
                process.wait()
                
            except KeyboardInterrupt:
                print("\n\nStreamlit server stopped by user.")
                if process:
                    process.terminate()
            except Exception as e:
                print(f"Failed to launch Streamlit: {e}")
                print("\nYou can manually run the visualization with:")
                print(f"streamlit run \"{streamlit_app}\" -- \"{results_file}\"")
        else:
            print(f"❌ Streamlit app not found at: {streamlit_app}")
            print("Please ensure the visualization_streamlit module is properly installed.")
            print("\nExpected structure:")
            print("backtest_framework/")
            print("├── core/")
            print("│   ├── visualization_streamlit/")
            print("│   │   ├── app.py")
            print("│   │   └── ...")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print(f"\n⏱️ Total execution time: {timer.elapsed_str()}")


if __name__ == "__main__":
    main()
