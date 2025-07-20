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
from backtest_framework.core.strategies.kdj_mfi_early_detection import KDJMFIEarlyDetectionStrategy
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
    print("KDJ MFI Early Detection Strategy Demo")
    print("=" * 45)
    
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
        data = loader.load(ticker, period="10y", resample_period="D")
        print(f"Loaded {len(data)} rows of data from {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
        
        # 2. Initialize strategy with parameters (including custom indicator parameters)
        print("\nInitializing KDJ MFI Early Detection strategy...")
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
        
        print(f"Strategy Configuration:")
        print(f"  • Required J up days: {strategy.required_up_days}")
        print(f"  • Signal limiting: DISABLED (showing all qualifying signals)")
        print(f"  • ADX down days: {strategy.required_adx_down_days}")
        print(f"  • Death cross buys: {'Enabled' if strategy.enable_death_cross_buys else 'Disabled'}")
        print(f"  • Required indicators: {len(strategy.required_indicators)}")
        
        # Print detailed indicator configuration
        print("\n📊 Indicator Parameter Configuration:")
        strategy.print_indicator_config()
        
        # 3. Setup backtest engine with configuration
        print("Setting up backtest engine...")
        engine = BacktestEngine(
            initial_capital=initial_capital, 
            commission=commission,
            leverage={"long": 1.0, "short": 1.0},    # 2x long leverage, 1x short leverage
            position_sizing=1.0,                     # Use 100% of capital per trade
            enable_short_selling=False                # Enable short selling for long/short strategy
        )
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
        print(f"   • Signal Limiting: DISABLED (all qualifying signals shown)")
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
        
        # J Momentum Analysis
        if 'j_consecutive_up_days' in results.columns:
            max_consecutive_up = results['j_consecutive_up_days'].max()
            avg_consecutive_up = results['j_consecutive_up_days'].mean()
            print(f"Max J Consecutive Up Days: {max_consecutive_up}")
            print(f"Avg J Consecutive Up Days: {avg_consecutive_up:.1f}")
        
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
        
        # Position analysis
        long_periods = (results['position_type'] == 'long').sum()
        short_periods = (results['position_type'] == 'short').sum()
        total_periods = len(results)
        
        print("\n📊 Position Analysis:")
        print(f"Long Periods: {long_periods} days ({long_periods/total_periods*100:.1f}%)")
        print(f"Short Periods: {short_periods} days ({short_periods/total_periods*100:.1f}%)")
        print(f"Cash Periods: {total_periods - long_periods - short_periods} days ({(total_periods - long_periods - short_periods)/total_periods*100:.1f}%)")
        
        # 6. Demonstrate Dynamic Parameter Adjustment
        print("\n🔧 Dynamic Parameter Adjustment Demo:")
        print("=" * 50)
        
        # Show current ADX parameters
        current_adx_params = strategy.get_indicator_params('ADX')
        print(f"Current ADX parameters: {current_adx_params}")
        
        # Demonstrate parameter adjustment (for next run)
        print("\nTo adjust parameters for next run using streamlined utility functions:")
        print("# Direct indicator parameter setting (fully standardized 'period' naming):")
        print("strategy.set_indicator_params('ADX', period=21, sma_period=7)")
        print("strategy.set_indicator_params('MONTHLY_KDJ', period=220, signal=73)")
        print("strategy.set_indicator_params('MFI', period=30)")
        print("strategy.set_indicator_params('RSI', period=21)")
        print("")
        print("# Using utility functions for batch updates:")
        print("from backtest_framework.core.utils.helpers import clean_params, filter_empty_dicts")
        print("batch_params = filter_empty_dicts({")
        print("    'ADX': clean_params(period=21, sma_period=7),")
        print("    'MFI': clean_params(period=30),")
        print("    'RSI': clean_params(period=21)")
        print("})") 
        print("strategy.update_indicator_params(batch_params)")
        print("")
        print("# Strategy-specific method:")
        print("strategy.set_strategy_specific_params(required_up_days=5, required_adx_down_days=3)")    
        # Alternative: full parameter dictionary approach
        print("\nAlternative - bulk parameter update:")
        print("custom_params = {")
        print("    'ADX': {'window': 21, 'sma_length': 7},")
        print("    'MONTHLY_KDJ': {'window': 14},")
        print("    'MFI': {'window': 20}")
        print("}")
        print("strategy.update_indicator_params(custom_params)")
        
        # 7. Demonstrate Dynamic Indicator System
        print("\n🔍 Dynamic Indicator System Demo:")
        print("=" * 50)
        
        # Import and test the dynamic indicator coordinator
        from backtest_framework.core.visualization.components.dynamic_indicators import DynamicIndicatorCoordinator
        from backtest_framework.core.indicators.registry import IndicatorRegistry
        
        # Create dynamic indicator coordinator
        dynamic_coordinator = DynamicIndicatorCoordinator(results)
        debug_info = dynamic_coordinator.get_debug_info()
        
        print("📊 Computed Indicators:")
        computed_indicators = debug_info['computed_indicators']
        for indicator in computed_indicators[:10]:  # Show first 10 to avoid clutter
            indicator_info = IndicatorRegistry.get(indicator)
            viz_class = indicator_info.get('visualization_class', 'None')
            print(f"  • {indicator}: outputs={indicator_info['outputs']}, viz_class={viz_class}")
        
        if len(computed_indicators) > 10:
            print(f"  ... and {len(computed_indicators) - 10} more indicators")
        
        print(f"\n🎨 Available Visualizations: {debug_info['available_visualizations']}")
        print(f"📋 Total Registered Indicators: {len(debug_info['registered_indicators'])}")
        print(f"🏗️  Visualization Registry: {debug_info['visualization_registry']}")
        
        # 8. Create visualization
        print("\nGenerating visualization...")
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
        os.makedirs(output_dir, exist_ok=True)
        
        # Create plotter with the data and results
        plotter = Plotter(data=data, results=results, engine=engine)
        
        # Create comprehensive chart using the new component-based system
        print("Creating comprehensive strategy analysis chart...")
        fig = plotter.create_comprehensive_chart(
            ticker=ticker, 
            base_strategy_name="KDJ MFI Early Detection", 
            log_scale=False
        )
        
        # Save chart
        output_file = os.path.join(output_dir, f"{ticker}_kdj_mfi_early_detection_strategy.html")
        plotter.save(output_file)
        print(f"Chart saved to: {output_file}")
        
        # Open chart in browser
        plotter.open_in_browser(output_file)
        
        # 9. Add debugging section for signal analysis
        print("\n🔍 SIGNAL DEBUGGING ANALYSIS:")
        print("=" * 50)
        
        # Find all buy signal dates
        buy_signal_dates = results[results['buy_signal'] == 1].index
        print(f"Total buy signals generated: {len(buy_signal_dates)}")
        
        if len(buy_signal_dates) > 0:
            print("\nAll buy signal dates:")
            for i, date in enumerate(buy_signal_dates[:20], 1):  # Show first 20
                print(f"  {i:2d}. {date.strftime('%Y-%m-%d')}")
            if len(buy_signal_dates) > 20:
                print(f"  ... and {len(buy_signal_dates) - 20} more signals")
        
        # Look specifically for December 2023 signals (your mentioned dates)
        dec_2023_signals = [d for d in buy_signal_dates if d.year == 2023 and d.month == 12]
        if dec_2023_signals:
            print(f"\n📅 December 2023 signals (including your mentioned dates):")
            for date in dec_2023_signals:
                idx = results.index.get_loc(date)
                date_str = date.strftime('%Y-%m-%d')
                print(f"\n  {date_str}:")
                print(f"    cross_status: {results['cross_status'].iloc[idx]}")
                print(f"    j_consecutive_up_days: {results['j_consecutive_up_days'].iloc[idx]}")
                print(f"    monthly_d_slope: {results['monthly_d_slope'].iloc[idx]:.6f}")
                
                # Find most recent golden cross
                recent_golden = results['golden_cross'].iloc[max(0, idx-30):idx+1]
                golden_dates = recent_golden[recent_golden == 1].index
                if len(golden_dates) > 0:
                    latest_golden = golden_dates[-1]
                    days_since = idx - results.index.get_loc(latest_golden)
                    print(f"    latest_golden_cross: {latest_golden.strftime('%Y-%m-%d')} ({days_since} days ago)")
        
        # Check for rapid-fire signals (within 5 days)
        print(f"\n⚡ RAPID-FIRE SIGNAL DETECTION:")
        rapid_fire_count = 0
        for i in range(len(buy_signal_dates) - 1):
            current_date = buy_signal_dates[i]
            next_date = buy_signal_dates[i + 1]
            days_apart = (next_date - current_date).days
            
            if days_apart <= 5:
                rapid_fire_count += 1
                print(f"  {current_date.strftime('%Y-%m-%d')} -> {next_date.strftime('%Y-%m-%d')} ({days_apart} days apart) ⚠️")
        
        if rapid_fire_count == 0:
            print("  ✅ No rapid-fire signals detected (all signals >5 days apart)")
        else:
            print(f"  ⚠️ Found {rapid_fire_count} rapid-fire signal pairs")
        
        # 10. Strategy insights and recommendations
        print(f"\n💡 Strategy Insights:")
        
        # Signal efficiency
        if buy_signals > 0:
            signal_efficiency = (results['trade_count'].iloc[-1] / buy_signals) * 100
            print(f"Signal Efficiency: {signal_efficiency:.1f}% (trades executed vs signals generated)")
        
        # Performance vs complexity
        total_indicators = len(strategy.required_indicators)
        print(f"Indicator Complexity: {total_indicators} indicators used")
        
        # Risk-adjusted performance
        if results['sharpe_ratio'].iloc[-1] > 1.0:
            print("✅ Strong risk-adjusted performance (Sharpe > 1.0)")
        elif results['sharpe_ratio'].iloc[-1] > 0.5:
            print("⚠️ Moderate risk-adjusted performance (Sharpe 0.5-1.0)")
        else:
            print("❌ Weak risk-adjusted performance (Sharpe < 0.5)")
        
        # Drawdown assessment
        max_dd = results['max_drawdown'].iloc[-1] * 100
        if max_dd < 10:
            print("✅ Low drawdown risk (<10%)")
        elif max_dd < 20:
            print("⚠️ Moderate drawdown risk (10-20%)")
        else:
            print("❌ High drawdown risk (>20%)")
        
        # Final summary for debugging
        print(f"\n📋 DEBUGGING SUMMARY:")
        print(f"Total buy signals: {len(buy_signal_dates)}")
        print(f"December 2023 signals: {len(dec_2023_signals)}")
        print(f"Rapid-fire signals: {rapid_fire_count}")
        if 'golden_cross' in results.columns:
            total_golden_crosses = (results['golden_cross'] == 1).sum()
            if total_golden_crosses > 0:
                avg_signals_per_cross = len(buy_signal_dates) / total_golden_crosses
                print(f"Average signals per golden cross: {avg_signals_per_cross:.2f}")
    
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print(f"\nCompleted in {timer.elapsed_str()}")


if __name__ == "__main__":
    main()
