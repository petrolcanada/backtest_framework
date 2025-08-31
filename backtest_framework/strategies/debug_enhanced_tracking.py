"""
Debug version to test enhanced tracking functionality.
"""
import os
import sys

# Add current directory and parent to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
backtest_framework_dir = os.path.dirname(current_dir)
scripts_dir = os.path.dirname(backtest_framework_dir)
sys.path.append(scripts_dir)

from backtest_framework.core.data.loader import DataLoader
from backtest_framework.core.signals.kdj_cross_depression import KDJCrossDepressionStrategy
from backtest_framework.core.backtest.engine import BacktestEngine
from backtest_framework.core.utils.helpers import Timer, suppress_warnings

# Import indicator modules
from backtest_framework.core.indicators import kdj, sma, adx, mfi, rsi
from backtest_framework.core.indicators import kdj_derived_factors, adx_derived_factors, mfi_derived_factors, rsi_derived_factors

suppress_warnings()

def main():
    """Run debug test for enhanced tracking."""
    timer = Timer()
    
    # Configuration
    ticker = "NVDA"
    initial_capital = 10000
    
    try:
        # Load data
        data_dir = os.path.join(os.path.expanduser("~"), "local_script", 
                               "Local Technical Indicator Data", "security_data")
        loader = DataLoader(data_dir=data_dir)
        data = loader.load(ticker, period="5y", resample_period="D", mode="no_reload")
        
        # Initialize strategy
        strategy = KDJCrossDepressionStrategy(
            depression_threshold=40,
            j_extreme_threshold=100.0,
            use_monthly_kdj=True,
            kdj_period=9 * 5,
            kdj_signal=3 * 5,
            daily_kdj_period=9,
            daily_kdj_signal=3,
        )
        
        # Setup backtest engine
        engine = BacktestEngine(
            initial_capital=initial_capital, 
            commission=0.001,
            slippage=0.001,
            leverage={"long": 1.0, "short": 1.0},
            position_sizing=1.0,
            enable_short_selling=False
        )
        
        # Run backtest
        print("Running backtest...")
        results = engine.run(strategy, data)
        
        # Debug: Check if enhanced tracking was used
        print(f"\nDEBUG - Trade Executor Stats:")
        trade_stats = engine.trade_executor.get_trade_stats()
        for key, value in trade_stats.items():
            print(f"  {key}: {value}")
        
        print(f"\nDEBUG - Enhanced Summary:")
        enhanced_summary = engine.trade_executor.get_enhanced_summary()
        for key, value in enhanced_summary.items():
            print(f"  {key}: {value}")
        
        # Debug: Check trade log
        print(f"\nDEBUG - Trade Log:")
        trade_log = engine.trade_executor.get_trade_log()
        print(f"Trade log shape: {trade_log.shape}")
        print("Trade log columns:", list(trade_log.columns))
        
        if not trade_log.empty:
            print("\nFirst trade details:")
            first_trade = trade_log.iloc[0]
            for col in trade_log.columns:
                print(f"  {col}: {first_trade[col]}")
        
        # Debug: Check equity curve
        equity_curve = engine.trade_executor.get_equity_curve()
        print(f"\nDEBUG - Equity Curve:")
        print(f"Equity curve shape: {equity_curve.shape}")
        if not equity_curve.empty:
            print("Equity curve columns:", list(equity_curve.columns))
            print("First few equity points:")
            print(equity_curve.head())
            print("Last few equity points:")
            print(equity_curve.tail())
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print(f"Completed in {timer.elapsed_str()}")

if __name__ == "__main__":
    main()
