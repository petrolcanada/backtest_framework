# Enhanced Backtest Framework - Max Drawdown & Trade Details

## Overview

This document describes the enhancements made to the backtest framework's `TradeTracker` and `TradeExecutor` modules to provide comprehensive trade detail information and max drawdown tracking.

## New Features

### 1. Enhanced Max Drawdown Tracking

The `TradeTracker` now provides real-time drawdown monitoring with the following capabilities:

#### Drawdown Metrics
- **Current Drawdown**: Real-time drawdown from equity peak
- **Max Drawdown**: Maximum drawdown experienced during the backtest
- **Drawdown Duration**: Current consecutive days in drawdown
- **Max Drawdown Duration**: Longest period spent in drawdown
- **Equity Peak**: Highest equity value achieved
- **Drawdown Start Date**: When current drawdown period began

#### Implementation Details
```python
# Update drawdown tracking during backtest loop
drawdown_metrics = trade_executor.update_trade_tracking(
    current_date=date,
    current_price=current_price,
    current_equity=current_portfolio_equity
)
```

### 2. Enhanced Trade Detail Information

#### Maximum Adverse/Favorable Excursion (MAE/MFE)
- **MAE**: Tracks the maximum loss experienced during each trade
- **MFE**: Tracks the maximum profit experienced during each trade
- Provides insights into trade timing and execution quality

#### Extended Trade Data
Each trade now records:
- Entry and exit equity levels
- Dividends received during the trade
- Borrowing costs incurred
- Maximum adverse excursion (MAE)
- Maximum favorable excursion (MFE)
- Trade duration in days

#### Consecutive Win/Loss Tracking
- Maximum consecutive winning trades
- Maximum consecutive losing trades
- Helps identify strategy momentum and risk periods

### 3. Enhanced Statistics and Analysis

#### Comprehensive Trade Statistics
```python
enhanced_stats = trade_executor.get_trade_stats()
```

Returns:
- Traditional metrics (win rate, profit factor, etc.)
- Drawdown metrics (max drawdown, duration)
- Execution metrics (MAE, MFE averages)
- Consecutive streaks (wins/losses)
- Equity tracking data

#### Detailed Trade Log
```python
trade_log = trade_executor.get_trade_log()
```

Provides a DataFrame with all trade details including:
- Entry/Exit dates and prices
- Position sizes and costs
- P&L and percentage returns
- MAE/MFE values
- Duration and equity levels

#### Drawdown Period Analysis
```python
drawdown_periods = trade_executor.get_drawdown_periods(min_duration_days=5)
```

Returns detailed information about significant drawdown periods:
- Start and end dates
- Duration in days
- Peak and trough equity values
- Maximum drawdown percentage

#### Equity Curve with Drawdown Data
```python
equity_curve = trade_executor.get_equity_curve()
```

Returns equity curve DataFrame with:
- Date-indexed equity values
- Running equity peaks
- Drawdown series (absolute and percentage)

### 4. Enhanced Summary Reporting

#### Comprehensive Summary
```python
summary = trade_executor.get_enhanced_summary()
```

Provides a complete performance overview including:
- **Trade Statistics**: Total trades, win rate, profit factor
- **P&L Metrics**: Average win/loss, best/worst trades
- **Drawdown Metrics**: Max drawdown, duration, equity peak
- **Execution Metrics**: MAE/MFE, consecutive streaks
- **Cost Metrics**: Dividends, borrowing costs
- **Equity Statistics**: Volatility, number of drawdown periods

## Key Modifications

### TradeTracker Class Enhancements

#### New Methods
- `update_trade_excursions(current_price)`: Updates MAE/MFE for active trade
- `update_equity_tracking(date, equity)`: Updates drawdown tracking
- `get_equity_curve()`: Returns equity curve with drawdown data
- `_calculate_consecutive_streaks()`: Calculates win/loss streaks

#### Enhanced Trade Data Structure
```python
@dataclass
class Trade:
    # ... existing fields ...
    max_adverse_excursion: Optional[float] = None
    max_favorable_excursion: Optional[float] = None
    entry_equity: Optional[float] = None
    exit_equity: Optional[float] = None
```

### TradeExecutor Class Enhancements

#### New Methods
- `update_trade_tracking()`: Updates trade execution and drawdown metrics
- `get_equity_curve()`: Access to equity curve data
- `get_drawdown_periods()`: Detailed drawdown period analysis
- `get_enhanced_summary()`: Comprehensive performance summary

#### Modified Methods
- `execute_buy_signal()` and `execute_sell_signal()`: Now accept current_equity parameter
- `get_trade_stats()`: Returns enhanced statistics including drawdown metrics

## Usage Example

```python
from core.backtest.trade_executor import TradeExecutor
from core.backtest.portfolio_manager import PortfolioManager
from core.backtest.cost_calculator import CostCalculator

# Initialize components
portfolio_manager = PortfolioManager(initial_capital=100000)
cost_calculator = CostCalculator(commission_rate=0.001)
trade_executor = TradeExecutor(portfolio_manager, cost_calculator)

# During backtest loop
for date, row in data.iterrows():
    current_price = row['Close']
    current_equity = portfolio_manager.get_current_equity(current_price)
    
    # Execute trades
    if row['buy_signal']:
        trade_executor.execute_buy_signal(
            current_price=current_price,
            execution_price=row['Open'],
            current_date=date,
            current_equity=current_equity
        )
    
    # Update tracking
    drawdown_metrics = trade_executor.update_trade_tracking(
        current_date=date,
        current_price=current_price,
        current_equity=current_equity
    )

# Analysis
enhanced_summary = trade_executor.get_enhanced_summary()
trade_log = trade_executor.get_trade_log()
drawdown_periods = trade_executor.get_drawdown_periods()
equity_curve = trade_executor.get_equity_curve()
```

## Benefits

1. **Risk Management**: Real-time drawdown monitoring helps assess strategy risk
2. **Trade Quality**: MAE/MFE analysis reveals execution timing quality
3. **Performance Analysis**: Comprehensive metrics for strategy evaluation
4. **Detailed Reporting**: Complete trade-by-trade analysis capability
5. **Drawdown Analysis**: Understand when and how losses occur
6. **Strategy Optimization**: Identify patterns in consecutive wins/losses

## Integration with Existing Code

The enhancements are backward compatible. Existing code will continue to work, but to access the new features, you need to:

1. Pass `current_equity` parameter to trade execution methods
2. Call `update_trade_tracking()` during the backtest loop
3. Use the new analysis methods for enhanced reporting

## File Changes

### Modified Files
- `core/backtest/trade_tracker.py`: Enhanced with drawdown tracking and MAE/MFE
- `core/backtest/trade_executor.py`: Added comprehensive analysis methods

### New Files
- `examples/enhanced_backtest_example.py`: Complete example demonstrating new features
- `ENHANCED_BACKTEST_FEATURES.md`: This documentation file

The enhanced framework now provides institutional-grade trade analysis capabilities while maintaining the simplicity and flexibility of the original design.
