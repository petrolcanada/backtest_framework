# Modular Backtest Engine

The backtest engine has been refactored into a modular architecture for better maintainability, testability, and extensibility.

## Architecture Overview

The engine is composed of several specialized modules, each handling a specific aspect of backtesting:

### Core Components

1. **BacktestEngine** (`engine.py`) - Main orchestrator
2. **PortfolioManager** (`portfolio_manager.py`) - Position and cash management
3. **CostCalculator** (`cost_calculator.py`) - Trading costs and fees
4. **TradeExecutor** (`trade_executor.py`) - Signal execution and trade management
5. **TradeTracker** (`trade_tracker.py`) - Individual trade tracking and P&L calculation
6. **PerformanceCalculator** (`performance_calculator.py`) - Metrics and analysis
7. **DataValidator** (`data_validator.py`) - Data validation and preparation
8. **RiskManagement** (`risk_management.py`) - Risk management strategies

## Module Responsibilities

### BacktestEngine
- **Purpose**: Main orchestrator that coordinates all other modules
- **Key Methods**:
  - `run()`: Execute a complete backtest
  - `add_risk_manager()`: Add risk management modules
  - `get_summary_stats()`: Get performance summary
  - `reset()`: Reset for new backtest

### PortfolioManager
- **Purpose**: Manages portfolio state including positions, cash, and margin
- **Key Features**:
  - Position sizing with leverage support (separate long/short leverage)
  - Margin debt tracking
  - Long/short position management
  - Dividend processing
  - MMF interest on cash balances
  - Accurate P&L calculation

### CostCalculator
- **Purpose**: Calculates all trading-related costs
- **Key Features**:
  - Commission and slippage calculations
  - Margin interest (with Federal Funds Rate integration)
  - Short borrowing costs
  - MMF (Money Market Fund) interest rates
  - Execution price calculations

### TradeExecutor
- **Purpose**: Handles trade execution logic and signal processing
- **Key Features**:
  - T+1 signal execution
  - Buy/sell signal processing with current date tracking
  - Integration with TradeTracker for accurate P&L
  - Daily cost and dividend application
  - Trade statistics with enhanced metrics

### TradeTracker
- **Purpose**: Tracks individual trades from entry to exit with accurate P&L calculation
- **Key Features**:
  - Individual trade lifecycle tracking
  - Separate P&L calculation for long and short trades
  - Dividend and borrowing cost attribution
  - Win rate calculation based on actual P&L
  - Comprehensive trade statistics (avg win/loss, profit factor, etc.)
  - Trade log generation for analysis

### PerformanceCalculator
- **Purpose**: Calculates comprehensive performance metrics
- **Key Features**:
  - Return and drawdown calculations
  - Risk metrics (Sharpe, Sortino ratios)
  - Benchmark comparison with dividend reinvestment
  - CAGR and other time-based metrics
  - Summary statistics generation

### DataValidator
- **Purpose**: Validates and prepares data for backtesting
- **Key Features**:
  - Data structure validation
  - Signal validation
  - Date filtering
  - Dividend data preparation
  - Multi-ticker alignment checks

### RiskManagement
- **Purpose**: Provides risk management strategies
- **Implementations**:
  - `DrawdownProtection`: Exit positions when drawdown exceeds threshold
  - `TrailingStop`: Trailing stop loss implementation

## Enhanced Trade Tracking System

The new trade tracking system provides accurate win rate calculation by:

1. **Trade Lifecycle**: Tracks each trade from entry to exit
2. **P&L Components**: 
   - Entry/exit prices with commission and slippage
   - Dividends received (long) or paid (short)
   - Borrowing costs (margin interest or short borrow costs)
3. **Win Rate**: Based on actual P&L including all costs and dividends

### Trade P&L Calculation

**Long Trades:**
```
P&L = Exit Proceeds - Entry Cost + Dividends - Borrowing Costs
```

**Short Trades:**
```
P&L = Entry Proceeds - Exit Cost - Dividends - Borrowing Costs
```

## Usage Examples

### Basic Usage
```python
from backtest_framework.core.backtest import BacktestEngine
from backtest_framework.core.strategies.kdj_cross import GoldenDeadCrossStrategyMonthly

# Create engine
engine = BacktestEngine(
    initial_capital=10000,
    commission=0.001,
    leverage={"long": 2.0, "short": 3.0},  # Separate leverage
    enable_short_selling=True,
    include_dividends=True
)

# Run backtest
strategy = GoldenDeadCrossStrategyMonthly()
results = engine.run(strategy, data)

# Get trade statistics
trade_stats = engine.trade_executor.get_trade_stats()
print(f"Win Rate: {trade_stats['win_rate'] * 100:.2f}%")
print(f"Profit Factor: {trade_stats['profit_factor']:.2f}")

# Get detailed trade log
trade_log = engine.trade_executor.get_trade_log()
trade_log.to_csv("trade_log.csv")
```

### Accessing Components
```python
# Access portfolio state
portfolio = engine.portfolio_manager
print(f"Current Position: {portfolio.position}")
print(f"Cash Balance: ${portfolio.cash:,.2f}")
print(f"Margin Debt: ${portfolio.margin_cash:,.2f}")

# Access trade tracker
tracker = engine.trade_executor.trade_tracker
print(f"Open Trade: {tracker.current_trade is not None}")

# Get comprehensive trade stats
stats = tracker.get_stats()
print(f"Average Win: ${stats['avg_win']:,.2f}")
print(f"Average Loss: ${stats['avg_loss']:,.2f}")
```

## File Structure

```
backtest/
├── __init__.py                 # Module exports
├── engine.py                   # Main BacktestEngine
├── portfolio_manager.py        # Portfolio state management
├── cost_calculator.py          # Cost calculations
├── trade_executor.py           # Trade execution with enhanced tracking
├── trade_tracker.py            # Individual trade tracking
├── performance_calculator.py   # Performance metrics
├── data_validator.py           # Data validation
├── risk_management.py          # Risk management strategies
└── README.md                   # This documentation
```

## Key Features

### 1. **Accurate Win Rate Calculation**
- Tracks each trade from entry to exit
- Includes all costs (commission, slippage, borrowing)
- Accounts for dividends received/paid
- Differentiates between long and short trade P&L

### 2. **Comprehensive Cost Tracking**
- Margin interest based on Federal Funds Rate
- Short borrowing costs
- MMF interest on cash balances
- Commission and slippage on every trade

### 3. **Flexible Leverage**
- Separate leverage for long and short positions
- Automatic margin debt tracking
- Interest charges on borrowed funds

### 4. **Enhanced Metrics**
- Win rate based on actual P&L
- Average win/loss amounts
- Profit factor
- Best/worst trade tracking
- Trade duration analysis

## Migration Notes

The engine maintains full backward compatibility while adding new features:

- All original parameters work unchanged
- New optional parameters for enhanced functionality
- Results DataFrame includes all original columns
- Additional metrics available through trade_executor

## Performance Considerations

The modular architecture maintains performance through:

- **Efficient state management**: Single-pass execution
- **Minimal overhead**: Clean module boundaries
- **Vectorized calculations**: Using pandas operations
- **Memory efficiency**: Same footprint as original

## Conclusion

The consolidated backtest engine provides:
- ✅ **Accurate win rate** calculation with full P&L tracking
- ✅ **Clean architecture** with consolidated modules
- ✅ **Enhanced metrics** for better strategy analysis
- ✅ **Full backward compatibility** with existing code
- ✅ **Comprehensive trade logging** for detailed analysis

The system now properly tracks every aspect of trading performance, from individual trades to portfolio-wide metrics.
