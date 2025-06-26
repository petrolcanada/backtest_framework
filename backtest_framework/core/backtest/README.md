# Modular Backtest Engine

The backtest engine has been refactored into a modular architecture for better maintainability, testability, and extensibility.

## Architecture Overview

The engine is now composed of several specialized modules, each handling a specific aspect of backtesting:

### Core Components

1. **BacktestEngine** (`engine.py`) - Main orchestrator
2. **PortfolioManager** (`portfolio_manager.py`) - Position and cash management
3. **CostCalculator** (`cost_calculator.py`) - Trading costs and fees
4. **TradeExecutor** (`trade_executor.py`) - Signal execution and trade management
5. **PerformanceCalculator** (`performance_calculator.py`) - Metrics and analysis
6. **DataValidator** (`data_validator.py`) - Data validation and preparation

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
  - Position sizing with leverage support
  - Margin debt tracking
  - Long/short position management
  - Dividend processing
  - Equity calculations

### CostCalculator
- **Purpose**: Calculates all trading-related costs
- **Key Features**:
  - Commission and slippage calculations
  - Margin interest (with Federal Funds Rate integration)
  - Short borrowing costs
  - Execution price calculations

### TradeExecutor
- **Purpose**: Handles trade execution logic and signal processing
- **Key Features**:
  - T+1 signal execution
  - Buy/sell signal processing
  - Trade statistics tracking
  - Daily cost application

### PerformanceCalculator
- **Purpose**: Calculates comprehensive performance metrics
- **Key Features**:
  - Return and drawdown calculations
  - Risk metrics (Sharpe, Sortino ratios)
  - Benchmark comparison
  - Summary statistics generation

### DataValidator
- **Purpose**: Validates and prepares data for backtesting
- **Key Features**:
  - Data structure validation
  - Signal validation
  - Date filtering
  - Dividend data preparation
  - Multi-ticker alignment checks

## Benefits of Modular Architecture

### 1. **Maintainability**
- Each module has a single responsibility
- Easy to locate and fix bugs
- Clear separation of concerns
- Smaller, more focused code files

### 2. **Testability**
- Each module can be unit tested independently
- Mock dependencies for isolated testing
- Better test coverage and reliability

### 3. **Extensibility**
- Easy to add new features to specific modules
- Replace individual components without affecting others
- Support for different portfolio management strategies
- Pluggable risk management systems

### 4. **Reusability**
- Components can be used independently
- Share modules across different backtesting systems
- Build specialized engines for specific use cases

## Usage Examples

### Basic Usage (Same as Before)
```python
from backtest_framework.core.backtest import BacktestEngine
from backtest_framework.core.strategies.kdj_cross import KDJCrossStrategy

# Create engine with same interface as before
engine = BacktestEngine(
    initial_capital=10000,
    commission=0.001,
    leverage=2.0,
    enable_short_selling=True
)

# Run backtest (unchanged interface)
strategy = KDJCrossStrategy()
results = engine.run(strategy, data)
```

### Advanced Usage - Custom Components
```python
from backtest_framework.core.backtest import (
    BacktestEngine, PortfolioManager, CostCalculator
)

# Create custom portfolio manager with different settings
custom_portfolio = PortfolioManager(
    initial_capital=50000,
    position_sizing=0.25,  # Use 25% of capital per trade
    long_leverage=3.0,
    short_leverage=1.5
)

# Create custom cost calculator
custom_costs = CostCalculator(
    commission=0.0005,  # Lower commission
    slippage=0.001,
    short_borrow_rate=0.03  # Higher borrow rate
)

# Engine will use custom components
engine = BacktestEngine()
engine.portfolio_manager = custom_portfolio
engine.cost_calculator = custom_costs
```

### Accessing Individual Components
```python
# Access components for analysis or customization
engine = BacktestEngine()

# Get performance calculator for custom metrics
perf_calc = engine.performance_calculator
summary = perf_calc.get_summary_stats(results)

# Access portfolio state during backtest
portfolio = engine.portfolio_manager
current_position = portfolio.position
cash_balance = portfolio.cash

# Customize cost calculations
cost_calc = engine.cost_calculator
execution_price = cost_calc.calculate_execution_price(100.0, is_buy=True)
```

## Migration from Original Engine

The modular engine maintains **full backward compatibility** with the original interface:

- All original `BacktestEngine` constructor parameters work unchanged
- The `run()` method has the same signature and returns the same format
- All result columns and metrics are preserved
- Risk managers work the same way

No changes required to existing code that uses the engine!

## File Structure

```
backtest/
├── __init__.py                 # Module exports
├── engine.py                   # Main BacktestEngine (NEW)
├── engine_orig.py              # Original engine (backup)
├── portfolio_manager.py        # Portfolio state management (NEW)
├── cost_calculator.py          # Cost calculations (NEW)
├── trade_executor.py           # Trade execution logic (NEW)
├── performance_calculator.py   # Performance metrics (NEW)
├── data_validator.py           # Data validation (NEW)
└── risk_management.py          # Risk management (unchanged)
```

## Future Enhancements

The modular architecture enables easy addition of:

1. **Alternative Portfolio Managers**
   - Kelly criterion position sizing
   - Risk parity allocation
   - Mean reversion strategies

2. **Enhanced Cost Models**
   - Dynamic commission structures
   - Market impact models
   - Real-time borrowing rates

3. **Advanced Execution Models**
   - Market orders vs limit orders
   - Partial fill simulation
   - Latency modeling

4. **Sophisticated Performance Analysis**
   - Factor attribution
   - Rolling performance metrics
   - Regime analysis

5. **Alternative Data Validators**
   - Real-time data validation
   - Corporate action adjustments
   - Alternative data sources

## Testing Strategy

Each module can be tested independently:

```python
# Example: Test PortfolioManager in isolation
import unittest
from backtest.portfolio_manager import PortfolioManager

class TestPortfolioManager(unittest.TestCase):
    def setUp(self):
        self.pm = PortfolioManager(10000, position_sizing=0.5)
    
    def test_long_position_entry(self):
        success = self.pm.enter_long_position(100.0)
        self.assertTrue(success)
        self.assertGreater(self.pm.position, 0)
    
    def test_equity_calculation(self):
        self.pm.enter_long_position(100.0)
        equity = self.pm.get_current_equity(105.0)
        self.assertGreater(equity, self.pm.initial_capital)
```

## Performance Considerations

The modular architecture maintains performance through:

- **Minimal overhead**: Module boundaries don't add computational cost
- **Efficient state management**: Single-pass execution with state updates
- **Vectorized calculations**: Performance metrics use pandas vectorization
- **Memory efficiency**: Same memory footprint as original engine

## Conclusion

The modular backtest engine provides:
- ✅ **Same functionality** as the original engine
- ✅ **Better maintainability** through separation of concerns
- ✅ **Enhanced testability** with isolated components
- ✅ **Greater extensibility** for future enhancements
- ✅ **Full backward compatibility** with existing code

The refactoring improves code quality while preserving all existing functionality and performance characteristics.
