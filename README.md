# Investment Backtesting Framework

A comprehensive, modular Python framework for backtesting trading strategies with automatic dependency resolution, risk management, and advanced visualization capabilities.

## 🚀 Features

### Core Capabilities
- **Multi-Asset Support**: Backtest single securities or portfolios
- **Leverage & Short Selling**: Configurable leverage ratios for long/short positions
- **Risk Management**: Built-in drawdown protection and trailing stops
- **Dynamic Position Sizing**: Percentage-based position sizing with leverage support
- **Professional Visualizations**: Interactive dark-themed charts with performance metrics

### Technical Analysis
- **Decorator-Based Indicators**: Easy indicator registration with automatic dependency resolution
- **Built-in Indicators**: KDJ (daily/monthly), SMA, slopes, and derivatives
- **Extensible Architecture**: Simple framework for adding custom indicators
- **Snake Case Convention**: Consistent naming across all indicators

### Data Management
- **Multiple Data Sources**: Yahoo Finance integration with CSV caching
- **Flexible Time Periods**: Support for various timeframes and date ranges
- **Data Resampling**: Convert between different frequencies (daily, weekly, monthly)
- **Auto-Caching**: Local CSV storage for faster subsequent loads

## 📁 Project Structure

```
backtest_framework/
├── core/
│   ├── backtest/
│   │   ├── engine.py              # Main backtesting engine
│   │   └── risk_management.py     # Risk management modules
│   ├── data/
│   │   └── loader.py              # Data loading and caching
│   ├── indicators/
│   │   ├── calculator.py          # Indicator computation engine
│   │   ├── registry.py            # Decorator-based indicator registry
│   │   ├── resolver.py            # Dependency resolution
│   │   ├── kdj.py                 # KDJ indicator implementations
│   │   └── sma.py                 # SMA indicator implementations
│   ├── strategies/
│   │   ├── base.py                # Abstract strategy base class
│   │   └── kdj_cross.py           # KDJ golden/dead cross strategies
│   ├── utils/
│   │   └── helpers.py             # Utility functions and helpers
│   └── visualization/
│       └── plotter.py             # Interactive plotting engine
└── __init__.py                    # Framework exports
```

## 🛠 Installation

### Prerequisites
```bash
pip install pandas numpy yfinance pandas-ta plotly
```

### Framework Setup
The framework is designed to be imported directly from your local directory:

```python
import sys
sys.path.append(r'C:\Users\peli\local_script\Local Technical Indicator Data\scripts')

from backtest_framework import (
    DataLoader, BacktestEngine, GoldenDeadCrossStrategyMonthly,
    DrawdownProtection, Plotter
)
```

## 📊 Quick Start Example

```python
# 1. Load data
loader = DataLoader()
data = loader.load('AAPL', period='2y')

# 2. Create strategy
strategy = GoldenDeadCrossStrategyMonthly()

# 3. Configure backtest engine
engine = BacktestEngine(
    initial_capital=10000,
    commission=0.001,
    leverage={'long': 2.0, 'short': 1.5},
    enable_short_selling=True,
    position_sizing=0.8
)

# 4. Add risk management
risk_manager = DrawdownProtection(threshold=0.15)
engine.add_risk_manager(risk_manager)

# 5. Run backtest
results = engine.run(strategy, data)

# 6. Visualize results
plotter = Plotter(data, results, engine)
fig = plotter.plot_chart_with_benchmark(ticker='AAPL', base_strategy_name='Monthly KDJ')
fig.show()
```

## 🔧 Core Components

### 1. Data Loader (`DataLoader`)

Handles data acquisition with intelligent caching:

```python
# Single ticker
data = loader.load('TSLA', period='1y', resample_period='D')

# Multiple tickers
portfolio_data = loader.load(['AAPL', 'MSFT', 'GOOGL'], period='6mo')

# Custom date range
data = loader.load('SPY', start_date='2020-01-01', end_date='2023-12-31')

# Force fresh download
data = loader.load('NVDA', force_download=True)
```

### 2. Backtest Engine (`BacktestEngine`)

Sophisticated backtesting with leverage and risk management:

```python
engine = BacktestEngine(
    initial_capital=50000,           # Starting capital
    commission=0.001,                # 0.1% commission
    slippage=0.0005,                # 0.05% slippage
    leverage={'long': 3.0, 'short': 2.0},  # Asymmetric leverage
    enable_short_selling=True,       # Allow short positions
    position_sizing=0.6              # Use 60% of capital per trade
)
```

**Key Features:**
- **Dynamic Position Sizing**: Adjusts position size based on current portfolio value
- **Leverage Support**: Different leverage ratios for long/short positions
- **Realistic Execution**: Uses next-day open prices for signal execution
- **Performance Metrics**: Automatic calculation of Sharpe ratio, CAGR, max drawdown

### 3. Indicator System

#### Decorator-Based Registration
```python
from backtest_framework.core.indicators.registry import IndicatorRegistry

@IndicatorRegistry.register(
    name="CUSTOM_INDICATOR",
    inputs=["High", "Low", "Close"],
    params={"period": 14},
    outputs=["custom_value"]
)
def calculate_custom_indicator(data, period=14):
    # Your indicator logic here
    result = pd.DataFrame(index=data.index)
    result['custom_value'] = data['Close'].rolling(period).mean()
    return result
```

#### Available Indicators
- **KDJ**: `KDJ`, `MONTHLY_KDJ` with slopes
- **SMA**: `SMA`, `SMA_SLOPE`, `CLOSE_MINUS_SMA`
- **Extensible**: Easy to add custom indicators

### 4. Strategy Framework

#### Base Strategy Class
```python
from backtest_framework.core.strategies.base import BaseStrategy

class MyCustomStrategy(BaseStrategy):
    @property
    def required_indicators(self):
        return ["KDJ", "SMA"]
    
    def generate_signals(self, data):
        data['buy_signal'] = 0
        data['sell_signal'] = 0
        
        # Your signal logic here
        buy_condition = (data['j'] > data['k']) & (data['Close'] > data['sma'])
        data.loc[buy_condition, 'buy_signal'] = 1
        
        sell_condition = (data['j'] < data['k'])
        data.loc[sell_condition, 'sell_signal'] = 1
        
        return data
```

### 5. Risk Management

#### Drawdown Protection
```python
drawdown_protection = DrawdownProtection(threshold=0.20)  # 20% max drawdown
engine.add_risk_manager(drawdown_protection)
```

#### Trailing Stop
```python
trailing_stop = TrailingStop(
    initial_threshold=0.10,  # 10% initial stop
    trail_percent=0.50       # Protect 50% of gains
)
engine.add_risk_manager(trailing_stop)
```

### 6. Advanced Visualization

The framework includes a sophisticated plotting system with:

```python
plotter = Plotter(data, results, engine)

# Comprehensive chart with benchmark comparison
fig = plotter.plot_chart_with_benchmark(
    ticker='AAPL',
    base_strategy_name='Monthly KDJ',
    log_scale=True
)

# Features:
# - Price candlesticks with signals
# - Strategy vs Buy & Hold performance
# - Separate drawdown visualization
# - Technical indicators panel
# - Dynamic titles with configuration
# - Performance metrics integration
```

## 📈 Supported Strategies

### KDJ Golden/Dead Cross Strategies

#### Monthly KDJ Strategy
```python
strategy = GoldenDeadCrossStrategyMonthly()
# Uses monthly-equivalent periods (198/66 days)
# Signals: J line crosses above/below K line
# Filters: K and D slopes must be positive for buy signals
```

#### Daily KDJ Strategy
```python
strategy = GoldenDeadCrossStrategyDaily()
# Uses standard KDJ periods (9/3 days)
# Same signal logic as monthly version
```

## 🎯 Configuration Options

### Engine Configuration
```python
engine = BacktestEngine(
    initial_capital=100000,          # Starting capital
    commission=0.001,                # Commission rate (0.1%)
    slippage=0.0005,                # Slippage rate (0.05%)
    leverage=2.0,                   # Uniform leverage, or
    leverage={'long': 2.0, 'short': 1.5},  # Asymmetric leverage
    enable_short_selling=True,       # Enable/disable short selling
    position_sizing=0.8             # Fraction of capital per trade
)
```

### Portfolio Backtesting
```python
# Multiple tickers with different rebalancing methods
portfolio_data = {
    'AAPL': loader.load('AAPL', period='1y'),
    'MSFT': loader.load('MSFT', period='1y'),
    'GOOGL': loader.load('GOOGL', period='1y')
}

results = engine.run(
    strategy, 
    portfolio_data, 
    portfolio_rebalance='buy_sell'  # or 'equal_weight', 'buy_hold'
)
```

## 📊 Performance Metrics

The framework automatically calculates:

- **Total Return**: Overall portfolio performance
- **CAGR**: Compound Annual Growth Rate
- **Sharpe Ratio**: Risk-adjusted returns
- **Sortino Ratio**: Downside risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Trade Count**: Total number of trades executed

## 🔍 Advanced Features

### 1. Automatic Dependency Resolution
The framework automatically determines the correct order to compute indicators based on their dependencies.

### 2. Data Caching
Downloaded data is automatically cached to local CSV files for faster subsequent access.

### 3. Flexible Time Periods
Support for various Yahoo Finance period formats: '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'max'

### 4. Interactive Visualizations
All charts are interactive Plotly figures with:
- Zoom and pan capabilities
- Hover information
- Crosshair cursors
- Performance annotations
- Dark theme optimized for financial data

### 5. Realistic Trading Simulation
- Uses next-day open prices for signal execution
- Accounts for commission and slippage
- Supports partial position sizing
- Handles margin requirements for leverage

## 🐛 Error Handling

The framework includes comprehensive error handling:

```python
# Automatic fallback for data loading
try:
    data = loader.load('INVALID_TICKER')
except Exception as e:
    print(f"Error loading data: {e}")
    # Framework continues with empty DataFrame

# Dependency validation
try:
    IndicatorCalculator.compute(data, ['INVALID_INDICATOR'])
except ValueError as e:
    print(f"Indicator error: {e}")
    # Clear error messages about missing dependencies
```

## 🚧 Extending the Framework

### Adding Custom Indicators
1. Create a new file in `core/indicators/`
2. Use the `@IndicatorRegistry.register` decorator
3. Define inputs, parameters, and outputs
4. Implement the calculation function

### Adding Custom Strategies
1. Inherit from `BaseStrategy`
2. Implement `required_indicators` property
3. Implement `generate_signals` method
4. Return DataFrame with `buy_signal` and `sell_signal` columns

### Adding Custom Risk Managers
1. Create a class with an `apply(data)` method
2. Modify signal columns based on risk rules
3. Add to engine with `engine.add_risk_manager()`

## 📝 Best Practices

1. **Use Snake Case**: All new indicators should follow snake_case naming
2. **Separate Concerns**: Keep signal generation separate from risk management
3. **Test Incrementally**: Test strategies on small datasets first
4. **Cache Data**: Use the built-in caching for faster development iterations
5. **Visualize Results**: Always plot results to verify strategy behavior
6. **Risk Management**: Always use appropriate position sizing and risk controls

## 🤝 Contributing

When extending the framework:
1. Follow the existing architectural patterns
2. Use the decorator-based registration system
3. Add appropriate error handling
4. Update documentation for new features
5. Test with multiple securities and time periods

## 📄 License

This framework is designed for educational and research purposes. Please ensure compliance with data provider terms of service when using market data.

---

**Happy Backtesting! 📈**