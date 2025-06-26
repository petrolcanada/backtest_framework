# Investment Backtesting Framework

A comprehensive, modular Python framework for backtesting trading strategies with automatic dependency resolution, advanced risk management, realistic cost modeling, and professional-grade visualizations.

## 🚀 Key Features

### Core Capabilities
- **Multi-Asset Support**: Backtest single securities or portfolios with sophisticated rebalancing
- **Advanced Leverage & Short Selling**: Asymmetric leverage ratios for long/short positions
- **Realistic Cost Modeling**: Federal Funds Rate-based margin costs, short borrowing fees, and MMF interest
- **T+1 Execution**: Realistic next-day signal execution with commission and slippage
- **Dynamic Position Sizing**: Percentage-based position sizing with leverage support
- **Professional Visualizations**: Interactive dark-themed charts with comprehensive performance analysis

### Technical Analysis System
- **Decorator-Based Indicators**: Automatic registration with dependency resolution
- **Built-in Indicators**: KDJ (daily/monthly), SMA, slopes, and derivatives
- **Extensible Architecture**: Simple framework for adding custom indicators
- **Snake Case Convention**: Consistent naming across all indicators

### Advanced Risk Management
- **Drawdown Protection**: Automatic position exits on excessive drawdowns
- **Trailing Stops**: Profit protection with dynamic stop levels
- **Margin Requirements**: Realistic margin calculations with Federal Funds Rate integration
- **Short Borrowing Costs**: Time-based borrowing fees for short positions

### Data Management
- **Multiple Data Sources**: Yahoo Finance with intelligent CSV caching
- **Federal Funds Rate Integration**: Real-time FFR data from FRED for margin cost calculations
- **Flexible Time Periods**: Support for various timeframes and date ranges
- **Auto-Caching**: Local storage for faster subsequent loads
- **Dividend Support**: Comprehensive dividend processing and analysis

## 📁 Project Structure

```
backtest_framework/
├── core/
│   ├── backtest/                   # Modular backtest engine
│   │   ├── engine.py              # Main orchestrator
│   │   ├── portfolio_manager.py    # Position & cash management
│   │   ├── cost_calculator.py      # Trading costs with FFR integration
│   │   ├── trade_executor.py       # Signal execution & T+1 logic
│   │   ├── performance_calculator.py # Metrics & analysis
│   │   ├── data_validator.py       # Data validation & preparation
│   │   └── risk_management.py      # Risk management modules
│   ├── data/
│   │   └── loader.py              # Data loading with FFR support
│   ├── indicators/
│   │   ├── calculator.py          # Indicator computation engine
│   │   ├── registry.py            # Decorator-based registration
│   │   ├── resolver.py            # Dependency resolution
│   │   ├── kdj.py                 # KDJ indicator implementations
│   │   └── sma.py                 # SMA indicator implementations
│   ├── strategies/
│   │   ├── base.py                # Abstract strategy base class
│   │   └── kdj_cross.py           # KDJ golden/dead cross strategies
│   ├── utils/
│   │   ├── helpers.py             # Utility functions
│   │   └── chart_utils.py         # Chart utility functions
│   └── visualization/
│       ├── plotter.py             # Main plotting engine
│       └── components/            # Modular chart components
│           ├── allocation.py      # Capital allocation plots
│           ├── chart_elements.py  # Basic chart elements
│           ├── indicators.py      # Technical indicator plots
│           ├── performance.py     # Performance comparison plots
│           ├── styling.py         # Chart styling and themes
│           └── titles.py          # Dynamic title generation
├── kdj_monthly_demo_simplified.py # Complete demo script
└── __init__.py                    # Framework exports
```

## 🛠 Installation

### Prerequisites
```bash
pip install pandas numpy yfinance pandas-ta plotly pandas-datareader
```

### Framework Setup
The framework is designed to be imported directly:

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
# 1. Load data with Federal Funds Rate support
loader = DataLoader()
data = loader.load('SPY', period='2y')

# 2. Create strategy
strategy = GoldenDeadCrossStrategyMonthly()

# 3. Configure backtest engine with realistic costs
engine = BacktestEngine(
    initial_capital=10000,
    commission=0.001,                        # 0.1% commission
    slippage=0.0005,                        # 0.05% slippage
    leverage={'long': 2.0, 'short': 1.5},   # Asymmetric leverage
    enable_short_selling=True,               # Enable short selling
    position_sizing=0.8,                     # Use 80% of capital per trade
    include_dividends=True,                  # Include dividend analysis
    short_borrow_rate=0.02,                 # 2% annual short borrow rate
    use_margin_costs=True                    # Include FFR-based margin costs
)

# 4. Add risk management
risk_manager = DrawdownProtection(threshold=0.15)
engine.add_risk_manager(risk_manager)

# 5. Run backtest with realistic T+1 execution
results = engine.run(strategy, data)

# 6. Create comprehensive visualization
plotter = Plotter(data=results, results=results, engine=engine)
fig = plotter.create_comprehensive_chart(
    ticker='SPY', 
    base_strategy_name='Monthly KDJ',
    log_scale=True
)
fig.show()
```

## 🔧 Core Components

### 1. Modular Backtest Engine

The engine has been completely refactored into specialized modules:

#### BacktestEngine (Main Orchestrator)
```python
engine = BacktestEngine(
    initial_capital=50000,                   # Starting capital
    commission=0.001,                        # Commission rate
    slippage=0.0005,                        # Slippage rate
    leverage={'long': 3.0, 'short': 2.0},   # Asymmetric leverage
    enable_short_selling=True,               # Allow short positions
    position_sizing=0.6,                     # Use 60% of capital per trade
    include_dividends=True,                  # Process dividends
    short_borrow_rate=0.025,                # 2.5% annual borrow rate
    use_margin_costs=True                    # Include FFR-based costs
)
```

#### Portfolio Manager
- **Dynamic Position Sizing**: Adjusts based on current portfolio value
- **Margin Debt Tracking**: Realistic margin calculations with interest
- **Long/Short Management**: Separate leverage ratios for different position types
- **Dividend Processing**: Automatic dividend collection and analysis

#### Cost Calculator
- **Federal Funds Rate Integration**: Real-time FFR data from FRED
- **Margin Interest**: Daily margin costs based on FFR + spread
- **Short Borrowing Costs**: Time-based borrowing fees for short positions
- **Money Market Fund Interest**: Interest on cash balances

#### Trade Executor
- **T+1 Execution**: Uses next-day open prices for realistic execution
- **Signal Processing**: Handles buy/sell signal execution with proper timing
- **Trade Statistics**: Tracks win/loss ratios and trade counts

### 2. Enhanced Data Loader

#### Federal Funds Rate Integration
```python
loader = DataLoader()

# Automatic FFR data loading for margin cost calculations
data = loader.load('AAPL', period='1y')

# Manual FFR data loading
ffr_data = loader.load_fed_funds_rate(start_date='2020-01-01')
```

**Features:**
- **FRED Integration**: Loads FFR data from Federal Reserve Economic Data
- **Auto-Backfill**: Extends FFR data to current date using latest rate
- **Smart Caching**: Updates FFR data only when needed
- **Fallback Data**: Provides realistic historical approximations if FRED is unavailable

### 3. Advanced Indicator System

#### Decorator-Based Registration
```python
from backtest_framework.core.indicators.registry import IndicatorRegistry

@IndicatorRegistry.register(
    name="CUSTOM_RSI",
    inputs=["Close"],
    params={"period": 14},
    outputs=["rsi"]
)
def calculate_custom_rsi(data, period=14):
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    result = pd.DataFrame(index=data.index)
    result['rsi'] = rsi
    return result
```

#### Available Indicators
- **KDJ Indicators**: `KDJ`, `MONTHLY_KDJ` with slopes and derivatives
- **SMA Indicators**: `SMA`, `SMA_SLOPE`, `CLOSE_MINUS_SMA`
- **Automatic Dependencies**: Resolves indicator dependencies automatically

### 4. Professional Visualization System

#### Component-Based Architecture
```python
plotter = Plotter(data=results, results=results, engine=engine)

# Create comprehensive 5-panel chart
fig = plotter.create_comprehensive_chart(
    ticker='AAPL',
    base_strategy_name='Monthly KDJ',
    log_scale=True
)

# Individual components also available
# plotter.performance.add_performance_comparison(fig, row=2, col=1)
# plotter.allocation.add_capital_allocation(fig, row=5, col=1)
```

**Chart Panels:**
1. **Price Panel**: Candlesticks, signals, SMA overlay, dividend markers
2. **Performance Panel**: Strategy vs benchmark with return curves
3. **Drawdown Panel**: Strategy vs benchmark drawdown comparison
4. **Indicators Panel**: KDJ indicators with signal levels
5. **Allocation Panel**: Capital allocation between cash, long, and short positions

### 5. Comprehensive Risk Management

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

## 📈 Advanced Strategies

### KDJ Golden/Dead Cross Strategies

#### Monthly KDJ Strategy
```python
from backtest_framework.core.strategies.kdj_cross import GoldenDeadCrossStrategyMonthly

strategy = GoldenDeadCrossStrategyMonthly()
# Uses monthly-equivalent periods (198/66 days)
# Signals: J line crosses above/below K line
# Filters: K and D slopes must be positive for buy signals
```

#### Daily KDJ Strategy
```python
from backtest_framework.core.strategies.kdj_cross import GoldenDeadCrossStrategyDaily

strategy = GoldenDeadCrossStrategyDaily()
# Uses standard KDJ periods (9/3 days)
# Same signal logic as monthly version
```

## 🎯 Advanced Configuration

### Realistic Cost Modeling
```python
engine = BacktestEngine(
    initial_capital=100000,
    commission=0.001,                        # 0.1% commission
    slippage=0.0005,                        # 0.05% slippage
    leverage={'long': 2.0, 'short': 1.5},   # Asymmetric leverage
    enable_short_selling=True,               # Enable short selling
    position_sizing=0.8,                     # 80% capital per trade
    include_dividends=True,                  # Include dividends
    short_borrow_rate=0.02,                 # 2% annual borrow rate
    use_margin_costs=True                    # FFR-based margin costs
)
```

### Portfolio Backtesting
```python
# Multi-ticker portfolio with rebalancing
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

## 📊 Comprehensive Performance Metrics

The framework automatically calculates:

### Return Metrics
- **Total Return**: Overall portfolio performance including dividends
- **CAGR**: Compound Annual Growth Rate
- **Benchmark Comparison**: Strategy vs Buy & Hold performance
- **Outperformance**: Strategy alpha over benchmark

### Risk Metrics
- **Sharpe Ratio**: Risk-adjusted returns
- **Sortino Ratio**: Downside risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Volatility**: Annualized return volatility

### Trading Metrics
- **Win Rate**: Percentage of profitable trades
- **Trade Count**: Total number of trades executed
- **Average Trade**: Mean trade return
- **Profit Factor**: Ratio of gross profits to gross losses

### Cost Analysis
- **Total Borrowing Costs**: Margin interest and short borrowing fees
- **Commission Impact**: Total commission costs
- **Dividend Income**: Total dividends received
- **MMF Interest**: Interest earned on cash balances

## 🔍 Advanced Features

### 1. Federal Funds Rate Integration
Real-time margin cost calculations using FRED data:
- Daily margin interest based on FFR + spread
- Automatic FFR data updates and caching
- Historical FFR approximation for older periods

### 2. T+1 Execution Simulation
Realistic trading simulation:
- Signals generated on day T
- Execution occurs at day T+1 open price
- Accounts for market gaps and overnight moves

### 3. Component-Based Visualization
Modular chart system:
- Specialized components for different chart elements
- Easy to customize and extend
- Professional dark theme optimized for financial data

### 4. Comprehensive Dividend Handling
- Automatic dividend detection and processing
- Ex-dividend date adjustments
- Dividend yield analysis
- Impact on strategy performance

### 5. Advanced Position Management
- Asymmetric leverage for long/short positions
- Dynamic position sizing based on portfolio value
- Margin debt tracking with realistic interest costs
- Money market fund interest on cash balances

## 🚧 Extending the Framework

### Adding Custom Indicators
1. Use the `@IndicatorRegistry.register` decorator
2. Define inputs, parameters, and outputs
3. Implement calculation function
4. Automatic dependency resolution

### Adding Custom Strategies
1. Inherit from `BaseStrategy`
2. Implement `required_indicators` property
3. Implement `generate_signals` method
4. Return DataFrame with signal columns

### Adding Custom Risk Managers
1. Create class with `apply(data)` method
2. Modify signal columns based on risk rules
3. Add to engine with `engine.add_risk_manager()`

### Adding Custom Visualization Components
1. Create component in `visualization/components/`
2. Follow existing component patterns
3. Add to main `Plotter` class
4. Maintain consistent styling

## 📝 Best Practices

1. **Use Realistic Costs**: Include commission, slippage, and borrowing costs
2. **Enable T+1 Execution**: Use realistic signal-to-execution timing
3. **Include Risk Management**: Always use appropriate position sizing and risk controls
4. **Test Incrementally**: Test strategies on small datasets first
5. **Visualize Results**: Always plot results to verify strategy behavior
6. **Monitor FFR Data**: Ensure Federal Funds Rate data is current for accurate costs
7. **Validate Signals**: Check signal generation and execution logic
8. **Analyze Costs**: Review borrowing costs and their impact on performance

## 🤝 Contributing

When extending the framework:
1. Follow the modular architecture patterns
2. Use the decorator-based registration system
3. Add comprehensive error handling
4. Update documentation for new features
5. Test with multiple securities and time periods
6. Maintain backward compatibility
7. Follow snake_case naming conventions

## 📄 Version History

- **v1.1.0**: Modular architecture with FFR integration, T+1 execution, and component-based visualization
- **v1.0.0**: Initial release with basic backtesting capabilities

## 📄 License

This framework is designed for educational and research purposes. Please ensure compliance with data provider terms of service when using market data.

---

**Happy Backtesting! 📈**

*A sophisticated, production-ready backtesting framework with realistic cost modeling and professional-grade analysis capabilities.*