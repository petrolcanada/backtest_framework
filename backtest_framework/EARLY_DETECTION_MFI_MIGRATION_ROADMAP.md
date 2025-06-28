# Early Detection 3-Stages MFI Strategy Migration Roadmap

## Overview

This document provides a comprehensive roadmap for migrating the legacy `generate_buy_sell_signals_early_detection_3_stages_MFI` strategy from `trading_processor.py` into the modular backtest framework.

## Legacy Strategy Analysis

### Source Function
- **File**: `scripts/trading_processor.py`
- **Function**: `generate_buy_sell_signals_early_detection_3_stages_MFI`
- **Line**: ~1980+ (approximate)

### Strategy Characteristics
- **Type**: 3-stage trading system (BUY → HOLD → SELL)
- **Primary Indicators**: Monthly KDJ, ADX, MFI
- **Signal Logic**: Golden/Death cross detection with early entry/exit
- **Risk Management**: Drawdown protection, position state tracking
- **Exit Conditions**: Multiple exit triggers (cross-based, drawdown-based)

### Key Parameters
```python
n_days = 10                    # Lookback window for pattern detection
drawdown_threshold = 1.0       # Maximum allowed drawdown (100%)
required_down_days = 4         # Required consecutive down days
required_up_days = 7           # Required consecutive up days
required_ADX_down_days = 5     # Required ADX down days
```

## Phase 1: Strategy Analysis & Structure Planning

### 1.1 Core Strategy Components

#### Signal Generation Logic
- **Buy Conditions**: 
  - During golden cross: `days_J_goes_up()` + Monthly_D increasing + buy signal count ≤ 1
  - During death cross: `days_J_goes_up_since_flip()` + `days_ADX_goes_down()`
- **Sell Conditions**: Death cross transitions
- **Hold Conditions**: Intermediate state management

#### State Management
```python
# Key state variables to track:
latest_buy_index = None
latest_hold_index = None  
latest_sell_index = None
is_in_golden_cross = False
last_golden_cross_index = None
number_buy_signal_since_golden_cross = None
```

#### Helper Functions to Extract
1. `days_ADX_goes_down(i)` - Check ADX consecutive decline
2. `days_J_goes_up(i)` - Check J-line consecutive increase
3. `days_J_goes_up_since_flip(i)` - Check J-line increase since flip point
4. `days_J_goes_down_since_flip(i)` - Check J-line decrease since flip point
5. `can_generate_buy_signal(i)` - Validate buy signal generation rules

### 1.2 Framework Integration Points

#### Required Indicators
- **MONTHLY_KDJ**: Monthly K, D, J values and slopes
- **ADX**: Average Directional Index with slope calculation
- **MFI**: Money Flow Index (currently calculated but not used in core logic)

#### Signal Columns to Generate
```python
# Primary signals
'BUY_SIGNAL' -> 'buy_signal'
'SELL_SIGNAL' -> 'sell_signal'  
'HOLD_SIGNAL' -> 'hold_signal'

# Exit signals
'BUY_EXIT_SIGNAL' -> 'buy_exit_signal'
'HOLD_EXIT_SIGNAL' -> 'hold_exit_signal'
'SELL_EXIT_SIGNAL' -> 'sell_exit_signal'

# Debug/state columns
'Golden_Cross' -> 'golden_cross'
'Death_Cross' -> 'death_cross'
'Cross_Status' -> 'cross_status'
'Buy_Signal_Count' -> 'buy_signal_count'

# Performance tracking
'DrawDown_Since_Latest_Buy' -> 'drawdown_since_latest_buy'
'DrawDown_Life_Time' -> 'drawdown_life_time'
```

## Phase 2: Framework Integration

### 2.1 Create New Strategy Class

**File**: `backtest_framework/core/strategies/early_detection_mfi.py`

```python
"""
Early Detection 3-Stages MFI Strategy implementation.
"""
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from backtest_framework.core.strategies.base import BaseStrategy

class EarlyDetection3StagesMFIStrategy(BaseStrategy):
    """
    Early detection 3-stages strategy with MFI integration.
    
    Implements a sophisticated 3-stage trading system:
    - BUY: Early detection based on J-line patterns
    - HOLD: Intermediate state during transitions
    - SELL: Exit on death cross or drawdown protection
    
    Features:
    - Golden/Death cross state tracking
    - Multiple pattern detection algorithms
    - Drawdown protection
    - Signal count limits per golden cross period
    """
    
    def __init__(self, 
                 n_days: int = 10,
                 drawdown_threshold: float = 1.0,
                 required_down_days: int = 4,
                 required_up_days: int = 7,
                 required_ADX_down_days: int = 5):
        """
        Initialize the Early Detection 3-Stages MFI strategy.
        
        Args:
            n_days: Lookback window for pattern detection
            drawdown_threshold: Maximum allowed drawdown before exit
            required_down_days: Required consecutive down days for patterns
            required_up_days: Required consecutive up days for patterns
            required_ADX_down_days: Required ADX consecutive down days
        """
        self.n_days = n_days
        self.drawdown_threshold = drawdown_threshold
        self.required_down_days = required_down_days
        self.required_up_days = required_up_days
        self.required_ADX_down_days = required_ADX_down_days
    
    @property
    def required_indicators(self) -> List[str]:
        """
        List indicators required by this strategy.
        
        Returns:
            List of required indicator names
        """
        return ["MONTHLY_KDJ", "MONTHLY_KDJ_SLOPES", "ADX", "ADX_SLOPE", "MFI"]
    
    @property
    def name(self) -> str:
        """Strategy name."""
        return "EarlyDetection3StagesMFI"
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on the 3-stages logic.
        
        Args:
            data: DataFrame with OHLCV data and computed indicators
            
        Returns:
            DataFrame with added signal columns
        """
        # Ensure all required indicators are computed
        data = self.prepare_data(data)
        
        # Initialize signal columns
        data = self._initialize_signal_columns(data)
        
        # Run main strategy logic
        return self._run_3stage_logic(data)
    
    def _initialize_signal_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Initialize all signal and state tracking columns."""
        # Primary signals
        data['buy_signal'] = 0
        data['sell_signal'] = 0
        data['hold_signal'] = 0
        
        # Exit signals
        data['buy_exit_signal'] = 0
        data['hold_exit_signal'] = 0
        data['sell_exit_signal'] = 0
        
        # Price tracking
        data['buy_price'] = np.nan
        data['hold_price'] = np.nan
        data['sell_price'] = np.nan
        data['buy_exit_price'] = np.nan
        data['hold_exit_price'] = np.nan
        data['sell_exit_price'] = np.nan
        
        # State tracking
        data['golden_cross'] = 0
        data['death_cross'] = 0
        data['cross_status'] = 0
        data['buy_signal_count'] = 0
        
        # Performance tracking
        data['drawdown_since_latest_buy'] = 0.0
        data['drawdown_life_time'] = 0.0
        data['buy_stop_loss_price'] = np.nan
        
        return data
    
    def _run_3stage_logic(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Main implementation of the 3-stage strategy logic.
        
        This is the core method that implements the strategy extracted
        from the legacy function.
        """
        # Initialize state variables
        latest_buy_index = None
        latest_hold_index = None
        latest_sell_index = None
        peak_price_since_buy = None
        peak_price_lifetime = data['Close'].iloc[0]
        
        # Cross tracking state
        last_golden_cross_index = None
        number_buy_signal_since_golden_cross = None
        is_in_golden_cross = False
        
        # Initialize cross status based on initial J/K relationship
        if len(data) > 0 and data['monthly_j'].iloc[0] > data['monthly_k'].iloc[0]:
            is_in_golden_cross = True
            data.at[data.index[0], 'cross_status'] = 1
        
        # Main strategy loop
        for i in range(self.n_days, len(data)):
            current_date = data.index[i]
            current_price = data['Close'].iloc[i]
            
            # Update peak prices and drawdowns
            self._update_drawdowns(data, i, current_price, 
                                 peak_price_lifetime, peak_price_since_buy, 
                                 latest_buy_index)
            
            # Check for golden/death cross transitions
            is_in_golden_cross, last_golden_cross_index, number_buy_signal_since_golden_cross = \
                self._check_cross_transitions(data, i, is_in_golden_cross, 
                                            last_golden_cross_index, number_buy_signal_since_golden_cross)
            
            # Process signal generation based on current state
            latest_buy_index, latest_hold_index, latest_sell_index, peak_price_since_buy = \
                self._process_signal_generation(data, i, current_price,
                                              is_in_golden_cross, last_golden_cross_index,
                                              number_buy_signal_since_golden_cross,
                                              latest_buy_index, latest_hold_index, latest_sell_index)
            
            # Check exit conditions
            latest_buy_index = self._process_exit_conditions(data, i, current_price,
                                                           peak_price_since_buy, latest_buy_index)
            
            # Update peak price tracking
            peak_price_lifetime = max(peak_price_lifetime, current_price)
            if latest_buy_index is not None:
                peak_price_since_buy = max(peak_price_since_buy or current_price, current_price)
        
        return data
    
    # Additional helper methods would be implemented here
    # _update_drawdowns, _check_cross_transitions, etc.
```

### 2.2 Add Required Indicators

#### ADX Indicator
**File**: `backtest_framework/core/indicators/adx.py`

```python
"""
ADX (Average Directional Index) indicator calculation.
"""
import pandas as pd
import pandas_ta as ta

def calculate_adx(data: pd.DataFrame, window: int = 14*22, sma_length: int = 5) -> pd.DataFrame:
    """
    Calculate ADX with slope and flip detection.
    
    Args:
        data: DataFrame with High, Low, Close columns
        window: ADX calculation window (default: 14*22 for monthly equivalent)
        sma_length: SMA smoothing length for ADX
        
    Returns:
        DataFrame with ADX, ADX_Slope, and ADX_Flip columns
    """
    result = data.copy()
    
    # Calculate ADX using pandas_ta
    adx_data = ta.adx(data['High'], data['Low'], data['Close'], length=window)
    adx_column_name = f'ADX_{window}'
    
    if adx_column_name in adx_data.columns:
        result['ADX'] = adx_data[adx_column_name]
    else:
        # Fallback if column name format changes
        result['ADX'] = adx_data.iloc[:, 0]  # First column should be ADX
    
    # Calculate ADX SMA
    result['ADX_SMA'] = ta.sma(result['ADX'], length=sma_length)
    
    # Calculate ADX slope
    result['ADX_Slope'] = ta.percent_return(result['ADX'])
    
    # Initialize ADX_Flip with zeros
    result['ADX_Flip'] = 0
    
    # Calculate ADX_Flip (trend change detection)
    for i in range(1, len(result)):
        if (result['ADX_Slope'].iloc[i-1] < 0 and 
            result['ADX_Slope'].iloc[i] > 0):
            result.at[result.index[i], 'ADX_Flip'] = 1
        elif (result['ADX_Slope'].iloc[i-1] > 0 and 
              result['ADX_Slope'].iloc[i] < 0):
            result.at[result.index[i], 'ADX_Flip'] = -1
    
    return result[['ADX', 'ADX_SMA', 'ADX_Slope', 'ADX_Flip']]

def calculate_adx_slope(data: pd.DataFrame) -> pd.Series:
    """
    Calculate ADX slope as a separate indicator.
    
    Args:
        data: DataFrame with ADX column
        
    Returns:
        Series with ADX slope values
    """
    if 'ADX' not in data.columns:
        raise ValueError("ADX column not found in data")
    
    return ta.percent_return(data['ADX'])
```

#### MFI Indicator
**File**: `backtest_framework/core/indicators/mfi.py`

```python
"""
MFI (Money Flow Index) indicator calculation.
"""
import pandas as pd
import pandas_ta as ta

def calculate_mfi(data: pd.DataFrame, length: int = 48) -> pd.Series:
    """
    Calculate Money Flow Index using pandas_ta.
    
    Args:
        data: DataFrame with High, Low, Close, Volume columns
        length: lookback period for MFI calculation
        
    Returns:
        Series with MFI values
    """
    return ta.mfi(data['High'], data['Low'], data['Close'], data['Volume'], length=length)
```

#### Update Indicator Registry
**File**: `backtest_framework/core/indicators/registry.py`

Add the following registrations:

```python
# Register ADX indicators
IndicatorRegistry.register(
    name="ADX",
    function=calculate_adx,
    inputs=["High", "Low", "Close"],
    outputs=["ADX", "ADX_SMA", "ADX_Slope", "ADX_Flip"],
    params={"window": 14*22, "sma_length": 5}
)

IndicatorRegistry.register(
    name="ADX_SLOPE", 
    function=calculate_adx_slope,
    inputs=["ADX"],
    outputs=["ADX_Slope"],
    params={}
)

IndicatorRegistry.register(
    name="MFI",
    function=calculate_mfi,
    inputs=["High", "Low", "Close", "Volume"],
    outputs=["MFI"],
    params={"length": 48}
)
```

### 2.3 Risk Management Integration

**File**: `backtest_framework/core/backtest/early_detection_risk_manager.py`

```python
"""
Risk management specific to Early Detection MFI strategy.
"""
import pandas as pd
import numpy as np

class EarlyDetectionRiskManager:
    """
    Risk manager specifically designed for the Early Detection 3-Stages strategy.
    
    Handles:
    - Drawdown protection
    - Position sizing limits
    - Signal validation
    """
    
    def __init__(self, 
                 drawdown_threshold: float = 1.0,
                 enable_profit_protection: bool = True):
        """
        Initialize risk manager.
        
        Args:
            drawdown_threshold: Maximum drawdown before forced exit
            enable_profit_protection: Whether to enable profit protection logic
        """
        self.drawdown_threshold = drawdown_threshold
        self.enable_profit_protection = enable_profit_protection
    
    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply risk management rules to strategy signals.
        
        Args:
            data: DataFrame with strategy signals
            
        Returns:
            DataFrame with risk-adjusted signals
        """
        # This would implement additional risk checks
        # beyond what's already in the strategy
        return data
```

## Phase 3: Implementation Details

### 3.1 Helper Functions Template

Extract and implement these key helper functions:

```python
def days_ADX_goes_down(self, data: pd.DataFrame, i: int) -> bool:
    """
    Check if ADX has been consistently decreasing for required number of days.
    
    Args:
        data: DataFrame with ADX data
        i: Current index
        
    Returns:
        True if ADX has been going down for required_ADX_down_days
    """
    if i < self.required_ADX_down_days:
        return False
        
    # Count consecutive down days
    consecutive_down_days = 0
    for j in range(i, i - self.required_ADX_down_days - 1, -1):
        if j > 0 and data['ADX'].iloc[j] < data['ADX'].iloc[j-1]:
            consecutive_down_days += 1
        else:
            break
            
    return consecutive_down_days >= self.required_ADX_down_days

def days_J_goes_up(self, data: pd.DataFrame, i: int) -> bool:
    """
    Check if Monthly_J has been consistently increasing for required number of days.
    
    Args:
        data: DataFrame with Monthly_J data
        i: Current index
        
    Returns:
        True if J has been going up for required_up_days
    """
    if i < self.required_up_days:
        return False
        
    # Count consecutive up days
    consecutive_up_days = 0
    for j in range(i, i - self.required_up_days - 1, -1):
        if j > 0 and data['monthly_j'].iloc[j] > data['monthly_j'].iloc[j-1]:
            consecutive_up_days += 1
        else:
            break
            
    return consecutive_up_days >= self.required_up_days

def can_generate_buy_signal(self, is_in_golden_cross: bool, 
                          last_golden_cross_index: int, i: int,
                          number_buy_signal_since_golden_cross: int) -> bool:
    """
    Check if we can generate a buy signal based on golden cross conditions.
    
    Args:
        is_in_golden_cross: Whether currently in golden cross period
        last_golden_cross_index: Index of last golden cross
        i: Current index
        number_buy_signal_since_golden_cross: Count of buy signals since last golden cross
        
    Returns:
        True if buy signal can be generated
    """
    return (is_in_golden_cross and
            last_golden_cross_index is not None and 
            i > last_golden_cross_index and 
            number_buy_signal_since_golden_cross <= 1)
```

### 3.2 Demo Script Template

**File**: `backtest_framework/early_detection_mfi_demo.py`

```python
"""
Demo script for the Early Detection 3-Stages MFI strategy.
"""
import os
import sys

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
scripts_dir = os.path.dirname(current_dir)
sys.path.append(scripts_dir)

from backtest_framework.core.data.loader import DataLoader
from backtest_framework.core.strategies.early_detection_mfi import EarlyDetection3StagesMFIStrategy
from backtest_framework.core.backtest.engine import BacktestEngine
from backtest_framework.core.backtest.early_detection_risk_manager import EarlyDetectionRiskManager
from backtest_framework.core.visualization.plotter import Plotter
from backtest_framework.core.utils.helpers import Timer, suppress_warnings

# Import indicator modules to ensure they're registered
from backtest_framework.core.indicators import kdj, sma, adx, mfi

suppress_warnings()

def main():
    """Run Early Detection 3-Stages MFI strategy demo."""
    print("Early Detection 3-Stages MFI Strategy Demo")
    print("==========================================")
    
    timer = Timer()
    
    # Configuration
    ticker = "SPY"
    initial_capital = 10000
    commission = 0.001
    
    # Strategy parameters
    strategy_params = {
        'n_days': 10,
        'drawdown_threshold': 1.0,
        'required_down_days': 4,
        'required_up_days': 7,
        'required_ADX_down_days': 5
    }
    
    try:
        # 1. Load data
        print(f"\nLoading data for {ticker}...")
        data_dir = os.path.join(os.path.expanduser("~"), "local_script", 
                               "Local Technical Indicator Data", "security_data")
        loader = DataLoader(data_dir=data_dir)
        data = loader.load(ticker, period="5y", resample_period="D")
        print(f"Loaded {len(data)} rows of data")
        
        # 2. Initialize strategy
        print("\nInitializing Early Detection 3-Stages MFI strategy...")
        strategy = EarlyDetection3StagesMFIStrategy(**strategy_params)
        
        # 3. Setup backtest engine
        print("Setting up backtest engine...")
        engine = BacktestEngine(
            initial_capital=initial_capital,
            commission=commission,
            leverage={"long": 1.0, "short": 1.0},
            position_sizing=1.0,
            enable_short_selling=True
        )
        
        # Add risk manager
        risk_manager = EarlyDetectionRiskManager(
            drawdown_threshold=strategy_params['drawdown_threshold']
        )
        engine.add_risk_manager(risk_manager)
        
        # 4. Run backtest
        print("Running backtest...")
        results = engine.run(strategy, data)
        
        # 5. Display results
        print(f"\n{ticker} Early Detection 3-Stages MFI Results:")
        print("=" * 60)
        print(f"Final Equity: ${results['equity'].iloc[-1]:,.2f}")
        print(f"Total Return: {results['returns'].iloc[-1] * 100:.2f}%")
        print(f"CAGR: {results['cagr'].iloc[-1] * 100:.2f}%")
        print(f"Sharpe Ratio: {results['sharpe_ratio'].iloc[-1]:.2f}")
        print(f"Max Drawdown: {results['max_drawdown'].iloc[-1] * 100:.2f}%")
        print(f"Total Trades: {int(results['trade_count'].iloc[-1])}")
        
        # Signal analysis
        buy_signals = (results['buy_signal'] == 1).sum()
        sell_signals = (results['sell_signal'] == 1).sum()
        hold_signals = (results['hold_signal'] == 1).sum()
        
        print(f"\nSignal Analysis:")
        print(f"Buy Signals: {buy_signals}")
        print(f"Hold Signals: {hold_signals}")
        print(f"Sell Signals: {sell_signals}")
        
        # 6. Create visualization
        print("\nGenerating visualization...")
        output_dir = os.path.join(os.path.dirname(__file__), "output")
        os.makedirs(output_dir, exist_ok=True)
        
        plotter = Plotter(data=data, results=results, engine=engine)
        fig = plotter.create_comprehensive_chart(
            ticker=ticker,
            base_strategy_name="Early Detection 3-Stages MFI",
            log_scale=False
        )
        
        output_file = os.path.join(output_dir, f"{ticker}_early_detection_mfi.html")
        plotter.save(output_file)
        print(f"Chart saved to: {output_file}")
        
        plotter.open_in_browser(output_file)
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print(f"\nCompleted in {timer.elapsed_str()}")

if __name__ == "__main__":
    main()
```

## Phase 4: Testing & Validation

### 4.1 Unit Testing Strategy

Create test file: `tests/test_early_detection_mfi.py`

```python
"""
Unit tests for Early Detection 3-Stages MFI strategy.
"""
import unittest
import pandas as pd
import numpy as np
from backtest_framework.core.strategies.early_detection_mfi import EarlyDetection3StagesMFIStrategy

class TestEarlyDetectionMFI(unittest.TestCase):
    
    def setUp(self):
        """Set up test data."""
        # Create sample data for testing
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        self.test_data = pd.DataFrame({
            'Open': np.random.randn(100).cumsum() + 100,
            'High': np.random.randn(100).cumsum() + 102,
            'Low': np.random.randn(100).cumsum() + 98,
            'Close': np.random.randn(100).cumsum() + 100,
            'Volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        
        self.strategy = EarlyDetection3StagesMFIStrategy()
    
    def test_required_indicators(self):
        """Test that strategy returns correct required indicators."""
        expected = ["MONTHLY_KDJ", "MONTHLY_KDJ_SLOPES", "ADX", "ADX_SLOPE", "MFI"]
        self.assertEqual(self.strategy.required_indicators, expected)
    
    def test_signal_generation(self):
        """Test signal generation produces valid output."""
        # This would require mocking the indicators or using real data
        pass
    
    def test_helper_functions(self):
        """Test individual helper functions."""
        # Test days_ADX_goes_down, days_J_goes_up, etc.
        pass

if __name__ == '__main__':
    unittest.main()
```

### 4.2 Validation Checklist

- [ ] **Indicator Calculations**: Verify ADX, MFI calculations match legacy
- [ ] **Signal Generation**: Compare signal timing between legacy and new implementation
- [ ] **State Management**: Validate golden cross tracking and buy signal counting
- [ ] **Performance Metrics**: Ensure identical backtest results on same data
- [ ] **Edge Cases**: Test with insufficient data, missing indicators, etc.

## Phase 5: Advanced Features

### 5.1 Parameter Optimization

Once basic implementation is complete, add parameter optimization capabilities:

```python
# Parameter optimization example
from backtest_framework.core.optimization import ParameterOptimizer

optimizer = ParameterOptimizer(
    strategy_class=EarlyDetection3StagesMFIStrategy,
    parameter_ranges={
        'n_days': range(5, 21),
        'required_up_days': range(3, 11),
        'drawdown_threshold': [0.5, 0.8, 1.0, 1.2, 1.5]
    }
)

results = optimizer.optimize(data, metric='sharpe_ratio')
```

### 5.2 Multi-Asset Portfolio

Extend to portfolio-level backtesting:

```python
# Portfolio backtesting
tickers = ['SPY', 'QQQ', 'IWM', 'TLT']
portfolio_data = {ticker: loader.load(ticker) for ticker in tickers}

portfolio_results = engine.run(
    strategy, 
    portfolio_data, 
    portfolio_rebalance='equal_weight'
)
```

## Phase 6: Migration Benefits

After successful migration, you'll gain:

### 6.1 Modular Architecture
- **Separation of Concerns**: Strategy logic separate from risk management
- **Reusable Components**: Indicators can be used across multiple strategies
- **Easy Testing**: Individual components can be tested in isolation

### 6.2 Enhanced Capabilities
- **Professional Visualization**: Interactive charts with multiple panels
- **Comprehensive Analytics**: Detailed performance metrics and statistics
- **Risk Management**: Pluggable risk management modules
- **Portfolio Support**: Multi-asset backtesting capabilities

### 6.3 Development Efficiency
- **Parameter Optimization**: Built-in optimization framework
- **Easy Iteration**: Quick strategy modifications and testing
- **Documentation**: Self-documenting code with clear interfaces
- **Extensibility**: Easy to add new features and indicators

## Implementation Timeline

| Phase | Estimated Time | Key Deliverables |
|-------|---------------|------------------|
| 1-2 | 2-3 days | Strategy class structure, indicator implementations |
| 3 | 2-3 days | Core strategy logic migration, helper functions |
| 4 | 1-2 days | Testing and validation |
| 5 | 1-2 days | Demo script, documentation |
| 6 | 1 day | Final integration and optimization |

**Total Estimated Time**: 7-11 days

## Next Steps

1. **Start with Phase 2.2**: Implement ADX and MFI indicators first
2. **Create Strategy Shell**: Build the basic strategy class structure
3. **Migrate Core Logic**: Extract and adapt the main strategy loop
4. **Test Incrementally**: Validate each component as you build it
5. **Create Demo**: Build working demo to validate end-to-end functionality

## Notes

- Keep legacy function intact for comparison during development
- Use git branches to track migration progress
- Consider creating intermediate validation scripts to compare outputs
- Document any deviations from legacy behavior and reasons

---

*This roadmap serves as a comprehensive guide for migrating the Early Detection 3-Stages MFI strategy into the modular backtest framework. Follow the phases sequentially for best results.*