# KDJ MFI Early Detection Strategy Migration Roadmap (Revised)

## Overview
This roadmap outlines the steps to migrate the legacy `generate_buy_sell_signals_early_detection_3_stages_MFI` strategy into the new backtesting framework, fully leveraging its modular architecture.

## Key Framework Principles
1. **Strategies only generate entry signals** - Exit logic is handled by risk management
2. **Helper functions become indicators** - Consecutive days tracking, flip detection, etc. are factors/indicators
3. **Risk management is separate** - Drawdown protection, stop losses are handled by risk managers
4. **Everything is modular** - Indicators can be chained and reused

## Current Legacy Strategy Analysis

### Core Signal Logic
The strategy generates three types of positions (Buy, Hold, Sell) based on:
- Monthly KDJ golden/death crosses
- Consecutive days of J movement
- ADX trend confirmation
- Buy signal count limiting within golden cross periods

### Signal Generation Conditions
- **Buy Signal**: During golden cross period with J increasing for N days
- **Hold Signal**: When conditions transition (legacy logic to be simplified)
- **Sell Signal**: At death cross occurrence

## Migration Architecture

### Phase 1: Factor/Indicator Implementation (Priority: High)

#### 1.1 Core Technical Indicators
Create missing standard indicators:

**`backtest_framework/core/indicators/adx.py`**:
```python
@IndicatorRegistry.register(
    name="ADX",
    inputs=["High", "Low", "Close"],
    params={"window": 308, "sma_length": 5},
    outputs=["adx", "adx_sma"]
)
```

**`backtest_framework/core/indicators/mfi.py`**:
```python
@IndicatorRegistry.register(
    name="MFI",
    inputs=["High", "Low", "Close", "Volume"],
    params={"length": 48},
    outputs=["mfi"]
)
```

**`backtest_framework/core/indicators/rsi.py`**:
```python
@IndicatorRegistry.register(
    name="RSI",
    inputs=["Close"],
    params={"period": 14},
    outputs=["rsi"]
)
```

#### 1.2 Derived Factor Indicators
Create factor indicators that capture the signal generation logic:

**`backtest_framework/core/indicators/kdj_factors.py`**:
```python
@IndicatorRegistry.register(
    name="KDJ_CONSECUTIVE_DAYS",
    inputs=["monthly_j"],
    params={"up_days": 7, "down_days": 4},
    outputs=["j_consecutive_up_days", "j_consecutive_down_days", "j_up_flip", "j_down_flip"]
)
def calculate_kdj_consecutive_days(data: pd.DataFrame, up_days: int = 7, down_days: int = 4):
    """Calculate consecutive days and flip points for J line."""
    # Tracks consecutive up/down days
    # Identifies flip points (negative to positive, positive to negative)
    pass

@IndicatorRegistry.register(
    name="GOLDEN_DEATH_CROSS",
    inputs=["monthly_j", "monthly_k"],
    outputs=["golden_cross", "death_cross", "cross_status", "bars_since_golden_cross"]
)
def calculate_golden_death_cross(data: pd.DataFrame):
    """Identify golden/death crosses and track cross status."""
    # Returns: golden_cross (0/1), death_cross (0/1), 
    # cross_status (1=golden period, 0=death period)
    # bars_since_golden_cross (count)
    pass

@IndicatorRegistry.register(
    name="BUY_SIGNAL_COUNTER",
    inputs=["golden_cross", "bars_since_golden_cross"],
    params={"max_signals_per_cross": 2},
    outputs=["buy_signals_since_cross", "can_generate_buy"]
)
def calculate_buy_signal_counter(data: pd.DataFrame, max_signals_per_cross: int = 2):
    """Track buy signals per golden cross period."""
    pass
```

**`backtest_framework/core/indicators/adx_factors.py`**:
```python
@IndicatorRegistry.register(
    name="ADX_CONSECUTIVE_DOWN",
    inputs=["adx"],
    params={"required_days": 5},
    outputs=["adx_consecutive_down_days", "adx_down_condition_met"]
)
def calculate_adx_consecutive_down(data: pd.DataFrame, required_days: int = 5):
    """Track consecutive ADX down days."""
    pass
```

### Phase 2: Strategy Implementation (Priority: High)

#### 2.1 Simplified Strategy Class
Create `backtest_framework/core/strategies/kdj_mfi_early_detection.py`:

```python
class KDJMFIEarlyDetectionStrategy(BaseStrategy):
    """
    KDJ MFI Early Detection strategy that generates buy/sell entry signals only.
    Exit logic is handled by risk management modules.
    """
    
    def __init__(self, 
                 required_up_days: int = 7,
                 required_down_days: int = 4,
                 required_adx_down_days: int = 5,
                 max_buy_signals_per_cross: int = 2):
        self.required_up_days = required_up_days
        self.required_down_days = required_down_days
        self.required_adx_down_days = required_adx_down_days
        self.max_buy_signals_per_cross = max_buy_signals_per_cross
    
    @property
    def required_indicators(self) -> List[str]:
        """All indicators needed, including derived factors."""
        return [
            # Core indicators
            "MONTHLY_KDJ",
            "MONTHLY_KDJ_SLOPES", 
            "ADX",
            "MFI",
            "RSI",
            
            # Derived factors
            "KDJ_CONSECUTIVE_DAYS",
            "GOLDEN_DEATH_CROSS",
            "BUY_SIGNAL_COUNTER",
            "ADX_CONSECUTIVE_DOWN"
        ]
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate buy and sell entry signals only.
        No exit signals - handled by risk management.
        """
        # Initialize signal columns
        data['buy_signal'] = 0
        data['sell_signal'] = 0
        
        # Buy signal: Golden cross period + J up days + D slope > 0 + can generate buy
        buy_condition = (
            (data['cross_status'] == 1) &  # In golden cross period
            (data['j_consecutive_up_days'] >= self.required_up_days) &
            (data['monthly_d_slope'] > 0) &
            (data['can_generate_buy'] == 1)
        )
        
        # Sell signal: Death cross occurrence
        sell_condition = (data['death_cross'] == 1)
        
        # Alternative buy during death cross (if implemented)
        # death_cross_buy = (
        #     (data['cross_status'] == 0) &
        #     (data['j_up_flip'] == 1) &
        #     (data['adx_down_condition_met'] == 1)
        # )
        
        data.loc[buy_condition, 'buy_signal'] = 1
        data.loc[sell_condition, 'sell_signal'] = 1
        
        return data
```

### Phase 3: Risk Management Configuration (Priority: High)

#### 3.1 Use Existing Risk Managers
The framework already provides risk management. Configure for this strategy:

```python
# In backtest configuration
risk_manager = CompositeRiskManager([
    DrawdownRiskManager(max_drawdown=0.2),  # 20% drawdown protection
    TrailingStopRiskManager(trailing_percent=0.1),  # 10% trailing stop
    # Add more as needed
])
```

#### 3.2 Custom Risk Manager (Optional)
Only if specific behavior is needed:
```python
class GoldenCrossAwareRiskManager(BaseRiskManager):
    """Risk manager that considers golden/death cross status."""
    def check_exit_conditions(self, position, current_data):
        # Custom exit logic if needed
        pass
```

### Phase 4: Validation Strategy (Priority: High)

#### 4.1 Create Validation Script
Create `kdj_mfi_early_detection_validation.py` in the main backtest_framework directory:

```python
"""
Validation script to compare new implementation with legacy results.
"""
from backtest_framework.core.data.loader import DataLoader
from backtest_framework.core.strategies.kdj_mfi_early_detection import KDJMFIEarlyDetectionStrategy
from backtest_framework.core.backtest.engine import BacktestEngine
# Import legacy function for comparison
from trading_processor import generate_buy_sell_signals_early_detection_3_stages_MFI

def validate_strategy():
    # Load test data
    data = DataLoader.load_ticker("SPY", period="5y")
    
    # Run legacy strategy
    legacy_signals = generate_buy_sell_signals_early_detection_3_stages_MFI(data.copy())
    
    # Run new strategy
    strategy = KDJMFIEarlyDetectionStrategy()
    new_signals = strategy.run(data.copy())
    
    # Compare signals
    # ... comparison logic ...
```

#### 4.2 Create Example Backtest
Create `examples/kdj_mfi_early_detection_backtest.py`:

```python
# Load data
data = DataLoader.load_ticker("SPY", period="5y")

# Initialize strategy
strategy = KDJMFIEarlyDetectionStrategy(
    required_up_days=7,
    required_down_days=4
)

# Configure risk management
risk_manager = CompositeRiskManager([
    DrawdownRiskManager(max_drawdown=0.2),
    TrailingStopRiskManager(trailing_percent=0.1)
])

# Run backtest
engine = BacktestEngine(
    strategy=strategy,
    risk_manager=risk_manager,
    initial_capital=100000
)

results = engine.run(data)
```

## Implementation Order

### Week 1: Indicators as Building Blocks
1. Implement core technical indicators (ADX, MFI, RSI)
2. Implement KDJ factor indicators (consecutive days, flip detection)
3. Implement cross detection indicators
4. Validate each indicator output independently

### Week 2: Strategy Assembly
1. Create strategy class using indicator outputs
2. Implement clean signal generation logic
3. Configure risk management
4. Initial validation

### Week 3: Validation & Optimization
1. Compare outputs with legacy strategy
2. Fine-tune parameters
3. Performance optimization
4. Edge case handling

## Key Architectural Benefits

### 1. Modular Indicators
- `j_consecutive_up_days` is now a reusable indicator
- `golden_cross` detection can be used by other strategies
- Each factor is testable independently

### 2. Clean Separation
- Strategy only decides WHEN to enter positions
- Risk management decides WHEN to exit
- No mixed concerns in strategy code

### 3. Reusability
- All indicators are registered and available globally
- Other strategies can use golden cross detection
- Factor indicators can be combined differently

## Migration Checklist

### Indicators
- [ ] Implement ADX indicator
- [ ] Implement MFI indicator
- [ ] Implement RSI indicator
- [ ] Implement KDJ_CONSECUTIVE_DAYS indicator
- [ ] Implement GOLDEN_DEATH_CROSS indicator
- [ ] Implement BUY_SIGNAL_COUNTER indicator
- [ ] Implement ADX_CONSECUTIVE_DOWN indicator

### Strategy
- [ ] Create KDJMFIEarlyDetectionStrategy class
- [ ] Implement generate_signals() with entry logic only
- [ ] Remove all exit signal logic (moved to risk management)
- [ ] Validate signal generation

### Risk Management
- [ ] Configure DrawdownRiskManager
- [ ] Configure appropriate stop loss managers
- [ ] Test risk management integration

### Validation
- [ ] Create validation script to compare with legacy
- [ ] Create example backtest script
- [ ] Validate signal timing matches legacy
- [ ] Performance benchmarking

## Code Organization

```
backtest_framework/
├── core/
│   ├── indicators/
│   │   ├── adx.py          # ADX calculation
│   │   ├── mfi.py          # MFI calculation
│   │   ├── rsi.py          # RSI calculation
│   │   ├── kdj_factors.py  # KDJ-based derived factors
│   │   └── adx_factors.py  # ADX-based derived factors
│   └── strategies/
│       └── kdj_mfi_early_detection.py  # Main strategy
├── examples/
│   └── kdj_mfi_early_detection_backtest.py  # Example usage
└── kdj_mfi_early_detection_validation.py     # Validation script
```

## Success Criteria

1. **Signal Accuracy**: Buy/sell signals match legacy logic
2. **Clean Architecture**: Clear separation between signal generation and risk management
3. **Reusability**: All factors are available as independent indicators
4. **Performance**: Efficient computation with indicator caching
5. **Maintainability**: Easy to modify conditions and parameters

## Next Steps

1. Start with implementing factor indicators (they're the building blocks)
2. Validate each indicator output in isolation
3. Assemble the strategy using indicator outputs
4. Configure risk management to match legacy exit behavior
5. Run validation script to ensure correctness
