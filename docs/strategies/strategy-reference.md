# 📚 **Strategy Reference Guide**

## 🎯 **Complete Strategy Library**

The HyperOpt Strategy Platform includes **24 professional trading strategies** across 5 categories, each optimized for different market conditions and trading styles.

---

## 📋 **Strategy Quick Reference**

| # | Strategy Name | Category | Complexity | Best Markets | API Name |
|---|---------------|----------|------------|--------------|-----------|
| 1 | **MovingAverageCrossover** | Trend | ⭐⭐ | Trending | `MovingAverageCrossover` |
| 2 | **MACDStrategy** | Trend | ⭐⭐⭐ | Medium-term trends | `MACDStrategy` |
| 3 | **MomentumStrategy** | Trend | ⭐⭐ | High volatility | `MomentumStrategy` |
| 4 | **MTFTrendAnalysis** | Multi-TF | ⭐⭐⭐⭐ | All conditions | `MTFTrendAnalysisStrategy` |
| 5 | **RSIStrategy** | Mean Reversion | ⭐⭐ | Range-bound | `RSIStrategy` |
| 6 | **BollingerBands** | Mean Reversion | ⭐⭐⭐ | High volatility | `BollingerBandsStrategy` |
| 7 | **StochasticStrategy** | Mean Reversion | ⭐⭐⭐ | Sideways markets | `StochasticStrategy` |
| 8 | **WilliamsR** | Mean Reversion | ⭐⭐⭐ | Short-term | `WilliamsRStrategy` |
| 9 | **MTFRSIStrategy** | Multi-TF | ⭐⭐⭐⭐ | All conditions | `MTFRSIStrategy` |
| 10 | **MTFMACDStrategy** | Multi-TF | ⭐⭐⭐⭐⭐ | Complex analysis | `MTFMACDStrategy` |
| 11 | **VWAPStrategy** | Volume | ⭐⭐⭐ | High-volume | `VWAPStrategy` |
| 12 | **OBVStrategy** | Volume | ⭐⭐⭐ | Trending | `OBVStrategy` |
| 13 | **ADStrategy** | Volume | ⭐⭐⭐ | All types | `ADStrategy` |
| 14 | **CMFStrategy** | Volume | ⭐⭐⭐ | Volatile | `CMFStrategy` |
| 15 | **ATRStrategy** | Volatility | ⭐⭐⭐ | All types | `ATRStrategy` |
| 16 | **BollingerSqueeze** | Volatility | ⭐⭐⭐⭐ | Pre-breakout | `BollingerSqueezeStrategy` |
| 17 | **HistoricalVolatility** | Volatility | ⭐⭐⭐⭐ | Options/derivatives | `HistoricalVolatilityStrategy` |
| 18 | **SupportResistance** | Pattern | ⭐⭐⭐⭐ | All timeframes | `SupportResistanceStrategy` |
| 19 | **PivotPoints** | Pattern | ⭐⭐⭐ | Day trading | `PivotPointsStrategy` |
| 20 | **DoubleTopBottom** | Pattern | ⭐⭐⭐⭐⭐ | Swing trading | `DoubleTopBottomStrategy` |
| 21 | **FibonacciRetracement** | Pattern | ⭐⭐⭐⭐ | Trending | `FibonacciRetracementStrategy` |
| 22 | **ROCStrategy** | Momentum | ⭐⭐⭐ | High volatility | `ROCStrategy` |
| 23 | **UltimateOscillator** | Momentum | ⭐⭐⭐⭐⭐ | Complex analysis | `UltimateOscillatorStrategy` |
| 24 | **KeltnerChannel** | Volatility | ⭐⭐⭐ | Breakout trading | `KeltnerChannelStrategy` |

**Complexity Legend**: ⭐ = Beginner, ⭐⭐⭐ = Intermediate, ⭐⭐⭐⭐⭐ = Advanced

---

## 🔄 **Trend Following Strategies**

### 1. **MovingAverageCrossover** ⭐⭐

**Description**: Classic trend-following strategy using fast and slow moving average crossovers.

**Parameters**:
```json
{
  "fast_period": {"type": "int", "range": [5, 50], "default": 10},
  "slow_period": {"type": "int", "range": [20, 200], "default": 20},
  "ma_type": {"type": "str", "choices": ["SMA", "EMA", "WMA", "TEMA"], "default": "SMA"},
  "signal_threshold": {"type": "float", "range": [0.001, 0.02], "default": 0.005}
}
```

**Usage Example**:
```python
strategy = MovingAverageCrossoverStrategy(
    fast_period=12,
    slow_period=26,
    ma_type="EMA",
    signal_threshold=0.01
)
```

**Best For**: Strong trending markets, medium to long-term trading  
**Avoid In**: Sideways/choppy markets  
**Optimal Timeframes**: 4h, 1d

---

### 2. **MACDStrategy** ⭐⭐⭐

**Description**: MACD (Moving Average Convergence Divergence) with histogram analysis.

**Parameters**:
```json
{
  "fast_period": {"type": "int", "range": [8, 25], "default": 12},
  "slow_period": {"type": "int", "range": [20, 50], "default": 26},
  "signal_period": {"type": "int", "range": [5, 15], "default": 9},
  "signal_threshold": {"type": "float", "range": [0.001, 0.01], "default": 0.005}
}
```

**Usage Example**:
```python
strategy = MACDStrategy(
    fast_period=12,
    slow_period=26,
    signal_period=9,
    signal_threshold=0.005
)
```

**Best For**: Medium-term trend identification, momentum confirmation  
**Optimal Timeframes**: 1h, 4h

---

### 3. **MomentumStrategy** ⭐⭐

**Description**: Rate of Change (ROC) based momentum strategy.

**Parameters**:
```json
{
  "momentum_period": {"type": "int", "range": [10, 50], "default": 20},
  "threshold": {"type": "float", "range": [0.01, 0.1], "default": 0.05},
  "smoothing_period": {"type": "int", "range": [3, 10], "default": 5}
}
```

**Best For**: High volatility markets, breakout trading  
**Optimal Timeframes**: 15m, 1h, 4h

---

### 4. **MTFTrendAnalysis** ⭐⭐⭐⭐

**Description**: Multi-timeframe trend analysis with confirmation signals.

**Parameters**:
```json
{
  "primary_timeframe": {"type": "str", "choices": ["1h", "4h", "1d"], "default": "4h"},
  "secondary_timeframe": {"type": "str", "choices": ["15m", "1h", "4h"], "default": "1h"},
  "trend_strength_threshold": {"type": "float", "range": [0.1, 0.8], "default": 0.3},
  "confirmation_required": {"type": "bool", "default": true}
}
```

**Best For**: High-confidence trend signals, reduced false positives  
**Optimal Use**: All market conditions with proper timeframe selection

---

## ↔️ **Mean Reversion Strategies**

### 5. **RSIStrategy** ⭐⭐

**Description**: RSI (Relative Strength Index) overbought/oversold strategy.

**Parameters**:
```json
{
  "rsi_period": {"type": "int", "range": [10, 30], "default": 14},
  "buy_threshold": {"type": "float", "range": [20, 40], "default": 30},
  "sell_threshold": {"type": "float", "range": [60, 80], "default": 70},
  "exit_strategy": {"type": "str", "choices": ["opposite", "middle", "trailing"], "default": "opposite"}
}
```

**Usage Example**:
```python
strategy = RSIStrategy(
    rsi_period=14,
    buy_threshold=30,
    sell_threshold=70,
    exit_strategy="middle"
)
```

**Best For**: Range-bound markets, oversold/overbought conditions  
**Optimal Timeframes**: 15m, 1h, 4h

---

### 6. **BollingerBands** ⭐⭐⭐

**Description**: Bollinger Bands with both breakout and mean reversion modes.

**Parameters**:
```json
{
  "period": {"type": "int", "range": [15, 30], "default": 20},
  "std_dev": {"type": "float", "range": [1.5, 3.0], "default": 2.0},
  "strategy_type": {"type": "str", "choices": ["breakout", "mean_reversion"], "default": "mean_reversion"},
  "squeeze_detection": {"type": "bool", "default": true}
}
```

**Best For**: High volatility markets, squeeze breakouts  
**Optimal Timeframes**: 1h, 4h

---

### 7. **StochasticStrategy** ⭐⭐⭐

**Description**: Stochastic oscillator with %K and %D line analysis.

**Parameters**:
```json
{
  "k_period": {"type": "int", "range": [10, 20], "default": 14},
  "d_period": {"type": "int", "range": [3, 8], "default": 3},
  "buy_level": {"type": "float", "range": [15, 30], "default": 20},
  "sell_level": {"type": "float", "range": [70, 85], "default": 80}
}
```

**Best For**: Sideways markets, short-term trading  
**Optimal Timeframes**: 15m, 1h

---

### 8. **WilliamsR** ⭐⭐⭐

**Description**: Williams %R momentum oscillator for overbought/oversold conditions.

**Parameters**:
```json
{
  "period": {"type": "int", "range": [10, 25], "default": 14},
  "buy_threshold": {"type": "float", "range": [-90, -70], "default": -80},
  "sell_threshold": {"type": "float", "range": [-30, -10], "default": -20}
}
```

**Best For**: Short-term mean reversion, quick reversals  
**Optimal Timeframes**: 15m, 30m, 1h

---

## ⚡ **Volume-Based Strategies**

### 11. **VWAPStrategy** ⭐⭐⭐

**Description**: Volume Weighted Average Price with volume confirmation.

**Parameters**:
```json
{
  "period": {"type": "int", "range": [20, 100], "default": 50},
  "volume_threshold": {"type": "float", "range": [1.2, 3.0], "default": 1.5},
  "price_deviation": {"type": "float", "range": [0.005, 0.02], "default": 0.01}
}
```

**Best For**: High-volume assets, institutional trading levels  
**Optimal Timeframes**: 1h, 4h, 1d

---

### 12. **OBVStrategy** ⭐⭐⭐

**Description**: On-Balance Volume with trend confirmation.

**Parameters**:
```json
{
  "period": {"type": "int", "range": [15, 40], "default": 20},
  "signal_threshold": {"type": "float", "range": [0.01, 0.05], "default": 0.02},
  "trend_confirmation": {"type": "bool", "default": true}
}
```

**Best For**: Trend confirmation, volume divergence analysis  
**Optimal Timeframes**: 4h, 1d

---

### 13. **ADStrategy** ⭐⭐⭐

**Description**: Accumulation/Distribution Line for buying/selling pressure.

**Parameters**:
```json
{
  "period": {"type": "int", "range": [15, 35], "default": 21},
  "volume_factor": {"type": "float", "range": [0.1, 0.5], "default": 0.2},
  "signal_threshold": {"type": "float", "range": [0.01, 0.03], "default": 0.015}
}
```

**Best For**: All market types, accumulation/distribution phases  
**Optimal Timeframes**: 1h, 4h

---

### 14. **CMFStrategy** ⭐⭐⭐

**Description**: Chaikin Money Flow for volume-weighted momentum.

**Parameters**:
```json
{
  "period": {"type": "int", "range": [15, 25], "default": 21},
  "threshold": {"type": "float", "range": [0.05, 0.2], "default": 0.1},
  "zero_line_cross": {"type": "bool", "default": true}
}
```

**Best For**: Volatile markets, money flow analysis  
**Optimal Timeframes**: 1h, 4h

---

## 📈 **Volatility-Based Strategies**

### 15. **ATRStrategy** ⭐⭐⭐

**Description**: Average True Range for volatility-based position sizing and signals.

**Parameters**:
```json
{
  "atr_period": {"type": "int", "range": [10, 25], "default": 14},
  "multiplier": {"type": "float", "range": [1.5, 3.5], "default": 2.0},
  "volatility_threshold": {"type": "float", "range": [0.01, 0.05], "default": 0.02}
}
```

**Best For**: All market types, adaptive position sizing  
**Optimal Timeframes**: 1h, 4h, 1d

---

### 16. **BollingerSqueeze** ⭐⭐⭐⭐

**Description**: Volatility contraction/expansion detection for breakout trading.

**Parameters**:
```json
{
  "period": {"type": "int", "range": [15, 25], "default": 20},
  "squeeze_threshold": {"type": "float", "range": [0.8, 1.2], "default": 1.0},
  "breakout_confirmation": {"type": "bool", "default": true}
}
```

**Best For**: Pre-breakout phases, volatility cycles  
**Optimal Timeframes**: 4h, 1d

---

### 17. **HistoricalVolatility** ⭐⭐⭐⭐

**Description**: Statistical volatility analysis with percentile ranking.

**Parameters**:
```json
{
  "period": {"type": "int", "range": [20, 60], "default": 30},
  "threshold_percentile": {"type": "float", "range": [70, 95], "default": 80},
  "lookback_period": {"type": "int", "range": [100, 500], "default": 252}
}
```

**Best For**: Options trading, volatility regime identification  
**Optimal Timeframes**: 1d, 1w

---

## 🎯 **Pattern Recognition Strategies**

### 18. **SupportResistance** ⭐⭐⭐⭐

**Description**: Dynamic support and resistance level identification.

**Parameters**:
```json
{
  "lookback_period": {"type": "int", "range": [50, 200], "default": 100},
  "min_touches": {"type": "int", "range": [2, 5], "default": 3},
  "tolerance": {"type": "float", "range": [0.005, 0.02], "default": 0.01},
  "strength_threshold": {"type": "float", "range": [0.1, 0.8], "default": 0.3}
}
```

**Best For**: All timeframes, key level trading  
**Optimal Use**: Confluence with other strategies

---

### 19. **PivotPoints** ⭐⭐⭐

**Description**: Classic pivot point analysis with support/resistance levels.

**Parameters**:
```json
{
  "pivot_type": {"type": "str", "choices": ["classic", "fibonacci", "demark"], "default": "classic"},
  "sensitivity": {"type": "float", "range": [0.5, 2.0], "default": 1.0},
  "time_frame": {"type": "str", "choices": ["daily", "weekly", "monthly"], "default": "daily"}
}
```

**Best For**: Day trading, intraday levels  
**Optimal Timeframes**: 15m, 1h

---

### 20. **DoubleTopBottom** ⭐⭐⭐⭐⭐

**Description**: Advanced double top/bottom pattern recognition.

**Parameters**:
```json
{
  "pattern_period": {"type": "int", "range": [50, 150], "default": 100},
  "tolerance": {"type": "float", "range": [0.01, 0.05], "default": 0.02},
  "volume_confirmation": {"type": "bool", "default": true},
  "min_pattern_height": {"type": "float", "range": [0.02, 0.1], "default": 0.05}
}
```

**Best For**: Swing trading, reversal patterns  
**Optimal Timeframes**: 4h, 1d

---

### 21. **FibonacciRetracement** ⭐⭐⭐⭐

**Description**: Fibonacci retracement levels with swing point analysis.

**Parameters**:
```json
{
  "swing_period": {"type": "int", "range": [20, 60], "default": 40},
  "retracement_levels": {"type": "list", "default": [0.236, 0.382, 0.5, 0.618, 0.786]},
  "tolerance": {"type": "float", "range": [0.005, 0.015], "default": 0.01}
}
```

**Best For**: Trending markets, retracement entries  
**Optimal Timeframes**: 1h, 4h, 1d

---

## 🚀 **Advanced Multi-Timeframe Strategies**

### 9. **MTFRSIStrategy** ⭐⭐⭐⭐

**Description**: Multi-timeframe RSI analysis with trend confirmation.

**Parameters**:
```json
{
  "primary_timeframe": {"type": "str", "choices": ["1h", "4h", "1d"], "default": "4h"},
  "secondary_timeframe": {"type": "str", "choices": ["15m", "1h", "4h"], "default": "1h"},
  "rsi_period": {"type": "int", "range": [10, 20], "default": 14},
  "confirmation_threshold": {"type": "float", "range": [0.6, 0.9], "default": 0.8}
}
```

**Best For**: High-confidence signals, reduced false positives  
**Optimal Use**: All market conditions

---

### 10. **MTFMACDStrategy** ⭐⭐⭐⭐⭐

**Description**: Multi-timeframe MACD with advanced signal filtering.

**Parameters**:
```json
{
  "primary_timeframe": {"type": "str", "choices": ["4h", "1d"], "default": "4h"},
  "secondary_timeframe": {"type": "str", "choices": ["1h", "4h"], "default": "1h"},
  "tertiary_timeframe": {"type": "str", "choices": ["15m", "1h"], "default": "15m"},
  "divergence_detection": {"type": "bool", "default": true}
}
```

**Best For**: Complex trend analysis, professional trading  
**Optimal Use**: High-frequency trading with multiple confirmations

---

## 📊 **Strategy Usage Examples**

### **Basic Strategy Setup**

```python
# Import strategies
from strategies import MovingAverageCrossoverStrategy, RSIStrategy

# Create strategy with custom parameters
ma_strategy = MovingAverageCrossoverStrategy(
    fast_period=12,
    slow_period=26,
    ma_type="EMA",
    position_size_pct=0.95
)

# Initialize with data
ma_strategy.initialize(historical_data)

# Generate signals
for timestamp, row in data.iterrows():
    signal = ma_strategy.generate_signal(timestamp, row)
    if signal.action != "hold":
        print(f"{timestamp}: {signal.action} - Strength: {signal.strength:.2f}")
```

### **API Usage**

```bash
# List available strategies
curl -H "X-API-Key: dev_key_123" \
  http://localhost:8000/api/v1/strategies

# Optimize specific strategy
curl -X POST "http://localhost:8000/api/v1/optimize/single" \
  -H "X-API-Key: dev_key_123" \
  -d '{
    "strategy_name": "MovingAverageCrossover",
    "asset": "BTC",
    "timeframe": "4h",
    "optimization_config": {"max_evals": 100}
  }'
```

### **Multi-Strategy Portfolio**

```python
from strategies import *

# Create strategy portfolio
portfolio = [
    MovingAverageCrossoverStrategy(fast_period=12, slow_period=26),
    RSIStrategy(rsi_period=14, buy_threshold=30),
    MACDStrategy(fast_period=12, slow_period=26),
    VWAPStrategy(period=50, volume_threshold=1.5)
]

# Equal allocation
weights = [0.25, 0.25, 0.25, 0.25]

# Portfolio optimization
portfolio_optimizer = PortfolioOptimizer(portfolio, weights)
optimal_allocation = portfolio_optimizer.optimize(historical_data)
```

---

## 🎯 **Strategy Selection Matrix**

### **By Market Condition**

| Market Type | Primary Strategies | Secondary Strategies | Avoid |
|-------------|-------------------|---------------------|--------|
| **Strong Uptrend** | MA Crossover, MACD, Momentum | Volume-based confirmation | RSI, Stochastic |
| **Strong Downtrend** | Short MA Crossover, MACD | Volume divergence | Mean reversion |
| **Sideways/Range** | RSI, Bollinger Bands, Williams %R | Support/Resistance | Trend following |
| **High Volatility** | ATR, Bollinger Squeeze, Volume | Volatility-based | Fixed-parameter |
| **Low Volatility** | Pattern recognition, Support/Resistance | Squeeze detection | Momentum strategies |
| **Breakout Phase** | Bollinger Squeeze, Volume, ATR | Pattern confirmation | Mean reversion |

### **By Trading Style**

| Style | Timeframes | Recommended Strategies | Key Features |
|-------|------------|----------------------|--------------|
| **Scalping** | 1m, 5m | RSI, Stochastic, Williams %R | Quick reversals |
| **Day Trading** | 15m, 1h | Pivot Points, VWAP, RSI | Intraday levels |
| **Swing Trading** | 4h, 1d | MA Crossover, MACD, Patterns | Multi-day holds |
| **Position Trading** | 1d, 1w | MTF strategies, Trend following | Long-term trends |

### **By Experience Level**

| Level | Start With | Progress To | Master |
|-------|------------|-------------|---------|
| **Beginner** | MA Crossover, RSI | MACD, Bollinger Bands | Volume strategies |
| **Intermediate** | Volume strategies, VWAP | Pattern recognition | MTF strategies |
| **Advanced** | MTF strategies, Complex patterns | Portfolio optimization | Custom strategies |

---

## 🔗 **Additional Resources**

- **[Strategy Development Guide](strategy-development.md)**
- **[Parameter Optimization](../examples/optimization-workflow.md)**
- **[Backtesting Examples](../examples/backtesting-examples.md)**
- **[API Integration](../api/strategies-endpoint.md)**

---

**📈 Ready to start trading? Choose your strategies and begin optimization with our [Quick Start Guide](../getting-started/quick-start.md)!** 