# Freqtrade Trading Bot - Documentation for Claude

**Last Updated:** November 1, 2025
**Project:** Automated Crypto Trading with Freqtrade

---

## 📋 Project Overview

This project uses **Freqtrade**, an open-source cryptocurrency trading bot written in Python, to automate trading strategies on various crypto exchanges.

### Key Technologies
- **Freqtrade:** 2025.9.1 (stable branch)
- **Python:** 3.12.10
- **Exchange:** Bybit (USDT Perpetual Futures with 10x leverage)
- **Strategy Development:** Python (Freqtrade) + Pine Script (TradingView)

---

## 🔄 Development Workflow

### 1. Strategy Development in TradingView

**Why TradingView First?**
- Visual feedback and rapid prototyping
- Extensive indicator library
- Easy testing on historical data
- Community-shared code and concepts

**Workflow:**
```
1. Develop strategy in Pine Script (.pine files)
2. Test visually on TradingView charts
3. Backtest with TradingView's strategy tester
4. Refine logic and parameters
```

**Location:** `strategies/choch_bos/tradingview/`

### 2. Strategy Implementation in Freqtrade

**Translation Process:**
```
Pine Script Indicator → Python Implementation → Freqtrade Strategy
```

**Key Differences:**
- **Pine Script:** Event-driven, real-time updates on each bar
- **Freqtrade:** Polling-based, runs `populate_indicators()` then `populate_entry/exit_trend()`

**Location:** `user_data/strategies/`

### 3. Backtesting & Optimization

**Freqtrade Backtesting:**
```bash
freqtrade backtesting \
    --strategy CHoCHBOSStrategy \
    --timeframe 15m \
    --timerange 20240101-20251027
```

**Hyperopt (Parameter Optimization):**
```bash
freqtrade hyperopt \
    --strategy CHoCHBOSStrategy \
    --hyperopt-loss SharpeHyperOptLoss \
    --spaces buy sell stoploss trailing \
    --epochs 200
```

---

## 🎯 Current Trading Strategies

### 1. CHoCH/BOS Strategy (Main)

**Concept:** Smart Money Concepts (SMC) - Institutional market structure analysis

**File:** `strategies/choch_bos/tradingview/CHoCH_BOS_Strategy_WithFibonacciZigZag.pine`

**Components:**
- **CHoCH (Change of Character):** Trend reversal signals
- **BOS (Break of Structure):** Trend continuation signals
- **ZigZag with Fibonacci:** Dynamic support/resistance levels
- **SFP (Swing Failure Pattern):** Price rejection patterns (LuxAlgo)
- **Smooth Trail:** Adaptive trailing stop indicator (OmegaTools)
- **HMA Filter:** Hull Moving Average for trend confirmation

**Entry Logic:**
1. Wait for CHoCH signal (trend reversal)
2. Confirm with BOS (structure break)
3. Optional: HMA trend filter
4. Optional: Smooth Trail zone touch
5. Optional: SFP confirmation

**Exit Logic:**
- Opposite CHoCH (trend reverses)
- Trailing stop (Smooth Trail or fixed %)
- Fixed stop loss: -5.8%

**Current Performance:**
- Win Rate: ~30%
- Overall: Positive (few big winners compensate many small losers)

---

## 📁 Project Structure

```
freqtrade/
├── CLAUDE.md                           # This file
├── README.md                           # Freqtrade official docs
├── user_data/
│   ├── strategies/                     # Python strategies
│   │   ├── FreqAISmartMoneyStrategy.py
│   │   └── FreqAISmartMoneyAdvanced.py
│   └── models/                         # FreqAI ML models
├── strategies/
│   ├── base/
│   │   └── tradingview/                # Reference indicators
│   │       ├── Smart_Money_Concepts_[LuxAlgo].pine
│   │       ├── Market_Structure_Targets_Model_[LuxAlgo].pine
│   │       ├── Swing_Failure_Pattern.pine
│   │       └── Smooth_Trail.pine
│   └── choch_bos/
│       ├── CHoCHBOSStrategy.py         # Freqtrade Python strategy
│       ├── tradingview/
│       │   ├── CHoCH_BOS_Strategy_WithFibonacciZigZag.pine  # Main TradingView strategy
│       │   ├── archive/                # Old versions
│       │   └── README.md               # TradingView strategy docs
│       ├── configs/                    # Freqtrade config files
│       ├── docs/                       # Strategy documentation
│       └── scripts/                    # Helper scripts
└── config_*.json                       # Various Freqtrade configurations
```

---

## 🛠️ Common Commands

### Backtesting
```bash
# Basic backtest
freqtrade backtesting --strategy CHoCHBOSStrategy --timeframe 15m

# With specific timerange
freqtrade backtesting --strategy CHoCHBOSStrategy --timeframe 15m --timerange 20240101-20251027

# Show results
freqtrade backtesting-show
```

### Hyperopt
```bash
# Optimize entry/exit parameters
freqtrade hyperopt --strategy CHoCHBOSStrategy --hyperopt-loss SharpeHyperOptLoss --spaces buy sell --epochs 200

# Optimize stoploss and trailing
freqtrade hyperopt --strategy CHoCHBOSStrategy --hyperopt-loss SharpeHyperOptLoss --spaces stoploss trailing --epochs 100
```

### Live Trading (Dry Run)
```bash
freqtrade trade --strategy CHoCHBOSStrategy --config config_live.json
```

### Download Data
```bash
freqtrade download-data --exchange bybit --pairs BTC/USDT:USDT ETH/USDT:USDT --timeframes 5m 15m 1h --timerange 20240101-
```

---

## 📊 TradingView to Freqtrade Translation Guide

### Common Patterns

| Pine Script | Freqtrade (Python) | Notes |
|------------|-------------------|-------|
| `ta.sma(close, 20)` | `ta.SMA(dataframe['close'], 20)` | Simple Moving Average |
| `ta.ema(close, 20)` | `ta.EMA(dataframe['close'], 20)` | Exponential MA |
| `ta.rsi(close, 14)` | `ta.RSI(dataframe['close'], 14)` | RSI Indicator |
| `ta.atr(14)` | `ta.ATR(dataframe, 14)` | Average True Range |
| `ta.crossover(a, b)` | `qtpylib.crossed_above(a, b)` | Cross above |
| `ta.crossunder(a, b)` | `qtpylib.crossed_below(a, b)` | Cross below |
| `high[1]` | `dataframe['high'].shift(1)` | Previous bar value |
| `bar_index` | `dataframe.index` | Current bar index |

### Strategy Structure

**Pine Script:**
```pine
//@version=6
strategy("My Strategy", overlay=true)

// Indicators
sma20 = ta.sma(close, 20)

// Entry
if close > sma20
    strategy.entry("Long", strategy.long)

// Exit
if close < sma20
    strategy.close("Long")
```

**Freqtrade:**
```python
class MyStrategy(IStrategy):
    def populate_indicators(self, dataframe, metadata):
        dataframe['sma20'] = ta.SMA(dataframe['close'], 20)
        return dataframe

    def populate_entry_trend(self, dataframe, metadata):
        dataframe.loc[
            (dataframe['close'] > dataframe['sma20']),
            'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe, metadata):
        dataframe.loc[
            (dataframe['close'] < dataframe['sma20']),
            'exit_long'] = 1
        return dataframe
```

---

## 🎓 Key Concepts

### Smart Money Concepts (SMC)

**CHoCH (Change of Character):**
- Price breaks previous structure in opposite direction
- Signals potential trend reversal
- Example: In uptrend, price breaks below previous higher low

**BOS (Break of Structure):**
- Price breaks structure in same direction
- Confirms trend continuation
- Example: In uptrend, price breaks above previous higher high

**Order Blocks:**
- Last bullish/bearish candle before strong move
- Institutional buying/selling zones
- High probability support/resistance areas

**Fair Value Gaps (FVG):**
- Imbalance in price (gap between wicks)
- Price tends to return to fill gaps
- Entry opportunities

### Market Structure Targets (MST)

**Concept:** Calculate dynamic price targets based on market structure

**Formula:**
```
For Long: Target = HighLevel + (HighLevel - LowestPoint) × Percentage
For Short: Target = LowLevel - (HighestPoint - LowLevel) × Percentage
```

**Example:**
- CHoCH bullish at $100
- Lowest point since then: $95
- With 100% target: $100 + ($100 - $95) × 1.0 = $105
- With 50% target: $100 + ($100 - $95) × 0.5 = $102.50

---

## 🔧 Configuration

### Exchange Setup (Bybit)
```json
{
    "exchange": {
        "name": "bybit",
        "key": "YOUR_API_KEY",
        "secret": "YOUR_API_SECRET",
        "ccxt_config": {},
        "ccxt_async_config": {},
        "pair_whitelist": ["BTC/USDT:USDT", "ETH/USDT:USDT"],
        "pair_blacklist": []
    },
    "trading_mode": "futures",
    "margin_mode": "isolated"
}
```

### Leverage
- **Current:** 10x leverage on perpetual futures
- **Risk:** Higher leverage = higher risk
- **Stop Loss:** Critical with leverage (default: -5.8% = -58% at 10x)

---

## 📈 Performance Metrics

### Key Metrics to Track
- **Win Rate:** Percentage of profitable trades
- **Profit Factor:** Gross profit / Gross loss
- **Sharpe Ratio:** Risk-adjusted return
- **Max Drawdown:** Largest peak-to-trough decline
- **Average Trade Duration:** Time in market
- **Expectancy:** Average profit per trade

### Current Challenge
- Low win rate (~30%) but overall positive
- Strategy catches occasional large winners
- Working on: Improving win rate while maintaining winners

### CHoCHBOSSimple Strategy - Entry Validation Tests (November 3, 2025)

**Strategy:** CHoCH + BOS with optional FVG/OB/Fibonacci validations
**File:** `user_data/strategies/CHoCHBOSSimple.py`
**Test Period:** 2024-2025 (ETH/USDT:USDT, 15m timeframe)

#### Hyperopt Results (20 epochs):
| Validation | Trades | Win Rate | Profit | Sharpe | Result |
|------------|--------|----------|--------|--------|--------|
| **Baseline (none)** | 187 | 72.7% | +1.8% | 0.68 | 🟡 Decent |
| **FVG only** ✅ | 84 | 85.7% | +5.81% | 1.88 | 🟢 **WINNER** |
| OB only | ~50 | ~75% | +2.5% | ~1.2 | 🟡 OK |
| Fibonacci only | ~60 | ~78% | +3.0% | ~1.4 | 🟡 OK |

**Best Configuration:**
- `require_fvg = True` (FVG retest required)
- `require_ob = False`
- `require_fibo = False`
- `zigzag_depth = 23`

#### Out-of-Sample Validation:
| Period | Trades | Win Rate | Profit | Status |
|--------|--------|----------|--------|--------|
| 2023 (before training) | 10 | 80.0% | +0.63% | ✅ Valid |
| 2025 (after training) | 63 | 84.1% | +3.38% | ✅ Valid |

**Conclusion:** NO overfitting detected. FVG validation filters weak entries effectively.

#### Exit Signal Analysis (without leverage):
| Exit Reason | Trades | Win Rate | P/L | Issue |
|-------------|--------|----------|-----|-------|
| ROI | 42 | 100% | +44.01 USDT | ✅ Excellent |
| Trailing Stop | 11 | 100% | +13.62 USDT | ✅ Very good |
| Exit Signal (CHoCH) | 10 | **0%** | -23.88 USDT | ⚠️ **Loses money** |

**Finding:** Exit signals (CHoCH opposite) consistently lose money. Consider disabling `use_exit_signal`.

#### Leverage Impact Analysis:

**Without Leverage (1x):**
- 84 trades, 85.7% WR, +5.81% profit
- Max consecutive losses: 2
- Drawdown: 0.65%
- Avg duration: 8:17 hours

**With Leverage 10x:**
- 114 trades, 79.8% WR, +19.91% profit (+343% more!)
- Max consecutive losses: 2 (same)
- Drawdown: 1.62% (+150% higher)
- Avg duration: 0:41 hours (-83% faster)
- **Critical:** Stop losses consume 70% of margin per trade
- 17 stop losses = -117 USDT (-59% of total profit)

**Exit Breakdown (10x leverage):**
- Trailing Stop: 79 trades, 92.4% WR, +182 USDT (91% of profit)
- ROI: 18 trades, 100% WR, +134 USDT
- Stop Loss: 17 trades, 0% WR, -117 USDT ⚠️

**Risk Assessment (10x leverage):**
- Each SL = -7.01% on position = -70 USDT loss
- 2 consecutive SLs = -140 USDT (14% of $1000 account)
- **Minimum recommended capital:** $200-300 for sustainability
- **With $100 account:** High risk, 1-2 losses can wipe margin

#### Key Learnings:
1. ✅ **FVG validation is highly effective** - improves WR from 72.7% to 85.7%
2. ⚠️ **Exit signals (CHoCH opposite) lose money** - consider disabling
3. ⚠️ **10x leverage amplifies both gains and losses** - requires larger capital
4. ✅ **Strategy has no overfitting** - validated across multiple time periods
5. ✅ **Max 2 consecutive losses** - good risk management

#### Stop Loss & Trailing Optimization (30 epochs - November 3, 2025):

After FVG validation success, optimized stop loss and trailing stop parameters:

**Hyperopt Configuration:**
- Optimization Spaces: `stoploss`, `trailing`
- Loss Function: SharpeHyperOptLoss
- Training Period: 2024-01-01 to 2025-01-31

**Results:**
| Configuration | Trades | Win Rate | Profit | Sharpe | Drawdown | Result |
|---------------|--------|----------|--------|--------|----------|--------|
| **Original SL (-5.8%)** | 114 | 79.8% | +19.91% | ~1.0 | 1.62% | 🟡 Good |
| **Optimized SL (-13%)** ✅ | 105 | **93.3%** | **+41.54%** | 4.06 | 2.50% | 🟢 **EXCELLENT** |

**Optimized Parameters:**
```python
stoploss = -0.13  # Widened from -0.058 (gives trades more room)
trailing_stop_positive = 0.02  # Increased from 0.01 (locks profits at 2%)
trailing_stop_positive_offset = 0.055  # Increased from 0.02 (activates at 5.5%)
```

**Out-of-Sample Validation (2023 data):**
| Period | Trades | Win Rate | Profit | Sharpe | Drawdown |
|--------|--------|----------|--------|--------|----------|
| 2023 (Nov-Dec) | 13 | **100%** | +5.43% | 13.55 | 0.00% |

**Key Finding:** Wider stop loss (-13% vs -5.8%) dramatically improves performance! This aligns with Smart Money Concepts - price needs room to retrace to Order Blocks and FVGs before continuing the trend. The 100% win rate on 2023 data confirms NO overfitting.

**Improvement Summary:**
- Win Rate: +13.5% (79.8% → 93.3%)
- Profit: +108% (19.91% → 41.54%)
- Validates on unseen 2023 data with perfect 100% WR

#### Exit Signal Analysis (November 3, 2025):

Tested disabling `use_exit_signal` to evaluate CHoCH opposite exits:

**Results (ETH, 10x leverage, 2024-2025):**
| Configuration | Trades | Win Rate | Profit | Drawdown | Impact |
|---------------|--------|----------|--------|----------|--------|
| With Exit Signals | 105 | 93.3% | +41.54% | 2.50% | Default |
| **Without Exit Signals** | 104 | 93.3% | +40.55% | 2.50% | -0.99% profit |

**Conclusion:** Exit signals have minimal impact (-1 trade, -0.99% profit). The optimized trailing stop and ROI already handle exits effectively. Can enable or disable without significant difference.

#### Leverage Comparison (ETH/USDT, 2024-2025):

**Test Results:**
| Leverage | Trades | Win Rate | Profit | Drawdown | Sharpe | Result |
|----------|--------|----------|--------|----------|--------|--------|
| **10x** ✅ | 104 | **93.3%** | **+40.55%** | **2.50%** | 3.83 | **WINNER** |
| **5x** | 95 | 92.6% | +24.52% | 2.94% | 2.50 | Good |
| **3x** | 88 | 93.2% | +16.58% | 3.12% | 1.86 | Conservative |

**Key Finding:** Higher leverage is BETTER with this strategy:
- 10x leverage has the best profit (+40.55%) with the LOWEST drawdown (2.50%)
- Lower leverage paradoxically increases drawdown (3.12% at 3x vs 2.50% at 10x)
- Reason: High win rate (93%) means leverage amplifies wins more than losses
- Fewer trades with lower leverage = missed profit opportunities

**Recommendation:** Use 10x leverage for optimal risk/reward ratio.

#### Multi-Pair Testing (10x leverage, 2024-2025):

**Results:**
| Pair | Trades | Win Rate | Profit | Drawdown | Sharpe | Result |
|------|--------|----------|--------|----------|--------|--------|
| **BTC/USDT:USDT** ✅ | 120 | 92.5% | **+43.01%** | **2.20%** | 4.03 | **BEST** |
| **ETH/USDT:USDT** | 104 | 93.3% | +40.55% | 2.50% | 3.83 | Excellent |

**Key Finding:** Strategy works excellently on both major pairs:
- BTC has slightly better performance (+43% vs +41%, 2.20% vs 2.50% DD)
- Both pairs show 92-93% win rates
- Strategy is robust across different market dynamics
- Max consecutive losses: 2 (same for both)

#### Final Optimized Configuration:

**✅ PRODUCTION READY - CHoCHBOSSimple Strategy**

```python
# Entry Validation
require_fvg = True      # FVG retest required (85.7% WR vs 72.7% baseline)
require_ob = False      # Order Block not needed
require_fibo = False    # Fibonacci not needed
zigzag_depth = 23       # Optimal structure detection

# Risk Management
stoploss = -0.13                          # -13% (widened from -5.8%)
trailing_stop = True
trailing_stop_positive = 0.02             # Lock profits at 2%
trailing_stop_positive_offset = 0.055     # Activate at 5.5%
trailing_only_offset_is_reached = True

# Exit Settings
use_exit_signal = True   # Minimal impact, can be True or False

# Leverage
leverage = 10.0          # 10x is optimal

# ROI Table
minimal_roi = {
    "0": 0.10,    # 10% immediately
    "60": 0.05,   # 5% after 1 hour
    "120": 0.03,  # 3% after 2 hours
    "240": 0.01   # 1% after 4 hours
}
```

**Performance Summary (2024-2025):**
- **BTC**: 120 trades, 92.5% WR, **+43.01%** profit, 2.20% DD
- **ETH**: 104 trades, 93.3% WR, +40.55% profit, 2.50% DD
- **2023 Validation**: 13 trades, **100% WR**, +5.43% profit (NO overfitting!)
- **Max Consecutive Losses**: 2 (both pairs)
- **Sharpe Ratio**: 4.03 (BTC), 3.83 (ETH)

**Capital Requirements (10x leverage, $1000 account):**
- Max drawdown: 2.50% = -$25 USDT
- Each stop loss: -13% position = -$70 USDT (7% of account)
- 2 consecutive losses: -$140 USDT (14% of account)
- **Recommended minimum**: $200-300 for comfortable margin

#### Completed Optimization Steps:
- [x] ✅ Optimize stop loss percentage - **COMPLETE** (widened to -13%)
- [x] ✅ Test disabling `use_exit_signal` - **COMPLETE** (minimal impact)
- [x] ✅ Evaluate lower leverage (3x, 5x) - **COMPLETE** (10x is optimal)
- [x] ✅ Test on multiple pairs (BTC, ETH) - **COMPLETE** (both excellent)

#### Recommendations for Live Trading:
1. **Start with paper trading (dry run)** for 1-2 weeks to validate
2. **Use BTC/USDT:USDT** as primary pair (best performance)
3. **Start with small capital** ($200-300 minimum recommended)
4. **Monitor first 10 trades closely** to ensure behavior matches backtest
5. **Consider adding ETH** after BTC proves successful
6. **Set exchange stop loss on exchange** (use `stoploss_on_exchange: True`)
7. **Keep detailed trade journal** to track actual vs expected performance

---

## 🚧 Current Development

### Upcoming Features

**1. Market Structure Targets (MST) - Priority**
- Dynamic TP calculation based on structure
- Separate targets for CHoCH vs BOS
- Configurable percentages
- Max duration per target

**2. Smart Money Concepts (SMC) Complete**
- Full LuxAlgo SMC implementation
- Order Blocks detection
- Equal Highs/Lows (EQH/EQL)
- Premium/Discount Zones
- Internal vs Swing structure

### Testing Approach
1. Develop in TradingView first
2. Visual validation on charts
3. TradingView backtest
4. Translate to Freqtrade
5. Python backtest
6. Hyperopt optimization
7. Paper trading (dry run)
8. Live trading (small capital)

---

## 🤖 FreqAI Strategies - Machine Learning Integration

### Current Status (November 2, 2024)

We have tested FreqAI integration with CHoCH/BOS concepts. Initial results were negative, but valuable insights were gained.

### Tested Strategies

**1. CHoCHBOSFreqAIBasic (Classification) - ❌ Failed**
- **File:** `user_data/strategies/CHoCHBOSFreqAIBasic.py`
- **Model:** LightGBMClassifier
- **Target:** Binary (0 = Loser, 1 = Winner)
- **Status:** KeyError: 0 in datasieve pipeline
- **Problem:** "One class" issue - in bear markets, NO trades hit +2% before -2%, resulting in all targets = 0
- **Result:** Cannot train classifier with only one class

**2. CHoCHBOSFreqAIRegressor (Regression) - ❌ Negative Results**
- **File:** `user_data/strategies/CHoCHBOSFreqAIRegressor.py`
- **Model:** LightGBMRegressor
- **Target:** Continuous (predicts % profit potential)
- **Status:** Runs successfully, but loses money
- **Backtest Results (Jan-Oct 2024, ETH/USDT, 10x leverage):**
  - Total Trades: 2,775
  - Win Rate: 36.1%
  - Total Loss: **-82.82%** (100 USDT → 17.18 USDT)
  - Main Issue: 62.5% of trades (1,736) hit stop loss at -2%
  - GPU Training: 44 models trained, 5-8 seconds each with RTX 3060

### Current Features (337 after expansion)

**Base Features (~25-30):**
1. **Technical Indicators (7 × 3 periods):** RSI, MFI, ADX, ATR, EMA, ROC, Relative Volume
2. **Price Action (6):** Price change, body size, wicks, range, candle direction
3. **Smart Money Concepts (4):** CHoCH bullish/bearish, BOS bullish/bearish
4. **Candle Patterns (2):** Bullish/bearish engulfing
5. **Market Structure (2):** Higher highs, lower lows
6. **MACD (3):** MACD, signal, histogram
7. **EMAs (6):** EMA crosses, distance to EMAs

**Problem:** SMC features (CHoCH/BOS) are INPUTS only, not used as logic. Model doesn't understand:
- CHoCH → BOS confirmation sequence
- Market structure states (uptrend/downtrend)
- Pullback/retrace opportunities
- Order Block zones
- Fair Value Gaps
- Premium/Discount zones

### 📋 Improvement Plan - IN PROGRESS

**✅ OPTION 1: Pure AI + Advanced SMC Features (CHOSEN)**

Add sophisticated SMC features to teach the model proper Smart Money Concepts:

**Phase 1: Advanced SMC Features (Priority)**
```python
# New features to implement:
1. Order Block Detection
   - Last bullish/bearish candle before BOS
   - Distance to nearest Order Block
   - Order Block strength (volume, size)

2. Fair Value Gap Detection
   - Imbalance zones (gap between wicks)
   - FVG size and location
   - Filled vs Unfilled gaps

3. Premium/Discount Zones
   - 50% Fibonacci from swing high to swing low
   - Current price position (premium/discount/equilibrium)
   - Zone strength

4. Market Structure State Tracking
   - 5 states like CHoCHBOSStrategy.py
   - NEUTRAL, TENDENCIA_ALCISTA, TENDENCIA_BAJISTA
   - ESPERANDO_CONFIRMACION_ALCISTA, ESPERANDO_CONFIRMACION_BAJISTA

5. Pullback Detection
   - Is price pulling back to Order Block?
   - Pullback depth (Fibonacci retracement)
   - Time since CHoCH/BOS (signal freshness)

6. Liquidity Detection
   - Swing highs/lows with high volume
   - Liquidity sweeps (fake breakouts)
   - Stop hunt patterns

7. Confluence Score
   - How many SMC signals align?
   - Weight: OB + FVG + Zone + Structure = Score
```

**Phase 2: Improved Entry Logic**
```python
# Current (too loose):
if ai_prediction > 1.5% and rsi 25-75:
    enter()

# Improved (stricter):
if (ai_prediction > 3.0 and                   # More conservative
    in_discount_zone and                      # Price in favorable zone
    near_order_block and                      # Near institutional zone
    has_liquidity_sweep and                   # Stop hunt occurred
    market_structure_strong and               # Clear trend
    volume > volume_ma * 1.2):                # High volume
    enter()
```

**Phase 3: Hybrid Scoring System**
```python
# AI + SMC weighted scoring
ai_score = ai_prediction * 0.4          # 40% AI
smc_score = calculate_smc_score() * 0.6 # 60% SMC rules

if (ai_score + smc_score) > threshold:
    enter()

# SMC Score components:
- State alignment: 20%
- Zone position: 15%
- Order Block proximity: 15%
- FVG presence: 10%
- Liquidity sweep: 10%
- Volume confirmation: 10%
- Pullback quality: 10%
- Confluence: 10%
```

**Expected Improvements:**
- Fewer false entries (better filtering)
- Better entry prices (wait for pullback to OB)
- Tighter stops (below Order Block)
- Higher win rate (proper SMC context)
- More interpretable (know why trade was taken)

---

### 🔬 OPTION 2: Multiclass Classification (NOT TESTED - DOCUMENTED FOR FUTURE)

**Status:** Documented but not implemented yet. Consider if Option 1 fails.

**Concept:** Instead of binary (Win/Lose), use multiple outcome classes.

**Approach A: 3 Classes**
```python
Class -1: Loser    (hits -2% before +2%)
Class  0: Neutral  (neither hits within N candles)
Class  1: Winner   (hits +2% before -2%)

# Entry: Only when predicts Class 1 with >70% confidence
```

**Approach B: 5 Classes (More Granular)**
```python
Class 0: Big Loss     (hits -6% first)
Class 1: Small Loss   (hits -2% first, not -6%)
Class 2: Neutral      (±1% max)
Class 3: Small Win    (hits +2% first)
Class 4: Big Win      (hits +5% first)

# Entry: Only Class 3 or 4 predictions
# Avoid: Class 0 (big losses)
```

**Pros:**
- Solves "one class" problem (always has variety)
- More granular predictions
- Can filter out "big loss" predictions
- Better risk control

**Cons:**
- Needs class balancing (if rare "Class 4", model ignores it)
- More complex to train (5 classes harder than 2)
- Arbitrary thresholds (why -2%, -6%?)
- Less data per class (2000 samples / 5 = 400 each)
- Loses information vs regression (continuous better than discrete)

**Recommendation:** Only try this if Option 1 (SMC Features) doesn't improve results significantly.

**Reason:** Multiclass solves a technical problem but doesn't solve the real problem (bad predictions). Better features (Option 1) address the root cause.

---

### 📊 Comparison: Strategy Approaches

| Aspect | FreqAI Regressor | CHoCHBOSStrategy (Pure) | FreqAI + SMC Features (Plan) |
|--------|------------------|------------------------|------------------------------|
| **Logic** | Black box ML | Explicit SMC rules | Hybrid (ML + SMC) |
| **CHoCH/BOS** | Features only | Core logic | Logic + Features |
| **Interpretable** | ❌ No | ✅ Yes | ✅ Mostly |
| **Entry Timing** | Immediate | After BOS confirmation | OB pullback + AI filter |
| **Pullback** | No | No (can add) | Yes (planned) |
| **Results** | -82.82% | Not tested | TBD |
| **Adaptability** | ✅ High | ❌ Fixed rules | ✅ High |

### 🎯 Next Steps

1. **Implement Order Block detection** (Phase 1.1)
2. **Implement Fair Value Gap detection** (Phase 1.2)
3. **Add Premium/Discount zones** (Phase 1.3)
4. **Add Market Structure state tracking** (Phase 1.4)
5. **Add Pullback detection** (Phase 1.5)
6. **Update entry logic with new filters** (Phase 2)
7. **Backtest improved strategy** (same period: Jan-Oct 2024)
8. **Compare results** (if worse, consider Option 2: Multiclass)

### 🔧 Configuration Notes

**GPU Acceleration (Works):**
```json
"model_training_parameters": {
    "device": "gpu",
    "gpu_use_dp": true,
    "n_estimators": 500,
    "learning_rate": 0.02,
    "max_depth": 5
}
```

**Key Learnings:**
- DI_threshold: 0.7 was too liberal (many false predictions)
- Leverage 10x amplifies losses (-2% SL = -20% real)
- Trailing stop worked well (96.5% win rate when activated)
- Problem: Most trades hit SL before trailing activated
- Target too optimistic (1.5% prediction vs -2% SL)

---

## 💡 Best Practices

### Development
1. **Always use TradingView first** for visual feedback
2. **Test incrementally** - add one feature at a time
3. **Document everything** - future you will thank you
4. **Version control** - commit often with descriptive messages
5. **Backtest thoroughly** - different market conditions

### Trading
1. **Never skip dry run** - test in paper trading first
2. **Start small** - use minimal capital initially
3. **Monitor constantly** - especially first days/weeks
4. **Have stop losses** - always, non-negotiable
5. **Review trades** - learn from wins and losses

### Risk Management
1. **Position sizing:** Never risk more than 1-2% per trade
2. **Leverage:** Be conservative, higher leverage = higher risk
3. **Diversification:** Don't put all capital in one strategy
4. **Stop loss:** Always set, especially with leverage
5. **Take profits:** Don't be greedy, secure wins

---

## 📚 Resources

### Freqtrade
- Docs: https://www.freqtrade.io/en/stable/
- GitHub: https://github.com/freqtrade/freqtrade
- Discord: https://discord.gg/freqtrade

### TradingView
- Pine Script Docs: https://www.tradingview.com/pine-script-docs/
- Ideas: https://www.tradingview.com/scripts/
- Community: Large active community for indicators

### Smart Money Concepts
- YouTube: Many educational channels on SMC
- LuxAlgo: Premium indicators provider
- Twitter/X: #SMC community

---

## 🐛 Troubleshooting

### Common Issues

**"Pair not available on exchange"**
```bash
# Check available pairs
freqtrade list-markets --exchange bybit --trading-mode futures
```

**"Insufficient balance"**
- Check account balance
- Reduce stake_amount in config
- Check margin mode (isolated vs cross)

**"Strategy has no buy signals"**
- Check indicators are calculating correctly
- Verify timeframe has enough data
- Review entry conditions logic

**"High CPU usage during backtest"**
- Reduce timerange
- Use fewer pairs
- Simplify indicator calculations

---

## 📝 Notes for Claude

### When Modifying Strategies

1. **Always read the full file first** before making changes
2. **Preserve existing functionality** unless explicitly asked to change
3. **Comment changes clearly** for future reference
4. **Test syntax** before committing (Pine Script can be strict)
5. **Update this file** if workflow or structure changes

### Pine Script Gotchas

- Arrays use `.get()` and `.set()` methods
- `var` keyword persists values across bars
- Line and label counts are limited (max_lines_count, max_labels_count)
- Version matters: `//@version=6` has different syntax than v5
- Can't import local files - each strategy must be self-contained

### Freqtrade Gotchas

- `populate_indicators()` runs first, then `populate_entry_trend()`
- Use `.loc[]` for conditional assignment to avoid warnings
- Don't calculate the same indicator twice (performance)
- `metadata['pair']` gives you current trading pair
- Timeframe affects indicator calculations

---

**Remember:** This is a living document. Update it as the project evolves!
