# Freqtrade Trading Bot - Documentation for Claude

**Last Updated:** November 9, 2025
**Project:** Automated Crypto Trading with Freqtrade

---

## 📋 Project Overview

This project uses **Freqtrade**, an open-source cryptocurrency trading bot written in Python, to automate trading strategies on various crypto exchanges.

### Key Technologies
- **Freqtrade:** 2025.9.1 (stable branch)
- **Python:** 3.12.10
- **Exchange:** Binance (USDT Perpetual Futures with leverage)
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
    --strategy MyStrategy \
    --timeframe 15m \
    --timerange 20240101-20241231
```

**Hyperopt (Parameter Optimization):**
```bash
freqtrade hyperopt \
    --strategy MyStrategy \
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

---

## 📁 Project Structure

```
freqtrade/
├── CLAUDE.md                           # This file - Main project documentation
├── README.md                           # Freqtrade official docs
├── user_data/
│   ├── strategies/                     # Python strategies
│   │   ├── CHoCHBOSStrategy.py        # Main SMC strategy
│   │   └── [other strategies].py
│   └── models/                         # FreqAI ML models (if used)
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
freqtrade backtesting --strategy CHoCHBOSStrategy --timeframe 15m --timerange 20240101-20241231

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
freqtrade download-data --exchange binance --pairs BTC/USDT:USDT ETH/USDT:USDT --timeframes 5m 15m 1h --timerange 20240101-
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

### Exchange Setup (Binance)
```json
{
    "exchange": {
        "name": "binance",
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
- **Current:** Configurable leverage on perpetual futures
- **Risk:** Higher leverage = higher risk
- **Stop Loss:** Critical with leverage (default: -5.8%)

---

## ⚠️ IMPORTANT: Trading Mode Policy

**FUTURES TRADING ONLY**
- **Default trading mode:** Perpetual futures (config: `"trading_mode": "futures"`)
- **NO SPOT TRADING:** Explicitamente prohibido a menos que se indique lo contrario en la documentación
- **Available data:** BTC/USDT:USDT y ETH/USDT:USDT desde 2020
- **Leverage:** 10x default (isolated margin)
- **Error Handling:** Si FreqAI falla en futures, arreglar la configuración, no cambiar a spot

**Current Strategy Versions:**
- **v2:** `SuperTrendAIStrategy_HyperOpt.py` - Many trades (-37.68%)
- **v3 FINAL:** `SuperTrendAIStrategy_v3_FINAL.py` - **POSITIVE +0.18%**
- **FreqAI:** `SuperTrendAIStrategy_FreqAI.py` - ML Enhanced (en desarrollo)

---

## 📈 Performance Metrics

### Key Metrics to Track
- **Win Rate:** Percentage of profitable trades
- **Profit Factor:** Gross profit / Gross loss
- **Sharpe Ratio:** Risk-adjusted return
- **Max Drawdown:** Largest peak-to-trough decline
- **Average Trade Duration:** Time in market
- **Expectancy:** Average profit per trade

---

## 💡 Best Practices

### Development
1. **Always use TradingView first** for visual feedback
2. **Test incrementally** - add one feature at a time
3. **Document everything** - future you will thank you
4. **Version control** - commit often with descriptive messages
5. **Backtest thoroughly** - different market conditions
6. **ALWAYS prevent overfitting** - see anti-overfitting section below

### ⚠️ ANTI-OVERFITTING MANDATORY

**Critical Warning:**
- **Default leverage: 10x futures (isolated margin)**
- **Stop loss of -5% = -50% real loss at 10x leverage**
- **Over-optimized strategies can wipe account quickly**
- **Isolated margin: Risk limited to position margin only**

**Anti-Overfitting Measures Required:**
1. **Conservative parameters** - Avoid extreme optimization
2. **Walk-forward validation** - Test on out-of-sample data
3. **Multiple market conditions** - Bull, bear, sideways markets
4. **Reasonable trade frequency** - 5+ trades per week minimum
5. **Risk/reward balance** - 1:1.5 minimum ratio
6. **Avoid curve fitting** - Don't overfit to historical data
7. **Robust fallback logic** - Strategy must work without AI
8. **Simplified indicators** - Complex indicators increase overfitting risk

### Trading
1. **Never skip dry run** - test in paper trading first
2. **Start small** - use minimal capital initially
3. **Monitor constantly** - especially first days/weeks
4. **Have stop losses** - always, non-negotiable
5. **Review trades** - learn from wins and losses

### Risk Management (CRITICAL WITH 10x LEVERAGE)
1. **Position sizing:** Never risk more than 1-2% per trade
2. **Leverage warning:** Default 10x leverage amplifies all gains/losses by 10x
3. **Stop loss impact:** -5% stop = -50% real loss at 10x leverage
4. **Diversification:** Don't put all capital in one strategy
5. **Always use stop loss:** Non-negotiable with leverage
6. **Take profits:** Secure wins, don't be greedy with leveraged positions
7. **Account protection:** Stop trading if >20% account loss in a day

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
freqtrade list-markets --exchange binance --trading-mode futures
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

## 🚧 Current Development

### Current Status
- **Main Strategy**: CHoCH/BOS SMC implementation working
- **Testing**: Ongoing optimization and parameter tuning
- **Next Steps**: Market Structure Targets implementation

### Development Pipeline
1. ✅ TradingView strategy development
2. ✅ Freqtrade Python implementation
3. ✅ Basic backtesting
4. 🔄 Parameter optimization (hyperopt)
5. ⏳ Paper trading (dry run)
6. ⏳ Live trading (small capital)

---

**Remember:** This is a living document. Update it as the project evolves!