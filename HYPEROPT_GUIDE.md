# Hyperopt Guide - DynamicAggressiveHighTP Strategy

## Quick Start

### 1. Run Hyperopt (1000 epochs, ~20-40 minutes)

```powershell
# From PowerShell:
cd D:\Scripts\trading\freqtrade
.\run_hyperopt.bat
```

**Or manually:**
```powershell
.venv\Scripts\freqtrade.exe hyperopt --strategy DynamicAggressiveHighTP --hyperopt-loss SharpeHyperOptLoss --spaces buy sell --epochs 1000 --timerange 20240101-20240930 --min-trades 100
```

### 2. Validate Results (anti-overfitting)

After hyperopt completes:

```powershell
.\validate_hyperopt.bat
```

### 3. Compare Results

**Good Sign (Robust Parameters):**
```
Training (Jan-Sep):  +15% profit, 1.5 Sharpe
Test (Oct-Dec):      +12% profit, 1.3 Sharpe
→ Results are similar, parameters are robust ✅
```

**Bad Sign (Overfitting):**
```
Training (Jan-Sep):  +50% profit, 3.0 Sharpe
Test (Oct-Dec):      -20% profit, -0.5 Sharpe
→ Parameters are overfit to training data ❌
```

## Parameters Being Optimized

### Entry Parameters (Buy Space)
- **RSI Threshold**: 20-40 (default 30)
  - Lower = more aggressive entries
  - Higher = more conservative entries

- **Momentum Threshold**: -5% to -1% (default -2%)
  - More negative = wait for bigger dips
  - Less negative = enter on smaller dips

### Exit Parameters (Sell Space)
- **Stop Loss**: -10% to -2% PnL (default -5%)
  - Tighter = less risk, more frequent stops
  - Wider = more risk, fewer stops

- **Take Profit**: 30% to 150% PnL (default 80%)
  - Lower = take profits earlier
  - Higher = let winners run longer

- **Trailing Activation**: 1% to 5% PnL (default 2%)
  - Lower = activate trailing sooner
  - Higher = wait for bigger gains before trailing

### Fixed Parameters (Not Optimized)
- **Emergency Stop Loss**: -8% PnL (safety feature)
- **Dynamic Trailing Distances**: Scaled with profit (2% → 20%)
- **Max Open Trades**: 5
- **Leverage**: 10x (BTC/ETH/SOL/BNB), 5x (SUI/LTC)

## Understanding Hyperopt Output

### Best Result Example
```
Best result:
   159/1000:    156 trades. Avg profit  1.45%. Total profit  0.23456789 BTC (  23.46%).
                Avg duration 45.3 min. Objective: 2.34567
                Buy hyperspace params:
                {
                  'buy_rsi_threshold': 25,
                  'buy_momentum_threshold': -0.03
                }
                Sell hyperspace params:
                {
                  'stop_loss_pnl': -0.07,
                  'take_profit_pnl': 1.20,
                  'trailing_activation_pnl': 0.03
                }
```

**Key Metrics:**
- **Total profit**: Higher is better
- **Avg profit**: Higher is better (but watch trade count)
- **Objective**: Sharpe ratio (risk-adjusted returns)
  - > 2.0 = Excellent
  - 1.0-2.0 = Good
  - 0-1.0 = Acceptable
  - < 0 = Bad

### Trade Count
- Too many trades (>2000): Strategy is overtrading
- Too few trades (<100): Not enough data to validate
- Sweet spot: 500-1500 trades for 9 months

## Implementing Optimized Parameters

### Option A: Update Freqtrade Strategy (for further testing)

Edit `user_data/strategies/DynamicAggressiveHighTP.py`:

```python
# Change default values to optimized ones
buy_rsi_threshold = IntParameter(20, 40, default=25, space='buy', optimize=False)
buy_momentum_threshold = DecimalParameter(-0.05, -0.01, default=-0.03, decimals=3, space='buy', optimize=False)
stop_loss_pnl = DecimalParameter(-0.10, -0.02, default=-0.07, decimals=2, space='sell', optimize=False)
take_profit_pnl = DecimalParameter(0.30, 1.50, default=1.20, decimals=2, space='sell', optimize=False)
trailing_activation_pnl = DecimalParameter(0.01, 0.05, default=0.03, decimals=2, space='sell', optimize=False)
```

### Option B: Implement in Pacifica Bot (RECOMMENDED)

Edit `D:\Scripts\trading\pacifica-python-sdk\trading_bot_ws.py`:

1. Find the entry logic and update thresholds:
```python
# OLD
RSI_THRESHOLD = 30
MOMENTUM_THRESHOLD = -0.02

# NEW (from hyperopt)
RSI_THRESHOLD = 25  # From hyperopt results
MOMENTUM_THRESHOLD = -0.03  # From hyperopt results
```

2. Find the exit logic and update levels:
```python
# OLD
STOP_LOSS_PNL = -0.05
TAKE_PROFIT_PNL = 0.80
TRAILING_ACTIVATION_PNL = 0.02

# NEW (from hyperopt)
STOP_LOSS_PNL = -0.07  # From hyperopt results
TAKE_PROFIT_PNL = 1.20  # From hyperopt results
TRAILING_ACTIVATION_PNL = 0.03  # From hyperopt results
```

3. Test in paper trading mode first!

## Anti-Overfitting Best Practices

### 1. Walk-Forward Analysis
- Train on Jan-Sep (75% of data)
- Test on Oct-Dec (25% of data)
- If test results ≈ train results → GOOD
- If test results << train results → OVERFIT

### 2. Multiple Validation Runs
```powershell
# Test on different periods
.venv\Scripts\freqtrade.exe backtesting --strategy DynamicAggressiveHighTP --timerange 20240401-20240630
.venv\Scripts\freqtrade.exe backtesting --strategy DynamicAggressiveHighTP --timerange 20240701-20240930
```

If all periods show similar results → parameters are robust

### 3. Reasonable Ranges
- Don't optimize too many parameters at once
- Keep some parameters fixed (like emergency_sl_pnl)
- Use realistic ranges based on market behavior

### 4. Trade Count Validation
- Need minimum 100 trades to be statistically significant
- Too many trades (>3000) suggests overtrading
- Optimal: 500-1500 trades per 9 months

## Common Issues

### Issue: "No trades in training period"
**Solution**: Entry conditions too strict. Relax ranges:
```python
buy_rsi_threshold = IntParameter(15, 45, ...)  # Wider range
```

### Issue: "All trades are losses"
**Solution**: Exit conditions too tight or SL too close:
```python
stop_loss_pnl = DecimalParameter(-0.15, -0.02, ...)  # Allow wider stops
```

### Issue: "Hyperopt very slow"
**Solution**:
- Reduce epochs: `--epochs 500`
- Reduce timerange: `--timerange 20240101-20240630`
- Use fewer pairs in config.json

### Issue: "Best result has Sharpe < 0"
**Solution**: Strategy fundamentally doesn't work on this data.
- Try different loss function: `--hyperopt-loss SortinoHyperOptLoss`
- Try different pairs
- Reconsider strategy logic

## Loss Functions

### SharpeHyperOptLoss (Default)
- Optimizes risk-adjusted returns
- Penalizes high volatility
- Good for consistent strategies

### SortinoHyperOptLoss
- Like Sharpe but only penalizes downside volatility
- Good for asymmetric strategies

### MaxDrawDownRelativeHyperOptLoss
- Minimizes maximum drawdown
- Good for risk-averse strategies

### CalmarHyperOptLoss
- Optimizes return/max_drawdown ratio
- Good for capital preservation

## Next Steps After Successful Hyperopt

1. ✅ Validate on test period (Oct-Dec 2024)
2. ✅ Compare train vs test performance
3. ✅ If robust: implement in Pacifica bot
4. ✅ Paper trade for 1-2 weeks
5. ✅ If paper trading successful: go live
6. ✅ Monitor and iterate

## Documentation

- **Freqtrade Hyperopt Docs**: https://www.freqtrade.io/en/stable/hyperopt/
- **Parameter Optimization Guide**: https://www.freqtrade.io/en/stable/strategy-customization/#hyperopt-parameters
- **Loss Functions**: https://www.freqtrade.io/en/stable/hyperopt/#loss-functions

---

**Remember**: Hyperopt finds parameters that worked in the PAST. Always validate on out-of-sample data before using in production!
