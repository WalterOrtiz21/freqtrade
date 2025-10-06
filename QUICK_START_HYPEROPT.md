# Quick Start - Hyperopt 1000 Epochs

## Step 1: Run Hyperopt (~20-40 minutes)

```powershell
# From PowerShell (in D:\Scripts\trading\freqtrade):
python run_hyperopt.py
```

**Or with venv python directly:**
```powershell
.venv\Scripts\python.exe run_hyperopt.py
```

**What it does:**
- Tests 1000 different parameter combinations
- Training period: Jan-Sep 2024 (9 months)
- Optimizes: RSI threshold, momentum threshold, stop loss, take profit, trailing activation
- Finds parameters with best Sharpe ratio (risk-adjusted returns)

## Step 2: Wait for Results

You'll see output like:
```
Epoch   1/1000:  159 trades. Avg profit  0.45%. Total profit  0.12345 BTC (12.34%). Objective: 1.234
Epoch   2/1000:  167 trades. Avg profit  0.89%. Total profit  0.23456 BTC (23.46%). Objective: 2.345
...
Best result:
   456/1000:  178 trades. Avg profit  1.23%. Total profit  0.34567 BTC (34.57%). Objective: 3.456
   Buy params: {'buy_rsi_threshold': 25, 'buy_momentum_threshold': -0.03}
   Sell params: {'stop_loss_pnl': -0.07, 'take_profit_pnl': 1.20, 'trailing_activation_pnl': 0.03}
```

**Copy these parameters!** You'll need them for validation.

## Step 3: Update Strategy with Best Parameters

Edit `user_data/strategies/DynamicAggressiveHighTP.py`:

```python
# Find these lines and change DEFAULT values to hyperopt results:
buy_rsi_threshold = IntParameter(20, 40, default=25, ...)  # Change 30 → 25
buy_momentum_threshold = DecimalParameter(-0.05, -0.01, default=-0.03, ...)  # Change -0.02 → -0.03
stop_loss_pnl = DecimalParameter(-0.10, -0.02, default=-0.07, ...)  # Change -0.05 → -0.07
take_profit_pnl = DecimalParameter(0.30, 1.50, default=1.20, ...)  # Change 0.80 → 1.20
trailing_activation_pnl = DecimalParameter(0.01, 0.05, default=0.03, ...)  # Change 0.02 → 0.03
```

## Step 4: Validate on Out-of-Sample Data

```powershell
python validate_hyperopt.py
```

**What it does:**
- Tests optimized parameters on Oct-Dec 2024 (unseen data)
- Checks if parameters are overfit

**Expected output:**
```
Training (Jan-Sep): +34.57% profit, 3.456 Sharpe
Test (Oct-Dec):     +28.12% profit, 2.987 Sharpe  ← Should be similar!
```

**Interpretation:**
- ✅ Test ≈ Train → Parameters are ROBUST → Ready for implementation
- ❌ Test << Train → Parameters are OVERFIT → Try again with different settings

## Step 5: Implement in Pacifica Bot

If validation passes, update `D:\Scripts\trading\pacifica-python-sdk\trading_bot_ws.py`:

```python
# Find these constants and update:
RSI_THRESHOLD = 25  # From hyperopt
MOMENTUM_THRESHOLD = -0.03  # From hyperopt
STOP_LOSS_PNL = -0.07  # From hyperopt
TAKE_PROFIT_PNL = 1.20  # From hyperopt
TRAILING_ACTIVATION_PNL = 0.03  # From hyperopt
```

## Step 6: Paper Trade

Test in paper trading mode for 1-2 weeks before going live.

---

## Troubleshooting

**Issue**: `python: command not found`
**Solution**: Use `.venv\Scripts\python.exe run_hyperopt.py`

**Issue**: Script hangs or takes too long
**Solution**: Reduce epochs in `run_hyperopt.py` (change `EPOCHS = 1000` to `EPOCHS = 500`)

**Issue**: No trades in training period
**Solution**: Entry conditions too strict. Widen ranges in strategy file.

**Issue**: All results are negative
**Solution**: Strategy may not work on 2024 data. Consider different approach.

---

## Quick Reference

```powershell
# Run hyperopt
python run_hyperopt.py

# Validate results
python validate_hyperopt.py

# Test on full year (after validation passes)
.venv\Scripts\freqtrade.exe backtesting --strategy DynamicAggressiveHighTP --timerange 20240101-20241231
```

See `HYPEROPT_GUIDE.md` for full documentation.
