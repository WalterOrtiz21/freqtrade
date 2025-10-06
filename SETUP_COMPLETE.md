# Freqtrade Setup Complete! ✅

## Installation Summary

- **Python Version**: 3.11.4 ✅
- **Freqtrade Version**: 2025.10-dev ✅
- **Hyperopt**: Installed ✅
- **Virtual Environment**: `.venv` created and ready ✅

## Strategy Files Ready

- ✅ `user_data/strategies/DynamicAggressiveHighTP.py` - Strategy implementation
- ✅ `convert_pacifica_to_freqtrade.py` - Data conversion script
- ✅ `FREQTRADE_INTEGRATION.md` - Full integration guide

## Quick Start Guide

### Option 1: Direct Executable (FASTEST - PowerShell/CMD)

```powershell
# From PowerShell or Command Prompt:
cd D:\Scripts\trading\freqtrade

# Run freqtrade directly (~2 seconds, no activation needed)
.venv\Scripts\freqtrade.exe --version
.venv\Scripts\freqtrade.exe list-strategies
.venv\Scripts\freqtrade.exe backtesting --strategy DynamicAggressiveHighTP --timerange 20240101-20241231
```

### Option 2: Activate Virtual Environment (PowerShell)

```powershell
# Activate venv (for multiple commands)
cd D:\Scripts\trading\freqtrade
.venv\Scripts\Activate.ps1

# Now use freqtrade directly
freqtrade --version
freqtrade list-strategies
freqtrade backtesting --strategy DynamicAggressiveHighTP --timerange 20240101-20241231

# Deactivate when done
deactivate
```

### Option 3: Git Bash

```bash
# From Git Bash:
cd /d/Scripts/trading/freqtrade

# Direct executable
.venv/Scripts/freqtrade --version

# Or activate venv
source .venv/Scripts/activate
freqtrade --version
deactivate
```

## Next Steps

### 1. Convert Historical Data (Optional)

If you have historical CSV data from Pacifica:

```bash
python convert_pacifica_to_freqtrade.py
```

This will convert your Pacifica data to Freqtrade format in `user_data/data/binance/`.

### 2. Download Market Data

Freqtrade can download data directly from exchanges:

```powershell
# PowerShell/CMD (direct executable)
.venv\Scripts\freqtrade.exe download-data --exchange binance --pairs BTC/USDT ETH/USDT SOL/USDT BNB/USDT --timerange 20240101-20241231 --timeframe 5m
```

```bash
# Git Bash
.venv/Scripts/freqtrade download-data \
  --exchange binance \
  --pairs BTC/USDT ETH/USDT SOL/USDT BNB/USDT \
  --timerange 20240101-20241231 \
  --timeframe 5m
```

### 3. Run Initial Backtest

Test the DYNAMIC-Aggressive High TP strategy:

```powershell
# PowerShell/CMD
.venv\Scripts\freqtrade.exe backtesting --strategy DynamicAggressiveHighTP --timerange 20240101-20241231
```

### 4. Optimize Parameters with Hyperopt

**This is the main use case for Freqtrade!**

```powershell
# PowerShell/CMD - Optimize on training period (9 months)
.venv\Scripts\freqtrade.exe hyperopt --strategy DynamicAggressiveHighTP --hyperopt-loss SharpeHyperOptLoss --spaces buy sell --epochs 1000 --timerange 20240101-20240930

# Validate on test period (3 months)
.venv\Scripts\freqtrade.exe backtesting --strategy DynamicAggressiveHighTP --timerange 20241001-20241231
```

### 5. Implement Optimized Parameters

After hyperopt finds better parameters:
1. Review the results in the Freqtrade output
2. Compare train vs test performance (anti-overfitting check)
3. If consistent: implement in `../pacifica-python-sdk/trading_bot_ws.py`

## Useful Commands

```powershell
# PowerShell/CMD - List available strategies
.venv\Scripts\freqtrade.exe list-strategies

# Show strategy details
.venv\Scripts\freqtrade.exe show-strategy DynamicAggressiveHighTP

# List available exchanges
.venv\Scripts\freqtrade.exe list-exchanges

# Test strategy configuration
.venv\Scripts\freqtrade.exe test-pairlist --config user_data/config.json
```

## Configuration Files

Create a config file for your backtests:

```bash
# Generate default config
freqtrade new-config --config user_data/config.json
```

Edit `user_data/config.json` with your preferences:
- Exchange: `binance`
- Pairs: `BTC/USDT`, `ETH/USDT`, `SOL/USDT`, etc.
- Timeframe: `5m`
- Stake currency: `USDT`

## Documentation

- **Freqtrade Integration Guide**: `FREQTRADE_INTEGRATION.md`
- **Official Freqtrade Docs**: https://www.freqtrade.io/en/stable/
- **Hyperopt Guide**: https://www.freqtrade.io/en/stable/hyperopt/
- **Backtesting Guide**: https://www.freqtrade.io/en/stable/backtesting/

## Anti-Overfitting Best Practices

1. **Train/Test Split**: Always split data (75% train, 25% test)
2. **Walk-Forward Analysis**: Validate on unseen data
3. **Consistency Check**: Train results ≈ Test results = GOOD
4. **Multiple Symbols**: Test on different assets
5. **Out-of-Sample Period**: Keep 20%+ of data completely unseen

## Troubleshooting

**Issue**: `freqtrade: command not found`
- **Solution**: Use direct executable `.venv\Scripts\freqtrade.exe` (PowerShell/CMD)
- **Or**: Activate venv first: `.venv\Scripts\Activate.ps1` (PowerShell) or `source .venv/Scripts/activate` (Git Bash)

**Issue**: `No data available`
- **Solution**: Download data first with `.venv\Scripts\freqtrade.exe download-data`

**Issue**: Python version error
- **Solution**: This .venv uses Python 3.11.4 (correct version)

**Issue**: Command hangs or takes too long
- **Solution**: Use direct executable (`.venv\Scripts\freqtrade.exe`) instead of bat files (~2 seconds)

## Production Bot

The production bot is in `../pacifica-python-sdk/trading_bot_ws.py`:
- Already running the validated DYNAMIC-Aggressive High TP strategy
- Use Freqtrade to find BETTER parameters
- Implement validated improvements in the production bot

---

**Setup completed on**: 2025-10-06
**Python version**: 3.11.4
**Ready for**: Backtesting, Hyperopt optimization, Strategy validation
