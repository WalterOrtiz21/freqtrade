# CHoCH/BOS Strategy - Live Trading Guide

## Quick Start

### Option 1: Interactive Mode (Recommended for First Time)

```bash
./start_trading.sh
```

This will guide you through selecting:
1. Exchange (Binance Futures/Spot, Hyperliquid)
2. Timeframe (15m or 5m)
3. Mode (Dry Run or Live)

### Option 2: Command Line Mode

```bash
./start_trading.sh [exchange] [timeframe] [mode]
```

**Examples:**
```bash
# Dry run with Binance Futures on 15m (recommended for testing)
./start_trading.sh binance_futures 15m dry

# Dry run with Binance Spot on 5m
./start_trading.sh binance_spot 5m dry

# LIVE trading with Binance Futures on 15m (real money!)
./start_trading.sh binance_futures 15m live
```

### Option 3: Direct Freqtrade Command

```bash
# Using cmd.exe or Git Bash
cd /d/Scripts/freqtrade
/c/Users/walte/AppData/Local/Programs/Python/Python312/Scripts/freqtrade.exe trade \
    --strategy CHoCHBOSStrategy \
    --config config_choch_bos_live_binance_futures.json \
    --timeframe 15m
```

---

## Configuration Files

### 1. Binance Futures (Recommended)
**File:** `config_choch_bos_live_binance_futures.json`

**Features:**
- 3x leverage with isolated margin
- Optimized for: SOL, ADA, DOT, AVAX, LINK, AAVE, BNB
- Excludes BTC (poor backtest performance)
- Same settings as successful backtests (+20.54% on Altcoins)

**Setup:**
1. Replace `YOUR_BINANCE_API_KEY` and `YOUR_BINANCE_API_SECRET`
2. Enable Futures trading in your Binance account
3. Create API key with Futures permissions
4. Set `"dry_run": false` when ready for live trading

### 2. Binance Spot
**File:** `config_choch_bos_live_binance_spot.json`

**Features:**
- No leverage (safer for beginners)
- Uses existing API credentials from config_usdc_usdt_live.json
- Fixed stake amount: 100 USDT per trade
- Max 7 open trades

**Note:** Strategy was optimized for Futures, so performance may differ in Spot.

### 3. Hyperliquid (Future Use)
**File:** `config_choch_bos_live_hyperliquid.json`

**Features:**
- Perpetual futures on Hyperliquid DEX
- Uses USDC as collateral
- Requires wallet address and private key

**Setup:**
1. Install Hyperliquid support for ccxt (if available)
2. Replace `YOUR_HYPERLIQUID_WALLET_ADDRESS` and `YOUR_HYPERLIQUID_PRIVATE_KEY`
3. Test thoroughly in dry-run mode first

---

## Strategy Parameters (Optimized)

The strategy uses these parameters (saved in `CHoCHBOSStrategy.json`):

```json
{
  "params": {
    "buy": {
      "zigzag_depth": 35,
      "zigzag_deviation": 10
    },
    "sell": {
      "move_to_be_after_tp1": false,
      "tp1_percentage": 0.3,
      "tp2_percentage": 0.5,
      "use_bos_tps": false
    }
  }
}
```

**These parameters were optimized for 15m timeframe with 4.5 years of training data (2020-2024).**

### For 5m Timeframe:
- Parameters are NOT optimized yet for 5m
- Currently running hyperopt to find optimal 5m parameters
- Use 15m for production until 5m optimization completes

---

## Changing Configuration

### Switch Exchange

Edit the config file or use environment-specific configs:

```bash
# Binance Futures
./start_trading.sh binance_futures 15m dry

# Binance Spot
./start_trading.sh binance_spot 15m dry

# Hyperliquid
./start_trading.sh hyperliquid 15m dry
```

### Switch Timeframe

**15m (Recommended):**
```bash
./start_trading.sh binance_futures 15m dry
```

**5m (Experimental):**
```bash
./start_trading.sh binance_futures 5m dry
```

Note: Wait for hyperopt to complete for optimal 5m parameters.

### Switch Strategy

Edit `start_trading.sh` line 8:
```bash
STRATEGY="CHoCHBOSStrategy"  # Change to another strategy name
```

Or override in freqtrade command:
```bash
freqtrade.exe trade --strategy YourStrategyName --config config.json --timeframe 15m
```

### Modify Pair Whitelist

Edit the config file's `pair_whitelist` section:

```json
"pair_whitelist": [
    "SOL/USDT:USDT",
    "ADA/USDT:USDT",
    "ETH/USDT:USDT"  // Add or remove pairs
]
```

**Recommended pairs (based on backtest):**
- ✅ Best: SOL, ADA, LINK, AAVE (+6% to +13%)
- ✅ Good: DOT, BNB, ETH (+0.5% to +5%)
- ❌ Avoid: BTC (-5.89% in backtest)

---

## Pre-Launch Checklist

### Before Dry Run:
- [ ] Config file has correct API keys
- [ ] `"dry_run": true` is set
- [ ] Pair whitelist is configured
- [ ] Timeframe matches optimized parameters (15m recommended)
- [ ] Telegram notifications configured (optional)

### Before LIVE Trading:
- [ ] Tested in dry-run mode for at least 1 week
- [ ] Understand the strategy's risk (3% stoploss per trade)
- [ ] Account has sufficient balance
- [ ] Set `"dry_run": false` in config
- [ ] API key has correct permissions (Futures/Spot trading)
- [ ] API key IP whitelist configured (recommended)
- [ ] Withdraw permissions DISABLED on API key (security)
- [ ] Telegram notifications enabled for alerts
- [ ] Monitoring setup (FreqUI or API server)

---

## Monitoring

### FreqUI Web Interface

Access at: `http://127.0.0.1:8080` (or configured port)

**Default credentials:**
- Username: `freqtrader`
- Password: `CHANGE_THIS_PASSWORD` (change in config!)

### API Server

Ports:
- Binance Futures: 8080
- Binance Spot: 8081
- Hyperliquid: 8082

### Telegram Notifications

Enable in config:
```json
"telegram": {
    "enabled": true,
    "token": "YOUR_BOT_TOKEN",
    "chat_id": "YOUR_CHAT_ID"
}
```

Get notifications for:
- Trade entries/exits
- Profit/Loss updates
- Errors and warnings

### Log Files

Location: `user_data/logs/`

View latest:
```bash
tail -f user_data/logs/freqtrade.log
```

---

## Performance Expectations (Based on Backtests)

### 15m Timeframe (Optimized)

**Altcoins Group (SOL, ADA, DOT, AVAX):**
- Profit: +20.54% (4 months)
- Win Rate: 71.3%
- Trades: 143 (1.21/day)
- Best Pair: SOL +10.59%

**DeFi Group (ETH, LINK, AAVE):**
- Profit: +6.89% (4 months)
- Win Rate: 70.8%
- Trades: 96 (0.81/day)
- Best Pair: ETH +4.90%

**Combined (7 pairs excluding BTC):**
- Expected: +15-20% per 4 months
- Win Rate: 70-72%
- Trades: 2-3 per day

### 5m Timeframe (Not Optimized Yet)

**Current Performance:**
- Profit: +10.55% Altcoins, +5.92% DeFi
- Win Rate: 67.4%
- Trades: 3-4 per day

**After Optimization:**
- Expected: Similar or better than 15m
- More trading opportunities (3x more signals)

---

## Risk Management

### Per-Trade Risk:
- **Stoploss:** -3% (-9% with 3x leverage)
- **Target:** +1% to +10% (trailing stop)
- **Average:** +0.5% per winning trade

### Portfolio Risk:
- **Max Open Trades:** 4 (Futures) / 7 (Spot)
- **Max Drawdown (backtest):** ~11-12%
- **Tradable Balance:** 99% (Futures) / 95% (Spot)

### Leverage (Futures Only):
- **Mode:** Isolated (3x effective leverage)
- **Margin:** Separate per trade (safer than cross)
- **Liquidation Risk:** Lower with isolated margin

---

## Troubleshooting

### API Errors

**"Invalid API key":**
- Check key/secret in config
- Verify key permissions (Futures/Spot trading)
- Check IP whitelist

**"Insufficient balance":**
- Check account balance
- Verify `stake_amount` in config
- Reduce `max_open_trades`

### Strategy Errors

**"No buy signals":**
- Market may not have CHoCH/BOS patterns
- Check logs for indicator calculation errors
- Verify timeframe has sufficient data

**"Too many trades":**
- Reduce `max_open_trades`
- Add more strict filters (requires strategy modification)
- Switch to 15m from 5m (fewer signals)

### Performance Issues

**Strategy underperforming:**
- Verify you're using 15m timeframe with optimized parameters
- Check if market conditions changed (strategy optimized for 2020-2024)
- Consider walk-forward optimization with recent data

---

## Updating Parameters

### When 5m Hyperopt Completes:

1. Check hyperopt results:
```bash
freqtrade hyperopt-show --best
```

2. Update `CHoCHBOSStrategy.json` with new parameters

3. Backtest with test period to verify:
```bash
freqtrade backtesting --strategy CHoCHBOSStrategy --timeframe 5m --timerange 20250701-20251027
```

4. If results are good, restart live bot with 5m

### Custom Optimization:

To optimize for your preferred pairs/timeframe:
```bash
freqtrade hyperopt \
    --strategy CHoCHBOSStrategy \
    --config your_config.json \
    --timeframe 15m \
    --timerange 20200101-20240630 \
    --hyperopt-loss SharpeHyperOptLoss \
    --spaces buy sell \
    --epochs 200 \
    -j -1
```

---

## Emergency Stop

### Stop Bot:
```bash
# In terminal where bot is running:
CTRL+C

# Or kill process:
pkill -f freqtrade
```

### Cancel All Orders (FreqUI):
1. Go to http://127.0.0.1:8080
2. Click "Force Sell All"

### Cancel All Orders (Command):
```bash
freqtrade forceexit all --config your_config.json
```

---

## Support & Resources

### Freqtrade Documentation:
- Main: https://www.freqtrade.io/en/stable/
- Strategy: https://www.freqtrade.io/en/stable/strategy-customization/
- Live Trading: https://www.freqtrade.io/en/stable/bot-usage/

### Strategy Documentation:
- Analysis: `CHOCH_BOS_STRATEGY_ANALYSIS.md`
- FreqAI Notes: `CLAUDE.md`
- Comparison: Check backtest results in this guide

### Contact:
Check GitHub issues or Freqtrade Discord for support.

---

**⚠️ DISCLAIMER:**
Trading cryptocurrencies involves significant risk. Past performance does not guarantee future results. The strategy was backtested on historical data and may not perform the same in live trading. Only trade with money you can afford to lose. Do NOT start with live trading before thoroughly testing in dry-run mode.
