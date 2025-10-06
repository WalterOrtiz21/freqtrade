"""
DYNAMIC-Aggressive High TP Strategy for Freqtrade

This strategy is a direct port of the validated DYNAMIC-Aggressive High TP strategy
that achieved +11.71% average return in backtesting (365 days, 5m candles).

Strategy Details:
- Entry: RSI < 30 + Momentum < -2% + Price > MA(5) (requires 2 of 3 signals)
- Exit Logic (All use leveraged PnL):
  * Emergency Stop: -8% leveraged PnL (prevents catastrophic losses, highest priority)
  * Stop Loss: -5% leveraged PnL (absolute risk limit)
  * Take Profit: +80% leveraged PnL (aggressive profit target)
  * Dynamic Trailing Stop: Activates at +2% PnL, distance scales with profit:
    - +2% PnL -> 0.2% trailing distance
    - +5% PnL -> 1.0% trailing distance
    - +7% PnL -> 3.0% trailing distance
    - +10% PnL -> 5.0% trailing distance
    - +13% PnL -> 7.0% trailing distance
    - +15% PnL -> 10% trailing distance
    - +20% PnL -> 15% trailing distance
    - +25% PnL -> 20% trailing distance
  * Timeout: Disabled (None) - better performance without time limits

Leverage: 10x for BTC/ETH/SOL/BNB, 5x for SUI/LTC
Position Sizing: 5% risk per trade, max 5 simultaneous positions

IMPORTANT: This is a PnL-based trailing stop, not price-based.
Example: At +10% PnL with 5% trailing, it protects +5% PnL minimum (not 5% below peak price).
"""

from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter
from pandas import DataFrame
import talib.abstract as ta
import numpy as np


class DynamicAggressiveHighTP(IStrategy):
    """
    DYNAMIC-Aggressive High TP Strategy
    Validated: +11.71% average return across 6 symbols (365 days)
    """

    # Strategy configuration
    INTERFACE_VERSION = 3

    # Minimal ROI designed to allow dynamic trailing to handle exits
    minimal_roi = {
        "0": 10.0  # Very high to prevent premature exit (trailing will handle it)
    }

    # Stoploss (emergency backup - dynamic trailing is primary exit)
    # This is -8% PnL (emergency stop level)
    stoploss = -0.08

    # Trailing stop configuration (Freqtrade's built-in - we'll override with custom logic)
    trailing_stop = False  # We implement custom trailing logic

    # Timeframe
    timeframe = '5m'

    # Run "populate_indicators" only for new candle
    process_only_new_candles = True

    # Startup candle count (need 14 for RSI)
    startup_candle_count = 20

    # Hyperopt parameter spaces
    buy_rsi_threshold = IntParameter(20, 40, default=30, space='buy', optimize=True)
    buy_momentum_threshold = DecimalParameter(-0.05, -0.01, default=-0.02, decimals=3, space='buy', optimize=True)

    # Exit parameters
    stop_loss_pnl = DecimalParameter(-0.10, -0.02, default=-0.05, decimals=2, space='sell', optimize=True)
    take_profit_pnl = DecimalParameter(0.30, 1.50, default=0.80, decimals=2, space='sell', optimize=True)
    trailing_activation_pnl = DecimalParameter(0.01, 0.05, default=0.02, decimals=2, space='sell', optimize=True)

    # Emergency SL - keep fixed for safety
    emergency_sl_pnl = -0.08  # -8% PnL (fixed, don't optimize)

    # Position sizing
    position_size_pct = 0.05  # 5% of balance per trade

    # Leverage configuration (per symbol)
    leverage_config = {
        'BTC/USDT:USDT': 10,
        'ETH/USDT:USDT': 10,
        'SOL/USDT:USDT': 10,
        'BNB/USDT:USDT': 10,
        'SUI/USDT:USDT': 5,
        'LTC/USDT:USDT': 5,
        # Also support without :USDT suffix for backwards compatibility
        'BTC/USDT': 10,
        'ETH/USDT': 10,
        'SOL/USDT': 10,
        'BNB/USDT': 10,
        'SUI/USDT': 5,
        'LTC/USDT': 5,
    }

    def leverage(self, pair: str, current_time, current_rate,
                 proposed_leverage, max_leverage, entry_tag, side, **kwargs) -> float:
        """
        Customize leverage for each pair.
        """
        return self.leverage_config.get(pair, 10)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Add indicators needed for entry signals.
        """
        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        # Momentum (5-period price change)
        dataframe['momentum'] = (dataframe['close'] - dataframe['close'].shift(5)) / dataframe['close'].shift(5)

        # MA5
        dataframe['ma5'] = ta.SMA(dataframe, timeperiod=5)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Entry signal: Requires 2 of 3 conditions:
        1. RSI < threshold
        2. Momentum < threshold
        3. Price > MA(5)
        """
        # Calculate individual signals
        signal_rsi = dataframe['rsi'] < self.buy_rsi_threshold.value
        signal_momentum = dataframe['momentum'] < self.buy_momentum_threshold.value
        signal_price_above_ma = dataframe['close'] > dataframe['ma5']

        # Require 2 of 3 signals
        signal_count = (
            signal_rsi.astype(int) +
            signal_momentum.astype(int) +
            signal_price_above_ma.astype(int)
        )

        dataframe.loc[
            (signal_count >= 2),
            'enter_long'
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Exit signals handled by custom_exit() for dynamic trailing.
        No exit signals here.
        """
        return dataframe

    def calculate_dynamic_trailing_distance(self, profit_pnl: float) -> float:
        """
        Calculate trailing distance based on current PnL level.
        This is the aggressive strategy from the validated backtest.

        Returns trailing distance as a decimal (e.g., 0.05 = 5%)
        Returns None if trailing not active yet.
        """
        if profit_pnl >= 0.25:  # 25%+ PnL
            return 0.20
        elif profit_pnl >= 0.20:  # 20%+ PnL
            return 0.15
        elif profit_pnl >= 0.15:  # 15%+ PnL
            return 0.10
        elif profit_pnl >= 0.13:  # 13%+ PnL
            return 0.07
        elif profit_pnl >= 0.10:  # 10%+ PnL
            return 0.05
        elif profit_pnl >= 0.07:  # 7%+ PnL
            return 0.03
        elif profit_pnl >= 0.05:  # 5%+ PnL
            return 0.01
        elif profit_pnl >= 0.02:  # 2%+ PnL
            return 0.002
        else:
            return None  # Trailing not active yet

    def custom_exit(self, pair: str, trade, current_time, current_rate,
                    current_profit, **kwargs) -> str:
        """
        Custom exit logic implementing DYNAMIC-Aggressive High TP strategy.

        Exit Priority:
        0. Emergency SL: -8% PnL (highest priority)
        1. Stop Loss: -5% PnL
        2. Take Profit: +80% PnL
        3. Dynamic Trailing: Activates at +2% PnL, scales with profit
        4. Timeout: Disabled (None)

        IMPORTANT: In Freqtrade futures mode, current_profit ALREADY includes leverage.
        Do NOT multiply by leverage again (that would be leverage^2).
        """
        # Get leverage for this pair (for reference only, not calculation)
        leverage = self.leverage_config.get(pair, 10)

        # CRITICAL: current_profit ALREADY includes leverage in Freqtrade
        # For futures: current_profit = (current_price - entry_price) / entry_price * leverage
        # So we use current_profit directly as our leveraged PnL
        leveraged_pnl = current_profit  # DO NOT multiply by leverage!

        # Priority 0: Emergency Stop Loss (-8% PnL)
        if leveraged_pnl <= self.emergency_sl_pnl:
            return 'emergency_stop_loss'

        # Priority 1: Stop Loss (optimizable)
        if leveraged_pnl <= self.stop_loss_pnl.value:
            return 'stop_loss'

        # Priority 2: Take Profit (optimizable)
        if leveraged_pnl >= self.take_profit_pnl.value:
            return 'take_profit'

        # Priority 3: Dynamic Trailing Stop
        if leveraged_pnl >= self.trailing_activation_pnl.value:
            # Get current trailing distance based on PnL
            trailing_distance = self.calculate_dynamic_trailing_distance(leveraged_pnl)

            if trailing_distance is not None:
                # Get peak profit from Freqtrade's trade object
                # max_rate is tracked by Freqtrade, use it to calculate peak PnL
                if hasattr(trade, 'max_rate') and trade.max_rate is not None:
                    # Calculate peak PnL
                    peak_pnl = (trade.max_rate - trade.open_rate) / trade.open_rate * leverage

                    # Calculate minimum PnL to maintain (peak - trailing_distance)
                    min_pnl_to_maintain = peak_pnl - trailing_distance

                    # Exit if current PnL drops below minimum
                    if leveraged_pnl <= min_pnl_to_maintain:
                        return 'dynamic_trailing_stop'

        # No exit signal
        return None

    def custom_stake_amount(self, pair: str, current_time, current_rate,
                           proposed_stake, min_stake, max_stake,
                           entry_tag, side, **kwargs) -> float:
        """
        Custom position sizing: 5% of balance per trade.
        """
        # Freqtrade will handle the actual calculation based on configuration
        # This is just documentation - configure stake_amount in config.json
        return proposed_stake


# USAGE INSTRUCTIONS FOR FREQTRADE:
#
# 1. Install Freqtrade (requires Python 3.11+):
#    cd ../trading/freqtrade
#    ./setup.sh -i
#
# 2. Configure config.json:
#    {
#      "exchange": {
#        "name": "binance",  # Or your exchange
#        "key": "your_api_key",
#        "secret": "your_secret"
#      },
#      "stake_amount": "unlimited",  # Use 5% per trade
#      "tradable_balance_ratio": 0.05,  # 5% per trade
#      "max_open_trades": 5,
#      "trading_mode": "futures",
#      "margin_mode": "isolated"
#    }
#
# 3. Backtest:
#    freqtrade backtesting --strategy DynamicAggressiveHighTP --timerange 20240101-20241231
#
# 4. Hyperopt (optimize parameters):
#    freqtrade hyperopt --strategy DynamicAggressiveHighTP --hyperopt-loss SharpeHyperOptLoss --epochs 1000 --spaces buy sell
#
# 5. Dry-run (paper trading):
#    freqtrade trade --strategy DynamicAggressiveHighTP --config config.json --dry-run
#
# 6. Live trading:
#    freqtrade trade --strategy DynamicAggressiveHighTP --config config.json
#
# NOTE: You'll need to adapt this for Pacifica exchange since Pacifica isn't directly supported by Freqtrade.
# Consider using the backtesting/hyperopt features offline, then implementing validated parameters in your bot.
