"""
LuxAlgo Market Structure Trailing Stop Strategy
Pure implementation of: https://www.tradingview.com/script/c4Z9xThG-Market-Structure-Trailing-Stop-LuxAlgo/

Entry Logic:
- LONG: close > pivot_high (when previous trend was bearish)
- SHORT: close < pivot_low (when previous trend was bullish)

Exit Logic:
- LuxAlgo Market Structure Trailing Stop
- Initial SL: minimum/maximum since last pivot
- Trailing: ts += (max - max[1]) * increment / 100

No CHoCH/BOS confirmation, no MST targets, no filters - just pure LuxAlgo logic.
"""

from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter
from pandas import DataFrame
import talib.abstract as ta
import numpy as np
from typing import Optional
from freqtrade.persistence import Trade
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class CHoCHBOSLuxAlgo(IStrategy):
    """
    Pure LuxAlgo Market Structure Trailing Stop implementation
    """

    INTERFACE_VERSION = 3

    # ROI table - disabled (rely on exits only)
    minimal_roi = {
        "0": 100  # Effectively disabled
    }

    # Stoploss
    stoploss = -0.10  # Fallback, custom_stoploss will override

    # Trailing stop
    trailing_stop = False
    use_custom_stoploss = True

    # Optimal timeframe
    timeframe = '15m'

    # Run "populate_indicators()" only for new candle
    process_only_new_candles = True

    # Position adjustment
    position_adjustment_enable = True
    max_entry_position_adjustment = -1

    # Startup candle count
    startup_candle_count: int = 200

    # Order types
    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'limit',
        'stoploss_on_exchange': False,
        'stoploss_on_exchange_interval': 60
    }

    # Order time in force
    order_time_in_force = {
        'entry': 'GTC',
        'exit': 'GTC'
    }

    # ============================
    # Strategy Parameters
    # ============================

    # Pivot detection length (LuxAlgo default: 14)
    pivot_length = IntParameter(
        low=10,
        high=30,
        default=14,
        space="buy",
        optimize=False,
        load=True
    )

    # Increment factor for trailing stop (LuxAlgo default: 100)
    increment_factor = DecimalParameter(
        low=50.0,
        high=200.0,
        default=100.0,
        decimals=0,
        space="sell",
        optimize=False,
        load=True
    )

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Calculate pivots and market structure
        """
        length = self.pivot_length.value

        # Calculate pivot highs and lows
        # Using rolling window to find local maxima/minima
        dataframe['pivot_high'] = 0.0
        dataframe['pivot_low'] = 0.0

        # Calculate pivot highs: highest high in window
        for i in range(length, len(dataframe) - length):
            window_high = dataframe['high'].iloc[i - length:i + length + 1]
            center_high = dataframe['high'].iloc[i]

            if center_high == window_high.max():
                dataframe.loc[dataframe.index[i], 'pivot_high'] = center_high

        # Calculate pivot lows: lowest low in window
        for i in range(length, len(dataframe) - length):
            window_low = dataframe['low'].iloc[i - length:i + length + 1]
            center_low = dataframe['low'].iloc[i]

            if center_low == window_low.min():
                dataframe.loc[dataframe.index[i], 'pivot_low'] = center_low

        # Forward fill pivot values (to track last pivot)
        dataframe['last_pivot_high'] = dataframe['pivot_high'].replace(0, np.nan).ffill().fillna(0)
        dataframe['last_pivot_low'] = dataframe['pivot_low'].replace(0, np.nan).ffill().fillna(0)

        # Track market structure state
        # 0 = neutral, 1 = bullish, -1 = bearish
        dataframe['ms_state'] = 0

        prev_state = 0
        for i in range(len(dataframe)):
            close = dataframe['close'].iloc[i]
            last_ph = dataframe['last_pivot_high'].iloc[i]
            last_pl = dataframe['last_pivot_low'].iloc[i]

            # Bullish: close > pivot_high (only if previous state was bearish - resetOn='CHoCH')
            if close > last_ph and last_ph > 0 and prev_state == -1:
                prev_state = 1
            # Bearish: close < pivot_low (only if previous state was bullish - resetOn='CHoCH')
            elif close < last_pl and last_pl > 0 and prev_state == 1:
                prev_state = -1
            # Initial state detection (when prev_state is still 0)
            elif prev_state == 0:
                if close > last_ph and last_ph > 0:
                    prev_state = 1
                elif close < last_pl and last_pl > 0:
                    prev_state = -1

            dataframe.loc[dataframe.index[i], 'ms_state'] = prev_state

        # Detect state changes for entry signals
        dataframe['ms_change'] = dataframe['ms_state'].diff()

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Entry when close crosses pivot level (with CHoCH logic)
        """
        # LONG: ms_state changed from -1 to 1 (close crossed above pivot_high)
        dataframe.loc[
            (dataframe['ms_change'] == 2),  # -1 to 1 = change of 2
            'enter_long'] = 1

        # SHORT: ms_state changed from 1 to -1 (close crossed below pivot_low)
        dataframe.loc[
            (dataframe['ms_change'] == -2),  # 1 to -1 = change of -2
            'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Exits handled by custom_stoploss and adjust_trade_position
        """
        # Emergency exit on opposite structure
        dataframe.loc[
            (dataframe['ms_change'] == 2),  # Bearish to bullish
            'exit_short'] = 1

        dataframe.loc[
            (dataframe['ms_change'] == -2),  # Bullish to bearish
            'exit_long'] = 1

        return dataframe

    def adjust_trade_position(self, trade, current_time, current_rate, current_profit,
                              min_stake, max_stake, current_entry_rate, current_exit_rate,
                              current_entry_profit, current_exit_profit, **kwargs) -> Optional[float]:
        """
        Track structure levels and max/min for LuxAlgo trailing stop
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        is_short = getattr(trade, 'is_short', trade.trade_direction == 'short')

        # Initialize on entry
        if trade.get_custom_data('structure_level', None) is None:
            entry_idx = dataframe[dataframe['date'] <= trade.open_date_utc].index[-1]

            if not is_short:
                # LONG: Find last pivot_high, then find minimum from there to entry
                # Get all pivot highs before entry
                pivots = dataframe.loc[:entry_idx, 'pivot_high']
                pivot_indices = pivots[pivots > 0].index

                if len(pivot_indices) > 0:
                    last_pivot_idx = pivot_indices[-1]
                    pivot_price = dataframe.loc[last_pivot_idx, 'pivot_high']

                    # Find minimum between pivot and entry
                    local_low = dataframe.loc[last_pivot_idx:entry_idx, 'low'].min()

                    trade.set_custom_data('structure_level', float(local_low))
                    trade.set_custom_data('trailing_max', float(current_rate))
                    trade.set_custom_data('prev_candle_max', float(current_rate))

                    logger.info(f"[{trade.pair}] LONG entry at {current_rate:.4f}")
                    logger.info(f"  Pivot high: {pivot_price:.4f} at idx {last_pivot_idx}")
                    logger.info(f"  Structure SL: {local_low:.4f} ({((local_low - current_rate) / current_rate * 100):.2f}%)")
                else:
                    # No pivot found, use fallback
                    fallback_sl = current_rate * (1 + self.stoploss)
                    trade.set_custom_data('structure_level', float(fallback_sl))
                    trade.set_custom_data('trailing_max', float(current_rate))
                    trade.set_custom_data('prev_candle_max', float(current_rate))
                    logger.warning(f"[{trade.pair}] LONG - No pivot found, using fallback SL: {fallback_sl:.4f}")

            else:
                # SHORT: Find last pivot_low, then find maximum from there to entry
                pivots = dataframe.loc[:entry_idx, 'pivot_low']
                pivot_indices = pivots[pivots > 0].index

                if len(pivot_indices) > 0:
                    last_pivot_idx = pivot_indices[-1]
                    pivot_price = dataframe.loc[last_pivot_idx, 'pivot_low']

                    # Find maximum between pivot and entry
                    local_high = dataframe.loc[last_pivot_idx:entry_idx, 'high'].max()

                    trade.set_custom_data('structure_level', float(local_high))
                    trade.set_custom_data('trailing_min', float(current_rate))
                    trade.set_custom_data('prev_candle_min', float(current_rate))

                    logger.info(f"[{trade.pair}] SHORT entry at {current_rate:.4f}")
                    logger.info(f"  Pivot low: {pivot_price:.4f} at idx {last_pivot_idx}")
                    logger.info(f"  Structure SL: {local_high:.4f} ({((local_high - current_rate) / current_rate * 100):.2f}%)")
                else:
                    # No pivot found, use fallback
                    fallback_sl = current_rate * (1 - self.stoploss)
                    trade.set_custom_data('structure_level', float(fallback_sl))
                    trade.set_custom_data('trailing_min', float(current_rate))
                    trade.set_custom_data('prev_candle_min', float(current_rate))
                    logger.warning(f"[{trade.pair}] SHORT - No pivot found, using fallback SL: {fallback_sl:.4f}")

            return None

        # Update trailing max/min on each candle
        if not is_short:
            prev_candle_max = trade.get_custom_data('trailing_max', current_rate)
            new_max = max(last_candle['high'], prev_candle_max)

            trade.set_custom_data('prev_candle_max', float(prev_candle_max))
            trade.set_custom_data('trailing_max', float(new_max))
        else:
            prev_candle_min = trade.get_custom_data('trailing_min', current_rate)
            new_min = min(last_candle['low'], prev_candle_min)

            trade.set_custom_data('prev_candle_min', float(prev_candle_min))
            trade.set_custom_data('trailing_min', float(new_min))

        return None

    def custom_stoploss(self, pair: str, trade, current_time, current_rate, current_profit, **kwargs) -> float:
        """
        LuxAlgo Market Structure Trailing Stop
        ts += (max - max[1]) * increment_factor / 100
        """
        structure_level = trade.get_custom_data('structure_level', None)

        if structure_level is None:
            return self.stoploss

        is_short = getattr(trade, 'is_short', trade.trade_direction == 'short')
        increment = self.increment_factor.value / 100.0

        if not is_short:
            # LONG
            trailing_max = trade.get_custom_data('trailing_max', trade.open_rate)
            prev_candle_max = trade.get_custom_data('prev_candle_max', trade.open_rate)

            # Bar-to-bar increment: (max - max[1]) * increment / 100
            max_movement = trailing_max - prev_candle_max
            trailing_adjustment = max_movement * increment

            # Calculate new trailing stop
            prev_ts = trade.get_custom_data('trailing_stop_level', structure_level)
            new_ts = prev_ts + trailing_adjustment

            # Only move up
            new_ts = max(new_ts, structure_level)

            trade.set_custom_data('trailing_stop_level', float(new_ts))

            # Convert to stoploss percentage
            stoploss_pct = (new_ts - current_rate) / current_rate

            if trailing_adjustment > 0.01 or abs(stoploss_pct - self.stoploss) > 0.001:
                logger.info(f"[{pair}] LONG TS: {new_ts:.4f} ({stoploss_pct*100:.2f}%) | Max: {trailing_max:.4f} | Adj: +{trailing_adjustment:.4f}")

            return stoploss_pct

        else:
            # SHORT
            trailing_min = trade.get_custom_data('trailing_min', trade.open_rate)
            prev_candle_min = trade.get_custom_data('prev_candle_min', trade.open_rate)

            # Bar-to-bar increment: (min - min[1]) * increment / 100
            min_movement = trailing_min - prev_candle_min
            trailing_adjustment = min_movement * increment

            # Calculate new trailing stop
            prev_ts = trade.get_custom_data('trailing_stop_level', structure_level)
            new_ts = prev_ts + trailing_adjustment

            # Only move down
            new_ts = min(new_ts, structure_level)

            trade.set_custom_data('trailing_stop_level', float(new_ts))

            # Convert to stoploss percentage
            stoploss_pct = (current_rate - new_ts) / current_rate

            if abs(trailing_adjustment) > 0.01 or abs(stoploss_pct - self.stoploss) > 0.001:
                logger.info(f"[{pair}] SHORT TS: {new_ts:.4f} ({stoploss_pct*100:.2f}%) | Min: {trailing_min:.4f} | Adj: {trailing_adjustment:.4f}")

            return stoploss_pct
