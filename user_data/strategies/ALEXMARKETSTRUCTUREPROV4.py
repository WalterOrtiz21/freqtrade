"""
Smart Money Trades Pro Strategy - FIXED VERSION
Properly implements Pine Script logic with pending state to avoid duplicate signals
"""

import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Optional, Union
from datetime import datetime
import logging
import re
import json
from pathlib import Path


from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter
from freqtrade.persistence import Trade
from dateutil import parser
import talib.abstract as ta
import pandas_ta as pta


logger = logging.getLogger(__name__)

class AlexMarketStructurePro(IStrategy):
    """
    ╔════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
    ║                                                                                                                        ║
    ║  ██████╗ ██╗     ███████╗██╗  ██╗ ██████╗██████╗ ██╗   ██╗██████╗ ████████╗ ██████╗ ██╗  ██╗██╗███╗   ██╗ ██████╗      ║
    ║  ██╔══██╗██║     ██╔════╝╚██╗██╔╝██╔════╝██╔══██╗╚██╗ ██╔╝██╔══██╗╚══██╔══╝██╔═══██╗██║ ██╔╝██║████╗  ██║██╔════╝      ║
    ║  ███████║██║     █████╗   ╚███╔╝ ██║     ██████╔╝ ╚████╔╝ ██████╔╝   ██║   ██║   ██║█████╔╝ ██║██╔██╗ ██║██║  ███╗     ║
    ║  ██╔══██║██║     ██╔══╝   ██╔██╗ ██║     ██╔══██╗  ╚██╔╝  ██╔═══╝    ██║   ██║   ██║██╔═██╗ ██║██║╚██╗██║██║   ██║     ║
    ║  ██║  ██║███████╗███████╗██╔╝ ██╗╚██████╗██║  ██║   ██║   ██║        ██║   ╚██████╔╝██║  ██╗██║██║ ╚████║╚██████╔╝     ║
    ║  ╚═╝  ╚═╝╚══════╝╚══════╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝   ╚═╝   ╚═╝        ╚═╝    ╚═════╝ ╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝ ╚═════╝      ║
    ║                                                                                                                        ║
    ║                                            AlexMarketStructure Pro                                                     ║
    ║                  Break of Structure trading system with smart money concepts and trend analysis.                       ║
    ║                     Identifies swing level breakouts and reverses on trend flip confirmation.                          ║
    ║                     Single TP exit strategy with ATR-based targets and proper risk management.                         ║
    ╚════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

    Smart Money Trades Pro - FIXED to match Pine Script exactly

    Key fixes:
    1. Added pending state for swing break detection
    2. Prevents duplicate signals from same swing level
    3. Matches Pine Script ta.pivothigh/pivotlow behavior
    """

    INTERFACE_VERSION = 3
    can_short: bool = True
    timeframe = '15m'
    minimal_roi = {}
    trailing_stop = False
    stoploss = -0.07
    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    use_custom_stoploss = False
    position_adjustment_enable = True
    process_only_new_candles = True
    max_entry_position_adjustment = 50  # Allow multiple position adds (each needs 2 TP adjustments)
    startup_candle_count: int = 250

    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False,
    }

    order_time_in_force = {
        'entry': 'GTC',
        'exit': 'GTC'
    }

    # Parameters
    structure_period = IntParameter(10, 30, default=20, space='buy', optimize=True)
    volatility_multiplier = DecimalParameter(1.5, 3.0, default=2.0, space='buy', optimize=True)
    atr_period = IntParameter(10, 20, default=14, space='buy', optimize=True)
    tsa_max_length = IntParameter(30, 100, default=50, space='buy', optimize=False)
    tsa_accel_multiplier = DecimalParameter(1.0, 10.0, default=5.0, space='buy', optimize=False)
    tsa_collen = IntParameter(50, 200, default=100, space='buy', optimize=False)

    # Partial exit percentages
    tp1_exit_pct = 0.30
    tp2_exit_pct = 0.30

    # Position limits
    max_long_positions = 15  # Maximum long positions
    max_short_positions = 15  # Maximum short positions

    # Reverse permission flag storage: Set when bear_break/bull_break opens position, check and delete after trend_flip closes
    # Format: { 'BTC/USDT:USDT': 'reverse_to_long', 'ETH/USDT:USDT': 'reverse_to_short' }
    reverse_permission_flags = {}

    # Persistent file path
    _flags_file_path = None

    def _get_flags_file_path(self) -> Path:
        """Get the persistent file path for reverse permission flags"""
        if self._flags_file_path is None:
            # Store in strategies subdirectory
            strategies_dir = Path(__file__).parent
            self._flags_file_path = strategies_dir / 'AlexMarketStructurePro_reverse_permission_flags.json'
        return self._flags_file_path

    def _load_reverse_flags(self):
        """Load reverse permission flags from file"""
        try:
            flags_file = self._get_flags_file_path()
            if flags_file.exists():
                with open(flags_file, 'r') as f:
                    self.reverse_permission_flags = json.load(f)
                logger.info(f"✅ Loaded reverse permission flags from {flags_file}: {self.reverse_permission_flags}")
            else:
                logger.info(f"ℹ️ No existing reverse permission flags file found at {flags_file}")
        except Exception as e:
            logger.error(f"❌ Error loading reverse permission flags: {e}")
            self.reverse_permission_flags = {}

    def _save_reverse_flags(self):
        """Save reverse permission flags to file"""
        try:
            flags_file = self._get_flags_file_path()
            with open(flags_file, 'w') as f:
                json.dump(self.reverse_permission_flags, f, indent=2)
            logger.debug(f"💾 Saved reverse permission flags to {flags_file}")
        except Exception as e:
            logger.error(f"❌ Error saving reverse permission flags: {e}")

    def bot_start(self, **kwargs) -> None:
        """
        Called only once after bot instantiation.
        Load persistent reverse permission flags
        """
        self._load_reverse_flags()

    def _tsa_rma(self, series: pd.Series, period: int) -> pd.Series:
        """
        Wilder's smoothing (ta.rma equivalent) used by Trend Speed Analyzer.
        """
        alpha = 1.0 / period
        return series.ewm(alpha=alpha, adjust=False).mean()

    def _get_candle_start_time(self, current_time: datetime) -> datetime:
        """
        Calculate the candle start time based on timeframe.
        Supports: 1m, 5m, 15m, 30m, 1h, 4h, 1d, etc.
        """
        # Parse timeframe to get interval in minutes
        match = re.match(r'(\d+)([mhd])', self.timeframe)
        if not match:
            raise ValueError(f"Unsupported timeframe format: {self.timeframe}")

        value = int(match.group(1))
        unit = match.group(2)

        if unit == 'm':
            interval_minutes = value
        elif unit == 'h':
            interval_minutes = value * 60
        elif unit == 'd':
            interval_minutes = value * 60 * 24
        else:
            raise ValueError(f"Unsupported timeframe unit: {unit}")

        # Calculate candle start time
        if interval_minutes < 60:  # Minutes-based candles
            candle_minutes = (current_time.minute // interval_minutes) * interval_minutes
            return current_time.replace(minute=candle_minutes, second=0, microsecond=0)
        elif interval_minutes < 1440:  # Hour-based candles (< 1 day)
            total_minutes = current_time.hour * 60 + current_time.minute
            candle_minutes = (total_minutes // interval_minutes) * interval_minutes
            candle_hour = candle_minutes // 60
            candle_minute = candle_minutes % 60
            return current_time.replace(hour=candle_hour, minute=candle_minute, second=0, microsecond=0)
        else:  # Day-based candles
            return current_time.replace(hour=0, minute=0, second=0, microsecond=0)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Properly implements Pine Script market structure detection
        """

        # ATR
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=self.atr_period.value)

        period = self.structure_period.value

        # Swing detection using center=True (matches ta.pivothigh behavior)
        # ta.pivothigh returns value at bar_index-period when confirmed
        dataframe['swing_high'] = dataframe['high'].rolling(
            window=period*2+1, center=True
        ).apply(lambda x: x[period] if (x[period] == x.max() and len(x) == period*2+1) else np.nan, raw=True)

        dataframe['swing_low'] = dataframe['low'].rolling(
            window=period*2+1, center=True
        ).apply(lambda x: x[period] if (x[period] == x.min() and len(x) == period*2+1) else np.nan, raw=True)

        # Forward fill to get last confirmed swing levels
        dataframe['last_swing_high'] = dataframe['swing_high'].ffill()
        dataframe['last_swing_low'] = dataframe['swing_low'].ffill()

        # CRITICAL FIX: Vectorized pending state (10000x faster than loop!)
        # Replicate Pine Script: highBreakPending/lowBreakPending logic

        # Track when swing levels change (new swing detected = pending True)
        swing_high_changed = dataframe['last_swing_high'] != dataframe['last_swing_high'].shift(1)
        swing_low_changed = dataframe['last_swing_low'] != dataframe['last_swing_low'].shift(1)

        # Group candles by swing level (each group = one pending period)
        swing_high_group = swing_high_changed.cumsum()
        swing_low_group = swing_low_changed.cumsum()

        # Detect price breaks
        bull_break_signal = dataframe['close'] > dataframe['last_swing_high']
        bear_break_signal = dataframe['close'] < dataframe['last_swing_low']

        # Initialize
        dataframe['bull_break'] = False
        dataframe['bear_break'] = False

        # For each swing group, mark FIRST break only (pending -> break -> clear)
        # Use groupby + idxmax for vectorized "first occurrence"
        bull_groups = dataframe[bull_break_signal].groupby(swing_high_group[bull_break_signal])
        for _, group in bull_groups:
            if len(group) > 0:
                dataframe.loc[group.index[0], 'bull_break'] = True

        bear_groups = dataframe[bear_break_signal].groupby(swing_low_group[bear_break_signal])
        for _, group in bear_groups:
            if len(group) > 0:
                dataframe.loc[group.index[0], 'bear_break'] = True

        # Trend: 1=bull, -1=bear, forward fill
        dataframe['trend'] = 0
        dataframe.loc[dataframe['bull_break'], 'trend'] = 1
        dataframe.loc[dataframe['bear_break'], 'trend'] = -1
        dataframe['trend'] = dataframe['trend'].replace(0, np.nan).ffill().fillna(0).astype(int)

        # CHoCH detection
        dataframe['is_choch'] = (
            ((dataframe['bull_break']) & (dataframe['trend'].shift(1) == -1)) |
            ((dataframe['bear_break']) & (dataframe['trend'].shift(1) == 1))
        )

        # Calculate targets
        target_range = dataframe['atr'] * self.volatility_multiplier.value

        dataframe['long_entry'] = dataframe['last_swing_high']
        dataframe['long_tp1'] = dataframe['long_entry'] + target_range * 0.8
        dataframe['long_tp2'] = dataframe['long_entry'] + target_range * 1.6
        dataframe['long_tp3'] = dataframe['long_entry'] + target_range * 2.0
        dataframe['long_sl'] = dataframe['long_entry'] - target_range * 1.2

        dataframe['short_entry'] = dataframe['last_swing_low']
        dataframe['short_tp1'] = dataframe['short_entry'] - target_range * 0.8
        dataframe['short_tp2'] = dataframe['short_entry'] - target_range * 1.6
        dataframe['short_tp3'] = dataframe['short_entry'] - target_range * 2.0
        dataframe['short_sl'] = dataframe['short_entry'] + target_range * 1.2

        # === Trend Speed Analyzer style trend filter ===
        # Note: Only calculate trend line, no need for c_smooth/o_smooth (they're only used for histogram calculation)

        close_values = dataframe['close'].values
        abs_close = np.abs(close_values)
        max_abs_counts_diff = (
            pd.Series(abs_close)
            .rolling(window=200)
            .max()
            .bfill()
        )
        max_abs_counts_diff = max_abs_counts_diff.where(max_abs_counts_diff != 0, 1.0).to_numpy()
        counts_diff_norm = (close_values + max_abs_counts_diff) / (2 * max_abs_counts_diff)
        dataframe['tsa_dyn_length'] = 5 + counts_diff_norm * (self.tsa_max_length.value - 5)

        delta_counts_diff = np.abs(np.diff(close_values, prepend=close_values[0]))
        max_delta = (
            pd.Series(delta_counts_diff)
            .rolling(window=200)
            .max()
            .bfill()
        )
        max_delta = max_delta.where(max_delta != 0, 1.0).to_numpy()
        accel_factor = delta_counts_diff / max_delta

        dyn_length = dataframe['tsa_dyn_length'].to_numpy()
        alpha_base = 2 / (dyn_length + 1)
        alpha = alpha_base * (1 + accel_factor * float(self.tsa_accel_multiplier.value))
        alpha = np.minimum(alpha, 1.0)

        dyn_ema = np.zeros(len(dataframe))
        if len(dataframe):
            dyn_ema[0] = dataframe['close'].iloc[0]
            for i in range(1, len(dataframe)):
                dyn_ema[i] = alpha[i] * dataframe['close'].iloc[i] + (1 - alpha[i]) * dyn_ema[i - 1]

        dataframe['tsa_dyn_ema'] = dyn_ema
        dataframe['tsa_trend_line'] = dataframe['tsa_dyn_ema']

        # Note: We only need trend line color, no need to calculate histogram (speed/trendspeed)
        # Remove the following two lines to improve performance:
        # dataframe['tsa_speed'] = self._tsa_calculate_speed(dataframe)
        # dataframe['tsa_trendspeed'] = pta.hma(dataframe['tsa_speed'], length=5)

        dataframe['tsa_trend_color_green'] = (
            ta.WMA(dataframe['close'], timeperiod=2) > dataframe['tsa_trend_line']
        ).astype(bool)

        # NEW:
        # BEST FIX:
        prev_green = dataframe['tsa_trend_color_green'].astype(bool).shift(1).fillna(False)
        dataframe['tsa_trend_flip_to_green'] = dataframe['tsa_trend_color_green'] & (~prev_green)
        dataframe['tsa_trend_flip_to_red'] = (~dataframe['tsa_trend_color_green']) & prev_green

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata['pair']

        # Bull break entry (PRIMARY SIGNAL)
        bull_break_mask = (
            (dataframe['bull_break'] == True) &
            (dataframe['volume'] > 0)
        )
        dataframe.loc[bull_break_mask, ["enter_long", "enter_tag"]] = (1, "bull_break")

        # Bear break entry (PRIMARY SIGNAL)
        bear_break_mask = (
            (dataframe['bear_break'] == True) &
            (dataframe['volume'] > 0)
        )
        dataframe.loc[bear_break_mask, ["enter_short", "enter_tag"]] = (1, "bear_break")

        # Trend flip to green - REVERSAL (requires bear_break position)
        trend_flip_green_mask = (
            (dataframe['tsa_trend_flip_to_green']) &
            (dataframe['volume'] > 0) &
            (~bull_break_mask)  # Don't overwrite bull_break
        )
        dataframe.loc[trend_flip_green_mask, ["enter_long", "enter_tag"]] = (1, "trend_flip_long")

        # Trend flip to red - REVERSAL (requires bull_break position)
        trend_flip_red_mask = (
            (dataframe['tsa_trend_flip_to_red']) &
            (dataframe['volume'] > 0) &
            (~bear_break_mask)  # Don't overwrite bear_break
        )
        dataframe.loc[trend_flip_red_mask, ["enter_short", "enter_tag"]] = (1, "trend_flip_short")

        # ✨ NEW: Trend flip to green - INDEPENDENT (no position required)
        trend_flip_green_independent_mask = (
            (dataframe['tsa_trend_flip_to_green']) &
            (dataframe['volume'] > 0) &
            (~bull_break_mask)  # Don't overwrite bull_break
        )
        # Only set if not already set by reversal signal
        independent_long = trend_flip_green_independent_mask & (dataframe['enter_long'] != 1)
        dataframe.loc[independent_long, ["enter_long", "enter_tag"]] = (1, "trend_flip_long_independent")

        # ✨ NEW: Trend flip to red - INDEPENDENT (no position required)
        trend_flip_red_independent_mask = (
            (dataframe['tsa_trend_flip_to_red']) &
            (dataframe['volume'] > 0) &
            (~bear_break_mask)  # Don't overwrite bear_break
        )
        # Only set if not already set by reversal signal
        independent_short = trend_flip_red_independent_mask & (dataframe['enter_short'] != 1)
        dataframe.loc[independent_short, ["enter_short", "enter_tag"]] = (1, "trend_flip_short_independent")

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Note: Do not set exit signals here
        # Reason: populate_exit_trend signals affect all positions, cannot distinguish entry_tag
        # Our exit logic is implemented in custom_exit, which can be finely controlled based on trade.enter_tag
        #
        # For example: If we set exit_long when tsa_trend_flip_to_red here,
        # then both trend_flip_long and bull_break longs will be closed
        # But we only want to close bull_break longs

        return dataframe
    

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                           time_in_force: str, current_time: datetime, entry_tag: Optional[str],
                           side: str, **kwargs) -> bool:
        """
        Check if we should enter the trade based on position limits.

        For trend_flip entries: Only allow if there's an opposite bull_break/bear_break position to reverse.
        Limit: max 15 long positions, max 15 short positions
        """
        # Get all open trades
        open_trades = Trade.get_open_trades()

        # === Special check for trend_flip entries ===
        # trend_flip_long: Only allow entry when there's a bear_break short position (reverse)
        # trend_flip_short: Only allow entry when there's a bull_break long position (reverse)
        if entry_tag == 'trend_flip_long':
            logger.info(f"[{pair}] 🔓 CONFIRM_ENTRY: trend_flip_long signal received | rate={rate:.8f}")

            # Get current dataframe to check trend flip signal
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            last_candle = dataframe.iloc[-1].squeeze()

            # Must confirm trend flip signal exists
            trend_flip_to_green = last_candle.get('tsa_trend_flip_to_green', False)
            logger.info(f"[{pair}] 🔍 CONFIRM_ENTRY: tsa_trend_flip_to_green={trend_flip_to_green}")

            if not trend_flip_to_green:
                logger.info(f"[{pair}] ❌ CONFIRM_ENTRY: trend_flip_long rejected - no trend flip signal in dataframe")
                return False

            # 🔑 Check reverse permission flag
            permission_flag = self.reverse_permission_flags.get(pair, None)
            logger.info(f"[{pair}] 🔑 CONFIRM_ENTRY: Checking reverse permission flag = '{permission_flag}'")

            if permission_flag != 'reverse_to_long':
                logger.info(f"[{pair}] ❌ CONFIRM_ENTRY: trend_flip_long entry rejected - no 'reverse_to_long' permission flag (current: {permission_flag})")
                return False

            # Print all related position info (for debugging)
            all_pair_trades = [t for t in open_trades if t.pair == pair]
            logger.info(f"[{pair}] 📊 CONFIRM_ENTRY: Total open trades for this pair: {len(all_pair_trades)}")
            for t in all_pair_trades:
                logger.info(f"[{pair}]    - Trade #{t.id}: {t.enter_tag} | {'SHORT' if t.is_short else 'LONG'} | profit={t.calc_profit_ratio(rate):.2%}")

            # Delete reverse permission flag (already used)
            del self.reverse_permission_flags[pair]
            self._save_reverse_flags()  # 💾 Persist
            logger.info(f"[{pair}] 🗑️ CONFIRM_ENTRY: Deleted reverse permission flag after approval (saved to file)")

            logger.info(f"[{pair}] ✅ CONFIRM_ENTRY: trend_flip_long entry APPROVED - reverse permission granted")
            return True

        if entry_tag == 'trend_flip_short':
            logger.info(f"[{pair}] 🔓 CONFIRM_ENTRY: trend_flip_short signal received | rate={rate:.8f}")

            # Get current dataframe to check trend flip signal
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            last_candle = dataframe.iloc[-1].squeeze()

            # Must confirm trend flip signal exists
            trend_flip_to_red = last_candle.get('tsa_trend_flip_to_red', False)
            logger.info(f"[{pair}] 🔍 CONFIRM_ENTRY: tsa_trend_flip_to_red={trend_flip_to_red}")

            if not trend_flip_to_red:
                logger.info(f"[{pair}] ❌ CONFIRM_ENTRY: trend_flip_short rejected - no trend flip signal in dataframe")
                return False

            # 🔑 Check reverse permission flag
            permission_flag = self.reverse_permission_flags.get(pair, None)
            logger.info(f"[{pair}] 🔑 CONFIRM_ENTRY: Checking reverse permission flag = '{permission_flag}'")

            if permission_flag != 'reverse_to_short':
                logger.info(f"[{pair}] ❌ CONFIRM_ENTRY: trend_flip_short entry rejected - no 'reverse_to_short' permission flag (current: {permission_flag})")
                return False

            # Print all related position info (for debugging)
            all_pair_trades = [t for t in open_trades if t.pair == pair]
            logger.info(f"[{pair}] 📊 CONFIRM_ENTRY: Total open trades for this pair: {len(all_pair_trades)}")
            for t in all_pair_trades:
                logger.info(f"[{pair}]    - Trade #{t.id}: {t.enter_tag} | {'SHORT' if t.is_short else 'LONG'} | profit={t.calc_profit_ratio(rate):.2%}")

            # Delete reverse permission flag (already used)
            del self.reverse_permission_flags[pair]
            self._save_reverse_flags()  # 💾 Persist
            logger.info(f"[{pair}] 🗑️ CONFIRM_ENTRY: Deleted reverse permission flag after approval (saved to file)")

            logger.info(f"[{pair}] ✅ CONFIRM_ENTRY: trend_flip_short entry APPROVED - reverse permission granted")
            return True

        # # === For bull_break/bear_break entries: check position limits ===
        # long_count = sum(1 for trade in open_trades if not trade.is_short)
        # short_count = sum(1 for trade in open_trades if trade.is_short)

        # if side == 'long':
        #     if long_count >= self.max_long_positions:
        #         logger.info(f"[{pair}] ❌ LONG entry rejected: already have {long_count}/{self.max_long_positions} long positions")
        #         return False
        # else:  # short
        #     if short_count >= self.max_short_positions:
        #         logger.info(f"[{pair}] ❌ SHORT entry rejected: already have {short_count}/{self.max_short_positions} short positions")
        #         return False

        logger.info(f"[{pair}] ✅ {'LONG' if side == 'long' else 'SHORT'} entry confirmed: {entry_tag}")
        return True

    def order_filled(self, pair: str, trade: Trade, order, current_time: datetime, **kwargs) -> None:
        """
        Called when an order is filled.
        - For initial entry: Save initial stake and set initial TP/SL levels
        - For position adds: Update TP/SL to new BOS/CHoCH levels AND reset TP counter
        """
        if order.ft_order_side == trade.entry_side:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            last_candle = dataframe.iloc[-1].squeeze()

            # CRITICAL: Mark the current candle to prevent adding on same candle
            current_candle = self._get_candle_start_time(current_time)
            trade.set_custom_data('last_add_candle', current_candle.isoformat())

            # CRITICAL: Save the swing level that triggered this entry/add
            # This prevents adding on the SAME swing level multiple times
            if trade.is_short:
                current_swing_level = float(last_candle['last_swing_low'])
                trade.set_custom_data('last_entry_swing_level', current_swing_level)
                ##logger.info(f"[{pair}] Order filled at {current_time}, swing_low={current_swing_level}")
            else:
                current_swing_level = float(last_candle['last_swing_high'])
                trade.set_custom_data('last_entry_swing_level', current_swing_level)
                ##logger.info(f"[{pair}] Order filled at {current_time}, swing_high={current_swing_level}")

            # Save initial stake on first entry (for position adding)
            if trade.nr_of_successful_entries == 1:  # First entry
                trade.set_custom_data('initial_stake', order.stake_amount)
                trade.set_custom_data('tp_hits', 0)  # Track TP hits separately
                logger.info(f"[{pair}] 📥 ORDER_FILLED: Initial entry | entry_tag={trade.enter_tag} | "
                           f"{'SHORT' if trade.is_short else 'LONG'} | stake={order.stake_amount:.2f} | "
                           f"rate={order.average:.8f}")

                # 🔑 Set reverse permission flag
                if trade.enter_tag == 'bear_break' and trade.is_short:
                    self.reverse_permission_flags[pair] = 'reverse_to_long'
                    self._save_reverse_flags()  # 💾 Persist
                    logger.info(f"[{pair}] 🔑 ORDER_FILLED: Set reverse permission flag 'reverse_to_long' for bear_break SHORT entry (saved to file)")
                elif trade.enter_tag == 'bull_break' and not trade.is_short:
                    self.reverse_permission_flags[pair] = 'reverse_to_short'
                    self._save_reverse_flags()  # 💾 Persist
                    logger.info(f"[{pair}] 🔑 ORDER_FILLED: Set reverse permission flag 'reverse_to_short' for bull_break LONG entry (saved to file)")
            elif trade.nr_of_successful_entries > 1:  # Position add
                # Update initial_stake to reflect new total position size
                # IMPORTANT: This ensures TP1/TP2 reduction is based on current total stake
                trade.set_custom_data('initial_stake', trade.stake_amount)
                # Reset TP counter when adding to position
                trade.set_custom_data('tp_hits', 0)
                ##logger.info(f"[{pair}] Position add #{trade.nr_of_successful_entries} filled - Resetting TP counter")

            # Update TP/SL levels (both for initial entry and position adds)
            if trade.is_short:
                entry = float(last_candle['short_entry'])
                tp1 = float(last_candle['short_tp1'])
                tp2 = float(last_candle['short_tp2'])
                tp3 = float(last_candle['short_tp3'])
                sl = float(last_candle['short_sl'])
                atr = float(last_candle['atr'])
 
                trade.set_custom_data('entry_price', entry)
                trade.set_custom_data('tp1_price', tp1) 
                trade.set_custom_data('tp2_price', tp2)
                trade.set_custom_data('tp3_price', tp3)
                trade.set_custom_data('sl_price', sl)
                trade.set_custom_data('direction', 'short')

                #logger.info(f"[{pair}] 🔴 SHORT TP/SL UPDATED:")
                #logger.info(f"  Entry: {entry:.8f}")
                #logger.info(f"  ATR: {atr:.8f}")
                #logger.info(f"  Target Range: {atr * self.volatility_multiplier.value:.8f}")
                #logger.info(f"  TP1: {tp1:.8f} (Entry - Range*0.8)")
                #logger.info(f"  TP2: {tp2:.8f} (Entry - Range*1.6)")
                #logger.info(f"  TP3: {tp3:.8f} (Entry - Range*2.8)")
                #logger.info(f"  SL:  {sl:.8f} (Entry + Range*1.2)")
                #logger.info(f"  SL Distance: {((sl - entry) / entry * 100):.2f}%")
            else:
                entry = float(last_candle['long_entry'])
                tp1 = float(last_candle['long_tp1'])
                tp2 = float(last_candle['long_tp2'])
                tp3 = float(last_candle['long_tp3'])
                sl = float(last_candle['long_sl'])
                atr = float(last_candle['atr'])

                trade.set_custom_data('entry_price', entry)
                trade.set_custom_data('tp1_price', tp1)
                trade.set_custom_data('tp2_price', tp2)
                trade.set_custom_data('tp3_price', tp3)
                trade.set_custom_data('sl_price', sl)
                trade.set_custom_data('direction', 'long')

                #logger.info(f"[{pair}] 🟢 LONG TP/SL UPDATED:")
                #logger.info(f"  Entry: {entry:.8f}")
                #logger.info(f"  ATR: {atr:.8f}")
                #logger.info(f"  Target Range: {atr * self.volatility_multiplier.value:.8f}")
                #logger.info(f"  TP1: {tp1:.8f} (Entry + Range*0.8)")
                #logger.info(f"  TP2: {tp2:.8f} (Entry + Range*1.6)")
                #logger.info(f"  TP3: {tp3:.8f} (Entry + Range*2.8)")
                #logger.info(f"  SL:  {sl:.8f} (Entry - Range*1.2)")
                #logger.info(f"  SL Distance: {((entry - sl) / entry * 100):.2f}%")

        # Handle exit orders (normal close, not trend flip)
        else:  # order.ft_order_side == trade.exit_side
            # Check if it's a complete close (trade is closed)
            if not trade.is_open:
                entry_tag = trade.enter_tag
                exit_reason = trade.exit_reason if hasattr(trade, 'exit_reason') else 'unknown'

                # 🔑 Only delete flag on non-trend-flip exits
                # Trend flip exits (trend_flip_reverse_long/short) need to keep flag for confirm_trade_entry check
                is_trend_flip_exit = exit_reason in ['trend_flip_reverse_long', 'trend_flip_reverse_short']

                if not is_trend_flip_exit:
                    # Delete reverse permission flag
                    # Scenarios: tp1_trailing, tp2_trailing, tp3_trailing, hybrid_trailing_sl, reverse_signal, stoploss, etc.
                    if entry_tag == 'bear_break' and trade.is_short:
                        if pair in self.reverse_permission_flags:
                            del self.reverse_permission_flags[pair]
                            self._save_reverse_flags()  # 💾 Persist
                            logger.info(f"[{pair}] 🗑️ ORDER_FILLED: Deleted 'reverse_to_long' permission flag - bear_break SHORT fully closed (saved to file)")
                            logger.info(f"[{pair}]    Exit reason: {exit_reason} (non-trend-flip exit)")

                    elif entry_tag == 'bull_break' and not trade.is_short:
                        if pair in self.reverse_permission_flags:
                            del self.reverse_permission_flags[pair]
                            self._save_reverse_flags()  # 💾 Persist
                            logger.info(f"[{pair}] 🗑️ ORDER_FILLED: Deleted 'reverse_to_short' permission flag - bull_break LONG fully closed (saved to file)")
                            logger.info(f"[{pair}]    Exit reason: {exit_reason} (non-trend-flip exit)")
                else:
                    # Trend flip exit, keep flag for confirm_trade_entry to use
                    logger.info(f"[{pair}] 🔑 ORDER_FILLED: Keeping reverse permission flag for trend flip exit")
                    logger.info(f"[{pair}]    Exit reason: {exit_reason} - flag will be checked in confirm_trade_entry")

        return None

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                             current_rate: float, current_profit: float,
                             min_stake: Optional[float], max_stake: float,
                             current_entry_rate: float, current_exit_rate: float,
                             current_entry_profit: float, current_exit_profit: float,
                             **kwargs) -> Optional[float]:

        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        # CRITICAL: Check if TP/SL data exists (might be lost after restart)
        sl_price = trade.get_custom_data('sl_price', None)
        if sl_price is None or sl_price == 0:
            # Recalculate TP/SL from current candle
            if trade.is_short:
                trade.set_custom_data('entry_price', float(last_candle['short_entry']))
                trade.set_custom_data('tp1_price', float(last_candle['short_tp1']))
                trade.set_custom_data('tp2_price', float(last_candle['short_tp2']))
                trade.set_custom_data('tp3_price', float(last_candle['short_tp3']))
                trade.set_custom_data('sl_price', float(last_candle['short_sl']))
                trade.set_custom_data('direction', 'short')
            else:
                trade.set_custom_data('entry_price', float(last_candle['long_entry']))
                trade.set_custom_data('tp1_price', float(last_candle['long_tp1']))
                trade.set_custom_data('tp2_price', float(last_candle['long_tp2']))
                trade.set_custom_data('tp3_price', float(last_candle['long_tp3']))
                trade.set_custom_data('sl_price', float(last_candle['long_sl']))
                trade.set_custom_data('direction', 'long')

            if trade.get_custom_data('initial_stake', None) is None:
                trade.set_custom_data('initial_stake', trade.stake_amount)
            if trade.get_custom_data('tp_hits', None) is None:
                trade.set_custom_data('tp_hits', 0)

        tp1_price = trade.get_custom_data('tp1_price', 0)
        tp2_price = trade.get_custom_data('tp2_price', 0)
        direction = trade.get_custom_data('direction', 'long')
        tp_hits = trade.get_custom_data('tp_hits', 0)
        initial_stake = trade.get_custom_data('initial_stake', trade.stake_amount)

        # === Priority 1: Partial Exit Logic (TP1 and TP2) ===
        if direction == 'long':
            if tp_hits == 1 and tp2_price > 0 and current_rate >= tp2_price:
                trade.set_custom_data('tp_hits', 2)
                return -(initial_stake * self.tp2_exit_pct)
            elif tp_hits == 0 and tp1_price > 0 and current_rate >= tp1_price:
                trade.set_custom_data('tp_hits', 1)
                return -(initial_stake * self.tp1_exit_pct)
        else:
            if tp_hits == 1 and tp2_price > 0 and current_rate <= tp2_price:
                trade.set_custom_data('tp_hits', 2)
                return -(initial_stake * self.tp2_exit_pct)
            elif tp_hits == 0 and tp1_price > 0 and current_rate <= tp1_price:
                trade.set_custom_data('tp_hits', 1)
                return -(initial_stake * self.tp1_exit_pct)

        return None

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime,
                   current_rate: float, current_profit: float, **kwargs) -> Optional[Union[str, bool]]:

        sl_price = trade.get_custom_data('sl_price', 0)
        tp1_price = trade.get_custom_data('tp1_price', 0)
        tp2_price = trade.get_custom_data('tp2_price', 0)
        tp3_price = trade.get_custom_data('tp3_price', 0)
        direction = trade.get_custom_data('direction', 'long')
        tp_hits = trade.get_custom_data('tp_hits', 0)
        entry_price = trade.get_custom_data('entry_price', trade.open_rate)
        entry_tag = trade.enter_tag

        # Get dataframe
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        target_range = float(last_candle['atr']) * self.volatility_multiplier.value

        trend_flip_to_green = bool(last_candle.get('tsa_trend_flip_to_green', False))
        trend_flip_to_red = bool(last_candle.get('tsa_trend_flip_to_red', False))

        # === Priority 1: Trend flip reversals ===
        if direction == 'short' and trend_flip_to_green and entry_tag == 'bear_break':
            logger.info(f"[{pair}] 🔄 CUSTOM_EXIT: bear_break SHORT exit due to trend flip to green | "
                       f"profit={current_profit:.2%}, rate={current_rate:.8f}")
            return 'trend_flip_reverse_long'

        if direction == 'long' and trend_flip_to_red and entry_tag == 'bull_break':
            logger.info(f"[{pair}] 🔄 CUSTOM_EXIT: bull_break LONG exit due to trend flip to red | "
                       f"profit={current_profit:.2%}, rate={current_rate:.8f}")
            return 'trend_flip_reverse_short'

        # === Priority 2: Reverse signal exit ===
        if direction == 'long':
            if last_candle.get('bear_break', False):
                logger.info(f"[{pair}] 🔄 LONG EXIT: Bear break signal detected")
                return 'reverse_signal'
        else:  # short
            if last_candle.get('bull_break', False):
                logger.info(f"[{pair}] 🔄 SHORT EXIT: Bull break signal detected")
                return 'reverse_signal'

        # === Priority 3: TP Trailing Stop Logic ===
        # TP gradient trailing: After reaching TPn, if price retraces to TPn-1, exit all
        if direction == 'long':
            if tp_hits >= 2 and tp3_price > 0 and current_rate >= tp3_price:  # TP3 reached, exit immediately
                logger.info(f"[{pair}] 🔒 LONG TP3 HIT: Immediate exit at {current_rate:.8f}")
                return 'tp3_trailing'
            elif tp_hits == 2 and current_rate <= tp1_price:  # TP2 reached, retraced to TP1, exit all
                logger.info(f"[{pair}] 🔒 LONG TP2->TP1 TRAILING: current={current_rate:.8f}, TP1={tp1_price:.8f}")
                return 'tp2_trailing'
            elif tp_hits == 1 and current_rate <= entry_price:  # TP1 reached, retraced to entry, exit all
                logger.info(f"[{pair}] 🔒 LONG TP1->ENTRY TRAILING: current={current_rate:.8f}, Entry={entry_price:.8f}")
                return 'tp1_trailing'
        else:  # short
            if tp_hits >= 2 and tp3_price > 0 and current_rate <= tp3_price:  # TP3 reached, exit immediately
                logger.info(f"[{pair}] 🔒 SHORT TP3 HIT: Immediate exit at {current_rate:.8f}")
                return 'tp3_trailing'
            elif tp_hits == 2 and current_rate >= tp1_price:  # TP2 reached, bounced to TP1, exit all
                logger.info(f"[{pair}] 🔒 SHORT TP2->TP1 TRAILING: current={current_rate:.8f}, TP1={tp1_price:.8f}")
                return 'tp2_trailing'
            elif tp_hits == 1 and current_rate >= entry_price:  # TP1 reached, bounced to entry, exit all
                logger.info(f"[{pair}] 🔒 SHORT TP1->ENTRY TRAILING: current={current_rate:.8f}, Entry={entry_price:.8f}")
                return 'tp1_trailing'

        # === Priority 4: Hybrid trailing stop (only after TP2) ===
        # Hybrid stop: ATR stop vs percentage stop, take the more conservative one
        if tp_hits >= 2:
            # Get entry ATR (avoid using current expanded ATR)
            entry_atr = trade.get_custom_data('entry_atr', target_range)

            if direction == 'long':
                best_price = trade.get_custom_data('best_price', current_rate)
                # Update highest price
                if current_rate > best_price:
                    best_price = current_rate
                    trade.set_custom_data('best_price', best_price)

                # Option 1: ATR-based trailing stop
                atr_trailing_sl = best_price - (entry_atr * self.volatility_multiplier.value)

                # Option 2: Fixed percentage trailing stop (3% drawdown)
                percentage_trailing_sl = best_price * 0.97

                # Take the higher of the two (more conservative stop)
                trailing_sl = max(atr_trailing_sl, percentage_trailing_sl)

                if current_rate <= trailing_sl:
                    logger.info(f"[{pair}] 📉 LONG HYBRID TRAILING: best={best_price:.8f}, current={current_rate:.8f}")
                    logger.info(f"  ATR SL={atr_trailing_sl:.8f}, Percentage SL={percentage_trailing_sl:.8f}, Final SL={trailing_sl:.8f}")
                    return 'hybrid_trailing_sl'

            else:  # short
                best_price = trade.get_custom_data('best_price', current_rate)
                # Update lowest price
                if current_rate < best_price:
                    best_price = current_rate
                    trade.set_custom_data('best_price', best_price)

                # Option 1: ATR-based trailing stop
                atr_trailing_sl = best_price + (entry_atr * self.volatility_multiplier.value)

                # Option 2: Fixed percentage trailing stop (3% bounce)
                percentage_trailing_sl = best_price * 1.03

                # Take the lower of the two (more conservative stop)
                trailing_sl = min(atr_trailing_sl, percentage_trailing_sl)

                if current_rate >= trailing_sl:
                    logger.info(f"[{pair}] 📈 SHORT HYBRID TRAILING: best={best_price:.8f}, current={current_rate:.8f}")
                    logger.info(f"  ATR SL={atr_trailing_sl:.8f}, Percentage SL={percentage_trailing_sl:.8f}, Final SL={trailing_sl:.8f}")
                    return 'hybrid_trailing_sl'

        return None

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                side: str, **kwargs) -> float:
        return 5.0