"""
CHoCH/BOS Hyperopt Strategy - Advanced Multi-Filter Exit Strategy

Extended version of CHoCHBOSStrategy with additional entry filters and exit modes
for comprehensive hyperparameter optimization.

Entry Filters (Hyperoptable):
- HMA 200: Optional Hull Moving Average filter (separate for long/short)
- Smooth Trail: Optional SuperTrend-based momentum filter (separate for long/short)

Exit Modes (Hyperoptable):
- BOS Counting: Dynamic TP based on BOS count after entry (like base strategy)
- Trailing Only: Pure trailing stop without BOS TPs
- CHoCH Exit: Exit only on opposite CHoCH signal
- MST Projection: Static price target projection from structure

Risk Management:
- Stop Loss: Hyperoptable from -1% to -3%
- Trailing Stop: Configurable activation and distance
- Break Even: Optional after TP1 in BOS mode

Author: Based on CHoCHBOSStrategy with extended hyperopt capabilities
Recommended: Run hyperopt on 3+ months data to find optimal filter combinations
"""

import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from pandas import DataFrame
import pandas_ta as pta

from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter, CategoricalParameter

logger = logging.getLogger(__name__)


class CHoCHBOSHyperopt(IStrategy):
    """
    CHoCH/BOS Strategy with extensive hyperoptimization parameters
    """

    # Strategy configuration
    # ROI disabled - exit only on CHoCH + BOS
    minimal_roi = {
        "0": 100  # Effectively disabled (100% target)
    }

    # Hyperoptable stop loss (safety net only)
    stoploss = -0.03

    # Trailing stop DISABLED - exit only on CHoCH + BOS
    trailing_stop = False
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = True

    # Operational settings
    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    can_short = True
    position_adjustment_enable = True  # Enable BOS tracking (not used for partial exits)
    use_custom_stoploss = True  # LuxAlgo Market Structure Trailing Stop

    # Startup candles needed
    startup_candle_count: int = 200  # Increased for HMA 200

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                 side: str, **kwargs) -> float:
        """
        Set leverage to 10x for all pairs (matching production config)
        """
        return 10.0

    # ====================================================================================
    # HYPEROPTABLE PARAMETERS - STRUCTURE DETECTION
    # Using Pine Script default values (optimize=False for fixed parameters)
    # ====================================================================================

    zigzag_depth = IntParameter(
        low=20,
        high=45,
        default=23,  # Pine Script default for CHoCH/BOS detection
        space="buy",
        optimize=False,  # Use fixed value
        load=True
    )

    # ====================================================================================
    # HYPEROPTABLE PARAMETERS - ENTRY FILTERS
    # ====================================================================================

    # HMA 200 Filter (disabled - no entry filters)
    use_hma_filter_long = CategoricalParameter(
        [True, False],
        default=False,  # Disabled - pure CHoCH + BOS
        space="buy",
        optimize=False,  # Use fixed value
        load=True
    )

    use_hma_filter_short = CategoricalParameter(
        [True, False],
        default=False,  # Disabled - pure CHoCH + BOS
        space="buy",
        optimize=False,  # Use fixed value
        load=True
    )

    hma_length = IntParameter(
        low=100,
        high=300,
        default=200,  # Pine Script default
        space="buy",
        optimize=False,  # Use fixed value
        load=True
    )

    # Smooth Trail Filter (disabled - no entry filters)
    use_smooth_trail_long = CategoricalParameter(
        [True, False],
        default=False,  # Disabled - pure CHoCH + BOS
        space="buy",
        optimize=False,  # Use fixed value
        load=True
    )

    use_smooth_trail_short = CategoricalParameter(
        [True, False],
        default=False,  # Disabled - pure CHoCH + BOS
        space="buy",
        optimize=False,  # Use fixed value
        load=True
    )

    smooth_trail_multiplier = DecimalParameter(
        low=0.5,
        high=3.0,
        default=1.0,  # Pine Script default
        decimals=1,
        space="buy",
        optimize=False,  # Use fixed value
        load=True
    )

    smooth_trail_length = IntParameter(
        low=20,
        high=50,
        default=34,  # Pine Script default
        space="buy",
        optimize=False,  # Use fixed value
        load=True
    )

    # ====================================================================================
    # HYPEROPTABLE PARAMETERS - EXIT STRATEGY
    # ====================================================================================

    exit_mode = CategoricalParameter(
        ["bos_counting", "trailing_only", "choch_only", "mst_projection"],
        default="trailing_only",  # No partial exits, only exit_signal (CHoCH + BOS)
        space="sell",
        optimize=False,  # Use fixed value
        load=True
    )

    # Exit confirmation - matches Pine's exit_on_bos_opposite parameter
    require_bos_exit = CategoricalParameter(
        [True, False],
        default=True,  # Pine Script default (exit_on_bos_opposite = true)
        space="sell",
        optimize=False,  # Use fixed value
        load=True
    )

    # BOS Counting Mode Parameters
    tp1_percentage = DecimalParameter(
        low=0.3,
        high=0.7,
        default=0.5,  # Pine Script default (mst_tp1_percentage = 50.0)
        decimals=1,
        space="sell",
        optimize=False,  # Use fixed value
        load=True
    )

    tp2_percentage = DecimalParameter(
        low=0.2,
        high=0.5,
        default=0.3,  # Pine Script default (mst_tp2_percentage = 30.0)
        decimals=1,
        space="sell",
        optimize=False,  # Use fixed value
        load=True
    )

    move_to_be_after_tp1 = CategoricalParameter(
        [True, False],
        default=True,  # Pine Script default (mst_move_to_be = true)
        space="sell",
        optimize=False,  # Use fixed value
        load=True
    )

    # MST Projection Mode Parameters
    mst_projection_percentage = DecimalParameter(
        low=50.0,
        high=200.0,
        default=100.0,  # Pine Script default
        decimals=0,
        space="sell",
        optimize=False,  # Use fixed value
        load=True
    )

    # Hyperoptable Stop Loss
    # Pine Script original: -5.8% = -58% real loss with 10x leverage (aggressive)
    stoploss_param = DecimalParameter(
        low=-0.06,
        high=-0.02,
        default=-0.058,  # -5.8% (Pine Script default)
        decimals=3,
        space="sell",
        optimize=False,  # Use fixed value
        load=True
    )

    # LuxAlgo Market Structure Trailing Stop - Increment Factor
    # Controls how aggressively the trailing stop follows price movement
    # 100% = trail at same rate as price moves, 50% = trail at half rate
    increment_factor = DecimalParameter(
        low=50.0,
        high=200.0,
        default=95.0,  # Less aggressive trailing (was 100.0)
        decimals=0,
        space="sell",
        optimize=False,  # Use fixed value
        load=True
    )

    # Trailing Stop Parameters
    trailing_stop_positive_param = DecimalParameter(
        low=0.005,
        high=0.03,
        default=0.01,
        decimals=3,
        space="sell",
        optimize=True,
        load=True
    )

    trailing_stop_positive_offset_param = DecimalParameter(
        low=0.01,
        high=0.05,
        default=0.02,
        decimals=3,
        space="sell",
        optimize=True,
        load=True
    )

    # States
    STATE_NEUTRAL = 0
    STATE_TENDENCIA_ALCISTA = 1
    STATE_TENDENCIA_BAJISTA = 2
    STATE_ESPERANDO_CONFIRMACION_ALCISTA = 3
    STATE_ESPERANDO_CONFIRMACION_BAJISTA = 4

    plot_config = {
        "main_plot": {
            "last_sig_high": {"color": "green"},
            "last_sig_low": {"color": "red"},
            "hma": {"color": "blue"},
            "smooth_trail": {"color": "purple"},
        },
        "subplots": {
            "State": {
                "state": {"color": "blue"},
            },
            "Signals": {
                "choch_up": {"color": "lime"},
                "choch_down": {"color": "red"},
                "bos_up": {"color": "green"},
                "bos_down": {"color": "orange"},
            },
            "Filters": {
                "hma_touched_long": {"color": "cyan"},
                "hma_touched_short": {"color": "magenta"},
                "smooth_trail_bullish": {"color": "lime"},
                "smooth_trail_bearish": {"color": "red"},
            },
            "BOS_Count": {
                "bos_count_long": {"color": "lime"},
                "bos_count_short": {"color": "red"},
            }
        },
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Calculate all indicators including filters
        """
        if len(dataframe) < self.startup_candle_count:
            return dataframe

        # Apply dynamic stoploss from hyperopt
        self.stoploss = self.stoploss_param.value

        # Apply dynamic trailing stop from hyperopt
        self.trailing_stop_positive = self.trailing_stop_positive_param.value
        self.trailing_stop_positive_offset = self.trailing_stop_positive_offset_param.value

        # ========================================
        # STRUCTURE DETECTION (Same as base strategy)
        # ========================================

        depth = self.zigzag_depth.value

        dataframe['swing_high'] = 0.0
        dataframe['swing_low'] = 0.0

        # Calculate swing highs and lows
        for i in range(depth, len(dataframe) - depth):
            idx = dataframe.index[i]

            # Check swing high
            current_high = dataframe.loc[idx, 'high']
            is_swing_high = True

            for j in range(1, depth + 1):
                if dataframe.loc[dataframe.index[i - j], 'high'] >= current_high:
                    is_swing_high = False
                    break

            if is_swing_high:
                for j in range(1, depth + 1):
                    if dataframe.loc[dataframe.index[i + j], 'high'] >= current_high:
                        is_swing_high = False
                        break

            if is_swing_high:
                dataframe.loc[idx, 'swing_high'] = current_high

            # Check swing low
            current_low = dataframe.loc[idx, 'low']
            is_swing_low = True

            for j in range(1, depth + 1):
                if dataframe.loc[dataframe.index[i - j], 'low'] <= current_low:
                    is_swing_low = False
                    break

            if is_swing_low:
                for j in range(1, depth + 1):
                    if dataframe.loc[dataframe.index[i + j], 'low'] <= current_low:
                        is_swing_low = False
                        break

            if is_swing_low:
                dataframe.loc[idx, 'swing_low'] = current_low

        # Initialize state tracking
        dataframe['state'] = self.STATE_NEUTRAL
        dataframe['last_sig_high'] = np.nan
        dataframe['last_sig_low'] = np.nan
        dataframe['choch_up'] = 0
        dataframe['choch_down'] = 0
        dataframe['bos_up'] = 0
        dataframe['bos_down'] = 0

        # State machine logic (same as base strategy)
        current_state = self.STATE_NEUTRAL
        last_sig_high = np.nan
        last_sig_low = np.nan

        for i in range(len(dataframe)):
            idx = dataframe.index[i]
            current_high = dataframe.loc[idx, 'high']
            current_low = dataframe.loc[idx, 'low']
            swing_high = dataframe.loc[idx, 'swing_high']
            swing_low = dataframe.loc[idx, 'swing_low']

            # Update significant levels
            if swing_high > 0:
                last_sig_high = swing_high
            if swing_low > 0:
                last_sig_low = swing_low

            # State transitions
            if current_state == self.STATE_NEUTRAL:
                if not np.isnan(last_sig_high) and current_low < last_sig_high:
                    current_state = self.STATE_TENDENCIA_BAJISTA
                    dataframe.loc[idx, 'choch_down'] = 1
                elif not np.isnan(last_sig_low) and current_high > last_sig_low:
                    current_state = self.STATE_TENDENCIA_ALCISTA
                    dataframe.loc[idx, 'choch_up'] = 1

            elif current_state == self.STATE_TENDENCIA_ALCISTA:
                if not np.isnan(last_sig_low) and current_low < last_sig_low:
                    current_state = self.STATE_ESPERANDO_CONFIRMACION_BAJISTA
                    dataframe.loc[idx, 'choch_down'] = 1
                elif not np.isnan(last_sig_high) and current_high > last_sig_high:
                    dataframe.loc[idx, 'bos_up'] = 1

            elif current_state == self.STATE_TENDENCIA_BAJISTA:
                if not np.isnan(last_sig_high) and current_high > last_sig_high:
                    current_state = self.STATE_ESPERANDO_CONFIRMACION_ALCISTA
                    dataframe.loc[idx, 'choch_up'] = 1
                elif not np.isnan(last_sig_low) and current_low < last_sig_low:
                    dataframe.loc[idx, 'bos_down'] = 1

            elif current_state == self.STATE_ESPERANDO_CONFIRMACION_ALCISTA:
                if not np.isnan(last_sig_low) and current_low < last_sig_low:
                    current_state = self.STATE_TENDENCIA_BAJISTA
                    dataframe.loc[idx, 'choch_down'] = 1
                elif not np.isnan(last_sig_high) and current_high > last_sig_high:
                    current_state = self.STATE_TENDENCIA_ALCISTA
                    dataframe.loc[idx, 'bos_up'] = 1

            elif current_state == self.STATE_ESPERANDO_CONFIRMACION_BAJISTA:
                if not np.isnan(last_sig_high) and current_high > last_sig_high:
                    current_state = self.STATE_TENDENCIA_ALCISTA
                    dataframe.loc[idx, 'choch_up'] = 1
                elif not np.isnan(last_sig_low) and current_low < last_sig_low:
                    current_state = self.STATE_TENDENCIA_BAJISTA
                    dataframe.loc[idx, 'bos_down'] = 1

            dataframe.loc[idx, 'state'] = current_state
            dataframe.loc[idx, 'last_sig_high'] = last_sig_high
            dataframe.loc[idx, 'last_sig_low'] = last_sig_low

        # Create state_prev column for detecting state transitions
        dataframe['state_prev'] = dataframe['state'].shift(1).fillna(self.STATE_NEUTRAL)

        # ========================================
        # ENTRY FILTERS
        # ========================================

        # HMA Filter
        if self.use_hma_filter_long.value or self.use_hma_filter_short.value:
            hma_len = self.hma_length.value
            hma_wma1 = dataframe['close'].rolling(window=hma_len // 2).mean()
            hma_wma2 = dataframe['close'].rolling(window=hma_len).mean()
            hma_diff = 2 * hma_wma1 - hma_wma2
            dataframe['hma'] = hma_diff.rolling(window=int(np.sqrt(hma_len))).mean()

            # Detect HMA touch
            dataframe['hma_touched_long'] = ((dataframe['low'] <= dataframe['hma']) &
                                             (dataframe['close'] > dataframe['hma'])).astype(int)
            dataframe['hma_touched_short'] = ((dataframe['high'] >= dataframe['hma']) &
                                              (dataframe['close'] < dataframe['hma'])).astype(int)
        else:
            dataframe['hma'] = 0
            dataframe['hma_touched_long'] = 0
            dataframe['hma_touched_short'] = 0

        # Smooth Trail Filter
        if self.use_smooth_trail_long.value or self.use_smooth_trail_short.value:
            st_multiplier = self.smooth_trail_multiplier.value
            st_length = self.smooth_trail_length.value

            # Calculate ATR
            dataframe['atr'] = pta.atr(dataframe['high'], dataframe['low'], dataframe['close'], length=100)

            # SuperTrend calculation
            supertrend = pta.supertrend(dataframe['high'], dataframe['low'], dataframe['close'],
                                        length=st_length * 2, multiplier=round((7 + st_multiplier) / 2))
            dataframe['smooth_trail'] = supertrend[f'SUPERT_{st_length*2}_{round((7 + st_multiplier) / 2)}.0']
            dataframe['smooth_trail_direction'] = supertrend[f'SUPERTd_{st_length*2}_{round((7 + st_multiplier) / 2)}.0']

            dataframe['smooth_trail_bullish'] = (dataframe['smooth_trail_direction'] == 1).astype(int)
            dataframe['smooth_trail_bearish'] = (dataframe['smooth_trail_direction'] == -1).astype(int)
        else:
            dataframe['smooth_trail'] = 0
            dataframe['smooth_trail_bullish'] = 0
            dataframe['smooth_trail_bearish'] = 0

        # BOS counter for tracking
        dataframe['bos_count_long'] = 0
        dataframe['bos_count_short'] = 0

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Entry logic matching TradingView Pine original:
        - CHoCH triggers waiting_confirmation state
        - BOS confirms entry (if waiting_confirmation)
        - Optional filters (HMA, Smooth Trail) must be met

        Pine Logic:
        if choch_bullish → waiting_long_confirmation = true
        if waiting_long_confirmation and bos_bullish → enter (if filters met)

        Python equivalent using state machine:
        CHoCH changes state to ESPERANDO_CONFIRMACION_ALCISTA
        BOS while in that state → transition to TENDENCIA_ALCISTA (entry)
        """
        # Base condition: State transition from ESPERANDO_CONFIRMACION → TENDENCIA + BOS
        # This matches Pine's: waiting_long_confirmation && bos_bullish
        long_base_condition = (
            (dataframe['state'] == self.STATE_TENDENCIA_ALCISTA) &
            (dataframe['state_prev'] == self.STATE_ESPERANDO_CONFIRMACION_ALCISTA) &
            (dataframe['bos_up'] == 1)
        )

        short_base_condition = (
            (dataframe['state'] == self.STATE_TENDENCIA_BAJISTA) &
            (dataframe['state_prev'] == self.STATE_ESPERANDO_CONFIRMACION_BAJISTA) &
            (dataframe['bos_down'] == 1)
        )

        # Apply HMA filter for long (optional)
        if self.use_hma_filter_long.value:
            long_base_condition = long_base_condition & (dataframe['hma_touched_long'].rolling(10).sum() > 0)

        # Apply HMA filter for short (optional)
        if self.use_hma_filter_short.value:
            short_base_condition = short_base_condition & (dataframe['hma_touched_short'].rolling(10).sum() > 0)

        # Apply Smooth Trail filter for long (optional)
        if self.use_smooth_trail_long.value:
            long_base_condition = long_base_condition & (dataframe['smooth_trail_bullish'] == 1)

        # Apply Smooth Trail filter for short (optional)
        if self.use_smooth_trail_short.value:
            short_base_condition = short_base_condition & (dataframe['smooth_trail_bearish'] == 1)

        dataframe.loc[long_base_condition, 'enter_long'] = 1
        dataframe.loc[short_base_condition, 'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Exit logic based on selected mode and require_bos_exit parameter

        Matches Pine Script logic:
        - If require_bos_exit = True (default): CHoCH → wait BOS confirmation → exit
        - If require_bos_exit = False: CHoCH → exit immediately
        """
        if self.exit_mode.value == "trailing_only":
            return dataframe

        if self.require_bos_exit.value:
            # Conservative mode: Require BOS confirmation after CHoCH (Pine default)
            # Track waiting states
            dataframe['waiting_exit_long'] = 0
            dataframe['waiting_exit_short'] = 0

            waiting_exit_long = False
            waiting_exit_short = False

            for i in range(len(dataframe)):
                idx = dataframe.index[i]

                # CHoCH down triggers waiting for exit long confirmation
                if dataframe.loc[idx, 'choch_down'] == 1:
                    waiting_exit_long = True

                # CHoCH up cancels waiting for exit long
                if dataframe.loc[idx, 'choch_up'] == 1:
                    waiting_exit_long = False
                    waiting_exit_short = True

                # BOS down confirms exit long
                if waiting_exit_long and dataframe.loc[idx, 'bos_down'] == 1:
                    dataframe.loc[idx, 'exit_long'] = 1
                    waiting_exit_long = False

                # CHoCH up triggers waiting for exit short confirmation
                if dataframe.loc[idx, 'choch_up'] == 1:
                    waiting_exit_short = True

                # CHoCH down cancels waiting for exit short
                if dataframe.loc[idx, 'choch_down'] == 1:
                    waiting_exit_short = False
                    waiting_exit_long = True

                # BOS up confirms exit short
                if waiting_exit_short and dataframe.loc[idx, 'bos_up'] == 1:
                    dataframe.loc[idx, 'exit_short'] = 1
                    waiting_exit_short = False
        else:
            # Aggressive mode: Exit immediately on opposite CHoCH
            dataframe.loc[dataframe['choch_down'] == 1, 'exit_long'] = 1
            dataframe.loc[dataframe['choch_up'] == 1, 'exit_short'] = 1

        return dataframe

    def custom_exit(self, pair: str, trade, current_time, current_rate, current_profit, **kwargs) -> Optional[str]:
        """
        Custom exit logic for emergency CHoCH and MST projection
        NOTE: BOS counting partial exits are handled in adjust_trade_position(), not here
        NOTE: Normal CHoCH + BOS exits handled by exit_signal in populate_exit_trend()
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        # EMERGENCY EXIT: CHoCH reversal (early exit before BOS confirmation)
        # If we're long and CHoCH bearish appears, exit immediately
        # If we're short and CHoCH bullish appears, exit immediately
        is_short = getattr(trade, 'is_short', trade.trade_direction == 'short')

        if not is_short and last_candle['choch_down'] == 1:
            return "choch_emergency_exit_bearish"
        elif is_short and last_candle['choch_up'] == 1:
            return "choch_emergency_exit_bullish"

        # MST Projection Mode
        if self.exit_mode.value == "mst_projection":
            # Calculate target from entry
            entry_price = trade.open_rate

            # Get structure info from trade metadata
            structure_high = trade.get_custom_data('structure_high', entry_price * 1.05)
            structure_low = trade.get_custom_data('structure_low', entry_price * 0.95)

            is_short = getattr(trade, 'is_short', trade.trade_direction == 'short')
            if not is_short:
                price_diff = structure_high - structure_low
                target = structure_high + (price_diff * self.mst_projection_percentage.value / 100)
                if current_rate >= target:
                    return "mst_target_hit"
            else:
                price_diff = structure_high - structure_low
                target = structure_low - (price_diff * self.mst_projection_percentage.value / 100)
                if current_rate <= target:
                    return "mst_target_hit"

        return None

    def adjust_trade_position(self, trade, current_time, current_rate, current_profit,
                              min_stake, max_stake, current_entry_rate, current_exit_rate,
                              current_entry_profit, current_exit_profit, **kwargs) -> Optional[float]:
        """
        Track structure levels for LuxAlgo Market Structure Trailing Stop
        No partial exits - we only exit on CHoCH + BOS or emergency CHoCH
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        is_short = getattr(trade, 'is_short', trade.trade_direction == 'short')

        # Initialize on entry - find initial structure level (LuxAlgo logic)
        if trade.get_custom_data('structure_level', None) is None:
            entry_idx = dataframe[dataframe['date'] <= trade.open_date_utc].index[-1]

            if not is_short:
                # LONG: Find the last swing_high (zigzag_depth=23) before entry, then find minimum between that swing and entry
                # Uses same zigzag pivot calculation for both entry detection and trailing stop structure level
                swing_highs = dataframe.loc[:entry_idx, 'swing_high']
                swing_indices = swing_highs[swing_highs > 0].index

                if len(swing_indices) > 0:
                    last_swing_idx = swing_indices[-1]
                    swing_price = dataframe.loc[last_swing_idx, 'swing_high']

                    # Find minimum low between last swing high and entry
                    local_low = dataframe.loc[last_swing_idx:entry_idx, 'low'].min()
                    num_candles = entry_idx - last_swing_idx
                    sl_pct = ((local_low - current_rate) / current_rate) * 100

                    trade.set_custom_data('structure_level', float(local_low))
                    trade.set_custom_data('trailing_max', float(current_rate))
                    trade.set_custom_data('prev_candle_max', float(current_rate))

                    logger.info(f"[{trade.pair}] LONG entry at {current_rate:.4f}")
                    logger.info(f"  Swing high (zigzag depth={self.zigzag_depth.value}): {swing_price:.4f} at idx {last_swing_idx}")
                    logger.info(f"  Structure SL: {local_low:.4f} ({sl_pct:.2f}%) - {num_candles} candles")
                else:
                    # No swing found, use fallback
                    fallback_sl = current_rate * (1 + self.stoploss)
                    trade.set_custom_data('structure_level', float(fallback_sl))
                    trade.set_custom_data('trailing_max', float(current_rate))
                    trade.set_custom_data('prev_candle_max', float(current_rate))
                    logger.warning(f"[{trade.pair}] LONG - No swing found, using fallback SL: {fallback_sl:.4f}")
            else:
                # SHORT: Find the last swing_low (zigzag_depth=23) before entry, then find maximum between that swing and entry
                # Uses same zigzag pivot calculation for both entry detection and trailing stop structure level
                swing_lows = dataframe.loc[:entry_idx, 'swing_low']
                swing_indices = swing_lows[swing_lows > 0].index

                if len(swing_indices) > 0:
                    last_swing_idx = swing_indices[-1]
                    swing_price = dataframe.loc[last_swing_idx, 'swing_low']

                    # Find maximum high between last swing low and entry
                    local_high = dataframe.loc[last_swing_idx:entry_idx, 'high'].max()
                    num_candles = entry_idx - last_swing_idx
                    sl_pct = ((local_high - current_rate) / current_rate) * 100

                    trade.set_custom_data('structure_level', float(local_high))
                    trade.set_custom_data('trailing_min', float(current_rate))
                    trade.set_custom_data('prev_candle_min', float(current_rate))

                    logger.info(f"[{trade.pair}] SHORT entry at {current_rate:.4f}")
                    logger.info(f"  Swing low (zigzag depth={self.zigzag_depth.value}): {swing_price:.4f} at idx {last_swing_idx}")
                    logger.info(f"  Structure SL: {local_high:.4f} ({sl_pct:.2f}%) - {num_candles} candles")
                else:
                    # No swing found, use fallback
                    fallback_sl = current_rate * (1 - self.stoploss)
                    trade.set_custom_data('structure_level', float(fallback_sl))
                    trade.set_custom_data('trailing_min', float(current_rate))
                    trade.set_custom_data('prev_candle_min', float(current_rate))
                    logger.warning(f"[{trade.pair}] SHORT - No swing found, using fallback SL: {fallback_sl:.4f}")

            trade.set_custom_data('bos_count', 0)
            trade.set_custom_data('bos_prices', [])
            return None

        # Update trailing max/min for use in custom_stoploss
        # ALWAYS save previous candle's max/min (for bar-to-bar increment calculation)
        # This matches Pine Script: max[1] ALWAYS refers to previous bar, even if max didn't change
        if not is_short:
            prev_candle_max = trade.get_custom_data('trailing_max', current_rate)
            new_max = max(last_candle['high'], prev_candle_max)

            # ALWAYS update both prev and current, even if max didn't change
            trade.set_custom_data('prev_candle_max', float(prev_candle_max))  # Save for increment calc
            trade.set_custom_data('trailing_max', float(new_max))  # Update to new value
        else:
            prev_candle_min = trade.get_custom_data('trailing_min', current_rate)
            new_min = min(last_candle['low'], prev_candle_min)

            # ALWAYS update both prev and current, even if min didn't change
            trade.set_custom_data('prev_candle_min', float(prev_candle_min))  # Save for increment calc
            trade.set_custom_data('trailing_min', float(new_min))  # Update to new value

        # Track BOS for informational purposes (not used in SL calculation anymore)
        bos_count = trade.get_custom_data('bos_count', 0)
        bos_prices = trade.get_custom_data('bos_prices', [])
        last_processed_bos = trade.get_custom_data('last_processed_bos', None)
        current_candle_time = str(last_candle['date'])

        if not is_short and last_candle['bos_up'] == 1:
            if last_processed_bos != current_candle_time:
                trade.set_custom_data('last_processed_bos', current_candle_time)
                trade.set_custom_data('bos_count', bos_count + 1)
                bos_prices.append(float(last_candle['high']))
                trade.set_custom_data('bos_prices', bos_prices)
                logger.info(f"[{trade.pair}] BOS #{bos_count + 1} detected at {last_candle['high']:.4f}")

        elif is_short and last_candle['bos_down'] == 1:
            if last_processed_bos != current_candle_time:
                trade.set_custom_data('last_processed_bos', current_candle_time)
                trade.set_custom_data('bos_count', bos_count + 1)
                bos_prices.append(float(last_candle['low']))
                trade.set_custom_data('bos_prices', bos_prices)
                logger.info(f"[{trade.pair}] BOS #{bos_count + 1} detected at {last_candle['low']:.4f}")

        # No partial exits
        return None

    def custom_stoploss(self, pair: str, trade, current_time, current_rate, current_profit, **kwargs) -> float:
        """
        LuxAlgo Market Structure Trailing Stop

        Logic:
        - Initial SL: Set at structure level (swing low for long, swing high for short)
        - Trailing: As price makes new highs/lows, SL trails with increment_factor
        - Formula: ts += (max - prev_max) * increment_factor / 100

        Returns stoploss as percentage from current_rate (negative = loss, positive = profit)
        """
        structure_level = trade.get_custom_data('structure_level', None)

        # If no structure level set, use default SL
        if structure_level is None:
            return self.stoploss

        is_short = getattr(trade, 'is_short', trade.trade_direction == 'short')
        entry_price = trade.open_rate
        increment = self.increment_factor.value / 100.0

        if not is_short:
            # LONG POSITION
            # Get trailing max and previous candle's max
            trailing_max = trade.get_custom_data('trailing_max', entry_price)
            prev_candle_max = trade.get_custom_data('prev_candle_max', entry_price)

            # Calculate bar-to-bar increment: (max - max[1]) * increment / 100
            # This matches Pine Script: ts + (max - max[1]) * incr / 100
            max_movement = trailing_max - prev_candle_max
            trailing_adjustment = max_movement * increment

            # Calculate new trailing stop level
            prev_ts = trade.get_custom_data('trailing_stop_level', structure_level)
            new_ts = prev_ts + trailing_adjustment

            # Ensure trailing stop only moves up, never down
            new_ts = max(new_ts, structure_level)

            # Store for next iteration
            trade.set_custom_data('trailing_stop_level', float(new_ts))

            # Convert to stoploss percentage from current_rate
            # Freqtrade expects: (stop_price - current_rate) / current_rate
            # Negative value = stop below current price (loss)
            # Positive value = stop above current price (profit/breakeven)
            stoploss_pct = (new_ts - current_rate) / current_rate

            # Log when SL moves or significantly changes
            if trailing_adjustment > 0.01 or abs(stoploss_pct - self.stoploss) > 0.001:
                logger.info(f"[{pair}] LONG Trailing SL: {new_ts:.4f} ({stoploss_pct*100:.2f}%) | Max: {trailing_max:.4f} | Prev: {prev_candle_max:.4f} | Adj: {trailing_adjustment:.4f}")

            return stoploss_pct

        else:
            # SHORT POSITION
            # Get trailing min and previous candle's min
            trailing_min = trade.get_custom_data('trailing_min', entry_price)
            prev_candle_min = trade.get_custom_data('prev_candle_min', entry_price)

            # Calculate bar-to-bar increment: (min - min[1]) * increment / 100
            # For shorts, min goes down, so adjustment will be negative (trailing stop down)
            min_movement = trailing_min - prev_candle_min
            trailing_adjustment = min_movement * increment

            # Calculate new trailing stop level
            prev_ts = trade.get_custom_data('trailing_stop_level', structure_level)
            new_ts = prev_ts + trailing_adjustment

            # Ensure trailing stop only moves down, never up
            new_ts = min(new_ts, structure_level)

            # Store for next iteration
            trade.set_custom_data('trailing_stop_level', float(new_ts))

            # Convert to stoploss percentage from current_rate
            # For shorts: stop is above current price
            # (stop_price - current_rate) / current_rate will be positive = loss for short
            # We need to return negative of this for Freqtrade
            stoploss_pct = (current_rate - new_ts) / current_rate

            # Log when SL moves or significantly changes
            if abs(trailing_adjustment) > 0.01 or abs(stoploss_pct - self.stoploss) > 0.001:
                logger.info(f"[{pair}] SHORT Trailing SL: {new_ts:.4f} ({stoploss_pct*100:.2f}%) | Min: {trailing_min:.4f} | Prev: {prev_candle_min:.4f} | Adj: {trailing_adjustment:.4f}")

            return stoploss_pct
