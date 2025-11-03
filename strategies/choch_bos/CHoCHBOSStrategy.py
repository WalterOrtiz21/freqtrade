"""
CHoCH/BOS Strategy - Change of Character + Break of Structure

This strategy implements Smart Money Concepts focusing on:
- CHoCH (Change of Character): Potential trend reversal signal
- BOS (Break of Structure): Trend continuation/confirmation signal

The strategy uses a state machine to track market structure:
- NEUTRAL: Initial state, waiting for first confirmed trend
- TENDENCIA_ALCISTA: Uptrend confirmed, looking for BOS up or CHoCH down
- TENDENCIA_BAJISTA: Downtrend confirmed, looking for BOS down or CHoCH up
- ESPERANDO_CONFIRMACION_ALCISTA: CHoCH up detected, waiting for BOS up to enter LONG
- ESPERANDO_CONFIRMACION_BAJISTA: CHoCH down detected, waiting for BOS down to enter SHORT

Trading Logic:
1. Entry: Only after confirmation (CHoCH followed by BOS in same direction)
2. Exit Management (two modes):

   A. BOS-based TPs (use_bos_tps = True) - RECOMMENDED:
      - TP1: First BOS after entry → Close tp1_percentage (default 50%)
      - TP2: Second BOS after entry → Close tp2_percentage (default 30%)
      - TP3: Remaining position runs until CHoCH opposite or ROI
      - Break Even: Moves SL to entry after TP1 (if move_to_be_after_tp1 = True)

   B. Traditional Trailing Stop (use_bos_tps = False):
      - Uses standard trailing stop configuration
      - Exit on opposite CHoCH signal

3. Final Exit: CHoCH opposite (potential reversal) always closes remaining position

Hyperoptable Parameters:
- zigzag_depth: Swing detection lookback (5-50, default 23)
- zigzag_deviation: Minimum % move for pivot (3-15%, default 5%)
- use_bos_tps: Enable BOS-based TP system (True/False)
- tp1_percentage: % to close at TP1 (30-70%, default 50%)
- tp2_percentage: % to close at TP2 (20-50%, default 30%)
- move_to_be_after_tp1: Move SL to breakeven after TP1 (True/False)

Author: Based on Smart Money Concepts
Recommended timeframes: 15m (best results), 5m, 1h
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from pandas import DataFrame
import pandas_ta as pta

from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter, CategoricalParameter

logger = logging.getLogger(__name__)


class CHoCHBOSStrategy(IStrategy):
    """
    CHoCH/BOS Strategy with ZigZag-based pivots detection
    """

    # Strategy configuration
    minimal_roi = {
        "0": 0.10,   # 10% profit target
        "60": 0.05,  # 5% after 1 hour
        "120": 0.03, # 3% after 2 hours
        "240": 0.01  # 1% after 4 hours
    }

    stoploss = -0.03  # 3% stop loss

    # Trailing stop configuration
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = True

    # Operational settings
    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    can_short = True
    position_adjustment_enable = True  # Enable partial exits for TP system

    # Startup candles needed for ZigZag calculation
    startup_candle_count: int = 100

    # Hyperoptable parameters - Structure Detection
    zigzag_depth = IntParameter(
        low=5,
        high=50,
        default=23,
        space="buy",
        optimize=True,
        load=True
    )

    zigzag_deviation = IntParameter(
        low=3,
        high=15,
        default=5,
        space="buy",
        optimize=True,
        load=True
    )

    # Hyperoptable parameters - TP Management
    use_bos_tps = CategoricalParameter(
        [True, False],
        default=True,
        space="sell",
        optimize=True,
        load=True
    )

    tp1_percentage = DecimalParameter(
        low=0.3,
        high=0.7,
        default=0.5,
        decimals=1,
        space="sell",
        optimize=True,
        load=True
    )

    tp2_percentage = DecimalParameter(
        low=0.2,
        high=0.5,
        default=0.3,
        decimals=1,
        space="sell",
        optimize=True,
        load=True
    )

    move_to_be_after_tp1 = CategoricalParameter(
        [True, False],
        default=True,
        space="sell",
        optimize=True,
        load=True
    )

    # States (as constants for clarity)
    STATE_NEUTRAL = 0
    STATE_TENDENCIA_ALCISTA = 1
    STATE_TENDENCIA_BAJISTA = 2
    STATE_ESPERANDO_CONFIRMACION_ALCISTA = 3
    STATE_ESPERANDO_CONFIRMACION_BAJISTA = 4

    plot_config = {
        "main_plot": {
            "last_sig_high": {"color": "green"},
            "last_sig_low": {"color": "red"},
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
            "Pivots": {
                "swing_high": {"color": "cyan"},
                "swing_low": {"color": "magenta"},
            },
            "BOS_Count": {
                "bos_count_long": {"color": "lime"},
                "bos_count_short": {"color": "red"},
            }
        },
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Calculate swing highs/lows (pivot points) and implement state machine logic
        """
        if len(dataframe) < self.startup_candle_count:
            return dataframe

        # Detect swing highs and swing lows
        # A swing high is a peak where the high is higher than 'depth' candles before and after
        # A swing low is a trough where the low is lower than 'depth' candles before and after
        depth = self.zigzag_depth.value

        dataframe['swing_high'] = 0.0
        dataframe['swing_low'] = 0.0

        # Calculate swing highs and lows
        for i in range(depth, len(dataframe) - depth):
            idx = dataframe.index[i]

            # Check if current candle is a swing high
            current_high = dataframe.loc[idx, 'high']
            is_swing_high = True

            # Check if current high is higher than previous 'depth' highs
            for j in range(1, depth + 1):
                if dataframe.loc[dataframe.index[i - j], 'high'] >= current_high:
                    is_swing_high = False
                    break

            # Check if current high is higher than next 'depth' highs
            if is_swing_high:
                for j in range(1, depth + 1):
                    if dataframe.loc[dataframe.index[i + j], 'high'] >= current_high:
                        is_swing_high = False
                        break

            if is_swing_high:
                dataframe.loc[idx, 'swing_high'] = current_high

            # Check if current candle is a swing low
            current_low = dataframe.loc[idx, 'low']
            is_swing_low = True

            # Check if current low is lower than previous 'depth' lows
            for j in range(1, depth + 1):
                if dataframe.loc[dataframe.index[i - j], 'low'] <= current_low:
                    is_swing_low = False
                    break

            # Check if current low is lower than next 'depth' lows
            if is_swing_low:
                for j in range(1, depth + 1):
                    if dataframe.loc[dataframe.index[i + j], 'low'] <= current_low:
                        is_swing_low = False
                        break

            if is_swing_low:
                dataframe.loc[idx, 'swing_low'] = current_low

        # Initialize state tracking columns
        dataframe['state'] = self.STATE_NEUTRAL
        dataframe['last_sig_high'] = np.nan
        dataframe['last_sig_low'] = np.nan

        # Event detection columns
        dataframe['choch_up'] = 0    # CHoCH bullish (price breaks above previous high in downtrend)
        dataframe['choch_down'] = 0  # CHoCH bearish (price breaks below previous low in uptrend)
        dataframe['bos_up'] = 0      # BOS bullish (price breaks above previous high in uptrend)
        dataframe['bos_down'] = 0    # BOS bearish (price breaks below previous low in downtrend)

        # Note: swing_high and swing_low are already calculated above

        # State machine logic - iterate through each candle
        current_state = self.STATE_NEUTRAL
        last_high = np.nan
        last_low = np.nan

        for i in range(self.startup_candle_count, len(dataframe)):
            idx = dataframe.index[i]
            close = dataframe.loc[idx, 'close']

            # Update last significant high/low if new swing point detected
            if dataframe.loc[idx, 'swing_high'] > 0:
                last_high = dataframe.loc[idx, 'swing_high']
            if dataframe.loc[idx, 'swing_low'] > 0:
                last_low = dataframe.loc[idx, 'swing_low']

            # Store current values
            dataframe.loc[idx, 'last_sig_high'] = last_high
            dataframe.loc[idx, 'last_sig_low'] = last_low

            # Skip if we don't have both high and low yet
            if pd.isna(last_high) or pd.isna(last_low):
                dataframe.loc[idx, 'state'] = current_state
                continue

            # --- STATE MACHINE LOGIC ---

            if current_state == self.STATE_NEUTRAL:
                # Looking for first confirmed trend
                # If price breaks above last high -> potential uptrend
                if close > last_high:
                    dataframe.loc[idx, 'bos_up'] = 1
                    current_state = self.STATE_TENDENCIA_ALCISTA
                # If price breaks below last low -> potential downtrend
                elif close < last_low:
                    dataframe.loc[idx, 'bos_down'] = 1
                    current_state = self.STATE_TENDENCIA_BAJISTA

            elif current_state == self.STATE_TENDENCIA_BAJISTA:
                # In downtrend, looking for BOS down (continuation) or CHoCH up (reversal)

                # BOS Bajista: price breaks below last significant low
                if close < last_low:
                    dataframe.loc[idx, 'bos_down'] = 1
                    # State remains TENDENCIA_BAJISTA

                # CHoCH Alcista: price breaks above last significant high
                elif close > last_high:
                    dataframe.loc[idx, 'choch_up'] = 1
                    current_state = self.STATE_ESPERANDO_CONFIRMACION_ALCISTA

            elif current_state == self.STATE_TENDENCIA_ALCISTA:
                # In uptrend, looking for BOS up (continuation) or CHoCH down (reversal)

                # BOS Alcista: price breaks above last significant high
                if close > last_high:
                    dataframe.loc[idx, 'bos_up'] = 1
                    # State remains TENDENCIA_ALCISTA

                # CHoCH Bajista: price breaks below last significant low
                elif close < last_low:
                    dataframe.loc[idx, 'choch_down'] = 1
                    current_state = self.STATE_ESPERANDO_CONFIRMACION_BAJISTA

            elif current_state == self.STATE_ESPERANDO_CONFIRMACION_ALCISTA:
                # Waiting for BOS up to confirm new uptrend

                # BOS de Confirmación: price breaks above last significant high
                if close > last_high:
                    dataframe.loc[idx, 'bos_up'] = 1
                    current_state = self.STATE_TENDENCIA_ALCISTA
                    # Entry signal will be generated in populate_entry_trend

                # If price breaks below last low again, back to downtrend
                elif close < last_low:
                    dataframe.loc[idx, 'bos_down'] = 1
                    current_state = self.STATE_TENDENCIA_BAJISTA

            elif current_state == self.STATE_ESPERANDO_CONFIRMACION_BAJISTA:
                # Waiting for BOS down to confirm new downtrend

                # BOS de Confirmación: price breaks below last significant low
                if close < last_low:
                    dataframe.loc[idx, 'bos_down'] = 1
                    current_state = self.STATE_TENDENCIA_BAJISTA
                    # Entry signal will be generated in populate_entry_trend

                # If price breaks above last high again, back to uptrend
                elif close > last_high:
                    dataframe.loc[idx, 'bos_up'] = 1
                    current_state = self.STATE_TENDENCIA_ALCISTA

            # Store current state
            dataframe.loc[idx, 'state'] = current_state

        # Create helper columns for entry/exit logic
        dataframe['state_prev'] = dataframe['state'].shift(1)

        # Track BOS count for TP system (resets on each new entry signal)
        dataframe['bos_count_long'] = 0
        dataframe['bos_count_short'] = 0

        # Count BOS after entry for TP management
        bos_long_counter = 0
        bos_short_counter = 0
        in_long_trade = False
        in_short_trade = False

        for i in range(self.startup_candle_count, len(dataframe)):
            idx = dataframe.index[i]

            # Detect new LONG entry
            if (dataframe.loc[idx, 'state'] == self.STATE_TENDENCIA_ALCISTA and
                dataframe.loc[idx, 'state_prev'] == self.STATE_ESPERANDO_CONFIRMACION_ALCISTA):
                in_long_trade = True
                in_short_trade = False
                bos_long_counter = 0  # Reset counter on new entry
                bos_short_counter = 0

            # Detect new SHORT entry
            elif (dataframe.loc[idx, 'state'] == self.STATE_TENDENCIA_BAJISTA and
                  dataframe.loc[idx, 'state_prev'] == self.STATE_ESPERANDO_CONFIRMACION_BAJISTA):
                in_short_trade = True
                in_long_trade = False
                bos_short_counter = 0  # Reset counter on new entry
                bos_long_counter = 0

            # Detect exit signals (CHoCH opposite)
            elif (dataframe.loc[idx, 'state'] == self.STATE_ESPERANDO_CONFIRMACION_BAJISTA and
                  dataframe.loc[idx, 'state_prev'] == self.STATE_TENDENCIA_ALCISTA):
                in_long_trade = False
                bos_long_counter = 0

            elif (dataframe.loc[idx, 'state'] == self.STATE_ESPERANDO_CONFIRMACION_ALCISTA and
                  dataframe.loc[idx, 'state_prev'] == self.STATE_TENDENCIA_BAJISTA):
                in_short_trade = False
                bos_short_counter = 0

            # Count BOS while in trade
            if in_long_trade and dataframe.loc[idx, 'bos_up'] == 1:
                bos_long_counter += 1

            if in_short_trade and dataframe.loc[idx, 'bos_down'] == 1:
                bos_short_counter += 1

            dataframe.loc[idx, 'bos_count_long'] = bos_long_counter
            dataframe.loc[idx, 'bos_count_short'] = bos_short_counter

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Generate entry signals based on state transitions

        LONG Entry: State transitions from ESPERANDO_CONFIRMACION_ALCISTA to TENDENCIA_ALCISTA
        SHORT Entry: State transitions from ESPERANDO_CONFIRMACION_BAJISTA to TENDENCIA_BAJISTA
        """
        # LONG Entry: Just transitioned to TENDENCIA_ALCISTA from ESPERANDO_CONFIRMACION_ALCISTA
        dataframe.loc[
            (
                (dataframe['state'] == self.STATE_TENDENCIA_ALCISTA) &
                (dataframe['state_prev'] == self.STATE_ESPERANDO_CONFIRMACION_ALCISTA) &
                (dataframe['bos_up'] == 1) &
                (dataframe['volume'] > 0)  # Ensure volume exists
            ),
            'enter_long'] = 1

        # SHORT Entry: Just transitioned to TENDENCIA_BAJISTA from ESPERANDO_CONFIRMACION_BAJISTA
        dataframe.loc[
            (
                (dataframe['state'] == self.STATE_TENDENCIA_BAJISTA) &
                (dataframe['state_prev'] == self.STATE_ESPERANDO_CONFIRMACION_BAJISTA) &
                (dataframe['bos_down'] == 1) &
                (dataframe['volume'] > 0)  # Ensure volume exists
            ),
            'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Generate exit signals based on opposite CHoCH

        EXIT LONG: CHoCH down detected (state changes to ESPERANDO_CONFIRMACION_BAJISTA)
        EXIT SHORT: CHoCH up detected (state changes to ESPERANDO_CONFIRMACION_ALCISTA)
        """
        # EXIT LONG: State transitions to ESPERANDO_CONFIRMACION_BAJISTA (CHoCH down detected)
        dataframe.loc[
            (
                (dataframe['state'] == self.STATE_ESPERANDO_CONFIRMACION_BAJISTA) &
                (dataframe['state_prev'] == self.STATE_TENDENCIA_ALCISTA) &
                (dataframe['choch_down'] == 1)
            ),
            'exit_long'] = 1

        # EXIT SHORT: State transitions to ESPERANDO_CONFIRMACION_ALCISTA (CHoCH up detected)
        dataframe.loc[
            (
                (dataframe['state'] == self.STATE_ESPERANDO_CONFIRMACION_ALCISTA) &
                (dataframe['state_prev'] == self.STATE_TENDENCIA_BAJISTA) &
                (dataframe['choch_up'] == 1)
            ),
            'exit_short'] = 1

        return dataframe

    def leverage(self, pair: str, current_time: 'datetime', current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                 side: str, **kwargs) -> float:
        """
        Customize leverage for each trade. Default: 3x
        Can be optimized based on market conditions or pair.
        """
        return 3.0

    def custom_exit(self, pair: str, trade, current_time, current_rate, current_profit, **kwargs):
        """
        Custom exit logic based on BOS count for TP system

        Only used when use_bos_tps = True
        Returns exit reason string when TP is hit
        """
        if not self.use_bos_tps.value:
            return None

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        if trade.is_short:
            bos_count = last_candle['bos_count_short']

            # TP1: First BOS after entry
            if bos_count >= 1 and trade.nr_of_successful_exits == 0:
                return 'tp1_bos_short'

            # TP2: Second BOS after entry
            elif bos_count >= 2 and trade.nr_of_successful_exits == 1:
                return 'tp2_bos_short'

        else:  # Long trade
            bos_count = last_candle['bos_count_long']

            # TP1: First BOS after entry
            if bos_count >= 1 and trade.nr_of_successful_exits == 0:
                return 'tp1_bos_long'

            # TP2: Second BOS after entry
            elif bos_count >= 2 and trade.nr_of_successful_exits == 1:
                return 'tp2_bos_long'

        return None

    def adjust_trade_position(self, trade, current_time, current_rate, current_profit,
                               min_stake, max_stake, current_entry_rate, current_exit_rate,
                               current_entry_profit, current_exit_profit, **kwargs):
        """
        Adjust trade position for partial exits (TPs)

        Returns negative value to reduce position size
        """
        if not self.use_bos_tps.value:
            return None

        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        if trade.is_short:
            bos_count = last_candle['bos_count_short']
        else:
            bos_count = last_candle['bos_count_long']

        # TP1: Close tp1_percentage of position on first BOS
        if bos_count >= 1 and trade.nr_of_successful_exits == 0:
            # Return negative stake to close partial position
            # Example: if tp1_percentage = 0.5, close 50% of position
            return -(trade.stake_amount * self.tp1_percentage.value)

        # TP2: Close tp2_percentage of remaining position on second BOS
        elif bos_count >= 2 and trade.nr_of_successful_exits == 1:
            # Calculate remaining position after TP1
            remaining_percentage = 1.0 - self.tp1_percentage.value
            # Close tp2_percentage of remaining
            return -(trade.stake_amount * remaining_percentage * self.tp2_percentage.value)

        return None

    def custom_stoploss(self, pair: str, trade, current_time, current_rate, current_profit, **kwargs):
        """
        Custom stoploss logic - Move to Break Even after TP1

        Returns stoploss value (0.001 = breakeven + 0.1% profit)
        Returns None to use default stoploss
        """
        if not self.use_bos_tps.value or not self.move_to_be_after_tp1.value:
            return None

        # Move to breakeven after first successful exit (TP1)
        if trade.nr_of_successful_exits >= 1:
            # Return small positive value to lock in minimal profit
            return 0.001  # Breakeven + 0.1%

        return None
