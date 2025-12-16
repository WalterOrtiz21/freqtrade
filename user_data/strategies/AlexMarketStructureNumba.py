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
║                                       AlexMarketStructure Numba (FAST)                                                 ║
║                  Break of Structure trading system with smart money concepts and trend analysis.                       ║
║                     Numba-optimized version - 5-10x faster than original Python implementation.                        ║
╚════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

Same trading logic as AlexMarketStructurePro but with:
- Numba-accelerated indicator calculations (Embedded Library)
- In-memory reverse permission flags (no file I/O)
- Optimized callback logic (lazy loading)
"""

import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Optional, Union, Tuple
from datetime import datetime
import logging
import re

from numba import jit, int8, float64
from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter, BooleanParameter
from freqtrade.persistence import Trade

logger = logging.getLogger(__name__)


# =============================================================================
# NUMBA KERNELS (Integrated from alex_market_structure_numba.py)
# =============================================================================

@jit(nopython=True, cache=True)
def _swing_detection_kernel(
    high: np.ndarray,
    low: np.ndarray,
    period: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Detect swing highs and lows using centered rolling window.
    Matches Pine Script ta.pivothigh/ta.pivotlow behavior.
    """
    n = len(high)
    swing_high = np.full(n, np.nan)
    swing_low = np.full(n, np.nan)
    last_swing_high = np.full(n, np.nan)
    last_swing_low = np.full(n, np.nan)
    
    # Need at least period*2+1 candles
    window = period * 2 + 1
    
    current_sh = np.nan
    current_sl = np.nan
    
    for i in range(period, n - period):
        # Check if center point is highest in window
        center_high = high[i]
        is_swing_high = True
        for j in range(i - period, i + period + 1):
            if j != i and high[j] >= center_high:
                is_swing_high = False
                break
        
        if is_swing_high:
            swing_high[i] = center_high
            current_sh = center_high
        
        # Check if center point is lowest in window
        center_low = low[i]
        is_swing_low = True
        for j in range(i - period, i + period + 1):
            if j != i and low[j] <= center_low:
                is_swing_low = False
                break
        
        if is_swing_low:
            swing_low[i] = center_low
            current_sl = center_low
    
    # Forward fill swing levels
    for i in range(n):
        if not np.isnan(swing_high[i]):
            current_sh = swing_high[i]
        last_swing_high[i] = current_sh
        
        if not np.isnan(swing_low[i]):
            current_sl = swing_low[i]
        last_swing_low[i] = current_sl
    
    return swing_high, swing_low, last_swing_high, last_swing_low


@jit(nopython=True, cache=True)
def _bos_detection_kernel(
    close: np.ndarray,
    last_swing_high: np.ndarray,
    last_swing_low: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Detect Break of Structure (BOS) with pending state.
    Only fires ONCE per swing level (prevents duplicates).
    """
    n = len(close)
    bull_break = np.zeros(n, dtype=int8)
    bear_break = np.zeros(n, dtype=int8)
    trend = np.zeros(n, dtype=int8)
    
    # Track which swing level has been broken (pending state)
    prev_swing_high = np.nan
    prev_swing_low = np.nan
    high_pending = True
    low_pending = True
    current_trend = 0
    
    for i in range(1, n):
        # Check if swing level changed (new swing detected)
        if last_swing_high[i] != prev_swing_high:
            prev_swing_high = last_swing_high[i]
            high_pending = True
        
        if last_swing_low[i] != prev_swing_low:
            prev_swing_low = last_swing_low[i]
            low_pending = True
        
        # Bull break: close crosses above swing high
        if high_pending and not np.isnan(last_swing_high[i]):
            if close[i] > last_swing_high[i]:
                bull_break[i] = 1
                high_pending = False
                current_trend = 1
        
        # Bear break: close crosses below swing low
        if low_pending and not np.isnan(last_swing_low[i]):
            if close[i] < last_swing_low[i]:
                bear_break[i] = 1
                low_pending = False
                current_trend = -1
        
        trend[i] = current_trend
    
    return bull_break, bear_break, trend


@jit(nopython=True, cache=True)
def _choch_detection_kernel(
    bull_break: np.ndarray,
    bear_break: np.ndarray,
    trend: np.ndarray
) -> np.ndarray:
    """
    Detect Change of Character (CHoCH).
    CHoCH = break against previous trend direction.
    """
    n = len(bull_break)
    is_choch = np.zeros(n, dtype=int8)
    
    for i in range(1, n):
        prev_trend = trend[i - 1]
        # Bull break while previous trend was bearish = CHoCH
        if bull_break[i] == 1 and prev_trend == -1:
            is_choch[i] = 1
        # Bear break while previous trend was bullish = CHoCH
        if bear_break[i] == 1 and prev_trend == 1:
            is_choch[i] = 1
    
    return is_choch


@jit(nopython=True, cache=True)
def _tsa_kernel(
    close: np.ndarray,
    max_length: int,
    accel_multiplier: float,
    wma_period: int = 2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Trend Speed Analyzer with dynamic EMA.
    """
    n = len(close)
    trend_line = np.zeros(n)
    trend_color_green = np.zeros(n, dtype=int8)
    trend_flip_to_green = np.zeros(n, dtype=int8)
    trend_flip_to_red = np.zeros(n, dtype=int8)
    
    # Pre-calculate normalization values
    abs_close = np.abs(close)
    
    # Rolling max of abs close (200 period)
    max_abs = np.zeros(n)
    for i in range(n):
        start = max(0, i - 199)
        max_val = 0.0
        for j in range(start, i + 1):
            if abs_close[j] > max_val:
                max_val = abs_close[j]
        max_abs[i] = max(max_val, 1.0)
    
    # Normalize close
    counts_diff_norm = (close + max_abs) / (2 * max_abs)
    
    # Dynamic length
    dyn_length = 5 + counts_diff_norm * (max_length - 5)
    
    # Calculate delta and acceleration
    delta = np.zeros(n)
    delta[0] = 0.0
    for i in range(1, n):
        delta[i] = abs(close[i] - close[i - 1])
    
    # Rolling max of delta (200 period)
    max_delta = np.zeros(n)
    for i in range(n):
        start = max(0, i - 199)
        max_val = 0.0
        for j in range(start, i + 1):
            if delta[j] > max_val:
                max_val = delta[j]
        max_delta[i] = max(max_val, 1.0)
    
    # Acceleration factor
    accel_factor = delta / max_delta
    
    # Calculate dynamic EMA
    if n > 0:
        trend_line[0] = close[0]
        for i in range(1, n):
            alpha_base = 2.0 / (dyn_length[i] + 1.0)
            alpha = alpha_base * (1.0 + accel_factor[i] * accel_multiplier)
            alpha = min(alpha, 1.0)
            trend_line[i] = alpha * close[i] + (1.0 - alpha) * trend_line[i - 1]
    
    # Calculate WMA(close, 2)
    wma = np.zeros(n)
    for i in range(n):
        if i < 1:
            wma[i] = close[i]
        else:
            # WMA(2) = (2*close + 1*close[-1]) / 3
            wma[i] = (2.0 * close[i] + close[i - 1]) / 3.0
    
    # Trend color
    prev_green = 0
    for i in range(n):
        is_green = 1 if wma[i] > trend_line[i] else 0
        trend_color_green[i] = is_green
        
        if i > 0:
            if is_green == 1 and prev_green == 0:
                trend_flip_to_green[i] = 1
            if is_green == 0 and prev_green == 1:
                trend_flip_to_red[i] = 1
        
        prev_green = is_green
    
    return trend_line, trend_color_green, trend_flip_to_green, trend_flip_to_red


@jit(nopython=True, cache=True)
def _target_calc_kernel(
    entry_long: np.ndarray,
    entry_short: np.ndarray,
    atr: np.ndarray,
    volatility_multiplier: float
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,  # Long TP1, TP2, TP3, SL
    np.ndarray, np.ndarray, np.ndarray, np.ndarray   # Short TP1, TP2, TP3, SL
]:
    """
    Calculate TP and SL levels based on ATR.
    """
    n = len(entry_long)
    target_range = atr * volatility_multiplier
    
    # Long targets
    long_tp1 = entry_long + target_range * 0.8
    long_tp2 = entry_long + target_range * 1.6
    long_tp3 = entry_long + target_range * 2.0
    long_sl = entry_long - target_range * 1.2
    
    # Short targets
    short_tp1 = entry_short - target_range * 0.8
    short_tp2 = entry_short - target_range * 1.6
    short_tp3 = entry_short - target_range * 2.0
    short_sl = entry_short + target_range * 1.2
    
    return long_tp1, long_tp2, long_tp3, long_sl, short_tp1, short_tp2, short_tp3, short_sl


@jit(nopython=True, cache=True)
def _atr_kernel(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """
    Calculate ATR (Average True Range).
    """
    n = len(high)
    atr = np.zeros(n)
    tr = np.zeros(n)
    
    # True Range
    for i in range(n):
        if i == 0:
            tr[i] = high[i] - low[i]
        else:
            tr1 = high[i] - low[i]
            tr2 = abs(high[i] - close[i - 1])
            tr3 = abs(low[i] - close[i - 1])
            tr[i] = max(tr1, max(tr2, tr3))
    
    # RMA (Wilder's smoothing)
    alpha = 1.0 / period
    atr[0] = tr[0]
    for i in range(1, n):
        atr[i] = alpha * tr[i] + (1.0 - alpha) * atr[i - 1]
    
    return atr


# =============================================================================
# HELPER CLASS (Integrated)
# =============================================================================

class AlexSMC:
    """
    Numba-optimized Market Structure calculator helper class.
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        structure_period: int = 20,
        volatility_multiplier: float = 2.0,
        atr_period: int = 14,
        tsa_max_length: int = 50,
        tsa_accel_multiplier: float = 5.0
    ):
        self.df = df
        self.structure_period = structure_period
        self.volatility_multiplier = volatility_multiplier
        self.atr_period = atr_period
        self.tsa_max_length = tsa_max_length
        self.tsa_accel_multiplier = tsa_accel_multiplier
        
        # Extract numpy arrays
        self.high = df['high'].values.astype(np.float64)
        self.low = df['low'].values.astype(np.float64)
        self.close = df['close'].values.astype(np.float64)
        self.n = len(df)
    
    def get_signals(self) -> pd.DataFrame:
        """
        Calculate all signals and return as DataFrame.
        """
        # 1. ATR
        atr = _atr_kernel(self.high, self.low, self.close, self.atr_period)
        
        # 2. Swing Detection
        swing_high, swing_low, last_swing_high, last_swing_low = _swing_detection_kernel(
            self.high, self.low, self.structure_period
        )
        
        # 3. BOS Detection
        bull_break, bear_break, trend = _bos_detection_kernel(
            self.close, last_swing_high, last_swing_low
        )
        
        # 4. CHoCH Detection
        is_choch = _choch_detection_kernel(bull_break, bear_break, trend)
        
        # 5. TSA
        trend_line, trend_color_green, trend_flip_to_green, trend_flip_to_red = _tsa_kernel(
            self.close, self.tsa_max_length, self.tsa_accel_multiplier
        )
        
        # 6. Targets
        (
            long_tp1, long_tp2, long_tp3, long_sl,
            short_tp1, short_tp2, short_tp3, short_sl
        ) = _target_calc_kernel(
            last_swing_high, last_swing_low, atr, self.volatility_multiplier
        )
        
        # Build DataFrame
        result = pd.DataFrame(index=self.df.index)
        
        # Core signals
        result['atr'] = atr
        result['swing_high'] = swing_high
        result['swing_low'] = swing_low
        result['last_swing_high'] = last_swing_high
        result['last_swing_low'] = last_swing_low
        result['bull_break'] = bull_break.astype(bool)
        result['bear_break'] = bear_break.astype(bool)
        result['trend'] = trend
        result['is_choch'] = is_choch.astype(bool)
        
        # TSA signals
        result['tsa_trend_line'] = trend_line
        result['tsa_trend_color_green'] = trend_color_green.astype(bool)
        result['tsa_trend_flip_to_green'] = trend_flip_to_green.astype(bool)
        result['tsa_trend_flip_to_red'] = trend_flip_to_red.astype(bool)
        
        # Target levels
        result['long_entry'] = last_swing_high
        result['long_tp1'] = long_tp1
        result['long_tp2'] = long_tp2
        result['long_tp3'] = long_tp3
        result['long_sl'] = long_sl
        
        result['short_entry'] = last_swing_low
        result['short_tp1'] = short_tp1
        result['short_tp2'] = short_tp2
        result['short_tp3'] = short_tp3
        result['short_sl'] = short_sl
        
        return result


# =============================================================================
# STRATEGY CLASS
# =============================================================================

class AlexMarketStructureNumba(IStrategy):
    """
    Numba-optimized AlexMarketStructure strategy.
    
    Same trading logic as AlexMarketStructurePro with:
    - 5-10x faster backtesting via Numba JIT compilation
    - In-memory flags instead of file persistence
    - enable_logging parameter to control all log output
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
    max_entry_position_adjustment = 50
    startup_candle_count: int = 250

    # ==========================================================================
    # LOGGING CONTROL
    # ==========================================================================
    enable_logging = BooleanParameter(default=False, space='custom', optimize=False)

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

    # Parameters (same as original)
    structure_period = IntParameter(10, 30, default=20, space='buy', optimize=True)
    volatility_multiplier = DecimalParameter(1.5, 3.0, default=2.0, space='buy', optimize=True)
    atr_period = IntParameter(10, 20, default=14, space='buy', optimize=True)
    tsa_max_length = IntParameter(30, 100, default=50, space='buy', optimize=False)
    tsa_accel_multiplier = DecimalParameter(1.0, 10.0, default=5.0, space='buy', optimize=False)

    # Partial exit percentages
    tp1_exit_pct = 0.30
    tp2_exit_pct = 0.30

    # Position limits
    max_long_positions = 15
    max_short_positions = 15

    # IN-MEMORY reverse permission flags (no file I/O)
    reverse_permission_flags = {}

    def _get_candle_start_time(self, current_time: datetime) -> datetime:
        """Calculate the candle start time based on timeframe."""
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

        if interval_minutes < 60:
            candle_minutes = (current_time.minute // interval_minutes) * interval_minutes
            return current_time.replace(minute=candle_minutes, second=0, microsecond=0)
        elif interval_minutes < 1440:
            total_minutes = current_time.hour * 60 + current_time.minute
            candle_minutes = (total_minutes // interval_minutes) * interval_minutes
            candle_hour = candle_minutes // 60
            candle_minute = candle_minutes % 60
            return current_time.replace(hour=candle_hour, minute=candle_minute, second=0, microsecond=0)
        else:
            return current_time.replace(hour=0, minute=0, second=0, microsecond=0)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Calculate indicators using Numba-optimized library.
        """
        # Use Numba library for all calculations
        smc = AlexSMC(
            dataframe,
            structure_period=self.structure_period.value,
            volatility_multiplier=self.volatility_multiplier.value,
            atr_period=self.atr_period.value,
            tsa_max_length=self.tsa_max_length.value,
            tsa_accel_multiplier=self.tsa_accel_multiplier.value
        )
        
        signals = smc.get_signals()
        
        # Merge all signals into dataframe
        for col in signals.columns:
            dataframe[col] = signals[col].values
        
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
            (~bull_break_mask)
        )
        dataframe.loc[trend_flip_green_mask, ["enter_long", "enter_tag"]] = (1, "trend_flip_long")

        # Trend flip to red - REVERSAL (requires bull_break position)
        trend_flip_red_mask = (
            (dataframe['tsa_trend_flip_to_red']) &
            (dataframe['volume'] > 0) &
            (~bear_break_mask)
        )
        dataframe.loc[trend_flip_red_mask, ["enter_short", "enter_tag"]] = (1, "trend_flip_short")

        # Trend flip - INDEPENDENT signals
        independent_long = trend_flip_green_mask & (dataframe['enter_long'] != 1)
        dataframe.loc[independent_long, ["enter_long", "enter_tag"]] = (1, "trend_flip_long_independent")

        independent_short = trend_flip_red_mask & (dataframe['enter_short'] != 1)
        dataframe.loc[independent_short, ["enter_short", "enter_tag"]] = (1, "trend_flip_short_independent")

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Exit signals handled by custom_exit
        return dataframe

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                           time_in_force: str, current_time: datetime, entry_tag: Optional[str],
                           side: str, **kwargs) -> bool:
        """Check if we should enter the trade based on position limits and reverse permissions."""


        if entry_tag == 'trend_flip_long':
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            last_candle = dataframe.iloc[-1].squeeze()
            trend_flip_to_green = last_candle.get('tsa_trend_flip_to_green', False)

            if not trend_flip_to_green:
                return False

            permission_flag = self.reverse_permission_flags.get(pair, None)
            if permission_flag != 'reverse_to_long':
                return False

            del self.reverse_permission_flags[pair]
            if self.enable_logging.value:
                logger.info(f"[{pair}] ✅ trend_flip_long APPROVED")
            return True

        if entry_tag == 'trend_flip_short':
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            last_candle = dataframe.iloc[-1].squeeze()
            trend_flip_to_red = last_candle.get('tsa_trend_flip_to_red', False)

            if not trend_flip_to_red:
                return False

            permission_flag = self.reverse_permission_flags.get(pair, None)
            if permission_flag != 'reverse_to_short':
                return False

            del self.reverse_permission_flags[pair]
            if self.enable_logging.value:
                logger.info(f"[{pair}] ✅ trend_flip_short APPROVED")
            return True

        if self.enable_logging.value:
            logger.info(f"[{pair}] ✅ {'LONG' if side == 'long' else 'SHORT'} entry confirmed: {entry_tag}")
        return True

    def order_filled(self, pair: str, trade: Trade, order, current_time: datetime, **kwargs) -> None:
        """Handle order fills - set TP/SL levels and manage reverse flags."""
        if order.ft_order_side == trade.entry_side:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            last_candle = dataframe.iloc[-1].squeeze()

            current_candle = self._get_candle_start_time(current_time)
            trade.set_custom_data('last_add_candle', current_candle.isoformat())

            if trade.is_short:
                current_swing_level = float(last_candle['last_swing_low'])
            else:
                current_swing_level = float(last_candle['last_swing_high'])
            trade.set_custom_data('last_entry_swing_level', current_swing_level)

            if trade.nr_of_successful_entries == 1:
                trade.set_custom_data('initial_stake', order.stake_amount)
                trade.set_custom_data('tp_hits', 0)

                # Set reverse permission flag (in-memory only)
                if trade.enter_tag == 'bear_break' and trade.is_short:
                    self.reverse_permission_flags[pair] = 'reverse_to_long'
                elif trade.enter_tag == 'bull_break' and not trade.is_short:
                    self.reverse_permission_flags[pair] = 'reverse_to_short'
            else:
                trade.set_custom_data('initial_stake', trade.stake_amount)
                trade.set_custom_data('tp_hits', 0)

            # Set TP/SL levels
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

        else:
            if not trade.is_open:
                entry_tag = trade.enter_tag
                exit_reason = trade.exit_reason if hasattr(trade, 'exit_reason') else 'unknown'
                is_trend_flip_exit = exit_reason in ['trend_flip_reverse_long', 'trend_flip_reverse_short']

                if not is_trend_flip_exit:
                    if entry_tag == 'bear_break' and trade.is_short:
                        if pair in self.reverse_permission_flags:
                            del self.reverse_permission_flags[pair]
                    elif entry_tag == 'bull_break' and not trade.is_short:
                        if pair in self.reverse_permission_flags:
                            del self.reverse_permission_flags[pair]

        return None

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                             current_rate: float, current_profit: float,
                             min_stake: Optional[float], max_stake: float,
                             current_entry_rate: float, current_exit_rate: float,
                             current_entry_profit: float, current_exit_profit: float,
                             **kwargs) -> Optional[float]:


        sl_price = trade.get_custom_data('sl_price', None)
        if sl_price is None or sl_price == 0:
            # Fallback: Load dataframe only if data is missing
            dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
            last_candle = dataframe.iloc[-1].squeeze()

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

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        target_range = float(last_candle['atr']) * self.volatility_multiplier.value

        trend_flip_to_green = bool(last_candle.get('tsa_trend_flip_to_green', False))
        trend_flip_to_red = bool(last_candle.get('tsa_trend_flip_to_red', False))

        # Priority 1: Trend flip reversals
        if direction == 'short' and trend_flip_to_green and entry_tag == 'bear_break':
            return 'trend_flip_reverse_long'

        if direction == 'long' and trend_flip_to_red and entry_tag == 'bull_break':
            return 'trend_flip_reverse_short'

        # Priority 2: Reverse signal exit
        if direction == 'long':
            if last_candle.get('bear_break', False):
                return 'reverse_signal'
        else:
            if last_candle.get('bull_break', False):
                return 'reverse_signal'

        # Priority 3: TP Trailing Stop Logic
        if direction == 'long':
            if tp_hits >= 2 and tp3_price > 0 and current_rate >= tp3_price:
                return 'tp3_trailing'
            elif tp_hits == 2 and current_rate <= tp1_price:
                return 'tp2_trailing'
            elif tp_hits == 1 and current_rate <= entry_price:
                return 'tp1_trailing'
        else:
            if tp_hits >= 2 and tp3_price > 0 and current_rate <= tp3_price:
                return 'tp3_trailing'
            elif tp_hits == 2 and current_rate >= tp1_price:
                return 'tp2_trailing'
            elif tp_hits == 1 and current_rate >= entry_price:
                return 'tp1_trailing'

        # Priority 4: Hybrid trailing stop (after TP2)
        if tp_hits >= 2:
            entry_atr = trade.get_custom_data('entry_atr', target_range)

            if direction == 'long':
                best_price = trade.get_custom_data('best_price', current_rate)
                if current_rate > best_price:
                    best_price = current_rate
                    trade.set_custom_data('best_price', best_price)

                atr_trailing_sl = best_price - (entry_atr * self.volatility_multiplier.value)
                percentage_trailing_sl = best_price * 0.97
                trailing_sl = max(atr_trailing_sl, percentage_trailing_sl)

                if current_rate <= trailing_sl:
                    return 'hybrid_trailing_sl'

            else:
                best_price = trade.get_custom_data('best_price', current_rate)
                if current_rate < best_price:
                    best_price = current_rate
                    trade.set_custom_data('best_price', best_price)

                atr_trailing_sl = best_price + (entry_atr * self.volatility_multiplier.value)
                percentage_trailing_sl = best_price * 1.03
                trailing_sl = min(atr_trailing_sl, percentage_trailing_sl)

                if current_rate >= trailing_sl:
                    return 'hybrid_trailing_sl'

        return None

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                side: str, **kwargs) -> float:
        return self.config.get('leverage', 1.0)
