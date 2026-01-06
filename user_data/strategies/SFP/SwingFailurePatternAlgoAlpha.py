"""
SwingFailurePatternAlgoAlpha Strategy
=====================================

Faithful port of TradingView "Swing Failure Signals [AlgoAlpha]" indicator.
Uses Numba JIT compilation for high-performance backtesting/Hyperopt.

Detection Logic:
1. Pivot Detection: Symmetric pivothigh/pivotlow
2. Sweep Detection: Price breaks pivot within max_pivot_edge bars
3. CISD (Change in State of Delivery): Reversal candle pattern with tolerance filter
4. Signal: Sweep + CISD within patience bars

Entry: SFP signal (bullish/bearish)
Exit: Opposite SFP signal OR TP1 (price-based)
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime
from pandas import DataFrame
from typing import Optional, Tuple
from numba import njit

logger = logging.getLogger(__name__)

from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter, BooleanParameter
from freqtrade.persistence import Trade


# =============================================================================
# NUMBA-OPTIMIZED FUNCTIONS
# =============================================================================

@njit(cache=False)
def _pivot_high_numba(high: np.ndarray, left: int, right: int) -> np.ndarray:
    """Calculate pivot highs with symmetric lookback."""
    n = len(high)
    result = np.full(n, np.nan)
    
    for i in range(left, n - right):
        is_pivot = True
        pivot_val = high[i]
        
        for j in range(1, left + 1):
            if high[i - j] >= pivot_val:
                is_pivot = False
                break
        
        if is_pivot:
            for j in range(1, right + 1):
                if high[i + j] >= pivot_val:
                    is_pivot = False
                    break
        
        if is_pivot:
            result[i + right] = pivot_val
    
    return result


@njit(cache=False)
def _pivot_low_numba(low: np.ndarray, left: int, right: int) -> np.ndarray:
    """Calculate pivot lows with symmetric lookback."""
    n = len(low)
    result = np.full(n, np.nan)
    
    for i in range(left, n - right):
        is_pivot = True
        pivot_val = low[i]
        
        for j in range(1, left + 1):
            if low[i - j] <= pivot_val:
                is_pivot = False
                break
        
        if is_pivot:
            for j in range(1, right + 1):
                if low[i + j] <= pivot_val:
                    is_pivot = False
                    break
        
        if is_pivot:
            result[i + right] = pivot_val
    
    return result


@njit(cache=False)
def _calculate_sfp_signals(
    high: np.ndarray,
    low: np.ndarray,
    open_: np.ndarray,
    close: np.ndarray,
    pivot_high_arr: np.ndarray,
    pivot_low_arr: np.ndarray,
    length: int,
    max_edge: int,
    patience_bars: int,
    tolerance: float,
    # dates: np.ndarray = None # Removed debug arg
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate SFP signals using AlgoAlpha logic (Numba-optimized).
    
    Returns: (bullsfp, bearsfp, trend)
    """
    n = len(high)
    
    # Output arrays
    bullsfp = np.zeros(n, dtype=np.int64)
    bearsfp = np.zeros(n, dtype=np.int64)
    trend_out = np.zeros(n, dtype=np.int64)
    
    # State - pivot storage
    max_pivots = 1000
    pivhs_idx = np.zeros(max_pivots, dtype=np.int64)
    pivhs_price = np.zeros(max_pivots, dtype=np.float64)
    pivhs_count = 0
    
    pivls_idx = np.zeros(max_pivots, dtype=np.int64)
    pivls_price = np.zeros(max_pivots, dtype=np.float64)
    pivls_count = 0
    
    # CISD potentials
    max_potential = 1000
    bear_pot_val = np.zeros(max_potential, dtype=np.float64)
    bear_pot_idx = np.zeros(max_potential, dtype=np.int64)
    bear_pot_count = 0
    
    bull_pot_val = np.zeros(max_potential, dtype=np.float64)
    bull_pot_idx = np.zeros(max_potential, dtype=np.int64)
    bull_pot_count = 0
    
    bar_sweep_bull = 0
    bar_sweep_bear = 0
    trend = 0
    last_signal = 0  # Track last signal type: 1=bullish, -1=bearish, 0=none
    
    for i in range(length + 1, n):
        # PIVOT STORAGE
        if not np.isnan(pivot_high_arr[i]):
            if pivhs_count < max_pivots:
                pivhs_count += 1
            for k in range(pivhs_count - 1, 0, -1):
                pivhs_idx[k] = pivhs_idx[k - 1]
                pivhs_price[k] = pivhs_price[k - 1]
            pivhs_idx[0] = i - length
            pivhs_price[0] = pivot_high_arr[i]
        
        if not np.isnan(pivot_low_arr[i]):
            if pivls_count < max_pivots:
                pivls_count += 1
            for k in range(pivls_count - 1, 0, -1):
                pivls_idx[k] = pivls_idx[k - 1]
                pivls_price[k] = pivls_price[k - 1]
            pivls_idx[0] = i - length
            pivls_price[0] = pivot_low_arr[i]
        
        # SWEEP DETECTION - BEARISH
        lvl_bear = 0.0
        new_count = 0
        for j in range(pivhs_count):
            if high[i] > pivhs_price[j]:
                if i - pivhs_idx[j] < max_edge:
                    if pivhs_price[j] > lvl_bear:
                        lvl_bear = pivhs_price[j]
            else:
                pivhs_idx[new_count] = pivhs_idx[j]
                pivhs_price[new_count] = pivhs_price[j]
                new_count += 1
        pivhs_count = new_count
        if lvl_bear != 0.0:
            bar_sweep_bear = i
        
        # SWEEP DETECTION - BULLISH
        lvl_bull = 0.0
        new_count = 0
        for j in range(pivls_count):
            if low[i] < pivls_price[j]:
                if i - pivls_idx[j] < max_edge:
                    if pivls_price[j] < lvl_bull or lvl_bull == 0:
                        lvl_bull = pivls_price[j]
            else:
                pivls_idx[new_count] = pivls_idx[j]
                pivls_price[new_count] = pivls_price[j]
                new_count += 1
        pivls_count = new_count
        if lvl_bull != 0.0:
            bar_sweep_bull = i
        
        # CISD POTENTIAL DETECTION
        if i > 0:
            if close[i-1] < open_[i-1] and close[i] > open_[i]:
                if bear_pot_count < max_potential:
                    bear_pot_count += 1
                for k in range(bear_pot_count - 1, 0, -1):
                    bear_pot_val[k] = bear_pot_val[k - 1]
                    bear_pot_idx[k] = bear_pot_idx[k - 1]
                bear_pot_val[0] = open_[i]
                bear_pot_idx[0] = i
            
            if close[i-1] > open_[i-1] and close[i] < open_[i]:
                if bull_pot_count < max_potential:
                    bull_pot_count += 1
                for k in range(bull_pot_count - 1, 0, -1):
                    bull_pot_val[k] = bull_pot_val[k - 1]
                    bull_pot_idx[k] = bull_pot_idx[k - 1]
                bull_pot_val[0] = open_[i]
                bull_pot_idx[0] = i
        
        # TRIM REMOVED - using full buffer size
        # if bear_pot_count > 50: bear_pot_count = 50
        # if bull_pot_count > 50: bull_pot_count = 50
        
        # CISD CONFIRMATION - BEARISH
        cisd = 0
        while bear_pot_count > 0:
            p_val = bear_pot_val[0]
            p_idx = bear_pot_idx[0]
            
            if close[i] < p_val:
                len_check = i - p_idx
                if 0 <= len_check < 500:
                    highest = 0.0
                    for k in range(len_check + 1):
                        if close[i - k] > highest:
                            highest = close[i - k]
                    
                    init = len_check + 1
                    top = 0.0
                    while init < min(500, i):
                        idx = i - init
                        if idx >= 0 and close[idx] < open_[idx]:
                            top = open_[idx]
                            init += 1
                        else:
                            break
                    
                    denom = top - p_val
                    if denom != 0 and abs(denom) > 1e-10:
                        if (highest - p_val) / denom > tolerance:
                            cisd = 1
                            bear_pot_count = 0
                            break
                        else:
                            for k in range(bear_pot_count - 1):
                                bear_pot_val[k] = bear_pot_val[k + 1]
                                bear_pot_idx[k] = bear_pot_idx[k + 1]
                            bear_pot_count -= 1
                    else:
                        for k in range(bear_pot_count - 1):
                            bear_pot_val[k] = bear_pot_val[k + 1]
                            bear_pot_idx[k] = bear_pot_idx[k + 1]
                        bear_pot_count -= 1
                else:
                    for k in range(bear_pot_count - 1):
                        bear_pot_val[k] = bear_pot_val[k + 1]
                        bear_pot_idx[k] = bear_pot_idx[k + 1]
                    bear_pot_count -= 1
            else:
                break
        
        # CISD CONFIRMATION - BULLISH
        while bull_pot_count > 0:
            p_val = bull_pot_val[0]
            p_idx = bull_pot_idx[0]
            
            if close[i] > p_val:
                len_check = i - p_idx
                if 0 <= len_check < 500:
                    lowest = close[i]
                    for k in range(len_check + 1):
                        if close[i - k] < lowest:
                            lowest = close[i - k]
                    
                    init = len_check + 1
                    bottom = 0.0
                    while init < min(500, i):
                        idx = i - init
                        if idx >= 0 and close[idx] > open_[idx]:
                            bottom = open_[idx]
                            init += 1
                        else:
                            break
                    
                    denom = p_val - bottom
                    if denom != 0 and abs(denom) > 1e-10:
                        if (p_val - lowest) / denom > tolerance:
                            cisd = 2
                            bull_pot_count = 0
                            break
                        else:
                            for k in range(bull_pot_count - 1):
                                bull_pot_val[k] = bull_pot_val[k + 1]
                                bull_pot_idx[k] = bull_pot_idx[k + 1]
                            bull_pot_count -= 1
                    else:
                        for k in range(bull_pot_count - 1):
                            bull_pot_val[k] = bull_pot_val[k + 1]
                            bull_pot_idx[k] = bull_pot_idx[k + 1]
                        bull_pot_count -= 1
                else:
                    for k in range(bull_pot_count - 1):
                        bull_pot_val[k] = bull_pot_val[k + 1]
                        bull_pot_idx[k] = bull_pot_idx[k + 1]
                    bull_pot_count -= 1
            else:
                break
        
        # TREND UPDATE
        if cisd == 1:
            trend = -1
        elif cisd == 2:
            trend = 1
        trend_out[i] = trend
        
        # SIGNAL GENERATION
        # Use last_signal to prevent consecutive signals of same type (like TV strategy.entry)
        prev_trend = trend_out[i-1] if i > 0 else 0
        if prev_trend <= 0 and trend > 0:
            if bar_sweep_bull > 0 and (i - bar_sweep_bull) < patience_bars:
                if last_signal != 1:  # Only if last signal wasn't bullish
                    bullsfp[i] = 1
                    last_signal = 1
        if prev_trend >= 0 and trend < 0:
            if bar_sweep_bear > 0 and (i - bar_sweep_bear) < patience_bars:
                if last_signal != -1:  # Only if last signal wasn't bearish
                    bearsfp[i] = 1
                    last_signal = -1
    
    return bullsfp, bearsfp, trend_out


# =============================================================================
# STRATEGY CLASS
# =============================================================================

class SwingFailurePatternAlgoAlpha(IStrategy):
    """
    Swing Failure Pattern Strategy (AlgoAlpha)
    
    Exact replication of Pine Script "Swing Failure Signals [AlgoAlpha]"
    with Numba optimization for Hyperopt performance.
    """

    INTERFACE_VERSION = 3

    # Fixed Parameters
    minimal_roi = {"0": 1.0}
    stoploss = -0.05
    timeframe = '15m'
    
    trailing_stop = False
    use_exit_signal = True
    use_custom_stoploss = True
    position_adjustment_enable = True
    max_open_trades = 3
    startup_candle_count: int = 200
    can_short = True

    # SFP Parameters
    pivot_length = IntParameter(5, 30, default=12, space='buy', optimize=True)
    max_pivot_edge = IntParameter(20, 100, default=50, space='buy', optimize=True)
    patience = IntParameter(3, 15, default=7, space='buy', optimize=True)
    tolerance = DecimalParameter(0.3, 0.9, default=0.7, decimals=1, space='buy', optimize=True)

    # Exit Parameters (Price-Based)
    tp1_pct = DecimalParameter(0.005, 0.10, default=0.01, decimals=3, space='sell', optimize=True)
    tp1_enabled = BooleanParameter(default=True, space='sell', optimize=True)
    tp1_amount = DecimalParameter(0.0, 100.0, default=50.0, space='sell', optimize=True)
    move_be_at_tp1 = BooleanParameter(default=True, space='sell', optimize=True)
    
    # Stoploss Optimization
    stoploss = DecimalParameter(-0.06, -0.01, default=-0.03, space='stoploss', optimize=True)
    
    # Logging Control
    enable_logging = BooleanParameter(default=False, space='custom', optimize=False)

    def __init__(self, config: dict) -> None:
        super().__init__(config)
    
    def bot_start(self, **kwargs) -> None:
        """Strategy startup. Apply leverage to stoploss."""
        config_leverage = self.config.get('leverage', 1.0)
        raw_stoploss = self.config.get('stoploss', -0.05)
        self.stoploss = raw_stoploss * config_leverage
        
        logger.info(
            f"SwingFailurePatternAlgoAlpha Configured:"
            f"\n  Leverage: {config_leverage}x"
            f"\n  Base Stoploss (Price): {raw_stoploss:.2%}"
            f"\n  Effective Stoploss (PnL): {self.stoploss:.2%}"
        )

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Calculate SFP indicators using Numba-optimized functions."""
        
        length = self.pivot_length.value
        max_edge = self.max_pivot_edge.value
        patience_bars = self.patience.value
        tol = self.tolerance.value
        
        # Get numpy arrays
        high = dataframe['high'].values.astype(np.float64)
        low = dataframe['low'].values.astype(np.float64)
        open_ = dataframe['open'].values.astype(np.float64)
        close = dataframe['close'].values.astype(np.float64)
        # Calculate pivots (Numba)
        pivot_high = _pivot_high_numba(high, length, length)
        pivot_low = _pivot_low_numba(low, length, length)
        
        # Calculate SFP signals (Numba)
        bullsfp, bearsfp, trend = _calculate_sfp_signals(
            high, low, open_, close,
            pivot_high, pivot_low,
            length, max_edge, patience_bars, tol
        )
        
        # Store results
        dataframe['sfp_bullish'] = bullsfp
        dataframe['sfp_bearish'] = bearsfp
        dataframe['sfp_trend'] = trend
        dataframe['pivot_high'] = pivot_high
        dataframe['pivot_low'] = pivot_low
        
        # Logging
        if self.enable_logging.value:
            logger.info(
                f"SFP AlgoAlpha for {metadata['pair']}: "
                f"Bullish SFP={bullsfp.sum()}, Bearish SFP={bearsfp.sum()}"
            )
            
            # Log each signal with date and price
            # Skip only if index is within startup candle count to avoid logging warmup signals
            start_idx = self.startup_candle_count
            for i in range(len(dataframe)):
                if i < start_idx:
                    continue
                    
                if bullsfp[i] == 1:
                    dt = dataframe.index[i] if hasattr(dataframe.index[i], 'strftime') else dataframe['date'].iloc[i]
                    price = close[i]
                    logger.info(f"  📈 LONG  Signal @ {dt} | Price: {price:.2f}")
                if bearsfp[i] == 1:
                    dt = dataframe.index[i] if hasattr(dataframe.index[i], 'strftime') else dataframe['date'].iloc[i]
                    price = close[i]
                    logger.info(f"  📉 SHORT Signal @ {dt} | Price: {price:.2f}")
        
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Generate entry signals based on SFP."""
        dataframe.loc[dataframe['sfp_bullish'] == 1, 'enter_long'] = 1
        dataframe.loc[dataframe['sfp_bearish'] == 1, 'enter_short'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Exit on opposite SFP signal."""
        dataframe.loc[dataframe['sfp_bearish'] == 1, 'exit_long'] = 1
        dataframe.loc[dataframe['sfp_bullish'] == 1, 'exit_short'] = 1
        return dataframe
    
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: str,
                 side: str, **kwargs) -> float:
        return self.config.get('leverage', 10.0)
    
    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        """Breakeven logic - move SL to entry when TP1 hit."""
        if not self.move_be_at_tp1.value:
            return 1
        
        if trade.is_short:
            extremum = trade.min_rate if trade.min_rate else current_rate
            price_move = (trade.open_rate - extremum) / trade.open_rate
        else:
            extremum = trade.max_rate if trade.max_rate else current_rate
            price_move = (extremum - trade.open_rate) / trade.open_rate
        
        if price_move >= self.tp1_pct.value:
            if trade.is_short:
                sl_rel = (current_rate - trade.open_rate) / current_rate
            else:
                sl_rel = (trade.open_rate - current_rate) / current_rate
            if sl_rel < 0:
                return sl_rel
        
        return 1
    
    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float,
                              min_stake: float | None, max_stake: float,
                              current_entry_rate: float, current_exit_rate: float,
                              current_entry_profit: float, current_exit_profit: float,
                              **kwargs) -> float | None | tuple[float | None, str | None]:
        """Partial TP1 based on price movement."""
        if trade.is_short:
            extremum = trade.min_rate if trade.min_rate else current_rate
            price_move = (trade.open_rate - extremum) / trade.open_rate
        else:
            extremum = trade.max_rate if trade.max_rate else current_rate
            price_move = (extremum - trade.open_rate) / trade.open_rate
        
        if price_move > self.tp1_pct.value:
            if not self.tp1_enabled.value or self.tp1_amount.value == 0:
                return None
            
            exit_side = 'buy' if trade.is_short else 'sell'
            partial_exits = [
                o for o in trade.orders 
                if o.side == exit_side and o.status == 'closed' and o.ft_order_side == 'exit'
            ]
            
            if len(partial_exits) == 0:
                close_amount = trade.amount * (self.tp1_amount.value / 100.0)
                sell_value = close_amount * current_rate
                if self.enable_logging.value:
                    logger.info(
                        f"TP1 {trade.pair} ({'SHORT' if trade.is_short else 'LONG'}): "
                        f"price {price_move:.2%}, closing {self.tp1_amount.value:.0f}%"
                    )
                return (-sell_value, "TP1_partial")
        
        return None
