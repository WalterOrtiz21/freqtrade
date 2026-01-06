"""
Swing Failure Pattern Strategy (LuxAlgo)
=========================================
Port of the LuxAlgo 'Swing Failure Pattern [LuxAlgo]' indicator to Freqtrade.

Key Logic:
- Bearish SFP: high > swing_high AND open < swing_high AND close < swing_high
- Bullish SFP: low < swing_low AND open > swing_low AND close > swing_low
- Confirmation: Price breaks the "Opposite Point" (lowest low / highest high between pivot and SFP)

Features:
- Configurable require_confirmation toggle (entry on label vs confirmation)
- Price-based TP1 and Break Even
- Numba-optimized calculations
- Logging control
"""

import numpy as np
import logging
from datetime import datetime
from typing import Optional

from freqtrade.strategy import IStrategy, DecimalParameter, IntParameter, BooleanParameter
from freqtrade.persistence import Trade
from pandas import DataFrame

from numba import njit

logger = logging.getLogger(__name__)

# ============================================================================
# NUMBA OPTIMIZED FUNCTIONS
# ============================================================================

@njit(cache=False)
def _pivot_high_numba(high: np.ndarray, left: int, right: int) -> np.ndarray:
    """Detect pivot highs (symmetric lookback)."""
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
    """Detect pivot lows (symmetric lookback)."""
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
    require_confirmation: bool
) -> tuple:
    """
    Calculate SFP signals based on LuxAlgo logic.
    
    Key insight: Confirmation can ONLY happen on bars AFTER the SFP detection bar.
    """
    n = len(high)
    bullsfp = np.zeros(n, dtype=np.int64)
    bearsfp = np.zeros(n, dtype=np.int64)
    
    # Bearish SFP state
    bear_active = False
    bear_swing_prc = 0.0
    bear_oppos_prc = 0.0
    bear_detect_bar = 0  # Bar where SFP was detected
    
    # Bullish SFP state
    bull_active = False
    bull_swing_prc = 0.0
    bull_oppos_prc = 0.0
    bull_detect_bar = 0  # Bar where SFP was detected
    
    # Track last known swing high/low
    last_swing_high_prc = 0.0
    last_swing_high_bix = 0
    last_swing_low_prc = 0.0
    last_swing_low_bix = 0
    
    # Track last signal for alternation
    last_signal = 0  # 1 = bullish, -1 = bearish
    
    for i in range(length + 1, n):
        # Update swing high if detected
        if not np.isnan(pivot_high_arr[i]):
            last_swing_high_prc = pivot_high_arr[i]
            last_swing_high_bix = i - 1
        
        # Update swing low if detected
        if not np.isnan(pivot_low_arr[i]):
            last_swing_low_prc = pivot_low_arr[i]
            last_swing_low_bix = i - 1
        
        # ========== BEARISH SFP DETECTION ==========
        sw_h = last_swing_high_prc
        bx_h = last_swing_high_bix
        
        if sw_h > 0 and high[i] > sw_h and open_[i] < sw_h and close[i] < sw_h:
            # NEW SFP detected on THIS bar
            opposL = sw_h
            for j in range(1, max(1, i - bx_h)):
                if low[i - j] < opposL:
                    opposL = low[i - j]
            
            bear_active = True
            bear_swing_prc = sw_h
            bear_oppos_prc = opposL
            bear_detect_bar = i  # Track detection bar
            
            # If NOT requiring confirmation, signal IMMEDIATELY
            if not require_confirmation and last_signal != -1:
                bearsfp[i] = 1
                last_signal = -1
                bear_active = False
        
        # Check for confirmation (ONLY on bars AFTER detection)
        if bear_active and i > bear_detect_bar:
            if close[i] < bear_oppos_prc:
                if require_confirmation and last_signal != -1:
                    bearsfp[i] = 1
                    last_signal = -1
                bear_active = False
            
            # Invalidation
            if close[i] > bear_swing_prc or i - bear_detect_bar > 500:
                bear_active = False
        
        # ========== BULLISH SFP DETECTION ==========
        sw_l = last_swing_low_prc
        bx_l = last_swing_low_bix
        
        if sw_l > 0 and low[i] < sw_l and open_[i] > sw_l and close[i] > sw_l:
            # NEW SFP detected on THIS bar
            opposH = sw_l
            for j in range(1, max(1, i - bx_l)):
                if high[i - j] > opposH:
                    opposH = high[i - j]
            
            bull_active = True
            bull_swing_prc = sw_l
            bull_oppos_prc = opposH
            bull_detect_bar = i  # Track detection bar
            
            # If NOT requiring confirmation, signal IMMEDIATELY
            if not require_confirmation and last_signal != 1:
                bullsfp[i] = 1
                last_signal = 1
                bull_active = False
        
        # Check for confirmation (ONLY on bars AFTER detection)
        if bull_active and i > bull_detect_bar:
            if close[i] > bull_oppos_prc:
                if require_confirmation and last_signal != 1:
                    bullsfp[i] = 1
                    last_signal = 1
                bull_active = False
            
            # Invalidation
            if close[i] < bull_swing_prc or i - bull_detect_bar > 500:
                bull_active = False
    
    return bullsfp, bearsfp


# ============================================================================
# STRATEGY CLASS
# ============================================================================

class SwingFailurePatternLuxAlgo(IStrategy):
    """
    Swing Failure Pattern Strategy based on LuxAlgo indicator.
    
    Entry: On SFP detection (optionally wait for confirmation)
    Exit: On opposite SFP signal
    """
    
    INTERFACE_VERSION = 3
    
    # Strategy settings
    timeframe = '1h'
    can_short = True
    stoploss = -0.03
    minimal_roi = {"0": 99}
    process_only_new_candles = True
    use_exit_signal = True
    startup_candle_count = 200
    
    # ========== CONFIGURABLE PARAMETERS ==========
    
    # SFP Detection
    swing_length = IntParameter(3, 21, default=5, space='buy', optimize=True)
    require_confirmation = BooleanParameter(default=True, space='buy', optimize=True)
    
    # Take Profit / Break Even
    tp1_enabled = BooleanParameter(default=True, space='sell', optimize=True)
    tp1_pct = DecimalParameter(0.01, 0.10, default=0.02, space='sell', optimize=True)
    tp1_amount = DecimalParameter(0.0, 100.0, default=50.0, space='sell', optimize=True)
    move_be_at_tp1 = BooleanParameter(default=True, space='sell', optimize=True)
    
    # Stoploss Optimization
    stoploss = DecimalParameter(-0.10, -0.01, default=-0.03, space='protection', optimize=True)
    
    # Logging Control
    enable_logging = BooleanParameter(default=False, space='buy', optimize=False)
    
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        logger.info("SwingFailurePatternLuxAlgo Configured:")
        logger.info(f"  swing_length={self.swing_length.value}")
        logger.info(f"  require_confirmation={self.require_confirmation.value}")
        logger.info(f"  tp1_pct={self.tp1_pct.value}")
        logger.info(f"  stoploss={self.stoploss.value}")
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Calculate SFP indicators using Numba-optimized functions."""
        
        length = self.swing_length.value
        require_conf = self.require_confirmation.value
        
        # Extract numpy arrays
        high = dataframe['high'].values.astype(np.float64)
        low = dataframe['low'].values.astype(np.float64)
        open_ = dataframe['open'].values.astype(np.float64)
        close = dataframe['close'].values.astype(np.float64)
        
        # Calculate pivots (LuxAlgo uses right=1)
        pivot_high = _pivot_high_numba(high, length, 1)
        pivot_low = _pivot_low_numba(low, length, 1)
        
        # Calculate SFP signals
        bullsfp, bearsfp = _calculate_sfp_signals(
            high, low, open_, close,
            pivot_high, pivot_low,
            length, require_conf
        )
        
        # Store in dataframe
        dataframe['sfp_bullish'] = bullsfp
        dataframe['sfp_bearish'] = bearsfp
        
        # Logging
        if self.enable_logging.value:
            bull_count = int(np.sum(bullsfp))
            bear_count = int(np.sum(bearsfp))
            logger.info(
                f"SFP LuxAlgo for {metadata['pair']}: "
                f"Bullish SFP={bull_count}, Bearish SFP={bear_count} "
                f"(confirmation={'ON' if require_conf else 'OFF'})"
            )
            
            # Log each signal (skip startup period)
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
