"""
ICT (Inner Circle Trader) Concepts - Numba Optimized Module
============================================================

CALCULATION MODULE ONLY - No trading logic

This module provides Numba-optimized calculations for ICT concepts:
- Market Structure (BoS/MSS/CHoCH)
- Order Blocks (Bullish/Bearish zones)
- Breaker Blocks
- Fair Value Gaps (FVG zones)
- Liquidity Sweeps
- Strong Highs/Lows
- Premium/Discount Zones
- OTE (Optimal Trade Entry) Fibonacci levels
- Session & Macro time filters
- Equal Highs/Lows (liquidity targets)

Trading logic (Model 2022, Unicorn, etc.) should be in the strategy file.

Author: Ported from Pine Script by Trading IQ
License: Mozilla Public License 2.0
"""

import numpy as np
import pandas as pd
from numba import jit
from typing import Tuple


# =============================================================================
# HELPER KERNELS
# =============================================================================

@jit(nopython=True, cache=True)
def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Calculate ATR (Average True Range)."""
    n = len(high)
    tr = np.zeros(n)
    atr = np.zeros(n)
    
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, hc, lc)
    
    if n >= period:
        atr[period-1] = np.mean(tr[1:period])
        for i in range(period, n):
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    
    return atr


@jit(nopython=True, cache=True)
def _find_lowest_low(low: np.ndarray, start: int, end: int) -> int:
    """Find index of lowest low in range [start, end)."""
    min_val = 1e15
    min_idx = -1
    for i in range(start, min(end, len(low))):
        if low[i] < min_val:
            min_val = low[i]
            min_idx = i
    return min_idx


@jit(nopython=True, cache=True)
def _find_highest_high(high: np.ndarray, start: int, end: int) -> int:
    """Find index of highest high in range [start, end)."""
    max_val = -1.0
    max_idx = -1
    for i in range(start, min(end, len(high))):
        if high[i] > max_val:
            max_val = high[i]
            max_idx = i
    return max_idx


# =============================================================================
# SWING DETECTION KERNEL
# =============================================================================

@jit(nopython=True, cache=True)
def _swing_detection_kernel(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr: np.ndarray,
    n: int,
    atr_mult: float = 2.0,
    buffer: float = 0.0
) -> Tuple[
    np.ndarray, np.ndarray,  # swing_high, swing_low
    np.ndarray, np.ndarray,  # swing_high_idx, swing_low_idx
    np.ndarray               # direction
]:
    """ATR-based ZigZag swing detection."""
    swing_high = np.full(n, np.nan)
    swing_low = np.full(n, np.nan)
    swing_high_idx = np.zeros(n, dtype=np.int64)
    swing_low_idx = np.zeros(n, dtype=np.int64)
    direction = np.zeros(n, dtype=np.int8)
    
    current_dir = 0
    current_high = high[0]
    current_high_idx = 0
    current_low = low[0]
    current_low_idx = 0
    prev_swing_high = np.nan
    prev_swing_low = np.nan
    
    for i in range(1, n):
        threshold = atr[i] * atr_mult + buffer if atr[i] > 0 else 0.001
        
        if current_dir >= 0:
            if high[i] > current_high:
                current_high = high[i]
                current_high_idx = i
            if close[i] < current_high - threshold:
                prev_swing_high = current_high
                current_dir = -1
                current_low = low[i]
                current_low_idx = i
        
        if current_dir <= 0:
            if low[i] < current_low:
                current_low = low[i]
                current_low_idx = i
            if close[i] > current_low + threshold:
                prev_swing_low = current_low
                current_dir = 1
                current_high = high[i]
                current_high_idx = i
        
        swing_high[i] = prev_swing_high
        swing_low[i] = prev_swing_low
        swing_high_idx[i] = current_high_idx
        swing_low_idx[i] = current_low_idx
        direction[i] = current_dir
    
    return swing_high, swing_low, swing_high_idx, swing_low_idx, direction


# =============================================================================
# MARKET STRUCTURE KERNEL (BoS/MSS/CHoCH)
# =============================================================================

@jit(nopython=True, cache=True)
def _market_structure_kernel(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    swing_high: np.ndarray,
    swing_low: np.ndarray,
    swing_high_idx: np.ndarray,
    swing_low_idx: np.ndarray,
    n: int,
    lookback: int = 20
) -> Tuple[
    np.ndarray, np.ndarray,  # bos_bullish, bos_bearish
    np.ndarray, np.ndarray,  # mss_bullish, mss_bearish (CHoCH)
    np.ndarray               # structure_direction
]:
    """Detect Break of Structure and Market Structure Shift."""
    bos_bull = np.zeros(n, dtype=np.int8)
    bos_bear = np.zeros(n, dtype=np.int8)
    mss_bull = np.zeros(n, dtype=np.int8)
    mss_bear = np.zeros(n, dtype=np.int8)
    structure_dir = np.zeros(n, dtype=np.int8)
    
    current_trend = 0
    
    for i in range(lookback, n):
        prev_swing_high = swing_high[i]
        prev_swing_low = swing_low[i]
        
        if np.isnan(prev_swing_high) or np.isnan(prev_swing_low):
            structure_dir[i] = current_trend
            continue
        
        # Bullish break above swing high
        if close[i] > prev_swing_high:
            if current_trend == -1:
                mss_bull[i] = 1  # CHoCH
            else:
                bos_bull[i] = 1  # BoS
            current_trend = 1
        
        # Bearish break below swing low
        if close[i] < prev_swing_low:
            if current_trend == 1:
                mss_bear[i] = 1  # CHoCH
            else:
                bos_bear[i] = 1  # BoS
            current_trend = -1
        
        structure_dir[i] = current_trend
    
    return bos_bull, bos_bear, mss_bull, mss_bear, structure_dir


# =============================================================================
# ORDER BLOCKS KERNEL
# =============================================================================

@jit(nopython=True, cache=True)
def _order_blocks_kernel(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    bos_bull: np.ndarray,
    bos_bear: np.ndarray,
    mss_bull: np.ndarray,
    mss_bear: np.ndarray,
    n: int,
    lookback: int = 10
) -> Tuple[
    np.ndarray, np.ndarray,  # ob_bull_top, ob_bull_btm
    np.ndarray, np.ndarray   # ob_bear_top, ob_bear_btm
]:
    """Detect Order Blocks on structure breaks."""
    ob_bull_top = np.full(n, np.nan)
    ob_bull_btm = np.full(n, np.nan)
    ob_bear_top = np.full(n, np.nan)
    ob_bear_btm = np.full(n, np.nan)
    
    for i in range(lookback, n):
        # Bullish OB on bullish structure break
        if bos_bull[i] == 1 or mss_bull[i] == 1:
            ob_idx = _find_lowest_low(low, max(0, i - lookback), i)
            if ob_idx >= 0:
                ob_bull_top[i] = high[ob_idx]
                ob_bull_btm[i] = low[ob_idx]
        
        # Bearish OB on bearish structure break
        if bos_bear[i] == 1 or mss_bear[i] == 1:
            ob_idx = _find_highest_high(high, max(0, i - lookback), i)
            if ob_idx >= 0:
                ob_bear_top[i] = high[ob_idx]
                ob_bear_btm[i] = low[ob_idx]
    
    return ob_bull_top, ob_bull_btm, ob_bear_top, ob_bear_btm


# =============================================================================
# FAIR VALUE GAP (FVG) KERNEL
# =============================================================================

@jit(nopython=True, cache=True)
def _fvg_kernel(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    open_: np.ndarray,
    atr: np.ndarray,
    n: int,
    min_gap_atr: float = 0.1  # Minimum gap size as ATR fraction
) -> Tuple[
    np.ndarray, np.ndarray,  # fvg_bull_top, fvg_bull_btm
    np.ndarray, np.ndarray   # fvg_bear_top, fvg_bear_btm
]:
    """
    Detect Fair Value Gaps (3-candle imbalance).
    
    Practical ICT FVG Definition:
    - Look for strong displacement move (middle candle)
    - Gap zone exists between:
      * Bullish: candle[i-2] high and candle[i] low (if there's space)
      * Bearish: candle[i-2] low and candle[i] high
    
    This version uses displacement detection rather than strict gaps,
    which works better for crypto's 24/7 continuous trading.
    """
    fvg_bull_top = np.full(n, np.nan)
    fvg_bull_btm = np.full(n, np.nan)
    fvg_bear_top = np.full(n, np.nan)
    fvg_bear_btm = np.full(n, np.nan)
    
    for i in range(2, n):
        curr_atr = atr[i] if atr[i] > 0 else 0.001
        min_gap = curr_atr * min_gap_atr
        
        # ===== BULLISH FVG =====
        # Strong bullish move: middle candle closes strongly up
        mid_body = close[i-1] - open_[i-1]
        is_bullish_displacement = mid_body > curr_atr * 0.5  # Strong up candle
        
        if is_bullish_displacement:
            # Check if there's unfilled space between candle 1 high and candle 3 low
            gap_top = low[i]        # Current candle's low
            gap_btm = high[i-2]     # Candle 2 bars ago high
            gap_size = gap_top - gap_btm
            
            if gap_size > min_gap:
                fvg_bull_top[i] = gap_top
                fvg_bull_btm[i] = gap_btm
            elif gap_size > 0:
                # Small but valid gap
                fvg_bull_top[i] = gap_top
                fvg_bull_btm[i] = gap_btm
        
        # ===== BEARISH FVG =====
        mid_body_bear = open_[i-1] - close[i-1]
        is_bearish_displacement = mid_body_bear > curr_atr * 0.5
        
        if is_bearish_displacement:
            gap_top = low[i-2]      # Candle 2 bars ago low
            gap_btm = high[i]       # Current candle's high
            gap_size = gap_top - gap_btm
            
            if gap_size > min_gap:
                fvg_bear_top[i] = gap_top
                fvg_bear_btm[i] = gap_btm
            elif gap_size > 0:
                fvg_bear_top[i] = gap_top
                fvg_bear_btm[i] = gap_btm
    
    return fvg_bull_top, fvg_bull_btm, fvg_bear_top, fvg_bear_btm


# =============================================================================
# ACTIVE ZONES KERNEL (Track unmitigated OBs/FVGs)
# =============================================================================

@jit(nopython=True, cache=True)
def _active_zones_kernel(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    ob_bull_top: np.ndarray, ob_bull_btm: np.ndarray,
    ob_bear_top: np.ndarray, ob_bear_btm: np.ndarray,
    fvg_bull_top: np.ndarray, fvg_bull_btm: np.ndarray,
    fvg_bear_top: np.ndarray, fvg_bear_btm: np.ndarray,
    n: int,
    max_zones: int = 50 
) -> Tuple[
    np.ndarray, np.ndarray,  # active_ob_bull_top/btm
    np.ndarray, np.ndarray,  # active_ob_bear_top/btm
    np.ndarray, np.ndarray,  # in_bull_ob, in_bear_ob
    np.ndarray, np.ndarray,  # active_fvg_bull_top/btm
    np.ndarray, np.ndarray,  # active_fvg_bear_top/btm
    np.ndarray, np.ndarray   # in_bull_fvg, in_bear_fvg
]:
    """Track active (unmitigated) OBs and FVGs."""
    act_ob_bull_top = np.full(n, np.nan)
    act_ob_bull_btm = np.full(n, np.nan)
    act_ob_bear_top = np.full(n, np.nan)
    act_ob_bear_btm = np.full(n, np.nan)
    in_bull_ob = np.zeros(n, dtype=np.int8)
    in_bear_ob = np.zeros(n, dtype=np.int8)
    
    act_fvg_bull_top = np.full(n, np.nan)
    act_fvg_bull_btm = np.full(n, np.nan)
    act_fvg_bear_top = np.full(n, np.nan)
    act_fvg_bear_btm = np.full(n, np.nan)
    in_bull_fvg = np.zeros(n, dtype=np.int8)
    in_bear_fvg = np.zeros(n, dtype=np.int8)
    
    # Track zones with simple arrays. Size is max concurrent zones.
    ob_bull_zones = np.zeros((max_zones, 2))
    ob_bear_zones = np.zeros((max_zones, 2))
    fvg_bull_zones = np.zeros((max_zones, 2))
    fvg_bear_zones = np.zeros((max_zones, 2))
    
    for i in range(n):
        # 1. ADD NEW ZONES (Find empty slot)
        if not np.isnan(ob_bull_top[i]):
            for k in range(max_zones):
                if ob_bull_zones[k, 0] == 0:
                    ob_bull_zones[k, 0] = ob_bull_top[i]
                    ob_bull_zones[k, 1] = ob_bull_btm[i]
                    break
        
        if not np.isnan(ob_bear_top[i]):
            for k in range(max_zones):
                if ob_bear_zones[k, 0] == 0:
                    ob_bear_zones[k, 0] = ob_bear_top[i]
                    ob_bear_zones[k, 1] = ob_bear_btm[i]
                    break
            
        if not np.isnan(fvg_bull_top[i]):
            for k in range(max_zones):
                if fvg_bull_zones[k, 0] == 0:
                    fvg_bull_zones[k, 0] = fvg_bull_top[i]
                    fvg_bull_zones[k, 1] = fvg_bull_btm[i]
                    break
            
        if not np.isnan(fvg_bear_top[i]):
            for k in range(max_zones):
                if fvg_bear_zones[k, 0] == 0:
                    fvg_bear_zones[k, 0] = fvg_bear_top[i]
                    fvg_bear_zones[k, 1] = fvg_bear_btm[i]
                    break
        
        # 2. CHECK PRICE IN ZONES & MITIGATION
        # Loop through ALL slots, skip empty ones
        
        # Bull OBs
        for j in range(max_zones):
            if ob_bull_zones[j, 0] != 0:
                # Overlap
                if low[i] <= ob_bull_zones[j, 0]:
                    in_bull_ob[i] = 1
                    act_ob_bull_top[i] = ob_bull_zones[j, 0]
                    act_ob_bull_btm[i] = ob_bull_zones[j, 1]
                # Mitigation (Close/Low below bottom? Usually OB is invalid if closed below or wicked through)
                # Standard ICT: Wick below is mitigation/invalidation
                if low[i] <= ob_bull_zones[j, 1]:
                    ob_bull_zones[j, 0] = 0
        
        # Bear OBs
        for j in range(max_zones):
            if ob_bear_zones[j, 0] != 0:
                # Overlap
                if high[i] >= ob_bear_zones[j, 1]:
                    in_bear_ob[i] = 1
                    act_ob_bear_top[i] = ob_bear_zones[j, 0]
                    act_ob_bear_btm[i] = ob_bear_zones[j, 1]
                # Mitigation
                if high[i] >= ob_bear_zones[j, 0]:
                    ob_bear_zones[j, 0] = 0
        
        # Bull FVGs
        for j in range(max_zones):
            if fvg_bull_zones[j, 0] != 0:
                # Overlap (Low <= Top)
                if low[i] <= fvg_bull_zones[j, 0]:
                    in_bull_fvg[i] = 1
                    act_fvg_bull_top[i] = fvg_bull_zones[j, 0]
                    act_fvg_bull_btm[i] = fvg_bull_zones[j, 1]
                # Mitigation (Low <= Bottom)
                if low[i] <= fvg_bull_zones[j, 1]:
                    fvg_bull_zones[j, 0] = 0
        
        # Bear FVGs
        for j in range(max_zones):
            if fvg_bear_zones[j, 0] != 0:
                # Overlap (High >= Bottom)
                if high[i] >= fvg_bear_zones[j, 1]:
                    in_bear_fvg[i] = 1
                    act_fvg_bear_top[i] = fvg_bear_zones[j, 0]
                    act_fvg_bear_btm[i] = fvg_bear_zones[j, 1]
                # Mitigation (High >= Top)
                if high[i] >= fvg_bear_zones[j, 0]:
                    fvg_bear_zones[j, 0] = 0
    
    return (act_ob_bull_top, act_ob_bull_btm, act_ob_bear_top, act_ob_bear_btm,
            in_bull_ob, in_bear_ob,
            act_fvg_bull_top, act_fvg_bull_btm, act_fvg_bear_top, act_fvg_bear_btm,
            in_bull_fvg, in_bear_fvg)


# =============================================================================
# LIQUIDITY SWEEP KERNEL
# =============================================================================

@jit(nopython=True, cache=True)
def _liquidity_sweep_kernel(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    swing_high: np.ndarray,
    swing_low: np.ndarray,
    atr: np.ndarray,
    n: int,
    atr_mult: float = 5.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Detect liquidity sweeps (wick above/below swing + rejection)."""
    sweep_high = np.zeros(n, dtype=np.int8)
    sweep_low = np.zeros(n, dtype=np.int8)
    
    for i in range(1, n):
        if np.isnan(swing_high[i]) or np.isnan(swing_low[i]):
            continue
        
        threshold = atr[i] * atr_mult if atr[i] > 0 else 0.001
        
        # Sweep high: wick above, close below
        if high[i] > swing_high[i] and close[i] < swing_high[i]:
            sweep_high[i] = 1
        
        # Sweep low: wick below, close above
        if low[i] < swing_low[i] and close[i] > swing_low[i]:
            sweep_low[i] = 1
    
    return sweep_high, sweep_low


# =============================================================================
# PREMIUM/DISCOUNT KERNEL
# =============================================================================

@jit(nopython=True, cache=True)
def _premium_discount_kernel(
    close: np.ndarray,
    swing_high: np.ndarray,
    swing_low: np.ndarray,
    n: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate equilibrium and premium/discount zones."""
    equilibrium = np.full(n, np.nan)
    in_premium = np.zeros(n, dtype=np.int8)
    in_discount = np.zeros(n, dtype=np.int8)
    
    for i in range(n):
        if np.isnan(swing_high[i]) or np.isnan(swing_low[i]):
            continue
        
        eq = (swing_high[i] + swing_low[i]) / 2.0
        equilibrium[i] = eq
        
        if close[i] > eq:
            in_premium[i] = 1
        else:
            in_discount[i] = 1
    
    return equilibrium, in_premium, in_discount


# =============================================================================
# STRONG LEVELS KERNEL
# =============================================================================

@jit(nopython=True, cache=True)
def _strong_levels_kernel(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    bos_bull: np.ndarray,
    bos_bear: np.ndarray,
    swing_high_idx: np.ndarray,
    swing_low_idx: np.ndarray,
    n: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Identify strong highs/lows that caused structure breaks."""
    strong_high = np.full(n, np.nan)
    strong_high_idx = np.zeros(n, dtype=np.int64)
    strong_low = np.full(n, np.nan)
    strong_low_idx_out = np.zeros(n, dtype=np.int64)
    
    current_strong_high = np.nan
    current_strong_high_idx = 0
    current_strong_low = np.nan
    current_strong_low_idx = 0
    
    for i in range(n):
        if bos_bull[i] == 1:
            sh_idx = int(swing_high_idx[i])
            if sh_idx >= 0 and sh_idx < n:
                current_strong_high = high[sh_idx]
                current_strong_high_idx = sh_idx
        
        if bos_bear[i] == 1:
            sl_idx = int(swing_low_idx[i])
            if sl_idx >= 0 and sl_idx < n:
                current_strong_low = low[sl_idx]
                current_strong_low_idx = sl_idx
        
        strong_high[i] = current_strong_high
        strong_high_idx[i] = current_strong_high_idx
        strong_low[i] = current_strong_low
        strong_low_idx_out[i] = current_strong_low_idx
    
    return strong_high, strong_high_idx, strong_low, strong_low_idx_out


# =============================================================================
# BREAKER BLOCKS KERNEL
# =============================================================================

@jit(nopython=True, cache=True)
def _breaker_block_kernel(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    ob_bull_top: np.ndarray, ob_bull_btm: np.ndarray,
    ob_bear_top: np.ndarray, ob_bear_btm: np.ndarray,
    n: int,
    max_zones: int = 50
) -> Tuple[
    np.ndarray, np.ndarray,  # bb_bull_top, bb_bull_btm
    np.ndarray, np.ndarray,  # bb_bear_top, bb_bear_btm
    np.ndarray, np.ndarray   # in_bb_bull, in_bb_bear
]:
    """Detect Breaker Blocks (mitigated OBs that become S/R)."""
    bb_bull_top = np.full(n, np.nan)
    bb_bull_btm = np.full(n, np.nan)
    bb_bear_top = np.full(n, np.nan)
    bb_bear_btm = np.full(n, np.nan)
    in_bb_bull = np.zeros(n, dtype=np.int8)
    in_bb_bear = np.zeros(n, dtype=np.int8)
    
    # Active Breaker Blocks tracking
    bb_bull_zones = np.zeros((max_zones, 2))
    bb_bear_zones = np.zeros((max_zones, 2))
    
    # Last OBs for conversion
    last_bear_ob_top = np.nan
    last_bear_ob_btm = np.nan
    last_bull_ob_top = np.nan
    last_bull_ob_btm = np.nan
    
    for i in range(n):
        if not np.isnan(ob_bear_top[i]):
            last_bear_ob_top = ob_bear_top[i]
            last_bear_ob_btm = ob_bear_btm[i]
        
        if not np.isnan(ob_bull_top[i]):
            last_bull_ob_top = ob_bull_top[i]
            last_bull_ob_btm = ob_bull_btm[i]
        
        # 1. Detect NEW Breaker Blocks (conversion)
        
        # Bear OB mitigated (price closes above) -> becomes Bull BB
        if not np.isnan(last_bear_ob_top) and close[i] > last_bear_ob_top:
            # Add to active zones
            for k in range(max_zones):
                if bb_bull_zones[k, 0] == 0:
                    bb_bull_zones[k, 0] = last_bear_ob_top
                    bb_bull_zones[k, 1] = last_bear_ob_btm
                    break
            
            # Emit signal for creation
            bb_bull_top[i] = last_bear_ob_top
            bb_bull_btm[i] = last_bear_ob_btm
            last_bear_ob_top = np.nan # Reset processed OB
        
        # Bull OB mitigated (price closes below) -> becomes Bear BB
        if not np.isnan(last_bull_ob_btm) and close[i] < last_bull_ob_btm:
            # Add to active zones
            for k in range(max_zones):
                if bb_bear_zones[k, 0] == 0:
                    bb_bear_zones[k, 0] = last_bull_ob_top
                    bb_bear_zones[k, 1] = last_bull_ob_btm
                    break
            
            # Emit signal for creation
            bb_bear_top[i] = last_bull_ob_top
            bb_bear_btm[i] = last_bull_ob_btm
            last_bull_ob_btm = np.nan # Reset processed OB
            
        # 2. Check Price in Active Zones & Mitigation (Invalidation)
        
        # Bull BBs (Support)
        for j in range(max_zones):
            if bb_bull_zones[j, 0] != 0:
                # Retest: Low enters zone (top to btm)
                if low[i] <= bb_bull_zones[j, 0] and high[i] >= bb_bull_zones[j, 1]:
                    in_bb_bull[i] = 1
                    # Update output arrays to show ACTIVE zone levels
                    if np.isnan(bb_bull_top[i]): # Don't overwrite creation signal
                        bb_bull_top[i] = bb_bull_zones[j, 0]
                        bb_bull_btm[i] = bb_bull_zones[j, 1]

                # Invalidation: Close below bottom? Or wick? 
                # Standard: Close below invalidates support
                if close[i] < bb_bull_zones[j, 1]:
                    bb_bull_zones[j, 0] = 0
                    
        # Bear BBs (Resistance)
        for j in range(max_zones):
            if bb_bear_zones[j, 0] != 0:
                # Retest: High enters zone
                if high[i] >= bb_bear_zones[j, 1] and low[i] <= bb_bear_zones[j, 0]:
                    in_bb_bear[i] = 1
                    # Update output arrays
                    if np.isnan(bb_bear_top[i]):
                        bb_bear_top[i] = bb_bear_zones[j, 0]
                        bb_bear_btm[i] = bb_bear_zones[j, 1]

                # Invalidation: Close above top
                if close[i] > bb_bear_zones[j, 0]:
                    bb_bear_zones[j, 0] = 0
    
    return bb_bull_top, bb_bull_btm, bb_bear_top, bb_bear_btm, in_bb_bull, in_bb_bear


# =============================================================================
# REJECTION BLOCKS KERNEL
# =============================================================================

@jit(nopython=True, cache=True)
def _rejection_blocks_kernel(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    open_: np.ndarray,
    atr: np.ndarray,
    n: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Detect Rejection Blocks.
    
    Definition:
    - Long wick relative to body.
    - Captures the concept of price rejecting a level strongly.
    - Used as support/resistance zones.
    """
    rb_bull_top = np.full(n, np.nan)
    rb_bull_btm = np.full(n, np.nan)
    rb_bear_top = np.full(n, np.nan)
    rb_bear_btm = np.full(n, np.nan)
    
    for i in range(n):
        curr_atr = atr[i] if atr[i] > 0 else 0.001
        
        # Bullish Rejection (Low Wick)
        # Condition: Wick length > Body length AND Wick > ATR/3 (significant size)
        body_low = min(open_[i], close[i])
        body_high = max(open_[i], close[i])
        low_wick = body_low - low[i]
        
        if low_wick > curr_atr * 0.33:
            rb_bull_top[i] = body_low
            rb_bull_btm[i] = low[i]
            
        # Bearish Rejection (High Wick)
        high_wick = high[i] - body_high
        
        if high_wick > curr_atr * 0.33:
            rb_bear_top[i] = high[i]
            rb_bear_btm[i] = body_high
            
    return rb_bull_top, rb_bull_btm, rb_bear_top, rb_bear_btm


# =============================================================================
# POWER OF THREE (Po3) KERNEL
# =============================================================================

@jit(nopython=True, cache=True)
def _po3_kernel(
    open_: np.ndarray,
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    hours: np.ndarray,
    n: int,
    session_start_h: int = 0  # Midnight New York usually
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Identify AMD (Accumulation, Manipulation, Distribution) phases.
    
    Simplified Logic for Intraday:
    - Reference price: Daily Open (or Session Open)
    - Manipulation: Price moves opposite to Day/Session direction first.
    - Distribution: Expansion in direction.
    """
    day_open = np.full(n, np.nan)
    po3_manipulation = np.zeros(n, dtype=np.int8)
    po3_distribution = np.zeros(n, dtype=np.int8)
    
    curr_day_open = open_[0]
    
    for i in range(1, n):
        # New Day/Session Start
        if hours[i] == session_start_h and hours[i-1] != session_start_h:
            curr_day_open = open_[i]
            
        day_open[i] = curr_day_open
        
        # Determine likely direction based on current price vs open
        if close[i] > curr_day_open:
            # Current is Bullish relative to Open
            # manipulation would be the low below open
            if low[i] < curr_day_open:
                po3_manipulation[i] = 1 # Wick below open in bullish candle
            elif high[i] > curr_day_open and low[i] > curr_day_open:
                po3_distribution[i] = 1
                
        elif close[i] < curr_day_open:
            # Current is Bearish relative to Open
            # manipulation would be the high above open
            if high[i] > curr_day_open:
                po3_manipulation[i] = 1 # Wick above open in bearish candle
            elif low[i] < curr_day_open and high[i] < curr_day_open:
                po3_distribution[i] = 1
                
    return day_open, po3_manipulation, po3_distribution


# =============================================================================
# OTE (OPTIMAL TRADE ENTRY) KERNEL
# =============================================================================

@jit(nopython=True, cache=True)
def _ote_kernel(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    swing_high: np.ndarray,
    swing_low: np.ndarray,
    structure_dir: np.ndarray,
    n: int,
    fib_62: float = 0.62,
    fib_79: float = 0.79
) -> Tuple[
    np.ndarray, np.ndarray,  # ote_bull_top, ote_bull_btm
    np.ndarray, np.ndarray,  # ote_bear_top, ote_bear_btm
    np.ndarray, np.ndarray   # in_ote_bull, in_ote_bear
]:
    """Calculate Optimal Trade Entry (OTE) Fibonacci zones."""
    ote_bull_top = np.full(n, np.nan)
    ote_bull_btm = np.full(n, np.nan)
    ote_bear_top = np.full(n, np.nan)
    ote_bear_btm = np.full(n, np.nan)
    in_ote_bull = np.zeros(n, dtype=np.int8)
    in_ote_bear = np.zeros(n, dtype=np.int8)
    
    for i in range(n):
        if np.isnan(swing_high[i]) or np.isnan(swing_low[i]):
            continue
        
        range_size = swing_high[i] - swing_low[i]
        if range_size <= 0:
            continue
        
        # Bullish OTE: retracement in uptrend
        if structure_dir[i] == 1:
            ote_bull_top[i] = swing_high[i] - range_size * fib_62
            ote_bull_btm[i] = swing_high[i] - range_size * fib_79
            if low[i] <= ote_bull_top[i] and low[i] >= ote_bull_btm[i]:
                in_ote_bull[i] = 1
        
        # Bearish OTE: retracement in downtrend
        if structure_dir[i] == -1:
            ote_bear_btm[i] = swing_low[i] + range_size * fib_62
            ote_bear_top[i] = swing_low[i] + range_size * fib_79
            if high[i] >= ote_bear_btm[i] and high[i] <= ote_bear_top[i]:
                in_ote_bear[i] = 1
    
    return ote_bull_top, ote_bull_btm, ote_bear_top, ote_bear_btm, in_ote_bull, in_ote_bear


# =============================================================================
# SESSION FILTER KERNEL
# =============================================================================

@jit(nopython=True, cache=True)
def _session_filter_kernel(
    hours: np.ndarray,
    n: int,
    asia_start: int = 0, asia_end: int = 8,
    london_start: int = 7, london_end: int = 16,
    ny_start: int = 12, ny_end: int = 21,
    sb_london_start: int = 9, sb_london_end: int = 10,
    sb_ny_am_start: int = 14, sb_ny_am_end: int = 15,
    sb_ny_pm_start: int = 18, sb_ny_pm_end: int = 19
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray,  # in_asia, in_london, in_ny
    np.ndarray, np.ndarray, np.ndarray   # in_kill_zone, in_silver_bullet, session_vol
]:
    """Session-based volatility filter with configurable times for DST."""
    in_asia = np.zeros(n, dtype=np.int8)
    in_london = np.zeros(n, dtype=np.int8)
    in_ny = np.zeros(n, dtype=np.int8)
    in_kill_zone = np.zeros(n, dtype=np.int8)
    in_silver_bullet = np.zeros(n, dtype=np.int8)
    session_vol = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        h = hours[i]
        
        if asia_start <= h < asia_end:
            in_asia[i] = 1
            session_vol[i] = 0.5
        
        if london_start <= h < london_end:
            in_london[i] = 1
            session_vol[i] = 1.0
        
        if ny_start <= h < ny_end:
            in_ny[i] = 1
            session_vol[i] = 1.0
        
        if ny_start <= h < london_end:
            in_kill_zone[i] = 1
            session_vol[i] = 1.5
        
        if (sb_london_start <= h < sb_london_end or
            sb_ny_am_start <= h < sb_ny_am_end or
            sb_ny_pm_start <= h < sb_ny_pm_end):
            in_silver_bullet[i] = 1
    
    return in_asia, in_london, in_ny, in_kill_zone, in_silver_bullet, session_vol


# =============================================================================
# MACROS KERNEL
# =============================================================================

@jit(nopython=True, cache=True)
def _macros_kernel(
    hours: np.ndarray,
    minutes: np.ndarray,
    n: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ICT Macros - specific time windows for high-probability setups."""
    in_macro = np.zeros(n, dtype=np.int8)
    macro_id = np.zeros(n, dtype=np.int8)
    in_ny_macro = np.zeros(n, dtype=np.int8)
    
    # Macros: (start_h, start_m, end_h, end_m, is_ny)
    macros = [
        (7, 33, 8, 0, 0),      # London Open
        (9, 3, 9, 30, 0),      # London 2
        (13, 50, 14, 10, 1),   # NY AM Open
        (14, 50, 15, 10, 1),   # NY AM 2
        (15, 50, 16, 10, 1),   # NY AM 3
        (16, 50, 17, 10, 1),   # NY Lunch
        (18, 10, 18, 40, 1),   # NY PM
        (20, 15, 20, 45, 1),   # NY Last Hr
    ]
    
    for i in range(n):
        time_mins = hours[i] * 60 + minutes[i]
        
        for idx, (sh, sm, eh, em, is_ny) in enumerate(macros):
            start = sh * 60 + sm
            end = eh * 60 + em
            
            if start <= time_mins < end:
                in_macro[i] = 1
                macro_id[i] = idx + 1
                if is_ny == 1:
                    in_ny_macro[i] = 1
                break
    
    return in_macro, macro_id, in_ny_macro


# =============================================================================
# EQUAL HIGHS/LOWS KERNEL
# =============================================================================

@jit(nopython=True, cache=True)
def _equal_levels_kernel(
    high: np.ndarray,
    low: np.ndarray,
    atr: np.ndarray,
    n: int,
    lookback: int = 50,
    tolerance_mult: float = 0.067
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Detect Equal Highs/Lows (liquidity pools)."""
    equal_high = np.full(n, np.nan)
    equal_low = np.full(n, np.nan)
    equal_high_count = np.zeros(n, dtype=np.int64)
    equal_low_count = np.zeros(n, dtype=np.int64)
    
    for i in range(lookback, n):
        tolerance = atr[i] * tolerance_mult if atr[i] > 0 else 0.001
        ref_high = high[i-1]
        ref_low = low[i-1]
        
        count_h, sum_h = 0, 0.0
        count_l, sum_l = 0, 0.0
        
        for j in range(i-lookback, i):
            if abs(high[j] - ref_high) <= tolerance:
                count_h += 1
                sum_h += high[j]
            if abs(low[j] - ref_low) <= tolerance:
                count_l += 1
                sum_l += low[j]
        
        if count_h >= 2:
            equal_high[i] = sum_h / count_h
            equal_high_count[i] = count_h
        if count_l >= 2:
            equal_low[i] = sum_l / count_l
            equal_low_count[i] = count_l
    
    return equal_high, equal_low, equal_high_count, equal_low_count


# =============================================================================
# WRAPPER CLASS - Only Calculations
# =============================================================================

class ICTNumba:
    """
    Numba-optimized ICT concepts calculator.
    
    This class ONLY calculates raw ICT data. Trading logic (Model2022, Unicorn,
    etc.) should be implemented in the strategy file.
    
    Usage:
        ict = ICTNumba(df, atr_period=14, swing_atr_mult=2.0)
        signals = ict.get_signals()
        
        # Access raw data in strategy:
        # signals['in_bull_ob'], signals['in_bear_fvg'], etc.
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        atr_period: int = 14,
        swing_atr_mult: float = 2.0,
        liq_sweep_atr_mult: float = 5.0,
        # Session times (UTC) - configurable for DST
        asia_start: int = 0, asia_end: int = 8,
        london_start: int = 7, london_end: int = 16,
        ny_start: int = 12, ny_end: int = 21,
        sb_london_start: int = 9, sb_london_end: int = 10,
        sb_ny_am_start: int = 14, sb_ny_am_end: int = 15,
        sb_ny_pm_start: int = 18, sb_ny_pm_end: int = 19
    ):
        self.df = df
        self.atr_period = atr_period
        self.swing_atr_mult = swing_atr_mult
        self.liq_sweep_atr_mult = liq_sweep_atr_mult
        
        self.session_times = {
            'asia_start': asia_start, 'asia_end': asia_end,
            'london_start': london_start, 'london_end': london_end,
            'ny_start': ny_start, 'ny_end': ny_end,
            'sb_london_start': sb_london_start, 'sb_london_end': sb_london_end,
            'sb_ny_am_start': sb_ny_am_start, 'sb_ny_am_end': sb_ny_am_end,
            'sb_ny_pm_start': sb_ny_pm_start, 'sb_ny_pm_end': sb_ny_pm_end
        }
        
        self.open = df['open'].values.astype(np.float64)
        self.high = df['high'].values.astype(np.float64)
        self.low = df['low'].values.astype(np.float64)
        self.close = df['close'].values.astype(np.float64)
        self.n = len(df)
        self.atr = _atr(self.high, self.low, self.close, atr_period)
    
    def get_signals(self) -> pd.DataFrame:
        """Calculate all ICT data and return as DataFrame."""
        
        # 1. Swing Detection
        swing_high, swing_low, swing_high_idx, swing_low_idx, direction = _swing_detection_kernel(
            self.high, self.low, self.close, self.atr, self.n,
            atr_mult=self.swing_atr_mult
        )
        
        # 2. Market Structure
        bos_bull, bos_bear, mss_bull, mss_bear, structure_dir = _market_structure_kernel(
            self.high, self.low, self.close,
            swing_high, swing_low, swing_high_idx, swing_low_idx, self.n
        )
        
        # 3. Order Blocks
        ob_bull_top, ob_bull_btm, ob_bear_top, ob_bear_btm = _order_blocks_kernel(
            self.open, self.high, self.low, self.close,
            bos_bull, bos_bear, mss_bull, mss_bear, self.n
        )
        
        # 4. FVGs
        fvg_bull_top, fvg_bull_btm, fvg_bear_top, fvg_bear_btm = _fvg_kernel(
            self.high, self.low, self.close, self.open, self.atr, self.n
        )
        
        # 5. Active Zones
        (act_ob_bull_top, act_ob_bull_btm, act_ob_bear_top, act_ob_bear_btm,
         in_bull_ob, in_bear_ob,
         act_fvg_bull_top, act_fvg_bull_btm, act_fvg_bear_top, act_fvg_bear_btm,
         in_bull_fvg, in_bear_fvg) = _active_zones_kernel(
            self.high, self.low, self.close,
            ob_bull_top, ob_bull_btm, ob_bear_top, ob_bear_btm,
            fvg_bull_top, fvg_bull_btm, fvg_bear_top, fvg_bear_btm, self.n
        )
        
        # 6. Liquidity Sweeps
        sweep_high, sweep_low = _liquidity_sweep_kernel(
            self.high, self.low, self.close,
            swing_high, swing_low, self.atr, self.n, self.liq_sweep_atr_mult
        )
        
        # 7. Premium/Discount
        equilibrium, in_premium, in_discount = _premium_discount_kernel(
            self.close, swing_high, swing_low, self.n
        )
        
        # 8. Strong Levels
        strong_high, strong_high_idx_out, strong_low, strong_low_idx_out = _strong_levels_kernel(
            self.high, self.low, self.close,
            bos_bull, bos_bear, swing_high_idx, swing_low_idx, self.n
        )
        
        # 9. Breaker Blocks
        bb_bull_top, bb_bull_btm, bb_bear_top, bb_bear_btm, in_bb_bull, in_bb_bear = _breaker_block_kernel(
            self.high, self.low, self.close,
            ob_bull_top, ob_bull_btm, ob_bear_top, ob_bear_btm, self.n
        )
        
        # 10. OTE
        ote_bull_top, ote_bull_btm, ote_bear_top, ote_bear_btm, in_ote_bull, in_ote_bear = _ote_kernel(
            self.high, self.low, self.close,
            swing_high, swing_low, structure_dir, self.n
        )
        
        # 11. Sessions
        hours = self.df.index.hour.values.astype(np.int64) if hasattr(self.df.index, 'hour') else np.zeros(self.n, dtype=np.int64)
        st = self.session_times
        in_asia, in_london, in_ny, in_kill_zone, in_silver_bullet, session_vol = _session_filter_kernel(
            hours, self.n,
            st['asia_start'], st['asia_end'],
            st['london_start'], st['london_end'],
            st['ny_start'], st['ny_end'],
            st['sb_london_start'], st['sb_london_end'],
            st['sb_ny_am_start'], st['sb_ny_am_end'],
            st['sb_ny_pm_start'], st['sb_ny_pm_end']
        )
        
        # 12. Macros
        minutes = self.df.index.minute.values.astype(np.int64) if hasattr(self.df.index, 'minute') else np.zeros(self.n, dtype=np.int64)
        in_macro, macro_id, in_ny_macro = _macros_kernel(hours, minutes, self.n)
        
        # 13. Equal Levels
        equal_high, equal_low, equal_high_count, equal_low_count = _equal_levels_kernel(
            self.high, self.low, self.atr, self.n
        )
        
        # 14. Rejection Blocks
        rb_bull_top, rb_bull_btm, rb_bear_top, rb_bear_btm = _rejection_blocks_kernel(
            self.high, self.low, self.close, self.open, self.atr, self.n
        )
        
        # 15. Po3 (Power of Three) - Defaulting to Midnight NY (00:00 UTC approx if user configured)
        # Assuming data is in UTC and NY is UTC-4/-5. 
        # If user passes 0 as asia_start, we can use that as "Day Start" reference or let user config.
        # For now using 0 (Midnight UTC) as a simpler proxy for "Daily Open"
        day_open, po3_manipulation, po3_distribution = _po3_kernel(
            self.open, self.close, self.high, self.low, hours, self.n,
            session_start_h=0 
        )
        
        # Build DataFrame
        result = pd.DataFrame(index=self.df.index)
        
        # Core levels
        result['swing_high'] = swing_high
        result['swing_low'] = swing_low
        result['swing_direction'] = direction
        
        # Structure
        result['bos_bullish'] = bos_bull
        result['bos_bearish'] = bos_bear
        result['mss_bullish'] = mss_bull
        result['mss_bearish'] = mss_bear
        result['structure_direction'] = structure_dir
        
        # Order Blocks
        result['active_bull_ob_top'] = act_ob_bull_top
        result['active_bull_ob_btm'] = act_ob_bull_btm
        result['active_bear_ob_top'] = act_ob_bear_top
        result['active_bear_ob_btm'] = act_ob_bear_btm
        result['in_bull_ob'] = in_bull_ob
        result['in_bear_ob'] = in_bear_ob
        
        # FVGs
        result['active_bull_fvg_top'] = act_fvg_bull_top
        result['active_bull_fvg_btm'] = act_fvg_bull_btm
        result['active_bear_fvg_top'] = act_fvg_bear_top
        result['active_bear_fvg_btm'] = act_fvg_bear_btm
        result['in_bull_fvg'] = in_bull_fvg
        result['in_bear_fvg'] = in_bear_fvg
        
        # Sweeps
        result['sweep_high'] = sweep_high
        result['sweep_low'] = sweep_low
        
        # Premium/Discount
        result['equilibrium'] = equilibrium
        result['in_premium'] = in_premium
        result['in_discount'] = in_discount
        
        # Strong Levels
        result['strong_high'] = strong_high
        result['strong_low'] = strong_low
        
        # Breaker Blocks
        result['bb_bull_top'] = bb_bull_top
        result['bb_bull_btm'] = bb_bull_btm
        result['bb_bear_top'] = bb_bear_top
        result['bb_bear_btm'] = bb_bear_btm
        result['in_bb_bull'] = in_bb_bull
        result['in_bb_bear'] = in_bb_bear
        
        # OTE
        result['ote_bull_top'] = ote_bull_top
        result['ote_bull_btm'] = ote_bull_btm
        result['ote_bear_top'] = ote_bear_top
        result['ote_bear_btm'] = ote_bear_btm
        result['in_ote_bull'] = in_ote_bull
        result['in_ote_bear'] = in_ote_bear
        
        # Sessions
        result['in_asia'] = in_asia
        result['in_london'] = in_london
        result['in_ny'] = in_ny
        result['in_kill_zone'] = in_kill_zone
        result['in_silver_bullet'] = in_silver_bullet
        result['session_volatility'] = session_vol
        
        # Macros
        result['in_macro'] = in_macro
        result['macro_id'] = macro_id
        result['in_ny_macro'] = in_ny_macro
        
        # Equal Levels
        result['equal_high'] = equal_high
        result['equal_low'] = equal_low
        result['equal_high_count'] = equal_high_count
        result['equal_low_count'] = equal_low_count
        
        # Rejection Blocks
        result['rb_bull_top'] = rb_bull_top
        result['rb_bull_btm'] = rb_bull_btm
        result['rb_bear_top'] = rb_bear_top
        result['rb_bear_btm'] = rb_bear_btm
        
        # Po3
        result['day_open'] = day_open
        result['po3_manipulation'] = po3_manipulation
        result['po3_distribution'] = po3_distribution
        
        # ATR
        result['atr'] = self.atr
        
        return result
