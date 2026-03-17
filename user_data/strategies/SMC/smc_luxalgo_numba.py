
import numpy as np
import pandas as pd
from numba import jit, float64, int64, int8, boolean
from typing import Tuple

# =============================================================================
# NUMBA KERNELS
# =============================================================================

@jit(nopython=True, cache=True)
def _find_lowest_low(low: np.ndarray, start: int, end: int) -> int:
    """Find index of lowest low in range [start, end)."""
    min_val = 1e15 # Huge number
    min_idx = -1
    for i in range(start, end):
        if low[i] < min_val:
            min_val = low[i]
            min_idx = i
    return min_idx

@jit(nopython=True, cache=True)
def _find_highest_high(high: np.ndarray, start: int, end: int) -> int:
    """Find index of highest high in range [start, end)."""
    max_val = -1.0
    max_idx = -1
    for i in range(start, end):
        if high[i] > max_val:
            max_val = high[i]
            max_idx = i
    return max_idx

@jit(nopython=True, cache=True)
def _smc_signals_kernel(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    n: int,
    internal_length: int,
    swing_length: int
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, # Internal BOS/CHoCH Bull/Bear
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, # Swing BOS/CHoCH Bull/Bear
    np.ndarray, np.ndarray, # Internal/Swing Trend
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, # OB Top/Bottom/Vol
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, # FVG Top/Bottom
    np.ndarray, np.ndarray, # Swing High, Swing Low (Current Range)
    np.ndarray, np.ndarray, np.ndarray, np.ndarray  # Liquidity Sweeps (Int/Sw Bull/Bear)
]:
    # Output Arrays
    int_bos_bull = np.zeros(n, dtype=int8)
    int_bos_bear = np.zeros(n, dtype=int8)
    int_choch_bull = np.zeros(n, dtype=int8)
    int_choch_bear = np.zeros(n, dtype=int8)
    
    sw_bos_bull = np.zeros(n, dtype=int8)
    sw_bos_bear = np.zeros(n, dtype=int8)
    sw_choch_bull = np.zeros(n, dtype=int8)
    sw_choch_bear = np.zeros(n, dtype=int8)
    
    int_trend = np.zeros(n, dtype=int8)
    sw_trend = np.zeros(n, dtype=int8)

    # OB Storage
    ob_bull_top = np.full(n, np.nan)
    ob_bull_btm = np.full(n, np.nan)
    ob_bull_vol = np.zeros(n)
    ob_bear_top = np.full(n, np.nan)
    ob_bear_btm = np.full(n, np.nan)
    ob_bear_vol = np.zeros(n)
    
    # FVG Storage
    fvg_bull_top = np.full(n, np.nan)
    fvg_bull_btm = np.full(n, np.nan)
    fvg_bear_top = np.full(n, np.nan)
    fvg_bear_btm = np.full(n, np.nan)

    # Swing Levels Storage
    out_sw_high = np.full(n, np.nan)
    out_sw_low = np.full(n, np.nan)

    # Liquidity Sweep Storage
    int_sweep_bull = np.zeros(n, dtype=int8)
    int_sweep_bear = np.zeros(n, dtype=int8)
    sw_sweep_bull = np.zeros(n, dtype=int8)
    sw_sweep_bear = np.zeros(n, dtype=int8)

    # State Variables
    # Internal Pivot
    ip_high_level = np.nan
    ip_high_idx = -1
    ip_high_crossed = False
    
    ip_low_level = np.nan
    ip_low_idx = -1
    ip_low_crossed = False
    
    int_leg = 0
    curr_int_trend = 0
    
    # Swing Pivot
    sp_high_level = np.nan
    sp_high_idx = -1
    sp_high_crossed = False
    
    sp_low_level = np.nan
    sp_low_idx = -1
    sp_low_crossed = False
    
    sw_leg = 0
    curr_sw_trend = 0

    for i in range(2, n):
        curr_high = high[i]
        curr_low = low[i]
        curr_close = close[i]
        prev_close = close[i-1]
        
        # --- 1. LEG DETECTION & PIVOT UPDATES ---
        
        # Internal Leg
        new_int_leg = 0
        if i >= internal_length:
            ref_l = low[i - internal_length]
            win_l = 1e15
            for k in range(i - internal_length + 1, i + 1):
                if low[k] < win_l: win_l = low[k]
            
            if ref_l < win_l:
                new_int_leg = 1
            else:
                ref_h = high[i - internal_length]
                win_h = -1.0
                for k in range(i - internal_length + 1, i + 1):
                    if high[k] > win_h: win_h = high[k]
                
                if ref_h > win_h:
                    new_int_leg = -1
        
        if new_int_leg != 0 and new_int_leg != int_leg:
            int_leg = new_int_leg
            pivot_idx = i - internal_length
            if int_leg == 1:
                ip_low_level = low[pivot_idx]
                ip_low_idx = pivot_idx
                ip_low_crossed = False
            else:
                ip_high_level = high[pivot_idx]
                ip_high_idx = pivot_idx
                ip_high_crossed = False

        # Swing Leg
        new_sw_leg = 0
        if i >= swing_length:
            ref_l = low[i - swing_length]
            win_l = 1e15
            for k in range(i - swing_length + 1, i + 1):
                if low[k] < win_l: win_l = low[k]
            
            if ref_l < win_l:
                new_sw_leg = 1
            else:
                ref_h = high[i - swing_length]
                win_h = -1.0
                for k in range(i - swing_length + 1, i + 1):
                    if high[k] > win_h: win_h = high[k]
                
                if ref_h > win_h:
                    new_sw_leg = -1
        
        if new_sw_leg != 0 and new_sw_leg != sw_leg:
            sw_leg = new_sw_leg
            pivot_idx = i - swing_length
            if sw_leg == 1: # New Swing Low -> Previous Swing High is confirmed as Top of Range?
                # Actually, in SMC, a Swing Low forms the bottom of the current range defined by the LAST Swing High.
                sp_low_level = low[pivot_idx]
                sp_low_idx = pivot_idx
                sp_low_crossed = False
            else: # New Swing High -> Top of Range
                sp_high_level = high[pivot_idx]
                sp_high_idx = pivot_idx
                sp_high_crossed = False
                
        # Update Output Arrays for Swing Levels (Repeatedly write current known levels)
        # This gives us the "Current Trading Range" at any point in time.
        out_sw_high[i] = sp_high_level
        out_sw_low[i] = sp_low_level

        # --- 2. INTERNAL STRUCTURE ---
        
        # Bullish Break
        if not ip_high_crossed and not np.isnan(ip_high_level):
            if prev_close <= ip_high_level and curr_close > ip_high_level:
                ip_high_crossed = True
                if curr_int_trend == -1:
                    int_choch_bull[i] = 1
                    curr_int_trend = 1
                else:
                    int_bos_bull[i] = 1
                    curr_int_trend = 1
                
                if ip_high_idx < i:
                    ob_idx = _find_lowest_low(low, ip_high_idx, i)
                    if ob_idx != -1:
                        ob_bull_top[i] = high[ob_idx]
                        ob_bull_btm[i] = low[ob_idx]
                        ob_bull_vol[i] = volume[ob_idx]

        # Bearish Break
        if not ip_low_crossed and not np.isnan(ip_low_level):
            if prev_close >= ip_low_level and curr_close < ip_low_level:
                ip_low_crossed = True
                if curr_int_trend == 1:
                    int_choch_bear[i] = 1
                    curr_int_trend = -1
                else:
                    int_bos_bear[i] = 1
                    curr_int_trend = -1
                
                if ip_low_idx < i:
                    ob_idx = _find_highest_high(high, ip_low_idx, i)
                    if ob_idx != -1:
                        ob_bear_top[i] = high[ob_idx]
                        ob_bear_btm[i] = low[ob_idx]
                        ob_bear_vol[i] = volume[ob_idx]
        
        # --- INTERNAL LIQUIDITY SWEEPS ---
        # Bullish Sweep: Wick below internal pivot low, close above (took sellside liquidity)
        # Only if this candle did NOT break the level (mutually exclusive with BOS/CHoCH)
        if not ip_low_crossed and not np.isnan(ip_low_level):
            if curr_low <= ip_low_level and curr_close >= ip_low_level:
                int_sweep_bull[i] = 1
        
        # Bearish Sweep: Wick above internal pivot high, close below (took buyside liquidity)
        if not ip_high_crossed and not np.isnan(ip_high_level):
            if curr_high >= ip_high_level and curr_close <= ip_high_level:
                int_sweep_bear[i] = 1

        int_trend[i] = curr_int_trend

        # --- 3. SWING STRUCTURE ---
        
        if not sp_high_crossed and not np.isnan(sp_high_level):
            if prev_close <= sp_high_level and curr_close > sp_high_level:
                sp_high_crossed = True
                if curr_sw_trend == -1:
                    sw_choch_bull[i] = 1
                    curr_sw_trend = 1
                else:
                    sw_bos_bull[i] = 1
                    curr_sw_trend = 1
                
                if sp_high_idx < i:
                    ob_idx = _find_lowest_low(low, sp_high_idx, i)
                    if ob_idx != -1:
                        # FIX: Write at confirmation index i, not geometry index
                        ob_bull_top[i] = high[ob_idx]
                        ob_bull_btm[i] = low[ob_idx]
                        ob_bull_vol[i] = volume[ob_idx]

        if not sp_low_crossed and not np.isnan(sp_low_level):
            if prev_close >= sp_low_level and curr_close < sp_low_level:
                sp_low_crossed = True
                if curr_sw_trend == 1:
                    sw_choch_bear[i] = 1
                    curr_sw_trend = -1
                else:
                    sw_bos_bear[i] = 1
                    curr_sw_trend = -1
                
                if sp_low_idx < i:
                    ob_idx = _find_highest_high(high, sp_low_idx, i)
                    if ob_idx != -1:
                        # FIX: Write at confirmation index i, not geometry index
                        ob_bear_top[i] = high[ob_idx]
                        ob_bear_btm[i] = low[ob_idx]
                        ob_bear_vol[i] = volume[ob_idx]
        
        # --- SWING LIQUIDITY SWEEPS ---
        # Bullish Sweep: Wick below swing pivot low, close above
        if not sp_low_crossed and not np.isnan(sp_low_level):
            if curr_low <= sp_low_level and curr_close >= sp_low_level:
                sw_sweep_bull[i] = 1
        
        # Bearish Sweep: Wick above swing pivot high, close below
        if not sp_high_crossed and not np.isnan(sp_high_level):
            if curr_high >= sp_high_level and curr_close <= sp_high_level:
                sw_sweep_bear[i] = 1

        sw_trend[i] = curr_sw_trend
        
        # --- 4. FVG DETECTION ---
        if low[i] > high[i-2]:
            fvg_bull_top[i] = low[i]
            fvg_bull_btm[i] = high[i-2]
        
        if high[i] < low[i-2]:
            fvg_bear_top[i] = low[i-2]
            fvg_bear_btm[i] = high[i]

    return (
        int_bos_bull, int_bos_bear, int_choch_bull, int_choch_bear,
        sw_bos_bull, sw_bos_bear, sw_choch_bull, sw_choch_bear,
        int_trend, sw_trend,
        ob_bull_top, ob_bull_btm, ob_bear_top, ob_bear_btm, ob_bull_vol, ob_bear_vol,
        fvg_bull_top, fvg_bull_btm, fvg_bear_top, fvg_bear_btm,
        out_sw_high, out_sw_low,
        int_sweep_bull, int_sweep_bear, sw_sweep_bull, sw_sweep_bear
    )

@jit(nopython=True, cache=True)
def _smc_zones_kernel(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    ob_bull_top: np.ndarray, ob_bull_btm: np.ndarray, ob_bull_vol: np.ndarray,
    ob_bear_top: np.ndarray, ob_bear_btm: np.ndarray, ob_bear_vol: np.ndarray,
    fvg_bull_top: np.ndarray, fvg_bull_btm: np.ndarray,
    fvg_bear_top: np.ndarray, fvg_bear_btm: np.ndarray,
    n: int
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, # Breakers (OB)
    np.ndarray, np.ndarray, np.ndarray, np.ndarray  # Breakers (FVG) - NEW
]:
    
    # --- Output Arrays ---
    # OBs
    act_bull_ob_top = np.zeros(n)
    act_bull_ob_btm = np.zeros(n)
    act_bull_ob_vol = np.zeros(n)
    act_bear_ob_top = np.zeros(n)
    act_bear_ob_btm = np.zeros(n)
    act_bear_ob_vol = np.zeros(n)
    
    # FVGs
    act_bull_fvg_top = np.zeros(n)
    act_bull_fvg_btm = np.zeros(n)
    act_bear_fvg_top = np.zeros(n)
    act_bear_fvg_btm = np.zeros(n)
    
    # Breakers (OB)
    act_brk_bull_top = np.zeros(n)
    act_brk_bull_btm = np.zeros(n)
    act_brk_bear_top = np.zeros(n)
    act_brk_bear_btm = np.zeros(n)

    # Breakers (FVG) - NEW
    act_fvg_brk_bull_top = np.zeros(n)
    act_fvg_brk_bull_btm = np.zeros(n)
    act_fvg_brk_bear_top = np.zeros(n)
    act_fvg_brk_bear_btm = np.zeros(n)
    
    # --- State Lists (Indices) ---
    bull_ob_idxs = [np.int64(x) for x in range(0)]
    bear_ob_idxs = [np.int64(x) for x in range(0)]
    
    bull_fvg_idxs = [np.int64(x) for x in range(0)]
    bear_fvg_idxs = [np.int64(x) for x in range(0)]
    
    # Breaker Lists (OB)
    bull_brk_idxs = [np.int64(x) for x in range(0)]
    bear_brk_idxs = [np.int64(x) for x in range(0)]

    # Breaker Lists (FVG) - NEW
    # Bullish FVG Breaker = Old Bearish FVG broken up (Support)
    bull_fvg_brk_idxs = [np.int64(x) for x in range(0)]
    
    # Bearish FVG Breaker = Old Bullish FVG broken down (Resistance)
    bear_fvg_brk_idxs = [np.int64(x) for x in range(0)]
    
    for i in range(n):
        c_high = high[i]
        c_low = low[i]
        c_close = close[i]
        
        # --- 1. Process Existing Zones (Mitigation / Flip) ---
        
        # A) Bullish OBs (Support)
        next_bull_obs = [np.int64(x) for x in range(0)]
        best_bull_ob_idx = -1
        
        for idx in bull_ob_idxs:
            btm = ob_bull_btm[idx]
            if c_close < btm:
                # BROKEN! Flux to Bearish Breaker
                bear_brk_idxs.append(idx)
            else:
                 next_bull_obs.append(idx)
                 best_bull_ob_idx = idx
        bull_ob_idxs = next_bull_obs
        
        # B) Bearish OBs (Resistance)
        next_bear_obs = [np.int64(x) for x in range(0)]
        best_bear_ob_idx = -1
        
        for idx in bear_ob_idxs:
            top = ob_bear_top[idx]
            if c_close > top:
                # BROKEN! Flux to Bullish Breaker
                bull_brk_idxs.append(idx)
            else:
                next_bear_obs.append(idx)
                best_bear_ob_idx = idx
        bear_ob_idxs = next_bear_obs
        
        # C) FVGs (Standard mitigation + Flip)
        
        # Bullish FVG (Support) -> Becomes Bearish FVG Breaker if Close < Bottom
        # Note: FVG Bottom is high[i-2], Top is low[i]. Wait.
        # In kernel:
        # fvg_bull_top[i] = low[i] (The top of the gap)
        # fvg_bull_btm[i] = high[i-2] (The bottom of the gap)
        
        next_bull_fvgs = [np.int64(x) for x in range(0)]
        best_bull_fvg_idx = -1
        for idx in bull_fvg_idxs:
            top = fvg_bull_top[idx]
            btm = fvg_bull_btm[idx]
            
            # Mitigation Check:
            # Standard: If filled (Low <= Top)? No, keeps existing.
            # Inversion Check: If Close < Bottom (Fully crossed)
            
            if c_close < btm:
                # BROKEN! Flip to Bearish FVG Breaker (Resistance)
                bear_fvg_brk_idxs.append(idx)
            elif c_low <= top:
                 # Filled / Mitigated (Touched). 
                 # Standard logic: often removed if filled.
                 # User wants Breakers. If it's just filled but not broken (closed below), 
                 # does it stay? Usually yes until invalid.
                 # Let's keep it until Broken.
                 next_bull_fvgs.append(idx)
                 best_bull_fvg_idx = idx
            else:
                next_bull_fvgs.append(idx)
                best_bull_fvg_idx = idx
        bull_fvg_idxs = next_bull_fvgs
        
        # Bearish FVG (Resistance) -> Becomes Bullish FVG Breaker if Close > Top
        # fvg_bear_top[i] = low[i-2] (Top of gap)
        # fvg_bear_btm[i] = high[i] (Bottom of gap)
        
        next_bear_fvgs = [np.int64(x) for x in range(0)]
        best_bear_fvg_idx = -1
        for idx in bear_fvg_idxs:
            top = fvg_bear_top[idx]
            btm = fvg_bear_btm[idx]
            
            if c_close > top:
                # BROKEN! Flip to Bullish FVG Breaker (Support)
                bull_fvg_brk_idxs.append(idx)
            elif c_high >= btm:
                # Filled/Mitigated
                next_bear_fvgs.append(idx)
                best_bear_fvg_idx = idx
            else:
                next_bear_fvgs.append(idx)
                best_bear_fvg_idx = idx
        bear_fvg_idxs = next_bear_fvgs
        
        # D) Breakers Validation (OB)
        
        next_bull_brks = [np.int64(x) for x in range(0)]
        best_bull_brk_idx = -1
        for idx in bull_brk_idxs:
            btm = ob_bear_btm[idx] # Bear OB Btm
            if c_close < btm:
                pass # Broken again
            else:
                next_bull_brks.append(idx)
                best_bull_brk_idx = idx
        bull_brk_idxs = next_bull_brks
        
        next_bear_brks = [np.int64(x) for x in range(0)]
        best_bear_brk_idx = -1
        for idx in bear_brk_idxs:
            top = ob_bull_top[idx] # Bull OB Top
            if c_close > top:
                pass # Broken again
            else:
                next_bear_brks.append(idx)
                best_bear_brk_idx = idx
        bear_brk_idxs = next_bear_brks

        # E) Breakers Validation (FVG) - NEW
        
        # Bullish FVG Breaker (Support): Created from broken Bearish FVG.
        # Original Bear FVG: Top=Low[i-2], Btm=High[i]
        # Valid as Support unless broken back down (Close < Btm) 
        
        next_bull_fvg_brks = [np.int64(x) for x in range(0)]
        best_bull_fvg_brk_idx = -1
        for idx in bull_fvg_brk_idxs:
             # Use original Bear FVG coords
             # top = fvg_bear_top[idx]
             btm = fvg_bear_btm[idx]
             
             if c_close < btm:
                 pass # Failed support
             else:
                 next_bull_fvg_brks.append(idx)
                 best_bull_fvg_brk_idx = idx
        bull_fvg_brk_idxs = next_bull_fvg_brks

        # Bearish FVG Breaker (Resistance): Created from broken Bullish FVG.
        # Original Bull FVG: Top=Low[i], Btm=High[i-2]
        # Valid as Resistance unless broken back up (Close > Top)
        
        next_bear_fvg_brks = [np.int64(x) for x in range(0)]
        best_bear_fvg_brk_idx = -1
        for idx in bear_fvg_brk_idxs:
            # Use original Bull FVG coords
            top = fvg_bull_top[idx]
            # btm = fvg_bull_btm[idx]
            
            if c_close > top:
                pass # Failed resistance
            else:
                next_bear_fvg_brks.append(idx)
                best_bear_fvg_brk_idx = idx
        bear_fvg_brk_idxs = next_bear_fvg_brks
            
        
        # --- 2. Add New Zones (OBs/FVGs) ---
        if not np.isnan(ob_bull_top[i]): 
            bull_ob_idxs.append(i)
            best_bull_ob_idx = i
            
        if not np.isnan(ob_bear_top[i]): 
            bear_ob_idxs.append(i)
            best_bear_ob_idx = i
            
        if not np.isnan(fvg_bull_top[i]): 
            bull_fvg_idxs.append(i)
            best_bull_fvg_idx = i
            
        if not np.isnan(fvg_bear_top[i]): 
            bear_fvg_idxs.append(i)
            best_bear_fvg_idx = i
        
        # --- 3. Write Active Outputs ---
        if best_bull_ob_idx != -1:
            act_bull_ob_top[i] = ob_bull_top[best_bull_ob_idx]
            act_bull_ob_btm[i] = ob_bull_btm[best_bull_ob_idx]
            act_bull_ob_vol[i] = ob_bull_vol[best_bull_ob_idx]
            
        if best_bear_ob_idx != -1:
            act_bear_ob_top[i] = ob_bear_top[best_bear_ob_idx]
            act_bear_ob_btm[i] = ob_bear_btm[best_bear_ob_idx]
            act_bear_ob_vol[i] = ob_bear_vol[best_bear_ob_idx]

        if best_bull_fvg_idx != -1:
            act_bull_fvg_top[i] = fvg_bull_top[best_bull_fvg_idx]
            act_bull_fvg_btm[i] = fvg_bull_btm[best_bull_fvg_idx]
            
        if best_bear_fvg_idx != -1:
            act_bear_fvg_top[i] = fvg_bear_top[best_bear_fvg_idx]
            act_bear_fvg_btm[i] = fvg_bear_btm[best_bear_fvg_idx]
            
        # Breakers Output (OB)
        if best_bull_brk_idx != -1:
            act_brk_bull_top[i] = ob_bear_top[best_bull_brk_idx]
            act_brk_bull_btm[i] = ob_bear_btm[best_bull_brk_idx]
            
        if best_bear_brk_idx != -1:
            act_brk_bear_top[i] = ob_bull_top[best_bear_brk_idx]
            act_brk_bear_btm[i] = ob_bull_btm[best_bear_brk_idx]

        # Breakers Output (FVG) - NEW
        if best_bull_fvg_brk_idx != -1:
            # Bullish Breaker comes from Bear FVG array
            act_fvg_brk_bull_top[i] = fvg_bear_top[best_bull_fvg_brk_idx]
            act_fvg_brk_bull_btm[i] = fvg_bear_btm[best_bull_fvg_brk_idx]
            
        if best_bear_fvg_brk_idx != -1:
            # Bearish Breaker comes from Bull FVG array
            act_fvg_brk_bear_top[i] = fvg_bull_top[best_bear_fvg_brk_idx]
            act_fvg_brk_bear_btm[i] = fvg_bull_btm[best_bear_fvg_brk_idx]

    return (
        act_bull_ob_top, act_bull_ob_btm, act_bull_ob_vol,
        act_bear_ob_top, act_bear_ob_btm, act_bear_ob_vol,
        act_bull_fvg_top, act_bull_fvg_btm,
        act_bear_fvg_top, act_bear_fvg_btm,
        act_brk_bull_top, act_brk_bull_btm,
        act_brk_bear_top, act_brk_bear_btm,
        act_fvg_brk_bull_top, act_fvg_brk_bull_btm,
        act_fvg_brk_bear_top, act_fvg_brk_bear_btm
    )

# =============================================================================
# WRAPPER CLASS
# =============================================================================

class SMCLuxAlgoNumba:
    """Numba-optimized SMC replacement."""
    
    def __init__(self, df: pd.DataFrame, internal_length: int = 5, swing_length: int = 50):
        self.df = df
        self.internal_length = internal_length
        self.swing_length = swing_length
        
        # Prepare arrays
        self.high = df['high'].values.astype(float)
        self.low = df['low'].values.astype(float)
        self.close = df['close'].values.astype(float)
        self.volume = df['volume'].values.astype(float)
        self.n = len(df)
        
    def get_signals(self) -> pd.DataFrame:
        # 1. Run Core Kernel
        (
            ib_bull, ib_bear, ic_bull, ic_bear,
            sb_bull, sb_bear, sc_bull, sc_bear,
            i_trend, s_trend,
            ob_bt_raw, ob_bb_raw, ob_bet_raw, ob_beb_raw, ob_bull_vol_raw, ob_bear_vol_raw,
            fvg_bt_raw, fvg_bb_raw, fvg_bet_raw, fvg_beb_raw,
            sw_high, sw_low,
            int_sweep_bull, int_sweep_bear, sw_sweep_bull, sw_sweep_bear
        ) = _smc_signals_kernel(
            self.high, self.low, self.close, self.volume, self.n,
            self.internal_length, self.swing_length
        )
        
        # 2. Run Zone Kernel (Active Memory + Breakers)
        (
            act_ob_bt, act_ob_bb, act_ob_bvol,
            act_ob_bet, act_ob_beb, act_ob_bevol,
            act_fvg_bt, act_fvg_bb, act_fvg_bet, act_fvg_beb,
            act_brk_bull_t, act_brk_bull_b,
            act_brk_bear_t, act_brk_bear_b,
            act_fvg_brk_bull_t, act_fvg_brk_bull_b,
            act_fvg_brk_bear_t, act_fvg_brk_bear_b
        ) = _smc_zones_kernel(
            self.high, self.low, self.close,
            ob_bt_raw, ob_bb_raw, ob_bull_vol_raw,
            ob_bet_raw, ob_beb_raw, ob_bear_vol_raw,
            fvg_bt_raw, fvg_bb_raw, fvg_bet_raw, fvg_beb_raw,
            self.n
        )
        
        # 3. Construct DataFrame
        df = pd.DataFrame(index=self.df.index)
        
        # Signals
        df['internal_bos_bullish'] = ib_bull
        df['internal_bos_bearish'] = ib_bear
        df['internal_choch_bullish'] = ic_bull
        df['internal_choch_bearish'] = ic_bear
        
        df['swing_bos_bullish'] = sb_bull
        df['swing_bos_bearish'] = sb_bear
        df['swing_choch_bullish'] = sc_bull
        df['swing_choch_bearish'] = sc_bear
        
        df['internal_trend'] = i_trend
        df['swing_trend'] = s_trend
        
        # New P/D Logic Levels
        df['swing_high'] = sw_high
        df['swing_low'] = sw_low
        df['equilibrium'] = (df['swing_high'] + df['swing_low']) / 2.0
        
        # Active Zones (OBs)
        df['active_bullish_ob_top'] = act_ob_bt
        df['active_bullish_ob_bottom'] = act_ob_bb
        df['active_bullish_ob_vol'] = act_ob_bvol
        df['active_bearish_ob_top'] = act_ob_bet
        df['active_bearish_ob_bottom'] = act_ob_beb
        df['active_bearish_ob_vol'] = act_ob_bevol
        
        # Active Zones (FVGs)
        df['active_bullish_fvg_top'] = act_fvg_bt
        df['active_bullish_fvg_bottom'] = act_fvg_bb
        df['active_bearish_fvg_top'] = act_fvg_bet
        df['active_bearish_fvg_bottom'] = act_fvg_beb
        
        # Active Zones (Breakers - OB)
        df['active_bullish_breaker_top'] = act_brk_bull_t
        df['active_bullish_breaker_bottom'] = act_brk_bull_b
        df['active_bearish_breaker_top'] = act_brk_bear_t
        df['active_bearish_breaker_bottom'] = act_brk_bear_b

        # Active Zones (Breakers - FVG)
        df['active_bullish_fvg_breaker_top'] = act_fvg_brk_bull_t
        df['active_bullish_fvg_breaker_bottom'] = act_fvg_brk_bull_b
        df['active_bearish_fvg_breaker_top'] = act_fvg_brk_bear_t
        df['active_bearish_fvg_breaker_bottom'] = act_fvg_brk_bear_b

        # Liquidity Sweeps
        df['internal_sweep_bullish'] = int_sweep_bull
        df['internal_sweep_bearish'] = int_sweep_bear
        df['swing_sweep_bullish'] = sw_sweep_bull
        df['swing_sweep_bearish'] = sw_sweep_bear

        # ── Raw event arrays (for multi-zone tracking in strategy layer) ──
        # These mark the bar where each new zone was CONFIRMED.
        # NaN → no event that bar.  Strategy uses these to rebuild zone lists.
        df['ob_bull_top_raw'] = np.where(~np.isnan(ob_bt_raw), ob_bt_raw, 0.0)
        df['ob_bull_btm_raw'] = np.where(~np.isnan(ob_bb_raw), ob_bb_raw, 0.0)
        df['ob_bull_vol_raw'] = np.where(ob_bull_vol_raw > 0, ob_bull_vol_raw, 0.0)
        df['ob_bear_top_raw'] = np.where(~np.isnan(ob_bet_raw), ob_bet_raw, 0.0)
        df['ob_bear_btm_raw'] = np.where(~np.isnan(ob_beb_raw), ob_beb_raw, 0.0)
        df['ob_bear_vol_raw'] = np.where(ob_bear_vol_raw > 0, ob_bear_vol_raw, 0.0)

        df['fvg_bull_top_raw'] = np.where(~np.isnan(fvg_bt_raw), fvg_bt_raw, 0.0)
        df['fvg_bull_btm_raw'] = np.where(~np.isnan(fvg_bb_raw), fvg_bb_raw, 0.0)
        df['fvg_bear_top_raw'] = np.where(~np.isnan(fvg_bet_raw), fvg_bet_raw, 0.0)
        df['fvg_bear_btm_raw'] = np.where(~np.isnan(fvg_beb_raw), fvg_beb_raw, 0.0)

        # FVG impulse volume: candle i-1 (the middle bar of the 3-bar gap pattern)
        fvg_bull_ivol = np.zeros(self.n)
        fvg_bear_ivol = np.zeros(self.n)
        vol_arr = self.volume
        for idx in range(1, self.n):
            if not np.isnan(fvg_bt_raw[idx]) and fvg_bt_raw[idx] > 0:
                fvg_bull_ivol[idx] = vol_arr[idx - 1]
            if not np.isnan(fvg_bet_raw[idx]) and fvg_bet_raw[idx] > 0:
                fvg_bear_ivol[idx] = vol_arr[idx - 1]
        df['fvg_bull_impulse_vol_raw'] = fvg_bull_ivol
        df['fvg_bear_impulse_vol_raw'] = fvg_bear_ivol

        return df

