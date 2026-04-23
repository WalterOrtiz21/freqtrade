"""
SMC_Forge — Inducement (IDM) Layer
==================================

Detects IDM (Inducement) levels and sweep events.

Canon SMC: IDM is the small sub-pivot pool that smart money targets BEFORE
delivering the real structural move. In a pullback after a major swing
high (bullish setup), the IDM is the most recent minor sub-low. When that
sub-low is wick-swept (low pierced, close stays above), retail stops are
absorbed and the next leg up has fuel.

This module exposes:
  - active IDM level per bar
  - sweep events (wick-only crosses)
  - automatic invalidation by real close-break or context shift

The strategy combines `bull_idm_swept[X] AND choch_bull_disp[X+1..X+3] >= 1.0`
to form an A+ entry signal.

Public API
----------
    annotate_inducement(df, fractal_n_major=5, fractal_n_minor=2) -> DataFrame

Returns DataFrame with columns:
    pivot_high_major, pivot_low_major   Williams major pivots
    pivot_high_minor, pivot_low_minor   Williams minor pivots
    bull_idm_level                      Active IDM low (NaN if no bullish setup)
    bull_idm_swept                      1 if wick-swept this bar
    bear_idm_level                      Active IDM high (NaN if no bearish setup)
    bear_idm_swept                      1 if wick-swept this bar
"""

import numpy as np
import pandas as pd
from numba import jit

from forge_levels import _detect_pivots_kernel  # share fractal pivots


# =============================================================================
# NUMBA KERNEL
# =============================================================================

@jit(nopython=True, cache=True)
def _idm_kernel(
    pivot_high_major: np.ndarray, pivot_low_major: np.ndarray,
    pivot_high_minor: np.ndarray, pivot_low_minor: np.ndarray,
    high: np.ndarray, low: np.ndarray, close: np.ndarray,
    n: int,
):
    """
    State machine over confirmed pivots.

    Bullish setup (waiting for long entry):
      Activated by a new major swing high (= context: pullback expected).
      IDM low = most recent minor swing low after the major high.
      Sweep   = low pierces IDM but close stays above (canonical wick sweep).
      After sweep: IDM is LOCKED (no more updates from later minor lows).
      Invalidated by:
        - close > major_high (BoS up: context already shifted, no IDM needed)
        - close < idm_low    (real break: was not a sweep, just continuation down)

    Bearish setup: mirror.

    Returns 4 arrays of length n.
    """
    bull_idm_level = np.full(n, np.nan)
    bull_idm_swept = np.zeros(n, dtype=np.int8)
    bear_idm_level = np.full(n, np.nan)
    bear_idm_swept = np.zeros(n, dtype=np.int8)

    # Bullish state
    bull_active = False
    bull_major_high = np.nan
    bull_idm = np.nan
    bull_idm_locked = False

    # Bearish state
    bear_active = False
    bear_major_low = np.nan
    bear_idm = np.nan
    bear_idm_locked = False

    for i in range(n):
        c = close[i]
        h = high[i]
        l = low[i]

        # ── New major swing high → activate / refresh bullish setup ─
        if not np.isnan(pivot_high_major[i]):
            bull_active = True
            bull_major_high = pivot_high_major[i]
            bull_idm = np.nan
            bull_idm_locked = False

        # ── New major swing low → activate / refresh bearish setup ──
        if not np.isnan(pivot_low_major[i]):
            bear_active = True
            bear_major_low = pivot_low_major[i]
            bear_idm = np.nan
            bear_idm_locked = False

        # ── New minor pivot low while bullish active → update IDM ──
        if (bull_active and not bull_idm_locked
                and not np.isnan(pivot_low_minor[i])):
            bull_idm = pivot_low_minor[i]

        # ── New minor pivot high while bearish active → update IDM ──
        if (bear_active and not bear_idm_locked
                and not np.isnan(pivot_high_minor[i])):
            bear_idm = pivot_high_minor[i]

        # ── Bullish state checks ───────────────────────────────────
        if bull_active:
            # Invalidation: BoS up
            if c > bull_major_high:
                bull_active = False
                bull_major_high = np.nan
                bull_idm = np.nan
                bull_idm_locked = False
            # Invalidation: real close-break of IDM low
            elif not np.isnan(bull_idm) and c < bull_idm:
                bull_active = False
                bull_major_high = np.nan
                bull_idm = np.nan
                bull_idm_locked = False
            else:
                # Sweep check
                if not np.isnan(bull_idm) and not bull_idm_locked:
                    if l < bull_idm and c >= bull_idm:
                        bull_idm_swept[i] = 1
                        bull_idm_locked = True

        # ── Bearish state checks ───────────────────────────────────
        if bear_active:
            if c < bear_major_low:
                bear_active = False
                bear_major_low = np.nan
                bear_idm = np.nan
                bear_idm_locked = False
            elif not np.isnan(bear_idm) and c > bear_idm:
                bear_active = False
                bear_major_low = np.nan
                bear_idm = np.nan
                bear_idm_locked = False
            else:
                if not np.isnan(bear_idm) and not bear_idm_locked:
                    if h > bear_idm and c <= bear_idm:
                        bear_idm_swept[i] = 1
                        bear_idm_locked = True

        # ── Write per-bar state ────────────────────────────────────
        if bull_active and not np.isnan(bull_idm):
            bull_idm_level[i] = bull_idm
        if bear_active and not np.isnan(bear_idm):
            bear_idm_level[i] = bear_idm

    return bull_idm_level, bull_idm_swept, bear_idm_level, bear_idm_swept


# =============================================================================
# PUBLIC API
# =============================================================================

def annotate_inducement(
    df: pd.DataFrame,
    fractal_n_major: int = 5,
    fractal_n_minor: int = 2,
) -> pd.DataFrame:
    """
    Annotate `df` with IDM (Inducement) levels and sweep events.

    Parameters
    ----------
    df : OHLCV DataFrame.
    fractal_n_major : Williams half-window for MAJOR swing pivots (default 5).
    fractal_n_minor : Williams half-window for MINOR sub-pivots (default 2).

    Returns
    -------
    DataFrame with columns:
        pivot_high_major, pivot_low_major,
        pivot_high_minor, pivot_low_minor,
        bull_idm_level, bull_idm_swept,
        bear_idm_level, bear_idm_swept
    Index aligned with df.index.
    """
    n = len(df)
    high = df['high'].values.astype(np.float64)
    low = df['low'].values.astype(np.float64)
    close = df['close'].values.astype(np.float64)

    pH_maj, pL_maj = _detect_pivots_kernel(high, low, n, fractal_n_major)
    pH_min, pL_min = _detect_pivots_kernel(high, low, n, fractal_n_minor)

    bull_lvl, bull_sw, bear_lvl, bear_sw = _idm_kernel(
        pH_maj, pL_maj, pH_min, pL_min, high, low, close, n,
    )

    return pd.DataFrame({
        'pivot_high_major': pH_maj,
        'pivot_low_major': pL_maj,
        'pivot_high_minor': pH_min,
        'pivot_low_minor': pL_min,
        'bull_idm_level': bull_lvl,
        'bull_idm_swept': bull_sw,
        'bear_idm_level': bear_lvl,
        'bear_idm_swept': bear_sw,
    }, index=df.index)
