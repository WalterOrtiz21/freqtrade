"""
SMC_Forge — Liquidity Levels Layer
==================================

Detects EQUAL HIGHS (EQH) and EQUAL LOWS (EQL) — clusters of swing pivots
sitting at nearly the same price, which act as obvious pools of stop-loss
liquidity for retail traders. Smart money targets these pools before
delivering the next directional move.

Canon SMC: an EQH/EQL is a magnet. When the cluster gets swept (price wicks
through and reverses), it's a high-quality liquidity grab — typically the
prelude to a CHoCH + displacement in the opposite direction.

Public API
----------
    annotate_eqh_eql(df, atr_period=14, fractal_n=5, max_pivots=10,
                     tolerance_atr=0.10) -> DataFrame

Returns df enriched with:

    atr_{period}        Wilder ATR
    pivot_high_level    Level of confirmed swing high pivot (NaN if none at i)
    pivot_low_level     Level of confirmed swing low pivot (NaN if none at i)
    eqh_level           Active EQH cluster mean level (NaN if none/broken)
    eqh_count           # of pivots in active EQH cluster (0 if none, >=2 if active)
    eql_level           Active EQL cluster mean level
    eql_count           # of pivots in active EQL cluster

Notes
-----
- Pivots are Williams fractals with `fractal_n` bars on EACH side. They get
  confirmed `fractal_n` bars AFTER they form (no lookahead).
- A new pivot triggers cluster detection by scanning the last `max_pivots`
  confirmed pivots; if 2+ are within `tolerance_atr * ATR` of the new one,
  the cluster is registered and exposed as the active EQH/EQL.
- Only the MOST RECENT cluster is exposed (multi-cluster tracking is a
  future iteration). The active level persists bar-by-bar until invalidated
  by a `close` that breaks through (with the same tolerance).
"""

import numpy as np
import pandas as pd
from numba import jit

from forge_quality import _atr_kernel  # share ATR with quality layer


# =============================================================================
# NUMBA KERNELS
# =============================================================================

@jit(nopython=True, cache=True)
def _detect_pivots_kernel(high: np.ndarray, low: np.ndarray,
                          n: int, fractal_n: int):
    """
    Williams fractal pivot detection.

    Bar `c` is a swing high iff high[c] > high[k] for all k in
    [c-fractal_n, c+fractal_n] excluding c itself. Same for swing low (with
    low[c] < low[k]).

    The pivot is WRITTEN at the confirmation bar (c + fractal_n), not at c,
    so downstream consumers can use it bar-by-bar without lookahead.

    Returns (pivot_high, pivot_low) arrays of length n. Values are NaN where
    no pivot was confirmed at that bar.
    """
    pivot_high = np.full(n, np.nan)
    pivot_low = np.full(n, np.nan)

    for i in range(2 * fractal_n, n):
        c = i - fractal_n  # candidate pivot bar
        cand_high = high[c]
        cand_low = low[c]
        is_high = True
        is_low = True

        # Left side
        for k in range(c - fractal_n, c):
            if high[k] >= cand_high:
                is_high = False
            if low[k] <= cand_low:
                is_low = False

        # Right side (only if still possible)
        if is_high:
            for k in range(c + 1, c + fractal_n + 1):
                if high[k] >= cand_high:
                    is_high = False
                    break
        if is_low:
            for k in range(c + 1, c + fractal_n + 1):
                if low[k] <= cand_low:
                    is_low = False
                    break

        if is_high:
            pivot_high[i] = cand_high
        if is_low:
            pivot_low[i] = cand_low

    return pivot_high, pivot_low


@jit(nopython=True, cache=True)
def _eqh_eql_kernel(
    pivot_high: np.ndarray, pivot_low: np.ndarray,
    close: np.ndarray, atr: np.ndarray,
    n: int, max_pivots: int, tolerance_atr: float,
):
    """
    For each newly confirmed pivot, check ring buffer of last `max_pivots`
    pivots for matches within `tolerance_atr * ATR`. If a cluster of 2+ is
    formed, register the average level as the active EQH/EQL.

    Active level is invalidated when `close` breaks through (with tolerance).

    Returns four arrays of length n.
    """
    eqh_level = np.full(n, np.nan)
    eqh_count = np.zeros(n, dtype=np.int8)
    eql_level = np.full(n, np.nan)
    eql_count = np.zeros(n, dtype=np.int8)

    # Ring buffers
    rh = np.full(max_pivots, np.nan)
    rh_n = 0
    rl = np.full(max_pivots, np.nan)
    rl_n = 0

    # Active cluster state
    curr_eqh = np.nan
    curr_eqh_n = 0
    curr_eql = np.nan
    curr_eql_n = 0

    for i in range(n):
        a = atr[i]
        if np.isnan(a) or a <= 0.0:
            eps = 0.0
        else:
            eps = a * tolerance_atr

        # ── New pivot high? ─────────────────────────────────────
        new_ph = pivot_high[i]
        if not np.isnan(new_ph):
            # Append to ring buffer (shift if full)
            if rh_n < max_pivots:
                rh[rh_n] = new_ph
                rh_n += 1
            else:
                for k in range(max_pivots - 1):
                    rh[k] = rh[k + 1]
                rh[max_pivots - 1] = new_ph

            # Cluster detection
            if eps > 0.0 and rh_n >= 2:
                cluster_sum = new_ph
                cluster_n = 1
                for k in range(rh_n - 1):
                    if abs(rh[k] - new_ph) <= eps:
                        cluster_sum += rh[k]
                        cluster_n += 1
                if cluster_n >= 2:
                    curr_eqh = cluster_sum / cluster_n
                    curr_eqh_n = cluster_n

        # ── New pivot low? ─────────────────────────────────────
        new_pl = pivot_low[i]
        if not np.isnan(new_pl):
            if rl_n < max_pivots:
                rl[rl_n] = new_pl
                rl_n += 1
            else:
                for k in range(max_pivots - 1):
                    rl[k] = rl[k + 1]
                rl[max_pivots - 1] = new_pl

            if eps > 0.0 and rl_n >= 2:
                cluster_sum = new_pl
                cluster_n = 1
                for k in range(rl_n - 1):
                    if abs(rl[k] - new_pl) <= eps:
                        cluster_sum += rl[k]
                        cluster_n += 1
                if cluster_n >= 2:
                    curr_eql = cluster_sum / cluster_n
                    curr_eql_n = cluster_n

        # ── Invalidation by close break ────────────────────────
        if not np.isnan(curr_eqh) and close[i] > curr_eqh + eps:
            curr_eqh = np.nan
            curr_eqh_n = 0
        if not np.isnan(curr_eql) and close[i] < curr_eql - eps:
            curr_eql = np.nan
            curr_eql_n = 0

        # ── Write state ────────────────────────────────────────
        eqh_level[i] = curr_eqh
        eqh_count[i] = curr_eqh_n
        eql_level[i] = curr_eql
        eql_count[i] = curr_eql_n

    return eqh_level, eqh_count, eql_level, eql_count


# =============================================================================
# PUBLIC API
# =============================================================================

def annotate_eqh_eql(
    df: pd.DataFrame,
    atr_period: int = 14,
    fractal_n: int = 5,
    max_pivots: int = 10,
    tolerance_atr: float = 0.10,
) -> pd.DataFrame:
    """
    Annotate `df` with EQH/EQL cluster levels.

    Parameters
    ----------
    df : OHLCV DataFrame.
    atr_period : Wilder ATR period (default 14).
    fractal_n : Williams fractal half-window (default 5 = ICT-friendly).
    max_pivots : Recent pivots considered for clustering (default 10).
    tolerance_atr : Two pivots cluster if within this fraction of ATR
                    (default 0.10 = 10%).

    Returns
    -------
    DataFrame containing only the new columns:
        atr_{period}, pivot_high_level, pivot_low_level,
        eqh_level, eqh_count, eql_level, eql_count
    Index aligned with df.index. Caller may pd.concat to merge.
    """
    n = len(df)
    high = df['high'].values.astype(np.float64)
    low = df['low'].values.astype(np.float64)
    close = df['close'].values.astype(np.float64)

    atr = _atr_kernel(high, low, close, n, atr_period)
    pivot_high, pivot_low = _detect_pivots_kernel(high, low, n, fractal_n)
    eqh_level, eqh_count, eql_level, eql_count = _eqh_eql_kernel(
        pivot_high, pivot_low, close, atr, n, max_pivots, tolerance_atr
    )

    return pd.DataFrame({
        f'atr_{atr_period}': atr,
        'pivot_high_level': pivot_high,
        'pivot_low_level': pivot_low,
        'eqh_level': eqh_level,
        'eqh_count': eqh_count,
        'eql_level': eql_level,
        'eql_count': eql_count,
    }, index=df.index)
