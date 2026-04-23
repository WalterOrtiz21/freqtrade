"""
SMC_Forge — Quality Annotation Layer
====================================

Annotates structural events from forge_engine (BoS, CHoCH) with quality
metrics. The first metric is DISPLACEMENT.

Canon SMC: a structural break without displacement is low-quality (retail-
grade). Real institutional moves are accompanied by 1-3 candles whose bodies
sum to >= 1 ATR in the break direction. Without that impulse, the "break"
is likely a fakeout that smart money will fade.

Public API
----------
    annotate_displacement(df, signals, atr_period=14, lookback=3) -> DataFrame

The returned DataFrame is the input `signals` plus:

    atr_{period}                       Wilder ATR (NaN until enough data)
    internal_bos_bull_disp             score at internal BoS bullish bars
    internal_bos_bear_disp             score at internal BoS bearish bars
    internal_choch_bull_disp           score at internal CHoCH bullish bars
    internal_choch_bear_disp           score at internal CHoCH bearish bars
    swing_bos_bull_disp                score at swing BoS bullish bars
    swing_bos_bear_disp                score at swing BoS bearish bars
    swing_choch_bull_disp              score at swing CHoCH bullish bars
    swing_choch_bear_disp              score at swing CHoCH bearish bars

Each *_disp column is 0 where no event fired, else the displacement
magnitude. Strategy code thresholds the magnitude (canon: >= 1.0 = strong).
"""

import numpy as np
import pandas as pd
from numba import jit


# =============================================================================
# NUMBA KERNELS
# =============================================================================

@jit(nopython=True, cache=True)
def _atr_kernel(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                n: int, period: int) -> np.ndarray:
    """
    Wilder ATR. Returns array of length n; values before bar `period` are NaN.
    """
    atr = np.full(n, np.nan)
    if n <= period:
        return atr

    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        h = high[i]
        l = low[i]
        pc = close[i - 1]
        d1 = h - l
        d2 = abs(h - pc)
        d3 = abs(l - pc)
        m = d1
        if d2 > m:
            m = d2
        if d3 > m:
            m = d3
        tr[i] = m

    # Initial ATR = simple mean of first `period` TRs (skip tr[0] = warmup)
    s = 0.0
    for i in range(1, period + 1):
        s += tr[i]
    atr[period] = s / period

    # Wilder smoothing
    for i in range(period + 1, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

    return atr


@jit(nopython=True, cache=True)
def _displacement_kernel(
    open_arr: np.ndarray, close: np.ndarray, atr: np.ndarray,
    int_bos_bull: np.ndarray, int_bos_bear: np.ndarray,
    int_choch_bull: np.ndarray, int_choch_bear: np.ndarray,
    sw_bos_bull: np.ndarray, sw_bos_bear: np.ndarray,
    sw_choch_bull: np.ndarray, sw_choch_bear: np.ndarray,
    n: int, lookback: int,
):
    """
    For each bar where a structural event fires, compute:

        score = sum(bodies in event direction over [i-lookback+1, i]) / ATR[i]

    Bullish event uses sum of bullish bodies (close > open).
    Bearish event uses sum of bearish bodies (open > close).
    Bodies in the wrong direction are ignored, not subtracted.

    Score interpretation:
        >= 2.0 : very strong institutional impulse
        >= 1.0 : valid displacement (canon threshold)
        >= 0.5 : marginal
        <  0.5 : weak / no displacement, probable fakeout

    Returns 8 arrays (one per event type) of length n.
    """
    out_int_bos_bull = np.zeros(n)
    out_int_bos_bear = np.zeros(n)
    out_int_choch_bull = np.zeros(n)
    out_int_choch_bear = np.zeros(n)
    out_sw_bos_bull = np.zeros(n)
    out_sw_bos_bear = np.zeros(n)
    out_sw_choch_bull = np.zeros(n)
    out_sw_choch_bear = np.zeros(n)

    for i in range(n):
        any_bull = (int_bos_bull[i] != 0 or int_choch_bull[i] != 0 or
                    sw_bos_bull[i] != 0 or sw_choch_bull[i] != 0)
        any_bear = (int_bos_bear[i] != 0 or int_choch_bear[i] != 0 or
                    sw_bos_bear[i] != 0 or sw_choch_bear[i] != 0)
        if not any_bull and not any_bear:
            continue

        a = atr[i]
        if np.isnan(a) or a <= 0.0:
            continue

        start = i - lookback + 1
        if start < 0:
            start = 0

        sum_bull_body = 0.0
        sum_bear_body = 0.0
        for k in range(start, i + 1):
            body = close[k] - open_arr[k]
            if body > 0:
                sum_bull_body += body
            elif body < 0:
                sum_bear_body += -body

        score_bull = sum_bull_body / a
        score_bear = sum_bear_body / a

        if any_bull:
            if int_bos_bull[i] != 0:
                out_int_bos_bull[i] = score_bull
            if int_choch_bull[i] != 0:
                out_int_choch_bull[i] = score_bull
            if sw_bos_bull[i] != 0:
                out_sw_bos_bull[i] = score_bull
            if sw_choch_bull[i] != 0:
                out_sw_choch_bull[i] = score_bull

        if any_bear:
            if int_bos_bear[i] != 0:
                out_int_bos_bear[i] = score_bear
            if int_choch_bear[i] != 0:
                out_int_choch_bear[i] = score_bear
            if sw_bos_bear[i] != 0:
                out_sw_bos_bear[i] = score_bear
            if sw_choch_bear[i] != 0:
                out_sw_choch_bear[i] = score_bear

    return (
        out_int_bos_bull, out_int_bos_bear,
        out_int_choch_bull, out_int_choch_bear,
        out_sw_bos_bull, out_sw_bos_bear,
        out_sw_choch_bull, out_sw_choch_bear,
    )


# =============================================================================
# PUBLIC API
# =============================================================================

def annotate_displacement(
    df: pd.DataFrame,
    signals: pd.DataFrame,
    atr_period: int = 14,
    lookback: int = 3,
) -> pd.DataFrame:
    """
    Annotate signals (output of SMCEngine.get_signals()) with displacement
    magnitude per structural event.

    Parameters
    ----------
    df : OHLCV DataFrame with columns open, high, low, close, volume.
    signals : DataFrame from SMCEngine.get_signals() aligned with df.
    atr_period : Wilder ATR period (default 14).
    lookback : bars to sum bodies over, including the event bar (default 3).

    Returns
    -------
    DataFrame : signals + atr_{period} + 8 *_disp columns.
    """
    n = len(df)
    if n != len(signals):
        raise ValueError(
            f"df and signals must have same length: got {n} and {len(signals)}"
        )

    high = df['high'].values.astype(np.float64)
    low = df['low'].values.astype(np.float64)
    close = df['close'].values.astype(np.float64)
    open_arr = df['open'].values.astype(np.float64)

    atr = _atr_kernel(high, low, close, n, atr_period)

    (
        out_ibb, out_ibr,
        out_icb, out_icr,
        out_sbb, out_sbr,
        out_scb, out_scr,
    ) = _displacement_kernel(
        open_arr, close, atr,
        signals['internal_bos_bullish'].values.astype(np.int8),
        signals['internal_bos_bearish'].values.astype(np.int8),
        signals['internal_choch_bullish'].values.astype(np.int8),
        signals['internal_choch_bearish'].values.astype(np.int8),
        signals['swing_bos_bullish'].values.astype(np.int8),
        signals['swing_bos_bearish'].values.astype(np.int8),
        signals['swing_choch_bullish'].values.astype(np.int8),
        signals['swing_choch_bearish'].values.astype(np.int8),
        n, lookback,
    )

    out = signals.copy()
    out[f'atr_{atr_period}'] = atr
    out['internal_bos_bull_disp'] = out_ibb
    out['internal_bos_bear_disp'] = out_ibr
    out['internal_choch_bull_disp'] = out_icb
    out['internal_choch_bear_disp'] = out_icr
    out['swing_bos_bull_disp'] = out_sbb
    out['swing_bos_bear_disp'] = out_sbr
    out['swing_choch_bull_disp'] = out_scb
    out['swing_choch_bear_disp'] = out_scr

    return out
