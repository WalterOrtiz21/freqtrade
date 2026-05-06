"""
SMC_Forge — Approach Strength
=============================

Quantifies whether price arrived at a POI with momentum or exhausted.

Hypothesis (canon SMC, formalized): mean-reversion at a POI requires that the
flow that drove price to the POI has run out of fuel. Symptoms in the bars
immediately preceding the tap: bodies shrinking, wicks dominating, range
contracting, volume decaying. A high-quality counter-trend entry needs a low
approach_strength score.

Anti-lookahead (CRITICAL)
-------------------------
For bar i, the window is [i - lookback, i - 1] — it EXCLUDES the current bar.
At the close of bar i-1, a bot can compute approach_strength.iloc[i] using
only past data. Implementation uses df.shift(1).rolling(lookback) so the
rolling window at index i never reads index i.

Public API
----------
    compute_approach_strength(df, lookback=7, baseline_window=20) -> Series

The returned Series:
- Score in [0, 1]. High = strong approach. Low = weak approach.
- NaN for bars where the rolling baseline is undefined (first
  baseline_window + 1 bars).
- Strategy uses `score < approach_strength_thr` (default 0.55) as the
  weak-approach gate for SMCForgeReverse entries.

Components (weighted 0.25 each)
-------------------------------
- velocity_ratio   = recent range mean / baseline range mean, clipped to 1.0
- body_ratio       = recent body mean  / baseline body mean,  clipped to 1.0
- rvol_score       = recent (volume / vol_sma) mean, clipped to 1.5, /1.5
- body_dominance   = recent body / (body + wick_total) mean,  clipped to 1.0
"""

from __future__ import annotations

import numpy as np
import pandas as pd


_REQUIRED_COLS = ('open', 'high', 'low', 'close', 'volume')


def _compute_components(
    df: pd.DataFrame,
    lookback: int,
    baseline_window: int,
) -> dict[str, pd.Series]:
    """
    Compute the 4 approach components as Series aligned to df.index.
    Exposed for unit tests; not part of the public API.
    """
    body = (df['close'] - df['open']).abs()
    rng = (df['high'] - df['low']).clip(lower=0.0)
    wick_total = (rng - body).clip(lower=0.0)

    bd_denom = (body + wick_total).replace(0.0, np.nan)
    body_dom = (body / bd_denom).fillna(0.0)

    vol_sma = df['volume'].rolling(baseline_window).mean()
    rvol = df['volume'] / vol_sma.replace(0.0, np.nan)

    body_recent = body.shift(1).rolling(lookback).mean()
    rng_recent = rng.shift(1).rolling(lookback).mean()
    body_dom_recent = body_dom.shift(1).rolling(lookback).mean()
    rvol_recent = rvol.shift(1).rolling(lookback).mean()

    body_baseline = body.shift(1).rolling(baseline_window).mean()
    rng_baseline = rng.shift(1).rolling(baseline_window).mean()

    body_baseline_safe = body_baseline.replace(0.0, np.nan)
    rng_baseline_safe = rng_baseline.replace(0.0, np.nan)

    velocity_ratio = (rng_recent / rng_baseline_safe).clip(0.0, 1.0)
    body_ratio = (body_recent / body_baseline_safe).clip(0.0, 1.0)
    rvol_score = (rvol_recent.clip(0.0, 1.5) / 1.5)
    body_dom_score = body_dom_recent.clip(0.0, 1.0)

    return {
        'velocity_ratio': velocity_ratio,
        'body_ratio': body_ratio,
        'rvol_score': rvol_score,
        'body_dominance': body_dom_score,
    }


def compute_approach_strength(
    df: pd.DataFrame,
    lookback: int = 7,
    baseline_window: int = 20,
) -> pd.Series:
    """
    Compute approach_strength score in [0, 1] over a backward-only window.

    Parameters
    ----------
    df : DataFrame with columns 'open', 'high', 'low', 'close', 'volume'.
    lookback : recent window size (canon: 5-10 bars).
    baseline_window : long-window for ratio baseline (typ. 20).

    Returns
    -------
    Series named 'approach_strength', index aligned to df.index.
    NaN until baseline rolling window is full (first baseline_window + 1 bars).
    """
    if lookback < 2:
        raise ValueError(f"lookback must be >= 2, got {lookback}")
    if baseline_window <= lookback:
        raise ValueError(
            f"baseline_window ({baseline_window}) must be > lookback ({lookback})"
        )
    missing = [c for c in _REQUIRED_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"df missing required columns: {missing}")

    n = len(df)
    if n < baseline_window + 1:
        return pd.Series(np.nan, index=df.index, name='approach_strength')

    components = _compute_components(df, lookback, baseline_window)
    score = (
        0.25 * components['velocity_ratio']
        + 0.25 * components['body_ratio']
        + 0.25 * components['rvol_score']
        + 0.25 * components['body_dominance']
    ).clip(0.0, 1.0)

    return score.rename('approach_strength')
