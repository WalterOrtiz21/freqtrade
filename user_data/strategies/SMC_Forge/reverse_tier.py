"""
SMC_Forge — Reverse Tier Classifier
====================================

Classifies the active counter-trend POI per bar into a quality tier
(S / A / A_minus / B / B_minus / NONE) for SMCForgeReverse strategy.

Tier semantics (canon SMC, ranked by institutional conviction):
    S        Breaker + HTF Breaker confluence
             (Smart Money flipped position; max conviction)
    A        OB + HTF OB confluence
    A_minus  OB + FVG nested confluence (any TF), or Breaker without HTF
    B        FVG-Breaker (IFVG), or FVG + HTF FVG confluence
    B_minus  Bare OB or FVG without HTF backing
    NONE     No active POI in that direction at this bar

Inputs (column conventions from SMCForge populate_indicators):
- LTF active POIs (engine output):
    active_{bullish|bearish}_{ob|fvg|breaker|fvg_breaker}_{top|bottom}
- HTF mirrors via configurable suffix (default '_htf').

The classifier is purely vectorized over the dataframe — no per-trade calls.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


_TIER_RANK = {
    'S': 0,
    'A': 1,
    'A_minus': 2,
    'B': 3,
    'B_minus': 4,
    'NONE': 99,
}

_TIERS_ORDERED = ('S', 'A', 'A_minus', 'B', 'B_minus')


def _present(df: pd.DataFrame, col: str) -> pd.Series:
    """A POI is "present" when its anchor column is positive (engine convention)."""
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    s = df[col]
    return s.notna() & (s > 0)


def _classify_tier_vec(
    has_ob: pd.Series,
    has_fvg: pd.Series,
    has_brk: pd.Series,
    has_fbrk: pd.Series,
    has_ob_htf: pd.Series,
    has_fvg_htf: pd.Series,
    has_brk_htf: pd.Series,
    index: pd.Index,
) -> pd.Series:
    """
    Apply the tier table priority. Conditions evaluated in order; first match
    wins (np.select semantics).
    """
    conditions = [
        has_brk & has_brk_htf,            # S
        has_ob & has_ob_htf,              # A
        has_ob & has_fvg,                 # A_minus (OB+FVG nested)
        has_brk,                          # A_minus (Breaker no HTF)
        has_fbrk,                         # B (FVG-Breaker / IFVG)
        has_fvg & has_fvg_htf,            # B (FVG + HTF FVG)
        has_ob | has_fvg,                 # B_minus (bare OB or FVG)
    ]
    choices = ['S', 'A', 'A_minus', 'A_minus', 'B', 'B', 'B_minus']
    tier = np.select(conditions, choices, default='NONE')
    return pd.Series(tier, index=index)


def classify_long_tier(df: pd.DataFrame, htf_suffix: str = '_htf') -> pd.Series:
    """Per-bar tier of the best active BULLISH POI."""
    has_ob = _present(df, 'active_bullish_ob_top')
    has_fvg = _present(df, 'active_bullish_fvg_top')
    has_brk = _present(df, 'active_bullish_breaker_top')
    has_fbrk = _present(df, 'active_bullish_fvg_breaker_top')
    has_ob_htf = _present(df, f'active_bullish_ob_top{htf_suffix}')
    has_fvg_htf = _present(df, f'active_bullish_fvg_top{htf_suffix}')
    has_brk_htf = _present(df, f'active_bullish_breaker_top{htf_suffix}')
    return _classify_tier_vec(
        has_ob, has_fvg, has_brk, has_fbrk,
        has_ob_htf, has_fvg_htf, has_brk_htf,
        df.index,
    ).rename('reverse_tier_long')


def classify_short_tier(df: pd.DataFrame, htf_suffix: str = '_htf') -> pd.Series:
    """Per-bar tier of the best active BEARISH POI."""
    has_ob = _present(df, 'active_bearish_ob_top')
    has_fvg = _present(df, 'active_bearish_fvg_top')
    has_brk = _present(df, 'active_bearish_breaker_top')
    has_fbrk = _present(df, 'active_bearish_fvg_breaker_top')
    has_ob_htf = _present(df, f'active_bearish_ob_top{htf_suffix}')
    has_fvg_htf = _present(df, f'active_bearish_fvg_top{htf_suffix}')
    has_brk_htf = _present(df, f'active_bearish_breaker_top{htf_suffix}')
    return _classify_tier_vec(
        has_ob, has_fvg, has_brk, has_fbrk,
        has_ob_htf, has_fvg_htf, has_brk_htf,
        df.index,
    ).rename('reverse_tier_short')


def tier_passes_min(tier_series: pd.Series, tier_min: str) -> pd.Series:
    """
    Boolean mask: True where bar's tier is >= tier_min in quality
    (i.e., rank <= threshold). 'NONE' always fails.
    """
    if tier_min not in _TIER_RANK:
        raise ValueError(
            f"tier_min must be one of {list(_TIERS_ORDERED)}, got {tier_min!r}"
        )
    threshold = _TIER_RANK[tier_min]
    return tier_series.map(_TIER_RANK).fillna(99).astype(int) <= threshold


def tier_size_factor(tier: str) -> float:
    """
    Sizing multiplier per tier (consumed by custom_stake_amount in strategy).
    Hardcoded canon: S/A full, A_minus 0.8, B 0.6, B_minus 0.4, NONE 0.
    """
    return {
        'S': 1.0,
        'A': 1.0,
        'A_minus': 0.8,
        'B': 0.6,
        'B_minus': 0.4,
        'NONE': 0.0,
    }.get(tier, 0.0)
