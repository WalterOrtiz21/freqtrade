"""
SMCForgeExpC2 — Experiment C2: weak swings as TP targets at ENTRY TF.

Hypothesis: the value of weak-swing classification is at the ENTRY timeframe.
In Exp C (structural_htf), OB/FVG dominated pool selection and HTF weak swings
rarely won. C2 tests whether excluding STRONG swings from the entry-TF TP pool
(strong = structural support/resistance, price respects them) and targeting only
WEAK swings (likely to be swept) improves outcome vs the baseline.

Changes vs E0 (SMCForge baseline):
  - tp_mode = structural (entry TF, not HTF)
  - _LONG_TP_COLS: swap swing_high + pivot_high_level → swing_high_weak only
  - _SHORT_TP_COLS: swap swing_low + pivot_low_level  → swing_low_weak only
  - pivot_*_level excluded: minor fractal pivots re-introduce noise targets

Entry-TF weak swing columns derived in populate_indicators:
  swing_high_weak = swing_high level when swing_high_is_weak == 1, else 0.0
  swing_low_weak  = swing_low  level when swing_low_is_weak  == 1, else 0.0
  (0.0 is filtered by _safe_float so it acts as NaN for TP selection)

Reference for comparison:
  E0  (fixed TP):               +60.79%, PF 1.33, Calmar 20.28, DD 11.76%
  E2  (structural entry-TF):    ~+42% (historical — truncated by strong swings)
  ExpC (structural_htf):        +60.44%, PF 1.32, Calmar 20.12, DD 11.79%

Hypothesis C2: removing strong swings un-truncates the right tail vs E2.
"""

import numpy as np
import pandas as pd
from pandas import DataFrame

from SMCForge import SMCForge


class SMCForgeExpC2(SMCForge):
    """Experiment C2: entry-TF structural TP targeting weak swings only."""

    timeframe = '1h'

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Extend base indicators with entry-TF weak swing level columns."""
        df = super().populate_indicators(dataframe, metadata)

        # Derive weak swing level columns:
        #   swing_high_weak = swing_high price ONLY when the pivot is classified
        #                     as weak (likely liquidity target to be swept).
        #                     Returns 0.0 for strong pivots → _safe_float skips.
        #   swing_low_weak  = mirror for lows.
        # swing_high_is_weak and swing_low_is_weak come from forge_engine with
        # shift(1) already applied (no lookahead).
        df['swing_high_weak'] = np.where(
            df['swing_high_is_weak'] == 1,
            df['swing_high'],
            0.0,
        )
        df['swing_low_weak'] = np.where(
            df['swing_low_is_weak'] == 1,
            df['swing_low'],
            0.0,
        )
        return df

    # TP candidate columns — entry TF, weak swings only.
    # Long TP = bearish liquidity above price: EQH, OB tops, FVG tops,
    #            weak swing highs (price will sweep them).
    # Strong swing_high excluded (structural resistance — price respects it, not a TP target).
    # pivot_high_level excluded (minor fractal — re-introduces near noise targets).
    _LONG_TP_COLS = (
        'eqh_level',
        'active_bearish_ob_top',
        'active_bearish_fvg_top',
        'active_bearish_breaker_top',
        'active_bearish_fvg_breaker_top',
        'swing_high_weak',
    )
    # Short TP = bullish liquidity below price: EQL, OB bottoms, FVG bottoms,
    #             weak swing lows (price will sweep them).
    _SHORT_TP_COLS = (
        'eql_level',
        'active_bullish_ob_bottom',
        'active_bullish_fvg_bottom',
        'active_bullish_breaker_bottom',
        'active_bullish_fvg_breaker_bottom',
        'swing_low_weak',
    )
