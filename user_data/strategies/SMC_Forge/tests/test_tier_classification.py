"""
Tests for reverse_tier — POI tier classification.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reverse_tier import (  # noqa: E402
    _TIER_RANK,
    classify_long_tier,
    classify_short_tier,
    tier_passes_min,
    tier_size_factor,
)


# =============================================================================
# Helpers — build df with controlled POI presence
# =============================================================================

def _empty_long_df(n=10, htf_suffix='_htf'):
    """All POI columns set to 0 (no active POI)."""
    cols = {}
    for ob_type in ('ob', 'fvg', 'breaker', 'fvg_breaker'):
        for direction in ('bullish', 'bearish'):
            for level in ('top', 'bottom'):
                cols[f'active_{direction}_{ob_type}_{level}'] = [0.0] * n
                cols[f'active_{direction}_{ob_type}_{level}{htf_suffix}'] = [0.0] * n
    return pd.DataFrame(cols)


def _set_poi(df, direction, poi_type, indices, top=100.0, bottom=99.0, htf=False, htf_suffix='_htf'):
    """In-place set POI top/bottom at given indices for direction + type."""
    suffix = htf_suffix if htf else ''
    df.loc[indices, f'active_{direction}_{poi_type}_top{suffix}'] = top
    df.loc[indices, f'active_{direction}_{poi_type}_bottom{suffix}'] = bottom
    return df


# =============================================================================
# 1. NONE when no POI active
# =============================================================================

class TestNoneTier:

    def test_empty_df_returns_none_long(self):
        df = _empty_long_df(n=5)
        out = classify_long_tier(df)
        assert (out == 'NONE').all()

    def test_empty_df_returns_none_short(self):
        df = _empty_long_df(n=5)
        out = classify_short_tier(df)
        assert (out == 'NONE').all()

    def test_only_short_poi_does_not_set_long_tier(self):
        df = _empty_long_df(n=5)
        _set_poi(df, 'bearish', 'ob', [2, 3])
        long_tier = classify_long_tier(df)
        assert (long_tier == 'NONE').all()
        short_tier = classify_short_tier(df)
        assert short_tier.iloc[2] != 'NONE'


# =============================================================================
# 2. Tier S — Breaker + HTF Breaker
# =============================================================================

class TestTierS:

    def test_breaker_with_htf_breaker_is_S(self):
        df = _empty_long_df(n=5)
        _set_poi(df, 'bullish', 'breaker', [2])
        _set_poi(df, 'bullish', 'breaker', [2], htf=True)
        out = classify_long_tier(df)
        assert out.iloc[2] == 'S'

    def test_breaker_without_htf_is_not_S(self):
        df = _empty_long_df(n=5)
        _set_poi(df, 'bullish', 'breaker', [2])
        out = classify_long_tier(df)
        assert out.iloc[2] != 'S'

    def test_breaker_short_with_htf_is_S(self):
        df = _empty_long_df(n=5)
        _set_poi(df, 'bearish', 'breaker', [3])
        _set_poi(df, 'bearish', 'breaker', [3], htf=True)
        out = classify_short_tier(df)
        assert out.iloc[3] == 'S'


# =============================================================================
# 3. Tier A — OB + HTF OB
# =============================================================================

class TestTierA:

    def test_ob_with_htf_ob_is_A(self):
        df = _empty_long_df(n=5)
        _set_poi(df, 'bullish', 'ob', [1])
        _set_poi(df, 'bullish', 'ob', [1], htf=True)
        out = classify_long_tier(df)
        assert out.iloc[1] == 'A'

    def test_ob_without_htf_is_not_A(self):
        df = _empty_long_df(n=5)
        _set_poi(df, 'bullish', 'ob', [1])
        out = classify_long_tier(df)
        assert out.iloc[1] != 'A'


# =============================================================================
# 4. Tier A_minus
# =============================================================================

class TestTierAMinus:

    def test_ob_plus_fvg_nested_is_A_minus(self):
        df = _empty_long_df(n=5)
        _set_poi(df, 'bullish', 'ob', [2])
        _set_poi(df, 'bullish', 'fvg', [2])
        # No HTF for either — nested confluence alone gives A_minus.
        out = classify_long_tier(df)
        assert out.iloc[2] == 'A_minus'

    def test_breaker_without_htf_is_A_minus(self):
        df = _empty_long_df(n=5)
        _set_poi(df, 'bullish', 'breaker', [3])
        out = classify_long_tier(df)
        assert out.iloc[3] == 'A_minus'


# =============================================================================
# 5. Tier B
# =============================================================================

class TestTierB:

    def test_fvg_breaker_is_B(self):
        df = _empty_long_df(n=5)
        _set_poi(df, 'bullish', 'fvg_breaker', [1])
        out = classify_long_tier(df)
        assert out.iloc[1] == 'B'

    def test_fvg_with_htf_fvg_is_B(self):
        df = _empty_long_df(n=5)
        _set_poi(df, 'bullish', 'fvg', [2])
        _set_poi(df, 'bullish', 'fvg', [2], htf=True)
        out = classify_long_tier(df)
        assert out.iloc[2] == 'B'


# =============================================================================
# 6. Tier B_minus
# =============================================================================

class TestTierBMinus:

    def test_bare_fvg_is_B_minus(self):
        df = _empty_long_df(n=5)
        _set_poi(df, 'bullish', 'fvg', [4])
        out = classify_long_tier(df)
        assert out.iloc[4] == 'B_minus'

    def test_bare_ob_no_htf_no_nested_is_B_minus(self):
        df = _empty_long_df(n=5)
        _set_poi(df, 'bullish', 'ob', [4])
        out = classify_long_tier(df)
        assert out.iloc[4] == 'B_minus'


# =============================================================================
# 7. Priority resolution — multiple POIs at same bar
# =============================================================================

class TestPriority:

    def test_S_wins_over_A_when_both_present(self):
        df = _empty_long_df(n=5)
        # S setup
        _set_poi(df, 'bullish', 'breaker', [2])
        _set_poi(df, 'bullish', 'breaker', [2], htf=True)
        # A setup at same bar
        _set_poi(df, 'bullish', 'ob', [2])
        _set_poi(df, 'bullish', 'ob', [2], htf=True)
        out = classify_long_tier(df)
        assert out.iloc[2] == 'S'

    def test_A_wins_over_A_minus(self):
        df = _empty_long_df(n=5)
        # A setup
        _set_poi(df, 'bullish', 'ob', [2])
        _set_poi(df, 'bullish', 'ob', [2], htf=True)
        # A_minus setup at same bar (nested fvg + bare breaker)
        _set_poi(df, 'bullish', 'fvg', [2])
        _set_poi(df, 'bullish', 'breaker', [2])
        out = classify_long_tier(df)
        assert out.iloc[2] == 'A'

    def test_A_minus_wins_over_B(self):
        df = _empty_long_df(n=5)
        # A_minus: nested OB+FVG
        _set_poi(df, 'bullish', 'ob', [2])
        _set_poi(df, 'bullish', 'fvg', [2])
        # B setup also: FVG-breaker
        _set_poi(df, 'bullish', 'fvg_breaker', [2])
        out = classify_long_tier(df)
        assert out.iloc[2] == 'A_minus'


# =============================================================================
# 8. tier_passes_min
# =============================================================================

class TestTierPassesMin:

    def test_invalid_tier_min_raises(self):
        s = pd.Series(['S', 'A'])
        with pytest.raises(ValueError, match="tier_min must be one of"):
            tier_passes_min(s, 'INVALID')

    def test_tier_min_S_only_S_passes(self):
        s = pd.Series(['S', 'A', 'A_minus', 'B', 'B_minus', 'NONE'])
        out = tier_passes_min(s, 'S')
        assert out.tolist() == [True, False, False, False, False, False]

    def test_tier_min_A_minus_passes_S_A_AMinus(self):
        s = pd.Series(['S', 'A', 'A_minus', 'B', 'B_minus', 'NONE'])
        out = tier_passes_min(s, 'A_minus')
        assert out.tolist() == [True, True, True, False, False, False]

    def test_tier_min_B_minus_passes_all_active(self):
        s = pd.Series(['S', 'A', 'A_minus', 'B', 'B_minus', 'NONE'])
        out = tier_passes_min(s, 'B_minus')
        assert out.tolist() == [True, True, True, True, True, False]


# =============================================================================
# 9. tier_size_factor
# =============================================================================

class TestTierSizeFactor:

    @pytest.mark.parametrize("tier,expected", [
        ('S', 1.0),
        ('A', 1.0),
        ('A_minus', 0.8),
        ('B', 0.6),
        ('B_minus', 0.4),
        ('NONE', 0.0),
        ('UNKNOWN', 0.0),
    ])
    def test_factor_matches_canon(self, tier, expected):
        assert tier_size_factor(tier) == expected


# =============================================================================
# 10. HTF suffix configurable
# =============================================================================

class TestHtfSuffix:

    def test_h4_suffix_picks_up_correct_columns(self):
        df = _empty_long_df(n=5, htf_suffix='_h4')
        _set_poi(df, 'bullish', 'ob', [2], htf_suffix='_h4')
        _set_poi(df, 'bullish', 'ob', [2], htf=True, htf_suffix='_h4')
        out = classify_long_tier(df, htf_suffix='_h4')
        assert out.iloc[2] == 'A'

    def test_default_suffix_does_not_match_h4_columns(self):
        df = _empty_long_df(n=5, htf_suffix='_h4')
        _set_poi(df, 'bullish', 'ob', [2], htf_suffix='_h4')
        _set_poi(df, 'bullish', 'ob', [2], htf=True, htf_suffix='_h4')
        # Default suffix '_htf' — '_h4' columns won't be matched as HTF
        out = classify_long_tier(df, htf_suffix='_htf')
        # OB present at LTF (the '_h4' col is used as LTF here), so it's
        # downgraded to B_minus (bare OB, no HTF detected with '_htf' suffix).
        assert out.iloc[2] == 'B_minus'
