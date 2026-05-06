"""
Tests for reverse_approach.compute_approach_strength.

Critical invariant covered: NO LOOKAHEAD. The score at bar i must depend
only on bars [0, i-1].
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reverse_approach import (  # noqa: E402
    _compute_components,
    compute_approach_strength,
)


# =============================================================================
# Helpers
# =============================================================================

def make_df(bars):
    """bars: list of (open, high, low, close, volume)."""
    df = pd.DataFrame(bars, columns=['open', 'high', 'low', 'close', 'volume'])
    df['date'] = pd.date_range('2024-01-01', periods=len(df), freq='15min')
    return df


def constant_bars(n, open_=100.0, high=101.0, low=99.0, close=100.5, volume=100.0):
    """All identical bars — useful for stable baselines."""
    return [(open_, high, low, close, volume)] * n


# =============================================================================
# 1. Input validation
# =============================================================================

class TestInputValidation:

    def test_missing_columns_raises(self):
        df = pd.DataFrame({'open': [1.0], 'close': [1.0]})
        with pytest.raises(KeyError, match="missing required columns"):
            compute_approach_strength(df)

    def test_lookback_too_small_raises(self):
        df = make_df(constant_bars(50))
        with pytest.raises(ValueError, match="lookback must be >= 2"):
            compute_approach_strength(df, lookback=1)

    def test_baseline_not_greater_than_lookback_raises(self):
        df = make_df(constant_bars(50))
        with pytest.raises(ValueError, match="must be > lookback"):
            compute_approach_strength(df, lookback=10, baseline_window=10)

    def test_short_df_returns_all_nan(self):
        df = make_df(constant_bars(15))
        out = compute_approach_strength(df, lookback=7, baseline_window=20)
        assert len(out) == 15
        assert out.isna().all()


# =============================================================================
# 2. Warmup behavior
# =============================================================================

class TestWarmup:

    def test_first_bars_are_nan(self):
        df = make_df(constant_bars(50))
        lookback, baseline = 7, 20
        out = compute_approach_strength(df, lookback=lookback,
                                         baseline_window=baseline)
        # The limiting component is rvol_score: rvol depends on vol_sma which
        # is undefined for the first baseline-1 bars. Then rvol.shift(1)
        # .rolling(lookback) needs all past `lookback` bars valid, so the
        # first defined output bar is baseline + lookback - 1.
        first_valid = baseline + lookback - 1
        assert out.iloc[:first_valid].isna().all()
        assert not np.isnan(out.iloc[first_valid])

    def test_warmup_length_scales_with_baseline(self):
        df = make_df(constant_bars(80))
        short = compute_approach_strength(df, lookback=5, baseline_window=20)
        long = compute_approach_strength(df, lookback=5, baseline_window=40)
        # Larger baseline -> later first non-NaN.
        first_short = short.first_valid_index()
        first_long = long.first_valid_index()
        assert first_short < first_long


# =============================================================================
# 3. NO-LOOKAHEAD (critical)
# =============================================================================

class TestNoLookahead:
    """
    For each k where the score at index k is defined, the score must equal
    the score computed on df[:k+1] at index k. Future bars (>k) MUST NOT
    influence the score at k.
    """

    def test_score_is_invariant_under_truncation(self):
        rng = np.random.default_rng(seed=42)
        n = 80
        bars = []
        for _ in range(n):
            o = 100.0 + rng.normal(0, 1)
            c = o + rng.normal(0, 0.5)
            h = max(o, c) + abs(rng.normal(0, 0.3))
            l = min(o, c) - abs(rng.normal(0, 0.3))
            v = 100.0 * abs(rng.normal(1.0, 0.3))
            bars.append((o, h, l, c, v))
        df_full = make_df(bars)
        score_full = compute_approach_strength(df_full, lookback=5, baseline_window=20)

        # For every bar k where score_full[k] is defined, recompute on the
        # truncated frame df[:k+1] and assert equality at index k.
        for k in range(len(df_full)):
            v_full = score_full.iloc[k]
            if pd.isna(v_full):
                continue
            df_trunc = df_full.iloc[: k + 1].reset_index(drop=True)
            score_trunc = compute_approach_strength(
                df_trunc, lookback=5, baseline_window=20
            )
            v_trunc = score_trunc.iloc[k]
            if pd.isna(v_trunc):
                # Acceptable only if rolling baseline can't yet stabilize on
                # the shortest frame. Should not happen for k >= 20.
                assert k < 20, (
                    f"bar {k}: full had value {v_full} but truncated returned NaN"
                )
                continue
            assert v_full == pytest.approx(v_trunc, rel=1e-9, abs=1e-12), (
                f"lookahead violation at bar {k}: full={v_full}, trunc={v_trunc}"
            )

    def test_components_window_excludes_current_bar(self):
        """
        Build a frame whose first 30 bars are constant, then bar 30 is an
        explosive bar (huge body+volume). The score AT bar 30 must be
        unaffected by bar 30 itself: it should reflect the constant past.
        """
        n_calm = 30
        bars = constant_bars(n_calm)
        # Bar at index 30: 5x larger body, 5x volume.
        bars.append((100.0, 110.0, 95.0, 109.0, 500.0))
        # 5 more bars to keep going.
        bars.extend(constant_bars(5))
        df = make_df(bars)

        score = compute_approach_strength(df, lookback=5, baseline_window=20)
        # The score AT bar 30 uses the calm window [25..29] only.
        # Score AT bar 31 uses [26..30] which includes the explosive bar.
        s30 = score.iloc[30]
        s31 = score.iloc[31]
        assert not np.isnan(s30)
        assert not np.isnan(s31)
        # The explosive bar should bump body/range/volume metrics in s31.
        assert s31 > s30, (
            f"explosive bar at i=30 should appear in score[31] (={s31}) "
            f"and NOT in score[30] (={s30})"
        )


# =============================================================================
# 4. Component semantics
# =============================================================================

class TestComponents:

    def test_constant_data_gives_predictable_score(self):
        df = make_df(constant_bars(60, open_=100.0, high=101.0, low=99.0,
                                    close=100.5, volume=100.0))
        c = _compute_components(df, lookback=5, baseline_window=20)
        # Identical bars: recent / baseline ratios must be ~1.0.
        idx = 30
        assert c['velocity_ratio'].iloc[idx] == pytest.approx(1.0, abs=1e-9)
        assert c['body_ratio'].iloc[idx] == pytest.approx(1.0, abs=1e-9)
        # rvol_recent ~1, /1.5 ~ 0.667
        assert c['rvol_score'].iloc[idx] == pytest.approx(1.0 / 1.5, abs=1e-9)
        # body_dom: body=0.5, wick_total=2-0.5=1.5 -> 0.5/2 = 0.25
        assert c['body_dominance'].iloc[idx] == pytest.approx(0.25, abs=1e-9)

    def test_strong_approach_scores_high(self):
        # 30 calm bars (small bodies, low vol), then 10 bars of expanding
        # bodies + rising volume. Score on the last bar (using last 5
        # explosive bars) should be high.
        bars = constant_bars(30, open_=100.0, high=100.5, low=99.5,
                              close=100.05, volume=100.0)
        for i in range(10):
            o = 100.0
            c_close = 100.0 + (i + 1) * 0.8  # growing body
            h = c_close + 0.05
            l = o - 0.05
            v = 200.0 + i * 50.0  # rising volume
            bars.append((o, h, l, c_close, v))
        df = make_df(bars)
        score = compute_approach_strength(df, lookback=5, baseline_window=20)
        # Last bar's score uses [34..38] window — all explosive.
        assert score.iloc[-1] > 0.6, (
            f"expected strong approach > 0.6, got {score.iloc[-1]}"
        )

    def test_weak_approach_scores_low(self):
        # 30 wide-range, big-body, high-volume bars (baseline fierce),
        # then 10 narrow bars with shrinking volume.
        bars = []
        for _ in range(30):
            bars.append((100.0, 102.0, 98.0, 101.5, 200.0))  # big bodies
        for i in range(10):
            o = 100.0
            c_close = 100.0 + 0.05
            h = 100.1
            l = 99.9
            v = 30.0  # collapsed volume
            bars.append((o, h, l, c_close, v))
        df = make_df(bars)
        score = compute_approach_strength(df, lookback=5, baseline_window=20)
        last = score.iloc[-1]
        assert last < 0.5, (
            f"expected weak approach < 0.5, got {last}"
        )

    def test_zero_volume_does_not_crash(self):
        bars = constant_bars(30, volume=0.0)
        df = make_df(bars)
        # Should not raise; rvol becomes NaN, score may be NaN.
        out = compute_approach_strength(df, lookback=5, baseline_window=20)
        assert len(out) == 30
        # All values either NaN or finite, never inf.
        finite = out.dropna()
        assert np.all(np.isfinite(finite.values))


# =============================================================================
# 5. Output shape & dtype
# =============================================================================

class TestOutputShape:

    def test_returns_named_series(self):
        df = make_df(constant_bars(40))
        out = compute_approach_strength(df)
        assert isinstance(out, pd.Series)
        assert out.name == 'approach_strength'
        assert len(out) == len(df)
        assert out.index.equals(df.index)

    def test_score_in_unit_interval_when_defined(self):
        rng = np.random.default_rng(seed=7)
        n = 100
        bars = []
        for _ in range(n):
            o = 100.0 + rng.normal(0, 1)
            c = o + rng.normal(0, 0.5)
            h = max(o, c) + abs(rng.normal(0, 0.3))
            l = min(o, c) - abs(rng.normal(0, 0.3))
            v = 100.0 * abs(rng.normal(1.0, 0.5))
            bars.append((o, h, l, c, v))
        df = make_df(bars)
        out = compute_approach_strength(df).dropna()
        assert (out >= 0.0).all()
        assert (out <= 1.0).all()
