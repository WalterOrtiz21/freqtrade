"""
Tests for forge_levels: pivot detection and EQH/EQL clustering.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forge_levels import (  # noqa: E402
    annotate_eqh_eql,
    _detect_pivots_kernel,
    _eqh_eql_kernel,
)


# =============================================================================
# Helpers
# =============================================================================

def make_df(bars):
    df = pd.DataFrame(bars, columns=['open', 'high', 'low', 'close', 'volume'])
    df['date'] = pd.date_range('2024-01-01', periods=len(df), freq='15min')
    return df


def constant_bars(n, base=100.0, rng=1.0):
    return [(base, base + rng, base - rng, base, 100.0) for _ in range(n)]


def peak_bar(o, peak_high, c, low=None):
    """Build a bar with a peak high (for testing pivot detection)."""
    if low is None:
        low = min(o, c) - 0.2
    return (o, peak_high, low, c, 100.0)


def trough_bar(o, trough_low, c, high=None):
    if high is None:
        high = max(o, c) + 0.2
    return (o, high, trough_low, c, 100.0)


# =============================================================================
# Pivot detection
# =============================================================================

def test_pivot_high_at_isolated_peak():
    """A single bar with high above all neighbors → pivot high confirmed."""
    bars = []
    # 5 bars of low high, 1 bar with high peak, 5 more bars of low high
    for _ in range(5):
        bars.append((100.0, 100.5, 99.5, 100.0, 100.0))
    bars.append((100.0, 110.0, 99.5, 100.0, 100.0))  # peak at index 5
    for _ in range(5):
        bars.append((100.0, 100.5, 99.5, 100.0, 100.0))

    high = np.array([b[1] for b in bars], dtype=np.float64)
    low = np.array([b[2] for b in bars], dtype=np.float64)
    pivot_h, pivot_l = _detect_pivots_kernel(high, low, len(bars), fractal_n=5)

    # Pivot at index 5 should be confirmed at index 10 (5 + fractal_n)
    assert pivot_h[10] == 110.0, f"expected 110.0 at index 10, got {pivot_h[10]}"
    # No other pivot highs
    assert np.sum(~np.isnan(pivot_h)) == 1


def test_pivot_low_at_isolated_trough():
    bars = [(100.0, 100.5, 99.5, 100.0, 100.0)] * 5
    bars.append((100.0, 100.5, 90.0, 100.0, 100.0))  # trough at index 5
    bars += [(100.0, 100.5, 99.5, 100.0, 100.0)] * 5

    high = np.array([b[1] for b in bars], dtype=np.float64)
    low = np.array([b[2] for b in bars], dtype=np.float64)
    pivot_h, pivot_l = _detect_pivots_kernel(high, low, len(bars), fractal_n=5)

    assert pivot_l[10] == 90.0, f"expected 90.0 at index 10, got {pivot_l[10]}"
    assert np.sum(~np.isnan(pivot_l)) == 1


def test_no_pivots_in_constant_data():
    n = 30
    bars = constant_bars(n)
    high = np.array([b[1] for b in bars], dtype=np.float64)
    low = np.array([b[2] for b in bars], dtype=np.float64)
    pivot_h, pivot_l = _detect_pivots_kernel(high, low, n, fractal_n=5)

    # All bars have the same high and low. No strict pivots (uses > and <).
    assert np.all(np.isnan(pivot_h))
    assert np.all(np.isnan(pivot_l))


# =============================================================================
# EQH/EQL: detection
# =============================================================================

def test_double_top_detects_eqh():
    """Two pivot highs at the same level → EQH cluster of 2."""
    bars = []
    # Establish baseline
    bars += constant_bars(5)
    # Peak 1 at index 5
    bars.append((100.0, 110.0, 99.5, 100.0, 100.0))
    # 10 baseline bars (peak 1 confirmed at index 10)
    bars += constant_bars(10)
    # Peak 2 at index 16 (same level, 110.0)
    bars.append((100.0, 110.0, 99.5, 100.0, 100.0))
    # 10 baseline bars (peak 2 confirmed at index 21)
    bars += constant_bars(10)

    df = make_df(bars)
    out = annotate_eqh_eql(df, atr_period=14, fractal_n=5,
                            max_pivots=10, tolerance_atr=0.10)

    # At index 21 onwards, EQH should be active at level ~110.0, count >= 2
    assert not np.isnan(out['eqh_level'].iloc[25]), "EQH should be active"
    assert abs(out['eqh_level'].iloc[25] - 110.0) < 1e-6
    assert out['eqh_count'].iloc[25] >= 2


def test_triple_top_eqh_count_3():
    """Three pivot highs at same level → cluster of 3."""
    bars = []
    bars += constant_bars(5)
    for _ in range(3):
        bars.append((100.0, 110.0, 99.5, 100.0, 100.0))  # peak
        bars += constant_bars(10)
    bars += constant_bars(5)

    df = make_df(bars)
    out = annotate_eqh_eql(df, atr_period=14, fractal_n=5,
                            max_pivots=10, tolerance_atr=0.10)

    # By the end, all 3 peaks should have been confirmed and clustered
    final_count = out['eqh_count'].iloc[-1]
    final_level = out['eqh_level'].iloc[-1]
    assert final_count == 3, f"expected 3 in cluster, got {final_count}"
    assert abs(final_level - 110.0) < 1e-6


def test_within_tolerance_clusters():
    """Two highs slightly different but within tol*ATR → still EQH."""
    # ATR will be ~1.0 (range = 1.0 in baseline), tolerance 10% → eps = 0.1
    # Two peaks at 110.00 and 110.05 → diff 0.05 < 0.1 → cluster
    bars = []
    bars += constant_bars(5)
    bars.append((100.0, 110.00, 99.5, 100.0, 100.0))
    bars += constant_bars(10)
    bars.append((100.0, 110.05, 99.5, 100.0, 100.0))
    bars += constant_bars(10)

    df = make_df(bars)
    out = annotate_eqh_eql(df, atr_period=14, fractal_n=5,
                            max_pivots=10, tolerance_atr=0.10)

    assert out['eqh_count'].iloc[-1] >= 2
    # Mean of 110.00 and 110.05 ≈ 110.025
    assert abs(out['eqh_level'].iloc[-1] - 110.025) < 0.01


def test_outside_tolerance_no_cluster():
    """Two highs far enough apart to break tolerance → no EQH."""
    # ATR ≈ 1, tol*ATR = 0.1. Two peaks at 110 and 120 → far → no cluster.
    bars = []
    bars += constant_bars(5)
    bars.append((100.0, 110.0, 99.5, 100.0, 100.0))
    bars += constant_bars(10)
    bars.append((100.0, 120.0, 99.5, 100.0, 100.0))
    bars += constant_bars(10)

    df = make_df(bars)
    out = annotate_eqh_eql(df, atr_period=14, fractal_n=5,
                            max_pivots=10, tolerance_atr=0.10)

    # Each isolated pivot leaves count = 0 (cluster requires 2+)
    assert out['eqh_count'].iloc[-1] == 0
    assert np.isnan(out['eqh_level'].iloc[-1])


def test_double_bottom_detects_eql():
    """Mirror of double top: EQL cluster of 2."""
    bars = []
    bars += constant_bars(5)
    bars.append((100.0, 100.5, 90.0, 100.0, 100.0))  # trough 1
    bars += constant_bars(10)
    bars.append((100.0, 100.5, 90.0, 100.0, 100.0))  # trough 2
    bars += constant_bars(10)

    df = make_df(bars)
    out = annotate_eqh_eql(df, atr_period=14, fractal_n=5,
                            max_pivots=10, tolerance_atr=0.10)

    assert out['eql_count'].iloc[-1] >= 2
    assert abs(out['eql_level'].iloc[-1] - 90.0) < 1e-6


def test_eqh_invalidated_by_close_break():
    """Active EQH should be cleared when close breaks above it."""
    bars = []
    bars += constant_bars(5)
    bars.append((100.0, 110.0, 99.5, 100.0, 100.0))  # peak 1
    bars += constant_bars(10)
    bars.append((100.0, 110.0, 99.5, 100.0, 100.0))  # peak 2
    bars += constant_bars(5)
    # After EQH is active, a bar that closes way above 110 → invalidates
    bars.append((100.0, 115.5, 99.5, 115.0, 200.0))  # close = 115 > 110 + eps
    bars += constant_bars(5)

    df = make_df(bars)
    out = annotate_eqh_eql(df, atr_period=14, fractal_n=5,
                            max_pivots=10, tolerance_atr=0.10)

    # Find the break-bar index (the bar with close=115)
    break_idx = bars.index((100.0, 115.5, 99.5, 115.0, 200.0))
    # Before break: EQH should be active
    assert not np.isnan(out['eqh_level'].iloc[break_idx - 1])
    # At and after break: EQH cleared
    assert np.isnan(out['eqh_level'].iloc[break_idx])
    assert out['eqh_count'].iloc[break_idx] == 0
    assert np.isnan(out['eqh_level'].iloc[-1])


# =============================================================================
# Output schema
# =============================================================================

def test_output_columns_present():
    df = make_df(constant_bars(40))
    out = annotate_eqh_eql(df)
    expected = {
        'atr_14', 'pivot_high_level', 'pivot_low_level',
        'eqh_level', 'eqh_count', 'eql_level', 'eql_count',
    }
    assert expected.issubset(out.columns)


def test_constant_data_no_eqh_eql():
    df = make_df(constant_bars(50))
    out = annotate_eqh_eql(df)
    assert np.all(np.isnan(out['eqh_level']))
    assert np.all(np.isnan(out['eql_level']))
    assert (out['eqh_count'] == 0).all()
    assert (out['eql_count'] == 0).all()


def test_pivot_confirmation_lag():
    """Pivot at bar c is written at bar c + fractal_n, not at c."""
    bars = constant_bars(5)
    bars.append((100.0, 110.0, 99.5, 100.0, 100.0))  # peak at index 5
    bars += constant_bars(10)

    high = np.array([b[1] for b in bars], dtype=np.float64)
    low = np.array([b[2] for b in bars], dtype=np.float64)
    pivot_h, _ = _detect_pivots_kernel(high, low, len(bars), fractal_n=5)

    # Should be NaN at the pivot bar itself (no lookahead)
    assert np.isnan(pivot_h[5])
    # Should be 110.0 at confirmation bar (5 + 5 = 10)
    assert pivot_h[10] == 110.0
