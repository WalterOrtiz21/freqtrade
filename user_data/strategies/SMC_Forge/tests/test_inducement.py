"""
Tests for forge_inducement: IDM detection and sweep events.

Synthetic OHLC fixtures designed so each test isolates one IDM mechanic.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forge_inducement import annotate_inducement  # noqa: E402


# =============================================================================
# Helpers
# =============================================================================

def make_df(bars):
    df = pd.DataFrame(bars, columns=['open', 'high', 'low', 'close', 'volume'])
    df['date'] = pd.date_range('2024-01-01', periods=len(df), freq='15min')
    return df


def flat(n, base=100.0, rng=0.5):
    return [(base, base + rng, base - rng, base, 100.0) for _ in range(n)]


def bar(o, h, l, c):
    return (o, h, l, c, 100.0)


# =============================================================================
# Schema
# =============================================================================

def test_output_columns_present():
    df = make_df(flat(30))
    out = annotate_inducement(df)
    expected = {
        'pivot_high_major', 'pivot_low_major',
        'pivot_high_minor', 'pivot_low_minor',
        'bull_idm_level', 'bull_idm_swept',
        'bear_idm_level', 'bear_idm_swept',
    }
    assert expected.issubset(out.columns)


def test_flat_data_no_idm():
    df = make_df(flat(40))
    out = annotate_inducement(df)
    assert np.all(np.isnan(out['bull_idm_level']))
    assert np.all(np.isnan(out['bear_idm_level']))
    assert (out['bull_idm_swept'] == 0).all()
    assert (out['bear_idm_swept'] == 0).all()


# =============================================================================
# Bullish IDM lifecycle
# =============================================================================

def _bullish_setup_with_sweep():
    """
    Build OHLC where:
      [0..14]   flat warmup at 100
      [15]      MAJOR swing high at 110 (isolated, fractal_n_major=5)
      [16..25]  pullback down with a MINOR sub-low at index 21 (low=95)
      [26..30]  rally up
      [31]      sweep bar: low=94.5 (pierces IDM=95) but close=96 (above)
      [32..40]  continuation up

    Expected:
      bull_idm_level becomes ~95 at bar 23 (minor pivot confirmed at bar 21+2=23)
      bull_idm_swept == 1 at bar 31
      bull_idm_level remains 95 (locked) after sweep
    """
    bars = []
    bars += flat(15)
    # Major peak at index 15
    bars.append(bar(100.0, 110.0, 99.5, 100.0))
    # Pullback + minor sub-low
    bars += flat(5)                                          # 16..20: flat at 100
    bars.append(bar(100.0, 100.5, 95.0, 96.0))               # 21: minor low at 95
    bars += flat(4)                                          # 22..25: flat at 100
    # Rally back up
    rally_close = 100.0
    rally_bars = []
    for i in range(5):
        rally_close = 100 + i * 0.5
        rally_bars.append(bar(rally_close - 0.2, rally_close + 0.3, rally_close - 0.4, rally_close))
    bars += rally_bars                                       # 26..30
    # Sweep bar
    bars.append(bar(102.0, 102.3, 94.5, 96.0))               # 31: wick to 94.5, close 96
    # Continuation
    bars += [bar(96.0 + i * 0.3, 96.0 + i * 0.3 + 0.4, 96.0 + i * 0.3 - 0.2, 96.0 + (i + 1) * 0.3) for i in range(10)]
    return make_df(bars)


def test_bullish_setup_idm_level_appears():
    df = _bullish_setup_with_sweep()
    out = annotate_inducement(df, fractal_n_major=5, fractal_n_minor=2)

    # Major pivot high at index 15 → confirmed at index 20
    assert out['pivot_high_major'].iloc[20] == 110.0
    # Minor pivot low at index 21 → confirmed at index 23
    assert out['pivot_low_minor'].iloc[23] == 95.0

    # bull_idm_level should be 95 from bar 23 onwards (until invalidated)
    assert abs(out['bull_idm_level'].iloc[24] - 95.0) < 1e-6
    assert abs(out['bull_idm_level'].iloc[30] - 95.0) < 1e-6


def test_bullish_idm_swept_event_fires():
    df = _bullish_setup_with_sweep()
    out = annotate_inducement(df, fractal_n_major=5, fractal_n_minor=2)

    # Sweep at index 31
    assert out['bull_idm_swept'].iloc[31] == 1
    # Only one sweep event
    assert out['bull_idm_swept'].sum() == 1


def test_bullish_idm_locked_after_sweep():
    """After sweep, IDM stays at the same level (not updated by later sub-lows)."""
    df = _bullish_setup_with_sweep()
    out = annotate_inducement(df, fractal_n_major=5, fractal_n_minor=2)

    # IDM remains active and locked at 95.0 after sweep
    assert abs(out['bull_idm_level'].iloc[35] - 95.0) < 1e-6


def test_bullish_idm_invalidated_by_close_break():
    """If close drops below IDM (real break, not sweep), IDM is cleared."""
    bars = []
    bars += flat(15)
    bars.append(bar(100.0, 110.0, 99.5, 100.0))               # major high
    bars += flat(5)
    bars.append(bar(100.0, 100.5, 95.0, 96.0))                # minor low
    bars += flat(4)
    # Real break: close below 95
    bars.append(bar(96.0, 96.5, 90.0, 91.0))                  # close = 91 < 95
    bars += flat(5)
    df = make_df(bars)

    out = annotate_inducement(df, fractal_n_major=5, fractal_n_minor=2)

    break_idx = 26
    # IDM was active before break
    assert not np.isnan(out['bull_idm_level'].iloc[break_idx - 1])
    # IDM cleared at and after break
    assert np.isnan(out['bull_idm_level'].iloc[break_idx])
    assert np.isnan(out['bull_idm_level'].iloc[-1])
    # No false sweep event
    assert out['bull_idm_swept'].iloc[break_idx] == 0


def test_bullish_idm_invalidated_by_bos_up():
    """If close exceeds the major high, context shifted up → IDM cleared."""
    bars = []
    bars += flat(15)
    bars.append(bar(100.0, 110.0, 99.5, 100.0))               # major high at 110
    bars += flat(5)
    bars.append(bar(100.0, 100.5, 97.0, 98.0))                # minor low at 97
    bars += flat(4)
    # Strong rally that closes above 110 → BoS up
    bars.append(bar(98.0, 115.5, 97.5, 115.0))                # close = 115 > 110
    bars += flat(5)
    df = make_df(bars)

    out = annotate_inducement(df, fractal_n_major=5, fractal_n_minor=2)

    bos_idx = 26
    # Before BoS, IDM was active
    assert not np.isnan(out['bull_idm_level'].iloc[bos_idx - 1])
    # After BoS, IDM cleared
    assert np.isnan(out['bull_idm_level'].iloc[bos_idx])


# =============================================================================
# Bearish IDM lifecycle (mirror)
# =============================================================================

def _bearish_setup_with_sweep():
    bars = []
    bars += flat(15)
    bars.append(bar(100.0, 100.5, 90.0, 100.0))               # major low at 90
    bars += flat(5)
    bars.append(bar(100.0, 105.0, 99.5, 104.0))               # minor high at 105
    bars += flat(4)
    # Pullback down then sweep up
    rally_close = 100.0
    for i in range(5):
        rally_close = 100 - i * 0.5
        bars.append(bar(rally_close + 0.2, rally_close + 0.4, rally_close - 0.3, rally_close))
    # Sweep: high pierces 105, close stays below
    bars.append(bar(98.0, 105.5, 97.7, 104.0))                # 31: high 105.5, close 104
    # Continuation
    bars += [bar(104.0 - i * 0.3, 104.0 - i * 0.3 + 0.2, 104.0 - i * 0.3 - 0.4, 104.0 - (i + 1) * 0.3) for i in range(10)]
    return make_df(bars)


def test_bearish_setup_idm_level_appears():
    df = _bearish_setup_with_sweep()
    out = annotate_inducement(df, fractal_n_major=5, fractal_n_minor=2)

    # Major low at index 15 → confirmed at index 20
    assert out['pivot_low_major'].iloc[20] == 90.0
    # Minor high at index 21 → confirmed at index 23
    assert out['pivot_high_minor'].iloc[23] == 105.0

    assert abs(out['bear_idm_level'].iloc[24] - 105.0) < 1e-6


def test_bearish_idm_swept_event_fires():
    df = _bearish_setup_with_sweep()
    out = annotate_inducement(df, fractal_n_major=5, fractal_n_minor=2)
    assert out['bear_idm_swept'].iloc[31] == 1
    assert out['bear_idm_swept'].sum() == 1


def test_bearish_idm_invalidated_by_close_break():
    bars = []
    bars += flat(15)
    bars.append(bar(100.0, 100.5, 90.0, 100.0))               # major low
    bars += flat(5)
    bars.append(bar(100.0, 105.0, 99.5, 104.0))               # minor high
    bars += flat(4)
    bars.append(bar(104.0, 110.0, 103.5, 109.0))              # close > 105 → break
    bars += flat(5)
    df = make_df(bars)
    out = annotate_inducement(df, fractal_n_major=5, fractal_n_minor=2)

    break_idx = 26
    assert not np.isnan(out['bear_idm_level'].iloc[break_idx - 1])
    assert np.isnan(out['bear_idm_level'].iloc[break_idx])


# =============================================================================
# Cross-cutting
# =============================================================================

def test_no_idm_without_major_pivot():
    """Minor pivots alone (no major) should not produce IDM."""
    bars = flat(15)
    bars.append(bar(100.0, 100.5, 95.0, 96.0))                # minor low, no major above
    bars += flat(15)
    df = make_df(bars)
    out = annotate_inducement(df, fractal_n_major=5, fractal_n_minor=2)
    assert np.all(np.isnan(out['bull_idm_level']))
    assert (out['bull_idm_swept'] == 0).all()


def test_sweep_requires_close_above_idm():
    """If close goes BELOW the IDM (close-break, not sweep), no sweep event."""
    bars = []
    bars += flat(15)
    bars.append(bar(100.0, 110.0, 99.5, 100.0))               # major high
    bars += flat(5)
    bars.append(bar(100.0, 100.5, 95.0, 96.0))                # minor low at 95
    bars += flat(4)
    # Close below IDM = invalidation, NOT sweep
    bars.append(bar(96.0, 96.5, 93.0, 94.0))                  # close 94 < 95
    bars += flat(5)
    df = make_df(bars)
    out = annotate_inducement(df, fractal_n_major=5, fractal_n_minor=2)
    assert (out['bull_idm_swept'] == 0).all()
