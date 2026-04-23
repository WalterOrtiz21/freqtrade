"""
Tests for forge_quality.annotate_displacement and the underlying numba kernels.

Synthetic OHLC fixtures where we know exactly what the engine should detect.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Allow imports from SMC_Forge/ root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forge_engine import SMCEngine  # noqa: E402
from forge_quality import (  # noqa: E402
    annotate_displacement,
    _atr_kernel,
    _displacement_kernel,
)


# =============================================================================
# Helpers
# =============================================================================

def make_df(bars):
    """bars: list of (open, high, low, close, volume)."""
    df = pd.DataFrame(bars, columns=['open', 'high', 'low', 'close', 'volume'])
    df['date'] = pd.date_range('2024-01-01', periods=len(df), freq='15min')
    return df


def constant_bars(n, base_price=100.0, range_size=1.0):
    """n bars with stable range = stable ATR."""
    bars = []
    for _ in range(n):
        o = base_price
        h = base_price + range_size
        l = base_price - range_size
        c = base_price + 0.05  # tiny bullish body
        bars.append((o, h, l, c, 100.0))
    return bars


def downtrend_then_explosive_bull():
    """
    Construct OHLC with clearly defined swing structure:
      Bars 0-14  : flat warmup at 100 (ATR seed).
      Bars 15-24 : leg down 100 → 91 (forms internal pivot high at 100).
      Bars 25-29 : bounce 91 → 96 (forms internal pivot low at ~91).
      Bars 30-39 : leg down 96 → 81 (BREAKS below pivot low → BoS bear,
                   curr_int_trend becomes -1; new pivot high at ~96).
      Bar 40     : EXPLOSIVE bullish bar (o=81, c=105) — closes above the
                   most recent pivot high (~96) → fires CHoCH bullish
                   (because curr_int_trend was -1).
      Bars 41-60 : follow-through up.

    With internal_length=5, swing_length=10 the engine resolves pivots
    cleanly. Returns the DataFrame.
    """
    bars = []

    # Phase 0 (0-14): flat warmup
    for _ in range(15):
        bars.append((100.0, 100.5, 99.5, 100.0, 100.0))

    # Phase 1 (15-24): bear leg 100 → 91
    for i in range(10):
        c = 100.0 - i * 1.0
        o = c + 0.6
        h = o + 0.2
        l = c - 0.4
        bars.append((o, h, l, c, 100.0))

    # Phase 2 (25-29): bounce 91 → 96
    for i in range(5):
        c = 91.0 + (i + 1) * 1.0
        o = c - 0.6
        h = c + 0.2
        l = o - 0.2
        bars.append((o, h, l, c, 100.0))

    # Phase 3 (30-39): bear leg 96 → 81 (breaks pivot low)
    for i in range(10):
        c = 95.0 - i * 1.5
        o = c + 0.8
        h = o + 0.2
        l = c - 0.4
        bars.append((o, h, l, c, 100.0))

    # Phase 4 (40): explosive bull break
    bars.append((81.0, 105.5, 80.7, 105.0, 500.0))

    # Phase 5 (41-60): follow-through
    base = 105.0
    for i in range(20):
        c = base + (i + 1) * 0.5
        o = c - 0.3
        h = c + 0.2
        l = o - 0.2
        bars.append((o, h, l, c, 100.0))

    return make_df(bars)


def uptrend_then_explosive_bear():
    """Mirror of downtrend_then_explosive_bull."""
    bars = []

    for _ in range(15):
        bars.append((100.0, 100.5, 99.5, 100.0, 100.0))

    # bull leg 100 → 109
    for i in range(10):
        c = 100.0 + i * 1.0
        o = c - 0.6
        h = c + 0.4
        l = o - 0.2
        bars.append((o, h, l, c, 100.0))

    # pullback 109 → 104
    for i in range(5):
        c = 109.0 - (i + 1) * 1.0
        o = c + 0.6
        h = o + 0.2
        l = c - 0.2
        bars.append((o, h, l, c, 100.0))

    # bull leg 104 → 119 (breaks pivot high)
    for i in range(10):
        c = 105.0 + i * 1.5
        o = c - 0.8
        h = c + 0.4
        l = o - 0.2
        bars.append((o, h, l, c, 100.0))

    # explosive bear break
    bars.append((119.0, 119.3, 94.5, 95.0, 500.0))

    base = 95.0
    for i in range(20):
        c = base - (i + 1) * 0.5
        o = c + 0.3
        h = o + 0.2
        l = c - 0.2
        bars.append((o, h, l, c, 100.0))

    return make_df(bars)


# =============================================================================
# ATR kernel
# =============================================================================

def test_atr_basic_constant_range():
    n = 30
    df = make_df(constant_bars(n, base_price=100.0, range_size=2.0))
    atr = _atr_kernel(
        df['high'].values, df['low'].values, df['close'].values, n, 14
    )
    # Range = 4 (high - low). With tiny body, TR = max(4, |gap|) ≈ 4 most bars.
    # Some bars TR = 4 + tiny_body_offset between consecutive closes.
    # ATR after warmup should be very close to 4.
    assert abs(atr[14] - 4.0) < 0.1
    assert abs(atr[29] - 4.0) < 0.1


def test_atr_warmup_is_nan():
    n = 20
    df = make_df(constant_bars(n))
    atr = _atr_kernel(
        df['high'].values, df['low'].values, df['close'].values, n, 14
    )
    assert np.isnan(atr[0])
    assert np.isnan(atr[13])
    assert not np.isnan(atr[14])


def test_atr_short_data_all_nan():
    """If data length <= period, ATR is all NaN."""
    n = 10
    df = make_df(constant_bars(n))
    atr = _atr_kernel(
        df['high'].values, df['low'].values, df['close'].values, n, 14
    )
    assert np.all(np.isnan(atr))


# =============================================================================
# annotate_displacement: shape and direction
# =============================================================================

def test_displacement_columns_present():
    df = make_df(constant_bars(30))
    eng = SMCEngine(df, internal_length=5, swing_length=20)
    signals = eng.get_signals()
    annotated = annotate_displacement(df, signals, atr_period=14, lookback=3)

    expected = [
        'atr_14',
        'internal_bos_bull_disp', 'internal_bos_bear_disp',
        'internal_choch_bull_disp', 'internal_choch_bear_disp',
        'swing_bos_bull_disp', 'swing_bos_bear_disp',
        'swing_choch_bull_disp', 'swing_choch_bear_disp',
    ]
    for col in expected:
        assert col in annotated.columns, f"missing column: {col}"


def test_no_event_no_score():
    """Constant-range data: no structural breaks → all disp scores are 0."""
    df = make_df(constant_bars(60))
    eng = SMCEngine(df, internal_length=5, swing_length=20)
    signals = eng.get_signals()
    annotated = annotate_displacement(df, signals)

    for col in [
        'internal_bos_bull_disp', 'internal_bos_bear_disp',
        'internal_choch_bull_disp', 'internal_choch_bear_disp',
        'swing_bos_bull_disp', 'swing_bos_bear_disp',
        'swing_choch_bull_disp', 'swing_choch_bear_disp',
    ]:
        assert (annotated[col] == 0).all(), f"{col} should be all zeros"


def test_strong_bullish_event_scores_high():
    """Explosive bull breakout → at least one bullish event with score >= 1.0."""
    df = downtrend_then_explosive_bull()
    eng = SMCEngine(df, internal_length=5, swing_length=10)
    signals = eng.get_signals()
    annotated = annotate_displacement(df, signals, atr_period=14, lookback=3)

    bull_disp_cols = [
        'internal_bos_bull_disp', 'internal_choch_bull_disp',
        'swing_bos_bull_disp', 'swing_choch_bull_disp',
    ]

    max_score = max(annotated[c].max() for c in bull_disp_cols)
    assert max_score >= 1.0, (
        f"expected strong displacement (>= 1.0) from explosive bar, "
        f"got max score {max_score:.3f}"
    )


def test_strong_bearish_event_scores_high():
    df = uptrend_then_explosive_bear()
    eng = SMCEngine(df, internal_length=5, swing_length=10)
    signals = eng.get_signals()
    annotated = annotate_displacement(df, signals, atr_period=14, lookback=3)

    bear_disp_cols = [
        'internal_bos_bear_disp', 'internal_choch_bear_disp',
        'swing_bos_bear_disp', 'swing_choch_bear_disp',
    ]
    max_score = max(annotated[c].max() for c in bear_disp_cols)
    assert max_score >= 1.0, (
        f"expected strong bearish displacement (>= 1.0), got {max_score:.3f}"
    )


def test_bullish_event_does_not_set_bearish_disp():
    """A bar that fires a bullish event must not write bearish disp values."""
    df = downtrend_then_explosive_bull()
    eng = SMCEngine(df, internal_length=5, swing_length=10)
    signals = eng.get_signals()
    annotated = annotate_displacement(df, signals)

    bull_event_mask = (
        (signals['internal_bos_bullish'] == 1) |
        (signals['internal_choch_bullish'] == 1) |
        (signals['swing_bos_bullish'] == 1) |
        (signals['swing_choch_bullish'] == 1)
    )
    assert bull_event_mask.sum() > 0, "fixture should produce bullish events"

    # On bullish-event bars, no bearish-event was simultaneously fired by setup,
    # so bearish disp must be 0 on those bars.
    for idx in np.where(bull_event_mask.values)[0]:
        row = annotated.iloc[idx]
        for col in [
            'internal_bos_bear_disp', 'internal_choch_bear_disp',
            'swing_bos_bear_disp', 'swing_choch_bear_disp',
        ]:
            assert row[col] == 0, (
                f"bull event at bar {idx} unexpectedly set {col}={row[col]}"
            )


def test_disp_score_zero_when_no_event_at_bar():
    """For ANY bar without a structural event, all 8 disp columns must be 0."""
    df = downtrend_then_explosive_bull()
    eng = SMCEngine(df, internal_length=5, swing_length=10)
    signals = eng.get_signals()
    annotated = annotate_displacement(df, signals)

    no_event_mask = (
        (signals['internal_bos_bullish'] == 0) &
        (signals['internal_bos_bearish'] == 0) &
        (signals['internal_choch_bullish'] == 0) &
        (signals['internal_choch_bearish'] == 0) &
        (signals['swing_bos_bullish'] == 0) &
        (signals['swing_bos_bearish'] == 0) &
        (signals['swing_choch_bullish'] == 0) &
        (signals['swing_choch_bearish'] == 0)
    )
    no_event_idxs = np.where(no_event_mask.values)[0]
    assert len(no_event_idxs) > 0

    for col in [
        'internal_bos_bull_disp', 'internal_bos_bear_disp',
        'internal_choch_bull_disp', 'internal_choch_bear_disp',
        'swing_bos_bull_disp', 'swing_bos_bear_disp',
        'swing_choch_bull_disp', 'swing_choch_bear_disp',
    ]:
        assert (annotated.loc[no_event_idxs, col] == 0).all(), (
            f"{col} non-zero on bars without that event"
        )


# =============================================================================
# Validation
# =============================================================================

def test_mismatched_lengths_raise():
    df = make_df(constant_bars(20))
    eng = SMCEngine(df, internal_length=5, swing_length=10)
    signals = eng.get_signals()
    truncated = signals.iloc[:10].copy()
    with pytest.raises(ValueError):
        annotate_displacement(df, truncated)
