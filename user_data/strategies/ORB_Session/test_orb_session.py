"""Dense unit tests for ORBSession. Each test covers multiple invariants.

Run from /home/wortiz/Desktop/freqtrade:
    pytest user_data/strategies/ORB_Session/test_orb_session.py -v
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE))

from ORBSession import ORBSession  # noqa: E402
from test_helpers import make_15m_day  # noqa: E402


@pytest.fixture
def strat():
    return ORBSession(config={})


def test_populate_indicators_range_and_skip(strat):
    """Covers in one test:
      (a) range_high/low computed from max/min of 13:00-13:45 candles
      (b) range is NaN before 14:00, populated from 14:00 onwards
      (c) range value persists through 20:45 same day
      (d) range_pct > 1.5% -> orb_skipped True
      (e) range_pct <= 1.5% -> orb_skipped False
      (f) state resets cleanly the next UTC day
    """
    # Day 1: wide range (10%), should be skipped
    day1 = make_15m_day(
        '2026-04-15',
        base_price=100.0,
        overrides={
            '13:00': {'high': 105.0, 'low': 99.5, 'close': 100.0},
            '13:15': {'high': 105.5, 'low': 100.0, 'close': 100.2},
            '13:30': {'high': 103.0, 'low': 95.0, 'close': 100.5},
            '13:45': {'high': 102.0, 'low': 100.0, 'close': 100.7},
        },
    )
    # Day 2: narrow range (1%), should NOT be skipped
    day2 = make_15m_day(
        '2026-04-16',
        base_price=200.0,
        overrides={
            '13:00': {'high': 200.5, 'low': 199.5, 'close': 200.0},
            '13:15': {'high': 201.0, 'low': 199.8, 'close': 200.5},
            '13:30': {'high': 200.8, 'low': 199.0, 'close': 200.0},
            '13:45': {'high': 200.5, 'low': 199.5, 'close': 200.0},
        },
    )
    df = pd.concat([day1, day2], ignore_index=True)

    out = strat.populate_indicators(df, {'pair': 'BTC/USDT:USDT'})

    # (a) day1 range_high = max of 4 candle highs = 105.5; range_low = min of lows = 95.0
    day1_at_14 = out.loc[
        (out['date'].dt.date == pd.Timestamp('2026-04-15').date()) &
        (out['date'].dt.hour == 14) & (out['date'].dt.minute == 0)
    ].iloc[0]
    assert day1_at_14['orb_range_high'] == pytest.approx(105.5)
    assert day1_at_14['orb_range_low'] == pytest.approx(95.0)

    # (b) day1 at 13:30 must be NaN (range still forming)
    day1_at_13_30 = out.loc[
        (out['date'].dt.date == pd.Timestamp('2026-04-15').date()) &
        (out['date'].dt.hour == 13) & (out['date'].dt.minute == 30)
    ].iloc[0]
    assert pd.isna(day1_at_13_30['orb_range_high'])
    assert pd.isna(day1_at_13_30['orb_range_low'])

    # (c) day1 at 20:45 must still carry the same range
    day1_at_20_45 = out.loc[
        (out['date'].dt.date == pd.Timestamp('2026-04-15').date()) &
        (out['date'].dt.hour == 20) & (out['date'].dt.minute == 45)
    ].iloc[0]
    assert day1_at_20_45['orb_range_high'] == pytest.approx(105.5)
    assert day1_at_20_45['orb_range_low'] == pytest.approx(95.0)

    # (d) day1 skipped (range = 10.5/100.7 = 10.4% > 1.5%)
    assert bool(day1_at_14['orb_skipped']) is True

    # (e) day2 not skipped (range = 2/200 = 1.0% < 1.5%)
    day2_at_14 = out.loc[
        (out['date'].dt.date == pd.Timestamp('2026-04-16').date()) &
        (out['date'].dt.hour == 14) & (out['date'].dt.minute == 0)
    ].iloc[0]
    assert day2_at_14['orb_range_high'] == pytest.approx(201.0)
    assert day2_at_14['orb_range_low'] == pytest.approx(199.0)
    assert bool(day2_at_14['orb_skipped']) is False

    # (f) day boundary: day 2 at 13:30 must again be NaN, not carrying day 1 state
    day2_at_13_30 = out.loc[
        (out['date'].dt.date == pd.Timestamp('2026-04-16').date()) &
        (out['date'].dt.hour == 13) & (out['date'].dt.minute == 30)
    ].iloc[0]
    assert pd.isna(day2_at_13_30['orb_range_high'])
