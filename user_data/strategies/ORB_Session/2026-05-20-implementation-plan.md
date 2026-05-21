# ORBSession Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `ORBSession` — a freqtrade Opening Range Breakout strategy on the US session (13:00-21:00 UTC) for BTC/USDT:USDT and ETH/USDT:USDT perps, with zero hyperopt parameters in v1, to diversify the LIVE `SMC_Forge` strategy.

**Architecture:** Single-file `IStrategy` subclass (~150-200 LOC) at `user_data/strategies/ORBSession.py`. All session/range/breakout state lives in DataFrame columns (no external state). Range computed in `populate_indicators`, entries gated in `populate_entry_trend`, stop in `custom_stoploss`, forced exit in `populate_exit_trend`. Companion config + tests + analysis script in `user_data/strategies/ORB_Session/`.

**Tech Stack:** Python 3.11, freqtrade v3 (IStrategy INTERFACE_VERSION = 3), pandas, numpy, pytest.

**Reference spec:** `user_data/strategies/ORB_Session/2026-05-20-design.md`

**Testing approach:** Pragmatic TDD per Walter's preference. **3 dense unit tests** cover the only subtle logic (range calc with groupby/ffill, no-reentry suppression, custom_stoploss long/short asymmetry). Everything else is verified end-to-end by the integration backtest (Task 6) with visual chart-check of 3 random trades. No TDD for plumbing or obvious code.

---

## File Structure

**To create:**
- `user_data/strategies/ORBSession.py` — IStrategy class
- `user_data/strategies/ORB_Session/__init__.py` — empty marker
- `user_data/strategies/ORB_Session/test_helpers.py` — synthetic OHLCV builder
- `user_data/strategies/ORB_Session/test_orb_session.py` — 3 dense tests
- `user_data/strategies/ORB_Session/analyze_correlation.py` — diversification check vs SMC_Forge
- `user_data/strategies/ORB_Session/freqtrade-orb.service` — systemd template
- `ORBSession.json` — freqtrade run config (repo root)

**Already exists:**
- `user_data/strategies/ORB_Session/2026-05-20-design.md` — spec (do not modify)

**Commit convention:** one commit per task. Use `git add <specific files>`, never `git add .` (the repo has unrelated SMC_Forge uncommitted changes).

---

## Task 1: Scaffold class + test helpers + config

**Files:**
- Create: `user_data/strategies/ORBSession.py` (full skeleton with all class attrs)
- Create: `user_data/strategies/ORB_Session/__init__.py` (empty)
- Create: `user_data/strategies/ORB_Session/test_helpers.py`

- [ ] **Step 1.1: Create `__init__.py`**

Create empty file at `user_data/strategies/ORB_Session/__init__.py`.

- [ ] **Step 1.2: Create `test_helpers.py`**

Create `user_data/strategies/ORB_Session/test_helpers.py`:

```python
"""Synthetic OHLCV builders for ORBSession unit tests."""

from datetime import datetime, timedelta, timezone

import pandas as pd


def make_15m_day(
    date_str: str,
    base_price: float = 100.0,
    overrides: dict | None = None,
) -> pd.DataFrame:
    """
    Build one UTC day of 96 15m candles (00:00 through 23:45).

    Args:
        date_str: e.g. '2026-04-15'
        base_price: default OHLC = base_price (flat day)
        overrides: dict mapping 'HH:MM' -> dict of {open, high, low, close, volume}

    Returns:
        DataFrame with columns [date, open, high, low, close, volume],
        date is tz-aware UTC.
    """
    start = datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc)
    rows = []
    for i in range(96):
        ts = start + timedelta(minutes=15 * i)
        rows.append({
            'date': ts,
            'open': base_price,
            'high': base_price,
            'low': base_price,
            'close': base_price,
            'volume': 1000.0,
        })
    df = pd.DataFrame(rows)
    if overrides:
        for hhmm, ohlcv in overrides.items():
            hh, mm = map(int, hhmm.split(':'))
            mask = (df['date'].dt.hour == hh) & (df['date'].dt.minute == mm)
            for col, val in ohlcv.items():
                df.loc[mask, col] = val
    return df
```

- [ ] **Step 1.3: Create the full `ORBSession.py` skeleton**

Create `user_data/strategies/ORBSession.py`:

```python
"""
ORBSession — Opening Range Breakout on US Session
==================================================

KISS mechanical strategy to diversify SMC_Forge LIVE.
Session: 13:00-21:00 UTC. Range: 13:00-14:00 UTC (4 15m candles).
Entry: close above/below range. Stop: opposite side of range.
Exit: range stop OR forced close at 21:00 UTC.

No hyperopt parameters in v1 (anti-curve-fit).

See user_data/strategies/ORB_Session/2026-05-20-design.md
"""

import logging

import pandas as pd
from pandas import DataFrame

from freqtrade.strategy import IStrategy


logger = logging.getLogger(__name__)


class ORBSession(IStrategy):
    INTERFACE_VERSION = 3

    # =========================================================
    # Session / range parameters (UTC hours)
    # =========================================================
    SESSION_START_HOUR = 13
    SESSION_END_HOUR = 21
    RANGE_END_HOUR = 14

    # =========================================================
    # Filters (configurable)
    # =========================================================
    MAX_RANGE_PCT = 0.015   # skip session if range/price > 1.5%
    ALLOW_REENTRY = False
    SKIP_WEEKENDS = False

    # =========================================================
    # freqtrade mechanics
    # =========================================================
    timeframe = '15m'
    can_short = True
    use_exit_signal = True
    use_custom_stoploss = True
    process_only_new_candles = True
    startup_candle_count: int = 96   # 24h of 15m

    minimal_roi = {"0": 100}    # disabled; custom_stoploss + exit signal handle exits
    stoploss = -0.99            # placeholder; real stop in custom_stoploss

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Implemented in Task 2.
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Implemented in Task 3.
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Implemented in Task 5.
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0
        return dataframe
```

- [ ] **Step 1.4: Verify imports work**

```bash
cd /home/wortiz/Desktop/freqtrade
python -c "
import sys; sys.path.insert(0, 'user_data/strategies')
from ORBSession import ORBSession
s = ORBSession(config={})
assert s.timeframe == '15m'
assert s.can_short is True
assert s.MAX_RANGE_PCT == 0.015
print('OK')
"
```
Expected: prints `OK`.

- [ ] **Step 1.5: Commit**

```bash
git add user_data/strategies/ORBSession.py \
        user_data/strategies/ORB_Session/__init__.py \
        user_data/strategies/ORB_Session/test_helpers.py
git commit -m "ORBSession: scaffold IStrategy class + test helpers"
```

---

## Task 2: `populate_indicators` (range + skip filter) — TDD

This is the subtlest piece: groupby-cummax/cummin within session_date, then ffill forward from 14:00 only. A bug here silently breaks every downstream check. **One dense test** locks the key behaviors.

**Files:**
- Modify: `user_data/strategies/ORBSession.py`
- Create: `user_data/strategies/ORB_Session/test_orb_session.py`

- [ ] **Step 2.1: Write the failing test**

Create `user_data/strategies/ORB_Session/test_orb_session.py`:

```python
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
```

- [ ] **Step 2.2: Run, confirm fail**

```bash
cd /home/wortiz/Desktop/freqtrade
pytest user_data/strategies/ORB_Session/test_orb_session.py -v
```
Expected: 1 FAIL (KeyError on `orb_range_high`).

- [ ] **Step 2.3: Implement `populate_indicators`**

Replace the stub `populate_indicators` in `ORBSession.py` with:

```python
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe.copy()
        session_date = df['date'].dt.date

        # 1. Mark candles inside the range-forming hour (13:00-13:45 UTC)
        in_range_window = (df['date'].dt.hour == self.SESSION_START_HOUR)

        # 2. cummax/cummin within each UTC day, only over range candles
        range_high_seed = df['high'].where(in_range_window)
        range_low_seed = df['low'].where(in_range_window)
        df['orb_range_high'] = range_high_seed.groupby(session_date).cummax()
        df['orb_range_low'] = range_low_seed.groupby(session_date).cummin()

        # 3. Hide the partial values during 13:00-13:45 (range still forming).
        #    Range is only "valid" from 14:00 onwards.
        post_range = df['date'].dt.hour >= self.RANGE_END_HOUR
        df.loc[~post_range, 'orb_range_high'] = pd.NA
        df.loc[~post_range, 'orb_range_low'] = pd.NA
        df['orb_range_high'] = df.groupby(session_date)['orb_range_high'].ffill()
        df['orb_range_low'] = df.groupby(session_date)['orb_range_low'].ffill()

        # 4. Skip filter: reference price = close of 13:45 candle.
        ref_close = df['close'].where(
            (df['date'].dt.hour == self.SESSION_START_HOUR) &
            (df['date'].dt.minute == 45)
        )
        ref_close_filled = ref_close.groupby(session_date).bfill().ffill()
        range_pct = (df['orb_range_high'] - df['orb_range_low']) / ref_close_filled
        df['orb_range_pct'] = range_pct
        df['orb_skipped'] = (range_pct > self.MAX_RANGE_PCT).fillna(False)

        return df
```

- [ ] **Step 2.4: Run, confirm pass**

```bash
pytest user_data/strategies/ORB_Session/test_orb_session.py -v
```
Expected: 1 PASS.

- [ ] **Step 2.5: Commit**

```bash
git add user_data/strategies/ORBSession.py \
        user_data/strategies/ORB_Session/test_orb_session.py
git commit -m "ORBSession: populate_indicators computes range + skip filter"
```

---

## Task 3: `populate_entry_trend` (long + short + no-reentry) — TDD

The non-obvious bit is the no-reentry gate using `cumsum` over `any_breakout` per session-date — easy to get wrong (off-by-one in cumulative, signal leakage across days). One dense test covers the key invariants.

**Files:**
- Modify: `user_data/strategies/ORBSession.py`
- Modify: `user_data/strategies/ORB_Session/test_orb_session.py`

- [ ] **Step 3.1: Write the failing test**

Append to `test_orb_session.py`:

```python
def test_populate_entry_trend_long_short_and_no_reentry(strat):
    """Covers in one test:
      (a) close > range_high inside session window -> enter_long=1
      (b) close < range_low inside session window -> enter_short=1
      (c) close == range_high (strict inequality) -> no entry
      (d) breakout outside session window (e.g. 22:00) -> no entry
      (e) breakout when session skipped -> no entry
      (f) second breakout same session -> suppressed (no re-entry)
      (g) opposite-side breakout same session -> suppressed
      (h) day boundary: new day -> entry signals work again
    """
    # Day 1: long breakout at 14:15, then re-breakout at 15:00, then short at 16:00.
    # Expect exactly 1 long signal at 14:15, nothing else that day.
    day1 = make_15m_day(
        '2026-04-15',
        overrides={
            '13:00': {'high': 100.5, 'low': 99.5, 'close': 100.0},
            '14:15': {'close': 101.0},   # (a) long
            '15:00': {'close': 102.0},   # (f) second long - suppressed
            '16:00': {'close': 98.0},    # (g) opposite short - suppressed
            '22:00': {'close': 110.0},   # (d) outside session - suppressed
        },
    )
    # Day 2: short breakout at 14:30; long at 14:45 (suppressed - already shorted).
    day2 = make_15m_day(
        '2026-04-16',
        overrides={
            '13:00': {'high': 100.5, 'low': 99.5, 'close': 100.0},
            '14:30': {'close': 99.0},    # (b) short
            '14:45': {'close': 100.5},   # (c) close == range_high -> no entry anyway
            '15:00': {'close': 101.0},   # (f)+(g) suppressed
        },
    )
    # Day 3: wide range (skipped) with breakout.
    day3 = make_15m_day(
        '2026-04-17',
        overrides={
            '13:00': {'high': 105.0, 'low': 95.0, 'close': 100.0},
            '14:15': {'close': 106.0},   # (e) skipped session
        },
    )
    df = pd.concat([day1, day2, day3], ignore_index=True)

    out = strat.populate_indicators(df, {'pair': 'BTC/USDT:USDT'})
    out = strat.populate_entry_trend(out, {'pair': 'BTC/USDT:USDT'})

    def signals_on(date_str):
        mask = out['date'].dt.date == pd.Timestamp(date_str).date()
        return out.loc[mask, ['enter_long', 'enter_short']].sum().to_dict()

    assert signals_on('2026-04-15') == {'enter_long': 1, 'enter_short': 0}
    assert signals_on('2026-04-16') == {'enter_long': 0, 'enter_short': 1}
    assert signals_on('2026-04-17') == {'enter_long': 0, 'enter_short': 0}

    # (h) day boundary verified by day2 having its own signal independent of day1.
    # Specifically check the day2 short was emitted at 14:30.
    day2_14_30 = out.loc[
        (out['date'].dt.date == pd.Timestamp('2026-04-16').date()) &
        (out['date'].dt.hour == 14) & (out['date'].dt.minute == 30)
    ].iloc[0]
    assert day2_14_30['enter_short'] == 1
```

- [ ] **Step 3.2: Run, confirm fail**

```bash
pytest user_data/strategies/ORB_Session/test_orb_session.py -v -k "entry_trend"
```
Expected: FAIL (all signals 0 because stub returns zeros).

- [ ] **Step 3.3: Implement `populate_entry_trend`**

Replace the stub in `ORBSession.py`:

```python
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe
        in_session = (df['date'].dt.hour >= self.RANGE_END_HOUR) & \
                     (df['date'].dt.hour < self.SESSION_END_HOUR)
        not_skipped = ~df['orb_skipped'].fillna(False)
        has_range = df['orb_range_high'].notna() & df['orb_range_low'].notna()

        long_breakout = df['close'] > df['orb_range_high']
        short_breakout = df['close'] < df['orb_range_low']

        any_breakout = (long_breakout | short_breakout) & in_session & not_skipped & has_range

        if not self.ALLOW_REENTRY:
            session_date = df['date'].dt.date
            cum_signals = any_breakout.groupby(session_date).cumsum()
            first_signal_only = any_breakout & (cum_signals == 1)
        else:
            first_signal_only = any_breakout

        df['enter_long'] = (first_signal_only & long_breakout).astype(int)
        df['enter_short'] = (first_signal_only & short_breakout).astype(int)
        return df
```

- [ ] **Step 3.4: Run, confirm pass**

```bash
pytest user_data/strategies/ORB_Session/test_orb_session.py -v
```
Expected: 2 PASS.

- [ ] **Step 3.5: Commit**

```bash
git add user_data/strategies/ORBSession.py \
        user_data/strategies/ORB_Session/test_orb_session.py
git commit -m "ORBSession: populate_entry_trend with long/short + no-reentry"
```

---

## Task 4: `custom_stoploss` (long + short asymmetry) — TDD

freqtrade's `custom_stoploss` return convention differs subtly for shorts. **One test** with both legs locks the formula.

**Files:**
- Modify: `user_data/strategies/ORBSession.py`
- Modify: `user_data/strategies/ORB_Session/test_orb_session.py`

- [ ] **Step 4.1: Write the failing test**

Append to `test_orb_session.py`:

```python
from datetime import datetime, timezone
from unittest.mock import MagicMock


def _fake_trade(pair: str, is_short: bool, open_rate: float):
    t = MagicMock()
    t.pair = pair
    t.is_short = is_short
    t.open_rate = open_rate
    t.open_date_utc = datetime(2026, 4, 15, 14, 15, tzinfo=timezone.utc)
    return t


def test_custom_stoploss_long_and_short(strat):
    """Covers in one test:
      (a) Long: open_rate=101.0, range_low=99.5 -> sl = (99.5/101)-1 ≈ -0.01485
      (b) Short: open_rate=99.0, range_high=100.5 -> sl = (99-100.5)/99 ≈ -0.01515
      (c) If dataframe has no range data (defensive): falls back to class stoploss
    """
    df = make_15m_day(
        '2026-04-15',
        overrides={
            '13:00': {'high': 100.5, 'low': 99.5, 'close': 100.0},
            '14:15': {'close': 101.0},
        },
    )
    df_with_range = strat.populate_indicators(df, {'pair': 'BTC/USDT:USDT'})

    strat.dp = MagicMock()
    strat.dp.get_analyzed_dataframe = MagicMock(return_value=(df_with_range, None))

    # (a) Long
    trade_long = _fake_trade('BTC/USDT:USDT', is_short=False, open_rate=101.0)
    sl_long = strat.custom_stoploss(
        pair='BTC/USDT:USDT', trade=trade_long,
        current_time=datetime(2026, 4, 15, 14, 30, tzinfo=timezone.utc),
        current_rate=101.0, current_profit=0.0,
    )
    assert sl_long == pytest.approx((99.5 / 101.0) - 1, abs=1e-5)

    # (b) Short
    trade_short = _fake_trade('ETH/USDT:USDT', is_short=True, open_rate=99.0)
    sl_short = strat.custom_stoploss(
        pair='ETH/USDT:USDT', trade=trade_short,
        current_time=datetime(2026, 4, 15, 14, 30, tzinfo=timezone.utc),
        current_rate=99.0, current_profit=0.0,
    )
    assert sl_short == pytest.approx((99.0 - 100.5) / 99.0, abs=1e-5)

    # (c) Defensive fallback: pretend dataframe is empty
    strat.dp.get_analyzed_dataframe = MagicMock(return_value=(pd.DataFrame(), None))
    sl_fallback = strat.custom_stoploss(
        pair='BTC/USDT:USDT', trade=trade_long,
        current_time=datetime(2026, 4, 15, 14, 30, tzinfo=timezone.utc),
        current_rate=101.0, current_profit=0.0,
    )
    assert sl_fallback == strat.stoploss
```

- [ ] **Step 4.2: Run, confirm fail**

```bash
pytest user_data/strategies/ORB_Session/test_orb_session.py -v -k "custom_stoploss"
```
Expected: FAIL (`custom_stoploss` not defined).

- [ ] **Step 4.3: Implement `custom_stoploss`**

Append this method to `ORBSession` class:

```python
    def custom_stoploss(self, pair, trade, current_time, current_rate, current_profit, **kwargs):
        """Stop at opposite end of opening range.

        Long: stop = range_low  -> returns (range_low / open_rate) - 1   (negative)
        Short: stop = range_high -> returns (open_rate - range_high) / open_rate  (negative)

        If range data is missing (defensive), returns the class-level stoploss
        (-0.99 placeholder; should never trigger in practice).
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        if dataframe is None or dataframe.empty:
            return self.stoploss

        trade_date = trade.open_date_utc.date()
        same_day = dataframe['date'].dt.date == trade_date
        candidates = dataframe.loc[same_day & dataframe['orb_range_high'].notna()]
        if candidates.empty:
            return self.stoploss

        last = candidates.iloc[-1]
        range_high = last['orb_range_high']
        range_low = last['orb_range_low']

        if trade.is_short:
            return (trade.open_rate - range_high) / trade.open_rate
        return (range_low / trade.open_rate) - 1
```

- [ ] **Step 4.4: Run, confirm pass**

```bash
pytest user_data/strategies/ORB_Session/test_orb_session.py -v
```
Expected: 3 PASS.

- [ ] **Step 4.5: Commit**

```bash
git add user_data/strategies/ORBSession.py \
        user_data/strategies/ORB_Session/test_orb_session.py
git commit -m "ORBSession: custom_stoploss returns opposite-of-range (long+short)"
```

---

## Task 5: `populate_exit_trend` (forced exit 21:00) — no TDD

Trivial one-liner; verified visually in the integration backtest (Task 6).

**Files:**
- Modify: `user_data/strategies/ORBSession.py`

- [ ] **Step 5.1: Implement forced exit**

Replace the stub `populate_exit_trend` in `ORBSession.py`:

```python
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe
        # The 20:45 15m candle closes at 21:00 UTC -> emit exit signal there.
        is_last_session_candle = (
            (df['date'].dt.hour == (self.SESSION_END_HOUR - 1)) &
            (df['date'].dt.minute == 45)
        )
        df['exit_long'] = is_last_session_candle.astype(int)
        df['exit_short'] = is_last_session_candle.astype(int)
        return df
```

- [ ] **Step 5.2: Quick smoke check (no test file)**

```bash
cd /home/wortiz/Desktop/freqtrade
python -c "
import sys; sys.path.insert(0, 'user_data/strategies')
sys.path.insert(0, 'user_data/strategies/ORBSession')
from ORBSession import ORBSession
from test_helpers import make_15m_day
df = make_15m_day('2026-04-15', overrides={'13:00': {'high': 100.5, 'low': 99.5, 'close': 100.0}})
s = ORBSession(config={})
df = s.populate_indicators(df, {'pair': 'BTC/USDT:USDT'})
df = s.populate_exit_trend(df, {'pair': 'BTC/USDT:USDT'})
row = df.loc[(df['date'].dt.hour == 20) & (df['date'].dt.minute == 45)].iloc[0]
assert row['exit_long'] == 1 and row['exit_short'] == 1, row
print('OK')
"
```
Expected: prints `OK`.

- [ ] **Step 5.3: Commit**

```bash
git add user_data/strategies/ORBSession.py
git commit -m "ORBSession: forced exit at 20:45 (closes at 21:00 UTC)"
```

---

## Task 6: Freqtrade run config + integration backtest

This is the **end-to-end verification** that replaces additional unit tests. After this task, anything broken in the strategy will surface as wrong trade count / wrong entry timing / wrong stop placement against real chart data.

**Files:**
- Create: `ORBSession.json` (repo root)

- [ ] **Step 6.1: Create `ORBSession.json`**

Create `/home/wortiz/Desktop/freqtrade/ORBSession.json`:

```json
{
  "max_open_trades": 2,
  "stake_currency": "USDT",
  "stake_amount": 200,
  "tradable_balance_ratio": 0.99,
  "fiat_display_currency": "USD",
  "dry_run": true,
  "dry_run_wallet": 10000,
  "cancel_open_orders_on_exit": false,
  "trading_mode": "futures",
  "margin_mode": "isolated",
  "unfilledtimeout": {
    "entry": 10,
    "exit": 30,
    "exit_timeout_count": 0,
    "unit": "minutes"
  },
  "entry_pricing": {
    "price_side": "same",
    "use_order_book": true,
    "order_book_top": 1,
    "price_last_balance": 0.0,
    "check_depth_of_market": {
      "enabled": false,
      "bids_to_ask_delta": 1
    }
  },
  "exit_pricing": {
    "price_side": "same",
    "use_order_book": true,
    "order_book_top": 1
  },
  "exchange": {
    "name": "binance",
    "key": "",
    "secret": "",
    "ccxt_config": {},
    "ccxt_async_config": {},
    "pair_whitelist": [
      "BTC/USDT:USDT",
      "ETH/USDT:USDT"
    ],
    "pair_blacklist": []
  },
  "pairlists": [
    {"method": "StaticPairList"}
  ],
  "telegram": {
    "enabled": false,
    "token": "",
    "chat_id": ""
  },
  "api_server": {
    "enabled": true,
    "listen_ip_address": "127.0.0.1",
    "listen_port": 8082,
    "verbosity": "error",
    "enable_openapi": false,
    "jwt_secret_key": "REPLACE_WITH_RANDOM_STRING_BEFORE_LIVE",
    "CORS_origins": [],
    "username": "freqtrader",
    "password": "REPLACE_BEFORE_LIVE"
  },
  "bot_name": "ORBSession",
  "initial_state": "running",
  "force_entry_enable": false,
  "internals": {
    "process_throttle_secs": 5
  },
  "strategy": "ORBSession",
  "strategy_path": "user_data/strategies/",
  "db_url": "sqlite:///user_data/tradesv3_orb.dryrun.sqlite"
}
```

Notes: separate DB (`tradesv3_orb.dryrun.sqlite`), separate API port (8082), placeholder JWT/password (must be replaced before live).

- [ ] **Step 6.2: Validate config JSON**

```bash
cd /home/wortiz/Desktop/freqtrade
python -c "import json; json.load(open('ORBSession.json'))"
freqtrade show-config --config ORBSession.json 2>&1 | tail -10
```
Expected: no error; second command prints active strategy `ORBSession`.

- [ ] **Step 6.3: Download April 2026 data**

```bash
freqtrade download-data \
    --exchange binance \
    --trading-mode futures \
    --pairs BTC/USDT:USDT ETH/USDT:USDT \
    --timeframes 15m \
    --timerange 20260401-20260501
```
Expected: data files appear in `user_data/data/binance/futures/`.

- [ ] **Step 6.4: Run backtest**

```bash
freqtrade backtesting \
    --config ORBSession.json \
    --strategy ORBSession \
    --timerange 20260401-20260501 \
    --timeframe 15m \
    --export trades \
    --export-filename user_data/backtest_results/orb_april2026
```
Expected: completes without exceptions, prints trade summary.

- [ ] **Step 6.5: Sanity-check trade counts and durations**

```bash
python -c "
import json, statistics
data = json.load(open('user_data/backtest_results/orb_april2026.json'))
trades = data['strategy']['ORBSession']['trades']
print(f'Total trades: {len(trades)}')
by_pair = {}
for t in trades:
    by_pair[t['pair']] = by_pair.get(t['pair'], 0) + 1
print(f'By pair: {by_pair}')
if trades:
    dur_h = [t['trade_duration'] / 60 for t in trades]
    print(f'Duration h: median={statistics.median(dur_h):.2f}, max={max(dur_h):.2f}')
"
```

Expected ranges:
- Total trades: 20-65 (max 60 = 30 days × 2 pairs; some skipped sessions reduce this).
- Max duration: ≤ 7h (entry no earlier than 14:00, forced exit at 21:00).
- Both BTC and ETH should have non-zero counts.

If out of range, **stop and debug**. Likely culprits: timezone confusion in `populate_indicators`, wrong groupby key, off-by-one in session window.

- [ ] **Step 6.6: Visual verification of 3 trades**

Pick 3 trades at random from the backtest output. For each, **open the BTC/USDT or ETH/USDT 15m chart on TradingView** at the trade's timestamp and verify:

1. Entry candle is the first 15m close strictly above/below the 13:00-14:00 UTC range.
2. The stop level matches the opposite end of that range.
3. The trade closed by either touching the stop or at the 20:45 candle (closes 21:00).

If any of the 3 trades doesn't match: **do not proceed to Task 7**. Fix the strategy code and rerun the backtest.

- [ ] **Step 6.7: Commit**

```bash
git add ORBSession.json
git status --short user_data/backtest_results/ | head -5
# If backtest results are tracked (not in .gitignore):
git add user_data/backtest_results/orb_april2026.json \
        user_data/backtest_results/orb_april2026.meta.json 2>/dev/null || true
git commit -m "ORBSession: run config + April 2026 backtest baseline"
```

---

## Task 7: Correlation analysis script (vs SMC_Forge)

The diversification thesis (spec §9) requires `|corr| < 0.4`. The script computes daily-return Pearson correlation between ORBSession and SMC_Forge backtest equity curves.

**Files:**
- Create: `user_data/strategies/ORB_Session/analyze_correlation.py`

- [ ] **Step 7.1: Create the script**

Create `user_data/strategies/ORB_Session/analyze_correlation.py`:

```python
"""
Diversification check: correlate ORBSession equity curve vs SMC_Forge equity curve.

Usage:
    python user_data/strategies/ORB_Session/analyze_correlation.py \\
        --orb user_data/backtest_results/orb_april2026.json \\
        --smc user_data/backtest_results/smc_april2026.json

Verdict (spec §9):
    |corr| < 0.4         -> GREEN: diversification holds
    0.4 <= |corr| < 0.6  -> YELLOW: weak diversification
    |corr| >= 0.6        -> RED: diversification thesis fails
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def equity_curve_from_backtest(json_path: Path, strategy_name: str) -> pd.Series:
    """Daily equity curve from a freqtrade backtest result JSON."""
    data = json.loads(json_path.read_text())
    trades = data['strategy'][strategy_name]['trades']
    if not trades:
        raise SystemExit(f'no trades in {json_path}')

    df = pd.DataFrame(trades)
    df['close_date'] = pd.to_datetime(df['close_date'])
    df = df.sort_values('close_date')
    df['date'] = df['close_date'].dt.tz_localize(None).dt.normalize()
    return df.groupby('date')['profit_abs'].sum().cumsum()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--orb', required=True, type=Path)
    ap.add_argument('--smc', required=True, type=Path)
    ap.add_argument('--orb-strategy-name', default='ORBSession')
    ap.add_argument('--smc-strategy-name', default='SMCForge')
    args = ap.parse_args()

    orb = equity_curve_from_backtest(args.orb, args.orb_strategy_name)
    smc = equity_curve_from_backtest(args.smc, args.smc_strategy_name)

    aligned = pd.DataFrame({'orb': orb, 'smc': smc}).fillna(method='ffill').dropna()
    returns = aligned.diff().dropna()

    if len(returns) < 5:
        raise SystemExit(f'not enough overlapping days ({len(returns)})')

    corr = returns['orb'].corr(returns['smc'])
    abs_corr = abs(corr)

    print(f'Overlapping days: {len(returns)}')
    print(f'Pearson correlation (daily returns): {corr:+.3f}')
    print(f'|corr| = {abs_corr:.3f}')
    if abs_corr < 0.4:
        verdict = 'GREEN -- diversification holds (spec §9 target)'
    elif abs_corr < 0.6:
        verdict = 'YELLOW -- weak diversification, evaluate operational complexity'
    else:
        verdict = 'RED -- diversification thesis fails (spec §9)'
    print(f'Verdict: {verdict}')


if __name__ == '__main__':
    main()
```

- [ ] **Step 7.2: Smoke-test the script**

Ensure an SMC_Forge backtest covering the same range exists. If not, regenerate:

```bash
cd /home/wortiz/Desktop/freqtrade
ls user_data/backtest_results/ | grep -i smc | head -3
# If nothing covers April 2026:
freqtrade backtesting --config config_smc_forge.json --strategy SMCForge \
    --timerange 20260401-20260501 \
    --export trades --export-filename user_data/backtest_results/smc_april2026
```

Then:

```bash
python user_data/strategies/ORB_Session/analyze_correlation.py \
    --orb user_data/backtest_results/orb_april2026.json \
    --smc user_data/backtest_results/smc_april2026.json
```
Expected: prints overlapping days, correlation, verdict. 1-month is too short for a real verdict — this only confirms the script runs end-to-end.

- [ ] **Step 7.3: Commit**

```bash
git add user_data/strategies/ORB_Session/analyze_correlation.py
git commit -m "ORBSession: correlation analysis script vs SMC_Forge"
```

---

## Task 8: systemd unit template

**Files:**
- Create: `user_data/strategies/ORB_Session/freqtrade-orb.service`

- [ ] **Step 8.1: Create the unit file template**

Create `user_data/strategies/ORB_Session/freqtrade-orb.service`:

```ini
[Unit]
Description=Freqtrade ORBSession bot (paper / live)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=wortiz
Group=wortiz
WorkingDirectory=/home/wortiz/Desktop/freqtrade
Environment=PATH=/home/wortiz/Desktop/freqtrade/.venv/bin:/usr/bin:/bin
ExecStart=/home/wortiz/Desktop/freqtrade/.venv/bin/freqtrade trade \
    --config /home/wortiz/Desktop/freqtrade/ORBSession.json \
    --strategy ORBSession \
    --logfile /home/wortiz/Desktop/freqtrade/user_data/logs/orb.log
Restart=on-failure
RestartSec=15

NoNewPrivileges=true
ProtectSystem=full
ProtectHome=false

[Install]
WantedBy=multi-user.target
```

**Do not auto-deploy.** Walter installs it manually under sudo:

```bash
sudo cp user_data/strategies/ORB_Session/freqtrade-orb.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now freqtrade-orb
sudo systemctl status freqtrade-orb
```

Verify the venv path (`.venv/bin/freqtrade`) exists before enabling. Adjust unit file if freqtrade lives elsewhere.

- [ ] **Step 8.2: Commit**

```bash
git add user_data/strategies/ORB_Session/freqtrade-orb.service
git commit -m "ORBSession: systemd unit template (manual deploy)"
```

---

## Final verification

- [ ] **Step F.1: Run the test suite**

```bash
cd /home/wortiz/Desktop/freqtrade
pytest user_data/strategies/ORB_Session/test_orb_session.py -v
```
Expected: 3 PASS.

- [ ] **Step F.2: Re-run backtest end-to-end**

```bash
freqtrade backtesting --config ORBSession.json --timerange 20260401-20260501 --timeframe 15m 2>&1 | tail -10
```
Expected: backtest summary table.

- [ ] **Step F.3: Verify only ORBSession files were modified**

```bash
git log --oneline test_womx ^test_womx~10 2>/dev/null || git log --oneline -10
```
Confirm all commits are ORBSession-scoped; no unrelated SMC_Forge edits crept in.

---

## Out of scope (deferred to subsequent plans)

- **Hyperopt** — none in v1 (spec §10).
- **`bt-forensics` Gate 2** — separate plan, walk-forward + Monte Carlo over full 3-year window.
- **Paper Gate 3** — 30 days dry-run live feed, separate plan.
- **Live Gate 4** — sizing + monitoring procedures, separate plan.
- **Multi-session (Asia/EU) v2** — only if v1 produces measurable edge.
- **Additional pairs (SOL, etc.)** — v2 expansion.
- **Funding-rate aware filter** — spec §11.1, evaluate post-Gate 3.

---

## Self-review checklist

1. **Spec coverage:**
   - §3 Mechanics (range, entry, stop, exit, sizing) → Tasks 2, 3, 4, 5, 6. ✓
   - §4 Configuration → Tasks 1 (attrs) + 6 (freqtrade config). ✓
   - §5 Flow → covered structurally by Tasks 2-5. ✓
   - §6 Validation Gate 1 partial → Task 6 (1-month baseline). Full 3-year backtest deferred. ✓
   - §7 Risk / kill criteria → procedural, Walter-managed, no task. ✓
   - §8 Testing → 3 dense unit tests (Tasks 2-4) + visual integration (Task 6). ✓
   - §9 Diversification → Task 7. ✓
   - §10 Decisions, §11 Open questions → spec-only, no code. ✓

2. **Placeholder scan:** no "TBD", "TODO", or unimplemented references found. ✓

3. **Type consistency:**
   - `orb_range_high`, `orb_range_low`, `orb_range_pct`, `orb_skipped` — consistent across Tasks 2, 3, 4. ✓
   - Class attrs (`MAX_RANGE_PCT`, `ALLOW_REENTRY`, `SESSION_START_HOUR`, `RANGE_END_HOUR`, `SESSION_END_HOUR`) defined in Task 1, used in Tasks 2, 3, 5. ✓
   - `custom_stoploss` signature matches freqtrade v3. ✓
