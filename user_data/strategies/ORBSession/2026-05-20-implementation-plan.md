# ORBSession Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `ORBSession` — a freqtrade Opening Range Breakout strategy on the US session (13:00-21:00 UTC) for BTC/USDT:USDT and ETH/USDT:USDT perps, with zero hyperopt parameters in v1, designed to diversify the LIVE `SMC_Forge` strategy.

**Architecture:** Single-file `IStrategy` subclass (~150-200 LOC) at `user_data/strategies/ORBSession.py`. All session/range/breakout state lives in DataFrame columns (no external state). Range computed in `populate_indicators`, entries gated in `populate_entry_trend`, stop in `custom_stoploss`, forced exit in `populate_exit_trend`. Companion tests + config + analysis script live in `user_data/strategies/ORBSession/`.

**Tech Stack:** Python 3.11, freqtrade v3 (IStrategy INTERFACE_VERSION = 3), pandas, numpy, pytest. Exchange: Binance USDM perpetuals.

**Reference spec:** `user_data/strategies/ORBSession/2026-05-20-design.md`

---

## File Structure

**To create:**
- `user_data/strategies/ORBSession.py` — IStrategy class
- `user_data/strategies/ORBSession/__init__.py` — empty marker (allows pytest discovery)
- `user_data/strategies/ORBSession/test_orb_session.py` — unit tests
- `user_data/strategies/ORBSession/test_helpers.py` — synthetic OHLCV builders
- `user_data/strategies/ORBSession/analyze_correlation.py` — post-backtest diversification check vs SMC_Forge
- `ORBSession.json` — freqtrade run config (repo root, like `config_smc_forge.json`)
- `user_data/strategies/ORBSession/freqtrade-orb.service` — systemd unit (template; deploy with sudo)

**Already exists:**
- `user_data/strategies/ORBSession/2026-05-20-design.md` — spec (do not modify)

---

## Task Decomposition Notes

- Tasks 1-8: TDD cycle (write failing test → run fail → implement → run pass → commit). These are the *financial logic* — per Walter's pragmatic-TDD memory, this is exactly where TDD pays.
- Tasks 9-12: Plumbing (config, integration backtest, analysis script, systemd unit). No TDD; verify-and-commit pattern.
- One commit per task. Use `git add <specific files>`, never `git add .` (avoids polluting unrelated SMC_Forge uncommitted changes in the repo).

---

## Task 1: Scaffold ORBSession class + test infrastructure

**Files:**
- Create: `user_data/strategies/ORBSession.py`
- Create: `user_data/strategies/ORBSession/__init__.py`
- Create: `user_data/strategies/ORBSession/test_helpers.py`
- Create: `user_data/strategies/ORBSession/test_orb_session.py`

- [ ] **Step 1.1: Create the test infrastructure**

Create `user_data/strategies/ORBSession/__init__.py` as empty file.

Create `user_data/strategies/ORBSession/test_helpers.py`:

```python
"""Synthetic OHLCV builders for ORBSession unit tests."""

from datetime import datetime, timedelta, timezone

import numpy as np
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


def make_multi_day(date_strs: list[str], base_price: float = 100.0) -> pd.DataFrame:
    """Concatenate several days of flat 15m candles into one DataFrame."""
    parts = [make_15m_day(d, base_price=base_price) for d in date_strs]
    return pd.concat(parts, ignore_index=True)
```

Create `user_data/strategies/ORBSession/test_orb_session.py`:

```python
"""Unit tests for ORBSession strategy.

Run with: pytest user_data/strategies/ORBSession/test_orb_session.py -v
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

# Make strategies/ importable
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from ORBSession import ORBSession  # noqa: E402

from test_helpers import make_15m_day, make_multi_day  # noqa: E402


@pytest.fixture
def strat():
    """ORBSession with empty config (all defaults)."""
    return ORBSession(config={})


def test_scaffold_imports():
    """Smoke test: class imports and instantiates."""
    s = ORBSession(config={})
    assert s.timeframe == '15m'
    assert s.can_short is True
```

- [ ] **Step 1.2: Run the smoke test, confirm it fails (no class yet)**

Run from `/home/wortiz/Desktop/freqtrade`:
```bash
pytest user_data/strategies/ORBSession/test_orb_session.py::test_scaffold_imports -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'ORBSession'`.

- [ ] **Step 1.3: Write the minimal ORBSession class**

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

See user_data/strategies/ORBSession/2026-05-20-design.md
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
    startup_candle_count: int = 96   # 24h of 15m for warmup

    minimal_roi = {"0": 100}    # disabled; custom_stoploss + exit signal handle exits
    stoploss = -0.99            # placeholder; real stop in custom_stoploss

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0
        return dataframe
```

- [ ] **Step 1.4: Run the smoke test, confirm it passes**

```bash
pytest user_data/strategies/ORBSession/test_orb_session.py::test_scaffold_imports -v
```
Expected: PASS.

- [ ] **Step 1.5: Commit**

```bash
git add user_data/strategies/ORBSession.py \
        user_data/strategies/ORBSession/__init__.py \
        user_data/strategies/ORBSession/test_helpers.py \
        user_data/strategies/ORBSession/test_orb_session.py
git commit -m "ORBSession: scaffold IStrategy class + test infra"
```

---

## Task 2: Range calculation (13:00-14:00 UTC)

**Files:**
- Modify: `user_data/strategies/ORBSession.py` (add range calc in `populate_indicators`)
- Modify: `user_data/strategies/ORBSession/test_orb_session.py` (add range tests)

- [ ] **Step 2.1: Write the failing test for range calculation**

Append to `test_orb_session.py`:

```python
def test_range_high_low_computed_at_session_end(strat):
    """At 14:00 (after 13:00-13:45 candles close), range_high and range_low
    must equal the max/min of those 4 candles."""
    df = make_15m_day(
        '2026-04-15',
        base_price=100.0,
        overrides={
            '13:00': {'high': 101.0, 'low': 99.5},
            '13:15': {'high': 102.5, 'low': 100.0},
            '13:30': {'high': 101.8, 'low': 99.0},
            '13:45': {'high': 102.0, 'low': 100.2},
        },
    )
    out = strat.populate_indicators(df, {'pair': 'BTC/USDT:USDT'})
    row_14 = out.loc[(out['date'].dt.hour == 14) & (out['date'].dt.minute == 0)].iloc[0]
    assert row_14['orb_range_high'] == pytest.approx(102.5)
    assert row_14['orb_range_low'] == pytest.approx(99.0)


def test_range_nan_before_session_end(strat):
    """Before 14:00, range_high/low must be NaN (range not yet known)."""
    df = make_15m_day('2026-04-15')
    out = strat.populate_indicators(df, {'pair': 'BTC/USDT:USDT'})
    row_13_30 = out.loc[(out['date'].dt.hour == 13) & (out['date'].dt.minute == 30)].iloc[0]
    assert pd.isna(row_13_30['orb_range_high'])
    assert pd.isna(row_13_30['orb_range_low'])


def test_range_persists_until_next_session(strat):
    """Range computed at 14:00 must stay in effect through 23:45 same day."""
    df = make_15m_day(
        '2026-04-15',
        overrides={
            '13:00': {'high': 110.0, 'low': 90.0},
        },
    )
    out = strat.populate_indicators(df, {'pair': 'BTC/USDT:USDT'})
    row_20_45 = out.loc[(out['date'].dt.hour == 20) & (out['date'].dt.minute == 45)].iloc[0]
    assert row_20_45['orb_range_high'] == pytest.approx(110.0)
    assert row_20_45['orb_range_low'] == pytest.approx(90.0)
```

- [ ] **Step 2.2: Run tests, confirm they fail**

```bash
pytest user_data/strategies/ORBSession/test_orb_session.py -v -k "range"
```
Expected: 3 FAILs (KeyError on `orb_range_high`).

- [ ] **Step 2.3: Implement range calculation**

Replace the `populate_indicators` method in `ORBSession.py`:

```python
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe.copy()

        # Mark the 4 "range candles" (13:00, 13:15, 13:30, 13:45 UTC)
        in_range_window = (df['date'].dt.hour == self.SESSION_START_HOUR) & \
                          (df['date'].dt.minute < 60)
        # Group key = UTC date (so each day's range is independent)
        session_date = df['date'].dt.date

        # high/low restricted to range candles, NaN elsewhere
        range_high_candidates = df['high'].where(in_range_window)
        range_low_candidates  = df['low'].where(in_range_window)

        # cummax/cummin within each session_date, starting fresh each day
        df['orb_range_high'] = range_high_candidates.groupby(session_date).cummax()
        df['orb_range_low']  = range_low_candidates.groupby(session_date).cummin()

        # Forward-fill the final range value across the rest of the day.
        # We only want the value from 14:00 onwards (after all 4 range candles closed).
        # Mask: before 14:00 same day → NaN; from 14:00 onwards → ffill within day.
        post_range = df['date'].dt.hour >= self.RANGE_END_HOUR
        df.loc[~post_range, 'orb_range_high'] = pd.NA
        df.loc[~post_range, 'orb_range_low'] = pd.NA
        df['orb_range_high'] = df.groupby(session_date)['orb_range_high'].ffill()
        df['orb_range_low']  = df.groupby(session_date)['orb_range_low'].ffill()

        return df
```

- [ ] **Step 2.4: Run tests, confirm they pass**

```bash
pytest user_data/strategies/ORBSession/test_orb_session.py -v -k "range"
```
Expected: 3 PASS.

- [ ] **Step 2.5: Commit**

```bash
git add user_data/strategies/ORBSession.py \
        user_data/strategies/ORBSession/test_orb_session.py
git commit -m "ORBSession: compute 13:00-14:00 UTC range in populate_indicators"
```

---

## Task 3: Skip-wide-range filter

**Files:**
- Modify: `user_data/strategies/ORBSession.py`
- Modify: `user_data/strategies/ORBSession/test_orb_session.py`

- [ ] **Step 3.1: Write failing tests for skip filter**

Append to `test_orb_session.py`:

```python
def test_skip_when_range_exceeds_max_pct(strat):
    """If range_pct > MAX_RANGE_PCT (1.5%), orb_skipped must be True."""
    # Range = 102 - 98 = 4 on price ~100 → 4% range, > 1.5% threshold
    df = make_15m_day(
        '2026-04-15',
        base_price=100.0,
        overrides={
            '13:00': {'high': 102.0, 'low': 98.0, 'close': 100.0},
        },
    )
    out = strat.populate_indicators(df, {'pair': 'BTC/USDT:USDT'})
    row_14 = out.loc[(out['date'].dt.hour == 14) & (out['date'].dt.minute == 0)].iloc[0]
    assert row_14['orb_skipped'] is True or row_14['orb_skipped'] == True  # bool or numpy bool


def test_not_skipped_when_range_within_threshold(strat):
    """Range 0.5% (within 1.5%) → orb_skipped False."""
    df = make_15m_day(
        '2026-04-15',
        base_price=100.0,
        overrides={
            '13:00': {'high': 100.5, 'low': 100.0, 'close': 100.2},
        },
    )
    out = strat.populate_indicators(df, {'pair': 'BTC/USDT:USDT'})
    row_14 = out.loc[(out['date'].dt.hour == 14) & (out['date'].dt.minute == 0)].iloc[0]
    assert row_14['orb_skipped'] == False
```

- [ ] **Step 3.2: Run tests, confirm fail**

```bash
pytest user_data/strategies/ORBSession/test_orb_session.py -v -k "skip"
```
Expected: 2 FAILs (KeyError on `orb_skipped`).

- [ ] **Step 3.3: Implement skip filter**

In `populate_indicators`, before the `return df` line, append:

```python
        # Range % = (high - low) / price_at_range_end_close
        # Use close of 13:45 candle as reference price.
        ref_close = df['close'].where(
            (df['date'].dt.hour == self.SESSION_START_HOUR) &
            (df['date'].dt.minute == 45)
        )
        ref_close_filled = ref_close.groupby(session_date).bfill().ffill()

        range_width = df['orb_range_high'] - df['orb_range_low']
        range_pct = range_width / ref_close_filled

        df['orb_range_pct'] = range_pct
        df['orb_skipped'] = (range_pct > self.MAX_RANGE_PCT).fillna(False)
```

- [ ] **Step 3.4: Run tests, confirm pass**

```bash
pytest user_data/strategies/ORBSession/test_orb_session.py -v -k "skip"
```
Expected: 2 PASS.

- [ ] **Step 3.5: Commit**

```bash
git add user_data/strategies/ORBSession.py \
        user_data/strategies/ORBSession/test_orb_session.py
git commit -m "ORBSession: skip session when range > MAX_RANGE_PCT (1.5%)"
```

---

## Task 4: Entry long signal

**Files:**
- Modify: `user_data/strategies/ORBSession.py`
- Modify: `user_data/strategies/ORBSession/test_orb_session.py`

- [ ] **Step 4.1: Write failing tests**

Append to `test_orb_session.py`:

```python
def test_entry_long_on_breakout_above_range(strat):
    """Close at 14:15 strictly above range_high → enter_long = 1 on that candle."""
    df = make_15m_day(
        '2026-04-15',
        base_price=100.0,
        overrides={
            '13:00': {'high': 100.5, 'low': 99.5, 'close': 100.0},
            '14:15': {'high': 101.0, 'low': 100.5, 'close': 100.8},  # above 100.5
        },
    )
    out = strat.populate_indicators(df, {'pair': 'BTC/USDT:USDT'})
    out = strat.populate_entry_trend(out, {'pair': 'BTC/USDT:USDT'})
    row_14_15 = out.loc[(out['date'].dt.hour == 14) & (out['date'].dt.minute == 15)].iloc[0]
    assert row_14_15['enter_long'] == 1
    assert row_14_15['enter_short'] == 0


def test_no_entry_long_when_close_equals_range_high(strat):
    """Strict inequality: close == range_high → no entry."""
    df = make_15m_day(
        '2026-04-15',
        overrides={
            '13:00': {'high': 100.5, 'low': 99.5, 'close': 100.0},
            '14:15': {'close': 100.5},
        },
    )
    out = strat.populate_indicators(df, {'pair': 'BTC/USDT:USDT'})
    out = strat.populate_entry_trend(out, {'pair': 'BTC/USDT:USDT'})
    row_14_15 = out.loc[(out['date'].dt.hour == 14) & (out['date'].dt.minute == 15)].iloc[0]
    assert row_14_15['enter_long'] == 0


def test_no_entry_long_when_session_skipped(strat):
    """If session is skipped (wide range), no entries emitted even if breakout."""
    df = make_15m_day(
        '2026-04-15',
        overrides={
            '13:00': {'high': 105.0, 'low': 95.0, 'close': 100.0},  # 10% range, skipped
            '14:15': {'close': 106.0},  # would be breakout
        },
    )
    out = strat.populate_indicators(df, {'pair': 'BTC/USDT:USDT'})
    out = strat.populate_entry_trend(out, {'pair': 'BTC/USDT:USDT'})
    row_14_15 = out.loc[(out['date'].dt.hour == 14) & (out['date'].dt.minute == 15)].iloc[0]
    assert row_14_15['enter_long'] == 0


def test_no_entry_long_outside_session_window(strat):
    """Breakout at 22:00 (after SESSION_END_HOUR 21) → no entry."""
    df = make_15m_day(
        '2026-04-15',
        overrides={
            '13:00': {'high': 100.5, 'low': 99.5, 'close': 100.0},
            '22:00': {'close': 101.0},
        },
    )
    out = strat.populate_indicators(df, {'pair': 'BTC/USDT:USDT'})
    out = strat.populate_entry_trend(out, {'pair': 'BTC/USDT:USDT'})
    row_22 = out.loc[(out['date'].dt.hour == 22) & (out['date'].dt.minute == 0)].iloc[0]
    assert row_22['enter_long'] == 0
```

- [ ] **Step 4.2: Run tests, confirm fail**

```bash
pytest user_data/strategies/ORBSession/test_orb_session.py -v -k "entry_long"
```
Expected: 4 FAILs.

- [ ] **Step 4.3: Implement entry long logic**

Replace `populate_entry_trend` in `ORBSession.py`:

```python
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe
        in_session = (df['date'].dt.hour >= self.RANGE_END_HOUR) & \
                     (df['date'].dt.hour < self.SESSION_END_HOUR)
        not_skipped = ~df['orb_skipped'].fillna(False)
        has_range = df['orb_range_high'].notna() & df['orb_range_low'].notna()

        long_breakout = df['close'] > df['orb_range_high']
        short_breakout = df['close'] < df['orb_range_low']

        df['enter_long'] = (in_session & not_skipped & has_range & long_breakout).astype(int)
        df['enter_short'] = (in_session & not_skipped & has_range & short_breakout).astype(int)
        return df
```

- [ ] **Step 4.4: Run tests, confirm pass**

```bash
pytest user_data/strategies/ORBSession/test_orb_session.py -v -k "entry_long"
```
Expected: 4 PASS.

- [ ] **Step 4.5: Commit**

```bash
git add user_data/strategies/ORBSession.py \
        user_data/strategies/ORBSession/test_orb_session.py
git commit -m "ORBSession: emit enter_long on breakout above range in session"
```

---

## Task 5: Entry short signal (symmetric)

**Files:**
- Modify: `user_data/strategies/ORBSession/test_orb_session.py`

The implementation already covers shorts (Task 4 implemented both legs symmetrically). This task adds tests to lock the behavior and catch regressions.

- [ ] **Step 5.1: Write tests for short side**

Append to `test_orb_session.py`:

```python
def test_entry_short_on_breakout_below_range(strat):
    df = make_15m_day(
        '2026-04-15',
        base_price=100.0,
        overrides={
            '13:00': {'high': 100.5, 'low': 99.5, 'close': 100.0},
            '14:15': {'high': 99.5, 'low': 99.0, 'close': 99.2},  # below 99.5
        },
    )
    out = strat.populate_indicators(df, {'pair': 'BTC/USDT:USDT'})
    out = strat.populate_entry_trend(out, {'pair': 'BTC/USDT:USDT'})
    row_14_15 = out.loc[(out['date'].dt.hour == 14) & (out['date'].dt.minute == 15)].iloc[0]
    assert row_14_15['enter_short'] == 1
    assert row_14_15['enter_long'] == 0


def test_no_entry_short_when_close_equals_range_low(strat):
    df = make_15m_day(
        '2026-04-15',
        overrides={
            '13:00': {'high': 100.5, 'low': 99.5, 'close': 100.0},
            '14:15': {'close': 99.5},
        },
    )
    out = strat.populate_indicators(df, {'pair': 'BTC/USDT:USDT'})
    out = strat.populate_entry_trend(out, {'pair': 'BTC/USDT:USDT'})
    row_14_15 = out.loc[(out['date'].dt.hour == 14) & (out['date'].dt.minute == 15)].iloc[0]
    assert row_14_15['enter_short'] == 0
```

- [ ] **Step 5.2: Run tests, confirm pass (no impl change needed)**

```bash
pytest user_data/strategies/ORBSession/test_orb_session.py -v -k "entry_short"
```
Expected: 2 PASS.

- [ ] **Step 5.3: Commit**

```bash
git add user_data/strategies/ORBSession/test_orb_session.py
git commit -m "ORBSession: lock short-entry behavior with tests"
```

---

## Task 6: No re-entry within the same session

**Files:**
- Modify: `user_data/strategies/ORBSession.py`
- Modify: `user_data/strategies/ORBSession/test_orb_session.py`

- [ ] **Step 6.1: Write failing test**

Append to `test_orb_session.py`:

```python
def test_no_reentry_after_first_breakout(strat):
    """Once a signal is emitted in a session, no subsequent signal that day."""
    df = make_15m_day(
        '2026-04-15',
        overrides={
            '13:00': {'high': 100.5, 'low': 99.5, 'close': 100.0},
            '14:15': {'close': 101.0},  # first long breakout
            '15:00': {'close': 102.0},  # would re-trigger; must be suppressed
            '16:00': {'close': 98.0},   # opposite-side short; also suppressed
        },
    )
    out = strat.populate_indicators(df, {'pair': 'BTC/USDT:USDT'})
    out = strat.populate_entry_trend(out, {'pair': 'BTC/USDT:USDT'})

    longs_in_session = out.loc[
        (out['date'].dt.hour >= 14) & (out['date'].dt.hour < 21),
        'enter_long'
    ].sum()
    shorts_in_session = out.loc[
        (out['date'].dt.hour >= 14) & (out['date'].dt.hour < 21),
        'enter_short'
    ].sum()
    assert longs_in_session == 1
    assert shorts_in_session == 0
```

- [ ] **Step 6.2: Run test, confirm fail**

```bash
pytest user_data/strategies/ORBSession/test_orb_session.py -v -k "no_reentry"
```
Expected: FAIL (longs_in_session == 2).

- [ ] **Step 6.3: Implement no-reentry gate**

Modify `populate_entry_trend` in `ORBSession.py` to suppress signals after the first hit per session. Replace the method with:

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

        # Per UTC date, allow only the first breakout candle.
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

- [ ] **Step 6.4: Run all entry tests, confirm pass**

```bash
pytest user_data/strategies/ORBSession/test_orb_session.py -v -k "entry or reentry"
```
Expected: all PASS.

- [ ] **Step 6.5: Commit**

```bash
git add user_data/strategies/ORBSession.py \
        user_data/strategies/ORBSession/test_orb_session.py
git commit -m "ORBSession: suppress re-entries after first breakout per session"
```

---

## Task 7: Custom stoploss = opposite of range

**Files:**
- Modify: `user_data/strategies/ORBSession.py`
- Modify: `user_data/strategies/ORBSession/test_orb_session.py`

- [ ] **Step 7.1: Write failing test for custom_stoploss**

Append to `test_orb_session.py`:

```python
from datetime import datetime, timezone
from unittest.mock import MagicMock


def _fake_trade(pair: str, is_short: bool, open_rate: float):
    """Minimal Trade mock that custom_stoploss inspects."""
    t = MagicMock()
    t.pair = pair
    t.is_short = is_short
    t.open_rate = open_rate
    t.open_date_utc = datetime(2026, 4, 15, 14, 15, tzinfo=timezone.utc)
    return t


def test_custom_stoploss_long(strat, monkeypatch):
    """For a long entered at 101 with range_low=99.5, stop must be (99.5/101)-1 = -0.01485."""
    df = make_15m_day(
        '2026-04-15',
        overrides={
            '13:00': {'high': 100.5, 'low': 99.5, 'close': 100.0},
            '14:15': {'close': 101.0},
        },
    )
    df = strat.populate_indicators(df, {'pair': 'BTC/USDT:USDT'})

    # freqtrade calls dp.get_analyzed_dataframe internally; we monkeypatch.
    strat.dp = MagicMock()
    strat.dp.get_analyzed_dataframe = MagicMock(return_value=(df, None))

    trade = _fake_trade('BTC/USDT:USDT', is_short=False, open_rate=101.0)
    current_time = datetime(2026, 4, 15, 14, 30, tzinfo=timezone.utc)
    sl = strat.custom_stoploss(
        pair='BTC/USDT:USDT',
        trade=trade,
        current_time=current_time,
        current_rate=101.0,
        current_profit=0.0,
    )
    assert sl == pytest.approx((99.5 / 101.0) - 1, abs=1e-5)


def test_custom_stoploss_short(strat):
    df = make_15m_day(
        '2026-04-15',
        overrides={
            '13:00': {'high': 100.5, 'low': 99.5, 'close': 100.0},
            '14:15': {'close': 99.0},
        },
    )
    df = strat.populate_indicators(df, {'pair': 'ETH/USDT:USDT'})

    strat.dp = MagicMock()
    strat.dp.get_analyzed_dataframe = MagicMock(return_value=(df, None))

    trade = _fake_trade('ETH/USDT:USDT', is_short=True, open_rate=99.0)
    current_time = datetime(2026, 4, 15, 14, 30, tzinfo=timezone.utc)
    sl = strat.custom_stoploss(
        pair='ETH/USDT:USDT',
        trade=trade,
        current_time=current_time,
        current_rate=99.0,
        current_profit=0.0,
    )
    # For short: stop above open. Freqtrade convention: custom_stoploss returns the
    # negative loss percentage from open_rate; for shorts, "loss" = price rising.
    # stop_pct = (open - stop) / open = (99 - 100.5) / 99 = -0.01515
    assert sl == pytest.approx((99.0 - 100.5) / 99.0, abs=1e-5)
```

- [ ] **Step 7.2: Run tests, confirm fail**

```bash
pytest user_data/strategies/ORBSession/test_orb_session.py -v -k "custom_stoploss"
```
Expected: 2 FAILs.

- [ ] **Step 7.3: Implement `custom_stoploss`**

Append this method to `ORBSession`:

```python
    def custom_stoploss(self, pair, trade, current_time, current_rate, current_profit, **kwargs):
        """
        Stop at opposite end of the opening range.
        - Long: stop = range_low → returned as (range_low / open_rate) - 1 (negative)
        - Short: stop = range_high → returned as (open_rate - range_high) / open_rate (negative)

        If range data not available (shouldn't happen post-14:00 same day),
        fall back to class-level stoploss (-0.99 placeholder = effectively none).
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        if dataframe is None or dataframe.empty:
            return self.stoploss

        # Find the row matching the trade's open candle (same UTC date as open).
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
        else:
            return (range_low / trade.open_rate) - 1
```

- [ ] **Step 7.4: Run tests, confirm pass**

```bash
pytest user_data/strategies/ORBSession/test_orb_session.py -v -k "custom_stoploss"
```
Expected: 2 PASS.

- [ ] **Step 7.5: Commit**

```bash
git add user_data/strategies/ORBSession.py \
        user_data/strategies/ORBSession/test_orb_session.py
git commit -m "ORBSession: custom_stoploss returns opposite-of-range stop"
```

---

## Task 8: Forced exit at 21:00 UTC

**Files:**
- Modify: `user_data/strategies/ORBSession.py`
- Modify: `user_data/strategies/ORBSession/test_orb_session.py`

- [ ] **Step 8.1: Write failing test**

Append to `test_orb_session.py`:

```python
def test_exit_signal_emitted_on_last_session_candle(strat):
    """The 20:45 candle (last 15m candle that closes at 21:00) must emit exit signal
    for both long and short, so freqtrade closes any open position at session end."""
    df = make_15m_day(
        '2026-04-15',
        overrides={
            '13:00': {'high': 100.5, 'low': 99.5, 'close': 100.0},
        },
    )
    out = strat.populate_indicators(df, {'pair': 'BTC/USDT:USDT'})
    out = strat.populate_exit_trend(out, {'pair': 'BTC/USDT:USDT'})

    row_20_45 = out.loc[(out['date'].dt.hour == 20) & (out['date'].dt.minute == 45)].iloc[0]
    assert row_20_45['exit_long'] == 1
    assert row_20_45['exit_short'] == 1

    # And no exit signal at, say, 18:00
    row_18 = out.loc[(out['date'].dt.hour == 18) & (out['date'].dt.minute == 0)].iloc[0]
    assert row_18['exit_long'] == 0
    assert row_18['exit_short'] == 0
```

- [ ] **Step 8.2: Run, confirm fail**

```bash
pytest user_data/strategies/ORBSession/test_orb_session.py -v -k "exit_signal"
```
Expected: FAIL.

- [ ] **Step 8.3: Implement forced exit**

Replace `populate_exit_trend` in `ORBSession.py`:

```python
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe
        # The 20:45 15m candle closes at 21:00 → emit exit there.
        is_last_session_candle = (df['date'].dt.hour == (self.SESSION_END_HOUR - 1)) & \
                                  (df['date'].dt.minute == 45)
        df['exit_long'] = is_last_session_candle.astype(int)
        df['exit_short'] = is_last_session_candle.astype(int)
        return df
```

- [ ] **Step 8.4: Run tests, confirm pass**

```bash
pytest user_data/strategies/ORBSession/test_orb_session.py -v -k "exit_signal"
```
Expected: PASS.

- [ ] **Step 8.5: Commit**

```bash
git add user_data/strategies/ORBSession.py \
        user_data/strategies/ORBSession/test_orb_session.py
git commit -m "ORBSession: forced exit at 20:45 (closes at 21:00 UTC)"
```

---

## Task 9: Multi-day reset verification

**Files:**
- Modify: `user_data/strategies/ORBSession/test_orb_session.py`

This is a no-code task: a test asserting that the range/skip/no-reentry state correctly resets between days. The implementation already uses `groupby(session_date)` which handles this — this task locks the behavior.

- [ ] **Step 9.1: Write the multi-day test**

Append:

```python
def test_state_resets_across_days(strat):
    """Day 1 has wide range (skipped). Day 2 has narrow range with breakout.
    Day 2 entry must fire — state from day 1 must NOT carry over."""
    day1 = make_15m_day(
        '2026-04-15',
        overrides={
            '13:00': {'high': 110.0, 'low': 90.0, 'close': 100.0},  # 20% range → skipped
        },
    )
    day2 = make_15m_day(
        '2026-04-16',
        overrides={
            '13:00': {'high': 100.5, 'low': 99.5, 'close': 100.0},
            '14:15': {'close': 101.0},  # clean long breakout day 2
        },
    )
    df = pd.concat([day1, day2], ignore_index=True)
    out = strat.populate_indicators(df, {'pair': 'BTC/USDT:USDT'})
    out = strat.populate_entry_trend(out, {'pair': 'BTC/USDT:USDT'})

    # Day 1: skipped
    day1_signals = out.loc[
        (out['date'].dt.date == day1['date'].dt.date.iloc[0]),
        ['enter_long', 'enter_short']
    ].sum().sum()
    assert day1_signals == 0

    # Day 2: one long entry at 14:15
    day2_long = out.loc[
        (out['date'].dt.date == day2['date'].dt.date.iloc[0]) &
        (out['date'].dt.hour == 14) & (out['date'].dt.minute == 15),
        'enter_long'
    ].iloc[0]
    assert day2_long == 1
```

- [ ] **Step 9.2: Run, confirm pass without code change**

```bash
pytest user_data/strategies/ORBSession/test_orb_session.py -v -k "resets_across_days"
```
Expected: PASS (the `groupby(session_date)` logic already isolates days).

- [ ] **Step 9.3: Run the full test suite to make sure nothing broke**

```bash
pytest user_data/strategies/ORBSession/test_orb_session.py -v
```
Expected: all 14 tests PASS.

- [ ] **Step 9.4: Commit**

```bash
git add user_data/strategies/ORBSession/test_orb_session.py
git commit -m "ORBSession: lock multi-day state reset with test"
```

---

## Task 10: Freqtrade run config (`ORBSession.json`)

**Files:**
- Create: `ORBSession.json` (at repo root, like `config_smc_forge.json`)

- [ ] **Step 10.1: Inspect an existing config for the conventions in this repo**

Run:
```bash
cat /home/wortiz/Desktop/freqtrade/config_smc_forge.json | head -60
```

Identify: `exchange`, `pair_whitelist`, `stake_currency`, `stake_amount`, `dry_run`, `db_url`, `api_server` port. We will reuse the same exchange config but override pairs, strategy, db, and port.

- [ ] **Step 10.2: Create `ORBSession.json`**

Create at `/home/wortiz/Desktop/freqtrade/ORBSession.json`:

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

**Notes:**
- `dry_run: true` and `dry_run_wallet: 10000` for paper trading by default. Switch to `dry_run: false` for live (Gate 4).
- `stake_amount: 200` per spec §6 Gate 4 (the lower of $200 or 0.5% equity).
- `api_server.listen_port: 8082` to avoid colliding with SMC_Forge's instance (which usually runs on 8080).
- `db_url` is a separate SQLite file → isolated state from SMC_Forge.
- `jwt_secret_key` and `password` are intentional placeholders to fail loudly before live.

- [ ] **Step 10.3: Validate the config syntactically**

```bash
cd /home/wortiz/Desktop/freqtrade
python -c "import json; json.load(open('ORBSession.json'))"
```
Expected: no output (valid JSON).

- [ ] **Step 10.4: Sanity-check with freqtrade**

```bash
cd /home/wortiz/Desktop/freqtrade
freqtrade show-config --config ORBSession.json 2>&1 | tail -20
```
Expected: config loads, prints active strategy = `ORBSession`. If freqtrade complains about missing strategy class, fix imports before continuing.

- [ ] **Step 10.5: Commit**

```bash
git add ORBSession.json
git commit -m "ORBSession: dry-run config (paper trade, port 8082, isolated DB)"
```

---

## Task 11: Integration backtest — 1 month real data

**Files:**
- No new files (uses freqtrade CLI).

This task confirms the strategy end-to-end against real Binance USDM data. **It is the first time we observe what the strategy actually does, not what we hope it does.** Do not skip the visual sanity check (Step 11.4).

- [ ] **Step 11.1: Download 1 month of 15m data for both pairs**

```bash
cd /home/wortiz/Desktop/freqtrade
freqtrade download-data \
    --exchange binance \
    --trading-mode futures \
    --pairs BTC/USDT:USDT ETH/USDT:USDT \
    --timeframes 15m \
    --timerange 20260401-20260501
```
Expected: data files in `user_data/data/binance/futures/`.

- [ ] **Step 11.2: Run backtest on April 2026**

```bash
freqtrade backtesting \
    --config ORBSession.json \
    --strategy ORBSession \
    --timerange 20260401-20260501 \
    --timeframe 15m \
    --export trades \
    --export-filename user_data/backtest_results/orb_april2026
```
Expected: completes without exceptions, outputs trade summary table.

- [ ] **Step 11.3: Sanity-check trade counts**

In April 2026 (30 days), expected:
- Max trades = 30 days × 2 pairs × 1 trade/session = **60 trades**.
- Skipped sessions (range > 1.5%) reduce this. Typical: 5-15% of sessions skipped → expect **45-55 trades**.

If total trades < 20 or > 65, something is wrong — review `populate_indicators` output for that month before continuing.

```bash
python -c "
import json
data = json.load(open('user_data/backtest_results/orb_april2026.json'))
trades = data['strategy']['ORBSession']['trades']
print(f'Total trades: {len(trades)}')
print(f'By pair: ', {p: sum(1 for t in trades if t[\"pair\"] == p) for p in ['BTC/USDT:USDT', 'ETH/USDT:USDT']})
print(f'Trade duration distribution (hours):')
import statistics
durations_h = [t['trade_duration'] / 60 for t in trades]
if durations_h:
    print(f'  median: {statistics.median(durations_h):.2f}h, max: {max(durations_h):.2f}h')
"
```

Expected duration: 0.5-7h typical, max ≤ 7h (entries between 14:00-20:45, forced exit at 21:00).

- [ ] **Step 11.4: Visual sanity-check 3 random trades**

Pick 3 trades from the backtest output and **manually verify on a TradingView chart** (BTC/USDT perp 15m):
1. The trade entered at the close of the candle where price first crossed the 13:00-14:00 high/low.
2. The stop was at the opposite side of that 13:00-14:00 range.
3. The trade exited either by stop or by 21:00 forced close.

If any of the 3 trades doesn't match, **stop and debug before proceeding to Gate 2**.

- [ ] **Step 11.5: Commit the backtest results (do NOT commit downloaded data)**

```bash
git add user_data/backtest_results/orb_april2026.json \
        user_data/backtest_results/orb_april2026.meta.json 2>/dev/null || true
git commit -m "ORBSession: April 2026 integration backtest baseline"
```

(If `.gitignore` excludes `user_data/backtest_results/`, that's OK — skip this commit step.)

---

## Task 12: Correlation analysis script (vs SMC_Forge)

**Files:**
- Create: `user_data/strategies/ORBSession/analyze_correlation.py`

The diversification thesis (spec §9) requires `|corr(equity_curve_ORB, equity_curve_SMC_Forge)| < 0.4`. This script computes that from backtest JSON outputs.

- [ ] **Step 12.1: Create the script**

Create `user_data/strategies/ORBSession/analyze_correlation.py`:

```python
"""
Diversification check: correlate ORBSession equity curve vs SMC_Forge equity curve.

Inputs:
    --orb     path to ORBSession backtest result JSON
    --smc     path to SMC_Forge backtest result JSON (same time range)

Output:
    Prints daily equity-curve Pearson correlation and decision:
        corr < 0.4         → GREEN: diversification holds
        0.4 <= corr < 0.6  → YELLOW: weak diversification, evaluate
        corr >= 0.6        → RED: diversification thesis fails

Per spec §9.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def equity_curve_from_backtest(json_path: Path, strategy_name: str) -> pd.Series:
    """Build a daily equity curve from a freqtrade backtest result JSON."""
    data = json.loads(json_path.read_text())
    trades = data['strategy'][strategy_name]['trades']
    if not trades:
        raise SystemExit(f'no trades in {json_path}')

    df = pd.DataFrame(trades)
    df['close_date'] = pd.to_datetime(df['close_date'])
    df = df.sort_values('close_date')
    df['date'] = df['close_date'].dt.tz_localize(None).dt.normalize()
    daily_pnl = df.groupby('date')['profit_abs'].sum()
    equity = daily_pnl.cumsum()
    return equity


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--orb', required=True, type=Path)
    ap.add_argument('--smc', required=True, type=Path)
    ap.add_argument('--orb-strategy-name', default='ORBSession')
    ap.add_argument('--smc-strategy-name', default='SMCForge')
    args = ap.parse_args()

    orb = equity_curve_from_backtest(args.orb, args.orb_strategy_name)
    smc = equity_curve_from_backtest(args.smc, args.smc_strategy_name)

    # Align on shared dates and use daily returns (not equity levels) for corr.
    aligned = pd.DataFrame({'orb': orb, 'smc': smc}).fillna(method='ffill').dropna()
    returns = aligned.diff().dropna()

    if len(returns) < 5:
        raise SystemExit(f'not enough overlapping days ({len(returns)}) to compute correlation')

    corr = returns['orb'].corr(returns['smc'])
    abs_corr = abs(corr)

    print(f'Overlapping days: {len(returns)}')
    print(f'Pearson correlation (daily returns): {corr:+.3f}')
    print(f'|corr| = {abs_corr:.3f}')
    if abs_corr < 0.4:
        verdict = 'GREEN — diversification holds (spec §9 target)'
    elif abs_corr < 0.6:
        verdict = 'YELLOW — weak diversification, evaluate operational complexity'
    else:
        verdict = 'RED — diversification thesis fails, kill candidate (spec §9)'
    print(f'Verdict: {verdict}')


if __name__ == '__main__':
    main()
```

- [ ] **Step 12.2: Smoke-test the script on the April backtest (paired with whatever SMC_Forge backtest is available)**

```bash
cd /home/wortiz/Desktop/freqtrade
ls user_data/backtest_results/ | grep -i smc | head -3
```
Pick an SMC_Forge backtest result covering April 2026 (or the closest available period). If none exists, regenerate one with:

```bash
freqtrade backtesting --config config_smc_forge.json --strategy SMCForge \
    --timerange 20260401-20260501 \
    --export trades --export-filename user_data/backtest_results/smc_april2026
```

Then run the correlation script:

```bash
python user_data/strategies/ORBSession/analyze_correlation.py \
    --orb user_data/backtest_results/orb_april2026.json \
    --smc user_data/backtest_results/smc_april2026.json
```
Expected: prints overlapping days, correlation, verdict.

**Note:** 1 month of data is too short for a real diversification verdict. The Gate 2 corr check uses the full multi-year backtest. This smoke test only confirms the script runs end-to-end.

- [ ] **Step 12.3: Commit**

```bash
git add user_data/strategies/ORBSession/analyze_correlation.py
git commit -m "ORBSession: correlation analysis script vs SMC_Forge"
```

---

## Task 13: systemd service template

**Files:**
- Create: `user_data/strategies/ORBSession/freqtrade-orb.service` (template; deploy with sudo)

- [ ] **Step 13.1: Create the unit file template**

Create `user_data/strategies/ORBSession/freqtrade-orb.service`:

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

# Sandboxing (matches existing tradeco systemd conventions)
NoNewPrivileges=true
ProtectSystem=full
ProtectHome=false

[Install]
WantedBy=multi-user.target
```

- [ ] **Step 13.2: Document deployment (no auto-install)**

This file is a **template**. Do not deploy automatically — Walter will install it under sudo. Document the install steps as a comment block prepended to the file? No — keep the file clean. Instead, surface the install commands in the final report so Walter can copy them when ready:

```bash
sudo cp user_data/strategies/ORBSession/freqtrade-orb.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now freqtrade-orb
sudo systemctl status freqtrade-orb
```

**Reminder for Walter:** verify the venv path exists (`/home/wortiz/Desktop/freqtrade/.venv/bin/freqtrade`) before enabling. Adjust the path in the unit file if your freqtrade install lives elsewhere.

- [ ] **Step 13.3: Commit**

```bash
git add user_data/strategies/ORBSession/freqtrade-orb.service
git commit -m "ORBSession: systemd unit template (manual deploy)"
```

---

## Final verification

- [ ] **Step F.1: Run the full test suite**

```bash
cd /home/wortiz/Desktop/freqtrade
pytest user_data/strategies/ORBSession/test_orb_session.py -v
```
Expected: 14+ tests PASS.

- [ ] **Step F.2: Confirm config loads and backtest still runs**

```bash
freqtrade show-config --config ORBSession.json | head -5
freqtrade backtesting --config ORBSession.json --timerange 20260401-20260415 --timeframe 15m 2>&1 | tail -10
```
Expected: backtest summary table prints.

- [ ] **Step F.3: Final commit (if anything left untracked)**

```bash
git status --short
# If only test/strategy files appear, commit them. If unrelated repo state appears, DO NOT touch.
```

---

## Out of scope (deferred to subsequent plans)

- **Hyperopt** — none in v1 (spec §10).
- **`bt-forensics` Gate 2** — separate plan after Task 11 baseline shows minimal viability.
- **Paper Gate 3** — separate plan once Gate 2 passes.
- **Multi-session (Asia/EU) v2** — only if v1 produces measurable edge.
- **Additional pairs (SOL, etc.)** — v2 expansion.
- **Funding-rate aware filter** — open question §11.1, evaluate post-Gate 3.

---

## Self-review checklist (run before handing off)

1. **Spec coverage:**
   - §3 Mechanics (range, entry, stop, exit, sizing) → Tasks 2, 3, 4, 5, 6, 7, 8, 10. ✓
   - §4 Configuration (class attrs + freqtrade config) → Tasks 1 (attrs) + 10 (config file). ✓
   - §5 Flow → covered structurally by Tasks 2-8. ✓
   - §6 Validation gates → Gate 1 partial via Task 11 (1-month baseline). Full Gate 1 (3-year backtest) deferred — explicit out-of-scope. ✓
   - §7 Risk / kill criteria → not implemented in code (procedural, not strategy logic). Documented in spec, enforced by Walter's monitoring during Gate 4. ✓ (No task needed.)
   - §8 Testing → unit tests in Tasks 2-9; integration test in Task 11. ✓
   - §9 Diversification → Task 12. ✓
   - §10 Decisions (anti-revisitar) → no implementation; spec only. ✓
   - §11 Open questions → deferred (documented out-of-scope). ✓

2. **Placeholder scan:** no "TBD", "TODO", "fill in details", or unimplemented references found. ✓

3. **Type consistency:**
   - `orb_range_high`, `orb_range_low`, `orb_range_pct`, `orb_skipped` — consistent across Tasks 2, 3, 4, 6, 7. ✓
   - `MAX_RANGE_PCT`, `ALLOW_REENTRY`, `SESSION_START_HOUR`, `RANGE_END_HOUR`, `SESSION_END_HOUR` — class attrs defined in Task 1, used in Tasks 3, 4, 6, 8. ✓
   - `custom_stoploss` signature matches freqtrade v3 IStrategy contract. ✓
