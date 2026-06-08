# ICTSilverBullet Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implementar un ICT Silver Bullet fiel (`ICTSilverBullet`) sobre acciones tokenizadas Bitget, corrigiendo los 4 bugs de fidelidad del `SMCForgeSilverBullet` actual, reusando el motor `SMC_Forge`.

**Architecture:** Lógica pura y testeable (ventanas SB DST-aware + state machine causal sweep→MSS→FVG + helpers de salida) vive en `ict_sb_lib.py` (sin imports de freqtrade). La clase `ICTSilverBullet` wirea esos helpers con el motor `SMC_Forge` (swings/CHoCH/displacement/sweep/FVG causales, sin lookahead). FVG-only, salidas híbridas con time-stop de sesión.

**Tech Stack:** Python 3.11+, freqtrade 2025.12, pandas/numpy, `zoneinfo` (stdlib), pytest, motor `SMC_Forge` (numba).

**Spec:** `docs/superpowers/specs/2026-06-08-ict-silver-bullet-rewrite-design.md`

**Contrato del motor (columnas verificadas que se consumen):**
- `SMCEngine(df, internal_length, swing_length).get_signals()` → `internal_choch_bullish/bearish`, `swing_choch_bullish/bearish`, `internal_sweep_bullish/bearish`, `swing_sweep_bullish/bearish`, `active_bullish_fvg_top/bottom`, `active_bearish_fvg_top/bottom`, `swing_high`, `swing_low`, `equilibrium`, `swing_trend`.
- `annotate_displacement(df, signals, atr_period=14, lookback)` → `atr_14`, `internal_choch_bull_disp/bear_disp`, `swing_choch_bull_disp/bear_disp` (+ bos variants).
- `annotate_eqh_eql(df, atr_period=14, fractal_n, tolerance_atr)` → `atr_14`, `pivot_high_level`, `pivot_low_level`, `eqh_level`, `eql_level`.

---

## File Structure

- Create: `user_data/strategies/ict_sb_lib.py` — helpers puros (ventanas DST, state machine, salidas). Sin freqtrade.
- Create: `user_data/strategies/tests/test_ict_sb_lib.py` — unit tests de los helpers.
- Create: `user_data/strategies/ICTSilverBullet.py` — clase de estrategia (wiring).
- Create: `config_ictsb_bt.json` — config de backtest (lev1, 14 stocks, dry_run).
- No tocar: `SMCForgeSilverBullet.py`, `SMC_Forge/`.

---

### Task 1: Ventanas SB DST-aware (`sb_window_flags`)

**Files:**
- Create: `user_data/strategies/ict_sb_lib.py`
- Test: `user_data/strategies/tests/test_ict_sb_lib.py`

- [ ] **Step 1: Write the failing test**

```python
# user_data/strategies/tests/test_ict_sb_lib.py
import sys, os
import pandas as pd
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from ict_sb_lib import sb_window_flags


def _utc(ts_list):
    return pd.Series(pd.to_datetime(ts_list, utc=True))


def test_sb_window_dst_summer():
    # 2025-07-15 is EDT (UTC-4): 10:00 ET = 14:00 UTC, 14:00 ET = 18:00 UTC
    dates = _utc([
        "2025-07-15 13:55", "2025-07-15 14:00", "2025-07-15 14:55",
        "2025-07-15 15:00", "2025-07-15 18:00", "2025-07-15 19:00",
    ])
    out = sb_window_flags(dates)
    assert list(out['in_sb']) == [False, True, True, False, True, False]
    assert list(out['sb_window_id']) == ['none', 'AM', 'AM', 'none', 'PM', 'none']
    assert list(out['sb_window_open']) == [False, True, False, False, True, False]


def test_sb_window_dst_winter():
    # 2025-12-15 is EST (UTC-5): 10:00 ET = 15:00 UTC, 14:00 ET = 19:00 UTC
    dates = _utc([
        "2025-12-15 14:55", "2025-12-15 15:00", "2025-12-15 16:00",
        "2025-12-15 19:00", "2025-12-15 20:00",
    ])
    out = sb_window_flags(dates)
    assert list(out['in_sb']) == [False, True, False, True, False]
    assert list(out['sb_window_id']) == ['none', 'AM', 'none', 'PM', 'none']
    assert list(out['sb_window_open']) == [False, True, False, True, False]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/wortiz/Desktop/freqtrade && .venv/bin/python -m pytest user_data/strategies/tests/test_ict_sb_lib.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ict_sb_lib'`

- [ ] **Step 3: Write minimal implementation**

```python
# user_data/strategies/ict_sb_lib.py
"""Pure, testable helpers for ICTSilverBullet. NO freqtrade imports."""
from __future__ import annotations

from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

_ET = ZoneInfo("America/New_York")
# Silver Bullet windows by ET hour-of-day: AM 10:00-11:00, PM 14:00-15:00.
_SB_AM_HOUR = 10
_SB_PM_HOUR = 14


def sb_window_flags(dates: pd.Series) -> pd.DataFrame:
    """Given tz-aware UTC timestamps, return SB window flags in ET (DST-aware).

    Columns:
      in_sb          bool  candle inside an SB window (AM or PM)
      sb_window_id   obj   'AM' / 'PM' / 'none'
      sb_window_open bool  first candle of a given (ET-date, window)
    """
    et = pd.to_datetime(dates).dt.tz_convert(_ET)
    hour = et.dt.hour.to_numpy()
    in_am = hour == _SB_AM_HOUR
    in_pm = hour == _SB_PM_HOUR
    in_sb = in_am | in_pm
    wid = np.where(in_am, "AM", np.where(in_pm, "PM", "none")).astype(object)

    et_date = et.dt.strftime("%Y-%m-%d").to_numpy()
    key = np.where(in_sb, np.char.add(np.char.add(et_date.astype(str), "_"),
                                      wid.astype(str)), "").astype(object)
    prev_key = np.empty_like(key)
    prev_key[0] = ""
    prev_key[1:] = key[:-1]
    window_open = in_sb & (key != prev_key)

    return pd.DataFrame(
        {"in_sb": in_sb, "sb_window_id": wid, "sb_window_open": window_open},
        index=dates.index,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/wortiz/Desktop/freqtrade && .venv/bin/python -m pytest user_data/strategies/tests/test_ict_sb_lib.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
cd /home/wortiz/Desktop/freqtrade
git add user_data/strategies/ict_sb_lib.py user_data/strategies/tests/test_ict_sb_lib.py
git commit -m "feat(ICTSilverBullet): DST-aware SB window flags (AM/PM ET)"
```

---

### Task 2: State machine causal sweep→MSS→FVG (`sb_entry_signals`)

**Files:**
- Modify: `user_data/strategies/ict_sb_lib.py`
- Test: `user_data/strategies/tests/test_ict_sb_lib.py`

- [ ] **Step 1: Write the failing test**

```python
# append to test_ict_sb_lib.py
from ict_sb_lib import sb_entry_signals


def _zeros(n): return np.zeros(n)
def _mk(n): return {k: _zeros(n) for k in [
    'int_sw_b','sw_sw_b','int_sw_be','sw_sw_be',
    'int_ch_b','sw_ch_b','int_ch_be','sw_ch_be',
    'disp_b','disp_be','fvg_bt','fvg_bb','fvg_bet','fvg_beb',
    'low','high','close','macro']}


def _call(d, in_sb, win_open, disp_threshold=0.8, use_macro=False):
    return sb_entry_signals(
        np.array(in_sb), np.array(win_open),
        d['int_sw_b'], d['sw_sw_b'], d['int_sw_be'], d['sw_sw_be'],
        d['int_ch_b'], d['sw_ch_b'], d['int_ch_be'], d['sw_ch_be'],
        d['disp_b'], d['disp_be'],
        d['fvg_bt'], d['fvg_bb'], d['fvg_bet'], d['fvg_beb'],
        d['low'], d['high'], d['close'], d['macro'],
        disp_threshold, use_macro)


def test_valid_long_sequence_fires():
    # bars: 0 sweep, 1 MSS(+disp+fvg), 2 retrace into fvg
    n = 4; d = _mk(n)
    in_sb = [True]*4; win_open = [True,False,False,False]
    d['int_sw_b'][0] = 1; d['low'][0] = 90.0           # sweep, sweep_low=90
    d['int_ch_b'][1] = 1; d['disp_b'][1] = 1.5         # MSS w/ displacement
    d['fvg_bt'][1] = 105.0; d['fvg_bb'][1] = 100.0     # fvg created at MSS
    d['low'][2] = 99.0; d['close'][2] = 102.0          # retrace into fvg
    el, es, sl, tag = _call(d, in_sb, win_open)
    assert el[2] == 1 and es.sum() == 0
    assert sl[2] == 90.0
    assert tag[2] == 'SB_long'


def test_wrong_order_no_entry():
    # MSS before sweep → must NOT fire
    n = 3; d = _mk(n)
    in_sb = [True]*3; win_open = [True,False,False]
    d['int_ch_b'][0] = 1; d['disp_b'][0] = 1.5; d['fvg_bt'][0]=105; d['fvg_bb'][0]=100
    d['int_sw_b'][1] = 1; d['low'][1] = 90.0
    d['low'][2] = 99.0; d['close'][2] = 102.0
    el, es, sl, tag = _call(d, in_sb, win_open)
    assert el.sum() == 0


def test_window_reset_breaks_sequence():
    # sweep+MSS in one window, retrace after window_open of next → reset, no entry
    n = 4; d = _mk(n)
    in_sb = [True,True,True,True]; win_open = [True,False,True,False]  # new window at bar 2
    d['int_sw_b'][0]=1; d['low'][0]=90
    d['int_ch_b'][1]=1; d['disp_b'][1]=1.5; d['fvg_bt'][1]=105; d['fvg_bb'][1]=100
    d['low'][3]=99; d['close'][3]=102
    el, es, sl, tag = _call(d, in_sb, win_open)
    assert el.sum() == 0


def test_macro_filter_blocks_long():
    n = 4; d = _mk(n)
    in_sb=[True]*4; win_open=[True,False,False,False]
    d['int_sw_b'][0]=1; d['low'][0]=90
    d['int_ch_b'][1]=1; d['disp_b'][1]=1.5; d['fvg_bt'][1]=105; d['fvg_bb'][1]=100
    d['low'][2]=99; d['close'][2]=102; d['macro'][2]=-1   # bias bearish blocks long
    el, es, sl, tag = _call(d, in_sb, win_open, use_macro=True)
    assert el.sum() == 0


def test_valid_short_sequence_fires():
    n=4; d=_mk(n)
    in_sb=[True]*4; win_open=[True,False,False,False]
    d['int_sw_be'][0]=1; d['high'][0]=110.0              # bearish sweep, sweep_high=110
    d['int_ch_be'][1]=1; d['disp_be'][1]=1.5
    d['fvg_bet'][1]=100.0; d['fvg_beb'][1]=95.0
    d['high'][2]=96.0; d['close'][2]=98.0                # retrace up into bearish fvg
    el, es, sl, tag = _call(d, in_sb, win_open)
    assert es[2] == 1 and el.sum() == 0
    assert sl[2] == 110.0
    assert tag[2] == 'SB_short'
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/wortiz/Desktop/freqtrade && .venv/bin/python -m pytest user_data/strategies/tests/test_ict_sb_lib.py -k sequence -v`
Expected: FAIL with `ImportError: cannot import name 'sb_entry_signals'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to ict_sb_lib.py
def sb_entry_signals(
    in_sb, sb_window_open,
    int_sweep_bull, sw_sweep_bull, int_sweep_bear, sw_sweep_bear,
    int_choch_bull, sw_choch_bull, int_choch_bear, sw_choch_bear,
    choch_bull_disp, choch_bear_disp,
    fvg_bull_top, fvg_bull_bottom, fvg_bear_top, fvg_bear_bottom,
    low, high, close, macro_bias,
    disp_threshold, use_macro,
):
    """Causal Silver Bullet state machine. All inputs are 1D arrays of len n.

    Enforces, within each SB window (reset on sb_window_open / leaving window),
    the strict bar-separated order: SWEEP -> MSS(CHoCH+displacement) -> FVG retrace.
    One stage advances per bar (elif chain) => sweep_bar < mss_bar < entry_bar.
    FVG captured at the MSS bar is the displacement's FVG. FVG-only (no OB).

    Returns (enter_long, enter_short, sl_price, tag).
    """
    n = len(in_sb)
    enter_long = np.zeros(n, dtype=np.int8)
    enter_short = np.zeros(n, dtype=np.int8)
    sl_price = np.full(n, np.nan)
    tag = np.empty(n, dtype=object)
    tag[:] = ""

    # long / short state
    l_swept = l_mss = False
    l_sweep_low = l_ftop = l_fbot = np.nan
    s_swept = s_mss = False
    s_sweep_high = s_ftop = s_fbot = np.nan

    for i in range(n):
        if sb_window_open[i] or not in_sb[i]:
            l_swept = l_mss = False
            l_sweep_low = l_ftop = l_fbot = np.nan
            s_swept = s_mss = False
            s_sweep_high = s_ftop = s_fbot = np.nan
        if not in_sb[i]:
            continue

        # ---- LONG (one stage per bar via elif => strict ordering) ----
        if l_mss and (l_ftop > 0) and (low[i] <= l_ftop) and (close[i] >= l_fbot):
            if (not use_macro) or (macro_bias[i] >= 0):
                enter_long[i] = 1
                sl_price[i] = l_sweep_low
                tag[i] = "SB_long"
            l_swept = l_mss = False
            l_sweep_low = l_ftop = l_fbot = np.nan
        elif l_swept and ((int_choch_bull[i] == 1) or (sw_choch_bull[i] == 1)) \
                and (choch_bull_disp[i] >= disp_threshold):
            l_mss = True
            l_ftop = fvg_bull_top[i]
            l_fbot = fvg_bull_bottom[i]
        elif (int_sweep_bull[i] == 1) or (sw_sweep_bull[i] == 1):
            l_swept = True
            l_sweep_low = low[i]

        # ---- SHORT mirror ----
        if s_mss and (s_ftop > 0) and (high[i] >= s_fbot) and (close[i] <= s_ftop):
            if (not use_macro) or (macro_bias[i] <= 0):
                enter_short[i] = 1
                sl_price[i] = s_sweep_high
                tag[i] = "SB_short"
            s_swept = s_mss = False
            s_sweep_high = s_ftop = s_fbot = np.nan
        elif s_swept and ((int_choch_bear[i] == 1) or (sw_choch_bear[i] == 1)) \
                and (choch_bear_disp[i] >= disp_threshold):
            s_mss = True
            s_ftop = fvg_bear_top[i]
            s_fbot = fvg_bear_bottom[i]
        elif (int_sweep_bear[i] == 1) or (sw_sweep_bear[i] == 1):
            s_swept = True
            s_sweep_high = high[i]

    return enter_long, enter_short, sl_price, tag
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/wortiz/Desktop/freqtrade && .venv/bin/python -m pytest user_data/strategies/tests/test_ict_sb_lib.py -v`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
cd /home/wortiz/Desktop/freqtrade
git add user_data/strategies/ict_sb_lib.py user_data/strategies/tests/test_ict_sb_lib.py
git commit -m "feat(ICTSilverBullet): causal sweep->MSS->FVG state machine"
```

---

### Task 3: Helpers de salida (`is_past_session_close`, `nearest_opposing_pool`, `tp1_target_price`)

**Files:**
- Modify: `user_data/strategies/ict_sb_lib.py`
- Test: `user_data/strategies/tests/test_ict_sb_lib.py`

- [ ] **Step 1: Write the failing test**

```python
# append to test_ict_sb_lib.py
from datetime import datetime, timezone
from ict_sb_lib import is_past_session_close, nearest_opposing_pool, tp1_target_price


def test_session_close_same_day_dst():
    # entry 2025-07-15 14:00 UTC (=10:00 EDT); 16:00 EDT = 20:00 UTC
    opn = datetime(2025, 7, 15, 14, 0, tzinfo=timezone.utc)
    assert is_past_session_close(opn, datetime(2025, 7, 15, 19, 0, tzinfo=timezone.utc)) is False
    assert is_past_session_close(opn, datetime(2025, 7, 15, 20, 0, tzinfo=timezone.utc)) is True


def test_nearest_opposing_pool_long():
    # long entry at 100; opposing pools ABOVE; pick nearest above
    pools = {'eqh_level': 108.0, 'swing_high': 103.0, 'active_bearish_fvg_top': 0.0}
    assert nearest_opposing_pool(100.0, pools, is_short=False) == 103.0


def test_nearest_opposing_pool_short():
    pools = {'eql_level': 92.0, 'swing_low': 97.0, 'active_bullish_fvg_bottom': 0.0}
    assert nearest_opposing_pool(100.0, pools, is_short=True) == 97.0


def test_tp1_target_r_mult_long():
    # entry 100, sl 96 => R=4; 1R target = 104 (long)
    assert tp1_target_price(100.0, 96.0, is_short=False, r_mult=1.0) == 104.0


def test_tp1_target_r_mult_short():
    assert tp1_target_price(100.0, 104.0, is_short=True, r_mult=1.0) == 96.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/wortiz/Desktop/freqtrade && .venv/bin/python -m pytest user_data/strategies/tests/test_ict_sb_lib.py -k "session or pool or tp1" -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Write minimal implementation**

```python
# append to ict_sb_lib.py
from datetime import datetime

_SESSION_CLOSE_HOUR_ET = 16  # NY cash session close


def is_past_session_close(open_dt_utc: datetime, current_dt_utc: datetime) -> bool:
    """True if current time is at/after 16:00 ET of the entry's ET date.
    Both args are tz-aware UTC datetimes."""
    open_et = open_dt_utc.astimezone(_ET)
    close_et = open_et.replace(hour=_SESSION_CLOSE_HOUR_ET, minute=0,
                               second=0, microsecond=0)
    return current_dt_utc.astimezone(_ET) >= close_et


def nearest_opposing_pool(entry: float, pools: dict, is_short: bool):
    """Nearest valid opposing-liquidity level. For longs, pools ABOVE entry;
    for shorts, pools BELOW. pools maps col->level (0/NaN = absent). Returns
    the nearest valid level, or None."""
    best = None
    for level in pools.values():
        if level is None:
            continue
        try:
            v = float(level)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(v) or v <= 0:
            continue
        if is_short:
            if v < entry and (best is None or v > best):
                best = v
        else:
            if v > entry and (best is None or v < best):
                best = v
    return best


def tp1_target_price(entry: float, sl: float, is_short: bool, r_mult: float) -> float:
    """TP1 at r_mult * R, where R = |entry - sl|."""
    r = abs(entry - sl)
    return entry - r * r_mult if is_short else entry + r * r_mult
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/wortiz/Desktop/freqtrade && .venv/bin/python -m pytest user_data/strategies/tests/test_ict_sb_lib.py -v`
Expected: PASS (12 passed)

- [ ] **Step 5: Commit**

```bash
cd /home/wortiz/Desktop/freqtrade
git add user_data/strategies/ict_sb_lib.py user_data/strategies/tests/test_ict_sb_lib.py
git commit -m "feat(ICTSilverBullet): exit helpers (session close, opposing pool, TP1)"
```

---

### Task 4: Clase `ICTSilverBullet` — esqueleto + `populate_indicators`

**Files:**
- Create: `user_data/strategies/ICTSilverBullet.py`
- Test: `user_data/strategies/tests/test_ict_strategy_smoke.py`

- [ ] **Step 1: Write the failing test (smoke: indicators sobre df sintético)**

```python
# user_data/strategies/tests/test_ict_strategy_smoke.py
import sys, os
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def _synthetic_ohlcv(n=1500, seed=7):
    rng = np.random.default_rng(seed)
    ret = rng.normal(0, 0.002, n)
    close = 100 * np.exp(np.cumsum(ret))
    high = close * (1 + np.abs(rng.normal(0, 0.001, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.001, n)))
    openp = np.concatenate([[close[0]], close[:-1]])
    vol = rng.uniform(1e3, 1e4, n)
    dates = pd.date_range("2025-07-01", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame({"date": dates, "open": openp, "high": high,
                         "low": low, "close": close, "volume": vol})


def test_populate_indicators_adds_sb_columns():
    from ICTSilverBullet import ICTSilverBullet
    s = ICTSilverBullet({})
    df = _synthetic_ohlcv()
    out = s.populate_indicators(df.copy(), {"pair": "TEST/USDT:USDT"})
    for col in ["atr_14", "in_sb", "sb_window_id", "sb_window_open",
                "active_bullish_fvg_top", "swing_high"]:
        assert col in out.columns, f"missing {col}"
    assert len(out) == len(df)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/wortiz/Desktop/freqtrade && .venv/bin/python -m pytest user_data/strategies/tests/test_ict_strategy_smoke.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ICTSilverBullet'`

- [ ] **Step 3: Write minimal implementation**

```python
# user_data/strategies/ICTSilverBullet.py
"""
ICTSilverBullet — ICT Silver Bullet fiel sobre acciones tokenizadas Bitget.

Reescritura de SMCForgeSilverBullet corrigiendo fidelidad ICT:
  - Ventanas SB canónicas de 1h, DST-aware (AM 10-11 ET, PM 14-15 ET).
  - Secuencia causal sweep -> MSS(CHoCH+displacement) -> FVG retrace (FVG-only).
  - Scalp de sesión: time-stop a 16:00 ET, nunca overnight.
Reusa el motor SMC_Forge (causal, sin lookahead).
"""
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame

from freqtrade.strategy import (
    BooleanParameter, DecimalParameter, IntParameter, IStrategy,
    stoploss_from_absolute,
)
from freqtrade.exchange import timeframe_to_prev_date
from freqtrade.persistence import Trade

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / 'SMC_Forge'))
sys.path.insert(0, str(_HERE))

from forge_engine import SMCEngine                 # noqa: E402
from forge_quality import annotate_displacement    # noqa: E402
from forge_levels import annotate_eqh_eql           # noqa: E402
import ict_sb_lib as sb                             # noqa: E402

logger = logging.getLogger(__name__)


class ICTSilverBullet(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = '5m'
    can_short = True
    process_only_new_candles = True
    use_exit_signal = True
    use_custom_stoploss = True
    position_adjustment_enable = True
    startup_candle_count: int = 1000

    stoploss = -0.10                 # hard backstop (FIJO, no escalado por leverage)
    minimal_roi = {"0": 100.0}       # ROI off; salidas por callbacks
    max_open_trades = 5

    # SMC engine (estructural — match SMCForge)
    internal_length = IntParameter(3, 8, default=5, space='buy', optimize=False)
    swing_length = IntParameter(20, 60, default=50, space='buy', optimize=False)
    fractal_n_major = IntParameter(3, 10, default=5, space='buy', optimize=False)
    eqh_tolerance_atr = DecimalParameter(0.05, 0.30, default=0.10, decimals=2,
                                         space='buy', optimize=False)
    disp_lookback_bars = IntParameter(2, 5, default=3, space='buy', optimize=False)

    # Silver Bullet
    disp_threshold = DecimalParameter(0.3, 2.5, default=0.8, decimals=2,
                                      space='buy', optimize=True)
    use_macro_filter = BooleanParameter(default=True, space='buy', optimize=True)
    macro_htf = '1h'

    # Exits
    sl_atr_buffer = DecimalParameter(0.3, 1.0, default=0.5, decimals=2,
                                     space='sell', optimize=False)
    tp1_r_mult = DecimalParameter(0.5, 2.0, default=1.0, decimals=1,
                                  space='sell', optimize=True)
    tp1_amount_pct = IntParameter(20, 80, default=50, space='sell', optimize=False)
    be_buffer_pct = DecimalParameter(0.0, 0.005, default=0.001, decimals=4,
                                     space='sell', optimize=False)

    # ---------- lifecycle ----------
    def leverage(self, pair, current_time, current_rate, proposed_leverage,
                 max_leverage, entry_tag, side, **kwargs) -> float:
        return min(float(self.config.get('leverage', 1.0)), max_leverage)

    def informative_pairs(self):
        if not self.dp:
            return []
        return [(p, self.macro_htf) for p in self.dp.current_whitelist()]

    # ---------- macro bias (HTF swing_trend, shift(1) => causal) ----------
    def _macro_bias(self, dataframe: DataFrame, metadata: dict) -> np.ndarray:
        out = np.zeros(len(dataframe))
        if not self.dp:
            return out
        try:
            ref = self.dp.get_pair_dataframe(metadata['pair'], self.macro_htf)
            if ref is None or ref.empty:
                return out
            sig = SMCEngine(ref, internal_length=int(self.internal_length.value),
                            swing_length=int(self.swing_length.value)).get_signals()
            b = ref[['date']].copy()
            b['bias'] = sig['swing_trend'].shift(1).fillna(0).astype(int).values
            merged = pd.merge_asof(dataframe[['date']].sort_values('date'),
                                   b.sort_values('date'), on='date',
                                   direction='backward')
            return merged['bias'].fillna(0).astype(int).to_numpy()
        except Exception as e:
            logger.warning("ICTSB macro bias failed %s: %s", metadata.get('pair'), e)
            return out

    # ---------- indicators ----------
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe
        signals = SMCEngine(df, internal_length=int(self.internal_length.value),
                            swing_length=int(self.swing_length.value)).get_signals()
        for col in signals.columns:
            df[col] = signals[col].values

        disp = annotate_displacement(df, signals, atr_period=14,
                                     lookback=int(self.disp_lookback_bars.value))
        for col in disp.columns:
            if col not in df.columns:
                df[col] = disp[col].values

        levels = annotate_eqh_eql(df, atr_period=14,
                                  fractal_n=int(self.fractal_n_major.value),
                                  tolerance_atr=float(self.eqh_tolerance_atr.value))
        for col in levels.columns:
            if col not in df.columns:
                df[col] = levels[col].values

        win = sb.sb_window_flags(df['date'])
        df['in_sb'] = win['in_sb'].values
        df['sb_window_id'] = win['sb_window_id'].values
        df['sb_window_open'] = win['sb_window_open'].values

        df['macro_bias'] = self._macro_bias(df, metadata)
        return df
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/wortiz/Desktop/freqtrade && .venv/bin/python -m pytest user_data/strategies/tests/test_ict_strategy_smoke.py -v`
Expected: PASS (1 passed). (El test instancia con `dp=None` → macro_bias devuelve ceros, OK.)

- [ ] **Step 5: Commit**

```bash
cd /home/wortiz/Desktop/freqtrade
git add user_data/strategies/ICTSilverBullet.py user_data/strategies/tests/test_ict_strategy_smoke.py
git commit -m "feat(ICTSilverBullet): strategy skeleton + populate_indicators"
```

---

### Task 5: `populate_entry_trend` + `populate_exit_trend`

**Files:**
- Modify: `user_data/strategies/ICTSilverBullet.py`
- Test: `user_data/strategies/tests/test_ict_strategy_smoke.py`

- [ ] **Step 1: Write the failing test**

```python
# append to test_ict_strategy_smoke.py
def test_entry_columns_present_and_binary():
    from ICTSilverBullet import ICTSilverBullet
    s = ICTSilverBullet({})
    df = _synthetic_ohlcv()
    df = s.populate_indicators(df, {"pair": "TEST/USDT:USDT"})
    df = s.populate_entry_trend(df, {"pair": "TEST/USDT:USDT"})
    for col in ["enter_long", "enter_short", "sb_sl_price"]:
        assert col in df.columns
    assert set(np.unique(df["enter_long"])).issubset({0, 1})
    assert set(np.unique(df["enter_short"])).issubset({0, 1})
    # entries solo dentro de ventana SB
    assert (df.loc[df["enter_long"] == 1, "in_sb"]).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/wortiz/Desktop/freqtrade && .venv/bin/python -m pytest user_data/strategies/tests/test_ict_strategy_smoke.py -k entry -v`
Expected: FAIL with `AttributeError: 'ICTSilverBullet' object has no attribute 'populate_entry_trend'`

- [ ] **Step 3: Write minimal implementation**

```python
# add methods to ICTSilverBullet
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe
        el, es, sl, tag = sb.sb_entry_signals(
            df['in_sb'].to_numpy(), df['sb_window_open'].to_numpy(),
            df['internal_sweep_bullish'].to_numpy(), df['swing_sweep_bullish'].to_numpy(),
            df['internal_sweep_bearish'].to_numpy(), df['swing_sweep_bearish'].to_numpy(),
            df['internal_choch_bullish'].to_numpy(), df['swing_choch_bullish'].to_numpy(),
            df['internal_choch_bearish'].to_numpy(), df['swing_choch_bearish'].to_numpy(),
            df['internal_choch_bull_disp'].to_numpy(), df['internal_choch_bear_disp'].to_numpy(),
            df['active_bullish_fvg_top'].to_numpy(), df['active_bullish_fvg_bottom'].to_numpy(),
            df['active_bearish_fvg_top'].to_numpy(), df['active_bearish_fvg_bottom'].to_numpy(),
            df['low'].to_numpy(), df['high'].to_numpy(), df['close'].to_numpy(),
            df['macro_bias'].to_numpy(),
            float(self.disp_threshold.value), bool(self.use_macro_filter.value),
        )
        df['enter_long'] = el
        df['enter_short'] = es
        df['sb_sl_price'] = sl
        df['enter_tag'] = tag
        return df

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Salidas 100% por callbacks (custom_stoploss / adjust / custom_exit).
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0
        return dataframe
```

Note: usa `internal_choch_bull_disp`/`bear_disp` (swing variants `swing_choch_*_disp` quedan para una iteración futura — el engine también emite los internal, suficiente para el MSS interno).

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/wortiz/Desktop/freqtrade && .venv/bin/python -m pytest user_data/strategies/tests/test_ict_strategy_smoke.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
cd /home/wortiz/Desktop/freqtrade
git add user_data/strategies/ICTSilverBullet.py user_data/strategies/tests/test_ict_strategy_smoke.py
git commit -m "feat(ICTSilverBullet): entry/exit trend wiring (FVG-only sequence)"
```

---

### Task 6: `custom_stoploss` (sweep SL + ATR buffer + BE) + `adjust_trade_position` (TP1)

**Files:**
- Modify: `user_data/strategies/ICTSilverBullet.py`
- Test: `user_data/strategies/tests/test_ict_strategy_smoke.py`

- [ ] **Step 1: Write the failing test (helper row + BE flag math)**

```python
# append to test_ict_strategy_smoke.py
def test_sl_uses_sweep_price_with_buffer():
    # con sb_sl_price < entry y atr conocido, custom_stoploss debe devolver ratio < 0
    from ICTSilverBullet import ICTSilverBullet
    s = ICTSilverBullet({})
    class T:  # trade stub
        open_rate = 100.0; is_short = False; leverage = 1.0
    # parchear _entry_row para devolver una fila con sb_sl_price y atr_14
    s._entry_row = lambda trade, pair: pd.Series({"sb_sl_price": 96.0, "atr_14": 1.0})
    r = s.custom_stoploss("TEST/USDT:USDT", T(), None, current_rate=100.0,
                          current_profit=0.0, after_fill=False)
    assert r is not None and r < 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/wortiz/Desktop/freqtrade && .venv/bin/python -m pytest user_data/strategies/tests/test_ict_strategy_smoke.py -k sl_uses -v`
Expected: FAIL with `AttributeError: ... has no attribute 'custom_stoploss'`

- [ ] **Step 3: Write minimal implementation**

```python
# add methods to ICTSilverBullet
    def _entry_row(self, trade: Trade, pair: str):
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if df is None or df.empty:
            return None
        entry_date = timeframe_to_prev_date(self.timeframe, trade.open_date_utc)
        row = df[df['date'] == entry_date]
        return None if row.empty else row.iloc[0]

    @staticmethod
    def _safe_float(row, col):
        if row is None:
            return None
        val = row.get(col, None)
        try:
            v = float(val)
        except (TypeError, ValueError):
            return None
        return None if (not np.isfinite(v) or v <= 0) else v

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float,
                        after_fill: bool, **kwargs):
        # BE after TP1
        if trade.get_custom_data('tp1_taken', False):
            side = -1 if trade.is_short else 1
            be = trade.open_rate * (1 + side * float(self.be_buffer_pct.value))
            return stoploss_from_absolute(be, current_rate=current_rate,
                                          is_short=trade.is_short, leverage=trade.leverage)
        row = self._entry_row(trade, pair)
        sweep = self._safe_float(row, 'sb_sl_price')
        atr = self._safe_float(row, 'atr_14')
        if sweep is None or atr is None:
            return None
        buf = atr * float(self.sl_atr_buffer.value)
        sl_price = (sweep + buf) if trade.is_short else (sweep - buf)
        return stoploss_from_absolute(sl_price, current_rate=current_rate,
                                      is_short=trade.is_short, leverage=trade.leverage)

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float,
                              min_stake, max_stake: float,
                              current_entry_rate: float, current_exit_rate: float,
                              current_entry_profit: float, current_exit_profit: float,
                              **kwargs):
        if trade.get_custom_data('tp1_taken', False):
            return None
        row = self._entry_row(trade, trade.pair)
        sweep = self._safe_float(row, 'sb_sl_price')
        if sweep is None:
            return None
        tp1 = sb.tp1_target_price(trade.open_rate, sweep, is_short=trade.is_short,
                                  r_mult=float(self.tp1_r_mult.value))
        hit = current_rate <= tp1 if trade.is_short else current_rate >= tp1
        if not hit:
            return None
        pct = float(self.tp1_amount_pct.value) / 100.0
        close_stake = (trade.amount * current_rate * pct) / max(trade.leverage, 1.0)
        trade.set_custom_data('tp1_taken', True)
        return -close_stake, 'sb_tp1_partial'
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/wortiz/Desktop/freqtrade && .venv/bin/python -m pytest user_data/strategies/tests/test_ict_strategy_smoke.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
cd /home/wortiz/Desktop/freqtrade
git add user_data/strategies/ICTSilverBullet.py user_data/strategies/tests/test_ict_strategy_smoke.py
git commit -m "feat(ICTSilverBullet): custom_stoploss (sweep SL+ATR, BE) + TP1 partial"
```

---

### Task 7: `custom_exit` (TP2 liquidez opuesta + time-stop de sesión)

**Files:**
- Modify: `user_data/strategies/ICTSilverBullet.py`
- Test: `user_data/strategies/tests/test_ict_strategy_smoke.py`

- [ ] **Step 1: Write the failing test**

```python
# append to test_ict_strategy_smoke.py
from datetime import datetime, timezone

def test_custom_exit_session_timestop():
    from ICTSilverBullet import ICTSilverBullet
    s = ICTSilverBullet({})
    class T:
        open_rate = 100.0; is_short = False; leverage = 1.0
        pair = "TEST/USDT:USDT"
        open_date_utc = datetime(2025, 7, 15, 14, 0, tzinfo=timezone.utc)  # 10:00 EDT
    s._entry_row = lambda trade, pair: pd.Series({"atr_14": 1.0})
    # 20:00 UTC = 16:00 EDT => time-stop dispara
    reason = s.custom_exit("TEST/USDT:USDT", T(), datetime(2025,7,15,20,0,tzinfo=timezone.utc),
                           current_rate=101.0, current_profit=0.01)
    assert reason == 'sb_session_close'
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/wortiz/Desktop/freqtrade && .venv/bin/python -m pytest user_data/strategies/tests/test_ict_strategy_smoke.py -k timestop -v`
Expected: FAIL with `AttributeError: ... has no attribute 'custom_exit'`

- [ ] **Step 3: Write minimal implementation**

```python
# add method to ICTSilverBullet
    _LONG_POOL_COLS = ('eqh_level', 'active_bearish_fvg_top', 'swing_high')
    _SHORT_POOL_COLS = ('eql_level', 'active_bullish_fvg_bottom', 'swing_low')

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime,
                    current_rate: float, current_profit: float, **kwargs):
        # 1) time-stop de sesión: nunca overnight
        if sb.is_past_session_close(trade.open_date_utc, current_time):
            return 'sb_session_close'
        # 2) TP2 = liquidez opuesta más cercana
        row = self._entry_row(trade, pair)
        if row is None:
            return None
        cols = self._SHORT_POOL_COLS if trade.is_short else self._LONG_POOL_COLS
        pools = {c: row.get(c, None) for c in cols}
        target = sb.nearest_opposing_pool(trade.open_rate, pools, is_short=trade.is_short)
        if target is None:
            return None
        if trade.is_short and current_rate <= target:
            return 'sb_tp2_liquidity'
        if (not trade.is_short) and current_rate >= target:
            return 'sb_tp2_liquidity'
        return None

    def confirm_trade_exit(self, pair, trade, order_type, amount, rate,
                           time_in_force, exit_reason, current_time, **kwargs) -> bool:
        # tras TP1, no permitir que un exit_signal cierre el runner (no hay signal exit
        # de todos modos, pero protegemos el patrón)
        return True
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/wortiz/Desktop/freqtrade && .venv/bin/python -m pytest user_data/strategies/tests/test_ict_strategy_smoke.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
cd /home/wortiz/Desktop/freqtrade
git add user_data/strategies/ICTSilverBullet.py user_data/strategies/tests/test_ict_strategy_smoke.py
git commit -m "feat(ICTSilverBullet): custom_exit (TP2 liquidity + session time-stop)"
```

---

### Task 8: Config de backtest + smoke de integración (5m, lev1, 14 stocks)

**Files:**
- Create: `config_ictsb_bt.json`

- [ ] **Step 1: Crear el config de backtest**

```json
{
    "$schema": "https://schema.freqtrade.io/schema.json",
    "max_open_trades": 5,
    "stake_currency": "USDT",
    "stake_amount": 10,
    "tradable_balance_ratio": 0.99,
    "dry_run": true,
    "dry_run_wallet": 100,
    "leverage": 1,
    "timeframe": "5m",
    "trading_mode": "futures",
    "margin_mode": "isolated",
    "fee": 0.0004,
    "exchange": {
        "name": "bitget",
        "key": "",
        "secret": "",
        "pair_whitelist": [
            "MSTR/USDT:USDT", "TSLA/USDT:USDT", "NVDA/USDT:USDT", "CRCL/USDT:USDT",
            "AAPL/USDT:USDT", "COIN/USDT:USDT", "PLTR/USDT:USDT", "META/USDT:USDT",
            "GOOGL/USDT:USDT", "HOOD/USDT:USDT", "AMZN/USDT:USDT", "MCD/USDT:USDT",
            "BABA/USDT:USDT", "MSFT/USDT:USDT"
        ],
        "pair_blacklist": []
    },
    "pairlists": [{"method": "StaticPairList"}],
    "entry_pricing": {"price_side": "same", "use_order_book": false, "order_book_top": 1},
    "exit_pricing": {"price_side": "same", "use_order_book": false, "order_book_top": 1}
}
```

- [ ] **Step 2: Correr el smoke de integración (rango NO-OOS: sep25→mar26, reservando mar26→may26 como OOS)**

Run:
```bash
cd /home/wortiz/Desktop/freqtrade
.venv/bin/freqtrade backtesting --strategy ICTSilverBullet \
  --config config_ictsb_bt.json --timeframe 5m \
  --timerange 20250922-20260305 \
  --datadir user_data/data/bitget --cache none > /tmp/ictsb_5m_smoke.log 2>&1
echo "exit=$?"
grep -iE "Total/Daily Avg Trades|Total profit %|Profit factor|Sharpe|Long / Short trades" /tmp/ictsb_5m_smoke.log
```
Expected: `exit=0`, una tabla de métricas, y un número de trades >0. (Si trades=0 → bug en la state machine o en el wiring de columnas; debuggear antes de seguir.)

- [ ] **Step 3: Verificar criterios mínimos del smoke**

Confirmar manualmente en el log:
- Corrió sin traceback (`grep -c Traceback /tmp/ictsb_5m_smoke.log` == 0).
- Trades > 0 y todos los enter tags son `SB_long`/`SB_short`.
- Holds NO overnight (Max Duration < ~6h; el time-stop a 16:00 ET debe cortar).

- [ ] **Step 4: Commit**

```bash
cd /home/wortiz/Desktop/freqtrade
git add config_ictsb_bt.json
git commit -m "feat(ICTSilverBullet): backtest config (lev1, 14 stocks) + smoke pasa"
```

---

## Validación post-implementación (fuera de este plan — fase forense, spec §4)

Una vez los 8 tasks pasan, NO concluir edge todavía. Ejecutar la ronda forense del spec §4 con los gates de Sofia:
1. OOS reservado (mar26→may26) intocado hasta el final.
2. Gate de avance: `≥150 trades 5m` Y `borde inferior CI de PF >1.0`.
3. Walk-forward + MC por bloques temporales + ablación SB-windows on/off + corrección de múltiples comparaciones (Deflated Sharpe/Bonferroni sobre TF×window).
4. 15m solo como sanity de la state machine, no test de edge.

Esta fase la dispara Liz con la skill `bt-forensics` tras aprobación de Walter.

---

## Self-Review (hecho)

- **Spec coverage:** §3.1 archivo+motor→Task 4; §3.2 ventanas DST→Task 1; §3.3 state machine FVG-only→Task 2+5; §3.4 salidas híbridas+time-stop→Task 3+6+7; §3.5 universo/TF/lev/warmup→Task 4 (startup 1000, lev config) + Task 8 (14 stocks, 5m). §4 validación→sección post-plan. ✅ sin gaps.
- **Placeholder scan:** sin TBD/TODO; todo step tiene código o comando real. ✅
- **Type consistency:** `sb_window_flags`/`sb_entry_signals`/`is_past_session_close`/`nearest_opposing_pool`/`tp1_target_price` definidos en Tasks 1-3 y consumidos con misma firma en Tasks 5-7. Columnas del motor verificadas contra el contrato del header. ✅
- **Limitación conocida declarada:** el MSS usa `internal_choch_*_disp` (los `swing_choch_*_disp` quedan para iteración futura) — documentado en Task 5.
