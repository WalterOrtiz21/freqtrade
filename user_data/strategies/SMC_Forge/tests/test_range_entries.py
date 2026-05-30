import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from range_entries import (   # noqa: E402
    fade_long, fade_short, breakout_long, breakout_short, rr_long_fade,
)


def _df(rows):
    return pd.DataFrame(rows)


def test_fade_long_fires_on_sweep_not_break():
    # Row A: sweep del engine (internal_sweep_bullish=1) en mitad inferior -> fade dispara.
    # Row B: break (close debajo del piso, sweep=0 porque el engine no marca sweep en un break)
    #        -> fade NO dispara.
    df = _df([
        {'open': 101, 'high': 101.5, 'low': 99.5, 'close': 100.4,
         'range_active': 1, 'range_top': 104, 'range_bottom': 100, 'range_mid': 102,
         'internal_sweep_bullish': 1, 'swing_sweep_bullish': 0, 'atr_14': 1.0},
        {'open': 100.4, 'high': 100.6, 'low': 98.0, 'close': 99.0,
         'range_active': 1, 'range_top': 104, 'range_bottom': 100, 'range_mid': 102,
         'internal_sweep_bullish': 0, 'swing_sweep_bullish': 0, 'atr_14': 1.0},
    ])
    sig = fade_long(df)
    assert bool(sig.iloc[0]) is True, "sweep del engine en discount debe disparar fade long"
    assert bool(sig.iloc[1]) is False, "break (sweep=0) NO debe disparar fade long"


def test_fade_long_requires_lower_half():
    # sweep=1 pero close en la mitad SUPERIOR del rango -> NO es fade long (no es discount).
    df = _df([{'open': 103, 'high': 103.5, 'low': 102.5, 'close': 103.0,
               'range_active': 1, 'range_top': 104, 'range_bottom': 100, 'range_mid': 102,
               'internal_sweep_bullish': 1, 'swing_sweep_bullish': 0, 'atr_14': 1.0}])
    assert bool(fade_long(df).iloc[0]) is False


def test_fade_long_requires_active_range():
    df = _df([{'open': 101, 'high': 101.5, 'low': 99.5, 'close': 100.4,
               'range_active': 0, 'range_top': np.nan, 'range_bottom': np.nan,
               'range_mid': np.nan, 'internal_sweep_bullish': 1, 'swing_sweep_bullish': 0,
               'atr_14': 1.0}])
    assert bool(fade_long(df).iloc[0]) is False


def test_breakout_long_fires_after_active_range_close_above_top():
    df = _df([
        {'open': 103, 'high': 104, 'low': 102.5, 'close': 103.5,
         'range_active': 1, 'range_top': 104, 'range_bottom': 100, 'range_mid': 102, 'atr_14': 1.0},
        {'open': 103.5, 'high': 105.5, 'low': 103.4, 'close': 105.0,
         'range_active': 0, 'range_top': np.nan, 'range_bottom': np.nan, 'range_mid': np.nan, 'atr_14': 1.0},
    ])
    sig = breakout_long(df)
    assert bool(sig.iloc[1]) is True, "close>prev_top con rango activo previo = breakout long"


def test_rr_long_fade_gate():
    # entry=100.4, low=99.5, atr=1.0, buffer=0.3 -> SL=99.2 ; risk=1.2
    # range_mid=102 -> reward=1.6 ; rr=1.333
    df = _df([{'open': 101, 'high': 101.5, 'low': 99.5, 'close': 100.4,
               'range_active': 1, 'range_top': 104, 'range_bottom': 100, 'range_mid': 102, 'atr_14': 1.0}])
    rr = rr_long_fade(df, atr_col='atr_14', sl_buffer=0.3).iloc[0]
    assert rr == pytest.approx(1.6 / 1.2, abs=1e-6)
