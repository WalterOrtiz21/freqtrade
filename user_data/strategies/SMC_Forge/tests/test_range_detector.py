import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forge_engine import SMCEngine            # noqa: E402
from forge_levels import annotate_eqh_eql     # noqa: E402
from range_detector import annotate_range     # noqa: E402


def _synthetic_range_df(n=400, seed=7):
    """Oscilación dentro de ~[98,102] con ruido — debería producir EQH/EQL + rango."""
    rng = np.random.RandomState(seed)
    mid = 100.0
    closes = mid + 2.0 * np.sin(np.linspace(0, 30 * np.pi, n)) + rng.normal(0, 0.3, n)
    highs = closes + np.abs(rng.normal(0.4, 0.2, n))
    lows = closes - np.abs(rng.normal(0.4, 0.2, n))
    opens = np.concatenate([[closes[0]], closes[:-1]])
    df = pd.DataFrame({'open': opens, 'high': highs, 'low': lows,
                       'close': closes, 'volume': rng.uniform(50, 150, n)})
    df['date'] = pd.date_range('2024-01-01', periods=n, freq='1h')
    return df


def _build_inputs(df):
    sig = SMCEngine(df, internal_length=5, swing_length=50).get_signals()
    lvl = annotate_eqh_eql(df, atr_period=14, fractal_n=5, tolerance_atr=0.10)
    merged = df.copy()
    for c in sig.columns:
        merged[c] = sig[c].values
    for c in lvl.columns:
        if c not in merged.columns:
            merged[c] = lvl[c].values
    return merged


def test_no_lookahead_range_active():
    """range_active[i] no puede cambiar si trunco el df en i+1 (sin datos futuros)."""
    df = _synthetic_range_df()
    full = _build_inputs(df)
    out_full = annotate_range(full)
    k = 320
    trunc = _build_inputs(df.iloc[:k].reset_index(drop=True))
    out_trunc = annotate_range(trunc)
    a = out_full['range_active'].values[:k]
    b = out_trunc['range_active'].values[:k]
    # Permitir divergencia solo en zona de warmup del swing (primeras ~60 barras).
    mismatches = np.where(a[60:] != b[60:])[0]
    assert len(mismatches) == 0, f"lookahead en barras {mismatches + 60}"


def test_columns_present():
    df = _synthetic_range_df()
    out = annotate_range(_build_inputs(df))
    for c in ['range_active', 'range_top', 'range_bottom', 'range_mid',
              'range_width_pct', 'range_quality']:
        assert c in out.columns
    assert out['range_active'].isin([0, 1]).all()
