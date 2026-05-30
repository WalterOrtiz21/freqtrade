"""
SMC_Forge — Range Detector
==========================
Detecta un rango SMC activo por barra a partir de columnas del engine ya
presentes en df. Bordes = clusters EQH/EQL del engine (eqh_level=top,
eql_level=bottom; persisten bar-by-bar y se invalidan con close-break,
forge_levels.py:191-196). Régimen ranging = SIN swing BoS en las últimas
`bos_lookback` barras (swing_trend==0 NO sirve: el engine nunca lo resetea a
0 tras el primer break — forge_engine.py:247-270).

Anti-lookahead: todo input es señal de barra cerrada del engine; los rolling
son backward-only (default pandas). No se shiftan datos futuros.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_REQUIRED = ('close', 'eqh_level', 'eql_level', 'eqh_count', 'eql_count',
             'swing_bos_bullish', 'swing_bos_bearish')


def annotate_range(
    df: pd.DataFrame,
    bos_lookback: int = 20,
    width_min_pct: float = 0.015,
    width_max_pct: float = 0.30,
    min_touches: int = 2,
    containment_window: int = 20,
    containment_min: float = 0.70,
) -> pd.DataFrame:
    missing = [c for c in _REQUIRED if c not in df.columns]
    if missing:
        raise KeyError(f"annotate_range: faltan columnas {missing}")

    close = df['close']
    top = df['eqh_level']
    bottom = df['eql_level']
    eqh_count = df['eqh_count'].fillna(0)
    eql_count = df['eql_count'].fillna(0)

    # 1. Régimen: sin BoS de continuación en últimas L barras (backward rolling).
    bos_bull = df['swing_bos_bullish'].fillna(0.0)
    bos_bear = df['swing_bos_bearish'].fillna(0.0)
    no_recent_bos = (
        (bos_bull.rolling(bos_lookback, min_periods=1).max() == 0)
        & (bos_bear.rolling(bos_lookback, min_periods=1).max() == 0)
    )

    # 2. Ambos bordes activos, top>bottom, suficientes toques.
    both_present = top.notna() & bottom.notna() & (top > bottom)
    enough_touches = (eqh_count >= min_touches) & (eql_count >= min_touches)

    # 3. Ancho sano.
    mid = (top + bottom) / 2.0
    width_pct = ((top - bottom) / mid.replace(0.0, np.nan)).abs()
    width_ok = (width_pct >= width_min_pct) & (width_pct <= width_max_pct)

    # 4. Contención: fracción de closes recientes dentro de [bottom, top].
    inside = (close >= bottom) & (close <= top)
    contain_frac = inside.fillna(False).rolling(
        containment_window, min_periods=1).mean()
    contained = contain_frac >= containment_min

    range_active = (
        no_recent_bos & both_present & enough_touches & width_ok & contained
    ).fillna(False)

    touch_score = ((eqh_count + eql_count) / (4 * min_touches)).clip(0.0, 1.0)
    quality = (0.5 * touch_score + 0.5 * contain_frac.clip(0.0, 1.0)).clip(0.0, 1.0)

    out = pd.DataFrame(index=df.index)
    out['range_active'] = range_active.astype(int)
    out['range_top'] = top.where(range_active)
    out['range_bottom'] = bottom.where(range_active)
    out['range_mid'] = mid.where(range_active)
    out['range_width_pct'] = width_pct.where(range_active)
    out['range_quality'] = quality.where(range_active).fillna(0.0)
    return out
