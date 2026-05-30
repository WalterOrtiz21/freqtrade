"""
SMC_Forge — Range Entries
=========================
Señales puras fade/breakout, TF-agnósticas: consumen las columnas range_*
canónicas (ya seleccionadas por la estrategia) + OHLC + CHoCH del engine.
Cada función devuelve un pd.Series booleano alineado al índice.
"""
from __future__ import annotations

import pandas as pd


def fade_long(df: pd.DataFrame) -> pd.Series:
    """Sweep del piso: mecha bajo range_bottom, close de vuelta adentro."""
    ra = df['range_active'] == 1
    return (ra & (df['low'] < df['range_bottom'])
            & (df['close'] >= df['range_bottom'])).fillna(False)


def fade_short(df: pd.DataFrame) -> pd.Series:
    """Sweep del techo: mecha sobre range_top, close de vuelta adentro."""
    ra = df['range_active'] == 1
    return (ra & (df['high'] > df['range_top'])
            & (df['close'] <= df['range_top'])).fillna(False)


def breakout_long(df: pd.DataFrame) -> pd.Series:
    """Body-close sobre el techo del rango activo de la barra previa."""
    prev_active = df['range_active'].shift(1) == 1
    prev_top = df['range_top'].shift(1)
    return (prev_active & prev_top.notna() & (df['close'] > prev_top)).fillna(False)


def breakout_short(df: pd.DataFrame) -> pd.Series:
    prev_active = df['range_active'].shift(1) == 1
    prev_bot = df['range_bottom'].shift(1)
    return (prev_active & prev_bot.notna() & (df['close'] < prev_bot)).fillna(False)


def reclaim_long(df: pd.DataFrame, lookback: int = 3) -> pd.Series:
    """Confirmación opcional: CHoCH interno alcista en las últimas `lookback` barras."""
    ev = (df.get('internal_choch_bullish', pd.Series(0, index=df.index)).fillna(0) == 1)
    return ev.rolling(lookback, min_periods=1).max().fillna(0).astype(bool)


def reclaim_short(df: pd.DataFrame, lookback: int = 3) -> pd.Series:
    ev = (df.get('internal_choch_bearish', pd.Series(0, index=df.index)).fillna(0) == 1)
    return ev.rolling(lookback, min_periods=1).max().fillna(0).astype(bool)


def rr_long_fade(df: pd.DataFrame, atr_col: str, sl_buffer: float) -> pd.Series:
    """R:R de fade long: entry≈close, SL=low-atr*buf, TP1=range_mid."""
    entry = df['close']
    sl = df['low'] - df[atr_col] * sl_buffer
    risk = (entry - sl)
    reward = (df['range_mid'] - entry)
    return (reward / risk.where(risk > 0)).fillna(-1.0)


def rr_short_fade(df: pd.DataFrame, atr_col: str, sl_buffer: float) -> pd.Series:
    entry = df['close']
    sl = df['high'] + df[atr_col] * sl_buffer
    risk = (sl - entry)
    reward = (entry - df['range_mid'])
    return (reward / risk.where(risk > 0)).fillna(-1.0)
