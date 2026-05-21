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
