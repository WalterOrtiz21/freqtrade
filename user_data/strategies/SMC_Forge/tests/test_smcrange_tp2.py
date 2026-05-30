import sys
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1].parent))  # -> strategies/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))         # -> SMC_Forge/

from SMCRange import SMCRange  # noqa: E402


class _Trade:
    def __init__(self, is_short, open_rate, tp1_taken=False):
        self.is_short = is_short
        self.open_rate = open_rate
        self.leverage = 1.0
        self.pair = 'TEST/USDT:USDT'
        self.amount = 1.0
        self.enter_tag = 'FADE_L'
        self._cd = {'tp1_taken': tp1_taken}
        self.open_date_utc = datetime(2024, 1, 2, tzinfo=timezone.utc)

    def get_custom_data(self, k, default=None):
        return self._cd.get(k, default)

    def set_custom_data(self, k, v):
        self._cd[k] = v


def _strat_with_entry_row(row: dict):
    s = SMCRange.__new__(SMCRange)
    s._entry_row = lambda trade, pair: pd.Series(row)
    return s


def test_tp2_fires_for_long_fade_at_range_top():
    """TP2 (borde opuesto = range_top) DEBE devolver tag cuando current_rate lo alcanza.
    Este es el bug que mató al reverse (tp2 era código muerto)."""
    s = _strat_with_entry_row({
        'range_top': 104.0, 'range_bottom': 100.0, 'range_mid': 102.0,
        'low': 99.5, 'high': 100.6, 'atr_14': 1.0,
    })
    trade = _Trade(is_short=False, open_rate=100.4, tp1_taken=True)
    now = datetime(2024, 1, 3, tzinfo=timezone.utc)
    # current_rate por debajo de TP2 -> None
    assert s.custom_exit('TEST/USDT:USDT', trade, now, 103.0, 0.026) is None
    # current_rate alcanza range_top -> tag de TP2
    tag = s.custom_exit('TEST/USDT:USDT', trade, now, 104.1, 0.037)
    assert tag is not None and 'tp2' in tag.lower(), f"TP2 no disparó: {tag!r}"


def test_tp2_fires_for_short_fade_at_range_bottom():
    s = _strat_with_entry_row({
        'range_top': 104.0, 'range_bottom': 100.0, 'range_mid': 102.0,
        'low': 103.4, 'high': 104.5, 'atr_14': 1.0,
    })
    trade = _Trade(is_short=True, open_rate=103.6, tp1_taken=True)
    trade.enter_tag = 'FADE_S'
    now = datetime(2024, 1, 3, tzinfo=timezone.utc)
    assert s.custom_exit('TEST/USDT:USDT', trade, now, 101.0, 0.02) is None
    tag = s.custom_exit('TEST/USDT:USDT', trade, now, 99.9, 0.03)
    assert tag is not None and 'tp2' in tag.lower(), f"TP2 short no disparó: {tag!r}"
