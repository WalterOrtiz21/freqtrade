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
