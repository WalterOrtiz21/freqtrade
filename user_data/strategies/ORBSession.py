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
        df = dataframe.copy()
        session_date = df['date'].dt.date

        # 1. Mark candles inside the range-forming hour (13:00-13:45 UTC)
        in_range_window = (df['date'].dt.hour == self.SESSION_START_HOUR)

        # 2. cummax/cummin within each UTC day, only over range candles
        range_high_seed = df['high'].where(in_range_window)
        range_low_seed = df['low'].where(in_range_window)
        df['orb_range_high'] = range_high_seed.groupby(session_date).cummax()
        df['orb_range_low'] = range_low_seed.groupby(session_date).cummin()

        # 3. Keep only the final (13:45) accumulated value; wipe all other pre-14:00 values.
        #    Then ffill within each UTC day so the range carries forward from 14:00 onwards.
        is_range_end_candle = (
            (df['date'].dt.hour == self.SESSION_START_HOUR) &
            (df['date'].dt.minute == 45)
        )
        post_range = df['date'].dt.hour >= self.RANGE_END_HOUR
        keep = post_range | is_range_end_candle
        df.loc[~keep, 'orb_range_high'] = pd.NA
        df.loc[~keep, 'orb_range_low'] = pd.NA
        df['orb_range_high'] = df.groupby(session_date)['orb_range_high'].ffill()
        df['orb_range_low'] = df.groupby(session_date)['orb_range_low'].ffill()
        # Now hide the 13:45 seed row itself (still pre-14:00 from strategy's perspective)
        df.loc[is_range_end_candle & ~post_range, 'orb_range_high'] = pd.NA
        df.loc[is_range_end_candle & ~post_range, 'orb_range_low'] = pd.NA

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
