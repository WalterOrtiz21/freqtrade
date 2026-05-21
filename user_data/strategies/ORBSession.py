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
        df = dataframe
        in_session = (df['date'].dt.hour >= self.RANGE_END_HOUR) & \
                     (df['date'].dt.hour < self.SESSION_END_HOUR)
        not_skipped = ~df['orb_skipped'].fillna(False)
        has_range = df['orb_range_high'].notna() & df['orb_range_low'].notna()

        long_breakout = df['close'] > df['orb_range_high']
        short_breakout = df['close'] < df['orb_range_low']

        any_breakout = (long_breakout | short_breakout) & in_session & not_skipped & has_range

        if not self.ALLOW_REENTRY:
            session_date = df['date'].dt.date
            cum_signals = any_breakout.groupby(session_date).cumsum()
            first_signal_only = any_breakout & (cum_signals == 1)
        else:
            first_signal_only = any_breakout

        df['enter_long'] = (first_signal_only & long_breakout).astype(int)
        df['enter_short'] = (first_signal_only & short_breakout).astype(int)
        return df

    # after_fill is absorbed by **kwargs intentionally — range stop is fixed at
    # entry, not recomputed on fill events. Don't add after_fill explicitly: that
    # would flip freqtrade's _ft_stop_uses_after_fill flag and change framework behavior.
    def custom_stoploss(self, pair, trade, current_time, current_rate, current_profit, **kwargs):
        """Stop at opposite end of opening range.

        Long: stop = range_low  -> returns (range_low / open_rate) - 1   (negative)
        Short: stop = range_high -> returns (open_rate - range_high) / open_rate  (negative)

        If range data is missing (defensive), returns the class-level stoploss
        (-0.99 placeholder; should never trigger in practice).
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        if dataframe is None or dataframe.empty:
            return self.stoploss

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
        return (range_low / trade.open_rate) - 1

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe
        # The 20:45 15m candle closes at 21:00 UTC -> emit exit signal there.
        is_last_session_candle = (
            (df['date'].dt.hour == (self.SESSION_END_HOUR - 1)) &
            (df['date'].dt.minute == 45)
        )
        df['exit_long'] = is_last_session_candle.astype(int)
        df['exit_short'] = is_last_session_candle.astype(int)
        return df
