"""
XSMomentum — Cross-Sectional Momentum Rotation (Long/Short)

Edge thesis:
  Rank the full universe by risk-adjusted momentum (log-return / realized vol,
  same window L). Go long the top-N, short the bottom-N. The bet is the spread
  between strong and weak — not market direction. With balanced legs the book
  is quasi market-neutral.

Signal (anti-lookahead by design):
  - score = log_return(L) / realized_vol(L) computed on CLOSED candles (shift(1))
  - rank = ordinal percentile rank computed cross-sectionally at EACH timestamp
    by combining this pair's own score series with the score series of all other
    pairs fetched via dp.get_pair_dataframe (also shifted)
  - score matrix is cached per (candle_ts, L_candles, universe_size) → only
    computed once per bar for the first pair; all subsequent pairs read from cache

Portfolio construction:
  - long:  rank percentile >= 1 - N/U (top-N),   exit when rank pct < 1 - 2N/U
  - short: rank percentile <= N/U (bottom-N),     exit when rank pct > 2N/U
  - max_open_trades = 2N
  - equal stake per slot
  - backstop stop: -12% price via custom_stoploss + stoploss_from_absolute
    (leverage-independent, house pattern)
  - no trailing, no fixed TP — rank exit is the engine

Key implementation detail:
  In freqtrade backtest, populate_indicators is called ONCE per pair with the
  FULL historical dataframe. dp.get_pair_dataframe returns the FULL history for
  other pairs too (not sliced per-bar in backtest). This means we can build a
  proper TIME-SERIES of cross-sectional ranks by:
    1. For each pair X, compute its score time series (NxT matrix of scores
       across the full history)
    2. For each timestamp T, rank pair X against all other pairs' score at T
  This requires that when we process pair X, we can read the score series for
  ALL other pairs. The cache stores the full score matrix (pair -> score series)
  keyed by (L_candles, universe_size, universe_tuple) so it is computed only
  once across the full populate_indicators loop.

Parameters L and N are CategoricalParameter with optimize=False.
The experiment matrix (Phase 2) is run by editing defaults or a params JSON,
NOT via hyperopt. This preserves clean out-of-sample separation.

Timeframe: 4h. Leverage: 1x for all validation runs.
Framework: freqtrade INTERFACE_VERSION=3, Bitget USDT-perp futures.
"""

import logging
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame

from freqtrade.enums import RunMode
from freqtrade.strategy import (
    IStrategy, CategoricalParameter,
    stoploss_from_absolute,
)
from freqtrade.persistence import Trade

logger = logging.getLogger(__name__)


class XSMomentum(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = '4h'
    can_short = True
    process_only_new_candles = True
    use_exit_signal = True
    use_custom_stoploss = True
    position_adjustment_enable = False

    # Startup buffer: L_max=28d × 6 candles/day = 168 + 30 margin = ~200.
    # Bitget caps startup_candle_count at 999 (4h → 166 days); 200 fits.
    startup_candle_count: int = 200

    # Hard backstop stoploss (freqtrade field) — the real stop is the -12%
    # price stop in custom_stoploss. This is the emergency safety net only.
    stoploss = -0.25

    # ROI disabled: rank exit and backstop stop are the only exits.
    minimal_roi = {"0": 10.0}

    # =========================================================================
    # PARAMETERS (optimize=False — matrix run by editing defaults, not hyperopt)
    # =========================================================================
    # Lookback in days; converted to candles as L_days * 6 (4h candles per day).
    # Phase 2 matrix: 7, 14, 28.
    lookback_days = CategoricalParameter([7, 14, 28], default=14, space='buy', optimize=False)

    # Book size per leg (long N, short N). Phase 2 matrix: 3, 5.
    book_n = CategoricalParameter([3, 5], default=5, space='buy', optimize=False)

    # Backstop stop distance from entry price (leverage-independent).
    stop_pct = CategoricalParameter([0.10, 0.12, 0.15], default=0.12, space='sell', optimize=False)

    # =========================================================================
    # SCORE MATRIX CACHE
    # =========================================================================
    # Full score matrix: Dict[pair -> pd.Series of scores indexed by timestamp]
    # Keyed by (L_candles, universe_tuple) in backtest (latest_ts=None) —
    # computed once for the full run since populate_indicators is called once
    # per pair.
    # In live/dry the key also includes latest_ts (the calling pair's last
    # candle timestamp).  The first pair of each new 4h bar recomputes the
    # matrix; subsequent pairs in the same cycle hit the cache.  If pairs have
    # slightly different last-candle timestamps the key changes and recomputes —
    # acceptable; correctness is the priority.
    _score_matrix_cache: Dict[str, pd.Series] = {}
    _score_matrix_cache_key: Optional[Tuple] = None

    # =========================================================================
    # INFORMATIVE PAIRS
    # =========================================================================
    def informative_pairs(self):
        # No additional informative pairs — all ranking is done on the base 4h
        # data fetched via dp.get_pair_dataframe inside populate_indicators.
        return []

    # =========================================================================
    # SCORE SERIES HELPER
    # =========================================================================
    @staticmethod
    def _compute_score_series(close: pd.Series, L: int) -> pd.Series:
        """
        Compute risk-adjusted momentum score time series for a single pair.

        score[t] = log_return(L)[t] / realized_vol(L)[t]

        Anti-lookahead: both log_return and vol are computed on the shift(1)
        series — only CLOSED candles feed into the score. NaN where insufficient
        history.
        """
        shifted = close.shift(1)  # use only closed candles
        log_ret = np.log(shifted / shifted.shift(L))
        log_diff = np.log(shifted / shifted.shift(1))
        realized_vol = log_diff.rolling(L).std()

        score = log_ret / realized_vol
        # Replace inf/-inf (zero-vol edge cases) with NaN
        score = score.replace([np.inf, -np.inf], np.nan)
        return score

    # =========================================================================
    # SCORE MATRIX (computed once for the full universe)
    # =========================================================================
    def _get_score_matrix(
        self, L_candles: int, latest_ts: Optional[pd.Timestamp] = None
    ) -> Dict[str, pd.Series]:
        """
        Returns the full score matrix {pair: score_series} for the universe.

        In backtest/hyperopt, latest_ts is None and the cache key is
        (L_candles, universe_tuple).  populate_indicators is called once per
        pair with the full history, so the matrix is built once and reused for
        the entire run — O(1) for every pair after the first.

        In live/dry, latest_ts is the calling pair's last candle timestamp.
        The cache key becomes (L_candles, universe_tuple, latest_ts), so the
        matrix is recomputed the first time any pair is processed on a new
        candle.  Subsequent pairs on the same candle cycle hit the cache.
        This prevents the live freeze bug where stale score series caused
        KeyError lookups → rank_pct=0.5 / xs_score=NaN for every row.
        """
        if not self.dp:
            return {}

        universe = tuple(sorted(self.dp.current_whitelist()))
        cache_key = (L_candles, universe, latest_ts)

        if self._score_matrix_cache_key == cache_key:
            return self._score_matrix_cache

        matrix: Dict[str, pd.Series] = {}
        for pair in universe:
            try:
                df = self.dp.get_pair_dataframe(pair=pair, timeframe=self.timeframe)
                if df is None or len(df) < L_candles + 2:
                    continue
                score_series = self._compute_score_series(df['close'], L_candles)
                # Use timezone-aware index to match freqtrade's UTC timestamps.
                # df['date'] is datetime64[ns, UTC]; .dt.tz_localize/'values' strips tz.
                # Using pd.DatetimeIndex directly from the Series preserves tz.
                score_series.index = pd.DatetimeIndex(df['date'])
                matrix[pair] = score_series
            except Exception as exc:
                logger.warning(f"XSMomentum: score series failed for {pair}: {exc}")

        self._score_matrix_cache = matrix
        self._score_matrix_cache_key = cache_key
        return matrix

    # =========================================================================
    # RANK TIME SERIES FOR ONE PAIR
    # =========================================================================
    def _compute_rank_series(
        self,
        pair: str,
        score_matrix: Dict[str, pd.Series],
        pair_dates: pd.Index,
    ) -> Tuple[pd.Series, pd.Series]:
        """
        For each timestamp in pair_dates, compute:
          - rank_pct: this pair's percentile rank (0=weakest, 1=strongest) among
            all pairs in score_matrix that have a valid (non-NaN) score at that T.

        Returns (rank_pct_series, universe_size_series) aligned to pair_dates.
        """
        if pair not in score_matrix:
            # Not enough history; return neutral rank
            neutral = pd.Series(0.5, index=pair_dates)
            zero = pd.Series(0, index=pair_dates)
            return neutral, zero

        own_score = score_matrix[pair]

        rank_pcts = []
        universe_sizes = []

        for ts in pair_dates:
            # Get all pairs' scores at this timestamp
            scores_at_t = {}
            for p, s in score_matrix.items():
                try:
                    val = s.loc[ts]
                    if not np.isnan(val):
                        scores_at_t[p] = val
                except KeyError:
                    pass

            if len(scores_at_t) < 2:
                rank_pcts.append(0.5)
                universe_sizes.append(len(scores_at_t))
                continue

            own_val = scores_at_t.get(pair, np.nan)
            if np.isnan(own_val) if isinstance(own_val, float) else False:
                rank_pcts.append(0.5)
                universe_sizes.append(len(scores_at_t))
                continue

            # Percentile rank: fraction of universe that this pair beats
            all_vals = list(scores_at_t.values())
            n_beaten = sum(1 for v in all_vals if v < own_val)
            n_tied = sum(1 for v in all_vals if v == own_val)
            # Average rank method (consistent with pd.Series.rank(pct=True, method='average'))
            pct = (n_beaten + 0.5 * n_tied) / len(all_vals)
            rank_pcts.append(pct)
            universe_sizes.append(len(scores_at_t))

        return (
            pd.Series(rank_pcts, index=pair_dates),
            pd.Series(universe_sizes, index=pair_dates),
        )

    # =========================================================================
    # INDICATOR COMPUTATION
    # =========================================================================
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata['pair']
        L_candles = int(self.lookback_days.value) * 6  # 4h candles per day
        N = int(self.book_n.value)

        # In live/dry, pass the last candle timestamp so the cache invalidates
        # each new bar cycle.  In backtest/hyperopt use None (single compute
        # for the full historical run — preserves backtest equivalence).
        is_live = self.dp and self.dp.runmode.value in ('live', 'dry_run')
        latest_ts = pd.Timestamp(dataframe['date'].iloc[-1]) if is_live else None

        # Build/retrieve the full cross-sectional score matrix (O(1) after first pair)
        score_matrix = self._get_score_matrix(L_candles, latest_ts=latest_ts)

        # Compute rank time series for this pair
        # Use timezone-aware DatetimeIndex to match the score_matrix index.
        pair_dates = pd.DatetimeIndex(dataframe['date'])
        rank_pct_series, universe_size_series = self._compute_rank_series(
            pair, score_matrix, pair_dates
        )

        dataframe['xs_rank_pct'] = rank_pct_series.values
        dataframe['xs_universe_size'] = universe_size_series.values
        dataframe['xs_book_n'] = N

        # Score for diagnostics
        if pair in score_matrix:
            own_score = score_matrix[pair]
            # Align to dataframe dates (both are tz-aware UTC)
            try:
                dataframe['xs_score'] = own_score.reindex(
                    pd.DatetimeIndex(dataframe['date'])
                ).values
            except Exception:
                dataframe['xs_score'] = np.nan
        else:
            dataframe['xs_score'] = np.nan

        # Entry/exit zone flags based on rank percentile
        # top-N: rank_pct >= 1 - N/U (but U varies; use robust threshold based on N/U_approx)
        # We use N/universe_size as the threshold, clipped to sane bounds.
        u = dataframe['xs_universe_size'].replace(0, 40)
        n_over_u = N / u

        dataframe['xs_long_zone'] = dataframe['xs_rank_pct'] >= (1.0 - n_over_u)
        dataframe['xs_long_hold'] = dataframe['xs_rank_pct'] >= (1.0 - 2.0 * n_over_u)
        dataframe['xs_short_zone'] = dataframe['xs_rank_pct'] <= n_over_u
        dataframe['xs_short_hold'] = dataframe['xs_rank_pct'] <= (2.0 * n_over_u)

        return dataframe

    # =========================================================================
    # ENTRY SIGNALS
    # =========================================================================
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        dataframe['enter_tag'] = ''

        # Only enter on bars with a valid score and non-trivial universe
        has_score = dataframe['xs_score'].notna() & (dataframe['xs_universe_size'] >= 3)

        cond_long = has_score & dataframe['xs_long_zone']
        cond_short = has_score & dataframe['xs_short_zone']

        dataframe.loc[cond_long, ['enter_long', 'enter_tag']] = (1, 'xs_long')
        dataframe.loc[cond_short, ['enter_short', 'enter_tag']] = (1, 'xs_short')

        return dataframe

    # =========================================================================
    # EXIT SIGNALS (hysteresis rank exits — the engine)
    # =========================================================================
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0
        dataframe['exit_tag'] = ''

        # Gate: only fire exit signals when rank data is reliable.
        # Without this guard, stale/missing scores (NaN xs_score or empty
        # universe) would trigger mass exits on every row.
        has_valid_rank = (
            dataframe['xs_score'].notna()
            & (dataframe['xs_universe_size'] >= 3)
        )

        # Exit long when pair falls out of top-2N (hysteresis zone)
        exit_long = has_valid_rank & ~dataframe['xs_long_hold']

        # Exit short when pair rises above bottom-2N (hysteresis zone, mirror)
        exit_short = has_valid_rank & ~dataframe['xs_short_hold']

        # Set FLAGS for both exits (unchanged from pre-fix: both can be 1 on the
        # same row when the pair is in neither hold zone, i.e. the middle zone).
        dataframe.loc[exit_long, 'exit_long'] = 1
        dataframe.loc[exit_short, 'exit_short'] = 1

        # Set TAGs without overwriting: long tag is written first, short tag is
        # written only on rows where exit_long did NOT also fire.  A long trade
        # on a row where both flags are 1 keeps its 'rank_exit_long' tag.
        dataframe.loc[exit_long, 'exit_tag'] = 'rank_exit_long'
        dataframe.loc[exit_short & ~exit_long, 'exit_tag'] = 'rank_exit_short'

        return dataframe

    # =========================================================================
    # BACKSTOP STOP: -12% from entry price (leverage-independent, house pattern)
    # =========================================================================
    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float,
                        after_fill: bool, **kwargs) -> float:
        stop_distance = float(self.stop_pct.value)
        if trade.is_short:
            sl_price = trade.open_rate * (1.0 + stop_distance)
        else:
            sl_price = trade.open_rate * (1.0 - stop_distance)

        rel = stoploss_from_absolute(
            sl_price, current_rate,
            is_short=trade.is_short,
            leverage=trade.leverage,
        )
        return rel if rel else self.stoploss

    # =========================================================================
    # LEVERAGE: always 1x for validation (spec §4)
    # =========================================================================
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float,
                 entry_tag: Optional[str], side: str, **kwargs) -> float:
        return float(self.config.get('leverage', 1.0))

    # =========================================================================
    # LOG PARAMS AT START
    # =========================================================================
    def bot_start(self, **kwargs) -> None:
        N = int(self.book_n.value)
        logger.info(
            f"XSMomentum — L={self.lookback_days.value}d "
            f"({self.lookback_days.value * 6} candles 4h), "
            f"N={N}/N={N} (long/short), "
            f"max_open_trades={2 * N}, "
            f"stop={self.stop_pct.value:.0%}"
        )
