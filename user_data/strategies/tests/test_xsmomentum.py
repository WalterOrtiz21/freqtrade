"""
Tests for XSMomentum — cross-sectional momentum rotation strategy.

Coverage targets (edge-protecting, not ceremonial):
1. No-lookahead of score: changing the last close must NOT change that bar's score.
2. Cross-sectional rank ordinal correctness + NaN exclusion for short-history pairs.
3. Hysteresis: enter top-N, hold top-2N, exit below top-2N; mirror for shorts.
4. Score-matrix cache: same timestamp → same object, no recompute;
   new timestamp → cache invalidated.
5. Timezone regression: pd.DatetimeIndex preserves UTC tz;
   test fails if .values strips tz (the Phase-1 critical bug).
"""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Ensure the strategies dir is importable without a full freqtrade install
# assumption: freqtrade package is available in the venv.
# ---------------------------------------------------------------------------
STRATEGIES_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(STRATEGIES_DIR))

from freqtrade.enums import CandleType  # noqa: E402
from XSMomentum import XSMomentum  # noqa: E402


# ---------------------------------------------------------------------------
# Minimal freqtrade config dict required by IStrategy.__init__
# ---------------------------------------------------------------------------
_MINIMAL_CONFIG: dict = {
    "stake_currency": "USDT",
    "stake_amount": 10.0,
    "max_open_trades": 6,
    "candle_type_def": CandleType.FUTURES,
    "trading_mode": "futures",
    "margin_mode": "isolated",
    "internals": {},
    "runmode": "backtest",
    "original_config": {},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_close_series(n: int, start: float = 100.0, seed: int = 42) -> pd.Series:
    """Deterministic price series with a mild uptrend and noise."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(loc=0.0005, scale=0.01, size=n)
    prices = start * np.cumprod(1 + returns)
    return pd.Series(prices, dtype=float)


def _make_utc_index(n: int, freq: str = "4h") -> pd.DatetimeIndex:
    """UTC-aware DatetimeIndex with 4h frequency."""
    return pd.date_range("2024-01-01", periods=n, freq=freq, tz="UTC")


def _make_dataframe(n: int, pair: str = "BTC/USDT:USDT", seed: int = 42) -> pd.DataFrame:
    """Minimal OHLCV dataframe with UTC-aware 'date' column."""
    close = _make_close_series(n, seed=seed)
    dates = _make_utc_index(n)
    df = pd.DataFrame(
        {
            "date": dates,
            "open": close * 0.999,
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close.values,
            "volume": np.ones(n) * 1000.0,
        }
    )
    return df


def _fresh_strategy() -> XSMomentum:
    """Return a clean strategy instance with cleared cache."""
    strat = XSMomentum(_MINIMAL_CONFIG)
    strat._score_matrix_cache = {}
    strat._score_matrix_cache_key = None
    return strat


def _mock_dp(pair_dfs: dict) -> MagicMock:
    """
    Build a mock DataProvider whose current_whitelist() returns the pair_dfs
    keys and get_pair_dataframe(pair, timeframe) returns pair_dfs[pair].
    """
    dp = MagicMock()
    dp.current_whitelist.return_value = list(pair_dfs.keys())
    dp.get_pair_dataframe.side_effect = lambda pair, timeframe: pair_dfs.get(pair)
    return dp


# ---------------------------------------------------------------------------
# 1. No-lookahead of the score
# ---------------------------------------------------------------------------

class TestNoLookahead:
    """
    The score on bar t must only use close[0..t-1] (shift(1) applied).
    Changing close[t] (the last bar) must NOT affect score[t].
    """

    L = 14  # candles

    def test_last_close_does_not_affect_own_bar_score(self):
        n = 100
        close = _make_close_series(n)

        score_original = XSMomentum._compute_score_series(close, self.L)
        score_at_last_bar = score_original.iloc[-1]

        # Mutate the last close drastically
        close_mutated = close.copy()
        close_mutated.iloc[-1] *= 10.0

        score_mutated = XSMomentum._compute_score_series(close_mutated, self.L)
        score_at_last_bar_mutated = score_mutated.iloc[-1]

        assert score_at_last_bar == pytest.approx(score_at_last_bar_mutated, rel=1e-9), (
            "Score at bar t changed when close[t] was mutated — lookahead detected!"
        )

    def test_last_close_does_not_affect_second_to_last_score(self):
        """Confirm -2 bar as well — belt-and-suspenders."""
        n = 100
        close = _make_close_series(n)
        score_original = XSMomentum._compute_score_series(close, self.L)

        close_mutated = close.copy()
        close_mutated.iloc[-1] = close_mutated.iloc[-1] * 50.0

        score_mutated = XSMomentum._compute_score_series(close_mutated, self.L)

        assert score_original.iloc[-2] == pytest.approx(score_mutated.iloc[-2], rel=1e-9)

    def test_score_is_nan_within_warmup(self):
        """First L+1 bars should be NaN (insufficient history after shift)."""
        n = 50
        L = 14
        close = _make_close_series(n)
        score = XSMomentum._compute_score_series(close, L)
        # shifted by 1 then log_ret needs L more bars, so first L+1 are NaN
        assert score.iloc[: L + 1].isna().all(), (
            "Expected NaN in warmup period — score computed without sufficient history"
        )


# ---------------------------------------------------------------------------
# 2. Cross-sectional rank correctness
# ---------------------------------------------------------------------------

class TestRankCrossSectional:
    """
    With a synthetic universe of known scores, verify:
    - ordinal rank is correct (highest score → rank_pct near 1.0)
    - pairs with insufficient history (< L candles) produce NaN scores
      and are excluded from ranking
    """

    L = 14  # candles (days × 6 = 84 4h candles, but _compute_score_series works on raw count)
    N_BARS = 120

    def _build_score_matrix_known(self) -> dict:
        """
        Three pairs with deterministic monotone prices.
        Pair A: strong uptrend (highest momentum)
        Pair B: flat (neutral)
        Pair C: downtrend (lowest momentum)
        """
        n = self.N_BARS
        dates = _make_utc_index(n)

        def _series(prices):
            s = pd.Series(prices, dtype=float)
            score = XSMomentum._compute_score_series(s, self.L)
            score.index = dates
            return score

        # Steep up: log returns strongly positive
        close_a = pd.Series(100.0 * np.exp(np.linspace(0, 1.0, n)))
        # Flat
        close_b = pd.Series(np.full(n, 100.0))
        # Down
        close_c = pd.Series(100.0 * np.exp(np.linspace(0, -1.0, n)))

        return {
            "A/USDT:USDT": _series(close_a),
            "B/USDT:USDT": _series(close_b),
            "C/USDT:USDT": _series(close_c),
        }

    def test_rank_order_is_correct(self):
        """A should rank highest, C lowest at the final bar."""
        score_matrix = self._build_score_matrix_known()
        strat = _fresh_strategy()

        pair_dates = list(score_matrix.values())[0].index
        rank_a, _ = strat._compute_rank_series("A/USDT:USDT", score_matrix, pair_dates)
        rank_c, _ = strat._compute_rank_series("C/USDT:USDT", score_matrix, pair_dates)
        rank_b, _ = strat._compute_rank_series("B/USDT:USDT", score_matrix, pair_dates)

        # At the last valid bar: A > B > C
        last = -1
        assert rank_a.iloc[last] > rank_b.iloc[last], "A should rank above B"
        assert rank_b.iloc[last] > rank_c.iloc[last], "B should rank above C"

    def test_short_history_pair_gets_nan_score_and_excluded_from_rank(self):
        """
        A pair with fewer than L+2 bars in get_pair_dataframe should not
        appear in the score matrix (skipped in _get_score_matrix).
        """
        strat = _fresh_strategy()

        # Build one good pair and one short-history pair
        n_good = 120
        n_short = 5  # < L + 2

        df_good = _make_dataframe(n_good, pair="GOOD/USDT:USDT", seed=10)
        df_short = _make_dataframe(n_short, pair="SHORT/USDT:USDT", seed=20)

        pair_dfs = {
            "GOOD/USDT:USDT": df_good,
            "SHORT/USDT:USDT": df_short,
        }
        strat.dp = _mock_dp(pair_dfs)

        L_candles = 14
        matrix = strat._get_score_matrix(L_candles)

        assert "GOOD/USDT:USDT" in matrix, "Good pair should be in the score matrix"
        assert "SHORT/USDT:USDT" not in matrix, (
            "Short-history pair must be excluded from the score matrix"
        )

    def test_rank_pct_in_zero_one_range(self):
        """rank_pct should always be within [0, 1]."""
        score_matrix = self._build_score_matrix_known()
        strat = _fresh_strategy()
        pair_dates = list(score_matrix.values())[0].index

        for pair in score_matrix:
            rank, _ = strat._compute_rank_series(pair, score_matrix, pair_dates)
            valid = rank.dropna()
            assert (valid >= 0.0).all() and (valid <= 1.0).all(), (
                f"rank_pct out of [0,1] for {pair}"
            )


# ---------------------------------------------------------------------------
# 3. Hysteresis
# ---------------------------------------------------------------------------

class TestHysteresis:
    """
    Entry: rank_pct >= 1 - N/U (top-N)
    Hold:  rank_pct >= 1 - 2N/U (top-2N)
    Exit:  rank_pct < 1 - 2N/U
    Mirror for short leg.

    We test the flags produced by populate_indicators using a mocked dp.
    """

    def _run_populate_indicators(
        self, n_pairs: int = 6, N: int = 3, L_days: int = 14, n_bars: int = 200
    ):
        """
        Build a synthetic universe of n_pairs pairs with distinct momentum
        profiles. Return the strat and the dataframe for pair[0] (the winner).
        """
        strat = _fresh_strategy()
        strat.lookback_days = SimpleNamespace(value=L_days)
        strat.book_n = SimpleNamespace(value=N)
        strat.timeframe = "4h"

        seeds = range(n_pairs)
        pairs = [f"PAIR{i}/USDT:USDT" for i in range(n_pairs)]
        pair_dfs = {}
        for i, pair in enumerate(pairs):
            seed = i
            # Give pair0 a strong uptrend, pair(n-1) a downtrend
            df = _make_dataframe(n_bars, pair=pair, seed=seed)
            # Amplify trend for pair0 and pair(n-1) to make zones clear
            if i == 0:
                df["close"] = df["close"] * np.exp(np.linspace(0, 0.5, n_bars))
            elif i == n_pairs - 1:
                df["close"] = df["close"] * np.exp(np.linspace(0, -0.5, n_bars))
            pair_dfs[pair] = df

        strat.dp = _mock_dp(pair_dfs)

        df0 = pair_dfs[pairs[0]].copy()
        df0 = strat.populate_indicators(df0, {"pair": pairs[0]})
        return strat, df0

    def test_long_zone_is_subset_of_long_hold(self):
        """Every bar in long_zone must also be in long_hold (entry ⊆ hold)."""
        _, df = self._run_populate_indicators()
        in_zone = df["xs_long_zone"]
        in_hold = df["xs_long_hold"]
        assert (in_zone & ~in_hold).sum() == 0, (
            "Found bars where xs_long_zone=True but xs_long_hold=False — "
            "hysteresis violated: entry is not a subset of hold zone"
        )

    def test_short_zone_is_subset_of_short_hold(self):
        """Every bar in short_zone must also be in short_hold."""
        strat = _fresh_strategy()
        strat.lookback_days = SimpleNamespace(value=14)
        strat.book_n = SimpleNamespace(value=3)
        strat.timeframe = "4h"

        n_bars, n_pairs = 200, 6
        pairs = [f"PAIR{i}/USDT:USDT" for i in range(n_pairs)]
        pair_dfs = {}
        for i, pair in enumerate(pairs):
            df = _make_dataframe(n_bars, pair=pair, seed=i + 100)
            if i == n_pairs - 1:
                df["close"] = df["close"] * np.exp(np.linspace(0, -0.5, n_bars))
            pair_dfs[pair] = df

        strat.dp = _mock_dp(pair_dfs)

        # Inspect the bottom pair (worst momentum)
        df_bottom = pair_dfs[pairs[-1]].copy()
        df_bottom = strat.populate_indicators(df_bottom, {"pair": pairs[-1]})

        in_zone = df_bottom["xs_short_zone"]
        in_hold = df_bottom["xs_short_hold"]
        assert (in_zone & ~in_hold).sum() == 0, (
            "Found bars where xs_short_zone=True but xs_short_hold=False — "
            "hysteresis violated for short leg"
        )

    def test_hold_zone_is_wider_than_entry_zone(self):
        """
        The hold threshold is 2×N/U whereas entry is N/U.
        So hold zone must cover at least as many bars as entry zone.
        """
        _, df = self._run_populate_indicators()
        assert df["xs_long_hold"].sum() >= df["xs_long_zone"].sum(), (
            "Long hold zone is not wider than long entry zone"
        )
        assert df["xs_short_hold"].sum() >= df["xs_short_zone"].sum(), (
            "Short hold zone is not wider than short entry zone"
        )


# ---------------------------------------------------------------------------
# 4. Score-matrix cache
# ---------------------------------------------------------------------------

class TestScoreMatrixCache:
    """
    Same (L_candles, universe_tuple) → same dict object returned, no recompute.
    Different universe_tuple → cache invalidated.
    """

    def _build_strat_with_dp(self, pairs=None, n_bars=200):
        if pairs is None:
            pairs = ["BTC/USDT:USDT", "ETH/USDT:USDT"]
        pair_dfs = {p: _make_dataframe(n_bars, pair=p, seed=i) for i, p in enumerate(pairs)}
        strat = _fresh_strategy()
        strat.dp = _mock_dp(pair_dfs)
        return strat

    def test_same_call_returns_same_object(self):
        strat = self._build_strat_with_dp()
        m1 = strat._get_score_matrix(84)
        m2 = strat._get_score_matrix(84)
        assert m1 is m2, "Expected same object (cache hit) on second call"

    def test_get_pair_dataframe_called_once_for_two_calls(self):
        strat = self._build_strat_with_dp()
        strat._get_score_matrix(84)
        call_count_after_first = strat.dp.get_pair_dataframe.call_count
        strat._get_score_matrix(84)
        call_count_after_second = strat.dp.get_pair_dataframe.call_count
        assert call_count_after_first == call_count_after_second, (
            "dp.get_pair_dataframe was called again on cache hit — recompute happening"
        )

    def test_different_universe_invalidates_cache(self):
        pairs1 = ["BTC/USDT:USDT", "ETH/USDT:USDT"]
        pairs2 = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"]

        strat = self._build_strat_with_dp(pairs=pairs1)
        m1 = strat._get_score_matrix(84)

        # Swap to a larger universe
        strat.dp = _mock_dp(
            {p: _make_dataframe(200, pair=p, seed=i) for i, p in enumerate(pairs2)}
        )
        m2 = strat._get_score_matrix(84)

        assert m1 is not m2, "Expected cache invalidation when universe changes"
        assert len(m2) >= len(m1), "New matrix should have more pairs"

    def test_different_L_invalidates_cache(self):
        strat = self._build_strat_with_dp()
        m1 = strat._get_score_matrix(84)   # L=14d
        m2 = strat._get_score_matrix(168)  # L=28d
        assert m1 is not m2, "Expected cache invalidation when L_candles changes"


# ---------------------------------------------------------------------------
# 5. Timezone regression — the Phase-1 critical bug
# ---------------------------------------------------------------------------

class TestTimezoneRegression:
    """
    Regression test for the UTC timezone-stripping bug from Phase 1.

    The bug: using `.values` on a tz-aware DatetimeIndex converts it to
    a tz-naive numpy array, causing timestamp mismatches when looking up
    scores in the matrix (which uses tz-aware UTC index).

    The fix in _get_score_matrix is:
        score_series.index = pd.DatetimeIndex(df['date'])   ← preserves tz
    NOT:
        score_series.index = df['date'].values              ← strips tz

    This test builds a score series with a tz-aware index (as the strategy
    does) and verifies that a tz-aware lookup succeeds, while a tz-naive
    lookup raises or returns NaN (demonstrating why .values was wrong).
    """

    def test_tz_aware_index_preserved_in_score_series(self):
        """Score series index must be tz-aware (UTC) after _get_score_matrix."""
        n = 120
        df = _make_dataframe(n)  # df['date'] is UTC-aware
        strat = _fresh_strategy()
        strat.dp = _mock_dp({"BTC/USDT:USDT": df})

        matrix = strat._get_score_matrix(84)
        assert "BTC/USDT:USDT" in matrix

        idx = matrix["BTC/USDT:USDT"].index
        assert idx.tz is not None, (
            "Score series index is tz-naive — .values was used somewhere, "
            "stripping UTC tz. This is the Phase-1 lookahead/lookup bug."
        )
        assert str(idx.tz) in ("UTC", "pytz.UTC", "datetime.timezone.utc"), (
            f"Expected UTC timezone, got: {idx.tz}"
        )

    def test_tz_aware_lookup_succeeds(self):
        """
        A tz-aware timestamp lookup into the score matrix must succeed
        (not KeyError or NaN on every bar).
        """
        n = 120
        df = _make_dataframe(n)
        strat = _fresh_strategy()
        strat.dp = _mock_dp({"BTC/USDT:USDT": df})

        matrix = strat._get_score_matrix(84)
        score_series = matrix["BTC/USDT:USDT"]

        # Sample valid timestamps from the dataframe
        sample_ts = pd.DatetimeIndex(df["date"]).dropna()[-5:]
        for ts in sample_ts:
            try:
                val = score_series.loc[ts]
                # val may be NaN if in warmup, but the lookup itself must not raise
                assert isinstance(val, (float, np.floating)), (
                    f"Unexpected type {type(val)} for score at {ts}"
                )
            except KeyError:
                pytest.fail(
                    f"KeyError on tz-aware lookup at {ts} — "
                    "score index tz does not match. "
                    "The .values stripping bug may have been reintroduced."
                )

    def test_values_stripping_would_break_lookup(self):
        """
        Demonstrate (and document) that using .values to assign the index
        causes tz-naive index, which cannot match tz-aware lookup timestamps.
        This test does NOT call strategy code — it documents the invariant.
        """
        n = 50
        dates_aware = _make_utc_index(n)
        dates_naive = dates_aware.values  # strips tz

        s_aware = pd.Series(range(n), index=dates_aware, dtype=float)
        s_naive = pd.Series(range(n), index=pd.DatetimeIndex(dates_naive), dtype=float)

        ts_aware = dates_aware[25]

        # Aware lookup on aware series → works
        assert s_aware.loc[ts_aware] == 25.0

        # Aware lookup on naive series → raises TypeError or KeyError
        with pytest.raises((KeyError, TypeError)):
            _ = s_naive.loc[ts_aware]


# ---------------------------------------------------------------------------
# 6. Live cache invalidation — regression for the dry/live freeze bug
# ---------------------------------------------------------------------------

class TestLiveCacheInvalidation:
    """
    Regression tests for the live/dry cache-freeze bug.

    In live/dry the cache key includes latest_ts so the matrix is recomputed
    on each new candle cycle.  In backtest latest_ts=None keeps the original
    single-compute behaviour.

    (a) Calling _get_score_matrix with a NEW latest_ts must produce a cache
        miss (recompute — returned object is different).
    (b) Calling _get_score_matrix with the SAME latest_ts must hit the cache
        (same object returned, dp.get_pair_dataframe not called again).
    """

    def _build_live_strat(self, pairs=None, n_bars=200):
        """Return a strategy whose dp reports runmode=dry_run."""
        if pairs is None:
            pairs = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"]
        pair_dfs = {p: _make_dataframe(n_bars, pair=p, seed=i) for i, p in enumerate(pairs)}

        strat = _fresh_strategy()

        dp = _mock_dp(pair_dfs)
        # Simulate live/dry runmode so the strategy picks up latest_ts
        from freqtrade.enums import RunMode
        dp.runmode = RunMode.DRY_RUN

        strat.dp = dp
        return strat

    def test_new_latest_ts_causes_cache_miss(self):
        """
        (a) When latest_ts advances (new candle), the matrix must be
            recomputed — the returned object must be a *different* dict.
        """
        strat = self._build_live_strat()
        L = 84

        ts_old = pd.Timestamp("2025-01-01 00:00:00", tz="UTC")
        ts_new = pd.Timestamp("2025-01-01 04:00:00", tz="UTC")  # one 4h bar later

        m1 = strat._get_score_matrix(L, latest_ts=ts_old)
        call_count_after_first = strat.dp.get_pair_dataframe.call_count

        m2 = strat._get_score_matrix(L, latest_ts=ts_new)
        call_count_after_second = strat.dp.get_pair_dataframe.call_count

        assert m1 is not m2, (
            "Expected a NEW matrix object when latest_ts changed — cache should have missed"
        )
        assert call_count_after_second > call_count_after_first, (
            "dp.get_pair_dataframe was not called again after latest_ts changed"
        )

    def test_same_latest_ts_returns_cached_object(self):
        """
        (b) Calling with the same latest_ts twice must return the same dict
            object (cache hit) without calling dp.get_pair_dataframe again.
        """
        strat = self._build_live_strat()
        L = 84

        ts = pd.Timestamp("2025-01-01 00:00:00", tz="UTC")

        m1 = strat._get_score_matrix(L, latest_ts=ts)
        call_count_after_first = strat.dp.get_pair_dataframe.call_count

        m2 = strat._get_score_matrix(L, latest_ts=ts)
        call_count_after_second = strat.dp.get_pair_dataframe.call_count

        assert m1 is m2, (
            "Expected the SAME cached object on second call with identical latest_ts"
        )
        assert call_count_after_first == call_count_after_second, (
            "dp.get_pair_dataframe was called again on cache hit — recompute is happening"
        )
