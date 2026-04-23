"""
Money Flow Divergence Strategy
==============================

Based on "Money Flow Divergence Zones [AlgoAlpha]" Pine Script indicator.

Core Oscillator:
- MFI(14) smoothed by HMA(21)

Entry Signals (configurable):
1. Bullish/Bearish MFI divergences (price LL + osc HL at pivot lows, vice-versa)
2. Zone Touches — price re-enters divergence zone with a directional candle
3. Oscillator Crossovers — HMA-MFI changes direction

Zone Management:
- Bull/Bear zones created at each divergence pivot
- Mitigated when price closes through zone edge (or 2 consecutive closes, with allow_rejection)
- Auto-expired after expiry_age bars

Exit Logic (matching SMCWithMLLuxAlgo pattern):
- Opposite divergence / osc cross
- Take Profit 1 — partial close based on PRICE movement (not leveraged PnL)
- Break Even — SL moved to entry after TP1 triggered, PRICE-based

Dynamic Stoploss:
- Long  SL = bull zone bottom (lowest low) - buffer
- Short SL = bear zone top  (highest high) + buffer
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pandas import DataFrame

import talib.abstract as ta
from freqtrade.strategy import (
    IStrategy, IntParameter, DecimalParameter,
    BooleanParameter, CategoricalParameter, stoploss_from_absolute,
)
from freqtrade.persistence import Trade

logger = logging.getLogger(__name__)


class MFDivergenceStrategy(IStrategy):

    INTERFACE_VERSION = 3

    # ── Fixed defaults (leverage-adjusted in bot_start) ───────────────────────
    minimal_roi = {"0": 1.0}       # Disabled; custom exits used
    stoploss = -0.05               # Base price movement; scaled by leverage
    timeframe = "15m"

    trailing_stop = False
    use_exit_signal = True
    use_custom_stoploss = True
    position_adjustment_enable = True

    max_open_trades = 3
    startup_candle_count: int = 150
    can_short = True

    # ── MFI / Divergence ──────────────────────────────────────────────────────
    mfi_length    = IntParameter(8, 21,  default=14, space="buy", optimize=True)
    lookback_left = IntParameter(5, 21,  default=14, space="buy", optimize=True)
    lookback_right = IntParameter(3, 10, default=5,  space="buy", optimize=True)

    # ── Zone Management ───────────────────────────────────────────────────────
    expiry_age      = IntParameter(100, 2000, default=1000, space="buy", optimize=False)
    mitigate_type   = CategoricalParameter(["body", "wick"], default="body",  space="buy", optimize=False)
    allow_rejection = BooleanParameter(default=False, space="buy", optimize=False)

    # ── Entry ─────────────────────────────────────────────────────────────────
    entry_on_div       = BooleanParameter(default=True,  space="buy", optimize=False)
    entry_on_touch     = BooleanParameter(default=False, space="buy", optimize=True)
    entry_on_osc_cross = BooleanParameter(default=False, space="buy", optimize=True)

    # ── Exit ──────────────────────────────────────────────────────────────────
    exit_on_opposite  = BooleanParameter(default=True,  space="sell", optimize=False)
    exit_on_osc_cross = BooleanParameter(default=False, space="sell", optimize=True)

    # ── TP1 / BE (price-based, mirrors SMCWithMLLuxAlgo) ─────────────────────
    tp1_enabled  = BooleanParameter(default=True, space="sell", optimize=True)
    tp1_pct      = DecimalParameter(0.005, 0.10,  default=0.01, decimals=3, space="sell", optimize=True)
    tp1_amount   = DecimalParameter(10.0,  75.0,  default=50.0, decimals=1, space="sell", optimize=True)

    move_be_at_tp1 = BooleanParameter(default=True, space="sell", optimize=True)
    be_trigger_pct = DecimalParameter(0.0, 0.05, default=0.0, decimals=3, space="sell", optimize=True)

    use_dynamic_stoploss = BooleanParameter(default=True, space="sell", optimize=True)
    sl_buffer_pct        = DecimalParameter(0.001, 0.01, default=0.002, decimals=3, space="sell", optimize=True)

    # ── Misc ──────────────────────────────────────────────────────────────────
    enable_logging = BooleanParameter(default=True, space="custom", optimize=False)

    # =========================================================================
    # BOT START
    # =========================================================================

    def bot_start(self, **kwargs) -> None:
        """Scale stoploss by leverage (price-based)."""
        leverage = self.config.get("leverage", 1.0)
        raw_sl   = self.config.get("stoploss", -0.05)
        self.stoploss = raw_sl * leverage
        logger.info(
            f"MFDivergenceStrategy | leverage={leverage}x | "
            f"raw_sl={raw_sl:.2%} | effective_sl={self.stoploss:.2%}"
        )

    # =========================================================================
    # INTERNAL HELPERS
    # =========================================================================

    @staticmethod
    def _hma(series: pd.Series, period: int) -> pd.Series:
        """Hull Moving Average: WMA(2·WMA(n/2) − WMA(n), √n)."""
        half   = max(period // 2, 1)
        sqrt_p = max(int(np.sqrt(period)), 1)
        wma_h  = ta.WMA(series, timeperiod=half)
        wma_f  = ta.WMA(series, timeperiod=period)
        return ta.WMA(2 * wma_h - wma_f, timeperiod=sqrt_p)

    @staticmethod
    def _pivot_lows(series: pd.Series, left: int, right: int) -> pd.Series:
        """True at position i if series[i] is the minimum in [i-left, i+right]."""
        arr = series.values
        n   = len(arr)
        out = np.zeros(n, dtype=bool)
        for i in range(left, n - right):
            window = arr[i - left: i + right + 1]
            if not np.any(np.isnan(window)) and arr[i] == np.min(window):
                out[i] = True
        return pd.Series(out, index=series.index)

    @staticmethod
    def _pivot_highs(series: pd.Series, left: int, right: int) -> pd.Series:
        """True at position i if series[i] is the maximum in [i-left, i+right]."""
        arr = series.values
        n   = len(arr)
        out = np.zeros(n, dtype=bool)
        for i in range(left, n - right):
            window = arr[i - left: i + right + 1]
            if not np.any(np.isnan(window)) and arr[i] == np.max(window):
                out[i] = True
        return pd.Series(out, index=series.index)

    def _calculate_divergences(
        self, df: DataFrame, mfi: pd.Series, left: int, right: int
    ) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Detect bullish and bearish MFI divergences.

        In Pine Script, ta.pivotlow(v, left, right) fires at bar i when
        bar i-right is a confirmed pivot.  We replicate this by:
          1. Detecting actual pivots (no delay).
          2. Shifting forward by `right` bars → plFound/phFound.
          3. oscLBR / lowLBR / highLBR = values at the actual pivot bar.

        Divergence rules (identical to Pine):
          Bull: consecutive pivot lows where price makes LL but osc makes HL,
                and they are 5–60 bars apart.
          Bear: consecutive pivot highs where price makes HH but osc makes LH,
                same range constraint.
        """
        actual_pl = self._pivot_lows(mfi,  left, right)
        actual_ph = self._pivot_highs(mfi, left, right)

        # Delayed detection (as Pine does — confirmed right bars later)
        pl_found = actual_pl.shift(right).fillna(False).astype(bool)
        ph_found = actual_ph.shift(right).fillna(False).astype(bool)

        # Values AT the actual pivot bar
        osc_lbr  = mfi.shift(right).values
        low_lbr  = df["low"].shift(right).values
        high_lbr = df["high"].shift(right).values

        pl_pos = np.where(pl_found.values)[0]
        ph_pos = np.where(ph_found.values)[0]

        n        = len(df)
        bull_div = np.zeros(n, dtype=bool)
        bear_div = np.zeros(n, dtype=bool)

        for k in range(1, len(pl_pos)):
            curr, prev = pl_pos[k], pl_pos[k - 1]
            dist = curr - prev
            if not (5 <= dist <= 60):
                continue
            if any(np.isnan([osc_lbr[curr], osc_lbr[prev], low_lbr[curr], low_lbr[prev]])):
                continue
            if osc_lbr[curr] > osc_lbr[prev] and low_lbr[curr] < low_lbr[prev]:
                bull_div[curr] = True

        for k in range(1, len(ph_pos)):
            curr, prev = ph_pos[k], ph_pos[k - 1]
            dist = curr - prev
            if not (5 <= dist <= 60):
                continue
            if any(np.isnan([osc_lbr[curr], osc_lbr[prev], high_lbr[curr], high_lbr[prev]])):
                continue
            if osc_lbr[curr] < osc_lbr[prev] and high_lbr[curr] > high_lbr[prev]:
                bear_div[curr] = True

        return (
            pd.Series(bull_div, index=df.index),
            pd.Series(bear_div, index=df.index),
            pl_found,
            ph_found,
        )

    def _track_zones(
        self,
        df: DataFrame,
        bull_div: pd.Series,
        bear_div: pd.Series,
        right: int,
        zone_len: int,
        mitigate: str,
        allow_rejection: bool,
        expiry_age: int,
    ) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Forward-propagate divergence zones bar-by-bar, matching Pine logic:

        Bull zone:
          edge = bullExtreme     = lowest low of zone_len bars at pivot
          base = bullBodyExtreme = lowest min(open,close) of same bars
          Mitigation: close < edge  (body) | low < edge  (wick)

        Bear zone:
          edge = bearExtreme     = highest high of zone_len bars at pivot
          base = bearBodyExtreme = highest max(open,close) of same bars
          Mitigation: close > edge (body) | high > edge (wick)
        """
        n         = len(df)
        lows      = df["low"].values
        highs     = df["high"].values
        closes    = df["close"].values
        opens     = df["open"].values
        body_low  = np.minimum(opens, closes)
        body_high = np.maximum(opens, closes)

        bull_edge_arr = np.full(n, np.nan)
        bull_base_arr = np.full(n, np.nan)
        bear_edge_arr = np.full(n, np.nan)
        bear_base_arr = np.full(n, np.nan)

        active_bull: list[dict] = []  # [{edge, base, start_bar}]
        active_bear: list[dict] = []

        for i in range(n):
            cl      = closes[i]
            lo      = lows[i]
            hi      = highs[i]
            cl_prev = closes[i - 1] if i > 0 else cl
            lo_prev = lows[i - 1]   if i > 0 else lo
            hi_prev = highs[i - 1]  if i > 0 else hi

            # ── New bull zone ─────────────────────────────────────────────────
            if bull_div.iloc[i]:
                pbar  = i - right
                start = max(0, pbar - zone_len + 1)
                end   = pbar + 1
                if 0 < end <= n and start < n:
                    seg_low  = lows[start:end]
                    seg_body = body_low[start:end]
                    if len(seg_low) > 0:
                        active_bull.insert(0, {
                            "edge":      float(np.nanmin(seg_low)),
                            "base":      float(np.nanmin(seg_body)),
                            "start_bar": pbar,
                        })

            # ── New bear zone ─────────────────────────────────────────────────
            if bear_div.iloc[i]:
                pbar  = i - right
                start = max(0, pbar - zone_len + 1)
                end   = pbar + 1
                if 0 < end <= n and start < n:
                    seg_high = highs[start:end]
                    seg_body = body_high[start:end]
                    if len(seg_high) > 0:
                        active_bear.insert(0, {
                            "edge":      float(np.nanmax(seg_high)),
                            "base":      float(np.nanmax(seg_body)),
                            "start_bar": pbar,
                        })

            # ── Mitigate bull zones ───────────────────────────────────────────
            surviving_bull = []
            for z in active_bull:
                age  = i - z["start_bar"]
                yL   = z["edge"]
                sp   = cl   if mitigate == "body" else lo
                sp_p = cl_prev if mitigate == "body" else lo_prev

                if allow_rejection and mitigate == "body":
                    gone = sp < yL and sp_p < yL
                else:
                    gone = sp < yL

                if not gone and age <= expiry_age:
                    surviving_bull.append(z)
            active_bull = surviving_bull

            # ── Mitigate bear zones ───────────────────────────────────────────
            surviving_bear = []
            for z in active_bear:
                age  = i - z["start_bar"]
                yL   = z["edge"]
                sp   = cl   if mitigate == "body" else hi
                sp_p = cl_prev if mitigate == "body" else hi_prev

                if allow_rejection and mitigate == "body":
                    gone = sp > yL and sp_p > yL
                else:
                    gone = sp > yL

                if not gone and age <= expiry_age:
                    surviving_bear.append(z)
            active_bear = surviving_bear

            # ── Record most-recent active zone ────────────────────────────────
            if active_bull:
                bull_edge_arr[i] = active_bull[0]["edge"]
                bull_base_arr[i] = active_bull[0]["base"]
            if active_bear:
                bear_edge_arr[i] = active_bear[0]["edge"]
                bear_base_arr[i] = active_bear[0]["base"]

        idx = df.index
        return (
            pd.Series(bull_edge_arr, index=idx),
            pd.Series(bull_base_arr, index=idx),
            pd.Series(bear_edge_arr, index=idx),
            pd.Series(bear_base_arr, index=idx),
        )

    # =========================================================================
    # INDICATORS
    # =========================================================================

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        left  = self.lookback_left.value
        right = self.lookback_right.value

        # 1. MFI
        dataframe["mfi"] = ta.MFI(
            dataframe["high"], dataframe["low"],
            dataframe["close"], dataframe["volume"],
            timeperiod=self.mfi_length.value,
        )

        # 2. HMA(21) of MFI
        dataframe["mfi_hma"] = self._hma(dataframe["mfi"], 21)

        # 3. Oscillator direction-change signals
        # Pine: ta.crossover(v_smooth, v_smooth[1])  →  rising after being flat/falling
        hma     = dataframe["mfi_hma"]
        hma_1   = hma.shift(1)
        hma_2   = hma.shift(2)
        dataframe["osc_cross_up"]   = ((hma > hma_1) & (hma_1 <= hma_2)).astype(int)
        dataframe["osc_cross_down"] = ((hma < hma_1) & (hma_1 >= hma_2)).astype(int)

        # 4. Divergences
        bull_div, bear_div, pl_found, ph_found = self._calculate_divergences(
            dataframe, dataframe["mfi"], left, right
        )
        dataframe["bull_div"] = bull_div.astype(int)
        dataframe["bear_div"] = bear_div.astype(int)
        dataframe["pl_found"] = pl_found.astype(int)
        dataframe["ph_found"] = ph_found.astype(int)

        # 5. Zone tracking
        bull_edge, bull_base, bear_edge, bear_base = self._track_zones(
            dataframe, bull_div, bear_div, right,
            zone_len      = 5,
            mitigate      = self.mitigate_type.value,
            allow_rejection = self.allow_rejection.value,
            expiry_age    = self.expiry_age.value,
        )
        dataframe["bull_zone_edge"] = bull_edge   # lowest low  — mitigation threshold
        dataframe["bull_zone_base"] = bull_base   # lowest body — zone upper boundary
        dataframe["bear_zone_edge"] = bear_edge   # highest high — mitigation threshold
        dataframe["bear_zone_base"] = bear_base   # highest body — zone lower boundary

        # 6. Zone-touch flags
        # Bull zone: [bull_zone_edge (bottom), bull_zone_base (top)]
        bz_top = dataframe[["bull_zone_edge", "bull_zone_base"]].max(axis=1)
        bz_bot = dataframe[["bull_zone_edge", "bull_zone_base"]].min(axis=1)
        bull_in_zone = (
            dataframe["bull_zone_edge"].notna()
            & (dataframe["high"] >= bz_bot)
            & (dataframe["low"]  <= bz_top)
        )
        dataframe["bull_touch"] = (bull_in_zone & (dataframe["close"] > dataframe["open"])).astype(int)

        # Bear zone: [bear_zone_base (bottom), bear_zone_edge (top)]
        bez_top = dataframe[["bear_zone_edge", "bear_zone_base"]].max(axis=1)
        bez_bot = dataframe[["bear_zone_edge", "bear_zone_base"]].min(axis=1)
        bear_in_zone = (
            dataframe["bear_zone_edge"].notna()
            & (dataframe["high"] >= bez_bot)
            & (dataframe["low"]  <= bez_top)
        )
        dataframe["bear_touch"] = (bear_in_zone & (dataframe["close"] < dataframe["open"])).astype(int)

        # 7. Dynamic SL reference prices (stored once at entry candle)
        # Long  SL = bull zone bottom (edge = lowest low) minus buffer
        # Short SL = bear zone top   (edge = highest high) plus buffer
        buf = self.sl_buffer_pct.value
        dataframe["sl_long_price"]  = dataframe["bull_zone_edge"] * (1 - buf)
        dataframe["sl_short_price"] = dataframe["bear_zone_edge"] * (1 + buf)

        return dataframe

    # =========================================================================
    # ENTRY
    # =========================================================================

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        long_cond  = pd.Series(False, index=dataframe.index)
        short_cond = pd.Series(False, index=dataframe.index)

        if self.entry_on_div.value:
            long_cond  |= dataframe["bull_div"] == 1
            short_cond |= dataframe["bear_div"] == 1

        if self.entry_on_touch.value:
            long_cond  |= dataframe["bull_touch"] == 1
            short_cond |= dataframe["bear_touch"] == 1

        if self.entry_on_osc_cross.value:
            long_cond  |= dataframe["osc_cross_up"]   == 1
            short_cond |= dataframe["osc_cross_down"]  == 1

        vol_ok = dataframe["volume"] > 0

        dataframe.loc[long_cond  & vol_ok, "enter_long"]  = 1
        dataframe.loc[short_cond & vol_ok, "enter_short"] = 1

        # Entry tags
        dataframe["enter_tag"] = ""
        dataframe.loc[long_cond  & (dataframe["bull_div"]       == 1), "enter_tag"] = "bull_div"
        dataframe.loc[long_cond  & (dataframe["bull_touch"]     == 1) & (dataframe["bull_div"] != 1), "enter_tag"] = "bull_touch"
        dataframe.loc[long_cond  & (dataframe["osc_cross_up"]   == 1) & (dataframe["bull_div"] != 1) & (dataframe["bull_touch"] != 1), "enter_tag"] = "osc_up"
        dataframe.loc[short_cond & (dataframe["bear_div"]       == 1), "enter_tag"] = "bear_div"
        dataframe.loc[short_cond & (dataframe["bear_touch"]     == 1) & (dataframe["bear_div"] != 1), "enter_tag"] = "bear_touch"
        dataframe.loc[short_cond & (dataframe["osc_cross_down"] == 1) & (dataframe["bear_div"] != 1) & (dataframe["bear_touch"] != 1), "enter_tag"] = "osc_down"

        logger.info(
            f"MFD signals for {metadata['pair']}: "
            f"Long={long_cond.sum()}, Short={short_cond.sum()}"
        )
        return dataframe

    # =========================================================================
    # EXIT
    # =========================================================================

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        exit_long  = pd.Series(False, index=dataframe.index)
        exit_short = pd.Series(False, index=dataframe.index)

        if self.exit_on_opposite.value:
            exit_long  |= dataframe["bear_div"] == 1
            exit_short |= dataframe["bull_div"] == 1

        if self.exit_on_osc_cross.value:
            exit_long  |= dataframe["osc_cross_down"] == 1
            exit_short |= dataframe["osc_cross_up"]   == 1

        dataframe.loc[exit_long,  "exit_long"]  = 1
        dataframe.loc[exit_short, "exit_short"] = 1

        return dataframe

    # =========================================================================
    # LEVERAGE
    # =========================================================================

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: str,
                 side: str, **kwargs) -> float:
        return self.config.get("leverage", 1.0)

    def _get_timeframe_minutes(self) -> int:
        tf = self.timeframe
        if tf.endswith("m"):  return int(tf[:-1])
        if tf.endswith("h"):  return int(tf[:-1]) * 60
        if tf.endswith("d"):  return int(tf[:-1]) * 1440
        return 15

    # =========================================================================
    # CUSTOM STOPLOSS  —  Break Even + Dynamic SL (price-based, not PnL)
    # Mirrors SMCWithMLLuxAlgo.custom_stoploss()
    # =========================================================================

    def custom_stoploss(
        self, pair: str, trade: Trade, current_time: datetime,
        current_rate: float, current_profit: float, **kwargs
    ) -> float:

        be_activated = trade.get_custom_data("be_activated", default=False)

        # ── 1. Dynamic SL from structure (only before BE kicks in) ────────────
        if not be_activated and self.use_dynamic_stoploss.value:
            initial_sl = trade.get_custom_data("initial_sl_price")

            if initial_sl is None:
                try:
                    df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
                    candle = df.loc[df["date"] == trade.open_date_utc]
                    if not candle.empty:
                        row   = candle.iloc[0]
                        col   = "sl_short_price" if trade.is_short else "sl_long_price"
                        price = row.get(col, None)
                        if price and not pd.isna(price) and price > 0:
                            initial_sl = float(price)
                            trade.set_custom_data("initial_sl_price", initial_sl)
                            if self.enable_logging.value:
                                logger.info(f"Dynamic SL set for {pair}: {initial_sl:.6f}")
                except Exception:
                    pass

            if initial_sl and initial_sl > 0:
                return stoploss_from_absolute(
                    initial_sl, current_rate,
                    is_short=trade.is_short, leverage=trade.leverage,
                )

        # ── 2. Break Even ─────────────────────────────────────────────────────
        if trade.is_short:
            extremum       = trade.min_rate if trade.min_rate else current_rate
            price_movement = (trade.open_rate - extremum) / trade.open_rate
        else:
            extremum       = trade.max_rate if trade.max_rate else current_rate
            price_movement = (extremum - trade.open_rate) / trade.open_rate

        be_trigger = self.be_trigger_pct.value if self.be_trigger_pct.value > 0 else self.tp1_pct.value
        should_be  = self.move_be_at_tp1.value and price_movement >= be_trigger

        if not be_activated and should_be:
            fee_buf       = 0.001
            be_stop_price = (trade.open_rate * (1 - fee_buf) if trade.is_short
                             else trade.open_rate * (1 + fee_buf))
            trade.set_custom_data("be_activated", True)
            trade.set_custom_data("be_stop_price", be_stop_price)
            be_activated = True
            if self.enable_logging.value:
                logger.info(
                    f"BE activated for {pair} | move={price_movement:.2%} | "
                    f"stop={be_stop_price:.6f}"
                )

        if be_activated:
            be_stop_price = trade.get_custom_data("be_stop_price", default=trade.open_rate)
            return stoploss_from_absolute(
                be_stop_price, current_rate,
                is_short=trade.is_short, leverage=trade.leverage,
            )

        return 1  # fall back to strategy stoploss

    # =========================================================================
    # PARTIAL TP1  —  price movement, not leveraged PnL
    # Mirrors SMCWithMLLuxAlgo.adjust_trade_position()
    # =========================================================================

    def adjust_trade_position(
        self, trade: Trade, current_time: datetime,
        current_rate: float, current_profit: float,
        min_stake: float | None, max_stake: float,
        current_entry_rate: float, current_exit_rate: float,
        current_entry_profit: float, current_exit_profit: float,
        **kwargs,
    ) -> float | None | tuple[float | None, str | None]:

        if trade.is_short:
            extremum       = trade.min_rate if trade.min_rate else current_rate
            price_movement = (trade.open_rate - extremum) / trade.open_rate
        else:
            extremum       = trade.max_rate if trade.max_rate else current_rate
            price_movement = (extremum - trade.open_rate) / trade.open_rate

        tp1_taken = trade.get_custom_data("tp1_taken", default=False)

        if not tp1_taken and price_movement > self.tp1_pct.value:
            if not self.tp1_enabled.value or self.tp1_amount.value == 0:
                return None

            trade.set_custom_data("tp1_taken", True)

            original_amount = (trade.stake_amount * trade.leverage) / trade.open_rate
            close_amount    = original_amount * (self.tp1_amount.value / 100.0)
            close_amount    = min(close_amount, trade.amount * 0.99)

            sell_value   = close_amount * current_rate
            stake_change = sell_value / trade.leverage          # margin to release

            if self.enable_logging.value:
                logger.info(
                    f"TP1 {trade.pair} ({'SHORT' if trade.is_short else 'LONG'}): "
                    f"move={price_movement:.2%} | closing {self.tp1_amount.value:.0f}% "
                    f"| margin={stake_change:.2f}"
                )
            return (-stake_change, "TP1_partial")

        return None

    # =========================================================================
    # CONFIRM ENTRY  —  one trade per pair at a time
    # =========================================================================

    def confirm_trade_entry(
        self, pair: str, order_type: str, amount: float, rate: float,
        time_in_force: str, current_time: datetime,
        entry_tag: str | None, side: str, **kwargs,
    ) -> bool:
        if Trade.get_trades_proxy(pair=pair, is_open=True):
            return False
        return True
