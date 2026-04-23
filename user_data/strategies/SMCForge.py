"""
SMCForge — pure-rules SMC strategy v1
=====================================

Built on the SMC_Forge canonical layers (engine + quality + levels +
inducement). No ML, no LLM. Single goal: validate that the canonical
SMC recipe has edge before adding any overlay.

Recipe
------
LONG entry requires ALL:
  1. bull IDM swept in last `idm_sweep_lookback` bars (liquidity grab)
  2. CHoCH bullish (internal or swing) with displacement >= disp_threshold
     ATR (institutional impulse), within `idm_sweep_lookback` bars
  3. Price currently inside an active bullish POI (Order Block or FVG)
  4. Price in discount (range_pct < 50)

SHORT: mirror (bear IDM, CHoCH bear with displacement, bearish POI,
        price in premium).

Exits
-----
- Fixed SL: -3% (adjusted by leverage in bot_start)
- Fixed TP: +5% (via minimal_roi)
- Final exit: opposite-direction CHoCH (canon SMC)
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame

from freqtrade.strategy import (
    BooleanParameter,
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
    IStrategy,
    stoploss_from_absolute,
)
from freqtrade.exchange import timeframe_to_prev_date
from freqtrade.persistence import Trade
from datetime import datetime


# Import forge layers from sibling subdirectory
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / 'SMC_Forge'))

from forge_engine import SMCEngine            # noqa: E402
from forge_quality import annotate_displacement  # noqa: E402
from forge_levels import annotate_eqh_eql     # noqa: E402
from forge_inducement import annotate_inducement  # noqa: E402


logger = logging.getLogger(__name__)


class SMCForge(IStrategy):
    INTERFACE_VERSION = 3

    # =========================================================
    # Base config
    # =========================================================
    timeframe = '1h'
    can_short = True
    process_only_new_candles = True
    use_exit_signal = True          # honor opposite-CHoCH from populate_exit_trend
    use_custom_stoploss = True      # v2: ATR-adaptive SL via custom_stoploss
    position_adjustment_enable = True   # v3: TP1 partial + BE
    startup_candle_count: int = 200

    # Hard-cap safety net: custom_stoploss usually returns tighter values.
    # This is the worst-case fallback if ATR is NaN or helper fails.
    stoploss = -0.10
    minimal_roi = {"0": 100}        # effectively disabled; custom_exit handles TP when ATR mode on
    max_open_trades = 5

    # =========================================================
    # SMC params — todos configurables vía JSON.
    # Estructurales (engine/fractals) → optimize=False (esqueleto canon).
    # Tácticos (thresholds/filters) → optimize=True (tuneables).
    # =========================================================

    # Engine (structural)
    internal_length = IntParameter(3, 8, default=5, space='buy', optimize=False)
    swing_length = IntParameter(20, 60, default=50, space='buy', optimize=False)

    # Quality (displacement filter) — tactical
    disp_threshold = DecimalParameter(
        0.5, 2.5, default=1.0, decimals=2, space='buy', optimize=True,
    )
    disp_lookback_bars = IntParameter(2, 5, default=3, space='buy', optimize=False)

    # Levels (structural)
    fractal_n_major = IntParameter(3, 10, default=5, space='buy', optimize=False)
    fractal_n_minor = IntParameter(2, 4, default=2, space='buy', optimize=False)
    eqh_tolerance_atr = DecimalParameter(
        0.05, 0.30, default=0.10, decimals=2, space='buy', optimize=False,
    )

    # Entry recipe — tactical
    idm_sweep_lookback = IntParameter(2, 8, default=5, space='buy', optimize=True)
    use_choch_internal = BooleanParameter(default=True, space='buy', optimize=True)
    use_choch_swing = BooleanParameter(default=True, space='buy', optimize=True)
    require_poi = BooleanParameter(default=True, space='buy', optimize=True)
    require_pd_alignment = BooleanParameter(default=False, space='buy', optimize=True)

    # =========================================================
    # Macro filter — HTF bias gate via SMC dogfooding
    # Reference pair:
    #   off       : no filter (allow both directions always)
    #   btc_htf   : BTC as proxy — useful when trading highly-correlated
    #               majors (BTC/ETH/SOL typically move together >85% of time)
    #   self_htf  : each symbol uses its OWN HTF swing_trend — better when
    #               trading decorrelated pairs (memes, narrative tokens,
    #               RWA, etc.) whose structure diverges from BTC
    # Bias semantics (either ref):
    #   +1 → longs only   / -1 → shorts only   / 0 → ranging
    #   ranging: both sides allowed if macro_neutral_both_sides=True,
    #            no trade otherwise.
    # =========================================================
    # MODE (fixed — not hyperopted to avoid conditional-param waste)
    macro_filter_mode = CategoricalParameter(
        ["off", "btc_htf", "self_htf"],
        default="self_htf", space='buy', optimize=False,
    )
    # Active params inside the fixed mode
    macro_smc_htf = CategoricalParameter(
        ["1h", "4h", "1d"], default="4h", space='buy', optimize=True,
    )
    macro_neutral_both_sides = BooleanParameter(
        default=True, space='buy', optimize=True,
    )

    # =========================================================
    # Exits — TP via structural / ATR / fixed / none, SL via ATR / fixed
    # =========================================================
    # TP MODE
    #   structural : next opposing liquidity pool (EQH/EQL, OB, FVG, Breaker,
    #                swing_high/low) — SMC canon. Fallback to ATR if no pool.
    #   atr        : N × ATR(entry) — mechanical, vol-scaled
    #   fixed      : minimal_roi handles it (tp_pct × leverage)
    #   none       : no hard TP — trade runs until CHoCH opposite, BE, or
    #                hard-cap. Pairs well with TP1 > 0 that locks partial gain.
    # MODE (fixed — not hyperopted)
    tp_mode = CategoricalParameter(
        ["structural", "atr", "fixed", "none"], default="atr",
        space='sell', optimize=False,
    )
    sl_mode = CategoricalParameter(
        ["atr", "fixed"], default="fixed", space='sell', optimize=False,
    )

    # Active param inside tp_mode=atr
    tp_atr_mult = DecimalParameter(
        1.5, 6.0, default=3.0, decimals=1, space='sell', optimize=True,
    )
    # Active param inside sl_mode=fixed
    stoploss_pct = DecimalParameter(
        -0.05, -0.01, default=-0.03, decimals=3, space='sell', optimize=True,
    )

    # UNUSED under current modes — optimize=False to avoid wasted hyperopt epochs
    tp_structural_margin = DecimalParameter(
        0.80, 1.00, default=0.95, decimals=3, space='sell', optimize=False,
    )
    sl_atr_mult = DecimalParameter(
        1.0, 4.0, default=2.0, decimals=1, space='sell', optimize=False,
    )
    tp_pct = DecimalParameter(
        0.01, 0.10, default=0.05, decimals=3, space='sell', optimize=False,
    )

    # =========================================================
    # TP1 partial + Break-Even (v3) — canonical SMC money management
    # When enabled: close tp1_amount_pct of position at tp1 distance,
    # then move SL to entry+fee_buffer so the remainder runs to TP-final
    # risk-free.
    # =========================================================
    # MODE (fixed — not hyperopted)
    tp1_enabled = BooleanParameter(default=True, space='sell', optimize=False)
    tp1_mode = CategoricalParameter(
        ["atr", "pct"], default="atr", space='sell', optimize=False,
    )
    # Active params under tp1_mode=atr
    tp1_atr_mult = DecimalParameter(
        0.5, 4.0, default=3.0, decimals=1, space='sell', optimize=True,
    )
    tp1_amount_pct = IntParameter(
        20, 80, default=70, space='sell', optimize=True,
    )
    be_buffer_pct = DecimalParameter(
        0.0, 0.005, default=0.001, decimals=4, space='sell', optimize=True,
    )
    # UNUSED under tp1_mode=atr
    tp1_pct = DecimalParameter(
        0.005, 0.03, default=0.015, decimals=3, space='sell', optimize=False,
    )

    # =========================================================
    # Lifecycle
    # =========================================================

    def bot_start(self, **kwargs) -> None:
        cfg_lev = float(self.config.get('leverage', 1.0))
        # Hard-cap safety: used only if custom_stoploss returns None
        self.stoploss = -0.10 * cfg_lev

        sl_mode = str(self.sl_mode.value)
        tp_mode = str(self.tp_mode.value)

        # SL fixed mode → self.stoploss is the real SL (as %-PnL)
        if sl_mode == 'fixed':
            self.stoploss = float(self.stoploss_pct.value) * cfg_lev

        # TP fixed mode → minimal_roi dispatches TP
        if tp_mode == 'fixed':
            self.minimal_roi = {0: float(self.tp_pct.value) * cfg_lev}
        else:
            # atr / structural / none: minimal_roi disabled
            self.minimal_roi = {0: 100.0}

        logger.info(
            'SMCForge starting. tf=%s  leverage=%.2fx  '
            'SL=%s TP=%s | macro=%s htf=%s',
            self.timeframe, cfg_lev, sl_mode, tp_mode,
            self.macro_filter_mode.value, self.macro_smc_htf.value,
        )

    def leverage(self, pair, current_time, current_rate, proposed_leverage,
                 max_leverage, entry_tag, side, **kwargs) -> float:
        return float(self.config.get('leverage', 1.0))

    def informative_pairs(self):
        """
        Load HTF data for macro bias. We always declare BOTH potential refs
        (BTC + each whitelisted pair) at the configured HTF — this way the
        strategy can switch between btc_htf and self_htf modes without
        re-declaring. The cost is minor (data cached).
        """
        htf = str(self.macro_smc_htf.value)
        pairs_set: set = set()
        pairs_set.add(("BTC/USDT:USDT", htf))
        if self.dp:
            for p in self.dp.current_whitelist():
                pairs_set.add((p, htf))
        return list(pairs_set)

    # =========================================================
    # Macro bias — HTF swing_trend via forge_engine (dogfooding)
    # Ref pair is BTC (btc_htf) or the symbol itself (self_htf).
    # =========================================================

    def _compute_macro_bias(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds macro_bias column (+1/0/-1) from forge_engine swing_trend on the
        reference pair's HTF. Uses shift(1) + merge_asof(backward) to prevent
        lookahead.
        """
        dataframe['macro_bias'] = 0
        mode = str(self.macro_filter_mode.value)

        if not self.dp or mode == 'off':
            return dataframe

        if mode == 'btc_htf':
            ref_pair = 'BTC/USDT:USDT'
        elif mode == 'self_htf':
            ref_pair = metadata['pair']
        else:
            return dataframe

        htf = str(self.macro_smc_htf.value)

        try:
            ref_df = self.dp.get_pair_dataframe(ref_pair, htf)
            if ref_df is None or ref_df.empty:
                return dataframe

            eng = SMCEngine(
                ref_df,
                internal_length=int(self.internal_length.value),
                swing_length=int(self.swing_length.value),
            )
            sig = eng.get_signals()
            b = ref_df[['date']].copy()
            b['bias'] = sig['swing_trend'].shift(1).fillna(0).astype(int).values
            b = b.sort_values('date')

            ref_tz = dataframe['date'].dt.tz
            src_tz = b['date'].dt.tz
            if ref_tz is not None and src_tz is None:
                b['date'] = b['date'].dt.tz_localize('UTC')
            elif ref_tz is None and src_tz is not None:
                b['date'] = b['date'].dt.tz_convert(None)

            merged = pd.merge_asof(
                dataframe[['date']].copy().sort_values('date'),
                b, on='date', direction='backward',
            )
            dataframe['macro_bias'] = (
                merged['bias'].fillna(0).astype(int).values
            )
        except Exception as e:
            logger.warning(
                'Macro bias calc failed for %s (ref=%s, htf=%s): %s',
                metadata.get('pair', '?'), ref_pair, htf, e,
            )

        return dataframe

    # =========================================================
    # Indicators — orchestrate the 4 SMC_Forge layers
    # =========================================================

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe

        # Layer 1: structural events
        eng = SMCEngine(
            df,
            internal_length=int(self.internal_length.value),
            swing_length=int(self.swing_length.value),
        )
        signals = eng.get_signals()
        for col in signals.columns:
            df[col] = signals[col].values

        # Layer 2: displacement quality on each event
        disp = annotate_displacement(
            df, signals,
            atr_period=14,
            lookback=int(self.disp_lookback_bars.value),
        )
        for col in disp.columns:
            if col not in df.columns:
                df[col] = disp[col].values

        # Layer 3: EQH/EQL liquidity pools
        levels = annotate_eqh_eql(
            df,
            atr_period=14,
            fractal_n=int(self.fractal_n_major.value),
            tolerance_atr=float(self.eqh_tolerance_atr.value),
        )
        for col in levels.columns:
            if col not in df.columns:
                df[col] = levels[col].values

        # Layer 4: IDM (inducement) detector
        idm = annotate_inducement(
            df,
            fractal_n_major=int(self.fractal_n_major.value),
            fractal_n_minor=int(self.fractal_n_minor.value),
        )
        for col in idm.columns:
            if col not in df.columns:
                df[col] = idm[col].values

        # Premium / Discount within current MTF swing range
        rng = (df['swing_high'] - df['swing_low']).clip(lower=1e-10)
        df['range_pct'] = ((df['close'] - df['swing_low']) / rng * 100).fillna(50.0)
        df['in_discount'] = df['range_pct'] < 50.0
        df['in_premium'] = df['range_pct'] > 50.0

        # Macro filter — HTF swing_trend on BTC or self (dogfooding engine)
        df = self._compute_macro_bias(df, metadata)

        return df

    # =========================================================
    # Entry — canonical A+ recipe
    # =========================================================

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe
        lookback = int(self.idm_sweep_lookback.value)
        thr = float(self.disp_threshold.value)

        # ── LONG ─────────────────────────────────────────────
        # 1. IDM swept recently
        bull_sweep_recent = (
            df['bull_idm_swept'].rolling(lookback, min_periods=1).max() == 1
        )

        # 2. CHoCH bull with displacement
        choch_int_bull = (
            (df['internal_choch_bullish'] == 1) &
            (df['internal_choch_bull_disp'] >= thr)
            if self.use_choch_internal.value else pd.Series(False, index=df.index)
        )
        choch_sw_bull = (
            (df['swing_choch_bullish'] == 1) &
            (df['swing_choch_bull_disp'] >= thr)
            if self.use_choch_swing.value else pd.Series(False, index=df.index)
        )
        choch_bull_recent = (
            (choch_int_bull | choch_sw_bull)
            .rolling(lookback, min_periods=1).max() == 1
        )

        # 3. Inside bullish POI
        in_bull_ob = (
            (df['active_bullish_ob_top'] > 0) &
            (df['low'] <= df['active_bullish_ob_top']) &
            (df['close'] >= df['active_bullish_ob_bottom'])
        )
        in_bull_fvg = (
            (df['active_bullish_fvg_top'] > 0) &
            (df['low'] <= df['active_bullish_fvg_top']) &
            (df['close'] >= df['active_bullish_fvg_bottom'])
        )
        in_bull_poi = in_bull_ob | in_bull_fvg

        long_cond = bull_sweep_recent & choch_bull_recent
        if self.require_poi.value:
            long_cond = long_cond & in_bull_poi
        if self.require_pd_alignment.value:
            long_cond = long_cond & df['in_discount']

        df.loc[long_cond, 'enter_long'] = 1
        df.loc[long_cond, 'enter_tag'] = 'IDM+CHOCH+POI'

        # ── SHORT (mirror) ───────────────────────────────────
        bear_sweep_recent = (
            df['bear_idm_swept'].rolling(lookback, min_periods=1).max() == 1
        )
        choch_int_bear = (
            (df['internal_choch_bearish'] == 1) &
            (df['internal_choch_bear_disp'] >= thr)
            if self.use_choch_internal.value else pd.Series(False, index=df.index)
        )
        choch_sw_bear = (
            (df['swing_choch_bearish'] == 1) &
            (df['swing_choch_bear_disp'] >= thr)
            if self.use_choch_swing.value else pd.Series(False, index=df.index)
        )
        choch_bear_recent = (
            (choch_int_bear | choch_sw_bear)
            .rolling(lookback, min_periods=1).max() == 1
        )

        in_bear_ob = (
            (df['active_bearish_ob_top'] > 0) &
            (df['high'] >= df['active_bearish_ob_bottom']) &
            (df['close'] <= df['active_bearish_ob_top'])
        )
        in_bear_fvg = (
            (df['active_bearish_fvg_top'] > 0) &
            (df['high'] >= df['active_bearish_fvg_bottom']) &
            (df['close'] <= df['active_bearish_fvg_top'])
        )
        in_bear_poi = in_bear_ob | in_bear_fvg

        short_cond = bear_sweep_recent & choch_bear_recent
        if self.require_poi.value:
            short_cond = short_cond & in_bear_poi
        if self.require_pd_alignment.value:
            short_cond = short_cond & df['in_premium']

        df.loc[short_cond, 'enter_short'] = 1
        df.loc[short_cond, 'enter_tag'] = 'IDM+CHOCH+POI'

        # ── MACRO FILTER — HTF swing_trend (BTC or self) ────────
        if str(self.macro_filter_mode.value) != 'off' and 'macro_bias' in df.columns:
            bias = df['macro_bias']
            neutral_both = bool(self.macro_neutral_both_sides.value)
            if neutral_both:
                # 0 neutral → both sides allowed.
                # +1 → longs only (block shorts).
                # -1 → shorts only (block longs).
                df.loc[bias < 0, 'enter_long'] = 0
                df.loc[bias > 0, 'enter_short'] = 0
            else:
                # Only trade on clear directional bias (no trade in ranging).
                df.loc[bias != 1, 'enter_long'] = 0
                df.loc[bias != -1, 'enter_short'] = 0

        return df

    # =========================================================
    # ATR-adaptive exits (v2)
    # =========================================================

    def _entry_row(self, trade: Trade, pair: str):
        """Return the OHLC+indicators row corresponding to the trade's entry bar."""
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or dataframe.empty:
            return None
        entry_date = timeframe_to_prev_date(self.timeframe, trade.open_date_utc)
        row = dataframe[dataframe['date'] == entry_date]
        if row.empty:
            return None
        return row.iloc[0]

    @staticmethod
    def _safe_float(row, col: str) -> float | None:
        val = row.get(col, None) if row is not None else None
        if val is None:
            return None
        try:
            v = float(val)
        except (TypeError, ValueError):
            return None
        if np.isnan(v) or v <= 0:
            return None
        return v

    def _atr_tp_price(self, trade: Trade, row) -> float | None:
        atr = self._safe_float(row, 'atr_14')
        if atr is None:
            return None
        d = atr * float(self.tp_atr_mult.value)
        return trade.open_rate - d if trade.is_short else trade.open_rate + d

    # Candidate columns for the "next opposing pool" search.
    # Long TP targets = BEARISH zones above price.
    _LONG_TP_COLS = (
        'eqh_level',
        'active_bearish_ob_top',
        'active_bearish_fvg_top',
        'active_bearish_breaker_top',
        'active_bearish_fvg_breaker_top',
        'swing_high',
        'pivot_high_level',
    )
    # Short TP targets = BULLISH zones below price.
    _SHORT_TP_COLS = (
        'eql_level',
        'active_bullish_ob_bottom',
        'active_bullish_fvg_bottom',
        'active_bullish_breaker_bottom',
        'active_bullish_fvg_breaker_bottom',
        'swing_low',
        'pivot_low_level',
    )

    def _structural_tp_price(self, trade: Trade, row) -> float | None:
        """
        Nearest opposing liquidity pool past the entry price.
        Returns None if no valid candidate exists.
        """
        entry = trade.open_rate
        cols = self._SHORT_TP_COLS if trade.is_short else self._LONG_TP_COLS

        best: float | None = None
        for col in cols:
            val = self._safe_float(row, col)
            if val is None:
                continue
            if trade.is_short:
                # looking for pool BELOW entry; pick the HIGHEST (nearest)
                if val < entry and (best is None or val > best):
                    best = val
            else:
                # looking for pool ABOVE entry; pick the LOWEST (nearest)
                if val > entry and (best is None or val < best):
                    best = val
        return best

    def _apply_margin(self, trade: Trade, target: float) -> float:
        """Adjust target so TP triggers at `margin` fraction of distance."""
        m = float(self.tp_structural_margin.value)
        if trade.is_short:
            return trade.open_rate - (trade.open_rate - target) * m
        return trade.open_rate + (target - trade.open_rate) * m

    def _tp1_price(self, trade: Trade, row) -> float | None:
        """Compute TP1 target price based on tp1_mode (atr or pct)."""
        side = -1 if trade.is_short else 1
        mode = str(self.tp1_mode.value)
        if mode == 'atr':
            atr = self._safe_float(row, 'atr_14')
            if atr is None:
                return None
            dist = atr * float(self.tp1_atr_mult.value)
        elif mode == 'pct':
            dist = trade.open_rate * float(self.tp1_pct.value)
        else:
            return None
        return trade.open_rate + (side * dist)

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float,
                        after_fill: bool, **kwargs) -> float | None:
        """
        Stop-loss dispatcher:
          - If TP1 already taken → move SL to break-even (entry + fee_buffer).
            Protects the remainder of the position risk-free.
          - Else if sl_mode=atr → ATR-adaptive SL.
          - Else (sl_mode=fixed) → fall through to self.stoploss.
        """
        # BE after TP1
        if (self.tp1_enabled.value
                and trade.get_custom_data('tp1_taken', False)):
            buffer = float(self.be_buffer_pct.value)
            side = -1 if trade.is_short else 1
            be_price = trade.open_rate * (1 + side * buffer)
            return stoploss_from_absolute(
                be_price,
                current_rate=current_rate,
                is_short=trade.is_short,
                leverage=trade.leverage,
            )

        # Normal SL dispatch
        if str(self.sl_mode.value) != 'atr':
            return None

        row = self._entry_row(trade, pair)
        atr = self._safe_float(row, 'atr_14')
        if atr is None:
            return None

        sl_distance = atr * float(self.sl_atr_mult.value)
        side = 1 if trade.is_short else -1
        sl_price = trade.open_rate + (side * sl_distance)

        return stoploss_from_absolute(
            sl_price,
            current_rate=current_rate,
            is_short=trade.is_short,
            leverage=trade.leverage,
        )

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float,
                              min_stake: float | None, max_stake: float,
                              current_entry_rate: float, current_exit_rate: float,
                              current_entry_profit: float, current_exit_profit: float,
                              **kwargs) -> float | tuple[float, str] | None:
        """Close tp1_amount_pct of position when price crosses TP1 target."""
        if not self.tp1_enabled.value:
            return None
        if trade.get_custom_data('tp1_taken', False):
            return None

        row = self._entry_row(trade, trade.pair)
        tp1 = self._tp1_price(trade, row)
        if tp1 is None:
            return None

        triggered = (
            current_rate <= tp1 if trade.is_short else current_rate >= tp1
        )
        if not triggered:
            return None

        pct = float(self.tp1_amount_pct.value) / 100.0
        # adjust_trade_position expects STAKE (margin) change, not notional.
        # Position value = amount * current_rate; stake = value / leverage.
        close_value = trade.amount * current_rate * pct
        close_stake = close_value / max(trade.leverage, 1.0)

        trade.set_custom_data('tp1_taken', True)
        return -close_stake, 'tp1_partial'

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime,
                    current_rate: float, current_profit: float,
                    **kwargs) -> str | None:
        """
        TP dispatch by tp_mode:
          fixed      → None (minimal_roi handles it)
          atr        → N × ATR(entry)
          structural → nearest opposing pool, with fallback to ATR if no pool
        """
        mode = str(self.tp_mode.value)
        if mode in ('fixed', 'none'):
            # fixed → minimal_roi handles it. none → no hard TP, trade runs
            # until CHoCH opposite, BE after TP1, or hard-cap SL.
            return None

        row = self._entry_row(trade, pair)
        tag = None
        tp_price = None

        if mode == 'structural':
            tp_price = self._structural_tp_price(trade, row)
            if tp_price is not None:
                tp_price = self._apply_margin(trade, tp_price)
                tag = 'tp_structural'
            else:
                # fallback
                tp_price = self._atr_tp_price(trade, row)
                tag = 'tp_structural_fallback_atr' if tp_price is not None else None
        elif mode == 'atr':
            tp_price = self._atr_tp_price(trade, row)
            tag = 'tp_atr' if tp_price is not None else None

        if tp_price is None:
            return None

        if trade.is_short:
            if current_rate <= tp_price:
                return tag
        else:
            if current_rate >= tp_price:
                return tag
        return None

    # =========================================================
    # Exit — opposite CHoCH (canon SMC)
    # =========================================================

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe
        df.loc[
            (df['internal_choch_bearish'] == 1) | (df['swing_choch_bearish'] == 1),
            'exit_long',
        ] = 1
        df.loc[
            (df['internal_choch_bullish'] == 1) | (df['swing_choch_bullish'] == 1),
            'exit_short',
        ] = 1
        return df
