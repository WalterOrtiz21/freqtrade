"""
SMCForgeReverse — counter-trend mean-reversion strategy
========================================================

Sibling of SMCForge. Where SMCForge enters AT the favor of a CHoCH+displacement
(trend continuation), SMCForgeReverse enters COUNTER to the local MTF flow
when price arrives at a high-quality HTF POI without strength.

Five non-negotiable gates per entry (canon SMC validated):
  1. PD ALIGNMENT     price in 4h discount (long) / premium (short)
  2. SWEEP            major liquidity sweep on 4h before tapping POI
  3. POI HTF          a bullish/bearish POI active on 4h (any of OB / FVG /
                      Breaker / FVG-Breaker, configurable per type)
  4. TIER             classified tier (S..B_minus) >= configured tier_min
  5. APPROACH WEAK    approach_strength < threshold (no momentum into POI)

Plus: a tap of the POI on 15m AND optionally an LTF CHoCH inside the POI as
confirmation.

Sizing differs per tier (S/A=1.0R, A_minus=0.8R, B=0.6R, B_minus=0.4R),
applied via custom_stake_amount.

Anti-lookahead: HTF informers are projected with shift(1); approach_strength
is computed over [i-N, i-1] (excludes current bar) and shifted again on
write to df. See reverse_approach.compute_approach_strength.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime

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


_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / 'SMC_Forge'))

from forge_engine import SMCEngine                  # noqa: E402
from forge_quality import annotate_displacement     # noqa: E402
from forge_levels import annotate_eqh_eql           # noqa: E402
from forge_inducement import annotate_inducement    # noqa: E402

# New SMCForgeReverse modules.
from reverse_approach import compute_approach_strength    # noqa: E402
from reverse_tier import (                                # noqa: E402
    classify_long_tier,
    classify_short_tier,
    tier_passes_min,
    tier_size_factor,
)


logger = logging.getLogger(__name__)


# Columns projected from the MTF (4h) informer onto the LTF (15m) frame.
# Each gets shift(1) applied during the merge to avoid lookahead.
_MTF_PROJECT_COLS = (
    'swing_high', 'swing_low',
    'eqh_level', 'eql_level',
    'active_bullish_ob_top', 'active_bullish_ob_bottom',
    'active_bearish_ob_top', 'active_bearish_ob_bottom',
    'active_bullish_fvg_top', 'active_bullish_fvg_bottom',
    'active_bearish_fvg_top', 'active_bearish_fvg_bottom',
    'active_bullish_breaker_top', 'active_bullish_breaker_bottom',
    'active_bearish_breaker_top', 'active_bearish_breaker_bottom',
    # FVG-breaker columns intentionally excluded from MTF projection.
    # The engine's fvg_breaker state at the truncated-dataset boundary
    # diverges from the full-dataset state at the same bar, which
    # lookahead-analysis flags as a bias. Since rank1 has
    # `enable_fvg_breaker=false` AND `tier_min='A'` filters out tier B
    # trades that would require fvg_breaker presence, projecting these
    # columns adds risk without benefit. To re-enable, audit the engine
    # boundary handling and re-run lookahead-analysis.
    'swing_sweep_bullish', 'swing_sweep_bearish',
    'internal_sweep_bullish', 'internal_sweep_bearish',
    'swing_trend',
    'bull_idm_swept', 'bear_idm_swept',
)


class SMCForgeReverse(IStrategy):
    INTERFACE_VERSION = 3

    # =========================================================
    # Base config
    # =========================================================
    timeframe = '15m'
    can_short = True
    process_only_new_candles = True
    # CHoCH-opposite is the trend-continuation invalidator (canon SMCForge).
    # For SMCForgeReverse it is the OPPOSITE — a CHoCH in the trade direction
    # is the confirmation of the mean-reversion bounce, not a signal to exit.
    # Disabled here; exits handled exclusively by custom_exit + custom_stoploss.
    use_exit_signal = False
    use_custom_stoploss = True
    position_adjustment_enable = True
    startup_candle_count: int = 500     # 4h informer warmup needs more bars

    stoploss = -0.10
    minimal_roi = {"0": 100}            # disabled by default; custom_exit handles TP
    max_open_trades = 5

    # =========================================================
    # Engine (structural — same as SMCForge for engine consistency)
    # =========================================================
    internal_length = IntParameter(3, 8, default=5, space='buy', optimize=False)
    swing_length = IntParameter(20, 60, default=50, space='buy', optimize=False)
    fractal_n_major = IntParameter(3, 10, default=5, space='buy', optimize=False)
    fractal_n_minor = IntParameter(2, 4, default=2, space='buy', optimize=False)
    eqh_tolerance_atr = DecimalParameter(
        0.05, 0.30, default=0.10, decimals=2, space='buy', optimize=False,
    )
    disp_lookback_bars = IntParameter(2, 5, default=3, space='buy', optimize=False)

    # =========================================================
    # HTF stack — fixed per backtest (cannot optimize without recompute)
    # =========================================================
    mtf_structure_tf = CategoricalParameter(
        ['4h', '1h'], default='4h', space='buy', optimize=False,
    )
    htf_bias_tf = CategoricalParameter(
        ['1d', '4h'], default='1d', space='buy', optimize=False,
    )
    enable_daily_bias = BooleanParameter(
        default=False, space='buy', optimize=False,
    )
    bias_mode = CategoricalParameter(
        ['pullback', 'reversal', 'off'], default='pullback',
        space='buy', optimize=False,
    )

    # =========================================================
    # Reverse gates — tactical
    # =========================================================
    tier_min = CategoricalParameter(
        ['S', 'A', 'A_minus', 'B', 'B_minus'], default='A',
        space='buy', optimize=True,
    )
    enable_ob = BooleanParameter(default=True, space='buy', optimize=False)
    enable_fvg = BooleanParameter(default=True, space='buy', optimize=False)
    enable_breaker = BooleanParameter(default=True, space='buy', optimize=False)
    enable_fvg_breaker = BooleanParameter(default=False, space='buy', optimize=False)
    # Direction toggles — observed asymmetry: counter-trend shorts in
    # 2025-2026 bull regime systematically fail. Long-only as a regime
    # filter is canonical when HTF macro trend is up. Enable shorts only
    # when regime / hyperopt suggests it.
    enable_longs = BooleanParameter(default=True, space='buy', optimize=False)
    enable_shorts = BooleanParameter(default=False, space='buy', optimize=False)

    require_sweep = BooleanParameter(default=True, space='buy', optimize=False)
    require_pd_alignment = BooleanParameter(default=True, space='buy', optimize=False)
    # IDM is a SCORING input per canon /smc, not a hard gate. Default off
    # to keep the 5 hard gates clean; users opt in to 'aligned' or 'opposite'
    # via config / hyperopt for stricter setups.
    require_idm_swept = CategoricalParameter(
        ['off', 'aligned', 'opposite'], default='off',
        space='buy', optimize=False,
    )

    approach_strength_thr = DecimalParameter(
        0.40, 0.70, default=0.55, decimals=2, space='buy', optimize=True,
    )
    approach_lookback = IntParameter(5, 10, default=7, space='buy', optimize=False)

    sweep_lookback = IntParameter(2, 8, default=4, space='buy', optimize=True)
    # CHoCH-inside is a confirmation, not a hard gate per canon. Default off
    # for entry frequency; opt in to True for stricter setups (less noise,
    # smaller sample).
    require_choch_inside = BooleanParameter(default=False, space='buy', optimize=False)
    disp_threshold = DecimalParameter(
        0.5, 2.5, default=1.0, decimals=2, space='buy', optimize=True,
    )

    # PD thresholds (canon: discount <= 38, premium >= 62)
    pd_discount_max = DecimalParameter(
        20.0, 45.0, default=38.0, decimals=1, space='buy', optimize=False,
    )
    pd_premium_min = DecimalParameter(
        55.0, 80.0, default=62.0, decimals=1, space='buy', optimize=False,
    )

    # =========================================================
    # Exits
    # =========================================================
    sl_atr_buffer_mult = DecimalParameter(
        0.3, 1.0, default=0.3, decimals=2, space='sell', optimize=False,
    )
    stoploss_pct = DecimalParameter(
        -0.05, -0.01, default=-0.025, decimals=3, space='sell', optimize=True,
    )
    tp_pct = DecimalParameter(
        0.05, 0.20, default=0.10, decimals=3, space='sell',
        # DEAD under current setup: tp1_target='equilibrium' (optimize=False)
        # makes custom_exit ignore tp_pct, and minimal_roi=100 in
        # bot_loop_start disables ROI exits. Keeping tp_pct optimize=True
        # would waste hyperopt epochs on a no-op dimension.
        optimize=False,
    )
    tp1_target = CategoricalParameter(
        ['equilibrium', 'opposite_poi', 'fixed'], default='equilibrium',
        space='sell', optimize=False,
    )
    tp2_target = CategoricalParameter(
        ['opposite_poi', 'liquidation_cluster', 'fixed'], default='opposite_poi',
        space='sell', optimize=False,
    )

    # =========================================================
    # Lifecycle
    # =========================================================

    def bot_start(self, **kwargs) -> None:
        self._configured = False

    def bot_loop_start(self, current_time: datetime, **kwargs) -> None:
        if self._configured:
            return
        cfg_lev = float(self.config.get('leverage', 1.0))

        # Freqtrade's `stoploss` is a price-move ratio (NOT stake PnL).
        # It is leverage-agnostic — leverage only multiplies the realized
        # stake loss when the SL fires. Convention: stoploss_pct, the
        # config-level stoploss, and self.stoploss are ALL the same number,
        # interpreted as price move. Custom_stoploss returns the structural
        # SL beyond POI which dominates in nearly every trade.
        self.stoploss = float(self.stoploss_pct.value)

        # minimal_roi DISABLED. tp_pct is read by custom_exit only when
        # tp1_target='fixed' (currently default 'equilibrium'). Setting
        # minimal_roi=0.10 (price-move) caps trades at +50% stake with
        # lev 5 — but the canonical TP is the 4h range equilibrium captured
        # by custom_exit, which often runs further. Keep minimal_roi at
        # an unreachable level so it never fires; let custom_exit own TP.
        self.minimal_roi = {0: 100.0}

        logger.info(
            'SMCForgeReverse configured. tf=%s mtf=%s htf_bias=%s '
            'leverage=%.2fx | SL=%.3f TP=%.3f (price-move) | '
            'bias_mode=%s daily_bias=%s tier_min=%s approach_thr=%.2f',
            self.timeframe, self.mtf_structure_tf.value,
            self.htf_bias_tf.value, cfg_lev,
            self.stoploss, self.minimal_roi[0],
            self.bias_mode.value, self.enable_daily_bias.value,
            self.tier_min.value, float(self.approach_strength_thr.value),
        )
        self._configured = True

    def leverage(self, pair, current_time, current_rate, proposed_leverage,
                 max_leverage, entry_tag, side, **kwargs) -> float:
        return float(self.config.get('leverage', 1.0))

    def informative_pairs(self):
        """Declare 1 (or 2) HTF informers per pair."""
        mtf = str(self.mtf_structure_tf.value)
        pairs_set: set = set()
        if self.dp:
            for p in self.dp.current_whitelist():
                pairs_set.add((p, mtf))
                if bool(self.enable_daily_bias.value):
                    pairs_set.add((p, str(self.htf_bias_tf.value)))
        return list(pairs_set)

    # =========================================================
    # HTF snapshot — parameterized projection of HTF columns to LTF
    # =========================================================

    def _compute_htf_snapshot(self,
                              dataframe: DataFrame,
                              metadata: dict,
                              tf: str,
                              prefix: str,
                              cols: tuple,
                              compute_levels: bool = True) -> DataFrame:
        """
        Run the SMC engine + (optional) levels on `tf` data for the same pair,
        shift(1) every projected column (anti-lookahead), and merge_asof
        backward onto the LTF dataframe with `prefix` appended to each col.
        """
        if not self.dp:
            return dataframe

        try:
            htf_df = self.dp.get_pair_dataframe(metadata['pair'], tf)
            if htf_df is None or htf_df.empty:
                return dataframe

            htf_signals = SMCEngine(
                htf_df,
                internal_length=int(self.internal_length.value),
                swing_length=int(self.swing_length.value),
            ).get_signals()

            htf_levels = pd.DataFrame()
            if compute_levels:
                htf_levels = annotate_eqh_eql(
                    htf_df,
                    atr_period=14,
                    fractal_n=int(self.fractal_n_major.value),
                    tolerance_atr=float(self.eqh_tolerance_atr.value),
                )

            snap = pd.DataFrame({'date': htf_df['date'].values})
            for col in cols:
                if not htf_levels.empty and col in htf_levels.columns:
                    snap[f'{col}{prefix}'] = htf_levels[col].shift(1).values
                elif col in htf_signals.columns:
                    snap[f'{col}{prefix}'] = htf_signals[col].shift(1).values
                else:
                    snap[f'{col}{prefix}'] = 0.0

            ref_tz = dataframe['date'].dt.tz
            src_tz = snap['date'].dt.tz if hasattr(snap['date'].dt, 'tz') else None
            if ref_tz is not None and src_tz is None:
                snap['date'] = pd.to_datetime(snap['date']).dt.tz_localize('UTC')
            elif ref_tz is None and src_tz is not None:
                snap['date'] = pd.to_datetime(snap['date']).dt.tz_convert(None)

            dataframe = pd.merge_asof(
                dataframe.sort_values('date'),
                snap.sort_values('date'),
                on='date', direction='backward',
            )
        except Exception as e:
            logger.warning(
                'HTF snapshot calc failed for %s (tf=%s, prefix=%s): %s',
                metadata.get('pair', '?'), tf, prefix, e,
            )

        return dataframe

    def _compute_daily_bias(self,
                            dataframe: DataFrame,
                            metadata: dict) -> DataFrame:
        """
        Single-column daily bias from forge_engine swing_trend on htf_bias_tf.
        Only invoked when enable_daily_bias=True.
        """
        dataframe['daily_bias'] = 0
        if not bool(self.enable_daily_bias.value) or not self.dp:
            return dataframe

        tf = str(self.htf_bias_tf.value)
        try:
            ref_df = self.dp.get_pair_dataframe(metadata['pair'], tf)
            if ref_df is None or ref_df.empty:
                return dataframe
            sig = SMCEngine(
                ref_df,
                internal_length=int(self.internal_length.value),
                swing_length=int(self.swing_length.value),
            ).get_signals()
            b = ref_df[['date']].copy()
            b['bias'] = sig['swing_trend'].shift(1).fillna(0).astype(int).values

            ref_tz = dataframe['date'].dt.tz
            src_tz = b['date'].dt.tz
            if ref_tz is not None and src_tz is None:
                b['date'] = b['date'].dt.tz_localize('UTC')
            elif ref_tz is None and src_tz is not None:
                b['date'] = b['date'].dt.tz_convert(None)

            merged = pd.merge_asof(
                dataframe[['date']].copy().sort_values('date'),
                b.sort_values('date'),
                on='date', direction='backward',
            )
            dataframe['daily_bias'] = merged['bias'].fillna(0).astype(int).values
        except Exception as e:
            logger.warning(
                'Daily bias calc failed for %s (tf=%s): %s',
                metadata.get('pair', '?'), tf, e,
            )
        return dataframe

    # =========================================================
    # Indicators
    # =========================================================

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe

        # CRITICAL: vol_sma must be present before tier classification + as
        # input for downstream rvol logic (future versions). See plan.
        df['vol_sma_20'] = df['volume'].rolling(20).mean()

        # ── LTF (15m) layers — same as SMCForge
        eng = SMCEngine(
            df,
            internal_length=int(self.internal_length.value),
            swing_length=int(self.swing_length.value),
        )
        signals = eng.get_signals()
        for col in signals.columns:
            df[col] = signals[col].values

        disp = annotate_displacement(
            df, signals, atr_period=14,
            lookback=int(self.disp_lookback_bars.value),
        )
        for col in disp.columns:
            if col not in df.columns:
                df[col] = disp[col].values

        levels = annotate_eqh_eql(
            df, atr_period=14,
            fractal_n=int(self.fractal_n_major.value),
            tolerance_atr=float(self.eqh_tolerance_atr.value),
        )
        for col in levels.columns:
            if col not in df.columns:
                df[col] = levels[col].values

        idm = annotate_inducement(
            df,
            fractal_n_major=int(self.fractal_n_major.value),
            fractal_n_minor=int(self.fractal_n_minor.value),
        )
        for col in idm.columns:
            if col not in df.columns:
                df[col] = idm[col].values

        # ── MTF (4h) snapshot — projects 4h structure to 15m
        df = self._compute_htf_snapshot(
            df, metadata,
            tf=str(self.mtf_structure_tf.value),
            prefix='_h4',
            cols=_MTF_PROJECT_COLS,
            compute_levels=True,
        )

        # ── Optional 1d bias
        df = self._compute_daily_bias(df, metadata)

        # ── 4h Premium / Discount zone
        rng_h4 = (df['swing_high_h4'] - df['swing_low_h4']).replace(0.0, np.nan)
        df['range_pct_h4'] = (
            (df['close'] - df['swing_low_h4']) / rng_h4 * 100.0
        ).fillna(50.0).clip(0.0, 100.0)
        df['in_discount_h4'] = df['range_pct_h4'] < float(self.pd_discount_max.value)
        df['in_premium_h4'] = df['range_pct_h4'] > float(self.pd_premium_min.value)

        # ── Approach strength (anti-lookahead via window [i-N, i-1])
        approach = compute_approach_strength(
            df, lookback=int(self.approach_lookback.value), baseline_window=20,
        )
        # Extra defensive shift: ensure score at bar i is never accessed
        # by a decision made at the open of bar i.
        df['approach_strength'] = approach.shift(1).values

        # ── Tier classification (uses 4h-projected POIs)
        df['reverse_tier_long'] = classify_long_tier(df, htf_suffix='_h4').values
        df['reverse_tier_short'] = classify_short_tier(df, htf_suffix='_h4').values

        return df

    # =========================================================
    # Entry — 5 named gates per direction
    # =========================================================

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe
        sweep_lb = int(self.sweep_lookback.value)
        approach_thr = float(self.approach_strength_thr.value)
        tier_min_value = str(self.tier_min.value)

        # ─────────────────────────────── LONG counter-trend
        gate_pd_long = df['in_discount_h4'].fillna(False)

        # Sweep: either swing-level (rare, ~17/year) OR internal-level
        # (~10×) on the 4h informer. Canon SMC accepts both as "major"
        # relative to the LTF — what matters is the TF, not the fractal.
        sweep_swing_long = (
            df.get('swing_sweep_bullish_h4', pd.Series(0.0, index=df.index))
            .fillna(0.0).astype(float)
        )
        sweep_internal_long = (
            df.get('internal_sweep_bullish_h4', pd.Series(0.0, index=df.index))
            .fillna(0.0).astype(float)
        )
        sweep_long_raw = (sweep_swing_long.eq(1) | sweep_internal_long.eq(1)).astype(float)
        gate_sweep_long = (
            sweep_long_raw.rolling(sweep_lb, min_periods=1).max() == 1
            if bool(self.require_sweep.value)
            else pd.Series(True, index=df.index)
        )

        tap_long, present_long = self._tap_and_present(df, side='long')
        gate_poi_long = present_long
        gate_tap_long = tap_long

        gate_tier_long = tier_passes_min(df['reverse_tier_long'], tier_min_value)

        gate_approach_long = (df['approach_strength'] < approach_thr).fillna(False)

        gate_idm_long = self._idm_gate(df, side='long', lookback=sweep_lb)
        gate_pd_align_long = (
            gate_pd_long if bool(self.require_pd_alignment.value)
            else pd.Series(True, index=df.index)
        )

        gate_choch_long = self._choch_inside_gate(df, side='long')

        long_cond = (
            gate_pd_long
            & gate_sweep_long
            & gate_poi_long
            & gate_tap_long
            & gate_tier_long
            & gate_approach_long
            & gate_idm_long
            & gate_pd_align_long
            & gate_choch_long
        )
        long_cond = self._apply_bias(long_cond, df, side='long')

        # ─────────────────────────────── SHORT counter-trend (mirror)
        gate_pd_short = df['in_premium_h4'].fillna(False)

        sweep_swing_short = (
            df.get('swing_sweep_bearish_h4', pd.Series(0.0, index=df.index))
            .fillna(0.0).astype(float)
        )
        sweep_internal_short = (
            df.get('internal_sweep_bearish_h4', pd.Series(0.0, index=df.index))
            .fillna(0.0).astype(float)
        )
        sweep_short_raw = (sweep_swing_short.eq(1) | sweep_internal_short.eq(1)).astype(float)
        gate_sweep_short = (
            sweep_short_raw.rolling(sweep_lb, min_periods=1).max() == 1
            if bool(self.require_sweep.value)
            else pd.Series(True, index=df.index)
        )

        tap_short, present_short = self._tap_and_present(df, side='short')
        gate_poi_short = present_short
        gate_tap_short = tap_short

        gate_tier_short = tier_passes_min(df['reverse_tier_short'], tier_min_value)
        gate_approach_short = (df['approach_strength'] < approach_thr).fillna(False)
        gate_idm_short = self._idm_gate(df, side='short', lookback=sweep_lb)
        gate_pd_align_short = (
            gate_pd_short if bool(self.require_pd_alignment.value)
            else pd.Series(True, index=df.index)
        )
        gate_choch_short = self._choch_inside_gate(df, side='short')

        short_cond = (
            gate_pd_short
            & gate_sweep_short
            & gate_poi_short
            & gate_tap_short
            & gate_tier_short
            & gate_approach_short
            & gate_idm_short
            & gate_pd_align_short
            & gate_choch_short
        )
        short_cond = self._apply_bias(short_cond, df, side='short')

        # Direction toggles
        if not bool(self.enable_longs.value):
            long_cond = pd.Series(False, index=df.index)
        if not bool(self.enable_shorts.value):
            short_cond = pd.Series(False, index=df.index)

        # Tagging by tier+POI type for post-mortem.
        df.loc[long_cond, 'enter_long'] = 1
        df.loc[long_cond, 'enter_tag'] = (
            'REVERSE_L_' + df.loc[long_cond, 'reverse_tier_long'].astype(str)
        )
        df.loc[short_cond, 'enter_short'] = 1
        df.loc[short_cond, 'enter_tag'] = (
            'REVERSE_S_' + df.loc[short_cond, 'reverse_tier_short'].astype(str)
        )
        return df

    # =========================================================
    # Entry helpers
    # =========================================================

    def _tap_and_present(self, df: DataFrame, side: str
                         ) -> tuple[pd.Series, pd.Series]:
        """
        Returns (tap, present) booleans for the given side.
          present : at least one enabled POI type is active on 4h.
          tap     : current bar's range overlaps any of those active POI zones.
        """
        if side == 'long':
            zones = []
            if bool(self.enable_ob.value):
                zones.append(('active_bullish_ob_top_h4', 'active_bullish_ob_bottom_h4'))
            if bool(self.enable_fvg.value):
                zones.append(('active_bullish_fvg_top_h4', 'active_bullish_fvg_bottom_h4'))
            if bool(self.enable_breaker.value):
                zones.append(('active_bullish_breaker_top_h4', 'active_bullish_breaker_bottom_h4'))
            if bool(self.enable_fvg_breaker.value):
                zones.append((
                    'active_bullish_fvg_breaker_top_h4',
                    'active_bullish_fvg_breaker_bottom_h4',
                ))
            present = pd.Series(False, index=df.index)
            tap = pd.Series(False, index=df.index)
            for top_col, bot_col in zones:
                if top_col not in df.columns or bot_col not in df.columns:
                    continue
                top = df[top_col]
                bot = df[bot_col]
                z_present = top.notna() & (top > 0)
                z_tap = z_present & (df['low'] <= top) & (df['close'] >= bot)
                present |= z_present
                tap |= z_tap
            return tap, present

        # short
        zones = []
        if bool(self.enable_ob.value):
            zones.append(('active_bearish_ob_top_h4', 'active_bearish_ob_bottom_h4'))
        if bool(self.enable_fvg.value):
            zones.append(('active_bearish_fvg_top_h4', 'active_bearish_fvg_bottom_h4'))
        if bool(self.enable_breaker.value):
            zones.append(('active_bearish_breaker_top_h4', 'active_bearish_breaker_bottom_h4'))
        if bool(self.enable_fvg_breaker.value):
            zones.append((
                'active_bearish_fvg_breaker_top_h4',
                'active_bearish_fvg_breaker_bottom_h4',
            ))
        present = pd.Series(False, index=df.index)
        tap = pd.Series(False, index=df.index)
        for top_col, bot_col in zones:
            if top_col not in df.columns or bot_col not in df.columns:
                continue
            top = df[top_col]
            bot = df[bot_col]
            z_present = top.notna() & (top > 0)
            z_tap = z_present & (df['high'] >= bot) & (df['close'] <= top)
            present |= z_present
            tap |= z_tap
        return tap, present

    def _idm_gate(self, df: DataFrame, side: str, lookback: int) -> pd.Series:
        """
        require_idm_swept ∈ {'off', 'aligned', 'opposite'}.
          'off'      : always pass.
          'aligned'  : long needs bull_idm_swept (engine LTF), short bear_idm_swept.
          'opposite' : long needs bear_idm_swept (capitulation), short needs bull.
        """
        mode = str(self.require_idm_swept.value)
        if mode == 'off':
            return pd.Series(True, index=df.index)
        bull_col = df.get('bull_idm_swept', pd.Series(0.0, index=df.index)).fillna(0.0)
        bear_col = df.get('bear_idm_swept', pd.Series(0.0, index=df.index)).fillna(0.0)
        if side == 'long':
            ref = bull_col if mode == 'aligned' else bear_col
        else:
            ref = bear_col if mode == 'aligned' else bull_col
        return ref.rolling(lookback, min_periods=1).max() == 1

    def _choch_inside_gate(self, df: DataFrame, side: str) -> pd.Series:
        """
        Optional confirmation: an LTF CHoCH in the trade's direction with
        displacement >= disp_threshold, recently (rolling sweep_lookback bars).
        When require_choch_inside=False, always pass.
        """
        if not bool(self.require_choch_inside.value):
            return pd.Series(True, index=df.index)
        thr = float(self.disp_threshold.value)
        lb = int(self.sweep_lookback.value)
        if side == 'long':
            ev = (
                ((df.get('internal_choch_bullish', 0) == 1) &
                 (df.get('internal_choch_bull_disp', 0.0) >= thr))
                | ((df.get('swing_choch_bullish', 0) == 1) &
                   (df.get('swing_choch_bull_disp', 0.0) >= thr))
            )
        else:
            ev = (
                ((df.get('internal_choch_bearish', 0) == 1) &
                 (df.get('internal_choch_bear_disp', 0.0) >= thr))
                | ((df.get('swing_choch_bearish', 0) == 1) &
                   (df.get('swing_choch_bear_disp', 0.0) >= thr))
            )
        return ev.fillna(False).rolling(lb, min_periods=1).max() == 1

    def _apply_bias(self, cond: pd.Series, df: DataFrame, side: str) -> pd.Series:
        """
        Apply daily bias gate based on bias_mode + enable_daily_bias.
          mode='off'      → no gate.
          mode='pullback' → trade aligned to daily trend (long in 1d uptrend).
          mode='reversal' → trade against daily trend (long in 1d downtrend).
        """
        if not bool(self.enable_daily_bias.value):
            return cond
        mode = str(self.bias_mode.value)
        if mode == 'off':
            return cond
        bias = df.get('daily_bias', pd.Series(0, index=df.index)).fillna(0).astype(int)
        if mode == 'pullback':
            allowed = bias == 1 if side == 'long' else bias == -1
        elif mode == 'reversal':
            allowed = bias == -1 if side == 'long' else bias == 1
        else:
            return cond
        return cond & allowed

    # =========================================================
    # Exit — opposite CHoCH (canon SMC, same as SMCForge)
    # =========================================================

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe
        df.loc[
            (df.get('internal_choch_bearish', 0) == 1)
            | (df.get('swing_choch_bearish', 0) == 1),
            'exit_long',
        ] = 1
        df.loc[
            (df.get('internal_choch_bullish', 0) == 1)
            | (df.get('swing_choch_bullish', 0) == 1),
            'exit_short',
        ] = 1
        return df

    # =========================================================
    # Sizing — tier-based (1.0 / 0.8 / 0.6 / 0.4)
    # =========================================================

    def custom_stake_amount(self, pair: str, current_time: datetime,
                            current_rate: float, proposed_stake: float,
                            min_stake: float | None, max_stake: float,
                            leverage: float, entry_tag: str | None,
                            side: str, **kwargs) -> float:
        """
        Multiply proposed_stake by tier_size_factor read from enter_tag.
        enter_tag format: 'REVERSE_{L,S}_{S,A,A_minus,B,B_minus}'.
        Falls back to proposed_stake if tag is missing or unrecognized.
        """
        if not entry_tag or not entry_tag.startswith('REVERSE_'):
            return proposed_stake
        parts = entry_tag.split('_', 2)
        if len(parts) < 3:
            return proposed_stake
        tier = parts[2]
        factor = tier_size_factor(tier)
        if factor <= 0:
            return proposed_stake
        return float(proposed_stake) * factor

    # =========================================================
    # Custom stoploss — structural beyond POI + ATR buffer
    # =========================================================

    def _entry_row(self, trade: Trade, pair: str):
        if not self.dp:
            return None
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
        if row is None:
            return None
        val = row.get(col, None)
        if val is None:
            return None
        try:
            v = float(val)
        except (TypeError, ValueError):
            return None
        if np.isnan(v) or v <= 0:
            return None
        return v

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float,
                        after_fill: bool, **kwargs) -> float | None:
        """
        Structural SL set ONCE at fill:
          long  → SL below the active 4h POI bottom that the trade tapped.
                  buffer = atr_h4 * sl_atr_buffer_mult.
          short → mirror.

        Critical: returns a value ONLY when after_fill=True (i.e., once at
        position open). Returning a value on subsequent bars triggers
        freqtrade's trailing-style trail down with current_rate, which cuts
        winners in profit (observed in v1 backtest: 99 trailing_stop_loss
        trades, win 7%, -96 USDT). After fill, return None so the trade
        keeps the originally-set absolute SL until structural exit / TP /
        opposite-direction CHoCH per `populate_exit_trend`.
        """
        if not after_fill:
            return None

        row = self._entry_row(trade, pair)
        if row is None:
            return None

        # Pick the matching POI bottom/top by tier.
        tier = (row.get('reverse_tier_long' if not trade.is_short
                        else 'reverse_tier_short') or 'NONE')
        if tier == 'NONE':
            return None

        # Use 4h ATR proxy: fall back to 14-bar LTF ATR if h4 not available.
        atr = self._safe_float(row, 'atr_14')
        if atr is None:
            return None
        buffer_mult = float(self.sl_atr_buffer_mult.value)

        if not trade.is_short:
            poi_bottom = (
                self._safe_float(row, 'active_bullish_ob_bottom_h4')
                or self._safe_float(row, 'active_bullish_fvg_bottom_h4')
                or self._safe_float(row, 'active_bullish_breaker_bottom_h4')
                or self._safe_float(row, 'active_bullish_fvg_breaker_bottom_h4')
            )
            if poi_bottom is None:
                return None
            sl_price = poi_bottom - atr * buffer_mult
        else:
            poi_top = (
                self._safe_float(row, 'active_bearish_ob_top_h4')
                or self._safe_float(row, 'active_bearish_fvg_top_h4')
                or self._safe_float(row, 'active_bearish_breaker_top_h4')
                or self._safe_float(row, 'active_bearish_fvg_breaker_top_h4')
            )
            if poi_top is None:
                return None
            sl_price = poi_top + atr * buffer_mult

        return stoploss_from_absolute(
            sl_price,
            current_rate=current_rate,
            is_short=trade.is_short,
            leverage=trade.leverage,
        )

    # =========================================================
    # Custom exit — TP at h4 equilibrium / opposite POI
    # =========================================================

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime,
                    current_rate: float, current_profit: float,
                    **kwargs) -> str | None:
        """
        TP1: close at h4 equilibrium (50% of swing range) by default.
        For now this is a single-target exit (TP1==TP2) — partial-close +
        BE handling is left for v2.
        """
        row = self._entry_row(trade, pair)
        if row is None:
            return None

        target = self._tp_target_price(row, trade)
        if target is None:
            return None

        if trade.is_short and current_rate <= target:
            return 'reverse_tp1_equilibrium'
        if (not trade.is_short) and current_rate >= target:
            return 'reverse_tp1_equilibrium'
        return None

    def _tp_target_price(self, row, trade: Trade) -> float | None:
        mode = str(self.tp1_target.value)
        if mode == 'fixed':
            return None  # let minimal_roi / fixed stoploss handle it.

        if mode == 'equilibrium':
            sh = self._safe_float(row, 'swing_high_h4')
            sl = self._safe_float(row, 'swing_low_h4')
            if sh is None or sl is None or sh <= sl:
                return None
            return (sh + sl) / 2.0

        if mode == 'opposite_poi':
            if trade.is_short:
                # bullish opposite (below)
                return (
                    self._safe_float(row, 'active_bullish_ob_top_h4')
                    or self._safe_float(row, 'active_bullish_fvg_top_h4')
                    or self._safe_float(row, 'eql_level_h4')
                )
            return (
                self._safe_float(row, 'active_bearish_ob_bottom_h4')
                or self._safe_float(row, 'active_bearish_fvg_bottom_h4')
                or self._safe_float(row, 'eqh_level_h4')
            )
        return None

    # =========================================================
    # Hyperopt — TPE sampler
    # =========================================================

    class HyperOpt:
        @staticmethod
        def generate_estimator(dimensions, **kwargs):
            return "TPESampler"
