"""
SMCRange — estrategia para régimen chop/rango.
Fade extremos (sweep EQH/EQL + reclaim) + breakout-flip, toggleable.
Detección adaptativa multi-TF {1h, 4h}; TF base de ejecución 1h.
Arregla las 3 fallas de SMCForgeReverse: gate de régimen, TP2 real, gate R:R.
"""
import logging
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from pandas import DataFrame

from freqtrade.strategy import (
    BooleanParameter, CategoricalParameter, DecimalParameter, IntParameter,
    IStrategy, stoploss_from_absolute,
)
from freqtrade.exchange import timeframe_to_prev_date
from freqtrade.persistence import Trade

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / 'SMC_Forge'))

from forge_engine import SMCEngine                 # noqa: E402
from forge_quality import annotate_displacement    # noqa: E402
from forge_levels import annotate_eqh_eql          # noqa: E402
from range_detector import annotate_range          # noqa: E402
from range_entries import (                         # noqa: E402
    fade_long, fade_short, breakout_long, breakout_short,
    reclaim_long, reclaim_short, rr_long_fade, rr_short_fade,
)

logger = logging.getLogger(__name__)

# Columnas que se proyectan del informer 4h al 1h (con shift(1)).
_RANGE_PROJECT_COLS = (
    'range_active', 'range_top', 'range_bottom', 'range_mid',
    'range_width_pct', 'range_quality',
)


class SMCRange(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = '1h'
    can_short = True
    process_only_new_candles = True
    use_exit_signal = False          # exits 100% por callbacks
    use_custom_stoploss = True
    position_adjustment_enable = True
    startup_candle_count: int = 400  # warmup engine 4h + swing_length

    stoploss = -0.10
    minimal_roi = {"0": 100}
    max_open_trades = 8

    # Engine (estructural, optimize=False)
    internal_length = IntParameter(3, 8, default=5, space='buy', optimize=False)
    swing_length = IntParameter(20, 60, default=50, space='buy', optimize=False)
    fractal_n_major = IntParameter(3, 10, default=5, space='buy', optimize=False)
    eqh_tolerance_atr = DecimalParameter(0.05, 0.30, default=0.10, decimals=2, space='buy', optimize=False)
    disp_lookback_bars = IntParameter(2, 5, default=3, space='buy', optimize=False)

    # HTF stack (fixed)
    htf_tf = CategoricalParameter(['4h', '1d'], default='4h', space='buy', optimize=False)

    # Modos (fixed — no hyperopt para evitar conditional-param waste)
    range_mode = CategoricalParameter(['fade_only', 'breakout_only', 'hybrid'],
                                      default='fade_only', space='buy', optimize=False)
    enable_longs = BooleanParameter(default=True, space='buy', optimize=False)
    enable_shorts = BooleanParameter(default=True, space='buy', optimize=False)
    require_reclaim = BooleanParameter(default=True, space='buy', optimize=False)

    # Detector: optimize=False OBLIGATORIO — se consumen en _engine_plus_range()
    # llamado desde populate_indicators, que bajo hyperopt corre UNA sola vez
    # (cacheado entre épocas). Marcarlos optimize=True = trap AP-1b silente: el
    # buy-space sería un no-op que converge a ruido. Para hyperoptearlos de verdad
    # habría que refactorear con .range (costoso). Mismo blindaje que SMCForge:135-137.
    bos_lookback = IntParameter(10, 40, default=20, space='buy', optimize=False)
    width_min_pct = DecimalParameter(0.010, 0.050, default=0.015, decimals=3, space='buy', optimize=False)
    width_max_pct = DecimalParameter(0.15, 0.40, default=0.30, decimals=2, space='buy', optimize=False)
    containment_min = DecimalParameter(0.50, 0.90, default=0.70, decimals=2, space='buy', optimize=False)
    # reclaim_lookback SÍ es hyperopteable: se lee en populate_entry_trend (re-corre por época).
    reclaim_lookback = IntParameter(2, 6, default=3, space='buy', optimize=True)

    # Exits (tácticos)
    sl_atr_buffer = DecimalParameter(0.2, 1.0, default=0.4, decimals=2, space='sell', optimize=True)
    rr_min = DecimalParameter(1.2, 2.5, default=1.5, decimals=2, space='sell', optimize=True)
    tp1_amount_pct = IntParameter(30, 70, default=50, space='sell', optimize=False)
    be_buffer_pct = DecimalParameter(0.0, 0.005, default=0.001, decimals=4, space='sell', optimize=False)

    # =========================================================
    # Lifecycle (patrón SMCForge: params se leen tarde en backtest)
    # =========================================================
    def bot_start(self, **kwargs) -> None:
        self._configured = False

    def bot_loop_start(self, current_time: datetime, **kwargs) -> None:
        if self._configured:
            return
        cfg_lev = float(self.config.get('leverage', 1.0))
        self.stoploss = -0.10 * cfg_lev          # hard-cap si custom_stoploss da None
        self.minimal_roi = {0: 100.0}            # custom_exit/adjust son dueños del TP
        logger.info('SMCRange configured. tf=%s htf=%s mode=%s L/S=%s/%s rr_min=%.2f',
                    self.timeframe, self.htf_tf.value, self.range_mode.value,
                    self.enable_longs.value, self.enable_shorts.value, float(self.rr_min.value))
        self._configured = True

    def leverage(self, pair, current_time, current_rate, proposed_leverage,
                 max_leverage, entry_tag, side, **kwargs) -> float:
        return float(self.config.get('leverage', 1.0))

    def informative_pairs(self):
        htf = str(self.htf_tf.value)
        pairs_set = set()
        if self.dp:
            for p in self.dp.current_whitelist():
                pairs_set.add((p, htf))
        return list(pairs_set)

    # =========================================================
    # Cómputo de rango por TF
    # =========================================================
    def _engine_plus_range(self, base_df: DataFrame) -> DataFrame:
        """Corre engine+levels+displacement+annotate_range sobre un df de un TF.
        Devuelve el df con columnas del engine, atr_14, y range_*."""
        df = base_df.copy()
        sig = SMCEngine(df, internal_length=int(self.internal_length.value),
                        swing_length=int(self.swing_length.value)).get_signals()
        for c in sig.columns:
            df[c] = sig[c].values
        lvl = annotate_eqh_eql(df, atr_period=14, fractal_n=int(self.fractal_n_major.value),
                               tolerance_atr=float(self.eqh_tolerance_atr.value))
        for c in lvl.columns:
            if c not in df.columns:
                df[c] = lvl[c].values
        disp = annotate_displacement(df, sig, atr_period=14, lookback=int(self.disp_lookback_bars.value))
        for c in disp.columns:
            if c not in df.columns:
                df[c] = disp[c].values
        rng = annotate_range(df, bos_lookback=int(self.bos_lookback.value),
                             width_min_pct=float(self.width_min_pct.value),
                             width_max_pct=float(self.width_max_pct.value),
                             containment_min=float(self.containment_min.value))
        for c in rng.columns:
            df[c] = rng[c].values
        return df

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 1) Rango en el TF base (1h)
        df = self._engine_plus_range(dataframe)

        # 2) Rango en HTF (4h) proyectado con shift(1)+merge_asof (anti-lookahead)
        df = self._project_htf_range(df, metadata)

        # 3) Selección adaptativa de TF: preferir 4h si su rango está activo.
        htf_active = df.get('range_active_htf', pd.Series(0, index=df.index)).fillna(0) == 1
        for col in _RANGE_PROJECT_COLS:
            htf_col = f'{col}_htf'
            if htf_col in df.columns:
                df[col] = np.where(htf_active, df[htf_col], df[col])
        df['range_tf'] = np.where(htf_active, 4, 1)  # 4=4h ganó, 1=1h
        df['range_active'] = pd.Series(df['range_active'], index=df.index).fillna(0).astype(int)
        return df

    def _project_htf_range(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if not self.dp:
            for c in _RANGE_PROJECT_COLS:
                dataframe[f'{c}_htf'] = 0.0 if c == 'range_active' else np.nan
            return dataframe
        htf = str(self.htf_tf.value)
        try:
            htf_df = self.dp.get_pair_dataframe(metadata['pair'], htf)
            if htf_df is None or htf_df.empty:
                raise ValueError('htf vacío')
            htf_full = self._engine_plus_range(htf_df)
            snap = pd.DataFrame({'date': htf_full['date'].values})
            for c in _RANGE_PROJECT_COLS:
                snap[f'{c}_htf'] = htf_full[c].shift(1).values
            ref_tz = dataframe['date'].dt.tz
            src_tz = snap['date'].dt.tz if hasattr(snap['date'].dt, 'tz') else None
            if ref_tz is not None and src_tz is None:
                snap['date'] = pd.to_datetime(snap['date']).dt.tz_localize('UTC')
            elif ref_tz is None and src_tz is not None:
                snap['date'] = pd.to_datetime(snap['date']).dt.tz_convert(None)
            dataframe = pd.merge_asof(dataframe.sort_values('date'),
                                      snap.sort_values('date'), on='date', direction='backward')
        except Exception as e:
            logger.warning('HTF range proj falló %s (htf=%s): %s', metadata.get('pair', '?'), htf, e)
            for c in _RANGE_PROJECT_COLS:
                dataframe[f'{c}_htf'] = 0.0 if c == 'range_active' else np.nan
        return dataframe

    # =========================================================
    # Entradas — modos toggleables + gate R:R + tags
    # =========================================================
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe
        mode = str(self.range_mode.value)
        rr_min = float(self.rr_min.value)
        sl_buf = float(self.sl_atr_buffer.value)
        rclb = int(self.reclaim_lookback.value)

        do_fade = mode in ('fade_only', 'hybrid')
        do_brk = mode in ('breakout_only', 'hybrid')

        # ---- FADE ----
        fade_l = fade_long(df) if do_fade else pd.Series(False, index=df.index)
        fade_s = fade_short(df) if do_fade else pd.Series(False, index=df.index)
        if do_fade and bool(self.require_reclaim.value):
            fade_l = fade_l & reclaim_long(df, lookback=rclb)
            fade_s = fade_s & reclaim_short(df, lookback=rclb)
        if do_fade:
            rr_l = rr_long_fade(df, atr_col='atr_14', sl_buffer=sl_buf)
            rr_s = rr_short_fade(df, atr_col='atr_14', sl_buffer=sl_buf)
            fade_l = fade_l & (rr_l >= rr_min)
            fade_s = fade_s & (rr_s >= rr_min)

        # ---- BREAKOUT ----
        brk_l = breakout_long(df) if do_brk else pd.Series(False, index=df.index)
        brk_s = breakout_short(df) if do_brk else pd.Series(False, index=df.index)

        long_cond = (fade_l | brk_l)
        short_cond = (fade_s | brk_s)
        if not bool(self.enable_longs.value):
            long_cond = pd.Series(False, index=df.index)
        if not bool(self.enable_shorts.value):
            short_cond = pd.Series(False, index=df.index)

        # Tags: prioridad FADE > BRK (fade es el edge primario).
        df.loc[long_cond, 'enter_long'] = 1
        df.loc[fade_l, 'enter_tag'] = 'FADE_L'
        df.loc[brk_l & ~fade_l, 'enter_tag'] = 'BRK_L'
        df.loc[short_cond, 'enter_short'] = 1
        df.loc[fade_s, 'enter_tag'] = 'FADE_S'
        df.loc[brk_s & ~fade_s, 'enter_tag'] = 'BRK_S'
        return df

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe  # exits 100% por callbacks

    # =========================================================
    # Exit helpers
    # =========================================================
    @staticmethod
    def _safe_float(row, col: str):
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

    def _entry_row(self, trade: Trade, pair: str):
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or dataframe.empty:
            return None
        entry_date = timeframe_to_prev_date(self.timeframe, trade.open_date_utc)
        row = dataframe[dataframe['date'] == entry_date]
        if row.empty:
            return None
        return row.iloc[0]

    def _is_fade(self, trade: Trade) -> bool:
        return (trade.enter_tag or '').startswith('FADE')

    def _sl_price(self, trade: Trade, row):
        """SL estructural: fade = más allá del barrido; breakout = de vuelta al rango."""
        atr = self._safe_float(row, 'atr_14')
        buf = float(self.sl_atr_buffer.value)
        if self._is_fade(trade):
            if not trade.is_short:
                lo = self._safe_float(row, 'low')
                return None if (lo is None or atr is None) else lo - atr * buf
            hi = self._safe_float(row, 'high')
            return None if (hi is None or atr is None) else hi + atr * buf
        # breakout: SL al medio del rango roto
        return self._safe_float(row, 'range_mid')

    def _tp1_price(self, trade: Trade, row):
        if self._is_fade(trade):
            return self._safe_float(row, 'range_mid')
        top = self._safe_float(row, 'range_top')
        bot = self._safe_float(row, 'range_bottom')
        if top is None or bot is None:
            return None
        height = top - bot
        return (trade.open_rate - height * 0.5) if trade.is_short else (trade.open_rate + height * 0.5)

    def _tp2_price(self, trade: Trade, row):
        if self._is_fade(trade):
            return self._safe_float(row, 'range_top') if not trade.is_short \
                else self._safe_float(row, 'range_bottom')
        top = self._safe_float(row, 'range_top')
        bot = self._safe_float(row, 'range_bottom')
        if top is None or bot is None:
            return None
        height = top - bot
        return (trade.open_rate - height) if trade.is_short else (trade.open_rate + height)

    # =========================================================
    # custom_stoploss: estructural una vez + BE post-TP1
    # =========================================================
    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float,
                        after_fill: bool, **kwargs):
        if trade.get_custom_data('tp1_taken', False):
            buffer = float(self.be_buffer_pct.value)
            side = -1 if trade.is_short else 1
            be_price = trade.open_rate * (1 + side * buffer)
            return stoploss_from_absolute(be_price, current_rate=current_rate,
                                          is_short=trade.is_short, leverage=trade.leverage)
        if not after_fill:
            return None  # set SL una sola vez (fix del reverse: nada de trailing-down)
        row = self._entry_row(trade, pair)
        sl_price = self._sl_price(trade, row)
        if sl_price is None:
            return None
        return stoploss_from_absolute(sl_price, current_rate=current_rate,
                                      is_short=trade.is_short, leverage=trade.leverage)

    # =========================================================
    # TP1 parcial
    # =========================================================
    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float,
                              min_stake, max_stake, current_entry_rate,
                              current_exit_rate, current_entry_profit,
                              current_exit_profit, **kwargs):
        if trade.get_custom_data('tp1_taken', False):
            return None
        row = self._entry_row(trade, trade.pair)
        tp1 = self._tp1_price(trade, row)
        if tp1 is None:
            return None
        triggered = (current_rate <= tp1) if trade.is_short else (current_rate >= tp1)
        if not triggered:
            return None
        pct = float(self.tp1_amount_pct.value) / 100.0
        close_stake = (trade.amount * current_rate * pct) / max(trade.leverage, 1.0)
        trade.set_custom_data('tp1_taken', True)
        return -close_stake, 'tp1_partial'

    # =========================================================
    # TP2 final (CABLEADO DE VERDAD — fix del reverse)
    # =========================================================
    def custom_exit(self, pair: str, trade: Trade, current_time: datetime,
                    current_rate: float, current_profit: float, **kwargs):
        row = self._entry_row(trade, pair)
        tp2 = self._tp2_price(trade, row)
        if tp2 is None:
            return None
        if trade.is_short and current_rate <= tp2:
            return 'tp2_opposite_edge'
        if (not trade.is_short) and current_rate >= tp2:
            return 'tp2_opposite_edge'
        return None

    # =========================================================
    # Hyperopt sampler — TPE (consistente con SMCForge)
    # =========================================================
    class HyperOpt:
        @staticmethod
        def generate_estimator(dimensions, **kwargs):
            return "TPESampler"
