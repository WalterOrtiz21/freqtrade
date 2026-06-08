"""
MomentumRider — Trend-following / breakout long+short, de alta actividad.

Tesis de edge (distinta a TODO lo del repo, que es mean-reversion / SMC):
  El crypto tiene persistencia de tendencia intradía. En vez de fadear extremos
  (lo que hace AtrRsiScalper y falló por costos), MomentumRider COMPRA fuerza y
  VENDE debilidad:
    - Filtro macro 1h (precio vs EMA200 1h) → solo opera a favor del régimen.
    - Stack EMA alineado (20>50>200) + ADX>=22 creciente → evita el chop.
    - Disparo: breakout de Donchian (cruce del máx/mín de N velas) con volumen.
  Rentabilidad por R:R ASIMÉTRICO, no por win rate (~30%): se corta el perdedor
  barato con un stop FIJO de precio (stop_pct) y se deja correr al ganador hasta
  un timeout de N velas. SIN trailing — se midió que cortaba ganancias.

  ⚠️ El edge es FINO y SENSIBLE AL LEVERAGE: los fees escalan con el notional.
     Validado lev1-2 (all-weather, +PF en bull y bear). lev>=3 lo destruye por
     fee drag en años de churn (2025: lev1 +10% vs lev3 -63%). Ver doc en
     strategies/MomentumRider.md. NO subir leverage sin re-backtest del año stress.

15m base + 1h informativo. Futures Bitget, can_short, leverage desde config.
"""
import logging
from datetime import datetime

import numpy as np
from pandas import DataFrame

from freqtrade.strategy import (
    IStrategy, IntParameter, DecimalParameter, BooleanParameter,
    stoploss_from_absolute, merge_informative_pair,
)
from freqtrade.persistence import Trade
import talib.abstract as ta

logger = logging.getLogger(__name__)


class MomentumRider(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = '15m'
    can_short = True
    process_only_new_candles = True
    use_exit_signal = True
    use_custom_stoploss = True
    position_adjustment_enable = False
    # 1h EMA200 = 200 velas 1h = 800 velas 15m de warmup + colchón.
    startup_candle_count: int = 850

    # Backstop duro; el stop operativo es el fijo de precio en custom_stoploss (stop_pct).
    stoploss = -0.30
    # ROI desactivado: NO capar al ganador (lo cierra el timeout o el stop fijo).
    minimal_roi = {"0": 10.0}

    # ==========================================================================
    # PARÁMETROS HYPEROPTIMIZABLES (defaults manuales; sin hyperopt aún)
    # ==========================================================================
    # Entrada
    don_period = IntParameter(15, 60, default=20, space='buy', optimize=True)
    adx_min = IntParameter(18, 40, default=22, space='buy', optimize=True)
    vol_factor = DecimalParameter(1.0, 2.5, default=1.2, decimals=1, space='buy', optimize=True)
    min_atr_pct = DecimalParameter(0.002, 0.012, default=0.004, decimals=4, space='buy', optimize=True)
    max_atr_pct = DecimalParameter(0.020, 0.080, default=0.050, decimals=3, space='buy', optimize=True)
    use_htf_filter = BooleanParameter(default=True, space='buy', optimize=True)
    enable_short = BooleanParameter(default=True, space='buy', optimize=True)

    # Salida: stop fijo de PRECIO (corta perdedores barato) + hold hasta timeout (deja correr
    # ganadores). SIN trailing — se midió que cortaba ganancias (timeouts ganan, trailing sangra).
    stop_pct = DecimalParameter(0.02, 0.06, default=0.03, decimals=3, space='sell', optimize=True)
    max_candles_hold = IntParameter(96, 480, default=288, space='sell', optimize=True)

    # Circuit breaker: tras N pérdidas consecutivas (global), pausar entradas cooldown_h horas.
    max_consec_losses = IntParameter(4, 12, default=6, space='protection', optimize=False)
    cooldown_h = IntParameter(4, 48, default=12, space='protection', optimize=False)

    # ==========================================================================
    # LIFECYCLE
    # ==========================================================================
    def bot_start(self, **kwargs) -> None:
        self.config_leverage = float(self.config.get('leverage', 1.0))
        logger.info(f"MomentumRider init — leverage {self.config_leverage}x")

    def leverage(self, pair, current_time, current_rate, proposed_leverage,
                 max_leverage, entry_tag, side, **kwargs) -> float:
        return min(float(self.config.get('leverage', 1.0)), max_leverage)

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        return [(pair, '1h') for pair in pairs]

    # ==========================================================================
    # INDICADORES
    # ==========================================================================
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['ema_fast'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['ema_slow'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        dataframe['adx_rising'] = dataframe['adx'] > dataframe['adx'].shift(3)
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['atr_pct'] = dataframe['atr'] / dataframe['close']
        dataframe['vol_sma'] = dataframe['volume'].rolling(20).mean()

        # Donchian: máximo/mínimo de las N velas PREVIAS (shift(1) evita lookahead intrabar).
        dataframe['don_hi'] = dataframe['high'].rolling(self.don_period.value).max().shift(1)
        dataframe['don_lo'] = dataframe['low'].rolling(self.don_period.value).min().shift(1)

        # Informativo 1h: filtro macro de régimen.
        if self.dp:
            inf = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe='1h')
            if not inf.empty:
                inf['ema200'] = ta.EMA(inf, timeperiod=200)
                dataframe = merge_informative_pair(dataframe, inf, self.timeframe, '1h', ffill=True)
                dataframe['ema200_1h'] = dataframe['ema200_1h']
            else:
                dataframe['ema200_1h'] = np.nan
        else:
            dataframe['ema200_1h'] = np.nan

        return dataframe

    # ==========================================================================
    # ENTRADA
    # ==========================================================================
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        dataframe['enter_tag'] = ''

        vol_ok = (
            (dataframe['atr_pct'] >= self.min_atr_pct.value) &
            (dataframe['atr_pct'] <= self.max_atr_pct.value) &
            (dataframe['volume'] > self.vol_factor.value * dataframe['vol_sma'])
        )
        regime = (dataframe['adx'] >= self.adx_min.value)

        htf_up = (dataframe['close'] > dataframe['ema200_1h']) if self.use_htf_filter.value else True
        htf_dn = (dataframe['close'] < dataframe['ema200_1h']) if self.use_htf_filter.value else True

        # Breakout como EVENTO (cruce), no estado: la vela cruza el nivel que la previa no.
        cross_up = (dataframe['close'] > dataframe['don_hi']) & (dataframe['close'].shift(1) <= dataframe['don_hi'])
        cross_dn = (dataframe['close'] < dataframe['don_lo']) & (dataframe['close'].shift(1) >= dataframe['don_lo'])

        # Stack alcista completo: ema_fast > ema_slow > ema_200 local + precio arriba.
        stack_up = (
            (dataframe['ema_fast'] > dataframe['ema_slow']) &
            (dataframe['ema_slow'] > dataframe['ema_200']) &
            (dataframe['close'] > dataframe['ema_fast'])
        )
        stack_dn = (
            (dataframe['ema_fast'] < dataframe['ema_slow']) &
            (dataframe['ema_slow'] < dataframe['ema_200']) &
            (dataframe['close'] < dataframe['ema_fast'])
        )

        cond_long = (
            htf_up &
            regime &
            dataframe['adx_rising'] &        # tendencia fortaleciéndose
            cross_up &
            stack_up &                       # estructura limpia alcista
            vol_ok
        )
        cond_short = (
            (self.enable_short.value) &
            htf_dn &
            regime &
            dataframe['adx_rising'] &
            cross_dn &
            stack_dn &
            vol_ok
        )

        dataframe.loc[cond_long, ['enter_long', 'enter_tag']] = (1, 'mom_long')
        dataframe.loc[cond_short, ['enter_short', 'enter_tag']] = (1, 'mom_short')
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Sin señal de salida: las salidas son 100% por stop fijo (custom_stoploss) y timeout.
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0
        return dataframe

    # ==========================================================================
    # STOP FIJO DE PRECIO (leverage-independiente)
    # ==========================================================================
    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float,
                        after_fill: bool, **kwargs) -> float:
        # Stop a stop_pct del PRECIO de entrada, fijo. stoploss_from_absolute convierte a
        # ratio de margen según leverage → el stop es el mismo % de precio a cualquier apalancamiento.
        if trade.is_short:
            sl_price = trade.open_rate * (1 + self.stop_pct.value)
        else:
            sl_price = trade.open_rate * (1 - self.stop_pct.value)
        rel = stoploss_from_absolute(sl_price, current_rate, is_short=trade.is_short,
                                     leverage=trade.leverage)
        return rel if rel else self.stoploss

    # ==========================================================================
    # TIME-STOP (deja correr al ganador hasta N velas, luego cierra)
    # ==========================================================================
    def custom_exit(self, pair: str, trade: Trade, current_time: datetime,
                    current_rate: float, current_profit: float, **kwargs):
        held = (current_time - trade.open_date_utc).total_seconds() / 900.0  # 900s = 15m
        if held >= float(self.max_candles_hold.value):
            return 'timeout'
        return None

    # ==========================================================================
    # CIRCUIT BREAKER: pausa tras racha de pérdidas (correlación / régimen adverso)
    # ==========================================================================
    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                            time_in_force: str, current_time: datetime, entry_tag,
                            side: str, **kwargs) -> bool:
        closed = Trade.get_trades_proxy(is_open=False)
        if not closed:
            return True
        # Trades cerrados ordenados por fecha de cierre, más reciente primero.
        closed = sorted(
            [t for t in closed if t.close_date is not None],
            key=lambda t: t.close_date, reverse=True)
        streak = 0
        for t in closed:
            if (t.close_profit or 0) < 0:
                streak += 1
            else:
                break
        if streak >= int(self.max_consec_losses.value):
            # En cooldown si la última pérdida cerró hace < cooldown_h horas.
            last_close = closed[0].close_date
            if (current_time - last_close).total_seconds() < int(self.cooldown_h.value) * 3600:
                return False
        return True
