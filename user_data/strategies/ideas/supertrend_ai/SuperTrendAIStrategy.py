# Freqtrade: SuperTrend AI (Clustering) - LUXALGO EXACT REPLICA v2.2
# Added: Break Even Logic + Partial Take Profit (TP1 Sell %)

import numpy as np
import pandas as pd
from pandas import DataFrame
from datetime import datetime
from typing import Optional, Union
import logging
import talib.abstract as ta

from freqtrade.strategy import (IStrategy, IntParameter, DecimalParameter, 
                                BooleanParameter, CategoricalParameter,
                                stoploss_from_absolute)
from freqtrade.persistence import Trade

# Importación de Numba con manejo de errores
try:
    from numba import jit
    has_numba = True
except ImportError:
    print("WARNING: Numba not found. Simulation will be VERY SLOW.")
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

logger = logging.getLogger(__name__)

# =============================================================================
# LÓGICA NUMBA JIT (Intacta)
# =============================================================================

@jit(nopython=True)
def run_supertrend_ai_numba(
    open_p, high, low, close, atr, 
    min_f, max_f, step, perf_alpha,
    cluster_mode, max_iter, den_array
):
    n_bars = len(close)
    factors = np.arange(min_f, max_f + 0.00001, step)
    n_factors = len(factors)
    
    agent_upper = np.zeros(n_factors)
    agent_lower = np.zeros(n_factors)
    agent_trend = np.ones(n_factors, dtype=np.int64) 
    agent_perf = np.zeros(n_factors)
    agent_output = np.zeros(n_factors)
    
    final_st = np.full(n_bars, np.nan)
    final_trend = np.zeros(n_bars, dtype=np.int64)
    final_ama = np.full(n_bars, np.nan)
    
    target_upper = 0.0
    target_lower = 0.0
    target_trend = 1
    current_ama = close[0] 
    
    alpha_k = 2.0 / (perf_alpha + 1.0)

    for i in range(n_bars):
        if np.isnan(atr[i]):
            final_st[i] = np.nan
            final_ama[i] = current_ama
            continue
            
        curr_close = close[i]
        curr_atr = atr[i]
        hl2 = (high[i] + low[i]) / 2.0
        prev_close = close[i-1] if i > 0 else curr_close
        
        for k in range(n_factors):
            factor = factors[k]
            up = hl2 + curr_atr * factor
            dn = hl2 - curr_atr * factor
            
            curr_upper = agent_upper[k]
            curr_lower = agent_lower[k]
            curr_trend = agent_trend[k]
            curr_output = agent_output[k]
            
            if i == 0 or np.isnan(curr_upper):
                curr_upper = up
                curr_lower = dn
                curr_trend = 1 if curr_close > up else -1
                curr_output = curr_lower if curr_trend == 1 else curr_upper
            else:
                if prev_close < curr_upper:
                    curr_upper = min(up, curr_upper)
                else:
                    curr_upper = up
                if prev_close > curr_lower:
                    curr_lower = max(dn, curr_lower)
                else:
                    curr_lower = dn
                if curr_close > curr_upper:
                    curr_trend = 1
                elif curr_close < curr_lower:
                    curr_trend = -1
                
                new_output = curr_lower if curr_trend == 1 else curr_upper
                diff = 1.0 if prev_close >= curr_output else -1.0
                price_delta = curr_close - prev_close
                agent_perf[k] += alpha_k * (price_delta * diff - agent_perf[k])
                curr_output = new_output
            
            agent_upper[k] = curr_upper
            agent_lower[k] = curr_lower
            agent_trend[k] = curr_trend
            agent_output[k] = curr_output

        sorted_perfs = np.sort(agent_perf)
        p25_idx = int(n_factors * 0.25)
        p50_idx = int(n_factors * 0.50)
        p75_idx = int(n_factors * 0.75)
        
        c_worst = sorted_perfs[p25_idx]
        c_avg   = sorted_perfs[p50_idx]
        c_best  = sorted_perfs[p75_idx]
        
        for _ in range(max_iter):
            sum_worst, count_worst = 0.0, 0
            sum_avg, count_avg = 0.0, 0
            sum_best, count_best = 0.0, 0
            sum_f_target, count_f_target = 0.0, 0
            sum_p_target, count_p_target = 0.0, 0
            
            for k in range(n_factors):
                val = agent_perf[k]
                d1 = abs(val - c_worst)
                d2 = abs(val - c_avg)
                d3 = abs(val - c_best)
                is_target = False
                if d1 <= d2 and d1 <= d3: # Worst
                    sum_worst += val
                    count_worst += 1
                    if cluster_mode == 0: is_target = True
                elif d2 <= d1 and d2 <= d3: # Avg
                    sum_avg += val
                    count_avg += 1
                    if cluster_mode == 1: is_target = True
                else: # Best
                    sum_best += val
                    count_best += 1
                    if cluster_mode == 2: is_target = True
                
                if is_target:
                    sum_f_target += factors[k]
                    count_f_target += 1
                    sum_p_target += val
                    count_p_target += 1
            
            new_c_worst = sum_worst / count_worst if count_worst > 0 else c_worst
            new_c_avg = sum_avg / count_avg if count_avg > 0 else c_avg
            new_c_best = sum_best / count_best if count_best > 0 else c_best
            
            if abs(new_c_worst - c_worst) < 0.0001 and abs(new_c_avg - c_avg) < 0.0001 and abs(new_c_best - c_best) < 0.0001:
               break
            c_worst, c_avg, c_best = new_c_worst, new_c_avg, new_c_best
            
        target_factor = max_f 
        target_perf_avg = 0.0
        
        if count_f_target > 0:
            target_factor = sum_f_target / count_f_target
            target_perf_avg = sum_p_target / count_p_target
            
        f_up = hl2 + curr_atr * target_factor
        f_dn = hl2 - curr_atr * target_factor
        
        if i == 0:
            target_upper = f_up
            target_lower = f_dn
            target_trend = 1 if curr_close > f_up else -1
        else:
            if prev_close < target_upper:
                target_upper = min(f_up, target_upper)
            else:
                target_upper = f_up
            if prev_close > target_lower:
                target_lower = max(f_dn, target_lower)
            else:
                target_lower = f_dn
            if curr_close > target_upper:
                target_trend = 1
            elif curr_close < target_lower:
                target_trend = -1
        
        final_val = target_lower if target_trend == 1 else target_upper
        
        den_val = den_array[i]
        if den_val == 0: den_val = 0.00001
        perf_idx = max(target_perf_avg, 0.0) / den_val
        if perf_idx > 2.0: perf_idx = 2.0 
        
        if i == 0 or np.isnan(current_ama):
            current_ama = final_val
        else:
            current_ama = current_ama + perf_idx * (final_val - current_ama)
            
        final_st[i] = final_val
        final_trend[i] = target_trend
        final_ama[i] = current_ama

    return final_st, final_trend, final_ama


class SuperTrendAIStrategy(IStrategy):
    INTERFACE_VERSION = 3

    # Tracking de eventos logueados (para evitar duplicados)
    _breakeven_logged = set()
    _tp_logged = set()
    
    # Cache para stoploss calculado (evita recalculos innecesarios)
    # Formato: {trade_id: (fixed_stop_price, is_breakeven_active)}
    _sl_cache = {}

    # --- CONTROL DE LOGGING ---
    analysis_logging = BooleanParameter(default=False, space='buy', optimize=False)

    # --- PARÁMETROS SUPERTREND AI ---
    min_factor = DecimalParameter(1.0, 2.0, default=1.0, decimals=1, space='buy', optimize=True)
    max_factor = DecimalParameter(3.0, 6.0, default=5.0, decimals=1, space='buy', optimize=True)
    perf_alpha = IntParameter(10, 100, default=10, space='buy', optimize=True)
    atr_length = IntParameter(10, 30, default=10, space='buy', optimize=True)
    
    # --- FILTROS ---
    use_adx_filter = BooleanParameter(default=False, space='buy', optimize=True)
    adx_threshold = IntParameter(15, 50, default=30, space='buy', optimize=True)
    
    use_ema_filter = BooleanParameter(default=False, space='buy', optimize=True)
    ema_period = IntParameter(50, 300, default=200, space='buy', optimize=True)
    
    # Volatility Filter: ATR(1) > ATR(10) = volatilidad actual > promedio
    use_volatility_filter = BooleanParameter(default=False, space='buy', optimize=True)

    factor_step = DecimalParameter(0.1, 1.0, default=0.5, decimals=1, space='buy', optimize=False)
    cluster_selection = CategoricalParameter(['best', 'average', 'worst'], default='best', space='buy', optimize=False)
    kmeans_iterations = IntParameter(500, 2000, default=1000, space='buy', optimize=False)

    # --- EXIT MODE ---
    exit_mode = CategoricalParameter(['standard', 'ama', 'none'], default='none', space='sell', optimize=True)

    # --- PARÁMETROS: Stop Loss Dinámico (ATR-based) ---
    atr_stop_len = IntParameter(10, 30, default=14, space='protection', optimize=True)
    atr_stop_mult = DecimalParameter(1.0, 5.0, default=1.5, decimals=1, space='protection', optimize=True)
    
    # --- PARÁMETROS: Break Even (R:R based) ---
    use_breakeven = BooleanParameter(default=True, space='protection', optimize=True)
    be_rr_long = DecimalParameter(0.5, 3.0, default=1.0, decimals=1, space='protection', optimize=True)
    be_rr_short = DecimalParameter(0.5, 3.0, default=1.0, decimals=1, space='protection', optimize=True)
    
    # --- PARÁMETROS: Take Profit Parcial (R:R based) ---
    use_takeprofit = BooleanParameter(default=True, space='protection', optimize=True)
    tp_rr_long = DecimalParameter(1.0, 5.0, default=3.0, decimals=1, space='protection', optimize=True)
    tp_rr_short = DecimalParameter(1.0, 5.0, default=3.0, decimals=1, space='protection', optimize=True)
    tp_percent = DecimalParameter(10, 100, default=50, decimals=0, space='protection', optimize=True)

    # Habilitar ajuste de posiciones (Necesario para salidas parciales)
    position_adjustment_enable = True

    # --- GESTIÓN DE RIESGO ---
    minimal_roi = { "0": 100 }  # Desactivamos ROI, usamos Custom TP
    stoploss = -0.99           # Desactivamos SL fijo, usamos Custom ATR
    timeframe = '15m'
    
    # Desactivar trailing (usamos BE en su lugar)
    trailing_stop = False
    
    startup_candle_count: int = 200
    
    can_short = True
    use_custom_stoploss = True

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                 side: str, **kwargs) -> float:
        return self.config.get('leverage', 1.0)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=self.atr_length.value)
        dataframe['atr'] = dataframe['atr'].bfill().fillna(0)
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        
        open_p = dataframe['open'].values.astype(np.float64)
        high = dataframe['high'].values.astype(np.float64)
        low = dataframe['low'].values.astype(np.float64)
        close = dataframe['close'].values.astype(np.float64)
        atr = dataframe['atr'].values.astype(np.float64)
        
        min_f = float(self.min_factor.value)
        max_f = float(self.max_factor.value)
        alpha = float(self.perf_alpha.value)
        step = float(self.factor_step.value)
        max_iter = int(self.kmeans_iterations.value)
        
        cluster_map = {'worst': 0, 'average': 1, 'best': 2}
        c_mode = cluster_map.get(self.cluster_selection.value, 2)

        price_diff = dataframe['close'].diff().abs().fillna(0)
        den_series = price_diff.ewm(alpha=1.0/alpha, adjust=False).mean()
        den_array = den_series.values.astype(np.float64)

        st_val, st_trend, ama_val = run_supertrend_ai_numba(
            open_p, high, low, close, atr,
            min_f, max_f, step, alpha, c_mode, max_iter,
            den_array
        )
        
        dataframe['supertrend_ai'] = st_val
        dataframe['supertrend_direction'] = st_trend
        dataframe['ama'] = ama_val
        
        # ATR para Stop Loss dinámico (separado del ATR del SuperTrend)
        dataframe['atr_sl'] = ta.ATR(dataframe, timeperiod=self.atr_stop_len.value)
        
        # EMA para filtro de tendencia
        dataframe['ema_filter'] = ta.EMA(dataframe, timeperiod=self.ema_period.value)
        
        # Volatility Filter: ATR(1) > ATR(10) = volatilidad actual > promedio
        dataframe['atr_1'] = ta.ATR(dataframe, timeperiod=1)
        dataframe['atr_10'] = ta.ATR(dataframe, timeperiod=10)
        dataframe['volatility_ok'] = dataframe['atr_1'] > dataframe['atr_10']

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        # Señales base: cambio de dirección del SuperTrend AI
        long_signal = (
            (dataframe['supertrend_direction'] == 1) &
            (dataframe['supertrend_direction'].shift(1) == -1) &
            (dataframe['volume'] > 0)
        )
        
        short_signal = (
            (dataframe['supertrend_direction'] == -1) &
            (dataframe['supertrend_direction'].shift(1) == 1) &
            (dataframe['volume'] > 0)
        )
        
        # Filtro ADX: Solo entrar si hay tendencia fuerte
        if self.use_adx_filter.value:
            adx_condition = dataframe['adx'] > self.adx_threshold.value
            long_signal = long_signal & adx_condition
            short_signal = short_signal & adx_condition
        
        # Filtro EMA: Solo longs encima de EMA, shorts debajo
        if self.use_ema_filter.value:
            long_signal = long_signal & (dataframe['close'] > dataframe['ema_filter'])
            short_signal = short_signal & (dataframe['close'] < dataframe['ema_filter'])
        
        # Filtro Volatilidad: Solo entrar si volatilidad actual > promedio
        if self.use_volatility_filter.value:
            long_signal = long_signal & dataframe['volatility_ok']
            short_signal = short_signal & dataframe['volatility_ok']

        dataframe.loc[long_signal, 'enter_long'] = 1
        dataframe.loc[short_signal, 'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[:, 'exit_long'] = 0
        dataframe.loc[:, 'exit_short'] = 0
        
        mode = self.exit_mode.value
        
        if mode == 'standard':
            dataframe.loc[
                (dataframe['supertrend_direction'] == -1) & (dataframe['supertrend_direction'].shift(1) == 1),
                'exit_long'
            ] = 1
            dataframe.loc[
                (dataframe['supertrend_direction'] == 1) & (dataframe['supertrend_direction'].shift(1) == -1),
                'exit_short'
            ] = 1
            
        elif mode == 'ama':
            dataframe.loc[
                (dataframe['close'] < dataframe['ama']) & (dataframe['close'].shift(1) >= dataframe['ama'].shift(1)),
                'exit_long'
            ] = 1
            dataframe.loc[
                (dataframe['close'] > dataframe['ama']) & (dataframe['close'].shift(1) <= dataframe['ama'].shift(1)),
                'exit_short'
            ] = 1
            
        return dataframe

    # =========================================================================
    # HELPER: Obtener candle de entrada
    # =========================================================================
    def _get_entry_candle(self, pair, open_date):
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        # Calcular duración del timeframe en minutos
        tf_min = int(self.timeframe[:-1]) * (60 if 'h' in self.timeframe else 1)
        
        open_date = pd.to_datetime(open_date, utc=True)
        tf_delta = pd.Timedelta(minutes=tf_min)
        candle_date = open_date.floor(tf_delta)
        signal_date = candle_date - tf_delta
        
        try:
            row = dataframe.loc[dataframe['date'] == signal_date]
            if not row.empty:
                return row.iloc[0]
            
            row = dataframe.loc[dataframe['date'] == candle_date]
            if not row.empty:
                return row.iloc[0]
                
            possible_candles = dataframe[dataframe['date'] <= signal_date]
            if not possible_candles.empty:
                return possible_candles.iloc[-1]
                
            return None
        except Exception:
            return None

    # =========================================================================
    # LÓGICA DE VENTA PARCIAL (TP) - BASADO EN ATR R:R
    # =========================================================================
    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float,
                              min_stake: float, max_stake: float,
                              **kwargs) -> Union[float, int, None]:
        
        if not self.use_takeprofit.value or trade.nr_of_successful_exits > 0:
            return None
        
        entry_data = self._get_entry_candle(trade.pair, trade.open_date_utc)
        if entry_data is None:
            return None
        
        # Calcular distancia del SL basada en ATR
        atr_val = entry_data.get('atr_sl', 0)
        if atr_val == 0 or pd.isna(atr_val):
            return None
            
        sl_dist = atr_val * float(self.atr_stop_mult.value)
        entry_price = trade.open_rate
        
        if trade.is_short:
            # SHORT: TP está por debajo del entry
            tp_price = entry_price - (sl_dist * float(self.tp_rr_short.value))
            if current_rate <= tp_price:
                if trade.id not in self._tp_logged:
                    self._tp_logged.add(trade.id)
                    if self.analysis_logging.value:
                        logger.info(f"💰 Take Profit SHORT ({self.tp_percent.value}%): {trade.pair} @ {current_rate:.8f} | R:R={self.tp_rr_short.value}")
                return -(trade.stake_amount * (self.tp_percent.value / 100))
        else:
            # LONG: TP está por encima del entry
            tp_price = entry_price + (sl_dist * float(self.tp_rr_long.value))
            if current_rate >= tp_price:
                if trade.id not in self._tp_logged:
                    self._tp_logged.add(trade.id)
                    if self.analysis_logging.value:
                        logger.info(f"💰 Take Profit LONG ({self.tp_percent.value}%): {trade.pair} @ {current_rate:.8f} | R:R={self.tp_rr_long.value}")
                return -(trade.stake_amount * (self.tp_percent.value / 100))
        
        return None

    # =========================================================================
    # LÓGICA STOP LOSS DINÁMICO + BREAK EVEN - BASADO EN ATR R:R
    # =========================================================================
    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        
        entry_data = self._get_entry_candle(pair, trade.open_date_utc)
        if entry_data is None:
            return self.stoploss
        
        # Calcular distancia del SL basada en ATR
        atr_val = entry_data.get('atr_sl', 0)
        if atr_val == 0 or pd.isna(atr_val):
            return self.stoploss
            
        sl_dist = atr_val * float(self.atr_stop_mult.value)
        entry_price = trade.open_rate
        
        if sl_dist <= 0:
            return self.stoploss
        
        # Verificar si ya tenemos cache para este trade
        if trade.id in self._sl_cache:
            cached_stop_price, is_breakeven_active = self._sl_cache[trade.id]
            
            # Si el breakeven ya está activo, usar entry_price como stop
            if is_breakeven_active:
                return stoploss_from_absolute(cached_stop_price, current_rate, 
                                              is_short=trade.is_short, leverage=trade.leverage)
            
            # Verificar si ahora califica para activar BE
            if self.use_breakeven.value:
                if trade.is_short:
                    be_dist = sl_dist * float(self.be_rr_short.value)
                    be_trigger = entry_price - be_dist
                    if trade.min_rate is not None and trade.min_rate <= be_trigger:
                        self._sl_cache[trade.id] = (entry_price, True)
                        if trade.id not in self._breakeven_logged:
                            self._breakeven_logged.add(trade.id)
                            if self.analysis_logging.value:
                                logger.info(f"⚖️ Breakeven SHORT activated: {pair} @ {current_rate:.8f} | R:R={self.be_rr_short.value}")
                        return stoploss_from_absolute(entry_price, current_rate, 
                                                      is_short=trade.is_short, leverage=trade.leverage)
                else:
                    be_dist = sl_dist * float(self.be_rr_long.value)
                    be_trigger = entry_price + be_dist
                    if trade.max_rate is not None and trade.max_rate >= be_trigger:
                        self._sl_cache[trade.id] = (entry_price, True)
                        if trade.id not in self._breakeven_logged:
                            self._breakeven_logged.add(trade.id)
                            if self.analysis_logging.value:
                                logger.info(f"⚖️ Breakeven LONG activated: {pair} @ {current_rate:.8f} | R:R={self.be_rr_long.value}")
                        return stoploss_from_absolute(entry_price, current_rate, 
                                                      is_short=trade.is_short, leverage=trade.leverage)
            
            # Breakeven no activado, usar el precio de SL cacheado
            return stoploss_from_absolute(cached_stop_price, current_rate, 
                                          is_short=trade.is_short, leverage=trade.leverage)
        
        # PRIMERA VEZ: Calcular el SL inicial y cachearlo
        if trade.is_short:
            fixed_stop_price = entry_price + sl_dist
        else:
            fixed_stop_price = entry_price - sl_dist
            
        fixed_stop_price = round(fixed_stop_price, 8)
        self._sl_cache[trade.id] = (fixed_stop_price, False)
        
        sl_pct = (sl_dist / entry_price) * 100
        if self.analysis_logging.value:
            logger.info(f"📍 SL Calculated for {pair}: {'SHORT' if trade.is_short else 'LONG'} | "
                       f"Entry: {entry_price:.8f} | SL: {fixed_stop_price:.8f} ({sl_pct:.2f}%)")
        
        return stoploss_from_absolute(fixed_stop_price, current_rate, 
                                      is_short=trade.is_short, leverage=trade.leverage)