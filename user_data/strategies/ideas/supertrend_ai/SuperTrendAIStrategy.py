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
                                BooleanParameter, CategoricalParameter)
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

    # --- PARÁMETROS EXISTENTES ---
    min_factor = DecimalParameter(1.0, 2.0, default=1.6, decimals=1, space='buy', optimize=True)
    max_factor = DecimalParameter(3.0, 6.0, default=4.2, decimals=1, space='buy', optimize=True)
    perf_alpha = IntParameter(10, 100, default=56, space='buy', optimize=True)
    atr_length = IntParameter(10, 30, default=24, space='buy', optimize=True)
    
    use_adx_filter = BooleanParameter(default=True, space='buy', optimize=True)
    adx_threshold = IntParameter(15, 50, default=38, space='buy', optimize=True)

    factor_step = DecimalParameter(0.1, 1.0, default=0.5, decimals=1, space='buy', optimize=False)
    cluster_selection = CategoricalParameter(['best', 'average', 'worst'], default='best', space='buy', optimize=False)
    kmeans_iterations = IntParameter(500, 2000, default=1000, space='buy', optimize=False)

    exit_mode = CategoricalParameter(['standard', 'ama', 'none'], default='none', space='sell', optimize=True)

    # --- PARÁMETROS: Break Even & Partial TP ---
    use_break_even = BooleanParameter(default=True, space='protection', optimize=True)
    
    # Trigger: % de ganancia para activar BE y TP1 (Ej: 0.015 = 1.5%)
    be_trigger = DecimalParameter(0.01, 0.05, default=0.015, decimals=3, space='protection', optimize=True)
    
    # Offset: % por encima de la entrada para poner el SL (Ej: 0.002 = 0.2%)
    be_offset = DecimalParameter(0.001, 0.01, default=0.002, decimals=3, space='protection', optimize=True)
    
    # NEW: Porcentaje de la posición a VENDER al llegar al trigger (Ej: 0.5 = 50%)
    tp1_sell_pct = DecimalParameter(0.0, 1.0, default=0.5, decimals=1, space='protection', optimize=True)

    # Habilitar ajuste de posiciones (Necesario para salidas parciales)
    position_adjustment_enable = True

    # --- GESTIÓN DE RIESGO ---
    minimal_roi = { "0": 100 }
    stoploss = -0.05 
    timeframe = '15m'
    
    trailing_stop = True
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.03
    trailing_only_offset_is_reached = True
    
    leverage_value = 10.0
    startup_candle_count: int = 200

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                 side: str, **kwargs) -> float:
        return self.leverage_value

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

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
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
        
        if self.use_adx_filter.value:
            adx_condition = dataframe['adx'] > self.adx_threshold.value
            long_signal = long_signal & adx_condition
            short_signal = short_signal & adx_condition

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
    # LÓGICA DE VENTA PARCIAL (TP1)
    # =========================================================================
    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float,
                              min_stake: float, max_stake: float,
                              **kwargs) -> Union[float, int, None]:
        
        # Solo ejecutamos si tenemos activado el sistema BE y el porcentaje de venta > 0
        if self.use_break_even.value and self.tp1_sell_pct.value > 0:
            
            # Verificamos si ya hemos alcanzado el TP1
            if current_profit > self.be_trigger.value:
                # Verificamos que NO hayamos hecho ya una venta parcial
                if trade.nr_of_successful_exits == 0:
                    # Retornamos cantidad negativa para vender
                    # Ejemplo: - (100 USDT * 0.5) = -50 USDT (Vende la mitad)
                    return -(trade.stake_amount * self.tp1_sell_pct.value)
        
        return None

    # =========================================================================
    # LÓGICA BREAK EVEN (MOVER STOP LOSS)
    # =========================================================================
    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        
        if not self.use_break_even.value:
            return self.stoploss

        # Si ya pasó el TP1 (trigger), aseguramos ganancias con el Stop Loss
        if current_profit >= self.be_trigger.value:
            
            if trade.is_short:
                 stop_price = trade.open_rate * (1 - self.be_offset.value)
                 if current_rate < stop_price: 
                     return (stop_price / current_rate) - 1
            else:
                stop_price = trade.open_rate * (1 + self.be_offset.value)
                if current_rate > stop_price:
                    return (stop_price / current_rate) - 1

        return self.stoploss