# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: disable=F401
# isort: skip_file
# --- Do not remove these libs ---
"""
Machine Learning: Cosine Similarity & Euclidean + Lorentzian Distance Strategy

Based on the TradingView indicator by chhagansinghmeena:
https://www.tradingview.com/script/...

This strategy implements a KNN-based ML approach with multiple distance metrics:
- Lorentzian Distance
- Euclidean Distance
- Cosine Similarity

Features used:
- RSI (normalized)
- KST (Know Sure Thing)
- CPMA (Centered Price Moving Average)
- VWAP (Volume Weighted Average Price)
- FRAMA (Fractal Adaptive Moving Average)
- MACD (normalized)

Filters:
- Trend Filter (CPMA/FRAMA/RationalQuad)
- Candlestick Pattern Filter
- SuperTrend Filter
"""

import numpy as np
import pandas as pd
from pandas import DataFrame
from datetime import datetime, timedelta
from typing import Optional, Union

from freqtrade.strategy import (IStrategy, IntParameter, DecimalParameter, BooleanParameter, 
                                 CategoricalParameter, stoploss_from_absolute)
from freqtrade.persistence import Trade
import talib.abstract as ta
import logging

# =============================================================================
# DEPENDENCIAS
# =============================================================================
try:
    from numba import njit, prange
    _HAS_NUMBA = True
except ImportError:
    raise ImportError("Esta estrategia requiere numba. Ejecuta: pip install numba")

logger = logging.getLogger(__name__)

# =============================================================================
# MOTOR MATEMÁTICO (NUMBA OPTIMIZED)
# =============================================================================

@njit(fastmath=True)
def numba_cpma(close: np.ndarray, high: np.ndarray, low: np.ndarray, 
               open_price: np.ndarray, length: int) -> np.ndarray:
    """
    Centered Price Moving Average (CPMA)
    Calcula el promedio de 8 tipos de precios típicos y aplica suavizado EMA.
    """
    n = len(close)
    result = np.full(n, np.nan)
    
    # Calcular los 8 tipos de precio
    for i in range(n):
        hl2 = (high[i] + low[i]) / 2
        hlc3 = (high[i] + low[i] + close[i]) / 3
        ohlc4 = (open_price[i] + high[i] + low[i] + close[i]) / 4
        hlcc4 = (high[i] + low[i] + close[i] + close[i]) / 4
        
        # Weighted prices
        weighted1 = (high[i] + low[i] + close[i] * 2) / 4
        weighted2 = (high[i] + low[i] + open_price[i] + close[i] * 2) / 5
        
        avg_price = (close[i] + hl2 + hlc3 + ohlc4 + hlcc4 + weighted1 + weighted2 + open_price[i]) / 8
        result[i] = avg_price
    
    # Aplicar EMA-like smoothing
    alpha = 2.0 / (length + 1)
    ema = np.full(n, np.nan)
    ema[0] = result[0]
    for i in range(1, n):
        if not np.isnan(result[i]):
            if np.isnan(ema[i-1]):
                ema[i] = result[i]
            else:
                ema[i] = alpha * result[i] + (1 - alpha) * ema[i-1]
    
    return ema


@njit(fastmath=True)
def numba_frama(close: np.ndarray, length: int) -> np.ndarray:
    """
    Fractal Adaptive Moving Average (FRAMA)
    Media móvil que se adapta según la dimensión fractal del precio.
    """
    n = len(close)
    frama = np.full(n, np.nan)
    
    half = length // 2
    if half < 1:
        half = 1
    
    # Inicializar con el primer close válido
    frama[length-1] = close[length-1] if length <= n else close[0]
    
    for i in range(length, n):
        # Primera mitad
        h1 = np.max(close[i-length:i-half])
        l1 = np.min(close[i-length:i-half])
        n1 = (h1 - l1) / half if half > 0 else 0
        
        # Segunda mitad
        h2 = np.max(close[i-half:i])
        l2 = np.min(close[i-half:i])
        n2 = (h2 - l2) / half if half > 0 else 0
        
        # Rango total
        h = np.max(close[i-length:i])
        l = np.min(close[i-length:i])
        n3 = (h - l) / length if length > 0 else 0
        
        # Dimensión fractal
        if n1 + n2 > 0 and n3 > 0:
            d = (np.log(n1 + n2) - np.log(n3)) / np.log(2)
        else:
            d = 1.0
        
        # Alpha adaptativo
        alpha = np.exp(-4.6 * (d - 1))
        alpha = max(0.01, min(1.0, alpha))  # Clamp entre 0.01 y 1.0
        
        # FRAMA
        if not np.isnan(frama[i-1]):
            frama[i] = alpha * close[i] + (1 - alpha) * frama[i-1]
        else:
            frama[i] = close[i]
    
    return frama


@njit(fastmath=True)
def numba_kst(close: np.ndarray) -> np.ndarray:
    """
    Know Sure Thing (KST) Oscillator
    Momentum oscillator basado en múltiples ROCs suavizados.
    """
    n = len(close)
    kst = np.full(n, np.nan)
    
    # ROC lengths
    roc1_len, roc2_len, roc3_len, roc4_len = 10, 15, 20, 30
    sma_len = 3
    
    max_lookback = roc4_len + sma_len
    
    for i in range(max_lookback, n):
        # ROC calculations
        roc1 = (close[i] - close[i-roc1_len]) / close[i-roc1_len] if close[i-roc1_len] != 0 else 0
        roc2 = (close[i] - close[i-roc2_len]) / close[i-roc2_len] if close[i-roc2_len] != 0 else 0
        roc3 = (close[i] - close[i-roc3_len]) / close[i-roc3_len] if close[i-roc3_len] != 0 else 0
        roc4 = (close[i] - close[i-roc4_len]) / close[i-roc4_len] if close[i-roc4_len] != 0 else 0
        
        # Simple smoothing (approximate SMA of last 3 ROC values)
        kst_val = roc1 + 2 * roc2 + 3 * roc3 + 4 * roc4
        kst[i] = kst_val * 100
    
    return kst


@njit(fastmath=True)
def numba_rational_quadratic(src: np.ndarray, lookback: int, relative_weight: float, 
                              start_at_bar: int) -> np.ndarray:
    """
    Rational Quadratic Kernel (Nadaraya-Watson Estimator)
    Suavizado adaptativo basado en kernel RQ.
    """
    n = len(src)
    result = np.full(n, np.nan)
    
    denom_base = 2.0 * relative_weight * lookback * lookback
    
    for i in range(start_at_bar, n):
        current_weight = 0.0
        cumulative_weight = 0.0
        
        loop_end = min(i + 1, 5000)  # Safety limit
        
        for j in range(loop_end):
            if i - j < 0:
                break
            
            y = src[i - j]
            if np.isnan(y):
                continue
            
            w = (1.0 + (j * j) / denom_base) ** (-relative_weight)
            current_weight += y * w
            cumulative_weight += w
        
        if cumulative_weight > 0:
            result[i] = current_weight / cumulative_weight
        else:
            result[i] = src[i]
    
    return result


@njit(parallel=True, fastmath=True)
def numba_lorentzian_distance_knn(features: np.ndarray, labels: np.ndarray, 
                                   max_bars_back: int, neighbors_count: int) -> np.ndarray:
    """
    KNN con distancia Lorentziana.
    Pine: distance += math.log(1 + math.abs(feature_current - feature_historical))
    """
    n_rows = len(features)
    n_features = features.shape[1]
    predictions = np.zeros(n_rows)
    
    start_calculation = max_bars_back
    if start_calculation >= n_rows:
        return predictions
    
    for i in prange(start_calculation, n_rows):
        current_features = features[i]
        
        last_distance = -1.0
        distances_buffer = np.zeros(neighbors_count * 10)
        predictions_buffer = np.zeros(neighbors_count * 10)
        buffer_size = 0
        
        start_index = max(0, i - max_bars_back)
        end_index = i - 4  # Exclude recent bars to avoid lookahead
        
        if end_index < start_index:
            predictions[i] = 0.0
            continue
        
        for j in range(start_index, end_index):
            # Skip every 4th bar (Pine: if i % 4)
            if j % 4 == 0:
                continue
            
            # Lorentzian distance
            dist = 0.0
            for f in range(n_features):
                d = abs(current_features[f] - features[j, f])
                dist += np.log(1.0 + d)
            
            if dist >= last_distance:
                last_distance = dist
                
                if buffer_size < len(distances_buffer):
                    distances_buffer[buffer_size] = dist
                    predictions_buffer[buffer_size] = labels[j]
                    buffer_size += 1
                
                if buffer_size > neighbors_count:
                    idx = int(round(neighbors_count * 3.0 / 4.0))
                    if idx < buffer_size:
                        last_distance = distances_buffer[idx]
                    
                    for k in range(buffer_size - 1):
                        distances_buffer[k] = distances_buffer[k+1]
                        predictions_buffer[k] = predictions_buffer[k+1]
                    buffer_size -= 1
        
        final_prediction = 0.0
        for k in range(buffer_size):
            final_prediction += predictions_buffer[k]
        
        predictions[i] = final_prediction
    
    return predictions


@njit(parallel=True, fastmath=True)
def numba_euclidean_distance_knn(features: np.ndarray, labels: np.ndarray, 
                                  max_bars_back: int, neighbors_count: int) -> np.ndarray:
    """
    KNN con distancia Euclidiana.
    Pine: distance += math.pow(feature_current - feature_historical, 2), sqrt at end
    """
    n_rows = len(features)
    n_features = features.shape[1]
    predictions = np.zeros(n_rows)
    
    start_calculation = max_bars_back
    if start_calculation >= n_rows:
        return predictions
    
    for i in prange(start_calculation, n_rows):
        current_features = features[i]
        
        last_distance = -1.0
        distances_buffer = np.zeros(neighbors_count * 10)
        predictions_buffer = np.zeros(neighbors_count * 10)
        buffer_size = 0
        
        start_index = max(0, i - max_bars_back)
        end_index = i - 4
        
        if end_index < start_index:
            predictions[i] = 0.0
            continue
        
        for j in range(start_index, end_index):
            if j % 4 == 0:
                continue
            
            # Euclidean distance
            dist = 0.0
            for f in range(n_features):
                d = current_features[f] - features[j, f]
                dist += d * d
            dist = np.sqrt(dist)
            
            if dist >= last_distance:
                last_distance = dist
                
                if buffer_size < len(distances_buffer):
                    distances_buffer[buffer_size] = dist
                    predictions_buffer[buffer_size] = labels[j]
                    buffer_size += 1
                
                if buffer_size > neighbors_count:
                    idx = int(round(neighbors_count * 3.0 / 4.0))
                    if idx < buffer_size:
                        last_distance = distances_buffer[idx]
                    
                    for k in range(buffer_size - 1):
                        distances_buffer[k] = distances_buffer[k+1]
                        predictions_buffer[k] = predictions_buffer[k+1]
                    buffer_size -= 1
        
        final_prediction = 0.0
        for k in range(buffer_size):
            final_prediction += predictions_buffer[k]
        
        predictions[i] = final_prediction
    
    return predictions


@njit(parallel=True, fastmath=True)
def numba_cosine_similarity_knn(features: np.ndarray, labels: np.ndarray, 
                                 max_bars_back: int, neighbors_count: int) -> np.ndarray:
    """
    KNN con Cosine Similarity.
    Pine: dotProduct / (magnitudeSeries * magnitudeArray)
    """
    n_rows = len(features)
    n_features = features.shape[1]
    predictions = np.zeros(n_rows)
    
    start_calculation = max_bars_back
    if start_calculation >= n_rows:
        return predictions
    
    for i in prange(start_calculation, n_rows):
        current_features = features[i]
        
        # Pre-calculate magnitude of current features
        mag_current = 0.0
        for f in range(n_features):
            mag_current += current_features[f] * current_features[f]
        mag_current = np.sqrt(mag_current)
        
        if mag_current == 0:
            mag_current = 1e-10
        
        last_similarity = -2.0  # Cosine similarity ranges from -1 to 1
        similarities_buffer = np.zeros(neighbors_count * 10)
        predictions_buffer = np.zeros(neighbors_count * 10)
        buffer_size = 0
        
        start_index = max(0, i - max_bars_back)
        end_index = i - 4
        
        if end_index < start_index:
            predictions[i] = 0.0
            continue
        
        for j in range(start_index, end_index):
            if j % 4 == 0:
                continue
            
            # Cosine similarity
            dot_product = 0.0
            mag_historical = 0.0
            
            for f in range(n_features):
                dot_product += current_features[f] * features[j, f]
                mag_historical += features[j, f] * features[j, f]
            
            mag_historical = np.sqrt(mag_historical)
            
            if mag_historical == 0:
                mag_historical = 1e-10
            
            similarity = dot_product / (mag_current * mag_historical)
            
            if similarity >= last_similarity:
                last_similarity = similarity
                
                if buffer_size < len(similarities_buffer):
                    similarities_buffer[buffer_size] = similarity
                    predictions_buffer[buffer_size] = labels[j]
                    buffer_size += 1
                
                if buffer_size > neighbors_count:
                    idx = int(round(neighbors_count * 3.0 / 4.0))
                    if idx < buffer_size:
                        last_similarity = similarities_buffer[idx]
                    
                    for k in range(buffer_size - 1):
                        similarities_buffer[k] = similarities_buffer[k+1]
                        predictions_buffer[k] = predictions_buffer[k+1]
                    buffer_size -= 1
        
        final_prediction = 0.0
        for k in range(buffer_size):
            final_prediction += predictions_buffer[k]
        
        predictions[i] = final_prediction
    
    return predictions


@njit(fastmath=True)
def numba_calculate_latched_signal(predictions: np.ndarray, filter_all: np.ndarray) -> np.ndarray:
    """
    Lógica de enclavamiento de señal (Latched Signal)
    Pine: signal := prediction > 0 ... ? long : nz(signal[1])
    """
    n = len(predictions)
    signal = np.zeros(n, dtype=np.int32)
    last_signal = 0
    
    for i in range(1, n):
        pred = predictions[i]
        is_filtered = filter_all[i]
        
        if pred > 0 and is_filtered:
            last_signal = 1
        elif pred < 0 and is_filtered:
            last_signal = -1
        
        signal[i] = last_signal
    
    return signal


@njit(fastmath=True)
def numba_supertrend(close: np.ndarray, high: np.ndarray, low: np.ndarray, 
                      atr: np.ndarray, factor: float) -> tuple:
    """
    SuperTrend Indicator
    """
    n = len(close)
    supertrend = np.full(n, np.nan)
    direction = np.zeros(n)
    
    hl2 = (high + low) / 2
    upper = hl2 + factor * atr
    lower = hl2 - factor * atr
    
    supertrend[0] = upper[0]
    direction[0] = 1
    
    for i in range(1, n):
        prev_upper = upper[i-1]
        prev_lower = lower[i-1]
        prev_st = supertrend[i-1]
        prev_dir = direction[i-1]
        prev_close = close[i-1]
        curr_close = close[i]
        
        # Adjust bands
        if lower[i] > prev_lower or prev_close < prev_lower:
            eff_lower = lower[i]
        else:
            eff_lower = prev_lower
        
        if upper[i] < prev_upper or prev_close > prev_upper:
            eff_upper = upper[i]
        else:
            eff_upper = prev_upper
        
        # Determine direction
        curr_dir = prev_dir
        if prev_dir == -1 and curr_close > eff_upper:
            curr_dir = 1
        elif prev_dir == 1 and curr_close < eff_lower:
            curr_dir = -1
        
        direction[i] = curr_dir
        
        if curr_dir == 1:
            supertrend[i] = eff_lower
            lower[i] = eff_lower
        else:
            supertrend[i] = eff_upper
            upper[i] = eff_upper
    
    return supertrend, direction


# =============================================================================
# ESTRATEGIA
# =============================================================================

class MLCosineSimilarity(IStrategy):
    """
    Machine Learning: Cosine Similarity & Euclidean + Lorentzian Distance Strategy
    
    Based on chhagansinghmeena's TradingView indicator, ported to Freqtrade
    with the same pattern as LorentzianSuperTrend.py
    """
    
    INTERFACE_VERSION = 3
    
    # ================= PARÁMETROS PRINCIPALES =================
    
    # ML Settings
    distance_method = CategoricalParameter(
        ['lorentzian', 'euclidean', 'cosine'], 
        default='cosine', space='buy', optimize=True
    )
    history_lookback = IntParameter(500, 2000, default=1000, space='buy', optimize=False)
    neighbors_count = IntParameter(4, 16, default=8, space='buy', optimize=True)
    
    # Moving Average Selection
    trend_ma_type = CategoricalParameter(
        ['cpma', 'frama', 'rational_quad'], 
        default='rational_quad', space='buy', optimize=True
    )
    cpma_length = IntParameter(5, 20, default=9, space='buy', optimize=True)
    frama_length = IntParameter(10, 30, default=14, space='buy', optimize=True)
    
    # Filters
    use_trend_filter = BooleanParameter(default=True, space='buy', optimize=True)
    use_candle_filter = BooleanParameter(default=False, space='buy', optimize=True)
    use_supertrend_filter = BooleanParameter(default=True, space='buy', optimize=True)
    
    # SuperTrend Settings
    st_factor = DecimalParameter(1.0, 5.0, default=3.0, decimals=1, space='buy', optimize=True)
    st_atr_period = IntParameter(5, 20, default=10, space='buy', optimize=True)
    
    # Position Management (Price Movement Percentages - same for long/short)
    take_profit_pct = DecimalParameter(2.0, 15.0, default=5.0, decimals=1, space='sell', optimize=True)
    
    # TP1 - Partial Take Profit
    use_tp1 = BooleanParameter(default=True, space='sell', optimize=True)
    tp1_pct = DecimalParameter(1.0, 10.0, default=2.0, decimals=1, space='sell', optimize=True)  # Price movement % to trigger TP1
    tp1_exit_pct = DecimalParameter(10.0, 80.0, default=50.0, decimals=0, space='sell', optimize=True)  # % of position to close at TP1
    
    # Breakeven (based on price movement)
    use_breakeven = BooleanParameter(default=True, space='sell', optimize=True)
    be_trigger_pct = DecimalParameter(0.5, 5.0, default=1.5, decimals=1, space='sell', optimize=True)  # Price move % to trigger BE
    
    # Rational Quadratic Kernel settings
    rq_lookback = IntParameter(5, 15, default=8, space='buy', optimize=False)
    rq_weight = DecimalParameter(0.3, 1.0, default=0.5, decimals=1, space='buy', optimize=False)
    rq_start = IntParameter(20, 35, default=25, space='buy', optimize=False)
    
    # ================= CONFIGURACIÓN FREQTRADE =================
    
    timeframe = '4h'
    minimal_roi = {"0": 100}  # Desactivado, usamos Take Profit manual
    stoploss = -0.99  # Fallback, usamos custom_stoploss
    
    startup_candle_count = 1200
    can_short = True
    use_custom_stoploss = True
    process_only_new_candles = True
    position_adjustment_enable = True  # Required for partial exits (TP1)
    
    # Cache para stoploss y TP1
    _sl_cache = {}
    _be_activated = set()
    _tp1_taken = set()
    
    def _normalize_0_1(self, series: pd.Series, min_val: float = None, max_val: float = None) -> pd.Series:
        """Normaliza una serie al rango [0, 1]"""
        if min_val is not None and max_val is not None:
            return (series.clip(min_val, max_val) - min_val) / (max_val - min_val)
        else:
            hist_min = series.expanding().min()
            hist_max = series.expanding().max()
            denom = (hist_max - hist_min).replace(0, 1e-10)
            return (series - hist_min) / denom
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Calcula todos los indicadores necesarios para la estrategia.
        """
        
        # === FEATURE 1: RSI (normalized 0-100) ===
        rsi_raw = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi'] = rsi_raw
        dataframe['f1_norm'] = self._normalize_0_1(rsi_raw, 0, 100)
        
        # === FEATURE 2: KST ===
        kst = numba_kst(dataframe['close'].values)
        dataframe['kst'] = kst
        dataframe['f2_norm'] = self._normalize_0_1(pd.Series(kst))
        
        # === FEATURE 3: CPMA ===
        cpma = numba_cpma(
            dataframe['close'].values,
            dataframe['high'].values,
            dataframe['low'].values,
            dataframe['open'].values,
            self.cpma_length.value
        )
        dataframe['cpma'] = cpma
        dataframe['f3_norm'] = self._normalize_0_1(pd.Series(cpma))
        
        # === FEATURE 4: VWAP ===
        # Simplified VWAP (typical price * volume cumsum / volume cumsum)
        typical_price = (dataframe['high'] + dataframe['low'] + dataframe['close']) / 3
        dataframe['vwap'] = (typical_price * dataframe['volume']).cumsum() / dataframe['volume'].cumsum()
        dataframe['f4_norm'] = self._normalize_0_1(dataframe['vwap'])
        
        # === FEATURE 5: FRAMA ===
        frama = numba_frama(dataframe['close'].values, self.frama_length.value)
        dataframe['frama'] = frama
        dataframe['f5_norm'] = self._normalize_0_1(pd.Series(frama))
        
        # === FEATURE 6: MACD ===
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['f6_norm'] = self._normalize_0_1(dataframe['macd'])
        
        # === TREND CALCULATION ===
        if self.trend_ma_type.value == 'cpma':
            trend_src = dataframe['cpma']
        elif self.trend_ma_type.value == 'frama':
            trend_src = dataframe['frama']
        else:  # rational_quad
            rq = numba_rational_quadratic(
                dataframe['close'].values,
                self.rq_lookback.value,
                float(self.rq_weight.value),
                self.rq_start.value
            )
            trend_src = pd.Series(rq, index=dataframe.index)
        
        dataframe['trend'] = trend_src
        dataframe['trend_bullish'] = dataframe['close'] >= dataframe['trend']
        dataframe['trend_bearish'] = dataframe['close'] <= dataframe['trend']
        
        # === SUPERTREND ===
        dataframe['atr_st'] = ta.ATR(dataframe, timeperiod=self.st_atr_period.value)
        st, st_dir = numba_supertrend(
            dataframe['close'].values,
            dataframe['high'].values,
            dataframe['low'].values,
            dataframe['atr_st'].values,
            float(self.st_factor.value)
        )
        dataframe['supertrend'] = st
        dataframe['st_direction'] = st_dir
        dataframe['st_bullish'] = st_dir > 0
        dataframe['st_bearish'] = st_dir < 0
        
        # === CANDLESTICK PATTERNS (simplified) ===
        # Bullish
        dataframe['cdl_hammer'] = ta.CDLHAMMER(dataframe)
        dataframe['cdl_engulf_bull'] = ta.CDLENGULFING(dataframe) > 0
        dataframe['cdl_morning_star'] = ta.CDLMORNINGSTAR(dataframe)
        dataframe['cdl_bullish'] = (
            (dataframe['cdl_hammer'] != 0) | 
            dataframe['cdl_engulf_bull'] | 
            (dataframe['cdl_morning_star'] != 0)
        )
        
        # Bearish
        dataframe['cdl_shooting_star'] = ta.CDLSHOOTINGSTAR(dataframe)
        dataframe['cdl_engulf_bear'] = ta.CDLENGULFING(dataframe) < 0
        dataframe['cdl_evening_star'] = ta.CDLEVENINGSTAR(dataframe)
        dataframe['cdl_bearish'] = (
            (dataframe['cdl_shooting_star'] != 0) | 
            dataframe['cdl_engulf_bear'] | 
            (dataframe['cdl_evening_star'] != 0)
        )
        
        # === ML LABELS ===
        # Pine: y_train_series = src[4] < src[0] ? -1 : src[4] > src[0] ? 1 : 0
        past_close = dataframe['close'].shift(4)
        labels = np.where(past_close < dataframe['close'], -1.0,
                  np.where(past_close > dataframe['close'], 1.0, 0.0))
        labels = np.nan_to_num(labels, nan=0.0)
        
        # === ML FEATURES ===
        feature_cols = ['f1_norm', 'f2_norm', 'f3_norm', 'f4_norm', 'f5_norm', 'f6_norm']
        features = dataframe[feature_cols].fillna(0.5).values.astype(np.float64)
        
        # === ML PREDICTION ===
        method = self.distance_method.value
        max_bars = int(self.history_lookback.value)
        k = int(self.neighbors_count.value)
        
        if method == 'lorentzian':
            predictions = numba_lorentzian_distance_knn(features, labels, max_bars, k)
        elif method == 'euclidean':
            predictions = numba_euclidean_distance_knn(features, labels, max_bars, k)
        else:  # cosine
            predictions = numba_cosine_similarity_knn(features, labels, max_bars, k)
        
        dataframe['prediction'] = predictions
        
        # === FILTER ALL ===
        dataframe['filter_all'] = True  # Simplified, can add more filters
        
        # === SIGNAL LATCHING ===
        filter_all_np = dataframe['filter_all'].fillna(True).values.astype(bool)
        signal = numba_calculate_latched_signal(predictions, filter_all_np)
        dataframe['signal'] = signal
        
        # === SIGNAL CHANGE DETECTION ===
        dataframe['signal_changed'] = dataframe['signal'] != dataframe['signal'].shift(1)
        
        dataframe['is_new_buy_signal'] = (dataframe['signal'] == 1) & dataframe['signal_changed']
        dataframe['is_new_sell_signal'] = (dataframe['signal'] == -1) & dataframe['signal_changed']
        
        # === ATR for Stop Loss ===
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define las condiciones de entrada.
        """
        
        # === LONG CONDITIONS ===
        long_base = dataframe['is_new_buy_signal'].copy()
        
        # Apply filters
        if self.use_trend_filter.value:
            long_base &= dataframe['trend_bullish']
        
        if self.use_candle_filter.value:
            long_base &= dataframe['cdl_bullish']
        
        if self.use_supertrend_filter.value:
            long_base &= dataframe['st_bullish']
        
        dataframe.loc[long_base & (dataframe['volume'] > 0), 'enter_long'] = 1
        
        # === SHORT CONDITIONS ===
        short_base = dataframe['is_new_sell_signal'].copy()
        
        if self.use_trend_filter.value:
            short_base &= dataframe['trend_bearish']
        
        if self.use_candle_filter.value:
            short_base &= dataframe['cdl_bearish']
        
        if self.use_supertrend_filter.value:
            short_base &= dataframe['st_bearish']
        
        if self.can_short:
            dataframe.loc[short_base & (dataframe['volume'] > 0), 'enter_short'] = 1
        
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define las condiciones de salida por señal.
        """
        dataframe.loc[:, ['exit_long', 'exit_short']] = 0
        
        # Exit on opposite signal
        dataframe.loc[dataframe['is_new_sell_signal'], 'exit_long'] = 1
        dataframe.loc[dataframe['is_new_buy_signal'], 'exit_short'] = 1
        
        # Exit on SuperTrend flip (if filter enabled)
        if self.use_supertrend_filter.value:
            st_flip_bearish = (dataframe['st_direction'] < 0) & (dataframe['st_direction'].shift(1) > 0)
            st_flip_bullish = (dataframe['st_direction'] > 0) & (dataframe['st_direction'].shift(1) < 0)
            
            dataframe.loc[st_flip_bearish, 'exit_long'] = 1
            dataframe.loc[st_flip_bullish, 'exit_short'] = 1
        
        return dataframe
    
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Custom stoploss que solo maneja Breakeven.
        El SL base viene del stoploss nativo de Freqtrade (JSON: stoploss: -0.03)
        """
        entry_price = trade.open_rate
        
        # Breakeven Logic - basado en movimiento de precio
        if self.use_breakeven.value:
            be_trigger_pct = float(self.be_trigger_pct.value) / 100
            
            if trade.is_short:
                # Para short: trigger cuando el precio BAJA be_trigger_pct%
                be_trigger_price = entry_price * (1 - be_trigger_pct)
                if trade.min_rate <= be_trigger_price:
                    # Breakeven activated - SL moves to entry
                    if trade.id not in self._be_activated:
                        self._be_activated.add(trade.id)
                        logger.info(f"⚖️ BE SHORT: {pair} @ {current_rate:.8f} (trigger: {be_trigger_pct*100:.1f}% move)")
                    return stoploss_from_absolute(entry_price, current_rate, 
                                                  is_short=True, leverage=trade.leverage)
            else:
                # Para long: trigger cuando el precio SUBE be_trigger_pct%
                be_trigger_price = entry_price * (1 + be_trigger_pct)
                if trade.max_rate >= be_trigger_price:
                    # Breakeven activated - SL moves to entry
                    if trade.id not in self._be_activated:
                        self._be_activated.add(trade.id)
                        logger.info(f"⚖️ BE LONG: {pair} @ {current_rate:.8f} (trigger: {be_trigger_pct*100:.1f}% move)")
                    return stoploss_from_absolute(entry_price, current_rate, 
                                                  is_short=False, leverage=trade.leverage)
        
        # Si BE no está activado, usar el stoploss nativo de Freqtrade
        # Retornar -1 significa "usar el stoploss por defecto de la estrategia"
        return -1
    
    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float,
                              min_stake: float, max_stake: float,
                              current_entry_rate: float, current_exit_rate: float,
                              current_entry_profit: float, current_exit_profit: float,
                              **kwargs) -> Optional[float]:
        """
        TP1 - Salida parcial basada en movimiento de precio.
        """
        if not self.use_tp1.value:
            return None
        
        # Solo ejecutar TP1 una vez
        if trade.id in self._tp1_taken:
            return None
        
        entry_price = trade.open_rate
        tp1_move_pct = float(self.tp1_pct.value) / 100
        tp1_exit_amount = float(self.tp1_exit_pct.value) / 100
        
        if trade.is_short:
            # Para short: TP1 cuando el precio BAJA tp1_pct%
            tp1_price = entry_price * (1 - tp1_move_pct)
            if current_rate <= tp1_price:
                self._tp1_taken.add(trade.id)
                exit_stake = -(trade.stake_amount * tp1_exit_amount)
                logger.info(f"💵 TP1 SHORT: {pair} @ {current_rate:.8f} (move: {tp1_move_pct*100:.1f}%, exit: {tp1_exit_amount*100:.0f}%)")
                return exit_stake
        else:
            # Para long: TP1 cuando el precio SUBE tp1_pct%
            tp1_price = entry_price * (1 + tp1_move_pct)
            if current_rate >= tp1_price:
                self._tp1_taken.add(trade.id)
                exit_stake = -(trade.stake_amount * tp1_exit_amount)
                logger.info(f"💵 TP1 LONG: {pair} @ {current_rate:.8f} (move: {tp1_move_pct*100:.1f}%, exit: {tp1_exit_amount*100:.0f}%)")
                return exit_stake
        
        return None
    
    def custom_exit(self, pair: str, trade: 'Trade', current_time: datetime, 
                    current_rate: float, current_profit: float, **kwargs):
        """
        Take Profit final - basado en movimiento de precio.
        Igual para long y short.
        """
        entry_price = trade.open_rate
        tp_pct = float(self.take_profit_pct.value) / 100
        
        if trade.is_short:
            # Para short: TP cuando el precio BAJA tp_pct%
            tp_price = entry_price * (1 - tp_pct)
            if current_rate <= tp_price:
                logger.info(f"💰 TP SHORT: {pair} @ {current_rate:.8f} (move: {tp_pct*100:.1f}%)")
                return "take_profit"
        else:
            # Para long: TP cuando el precio SUBE tp_pct%
            tp_price = entry_price * (1 + tp_pct)
            if current_rate >= tp_price:
                logger.info(f"💰 TP LONG: {pair} @ {current_rate:.8f} (move: {tp_pct*100:.1f}%)")
                return "take_profit"
        
        return None
    
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: str | None, 
                 side: str, **kwargs) -> float:
        """
        Define el apalancamiento a usar.
        """
        return self.config.get('leverage', 1.0)
