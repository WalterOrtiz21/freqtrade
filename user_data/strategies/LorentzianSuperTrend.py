# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: disable=F401
# isort: skip_file
# --- Do not remove these libs ---
import numpy as np
import pandas as pd
from pandas import DataFrame
from datetime import datetime, timedelta
from typing import Optional, Union

from freqtrade.strategy import (IStrategy, IntParameter, DecimalParameter, BooleanParameter, CategoricalParameter)
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
# TRADE LOGGER - Para comparación con TradingView
# =============================================================================
import os
trade_logger = logging.getLogger('LorentzianSuperTrend.trades')
trade_logger.setLevel(logging.INFO)

# Crear directorio de logs si no existe
log_dir = 'user_data/logs/trades'
os.makedirs(log_dir, exist_ok=True)

# Handler para archivo CSV-like
trade_log_file = os.path.join(log_dir, 'lorentzian_trades.log')
trade_handler = logging.FileHandler(trade_log_file, mode='a')
trade_handler.setLevel(logging.INFO)

# Formato CSV-like para fácil comparación con TV
trade_formatter = logging.Formatter('%(asctime)s,%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
trade_handler.setFormatter(trade_formatter)
trade_logger.addHandler(trade_handler)
trade_logger.propagate = False  # No propagar a root logger

# =============================================================================
# 1. MOTOR MATEMÁTICO (NUMBA OPTIMIZED - FIEL A PINE SCRIPT)
# =============================================================================

@njit(parallel=True, fastmath=True)
def numba_lorentzian_distance_prediction(features_norm, labels, max_bars_back, neighbors_count):
    """
    Replica EXACTA de la lógica de Pine Script:
    - Ventana fija (desde el inicio hasta i)
    - Selección heurística de vecinos (lastDistance)
    - Salto de 4 en 4
    """
    n_rows = len(features_norm)
    n_features = features_norm.shape[1]
    predictions = np.zeros(n_rows)

    start_calculation = max_bars_back 
    if start_calculation >= n_rows:
        return predictions

    for i in prange(start_calculation, n_rows):
        current_features = features_norm[i]
        
        last_distance = -1.0
        distances_buffer = np.zeros(neighbors_count * 10) 
        predictions_buffer = np.zeros(neighbors_count * 10)
        buffer_size = 0
        
        start_index = i - max_bars_back
        if start_index < 0: start_index = 0
        end_index = i 
        
        for j in range(start_index, end_index):
            # Pine Logic: if i % 4 (meaning i % 4 != 0)
            # We want to process bars where index % 4 != 0.
            # j is the absolute index here.

            if j % 4 == 0:  # Skip 0, 4, 8... (Matches Pine 'if i%4' which skips when 0)
                continue
                
            dist = 0.0
            for f in range(n_features):
                d = abs(current_features[f] - features_norm[j, f])
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

@njit(fastmath=True)
def numba_calculate_latched_signal(predictions, filter_all):
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
        # Si no cumple filtro, mantiene last_signal
        
        signal[i] = last_signal
        
    return signal

@njit(fastmath=True)
def numba_rational_quadratic_kernel(src, h, r, x):
    """
    Kernel Racional Cuadrático - Versión corregida para matching.
    """
    n = len(src)
    yhat = np.full(n, np.nan)
    denom_base = 2.0 * r * h * h

    # Comenzamos calculo despues de X periodos
    for i in range(h + x, n):
        current_weight = 0.0
        cumulative_weight = 0.0

        # Pine loop: EXACT MATCHING - x + 2 = 27 iterations (when x=25)
        # Pine: for i = 0 to _size + startAtBar (0 to 1 + 25 = 26 inclusive)
        lookback_limit = x + 2
        for j in range(lookback_limit):
            # j representa la 'distancia' en el pasado
            # CORRECTED: Remove x offset (already applied in outer loop start)
            target_idx = i - j
            
            if target_idx < 0: 
                break
                
            y = src[target_idx]
            
            # Formula kernel Rational Quadratic
            # w = (1 + j^2 / (2*r*h^2)) ^ (-r)
            w = (1.0 + (j * j) / denom_base) ** (-r)
            
            current_weight += y * w
            cumulative_weight += w
            
        if cumulative_weight > 0:
            yhat[i] = current_weight / cumulative_weight
        else:
            yhat[i] = src[i] # Fallback
            
    return yhat

@njit(fastmath=True)
def numba_gaussian_kernel(src, h, x):
    """
    Kernel Gaussiano para el suavizado/filtro.
    """
    n = len(src)
    yhat = np.full(n, np.nan)
    denom = 2.0 * h * h

    for i in range(h + x, n):
        current_weight = 0.0
        cumulative_weight = 0.0
        # EXACT MATCHING: Same lookback as Rational Quadratic
        lookback_limit = x + 2
        for j in range(lookback_limit):
            # CORRECTED: Remove x offset (already applied in outer loop start)
            target_idx = i - j
            if target_idx < 0: break
            
            y = src[target_idx]
            w = np.exp(- (j*j) / denom)
            
            current_weight += y * w
            cumulative_weight += w
            
        if cumulative_weight > 0:
            yhat[i] = current_weight / cumulative_weight
        else:
            yhat[i] = src[i]
            
    return yhat

@njit(fastmath=True)
def _numba_supertrend_recalc(close, upper, lower):
    n = len(close)
    st = np.full(n, np.nan)
    direction = np.zeros(n) # 1 = up, -1 = down
    
    # Init simple
    st[0] = upper[0]
    direction[0] = 1
    
    for i in range(1, n):
        prev_upper = upper[i-1]
        prev_lower = lower[i-1]
        prev_st = st[i-1]
        prev_dir = direction[i-1]
        
        curr_close = close[i]
        prev_close = close[i-1]
        curr_upper = upper[i]
        curr_lower = lower[i]
        
        # Ajuste de bandas (no repintado, lógica ST estándar)
        if (curr_lower > prev_lower) or (prev_close < prev_lower):
            eff_lower = curr_lower
        else:
            eff_lower = prev_lower
            
        if (curr_upper < prev_upper) or (prev_close > prev_upper):
            eff_upper = curr_upper
        else:
            eff_upper = prev_upper
            
        # Determinar dirección
        curr_dir = prev_dir
        if prev_dir == -1 and curr_close > eff_upper:
            curr_dir = 1
        elif prev_dir == 1 and curr_close < eff_lower:
            curr_dir = -1
            
        direction[i] = curr_dir
        
        # Escribir para el siguiente ciclo
        if curr_dir == 1:
            st[i] = eff_lower
            lower[i] = eff_lower # persistencia
            # OJO: Upper se "resetea" en teoria, pero mantenemos estado en la variable local eff_upper para la logica
        else:
            st[i] = eff_upper
            upper[i] = eff_upper # persistencia
            
    return st, direction
# =============================================================================
# ESTRATEGIA
# =============================================================================

class LorentzianSuperTrend(IStrategy):
    INTERFACE_VERSION = 3

    # Tracking de eventos logueados (para evitar duplicados)
    _breakeven_logged = set()
    _tp_logged = set()
    _sl_logged = set()

    # ================= PARÁMETROS (Coinciden con JSON) =================
    
    # ML Settings
    neighbors_count = IntParameter(2, 20, default=8, space='buy', optimize=False)
    max_bars_back = IntParameter(500, 3000, default=2000, space='buy', optimize=False)
    
    # Features (Ajustado a Pine)
    f1_period = IntParameter(14, 14, default=14, space='buy', optimize=False)
    f1_smoothing = IntParameter(1, 1, default=1, space='buy', optimize=False) 
    
    f2_ch_len = IntParameter(10, 10, default=10, space='buy', optimize=False)
    f2_avg_len = IntParameter(11, 11, default=11, space='buy', optimize=False)
    
    f3_period = IntParameter(20, 20, default=20, space='buy', optimize=False)
    f3_smoothing = IntParameter(1, 1, default=1, space='buy', optimize=False) 
    
    f4_period = IntParameter(20, 20, default=20, space='buy', optimize=False)
    f4_smoothing = IntParameter(2, 2, default=2, space='buy', optimize=False) 
    
    f5_period = IntParameter(9, 9, default=9, space='buy', optimize=False)
    f5_smoothing = IntParameter(1, 1, default=1, space='buy', optimize=False) 

    # Kernel
    use_kernel_filter = BooleanParameter(default=True, space='buy', optimize=False)
    use_kernel_smoothing = BooleanParameter(default=False, space='buy', optimize=False)
    kernel_h = IntParameter(8, 8, default=8, space='buy', optimize=False)
    kernel_r = DecimalParameter(8.0, 8.0, default=8.0, decimals=1, space='buy', optimize=False)
    kernel_x = IntParameter(25, 25, default=25, space='buy', optimize=False)
    kernel_lag = IntParameter(2, 2, default=2, space='buy', optimize=False)

    # Filters (Sistema 1: ML Signal Filters - aplicados a isNewBuySignal/isNewSellSignal)
    use_volatility_filter = BooleanParameter(default=True, space='buy', optimize=False)
    use_regime_filter = BooleanParameter(default=True, space='buy', optimize=False)
    use_adx_filter = BooleanParameter(default=False, space='buy', optimize=False)
    regime_threshold = DecimalParameter(-0.1, -0.1, default=-0.1, decimals=2, space='buy', optimize=False)
    adx_threshold = IntParameter(20, 20, default=20, space='buy', optimize=False)

    # Sistema 1: EMA/SMA Filters para señales ML (Pine: useEmaFilter, useSmaFilter)
    use_ml_ema_filter = BooleanParameter(default=False, space='buy', optimize=False)  # Pine default: false
    use_ml_sma_filter = BooleanParameter(default=False, space='buy', optimize=False)  # Pine default: false
    ml_ema_period = IntParameter(200, 200, default=200, space='buy', optimize=False)
    ml_sma_period = IntParameter(200, 200, default=200, space='buy', optimize=False)

    # Sistema 2: EMA Filter para posiciones (Pine: ema_filter_long, ema_filter_short, bullish/bearish)
    use_ema_filter_long = BooleanParameter(default=True, space='buy', optimize=False)   # Pine default: true
    use_ema_filter_short = BooleanParameter(default=True, space='buy', optimize=False)  # Pine default: true
    position_ema_period = IntParameter(200, 200, default=200, space='buy', optimize=False)

    # Exits
    use_dynamic_exits = BooleanParameter(default=False, space='sell', optimize=False)
    close_with_supertrend = BooleanParameter(default=True, space='sell', optimize=False)
    close_only_tp = BooleanParameter(default=False, space='sell', optimize=False)  # Pine: close_only_tp

    # Protection & Risk
    stoploss_type = CategoricalParameter(['atr', 'swing'], default='atr', space='protection', optimize=False)
    atr_stop_len = IntParameter(14, 14, default=14, space='protection', optimize=False)
    atr_stop_mult = DecimalParameter(1.5, 1.5, default=1.5, decimals=1, space='protection', optimize=False)
    swing_bars = IntParameter(10, 10, default=10, space='protection', optimize=False)
    
    use_breakeven = BooleanParameter(default=True, space='protection', optimize=False)
    be_rr_long = DecimalParameter(1.0, 1.0, default=1.0, decimals=1, space='protection', optimize=False)
    be_rr_short = DecimalParameter(1.0, 1.0, default=1.0, decimals=1, space='protection', optimize=False)
    
    use_takeprofit = BooleanParameter(default=True, space='protection', optimize=False)
    tp_rr_long = DecimalParameter(3.0, 3.0, default=3.0, decimals=1, space='protection', optimize=False)
    tp_rr_short = DecimalParameter(3.0, 3.0, default=3.0, decimals=1, space='protection', optimize=False)
    tp_percent = DecimalParameter(50, 50, default=50, decimals=0, space='protection', optimize=False)
    close_only_tp = BooleanParameter(default=False, space='protection', optimize=False)

    st_atr_period = IntParameter(9, 9, default=9, space='buy', optimize=False)
    st_factor = DecimalParameter(2.5, 2.5, default=2.5, decimals=1, space='buy', optimize=False)

    # Configuración Freqtrade
    timeframe = '4h'
    minimal_roi = { "0": 100 } # Desactivamos ROI estandard, usamos Custom
    stoploss = -0.99           # Desactivamos SL fijo estandard, usamos Custom ATR
    
    # IMPORTANTE: Subir esto para tener min/max estables en los indicadores "expanding"
    startup_candle_count = 7000 
    
    # REGLA PINE SCRIPT CRUCIAL:
    # El script original dice: if strategy.openprofit > 0 -> close.
    # Esto asegura que no salgamos en pérdidas por cambios de señal o supertrend,
    # solo por Stop Loss.
    exit_profit_only = True 
    
    can_short = True
    use_custom_stoploss = True
    position_adjustment_enable = True 
    process_only_new_candles = True # Similar a process_orders_on_close de Pine

    # ---------------- FUNCIONES DE NORMALIZACION -----------------
    
    def _normalize_expanding(self, series: Union[pd.Series, np.ndarray]) -> pd.Series:
        """Replica normalize() de Pine para valores sin limites definidos."""
        # CORRECCION: Verificar si es numpy array y convertir a pandas Series
        if isinstance(series, np.ndarray):
            series = pd.Series(series)
            
        hist_min = series.expanding().min()
        hist_max = series.expanding().max()
        denom = (hist_max - hist_min).replace(0, 0.000001)
        return (series - hist_min) / denom

    def _normalize_fixed(self, series: Union[pd.Series, np.ndarray], min_val: float, max_val: float) -> pd.Series:
        """Replica rescale(x, min, max, 0, 1) de Pine para RSI y ADX."""
        # Aseguramos compatibilidad si llega numpy array
        if isinstance(series, np.ndarray):
            series = pd.Series(series)
            
        clamped = series.clip(min_val, max_val)
        return (clamped - min_val) / (max_val - min_val)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        # --- FEATURE 1: RSI ---
        rsi_raw = ta.RSI(dataframe, timeperiod=self.f1_period.value)
        rsi_smooth = ta.EMA(rsi_raw, timeperiod=self.f1_smoothing.value) if self.f1_smoothing.value > 1 else rsi_raw
        dataframe['rsi_f1'] = rsi_smooth
        dataframe['f1_norm'] = self._normalize_fixed(rsi_smooth, 0, 100)

        # --- FEATURE 2: WT (WaveTrend) ---
        ap = (dataframe['high'] + dataframe['low'] + dataframe['close']) / 3
        esa = ta.EMA(ap, timeperiod=self.f2_ch_len.value)
        d = ta.EMA(abs(ap - esa), timeperiod=self.f2_ch_len.value)
        ci = (ap - esa) / (0.015 * d)
        tci = ta.EMA(ci, timeperiod=self.f2_avg_len.value)
        wt1 = tci
        wt2 = ta.SMA(wt1, timeperiod=4)
        wt_diff = wt1 - wt2
        dataframe['f2_norm'] = self._normalize_expanding(wt_diff) 
        
        # --- FEATURE 3: CCI ---
        cci_raw = ta.CCI(dataframe, timeperiod=self.f3_period.value)
        cci_smooth = ta.EMA(cci_raw, timeperiod=self.f3_smoothing.value) if self.f3_smoothing.value > 1 else cci_raw
        dataframe['f3_norm'] = self._normalize_expanding(cci_smooth)

        # --- FEATURE 4: ADX ---
        adx_raw = ta.ADX(dataframe, timeperiod=self.f4_period.value)
        adx_smooth = ta.EMA(adx_raw, timeperiod=self.f4_smoothing.value) if self.f4_smoothing.value > 1 else adx_raw
        dataframe['adx_f4'] = adx_smooth
        dataframe['f4_norm'] = self._normalize_fixed(adx_smooth, 0, 100)

        # --- FEATURE 5: RSI ---
        rsi5_raw = ta.RSI(dataframe, timeperiod=self.f5_period.value)
        rsi5_smooth = ta.EMA(rsi5_raw, timeperiod=self.f5_smoothing.value) if self.f5_smoothing.value > 1 else rsi5_raw
        dataframe['f5_norm'] = self._normalize_fixed(rsi5_smooth, 0, 100)

        # --- FILTROS ---
        atr_1 = ta.ATR(dataframe, timeperiod=1)
        atr_10 = ta.ATR(dataframe, timeperiod=10)
        dataframe['volatility_filter_ok'] = atr_1 > atr_10

        # Regime Filter (Algoritmo JDhorty replicado)
        ohlc4 = (dataframe['open'] + dataframe['high'] + dataframe['low'] + dataframe['close']) / 4
        
        @njit
        def numba_regime_filter(src, high, low):
            n = len(src)
            klmf = np.zeros(n)
            val1 = np.zeros(n)
            val2 = np.zeros(n)
            
            # Inicializar con valores razonables
            val1[0] = 0.0
            val2[0] = 0.0
            klmf[0] = src[0]

            for i in range(1, n):
                val1[i] = 0.2 * (src[i] - src[i-1]) + 0.8 * val1[i-1]
                val2[i] = 0.1 * (high[i] - low[i]) + 0.8 * val2[i-1]
                
                omega = abs(val1[i] / val2[i]) if val2[i] != 0 else 0
                alpha = (-omega**2 + np.sqrt(omega**4 + 16 * omega**2)) / 8
                
                klmf[i] = alpha * src[i] + (1 - alpha) * klmf[i-1]
            return klmf

        dataframe['klmf'] = numba_regime_filter(ohlc4.values, dataframe['high'].values, dataframe['low'].values)
        
        # Slope Calculation
        # dataframe['klmf'] es una columna (Series), así que shift funciona
        slope = (dataframe['klmf'] - dataframe['klmf'].shift(1)).abs()
        
        # EMA devuelve numpy array.
        avg_slope = ta.EMA(slope, timeperiod=200)
        
        # CORRECCION DEL ERROR ANTERIOR: usar np.where en lugar de .replace para arrays
        # (avg_slope es un array de numpy aqui)
        avg_slope_safe = np.where(avg_slope == 0, 0.000001, avg_slope)
        normalized_slope = (slope - avg_slope) / avg_slope_safe
        
        dataframe['regime_filter_ok'] = normalized_slope >= self.regime_threshold.value
        
        dataframe['adx_filter_ok'] = dataframe['adx_f4'] > self.adx_threshold.value
        
        # Combinar filtros segun configuracion
        dataframe['filter_all'] = (
            (dataframe['volatility_filter_ok'] | (not self.use_volatility_filter.value)) &
            (dataframe['regime_filter_ok'] | (not self.use_regime_filter.value)) &
            (dataframe['adx_filter_ok'] | (not self.use_adx_filter.value))
        )

        # --- KERNEL (NADARAYA-WATSON) ---
        src_close = dataframe['close'].values.astype(np.float64)
        dataframe['yhat1'] = numba_rational_quadratic_kernel(
            src_close, 
            self.kernel_h.value, 
            float(self.kernel_r.value), 
            self.kernel_x.value
        )
        # kernel Gaussian para yhat2 con lag
        h_lag = max(1, self.kernel_h.value - self.kernel_lag.value)
        dataframe['yhat2'] = numba_gaussian_kernel(src_close, h_lag, self.kernel_x.value)
        
        is_bullish_smooth = dataframe['yhat2'] >= dataframe['yhat1']
        is_bearish_smooth = dataframe['yhat2'] <= dataframe['yhat1']
        is_bullish_rate = dataframe['yhat1'] > dataframe['yhat1'].shift(1)
        is_bearish_rate = dataframe['yhat1'] < dataframe['yhat1'].shift(1)

        if self.use_kernel_smoothing.value:
            dataframe['is_bullish_kernel'] = is_bullish_smooth
            dataframe['is_bearish_kernel'] = is_bearish_smooth
        else:
            dataframe['is_bullish_kernel'] = is_bullish_rate
            dataframe['is_bearish_kernel'] = is_bearish_rate

        # Detectar cambios de color del kernel
        dataframe['is_bearish_change'] = is_bearish_rate & (dataframe['yhat1'].shift(1) < dataframe['yhat1'].shift(2))
        dataframe['is_bullish_change'] = is_bullish_rate & (dataframe['yhat1'].shift(1) > dataframe['yhat1'].shift(2))

        # --- TREND FILTERS ---
        # Sistema 1: EMA/SMA Filters para señales ML (Pine: useEmaFilter, useSmaFilter)
        # Pine: isEmaUptrend = useEmaFilter ? close > ta.ema(close, emaPeriod) : true
        if self.use_ml_ema_filter.value:
            ml_ema = ta.EMA(dataframe, timeperiod=self.ml_ema_period.value)
            dataframe['is_ml_ema_uptrend'] = dataframe['close'] > ml_ema
            dataframe['is_ml_ema_downtrend'] = dataframe['close'] < ml_ema
        else:
            dataframe['is_ml_ema_uptrend'] = True  # Siempre pasa si desactivado
            dataframe['is_ml_ema_downtrend'] = True

        if self.use_ml_sma_filter.value:
            ml_sma = ta.SMA(dataframe, timeperiod=self.ml_sma_period.value)
            dataframe['is_ml_sma_uptrend'] = dataframe['close'] > ml_sma
            dataframe['is_ml_sma_downtrend'] = dataframe['close'] < ml_sma
        else:
            dataframe['is_ml_sma_uptrend'] = True  # Siempre pasa si desactivado
            dataframe['is_ml_sma_downtrend'] = True

        # Sistema 2: EMA para posiciones (Pine: bullish/bearish, ema200)
        # Pine: bullish = close > ema200 (SIEMPRE calculado)
        position_ema = ta.EMA(dataframe, timeperiod=self.position_ema_period.value)
        dataframe['bullish'] = dataframe['close'] > position_ema
        dataframe['bearish'] = dataframe['close'] < position_ema

        # --- ML LORENTZIAN PREDICTION ---
        # CRITICAL CORRECTION: Pine uses src[4] which is 4 bars in the FUTURE
        # Pine Strategy (line 233): y_train_series = src[4] < src[0] ? direction_s.short : src[4] > src[0] ? direction_s.long : direction_s.neutral
        # Where direction_s.short = -1, direction_s.long = 1, direction_s.neutral = 0
        # So:
        # - future < current → price will drop → label = -1 (short/bearish)
        # - future > current → price will rise → label = 1 (long/bullish)
        future_close = dataframe['close'].shift(-4)  # 4 bars in the FUTURE
        labels = np.where(future_close < dataframe['close'], -1.0,   # futuro < actual → bajará → short = -1
                  np.where(future_close > dataframe['close'], 1.0,   # futuro > actual → subirá → long = 1
                           0.0))
        labels = np.nan_to_num(labels, nan=0.0)
        
        # Preparar features
        feature_cols = ['f1_norm', 'f2_norm', 'f3_norm', 'f4_norm', 'f5_norm']
        features_data = dataframe[feature_cols].fillna(0.5).values.astype(np.float64)
        
        predictions = numba_lorentzian_distance_prediction(
            features_data,
            labels.astype(np.float64),
            int(self.max_bars_back.value),
            int(self.neighbors_count.value)
        )
        dataframe['prediction'] = predictions

        # Calcular Señal final (Signal Latching)
        filter_all_np = dataframe['filter_all'].fillna(False).values.astype(bool)
        dataframe['signal'] = numba_calculate_latched_signal(predictions, filter_all_np)
        
        # --- SUPERTREND ---
        dataframe['atr_st'] = ta.ATR(dataframe, timeperiod=self.st_atr_period.value)
        hl2 = (dataframe['high'] + dataframe['low']) / 2
        matr = float(self.st_factor.value) * dataframe['atr_st']
        upperband = hl2 + matr
        lowerband = hl2 - matr
        
        st, direction = _numba_supertrend_recalc(
            dataframe['close'].values.astype(np.float64),
            upperband.values.astype(np.float64),
            lowerband.values.astype(np.float64)
        )
        dataframe['supertrend'] = st
        dataframe['st_direction'] = direction

        # SuperTrend Crossovers (Pine: closelong_supertrend, closeshort_supertrend)
        # Pine: closelong_supertrend = ta.crossunder(close, supertrend)
        # Pine: closeshort_supertrend = ta.crossover(close, supertrend)
        import freqtrade.vendor.qtpylib.indicators as qtpylib
        dataframe['close_cross_below_st'] = qtpylib.crossed_below(dataframe['close'], dataframe['supertrend'])
        dataframe['close_cross_above_st'] = qtpylib.crossed_above(dataframe['close'], dataframe['supertrend'])

        # --- Swing Points (Para Stop Loss) ---
        dataframe['swing_high'] = dataframe['high'].rolling(self.swing_bars.value).max().shift(1)
        dataframe['swing_low'] = dataframe['low'].rolling(self.swing_bars.value).min().shift(1)
        dataframe['atr_sl'] = ta.ATR(dataframe, timeperiod=self.atr_stop_len.value)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Pine Strategy Entry Logic (líneas 288, 333):

        isBuySignal = signal == direction_s.long and isEmaUptrend and isSmaUptrend
        isNewBuySignal = isBuySignal and isDifferentSignalType

        startLongTrade = isNewBuySignal and isBullish and isEmaUptrend and isSmaUptrend

        Luego en strategy.entry (línea 726):
        if not bought and buy and ... and bullish and ema_filter_long
        """

        # Señales ML base
        ml_buy_signal = (dataframe['signal'] == 1)
        ml_sell_signal = (dataframe['signal'] == -1)

        # Pine: isBuySignal = signal == direction_s.long and isEmaUptrend and isSmaUptrend
        # Aplicar Sistema 1 (ML filters) a las señales
        is_buy_signal = (
            ml_buy_signal &
            dataframe['is_ml_ema_uptrend'] &
            dataframe['is_ml_sma_uptrend']
        )
        is_sell_signal = (
            ml_sell_signal &
            dataframe['is_ml_ema_downtrend'] &
            dataframe['is_ml_sma_downtrend']
        )

        # Pine: isNewBuySignal = isBuySignal and isDifferentSignalType
        dataframe['is_new_buy_signal'] = is_buy_signal & (is_buy_signal.shift(1) == False)
        dataframe['is_new_sell_signal'] = is_sell_signal & (is_sell_signal.shift(1) == False)

        # Pine (línea 333): startLongTrade = isNewBuySignal and isBullish and isEmaUptrend and isSmaUptrend
        # isBullish = kernel filter, isEmaUptrend/isSmaUptrend ya incluidos en isNewBuySignal
        start_long_trade = (
            dataframe['is_new_buy_signal'] &
            dataframe['is_bullish_kernel'] &  # Pine: isBullish
            dataframe['is_ml_ema_uptrend'] &  # Pine: isEmaUptrend (redundante pero matching)
            dataframe['is_ml_sma_uptrend']    # Pine: isSmaUptrend (redundante pero matching)
        )

        start_short_trade = (
            dataframe['is_new_sell_signal'] &
            dataframe['is_bearish_kernel'] &  # Pine: isBearish
            dataframe['is_ml_ema_downtrend'] & # Pine: isEmaDowntrend
            dataframe['is_ml_sma_downtrend']   # Pine: isSmaDowntrend
        )

        # Pine (línea 726): if not bought and buy and ... and bullish and ema_filter_long
        # Sistema 2: Filtro de posiciones (bullish/bearish)
        if self.use_ema_filter_long.value:
            long_position_allowed = start_long_trade & dataframe['bullish']
        else:
            long_position_allowed = start_long_trade

        if self.use_ema_filter_short.value:
            short_position_allowed = start_short_trade & dataframe['bearish']
        else:
            short_position_allowed = start_short_trade

        # Aplicar condiciones finales
        dataframe.loc[long_position_allowed & (dataframe['volume'] > 0), 'enter_long'] = 1

        if self.can_short:
            dataframe.loc[short_position_allowed & (dataframe['volume'] > 0), 'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Pine Strategy Exit Logic (líneas 665, 700):

        Long exits:
        if bought and sell and strategy.openprofit>0 and not close_only_tp or
           bought and closelong_supertrend and close_withsupertrend and strategy.openprofit>0 and not close_only_tp

        Short exits:
        if sold and buy and strategy.openprofit>0 and not close_only_tp or
           sold and closeshort_supertrend and close_withsupertrend and strategy.openprofit>0 and not close_only_tp

        NOTA: La verificación de profit (strategy.openprofit>0) se maneja en custom_exit()
              porque populate_exit_trend no tiene acceso a current_profit.
              También usamos exit_profit_only = True para asegurar exits solo con profit.
        """
        dataframe.loc[:, ['exit_long', 'exit_short']] = 0

        # NO aplicar si close_only_tp está activado
        if not self.close_only_tp.value:
            # Exit por señal opuesta (Pine: bought and sell)
            dataframe.loc[dataframe['is_new_sell_signal'], 'exit_long'] = 1
            dataframe.loc[dataframe['is_new_buy_signal'], 'exit_short'] = 1

            # Exit por SuperTrend crossover/crossunder (Pine: closelong_supertrend, closeshort_supertrend)
            if self.close_with_supertrend.value:
                # Pine: closelong_supertrend = ta.crossunder(close, supertrend)
                dataframe.loc[dataframe['close_cross_below_st'], 'exit_long'] = 1
                # Pine: closeshort_supertrend = ta.crossover(close, supertrend)
                dataframe.loc[dataframe['close_cross_above_st'], 'exit_short'] = 1

        # Dynamic exits (kernel change)
        if self.use_dynamic_exits.value:
            dataframe.loc[dataframe['is_bearish_change'] & dataframe['filter_all'], 'exit_long'] = 1
            dataframe.loc[dataframe['is_bullish_change'] & dataframe['filter_all'], 'exit_short'] = 1

        return dataframe

    def _get_entry_candle(self, pair, open_date):
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        tf_min = int(self.timeframe[:-1]) * (60 if 'h' in self.timeframe else 1)
        signal_date = open_date - timedelta(minutes=tf_min)
        try:
            return dataframe.loc[dataframe['date'] == signal_date].iloc[0]
        except:
            return None

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        
        entry_data = self._get_entry_candle(pair, trade.open_date_utc)
        if entry_data is None: return self.stoploss 
        
        entry_price = trade.open_rate
        sl_dist = 0.0
        
        if self.stoploss_type.value == 'atr':
            sl_dist = entry_data['atr_sl'] * float(self.atr_stop_mult.value)
        else:
            if trade.is_short:
                sl_dist = abs(entry_data['swing_high'] - entry_price)
            else:
                sl_dist = abs(entry_price - entry_data['swing_low'])

        if sl_dist <= 0: return self.stoploss

        if trade.is_short:
            fixed_stop_price = entry_price + sl_dist
        else:
            fixed_stop_price = entry_price - sl_dist
            
        if self.use_breakeven.value:
            if trade.is_short:
                be_dist = sl_dist * float(self.be_rr_short.value)
                be_trigger = entry_price - be_dist
                if trade.min_rate <= be_trigger:
                    # Pine: sl_short:=strategy.position_avg_price (exactamente el entry, sin buffer)
                    fixed_stop_price = entry_price
                    # Log breakeven activation (una sola vez)
                    if trade.id not in self._breakeven_logged:
                        self._breakeven_logged.add(trade.id)
                        profit_pct = ((entry_price - current_rate) / entry_price) * 100
                        be_date = current_time.strftime('%Y-%m-%d %H:%M:%S')
                        entry_date = trade.open_date_utc.strftime('%Y-%m-%d %H:%M:%S')
                        trade_logger.info(f"BE_ACTIVATED,SHORT,{pair},{current_rate:.8f},{profit_pct:.2f},R:R={self.be_rr_short.value},Date={be_date},EntryDate={entry_date}")
                        logger.info(f"⚖️ Breakeven SHORT activated: {pair} @ {current_rate:.8f} | R:R={self.be_rr_short.value} | Date: {be_date}")
            else:
                be_dist = sl_dist * float(self.be_rr_long.value)
                be_trigger = entry_price + be_dist
                if trade.max_rate >= be_trigger:
                    # Pine: sl_long:=strategy.position_avg_price (exactamente el entry, sin buffer)
                    fixed_stop_price = entry_price
                    # Log breakeven activation (una sola vez)
                    if trade.id not in self._breakeven_logged:
                        self._breakeven_logged.add(trade.id)
                        profit_pct = ((current_rate - entry_price) / entry_price) * 100
                        be_date = current_time.strftime('%Y-%m-%d %H:%M:%S')
                        entry_date = trade.open_date_utc.strftime('%Y-%m-%d %H:%M:%S')
                        trade_logger.info(f"BE_ACTIVATED,LONG,{pair},{current_rate:.8f},{profit_pct:.2f},R:R={self.be_rr_long.value},Date={be_date},EntryDate={entry_date}")
                        logger.info(f"⚖️ Breakeven LONG activated: {pair} @ {current_rate:.8f} | R:R={self.be_rr_long.value} | Date: {be_date}")

        # Retornar SL como porcentaje relativo al precio actual
        # Para LONG: SL abajo del current → negativo
        # Para SHORT: SL arriba del current → positivo; SL abajo del current → negativo (debería cerrar)
        if trade.is_short:
             return (fixed_stop_price - current_rate) / current_rate
        else:
             return (fixed_stop_price - current_rate) / current_rate

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: str | None, side: str,
                 **kwargs) -> float:
        """
        Customize leverage for each new trade. This method is only called in futures mode.
        """
        return self.config.get('leverage', 1.0)

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float,
                              min_stake: float, max_stake: float,
                              current_entry_rate: float, current_exit_rate: float,
                              current_entry_profit: float, current_exit_profit: float,
                              **kwargs) -> Optional[float]:
        
        if self.use_takeprofit.value and trade.nr_of_successful_exits == 0:
            entry_data = self._get_entry_candle(trade.pair, trade.open_date_utc)
            if entry_data is None: return None
            
            sl_dist = 0.0
            if self.stoploss_type.value == 'atr':
                sl_dist = entry_data['atr_sl'] * float(self.atr_stop_mult.value)
            else:
                if trade.is_short: sl_dist = abs(entry_data['swing_high'] - trade.open_rate)
                else: sl_dist = abs(trade.open_rate - entry_data['swing_low'])
            
            if sl_dist == 0: return None

            if trade.is_short:
                tp_price = trade.open_rate - (sl_dist * float(self.tp_rr_short.value))
                if current_rate <= tp_price:
                    # Log take profit (una sola vez)
                    if trade.id not in self._tp_logged:
                        self._tp_logged.add(trade.id)
                        profit_pct = current_profit * 100
                        profit_abs = trade.calc_profit(current_rate) * (self.tp_percent.value / 100)
                        tp_date = current_time.strftime('%Y-%m-%d %H:%M:%S')
                        entry_date = trade.open_date_utc.strftime('%Y-%m-%d %H:%M:%S')
                        trade_logger.info(f"TP_PARTIAL,SHORT,{trade.pair},{current_rate:.8f},{profit_pct:.2f},{profit_abs:.8f},R:R={self.tp_rr_short.value},{self.tp_percent.value}%,Date={tp_date},EntryDate={entry_date}")
                        logger.info(f"💰 Take Profit SHORT ({self.tp_percent.value}%): {trade.pair} @ {current_rate:.8f} | "
                                   f"Profit: {profit_pct:.2f}% | R:R={self.tp_rr_short.value} | Date: {tp_date}")
                    return -(trade.stake_amount * (self.tp_percent.value / 100))
            else:
                tp_price = trade.open_rate + (sl_dist * float(self.tp_rr_long.value))
                if current_rate >= tp_price:
                    # Log take profit (una sola vez)
                    if trade.id not in self._tp_logged:
                        self._tp_logged.add(trade.id)
                        profit_pct = current_profit * 100
                        profit_abs = trade.calc_profit(current_rate) * (self.tp_percent.value / 100)
                        tp_date = current_time.strftime('%Y-%m-%d %H:%M:%S')
                        entry_date = trade.open_date_utc.strftime('%Y-%m-%d %H:%M:%S')
                        trade_logger.info(f"TP_PARTIAL,LONG,{trade.pair},{current_rate:.8f},{profit_pct:.2f},{profit_abs:.8f},R:R={self.tp_rr_long.value},{self.tp_percent.value}%,Date={tp_date},EntryDate={entry_date}")
                        logger.info(f"💰 Take Profit LONG ({self.tp_percent.value}%): {trade.pair} @ {current_rate:.8f} | "
                                   f"Profit: {profit_pct:.2f}% | R:R={self.tp_rr_long.value} | Date: {tp_date}")
                    return -(trade.stake_amount * (self.tp_percent.value / 100))

        return None

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                            time_in_force: str, current_time: datetime, entry_tag: Optional[str],
                            side: str, **kwargs) -> bool:
        """
        Logging de entradas para comparación con TradingView
        Incluye nivel de SL calculado
        """
        # Determinar tipo de entrada
        direction = "LONG" if side == "long" else "SHORT"
        entry_date = current_time.strftime('%Y-%m-%d %H:%M:%S')

        # Calcular nivel de SL para logging
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        # Obtener el último candle completo (candle de señal)
        last_candle = dataframe.iloc[-1].squeeze()

        # Calcular SL según configuración (igual que en custom_stoploss)
        sl_price = None
        sl_dist = None
        sl_pct = None

        if self.stoploss_type.value == 'atr':
            # Pine: longStopLoss = close - ta.atr(14) * multiplier
            # Pine: shortStopLoss = close + ta.atr(14) * multiplier
            atr_value = last_candle.get('atr_sl', 0)
            sl_dist = atr_value * float(self.atr_stop_mult.value)

            if side == "short":
                sl_price = rate + sl_dist
                sl_pct = (sl_dist / rate) * 100
            else:
                sl_price = rate - sl_dist
                sl_pct = (sl_dist / rate) * 100

            sl_type = f"ATR({self.atr_stop_len.value})*{self.atr_stop_mult.value}"
        else:
            # Swing High/Low
            if side == "short":
                swing_high = last_candle.get('swing_high', rate)
                sl_dist = abs(swing_high - rate)
                sl_price = swing_high
            else:
                swing_low = last_candle.get('swing_low', rate)
                sl_dist = abs(rate - swing_low)
                sl_price = swing_low

            sl_pct = (sl_dist / rate) * 100
            sl_type = f"SWING({self.swing_bars.value})"

        # Calcular niveles de BE y TP
        be_price = None
        tp_price = None

        if sl_dist and sl_dist > 0:
            if side == "short":
                be_price = rate - (sl_dist * float(self.be_rr_short.value))
                tp_price = rate - (sl_dist * float(self.tp_rr_short.value))
            else:
                be_price = rate + (sl_dist * float(self.be_rr_long.value))
                tp_price = rate + (sl_dist * float(self.tp_rr_long.value))

        # Formatear valores para logging
        sl_str = f"{sl_price:.8f}" if sl_price else "N/A"
        sl_pct_str = f"{sl_pct:.2f}" if sl_pct else "N/A"
        be_str = f"{be_price:.8f}" if be_price else "N/A"
        tp_str = f"{tp_price:.8f}" if tp_price else "N/A"
        sl_type_str = sl_type if sl_type else "N/A"

        # Log formato CSV: Type,Direction,Pair,EntryPrice,Amount,EntryDate,SL_Price,SL_Pct,BE_Price,TP_Price,Tag
        trade_logger.info(f"ENTRY,{direction},{pair},{rate:.8f},{amount:.8f},{entry_date},"
                         f"SL={sl_str}({sl_pct_str}% {sl_type_str}),"
                         f"BE={be_str},"
                         f"TP={tp_str},"
                         f"{entry_tag or 'manual'}")

        logger.info(f"✅ {direction} Entry: {pair} @ {rate:.8f} | Amount: {amount:.8f} | Date: {entry_date}\n"
                   f"   💠 SL: {sl_str} ({sl_pct_str}% / {sl_type_str})\n"
                   f"   🎯 BE: {be_str} | TP: {tp_str} | Tag: {entry_tag}")

        return True

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                          rate: float, time_in_force: str, exit_reason: str,
                          current_time: datetime, **kwargs) -> bool:
        """
        LOGGING DE TODAS LAS SALIDAS (SL, TP, ROI, Señal, etc.)
        Se ejecuta SIEMPRE antes de cerrar una operación, sin importar la razón.
        """
        direction = "SHORT" if trade.is_short else "LONG"
        entry_date = trade.open_date_utc.strftime('%Y-%m-%d %H:%M:%S')
        entry_price = trade.open_rate
        exit_date = current_time.strftime('%Y-%m-%d %H:%M:%S')

        # Calcular profit
        current_profit_pct = trade.calc_profit_ratio(rate) * 100
        current_profit_abs = trade.calc_profit(rate)

        # Determinar el tipo de salida basado en exit_reason
        exit_type = "EXIT"
        if "stop_loss" in exit_reason.lower() or "stoploss" in exit_reason.lower():
            exit_type = "SL_EXIT"
        elif "roi" in exit_reason.lower():
            exit_type = "ROI_EXIT"
        elif "trailing" in exit_reason.lower():
            exit_type = "TRAILING_EXIT"
        elif exit_reason in ["exit_long", "exit_short"]:
            exit_type = "SIGNAL_EXIT"
        elif "partial" in exit_reason.lower():
            exit_type = "TP_PARTIAL"

        # Log formato CSV: Type,Direction,Reason,Pair,ExitPrice,ExitDate,Profit%,ProfitAbs,EntryDate,EntryPrice
        trade_logger.info(f"{exit_type},{direction},{exit_reason},{pair},{rate:.8f},{exit_date},"
                         f"{current_profit_pct:.2f},{current_profit_abs:.8f},{entry_date},{entry_price:.8f}")

        profit_emoji = "🟢" if current_profit_abs > 0 else "🔴"
        logger.info(f"{profit_emoji} {direction} {exit_type} ({exit_reason}): {pair} @ {rate:.8f} | "
                   f"Profit: {current_profit_pct:.2f}% ({current_profit_abs:.8f}) | "
                   f"Date: {exit_date} | EntryPrice: {entry_price:.8f}")

        return True

    def custom_exit(self, pair: str, trade: 'Trade', current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs):
        """
        Logging de salidas con razón específica (incluye detección de Stop Loss)
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        exit_reason = None
        direction = "SHORT" if trade.is_short else "LONG"
        entry_date = trade.open_date_utc.strftime('%Y-%m-%d %H:%M:%S')
        entry_price = trade.open_rate

        # Detectar si estamos cerca del Stop Loss (pérdida >= 80% del SL)
        # Esto loguea ANTES de que Freqtrade ejecute el SL automático
        if current_profit < -0.01:  # En pérdida > 1%
            # Calcular nivel de SL esperado
            entry_data = self._get_entry_candle(pair, trade.open_date_utc)
            if entry_data is not None:
                sl_dist = 0.0
                if self.stoploss_type.value == 'atr':
                    sl_dist = entry_data['atr_sl'] * float(self.atr_stop_mult.value)
                else:
                    if trade.is_short:
                        sl_dist = abs(entry_data['swing_high'] - entry_price)
                    else:
                        sl_dist = abs(entry_price - entry_data['swing_low'])

                if sl_dist > 0:
                    if trade.is_short:
                        sl_level = (entry_price + sl_dist - current_rate) / current_rate
                    else:
                        sl_level = (entry_price - sl_dist - current_rate) / current_rate

                    # Si estamos muy cerca del SL (80%+), loguear
                    if abs(current_profit) >= abs(sl_level) * 0.8:
                        if trade.id not in getattr(self, '_sl_logged', set()):
                            if not hasattr(self, '_sl_logged'):
                                self._sl_logged = set()
                            self._sl_logged.add(trade.id)

                            profit_pct = current_profit * 100
                            profit_abs = trade.calc_profit(current_rate)
                            sl_type = self.stoploss_type.value.upper()

                            trade_logger.info(f"SL_HIT,{direction},{sl_type},{pair},{current_rate:.8f},{profit_pct:.2f},{profit_abs:.8f},EntryDate={entry_date},EntryPrice={entry_price:.8f}")
                            logger.info(f"🛑 Stop Loss {direction} ({sl_type}): {pair} @ {current_rate:.8f} | "
                                       f"Loss: {profit_pct:.2f}% ({profit_abs:.8f}) | Entry: {entry_date} @ {entry_price:.8f}")

        # Detectar razón de salida normal (señal/supertrend)
        if last_candle.get('is_new_sell_signal', False) and trade.is_short == False:
            exit_reason = "SIGNAL_FLIP_SHORT"
        elif last_candle.get('is_new_buy_signal', False) and trade.is_short == True:
            exit_reason = "SIGNAL_FLIP_LONG"
        elif last_candle.get('close_cross_below_st', False) and trade.is_short == False:
            exit_reason = "SUPERTREND_CROSS_BELOW"
        elif last_candle.get('close_cross_above_st', False) and trade.is_short == True:
            exit_reason = "SUPERTREND_CROSS_ABOVE"

        # Si hay razón de exit y tenemos profit, loguear y salir
        if exit_reason and current_profit > 0 and not self.close_only_tp.value:
            profit_pct = current_profit * 100
            profit_abs = trade.calc_profit(current_rate)

            # Log formato CSV: Type,Direction,Reason,Pair,ExitPrice,Profit%,ProfitAbs,EntryDate,EntryPrice
            trade_logger.info(f"EXIT,{direction},{exit_reason},{pair},{current_rate:.8f},{profit_pct:.2f},{profit_abs:.8f},EntryDate={entry_date},EntryPrice={entry_price:.8f}")

            logger.info(f"🔴 {direction} Exit ({exit_reason}): {pair} @ {current_rate:.8f} | "
                       f"Profit: {profit_pct:.2f}% ({profit_abs:.8f}) | Entry: {entry_date} @ {entry_price:.8f}")

            return exit_reason

        return None
