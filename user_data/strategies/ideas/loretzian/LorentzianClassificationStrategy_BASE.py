import numpy as np
import pandas as pd
from pandas import DataFrame
import math
import logging
from datetime import datetime
from typing import Optional, Union

from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter, 
                                IntParameter, RealParameter)
from freqtrade.persistence import Trade
import talib.abstract as ta

# --- IMPORTACIONES NUMBA ---
try:
    from numba import njit, prange
    _HAS_NUMBA = True
except ImportError:
    print("FATAL: Numba is required for Lorentzian Classification.")
    raise ImportError("Please install numba: pip install numba")

logger = logging.getLogger(__name__)

# =============================================================================
# MOTORES MATEMÁTICOS OPTIMIZADOS (NUMBA - JIT)
# =============================================================================

@njit(parallel=True, fastmath=True)
def numba_lorentzian_prediction(features_norm, labels, max_bars_back, neighbors_count):
    """
    Replica EXACTA de la lógica de Pine Script:
    - Ventana fija (desde el inicio hasta i)
    - Selección heurística de vecinos (lastDistance)
    - Salto de 4 en 4 (Pine: i%4, que es true para 1, 2, 3, 5... y false para 0, 4...)
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
            
            if j % 4 == 0: # Skip 0, 4, 8... (Matches Pine 'if i%4' which skips when 0)
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
def numba_rational_quadratic_kernel(src, h, r, x):
    """
    Kernel Racional Cuadrático - Versión corregida para matching.
    Pine: Loop i=0 to size. y = src[i]. w uses i.
    Python: i is current bar. j is lag. target = i - j.
    """
    n = len(src)
    yhat = np.full(n, np.nan)
    denom_base = 2.0 * r * h * h
    
    # Comenzamos calculo despues de X periodos (aunque Pine empieza en 0, aqui evitamos errores de indice)
    for i in range(h + x, n):
        current_weight = 0.0
        cumulative_weight = 0.0
        
        # Pine uses _size + startAtBar as loop limit.
        # _size = 1 (always), startAtBar = x = 25 (default).
        # Pine: for i = 0 to (1 + 25) = for i = 0 to 26 (inclusive).
        # This means 27 iterations: 0, 1, 2, ..., 26.
        # Python equivalent: range(27) = 0 to 26.
        # Formula: lookback_limit = _size + x + 1 = 1 + x + 1 = x + 2.
        lookback_limit = x + 2  # EXACT MATCHING: 27 iterations (x=25 -> 27)
        
        for j in range(lookback_limit):
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
def numba_kalman_filter(src, high, low):
    n = len(src)
    klmf = np.zeros(n)
    value1 = np.zeros(n)
    value2 = np.zeros(n)
    klmf[0] = src[0]
    
    for i in range(1, n):
        value1[i] = 0.2 * (src[i] - src[i-1]) + 0.8 * value1[i-1]
        value2[i] = 0.1 * (high[i] - low[i]) + 0.8 * value2[i-1]
        
        omega = np.abs(value1[i] / value2[i]) if value2[i] != 0 else 0
        alpha = (-omega**2 + np.sqrt(omega**4 + 16 * omega**2)) / 8
        klmf[i] = alpha * src[i] + (1 - alpha) * klmf[i-1]
        
    return klmf

# =============================================================================
# ESTRATEGIA
# =============================================================================

class LorentzianClassificationStrategy(IStrategy):
    INTERFACE_VERSION = 3

    # --- PARÁMETROS ML (BUY) ---
    neighbors_count = IntParameter(4, 16, default=8, space='buy', optimize=True)
    buy_min_score = IntParameter(0, neighbors_count, default=0, space='buy', optimize=True)
    max_bars_back = IntParameter(1500, 5000, default=2000, space='buy', optimize=False)
    # feature_lookback REMOVED
    
    # --- PARÁMETROS KERNEL (BUY) ---
    kernel_lookback = IntParameter(2, 50, default=8, space='buy', optimize=True) # Pine default: 8
    kernel_weighting = RealParameter(0.1, 10.0, default=8.0, space='buy', optimize=True)
    kernel_regression = IntParameter(10, 50, default=25, space='buy', optimize=False)
    use_kernel_filter = BooleanParameter(default=True, space='buy', optimize=True)
    
    # --- FILTROS ADICIONALES (BUY) ---
    use_volatility_filter = BooleanParameter(default=True, space='buy', optimize=True)
    use_regime_filter = BooleanParameter(default=True, space='buy', optimize=True)
    regime_threshold = RealParameter(-0.2, 0.3, default=-0.1, space='buy', optimize=True) # Pine default: -0.1
    use_adx_filter = BooleanParameter(default=False, space='buy', optimize=True)
    use_ema_filter = BooleanParameter(default=False, space='buy', optimize=True) # Pine default: False
    use_sma_filter = BooleanParameter(default=False, space='buy', optimize=True) # New: SMA Filter
    
    # --- PULLBACK (BUY) ---
    use_kernel_pullback_filter = BooleanParameter(default=True, space='buy', optimize=True)
    kernel_pullback_tolerance = RealParameter(0.001, 0.05, default=0.01, space='buy', optimize=True)
    
    # --- INDICADORES (BUY) ---
    rsi_period_1 = IntParameter(5, 30, default=14, space='buy', optimize=True)
    rsi_smoothing_1 = IntParameter(1, 10, default=1, space='buy', optimize=True) # f1_paramB
    rsi_period_2 = IntParameter(5, 30, default=9, space='buy', optimize=True)
    rsi_smoothing_2 = IntParameter(1, 10, default=1, space='buy', optimize=True) # f5_paramB
    wt_channel_length = IntParameter(5, 30, default=10, space='buy', optimize=True)
    wt_average_length = IntParameter(5, 30, default=11, space='buy', optimize=True)
    cci_period = IntParameter(10, 40, default=20, space='buy', optimize=True)
    cci_smoothing = IntParameter(1, 10, default=1, space='buy', optimize=True) # f3_paramB
    adx_period = IntParameter(10, 40, default=20, space='buy', optimize=True)

    # --- SALIDAS (SELL) ---
    exit_mode = CategoricalParameter(['fixed_bars', 'signal', 'dynamic', 'all'], default='signal', space='sell', optimize=True)
    fixed_exit_bars = IntParameter(1, 20, default=4, space='sell', optimize=True)
    
    # Break Even
    use_break_even = BooleanParameter(default=True, space='sell', optimize=True)
    break_even_profit_threshold = DecimalParameter(0.005, 0.05, default=0.015, space='sell', optimize=True)
    break_even_offset = DecimalParameter(0.001, 0.02, default=0.003, space='sell', optimize=True)

    # CONFIGURACIÓN
    timeframe = '1h'
    minimal_roi = { "0": 100.0 } 
    stoploss = -0.99 
    
    startup_candle_count: int = 5000 
    
    can_short = True
    
    order_types = {
        'entry': 'market',
        'exit': 'market',
        'stoploss': 'market',
        'stoploss_on_exchange': False,
    }

    # ---------------- FUNCIONES DE NORMALIZACION -----------------
    
    def _normalize_expanding(self, series: Union[pd.Series, np.ndarray]) -> pd.Series:
        """Replica normalize() de Pine para valores sin limites definidos."""
        if isinstance(series, np.ndarray):
            series = pd.Series(series)
        hist_min = series.expanding().min()
        hist_max = series.expanding().max()
        denom = (hist_max - hist_min).replace(0, 0.000001)
        return (series - hist_min) / denom

    def _normalize_fixed(self, series: Union[pd.Series, np.ndarray], min_val: float, max_val: float) -> pd.Series:
        """Replica rescale(x, min, max, 0, 1) de Pine para RSI y ADX."""
        if isinstance(series, np.ndarray):
            series = pd.Series(series)
        clamped = series.clip(min_val, max_val)
        return (clamped - min_val) / (max_val - min_val)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # === 1. Features Normalizados para ML (CORREGIDO) ===
        
        # RSI 1 (Fixed 0-100)
        rsi1 = ta.RSI(dataframe, timeperiod=self.rsi_period_1.value)
        # Smoothing RSI 1
        if self.rsi_smoothing_1.value > 1:
            rsi1 = ta.EMA(rsi1, timeperiod=self.rsi_smoothing_1.value)
        dataframe['rsi_norm_1'] = self._normalize_fixed(rsi1, 0, 100)
        
        # RSI 2 (Fixed 0-100)
        rsi2 = ta.RSI(dataframe, timeperiod=self.rsi_period_2.value)
        # Smoothing RSI 2 (Added per verification)
        if self.rsi_smoothing_2.value > 1:
            rsi2 = ta.EMA(rsi2, timeperiod=self.rsi_smoothing_2.value)
        dataframe['rsi_norm_5'] = self._normalize_fixed(rsi2, 0, 100)
        
        # CCI (Expanding)
        cci = ta.CCI(dataframe, timeperiod=self.cci_period.value)
        # Smoothing CCI
        if self.cci_smoothing.value > 1:
            cci = ta.EMA(cci, timeperiod=self.cci_smoothing.value)
        dataframe['cci_norm'] = self._normalize_expanding(cci)
        
        # ADX (Fixed 0-100)
        adx = ta.ADX(dataframe, timeperiod=self.adx_period.value)
        dataframe['adx_norm'] = self._normalize_fixed(adx, 0, 100)
        
        # WT (Expanding)
        ap = (dataframe['high'] + dataframe['low'] + dataframe['close']) / 3
        esa = ta.EMA(ap, timeperiod=self.wt_channel_length.value)
        d = ta.EMA(abs(ap - esa), timeperiod=self.wt_channel_length.value)
        ci = (ap - esa) / (0.015 * d)
        tci = ta.EMA(ci, timeperiod=self.wt_average_length.value)
        wt1 = tci
        wt2 = ta.SMA(wt1, timeperiod=4)
        wt_diff = wt1 - wt2
        dataframe['wt_norm'] = self._normalize_expanding(wt_diff)

        # === 2. Indicadores Estándar ===
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['adx'] = adx # Raw ADX for filter
        dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)
        dataframe['sma_200'] = ta.SMA(dataframe, timeperiod=200) # New: SMA for filter

        # === 3. Filtros (CORREGIDO) ===
        
        # Volatility Filter: ATR(1) > ATR(10)
        atr1 = ta.ATR(dataframe, timeperiod=1)
        atr10 = ta.ATR(dataframe, timeperiod=10)
        dataframe['volatility_ok'] = atr1 > atr10
        
        # Regime Filter (Kalman)
        ohlc4 = (dataframe['open'] + dataframe['high'] + dataframe['low'] + dataframe['close']) / 4
        klmf = numba_kalman_filter(ohlc4.values, dataframe['high'].values, dataframe['low'].values)
        klmf_s = pd.Series(klmf, index=dataframe.index)
        slope = (klmf_s - klmf_s.shift(1))
        
        # En Pine: filter.regime = val > threshold. Pero aqui 'val' es el slope normalizado?
        # Pine: normalized_slope_decline = (absCurveSlope - exponentialAverageAbsCurveSlope) / exponentialAverageAbsCurveSlope
        # FIX: Use slope.abs() to match Pine's absCurveSlope
        avg_slope = ta.EMA(slope.abs(), timeperiod=200)
        avg_slope_safe = np.where(avg_slope == 0, 0.000001, avg_slope)
        normalized_slope = (slope.abs() - avg_slope) / avg_slope_safe
        dataframe['regime_ok'] = normalized_slope > self.regime_threshold.value

        # EMA Filter
        dataframe['ema_uptrend'] = dataframe['close'] > dataframe['ema_200']
        dataframe['ema_downtrend'] = dataframe['close'] < dataframe['ema_200']
        
        # SMA Filter
        dataframe['sma_uptrend'] = dataframe['close'] > dataframe['sma_200']
        dataframe['sma_downtrend'] = dataframe['close'] < dataframe['sma_200']

        # === 4. KERNEL (Numba) ===
        src_kernel = dataframe['close'].values.astype(np.float64)
        k_lookback = int(self.kernel_lookback.value)
        k_weight = float(self.kernel_weighting.value)
        k_reg = int(self.kernel_regression.value)
        
        kernel_est = numba_rational_quadratic_kernel(src_kernel, k_lookback, k_weight, k_reg)
            
        dataframe['kernel_estimate'] = np.where(np.isnan(kernel_est), src_kernel, kernel_est)
        # Pine: isBullishRate = yhat1[1] < yhat1 (kernel subiendo)
        # Pine: isBearishRate = yhat1[1] > yhat1 (kernel bajando)
        dataframe['kernel_bullish'] = dataframe['kernel_estimate'] > dataframe['kernel_estimate'].shift(1)
        dataframe['kernel_bearish'] = dataframe['kernel_estimate'] < dataframe['kernel_estimate'].shift(1)

        # === 5. ML Prediction (Lorentzian) ===
        self.calculate_ml_prediction(dataframe)
        
        # === 6. Pullback Filter ===
        if self.use_kernel_pullback_filter.value:
            dataframe['kernel_pullback_ok'] = self.calculate_kernel_pullback_filter(dataframe)
        else:
            dataframe['kernel_pullback_ok'] = True

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0

        if len(dataframe) < self.max_bars_back.value:
            return dataframe

        f_vol = dataframe['volatility_ok'] if self.use_volatility_filter.value else True
        f_reg = dataframe['regime_ok'] if self.use_regime_filter.value else True
        f_adx = dataframe['adx_ok'] if self.use_adx_filter.value else True
        f_pull = dataframe['kernel_pullback_ok']

        f_ema_long = dataframe['ema_uptrend'] if self.use_ema_filter.value else True
        f_sma_long = dataframe['sma_uptrend'] if self.use_sma_filter.value else True
        f_kernel_long = dataframe['kernel_bullish'] if self.use_kernel_filter.value else True
        
        # === LOGICA MEJORADA DE ENTRADA ===
        min_score = self.buy_min_score.value

        # Debugging Counts
        c_ml = (dataframe['ml_prediction'] > min_score)
        c_vol = c_ml & f_vol
        c_reg = c_vol & f_reg
        c_adx = c_reg & f_adx
        c_pull = c_adx & f_pull
        c_ema = c_pull & f_ema_long & f_sma_long # Added SMA
        c_kernel = c_ema & f_kernel_long
        
        logger.info(f"DEBUG: Total Bars: {len(dataframe)}")
        logger.info(f"DEBUG: ML > {min_score}: {c_ml.sum()}")
        logger.info(f"DEBUG: + Volatility: {c_vol.sum()}")
        logger.info(f"DEBUG: + Regime: {c_reg.sum()}")
        logger.info(f"DEBUG: + ADX: {c_adx.sum()}")
        logger.info(f"DEBUG: + Pullback: {c_pull.sum()}")
        logger.info(f"DEBUG: + EMA/SMA: {c_ema.sum()}")
        logger.info(f"DEBUG: + Kernel: {c_kernel.sum()} (Final Long)")

        long_cond = c_kernel
        
        f_ema_short = dataframe['ema_downtrend'] if self.use_ema_filter.value else True
        f_sma_short = dataframe['sma_downtrend'] if self.use_sma_filter.value else True
        f_kernel_short = dataframe['kernel_bearish'] if self.use_kernel_filter.value else True
        
        # Debugging Counts Short
        # Pine: prediction < 0 (any negative prediction)
        c_ml_s = (dataframe['ml_prediction'] < 0)  # EXACT MATCHING (was: < -min_score)
        c_vol_s = c_ml_s & f_vol
        c_reg_s = c_vol_s & f_reg
        c_adx_s = c_reg_s & f_adx
        c_pull_s = c_adx_s & f_pull
        c_ema_s = c_pull_s & f_ema_short & f_sma_short # Added SMA
        c_kernel_s = c_ema_s & f_kernel_short

        logger.info(f"DEBUG: ML < {-min_score}: {c_ml_s.sum()}")
        logger.info(f"DEBUG: + Volatility: {c_vol_s.sum()}")
        logger.info(f"DEBUG: + Regime: {c_reg_s.sum()}")
        logger.info(f"DEBUG: + ADX: {c_adx_s.sum()}")
        logger.info(f"DEBUG: + Pullback: {c_pull_s.sum()}")
        logger.info(f"DEBUG: + EMA/SMA: {c_ema_s.sum()}")
        logger.info(f"DEBUG: + Kernel: {c_kernel_s.sum()} (Final Short)")

        short_cond = c_kernel_s
        
        dataframe.loc[long_cond, 'enter_long'] = 1
        if self.can_short:
            dataframe.loc[short_cond, 'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[:, 'exit_long'] = 0
        dataframe.loc[:, 'exit_short'] = 0
        
        mode = self.exit_mode.value
        
        # --- 1. Signal Exit (ML Flip) ---
        ml_flip_long = (dataframe['ml_prediction'] < 0) & (dataframe['ml_prediction'].shift(1) > 0)
        ml_flip_short = (dataframe['ml_prediction'] > 0) & (dataframe['ml_prediction'].shift(1) < 0)
        
        # --- 2. Dynamic Exit (Kernel Color Change) ---
        kernel = dataframe['kernel_estimate']
        kernel_prev = kernel.shift(1)
        kernel_prev2 = kernel.shift(2)
        
        is_bearish_rate = kernel_prev > kernel
        is_bullish_rate = kernel_prev < kernel
        
        was_bearish_rate = kernel_prev2 > kernel_prev
        was_bullish_rate = kernel_prev2 < kernel_prev
        
        is_bearish_change = is_bearish_rate & was_bullish_rate
        is_bullish_change = is_bullish_rate & was_bearish_rate
        
        dyn_exit_long = is_bearish_change
        dyn_exit_short = is_bullish_change
        
        if mode in ['signal', 'all']:
            dataframe.loc[ml_flip_long, 'exit_long'] = 1
            dataframe.loc[ml_flip_short, 'exit_short'] = 1
            
        if mode in ['dynamic', 'all']:
            dataframe.loc[dyn_exit_long, 'exit_long'] = 1
            dataframe.loc[dyn_exit_short, 'exit_short'] = 1
            
        return dataframe

    def custom_exit(self, pair: str, trade: 'Trade', current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs):
        
        mode = self.exit_mode.value
        
        if mode == 'fixed_bars':
            time_diff = (current_time - trade.open_date_utc)
            tf_minutes = 60 
            if self.timeframe.endswith('m'):
                tf_minutes = int(self.timeframe[:-1])
            elif self.timeframe.endswith('h'):
                tf_minutes = int(self.timeframe[:-1]) * 60
                
            candles_passed = time_diff.total_seconds() / 60 / tf_minutes
            
            if candles_passed >= self.fixed_exit_bars.value:
                logger.info(f"{pair}: Fixed Exit triggered after {candles_passed:.1f} candles")
                return "fixed_bars_exit"
 
        if self.use_break_even.value:
            max_profit = trade.calc_profit_ratio(trade.max_rate)
            if max_profit >= self.break_even_profit_threshold.value:
                if current_profit < self.break_even_offset.value:
                    logger.info(f"{pair}: Break Even Exit triggered. Max profit: {max_profit:.2%}, Current: {current_profit:.2%}")
                    return "break_even_exit"
                    
        return None

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                            time_in_force: str, current_time: datetime, entry_tag: Optional[str],
                            side: str, **kwargs) -> bool:
        
        logger.info(f"{pair}: Entering {side} trade. Rate: {rate}, Tag: {entry_tag}")
        return True

    # =========================================================================
    # HELPERS
    # =========================================================================

    def calculate_ml_prediction(self, dataframe: DataFrame):
        feats = ['rsi_norm_1', 'wt_norm', 'cci_norm', 'adx_norm', 'rsi_norm_5']
        
        features_data = dataframe[feats].fillna(0.5).values.astype(np.float64)
        
        y_train = self.create_training_labels(dataframe)
        
        preds = numba_lorentzian_prediction(
            features_data, 
            y_train, 
            int(self.max_bars_back.value), 
            int(self.neighbors_count.value)
        )
        dataframe['ml_prediction'] = preds

    def create_training_labels(self, df):
        future = df['close'].shift(-4)
        conds = [
            (future < df['close']), 
            (future > df['close'])  
        ]
        return np.select(conds, [-1.0, 1.0], default=0.0).astype(np.float64)

    def calculate_kernel_pullback_filter(self, df) -> pd.Series:
        kernel = df['kernel_estimate']
        close = df['close']
        tol = self.kernel_pullback_tolerance.value
        
        dist_pct = abs(close - kernel) / kernel
        is_near = dist_pct <= tol
        
        return is_near.rolling(window=3, min_periods=1).max() > 0