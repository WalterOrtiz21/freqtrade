# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: disable=F401
# isort: skip_file
# --- Do not remove these libs ---
import numpy as np
import pandas as pd
from pandas import DataFrame
from datetime import datetime, timedelta
from typing import Optional, Union

from freqtrade.strategy import (IStrategy, IntParameter, DecimalParameter, BooleanParameter, CategoricalParameter, stoploss_from_absolute)
from freqtrade.persistence import Trade
import talib.abstract as ta
import logging

# Exit Manager - Intelligent Exit System
try:
    from ExitManager import ExitManager, ExitAction, create_exit_manager
    HAS_EXIT_MANAGER = True
except ImportError:
    HAS_EXIT_MANAGER = False
    ExitManager = None

# =============================================================================
# DEPENDENCIAS
# =============================================================================
try:
    from numba import njit
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

@njit(fastmath=True)
def numba_lorentzian_distance_prediction(features_norm, labels, max_bars_back, neighbors_count, prediction_horizon=4):
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

    for i in range(start_calculation, n_rows):
        current_features = features_norm[i]
        
        last_distance = -1.0
        distances_buffer = np.zeros(neighbors_count * 10) 
        predictions_buffer = np.zeros(neighbors_count * 10)
        buffer_size = 0
        
        start_index = i - max_bars_back
        if start_index < 0: start_index = 0
        
        # FIX LOOKAHEAD BIAS:
        # Labels are defined as shift(-4) (future).
        # So label[k] contains information about price at k+4.
        # If we are at index i, we cannot know label[i-1] because that requires price at i+3 (future).
        # We must stop searching for neighbors at i - prediction_horizon.
        end_index = i - prediction_horizon
        
        if end_index < start_index:
             predictions[i] = 0.0
             continue 
        
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
    
    # Cache para stoploss calculado (evita recalculos innecesarios)
    # Formato: {trade_id: (fixed_stop_price, is_breakeven_active)}
    _sl_cache = {}
    
    # Exit Manager instance (initialized in bot_start if enabled)
    _exit_manager = None
    
    # Exit Manager Parameters
    use_exit_manager = BooleanParameter(default=True, space='sell', optimize=False)
    exit_manager_model_path = 'user_data/strategies/models/exit_manager.pkl'

    # ================= PARÁMETROS (Coinciden con JSON) =================
    
    # ML Settings
    source_type = CategoricalParameter(['close', 'hlc3', 'ohlc4', 'hl2', 'open', 'high', 'low'], default='close', space='buy', optimize=False)
    neighbors_count = IntParameter(2, 20, default=8, space='buy', optimize=False)
    source_type = CategoricalParameter(['close', 'hlc3', 'ohlc4', 'hl2', 'open', 'high', 'low'], default='close', space='buy', optimize=False)
    max_bars_back = IntParameter(500, 3000, default=1000, space='buy', optimize=False)


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
    use_kernel_filter = BooleanParameter(default=True, space='buy', optimize=True)
    use_kernel_smoothing = BooleanParameter(default=False, space='buy', optimize=True)
    kernel_h = IntParameter(3, 20, default=8, space='buy', optimize=True)  # Doc: 3-50, using 3-20
    kernel_r = DecimalParameter(2.0, 15.0, default=8.0, decimals=1, space='buy', optimize=True)  # Doc: 0.25-25
    kernel_x = IntParameter(5, 30, default=25, space='buy', optimize=True)  # Doc: 2-25, using 5-30
    kernel_lag = IntParameter(1, 3, default=2, space='buy', optimize=True)  # Doc: 1-2

    # Filters (Sistema 1: ML Signal Filters - aplicados a isNewBuySignal/isNewSellSignal)
    use_volatility_filter = BooleanParameter(default=True, space='buy', optimize=True)
    use_regime_filter = BooleanParameter(default=True, space='buy', optimize=True)
    use_adx_filter = BooleanParameter(default=False, space='buy', optimize=True)
    regime_threshold = DecimalParameter(-0.3, 0.3, default=-0.1, decimals=2, space='buy', optimize=True)
    adx_threshold = IntParameter(15, 30, default=20, space='buy', optimize=True)  # Doc: 15-25

    # Sistema 1: EMA/SMA Filters para señales ML (Pine: useEmaFilter, useSmaFilter)
    use_ml_ema_filter = BooleanParameter(default=False, space='buy', optimize=True)  # Pine default: false
    use_ml_sma_filter = BooleanParameter(default=False, space='buy', optimize=True)  # Pine default: false
    ml_ema_period = IntParameter(50, 200, default=200, space='buy', optimize=True)  # Doc: 20-100 for EMA
    ml_sma_period = IntParameter(100, 300, default=200, space='buy', optimize=True)  # Doc: 50-200 for SMA

    # Sistema 2: EMA Filter para posiciones (Pine: ema_filter_long, ema_filter_short, bullish/bearish)
    use_ema_filter_long = BooleanParameter(default=True, space='buy', optimize=True)   # Pine default: true
    use_ema_filter_short = BooleanParameter(default=True, space='buy', optimize=True)  # Pine default: true
    position_ema_period = IntParameter(20, 400, default=200, space='buy', optimize=True)

    # Exits
    use_dynamic_exits = BooleanParameter(default=False, space='sell', optimize=True)
    close_with_supertrend = BooleanParameter(default=True, space='sell', optimize=True)


    # Protection & Risk
    stoploss_type = CategoricalParameter(['atr', 'swing'], default='atr', space='protection', optimize=True)
    atr_stop_len = IntParameter(10, 30, default=14, space='protection', optimize=True)
    atr_stop_mult = DecimalParameter(1.0, 5.0, default=1.5, decimals=1, space='protection', optimize=True)
    swing_bars = IntParameter(5, 20, default=10, space='protection', optimize=True)
    
    use_breakeven = BooleanParameter(default=True, space='protection', optimize=True)
    be_rr_long = DecimalParameter(0.5, 3.0, default=1.0, decimals=1, space='protection', optimize=True)
    be_rr_short = DecimalParameter(0.5, 3.0, default=1.0, decimals=1, space='protection', optimize=True)
    
    use_takeprofit = BooleanParameter(default=True, space='protection', optimize=True)
    tp_rr_long = DecimalParameter(1.0, 5.0, default=3.0, decimals=1, space='protection', optimize=True)
    tp_rr_short = DecimalParameter(1.0, 5.0, default=3.0, decimals=1, space='protection', optimize=True)
    tp_percent = DecimalParameter(10, 100, default=50, decimals=0, space='protection', optimize=True)
    close_only_tp = BooleanParameter(default=False, space='protection', optimize=True)

    st_atr_period = IntParameter(5, 20, default=9, space='buy', optimize=True)
    st_factor = DecimalParameter(1.0, 5.0, default=2.5, decimals=1, space='buy', optimize=True)

    # ==========================================================================
    # HTF TIMEFRAMES (SuperTrend Filter)
    # ==========================================================================
    HTF_OPTIONS = ['none', '5m', '15m', '30m', '1h', '2h', '4h', '8h', '12h', '1d']
    
    htf_1 = CategoricalParameter(HTF_OPTIONS, default='1h', space='buy', optimize=False)
    htf_2 = CategoricalParameter(HTF_OPTIONS, default='4h', space='buy', optimize=False)
    use_htf_filter = BooleanParameter(default=True, space='buy', optimize=True)

    # Configuración Freqtrade
    timeframe = '4h'
    minimal_roi = { "0": 100 } # Desactivamos ROI estandard, usamos Custom
    stoploss = -0.99           # Desactivamos SL fijo estandard, usamos Custom ATR
    
    # Con normalización fija (constantes históricas), solo necesitamos velas
    # para los indicadores técnicos (EMA200, ATR, etc.)
    # Con normalización fija (constantes históricas), solo necesitamos velas
    # para los indicadores técnicos (EMA200, ATR, etc.)
    startup_candle_count = 1000 
    
    # REGLA PINE SCRIPT CRUCIAL:
    # El script original dice: if strategy.openprofit > 0 -> close.
    # Esto asegura que no salgamos en pérdidas por cambios de señal o supertrend,
    # solo por Stop Loss.
    exit_profit_only = True 
    
    can_short = True
    use_custom_stoploss = True
    position_adjustment_enable = True 
    process_only_new_candles = True # Similar a process_orders_on_close de Pine

    # Fecha desde la cual calcular min/max históricos para normalización
    # Esto replica el comportamiento de Pine Script que usa todo el historial del chart
    HISTORIC_START_DATE = "2020-01-01"
    
    # Cache para min/max calculados (se calculan una sola vez)
    _historic_calculated = False
    _historic_wt_min = -53.0   # Valores por defecto
    _historic_wt_max = 47.0
    _historic_cci_min = -550.0
    _historic_cci_max = 550.0

    # Logging Control
    analysis_logging = BooleanParameter(default=False, space='protection', optimize=False)

    # ---------------- FUNCIONES DE NORMALIZACION -----------------
    
    def _calculate_historic_minmax(self, pair: str):
        """
        Calcula min/max históricos desde HISTORIC_START_DATE.
        Carga datos directamente del archivo feather.
        Se ejecuta una sola vez al inicio.
        """
        if self._historic_calculated:
            return
            
        try:
            from pathlib import Path
            
            # Construir ruta al archivo de datos
            # Construir ruta al archivo de datos
            # Formato: user_data/data/{exchange}/futures/{PAIR}-{timeframe}-futures.feather
            exchange_name = self.config.get('exchange', {}).get('name', 'binance')
            pair_filename = pair.replace("/", "_").replace(":", "_")
            data_path = Path(f"user_data/data/{exchange_name}/futures") / f"{pair_filename}-{self.timeframe}-futures.feather"
            
            if not data_path.exists():
                logger.warning(f"Archivo de datos no encontrado: {data_path}")
                logger.info(f"   Usando valores por defecto para min/max")
                self._historic_calculated = True
                return
            
            # Cargar datos desde archivo
            df = pd.read_feather(data_path)
            
            # Filtrar desde la fecha de inicio
            df = df[df['date'] >= self.HISTORIC_START_DATE].copy()
            
            if len(df) < 100:
                logger.warning(f"Pocos datos históricos desde {self.HISTORIC_START_DATE}: {len(df)} filas")
                self._historic_calculated = True
                return
            
            # Calcular WaveTrend
            ap = (df['high'] + df['low'] + df['close']) / 3
            esa = ta.EMA(ap, timeperiod=self.f2_ch_len.value)
            d = ta.EMA(np.abs(ap - esa), timeperiod=self.f2_ch_len.value)
            d = np.where(d == 0, 0.0001, d)
            ci = (ap - esa) / (0.015 * d)
            tci = ta.EMA(ci, timeperiod=self.f2_avg_len.value)
            wt1 = tci
            wt2 = ta.SMA(wt1, timeperiod=4)
            wt_diff = wt1 - wt2
            
            self._historic_wt_min = float(np.nanmin(wt_diff))
            self._historic_wt_max = float(np.nanmax(wt_diff))
            
            # Calcular CCI
            cci_raw = ta.CCI(df, timeperiod=self.f3_period.value)
            if self.f3_smoothing.value > 1:
                cci_smooth = ta.EMA(cci_raw, timeperiod=self.f3_smoothing.value)
            else:
                cci_smooth = cci_raw
            
            self._historic_cci_min = float(np.nanmin(cci_smooth))
            self._historic_cci_max = float(np.nanmax(cci_smooth))
            
            if self.analysis_logging.value:
                logger.info(f"📊 Min/Max históricos calculados desde {self.HISTORIC_START_DATE} ({len(df)} filas):")
                logger.info(f"   WaveTrend: [{self._historic_wt_min:.2f}, {self._historic_wt_max:.2f}]")
                logger.info(f"   CCI: [{self._historic_cci_min:.2f}, {self._historic_cci_max:.2f}]")
            
        except Exception as e:
            logger.error(f"Error calculando min/max históricos: {e}")
        
        self._historic_calculated = True
    
    def _get_active_htf_list(self) -> list:
        """Build list of active HTF timeframes from parameters."""
        htf_list = []
        if self.htf_1.value != 'none':
            htf_list.append(self.htf_1.value)
        if self.htf_2.value != 'none':
            htf_list.append(self.htf_2.value)
        return htf_list

    def informative_pairs(self):
        """
        Define pairs to load based on htf_1 and htf_2 parameters.
        """
        pairs = self.dp.current_whitelist()
        htf_list = self._get_active_htf_list()
        
        informative_pairs = []
        for tf in htf_list:
            informative_pairs += [(pair, tf) for pair in pairs]
        
        return informative_pairs

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
        
        # Calcular min/max históricos (solo la primera vez)
        self._calculate_historic_minmax(metadata['pair'])
        
        # --- SELECCIÓN DE SOURCE ---
        st = self.source_type.value
        
        if st == 'close':
            dataframe['source'] = dataframe['close']
        elif st == 'hlc3':
            dataframe['source'] = (dataframe['high'] + dataframe['low'] + dataframe['close']) / 3
        elif st == 'ohlc4':
            dataframe['source'] = (dataframe['open'] + dataframe['high'] + dataframe['low'] + dataframe['close']) / 4
        elif st == 'hl2':
            dataframe['source'] = (dataframe['high'] + dataframe['low']) / 2
        elif st == 'open':
            dataframe['source'] = dataframe['open']
        elif st == 'high':
            dataframe['source'] = dataframe['high']
        elif st == 'low':
            dataframe['source'] = dataframe['low']
        else:
            dataframe['source'] = dataframe['close']

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
        # Usamos min/max históricos calculados dinámicamente
        dataframe['f2_norm'] = self._normalize_fixed(wt_diff, self._historic_wt_min, self._historic_wt_max) 
        
        # --- FEATURE 3: CCI ---
        cci_raw = ta.CCI(dataframe, timeperiod=self.f3_period.value)
        cci_smooth = ta.EMA(cci_raw, timeperiod=self.f3_smoothing.value) if self.f3_smoothing.value > 1 else cci_raw
        # Usamos min/max históricos calculados dinámicamente
        dataframe['f3_norm'] = self._normalize_fixed(cci_smooth, self._historic_cci_min, self._historic_cci_max)

        # --- FEATURE 4: ADX ---
        adx_raw = ta.ADX(dataframe, timeperiod=self.f4_period.value)
        adx_smooth = ta.EMA(adx_raw, timeperiod=self.f4_smoothing.value) if self.f4_smoothing.value > 1 else adx_raw
        dataframe['adx_f4'] = adx_smooth
        dataframe['f4_norm'] = self._normalize_fixed(adx_smooth, 0, 100)

        # --- HTF SUPERTREND CALCULATION ---
        if self.dp:
            htf_list = self._get_active_htf_list()
            for htf in htf_list:
                try:
                    inf_htf = self.dp.get_pair_dataframe(metadata['pair'], htf)
                    if inf_htf.empty:
                        logger.warning(f"No data for {metadata['pair']} {htf}")
                        continue
                    
                    # Calculate SuperTrend on HTF
                    inf_htf['atr_st'] = ta.ATR(inf_htf, timeperiod=self.st_atr_period.value)
                    hl2_htf = (inf_htf['high'] + inf_htf['low']) / 2
                    matr_htf = float(self.st_factor.value) * inf_htf['atr_st']
                    upperband_htf = hl2_htf + matr_htf
                    lowerband_htf = hl2_htf - matr_htf
                    
                    st_htf, direction_htf = _numba_supertrend_recalc(
                        inf_htf['close'].values.astype(np.float64),
                        upperband_htf.values.astype(np.float64),
                        lowerband_htf.values.astype(np.float64)
                    )
                    
                    # Store trend direction
                    inf_htf[f'{htf}_st_direction'] = direction_htf
                    
                    # Prepare for merge
                    inf_htf = inf_htf[['date', f'{htf}_st_direction']].copy()
                    
                    # Merge
                    dataframe = pd.merge(dataframe, inf_htf, on='date', how='left')
                    dataframe[f'{htf}_st_direction'] = dataframe[f'{htf}_st_direction'].ffill()
                    
                except Exception as e:
                    logger.error(f"Error processing HTF {htf}: {e}")

        # --- FILTROS ---
        atr_1 = ta.ATR(dataframe, timeperiod=1)
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
        # Usamos 'source' seleccionado por el usuario
        src_kernel = dataframe['source'].values.astype(np.float64)
        dataframe['yhat1'] = numba_rational_quadratic_kernel(
            src_kernel, 
            self.kernel_h.value, 
            float(self.kernel_r.value), 
            self.kernel_x.value
        )
        # kernel Gaussian para yhat2 con lag
        h_lag = max(1, self.kernel_h.value - self.kernel_lag.value)
        dataframe['yhat2'] = numba_gaussian_kernel(src_kernel, h_lag, self.kernel_x.value)
        
        is_bullish_smooth = dataframe['yhat2'] >= dataframe['yhat1']
        is_bearish_smooth = dataframe['yhat2'] <= dataframe['yhat1']
        is_bullish_rate = dataframe['yhat1'] > dataframe['yhat1'].shift(1)
        is_bearish_rate = dataframe['yhat1'] < dataframe['yhat1'].shift(1)

        if self.use_kernel_filter.value:
            if self.use_kernel_smoothing.value:
                dataframe['is_bullish_kernel'] = is_bullish_smooth
                dataframe['is_bearish_kernel'] = is_bearish_smooth
            else:
                dataframe['is_bullish_kernel'] = is_bullish_rate
                dataframe['is_bearish_kernel'] = is_bearish_rate
        else:
            # Pine: isBullish = useKernelFilter ? ... : true
            dataframe['is_bullish_kernel'] = np.ones(len(dataframe), dtype=bool)
            dataframe['is_bearish_kernel'] = np.ones(len(dataframe), dtype=bool)

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
            # CRITICAL: Usar numpy array explícito, no escalar True
            # El escalar True puede causar problemas con operaciones & en pandas
            dataframe['is_ml_ema_uptrend'] = np.ones(len(dataframe), dtype=bool)
            dataframe['is_ml_ema_downtrend'] = np.ones(len(dataframe), dtype=bool)

        if self.use_ml_sma_filter.value:
            ml_sma = ta.SMA(dataframe, timeperiod=self.ml_sma_period.value)
            dataframe['is_ml_sma_uptrend'] = dataframe['close'] > ml_sma
            dataframe['is_ml_sma_downtrend'] = dataframe['close'] < ml_sma
        else:
            dataframe['is_ml_sma_uptrend'] = np.ones(len(dataframe), dtype=bool)
            dataframe['is_ml_sma_downtrend'] = np.ones(len(dataframe), dtype=bool)

        # Sistema 2: EMA para posiciones (Pine: bullish/bearish, ema200)
        # Pine: bullish = close > ema200 (SIEMPRE calculado)
        position_ema = ta.EMA(dataframe, timeperiod=self.position_ema_period.value)
        dataframe['bullish'] = dataframe['close'] > position_ema
        dataframe['bearish'] = dataframe['close'] < position_ema

        # --- ML LORENTZIAN PREDICTION ---
        # CRITICAL CORRECTION: Pine uses src[4] which is 4 bars in the PAST (history referencing)
        # Pine Strategy (line 233): y_train_series = src[4] < src[0] ? direction_s.short : src[4] > src[0] ? direction_s.long : direction_s.neutral
        # This compares Price 4 bars ago vs Current Price.
        # If Price(t-4) < Price(t): Price ROSE. Label = Short (-1). (Contrarian/Mean Reversion)
        # If Price(t-4) > Price(t): Price FELL. Label = Long (1).
        
        # Usamos 'source' seleccionado por el usuario
        past_source = dataframe['source'].shift(4)  # 4 bars in the PAST
        labels = np.where(past_source < dataframe['source'], -1.0,   # Pasado < Actual → Subió → Short = -1
                  np.where(past_source > dataframe['source'], 1.0,   # Pasado > Actual → Bajó → Long = 1
                           0.0))
        labels = np.nan_to_num(labels, nan=0.0)
        
        # Preparar features
        feature_cols = ['f1_norm', 'f2_norm', 'f3_norm', 'f4_norm', 'f5_norm']
        features_data = dataframe[feature_cols].fillna(0.5).values.astype(np.float64)
        
        predictions = numba_lorentzian_distance_prediction(
            features_data,
            labels.astype(np.float64),
            int(self.max_bars_back.value),
            int(self.neighbors_count.value),
            -1 # prediction_horizon = -1 to include SELF (i) in neighbors, matching Pine behavior
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
        # Pine: isNewBuySignal = isBuySignal and isDifferentSignalType
        # Pine: isDifferentSignalType = ta.change(signal)
        
        # CRITICAL FIX: Strictly match Pine Script logic.
        # Previously, we were checking if the *filtered* signal changed, which meant that if the 
        # EMA filter became valid later, it would trigger a trade (re-entry).
        # Pine Script DOES NOT do this. It only trades if the signal changes AND the filter is valid AT THAT MOMENT.
        
        # 1. Detect signal changes (Pine: isDifferentSignalType)
        dataframe['signal_changed'] = dataframe['signal'] != dataframe['signal'].shift(1)

        # 2. Define New Buy/Sell Signals (Pine: isNewBuySignal)
        # isNewBuySignal = (Signal is Buy) AND (Signal Changed) AND (Filters are Valid)
        
        dataframe['is_new_buy_signal'] = (
            (dataframe['signal'] == 1) & 
            dataframe['signal_changed'] &
            dataframe['is_ml_ema_uptrend'] &
            dataframe['is_ml_sma_uptrend']
        )
        
        dataframe['is_new_sell_signal'] = (
            (dataframe['signal'] == -1) & 
            dataframe['signal_changed'] &
            dataframe['is_ml_ema_downtrend'] &
            dataframe['is_ml_sma_downtrend']
        )

        # Pine (línea 333): startLongTrade = isNewBuySignal and isBullish and isEmaUptrend and isSmaUptrend
        # isBullish = kernel filter, isEmaUptrend/isSmaUptrend ya incluidos en isNewBuySignal
        start_long_trade = (
            dataframe['is_new_buy_signal'] &
            dataframe['is_bullish_kernel']    # Pine: isBullish
        )

        start_short_trade = (
            dataframe['is_new_sell_signal'] &
            dataframe['is_bearish_kernel']    # Pine: isBearish
        )

        # HTF Filter
        if self.use_htf_filter.value:
            htf_list = self._get_active_htf_list()
            for htf in htf_list:
                col = f'{htf}_st_direction'
                if col in dataframe.columns:
                    # 1 = Bullish, -1 = Bearish
                    start_long_trade &= (dataframe[col] == 1)
                    start_short_trade &= (dataframe[col] == -1)

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
        
        # Calcular duración del timeframe en minutos
        tf_min = int(self.timeframe[:-1]) * (60 if 'h' in self.timeframe else 1)
        
        # Asegurar que open_date sea compatible (pd.Timestamp UTC)
        # En backtesting y dry-run, freqtrade usa datetimes con UTC
        open_date = pd.to_datetime(open_date, utc=True)
            
        # Lógica de redondeo robusta usando pd.Timestamp
        # Floor al inicio del timeframe
        tf_delta = pd.Timedelta(minutes=tf_min)
        candle_date = open_date.floor(tf_delta)
        
        # La vela de SEÑAL suele ser la anterior a la vela actual (entry candle)
        signal_date = candle_date - tf_delta
        
        try:
            # Intentar match exacto con signal_date
            row = dataframe.loc[dataframe['date'] == signal_date]
            if not row.empty:
                return row.iloc[0]
            
            # Fallback: Intentar con candle_date
            row = dataframe.loc[dataframe['date'] == candle_date]
            if not row.empty:
                return row.iloc[0]
                
            # Fallback 2: Buscar la última vela disponible anterior o igual a signal_date
            # Usando asof (más eficiente y robusto)
            # Necesitamos que dataframe esté indexado por fecha o usar searchsorted
            # Pero 'date' es una columna normal aquí. Usamos boolean masking seguro.
            
            possible_candles = dataframe[dataframe['date'] <= signal_date]
            if not possible_candles.empty:
                return possible_candles.iloc[-1]
                
            return None
        except Exception as e:
            # Evitar spam de logs en backtesting si faltan datos al inicio
            # logger.error(f"Error getting entry candle for {pair} at {open_date}: {e}")
            return None

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Custom stoploss con cache para evitar recalculos innecesarios.
        """
        
        # Obtener datos de la vela de entrada
        entry_data = self._get_entry_candle(pair, trade.open_date_utc)
        if entry_data is None: 
            # logger.warning(f"Stoploss: No entry candle found for {pair}")
            return self.stoploss 
        
        entry_price = trade.open_rate
        
        # Calcular distancia del SL (solo necesitamos esto para el cálculo inicial y breakeven)
        sl_dist = 0.0
        atr_val = entry_data.get('atr_sl', 0)
        swing_val = 0
        
        if self.stoploss_type.value == 'atr':
            sl_dist = atr_val * float(self.atr_stop_mult.value)
        else:
            if trade.is_short:
                swing_val = entry_data.get('swing_high', 0)
                sl_dist = abs(swing_val - entry_price)
            else:
                swing_val = entry_data.get('swing_low', 0)
                sl_dist = abs(entry_price - swing_val)

        if sl_dist <= 0: 
            return self.stoploss
        
        # Verificar si ya tenemos cache para este trade
        if trade.id in self._sl_cache:
            cached_stop_price, is_breakeven_active = self._sl_cache[trade.id]
            
            # Si el breakeven ya está activo, usar el precio cacheado (entry_price)
            if is_breakeven_active:
                return stoploss_from_absolute(cached_stop_price, current_rate, 
                                              is_short=trade.is_short, leverage=trade.leverage)
            
            # Si el breakeven NO está activo, verificar si ahora califica para activarlo
            if self.use_breakeven.value:
                if trade.is_short:
                    be_dist = sl_dist * float(self.be_rr_short.value)
                    be_trigger = entry_price - be_dist
                    if trade.min_rate <= be_trigger:
                        # ¡Activar breakeven! Actualizar cache con entry_price
                        self._sl_cache[trade.id] = (entry_price, True)
                        
                        # Log breakeven activation (una sola vez)
                        if trade.id not in self._breakeven_logged:
                            self._breakeven_logged.add(trade.id)
                            profit_pct = ((entry_price - current_rate) / entry_price) * 100
                            be_date = current_time.strftime('%Y-%m-%d %H:%M:%S')
                            entry_date = trade.open_date_utc.strftime('%Y-%m-%d %H:%M:%S')
                            if self.analysis_logging.value:
                                trade_logger.info(f"BE_ACTIVATED,SHORT,{pair},{current_rate:.8f},{profit_pct:.2f},R:R={self.be_rr_short.value},Date={be_date},EntryDate={entry_date}")
                                logger.info(f"⚖️ Breakeven SHORT activated: {pair} @ {current_rate:.8f} | R:R={self.be_rr_short.value} | Date: {be_date}")
                        
                        return stoploss_from_absolute(entry_price, current_rate, 
                                                      is_short=trade.is_short, leverage=trade.leverage)
                else:
                    be_dist = sl_dist * float(self.be_rr_long.value)
                    be_trigger = entry_price + be_dist
                    if trade.max_rate >= be_trigger:
                        # ¡Activar breakeven! Actualizar cache con entry_price
                        self._sl_cache[trade.id] = (entry_price, True)
                        
                        # Log breakeven activation (una sola vez)
                        if trade.id not in self._breakeven_logged:
                            self._breakeven_logged.add(trade.id)
                            profit_pct = ((current_rate - entry_price) / entry_price) * 100
                            be_date = current_time.strftime('%Y-%m-%d %H:%M:%S')
                            entry_date = trade.open_date_utc.strftime('%Y-%m-%d %H:%M:%S')
                            if self.analysis_logging.value:
                                trade_logger.info(f"BE_ACTIVATED,LONG,{pair},{current_rate:.8f},{profit_pct:.2f},R:R={self.be_rr_long.value},Date={be_date},EntryDate={entry_date}")
                                logger.info(f"⚖️ Breakeven LONG activated: {pair} @ {current_rate:.8f} | R:R={self.be_rr_long.value} | Date: {be_date}")
                        
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
            
        # Redondear para evitar actualizaciones innecesarias por precision flotante
        fixed_stop_price = round(fixed_stop_price, 8)
        
        # Cachear el SL inicial (sin breakeven activo)
        self._sl_cache[trade.id] = (fixed_stop_price, False)
        
        if self.analysis_logging.value:
            sl_pct = (sl_dist / entry_price) * 100
            logger.info(f"📍 SL Calculated for {pair} (Trade #{trade.id}): "
                       f"{'SHORT' if trade.is_short else 'LONG'} | "
                       f"Entry: {entry_price:.8f} | SL Price: {fixed_stop_price:.8f} | SL Dist: {sl_dist:.8f} ({sl_pct:.2f}%) | Current: {current_rate:.8f}")
        
        # Retornar SL inicial
        return stoploss_from_absolute(fixed_stop_price, current_rate, 
                                      is_short=trade.is_short, leverage=trade.leverage)

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
        
        # === EXIT MANAGER PARTIAL EXIT ===
        if self.use_exit_manager.value and HAS_EXIT_MANAGER:
            # Initialize ExitManager if not already done
            if self._exit_manager is None:
                self._exit_manager = ExitManager(
                    model_path=self.exit_manager_model_path,
                    config={
                        'partial_exit_profit': 0.03,
                        'partial_exit_amount': 0.50,
                        'tighten_sl_drawdown': 0.02,
                    }
                )
            
            dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
            
            # Get recommendation from ExitManager
            recommendation = self._exit_manager.predict_action(
                trade=trade,
                current_rate=current_rate,
                dataframe=dataframe,
                current_profit=current_profit
            )
            
            # Act on PARTIAL_EXIT recommendation (before standard TP logic)
            if recommendation.action == ExitAction.PARTIAL_EXIT and trade.nr_of_successful_exits == 0:
                if recommendation.confidence > 0.6:
                    partial_amount = recommendation.details.get('amount', 0.50)
                    exit_stake = -(trade.stake_amount * partial_amount)
                    
                    if self.analysis_logging.value:
                        logger.info(f"🤖 ExitManager PARTIAL_EXIT: {trade.pair} ({partial_amount*100:.0f}%, conf={recommendation.confidence:.2f})")
                    return exit_stake
        
        # === ORIGINAL TP LOGIC ===
        if self.use_takeprofit.value and trade.nr_of_successful_exits == 0:
            entry_data = self._get_entry_candle(trade.pair, trade.open_date_utc)
            if entry_data is None: 
                # Debug logging para entender por qué falla
                if self.analysis_logging.value:
                     logger.warning(f"TP Check Failed: No entry candle found for {trade.pair} (Open: {trade.open_date_utc})")
                return None
            
            sl_dist = 0.0
            if self.stoploss_type.value == 'atr':
                sl_dist = entry_data.get('atr_sl', 0) * float(self.atr_stop_mult.value)
            else:
                if trade.is_short: sl_dist = abs(entry_data.get('swing_high', 0) - trade.open_rate)
                else: sl_dist = abs(trade.open_rate - entry_data.get('swing_low', 0))
            
            if sl_dist == 0: 
                return None

            if trade.is_short:
                tp_price = trade.open_rate - (sl_dist * float(self.tp_rr_short.value))
                
                # Debug ocasional (cada ~5% de cambio o algo asi? No, muy complejo. Loguear si está cerca)
                # if self.analysis_logging.value and abs(current_rate - tp_price) / tp_price < 0.01:
                #    logger.info(f"TP Check SHORT {trade.pair}: Price {current_rate:.8f} vs Target {tp_price:.8f} (Dist: {sl_dist:.8f})")

                if current_rate <= tp_price:
                    # Log take profit (una sola vez)
                    if trade.id not in self._tp_logged:
                        self._tp_logged.add(trade.id)
                        profit_pct = current_profit * 100
                        profit_abs = trade.calc_profit(current_rate) * (self.tp_percent.value / 100)
                        tp_date = current_time.strftime('%Y-%m-%d %H:%M:%S')
                        entry_date = trade.open_date_utc.strftime('%Y-%m-%d %H:%M:%S')
                        if self.analysis_logging.value:
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
                        if self.analysis_logging.value:
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
        # Log formato CSV: Type,Direction,Pair,EntryPrice,Amount,EntryDate,SL_Price,SL_Pct,BE_Price,TP_Price,Tag
        if self.analysis_logging.value:
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
        if self.analysis_logging.value:
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
        Intelligent exit with ExitManager + original signal logic.
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        
        # === EXIT MANAGER CONSULTATION ===
        if self.use_exit_manager.value and HAS_EXIT_MANAGER:
            # Initialize ExitManager if not already done
            if self._exit_manager is None:
                self._exit_manager = ExitManager(
                    model_path=self.exit_manager_model_path,
                    config={
                        'partial_exit_profit': 0.03,
                        'partial_exit_amount': 0.50,
                        'tighten_sl_drawdown': 0.02,
                    }
                )
            
            # Get recommendation from ExitManager
            recommendation = self._exit_manager.predict_action(
                trade=trade,
                current_rate=current_rate,
                dataframe=dataframe,
                current_profit=current_profit
            )
            
            # Act on FULL_EXIT recommendation
            if recommendation.action == ExitAction.FULL_EXIT and recommendation.confidence > 0.6:
                if self.analysis_logging.value:
                    logger.info(f"🤖 ExitManager FULL_EXIT: {pair} (conf={recommendation.confidence:.2f}, reason={recommendation.details.get('reason', 'ml')})")
                return f"exit_manager_{recommendation.details.get('reason', 'ml')}"

        exit_reason = None
        direction = "SHORT" if trade.is_short else "LONG"
        entry_date = trade.open_date_utc.strftime('%Y-%m-%d %H:%M:%S')
        entry_price = trade.open_rate

        # NOTA: El log de SL real se maneja en confirm_trade_exit cuando el trade
        # realmente se cierra por trailing_stop_loss. No logueamos aquí para evitar
        # mensajes confusos cuando el precio se acerca al SL pero luego se recupera.

        # Detectar razón de salida normal (señal/supertrend)
        # 1. Signal Flips (Dynamic Exits)
        if self.use_dynamic_exits.value:
            if last_candle.get('is_new_sell_signal', False) and trade.is_short == False:
                exit_reason = "SIGNAL_FLIP_SHORT"
            elif last_candle.get('is_new_buy_signal', False) and trade.is_short == True:
                exit_reason = "SIGNAL_FLIP_LONG"
        
        # 2. SuperTrend Cross (Close with SuperTrend)
        if self.close_with_supertrend.value and not exit_reason:
            if last_candle.get('close_cross_below_st', False) and trade.is_short == False:
                exit_reason = "SUPERTREND_CROSS_BELOW"
            elif last_candle.get('close_cross_above_st', False) and trade.is_short == True:
                exit_reason = "SUPERTREND_CROSS_ABOVE"

        # Si hay razón de exit y tenemos profit, loguear y salir
        if exit_reason and current_profit > 0 and not self.close_only_tp.value:
            profit_pct = current_profit * 100
            profit_abs = trade.calc_profit(current_rate)

            # Log formato CSV: Type,Direction,Reason,Pair,ExitPrice,Profit%,ProfitAbs,EntryDate,EntryPrice
            if self.analysis_logging.value:
                trade_logger.info(f"EXIT,{direction},{exit_reason},{pair},{current_rate:.8f},{profit_pct:.2f},{profit_abs:.8f},EntryDate={entry_date},EntryPrice={entry_price:.8f}")
                logger.info(f"🔴 {direction} Exit ({exit_reason}): {pair} @ {current_rate:.8f} | "
                           f"Profit: {profit_pct:.2f}% ({profit_abs:.8f}) | Entry: {entry_date} @ {entry_price:.8f}")

            return exit_reason

        return None
