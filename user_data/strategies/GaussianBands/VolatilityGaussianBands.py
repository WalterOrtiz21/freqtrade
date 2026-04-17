import logging
import numpy as np
import pandas as pd
from datetime import datetime
from pandas import DataFrame
from typing import Optional

from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter, BooleanParameter, CategoricalParameter, stoploss_from_absolute, merge_informative_pair
from freqtrade.persistence import Trade
import talib.abstract as ta
from numba import njit
import os
import pickle

logger = logging.getLogger(__name__)

class VolatilityGaussianBands(IStrategy):
    """
    Volatility Gaussian Bands [BigBeluga]
    Ported to Freqtrade.
    
    Includes Retest options and Partial Take Profit (TP1) & Break Even (BE)
    logic based on absolute percentage movement (ignoring leverage).
    """

    INTERFACE_VERSION = 3

    # ==========================================================================
    # GAUSSIAN BANDS PARAMETERS
    # ==========================================================================
    len_gaussian = IntParameter(5, 50, default=20, space='buy', optimize=True)
    distance = DecimalParameter(0.1, 3.0, default=1.0, decimals=1, space='buy', optimize=True)
    
    # ==========================================================================
    # SIGNAL PARAMETERS & FILTERS
    # ==========================================================================
    enable_crossover_entry = BooleanParameter(default=True, space='buy', optimize=True)
    enable_retest_entry = BooleanParameter(default=True, space='buy', optimize=True)
    
    # HTF Filter
    use_htf_filter = BooleanParameter(default=True, space='buy', optimize=True)
    htf_timeframe = CategoricalParameter(['none', '1h', '4h', '1d'], default='4h', space='buy', optimize=False)
    
    # ML Filter (Requires trained models from GaussianBands)
    use_ml_filter = BooleanParameter(default=True, space='buy', optimize=False)
    use_per_symbol_models = BooleanParameter(default=True, space='buy', optimize=False)
    ml_threshold = DecimalParameter(0.05, 0.50, default=0.15, decimals=2, space='buy', optimize=True)
    ml_model_path = "user_data/strategies/GaussianBands/models"
    _ml_model = None
    
    # ==========================================================================
    # EXIT & RISK MANAGEMENT (TP1 & Break Even)
    # ==========================================================================
    # Take Profit 1 (Partial Exit)
    tp1_enabled = BooleanParameter(default=True, space='sell', optimize=True)
    tp1_pct = DecimalParameter(0.005, 0.05, default=0.01, decimals=3, space='sell', optimize=True) # 1% Target
    tp1_amount = DecimalParameter(0.1, 1.0, default=0.5, decimals=1, space='sell', optimize=True)  # Sell 50%
    
    # Break Even Settings
    move_be_at_tp1 = BooleanParameter(default=True, space='sell', optimize=True)
    be_trigger_pct = DecimalParameter(0.0, 0.05, default=0.0, decimals=3, space='sell', optimize=True) 
    be_offset_pct = DecimalParameter(0.001, 0.01, default=0.002, decimals=3, space='sell', optimize=True)
    
    # Base Stoploss (Ajustable dinámicamente)
    use_custom_stoploss = True
    stoploss = -0.05 
    position_adjustment_enable = True
    
    # General Freqtrade configs
    timeframe = '1h'
    startup_candle_count: int = 150
    can_short = True

    def bot_start(self, **kwargs) -> None:
        """Called automatically on bot startup"""
        self.config_leverage = self.config.get('leverage', 1.0)
        logger.info(f"Initialized VolatilityGaussianBands with {self.config_leverage}x leverage. Custom exits adjust for this.")
        
        # Preload ML general model if available to save time
        self._load_ml_model()

    def informative_pairs(self):
        """Define pairs to load for HTF filtering."""
        pairs = self.dp.current_whitelist()
        informative_pairs = []
        if self.htf_timeframe.value != 'none':
            informative_pairs += [(pair, self.htf_timeframe.value) for pair in pairs]
        return informative_pairs

    # ==========================================================================
    # ML INTEGRATION
    # ==========================================================================
    def _load_ml_model(self):
        if not hasattr(self, '_models_cache'):
            self._models_cache = {}
            
        general_model_file = os.path.join(self.ml_model_path, "gaussian_xgboost_model.pkl")
        if os.path.exists(general_model_file) and 'general' not in self._models_cache:
            if self.use_ml_filter.value:
                try:
                    with open(general_model_file, 'rb') as f:
                        self._models_cache['general'] = pickle.load(f)
                    logger.info(f"✅ General ML Model loaded from {general_model_file}")
                except Exception as e:
                    logger.error(f"❌ Failed to load general ML model: {e}")
                    self._models_cache['general'] = None
        if 'general' in self._models_cache:
            self._ml_model = self._models_cache['general']
        else:
            self._ml_model = None
            
    def _get_ml_model_for_pair(self, pair: str):
        if not self.use_ml_filter.value:
            return None
        if not hasattr(self, '_models_cache'):
            self._models_cache = {}
        pair_key = pair.replace('/', '_').replace(':', '_').split('_USDT')[0] + '_USDT'
        
        if pair_key in self._models_cache:
            return self._models_cache[pair_key]
        
        if self.use_per_symbol_models.value:
            symbol_model_file = os.path.join(self.ml_model_path, f"gaussian_xgboost_{pair_key}.pkl")
            if os.path.exists(symbol_model_file):
                try:
                    with open(symbol_model_file, 'rb') as f:
                        model = pickle.load(f)
                    self._models_cache[pair_key] = model
                    return model
                except Exception:
                    pass
        
        if 'general' not in self._models_cache:
            general_model_file = os.path.join(self.ml_model_path, "gaussian_xgboost_model.pkl")
            if os.path.exists(general_model_file):
                try:
                    with open(general_model_file, 'rb') as f:
                        self._models_cache['general'] = pickle.load(f)
                except Exception:
                    self._models_cache['general'] = None
            else:
                self._models_cache['general'] = None
        
        self._models_cache[pair_key] = self._models_cache.get('general')
        return self._models_cache[pair_key]

    # ==========================================================================
    # VOLATILITY GAUSSIAN CALCULATION (NUMBA OPTIMIZED)
    # ==========================================================================
    def calculate_gaussian_multi_trend(self, df: DataFrame, period: int) -> pd.Series:
        """
        Replicates the 'multi_trend' loop from PineScript using Numba for maximum speed.
        """
        close_array = df['close'].values
        result = _calculate_gaussian_multi_trend_numba(close_array, period)
        return pd.Series(result, index=df.index)

    # ==========================================================================
    # INDICATORS & SIGNALS
    # ==========================================================================
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe.copy()
        period = self.len_gaussian.value
        
        # 1. Base Gaussian Trend Average
        df['gaussian_avg'] = self.calculate_gaussian_multi_trend(df, period)
        
        # 2. Volatility Range (ATR-like approach from Pine: SMA of High-Low)
        df['volatility_hl'] = (df['high'] - df['low']).rolling(window=100).mean()
        
        # 3. Dynamic Bands
        dist = self.distance.value
        df['lower_band'] = df['gaussian_avg'] - (df['volatility_hl'] * dist)
        df['upper_band'] = df['gaussian_avg'] + (df['volatility_hl'] * dist)
        
        # 4. Determine Global Trend (1 = Bullish, -1 = Bearish)
        # Shift 1 in comparison to ensure 'crossover/under' logic (started below -> crosses above)
        df['cross_up'] = (df['close'] > df['upper_band']) & (df['close'].shift(1) <= df['upper_band'].shift(1))
        df['cross_dn'] = (df['close'] < df['lower_band']) & (df['close'].shift(1) >= df['lower_band'].shift(1))
        
        df['trend_signal'] = 0
        df.loc[df['cross_up'], 'trend_signal'] = 1
        df.loc[df['cross_dn'], 'trend_signal'] = -1
        # Forward fill the state
        df['trend'] = df['trend_signal'].replace(0, np.nan).ffill().fillna(0)
        
        # 5. ENTRY SIGNALS
        # A. Crossover Entry (Entering just when the trend turns)
        df['signal_crossover_long'] = (df['trend'] == 1) & (df['trend'].shift(1) != 1)
        df['signal_crossover_short'] = (df['trend'] == -1) & (df['trend'].shift(1) != -1)
        
        # B. Retest Entry (Entering on pullback towards gaussian_avg)
        # Alcista: In uptrend, price tests the median (falls below), but closes back above
        df['close_cross_up_avg'] = (df['close'] > df['gaussian_avg']) & (df['close'].shift(1) <= df['gaussian_avg'].shift(1))
        df['signal_retest_long'] = (df['trend'] == 1) & df['close_cross_up_avg']
        
        # Bajista: In downtrend, price tests the median (rises above), but closes back below
        df['close_cross_dn_avg'] = (df['close'] < df['gaussian_avg']) & (df['close'].shift(1) >= df['gaussian_avg'].shift(1))
        df['signal_retest_short'] = (df['trend'] == -1) & df['close_cross_dn_avg']
        
        # 6. HTF FILTERS (Information extraction)
        if self.dp and self.htf_timeframe.value != 'none':
            htf = self.htf_timeframe.value
            inf_htf = self.dp.get_pair_dataframe(metadata['pair'], htf)
            if not inf_htf.empty:
                # Calculate HTF Gaussian
                inf_htf['htf_gaussian_avg'] = self.calculate_gaussian_multi_trend(inf_htf, period)
                inf_htf['htf_volatility_hl'] = (inf_htf['high'] - inf_htf['low']).rolling(window=100).mean()
                inf_htf['htf_lower_band'] = inf_htf['htf_gaussian_avg'] - (inf_htf['htf_volatility_hl'] * dist)
                inf_htf['htf_upper_band'] = inf_htf['htf_gaussian_avg'] + (inf_htf['htf_volatility_hl'] * dist)
                
                inf_htf['htf_cross_up'] = (inf_htf['close'] > inf_htf['htf_upper_band']) & (inf_htf['close'].shift(1) <= inf_htf['htf_upper_band'].shift(1))
                inf_htf['htf_cross_dn'] = (inf_htf['close'] < inf_htf['htf_lower_band']) & (inf_htf['close'].shift(1) >= inf_htf['htf_lower_band'].shift(1))
                
                inf_htf['htf_trend_signal'] = 0
                inf_htf.loc[inf_htf['htf_cross_up'], 'htf_trend_signal'] = 1
                inf_htf.loc[inf_htf['htf_cross_dn'], 'htf_trend_signal'] = -1
                inf_htf['htf_trend'] = inf_htf['htf_trend_signal'].replace(0, np.nan).ffill().fillna(0)
                
                # Merge with proper shift — merge_informative_pair shifts HTF dates
                # forward by 1 HTF period so each base candle only sees CLOSED HTF candles.
                # pd.merge on 'date' was lookahead: 1H candle at 00:00 was getting the
                # 4H trend from the candle that opens at 00:00 but closes at 04:00.
                inf_htf_merge = inf_htf[['date', 'htf_trend']].copy()
                df = merge_informative_pair(df, inf_htf_merge, self.timeframe, htf, ffill=True)
                # merge_informative_pair appends the HTF suffix; rename back for downstream code
                df['htf_trend'] = df[f'htf_trend_{htf}'].fillna(0)
                df.drop(columns=[f'htf_trend_{htf}', f'date_{htf}'], errors='ignore', inplace=True)
            else:
                df['htf_trend'] = 0
        else:
            df['htf_trend'] = 0

        # 7. ML / CONTEXT FEATURES (Native for Gaussian Mean Reversion)
        # Squeeze normalized by close price
        df['ml_vol_squeeze'] = (df['upper_band'] - df['lower_band']) / df['close']
        # Distance to gaussian average
        df['ml_dist_to_avg'] = (df['close'] - df['gaussian_avg']) / df['gaussian_avg']
        # Position within the bands (0 = lower, 1 = upper, >1 above upper)
        band_range = (df['upper_band'] - df['lower_band']).replace(0, np.nan)
        df['ml_close_pos'] = ((df['close'] - df['lower_band']) / band_range).fillna(0.5)
        
        # Trend HTF Alignment (1, 0, -1)
        df['ml_htf_trend'] = df['htf_trend']
        
        df['rsi'] = ta.RSI(df['close'], timeperiod=14)
        df['ml_rsi'] = df['rsi'] / 100.0
        
        df['adx'] = ta.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        df['ml_adx'] = df['adx'] / 100.0
        
        df['atr'] = ta.ATR(df, timeperiod=14)
        df['ml_atr_pct'] = df['atr'] / df['close']
        
        # Native ML features strictly built for XGBoost Gaussian training
        expected_ml_features = [
            'ml_vol_squeeze', 'ml_dist_to_avg', 'ml_close_pos', 'ml_htf_trend',
            'ml_rsi', 'ml_adx', 'ml_atr_pct'
        ]
        
        for col in expected_ml_features:
            if col not in df.columns:
                df[col] = 0.0 # Null fallback for indicators missing in this barebone script
                
        # Fill NaN
        feature_cols = [c for c in df.columns if c.startswith('ml_')]
        df[feature_cols] = df[feature_cols].fillna(0)

        # Execute ML Inference
        df['ml_predict_prob'] = 0.5
        if self.use_ml_filter.value:
            model = self._get_ml_model_for_pair(metadata['pair'])
            if model:
                try:
                    df['ml_predict_prob'] = model.predict_proba(df[expected_ml_features])[:, 1]
                except Exception as e:
                    logger.warning(f"ML Inference failed for {metadata['pair']}: {e}")
        
        return df

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe.copy()
        
        # Initialize
        df['enter_long'] = False
        df['enter_short'] = False
        
        # 1. Base Signals
        enter_long_cond = pd.Series(False, index=df.index)
        enter_short_cond = pd.Series(False, index=df.index)
        
        if self.enable_crossover_entry.value:
            enter_long_cond |= df['signal_crossover_long']
            enter_short_cond |= df['signal_crossover_short']
            
        if self.enable_retest_entry.value:
            enter_long_cond |= df['signal_retest_long']
            enter_short_cond |= df['signal_retest_short']
            
        # 2. Apply HTF Filter (Trend Alignment)
        if self.use_htf_filter.value and self.htf_timeframe.value != 'none':
            enter_long_cond &= (df['htf_trend'] == 1)
            enter_short_cond &= (df['htf_trend'] == -1)
            
        # 3. Apply ML Probability Filter
        if self.use_ml_filter.value:
            enter_long_cond &= (df['ml_predict_prob'] > self.ml_threshold.value)
            # ML Model was trained on Longs (TP1 vs SL). For shorts we block if Long is highly probable
            enter_short_cond &= (df['ml_predict_prob'] < (1.0 - self.ml_threshold.value))
            
        # Set final signals
        df.loc[enter_long_cond, 'enter_long'] = True
        df.loc[enter_short_cond, 'enter_short'] = True
            
        return df

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Emergency exit when the trend completely reverses across the bands
        dataframe['exit_long'] = (dataframe['trend'] == -1) & (dataframe['trend'].shift(1) == 1)
        dataframe['exit_short'] = (dataframe['trend'] == 1) & (dataframe['trend'].shift(1) == -1)
        return dataframe

    # ==========================================================================
    # EXIT LOGIC (TAKE PROFIT 1 & BREAK EVEN) - Leverage Agnostic
    # ==========================================================================
    def adjust_trade_position(self, trade: 'Trade', current_time: datetime,
                              current_rate: float, current_profit: float,
                              min_stake: Optional[float], max_stake: float,
                              current_entry_rate: float, current_exit_rate: float,
                              current_entry_profit: float, current_exit_profit: float,
                              **kwargs) -> Optional[float]:
        """
        Calculates Take Profit 1 (Sell % Amount of Position at % Target).
        """
        if not self.tp1_enabled.value:
            return None

        # Ya realizamos TP1 parcial?
        if trade.nr_of_successful_exits > 0:
            return None
            
        # Calcular ROI sin considerar el apalancamiento (Movimiento base de precio)
        if hasattr(trade, 'open_rate'):
            price_moved_pct = (current_rate - trade.open_rate) / trade.open_rate if not trade.is_short else (trade.open_rate - current_rate) / trade.open_rate
        else:
            price_moved_pct = current_profit / self.config_leverage
        
        # Si llegamos al Target de TP1:
        if price_moved_pct >= self.tp1_pct.value:
            # Retornar cantidad negativa para vender una fracción del stake
            return -(trade.stake_amount * self.tp1_amount.value)

        return None

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Manages Break-Even dynamically independently of Freqtrade's default configs.
        """
        if hasattr(trade, 'open_rate'):
            price_moved_pct = (current_rate - trade.open_rate) / trade.open_rate if not trade.is_short else (trade.open_rate - current_rate) / trade.open_rate
        else:
            price_moved_pct = current_profit / self.config_leverage
            
        # Determinar el Trigger (%) para deslizar el BE. Si es 0, usamos TP1.
        trigger = self.be_trigger_pct.value if self.be_trigger_pct.value > 0 else self.tp1_pct.value
        
        # Mover a Break Even si el beneficio alcanzó el target o el trade ya ejecutó el TP1
        tp1_hit = trade.nr_of_successful_exits > 0
        
        if self.move_be_at_tp1.value and (price_moved_pct >= trigger or tp1_hit):
            # Calcular stop al punto de entrada + offset de comisiones
            if not trade.is_short:
                limit = trade.open_rate * (1 + self.be_offset_pct.value)
            else:
                limit = trade.open_rate * (1 - self.be_offset_pct.value)
                
            return stoploss_from_absolute(limit, current_rate, is_short=trade.is_short)
            
        return self.stoploss

# ==============================================================================
# NUMBA OPTIMIZED FUNCTIONS (OUTSIDE CLASS)
# ==============================================================================
@njit
def _get_gaussian_weights_numba(length: int, sigma: float) -> np.ndarray:
    pi = np.pi
    weights = np.zeros(length)
    for i in range(length):
        weight = np.exp(-0.5 * (((i - length / 2) / sigma) ** 2)) / np.sqrt(sigma * 2.0 * pi)
        weights[i] = weight
    return weights / np.sum(weights)

@njit
def _calculate_gaussian_multi_trend_numba(close_array: np.ndarray, period: int) -> np.ndarray:
    n = len(close_array)
    steps = 21
    
    # Pre-calculate all weights for all steps
    # Max length is period + 20
    # Store all filtered results to average them later
    all_filters = np.zeros((steps, n))
    
    for step in range(steps):
        length = period + step
        w = _get_gaussian_weights_numba(length, 10.0)
        
        filtered = np.full(n, np.nan)
        
        # Apply the filter only where we have enough data (length)
        for i in range(length - 1, n):
            # Sum over the window
            sum_val = 0.0
            # w[0] corresponds to newest data close_array[i]
            # w[1] corresponds to close_array[i-1]
            for j in range(length):
                sum_val += w[j] * close_array[i - j]
            filtered[i] = sum_val
            
        all_filters[step] = filtered
        
    # Average across all 21 steps for each time point
    result = np.full(n, np.nan)
    for i in range(n):
        sum_val = 0.0
        count = 0
        for step in range(steps):
            val = all_filters[step, i]
            if not np.isnan(val):
                sum_val += val
                count += 1
        if count > 0:
            result[i] = sum_val / count
            
    return result
