"""
SMCWithMLLuxAlgo Strategy
=========================

SMC Strategy aligned with LuxAlgo TradingView implementation.
"Smart Money Concepts [LuxAlgo]"

Key Features:
- Dual structure: Internal (reactive) + Swing (stable)
- CHoCH for reversals (Entry Signal)
- Real-time detection (close crosses level)
- Trend tracking for context
- TP and BE logic matching Pine Script execution

Entry Logic (Faithful to Pine):
- Long: CHoCH bullish (Internal OR Swing, depending on config)
- Short: CHoCH bearish (Internal OR Swing, depending on config)
- NO BOS entries by default (Pine only uses CHoCH).
- NO Order Block / FVG requirement by default (Pine entry section does not check these).

Exit Logic:
- Opposite CHoCH (trend reversal)
- OR Take Profit 1 (Partial)
- OR Break Even (after TP1)
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pandas import DataFrame
from typing import Optional, Dict, List
import pickle
import os

logger = logging.getLogger(__name__)

from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter, BooleanParameter, CategoricalParameter, stoploss_from_absolute
from freqtrade.persistence import Trade
import talib.abstract as ta

# Import LuxAlgo-style SMC library
# Module copied to user_data/strategies/ for Hyperopt compatibility
try:
    from smc_luxalgo_numba import SMCLuxAlgoNumba as SMCLuxAlgo
except ImportError:
    # Fallback: try local import
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from smc_luxalgo_numba import SMCLuxAlgoNumba as SMCLuxAlgo

# Import Training Logic
try:
    from user_data.strategies.SMC.train_smc_model import train_model
except ImportError:
    # Fallback if path parsing fails (e.g. running from different root)
    logger.warning("Could not import train_model. Auto-training disabled.")
    train_model = None


class SMCWithMLLuxAlgo(IStrategy):
    """
    SMC Strategy (LuxAlgo Aligned)
    
    Verified against 'Smart_Money_Concepts_[LuxAlgo]_Strategy.pine'
    """

    INTERFACE_VERSION = 3

    # ==========================================================================
    # FIXED PARAMETERS (adjusted dynamically by leverage in bot_start)
    # ==========================================================================
    
    # These are BASE values for 1x leverage - will be multiplied by leverage
    minimal_roi = {"0": 1.0}  # Disabled, using custom exits
    stoploss = -0.03  # Base: -3% price movement (adjusted by leverage in bot_start)
    timeframe = '15m'
    
    trailing_stop = False
    trailing_stop_positive = 0.005  # 0.5% price movement
    trailing_stop_positive_offset = 0.01  # 1% price movement
    trailing_only_offset_is_reached = True
    
    
    use_exit_signal = True
    use_custom_stoploss = True  # Enable for breakeven
    position_adjustment_enable = True  # Enable for partial TPs
    max_open_trades = 3
    startup_candle_count: int = 250
    can_short = True
    
    # ==========================================================================
    # HTF TIMEFRAMES (Flexible selection - works with JSON buy params)
    # ==========================================================================
    # Set to 'none' to disable that slot
    # Common options: '5m', '15m', '30m', '1h', '2h', '4h', '8h', '12h', '1d'
    HTF_OPTIONS = ['none', '5m', '15m', '30m', '1h', '2h', '4h', '8h', '12h', '1d']
    
    htf_1 = CategoricalParameter(HTF_OPTIONS, default='1h', space='buy', optimize=False)
    htf_2 = CategoricalParameter(HTF_OPTIONS, default='4h', space='buy', optimize=False)

    # ==========================================================================
    # SMC PARAMETERS (Optimizable)
    # ==========================================================================
    
    # Structure Mode: "Internal" or "Swing" (Pine 'stratStructureType')
    # Pine Default: "Swing".
    # To match Pine Default: set use_swing_signals=True, use_internal_signals=False.
    
    # Internal structure length (reactive)
    internal_length = IntParameter(3, 10, default=5, space='buy', optimize=True)
    
    # Swing structure length (stable)
    swing_length = IntParameter(20, 100, default=50, space='buy', optimize=True)
    
    # Use internal or swing signals for entry
    # Note: Pine Script uses a dropdown XOR selection. Here we allow both via flags.
    # To match Pine exactly: enable ONE, disable the other.
    use_internal_signals = BooleanParameter(default=False, space='buy', optimize=True)
    use_swing_signals = BooleanParameter(default=True, space='buy', optimize=True)
    
    # Only trade CHoCH (reversals) or also BOS (continuations)?
    # Pine Strategy ONLY trades CHoCH for entries. Default to True to match.
    require_choch = BooleanParameter(default=True, space='buy', optimize=True)
    
    # Zone requirements - disabled by default to match Pine Strategy execution logic
    # Zone Filters
    require_ob_zone = BooleanParameter(default=True, space='buy', optimize=True)
    require_fvg_zone = BooleanParameter(default=True, space='buy', optimize=True)
    
    # NEW: Volumetric Order Blocks (BigBeluga Style)
    require_volumetric_ob = BooleanParameter(default=True, space='buy', optimize=True)
    vol_sma_period = IntParameter(10, 50, default=20, space='buy', optimize=True)
    ob_rvol_threshold = DecimalParameter(1.2, 3.5, default=1.8, decimals=1, space='buy', optimize=True)
    
    # Trend filter - disabled by default to match Pine Strategy execution logic
    # (Pine indicator shows trend color, but strategy entry block does not enforce it)
    trade_with_trend = BooleanParameter(default=False, space='buy', optimize=True)
    
    # Lookback for recent signals
    # 1 = Instant Entry (Signal + Zone on same candle)
    # > 1 = Retest Logic (Signal happened X bars ago, entering now on Zone or Pullback)
    entry_signal_lookback = IntParameter(1, 24, default=1, space='buy', optimize=True)
    
    # NEW: Toggle for HTF (4h) Filter
    use_htf_filter = BooleanParameter(default=True, space='buy', optimize=True)
    
    # Per-HTF: Use internal_trend (faster, reacts to CHoCH) vs swing_trend (slower, more reliable)
    htf_1_use_internal = BooleanParameter(default=False, space='buy', optimize=True)
    htf_2_use_internal = BooleanParameter(default=True, space='buy', optimize=True)
    
    # NEW: Dynamic Stoploss Toggle & Offset
    use_dynamic_stoploss = BooleanParameter(default=True, space='sell', optimize=True)
    sl_buffer_pct = DecimalParameter(0.001, 0.01, default=0.002, decimals=3, space='sell', optimize=True)
    
    # NEW: FVG ATR Filter (BigBeluga Style)
    # Filter out based on volatility (ATR) rather than percentage.
    # Threshold = ATR(200) * fvg_atr_threshold.
    fvg_atr_threshold = DecimalParameter(0.05, 0.50, default=0.1, decimals=2, space='buy', optimize=True)
    
    # NEW: Conservative Entry Mode
    # If False, ignores signals on the current candle (0 bars ago), forcing a wait.
    allow_immediate_entry = BooleanParameter(default=True, space='buy', optimize=True)

    # Liquidity Sweep: If detected + CHoCH, bypass zone requirement
    use_sweep_bypass = BooleanParameter(default=True, space='buy', optimize=True)
    sweep_lookback = IntParameter(1, 10, default=3, space='buy', optimize=True)

    # Logging Control
    enable_logging = BooleanParameter(default=True, space='custom', optimize=False)
    
    # Circuit Breaker (Panic Mode) - Blocks trades during high volatility
    circuit_breaker_enabled = BooleanParameter(default=True, space='custom', optimize=False)
    circuit_breaker_window = IntParameter(15, 120, default=30, space='custom', optimize=False)  # Minutes
    circuit_breaker_limit = IntParameter(2, 10, default=3, space='custom', optimize=False)  # Max SL hits

    # ==========================================================================
    # ML PARAMETERS (Placeholder)
    # ==========================================================================
    # Note: ML logic is currently disabled in entry generation to ensure 
    # strict fidelity to Pine Script logic.
    
    use_ml_filter = BooleanParameter(default=False, space='buy', optimize=False)
    use_per_symbol_models = BooleanParameter(default=False, space='buy', optimize=False)  # True = per-symbol, False = general
    ml_threshold = DecimalParameter(0.05, 0.50, default=0.15, decimals=2, space='buy', optimize=True)
    ml_model_path = "user_data/strategies/SMC/models"
    enable_auto_training = BooleanParameter(default=False, space='buy', optimize=False)
    _ml_model = None

    # ==========================================================================
    # EXIT PARAMETERS (Pine Strategy Alignment)
    # ==========================================================================
    
    # TP and BE logic aligned with Pine 'STRAT_GROUP':
    # "Usar Take Profit 1" (stratUseTP)
    # "Mover a Break Even al tocar TP1" (stratMoveBE)
    
    # Pine Default: stratUseTP=False, stratMoveBE=False.
    # We set defaults logic, but users likely want these features enabled in Freqtrade.
    
    # TP1: % movement of price (not leveraged PnL)
    # Pine Default: 1.0%. Adjusted Range to allow "high value" (e.g. 0.50 = 50% move).
    tp1_pct = DecimalParameter(0.005, 0.50, default=0.01, decimals=3, space='sell', optimize=True)
    
    # TP1 Enabled: Bool to turn on/off the partial exit
    tp1_enabled = BooleanParameter(default=True, space='sell', optimize=True)
    
    # TP1 Amount: % of position to close
    # Pine Default: 50%. Allow 0.0 to disable partials.
    tp1_amount = DecimalParameter(0.0, 100.0, default=50.0, space='sell', optimize=True)
    
    # Break Even (Pine 'stratMoveBE')
    # Pine Default: False
    move_be_at_tp1 = BooleanParameter(default=False, space='sell', optimize=True)
    
    # Custom Break Even Target (Optional)
    # If > 0, BE is activated cuando el precio se mueve este %, ignorando el tp1_pct.
    be_trigger_pct = DecimalParameter(0.0, 0.50, default=0.0, decimals=3, space='sell', optimize=True)
    
    # Final Exit (Reversal)
    # Granular control over which CHoCH triggers an exit
    exit_on_internal_choch = BooleanParameter(default=True, space='sell', optimize=True)
    exit_on_swing_choch = BooleanParameter(default=True, space='sell', optimize=True)

    # ==========================================================================
    # INITIALIZATION
    # ==========================================================================
    
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        # htf_timeframes is now built dynamically from boolean params
        # No need to read from config anymore
    
    def bot_start(self, **kwargs) -> None:
        """
        Strategy startup.
        Applies leverage to risk parameters defined in JSON.
        """
        print(f"🤖 DEBUG: Bot Start. use_ml_filter: {self.use_ml_filter.value}, auto_train: {self.enable_auto_training.value}")
        logger.info(f"🤖 Bot Start. use_ml_filter: {self.use_ml_filter.value}, auto_train: {self.enable_auto_training.value}")
        
        # 0. Auto-Train Model (Only if explicitely enabled)
        if self.use_ml_filter.value and self.enable_auto_training.value and train_model:
            print("🚀 DEBUG: Starting Auto-Training...")
            logger.info("🚀 Starting Auto-Training of ML Model (Hyperopt enabled)...")
            success = train_model()
            if success:
                logger.info("✅ Auto-Training Complete.")
                print("✅ DEBUG: Auto-Training Complete.")
            else:
                logger.error("❌ Auto-Training Failed. Using existing model if available.")
                print("❌ DEBUG: Auto-Training Failed.")
        elif self.use_ml_filter.value and not self.enable_auto_training.value:
            logger.info("ℹ️ Auto-Training disabled. Loading existing model.")
            print("ℹ️ DEBUG: Auto-Training disabled.")

        # Load ML model here (after params are loaded)
        self._load_ml_model()
        
        # 1. Get Leverage
        config_leverage = self.config.get('leverage', 1.0)
        
        # 2. Get Raw Values from Config (assumed to be Price Distance)
        raw_stoploss = self.config.get('stoploss', -0.05)
        raw_trailing_pos = self.config.get('trailing_stop_positive', 0.005)
        raw_trailing_offset = self.config.get('trailing_stop_positive_offset', 0.01)
        
        # 3. Apply Leverage to Calculate PnL thresholds
        self.stoploss = raw_stoploss * config_leverage
        
        # Adjust trailing settings if they are set in config
        if self.config.get('trailing_stop', False):
            self.trailing_stop_positive = raw_trailing_pos * config_leverage
            self.trailing_stop_positive_offset = raw_trailing_offset * config_leverage
        
        
        logger.info(
            f"SMCWithMLLuxAlgo (Neptune) Configured:"
            f"\n  Leverage: {config_leverage}x"
            f"\n  Base Stoploss (Price): {raw_stoploss:.2%}"
            f"\n  Effective Stoploss (PnL): {self.stoploss:.2%}"
            f"\n  Structure: MTF (15m) + HTF (4h)"
        )

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
        
        logger.info(f"Informative pairs configured for timeframes: {htf_list}")
        return informative_pairs

    
    def _load_ml_model(self):
        """Load pre-trained ML model if exists.
        
        Tries to load in order:
        1. Per-symbol model: smc_xgboost_{PAIR}.pkl
        2. General model: smc_xgboost_model.pkl
        """
        if not hasattr(self, '_models_cache'):
            self._models_cache = {}
            
        # General model (fallback) - load once
        general_model_file = os.path.join(self.ml_model_path, "smc_xgboost_model.pkl")
        if os.path.exists(general_model_file) and 'general' not in self._models_cache:
            if self.use_ml_filter.value:
                try:
                    with open(general_model_file, 'rb') as f:
                        self._models_cache['general'] = pickle.load(f)
                    logger.info(f"✅ General ML Model loaded from {general_model_file}")
                except Exception as e:
                    logger.error(f"❌ Failed to load general ML model: {e}")
                    self._models_cache['general'] = None
        
        # Set default model to general if exists
        if 'general' in self._models_cache:
            self._ml_model = self._models_cache['general']
        else:
            self._ml_model = None
            
    def _get_ml_model_for_pair(self, pair: str):
        """Get ML model for specific pair (with caching and fallback).
        
        If use_per_symbol_models=True: Try per-symbol model, fallback to general
        If use_per_symbol_models=False: Only use general model
        """
        if not self.use_ml_filter.value:
            return None
            
        if not hasattr(self, '_models_cache'):
            self._models_cache = {}
        
        # Convert pair to filename: 'BTC/USDT:USDT' -> 'BTC_USDT'
        pair_key = pair.replace('/', '_').replace(':', '_').split('_USDT')[0] + '_USDT'
        
        # Check cache first
        if pair_key in self._models_cache:
            return self._models_cache[pair_key]
        
        # If per-symbol models enabled, try to load specific model
        if self.use_per_symbol_models.value:
            symbol_model_file = os.path.join(self.ml_model_path, f"smc_xgboost_{pair_key}.pkl")
            if os.path.exists(symbol_model_file):
                try:
                    with open(symbol_model_file, 'rb') as f:
                        model = pickle.load(f)
                    self._models_cache[pair_key] = model
                    logger.info(f"✅ Per-symbol ML Model loaded for {pair}: {symbol_model_file}")
                    return model
                except Exception as e:
                    logger.warning(f"Failed to load per-symbol model for {pair}: {e}")
        
        # Fallback to general model (or use it directly if per_symbol disabled)
        if 'general' not in self._models_cache:
            general_model_file = os.path.join(self.ml_model_path, "smc_xgboost_model.pkl")
            if os.path.exists(general_model_file):
                try:
                    with open(general_model_file, 'rb') as f:
                        self._models_cache['general'] = pickle.load(f)
                    logger.info(f"✅ General ML Model loaded: {general_model_file}")
                except Exception:
                    self._models_cache['general'] = None
            else:
                self._models_cache['general'] = None
        
        # Cache that this pair uses general model
        self._models_cache[pair_key] = self._models_cache.get('general')
        return self._models_cache[pair_key]

    # ==========================================================================
    # INDICATOR CALCULATION
    # ==========================================================================
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Calculate SMC indicators using LuxAlgo-style library."""
        
        # --- 1. MTF (15m) Calculations ---
        # Calculate SMC signals (Numba)
        # Returns signals AND active zones
        smc = SMCLuxAlgo(
            dataframe,
            internal_length=self.internal_length.value,
            swing_length=self.swing_length.value,
        )
        signals = smc.get_signals()
        
        # Merge signals into dataframe
        for col in signals.columns:
            dataframe[col] = signals[col].values
            
        # --- FVG ATR FILTER (BigBeluga Style) ---
        # Zero out FVGs that are smaller than Threshold * ATR
        # Using ATR(200) to match BigBeluga's volatility reference for "noise".
        if self.fvg_atr_threshold.value > 0:
            # Calculate ATR(200) locally for this filter
            atr_200 = ta.ATR(dataframe, timeperiod=200)
            # Handle potential NaNs at start of data
            atr_200 = atr_200.bfill().ffill()
            
            fvg_threshold = atr_200 * self.fvg_atr_threshold.value
            
            # Bullish FVGs
            bull_range = dataframe['active_bullish_fvg_top'] - dataframe['active_bullish_fvg_bottom']
            mask_small_bull = (dataframe['active_bullish_fvg_top'] > 0) & (bull_range < fvg_threshold)
            dataframe.loc[mask_small_bull, ['active_bullish_fvg_top', 'active_bullish_fvg_bottom']] = 0
            
            # Bearish FVGs
            bear_range = dataframe['active_bearish_fvg_top'] - dataframe['active_bearish_fvg_bottom']
            mask_small_bear = (dataframe['active_bearish_fvg_top'] > 0) & (bear_range < fvg_threshold)
            dataframe.loc[mask_small_bear, ['active_bearish_fvg_top', 'active_bearish_fvg_bottom']] = 0
        
        # --- LOOKAHEAD BIAS FIX ---
        # Zone columns (OB/FVG) must be shifted by 1 to avoid using zones
        # that are created by the CURRENT candle for entry on that same candle.
        # We evaluate entry at candle close, but should only use zones that
        # existed BEFORE the candle started.
        zone_cols = [c for c in dataframe.columns if c.startswith('active_')]
        for col in zone_cols:
            dataframe[col] = dataframe[col].shift(1).fillna(0)
            
        # --- RVOL CALCULATION FOR OBs (BigBeluga Style) ---
        # The volume output array stores the exact volume of the candle that created the OB.
        # RVOL = Volume del OB / Volumen Promedio Diario (o SMA Period)
        dataframe['vol_sma'] = ta.SMA(dataframe['volume'], timeperiod=self.vol_sma_period.value)
        dataframe['vol_sma'] = dataframe['vol_sma'].replace(0, np.nan) # Prevent division by zero
        
        # We calculate the Relative Volume (RVOL) multiplier of the Order Block
        dataframe['bull_ob_rvol'] = dataframe['active_bullish_ob_vol'] / dataframe['vol_sma']
        dataframe['bear_ob_rvol'] = dataframe['active_bearish_ob_vol'] / dataframe['vol_sma']
        
        # Replace NaNs with 0 (for early candles where SMA is not yet formed)
        dataframe['bull_ob_rvol'] = dataframe['bull_ob_rvol'].fillna(0)
        dataframe['bear_ob_rvol'] = dataframe['bear_ob_rvol'].fillna(0)
            
        # --- 2. HTF Calculations (All configured timeframes) ---
        if self.dp:
            htf_list = self._get_active_htf_list()
            for htf in htf_list:
                try:
                    inf_htf = self.dp.get_pair_dataframe(metadata['pair'], htf)
                    if inf_htf.empty:
                        logger.warning(f"No data for {metadata['pair']} {htf}")
                        continue
                        
                    # Calculate SMC on HTF
                    smc_htf = SMCLuxAlgo(
                        inf_htf,
                        internal_length=self.internal_length.value,
                        swing_length=self.swing_length.value,
                    )
                    signals_htf = smc_htf.get_signals()
                    
                    # Create column names with timeframe prefix
                    # e.g., '4h_swing_trend', '1h_swing_trend'
                    swing_col = f'{htf}_swing_trend'
                    internal_col = f'{htf}_internal_trend'
                    
                    inf_htf[swing_col] = signals_htf['swing_trend']
                    inf_htf[internal_col] = signals_htf['internal_trend']
                    
                    # Prepare for merge
                    inf_htf = inf_htf[['date', swing_col, internal_col]].copy()
                    
                    # Merge
                    dataframe = pd.merge(dataframe, inf_htf, on='date', how='left')
                    dataframe[swing_col] = dataframe[swing_col].ffill()
                    dataframe[internal_col] = dataframe[internal_col].ffill()
                    
                except Exception as e:
                    logger.error(f"Error processing HTF {htf}: {e}")
            
        # Add ML Features (Context & Technicals)
        dataframe = self._add_ml_features(dataframe)
        
        # Log summary
        if self.dp and self.enable_logging.value: # Only log if not backtesting or sparse (controlled by param)
            logger.info(
                f"SMC LuxAlgo indicators for {metadata['pair']}: "
                f"MTF Trend Valid={dataframe['swing_trend'].notna().sum()}/{len(dataframe)} "
                f"HTF Trend Valid={dataframe.get('4h_swing_trend', pd.Series()).notna().sum()}/{len(dataframe)}"
            )
        
        return dataframe

    def _add_ml_features(self, df: DataFrame) -> DataFrame:
        """
        Generate consistent ML features for both Training and Inference.
        Must match logic in train_smc_model.py.
        """
        # --- 1. Technical Indicators (Momentum & Volatility) ---
        
        # RSI
        df['rsi'] = ta.RSI(df['close'], timeperiod=14)
        df['ml_rsi'] = df['rsi'] / 100.0
        
        # ADX (Trend Strength) - NEW
        df['adx'] = ta.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        df['ml_adx'] = df['adx'] / 100.0
        
        # CCI (Cyclical) - NEW
        df['cci'] = ta.CCI(df['high'], df['low'], df['close'], timeperiod=20)
        df['ml_cci'] = df['cci'] / 300.0 # Normalize -1 to 1

        # WaveTrend (Oscillator) - NEW
        n1 = 10
        n2 = 21
        ap = (df['high'] + df['low'] + df['close']) / 3
        esa = ta.EMA(ap, timeperiod=n1)
        d = ta.EMA((ap - esa).abs(), timeperiod=n1)
        ci = (ap - esa) / (0.015 * d)
        df['wt1'] = ta.EMA(ci, timeperiod=n2)
        df['wt2'] = ta.SMA(df['wt1'], timeperiod=4)
        
        df['ml_wt1'] = df['wt1'] / 100.0
        df['ml_wt2'] = df['wt2'] / 100.0
        df['ml_wt_diff'] = (df['wt1'] - df['wt2']) / 100.0

        # EMAs
        df['ema_50'] = ta.EMA(df['close'], timeperiod=50)
        df['ema_200'] = ta.EMA(df['close'], timeperiod=200)
        df['ml_dist_ema50'] = (df['close'] - df['ema_50']) / df['ema_50']
        df['ml_dist_ema200'] = (df['close'] - df['ema_200']) / df['ema_200']
        
        # ATR
        df['atr'] = ta.ATR(df, timeperiod=14)
        df['ml_atr_pct'] = df['atr'] / df['close']
        
        # Volatility Filter (Lorentzian Style: atr_1 > atr_10)
        atr_1 = ta.ATR(df, timeperiod=1)
        atr_10 = ta.ATR(df, timeperiod=10)
        df['ml_volatility_high'] = (atr_1 > atr_10).astype(float)
        
        # Volume
        df['volume_ma'] = ta.SMA(df['volume'], timeperiod=20)
        df['ml_volume_ratio'] = df['volume'] / df['volume_ma'].replace(0, 1)
        
        # RSI(9) - Feature 5 from Lorentzian (separate from RSI 14)
        df['rsi_9'] = ta.RSI(df['close'], timeperiod=9)
        df['ml_rsi_9'] = df['rsi_9'] / 100.0
        
        # --- 2. SMC Context ---
        
        # Bars Since Signals
        def bars_since(series):
            return series.cumsum().groupby(series.cumsum()).cumcount()

        df['ml_bars_since_int_bull_choch'] = bars_since(df['internal_choch_bullish'])
        df['ml_bars_since_int_bear_choch'] = bars_since(df['internal_choch_bearish'])
        
        # Trend and Swing
        df['ml_swing_trend'] = df['swing_trend']
        
        # Zone Interaction
        df['ml_in_bull_ob'] = ((df['active_bullish_ob_top'] > 0) & (df['low'] <= df['active_bullish_ob_top'])).astype(int)
        df['ml_in_bear_ob'] = ((df['active_bearish_ob_top'] > 0) & (df['high'] >= df['active_bearish_ob_bottom'])).astype(int)

        # Liquidity Sweeps (Rolling window for ML context)
        if 'internal_sweep_bullish' in df.columns:
            df['ml_recent_bull_sweep'] = df['internal_sweep_bullish'].rolling(3, min_periods=1).max().fillna(0)
            df['ml_recent_bear_sweep'] = df['internal_sweep_bearish'].rolling(3, min_periods=1).max().fillna(0)
        else:
            df['ml_recent_bull_sweep'] = 0
            df['ml_recent_bear_sweep'] = 0
       
        # Fill NaNs created by indicators
        feature_cols = [c for c in df.columns if c.startswith('ml_')]
        df[feature_cols] = df[feature_cols].fillna(0)
        
        return df
    
    def _apply_zone_memory(self, dataframe: DataFrame, smc: SMCLuxAlgo) -> DataFrame:
        """Apply zone memory for OBs and FVGs."""
        n = len(dataframe)
        
        # Initialize zone columns
        dataframe['active_bullish_ob_top'] = 0.0
        dataframe['active_bullish_ob_bottom'] = 0.0
        dataframe['active_bearish_ob_top'] = 0.0
        dataframe['active_bearish_ob_bottom'] = 0.0
        
        dataframe['active_bullish_fvg_top'] = 0.0
        dataframe['active_bullish_fvg_bottom'] = 0.0
        dataframe['active_bearish_fvg_top'] = 0.0
        dataframe['active_bearish_fvg_bottom'] = 0.0
        
        # Process each bar to find active zones
        for i in range(n):
            # Find active OBs at this bar
            active_obs = smc.get_active_order_blocks(as_of_index=i)
            
            for ob in active_obs:
                if ob.bar_index <= i:
                    if ob.bias == 1:  # Bullish
                        if dataframe.loc[dataframe.index[i], 'active_bullish_ob_top'] == 0:
                            dataframe.loc[dataframe.index[i], 'active_bullish_ob_top'] = ob.top
                            dataframe.loc[dataframe.index[i], 'active_bullish_ob_bottom'] = ob.bottom
                    else:  # Bearish
                        if dataframe.loc[dataframe.index[i], 'active_bearish_ob_top'] == 0:
                            dataframe.loc[dataframe.index[i], 'active_bearish_ob_top'] = ob.top
                            dataframe.loc[dataframe.index[i], 'active_bearish_ob_bottom'] = ob.bottom
            
            # Find active FVGs at this bar
            active_fvgs = smc.get_active_fvgs(as_of_index=i)
            
            for fvg in active_fvgs:
                if fvg.bar_index <= i:
                    if fvg.bias == 1:  # Bullish
                        if dataframe.loc[dataframe.index[i], 'active_bullish_fvg_top'] == 0:
                            dataframe.loc[dataframe.index[i], 'active_bullish_fvg_top'] = fvg.top
                            dataframe.loc[dataframe.index[i], 'active_bullish_fvg_bottom'] = fvg.bottom
                    else:  # Bearish
                        if dataframe.loc[dataframe.index[i], 'active_bearish_fvg_top'] == 0:
                            dataframe.loc[dataframe.index[i], 'active_bearish_fvg_top'] = fvg.top
                            dataframe.loc[dataframe.index[i], 'active_bearish_fvg_bottom'] = fvg.bottom
        
        return dataframe

    # ==========================================================================
    # ENTRY LOGIC
    # ==========================================================================
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Generate entry signals based on SMC LuxAlgo logic."""
        
        lookback = self.entry_signal_lookback.value
        
        # ===== BULLISH STRUCTURE SIGNALS =====
        # Pine Logic: if stratStructureType == "Internal" -> internalBullishCHoCH else swingBullishCHoCH
        
        # Determine strict CHoCH signals (Reversals)
        internal_choch_bull = dataframe['internal_choch_bullish'] == 1
        swing_choch_bull = dataframe['swing_choch_bullish'] == 1
        
        if lookback > 1:
            # Bullish CHoCH
            ic_bull_rolled = internal_choch_bull.rolling(lookback).max() == 1
            sc_bull_rolled = swing_choch_bull.rolling(lookback).max() == 1
            
            if not self.allow_immediate_entry.value:
                # Conservative: Signal must be historical (not current candle)
                internal_choch_bull = ic_bull_rolled & (internal_choch_bull == False)
                swing_choch_bull = sc_bull_rolled & (swing_choch_bull == False)
            else:
                internal_choch_bull = ic_bull_rolled
                swing_choch_bull = sc_bull_rolled
            
        # Determine BOS signals (Continuations) - validated by require_choch=False
        internal_bos_bull = dataframe['internal_bos_bullish'] == 1
        swing_bos_bull = dataframe['swing_bos_bullish'] == 1
        
        if lookback > 1:
            ib_bull_rolled = internal_bos_bull.rolling(lookback).max() == 1
            sb_bull_rolled = swing_bos_bull.rolling(lookback).max() == 1
            
            if not self.allow_immediate_entry.value:
                internal_bos_bull = ib_bull_rolled & (internal_bos_bull == False)
                swing_bos_bull = sb_bull_rolled & (swing_bos_bull == False)
            else:
                internal_bos_bull = ib_bull_rolled
                swing_bos_bull = sb_bull_rolled

        # Combine based on configuration
        bullish_structure = pd.Series(False, index=dataframe.index)
        
        # Internal Signals
        if self.use_internal_signals.value:
            if self.require_choch.value:
                bullish_structure |= internal_choch_bull
            else:
                bullish_structure |= (internal_choch_bull | internal_bos_bull)
                
        # Swing Signals
        if self.use_swing_signals.value:
            if self.require_choch.value:
                bullish_structure |= swing_choch_bull
            else:
                bullish_structure |= (swing_choch_bull | swing_bos_bull)
        
        # ===== BEARISH STRUCTURE SIGNALS =====
        internal_choch_bear = dataframe['internal_choch_bearish'] == 1
        swing_choch_bear = dataframe['swing_choch_bearish'] == 1
        
        if lookback > 1:
            ic_bear_rolled = internal_choch_bear.rolling(lookback).max() == 1
            sc_bear_rolled = swing_choch_bear.rolling(lookback).max() == 1
            
            if not self.allow_immediate_entry.value:
                internal_choch_bear = ic_bear_rolled & (internal_choch_bear == False)
                swing_choch_bear = sc_bear_rolled & (swing_choch_bear == False)
            else:
                internal_choch_bear = ic_bear_rolled
                swing_choch_bear = sc_bear_rolled
            
        internal_bos_bear = dataframe['internal_bos_bearish'] == 1
        swing_bos_bear = dataframe['swing_bos_bearish'] == 1
        
        if lookback > 1:
            ib_bear_rolled = internal_bos_bear.rolling(lookback).max() == 1
            sb_bear_rolled = swing_bos_bear.rolling(lookback).max() == 1
            
            if not self.allow_immediate_entry.value:
                internal_bos_bear = ib_bear_rolled & (internal_bos_bear == False)
                swing_bos_bear = sb_bear_rolled & (swing_bos_bear == False)
            else:
                internal_bos_bear = ib_bear_rolled
                swing_bos_bear = sb_bear_rolled

        bearish_structure = pd.Series(False, index=dataframe.index)
        
        # Internal Signals
        if self.use_internal_signals.value:
            if self.require_choch.value:
                bearish_structure |= internal_choch_bear
            else:
                bearish_structure |= (internal_choch_bear | internal_bos_bear)
        
        # Swing Signals
        if self.use_swing_signals.value:
            if self.require_choch.value:
                bearish_structure |= swing_choch_bear
            else:
                bearish_structure |= (swing_choch_bear | swing_bos_bear)
        
        # ===== ZONE CONDITIONS (Optional Filters) =====
        # Price in bullish OB zone (Must close ABOVE bottom)
        
        in_bull_ob = (
            (dataframe['active_bullish_ob_top'] > 0) &
            (dataframe['low'] <= dataframe['active_bullish_ob_top']) &
            (dataframe['high'] >= dataframe['active_bullish_ob_bottom']) &
            (dataframe['close'] >= dataframe['active_bullish_ob_bottom'])
        )
        
        # Apply RVOL Volumetric Filter to OBs
        if self.require_volumetric_ob.value:
            in_bull_ob &= (dataframe['bull_ob_rvol'] >= self.ob_rvol_threshold.value)
            
        dataframe['in_bullish_ob'] = in_bull_ob
        
        # Price in bearish OB zone (Must close BELOW top)
        in_bear_ob = (
            (dataframe['active_bearish_ob_top'] > 0) &
            (dataframe['high'] >= dataframe['active_bearish_ob_bottom']) &
            (dataframe['low'] <= dataframe['active_bearish_ob_top']) &
            (dataframe['close'] <= dataframe['active_bearish_ob_top'])
        )
        
        # Apply RVOL Volumetric Filter to OBs
        if self.require_volumetric_ob.value:
            in_bear_ob &= (dataframe['bear_ob_rvol'] >= self.ob_rvol_threshold.value)
            
        dataframe['in_bearish_ob'] = in_bear_ob
        
        # Price in bullish FVG zone (Must close ABOVE bottom to be valid)
        dataframe['in_bullish_fvg'] = (
            (dataframe['active_bullish_fvg_top'] > 0) &
            (dataframe['low'] <= dataframe['active_bullish_fvg_top']) &
            (dataframe['close'] >= dataframe['active_bullish_fvg_bottom'])
        )
        
        # Price in bearish FVG zone (Must close BELOW top to be valid)
        dataframe['in_bearish_fvg'] = (
            (dataframe['active_bearish_fvg_top'] > 0) &
            (dataframe['high'] >= dataframe['active_bearish_fvg_bottom']) &
            (dataframe['close'] <= dataframe['active_bearish_fvg_top'])
        )

        # Initialize Breaker columns (defaults to False)
        dataframe['in_bull_breaker'] = False
        dataframe['in_bear_breaker'] = False
        dataframe['in_bull_fvg_breaker'] = False
        dataframe['in_bear_fvg_breaker'] = False
        
        # Zone requirements - OR logic (any enabled filter can pass)
        # If NO zone filter is enabled, all bars pass (default True)
        any_zone_enabled = self.require_ob_zone.value or self.require_fvg_zone.value
        
        if any_zone_enabled:
            # Start with False, use OR to add conditions
            bullish_zone = pd.Series(False, index=dataframe.index)
            bearish_zone = pd.Series(False, index=dataframe.index)
            
            if self.require_ob_zone.value:
                bullish_zone |= dataframe['in_bullish_ob']
                bearish_zone |= dataframe['in_bearish_ob']
                
                # Add Breakers to OB logic (User description: Breaker OB is a key entry point)
                if 'active_bullish_breaker_top' in dataframe.columns:
                     in_bull_breaker = (
                        (dataframe['active_bullish_breaker_top'] > 0) & 
                        # Price touching breaker (Breaker acts as Support)
                        (dataframe['low'] <= dataframe['active_bullish_breaker_top']) & 
                        (dataframe['high'] >= dataframe['active_bullish_breaker_bottom']) &
                        (dataframe['close'] >= dataframe['active_bullish_breaker_bottom'])
                     )
                     
                     if self.require_volumetric_ob.value:
                         # A bullish breaker comes from a broken bearish OB. Use bearish RVOL.
                         in_bull_breaker &= (dataframe['bear_ob_rvol'] >= self.ob_rvol_threshold.value)
                         
                     dataframe['in_bull_breaker'] = in_bull_breaker
                     bullish_zone |= dataframe['in_bull_breaker']
                     
                     in_bear_breaker = (
                        (dataframe['active_bearish_breaker_top'] > 0) & 
                        # Price touching breaker (Breaker acts as Resistance)
                        (dataframe['high'] >= dataframe['active_bearish_breaker_bottom']) &
                        (dataframe['low'] <= dataframe['active_bearish_breaker_top']) &
                        (dataframe['close'] <= dataframe['active_bearish_breaker_top'])
                     )
                     
                     if self.require_volumetric_ob.value:
                         # A bearish breaker comes from a broken bullish OB. Use bullish RVOL.
                         in_bear_breaker &= (dataframe['bull_ob_rvol'] >= self.ob_rvol_threshold.value)
                     
                     dataframe['in_bear_breaker'] = in_bear_breaker
                     bearish_zone |= dataframe['in_bear_breaker']
            
            if self.require_fvg_zone.value:
                bullish_zone |= dataframe['in_bullish_fvg']
                bearish_zone |= dataframe['in_bearish_fvg']
                
                # Add FVG Breakers logic
                if 'active_bullish_fvg_breaker_top' in dataframe.columns:
                     dataframe['in_bull_fvg_breaker'] = (
                        (dataframe['active_bullish_fvg_breaker_top'] > 0) & 
                        (dataframe['low'] <= dataframe['active_bullish_fvg_breaker_top']) & 
                        (dataframe['high'] >= dataframe['active_bullish_fvg_breaker_bottom']) &
                        (dataframe['close'] >= dataframe['active_bullish_fvg_breaker_bottom'])
                     )
                     bullish_zone |= dataframe['in_bull_fvg_breaker']
                     
                     dataframe['in_bear_fvg_breaker'] = (
                        (dataframe['active_bearish_fvg_breaker_top'] > 0) & 
                        (dataframe['high'] >= dataframe['active_bearish_fvg_breaker_bottom']) &
                        (dataframe['low'] <= dataframe['active_bearish_fvg_breaker_top']) &
                        (dataframe['close'] <= dataframe['active_bearish_fvg_breaker_top'])
                     )
                     bearish_zone |= dataframe['in_bear_fvg_breaker']

            # --- DYNAMIC STOPLOSS CALCULATION (For DataFrame) ---
            # We calculate what the SL PRICE would be for this candle if we entered.
            # Long SL = Lowest Support Zone Bottom - offset
            # Short SL = Highest Resistance Zone Top + offset
            
            # Helper to get "Best" zone bottom for Long (OB or Breaker)
            # We want the zone that is "active" and closest to price?
            # Actually, we want the zone that is "holding" the price.
            # If multiple overlapping, use the lowest bottom for safety?
            # Or the bottom of the one we are touching.
            
            # Simple approach: Max of available support zone bottoms? No, SL is below bottom.
            # So, Min of bottoms?
            
            # Create temporary series
            bull_sl_price = pd.Series(np.nan, index=dataframe.index)
            bear_sl_price = pd.Series(np.nan, index=dataframe.index)
            
            # Find relevant zone limit for SL
            # Bull OB Bottom
            mask_bull_ob = (dataframe['active_bullish_ob_bottom'] > 0)
            bull_sl_price[mask_bull_ob] = dataframe.loc[mask_bull_ob, 'active_bullish_ob_bottom']
            
            # Bull Breaker Bottom (might overwrite if present - usually we touch one or other)
            if 'active_bullish_breaker_bottom' in dataframe.columns:
                mask_bull_brk = (dataframe['active_bullish_breaker_bottom'] > 0)
                bull_sl_price = np.fmin(bull_sl_price, dataframe['active_bullish_breaker_bottom'])

            # FVG Breaker Bottom (NEW)
            if 'active_bullish_fvg_breaker_bottom' in dataframe.columns:
                 # If active, use its bottom
                 bull_sl_price = np.fmin(bull_sl_price, dataframe['active_bullish_fvg_breaker_bottom'])

            # Apply offset
            dataframe['sl_long_price'] = bull_sl_price * (1 - self.sl_buffer_pct.value)
            
            # Same for Shorts (Top + offset)
            bear_sl_price[mask_bull_ob] = np.nan # reset logic
            bear_sl_price = pd.Series(np.nan, index=dataframe.index)
            
            mask_bear_ob = (dataframe['active_bearish_ob_top'] > 0)
            bear_sl_price[mask_bear_ob] = dataframe.loc[mask_bear_ob, 'active_bearish_ob_top']
            
            if 'active_bearish_breaker_top' in dataframe.columns:
                 bear_sl_price = np.fmax(bear_sl_price, dataframe['active_bearish_breaker_top'])

            # FVG Breaker Top (NEW)
            if 'active_bullish_fvg_breaker_top' in dataframe.columns: # wait, Bearish FVG Breaker for Shorts
                 if 'active_bearish_fvg_breaker_top' in dataframe.columns:
                      bear_sl_price = np.fmax(bear_sl_price, dataframe['active_bearish_fvg_breaker_top'])
            
            dataframe['sl_short_price'] = bear_sl_price * (1 + self.sl_buffer_pct.value)
            
            # ===== LIQUIDITY SWEEP BYPASS =====
            # If a recent sweep is detected, bypass zone requirement
            # Sweep + CHoCH = high confidence entry without needing OB/FVG retest
            if self.use_sweep_bypass.value:
                sweep_lb = self.sweep_lookback.value
                
                # Recent bullish sweep (wick took sellside liquidity below pivot low)
                recent_bull_sweep = (
                    (dataframe['internal_sweep_bullish'].rolling(sweep_lb, min_periods=1).max() == 1) |
                    (dataframe['swing_sweep_bullish'].rolling(sweep_lb, min_periods=1).max() == 1)
                )
                
                # Recent bearish sweep (wick took buyside liquidity above pivot high)
                recent_bear_sweep = (
                    (dataframe['internal_sweep_bearish'].rolling(sweep_lb, min_periods=1).max() == 1) |
                    (dataframe['swing_sweep_bearish'].rolling(sweep_lb, min_periods=1).max() == 1)
                )

                # Sweep bypasses zone requirement (INCLUDING volumetric filter)
                bullish_zone |= recent_bull_sweep
                bearish_zone |= recent_bear_sweep
                
                final_bull_zone = bullish_zone
                final_bear_zone = bearish_zone
            else:
                final_bull_zone = bullish_zone
                final_bear_zone = bearish_zone
        else:
            final_bull_zone = bullish_zone
            final_bear_zone = bearish_zone
        
        # Debugging Zone Counts (Only if backtesting/dry)
        if self.dp: 
            n_bull_zones = bullish_zone.sum()
            n_bear_zones = bearish_zone.sum()
            if (n_bull_zones == 0 or n_bear_zones == 0) and (self.require_ob_zone.value or self.require_fvg_zone.value):
                logger.warning(
                    f"⚠️ ZONES EMPTY! Bull: {n_bull_zones}, Bear: {n_bear_zones}. "
                    f"OB Req: {self.require_ob_zone.value}, FVG Req: {self.require_fvg_zone.value}"
                )
                logger.warning(f"   Active Bull OB Candles: {(dataframe['active_bullish_ob_top'] > 0).sum()}")
                logger.warning(f"   Active Bull FVG Candles: {(dataframe['active_bullish_fvg_top'] > 0).sum()}")
        
        # ===== HTF TREND FILTER =====
        if self.trade_with_trend.value:
            # Local Trend Filter (MTF)
            bullish_trend_ok = dataframe['swing_trend'] == 1
            bearish_trend_ok = dataframe['swing_trend'] == -1
            
            # HTF Filter - Check ALL configured timeframes
            if self.use_htf_filter.value:
                htf_list = self._get_active_htf_list()
                # Per-HTF internal trend toggle mapping
                htf_internal_map = {
                    0: self.htf_1_use_internal.value,
                    1: self.htf_2_use_internal.value,
                }
                for idx, htf in enumerate(htf_list):
                    use_internal = htf_internal_map.get(idx, False)
                    trend_type = 'internal_trend' if use_internal else 'swing_trend'
                    trend_col = f'{htf}_{trend_type}'
                    
                    if trend_col in dataframe.columns:
                        htf_bull = dataframe[trend_col] == 1
                        htf_bear = dataframe[trend_col] == -1
                        
                        # AND logic: ALL HTFs must align
                        bullish_trend_ok &= htf_bull
                        bearish_trend_ok &= htf_bear
        else:
            bullish_trend_ok = True
            bearish_trend_ok = True
        
        # ===== FINAL CONDITIONS =====
        long_condition = bullish_structure & bullish_zone & bullish_trend_ok
        short_condition = bearish_structure & bearish_zone & bearish_trend_ok
        
        # ===== ML FILTER =====
        ml_model = self._get_ml_model_for_pair(metadata['pair'])
        if self.use_ml_filter.value and ml_model:
            try:
                # Prepare Features
                feature_cols = [c for c in dataframe.columns if c.startswith('ml_')]
                
                # Predict Longs (direction = 1)
                X_long = dataframe[feature_cols].copy()
                X_long['direction'] = 1
                long_proba = ml_model.predict_proba(X_long)[:, 1]
                ml_long_ok = long_proba > self.ml_threshold.value
                
                # Predict Shorts (direction = -1)
                X_short = dataframe[feature_cols].copy()
                X_short['direction'] = -1
                short_proba = ml_model.predict_proba(X_short)[:, 1]
                ml_short_ok = short_proba > self.ml_threshold.value
                
                # Apply Filter
                long_signals_count = long_condition.sum()
                short_signals_count = short_condition.sum()
                
                long_condition &= ml_long_ok
                short_condition &= ml_short_ok
                
                # Log only if a signal was blocked
                blocked_long = long_signals_count - long_condition.sum()
                blocked_short = short_signals_count - short_condition.sum()
                
                if blocked_long > 0 or blocked_short > 0:
                     logger.info(f"🤖 ML Filter ({metadata['pair']}) Blocked {blocked_long} Longs, {blocked_short} Shorts. Threshold: {self.ml_threshold.value}")

            except Exception as e:
                logger.error(f"❌ ML Inference Failed for {metadata['pair']}: {e}")
                pass
        
        # Apply signals
        dataframe.loc[long_condition, 'enter_long'] = 1
        dataframe.loc[short_condition, 'enter_short'] = 1
        
        # === DETAILED ENTRY LOGGING ===
        # Log each entry with zone details (only if logging enabled)
        # 
        # NOTE: Future consideration - stricter zone validation:
        # Current logic: high >= fvg_bottom (price touches or passes zone)
        # Stricter: (high >= fvg_bottom) & (high <= fvg_top) (price INSIDE zone)
        #
        if self.enable_logging.value:
            lookback = self.entry_signal_lookback.value
            
            short_entries = dataframe[short_condition].copy()
            for idx, row in short_entries.iterrows():
                # Find Signal Source (Bars Ago)
                bars_ago = 0
                signal_type = "None"
                signal_close = 0.0
                
                # We need the integer index to look back
                # dataframe index might be date, so we use get_loc if needed, 
                # but iterrows gives index. If index is date, we need strict integer access.
                # Safer to use the 'internal_choch_bearish' column directly on the row 
                # BUT row is just one slice. We need context.
                # Actually, 'short_entries' is a slice. accessing global 'dataframe' by index is better.
                
                # Optimization: Check current row first
                if row.get('internal_choch_bearish', 0) == 1:
                    signal_type = "IntCHoCH"
                    bars_ago = 0
                    signal_close = row['close']
                elif row.get('swing_choch_bearish', 0) == 1:
                    signal_type = "SwingCHoCH"
                    bars_ago = 0
                    signal_close = row['close']
                else:
                    # Look back
                    # This is slow in a loop but fine for logging only
                    # We need the integer position of 'idx' in 'dataframe'
                    try:
                        i = dataframe.index.get_loc(idx)
                        for k in range(1, lookback + 1):
                            if i - k >= 0:
                                prev_row = dataframe.iloc[i - k]
                                if prev_row['internal_choch_bearish'] == 1:
                                    signal_type = "IntCHoCH"
                                    bars_ago = k
                                    signal_close = prev_row['close']
                                    break
                                elif prev_row['swing_choch_bearish'] == 1:
                                    signal_type = "SwingCHoCH"
                                    bars_ago = k
                                    signal_close = prev_row['close']
                                    break
                    except Exception:
                        pass

                reasons = []
                if row.get('in_bearish_ob', False): reasons.append("OB")
                if row.get('in_bearish_fvg', False): reasons.append("FVG")
                if row.get('in_bear_breaker', False): reasons.append("Breaker")
                if row.get('in_bear_fvg_breaker', False): reasons.append("FVG-Breaker")
                
                reason_str = "+".join(reasons) if reasons else "StructureOnly"
                
                bear_fvg = f"[{row.get('active_bearish_fvg_bottom', 0):.8f}, {row.get('active_bearish_fvg_top', 0):.8f}]" if row.get('active_bearish_fvg_top', 0) > 0 else "None"
                bear_ob = f"[{row.get('active_bearish_ob_bottom', 0):.8f}, {row.get('active_bearish_ob_top', 0):.8f}]" if row.get('active_bearish_ob_top', 0) > 0 else "None"
                bear_brk = f"[{row.get('active_bearish_breaker_bottom', 0):.8f}, {row.get('active_bearish_breaker_top', 0):.8f}]" if row.get('active_bearish_breaker_top', 0) > 0 else "None"
                bear_fvg_brk = f"[{row.get('active_bearish_fvg_breaker_bottom', 0):.8f}, {row.get('active_bearish_fvg_breaker_top', 0):.8f}]" if row.get('active_bearish_fvg_breaker_top', 0) > 0 else "None"
                
                logger.info(
                    f"📉 SHORT ENTRY: {metadata['pair']} @ {row['date']} | "
                    f"Signal={signal_type} ({bars_ago} bars ago, Close={signal_close:.8f}) | Reason={reason_str} | "
                    f"C={row['close']:.8f} H={row['high']:.8f} | "
                    f"FVG={bear_fvg} OB={bear_ob} BRK={bear_brk} FVG-BRK={bear_fvg_brk}"
                )
            
            long_entries = dataframe[long_condition].copy()
            for idx, row in long_entries.iterrows():
                # Find Signal Source (Bars Ago)
                bars_ago = 0
                signal_type = "None"
                signal_close = 0.0
                
                if row.get('internal_choch_bullish', 0) == 1:
                    signal_type = "IntCHoCH"
                    bars_ago = 0
                    signal_close = row['close']
                elif row.get('swing_choch_bullish', 0) == 1:
                    signal_type = "SwingCHoCH"
                    bars_ago = 0
                    signal_close = row['close']
                else:
                    try:
                        i = dataframe.index.get_loc(idx)
                        for k in range(1, lookback + 1):
                            if i - k >= 0:
                                prev_row = dataframe.iloc[i - k]
                                if prev_row['internal_choch_bullish'] == 1:
                                    signal_type = "IntCHoCH"
                                    bars_ago = k
                                    signal_close = prev_row['close']
                                    break
                                elif prev_row['swing_choch_bullish'] == 1:
                                    signal_type = "SwingCHoCH"
                                    bars_ago = k
                                    signal_close = prev_row['close']
                                    break
                    except Exception:
                        pass
                        
                reasons = []
                if row.get('in_bullish_ob', False): reasons.append("OB")
                if row.get('in_bullish_fvg', False): reasons.append("FVG")
                if row.get('in_bull_breaker', False): reasons.append("Breaker")
                if row.get('in_bull_fvg_breaker', False): reasons.append("FVG-Breaker")
                
                reason_str = "+".join(reasons) if reasons else "StructureOnly"

                bull_fvg = f"[{row.get('active_bullish_fvg_bottom', 0):.8f}, {row.get('active_bullish_fvg_top', 0):.8f}]" if row.get('active_bullish_fvg_top', 0) > 0 else "None"
                bull_ob = f"[{row.get('active_bullish_ob_bottom', 0):.8f}, {row.get('active_bullish_ob_top', 0):.8f}]" if row.get('active_bullish_ob_top', 0) > 0 else "None"
                bull_brk = f"[{row.get('active_bullish_breaker_bottom', 0):.8f}, {row.get('active_bullish_breaker_top', 0):.8f}]" if row.get('active_bullish_breaker_top', 0) > 0 else "None"
                bull_fvg_brk = f"[{row.get('active_bullish_fvg_breaker_bottom', 0):.8f}, {row.get('active_bullish_fvg_breaker_top', 0):.8f}]" if row.get('active_bullish_fvg_breaker_top', 0) > 0 else "None"
                
                logger.info(
                    f"📈 LONG ENTRY: {metadata['pair']} @ {row['date']} | "
                    f"Signal={signal_type} ({bars_ago} bars ago, Close={signal_close:.8f}) | Reason={reason_str} | "
                    f"C={row['close']:.8f} L={row['low']:.8f} | "
                    f"FVG={bull_fvg} OB={bull_ob} BRK={bull_brk} FVG-BRK={bull_fvg_brk}"
                )
        
        # Log stats
        logger.info(
            f"SMC LuxAlgo signals for {metadata['pair']}: "
            f"Long={long_condition.sum()}, Short={short_condition.sum()}"
        )
        
        return dataframe

    # ==========================================================================
    # EXIT LOGIC
    # ==========================================================================
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Generate exit signals on opposite CHoCH (trend reversal)."""
        
        if True: # Always check for exits if parameters are enabled
            
            # Exit long on bearish CHoCH (trend reversal)
            # Check configured exit triggers
            exit_long = pd.Series(False, index=dataframe.index)
            if self.exit_on_internal_choch.value:
                exit_long |= (dataframe['internal_choch_bearish'] == 1)
            if self.exit_on_swing_choch.value:
                exit_long |= (dataframe['swing_choch_bearish'] == 1)
            
            # Exit short on bullish CHoCH
            exit_short = pd.Series(False, index=dataframe.index)
            if self.exit_on_internal_choch.value:
                exit_short |= (dataframe['internal_choch_bullish'] == 1)
            if self.exit_on_swing_choch.value:
                exit_short |= (dataframe['swing_choch_bullish'] == 1)
        else:
            exit_long = pd.Series(False, index=dataframe.index)
            exit_short = pd.Series(False, index=dataframe.index)
        
        dataframe.loc[exit_long, 'exit_long'] = 1
        dataframe.loc[exit_short, 'exit_short'] = 1
        
        return dataframe
    
    # ==========================================================================
    # LEVERAGE & RISK MANAGEMENT
    # ==========================================================================
    
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: str,
                 side: str, **kwargs) -> float:
        """Get leverage from config."""
        return self.config.get('leverage', 10.0)
    
    def _get_timeframe_minutes(self) -> int:
        """Convert timeframe string to minutes."""
        tf = self.timeframe
        if tf.endswith('m'):
            return int(tf[:-1])
        elif tf.endswith('h'):
            return int(tf[:-1]) * 60
        elif tf.endswith('d'):
            return int(tf[:-1]) * 1440
        return 15  # Default fallback
    
    # --------------------------------------------------------------------------
    # Circuit Breaker & Risk Helper
    # --------------------------------------------------------------------------
    def _check_market_panic(self, current_time: datetime) -> bool:
        """
        Check if market is in panic mode (multiple recent SL hits).
        Values cached for 1 minute to allow blocking and tight control.
        """
        # Cache mechanism using instance attributes
        last_check = getattr(self, '_panic_last_check', None)
        if last_check and (current_time - last_check).total_seconds() < 60:
            return getattr(self, '_panic_active', False)
            
        # Perform check
        self._panic_last_check = current_time
        self._panic_active = False # Default
        
        if not self.dp:
            return False
        
        # Check if Circuit Breaker is enabled
        if not self.circuit_breaker_enabled.value:
            return False
            
        # Use configurable parameters
        PANIC_WINDOW_MIN = self.circuit_breaker_window.value
        PANIC_SL_LIMIT = self.circuit_breaker_limit.value
        
        lookback = current_time - timedelta(minutes=PANIC_WINDOW_MIN)
        
        # Robust timezone handling
        if lookback.tzinfo is None and current_time.tzinfo is not None:
             lookback = lookback.replace(tzinfo=current_time.tzinfo)
        
        trades = Trade.get_trades_proxy(is_open=False)
        
        sl_count = 0
        for t in trades:
            if t.close_date:
                c_date = t.close_date
                # Sync timezones if needed
                if c_date.tzinfo is None and lookback.tzinfo is not None:
                     c_date = c_date.replace(tzinfo=lookback.tzinfo)
                elif c_date.tzinfo is not None and lookback.tzinfo is None:
                     c_date = c_date.replace(tzinfo=None)
                     
                if c_date >= lookback:
                    # Count SL hits (Exchange SL, Strategy SL, or forced exit with loss)
                    if (t.exit_reason in ['stop_loss', 'stoploss_on_exchange', 'force_exit', 'emergency_exit']) and (t.close_profit < 0):
                        sl_count += 1
                        
        if sl_count >= PANIC_SL_LIMIT:
             self._panic_active = True
             if self.enable_logging.value:
                 logger.warning(f"🚨 CIRCUIT BREAKER ACTIVE: {sl_count} Losses in last {PANIC_WINDOW_MIN}m. Market is volatile.")
                 
        return self._panic_active

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                           time_in_force: str, current_time: datetime, entry_tag: str | None,
                           side: str, **kwargs) -> bool:
        """
        Prevent entry if:
        1. CIRCUIT BREAKER (High Volatility/Panic) - New
        2. Already have an open trade for this pair
        3. Recently closed a trade for this pair (cooldown based on entry_signal_lookback)
        
        This ensures ONE TRADE PER SYMBOL at a time and prevents re-entry
        on the same signal after SL hit.
        """
        # 1. CIRCUIT BREAKER (Global Panic Switch)
        if self._check_market_panic(current_time):
             if self.enable_logging.value:
                  logger.info(f"⛔ Entry blocked for {pair} due to Circuit Breaker (High Volatility/Panic).")
             return False

        # 2. Check for existing open trades on this pair
        open_trades = Trade.get_trades_proxy(pair=pair, is_open=True)
        if open_trades:
            logger.info(f"Entry blocked for {pair}: Already have an open trade")
            return False
        
        # 2. Cooldown after closed trade (prevent immediate re-entry on same signal)
        # Cooldown = entry_signal_lookback * timeframe_minutes
        lookback_minutes = self.entry_signal_lookback.value * self._get_timeframe_minutes()
        cooldown_start = current_time - timedelta(minutes=lookback_minutes)
        
        recent_trades = Trade.get_trades_proxy(
            pair=pair,
            is_open=False,
        )
        
        # Filter trades that closed within the cooldown period
        for trade in recent_trades:
            if trade.close_date:
                # Normalize timezones for comparison
                c_date = trade.close_date
                c_start = cooldown_start
                if c_date.tzinfo is None and c_start.tzinfo is not None:
                    c_date = c_date.replace(tzinfo=c_start.tzinfo)
                elif c_date.tzinfo is not None and c_start.tzinfo is None:
                    c_start = c_start.replace(tzinfo=c_date.tzinfo)
                
                if c_date >= c_start:
                    # Use c_date (normalized) to avoid timezone mismatch with current_time
                    c_time = current_time
                    if c_time.tzinfo is None and c_date.tzinfo is not None:
                        c_time = c_time.replace(tzinfo=c_date.tzinfo)
                    elif c_time.tzinfo is not None and c_date.tzinfo is None:
                        c_date = c_date.replace(tzinfo=c_time.tzinfo)
                    minutes_since_close = (c_time - c_date).total_seconds() / 60
                    logger.info(
                        f"Entry blocked for {pair}: Trade closed {minutes_since_close:.0f}m ago, "
                        f"cooldown is {lookback_minutes}m (lookback={self.entry_signal_lookback.value})"
                    )
                    return False
        
        return True
    
    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Break Even Logic with persistent state.
        Once BE is activated (when price reaches TP1), the stoploss is fixed at entry price + small buffer for fees.
        """
        if not self.move_be_at_tp1.value and not self.use_custom_stoploss:
            return 1  # Use default stoploss
            
        # --- 1. INITIAL DYNAMIC STOPLOSS (At Entry) ---
        # If trade just opened (no BE yet), we check for Dynamic SL from structure
        be_activated = trade.get_custom_data('be_activated', default=False)
        
        if not be_activated and self.use_dynamic_stoploss.value:
            # Check if we already have the initial SL price stored
            # This ensures we only read the dataframe ONCE at the start of the trade
            initial_sl_price = trade.get_custom_data('initial_sl_price')
            
            # If not stored, try to find it in the dataframe
            if initial_sl_price is None:
                try:
                    # We need the dataframe. 
                    # Optimization: Only load analyzed dataframe if we haven't stored the SL yet
                    dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
                    
                    # Find the candle where the trade opened
                    candle = dataframe.loc[dataframe['date'] == trade.open_date_utc]
                    
                    if not candle.empty:
                        row = candle.iloc[0]
                        price_found = 0.0
                        
                        if trade.is_short:
                            if 'sl_short_price' in row and not pd.isna(row['sl_short_price']):
                                 price_found = row['sl_short_price']
                        else:
                            if 'sl_long_price' in row and not pd.isna(row['sl_long_price']):
                                 price_found = row['sl_long_price']
                        
                        if price_found > 0:
                            initial_sl_price = price_found
                            # STORE IT so we don't recalculate again
                            trade.set_custom_data('initial_sl_price', initial_sl_price)
                            logger.info(f"Initial Dynamic SL found for {pair}: {initial_sl_price}")
                except Exception as e:
                    # Fallback to default
                    pass

            # If we have a valid initial SL price (either from storage or just found)
            if initial_sl_price and initial_sl_price > 0:
                 # Use Freqtrade's official helper function
                 logger.debug(f"Dynamic SL for {pair}: sl_price={initial_sl_price:.4f}, current_rate={current_rate:.4f}")
                 return stoploss_from_absolute(initial_sl_price, current_rate, is_short=trade.is_short, leverage=trade.leverage)

        # --- 2. BREAK EVEN LOGIC ---
        # Check if BE was already activated (persistent across ticks)
        
        # Calculate price movement to check if we should activate BE
        if trade.is_short:
            current_extremum = trade.min_rate if trade.min_rate is not None else current_rate
            price_movement = (trade.open_rate - current_extremum) / trade.open_rate
        else:
            current_extremum = trade.max_rate if trade.max_rate is not None else current_rate
            price_movement = (current_extremum - trade.open_rate) / trade.open_rate
        
        # Activate BE if price moved past TP1 and not already activated
        # OR if Market Panic (Circuit Breaker active) and we have some profit
        market_panic = self._check_market_panic(current_time)
        should_activate_be = False
        
        # Determine the target percentage to trigger Break Even
        be_trigger = self.be_trigger_pct.value if self.be_trigger_pct.value > 0 else self.tp1_pct.value
        
        if self.move_be_at_tp1.value and price_movement >= be_trigger:
            should_activate_be = True
        elif market_panic and price_movement >= 0.005: # Panic: Force BE at 0.5% profit
            should_activate_be = True
            if self.enable_logging.value and not be_activated:
                 logger.info(f"🚨 Panic Mode: Forcing BE for {pair} at {price_movement:.2%} profit.")

        if not be_activated and should_activate_be:
            # Calculate and STORE the BE stop price ONCE
            fee_buffer_pct = 0.001  # 0.1% price buffer to cover fees
            
            if trade.is_short:
                # For SHORT: set stop slightly BELOW entry so if triggered, small profit covers fees
                # e.g., entry=1.988, stop=1.986 -> if price rises to 1.986, exit with 0.1% profit
                be_stop_price = trade.open_rate * (1 - fee_buffer_pct)
            else:
                # For LONG: set stop slightly ABOVE entry so if triggered, small profit covers fees
                # e.g., entry=100, stop=100.1 -> if price drops to 100.1, exit with 0.1% profit
                be_stop_price = trade.open_rate * (1 + fee_buffer_pct)
            
            trade.set_custom_data('be_activated', True)
            trade.set_custom_data('be_stop_price', be_stop_price)
            be_activated = True
            if self.enable_logging.value:
                logger.info(f"BE activated for {pair} at price move {price_movement:.2%}. Stop set at {be_stop_price:.4f} (entry: {trade.open_rate:.4f})")
        
        # If BE is activated, use the STORED stop price
        if be_activated:
            # Retrieve the stored BE stop price (calculated once when BE activated)
            be_stop_price = trade.get_custom_data('be_stop_price', default=trade.open_rate)
            
            # Use Freqtrade's official helper function for absolute price stoploss
            if self.enable_logging.value:
                logger.info(
                    f"BE SL for {pair}: direction={'SHORT' if trade.is_short else 'LONG'}, "
                    f"be_stop_price={be_stop_price:.6f}, open_rate={trade.open_rate:.6f}, "
                    f"current_rate={current_rate:.6f}"
                )
            
            return stoploss_from_absolute(be_stop_price, current_rate, is_short=trade.is_short, leverage=trade.leverage)
        
        return 1  # Use default stoploss
    
    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float,
                              min_stake: float | None, max_stake: float,
                              current_entry_rate: float, current_exit_rate: float,
                              current_entry_profit: float, current_exit_profit: float,
                              **kwargs) -> float | None | tuple[float | None, str | None]:
        """
        Partial Take Profits based on PRICE movement (not leveraged profit).
        
        TP1: Close tp1_amount% when price moves tp1_pct%
        TP2: Close remaining when price moves tp2_pct% (handled by custom_exit)
        """
        # Calculate max price movement based on HIGH/LOW of trade to capture intra-candle touches
        if trade.is_short:
            current_extremum = trade.min_rate if trade.min_rate is not None else current_rate
            price_movement = (trade.open_rate - current_extremum) / trade.open_rate
        else:
            current_extremum = trade.max_rate if trade.max_rate is not None else current_rate
            price_movement = (current_extremum - trade.open_rate) / trade.open_rate
        
        # Check if TP1 was already taken (persistent flag)
        tp1_taken = trade.get_custom_data('tp1_taken', default=False)
        
        # Check for TP1 (only if not already taken)
        if not tp1_taken and price_movement > self.tp1_pct.value:
            # Check if TP1 is enabled via boolean or amount
            if not self.tp1_enabled.value or self.tp1_amount.value == 0:
                return None
            
            # Mark TP1 as taken BEFORE placing the order (prevents recursion)
            trade.set_custom_data('tp1_taken', True)
            
            # Close tp1_amount% of ORIGINAL position
            # Note: trade.amount might already be reduced if previous partial filled
            # So we use stake_amount to calculate original size
            original_amount = (trade.stake_amount * trade.leverage) / trade.open_rate
            close_amount = original_amount * (self.tp1_amount.value / 100.0)
            
            # Make sure we don't try to close more than available
            close_amount = min(close_amount, trade.amount * 0.99)  # Leave 1% buffer for rounding
            
            sell_value = close_amount * current_rate
            
            # FIX: adjust_trade_position expects change in STAKE (margin), not notional value.
            # Must divide by leverage to get the margin amount to remove.
            stake_change = sell_value / trade.leverage
            
            logger.info(
                f"TP1 for {trade.pair} ({'SHORT' if trade.is_short else 'LONG'}): "
                f"price moved {price_movement:.2%}, closing {self.tp1_amount.value:.0f}% "
                f"(Notional: {sell_value:.2f}, Margin: {stake_change:.2f})"
            )
            return (-stake_change, "TP1_partial")
        
        return None
    
