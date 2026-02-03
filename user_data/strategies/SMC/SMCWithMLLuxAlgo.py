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
    require_premium_discount = BooleanParameter(default=False, space='buy', optimize=True)
    premium_discount_threshold = DecimalParameter(0.05, 0.5, default=0.5, space='buy', optimize=True)
    
    # Trend filter - disabled by default to match Pine Strategy execution logic
    # (Pine indicator shows trend color, but strategy entry block does not enforce it)
    trade_with_trend = BooleanParameter(default=False, space='buy', optimize=True)
    
    # Lookback for recent signals
    # 1 = Instant Entry (Signal + Zone on same candle)
    # > 1 = Retest Logic (Signal happened X bars ago, entering now on Zone or Pullback)
    entry_signal_lookback = IntParameter(1, 24, default=1, space='buy', optimize=True)
    
    # NEW: Toggle for HTF (4h) Filter
    use_htf_filter = BooleanParameter(default=True, space='buy', optimize=True)
    
    # NEW: Dynamic Stoploss Toggle & Offset
    use_dynamic_stoploss = BooleanParameter(default=True, space='sell', optimize=True)
    sl_buffer_pct = DecimalParameter(0.001, 0.01, default=0.002, decimals=3, space='sell', optimize=True)
    
    # Logging Control
    enable_logging = BooleanParameter(default=True, space='custom', optimize=False)

    # ==========================================================================
    # ML PARAMETERS (Placeholder)
    # ==========================================================================
    # Note: ML logic is currently disabled in entry generation to ensure 
    # strict fidelity to Pine Script logic.
    
    use_ml_filter = BooleanParameter(default=True, space='buy', optimize=False)
    use_per_symbol_models = BooleanParameter(default=True, space='buy', optimize=False)  # True = per-symbol, False = general
    ml_threshold = DecimalParameter(0.05, 0.50, default=0.15, decimals=2, space='buy', optimize=True)
    ml_model_path = "user_data/strategies/SMC/models"
    enable_auto_training = BooleanParameter(default=True, space='buy', optimize=False)
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
    
    # Final Exit (Reversal)
    # Granular control over which CHoCH triggers an exit
    exit_on_internal_choch = BooleanParameter(default=True, space='sell', optimize=True)
    exit_on_swing_choch = BooleanParameter(default=True, space='sell', optimize=True)
    
    # TP2 (Full Exit) - Optional extention
    # Pine does not have explicit TP2, it holds until reversal or manual close.
    # We set a high default or rely on reversal.
    tp2_pct = DecimalParameter(0.02, 1.00, default=0.05, decimals=3, space='sell', optimize=True)

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
        
        # Premium/Discount Factor - NEW
        if 'swing_high' in df.columns and 'swing_low' in df.columns:
            swing_range = df['swing_high'] - df['swing_low']
            swing_range = swing_range.replace(0, np.nan)
            df['ml_pd_factor'] = (df['close'] - df['swing_low']) / swing_range
            df['ml_pd_factor'] = df['ml_pd_factor'].fillna(0.5).clip(0, 1)
        else:
             df['ml_pd_factor'] = 0.5

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
            internal_choch_bull = internal_choch_bull.rolling(lookback).max() == 1
            swing_choch_bull = swing_choch_bull.rolling(lookback).max() == 1
            
        # Determine BOS signals (Continuations) - validated by require_choch=False
        internal_bos_bull = dataframe['internal_bos_bullish'] == 1
        swing_bos_bull = dataframe['swing_bos_bullish'] == 1
        
        if lookback > 1:
            internal_bos_bull = internal_bos_bull.rolling(lookback).max() == 1
            swing_bos_bull = swing_bos_bull.rolling(lookback).max() == 1

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
            internal_choch_bear = internal_choch_bear.rolling(lookback).max() == 1
            swing_choch_bear = swing_choch_bear.rolling(lookback).max() == 1
            
        internal_bos_bear = dataframe['internal_bos_bearish'] == 1
        swing_bos_bear = dataframe['swing_bos_bearish'] == 1
        
        if lookback > 1:
            internal_bos_bear = internal_bos_bear.rolling(lookback).max() == 1
            swing_bos_bear = swing_bos_bear.rolling(lookback).max() == 1

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
        # Price in bullish OB zone
        in_bullish_ob = (
            (dataframe['active_bullish_ob_top'] > 0) &
            (dataframe['low'] <= dataframe['active_bullish_ob_top']) &
            (dataframe['high'] >= dataframe['active_bullish_ob_bottom'])
        )
        
        # Price in bearish OB zone
        in_bearish_ob = (
            (dataframe['active_bearish_ob_top'] > 0) &
            (dataframe['high'] >= dataframe['active_bearish_ob_bottom']) &
            (dataframe['low'] <= dataframe['active_bearish_ob_top'])
        )
        
        # Price in bullish FVG zone
        in_bullish_fvg = (
            (dataframe['active_bullish_fvg_top'] > 0) &
            (dataframe['low'] <= dataframe['active_bullish_fvg_top'])
        )
        
        # Price in bearish FVG zone
        in_bearish_fvg = (
            (dataframe['active_bearish_fvg_top'] > 0) &
            (dataframe['high'] >= dataframe['active_bearish_fvg_bottom'])
        )
        
        # Zone requirements - OR logic (any enabled filter can pass)
        # If NO zone filter is enabled, all bars pass (default True)
        any_zone_enabled = self.require_ob_zone.value or self.require_fvg_zone.value or self.require_premium_discount.value
        
        # P/D Logic with Threshold:
        # Discount: Price in bottom X% of range (e.g., < 0.05 or < 0.5)
        # Premium: Price in top X% of range
        
        swing_range = dataframe['swing_high'] - dataframe['swing_low']
        discount_limit = dataframe['swing_low'] + (swing_range * self.premium_discount_threshold.value)
        premium_limit = dataframe['swing_high'] - (swing_range * self.premium_discount_threshold.value)
        
        in_discount = dataframe['close'] < discount_limit
        in_premium = dataframe['close'] > premium_limit
        
        if any_zone_enabled:
            # Start with False, use OR to add conditions
            bullish_zone = pd.Series(False, index=dataframe.index)
            bearish_zone = pd.Series(False, index=dataframe.index)
            
            if self.require_ob_zone.value:
                bullish_zone |= in_bullish_ob
                bearish_zone |= in_bearish_ob
                
                # Add Breakers to OB logic (User description: Breaker OB is a key entry point)
                if 'active_bullish_breaker_top' in dataframe.columns:
                     in_bull_breaker = (
                        (dataframe['active_bullish_breaker_top'] > 0) & 
                        # Price touching breaker (Breaker acts as Support)
                        (dataframe['low'] <= dataframe['active_bullish_breaker_top']) & 
                        (dataframe['high'] >= dataframe['active_bullish_breaker_bottom'])
                     )
                     bullish_zone |= in_bull_breaker
                     
                     in_bear_breaker = (
                        (dataframe['active_bearish_breaker_top'] > 0) & 
                        # Price touching breaker (Breaker acts as Resistance)
                        (dataframe['high'] >= dataframe['active_bearish_breaker_bottom']) &
                        (dataframe['low'] <= dataframe['active_bearish_breaker_top'])
                     )
                     bearish_zone |= in_bear_breaker
            
            if self.require_fvg_zone.value:
                bullish_zone |= in_bullish_fvg
                bearish_zone |= in_bearish_fvg
                
                # Add FVG Breakers logic
                if 'active_bullish_fvg_breaker_top' in dataframe.columns:
                     in_bull_fvg_breaker = (
                        (dataframe['active_bullish_fvg_breaker_top'] > 0) & 
                        (dataframe['low'] <= dataframe['active_bullish_fvg_breaker_top']) & 
                        (dataframe['high'] >= dataframe['active_bullish_fvg_breaker_bottom'])
                     )
                     bullish_zone |= in_bull_fvg_breaker
                     
                     in_bear_fvg_breaker = (
                        (dataframe['active_bearish_fvg_breaker_top'] > 0) & 
                        (dataframe['high'] >= dataframe['active_bearish_fvg_breaker_bottom']) &
                        (dataframe['low'] <= dataframe['active_bearish_fvg_breaker_top'])
                     )
                     bearish_zone |= in_bear_fvg_breaker

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
            
            if self.require_premium_discount.value:
                bullish_zone |= in_discount
                bearish_zone |= in_premium
                
        else:
            # No zone filters enabled - all pass
            bullish_zone = pd.Series(True, index=dataframe.index)
            bearish_zone = pd.Series(True, index=dataframe.index)
        
        # Debugging Zone Counts (Only if backtesting/dry)
        if self.dp: 
            n_bull_zones = bullish_zone.sum()
            n_bear_zones = bearish_zone.sum()
            if (n_bull_zones == 0 or n_bear_zones == 0) and (self.require_ob_zone.value or self.require_fvg_zone.value or self.require_premium_discount.value):
                logger.warning(
                    f"⚠️ ZONES EMPTY! Bull: {n_bull_zones}, Bear: {n_bear_zones}. "
                    f"OB Req: {self.require_ob_zone.value}, FVG Req: {self.require_fvg_zone.value}, PD Req: {self.require_premium_discount.value}"
                )
                logger.warning(f"   Active Bull OB Candles: {(dataframe['active_bullish_ob_top'] > 0).sum()}")
                logger.warning(f"   Active Bull FVG Candles: {(dataframe['active_bullish_fvg_top'] > 0).sum()}")
        
        # ===== TREND FILTER (Neptune Logic: HTF Alignment) =====
        # HTF (4h) Filter:
        # Long only if 4h is Bullish (1) or Neutral/Internal-Bullish
        # Strict mode: 4h Swing Trend must be 1.
        
        # ===== TREND FILTER (Neptune Logic: HTF Alignment) =====
        # HTF (4h) Filter:
        # Long only if 4h is Bullish (1) or Neutral/Internal-Bullish
        # Strict mode: 4h Swing Trend must be 1.
        
        if self.trade_with_trend.value:
            # Local Trend Filter (MTF)
            bullish_trend_ok = dataframe['swing_trend'] == 1
            bearish_trend_ok = dataframe['swing_trend'] == -1
            
            # HTF Filter - Check ALL configured timeframes
            if self.use_htf_filter.value:
                htf_list = self._get_active_htf_list()
                for htf in htf_list:
                    swing_col = f'{htf}_swing_trend'
                    if swing_col in dataframe.columns:
                        htf_bull = dataframe[swing_col] == 1
                        htf_bear = dataframe[swing_col] == -1
                        
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
    
    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                           time_in_force: str, current_time: datetime, entry_tag: str | None,
                           side: str, **kwargs) -> bool:
        """
        Prevent entry if:
        1. Already have an open trade for this pair
        2. Recently closed a trade for this pair (cooldown based on entry_signal_lookback)
        
        This ensures ONE TRADE PER SYMBOL at a time and prevents re-entry
        on the same signal after SL hit.
        """
        # 1. Check for existing open trades on this pair
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
            if trade.close_date and trade.close_date >= cooldown_start:
                minutes_since_close = (current_time - trade.close_date).total_seconds() / 60
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
        if self.move_be_at_tp1.value and not be_activated and price_movement >= self.tp1_pct.value:
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
    
    def custom_exit(self, pair: str, trade: Trade, current_time: datetime,
                   current_rate: float, current_profit: float, **kwargs) -> Optional[str]:
        """
        Custom exit for TP2 based on PRICE movement.
        """
        # Calculate price movement (remove leverage)
        price_movement = current_profit / trade.leverage
        
        # TP2: Full exit
        if price_movement >= self.tp2_pct.value:
            return "TP2_exit"
        
        return None
