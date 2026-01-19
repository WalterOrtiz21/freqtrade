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
from datetime import datetime
from pandas import DataFrame
from typing import Optional, Dict, List
import pickle
import os

logger = logging.getLogger(__name__)

from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter, BooleanParameter
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
    startup_candle_count: int = 200
    can_short = True

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
    require_ob_zone = BooleanParameter(default=False, space='buy', optimize=True)
    require_fvg_zone = BooleanParameter(default=False, space='buy', optimize=True)
    require_premium_discount = BooleanParameter(default=False, space='buy', optimize=True)
    
    # Trend filter - disabled by default to match Pine Strategy execution logic
    # (Pine indicator shows trend color, but strategy entry block does not enforce it)
    trade_with_trend = BooleanParameter(default=False, space='buy', optimize=True)
    
    # Lookback for recent signals - Pine uses immediate alerts (current bar)
    # kept slightly > 1 to avoid missing signals due to candle closing timing
    signal_lookback = IntParameter(1, 3, default=1, space='buy', optimize=True)

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
    # Pine Strategy implicitly exits/reverses on opposite CHoCH.
    exit_on_opposite_choch = BooleanParameter(default=True, space='sell', optimize=True)
    
    # TP2 (Full Exit) - Optional extention
    # Pine does not have explicit TP2, it holds until reversal or manual close.
    # We set a high default or rely on reversal.
    tp2_pct = DecimalParameter(0.02, 1.00, default=0.05, decimals=3, space='sell', optimize=True)

    # ==========================================================================
    # INITIALIZATION
    # ==========================================================================
    
    def __init__(self, config: dict) -> None:
        super().__init__(config)
    
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

    def informative_pairs(self):
        """
        Define pairs to load. We need HTF (4h) for trend context.
        """
        pairs = self.dp.current_whitelist()
        # Ensure we have 4h and 1h candles
        informative_pairs = [(pair, '1h') for pair in pairs]
        informative_pairs += [(pair, '4h') for pair in pairs]
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
            
        # --- 2. HTF (4h) Calculations ---
        # Get Informative 4h Pair
        if self.dp:
            inf_4h = self.dp.get_pair_dataframe(metadata['pair'], '4h')
            # Calculate SMC on 4h
            smc_4h = SMCLuxAlgo(
                inf_4h,
                internal_length=self.internal_length.value, # Use same params or different? Usually HTF is same logic on larger bars.
                swing_length=self.swing_length.value,
            )
            signals_4h = smc_4h.get_signals()
            
            # We only need the Trend and maybe major zones
            # Rename columns to avoid collision
            inf_4h['4h_swing_trend'] = signals_4h['swing_trend']
            inf_4h['4h_internal_trend'] = signals_4h['internal_trend']
            
            # Merge 4h Trend into 15m dataframe (FFILL to propagate latest 4h state)
            # Use merge_informative_pair or simple merge_asof?
            # Freqtrade standard is using dataframe.merge relying on 'date'
            
            # Prepare for merge
            inf_4h = inf_4h[['date', '4h_swing_trend', '4h_internal_trend']].copy()
            
            # Merge
            dataframe = pd.merge(dataframe, inf_4h, on='date', how='left')
            dataframe['4h_swing_trend'] = dataframe['4h_swing_trend'].ffill()
            dataframe['4h_internal_trend'] = dataframe['4h_internal_trend'].ffill()
            
        # Add ML Features (Context & Technicals)
        dataframe = self._add_ml_features(dataframe)
        
        # Log summary
        if self.dp: # Only log if not backtesting or sparse
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
        
        lookback = self.signal_lookback.value
        
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
        
        # P/D Logic:
        # Discount: Close < Equilibrium (<0.5) - Good for LONG
        # Premium: Close > Equilibrium (>0.5) - Good for SHORT
        in_discount = dataframe['close'] < dataframe['equilibrium']
        in_premium = dataframe['close'] > dataframe['equilibrium']
        
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
        
        if self.trade_with_trend.value:
            # Check if we have 4h data
            if '4h_swing_trend' in dataframe.columns:
                # Long: HTF is Bullish
                htf_bull = dataframe['4h_swing_trend'] == 1
                # Short: HTF is Bearish
                htf_bear = dataframe['4h_swing_trend'] == -1
                
                bullish_trend_ok = htf_bull
                bearish_trend_ok = htf_bear
            else:
                # Fallback to local trend if 4h missing (backtest safety)
                bullish_trend_ok = dataframe['swing_trend'] == 1
                bearish_trend_ok = dataframe['swing_trend'] == -1
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
                long_condition &= ml_long_ok
                short_condition &= ml_short_ok
                
                # Log only if a signal was blocked significantly (optional)
                if (long_condition.sum() < ml_long_ok.sum()) or (short_condition.sum() < ml_short_ok.sum()):
                     logger.info(f"🤖 ML Filter ({metadata['pair']}) Blocked Signals. Threshold: {self.ml_threshold.value}")

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
        
        if self.exit_on_opposite_choch.value:
            # Exit long on bearish CHoCH (trend reversal)
            exit_long = (
                (dataframe['internal_choch_bearish'] == 1) |
                (dataframe['swing_choch_bearish'] == 1)
            )
            
            # Exit short on bullish CHoCH
            exit_short = (
                (dataframe['internal_choch_bullish'] == 1) |
                (dataframe['swing_choch_bullish'] == 1)
            )
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
    
    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Break Even Logic with persistent state.
        Once BE is activated (when price reaches TP1), the stoploss is fixed at entry price + small buffer for fees.
        """
        if not self.move_be_at_tp1.value:
            return 1  # Use default stoploss
        
        # Check if BE was already activated (persistent across ticks)
        be_activated = trade.get_custom_data('be_activated', default=False)
        
        # Calculate price movement to check if we should activate BE
        if trade.is_short:
            current_extremum = trade.min_rate if trade.min_rate is not None else current_rate
            price_movement = (trade.open_rate - current_extremum) / trade.open_rate
        else:
            current_extremum = trade.max_rate if trade.max_rate is not None else current_rate
            price_movement = (current_extremum - trade.open_rate) / trade.open_rate
        
        # Activate BE if price moved past TP1 and not already activated
        if not be_activated and price_movement >= self.tp1_pct.value:
            trade.set_custom_data('be_activated', True)
            be_activated = True
            logger.info(f"BE activated for {pair} at price move {price_movement:.2%}")
        
        # If BE is activated, return fixed stoploss at entry + small buffer
        if be_activated:
            # Add 0.1% buffer to cover fees (in price terms)
            fee_buffer_pct = 0.001  # 0.1% price buffer
            
            if trade.is_short:
                # For short, stop is ABOVE entry. We want stop at entry - buffer (price below entry = small profit)
                target_stop = trade.open_rate * (1 - fee_buffer_pct)
                # Stoploss relative to current rate
                sl_relative = (target_stop - current_rate) / current_rate
            else:
                # For long, stop is BELOW entry. We want stop at entry + buffer (price above entry = small profit)
                target_stop = trade.open_rate * (1 + fee_buffer_pct)
                # Stoploss relative to current rate
                sl_relative = (target_stop - current_rate) / current_rate
            
            return sl_relative
        
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
        
        # Check for TP1
        if price_movement > self.tp1_pct.value:
            # Check if TP1 is enabled via boolean or amount
            if not self.tp1_enabled.value or self.tp1_amount.value == 0:
                return None
            
            # For LONG: closing partial = sell order
            # For SHORT: closing partial = buy order (re-buy to reduce short position)
            exit_side = 'buy' if trade.is_short else 'sell'
            
            # Count how many partial exits we've done (excluding entry orders)
            partial_exits = [
                o for o in trade.orders 
                if o.side == exit_side 
                and o.status == 'closed'
                and o.ft_order_side == 'exit'  # Only exit orders, not entries
            ]
            
            if len(partial_exits) == 0:
                # Close tp1_amount% of position
                close_amount = trade.amount * (self.tp1_amount.value / 100.0)
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
