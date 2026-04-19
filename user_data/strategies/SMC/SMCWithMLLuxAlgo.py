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
from datetime import datetime, timedelta, timezone
from pathlib import Path
from pandas import DataFrame
from typing import Optional, Dict, List
import pickle
import os

logger = logging.getLogger(__name__)

from freqtrade.strategy import (
    IStrategy,
    IntParameter,
    DecimalParameter,
    BooleanParameter,
    CategoricalParameter,
    stoploss_from_absolute,
)
from freqtrade.persistence import Trade
import talib.abstract as ta

# Import SMC engine (Numba-optimized: CHoCH, BOS, OBs, FVGs, Sweeps)
# Module copied to user_data/strategies/ for Hyperopt compatibility
try:
    from smc_engine import SMCEngine as SMCEngine
except ImportError:
    # Fallback: try local import
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from smc_engine import SMCEngine as SMCEngine

# LLM Confluence Filter (OpenRouter API)
try:
    from llm_filter import LLMConfluenceFilter
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from llm_filter import LLMConfluenceFilter

# train_model is imported lazily inside bot_start only when auto-training is enabled.


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
    timeframe = "15m"

    trailing_stop = False
    trailing_stop_positive = 0.005  # 0.5% price movement
    trailing_stop_positive_offset = 0.01  # 1% price movement
    trailing_only_offset_is_reached = True

    use_exit_signal = True
    use_custom_stoploss = True  # Enable for breakeven
    position_adjustment_enable = True  # Enable for partial TPs
    process_only_new_candles = True  # Avoid reprocessing closed candles on every tick
    max_open_trades = 5
    startup_candle_count: int = 400  # EMA(200) + ATR(200) + swing_length up to 100 (×2 for confirmation)
    can_short = True

    # ==========================================================================
    # HTF TIMEFRAMES (Flexible selection - works with JSON buy params)
    # ==========================================================================
    # Set to 'none' to disable that slot
    # Common options: '5m', '15m', '30m', '1h', '2h', '4h', '8h', '12h', '1d'
    HTF_OPTIONS = ["none", "5m", "15m", "30m", "1h", "2h", "4h", "8h", "12h", "1d"]

    htf_1 = CategoricalParameter(HTF_OPTIONS, default="1h", space="buy", optimize=False)
    htf_2 = CategoricalParameter(HTF_OPTIONS, default="4h", space="buy", optimize=False)

    # ==========================================================================
    # SMC PARAMETERS (Optimizable)
    # ==========================================================================

    # Structure Mode: "Internal" or "Swing" (Pine 'stratStructureType')
    # Pine Default: "Swing".
    # To match Pine Default: set use_swing_signals=True, use_internal_signals=False.

    # Internal structure length (reactive)
    internal_length = IntParameter(3, 8, default=5, space="buy", optimize=True)

    # Swing structure length (stable)
    swing_length = IntParameter(20, 60, default=50, space="buy", optimize=True)

    # Use internal or swing signals for entry
    # Note: Pine Script uses a dropdown XOR selection. Here we allow both via categorical.
    entry_signal_type = CategoricalParameter(
        ["swing_only", "internal_only", "both"],
        default="swing_only",
        space="buy",
        optimize=True,
    )

    # Only trade CHoCH (reversals) or also BOS (continuations)?
    # Pine Strategy ONLY trades CHoCH for entries. Default to True to match.
    require_choch = BooleanParameter(default=True, space="buy", optimize=False)

    # Zone requirements - combined to avoid flat spaces in Hyperopt
    required_zone = CategoricalParameter(
        ["ob_only", "fvg_only", "any"],
        default="any",
        space="buy",
        optimize=True,
    )

    # NEW: Volumetric Order Blocks (BigBeluga Style)
    require_volumetric_ob = BooleanParameter(default=True, space="buy", optimize=False)
    vol_sma_period = IntParameter(10, 30, default=20, space="buy", optimize=True)
    ob_rvol_threshold = DecimalParameter(
        1.2, 2.5, default=1.5, decimals=1, space="buy", optimize=True
    )

    # Trend filter - disabled by default to match Pine Strategy execution logic
    # (Pine indicator shows trend color, but strategy entry block does not enforce it)
    trade_with_trend = BooleanParameter(default=True, space="buy", optimize=False)

    # Lookback for recent signals
    # 1 = Instant Entry (Signal + Zone on same candle)
    # > 1 = Retest Logic (Signal happened X bars ago, entering now on Zone or Pullback)
    entry_signal_lookback = IntParameter(1, 6, default=2, space="buy", optimize=True)

    # Per-HTF: Use internal_trend (faster, reacts to CHoCH) vs swing_trend (slower, more reliable)
    htf_1_use_internal = BooleanParameter(default=False, space="buy", optimize=False)
    htf_2_use_internal = BooleanParameter(default=False, space="buy", optimize=False)

    # NEW: Dynamic Stoploss Toggle & Offset
    use_dynamic_stoploss = BooleanParameter(default=True, space="sell", optimize=False)
    sl_buffer_pct = DecimalParameter(
        0.001, 0.008, default=0.003, decimals=3, space="sell", optimize=True
    )

    # NEW: FVG ATR Filter (BigBeluga Style)
    # Filter out based on volatility (ATR) rather than percentage.
    # Threshold = ATR(200) * fvg_atr_threshold.
    fvg_atr_threshold = DecimalParameter(
        0.05, 0.25, default=0.10, decimals=2, space="buy", optimize=True
    )

    # NEW: Conservative Entry Mode
    # If False, ignores signals on the current candle (0 bars ago), forcing a wait.
    allow_immediate_entry = BooleanParameter(default=False, space="buy", optimize=False)

    # Liquidity Sweep: If detected + CHoCH, bypass zone requirement
    use_sweep_bypass = BooleanParameter(default=False, space="buy", optimize=False)
    sweep_lookback = IntParameter(1, 10, default=3, space="buy", optimize=False)

    # Dynamic TP1 / BE adjustment based on opposing zone proximity.
    # If a strong OB/FVG sits between entry and default TP1, reduce TP1 to just below
    # that zone and trigger BE there. Allows entering the trade but adapts the exit
    # to the actual structure instead of a fixed percentage.
    use_dynamic_tp_adj = BooleanParameter(default=True, space="sell", optimize=False)
    tp_adj_min_rvol = DecimalParameter(
        1.2, 2.5, default=1.5, decimals=1, space="sell", optimize=True
    )
    tp_adj_margin = DecimalParameter(
        0.50, 0.90, default=0.80, decimals=2, space="sell", optimize=True
    )  # Take profit at X% of zone distance

    # Logging Control
    enable_logging = BooleanParameter(default=True, space="custom", optimize=False)

    # Circuit Breaker (Panic Mode) - Blocks trades during high volatility
    circuit_breaker_enabled = BooleanParameter(default=True, space="custom", optimize=False)
    circuit_breaker_window = IntParameter(
        15, 120, default=30, space="custom", optimize=False
    )  # Minutes
    circuit_breaker_limit = IntParameter(
        2, 10, default=3, space="custom", optimize=False
    )  # Max SL hits

    # ── Entry Filters (context-based, driven by backtest analysis) ────────────
    # All default=True to apply the filters found in analysis.
    # Set to False in JSON to disable individually.
    filter_block_kz_ldn             = BooleanParameter(default=False, space="custom", optimize=False)
    filter_block_off_hours          = BooleanParameter(default=False, space="custom", optimize=False)
    filter_block_sweep              = BooleanParameter(default=False, space="custom", optimize=False)
    filter_block_fvg_stale_htf_part = BooleanParameter(default=False, space="custom", optimize=False)
    filter_friday_reduce_stake      = BooleanParameter(default=False, space="custom", optimize=False)
    filter_friday_stake_ratio       = DecimalParameter(
        0.1, 1.0, default=0.5, decimals=1, space="custom", optimize=False
    )  # Fraction of normal stake on Fridays

    # ── Indicator Export (saves parquet per pair under analisis/) ─────────────
    export_indicator_data = BooleanParameter(default=False, space="custom", optimize=False)

    # ==========================================================================
    # ML PARAMETERS (Placeholder)
    # ==========================================================================
    # Note: ML logic is currently disabled in entry generation to ensure
    # strict fidelity to Pine Script logic.

    use_ml_filter = BooleanParameter(default=False, space="buy", optimize=False)
    use_per_symbol_models = BooleanParameter(
        default=False, space="buy", optimize=False
    )  # True = per-symbol, False = general
    ml_threshold = DecimalParameter(
        0.05, 0.50, default=0.15, decimals=2, space="buy", optimize=False
    )
    ml_model_path = "user_data/strategies/SMC/models"
    enable_auto_training = BooleanParameter(default=False, space="buy", optimize=False)
    _ml_model = None

    # ==========================================================================
    # LLM CONFLUENCE FILTER PARAMETERS
    # ==========================================================================
    use_llm_filter = BooleanParameter(default=False, space="buy", optimize=False)
    use_llm_shadow = BooleanParameter(default=False, space="buy", optimize=False)
    llm_confidence_threshold = DecimalParameter(
        0.3, 0.8, default=0.6, decimals=2, space="buy", optimize=False
    )
    llm_model_name = CategoricalParameter(
        ["z-ai/glm-5-turbo",
         "openai/gpt-oss-20b:free",
         "openai/gpt-oss-120b:free",
         "nvidia/nemotron-3-super-120b-a12b:free"],
        default="z-ai/glm-5-turbo",
        space="buy",
        optimize=False,
    )

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
    tp1_pct = DecimalParameter(0.005, 0.05, default=0.015, decimals=3, space="sell", optimize=True)

    # TP1 Enabled: Bool to turn on/off the partial exit
    tp1_enabled = BooleanParameter(default=True, space="sell", optimize=False)

    # TP1 Amount: % of position to close
    # Pine Default: 50%. Allow 0.0 to disable partials.
    tp1_amount = CategoricalParameter([25, 33, 50, 67, 75], default=50, space="sell", optimize=True)

    # Break Even (Pine 'stratMoveBE')
    # Pine Default: False
    move_be_at_tp1 = BooleanParameter(default=True, space="sell", optimize=False)

    # Custom Break Even Target (Optional)
    # If > 0, BE is activated cuando el precio se mueve este %, ignorando el tp1_pct.
    be_trigger_pct = DecimalParameter(
        0.0, 0.04, default=0.0, decimals=3, space="sell", optimize=True
    )

    # Final Exit (Reversal)
    # Granular control over which CHoCH triggers an exit
    exit_trend_type = CategoricalParameter(
        ["swing", "internal", "both"],
        default="both",
        space="sell",
        optimize=True,
    )

    # HTF Zone Requirement
    # enabled: price must be inside an active HTF OB/FVG/Breaker to enter
    # disabled: HTF acts as direction filter only (legacy behavior)
    require_htf_zone = CategoricalParameter(
        ["enabled", "disabled"],
        default="enabled",
        space="buy",
        optimize=False,
    )

    # ==========================================================================
    # MACRO REGIME FILTER
    # ==========================================================================
    # Uses BTC Weekly EMA(21) as macro bias arbiter.
    # Bull bias  (BTC > EMA21 * 1.01): longs allowed, shorts blocked
    # Bear bias  (BTC < EMA21 * 0.99): shorts allowed, longs blocked
    # Neutral    (within ±1% of EMA21): both directions allowed
    # BTC/USDT:USDT is always loaded as informative regardless of whitelist.

    use_macro_filter = CategoricalParameter(
        ["enabled", "disabled"],
        default="enabled",
        space="buy",
        optimize=False,
    )

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
        logger.info(
            f"SMCWithMLLuxAlgo starting. use_ml_filter={self.use_ml_filter.value}, "
            f"use_llm_filter={self.use_llm_filter.value}, "
            f"auto_train={self.enable_auto_training.value}"
        )

        # 0. Auto-Train Model (Only if explicitly enabled)
        if self.use_ml_filter.value and self.enable_auto_training.value:
            try:
                from user_data.strategies.SMC.train_smc_model import train_model
                logger.info("Starting Auto-Training of ML Model...")
                success = train_model()
                if success:
                    logger.info("Auto-Training Complete.")
                else:
                    logger.error("Auto-Training Failed. Using existing model if available.")
            except ImportError:
                logger.error("Could not import train_smc_model. Auto-training disabled.")

        # Load ML model here (after params are loaded)
        self._load_ml_model()

        # LLM filter init is deferred to _init_llm_filter() (lazy init)
        # because freqtrade loads JSON params AFTER bot_start()
        self._llm_filter = None
        self._llm_initialized = False

        # 1. Get Leverage
        config_leverage = self.config.get("leverage", 1.0)

        # 2. Get Raw Values from Config (assumed to be Price Distance)
        raw_stoploss = self.config.get("stoploss", -0.05)
        raw_trailing_pos = self.config.get("trailing_stop_positive", 0.005)
        raw_trailing_offset = self.config.get("trailing_stop_positive_offset", 0.01)

        # 3. Apply Leverage to Calculate PnL thresholds
        self.stoploss = raw_stoploss * config_leverage

        # Adjust trailing settings if they are set in config
        if self.config.get("trailing_stop", False):
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
        if self.htf_1.value != "none":
            htf_list.append(self.htf_1.value)
        if self.htf_2.value != "none":
            htf_list.append(self.htf_2.value)
        return htf_list

    def informative_pairs(self):
        """
        Define pairs to load based on htf_1 and htf_2 parameters.
        BTC/USDT:USDT 1D and 4H are always loaded for the macro regime filter.
        """
        pairs = self.dp.current_whitelist()
        htf_list = self._get_active_htf_list()

        informative_pairs = []
        for tf in htf_list:
            informative_pairs += [(pair, tf) for pair in pairs]

        # BTC macro reference — always loaded for all pairs regardless of whitelist
        informative_pairs += [
            ("BTC/USDT:USDT", "1d"),
            ("BTC/USDT:USDT", "4h"),
        ]

        logger.info(f"Informative pairs configured for timeframes: {htf_list}")
        return informative_pairs

    def _load_ml_model(self):
        """Load pre-trained ML model if exists.

        Tries to load in order:
        1. Per-symbol model: smc_xgboost_{PAIR}.pkl
        2. General model: smc_xgboost_model.pkl
        """
        if not hasattr(self, "_models_cache"):
            self._models_cache = {}

        # General model (fallback) - load once
        general_model_file = os.path.join(self.ml_model_path, "smc_xgboost_model.pkl")
        if os.path.exists(general_model_file) and "general" not in self._models_cache:
            if self.use_ml_filter.value:
                try:
                    with open(general_model_file, "rb") as f:
                        self._models_cache["general"] = pickle.load(f)
                    logger.info(f"✅ General ML Model loaded from {general_model_file}")
                except Exception as e:
                    logger.error(f"❌ Failed to load general ML model: {e}")
                    self._models_cache["general"] = None

        # Load optimal threshold — only needed when ML filter is active
        self._ml_optimal_threshold = None
        if self.use_ml_filter.value:
            import json as _json
            threshold_file = os.path.join(self.ml_model_path, "smc_xgboost_threshold.json")
            if os.path.exists(threshold_file):
                try:
                    with open(threshold_file, "r") as f:
                        tdata = _json.load(f)
                    self._ml_optimal_threshold = tdata.get("threshold", None)
                    logger.info(
                        f"ML Threshold loaded: {self._ml_optimal_threshold:.4f} "
                        f"(F1={tdata.get('f1', 0):.4f}, AUC={tdata.get('auc', 0):.4f})"
                    )
                except Exception as e:
                    logger.warning(f"Could not load ML threshold file: {e}")

        # Set default model to general if exists
        if "general" in self._models_cache:
            self._ml_model = self._models_cache["general"]
        else:
            self._ml_model = None

    def _get_ml_model_for_pair(self, pair: str):
        """Get ML model for specific pair (with caching and fallback).

        If use_per_symbol_models=True: Try per-symbol model, fallback to general
        If use_per_symbol_models=False: Only use general model
        """
        if not self.use_ml_filter.value:
            return None

        if not hasattr(self, "_models_cache"):
            self._models_cache = {}

        # Convert pair to filename: 'BTC/USDT:USDT' -> 'BTC_USDT'
        pair_key = pair.replace("/", "_").replace(":", "_").split("_USDT")[0] + "_USDT"

        # Check cache first
        if pair_key in self._models_cache:
            return self._models_cache[pair_key]

        # If per-symbol models enabled, try to load specific model
        if self.use_per_symbol_models.value:
            symbol_model_file = os.path.join(self.ml_model_path, f"smc_xgboost_{pair_key}.pkl")
            if os.path.exists(symbol_model_file):
                try:
                    with open(symbol_model_file, "rb") as f:
                        model = pickle.load(f)
                    self._models_cache[pair_key] = model
                    logger.info(f"✅ Per-symbol ML Model loaded for {pair}: {symbol_model_file}")
                    return model
                except Exception as e:
                    logger.warning(f"Failed to load per-symbol model for {pair}: {e}")

        # Fallback to general model (or use it directly if per_symbol disabled)
        if "general" not in self._models_cache:
            general_model_file = os.path.join(self.ml_model_path, "smc_xgboost_model.pkl")
            if os.path.exists(general_model_file):
                try:
                    with open(general_model_file, "rb") as f:
                        self._models_cache["general"] = pickle.load(f)
                    logger.info(f"✅ General ML Model loaded: {general_model_file}")
                except Exception:
                    self._models_cache["general"] = None
            else:
                self._models_cache["general"] = None

        # Cache that this pair uses general model
        self._models_cache[pair_key] = self._models_cache.get("general")
        return self._models_cache[pair_key]

    # ==========================================================================
    # INDICATOR CALCULATION
    # ==========================================================================

    def _init_llm_filter(self) -> None:
        """Lazy init of LLM filter — called on first populate_indicators.

        freqtrade loads JSON params AFTER bot_start(), so we can't read
        use_llm_shadow / use_llm_filter there. This runs once, when params
        are already loaded.
        """
        if self._llm_initialized:
            return
        self._llm_initialized = True

        if not (self.use_llm_filter.value or self.use_llm_shadow.value):
            return

        if self.dp and self.dp.runmode.value in ("backtest", "hyperopt"):
            logger.info("LLM filter disabled for backtest/hyperopt mode")
            return

        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            logger.warning(
                "use_llm_filter/shadow=True but OPENROUTER_API_KEY not set. "
                "LLM filter disabled."
            )
            return

        shadow = self.use_llm_shadow.value and not self.use_llm_filter.value
        log_dir = ""
        if shadow:
            log_dir = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..", "logs", "llm_shadow")
            )
            try:
                os.makedirs(log_dir, exist_ok=True)
                marker = os.path.join(log_dir, "_init.log")
                with open(marker, "a") as f:
                    f.write(
                        f"{datetime.now(timezone.utc).isoformat()} "
                        f"init model={self.llm_model_name.value} "
                        f"threshold={self.llm_confidence_threshold.value:.2f} "
                        f"runmode={self.dp.runmode.value if self.dp else 'unknown'}\n"
                    )
            except Exception as e:
                logger.warning(f"LLM shadow log dir not writable ({log_dir}): {e}")
        try:
            self._llm_filter = LLMConfluenceFilter(
                api_key=api_key,
                model=self.llm_model_name.value,
                confidence_threshold=self.llm_confidence_threshold.value,
                timeout=25,
                max_retries=2,
                cache_ttl=900,
                shadow_mode=shadow,
                log_dir=log_dir,
            )
            mode_str = "SHADOW (log only)" if shadow else "FILTER (blocking)"
            logger.info(
                f"LLM Confluence Filter [{mode_str}]: "
                f"model={self.llm_model_name.value}, "
                f"threshold={self.llm_confidence_threshold.value:.2f}, "
                f"log_dir={log_dir or 'N/A'}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize LLM filter: {e}")
            self._llm_filter = None

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Calculate SMC indicators using LuxAlgo-style library."""

        # Lazy init LLM filter (params loaded after bot_start)
        self._init_llm_filter()

        # --- 1. MTF (15m) Calculations ---
        # Calculate SMC signals (Numba)
        # Returns signals AND active zones
        smc = SMCEngine(
            dataframe,
            internal_length=self.internal_length.value,
            swing_length=self.swing_length.value,
        )
        signals = smc.get_signals()

        # Merge signals into dataframe
        for col in signals.columns:
            dataframe[col] = signals[col].values

        # --- vol_sma: must be computed BEFORE multi-zone tracker ─────────
        # (tracker needs it to compute per-zone RVOL at creation time)
        dataframe["vol_sma"] = ta.SMA(dataframe["volume"], timeperiod=self.vol_sma_period.value)
        dataframe["vol_sma"] = dataframe["vol_sma"].replace(0, np.nan).bfill().ffill()

        # --- FVG ATR FILTER (BigBeluga Style) ---
        # Zero out FVGs that are smaller than Threshold * ATR
        # Using ATR(200) to match BigBeluga's volatility reference for "noise".
        if self.fvg_atr_threshold.value > 0:
            atr_200 = ta.ATR(dataframe, timeperiod=200)
            atr_200 = atr_200.bfill().ffill()
            fvg_threshold = atr_200 * self.fvg_atr_threshold.value

            bull_range = (
                dataframe["active_bullish_fvg_top"] - dataframe["active_bullish_fvg_bottom"]
            )
            mask_small_bull = (dataframe["active_bullish_fvg_top"] > 0) & (
                bull_range < fvg_threshold
            )
            dataframe.loc[
                mask_small_bull, ["active_bullish_fvg_top", "active_bullish_fvg_bottom"]
            ] = 0

            bear_range = (
                dataframe["active_bearish_fvg_top"] - dataframe["active_bearish_fvg_bottom"]
            )
            mask_small_bear = (dataframe["active_bearish_fvg_top"] > 0) & (
                bear_range < fvg_threshold
            )
            dataframe.loc[
                mask_small_bear, ["active_bearish_fvg_top", "active_bearish_fvg_bottom"]
            ] = 0

        # --- MULTI-ZONE CONTEXT (new features, shift applied inside) ─────
        dataframe = self._build_multi_zone_context(dataframe)

        # --- LOOKAHEAD BIAS FIX (existing active_* columns) ──────────────
        # Multi-zone columns already shifted inside _build_multi_zone_context.
        zone_cols = [c for c in dataframe.columns if c.startswith("active_")]
        for col in zone_cols:
            dataframe[col] = dataframe[col].shift(1).fillna(0)

        # --- RVOL for single-zone OBs (uses already-shifted active_*_vol) -
        dataframe["bull_ob_rvol"] = (
            dataframe["active_bullish_ob_vol"] / dataframe["vol_sma"]
        ).fillna(0)
        dataframe["bear_ob_rvol"] = (
            dataframe["active_bearish_ob_vol"] / dataframe["vol_sma"]
        ).fillna(0)

        # --- 2. HTF Calculations (All configured timeframes) ---
        if self.dp:
            htf_list = self._get_active_htf_list()
            for htf in htf_list:
                try:
                    inf_htf = self.dp.get_pair_dataframe(metadata["pair"], htf)
                    if inf_htf.empty:
                        logger.warning(f"No data for {metadata['pair']} {htf}")
                        continue

                    smc_htf = SMCEngine(
                        inf_htf,
                        internal_length=self.internal_length.value,
                        swing_length=self.swing_length.value,
                    )
                    signals_htf = smc_htf.get_signals()

                    swing_col    = f"{htf}_swing_trend"
                    internal_col = f"{htf}_internal_trend"
                    sh_col       = f"{htf}_swing_high"
                    sl_col       = f"{htf}_swing_low"

                    inf_htf[swing_col]    = signals_htf["swing_trend"]
                    inf_htf[internal_col] = signals_htf["internal_trend"]
                    # swing_high/low carry the current HTF trading range (used for
                    # HTF premium/discount). Not zone columns — no shift(1) needed.
                    inf_htf[sh_col] = signals_htf["swing_high"]
                    inf_htf[sl_col] = signals_htf["swing_low"]

                    # HTF Zone columns: OB / FVG / Breaker / FVG-Breaker
                    # Naming: {htf}_bull_ob_top, {htf}_bull_ob_bot, etc.
                    htf_zone_map = {
                        f"{htf}_bull_ob_top":   "active_bullish_ob_top",
                        f"{htf}_bull_ob_bot":   "active_bullish_ob_bottom",
                        f"{htf}_bear_ob_top":   "active_bearish_ob_top",
                        f"{htf}_bear_ob_bot":   "active_bearish_ob_bottom",
                        f"{htf}_bull_fvg_top":  "active_bullish_fvg_top",
                        f"{htf}_bull_fvg_bot":  "active_bullish_fvg_bottom",
                        f"{htf}_bear_fvg_top":  "active_bearish_fvg_top",
                        f"{htf}_bear_fvg_bot":  "active_bearish_fvg_bottom",
                        f"{htf}_bull_brk_top":  "active_bullish_breaker_top",
                        f"{htf}_bull_brk_bot":  "active_bullish_breaker_bottom",
                        f"{htf}_bear_brk_top":  "active_bearish_breaker_top",
                        f"{htf}_bear_brk_bot":  "active_bearish_breaker_bottom",
                        f"{htf}_bull_fbrk_top": "active_bullish_fvg_breaker_top",
                        f"{htf}_bull_fbrk_bot": "active_bullish_fvg_breaker_bottom",
                        f"{htf}_bear_fbrk_top": "active_bearish_fvg_breaker_top",
                        f"{htf}_bear_fbrk_bot": "active_bearish_fvg_breaker_bottom",
                    }
                    for dest_col, src_col in htf_zone_map.items():
                        if src_col in signals_htf.columns:
                            inf_htf[dest_col] = signals_htf[src_col].values

                    all_htf_cols = (
                        [swing_col, internal_col, sh_col, sl_col]
                        + list(htf_zone_map.keys())
                    )
                    inf_htf = inf_htf[["date"] + all_htf_cols].copy()

                    # FIX: shift(1) all HTF columns BEFORE merging so a 15m candle
                    # at HH:00 never reads the HTF candle that opened at HH:00 but
                    # hasn't closed yet (lookahead bias).
                    for col in all_htf_cols:
                        inf_htf[col] = inf_htf[col].shift(1)

                    dataframe = pd.merge(dataframe, inf_htf, on="date", how="left")
                    for col in all_htf_cols:
                        dataframe[col] = dataframe[col].ffill()
                    # Zone columns default to 0 (no zone) rather than NaN
                    for dest_col in htf_zone_map.keys():
                        dataframe[dest_col] = dataframe[dest_col].fillna(0)

                except Exception as e:
                    logger.error(f"Error processing HTF {htf}: {e}")

        # --- 3. Macro Regime Filter (BTC Weekly EMA21) ───────────────────
        dataframe = self._add_macro_bias(dataframe, metadata)

        # --- 4. Enrichment layers (new features) ─────────────────────────
        dataframe = self._add_pdh_pdl(dataframe, metadata)
        dataframe = self._add_premium_discount(dataframe)
        dataframe = self._add_session_context(dataframe)
        dataframe = self._add_entry_context(dataframe)

        # --- 5. ML Features (only when ML filter is active) ─────────────
        if self.use_ml_filter.value:
            dataframe = self._add_ml_features(dataframe)

        # Log summary
        if self.dp and self.enable_logging.value:
            logger.info(
                f"SMC LuxAlgo indicators for {metadata['pair']}: "
                f"MTF Trend Valid={dataframe['swing_trend'].notna().sum()}/{len(dataframe)} "
                f"HTF Trend Valid={dataframe.get('4h_swing_trend', pd.Series()).notna().sum()}/{len(dataframe)} "
                f"MultiZone Bull OBs={dataframe['n_active_bull_obs'].max():.0f} max active"
            )

        # Export indicator snapshot for post-analysis (if enabled)
        if self.export_indicator_data.value:
            self._save_indicator_parquet(dataframe, metadata["pair"])

        return dataframe

    def _add_ml_features(self, df: DataFrame) -> DataFrame:
        """
        Generate consistent ML features for both Training and Inference.
        Must match logic in train_smc_model.py.

        All new columns are accumulated in a dict and written via a single
        pd.concat to avoid the PerformanceWarning from repeated df[col] inserts.
        """
        nd: dict = {}   # new_data — all columns go here, df stays read-only

        # --- 1. Technical Indicators (Momentum & Volatility) ---

        rsi     = ta.RSI(df["close"], timeperiod=14)
        nd["rsi"]    = rsi
        nd["ml_rsi"] = rsi / 100.0

        adx          = ta.ADX(df["high"], df["low"], df["close"], timeperiod=14)
        nd["adx"]    = adx
        nd["ml_adx"] = adx / 100.0

        cci          = ta.CCI(df["high"], df["low"], df["close"], timeperiod=20)
        nd["cci"]    = cci
        nd["ml_cci"] = cci / 300.0

        # WaveTrend oscillator
        n1  = 10
        n2  = 21
        ap  = (df["high"] + df["low"] + df["close"]) / 3
        esa = ta.EMA(ap, timeperiod=n1)
        d_  = ta.EMA((ap - esa).abs(), timeperiod=n1)
        ci  = (ap - esa) / (0.015 * d_)
        wt1 = ta.EMA(ci, timeperiod=n2)
        wt2 = ta.SMA(wt1, timeperiod=4)
        nd["wt1"]        = wt1
        nd["wt2"]        = wt2
        nd["ml_wt1"]     = wt1 / 100.0
        nd["ml_wt2"]     = wt2 / 100.0
        nd["ml_wt_diff"] = (wt1 - wt2) / 100.0

        ema_50  = ta.EMA(df["close"], timeperiod=50)
        ema_200 = ta.EMA(df["close"], timeperiod=200)
        nd["ema_50"]         = ema_50
        nd["ema_200"]        = ema_200
        nd["ml_dist_ema50"]  = (df["close"] - ema_50)  / ema_50
        nd["ml_dist_ema200"] = (df["close"] - ema_200) / ema_200

        atr          = ta.ATR(df, timeperiod=14)
        nd["atr"]    = atr
        nd["ml_atr_pct"] = atr / df["close"]

        atr_1  = ta.ATR(df, timeperiod=1)
        atr_10 = ta.ATR(df, timeperiod=10)
        nd["ml_volatility_high"] = (atr_1 > atr_10).astype(float)

        vol_ma = ta.SMA(df["volume"], timeperiod=20)
        nd["volume_ma"]       = vol_ma
        nd["ml_volume_ratio"] = df["volume"] / np.where(vol_ma == 0, 1, vol_ma)

        rsi_9         = ta.RSI(df["close"], timeperiod=9)
        nd["rsi_9"]   = rsi_9
        nd["ml_rsi_9"] = rsi_9 / 100.0

        # --- 2. SMC Context ---

        def bars_since(series):
            return series.cumsum().groupby(series.cumsum()).cumcount()

        nd["ml_bars_since_int_bull_choch"] = bars_since(df["internal_choch_bullish"])
        nd["ml_bars_since_int_bear_choch"] = bars_since(df["internal_choch_bearish"])

        nd["ml_swing_trend"] = df["swing_trend"]

        nd["ml_in_bull_ob"] = (
            (df["active_bullish_ob_top"] > 0) & (df["low"] <= df["active_bullish_ob_top"])
        ).astype(int)
        nd["ml_in_bear_ob"] = (
            (df["active_bearish_ob_top"] > 0) & (df["high"] >= df["active_bearish_ob_bottom"])
        ).astype(int)

        if "internal_sweep_bullish" in df.columns:
            nd["ml_recent_bull_sweep"] = (
                df["internal_sweep_bullish"].rolling(3, min_periods=1).max().fillna(0)
            )
            nd["ml_recent_bear_sweep"] = (
                df["internal_sweep_bearish"].rolling(3, min_periods=1).max().fillna(0)
            )
        else:
            nd["ml_recent_bull_sweep"] = 0
            nd["ml_recent_bear_sweep"] = 0

        # --- 3. Opposing Zone Obstacle Features ---
        NO_ZONE_DIST = 0.5

        bear_ob_above = (df["active_bearish_ob_bottom"] > 0) & (
            df["active_bearish_ob_bottom"] > df["close"]
        )
        ml_bear_ob_above_dist = np.where(
            bear_ob_above,
            (df["active_bearish_ob_bottom"] - df["close"]) / df["close"],
            NO_ZONE_DIST,
        )
        ml_bear_ob_above_rvol = np.where(bear_ob_above, df["bear_ob_rvol"], 0.0)
        nd["ml_bear_ob_above_dist"] = ml_bear_ob_above_dist
        nd["ml_bear_ob_above_rvol"] = ml_bear_ob_above_rvol

        bear_fvg_above = (df["active_bearish_fvg_bottom"] > 0) & (
            df["active_bearish_fvg_bottom"] > df["close"]
        )
        ml_bear_fvg_above_dist = np.where(
            bear_fvg_above,
            (df["active_bearish_fvg_bottom"] - df["close"]) / df["close"],
            NO_ZONE_DIST,
        )
        nd["ml_bear_fvg_above_dist"] = ml_bear_fvg_above_dist

        bull_ob_below = (df["active_bullish_ob_top"] > 0) & (
            df["active_bullish_ob_top"] < df["close"]
        )
        ml_bull_ob_below_dist = np.where(
            bull_ob_below,
            (df["close"] - df["active_bullish_ob_top"]) / df["close"],
            NO_ZONE_DIST,
        )
        ml_bull_ob_below_rvol = np.where(bull_ob_below, df["bull_ob_rvol"], 0.0)
        nd["ml_bull_ob_below_dist"] = ml_bull_ob_below_dist
        nd["ml_bull_ob_below_rvol"] = ml_bull_ob_below_rvol

        bull_fvg_below = (df["active_bullish_fvg_top"] > 0) & (
            df["active_bullish_fvg_top"] < df["close"]
        )
        ml_bull_fvg_below_dist = np.where(
            bull_fvg_below,
            (df["close"] - df["active_bullish_fvg_top"]) / df["close"],
            NO_ZONE_DIST,
        )
        nd["ml_bull_fvg_below_dist"] = ml_bull_fvg_below_dist

        # Aggregated obstacle distances
        nd["ml_min_resist_above_dist"] = np.minimum(ml_bear_ob_above_dist, ml_bear_fvg_above_dist)
        nd["ml_max_resist_above_rvol"] = ml_bear_ob_above_rvol
        nd["ml_min_support_below_dist"] = np.minimum(ml_bull_ob_below_dist, ml_bull_fvg_below_dist)
        nd["ml_max_support_below_rvol"] = ml_bull_ob_below_rvol

        # --- 4. Single concat + fillna -----------------------------------------
        new_df = pd.DataFrame(nd, index=df.index).fillna(0)
        return pd.concat([df, new_df], axis=1)

    # ── Indicator Export ──────────────────────────────────────────────────────

    _EXPORT_COLS = [
        "bull_ob_0_rvol", "bear_ob_0_rvol",
        "bull_fvg_0_rvol", "bear_fvg_0_rvol",
        "bull_fvg_0_filled_pct", "bear_fvg_0_filled_pct",
        "bull_ob_0_touches", "bear_ob_0_touches",
        "n_active_bull_obs", "n_active_bear_obs",
        "htf_range_position_pct", "daily_range_position_pct",
    ]

    def _save_indicator_parquet(self, dataframe: DataFrame, pair: str) -> None:
        """Save a snapshot of key indicator columns to parquet for post-analysis.

        File: user_data/strategies/SMC/analisis/indicators_<PAIR>.parquet
        Key: date (UTC, matching the signal candle).
        analyze_backtest.py joins on pair + (open_date - 1 timeframe).
        """
        out_dir = Path("user_data/strategies/SMC/analisis")
        out_dir.mkdir(parents=True, exist_ok=True)

        safe_pair = pair.replace("/", "_").replace(":", "_")
        out_path  = out_dir / f"indicators_{safe_pair}.parquet"

        cols = ["date"] + [c for c in self._EXPORT_COLS if c in dataframe.columns]
        dataframe[cols].to_parquet(out_path, index=False)

    # ==========================================================================
    # MULTI-ZONE CONTEXT  (Priority 1 & 3 features)
    # ==========================================================================

    # Slots exported per direction
    _MZ_MAX_OBS  = 10
    _MZ_MAX_FVGS =  5
    _MZ_MAX_BRKS =  3

    def _build_multi_zone_context(self, dataframe: DataFrame) -> DataFrame:
        """
        Pure-Python zone tracker built on top of the raw event arrays exposed
        by smc_engine.get_signals().

        For each bar it maintains running lists of active zones and writes
        indexed columns (bull_ob_0_top … bull_ob_9_top, etc.).

        Zone quality per slot (k=0 → most recent):
          OB:  top, bottom, rvol, age_bars, touches, mitigated (0/1)
          FVG: top, bottom, rvol, filled_pct, in_ob (0/1)
          BRK: top, bottom

        NOTE: shift(1) is applied to all new columns at the end so they obey
        the same lookahead-bias rule as the existing active_* columns.
        """
        # Guard: if raw columns are missing, return silently
        if "ob_bull_top_raw" not in dataframe.columns:
            logger.warning("Multi-zone: raw event columns not found – skipping.")
            return dataframe

        # When ML is off only slot-0 is used in entry/exit — skip the rest
        MAX_OBS  = self._MZ_MAX_OBS  if self.use_ml_filter.value else 1
        MAX_FVGS = self._MZ_MAX_FVGS if self.use_ml_filter.value else 1
        MAX_BRKS = self._MZ_MAX_BRKS if self.use_ml_filter.value else 1

        n      = len(dataframe)
        high   = dataframe["high"].values
        low    = dataframe["low"].values
        close  = dataframe["close"].values
        vs_arr = dataframe["vol_sma"].values  # already computed upstream

        ob_bt  = dataframe["ob_bull_top_raw"].values
        ob_bb  = dataframe["ob_bull_btm_raw"].values
        ob_bv  = dataframe["ob_bull_vol_raw"].values
        ob_et  = dataframe["ob_bear_top_raw"].values
        ob_eb  = dataframe["ob_bear_btm_raw"].values
        ob_ev  = dataframe["ob_bear_vol_raw"].values

        fvg_bt  = dataframe["fvg_bull_top_raw"].values
        fvg_bb  = dataframe["fvg_bull_btm_raw"].values
        fvg_biv = dataframe["fvg_bull_impulse_vol_raw"].values
        fvg_et  = dataframe["fvg_bear_top_raw"].values
        fvg_eb  = dataframe["fvg_bear_btm_raw"].values
        fvg_eiv = dataframe["fvg_bear_impulse_vol_raw"].values

        # ── Output arrays (n × MAX_*) ──────────────────────────────────────
        bull_ob_top  = np.zeros((n, MAX_OBS)); bull_ob_btm  = np.zeros((n, MAX_OBS))
        bull_ob_rvol = np.zeros((n, MAX_OBS)); bull_ob_age  = np.zeros((n, MAX_OBS))
        bull_ob_tch  = np.zeros((n, MAX_OBS)); bull_ob_mit  = np.zeros((n, MAX_OBS))

        bear_ob_top  = np.zeros((n, MAX_OBS)); bear_ob_btm  = np.zeros((n, MAX_OBS))
        bear_ob_rvol = np.zeros((n, MAX_OBS)); bear_ob_age  = np.zeros((n, MAX_OBS))
        bear_ob_tch  = np.zeros((n, MAX_OBS)); bear_ob_mit  = np.zeros((n, MAX_OBS))

        bull_fvg_top  = np.zeros((n, MAX_FVGS)); bull_fvg_btm    = np.zeros((n, MAX_FVGS))
        bull_fvg_rvol = np.zeros((n, MAX_FVGS)); bull_fvg_filled = np.zeros((n, MAX_FVGS))
        bull_fvg_inob = np.zeros((n, MAX_FVGS))

        bear_fvg_top  = np.zeros((n, MAX_FVGS)); bear_fvg_btm    = np.zeros((n, MAX_FVGS))
        bear_fvg_rvol = np.zeros((n, MAX_FVGS)); bear_fvg_filled = np.zeros((n, MAX_FVGS))
        bear_fvg_inob = np.zeros((n, MAX_FVGS))

        bull_brk_top = np.zeros((n, MAX_BRKS)); bull_brk_btm = np.zeros((n, MAX_BRKS))
        bear_brk_top = np.zeros((n, MAX_BRKS)); bear_brk_btm = np.zeros((n, MAX_BRKS))

        # ── In-flight zone state (list of dicts) ────────────────────────────
        # Each dict: top, bottom, rvol, bar_idx, age, touches, inside,
        #            mid_touched, min_low (FVG fill tracker), max_high
        act_bull_obs  = []
        act_bear_obs  = []
        act_bull_fvgs = []
        act_bear_fvgs = []
        act_bull_brks = []   # created when bear OB is broken upward
        act_bear_brks = []   # created when bull OB is broken downward

        for i in range(n):
            c  = close[i]
            h  = high[i]
            l  = low[i]
            vs = vs_arr[i] if (vs_arr[i] > 0 and not np.isnan(vs_arr[i])) else 1.0

            # ── 1. Update existing zones ──────────────────────────────────

            # --- Bull OBs (support) ---
            next_bull_obs = []
            for z in act_bull_obs:
                if c < z["bottom"]:
                    # Invalidated → flip to bearish breaker
                    act_bear_brks.append({"top": z["top"], "bottom": z["bottom"]})
                    if len(act_bear_brks) > MAX_BRKS:
                        act_bear_brks = act_bear_brks[-MAX_BRKS:]
                else:
                    was_inside = z["inside"]
                    z["inside"] = (l <= z["top"]) and (c >= z["bottom"])
                    if z["inside"]:
                        if not was_inside:
                            z["touches"] += 1
                        mid = (z["top"] + z["bottom"]) / 2
                        if l <= mid:
                            z["mid_touched"] = True
                    z["age"] = i - z["bar_idx"]
                    next_bull_obs.append(z)
            act_bull_obs = next_bull_obs

            # --- Bear OBs (resistance) ---
            next_bear_obs = []
            for z in act_bear_obs:
                if c > z["top"]:
                    # Invalidated → flip to bullish breaker
                    act_bull_brks.append({"top": z["top"], "bottom": z["bottom"]})
                    if len(act_bull_brks) > MAX_BRKS:
                        act_bull_brks = act_bull_brks[-MAX_BRKS:]
                else:
                    was_inside = z["inside"]
                    z["inside"] = (h >= z["bottom"]) and (c <= z["top"])
                    if z["inside"]:
                        if not was_inside:
                            z["touches"] += 1
                        mid = (z["top"] + z["bottom"]) / 2
                        if h >= mid:
                            z["mid_touched"] = True
                    z["age"] = i - z["bar_idx"]
                    next_bear_obs.append(z)
            act_bear_obs = next_bear_obs

            # --- Bull FVGs (support gap; price enters from above) ---
            # fvg_bull: bottom=high[i-2], top=low[i] → zone BELOW price, fills as price retraces down
            next_bull_fvgs = []
            for z in act_bull_fvgs:
                if c < z["bottom"]:
                    pass  # fully broken → discard
                else:
                    if l <= z["top"]:
                        # Price entered the gap (came down from above)
                        deepest = min(l, z.get("min_low", z["top"]))
                        z["min_low"] = deepest
                        gap = z["top"] - z["bottom"]
                        z["filled_pct"] = (z["top"] - deepest) / gap if gap > 0 else 0.0
                        z["filled_pct"] = min(1.0, z["filled_pct"])
                    z["age"] = i - z["bar_idx"]
                    next_bull_fvgs.append(z)
            act_bull_fvgs = next_bull_fvgs

            # --- Bear FVGs (resistance gap; price enters from below) ---
            next_bear_fvgs = []
            for z in act_bear_fvgs:
                if c > z["top"]:
                    pass  # fully broken → discard
                else:
                    if h >= z["bottom"]:
                        deepest = max(h, z.get("max_high", z["bottom"]))
                        z["max_high"] = deepest
                        gap = z["top"] - z["bottom"]
                        z["filled_pct"] = (deepest - z["bottom"]) / gap if gap > 0 else 0.0
                        z["filled_pct"] = min(1.0, z["filled_pct"])
                    z["age"] = i - z["bar_idx"]
                    next_bear_fvgs.append(z)
            act_bear_fvgs = next_bear_fvgs

            # --- Breakers validation ---
            act_bull_brks = [z for z in act_bull_brks if c >= z["bottom"]]
            act_bear_brks = [z for z in act_bear_brks if c <= z["top"]]

            # ── 2. Add new zones ──────────────────────────────────────────

            if ob_bt[i] > 0:
                act_bull_obs.append({
                    "top": ob_bt[i], "bottom": ob_bb[i],
                    "rvol": ob_bv[i] / vs,
                    "bar_idx": i, "age": 0,
                    "touches": 0, "inside": False, "mid_touched": False,
                })
                if len(act_bull_obs) > MAX_OBS:
                    act_bull_obs = act_bull_obs[-MAX_OBS:]

            if ob_et[i] > 0:
                act_bear_obs.append({
                    "top": ob_et[i], "bottom": ob_eb[i],
                    "rvol": ob_ev[i] / vs,
                    "bar_idx": i, "age": 0,
                    "touches": 0, "inside": False, "mid_touched": False,
                })
                if len(act_bear_obs) > MAX_OBS:
                    act_bear_obs = act_bear_obs[-MAX_OBS:]

            if fvg_bt[i] > 0:
                act_bull_fvgs.append({
                    "top": fvg_bt[i], "bottom": fvg_bb[i],
                    "rvol": fvg_biv[i] / vs,
                    "bar_idx": i, "age": 0,
                    "filled_pct": 0.0, "min_low": fvg_bt[i],
                })
                if len(act_bull_fvgs) > MAX_FVGS:
                    act_bull_fvgs = act_bull_fvgs[-MAX_FVGS:]

            if fvg_et[i] > 0:
                act_bear_fvgs.append({
                    "top": fvg_et[i], "bottom": fvg_eb[i],
                    "rvol": fvg_eiv[i] / vs,
                    "bar_idx": i, "age": 0,
                    "filled_pct": 0.0, "max_high": fvg_eb[i],
                })
                if len(act_bear_fvgs) > MAX_FVGS:
                    act_bear_fvgs = act_bear_fvgs[-MAX_FVGS:]

            # ── 3. Write outputs (slot 0 = most recent) ───────────────────

            for k, z in enumerate(reversed(act_bull_obs)):
                if k >= MAX_OBS: break
                bull_ob_top[i, k] = z["top"];    bull_ob_btm[i, k]  = z["bottom"]
                bull_ob_rvol[i, k] = z["rvol"];  bull_ob_age[i, k]  = z["age"]
                bull_ob_tch[i, k]  = z["touches"]; bull_ob_mit[i, k] = int(z["mid_touched"])

            for k, z in enumerate(reversed(act_bear_obs)):
                if k >= MAX_OBS: break
                bear_ob_top[i, k] = z["top"];    bear_ob_btm[i, k]  = z["bottom"]
                bear_ob_rvol[i, k] = z["rvol"];  bear_ob_age[i, k]  = z["age"]
                bear_ob_tch[i, k]  = z["touches"]; bear_ob_mit[i, k] = int(z["mid_touched"])

            for k, z in enumerate(reversed(act_bull_fvgs)):
                if k >= MAX_FVGS: break
                bull_fvg_top[i, k]    = z["top"];    bull_fvg_btm[i, k]    = z["bottom"]
                bull_fvg_rvol[i, k]   = z["rvol"];   bull_fvg_filled[i, k] = z["filled_pct"]
                # in_ob: FVG entirely inside an active bull OB
                for ob in act_bull_obs:
                    if z["top"] <= ob["top"] and z["bottom"] >= ob["bottom"]:
                        bull_fvg_inob[i, k] = 1
                        break

            for k, z in enumerate(reversed(act_bear_fvgs)):
                if k >= MAX_FVGS: break
                bear_fvg_top[i, k]    = z["top"];    bear_fvg_btm[i, k]    = z["bottom"]
                bear_fvg_rvol[i, k]   = z["rvol"];   bear_fvg_filled[i, k] = z["filled_pct"]
                for ob in act_bear_obs:
                    if z["top"] <= ob["top"] and z["bottom"] >= ob["bottom"]:
                        bear_fvg_inob[i, k] = 1
                        break

            for k, z in enumerate(reversed(act_bull_brks)):
                if k >= MAX_BRKS: break
                bull_brk_top[i, k] = z["top"]; bull_brk_btm[i, k] = z["bottom"]

            for k, z in enumerate(reversed(act_bear_brks)):
                if k >= MAX_BRKS: break
                bear_brk_top[i, k] = z["top"]; bear_brk_btm[i, k] = z["bottom"]

        # ── 4. Build all new columns as a dict → single pd.concat ───────────
        # Avoids the PerformanceWarning from repeated df[col] = ... inserts.
        new_data: dict[str, np.ndarray] = {}

        for k in range(MAX_OBS):
            new_data[f"bull_ob_{k}_top"]       = bull_ob_top[:, k]
            new_data[f"bull_ob_{k}_bottom"]    = bull_ob_btm[:, k]
            new_data[f"bull_ob_{k}_rvol"]      = bull_ob_rvol[:, k]
            new_data[f"bull_ob_{k}_age"]       = bull_ob_age[:, k]
            new_data[f"bull_ob_{k}_touches"]   = bull_ob_tch[:, k]
            new_data[f"bull_ob_{k}_mitigated"] = bull_ob_mit[:, k]

        for k in range(MAX_OBS):
            new_data[f"bear_ob_{k}_top"]       = bear_ob_top[:, k]
            new_data[f"bear_ob_{k}_bottom"]    = bear_ob_btm[:, k]
            new_data[f"bear_ob_{k}_rvol"]      = bear_ob_rvol[:, k]
            new_data[f"bear_ob_{k}_age"]       = bear_ob_age[:, k]
            new_data[f"bear_ob_{k}_touches"]   = bear_ob_tch[:, k]
            new_data[f"bear_ob_{k}_mitigated"] = bear_ob_mit[:, k]

        for k in range(MAX_FVGS):
            new_data[f"bull_fvg_{k}_top"]        = bull_fvg_top[:, k]
            new_data[f"bull_fvg_{k}_bottom"]     = bull_fvg_btm[:, k]
            new_data[f"bull_fvg_{k}_rvol"]       = bull_fvg_rvol[:, k]
            new_data[f"bull_fvg_{k}_filled_pct"] = bull_fvg_filled[:, k]
            new_data[f"bull_fvg_{k}_in_ob"]      = bull_fvg_inob[:, k]

        for k in range(MAX_FVGS):
            new_data[f"bear_fvg_{k}_top"]        = bear_fvg_top[:, k]
            new_data[f"bear_fvg_{k}_bottom"]     = bear_fvg_btm[:, k]
            new_data[f"bear_fvg_{k}_rvol"]       = bear_fvg_rvol[:, k]
            new_data[f"bear_fvg_{k}_filled_pct"] = bear_fvg_filled[:, k]
            new_data[f"bear_fvg_{k}_in_ob"]      = bear_fvg_inob[:, k]

        for k in range(MAX_BRKS):
            new_data[f"bull_brk_{k}_top"]    = bull_brk_top[:, k]
            new_data[f"bull_brk_{k}_bottom"] = bull_brk_btm[:, k]
            new_data[f"bear_brk_{k}_top"]    = bear_brk_top[:, k]
            new_data[f"bear_brk_{k}_bottom"] = bear_brk_btm[:, k]

        # Summary counts
        new_data["n_active_bull_obs"]  = (bull_ob_top  > 0).sum(axis=1)
        new_data["n_active_bear_obs"]  = (bear_ob_top  > 0).sum(axis=1)
        new_data["n_active_bull_fvgs"] = (bull_fvg_top > 0).sum(axis=1)
        new_data["n_active_bear_fvgs"] = (bear_fvg_top > 0).sum(axis=1)

        # ── 5. Shift(1) en bloque + concat único ─────────────────────────
        # shift() on the whole DataFrame at once → single memory allocation.
        new_df = pd.DataFrame(new_data, index=dataframe.index)
        new_df = new_df.shift(1).fillna(0)

        dataframe = pd.concat([dataframe, new_df], axis=1)
        return dataframe

    # ==========================================================================
    # MACRO REGIME FILTER  (BTC EMA200 Daily + BTC 4H Swing Trend)
    # ==========================================================================

    def _add_macro_bias(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds 'btc_macro_bias' column: +1 = bull, 0 = neutral, -1 = bear.

        Uses BTC/USDT:USDT as institutional macro reference (not the traded pair):
        - Daily EMA200: close > EMA200*(1+0.5%) → +1 bull, close < EMA200*(1-0.5%) → -1 bear
        - 4H swing_trend from SMCEngine kernel: +1 or -1
        - Combined bias: +1 only if both agree bull, -1 only if both agree bear, else 0 (neutral)

        Using BTC as reference instead of per-pair weekly EMA avoids the 2024 regime trap:
        ETH can be in a local pullback (weekly bearish) while BTC is at ATH → per-pair
        bias would block longs/take shorts on ETH, losing against the macro trend.

        ewm(span=200, min_periods=1): starts from first bar, converges to true EMA200.
        shift(1) on both biases ensures only *closed* candles influence entries (no lookahead).
        merge_asof(direction="backward") handles timestamp alignment across timeframes.
        Falls back to 0 (neutral = both directions allowed) on any error.
        """
        dataframe["btc_macro_bias"] = 0  # default: neutral

        if not self.dp:
            return dataframe

        try:
            # ── BTC Daily EMA200 ──────────────────────────────────────────────
            btc_1d = self.dp.get_pair_dataframe("BTC/USDT:USDT", "1d")
            if btc_1d is None or btc_1d.empty:
                logger.warning("Macro bias: BTC/USDT:USDT 1D not available, defaulting to neutral.")
                return dataframe

            btc_1d = btc_1d[["date", "close"]].copy()
            btc_1d["ema200"] = btc_1d["close"].ewm(span=200, min_periods=1).mean()
            btc_1d["bias_1d"] = np.where(
                btc_1d["close"] > btc_1d["ema200"] * 1.005,  1,
                np.where(btc_1d["close"] < btc_1d["ema200"] * 0.995, -1, 0),
            )
            # shift(1): only the *closed* daily candle contributes — the current one is forming
            btc_1d["bias_1d"] = btc_1d["bias_1d"].shift(1).fillna(0).astype(int)

            # ── BTC 4H Swing Trend ────────────────────────────────────────────
            btc_4h = self.dp.get_pair_dataframe("BTC/USDT:USDT", "4h")
            if btc_4h is None or btc_4h.empty:
                logger.warning("Macro bias: BTC/USDT:USDT 4H not available, defaulting to neutral.")
                return dataframe

            smc_btc = SMCEngine(
                btc_4h,
                internal_length=self.internal_length.value,
                swing_length=self.swing_length.value,
            )
            signals_btc = smc_btc.get_signals()
            btc_4h = btc_4h[["date"]].copy()
            # shift(1): 4H candle's trend is known only after it closes
            btc_4h["bias_4h"] = signals_btc["swing_trend"].shift(1).fillna(0).astype(int)

            # ── Align timezones before merge_asof ────────────────────────────
            ref_tz = dataframe["date"].dt.tz
            for src_df in (btc_1d, btc_4h):
                src_tz = src_df["date"].dt.tz
                if ref_tz is not None and src_tz is None:
                    src_df["date"] = src_df["date"].dt.tz_localize("UTC")
                elif ref_tz is None and src_tz is not None:
                    src_df["date"] = src_df["date"].dt.tz_convert(None)

            # ── merge_asof: propagate each bias to all 15m candles ────────────
            # dataframe is already sorted by date in populate_indicators.
            # merge_asof finds the most recent key <= each 15m timestamp (no lookahead).
            df_base = dataframe[["date"]].copy()
            df_base = pd.merge_asof(
                df_base,
                btc_1d.sort_values("date")[["date", "bias_1d"]],
                on="date", direction="backward",
            )
            df_base = pd.merge_asof(
                df_base,
                btc_4h.sort_values("date")[["date", "bias_4h"]],
                on="date", direction="backward",
            )

            bias_1d = df_base["bias_1d"].fillna(0).astype(int)
            bias_4h = df_base["bias_4h"].fillna(0).astype(int)

            # Require consensus: both timeframes must agree for a directional bias
            dataframe["btc_macro_bias"] = np.where(
                (bias_1d == 1)  & (bias_4h == 1),   1,
                np.where((bias_1d == -1) & (bias_4h == -1), -1, 0),
            )

        except Exception as e:
            logger.warning(f"Macro bias calculation failed, defaulting to neutral: {e}")
            dataframe["btc_macro_bias"] = 0

        return dataframe

    # ==========================================================================
    # PDH / PDL  (Priority 4)
    # ==========================================================================

    def _add_pdh_pdl(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Previous Day High / Low computed from 1h HTF data (already loaded).
        Falls back to grouping the 15m data by UTC day if 1h is unavailable.
        """
        pdh = np.full(len(dataframe), np.nan)
        pdl = np.full(len(dataframe), np.nan)

        try:
            if self.dp:
                # Prefer 1h (always loaded as HTF), fall back to 4h or raw 15m
                src_tf = None
                for tf in ["1h", "4h"]:
                    if tf in self._get_active_htf_list():
                        src_tf = tf
                        break

                if src_tf:
                    htf = self.dp.get_pair_dataframe(metadata["pair"], src_tf)
                else:
                    htf = dataframe[["date", "high", "low"]].copy()
                    src_tf = self.timeframe

                if not htf.empty:
                    htf = htf.copy()
                    htf["date"] = pd.to_datetime(htf["date"], utc=True)
                    htf["_day"] = htf["date"].dt.floor("D")

                    daily = (
                        htf.groupby("_day")
                        .agg(day_high=("high", "max"), day_low=("low", "min"))
                        .reset_index()
                    )
                    daily["pdh"] = daily["day_high"].shift(1)
                    daily["pdl"] = daily["day_low"].shift(1)

                    # Merge into MTF dataframe
                    main = dataframe.copy()
                    main["date"] = pd.to_datetime(main["date"], utc=True)
                    main["_day"] = main["date"].dt.floor("D")
                    main = main.merge(daily[["_day", "pdh", "pdl"]], on="_day", how="left")
                    pdh = main["pdh"].ffill().values
                    pdl = main["pdl"].ffill().values

        except Exception as e:
            logger.warning(f"PDH/PDL calc failed: {e}")

        dataframe["pdh"] = pdh
        dataframe["pdl"] = pdl
        dataframe["above_pdh"] = (dataframe["close"] > dataframe["pdh"]).fillna(False)
        dataframe["below_pdl"] = (dataframe["close"] < dataframe["pdl"]).fillna(False)

        # -- Daily premium/discount (using PDH/PDL as the daily range) --------
        d_rng = (dataframe["pdh"] - dataframe["pdl"]).clip(lower=1e-10)
        d_pct = (dataframe["close"] - dataframe["pdl"]) / d_rng * 100
        dataframe["daily_equilibrium"]       = (dataframe["pdh"] + dataframe["pdl"]) / 2
        dataframe["daily_range_position_pct"] = d_pct.fillna(50.0)
        dataframe["daily_prem_disc_label"]   = self._pct_to_pd_label(d_pct)

        # -- Weekly Previous High / Low (PWH / PWL) ----------------------------
        pwh = np.full(len(dataframe), np.nan)
        pwl = np.full(len(dataframe), np.nan)
        try:
            if self.dp:
                src_tf = None
                for tf in ["1h", "4h"]:
                    if tf in self._get_active_htf_list():
                        src_tf = tf
                        break
                if src_tf:
                    wkly_src = self.dp.get_pair_dataframe(metadata["pair"], src_tf)
                else:
                    wkly_src = dataframe[["date", "high", "low"]].copy()

                if wkly_src is not None and not wkly_src.empty:
                    ws = wkly_src.copy()
                    ws["date"] = pd.to_datetime(ws["date"], utc=True)
                    # ISO week start (Monday 00:00 UTC) via date arithmetic.
                    # Avoids dt.to_period() which silently drops timezone info.
                    d = ws["date"]
                    ws["_week"] = (d - pd.to_timedelta(d.dt.weekday, unit="D")).dt.normalize()

                    weekly = (
                        ws.groupby("_week")
                        .agg(wk_high=("high", "max"), wk_low=("low", "min"))
                        .reset_index()
                    )
                    weekly["pwh"] = weekly["wk_high"].shift(1)
                    weekly["pwl"] = weekly["wk_low"].shift(1)

                    main_w = dataframe.copy()
                    main_w["date"] = pd.to_datetime(main_w["date"], utc=True)
                    d2 = main_w["date"]
                    main_w["_week"] = (d2 - pd.to_timedelta(d2.dt.weekday, unit="D")).dt.normalize()

                    main_w = main_w.merge(
                        weekly[["_week", "pwh", "pwl"]], on="_week", how="left"
                    )
                    pwh = main_w["pwh"].ffill().values
                    pwl = main_w["pwl"].ffill().values
        except Exception as e:
            logger.warning(f"PWH/PWL calc failed: {e}")

        dataframe["pwh"] = pwh
        dataframe["pwl"] = pwl

        w_rng = (dataframe["pwh"] - dataframe["pwl"]).clip(lower=1e-10)
        w_pct = (dataframe["close"] - dataframe["pwl"]) / w_rng * 100
        dataframe["weekly_equilibrium"]        = (dataframe["pwh"] + dataframe["pwl"]) / 2
        dataframe["weekly_range_position_pct"] = w_pct.fillna(50.0)
        dataframe["weekly_prem_disc_label"]    = self._pct_to_pd_label(w_pct)

        return dataframe

    @staticmethod
    def _pct_to_pd_label(pct: pd.Series) -> pd.Series:
        """Convert a 0-100 range position to deep_discount/discount/premium/deep_premium."""
        lbl = pd.Series("premium", index=pct.index, dtype=object)
        lbl[pct < 50]  = "discount"
        lbl[pct >= 75] = "deep_premium"
        lbl[pct < 25]  = "deep_discount"
        return lbl

    # ==========================================================================
    # PREMIUM / DISCOUNT  (Priority 5)
    # ==========================================================================

    def _add_premium_discount(self, dataframe: DataFrame) -> DataFrame:
        """
        Three-tier premium/discount framework (ICT style).

        A) Swing / MTF (15m internal range) — noisy, changes with each swing.
           Kept as reference, clearly labelled as swing_*.

           swing_range_position_pct   : 0-100 within [swing_low, swing_high]
           swing_prem_disc_label      : deep_discount / discount / premium / deep_premium
           swing_fib_236 … swing_fib_786

        B) HTF (4h or 1h, whichever is configured) — stable institutional range.
           The range is the last confirmed 4h swing_high / swing_low pair.
           This range stays fixed until a new HTF swing is confirmed.

           htf_range_high / htf_range_low
           htf_equilibrium
           htf_range_position_pct
           htf_prem_disc_label
           htf_fib_236 … htf_fib_786

        C) Daily and Weekly ranges are handled in _add_pdh_pdl (they need the
           raw OHLCV data from the HTF source). This method reads those results.
        """
        # -- A. Swing / MTF range ---------------------------------------------
        sh  = dataframe["swing_high"].ffill()
        sl  = dataframe["swing_low"].ffill()
        rng = (sh - sl).clip(lower=1e-10)

        pct = (dataframe["close"] - sl) / rng * 100
        dataframe["swing_range_position_pct"] = pct.fillna(50.0)
        dataframe["swing_prem_disc_label"]    = self._pct_to_pd_label(pct)

        for level in [0.236, 0.382, 0.500, 0.618, 0.786]:
            dataframe[f"swing_fib_{int(level * 1000):03d}"] = sl + rng * level

        # -- B. HTF range (4h preferred, fall back to 1h then swing) ----------
        # Determine which HTF columns are available
        primary_htf = None
        for candidate in ["4h", "1h"]:
            if f"{candidate}_swing_high" in dataframe.columns:
                primary_htf = candidate
                break

        if primary_htf is not None:
            htf_sh  = dataframe[f"{primary_htf}_swing_high"].ffill()
            htf_sl  = dataframe[f"{primary_htf}_swing_low"].ffill()
            htf_rng = (htf_sh - htf_sl).clip(lower=1e-10)
            htf_pct = (dataframe["close"] - htf_sl) / htf_rng * 100
        else:
            # Fallback: use the same MTF swing range
            htf_sh  = sh
            htf_sl  = sl
            htf_rng = rng
            htf_pct = pct

        dataframe["htf_range_high"]        = htf_sh
        dataframe["htf_range_low"]         = htf_sl
        dataframe["htf_equilibrium"]       = (htf_sh + htf_sl) / 2
        dataframe["htf_range_position_pct"] = htf_pct.fillna(50.0)
        dataframe["htf_prem_disc_label"]   = self._pct_to_pd_label(htf_pct)

        for level in [0.236, 0.382, 0.500, 0.618, 0.786]:
            dataframe[f"htf_fib_{int(level * 1000):03d}"] = htf_sl + htf_rng * level

        return dataframe

    # ==========================================================================
    # SESSION CONTEXT  (Priority 6)
    # ==========================================================================

    def _add_session_context(self, dataframe: DataFrame) -> DataFrame:
        """
        Label each bar with its UTC trading session and ICT kill-zone flags.

        Sessions (UTC)
        --------------
        asia        : 00:00 – 07:00
        london      : 07:00 – 13:00
        london_ny   : 13:00 – 16:00  (overlap)
        new_york    : 16:00 – 20:00
        new_york_pm : 20:00 – 22:00
        off_hours   : 22:00 – 00:00

        Kill zones (ICT standard)
        -------------------------
        London open : 07:00 – 08:30 UTC
        NY open     : 13:00 – 14:30 UTC
        """
        dt = pd.to_datetime(dataframe["date"], utc=True)
        mins = dt.dt.hour * 60 + dt.dt.minute   # 0 – 1439

        # Session buckets
        session = pd.Series("off_hours", index=dataframe.index, dtype=object)
        session[mins <  7 * 60]                           = "asia"
        session[(mins >= 7*60) & (mins < 13*60)]          = "london"
        session[(mins >= 13*60) & (mins < 16*60)]         = "london_ny"
        session[(mins >= 16*60) & (mins < 20*60)]         = "new_york"
        session[(mins >= 20*60) & (mins < 22*60)]         = "new_york_pm"
        dataframe["session"] = session.values

        # Kill zones
        in_lkz = (mins >= 7*60)  & (mins < 8*60 + 30)   # London open
        in_nykz = (mins >= 13*60) & (mins < 14*60 + 30)  # NY open
        dataframe["in_kill_zone"] = (in_lkz | in_nykz).values

        dow = dt.dt.dayofweek
        dataframe["is_monday"] = (dow == 0).values
        dataframe["is_friday"] = (dow == 4).values

        return dataframe

    def _add_entry_context(self, dataframe: DataFrame) -> DataFrame:
        """
        Extra entry context for LLM filter / post-analysis:
          - bars_since_{internal,swing}_choch_{bull,bear}: freshness of the structure break
          - atr_pct_rank: percentile rank of ATR(14) over last 100 bars (0-1)
        """
        def bars_since(series: pd.Series) -> pd.Series:
            grp = series.cumsum()
            return grp.groupby(grp).cumcount()

        for col in ("internal_choch_bullish", "internal_choch_bearish",
                    "swing_choch_bullish", "swing_choch_bearish"):
            if col in dataframe.columns:
                out = f"bars_since_{col.replace('_bullish','_bull').replace('_bearish','_bear')}"
                dataframe[out] = bars_since(dataframe[col].fillna(0).astype(int))

        if "atr" in dataframe.columns:
            atr = dataframe["atr"].astype(float)
        else:
            import talib.abstract as ta
            atr = ta.ATR(dataframe, timeperiod=14)
        dataframe["atr_pct_rank"] = atr.rolling(100, min_periods=20).rank(pct=True).fillna(0.5)

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
        internal_choch_bull = dataframe["internal_choch_bullish"] == 1
        swing_choch_bull = dataframe["swing_choch_bullish"] == 1

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
        internal_bos_bull = dataframe["internal_bos_bullish"] == 1
        swing_bos_bull = dataframe["swing_bos_bullish"] == 1

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
        if self.entry_signal_type.value in ["internal_only", "both"]:
            if self.require_choch.value:
                bullish_structure |= internal_choch_bull
            else:
                bullish_structure |= internal_choch_bull | internal_bos_bull

        # Swing Signals
        if self.entry_signal_type.value in ["swing_only", "both"]:
            if self.require_choch.value:
                bullish_structure |= swing_choch_bull
            else:
                bullish_structure |= swing_choch_bull | swing_bos_bull

        # ===== BEARISH STRUCTURE SIGNALS =====
        internal_choch_bear = dataframe["internal_choch_bearish"] == 1
        swing_choch_bear = dataframe["swing_choch_bearish"] == 1

        if lookback > 1:
            ic_bear_rolled = internal_choch_bear.rolling(lookback).max() == 1
            sc_bear_rolled = swing_choch_bear.rolling(lookback).max() == 1

            if not self.allow_immediate_entry.value:
                internal_choch_bear = ic_bear_rolled & (internal_choch_bear == False)
                swing_choch_bear = sc_bear_rolled & (swing_choch_bear == False)
            else:
                internal_choch_bear = ic_bear_rolled
                swing_choch_bear = sc_bear_rolled

        internal_bos_bear = dataframe["internal_bos_bearish"] == 1
        swing_bos_bear = dataframe["swing_bos_bearish"] == 1

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
        if self.entry_signal_type.value in ["internal_only", "both"]:
            if self.require_choch.value:
                bearish_structure |= internal_choch_bear
            else:
                bearish_structure |= internal_choch_bear | internal_bos_bear

        # Swing Signals
        if self.entry_signal_type.value in ["swing_only", "both"]:
            if self.require_choch.value:
                bearish_structure |= swing_choch_bear
            else:
                bearish_structure |= swing_choch_bear | swing_bos_bear

        # ===== ZONE CONDITIONS (Optional Filters) =====
        # Price in bullish OB zone (Must close ABOVE bottom)

        in_bull_ob = (
            (dataframe["active_bullish_ob_top"] > 0)
            & (dataframe["low"] <= dataframe["active_bullish_ob_top"])
            & (dataframe["high"] >= dataframe["active_bullish_ob_bottom"])
            & (dataframe["close"] >= dataframe["active_bullish_ob_bottom"])
        )

        # Apply RVOL Volumetric Filter to OBs
        if self.require_volumetric_ob.value:
            in_bull_ob &= dataframe["bull_ob_rvol"] >= self.ob_rvol_threshold.value

        dataframe["in_bullish_ob"] = in_bull_ob

        # Price in bearish OB zone (Must close BELOW top)
        in_bear_ob = (
            (dataframe["active_bearish_ob_top"] > 0)
            & (dataframe["high"] >= dataframe["active_bearish_ob_bottom"])
            & (dataframe["low"] <= dataframe["active_bearish_ob_top"])
            & (dataframe["close"] <= dataframe["active_bearish_ob_top"])
        )

        # Apply RVOL Volumetric Filter to OBs
        if self.require_volumetric_ob.value:
            in_bear_ob &= dataframe["bear_ob_rvol"] >= self.ob_rvol_threshold.value

        dataframe["in_bearish_ob"] = in_bear_ob

        # Price in bullish FVG zone (Must close ABOVE bottom to be valid)
        dataframe["in_bullish_fvg"] = (
            (dataframe["active_bullish_fvg_top"] > 0)
            & (dataframe["low"] <= dataframe["active_bullish_fvg_top"])
            & (dataframe["close"] >= dataframe["active_bullish_fvg_bottom"])
        )

        # Price in bearish FVG zone (Must close BELOW top to be valid)
        dataframe["in_bearish_fvg"] = (
            (dataframe["active_bearish_fvg_top"] > 0)
            & (dataframe["high"] >= dataframe["active_bearish_fvg_bottom"])
            & (dataframe["close"] <= dataframe["active_bearish_fvg_top"])
        )

        # Initialize Breaker columns (defaults to False)
        dataframe["in_bull_breaker"] = False
        dataframe["in_bear_breaker"] = False
        dataframe["in_bull_fvg_breaker"] = False
        dataframe["in_bear_fvg_breaker"] = False

        # Zone requirements
        # If NO zone filter is enabled, all bars pass (default True)
        any_zone_enabled = self.required_zone.value != "none"

        if any_zone_enabled:
            # Start with False, use OR to add conditions
            bullish_zone = pd.Series(False, index=dataframe.index)
            bearish_zone = pd.Series(False, index=dataframe.index)

            if self.required_zone.value in ["ob_only", "any"]:
                bullish_zone |= dataframe["in_bullish_ob"]
                bearish_zone |= dataframe["in_bearish_ob"]

                # Add Breakers to OB logic (User description: Breaker OB is a key entry point)
                if "active_bullish_breaker_top" in dataframe.columns:
                    in_bull_breaker = (
                        (dataframe["active_bullish_breaker_top"] > 0)
                        &
                        # Price touching breaker (Breaker acts as Support)
                        (dataframe["low"] <= dataframe["active_bullish_breaker_top"])
                        & (dataframe["high"] >= dataframe["active_bullish_breaker_bottom"])
                        & (dataframe["close"] >= dataframe["active_bullish_breaker_bottom"])
                    )

                    if self.require_volumetric_ob.value:
                        # A bullish breaker comes from a broken bearish OB. Use bearish RVOL.
                        in_bull_breaker &= dataframe["bear_ob_rvol"] >= self.ob_rvol_threshold.value

                    dataframe["in_bull_breaker"] = in_bull_breaker
                    bullish_zone |= dataframe["in_bull_breaker"]

                    in_bear_breaker = (
                        (dataframe["active_bearish_breaker_top"] > 0)
                        &
                        # Price touching breaker (Breaker acts as Resistance)
                        (dataframe["high"] >= dataframe["active_bearish_breaker_bottom"])
                        & (dataframe["low"] <= dataframe["active_bearish_breaker_top"])
                        & (dataframe["close"] <= dataframe["active_bearish_breaker_top"])
                    )

                    if self.require_volumetric_ob.value:
                        # A bearish breaker comes from a broken bullish OB. Use bullish RVOL.
                        in_bear_breaker &= dataframe["bull_ob_rvol"] >= self.ob_rvol_threshold.value

                    dataframe["in_bear_breaker"] = in_bear_breaker
                    bearish_zone |= dataframe["in_bear_breaker"]

            if self.required_zone.value in ["fvg_only", "any"]:
                bullish_zone |= dataframe["in_bullish_fvg"]
                bearish_zone |= dataframe["in_bearish_fvg"]

                # Add FVG Breakers logic
                if "active_bullish_fvg_breaker_top" in dataframe.columns:
                    dataframe["in_bull_fvg_breaker"] = (
                        (dataframe["active_bullish_fvg_breaker_top"] > 0)
                        & (dataframe["low"] <= dataframe["active_bullish_fvg_breaker_top"])
                        & (dataframe["high"] >= dataframe["active_bullish_fvg_breaker_bottom"])
                        & (dataframe["close"] >= dataframe["active_bullish_fvg_breaker_bottom"])
                    )
                    bullish_zone |= dataframe["in_bull_fvg_breaker"]

                    dataframe["in_bear_fvg_breaker"] = (
                        (dataframe["active_bearish_fvg_breaker_top"] > 0)
                        & (dataframe["high"] >= dataframe["active_bearish_fvg_breaker_bottom"])
                        & (dataframe["low"] <= dataframe["active_bearish_fvg_breaker_top"])
                        & (dataframe["close"] <= dataframe["active_bearish_fvg_breaker_top"])
                    )
                    bearish_zone |= dataframe["in_bear_fvg_breaker"]

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
            mask_bull_ob = dataframe["active_bullish_ob_bottom"] > 0
            bull_sl_price[mask_bull_ob] = dataframe.loc[mask_bull_ob, "active_bullish_ob_bottom"]

            # Bull Breaker Bottom (might overwrite if present - usually we touch one or other)
            if "active_bullish_breaker_bottom" in dataframe.columns:
                mask_bull_brk = dataframe["active_bullish_breaker_bottom"] > 0
                bull_sl_price = np.fmin(bull_sl_price, dataframe["active_bullish_breaker_bottom"])

            # FVG Breaker Bottom (NEW)
            if "active_bullish_fvg_breaker_bottom" in dataframe.columns:
                # If active, use its bottom
                bull_sl_price = np.fmin(
                    bull_sl_price, dataframe["active_bullish_fvg_breaker_bottom"]
                )

            # Apply offset
            dataframe["sl_long_price"] = bull_sl_price * (1 - self.sl_buffer_pct.value)

            # Same for Shorts: SL is ABOVE the bearish zone top (+ offset).
            # np.fmax keeps the highest top across zone types (most conservative SL for shorts).
            bear_sl_price = pd.Series(np.nan, index=dataframe.index)

            mask_bear_ob = dataframe["active_bearish_ob_top"] > 0
            bear_sl_price[mask_bear_ob] = dataframe.loc[mask_bear_ob, "active_bearish_ob_top"]

            if "active_bearish_breaker_top" in dataframe.columns:
                bear_sl_price = np.fmax(bear_sl_price, dataframe["active_bearish_breaker_top"])

            if "active_bearish_fvg_breaker_top" in dataframe.columns:
                bear_sl_price = np.fmax(
                    bear_sl_price, dataframe["active_bearish_fvg_breaker_top"]
                )

            dataframe["sl_short_price"] = bear_sl_price * (1 + self.sl_buffer_pct.value)

            # ===== LIQUIDITY SWEEP BYPASS =====
            # If a recent sweep is detected, bypass zone requirement
            # Sweep + CHoCH = high confidence entry without needing OB/FVG retest
            if self.use_sweep_bypass.value:
                sweep_lb = self.sweep_lookback.value

                # Recent bullish sweep (wick took sellside liquidity below pivot low)
                recent_bull_sweep = (
                    dataframe["internal_sweep_bullish"].rolling(sweep_lb, min_periods=1).max() == 1
                ) | (dataframe["swing_sweep_bullish"].rolling(sweep_lb, min_periods=1).max() == 1)

                # Recent bearish sweep (wick took buyside liquidity above pivot high)
                recent_bear_sweep = (
                    dataframe["internal_sweep_bearish"].rolling(sweep_lb, min_periods=1).max() == 1
                ) | (dataframe["swing_sweep_bearish"].rolling(sweep_lb, min_periods=1).max() == 1)

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
            if (n_bull_zones == 0 or n_bear_zones == 0) and (
                self.required_zone.value != "none"
            ):
                logger.warning(
                    f"⚠️ ZONES EMPTY! Bull: {n_bull_zones}, Bear: {n_bear_zones}. "
                    f"Zones: {self.required_zone.value}"
                )
                logger.warning(
                    f"   Active Bull OB Candles: {(dataframe['active_bullish_ob_top'] > 0).sum()}"
                )
                logger.warning(
                    f"   Active Bull FVG Candles: {(dataframe['active_bullish_fvg_top'] > 0).sum()}"
                )

        # ===== HTF TREND FILTER =====
        if self.trade_with_trend.value:
            # Local Trend Filter (MTF)
            bullish_trend_ok = dataframe["swing_trend"] == 1
            bearish_trend_ok = dataframe["swing_trend"] == -1

            # HTF Filter - Check ALL configured timeframes
            if self.htf_1.value != "none":
                htf_list = self._get_active_htf_list()
                # Per-HTF internal trend toggle mapping
                htf_internal_map = {
                    0: self.htf_1_use_internal.value,
                    1: self.htf_2_use_internal.value,
                }
                for idx, htf in enumerate(htf_list):
                    use_internal = htf_internal_map.get(idx, False)
                    trend_type = "internal_trend" if use_internal else "swing_trend"
                    trend_col = f"{htf}_{trend_type}"

                    if trend_col in dataframe.columns:
                        htf_bull = dataframe[trend_col] == 1
                        htf_bear = dataframe[trend_col] == -1

                        # AND logic: ALL HTFs must align
                        bullish_trend_ok &= htf_bull
                        bearish_trend_ok &= htf_bear
        else:
            bullish_trend_ok = True
            bearish_trend_ok = True

        # ===== HTF ZONE FILTER =====
        # Price must overlap with an active HTF zone (OB/FVG/Breaker/FVG-Breaker).
        # This is the architectural core: 15m CHoCH is the trigger; the HTF zone
        # is the "why" (institutional presence). Without a zone, the signal has no
        # structural backing at the higher timeframe.
        htf_bull_zone_ok = pd.Series(True, index=dataframe.index)
        htf_bear_zone_ok = pd.Series(True, index=dataframe.index)

        if self.require_htf_zone.value == "enabled" and self.trade_with_trend.value:
            htf_list = self._get_active_htf_list()
            if htf_list:
                htf_bull_zone_ok = pd.Series(False, index=dataframe.index)
                htf_bear_zone_ok = pd.Series(False, index=dataframe.index)

                for htf in htf_list:
                    bull_zone_pairs = [
                        (f"{htf}_bull_ob_top",   f"{htf}_bull_ob_bot"),
                        (f"{htf}_bull_fvg_top",  f"{htf}_bull_fvg_bot"),
                        (f"{htf}_bull_brk_top",  f"{htf}_bull_brk_bot"),
                        (f"{htf}_bull_fbrk_top", f"{htf}_bull_fbrk_bot"),
                    ]
                    bear_zone_pairs = [
                        (f"{htf}_bear_ob_top",   f"{htf}_bear_ob_bot"),
                        (f"{htf}_bear_fvg_top",  f"{htf}_bear_fvg_bot"),
                        (f"{htf}_bear_brk_top",  f"{htf}_bear_brk_bot"),
                        (f"{htf}_bear_fbrk_top", f"{htf}_bear_fbrk_bot"),
                    ]

                    for top_col, bot_col in bull_zone_pairs:
                        if top_col in dataframe.columns and bot_col in dataframe.columns:
                            zone_active = dataframe[top_col] > 0
                            price_in_zone = (
                                (dataframe["low"] <= dataframe[top_col])
                                & (dataframe["close"] >= dataframe[bot_col])
                            )
                            htf_bull_zone_ok |= zone_active & price_in_zone

                    for top_col, bot_col in bear_zone_pairs:
                        if top_col in dataframe.columns and bot_col in dataframe.columns:
                            zone_active = dataframe[top_col] > 0
                            price_in_zone = (
                                (dataframe["high"] >= dataframe[bot_col])
                                & (dataframe["close"] <= dataframe[top_col])
                            )
                            htf_bear_zone_ok |= zone_active & price_in_zone

        # ===== DYNAMIC TP1 ADJUSTMENT (per-candle, stored as column) =====
        # Instead of blocking entries, adapt TP1/BE to the nearest opposing zone.
        # Long: if bearish OB/FVG is between price and default TP1 → take partial just before it.
        # Short: if bullish OB/FVG is between price and default TP1 → take partial just before it.
        # The adjusted value is stored in the dataframe so custom_stoploss/adjust_trade_position
        # can read it once at trade open and store it as custom trade data.

        default_tp1 = self.tp1_pct.value
        margin = self.tp_adj_margin.value
        min_rvol = self.tp_adj_min_rvol.value

        # --- Long TP1 adjustment ---
        bear_ob_dist_raw = (dataframe["active_bearish_ob_bottom"] - dataframe["close"]) / dataframe[
            "close"
        ]
        has_bear_ob_in_range = (
            (dataframe["active_bearish_ob_bottom"] > dataframe["close"])
            & (bear_ob_dist_raw > 0)
            & (bear_ob_dist_raw < default_tp1)
            & (dataframe["bear_ob_rvol"] >= min_rvol)
        )
        bear_fvg_dist_raw = (
            dataframe["active_bearish_fvg_bottom"] - dataframe["close"]
        ) / dataframe["close"]
        has_bear_fvg_in_range = (
            (dataframe["active_bearish_fvg_bottom"] > dataframe["close"])
            & (bear_fvg_dist_raw > 0)
            & (bear_fvg_dist_raw < default_tp1)
        )
        # Pick the nearest obstacle (min dist), apply margin
        nearest_long_obstacle = pd.Series(
            np.where(
                has_bear_ob_in_range & has_bear_fvg_in_range,
                np.minimum(bear_ob_dist_raw, bear_fvg_dist_raw) * margin,
                np.where(
                    has_bear_ob_in_range,
                    bear_ob_dist_raw * margin,
                    np.where(has_bear_fvg_in_range, bear_fvg_dist_raw * margin, default_tp1),
                ),
            ),
            index=dataframe.index,
        )
        dataframe["tp1_adj_long"] = nearest_long_obstacle

        # --- Short TP1 adjustment ---
        bull_ob_dist_raw = (dataframe["close"] - dataframe["active_bullish_ob_top"]) / dataframe[
            "close"
        ]
        has_bull_ob_in_range = (
            (dataframe["active_bullish_ob_top"] > 0)
            & (dataframe["active_bullish_ob_top"] < dataframe["close"])
            & (bull_ob_dist_raw > 0)
            & (bull_ob_dist_raw < default_tp1)
            & (dataframe["bull_ob_rvol"] >= min_rvol)
        )
        bull_fvg_dist_raw = (dataframe["close"] - dataframe["active_bullish_fvg_top"]) / dataframe[
            "close"
        ]
        has_bull_fvg_in_range = (
            (dataframe["active_bullish_fvg_top"] > 0)
            & (dataframe["active_bullish_fvg_top"] < dataframe["close"])
            & (bull_fvg_dist_raw > 0)
            & (bull_fvg_dist_raw < default_tp1)
        )
        nearest_short_obstacle = pd.Series(
            np.where(
                has_bull_ob_in_range & has_bull_fvg_in_range,
                np.minimum(bull_ob_dist_raw, bull_fvg_dist_raw) * margin,
                np.where(
                    has_bull_ob_in_range,
                    bull_ob_dist_raw * margin,
                    np.where(has_bull_fvg_in_range, bull_fvg_dist_raw * margin, default_tp1),
                ),
            ),
            index=dataframe.index,
        )
        dataframe["tp1_adj_short"] = nearest_short_obstacle

        # ===== FINAL CONDITIONS =====
        long_condition = bullish_structure & bullish_zone & bullish_trend_ok & htf_bull_zone_ok
        short_condition = bearish_structure & bearish_zone & bearish_trend_ok & htf_bear_zone_ok

        # ===== ML FILTER =====
        ml_model = self._get_ml_model_for_pair(metadata["pair"])
        if self.use_ml_filter.value and ml_model:
            try:
                # Use optimal threshold from training if available, else fall back to param
                effective_threshold = (
                    getattr(self, "_ml_optimal_threshold", None) or self.ml_threshold.value
                )

                # Prepare Features
                feature_cols = [c for c in dataframe.columns if c.startswith("ml_")]

                # Predict Longs (direction = 1)
                X_long = dataframe[feature_cols].copy()
                X_long["direction"] = 1
                long_proba = ml_model.predict_proba(X_long)[:, 1]
                ml_long_ok = long_proba > effective_threshold

                # Predict Shorts (direction = -1)
                X_short = dataframe[feature_cols].copy()
                X_short["direction"] = -1
                short_proba = ml_model.predict_proba(X_short)[:, 1]
                ml_short_ok = short_proba > effective_threshold

                # Apply Filter
                long_signals_count = long_condition.sum()
                short_signals_count = short_condition.sum()

                long_condition &= ml_long_ok
                short_condition &= ml_short_ok

                # Log only if a signal was blocked
                blocked_long = long_signals_count - long_condition.sum()
                blocked_short = short_signals_count - short_condition.sum()

                if blocked_long > 0 or blocked_short > 0:
                    logger.info(
                        f"🤖 ML Filter ({metadata['pair']}) Blocked {blocked_long} Longs, {blocked_short} Shorts. Threshold: {effective_threshold:.4f}"
                    )

            except Exception as e:
                logger.error(f"❌ ML Inference Failed for {metadata['pair']}: {e}")
                pass

        # ===== LLM CONFLUENCE FILTER / SHADOW =====
        # Moved to confirm_trade_entry: LLM now runs once per confirmed trade
        # instead of iterating every signal candle on every refresh cycle.

        # ===== MACRO REGIME FILTER (BTC Weekly EMA21) =====
        if self.use_macro_filter.value == "enabled":
            macro_long_ok  = dataframe["btc_macro_bias"] >= 0   # bull or neutral → longs allowed
            macro_short_ok = dataframe["btc_macro_bias"] <= 0   # bear or neutral → shorts allowed

            blocked_macro_long  = long_condition.sum()  - (long_condition  & macro_long_ok).sum()
            blocked_macro_short = short_condition.sum() - (short_condition & macro_short_ok).sum()

            long_condition  = long_condition  & macro_long_ok
            short_condition = short_condition & macro_short_ok

            if (blocked_macro_long > 0 or blocked_macro_short > 0) and self.enable_logging.value:
                logger.info(
                    f"🌍 Macro Filter ({metadata['pair']}) blocked "
                    f"{blocked_macro_long} longs, {blocked_macro_short} shorts. "
                    f"BTC macro bias: {dataframe['btc_macro_bias'].iloc[-1]}"
                )

        # Apply signals
        dataframe.loc[long_condition, "enter_long"] = 1
        dataframe.loc[short_condition, "enter_short"] = 1

        # --- Populate enter_tag String ---
        # Freqtrade expects the string directly in 'enter_tag' column.
        dataframe["enter_tag"] = ""

        # Long Tags
        long_mask = long_condition
        dataframe.loc[long_mask & dataframe["in_bullish_ob"], "enter_tag"] += "OB+"
        dataframe.loc[long_mask & dataframe["in_bull_breaker"], "enter_tag"] += "BRK+"
        dataframe.loc[long_mask & dataframe["in_bullish_fvg"], "enter_tag"] += "FVG+"
        dataframe.loc[long_mask & dataframe["in_bull_fvg_breaker"], "enter_tag"] += "FVGBRK+"
        dataframe.loc[
            long_mask
            & ~dataframe["in_bullish_ob"]
            & ~dataframe["in_bull_breaker"]
            & ~dataframe["in_bullish_fvg"]
            & ~dataframe["in_bull_fvg_breaker"],
            "enter_tag",
        ] = "NoZone"

        # Short Tags
        short_mask = short_condition
        dataframe.loc[short_mask & dataframe["in_bearish_ob"], "enter_tag"] += "OB+"
        dataframe.loc[short_mask & dataframe["in_bear_breaker"], "enter_tag"] += "BRK+"
        dataframe.loc[short_mask & dataframe["in_bearish_fvg"], "enter_tag"] += "FVG+"
        dataframe.loc[short_mask & dataframe["in_bear_fvg_breaker"], "enter_tag"] += "FVGBRK+"
        dataframe.loc[
            short_mask
            & ~dataframe["in_bearish_ob"]
            & ~dataframe["in_bear_breaker"]
            & ~dataframe["in_bearish_fvg"]
            & ~dataframe["in_bear_fvg_breaker"],
            "enter_tag",
        ] = "NoZone"

        # Clean trailing '+'
        dataframe["enter_tag"] = dataframe["enter_tag"].str.rstrip("+")

        # === GRANULAR CONTEXT TAGS (components A-F) ===
        # Each component appends a "+TAG" suffix to the existing zone tag.
        # All operations are vectorized; no Python loops.
        dataframe = self._append_context_tags(
            dataframe, long_condition, short_condition
        )

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
                if row.get("internal_choch_bearish", 0) == 1:
                    signal_type = "IntCHoCH"
                    bars_ago = 0
                    signal_close = row["close"]
                elif row.get("swing_choch_bearish", 0) == 1:
                    signal_type = "SwingCHoCH"
                    bars_ago = 0
                    signal_close = row["close"]
                else:
                    # Look back
                    # This is slow in a loop but fine for logging only
                    # We need the integer position of 'idx' in 'dataframe'
                    try:
                        i = dataframe.index.get_loc(idx)
                        for k in range(1, lookback + 1):
                            if i - k >= 0:
                                prev_row = dataframe.iloc[i - k]
                                if prev_row["internal_choch_bearish"] == 1:
                                    signal_type = "IntCHoCH"
                                    bars_ago = k
                                    signal_close = prev_row["close"]
                                    break
                                elif prev_row["swing_choch_bearish"] == 1:
                                    signal_type = "SwingCHoCH"
                                    bars_ago = k
                                    signal_close = prev_row["close"]
                                    break
                    except Exception:
                        pass

                reasons = []
                if row.get("in_bearish_ob", False):
                    reasons.append("OB")
                if row.get("in_bearish_fvg", False):
                    reasons.append("FVG")
                if row.get("in_bear_breaker", False):
                    reasons.append("Breaker")
                if row.get("in_bear_fvg_breaker", False):
                    reasons.append("FVG-Breaker")

                reason_str = "+".join(reasons) if reasons else "StructureOnly"

                bear_fvg = (
                    f"[{row.get('active_bearish_fvg_bottom', 0):.8f}, {row.get('active_bearish_fvg_top', 0):.8f}]"
                    if row.get("active_bearish_fvg_top", 0) > 0
                    else "None"
                )
                bear_ob = (
                    f"[{row.get('active_bearish_ob_bottom', 0):.8f}, {row.get('active_bearish_ob_top', 0):.8f}]"
                    if row.get("active_bearish_ob_top", 0) > 0
                    else "None"
                )
                bear_brk = (
                    f"[{row.get('active_bearish_breaker_bottom', 0):.8f}, {row.get('active_bearish_breaker_top', 0):.8f}]"
                    if row.get("active_bearish_breaker_top", 0) > 0
                    else "None"
                )
                bear_fvg_brk = (
                    f"[{row.get('active_bearish_fvg_breaker_bottom', 0):.8f}, {row.get('active_bearish_fvg_breaker_top', 0):.8f}]"
                    if row.get("active_bearish_fvg_breaker_top", 0) > 0
                    else "None"
                )

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

                if row.get("internal_choch_bullish", 0) == 1:
                    signal_type = "IntCHoCH"
                    bars_ago = 0
                    signal_close = row["close"]
                elif row.get("swing_choch_bullish", 0) == 1:
                    signal_type = "SwingCHoCH"
                    bars_ago = 0
                    signal_close = row["close"]
                else:
                    try:
                        i = dataframe.index.get_loc(idx)
                        for k in range(1, lookback + 1):
                            if i - k >= 0:
                                prev_row = dataframe.iloc[i - k]
                                if prev_row["internal_choch_bullish"] == 1:
                                    signal_type = "IntCHoCH"
                                    bars_ago = k
                                    signal_close = prev_row["close"]
                                    break
                                elif prev_row["swing_choch_bullish"] == 1:
                                    signal_type = "SwingCHoCH"
                                    bars_ago = k
                                    signal_close = prev_row["close"]
                                    break
                    except Exception:
                        pass

                reasons = []
                if row.get("in_bullish_ob", False):
                    reasons.append("OB")
                if row.get("in_bullish_fvg", False):
                    reasons.append("FVG")
                if row.get("in_bull_breaker", False):
                    reasons.append("Breaker")
                if row.get("in_bull_fvg_breaker", False):
                    reasons.append("FVG-Breaker")

                reason_str = "+".join(reasons) if reasons else "StructureOnly"

                bull_fvg = (
                    f"[{row.get('active_bullish_fvg_bottom', 0):.8f}, {row.get('active_bullish_fvg_top', 0):.8f}]"
                    if row.get("active_bullish_fvg_top", 0) > 0
                    else "None"
                )
                bull_ob = (
                    f"[{row.get('active_bullish_ob_bottom', 0):.8f}, {row.get('active_bullish_ob_top', 0):.8f}]"
                    if row.get("active_bullish_ob_top", 0) > 0
                    else "None"
                )
                bull_brk = (
                    f"[{row.get('active_bullish_breaker_bottom', 0):.8f}, {row.get('active_bullish_breaker_top', 0):.8f}]"
                    if row.get("active_bullish_breaker_top", 0) > 0
                    else "None"
                )
                bull_fvg_brk = (
                    f"[{row.get('active_bullish_fvg_breaker_bottom', 0):.8f}, {row.get('active_bullish_fvg_breaker_top', 0):.8f}]"
                    if row.get("active_bullish_fvg_breaker_top", 0) > 0
                    else "None"
                )

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
    # GRANULAR CONTEXT TAGS  (Tarea 2)
    # ==========================================================================

    def _append_context_tags(
        self,
        dataframe: DataFrame,
        long_condition: pd.Series,
        short_condition: pd.Series,
    ) -> DataFrame:
        """
        Append diagnostic context suffixes to enter_tag for every entry signal.

        Components
        ----------
        A  OB quality  : OB_FRESH / OB_TOUCHED / OB_MITIGATED
        B  P/D context : PD_DEEP_DISC / PD_DISC / PD_NEUTRAL / PD_PREM / PD_DEEP_PREM
        C  Session     : KZ_LDN / KZ_NY / SESSION / OFF
        D  FVG quality : FVG_CLEAN / FVG_PARTIAL / FVG_STALE
        E  Structure   : SWEEP / HTF_ALIGN / HTF_PART / HTF_CONF
        F  Day         : MON / FRI  (nothing on Tue/Wed/Thu)

        Each active component is joined with '+'. The result is appended to the
        existing zone tag: "OB" + "+OB_FRESH+PD_DISC+KZ_NY+..." = full tag.
        """
        idx = dataframe.index
        lc  = long_condition
        sc  = short_condition
        any_entry = lc | sc

        def _safe(col: str, default=0.0) -> pd.Series:
            """Return column or a constant Series when the column is absent."""
            if col in dataframe.columns:
                return dataframe[col].fillna(default)
            return pd.Series(default, index=idx, dtype=float)

        def _tag(mask: pd.Series, label: str) -> None:
            """Append '+label' to enter_tag wherever mask is True."""
            dataframe.loc[mask, "enter_tag"] += f"+{label}"

        # ------------------------------------------------------------------
        # A. OB quality (applies when entry zone is OB or Breaker)
        # ------------------------------------------------------------------
        in_ob_long  = lc & (
            dataframe.get("in_bullish_ob", pd.Series(False, index=idx)).fillna(False)
            | dataframe.get("in_bull_breaker", pd.Series(False, index=idx)).fillna(False)
        )
        in_ob_short = sc & (
            dataframe.get("in_bearish_ob", pd.Series(False, index=idx)).fillna(False)
            | dataframe.get("in_bear_breaker", pd.Series(False, index=idx)).fillna(False)
        )

        # Pull directional OB metrics (slot 0 = most recent active OB)
        ob_age     = pd.Series(99.0, index=idx)
        ob_touches = pd.Series(0.0,  index=idx)
        ob_mit     = pd.Series(0.0,  index=idx)
        ob_age[lc]     = _safe("bull_ob_0_age",       99.0)[lc]
        ob_age[sc]     = _safe("bear_ob_0_age",       99.0)[sc]
        ob_touches[lc] = _safe("bull_ob_0_touches",   0.0)[lc]
        ob_touches[sc] = _safe("bear_ob_0_touches",   0.0)[sc]
        ob_mit[lc]     = _safe("bull_ob_0_mitigated", 0.0)[lc]
        ob_mit[sc]     = _safe("bear_ob_0_mitigated", 0.0)[sc]

        in_ob = in_ob_long | in_ob_short
        # Priority: OB_MITIGATED > OB_TOUCHED > OB_FRESH > (no tag)
        _tag(in_ob & (ob_mit == 1),                              "OB_MITIGATED")
        _tag(in_ob & (ob_mit == 0) & (ob_touches >= 1),         "OB_TOUCHED")
        _tag(in_ob & (ob_mit == 0) & (ob_touches == 0) & (ob_age < 20), "OB_FRESH")

        # ------------------------------------------------------------------
        # B. Premium/Discount context (daily range, primary source)
        # ------------------------------------------------------------------
        daily_eq  = _safe("daily_equilibrium", 0.0)
        close_col = dataframe["close"]
        daily_pd  = _safe("daily_prem_disc_label", "discount").astype(str)

        # Neutral overrides the label: price within 5% of daily equilibrium
        neutral_mask = any_entry & (daily_eq > 0) & (
            (close_col - daily_eq).abs() / daily_eq.clip(lower=1e-10) < 0.05
        )
        _tag(any_entry & ~neutral_mask & (daily_pd == "deep_discount"), "PD_DEEP_DISC")
        _tag(any_entry & ~neutral_mask & (daily_pd == "discount"),      "PD_DISC")
        _tag(neutral_mask,                                               "PD_NEUTRAL")
        _tag(any_entry & ~neutral_mask & (daily_pd == "premium"),       "PD_PREM")
        _tag(any_entry & ~neutral_mask & (daily_pd == "deep_premium"),  "PD_DEEP_PREM")

        # ------------------------------------------------------------------
        # C. Session context
        # ------------------------------------------------------------------
        session = _safe("session", "off_hours").astype(str)
        in_kz   = _safe("in_kill_zone", 0.0).astype(bool)

        kz_ldn = any_entry & in_kz & (session == "london")
        kz_ny  = any_entry & in_kz & (
            (session == "london_ny") | (session == "new_york")
        )
        active  = any_entry & ~in_kz & (session != "off_hours")
        off_hrs = any_entry & (session == "off_hours")

        _tag(kz_ldn,  "KZ_LDN")
        _tag(kz_ny,   "KZ_NY")
        _tag(active,  "SESSION")
        _tag(off_hrs, "OFF")

        # ------------------------------------------------------------------
        # D. FVG quality (applies when entry zone includes FVG or FVG-Breaker)
        # ------------------------------------------------------------------
        in_fvg_long  = lc & (
            dataframe.get("in_bullish_fvg",    pd.Series(False, index=idx)).fillna(False)
            | dataframe.get("in_bull_fvg_breaker", pd.Series(False, index=idx)).fillna(False)
        )
        in_fvg_short = sc & (
            dataframe.get("in_bearish_fvg",    pd.Series(False, index=idx)).fillna(False)
            | dataframe.get("in_bear_fvg_breaker", pd.Series(False, index=idx)).fillna(False)
        )
        in_fvg = in_fvg_long | in_fvg_short

        fvg_filled = pd.Series(0.0, index=idx)
        fvg_rvol   = pd.Series(0.0, index=idx)
        fvg_filled[lc] = _safe("bull_fvg_0_filled_pct", 0.0)[lc]
        fvg_filled[sc] = _safe("bear_fvg_0_filled_pct", 0.0)[sc]
        fvg_rvol[lc]   = _safe("bull_fvg_0_rvol", 0.0)[lc]
        fvg_rvol[sc]   = _safe("bear_fvg_0_rvol", 0.0)[sc]

        _tag(in_fvg & (fvg_filled < 0.20) & (fvg_rvol > 1.5),          "FVG_CLEAN")
        _tag(in_fvg & (fvg_filled >= 0.20) & (fvg_filled < 0.70),      "FVG_PARTIAL")
        _tag(in_fvg & (fvg_filled >= 0.70),                              "FVG_STALE")

        # ------------------------------------------------------------------
        # E. Structural context
        # ------------------------------------------------------------------
        # SWEEP: a liquidity sweep of the appropriate direction occurred recently
        sweep_lb = self.sweep_lookback.value
        bull_sweep = (
            (_safe("internal_sweep_bullish", 0).rolling(sweep_lb, min_periods=1).max() > 0)
            | (_safe("swing_sweep_bullish",  0).rolling(sweep_lb, min_periods=1).max() > 0)
        )
        bear_sweep = (
            (_safe("internal_sweep_bearish", 0).rolling(sweep_lb, min_periods=1).max() > 0)
            | (_safe("swing_sweep_bearish",  0).rolling(sweep_lb, min_periods=1).max() > 0)
        )
        _tag((lc & bull_sweep) | (sc & bear_sweep), "SWEEP")

        # HTF trend alignment: compare each configured HTF trend to MTF swing_trend
        mtf_trend = _safe("swing_trend", 0)
        htf_list  = self._get_active_htf_list()

        # Primary HTF = highest timeframe (rightmost after sort by minutes)
        _TF_MIN = {"5m": 5, "15m": 15, "30m": 30, "1h": 60,
                   "2h": 120, "4h": 240, "8h": 480, "12h": 720, "1d": 1440}
        htf_sorted = sorted(
            [tf for tf in htf_list if f"{tf}_swing_trend" in dataframe.columns],
            key=lambda t: _TF_MIN.get(t, 0),
            reverse=True,
        )

        if htf_sorted:
            primary_htf_trend = _safe(f"{htf_sorted[0]}_swing_trend", 0)
            # Count how many HTFs agree with MTF direction
            agree = sum(
                ((_safe(f"{tf}_swing_trend", 0) == mtf_trend).astype(int))
                for tf in htf_sorted
            )
            n_htfs = len(htf_sorted)

            # Entry direction: +1 for long, -1 for short
            entry_dir = pd.Series(0.0, index=idx)
            entry_dir[lc] =  1.0
            entry_dir[sc] = -1.0

            # HTF_CONF: primary HTF trend is opposite to the entry direction
            htf_conf = any_entry & (primary_htf_trend * entry_dir < 0)
            htf_align = any_entry & ~htf_conf & (agree == n_htfs)
            htf_part  = any_entry & ~htf_conf & ~htf_align & (agree > 0)

            _tag(htf_align, "HTF_ALIGN")
            _tag(htf_part,  "HTF_PART")
            _tag(htf_conf,  "HTF_CONF")

        # ------------------------------------------------------------------
        # F. Day of week (only Mon and Fri are tagged; Tue/Wed/Thu silent)
        # ------------------------------------------------------------------
        is_mon = _safe("is_monday", 0).astype(bool)
        is_fri = _safe("is_friday", 0).astype(bool)
        _tag(any_entry & is_mon, "MON")
        _tag(any_entry & is_fri, "FRI")

        return dataframe

    # ==========================================================================
    # EXIT LOGIC
    # ==========================================================================

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Generate exit signals on opposite CHoCH (trend reversal)."""

        if True:  # Always check for exits if parameters are enabled
            # Exit long on bearish CHoCH (trend reversal)
            # Check configured exit triggers
            exit_long = pd.Series(False, index=dataframe.index)
            if self.exit_trend_type.value in ["internal", "both"]:
                exit_long |= dataframe["internal_choch_bearish"] == 1
            if self.exit_trend_type.value in ["swing", "both"]:
                exit_long |= dataframe["swing_choch_bearish"] == 1

            # Exit short on bullish CHoCH
            exit_short = pd.Series(False, index=dataframe.index)
            if self.exit_trend_type.value in ["internal", "both"]:
                exit_short |= dataframe["internal_choch_bullish"] == 1
            if self.exit_trend_type.value in ["swing", "both"]:
                exit_short |= dataframe["swing_choch_bullish"] == 1
        else:
            exit_long = pd.Series(False, index=dataframe.index)
            exit_short = pd.Series(False, index=dataframe.index)

        dataframe.loc[exit_long, "exit_long"] = 1
        dataframe.loc[exit_short, "exit_short"] = 1

        return dataframe

    # ==========================================================================
    # LEVERAGE & RISK MANAGEMENT
    # ==========================================================================

    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: str,
        side: str,
        **kwargs,
    ) -> float:
        """Get leverage from config. Default matches bot_start to avoid stoploss miscalculation."""
        return self.config.get("leverage", 1.0)

    def _get_timeframe_minutes(self) -> int:
        """Convert timeframe string to minutes."""
        tf = self.timeframe
        if tf.endswith("m"):
            return int(tf[:-1])
        elif tf.endswith("h"):
            return int(tf[:-1]) * 60
        elif tf.endswith("d"):
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
        last_check = getattr(self, "_panic_last_check", None)
        if last_check and (current_time - last_check).total_seconds() < 60:
            return getattr(self, "_panic_active", False)

        # Perform check
        self._panic_last_check = current_time
        self._panic_active = False  # Default

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
                    if (
                        t.exit_reason
                        in ["stop_loss", "stoploss_on_exchange", "force_exit", "emergency_exit"]
                    ) and (t.close_profit < 0):
                        sl_count += 1

        if sl_count >= PANIC_SL_LIMIT:
            self._panic_active = True
            if self.enable_logging.value:
                logger.warning(
                    f"🚨 CIRCUIT BREAKER ACTIVE: {sl_count} Losses in last {PANIC_WINDOW_MIN}m. Market is volatile."
                )

        return self._panic_active

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> bool:
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
                logger.info(
                    f"⛔ Entry blocked for {pair} due to Circuit Breaker (High Volatility/Panic)."
                )
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

        # ── Context-based entry filters (driven by backtest analysis) ──────────
        tag = entry_tag or ""

        # KZ_LDN: London kill zone, avg profit ~+1% vs +9.5% rest
        if self.filter_block_kz_ldn.value and "KZ_LDN" in tag:
            if self.enable_logging.value:
                logger.info(f"Entry blocked for {pair}: KZ_LDN filter")
            return False

        # OFF: off-hours session, lower win rate vs SESSION
        if self.filter_block_off_hours.value and "+OFF" in tag:
            if self.enable_logging.value:
                logger.info(f"Entry blocked for {pair}: OFF hours filter")
            return False

        # SWEEP: liquidity sweep entries, WR 38.3% consistently below average
        if self.filter_block_sweep.value and "+SWEEP" in tag:
            if self.enable_logging.value:
                logger.info(f"Entry blocked for {pair}: SWEEP filter")
            return False

        # FVG_STALE + HTF_PART combo: 17 trades avg +0.15%, not worth the risk
        if (self.filter_block_fvg_stale_htf_part.value
                and "FVG_STALE" in tag and "HTF_PART" in tag):
            if self.enable_logging.value:
                logger.info(f"Entry blocked for {pair}: FVG_STALE+HTF_PART combo filter")
            return False

        # ── LLM confluence filter (once per confirmed trade) ───────────────────
        # Runs here instead of populate_entry_trend to avoid per-candle LLM calls.
        # Fail-open: API/parse errors allow the trade through.
        if (self.use_llm_filter.value or self.use_llm_shadow.value) and self._llm_filter is not None:
            try:
                df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
                if df is not None and len(df) > 0:
                    direction = "LONG" if side == "long" else "SHORT"
                    signal_col = "enter_long" if side == "long" else "enter_short"

                    # Signal may be up to `entry_signal_lookback` bars old.
                    lookback = int(self.entry_signal_lookback.value)
                    tail = df.tail(lookback + 2)
                    signal_rows = tail[tail.get(signal_col, 0) == 1]
                    row = signal_rows.iloc[-1] if not signal_rows.empty else df.iloc[-1]

                    allow, confidence, reason = self._llm_filter.evaluate_trade(
                        pair, row, direction
                    )

                    if confidence is not None:
                        mode = "SHADOW" if self.use_llm_shadow.value else "FILTER"
                        thr = self.llm_confidence_threshold.value
                        if (not allow) and self.use_llm_filter.value:
                            logger.info(
                                f"⛔ LLM [{mode}] blocked {direction} {pair}: "
                                f"conf={confidence:.2f} < {thr:.2f} | {reason[:120]}"
                            )
                            return False
                        if self.enable_logging.value:
                            logger.info(
                                f"LLM [{mode}] {direction} {pair} tag={tag} "
                                f"conf={confidence:.2f} thr={thr:.2f} | {reason[:100]}"
                            )
            except Exception as e:
                logger.error(f"LLM filter failed in confirm_trade_entry for {pair}: {e}")
                # fail-open

        return True

    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: Optional[float],
        max_stake: float,
        leverage: float,
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> float:
        """Reduce stake on Fridays (WR 39.6%, avg +5.5% vs +9.5% non-Friday)."""
        if self.filter_friday_reduce_stake.value and current_time.weekday() == 4:
            reduced = proposed_stake * float(self.filter_friday_stake_ratio.value)
            if min_stake and reduced < min_stake:
                reduced = min_stake
            return reduced
        return proposed_stake

    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> float:
        """
        Break Even Logic with persistent state.
        Once BE is activated (when price reaches TP1), the stoploss is fixed at entry price + small buffer for fees.
        """
        if not self.move_be_at_tp1.value and not self.use_custom_stoploss:
            return 1  # Use default stoploss

        # --- 1. INITIAL DYNAMIC STOPLOSS (At Entry) ---
        # If trade just opened (no BE yet), we check for Dynamic SL from structure
        be_activated = trade.get_custom_data("be_activated", default=False)

        # --- 1a. STORE DYNAMIC TP1 ADJUSTMENT (independent of dynamic SL) ---
        # Read tp1_adj from the entry candle once and cache it as custom trade data.
        if not be_activated and self.use_dynamic_tp_adj.value and trade.get_custom_data("tp1_adj") is None:
            try:
                dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
                candle = dataframe.loc[dataframe["date"] == trade.open_date_utc]
                if candle.empty:
                    candle = dataframe.iloc[[-1]]
                if not candle.empty:
                    row = candle.iloc[0]
                    adj_col = "tp1_adj_short" if trade.is_short else "tp1_adj_long"
                    if adj_col in row and not pd.isna(row[adj_col]) and row[adj_col] > 0:
                        trade.set_custom_data("tp1_adj", float(row[adj_col]))
                        if row[adj_col] < self.tp1_pct.value:
                            logger.info(
                                f"🎯 Dynamic TP1 for {pair}: {row[adj_col]:.2%} (opposing zone detected, default was {self.tp1_pct.value:.2%})"
                            )
            except Exception:
                pass

        # --- 2. BREAK EVEN LOGIC (BEFORE dynamic SL — priority once TP1 hit) ---
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
        # Priority: 1) be_trigger_pct if set  2) dynamic tp1_adj  3) tp1_pct default
        if self.be_trigger_pct.value > 0:
            be_trigger = self.be_trigger_pct.value
        elif self.use_dynamic_tp_adj.value:
            be_trigger = trade.get_custom_data("tp1_adj", default=self.tp1_pct.value)
        else:
            be_trigger = self.tp1_pct.value

        if self.move_be_at_tp1.value and price_movement >= be_trigger:
            should_activate_be = True
        elif market_panic and price_movement >= 0.005:  # Panic: Force BE at 0.5% profit
            should_activate_be = True
            if self.enable_logging.value and not be_activated:
                logger.info(f"🚨 Panic Mode: Forcing BE for {pair} at {price_movement:.2%} profit.")

        if not be_activated and should_activate_be:
            # Calculate and STORE the BE stop price ONCE
            fee_buffer_pct = 0.001  # 0.1% price buffer to cover fees

            if trade.is_short:
                be_stop_price = trade.open_rate * (1 - fee_buffer_pct)
            else:
                be_stop_price = trade.open_rate * (1 + fee_buffer_pct)

            trade.set_custom_data("be_activated", True)
            trade.set_custom_data("be_stop_price", be_stop_price)
            be_activated = True
            if self.enable_logging.value:
                logger.info(
                    f"BE activated for {pair} at price move {price_movement:.2%}. Stop set at {be_stop_price:.4f} (entry: {trade.open_rate:.4f})"
                )

        # If BE is activated, use the STORED stop price — this takes priority over dynamic SL
        if be_activated:
            be_stop_price = trade.get_custom_data("be_stop_price", default=trade.open_rate)

            if self.enable_logging.value:
                logger.info(
                    f"BE SL for {pair}: direction={'SHORT' if trade.is_short else 'LONG'}, "
                    f"be_stop_price={be_stop_price:.6f}, open_rate={trade.open_rate:.6f}, "
                    f"current_rate={current_rate:.6f}"
                )

            return stoploss_from_absolute(
                be_stop_price, current_rate, is_short=trade.is_short, leverage=trade.leverage
            )

        # --- 1. INITIAL DYNAMIC STOPLOSS (fallback when BE not yet activated) ---
        if not be_activated and self.use_dynamic_stoploss.value:
            initial_sl_price = trade.get_custom_data("initial_sl_price")

            if initial_sl_price is None:
                try:
                    dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

                    candle = dataframe.loc[dataframe["date"] == trade.open_date_utc]
                    if candle.empty:
                        candle = dataframe.iloc[[-1]]
                        logger.debug(
                            f"Entry candle not in analyzed window for {pair}, "
                            f"using latest candle for dynamic SL lookup."
                        )

                    if not candle.empty:
                        row = candle.iloc[0]
                        price_found = 0.0

                        if trade.is_short:
                            if "sl_short_price" in row and not pd.isna(row["sl_short_price"]):
                                price_found = row["sl_short_price"]
                        else:
                            if "sl_long_price" in row and not pd.isna(row["sl_long_price"]):
                                price_found = row["sl_long_price"]

                        if price_found > 0:
                            initial_sl_price = price_found
                            trade.set_custom_data("initial_sl_price", initial_sl_price)
                            logger.info(f"Initial Dynamic SL found for {pair}: {initial_sl_price}")
                except Exception:
                    pass

            if initial_sl_price and initial_sl_price > 0:
                logger.debug(
                    f"Dynamic SL for {pair}: sl_price={initial_sl_price:.4f}, current_rate={current_rate:.4f}"
                )
                return stoploss_from_absolute(
                    initial_sl_price, current_rate, is_short=trade.is_short, leverage=trade.leverage
                )

        return 1  # Use default stoploss

    def adjust_trade_position(
        self,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        min_stake: float | None,
        max_stake: float,
        current_entry_rate: float,
        current_exit_rate: float,
        current_entry_profit: float,
        current_exit_profit: float,
        **kwargs,
    ) -> float | None | tuple[float | None, str | None]:
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
        tp1_taken = trade.get_custom_data("tp1_taken", default=False)

        # Use dynamic TP1 if available, else fall back to configured tp1_pct
        effective_tp1 = self.tp1_pct.value
        if self.use_dynamic_tp_adj.value:
            effective_tp1 = trade.get_custom_data("tp1_adj", default=self.tp1_pct.value)

        # Check for TP1 (only if not already taken)
        if not tp1_taken and price_movement > effective_tp1:
            # Check if TP1 is enabled via boolean or amount
            if not self.tp1_enabled.value or self.tp1_amount.value == 0:
                return None

            # Mark TP1 as taken BEFORE placing the order (prevents recursion)
            trade.set_custom_data("tp1_taken", True)

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
                f"price moved {price_movement:.2%} (trigger={effective_tp1:.2%}), closing {self.tp1_amount.value:.0f}% "
                f"(Notional: {sell_value:.2f}, Margin: {stake_change:.2f})"
            )
            return (-stake_change, "TP1_partial")

        return None
