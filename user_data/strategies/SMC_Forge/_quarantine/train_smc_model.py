"""
SMC ML Training Script (LuxAlgo Version)
========================================

Trains XGBoost using SMCEngine signals to match the strategy.

Usage:
1. Ensure data is available (freqtrade download-data).
2. Run: python user_data/strategies/SMC/train_smc_model.py
"""

import os
import sys
import json
import pickle
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import talib.abstract as ta
import warnings

# Suppress XGBoost/Sklearn device mismatch warning
warnings.filterwarnings("ignore", message=".*Falling back to prediction using DMatrix.*")

# Setup logging
logger = logging.getLogger(__name__)

# Add strategy path
sys.path.insert(0, str(Path(__file__).parent))
# Import NEW library (Numba Optimized)
from smc_engine import SMCEngine as SMCEngine

try:
    from xgboost import XGBClassifier
    from sklearn.model_selection import TimeSeriesSplit, cross_val_score
    from sklearn.metrics import (
        classification_report,
        roc_auc_score,
        precision_recall_curve,
        average_precision_score,
    )
    import optuna
except ImportError as e:
    logger.error(f"Missing import: {e}. Run pip install xgboost scikit-learn")
    sys.exit(1)


# =============================================================================
# CONFIGURATION
# =============================================================================


def load_config():
    """Load config dynamically from JSON files."""

    # Dedicated training pairs — broader than the live bot whitelist.
    # Covers large caps (stable structure), mid caps, and memes (noisy / high vol).
    # All pairs confirmed to have 15m data from 2024-01-01 on Bitget futures.
    training_pairs = [
        # --- Large caps / low volatility ---
        "BTC/USDT:USDT",
        "ETH/USDT:USDT",
        "BNB/USDT:USDT",
        "SOL/USDT:USDT",
        "XRP/USDT:USDT",
        "ADA/USDT:USDT",
        "AVAX/USDT:USDT",
        "LINK/USDT:USDT",
        "DOT/USDT:USDT",
        "LTC/USDT:USDT",
        "ATOM/USDT:USDT",
        "TRX/USDT:USDT",
        # --- Mid caps / medium volatility ---
        "ARB/USDT:USDT",
        "OP/USDT:USDT",
        "NEAR/USDT:USDT",
        "APT/USDT:USDT",
        "SUI/USDT:USDT",
        "FIL/USDT:USDT",
        "ICP/USDT:USDT",
        "HBAR/USDT:USDT",
        "VET/USDT:USDT",
        "RUNE/USDT:USDT",
        "PENDLE/USDT:USDT",
        # --- Memes / high volatility ---
        "DOGE/USDT:USDT",
        "PEPE/USDT:USDT",
        "FLOKI/USDT:USDT",
        "1000BONK/USDT:USDT",
        "STX/USDT:USDT",
        "WIF/USDT:USDT",
        # --- Speculative / narrative ---
        "TAO/USDT:USDT",
        "JTO/USDT:USDT",
    ]

    # Base config (static values)
    config = {
        # Paths
        "data_dir": "user_data/data/bitget/futures",
        "model_output_dir": "user_data/strategies/SMC/models",
        # Training pairs — dedicated list, ignores config.json whitelist
        "pairs": training_pairs,
        # Training date range — use full available history (2024-01 to 2025-12)
        "training_start": "2024-01-01",
        "training_end": "2025-12-31",
        # Target definition - aligned with strategy JSON (tp1_pct / stoploss)
        "profit_threshold": 0.05,  # 5% target (matches tp1_pct in strategy JSON)
        "stop_loss_threshold": 0.02,  # 2% stop (matches stoploss in strategy JSON)
        "future_window": 96,  # 24h at 15m (enough time for 5% moves)
        # ML parameters
        "n_splits": 5,
        "hyperopt": True,
        "hyperopt_iter": 100,
        # Per-symbol training mode
        "per_symbol_models": False,  # If True, train one model per symbol
    }

    # Load from config.json
    try:
        with open("config.json", "r") as f:
            main_config = json.load(f)

            # Load timeframe
            config["timeframe"] = main_config.get("timeframe", "15m")
            logger.info(f"📖 Loaded timeframe from config.json: {config['timeframe']}")

            # Load exchange and build data_dir dynamically
            exchange_config = main_config.get("exchange", {})
            exchange_name = exchange_config.get("name", "bitget").lower()

            # Check for trading_mode to determine subfolder (spot vs futures)
            trading_mode = main_config.get("trading_mode", "spot")
            if trading_mode == "futures":
                config["data_dir"] = f"user_data/data/{exchange_name}/futures"
            else:
                config["data_dir"] = f"user_data/data/{exchange_name}"

            logger.info(f"📖 Data directory from config: {config['data_dir']}")

            # Training uses dedicated training_pairs list (not the live bot whitelist)
            logger.info(
                f"📖 Training pairs: {len(config['pairs'])} (dedicated list, not from config.json whitelist)"
            )

    except Exception as e:
        logger.warning(f"Could not load config.json, using defaults: {e}")
        config["timeframe"] = "15m"

    # Load SMC parameters from strategy JSON
    try:
        strat_json_path = "user_data/strategies/SMC/SMCWithMLLuxAlgo.json"
        with open(strat_json_path, "r") as f:
            strat_config = json.load(f)
            buy_params = strat_config.get("params", {}).get("buy", {})

            config["internal_length"] = buy_params.get("internal_length", 5)
            config["swing_length"] = buy_params.get("swing_length", 25)

            # Align label targets with live strategy parameters
            sell_params = strat_config.get("params", {}).get("sell", {})
            stoploss_params = strat_config.get("params", {}).get("stoploss", {})
            tp1_pct = sell_params.get("tp1_pct", 0.05)
            sl_pct = abs(stoploss_params.get("stoploss", -0.02))
            config["profit_threshold"] = tp1_pct
            config["stop_loss_threshold"] = sl_pct
            logger.info(f"📖 Label targets from strategy JSON: TP={tp1_pct:.2%}, SL={sl_pct:.2%}")

            # Check for GPU config
            config["use_gpu"] = buy_params.get("use_gpu", False)
            if config["use_gpu"]:
                logger.info("🚀 GPU Training Enabled via JSON")

            logger.info(
                f"📖 Loaded SMC params from JSON: internal={config['internal_length']}, swing={config['swing_length']}"
            )
    except Exception as e:
        logger.warning(f"Could not load strategy JSON, using defaults: {e}")
        config["internal_length"] = 5
        config["swing_length"] = 25

    return config


# Load config at module level
CONFIG = load_config()

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================


def calculate_ml_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate ML Features.
    Must be compatible with Strategy's populate_entry_trend.
    """
    df = dataframe.copy()

    # --- 1. Technical Indicators (Momentum & Volatility) ---

    # RSI
    df["rsi"] = ta.RSI(df["close"], timeperiod=14)
    df["ml_rsi"] = df["rsi"] / 100.0

    # ADX (Trend Strength) - NEW
    df["adx"] = ta.ADX(df["high"], df["low"], df["close"], timeperiod=14)
    df["ml_adx"] = df["adx"] / 100.0

    # CCI (Cyclical) - NEW
    df["cci"] = ta.CCI(df["high"], df["low"], df["close"], timeperiod=20)
    df["ml_cci"] = df["cci"] / 300.0  # Normalize -1 to 1

    # WaveTrend (Oscillator)
    n1 = 10
    n2 = 21
    ap = (df["high"] + df["low"] + df["close"]) / 3
    esa = ta.EMA(ap, timeperiod=n1)
    d = ta.EMA((ap - esa).abs(), timeperiod=n1)
    ci = (ap - esa) / (0.015 * d)
    df["wt1"] = ta.EMA(ci, timeperiod=n2)
    df["wt2"] = ta.SMA(df["wt1"], timeperiod=4)

    df["ml_wt1"] = df["wt1"] / 100.0
    df["ml_wt2"] = df["wt2"] / 100.0
    df["ml_wt_diff"] = (df["wt1"] - df["wt2"]) / 100.0

    # EMAs
    df["ema_50"] = ta.EMA(df["close"], timeperiod=50)
    df["ema_200"] = ta.EMA(df["close"], timeperiod=200)
    df["ml_dist_ema50"] = (df["close"] - df["ema_50"]) / df["ema_50"]
    df["ml_dist_ema200"] = (df["close"] - df["ema_200"]) / df["ema_200"]

    # ATR
    df["atr"] = ta.ATR(df, timeperiod=14)
    df["ml_atr_pct"] = df["atr"] / df["close"]

    # Volatility Filter (Lorentzian Style: atr_1 > atr_10)
    atr_1 = ta.ATR(df, timeperiod=1)
    atr_10 = ta.ATR(df, timeperiod=10)
    df["ml_volatility_high"] = (atr_1 > atr_10).astype(float)

    # Volume
    df["volume_ma"] = ta.SMA(df["volume"], timeperiod=20)
    df["ml_volume_ratio"] = df["volume"] / df["volume_ma"].replace(0, 1)

    # OB Relative Volume (RVOL) — mirrors populate_indicators in the strategy
    vol_sma = pd.Series(ta.SMA(df["volume"], timeperiod=20), index=df.index)
    vol_sma = vol_sma.replace(0, np.nan)
    if "active_bullish_ob_vol" in df.columns:
        df["bull_ob_rvol"] = (df["active_bullish_ob_vol"] / vol_sma).fillna(0)
    else:
        df["bull_ob_rvol"] = 0.0
    if "active_bearish_ob_vol" in df.columns:
        df["bear_ob_rvol"] = (df["active_bearish_ob_vol"] / vol_sma).fillna(0)
    else:
        df["bear_ob_rvol"] = 0.0

    # RSI(9) - Feature 5 from Lorentzian (separate from RSI 14)
    df["rsi_9"] = ta.RSI(df["close"], timeperiod=9)
    df["ml_rsi_9"] = df["rsi_9"] / 100.0

    # --- 2. SMC Context ---

    # Time since last signal (Vectorized)
    def bars_since(series):
        # cumcount starts at 0 for each group.
        # groups are formed by cumsum.
        return series.cumsum().groupby(series.cumsum()).cumcount()

    df["ml_bars_since_int_bull_choch"] = bars_since(df["internal_choch_bullish"])
    df["ml_bars_since_int_bear_choch"] = bars_since(df["internal_choch_bearish"])

    # Trend alignment
    df["ml_swing_trend"] = df["swing_trend"]

    # Zone Interaction
    df["ml_in_bull_ob"] = (
        (df["active_bullish_ob_top"] > 0) & (df["low"] <= df["active_bullish_ob_top"])
    ).astype(int)
    df["ml_in_bear_ob"] = (
        (df["active_bearish_ob_top"] > 0) & (df["high"] >= df["active_bearish_ob_bottom"])
    ).astype(int)

    # Liquidity Sweeps (Rolling window for ML context)
    if "internal_sweep_bullish" in df.columns:
        df["ml_recent_bull_sweep"] = (
            df["internal_sweep_bullish"].rolling(3, min_periods=1).max().fillna(0)
        )
        df["ml_recent_bear_sweep"] = (
            df["internal_sweep_bearish"].rolling(3, min_periods=1).max().fillna(0)
        )
    else:
        df["ml_recent_bull_sweep"] = 0
        df["ml_recent_bear_sweep"] = 0

    # --- 3. Opposing Zone Obstacle Features ---
    # Captures whether a strong opposing OB/FVG blocks the path to TP.
    # For LONGS: obstacle = bearish zone ABOVE price (resistance)
    # For SHORTS: obstacle = bullish zone BELOW price (support)
    # The model uses 'direction' to interpret these correctly.

    NO_ZONE_DIST = 0.5  # Fill when no zone exists (effectively no obstacle)

    # Bearish OB above price
    bear_ob_active = df["active_bearish_ob_bottom"] > 0
    bear_ob_above = bear_ob_active & (df["active_bearish_ob_bottom"] > df["close"])
    df["ml_bear_ob_above_dist"] = np.where(
        bear_ob_above, (df["active_bearish_ob_bottom"] - df["close"]) / df["close"], NO_ZONE_DIST
    )
    df["ml_bear_ob_above_rvol"] = np.where(bear_ob_above, df["bear_ob_rvol"], 0.0)

    # Bearish FVG above price
    bear_fvg_active = df["active_bearish_fvg_bottom"] > 0
    bear_fvg_above = bear_fvg_active & (df["active_bearish_fvg_bottom"] > df["close"])
    df["ml_bear_fvg_above_dist"] = np.where(
        bear_fvg_above, (df["active_bearish_fvg_bottom"] - df["close"]) / df["close"], NO_ZONE_DIST
    )

    # Bullish OB below price
    bull_ob_active = df["active_bullish_ob_top"] > 0
    bull_ob_below = bull_ob_active & (df["active_bullish_ob_top"] < df["close"])
    df["ml_bull_ob_below_dist"] = np.where(
        bull_ob_below, (df["close"] - df["active_bullish_ob_top"]) / df["close"], NO_ZONE_DIST
    )
    df["ml_bull_ob_below_rvol"] = np.where(bull_ob_below, df["bull_ob_rvol"], 0.0)

    # Bullish FVG below price
    bull_fvg_active = df["active_bullish_fvg_top"] > 0
    bull_fvg_below = bull_fvg_active & (df["active_bullish_fvg_top"] < df["close"])
    df["ml_bull_fvg_below_dist"] = np.where(
        bull_fvg_below, (df["close"] - df["active_bullish_fvg_top"]) / df["close"], NO_ZONE_DIST
    )

    # Aggregated: nearest obstacle in each direction
    df["ml_min_resist_above_dist"] = df[["ml_bear_ob_above_dist", "ml_bear_fvg_above_dist"]].min(
        axis=1
    )
    df["ml_max_resist_above_rvol"] = df["ml_bear_ob_above_rvol"]

    df["ml_min_support_below_dist"] = df[["ml_bull_ob_below_dist", "ml_bull_fvg_below_dist"]].min(
        axis=1
    )
    df["ml_max_support_below_rvol"] = df["ml_bull_ob_below_rvol"]

    # Fill NaNs
    feature_cols = [c for c in df.columns if c.startswith("ml_")]
    df[feature_cols] = df[feature_cols].fillna(0)

    return df


# =============================================================================
# DATA LOADING & PREP
# =============================================================================


def load_data(config: dict) -> pd.DataFrame:
    all_data = []

    for pair in config["pairs"]:
        # Freqtrade feather format: pair-timeframe.feather
        # User pair like BTC/USDT:USDT -> clean to BTC_USDT_USDT or similar depending on download
        # Assuming generic format for now
        filename_pair = pair.replace("/", "_").replace(":", "_")
        filename = f"{filename_pair}-{config['timeframe']}.feather"  # Try standard
        filepath = Path(config["data_dir"]) / filename

        if not filepath.exists():
            # Try futures specific
            filename = f"{filename_pair}-{config['timeframe']}-futures.feather"
            filepath = Path(config["data_dir"]) / filename

        if not filepath.exists():
            logger.warning(f"File not found for {pair}: {filepath}")
            continue

        df = pd.read_feather(filepath)
        df["date"] = pd.to_datetime(df["date"])
        df = df[(df["date"] >= config["training_start"]) & (df["date"] < config["training_end"])]
        df = df.reset_index(drop=True)
        df["pair"] = pair
        all_data.append(df)

    if not all_data:
        raise ValueError("No data loaded")

    return pd.concat(all_data, ignore_index=True)


def generate_labels(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Generate signals (Entry) and labels (Success/Fail)."""

    # 1. Calculate Indicators
    # Numba version returns active zones in the DF
    smc = SMCEngine(df, config["internal_length"], config["swing_length"])
    signals = smc.get_signals()
    for col in signals.columns:
        df[col] = signals[col].values

    # Zone memory is now built-in to 'signals' DF output from Numba lib
    # df = apply_zone_memory(df, smc) <-- REMOVED

    # Apply zone shift (matches strategy's populate_indicators lookahead bias fix)
    zone_cols = [c for c in df.columns if c.startswith("active_")]
    for col in zone_cols:
        df[col] = df[col].shift(1).fillna(0)

    # 2. Identify Potential Entries (CHoCH only, matching Strategy)

    # We want to train on ALL CHoCH signals to learn which are fake.
    # Entry conditions:
    df["signal_long"] = (df["internal_choch_bullish"] == 1) | (df["swing_choch_bullish"] == 1)
    df["signal_short"] = (df["internal_choch_bearish"] == 1) | (df["swing_choch_bearish"] == 1)

    # 3. Labeling (Lookahead)
    # If Price hits TP before SL -> 1, else 0.

    future_window = config.get("future_window", 96)  # 24h at 15m
    threshold = config["profit_threshold"]
    stop = config["stop_loss_threshold"]

    labels = []

    # Optimization: Iterate only on signals
    # (Doing full vector check for accurate TP/SL is complex, doing iterative for now)

    # We create a generic 'label' column
    df["label"] = 0
    df["direction"] = 0  # 1 long, -1 short

    # Scan for Longs
    long_idxs = df.index[df["signal_long"]].tolist()
    for idx in long_idxs:
        if idx + future_window >= len(df):
            continue

        entry_price = df.loc[idx, "close"]
        future_prices = df.loc[idx + 1 : idx + future_window]

        # Check Long Success
        # Hit TP?
        hit_tp = (future_prices["high"] >= entry_price * (1 + threshold)).any()
        # Hit SL? (Simplification: check if SL hit before TP)
        # We need timing.

        # Vectorized-ish check inside loop
        # Find first index where TP or SL hit
        tp_idx = future_prices[future_prices["high"] >= entry_price * (1 + threshold)].index.min()
        sl_idx = future_prices[future_prices["low"] <= entry_price * (1 - stop)].index.min()

        success = False
        if pd.notna(tp_idx):
            if pd.isna(sl_idx) or tp_idx < sl_idx:
                success = True

        if success:
            df.loc[idx, "label"] = 1

        df.loc[idx, "direction"] = 1

    # Scan for Shorts
    short_idxs = df.index[df["signal_short"]].tolist()
    for idx in short_idxs:
        if idx + future_window >= len(df):
            continue

        entry_price = df.loc[idx, "close"]
        future_prices = df.loc[idx + 1 : idx + future_window]

        tp_idx = future_prices[future_prices["low"] <= entry_price * (1 - threshold)].index.min()
        sl_idx = future_prices[future_prices["high"] >= entry_price * (1 + stop)].index.min()

        success = False
        if pd.notna(tp_idx):
            if pd.isna(sl_idx) or tp_idx < sl_idx:
                success = True

        if success:
            df.loc[idx, "label"] = 1

        df.loc[idx, "direction"] = -1

    return df


# =============================================================================
# MAIN PIPELINE
# =============================================================================


def train_model():
    """Execute the full training pipeline."""
    try:
        logger.info("Starting Training Pipeline...")

        # 1. Load & Prep
        df_raw = load_data(CONFIG)
        logger.info(f"Loaded {len(df_raw)} rows.")

        # 2. Features & Labels
        logger.info("Generating signals and features...")
        df_labeled = generate_labels(df_raw, CONFIG)
        df_features = calculate_ml_features(df_labeled)

        # 3. Filter for Training
        valid_entries = df_features[
            (df_features["signal_long"]) | (df_features["signal_short"])
        ].copy()

        if len(valid_entries) < 100:
            logger.error(f"Not enough signals found ({len(valid_entries)}). Check Data/Params.")
            return False

        logger.info(f"Training on {len(valid_entries)} signals.")
        logger.info(f"Class Balance: {valid_entries['label'].mean():.2%}")

        # 4. Train
        feature_cols = [c for c in valid_entries.columns if c.startswith("ml_")]
        feature_cols.append("direction")  # Explicitly add direction

        X = valid_entries[feature_cols]
        y = valid_entries["label"]

        # Split for final evaluation (after hyperopt)
        split = int(len(X) * 0.8)
        X_train_full, X_test = X.iloc[:split], X.iloc[split:]
        y_train_full, y_test = y.iloc[:split], y.iloc[split:]

        # Handle class imbalance: give winners proportionally more weight
        # Without this, XGBoost learns to predict "loser" for everything
        n_neg = int((y_train_full == 0).sum())
        n_pos = int((y_train_full == 1).sum())
        scale_pos_weight = n_neg / max(n_pos, 1)
        logger.info(
            f"⚖️  Class imbalance: {n_pos} winners / {n_neg} losers → scale_pos_weight={scale_pos_weight:.2f}"
        )

        best_params = {
            "n_estimators": 200,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "gamma": 0,
        }

        if CONFIG.get("hyperopt", False):
            logger.info(f"🔎 Starting Hyperopt (Optuna) with {CONFIG['hyperopt_iter']} trials...")

            optuna.logging.set_verbosity(optuna.logging.WARNING)

            def objective(trial):
                param = {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                    "gamma": trial.suggest_float("gamma", 0.0, 1.0),
                    "min_child_weight": trial.suggest_int("min_child_weight", 1, 5),
                    # Imbalance correction (constant, derived from data)
                    "scale_pos_weight": scale_pos_weight,
                    "eval_metric": "aucpr",  # PR-AUC: better metric for imbalanced data
                    "n_jobs": -1,
                }

                if CONFIG.get("use_gpu", False):
                    param.update({"tree_method": "hist", "device": "cuda"})

                model = XGBClassifier(**param)

                tscv = TimeSeriesSplit(n_splits=5)
                cv_jobs = 1 if CONFIG.get("use_gpu", False) else -1

                # average_precision = PR-AUC: penalizes missing the minority class
                scores = cross_val_score(
                    model,
                    X_train_full,
                    y_train_full,
                    cv=tscv,
                    scoring="average_precision",
                    n_jobs=cv_jobs,
                )
                return scores.mean()

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=CONFIG["hyperopt_iter"])

            best_params = study.best_trial.params
            logger.info(f"✅ Best Params Found: {json.dumps(best_params, indent=2)}")
            logger.info(f"   Best Validation PR-AUC: {study.best_value:.4f}")

            final_args = {
                **best_params,
                "scale_pos_weight": scale_pos_weight,
                "eval_metric": "aucpr",
                "n_jobs": -1,
            }
            if CONFIG.get("use_gpu", False):
                final_args.update({"tree_method": "hist", "device": "cuda"})

            model = XGBClassifier(**final_args)
            model.fit(X_train_full, y_train_full)

        else:
            logger.info("Using fixed default parameters.")

            xgb_args = {
                **best_params,
                "scale_pos_weight": scale_pos_weight,
                "eval_metric": "aucpr",
                "n_jobs": -1,
            }
            if CONFIG.get("use_gpu", False):
                xgb_args.update({"tree_method": "hist", "device": "cuda"})

            model = XGBClassifier(**xgb_args)
            model.fit(X_train_full, y_train_full)

        # 5. Evaluate on Holdout Test Set
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]
        try:
            auc = roc_auc_score(y_test, probs)
            pr_auc = average_precision_score(y_test, probs)
            logger.info(f"Test Set ROC-AUC: {auc:.4f} | PR-AUC: {pr_auc:.4f}")

            # Find Optimal Threshold (Maximize F1)
            precision, recall, thresholds = precision_recall_curve(y_test, probs)
            f1_scores = 2 * (precision * recall) / (precision + recall)
            f1_scores = np.nan_to_num(f1_scores)  # Handle division by zero
            best_idx = np.argmax(f1_scores)

            # thresholds is 1 shorter than precision/recall, but argmax usually works fine if we match dims
            if best_idx < len(thresholds):
                best_threshold = thresholds[best_idx]
            else:
                best_threshold = 0.5

            best_f1 = f1_scores[best_idx]

            logger.info(f"💡 Recommended Threshold: {best_threshold:.4f} (Max F1: {best_f1:.4f})")

            # Save optimal threshold alongside model
            threshold_path = Path(CONFIG["model_output_dir"]) / "smc_xgboost_threshold.json"
            os.makedirs(CONFIG["model_output_dir"], exist_ok=True)
            with open(threshold_path, "w") as f:
                json.dump(
                    {
                        "threshold": float(best_threshold),
                        "f1": float(best_f1),
                        "roc_auc": float(auc),
                        "pr_auc": float(pr_auc),
                    },
                    f,
                    indent=2,
                )
            logger.info(f"💾 Threshold saved to {threshold_path}")

        except ValueError:
            logger.info("Test AUC: N/A (Only one class present in test set)")
            best_threshold = 0.5

        logger.info("\n" + classification_report(y_test, preds, zero_division=0))

        # 6. Save
        os.makedirs(CONFIG["model_output_dir"], exist_ok=True)
        model_path = Path(CONFIG["model_output_dir"]) / "smc_xgboost_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        logger.info(f"Model saved to {model_path}")
        return model  # Return model for per-symbol training

    except Exception as e:
        logger.error(f"Training Failed: {e}")
        return False


def pair_to_filename(pair: str) -> str:
    """Convert pair name to safe filename. E.g. 'BTC/USDT:USDT' -> 'BTC_USDT'"""
    return pair.replace("/", "_").replace(":", "_").split("_USDT")[0] + "_USDT"


def train_per_symbol():
    """Train individual models for each symbol."""
    global CONFIG
    CONFIG = load_config()

    pairs = CONFIG["pairs"]
    logger.info(f"🚀 Per-Symbol Training Mode: {len(pairs)} pairs")

    successful = 0
    failed = []

    for i, pair in enumerate(pairs, 1):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"📊 Training [{i}/{len(pairs)}]: {pair}")
        logger.info(f"{'=' * 60}")

        # Override pairs to train only this one
        original_pairs = CONFIG["pairs"]
        CONFIG["pairs"] = [pair]

        try:
            model = train_model()

            if model:
                # Save with pair-specific name
                pair_filename = pair_to_filename(pair)
                model_path = Path(CONFIG["model_output_dir"]) / f"smc_xgboost_{pair_filename}.pkl"

                with open(model_path, "wb") as f:
                    pickle.dump(model, f)

                logger.info(f"✅ Model saved: {model_path}")
                successful += 1
            else:
                failed.append(pair)

        except Exception as e:
            logger.error(f"❌ Failed training {pair}: {e}")
            failed.append(pair)

        # Restore pairs for next iteration
        CONFIG["pairs"] = original_pairs

    # Summary
    logger.info(f"\n{'=' * 60}")
    logger.info(f"📈 TRAINING COMPLETE")
    logger.info(f"{'=' * 60}")
    logger.info(f"✅ Successful: {successful}/{len(pairs)}")
    if failed:
        logger.info(f"❌ Failed: {failed}")

    return successful > 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Reload config to get latest settings
    CONFIG = load_config()

    if CONFIG.get("per_symbol_models", False):
        train_per_symbol()
    else:
        train_model()
