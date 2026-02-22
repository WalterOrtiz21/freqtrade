"""
Gaussian Bands ML Training Script
========================================

Trains XGBoost using Volatility Gaussian Bands logic to predict trade success.

Usage:
1. Ensure data is available (freqtrade download-data).
2. Run: python user_data/strategies/GaussianBands/train_gaussian_model.py
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

warnings.filterwarnings("ignore", message=".*Falling back to prediction using DMatrix.*")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))
# Import the exact compiled function from the strategy to ensure identical mathematical results
from VolatilityGaussianBands import _calculate_gaussian_multi_trend_numba

try:
    from xgboost import XGBClassifier
    from sklearn.model_selection import TimeSeriesSplit, cross_val_score
    from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
    import optuna
except ImportError as e:
    logger.error(f"Missing import: {e}. Run pip install xgboost scikit-learn optuna")
    sys.exit(1)


# =============================================================================
# CONFIGURATION
# =============================================================================
def load_config():
    """Load config dynamically from JSON files."""
    default_pairs = [
        'BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT', 
        'BNB/USDT:USDT', 'DOGE/USDT:USDT', 'SUI/USDT:USDT', 'STX/USDT:USDT'
    ]
    
    config = {
        'data_dir': 'user_data/data/bitget/futures',
        'model_output_dir': 'user_data/strategies/GaussianBands/models',
        'pairs': default_pairs,
        'training_start': '2020-01-01',
        'training_end': '2024-12-31',
        # Target definition for TP1 (Gaussian Strategy usually targets 5% movement)
        'profit_threshold': 0.05,  
        'stop_loss_threshold': 0.04, 
        'hyperopt': True,
        'hyperopt_iter': 100,
        'per_symbol_models': False,
    }
    
    try:
        with open('config_gaussian.json', 'r') as f:
            main_config = json.load(f)
            config['timeframe'] = main_config.get('timeframe', '1h')
            exchange_config = main_config.get('exchange', {})
            exchange_name = exchange_config.get('name', 'bitget').lower()
            trading_mode = main_config.get('trading_mode', 'spot')
            
            if trading_mode == 'futures':
                config['data_dir'] = f'user_data/data/{exchange_name}/futures'
            else:
                config['data_dir'] = f'user_data/data/{exchange_name}'
            
            pairs_whitelist = exchange_config.get('pair_whitelist', [])
            if pairs_whitelist: config['pairs'] = pairs_whitelist
    except Exception as e:
        config['timeframe'] = '1h'
    
    try:
        strat_json_path = 'user_data/strategies/GaussianBands/VolatilityGaussianBands.json'
        with open(strat_json_path, 'r') as f:
            strat_config = json.load(f)
            buy_params = strat_config.get('params', {}).get('buy', {})
            config['len_gaussian'] = buy_params.get('len_gaussian', 21)
            config['distance'] = buy_params.get('distance', 0.9)
    except Exception as e:
        config['len_gaussian'] = 21
        config['distance'] = 0.9
    
    return config

CONFIG = load_config()

# =============================================================================
# FEATURE ENGINEERING & LOGIC (Identical to Freqtrade Strategy)
# =============================================================================
def calculate_ml_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    period = config['len_gaussian']
    dist = config['distance']
    
    # 1. Base Gaussian Trend Average & Bands
    close_array = df['close'].values
    df['gaussian_avg'] = _calculate_gaussian_multi_trend_numba(close_array, period)
    df['volatility_hl'] = (df['high'] - df['low']).rolling(window=100).mean()
    df['lower_band'] = df['gaussian_avg'] - (df['volatility_hl'] * dist)
    df['upper_band'] = df['gaussian_avg'] + (df['volatility_hl'] * dist)
    
    # 2. Determine Trend
    df['cross_up'] = (df['close'] > df['upper_band']) & (df['close'].shift(1) <= df['upper_band'].shift(1))
    df['cross_dn'] = (df['close'] < df['lower_band']) & (df['close'].shift(1) >= df['lower_band'].shift(1))
    
    df['trend_signal'] = 0
    df.loc[df['cross_up'], 'trend_signal'] = 1
    df.loc[df['cross_dn'], 'trend_signal'] = -1
    df['trend'] = df['trend_signal'].replace(0, np.nan).ffill().fillna(0)
    
    # 3. Signals (Retest Only - This defines when the ML will test an entry)
    df['close_cross_up_avg'] = (df['close'] > df['gaussian_avg']) & (df['close'].shift(1) <= df['gaussian_avg'].shift(1))
    df['signal_long'] = (df['trend'] == 1) & df['close_cross_up_avg']
    
    df['close_cross_dn_avg'] = (df['close'] < df['gaussian_avg']) & (df['close'].shift(1) >= df['gaussian_avg'].shift(1))
    df['signal_short'] = (df['trend'] == -1) & df['close_cross_dn_avg']
    
    # 4. ML Features (Normalized variables)
    df['ml_vol_squeeze'] = (df['upper_band'] - df['lower_band']) / df['close']
    df['ml_dist_to_avg'] = (df['close'] - df['gaussian_avg']) / df['gaussian_avg']
    band_range = (df['upper_band'] - df['lower_band']).replace(0, np.nan)
    df['ml_close_pos'] = ((df['close'] - df['lower_band']) / band_range).fillna(0.5)
    df['ml_htf_trend'] = df['trend'] # Self-proxy for training speed since simulated HTF requires complex resamples
    
    df['rsi'] = ta.RSI(df['close'], timeperiod=14)
    df['ml_rsi'] = df['rsi'] / 100.0
    
    df['adx'] = ta.ADX(df['high'], df['low'], df['close'], timeperiod=14)
    df['ml_adx'] = df['adx'] / 100.0
    
    df['atr'] = ta.ATR(df, timeperiod=14)
    df['ml_atr_pct'] = df['atr'] / df['close']
    
    return df

# =============================================================================
# DATA LOADING & PREP
# =============================================================================
def load_data(config: dict) -> pd.DataFrame:
    all_data = []
    
    for pair in config['pairs']:
        filename_pair = pair.replace('/', '_').replace(':', '_')
        filename = f"{filename_pair}-{config['timeframe']}.feather"
        filepath = Path(config['data_dir']) / filename
        
        if not filepath.exists():
            filename = f"{filename_pair}-{config['timeframe']}-futures.feather"
            filepath = Path(config['data_dir']) / filename
            
        if not filepath.exists():
            logger.warning(f"File not found for {pair}: {filepath}")
            continue
            
        df = pd.read_feather(filepath)
        df['date'] = pd.to_datetime(df['date'])
        df = df[(df['date'] >= config['training_start']) & (df['date'] < config['training_end'])]
        df = df.reset_index(drop=True)
        df['pair'] = pair
        all_data.append(df)
        
    if not all_data:
        raise ValueError("No data loaded")
    return pd.concat(all_data, ignore_index=True)

def generate_labels(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Evaluate signals (Entry) to label Success (1) or Fail (0)."""
    future_window = 48 # Evaluate next 48 hours for 1h candles
    threshold = config['profit_threshold']
    stop = config['stop_loss_threshold']
    
    df['label'] = 0
    df['direction'] = 0
    
    # Vectorized loop for longs
    long_idxs = df.index[df['signal_long']].tolist()
    for idx in long_idxs:
        if idx + future_window >= len(df): continue
        entry_price = df.loc[idx, 'close']
        future_prices = df.loc[idx+1 : idx+future_window]
        
        tp_idx = future_prices[future_prices['high'] >= entry_price * (1 + threshold)].index.min()
        sl_idx = future_prices[future_prices['low'] <= entry_price * (1 - stop)].index.min()
        
        success = False
        if pd.notna(tp_idx) and (pd.isna(sl_idx) or tp_idx < sl_idx):
            success = True
            
        if success: df.loc[idx, 'label'] = 1
        df.loc[idx, 'direction'] = 1

    # Vectorized loop for shorts
    short_idxs = df.index[df['signal_short']].tolist()
    for idx in short_idxs:
        if idx + future_window >= len(df): continue
        entry_price = df.loc[idx, 'close']
        future_prices = df.loc[idx+1 : idx+future_window]
        
        tp_idx = future_prices[future_prices['low'] <= entry_price * (1 - threshold)].index.min()
        sl_idx = future_prices[future_prices['high'] >= entry_price * (1 + stop)].index.min()
        
        success = False
        if pd.notna(tp_idx) and (pd.isna(sl_idx) or tp_idx < sl_idx):
            success = True
                
        if success: df.loc[idx, 'label'] = 1
        df.loc[idx, 'direction'] = -1
        
    return df

# =============================================================================
# MAIN PIPELINE
# =============================================================================
def train_model():
    try:
        logger.info("Starting Gaussian ML Training Pipeline...")
        df_raw = load_data(CONFIG)
        logger.info(f"Loaded {len(df_raw)} rows.")
        
        logger.info("Generating signals and calculating Numba Gaussian variables...")
        df_features = calculate_ml_features(df_raw, CONFIG)
        df_labeled = generate_labels(df_features, CONFIG)
        
        # We drop NA features because the initial 100 rows per pair lack rolling data
        valid_entries = df_labeled.dropna(subset=['ml_atr_pct', 'ml_vol_squeeze']).copy()
        valid_entries = valid_entries[(valid_entries['signal_long']) | (valid_entries['signal_short'])].copy()
        
        if len(valid_entries) < 100:
            logger.error(f"Not enough signals found ({len(valid_entries)}). Check Data/Params.")
            return False
            
        logger.info(f"Training on {len(valid_entries)} Gaussian Retest signals.")
        logger.info(f"Class Balance (Target % Win Rate): {valid_entries['label'].mean():.2%}")
        
        feature_cols = [c for c in valid_entries.columns if c.startswith('ml_')]
        # feature_cols.append('direction') # Removed for vectorized inference
        
        X = valid_entries[feature_cols]
        y = valid_entries['label']
        
        split = int(len(X) * 0.8)
        X_train_full, X_test = X.iloc[:split], X.iloc[split:]
        y_train_full, y_test = y.iloc[:split], y.iloc[split:]

        best_params = {
            'n_estimators': 200, 'max_depth': 4, 'learning_rate': 0.05,
            'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0
        }
        
        if CONFIG.get('hyperopt', False):
            logger.info("🔎 Optimizing Hyperparameters via Optuna...")
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            
            def objective(trial):
                param = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'gamma': trial.suggest_float('gamma', 0.0, 1.0),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 5),
                    'eval_metric': 'logloss', 'n_jobs': -1 
                }
                model = XGBClassifier(**param)
                tscv = TimeSeriesSplit(n_splits=5)
                scores = cross_val_score(model, X_train_full, y_train_full, cv=tscv, scoring='roc_auc', n_jobs=-1)
                return scores.mean()

            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=CONFIG['hyperopt_iter'])
            best_params = study.best_trial.params
            
            final_args = {**best_params, 'eval_metric': 'logloss', 'n_jobs': -1}
            model = XGBClassifier(**final_args)
            model.fit(X_train_full, y_train_full)
        else:
            model = XGBClassifier(**{**best_params, 'eval_metric': 'logloss', 'n_jobs': -1})
            model.fit(X_train_full, y_train_full)
        
        logger.info("Evaluating optimal classification thresholds on out-of-sample data...")
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]
        try:
            auc = roc_auc_score(y_test, probs)
            logger.info(f"Test Set AUC: {auc:.4f}")
            
            precision, recall, thresholds = precision_recall_curve(y_test, probs)
            f1_scores = 2 * (precision * recall) / (precision + recall)
            f1_scores = np.nan_to_num(f1_scores)
            best_idx = np.argmax(f1_scores)
            best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
            
            logger.info(f"💡 Recommended Threshold for freqtrade JSON: {best_threshold:.4f} (Max F1: {f1_scores[best_idx]:.4f})")
        except ValueError:
            pass
            
        logger.info("\n" + classification_report(y_test, preds, zero_division=0))
        
        os.makedirs(CONFIG['model_output_dir'], exist_ok=True)
        model_path = Path(CONFIG['model_output_dir']) / 'gaussian_xgboost_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
            
        logger.info(f"✅ Gaussian Model saved successfully to {model_path}")
        return model
        
    except Exception as e:
        logger.error(f"Training Failed: {e}")
        return False

def pair_to_filename(pair: str) -> str:
    return pair.replace('/', '_').replace(':', '_').split('_USDT')[0] + '_USDT'

def train_per_symbol():
    global CONFIG
    pairs = CONFIG['pairs']
    successful = 0
    for i, pair in enumerate(pairs, 1):
        logger.info(f"\n📊 Training [{i}/{len(pairs)}]: {pair}")
        original_pairs = CONFIG['pairs']
        CONFIG['pairs'] = [pair]
        
        try:
            model = train_model()
            if model:
                model_path = Path(CONFIG['model_output_dir']) / f'gaussian_xgboost_{pair_to_filename(pair)}.pkl'
                with open(model_path, 'wb') as f: pickle.dump(model, f)
                successful += 1
        except Exception as e:
            logger.error(f"❌ Failed training {pair}: {e}")
        
        CONFIG['pairs'] = original_pairs
    
    logger.info(f"✅ Successful Symbol Models: {successful}/{len(pairs)}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    if CONFIG.get('per_symbol_models', False):
        train_per_symbol()
    else:
        train_model()
