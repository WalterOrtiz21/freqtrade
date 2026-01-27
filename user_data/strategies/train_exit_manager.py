"""
Exit Manager Training Script
============================

Trains an XGBoost model to predict optimal exit actions based on historical trade data.

Usage:
    python train_exit_manager.py --input backtest_results.json --output models/exit_manager.pkl

Data Source:
    Freqtrade backtest export with detailed trade history.
    Run: freqtrade backtesting --export trades --export-filename results.json
"""

import argparse
import json
import logging
import pickle
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd

# ML imports
try:
    from xgboost import XGBClassifier
    from sklearn.model_selection import TimeSeriesSplit, cross_val_score
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.preprocessing import LabelEncoder
    HAS_ML_LIBS = True
except ImportError as e:
    print(f"Missing ML libraries: {e}")
    print("Install with: pip install xgboost scikit-learn")
    HAS_ML_LIBS = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    'model_output_dir': 'user_data/strategies/models',
    'model_filename': 'exit_manager.pkl',
    
    # Labeling thresholds (for creating target variable)
    'partial_exit_threshold': 0.03,    # Label as PARTIAL_EXIT if profit > 3%
    'tighten_sl_threshold': 0.02,      # Label as TIGHTEN_SL if drawdown > 2%
    'full_exit_profit_loss': 0.01,     # Min profit to consider FULL_EXIT valid
    
    # Model hyperparameters
    'xgb_params': {
        'n_estimators': 200,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'use_label_encoder': False,
        'eval_metric': 'mlogloss',
    },
    
    # Cross-validation
    'cv_splits': 5,
}


# =============================================================================
# DATA PROCESSING
# =============================================================================

@dataclass
class TradeSnapshot:
    """Represents a single snapshot of a trade at a point in time."""
    trade_id: str
    timestamp: pd.Timestamp
    current_profit: float
    max_profit: float
    trade_duration: float  # hours
    profit_velocity: float
    drawdown_from_max: float
    dist_to_entry: float
    # Market features (to be populated from OHLCV data)
    rsi: float = 0.5
    adx: float = 0.25
    atr_pct: float = 0.01
    dist_to_swing_high: float = 0.02
    dist_to_swing_low: float = 0.02
    kernel_direction: int = 1
    signal_current: int = 0
    volatility_high: int = 0
    # Label
    optimal_action: int = 0  # 0=HOLD, 1=PARTIAL, 2=TIGHTEN, 3=FULL_EXIT


def load_backtest_results(filepath: str) -> pd.DataFrame:
    """
    Load backtest results from Freqtrade JSON or ZIP export.
    
    Expected structure:
    {
        "trades": [
            {
                "pair": "BTC/USDT",
                "open_date": "2024-01-01 00:00:00",
                "close_date": "2024-01-02 00:00:00",
                "open_rate": 40000.0,
                "close_rate": 41000.0,
                "profit_ratio": 0.025,
                "max_rate": 41500.0,
                "min_rate": 39500.0,
                ...
            },
            ...
        ]
    }
    """
    import zipfile
    
    logger.info(f"Loading backtest results from {filepath}")
    
    # Handle ZIP files
    if filepath.endswith('.zip'):
        with zipfile.ZipFile(filepath, 'r') as zf:
            # Find JSON file inside ZIP
            json_files = [n for n in zf.namelist() if n.endswith('.json')]
            if not json_files:
                raise ValueError(f"No JSON file found inside {filepath}")
            
            # Read the first JSON file
            with zf.open(json_files[0]) as f:
                data = json.load(f)
    else:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    
    # Handle different export formats
    if isinstance(data, dict):
        if 'trades' in data:
            trades = data['trades']
        elif 'strategy' in data:
            # Freqtrade 2024+ format: {'strategy': {'StrategyName': {'trades': [...]}}}
            strategy_dict = data['strategy']
            if isinstance(strategy_dict, dict):
                # Get first strategy's data
                first_strategy = list(strategy_dict.values())[0]
                if isinstance(first_strategy, dict):
                    trades = first_strategy.get('trades', [])
                else:
                    trades = []
            else:
                trades = []
        else:
            # Try to find trades in nested structure (strategy name as key)
            trades = []
            for key, value in data.items():
                if isinstance(value, dict) and 'trades' in value:
                    trades = value['trades']
                    break
    elif isinstance(data, list):
        trades = data
    else:
        raise ValueError(f"Unknown backtest format in {filepath}")
    
    if not trades:
        logger.warning("No trades found in backtest results!")
        return pd.DataFrame()
    
    df = pd.DataFrame(trades)
    
    # Convert dates
    for col in ['open_date', 'close_date']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    
    logger.info(f"Loaded {len(df)} trades from backtest")
    return df


def generate_training_samples(trades_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate training samples from trade history.
    
    For each trade, we generate multiple snapshots at different profit levels
    and label what the OPTIMAL action would have been (with hindsight).
    """
    samples = []
    
    for _, trade in trades_df.iterrows():
        trade_samples = generate_trade_snapshots(trade)
        samples.extend(trade_samples)
    
    if not samples:
        return pd.DataFrame()
    
    return pd.DataFrame([s.__dict__ for s in samples])


def generate_trade_snapshots(trade: pd.Series) -> List[TradeSnapshot]:
    """
    Generate labeled snapshots for a single trade.
    
    We simulate what we would have known at different points during the trade
    and label what the optimal action would have been given the final outcome.
    """
    snapshots = []
    
    # Extract trade info
    trade_id = str(trade.get('trade_id', trade.name))
    open_rate = trade.get('open_rate', 0)
    close_rate = trade.get('close_rate', 0)
    max_rate = trade.get('max_rate', close_rate)
    min_rate = trade.get('min_rate', close_rate)
    final_profit = trade.get('profit_ratio', 0)
    is_short = trade.get('is_short', False)
    
    open_date = trade.get('open_date')
    close_date = trade.get('close_date')
    
    if not open_rate or not close_rate:
        return snapshots
    
    # Calculate key metrics
    if is_short:
        max_profit = (open_rate - min_rate) / open_rate if open_rate else 0
    else:
        max_profit = (max_rate - open_rate) / open_rate if open_rate else 0
    
    trade_duration_hours = 0
    if open_date and close_date:
        trade_duration_hours = (close_date - open_date).total_seconds() / 3600
    
    # Generate snapshots at different profit levels
    profit_checkpoints = [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, max_profit * 0.5, max_profit]
    profit_checkpoints = sorted(set([p for p in profit_checkpoints if 0 <= p <= max_profit]))
    
    for i, checkpoint_profit in enumerate(profit_checkpoints):
        # Determine optimal action with hindsight
        optimal_action = determine_optimal_action(
            checkpoint_profit=checkpoint_profit,
            max_profit=max_profit,
            final_profit=final_profit,
            is_short=is_short
        )
        
        # Estimate time at this checkpoint (linear interpolation)
        if max_profit > 0:
            time_fraction = checkpoint_profit / max_profit
        else:
            time_fraction = 0
        
        snapshot = TradeSnapshot(
            trade_id=trade_id,
            timestamp=open_date + pd.Timedelta(hours=trade_duration_hours * time_fraction) if open_date else pd.Timestamp.now(),
            current_profit=checkpoint_profit,
            max_profit=checkpoint_profit,  # At this point, current IS the max
            trade_duration=trade_duration_hours * time_fraction,
            profit_velocity=checkpoint_profit / max(trade_duration_hours * time_fraction, 0.01),
            drawdown_from_max=0,  # No drawdown yet at peak
            dist_to_entry=checkpoint_profit,  # Simplified
            optimal_action=optimal_action,
        )
        snapshots.append(snapshot)
        
        # Also generate drawdown scenarios (after peak)
        if checkpoint_profit == max_profit and max_profit > 0:
            drawdown_levels = [0.01, 0.02, 0.03, 0.05]
            for dd in drawdown_levels:
                profit_after_dd = max_profit - dd
                if profit_after_dd < 0:
                    continue
                    
                # What should we have done during the drawdown?
                dd_action = determine_optimal_action_during_drawdown(
                    current_profit=profit_after_dd,
                    max_profit=max_profit,
                    final_profit=final_profit,
                    drawdown=dd
                )
                
                dd_snapshot = TradeSnapshot(
                    trade_id=f"{trade_id}_dd_{dd}",
                    timestamp=open_date + pd.Timedelta(hours=trade_duration_hours * 0.8) if open_date else pd.Timestamp.now(),
                    current_profit=profit_after_dd,
                    max_profit=max_profit,
                    trade_duration=trade_duration_hours * 0.8,
                    profit_velocity=-dd / (trade_duration_hours * 0.2) if trade_duration_hours else 0,
                    drawdown_from_max=dd,
                    dist_to_entry=profit_after_dd,
                    optimal_action=dd_action,
                )
                snapshots.append(dd_snapshot)
    
    return snapshots


def determine_optimal_action(checkpoint_profit: float, max_profit: float, 
                             final_profit: float, is_short: bool) -> int:
    """
    Determine optimal action with hindsight.
    
    Actions: 0=HOLD, 1=PARTIAL_EXIT, 2=TIGHTEN_SL, 3=FULL_EXIT
    """
    # If we eventually lost money but had profit here -> should have exited
    if checkpoint_profit > 0.01 and final_profit < 0:
        return 3  # FULL_EXIT
    
    # If this is near the peak and profit dropped after -> should have taken partial
    if checkpoint_profit > 0.03 and (max_profit - checkpoint_profit) < 0.01 and final_profit < max_profit * 0.5:
        return 1  # PARTIAL_EXIT
    
    # If we had good profit and it went even higher -> HOLD was right
    if checkpoint_profit > 0 and final_profit > checkpoint_profit:
        return 0  # HOLD
    
    # If we're in profit but final is less -> partial would have been good
    if checkpoint_profit > 0.02 and final_profit < checkpoint_profit * 0.7:
        return 1  # PARTIAL_EXIT
    
    # Default: HOLD
    return 0


def determine_optimal_action_during_drawdown(current_profit: float, max_profit: float,
                                             final_profit: float, drawdown: float) -> int:
    """Determine optimal action during drawdown phase."""
    
    # If final profit is even worse -> should have tightened or exited
    if final_profit < current_profit:
        if drawdown > 0.03:
            return 3  # FULL_EXIT
        else:
            return 2  # TIGHTEN_SL
    
    # If price recovered -> HOLD was fine
    if final_profit > current_profit:
        return 0  # HOLD
    
    # If significant drawdown from max -> TIGHTEN_SL
    if drawdown > 0.02:
        return 2  # TIGHTEN_SL
    
    return 0  # HOLD


# =============================================================================
# MODEL TRAINING
# =============================================================================

def train_model(samples_df: pd.DataFrame) -> Optional[XGBClassifier]:
    """Train XGBoost classifier on the training samples."""
    
    if not HAS_ML_LIBS:
        logger.error("ML libraries not available. Cannot train model.")
        return None
    
    if samples_df.empty:
        logger.error("No training samples available!")
        return None
    
    # Feature columns
    feature_cols = [
        'current_profit', 'max_profit', 'trade_duration', 'profit_velocity',
        'drawdown_from_max', 'dist_to_entry', 'rsi', 'adx', 'atr_pct',
        'dist_to_swing_high', 'dist_to_swing_low', 'kernel_direction',
        'signal_current', 'volatility_high'
    ]
    
    # Filter to available columns
    available_features = [c for c in feature_cols if c in samples_df.columns]
    
    X = samples_df[available_features].fillna(0)
    y = samples_df['optimal_action']
    
    logger.info(f"Training on {len(X)} samples with {len(available_features)} features")
    logger.info(f"Class distribution:\n{y.value_counts()}")
    
    # Train model
    model = XGBClassifier(**CONFIG['xgb_params'])
    
    # Cross-validation
    tscv = TimeSeriesSplit(n_splits=CONFIG['cv_splits'])
    
    try:
        cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='accuracy')
        logger.info(f"Cross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
    except Exception as e:
        logger.warning(f"Cross-validation failed: {e}")
    
    # Fit on all data
    model.fit(X, y)
    
    # Evaluation
    y_pred = model.predict(X)
    logger.info("\nClassification Report (Training Data):")
    print(classification_report(y, y_pred, target_names=['HOLD', 'PARTIAL', 'TIGHTEN_SL', 'FULL_EXIT']))
    
    # Feature importance
    logger.info("\nFeature Importance:")
    importance = pd.DataFrame({
        'feature': available_features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(importance.to_string(index=False))
    
    return model


def save_model(model: XGBClassifier, output_path: str) -> bool:
    """Save trained model to disk."""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"✅ Model saved to {output_path}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to save model: {e}")
        return False


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train Exit Manager ML Model')
    parser.add_argument('--input', '-i', required=True, 
                        help='Path to backtest results JSON file')
    parser.add_argument('--output', '-o', 
                        default=os.path.join(CONFIG['model_output_dir'], CONFIG['model_filename']),
                        help='Output path for trained model')
    args = parser.parse_args()
    
    # Load data
    trades_df = load_backtest_results(args.input)
    
    if trades_df.empty:
        logger.error("No trades to train on!")
        return False
    
    # Generate training samples
    logger.info("Generating training samples from trades...")
    samples_df = generate_training_samples(trades_df)
    
    if samples_df.empty:
        logger.error("Failed to generate training samples!")
        return False
    
    logger.info(f"Generated {len(samples_df)} training samples from {len(trades_df)} trades")
    
    # Train model
    model = train_model(samples_df)
    
    if model is None:
        return False
    
    # Save model
    return save_model(model, args.output)


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
