"""
Intelligent Exit Manager for Freqtrade
=======================================

ML-based trade exit management system that recommends optimal exit actions
based on trade context and market conditions.

Actions:
- HOLD: Keep position unchanged
- PARTIAL_EXIT: Close X% of position
- TIGHTEN_SL: Move stoploss closer to current price
- FULL_EXIT: Close position completely

Usage:
    from ExitManager import ExitManager
    
    exit_manager = ExitManager(model_path='user_data/strategies/models/exit_manager.pkl')
    
    # In custom_exit or adjust_trade_position:
    action = exit_manager.predict_action(trade, current_rate, dataframe, current_profit)
"""

import logging
import pickle
import os
from typing import Optional, Dict, Any, Literal
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ExitAction(Enum):
    """Possible exit actions recommended by the manager."""
    HOLD = "hold"
    PARTIAL_EXIT = "partial_exit"
    TIGHTEN_SL = "tighten_sl"
    FULL_EXIT = "full_exit"


@dataclass
class ExitRecommendation:
    """Structured recommendation from ExitManager."""
    action: ExitAction
    confidence: float  # 0.0 to 1.0
    details: Dict[str, Any]  # Action-specific parameters
    
    def __repr__(self):
        return f"ExitRecommendation({self.action.value}, conf={self.confidence:.2f})"


class ExitManager:
    """
    Intelligent Exit Manager using ML to recommend trade exit actions.
    
    Features extracted from:
    - Trade context (profit, duration, drawdown)
    - Market conditions (indicators, swing levels)
    - Strategy signals (kernel direction, ML signal)
    """
    
    # Feature columns expected by the model
    TRADE_FEATURES = [
        'current_profit',      # Current profit ratio
        'max_profit',          # Maximum profit reached
        'trade_duration',      # Duration in candles
        'profit_velocity',     # Rate of profit change
        'drawdown_from_max',   # Drawdown from peak profit
        'dist_to_entry',       # Price distance from entry
    ]
    
    MARKET_FEATURES = [
        'rsi',                 # RSI value (0-100)
        'adx',                 # ADX value (0-100)
        'atr_pct',             # ATR as % of price
        'dist_to_swing_high',  # Distance to resistance
        'dist_to_swing_low',   # Distance to support
        'kernel_direction',    # 1 (bullish) or -1 (bearish)
        'signal_current',      # Current ML signal
        'volatility_high',     # Volatility filter status
    ]
    
    ALL_FEATURES = TRADE_FEATURES + MARKET_FEATURES
    
    def __init__(self, model_path: Optional[str] = None, config: Optional[Dict] = None):
        """
        Initialize Exit Manager.
        
        Args:
            model_path: Path to trained model file (.pkl)
            config: Configuration dict with thresholds and parameters
        """
        self.model = None
        self.model_path = model_path
        
        # Merge provided config with defaults
        self.config = self._default_config()
        if config:
            self.config.update(config)
        
        # Rule-based fallback thresholds (cached for speed)
        self.partial_exit_profit = self.config.get('partial_exit_profit', 0.03)  # 3%
        self.tighten_sl_drawdown = self.config.get('tighten_sl_drawdown', 0.02)  # 2%
        self.full_exit_signal_weight = self.config.get('full_exit_signal_weight', 0.7)
        
        # Load model if path provided
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
        else:
            logger.info("ExitManager initialized in rule-based mode (no ML model)")
    
    def _default_config(self) -> Dict:
        """Default configuration for rule-based fallback."""
        return {
            'partial_exit_profit': 0.03,      # Trigger partial at 3% profit
            'partial_exit_amount': 0.50,      # Close 50% of position
            'tighten_sl_drawdown': 0.02,      # Tighten SL after 2% drawdown from max
            'tighten_sl_factor': 0.5,         # Move SL to 50% of distance
            'full_exit_signal_weight': 0.7,   # Weight for signal-based exit
            'min_profit_for_exit': 0.001,     # Minimum profit for full exit
        }
    
    def _load_model(self, path: str) -> bool:
        """Load trained ML model from disk."""
        try:
            with open(path, 'rb') as f:
                self.model = pickle.load(f)
            logger.info(f"✅ ExitManager model loaded from {path}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load ExitManager model: {e}")
            self.model = None
            return False
    
    def extract_features(self, trade: Any, current_rate: float, 
                         dataframe: pd.DataFrame, current_profit: float) -> Dict[str, float]:
        """
        Extract features for ML prediction.
        
        Args:
            trade: Freqtrade Trade object
            current_rate: Current market price
            dataframe: Analyzed dataframe from strategy
            current_profit: Current profit ratio (e.g., 0.05 = 5%)
            
        Returns:
            Dictionary of feature values
        """
        features = {}
        
        # Trade Context Features
        features['current_profit'] = current_profit
        
        # Max profit (from trade object if available)
        if hasattr(trade, 'max_rate') and trade.max_rate:
            if trade.is_short:
                max_profit = (trade.open_rate - trade.min_rate) / trade.open_rate
            else:
                max_profit = (trade.max_rate - trade.open_rate) / trade.open_rate
        else:
            max_profit = current_profit
        features['max_profit'] = max_profit
        
        # Trade duration (in candles, approximated)
        if hasattr(trade, 'open_date_utc'):
            trade_duration_hours = (pd.Timestamp.utcnow() - pd.Timestamp(trade.open_date_utc)).total_seconds() / 3600
            # Approximate candles based on 1h timeframe (adjust if needed)
            features['trade_duration'] = trade_duration_hours
        else:
            features['trade_duration'] = 0
        
        # Profit velocity (simplified: current_profit / duration)
        if features['trade_duration'] > 0:
            features['profit_velocity'] = current_profit / features['trade_duration']
        else:
            features['profit_velocity'] = 0
        
        # Drawdown from max
        features['drawdown_from_max'] = max_profit - current_profit
        
        # Distance to entry
        features['dist_to_entry'] = (current_rate - trade.open_rate) / trade.open_rate
        
        # Market Features (from dataframe)
        last_candle = dataframe.iloc[-1] if len(dataframe) > 0 else {}
        
        # Indicators
        features['rsi'] = last_candle.get('rsi_f1', last_candle.get('rsi', 50)) / 100
        features['adx'] = last_candle.get('adx_f4', last_candle.get('adx', 25)) / 100
        features['atr_pct'] = last_candle.get('atr_sl', 0) / current_rate if current_rate > 0 else 0
        
        # Swing levels
        swing_high = last_candle.get('swing_high', current_rate * 1.05)
        swing_low = last_candle.get('swing_low', current_rate * 0.95)
        features['dist_to_swing_high'] = (swing_high - current_rate) / current_rate
        features['dist_to_swing_low'] = (current_rate - swing_low) / current_rate
        
        # Strategy signals
        features['kernel_direction'] = 1 if last_candle.get('is_bullish_kernel', True) else -1
        features['signal_current'] = last_candle.get('signal', 0)
        features['volatility_high'] = 1 if last_candle.get('volatility_filter_ok', False) else 0
        
        return features
    
    def predict_action(self, trade: Any, current_rate: float,
                       dataframe: pd.DataFrame, current_profit: float) -> ExitRecommendation:
        """
        Predict the optimal exit action for the current trade state.
        
        Args:
            trade: Freqtrade Trade object
            current_rate: Current market price
            dataframe: Analyzed dataframe from strategy
            current_profit: Current profit ratio
            
        Returns:
            ExitRecommendation with action, confidence, and details
        """
        # Extract features
        features = self.extract_features(trade, current_rate, dataframe, current_profit)
        
        # Use ML model if available
        if self.model is not None:
            return self._predict_ml(features, trade.is_short)
        else:
            return self._predict_rules(features, trade.is_short)
    
    def _predict_ml(self, features: Dict[str, float], is_short: bool) -> ExitRecommendation:
        """ML-based prediction using trained model."""
        try:
            # Prepare feature vector
            X = pd.DataFrame([features])[self.ALL_FEATURES]
            X = X.fillna(0)
            
            # Get prediction and probabilities
            action_idx = self.model.predict(X)[0]
            probas = self.model.predict_proba(X)[0]
            
            # Map index to action
            action_map = {
                0: ExitAction.HOLD,
                1: ExitAction.PARTIAL_EXIT,
                2: ExitAction.TIGHTEN_SL,
                3: ExitAction.FULL_EXIT,
            }
            action = action_map.get(action_idx, ExitAction.HOLD)
            confidence = probas[action_idx]
            
            # Generate details based on action
            details = self._generate_action_details(action, features, is_short)
            
            return ExitRecommendation(action=action, confidence=confidence, details=details)
            
        except Exception as e:
            logger.error(f"ML prediction failed: {e}, falling back to rules")
            return self._predict_rules(features, is_short)
    
    def _predict_rules(self, features: Dict[str, float], is_short: bool) -> ExitRecommendation:
        """Rule-based prediction as fallback."""
        current_profit = features['current_profit']
        drawdown = features['drawdown_from_max']
        kernel_direction = features['kernel_direction']
        signal = features['signal_current']
        
        # Rule 1: Full exit on signal flip with profit
        expected_signal = -1 if is_short else 1
        if signal != 0 and signal != expected_signal and current_profit > self.config['min_profit_for_exit']:
            return ExitRecommendation(
                action=ExitAction.FULL_EXIT,
                confidence=0.85,
                details={'reason': 'signal_flip', 'signal': signal}
            )
        
        # Rule 2: Tighten SL after drawdown from max (with profit)
        if drawdown > self.config['tighten_sl_drawdown'] and current_profit > 0:
            return ExitRecommendation(
                action=ExitAction.TIGHTEN_SL,
                confidence=0.70,
                details={
                    'reason': 'drawdown_protection',
                    'drawdown': drawdown,
                    'factor': self.config['tighten_sl_factor']
                }
            )
        
        # Rule 3: Partial exit at profit threshold
        if current_profit >= self.config['partial_exit_profit']:
            return ExitRecommendation(
                action=ExitAction.PARTIAL_EXIT,
                confidence=0.75,
                details={
                    'reason': 'profit_target',
                    'amount': self.config['partial_exit_amount']
                }
            )
        
        # Rule 4: Full exit on kernel direction change (with profit)
        expected_kernel = -1 if is_short else 1
        if kernel_direction != expected_kernel and current_profit > self.config['min_profit_for_exit']:
            return ExitRecommendation(
                action=ExitAction.FULL_EXIT,
                confidence=0.65,
                details={'reason': 'kernel_flip', 'kernel': kernel_direction}
            )
        
        # Default: HOLD
        return ExitRecommendation(
            action=ExitAction.HOLD,
            confidence=0.60,
            details={'reason': 'no_action_triggered'}
        )
    
    def _generate_action_details(self, action: ExitAction, 
                                  features: Dict[str, float], is_short: bool) -> Dict:
        """Generate action-specific details."""
        details = {}
        
        if action == ExitAction.PARTIAL_EXIT:
            details['amount'] = self.config['partial_exit_amount']
            details['profit'] = features['current_profit']
            
        elif action == ExitAction.TIGHTEN_SL:
            details['factor'] = self.config['tighten_sl_factor']
            details['drawdown'] = features['drawdown_from_max']
            
        elif action == ExitAction.FULL_EXIT:
            details['profit'] = features['current_profit']
            details['signal'] = features['signal_current']
            
        details['is_short'] = is_short
        return details


# Convenience functions for strategy integration
def create_exit_manager(strategy_config: Dict) -> ExitManager:
    """
    Factory function to create ExitManager from strategy config.
    
    Usage in strategy __init__:
        self.exit_manager = create_exit_manager(config)
    """
    model_path = strategy_config.get('exit_manager_model', None)
    exit_config = strategy_config.get('exit_manager_config', {})
    
    return ExitManager(model_path=model_path, config=exit_config)
