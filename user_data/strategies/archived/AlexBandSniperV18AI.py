# --- Do not remove these libs ---
from freqtrade.exchange import timeframe_to_minutes, timeframe_to_prev_date
from pathlib import Path
import logging
import time
current_time = time.time()
import datetime
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import csv
pd.options.mode.chained_assignment = None
from pandas import DataFrame, Series
from technical.util import resample_to_interval, resampled_merge
from freqtrade.strategy import (IStrategy, BooleanParameter, CategoricalParameter, DecimalParameter,
                                IntParameter, RealParameter, merge_informative_pair, stoploss_from_open,
                                stoploss_from_absolute, merge_informative_pair)
from freqtrade.persistence import Trade
from typing import List, Tuple, Optional, Dict, Any
from freqtrade.optimize.space import Categorical, Dimension, Integer, SKDecimal
import json
import os
# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from collections import deque
import optuna
from optuna.samplers import TPESampler
from optuna.exceptions import OptunaError
import warnings
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

logger = logging.getLogger(__name__)
ASCII_ART = '''
    ╔════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
    ║                                                                                                                        ║
    ║  ██████╗ ██╗     ███████╗██╗  ██╗ ██████╗██████╗ ██╗   ██╗██████╗ ████████╗ ██████╗ ██╗  ██╗██╗███╗   ██╗ ██████╗      ║
    ║  ██╔══██╗██║     ██╔════╝╚██╗██╔╝██╔════╝██╔══██╗╚██╗ ██╔╝██╔══██╗╚══██╔══╝██╔═══██╗██║ ██╔╝██║████╗  ██║██╔════╝      ║
    ║  ███████║██║     █████╗   ╚███╔╝ ██║     ██████╔╝ ╚████╔╝ ██████╔╝   ██║   ██║   ██║█████╔╝ ██║██╔██╗ ██║██║  ███╗     ║
    ║  ██╔══██║██║     ██╔══╝   ██╔██╗ ██║     ██╔══██╗  ╚██╔╝  ██╔═══╝    ██║   ██║   ██║██╔═██╗ ██║██║╚██╗██║██║   ██║     ║
    ║  ██║  ██║███████╗███████╗██╔╝ ██╗╚██████╗██║  ██║   ██║   ██║        ██║   ╚██████╔╝██║  ██╗██║██║ ╚████║╚██████╔╝     ║
    ║  ╚═╝  ╚═╝╚══════╝╚══════╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝   ╚═╝   ╚═╝        ╚═╝    ╚═════╝ ╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝ ╚═════╝      ║
    ║                                                                                                                        ║
    ║                                       PURE OPTUNA OPTIMIZING STRATEGY                                                  ║
    ║                                         Enhanced Multi-Feature Optimazion                                              ║
    ║                                           Machine Learning Powered                                                     ║
    ║                                                                                                                        ║
    ╚════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
'''
# FIX: Remove the problematic relative import and create a simple inline replacement
# from .optuna_manager import OptunaManager
class RealOptunaManager:
    """Real Optuna-based optimization manager"""
    
    def __init__(self, strategy_name: str):
        self.strategy_name = strategy_name
        self.studies = {}  # Store studies per pair
        self.best_params_cache = {}
        self.performance_history = {}
        self.optimization_trigger_count = {}
        self.pretraining_enabled = True
        self.pretraining_days = 90
        self.pretraining_done = {}
        
        logger.info(f"📊 [OPTUNA] Initializing RealOptunaManager for strategy: {strategy_name}")
        logger.info(f"🎯 [OPTUNA] Real Optuna optimization enabled")
    
    def get_best_params(self, pair: str) -> Dict[str, Any]:
        """Get best parameters for a pair"""
        if pair in self.best_params_cache:
            return self.best_params_cache[pair]
        
        # Try to load from existing study
        try:
            study_name = f"{self.strategy_name}_{pair.replace('/', '_').replace(':', '_')}"
            study = optuna.load_study(
                study_name=study_name,
                storage=f"sqlite:///user_data/strategies/optuna_studies/optuna_{study_name}.db"
            )
            if study.trials:
                self.studies[pair] = study
                self.best_params_cache[pair] = study.best_params
                return study.best_params
        except:
            pass
        
        return None
    
    def should_optimize(self, pair: str) -> bool:
        """Check if optimization should be triggered"""
        if pair not in self.best_params_cache:
            return True
        
        if pair in self.performance_history:
            recent_performance = self.performance_history[pair][-10:]
            if len(recent_performance) >= 5:
                avg_performance = sum(recent_performance) / len(recent_performance)
                if avg_performance < -0.02:
                    return True
        
        return False
    
    def should_optimize_based_on_performance(self, pair: str) -> bool:
        """Check if performance-based optimization is needed"""
        if pair not in self.performance_history:
            return False
        
        recent_trades = self.performance_history[pair][-10:]
        if len(recent_trades) >= 5:
            avg_performance = sum(recent_trades) / len(recent_trades)
            return avg_performance < -0.015
        
        return False
    
    def update_performance(self, pair: str, profit_ratio: float):
        """Update performance tracking"""
        if pair not in self.performance_history:
            self.performance_history[pair] = []
        
        self.performance_history[pair].append(profit_ratio)
        
        if len(self.performance_history[pair]) > 50:
            self.performance_history[pair] = self.performance_history[pair][-50:]
        
        logger.debug(f"📈 [OPTUNA] Updated performance for {pair}: {profit_ratio:.4f}")
    
    def optimize_coin(self, pair: str, objective_func, n_trials: int = 15):
        """Optimize parameters for a specific coin using real Optuna"""
        logger.info(f"🚀 [OPTUNA] Starting REAL optimization for {pair} with {n_trials} trials")
        
        try:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            from pathlib import Path
            study_name = f"{self.strategy_name}_{pair.replace('/', '_').replace(':', '_')}"
            
            Path("user_data/strategies/optuna_studies").mkdir(parents=True, exist_ok=True)
            study = optuna.create_study(
                study_name=study_name,
                direction='maximize',
                sampler=TPESampler(),
                storage=f"sqlite:///user_data/strategies/optuna_studies/optuna_{study_name}.db",
                load_if_exists=True
            )
            
            study.optimize(objective_func, n_trials=n_trials)
            
            self.studies[pair] = study
            self.best_params_cache[pair] = study.best_params
            
            logger.info(f"✅ [OPTUNA] REAL optimization completed for {pair}")
            logger.info(f"🏆 [OPTUNA] Best value: {study.best_value:.4f}")
            self._log_formatted_parameters(pair, study.best_params)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ [OPTUNA] Real optimization failed for {pair}: {e}")
            return False

    def _log_formatted_parameters(self, pair: str, params: Dict[str, Any]):
        """Log parameters in a nicely formatted way - Enhanced with all optimized parameters"""
        logger.info(f"📈 [OPTUNA] Optimized parameters for {pair}:")
        
        # === INDICATOR PARAMETERS ===
        logger.info(f"┌─ 📊 TECHNICAL INDICATOR PARAMETERS")
        logger.info(f"│  • rsi_period: {params.get('rsi_period', 'N/A')}")
        logger.info(f"│  • willr_period: {params.get('willr_period', 'N/A')}")
        logger.info(f"│  • cci_period: {params.get('cci_period', 'N/A')}")
        logger.info(f"│  • bb_period: {params.get('bb_period', 'N/A')}")
        logger.info(f"│  • bb_std: {params.get('bb_std', 'N/A')}")
        logger.info(f"│  • macd_fast/slow/signal: {params.get('macd_fast', 'N/A')}/{params.get('macd_slow', 'N/A')}/{params.get('macd_signal', 'N/A')}")
        logger.info(f"│  • ema_short/long: {params.get('ema_short_period', 'N/A')}/{params.get('ema_long_period', 'N/A')}")
        logger.info(f"│  • volume_sma_period: {params.get('volume_sma_period', 'N/A')}")
        logger.info(f"│  • atr_period: {params.get('atr_period', 'N/A')}")
        logger.info(f"│  • pivot_window: {params.get('pivot_window', 'N/A')}")
        logger.info(f"│  • swing_period: {params.get('swing_period', 'N/A')}")
        
        # === CORE ENTRY PARAMETERS ===
        logger.info(f"├─ 🎯 CORE ENTRY PARAMETERS")
        logger.info(f"│  • min_divergence_count: {params.get('min_divergence_count', 'N/A')}")
        logger.info(f"│  • min_signal_strength: {params.get('min_signal_strength', 'N/A')}")
        logger.info(f"│  • volume_threshold: {params.get('volume_threshold', 'N/A')}")
        logger.info(f"│  • volume_factor: {params.get('volume_factor', 'N/A')}")
        logger.info(f"│  • atr_multiplier: {params.get('atr_multiplier', 'N/A')}")
        logger.info(f"│  • adx_threshold: {params.get('adx_threshold', 'N/A')}")
        logger.info(f"│  • max_volatility: {params.get('max_volatility', 'N/A')}")
        logger.info(f"│  • min_volatility: {params.get('min_volatility', 'N/A')}")
        
        # === RSI PARAMETERS ===
        logger.info(f"├─ 📈 RSI TIMING PARAMETERS")
        logger.info(f"│  • rsi_overbought: {params.get('rsi_overbought', 'N/A')}")
        logger.info(f"│  • rsi_oversold: {params.get('rsi_oversold', 'N/A')}")
        logger.info(f"│  • rsi_long_upper_tight: {params.get('rsi_long_upper_tight', 'N/A')}")
        logger.info(f"│  • rsi_short_lower_tight: {params.get('rsi_short_lower_tight', 'N/A')}")
        logger.info(f"│  • rsi_recovery_periods: {params.get('rsi_recovery_periods', 'N/A')}")
        
        # === VOLUME PARAMETERS ===
        logger.info(f"├─ 📊 VOLUME PARAMETERS")
        logger.info(f"│  • volume_ratio_min: {params.get('volume_ratio_min', 'N/A')}")
        logger.info(f"│  • volume_trend_multiplier: {params.get('volume_trend_multiplier', 'N/A')}")
        logger.info(f"│  • volume_momentum_multiplier: {params.get('volume_momentum_multiplier', 'N/A')}")
        logger.info(f"│  • volume_breakout_multiplier: {params.get('volume_breakout_multiplier', 'N/A')}")
        logger.info(f"│  • volume_breakout_rsi_min/max: {params.get('volume_breakout_rsi_min', 'N/A')}/{params.get('volume_breakout_rsi_max', 'N/A')}")
        
        # === POSITION TIMING ===
        logger.info(f"├─ ⏰ POSITION TIMING")
        logger.info(f"│  • bb_percent_long_max: {params.get('bb_percent_long_max', 'N/A')}")
        logger.info(f"│  • bb_percent_short_min: {params.get('bb_percent_short_min', 'N/A')}")
        logger.info(f"│  • price_position_long_max: {params.get('price_position_long_max', 'N/A')}")
        logger.info(f"│  • price_position_short_min: {params.get('price_position_short_min', 'N/A')}")
        
        # === MARKET STRUCTURE ===
        logger.info(f"├─ 🏗️ MARKET STRUCTURE")
        logger.info(f"│  • adx_trending_min: {params.get('adx_trending_min', 'N/A')}")
        logger.info(f"│  • chop_threshold: {params.get('chop_threshold', 'N/A')}")
        logger.info(f"│  • min_oversold_conditions: {params.get('min_oversold_conditions', 'N/A')}")
        logger.info(f"│  • min_overbought_conditions: {params.get('min_overbought_conditions', 'N/A')}")
        logger.info(f"│  • min_reversal_signals: {params.get('min_reversal_signals', 'N/A')}")
        logger.info(f"│  • min_trend_filters: {params.get('min_trend_filters', 'N/A')}")
        
        # === MEAN REVERSION PARAMETERS ===
        logger.info(f"├─ 🔄 MEAN REVERSION PARAMETERS")
        logger.info(f"│  • bb_oversold_threshold: {params.get('bb_oversold_threshold', 'N/A')}")
        logger.info(f"│  • bb_overbought_threshold: {params.get('bb_overbought_threshold', 'N/A')}")
        logger.info(f"│  • mean_reversion_rsi_oversold: {params.get('mean_reversion_rsi_oversold', 'N/A')}")
        logger.info(f"│  • mean_reversion_rsi_overbought: {params.get('mean_reversion_rsi_overbought', 'N/A')}")
        
        # === MOMENTUM CONTINUATION PARAMETERS ===
        logger.info(f"├─ 🚀 MOMENTUM CONTINUATION PARAMETERS")
        logger.info(f"│  • momentum_ema_periods: {params.get('momentum_ema_periods', 'N/A')}")
        logger.info(f"│  • momentum_strength_min: {params.get('momentum_strength_min', 'N/A')}")
        logger.info(f"│  • momentum_pullback_max: {params.get('momentum_pullback_max', 'N/A')}")
        logger.info(f"│  • momentum_pullback_min: {params.get('momentum_pullback_min', 'N/A')}")
        
        # === EXIT PARAMETERS ===
        logger.info(f"├─ 🎯 GLOBAL EXIT SETTINGS")
        logger.info(f"│  • session_multiplier_overlap: {params.get('session_multiplier_overlap', 'N/A')}")
        logger.info(f"│  • session_multiplier_major: {params.get('session_multiplier_major', 'N/A')}")
        logger.info(f"│  • volatility_sensitivity: {params.get('volatility_sensitivity', 'N/A')}")
        logger.info(f"│  • high_signal_multiplier: {params.get('high_signal_multiplier', 'N/A')}")
        logger.info(f"│  • low_signal_multiplier: {params.get('low_signal_multiplier', 'N/A')}")
        
        logger.info(f"├─ 🔄 MEAN REVERSION (MR1) EXITS")
        logger.info(f"│  • mr1_max_hold_minutes: {params.get('mr1_max_hold_minutes', 'N/A')} min")
        logger.info(f"│  • mr1_rsi_exit_long/short: {params.get('mr1_rsi_exit_long', 'N/A')}/{params.get('mr1_rsi_exit_short', 'N/A')}")
        logger.info(f"│  • mr1_quick_profit_target: {params.get('mr1_quick_profit_target', 'N/A')}")
        logger.info(f"│  • mr1_timeout_min_profit: {params.get('mr1_timeout_min_profit', 'N/A')}")
        
        logger.info(f"├─ 🚀 MOMENTUM CONTINUATION (MC1) EXITS")
        logger.info(f"│  • mc1_profit_target: {params.get('mc1_profit_target', 'N/A')}")
        logger.info(f"│  • mc1_max_hold_minutes: {params.get('mc1_max_hold_minutes', 'N/A')} min")
        logger.info(f"│  • mc1_adx_exit_threshold: {params.get('mc1_adx_exit_threshold', 'N/A')}")
        logger.info(f"│  • mc1_timeout_min_profit: {params.get('mc1_timeout_min_profit', 'N/A')}")
        
        logger.info(f"├─ 📊 VOLUME BREAKOUT (VB1) EXITS")
        logger.info(f"│  • vb1_volume_fade_threshold: {params.get('vb1_volume_fade_threshold', 'N/A')}")
        logger.info(f"│  • vb1_profit_target: {params.get('vb1_profit_target', 'N/A')}")
        logger.info(f"│  • vb1_quick_profit: {params.get('vb1_quick_profit', 'N/A')}")
        
        logger.info(f"├─ 🔄 REVERSAL (RSV1) EXITS")
        logger.info(f"│  • rsv1_rsi_recovery_threshold: {params.get('rsv1_rsi_recovery_threshold', 'N/A')}")
        logger.info(f"│  • rsv1_profit_target: {params.get('rsv1_profit_target', 'N/A')}")
        logger.info(f"│  • rsv1_max_hold_minutes: {params.get('rsv1_max_hold_minutes', 'N/A')} min")
        logger.info(f"│  • rsv1_min_timeout_profit: {params.get('rsv1_min_timeout_profit', 'N/A')}")
        
        logger.info(f"├─ 📈 TREND FOLLOWING EXITS")
        logger.info(f"│  • trend_profit_target: {params.get('trend_profit_target', 'N/A')}")
        logger.info(f"│  • trend_ema_break_periods: {params.get('trend_ema_break_periods', 'N/A')}")
        
        logger.info(f"├─ ⏰ UNIVERSAL EXIT PARAMETERS")
        logger.info(f"│  • medium_term_base_target: {params.get('medium_term_base_target', 'N/A')}")
        logger.info(f"│  • rsi_exit_overbought/oversold: {params.get('rsi_exit_overbought', 'N/A')}/{params.get('rsi_exit_oversold', 'N/A')}")
        logger.info(f"│  • momentum_fade_threshold: {params.get('momentum_fade_threshold', 'N/A')}")
        
        logger.info(f"├─ 🕐 EXTENDED DURATION MANAGEMENT")
        logger.info(f"│  • extended_2hr_target: {params.get('extended_2hr_target', 'N/A')}")
        logger.info(f"│  • extended_3hr_target: {params.get('extended_3hr_target', 'N/A')}")
        logger.info(f"│  • extended_5hr_target: {params.get('extended_5hr_target', 'N/A')}")
        
        logger.info(f"├─ 🛡️ MARKET PROTECTION")
        logger.info(f"│  • friday_close_min_profit: {params.get('friday_close_min_profit', 'N/A')}")
        logger.info(f"│  • overnight_min_profit: {params.get('overnight_min_profit', 'N/A')}")
        logger.info(f"│  • low_liquidity_min_profit: {params.get('low_liquidity_min_profit', 'N/A')}")
        
        # === RSV1 SPECIFIC PARAMETERS ===
        logger.info(f"├─ 🔄 REVERSAL (RSV1) ENTRY PARAMETERS")
        logger.info(f"│  • rsv1_rsi_oversold/overbought: {params.get('rsv1_rsi_oversold', 'N/A')}/{params.get('rsv1_rsi_overbought', 'N/A')}")
        logger.info(f"│  • rsv1_willr_oversold/overbought: {params.get('rsv1_willr_oversold', 'N/A')}/{params.get('rsv1_willr_overbought', 'N/A')}")
        logger.info(f"│  • rsv1_cci_oversold/overbought: {params.get('rsv1_cci_oversold', 'N/A')}/{params.get('rsv1_cci_overbought', 'N/A')}")
        logger.info(f"│  • rsv1_volume_factor: {params.get('rsv1_volume_factor', 'N/A')}")
        logger.info(f"│  • rsv1_min_conditions: {params.get('rsv1_min_oversold_conditions', 'N/A')}/{params.get('rsv1_min_overbought_conditions', 'N/A')}")
        
        # === ADVANCED ENTRY PARAMETERS ===
        logger.info(f"├─ 🎯 E1 PRIMARY ENTRY PARAMETERS")
        logger.info(f"│  • e1_rsi_thresholds: {params.get('e1_rsi_oversold_threshold', 'N/A')}-{params.get('e1_rsi_overbought_threshold', 'N/A')}")
        logger.info(f"│  • e1_rsi_buffers: {params.get('e1_rsi_oversold_buffer', 'N/A')}/{params.get('e1_rsi_overbought_buffer', 'N/A')}")
        logger.info(f"│  • volatility_multipliers: {params.get('volatility_min_multiplier', 'N/A')}-{params.get('volatility_max_multiplier', 'N/A')}")
        
        logger.info(f"├─ 📈 TREND BREAKOUT PARAMETERS")
        logger.info(f"│  • trend_volume_window/multiplier: {params.get('trend_volume_window', 'N/A')}/{params.get('trend_volume_multiplier', 'N/A')}")
        logger.info(f"│  • trend_rsi_range: {params.get('trend_rsi_min', 'N/A')}-{params.get('trend_rsi_max', 'N/A')}")
        logger.info(f"│  • trend_adx_min: {params.get('trend_adx_min', 'N/A')}")
        
        logger.info(f"├─ 🚀 MOMENTUM ENTRY PARAMETERS")
        logger.info(f"│  • momentum_rsi_pivot/trigger: {params.get('momentum_rsi_pivot', 'N/A')}/{params.get('momentum_rsi_trigger', 'N/A')}")
        logger.info(f"│  • momentum_volume_multiplier: {params.get('momentum_volume_multiplier', 'N/A')}")
        logger.info(f"│  • momentum_adx_min: {params.get('momentum_adx_min', 'N/A')}")
        
        # === SUPPORT/RESISTANCE & BREAKDOWN ===
        logger.info(f"├─ 📉 BREAKDOWN/BREAKOUT PARAMETERS")
        logger.info(f"│  • breakdown_volume_threshold: {params.get('breakdown_volume_threshold', 'N/A')}")
        logger.info(f"│  • breakdown_adx_threshold: {params.get('breakdown_adx_threshold', 'N/A')}")
        logger.info(f"│  • breakdown_rsi_range: {params.get('breakdown_rsi_lower', 'N/A')}-{params.get('breakdown_rsi_upper', 'N/A')}")
        logger.info(f"│  • support_tolerance: {params.get('support_tolerance', 'N/A')}")
        logger.info(f"│  • resistance_tolerance: {params.get('resistance_tolerance', 'N/A')}")
        
        # === SECONDARY/TERTIARY ENTRIES ===
        logger.info(f"├─ 📊 SECONDARY/TERTIARY ENTRIES")
        logger.info(f"│  • secondary_rsi_range: {params.get('secondary_rsi_min', 'N/A')}-{params.get('secondary_rsi_max', 'N/A')}")
        logger.info(f"│  • tertiary_volume_multiplier: {params.get('tertiary_volume_multiplier', 'N/A')}")
        logger.info(f"│  • sixth_min_div_count: {params.get('sixth_min_div_count', 'N/A')}")
        logger.info(f"│  • seventh_atr_multiplier: {params.get('seventh_atr_multiplier', 'N/A')}")
        
        # === SUMMARY ===
        logger.info(f"├─ 📋 OPTIMIZATION SUMMARY")
        entry_params = len([k for k in params.keys() if any(x in k for x in ['e1_', 'trend_', 'momentum_', 'secondary_', 'tertiary_', 'rsv1_rsi_', 'breakdown_', 'div_'])])
        exit_params = len([k for k in params.keys() if any(x in k for x in ['mr1_', 'mc1_', 'vb1_', 'rsv1_profit', 'extended_', 'session_'])])
        indicator_params = len([k for k in params.keys() if any(x in k for x in ['rsi_period', 'bb_period', 'macd_', 'ema_', 'volume_sma', 'atr_period'])])
        
        logger.info(f"│  • Entry parameters optimized: {entry_params}")
        logger.info(f"│  • Exit parameters optimized: {exit_params}")
        logger.info(f"│  • Indicator parameters optimized: {indicator_params}")
        logger.info(f"│  • Total parameters in study: {len(params)}")
        logger.info(f"└─ 🎯 Strategy fully optimized with {len(params)} parameters")
        logger.info(f"📊 [OPTUNA] Comprehensive optimization completed for {pair}")

    def pretrain_with_historical_data(self, pair: str, strategy_instance) -> bool:
        """Stub for compatibility - real Optuna doesn't need pre-training"""
        return False

class PlotConfig():
    def __init__(self):
        self.config = {
            'main_plot': {
                # Try direct column names first to test
                'bollinger_upperband': {'color': 'rgba(4,137,122,0.7)'},
                'kc_upperband': {'color': 'rgba(4,146,250,0.7)'},
                'kc_middleband': {'color': 'rgba(4,146,250,0.7)'},
                'kc_lowerband': {'color': 'rgba(4,146,250,0.7)'},
                'bollinger_lowerband': {
                    'color': 'rgba(4,137,122,0.7)',
                    'fill_to': 'bollinger_upperband',
                    'fill_color': 'rgba(4,137,122,0.07)'
                },
                'ema9': {'color': 'purple'},
                'ema20': {'color': 'yellow'},
                'ema50': {'color': 'red'},
                'ema200': {'color': 'white'},
                'trend_1h_1h': {'color': 'orange'},
            },
            'subplots': {
                "RSI": {
                    'rsi': {'color': 'green'}
                },
                "ATR": {
                    'atr': {'color': 'firebrick'}
                },
                "Signal Strength": {
                    'signal_strength': {'color': 'blue'}
                }
            }
        }
    
    def add_total_divergences_in_config(self, dataframe):
        # Test if columns exist before adding them
        if 'total_bullish_divergences' in dataframe.columns:
            self.config['main_plot']['total_bullish_divergences'] = {
                "plotly": {
                    'mode': 'markers',
                    'marker': {
                        'symbol': 'diamond',
                        'size': 11,
                        'color': 'green'
                    }
                }
            }
        
        if 'total_bearish_divergences' in dataframe.columns:
            self.config['main_plot']['total_bearish_divergences'] = {
                "plotly": {
                    'mode': 'markers',
                    'marker': {
                        'symbol': 'diamond',
                        'size': 11,
                        'color': 'crimson'
                    }
                }
            }
        
        return self

class AlexBandSniperV18AI(IStrategy):
    """
    ╔════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
    ║                                                                                                                        ║
    ║  ██████╗ ██╗     ███████╗██╗  ██╗ ██████╗██████╗ ██╗   ██╗██████╗ ████████╗ ██████╗ ██╗  ██╗██╗███╗   ██╗ ██████╗      ║
    ║  ██╔══██╗██║     ██╔════╝╚██╗██╔╝██╔════╝██╔══██╗╚██╗ ██╔╝██╔══██╗╚══██╔══╝██╔═══██╗██║ ██╔╝██║████╗  ██║██╔════╝      ║
    ║  ███████║██║     █████╗   ╚███╔╝ ██║     ██████╔╝ ╚████╔╝ ██████╔╝   ██║   ██║   ██║█████╔╝ ██║██╔██╗ ██║██║  ███╗     ║
    ║  ██╔══██║██║     ██╔══╝   ██╔██╗ ██║     ██╔══██╗  ╚██╔╝  ██╔═══╝    ██║   ██║   ██║██╔═██╗ ██║██║╚██╗██║██║   ██║     ║
    ║  ██║  ██║███████╗███████╗██╔╝ ██╗╚██████╗██║  ██║   ██║   ██║        ██║   ╚██████╔╝██║  ██╗██║██║ ╚████║╚██████╔╝     ║
    ║  ╚═╝  ╚═╝╚══════╝╚══════╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝   ╚═╝   ╚═╝        ╚═╝    ╚═════╝ ╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝ ╚═════╝      ║
    ║                                                                                                                        ║
    ║                                            Advanced RL + Optuna Trading Strategy                                       ║
    ║                                               Enhanced Multi-Pair Optimization                                         ║
    ║                                                 Machine Learning Powered                                               ║
    ║                                                                                                                        ║
    ╚════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

    Alex BandSniper on 15m Timeframe - OPTIMIZED VERSION WITH OPTUNA INTEGRATION
    Version 106CO-Optuna - Claude optimized Entry & Exit with Optuna Management  # CHANGED: Versionshinweis angepasst
    Key improvements:
    - Trend Singals adaption
    - Fixed Entry Tags Assignment
    - Removed - Bull Pullbak Momentum & Bear Momentum Breakout
    - Bull Pullbak Momentum & Bear Momentum Breakout
    - Removed RSV1 Signals because of worse results
    - Added better Custom_Exit
    - Added Signals from V52C

    - Enabled ROI and TRAILING
    - Dynamic Trailing & New Custom Exits
    - Integrated OPTUNA Fully
    - Opimizing Daily Optimizing
    - Enhanced Parameters for Optuna
    - Optuna integration
    - Fixed ROI and Trailing adjusted Custom Exits
    - Fixed Entry Signals
    - Included 1h Informative Timeframe
    - Multi-timeframe analysis (1h trend confirmation)
    - Enhanced signal filtering with minimum divergence counts
    - Volume and volatility filters
    - Adaptive position sizing based on signal strength
    - Improved risk management
    - ADDED: Optuna-based parameter optimization per coin  # ADDED: Neuer Kommentar
    - ADDED: Historical pre-training for better startup performance  # ADDED: Neuer Kommentar
    - ADDED: Dynamic parameter adjustment based on performance  # ADDED: Neuer Kommentar
    """
    INTERFACE_VERSION = 3

    def version(self) -> str:
        return "v5-Optuna"

    class HyperOpt:
        # Define a custom stoploss space.
        def stoploss_space():
            return [SKDecimal(-0.15, -0.03, decimals=2, name='stoploss')]

        # Define a custom max_open_trades space
        def max_open_trades_space() -> List[Dimension]:
            return [
                Integer(3, 8, name='max_open_trades'),
            ]
        
        def trailing_space() -> List[Dimension]:
            return [
                Categorical([True], name='trailing_stop'),
                SKDecimal(0.02, 0.3, decimals=2, name='trailing_stop_positive'),
                SKDecimal(0.03, 0.1, decimals=2, name='trailing_stop_positive_offset_p1'),
                Categorical([True, False], name='trailing_only_offset_is_reached'),
            ]
    
    # Minimal ROI designed for the strategy.
        minimal_roi = {
        "0": 0.336,
        "441": 0.131,
        "640": 0.061,
        "1932": 0
    }
    
    # Optimal stoploss designed for the strategy.
    stoploss = -0.07
    can_short = True
    use_custom_stoploss = False
    leverage_value = 10.0  # Reduced leverage for better risk management
    #trailing_stop = True
    #trailing_stop_positive = 0.26  # Start trailing at 2.5% profit
    #trailing_stop_positive_offset = 0.34  # Trail 0.8% behind peak
    #trailing_stop = True
    #trailing_stop_positive = 0.26        # Only trail after 40% profit (very high)
    #trailing_stop_positive_offset = 0.34 # Start trailing at 45% profit
    #trailing_only_offset_is_reached = True

    # Optimal timeframe for the strategy.
    timeframe = '15m'
    timeframe_minutes = timeframe_to_minutes(timeframe)

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the "exit_pricing" section in the config.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    use_custom_exits = True
    use_entry_signal = True
    # In your hyperopt parameters, consider these more permissive defaults:
    min_divergence_count = IntParameter(1, 3, default=1, space='buy', optimize=True, load=True)  # Reduced from 2-5
    min_signal_strength = IntParameter(1, 5, default=5, space='buy', optimize=True, load=True)   # Reduced from 3-10
    volume_threshold = DecimalParameter(1.0, 1.5, default=1.3, decimals=1, space='buy', optimize=True, load=True)  # Reduced from 1.1-2.5

    # Make ADX less restrictive
    adx_threshold = IntParameter(15, 30, default=21, space='buy', optimize=True, load=True)  # Reduced from 25-45
  
    # Market Condition Filters
    rsi_overbought = DecimalParameter(65.0, 85.0, default=78.3, decimals=1, space='buy', optimize=True, load=True)
    rsi_oversold = DecimalParameter(15.0, 35.0, default=18.7, decimals=1, space='buy', optimize=True, load=True)
    
    # Volatility Filters
    max_volatility = DecimalParameter(0.015, 0.035, default=0.032, decimals=3, space='buy', optimize=True, load=True)
    min_volatility = DecimalParameter(0.003, 0.008, default=0.007, decimals=3, space='buy', optimize=True, load=True)
    
    # Exit Parameters
    rsi_exit_overbought = DecimalParameter(70.0, 90.0, default=89.8, decimals=1, space='sell', optimize=True, load=True)
    rsi_exit_oversold = DecimalParameter(10.0, 30.0, default=19.2, decimals=1, space='sell', optimize=True, load=True)
    adx_exit_threshold = IntParameter(15, 30, default=21, space='sell', optimize=True, load=True)
    
    # Trend Confirmation Parameters
    trend_strength_threshold = IntParameter(20, 40, default=26, space='buy', optimize=True, load=True)
    
    # Technical Parameters
    window = IntParameter(3, 6, default=5, space="buy", optimize=True, load=True)
    zigzag_period = IntParameter(10, 50, default=49, space="buy", optimize=True, load=True)
    index_range = IntParameter(20, 50, default=36, space='buy', optimize=True, load=True)

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 2000

    # Protection parameters
    cooldown_lookback = IntParameter(2, 48, default=1, space="protection", optimize=True)
    stop_duration = IntParameter(12, 200, default=20, space="protection", optimize=True)
    use_stop_protection = BooleanParameter(default=False, space="protection", optimize=True)
    use_cooldown_protection = BooleanParameter(default=False, space="protection", optimize=True)

    # Enhanced protection parameters
    use_max_drawdown_protection = BooleanParameter(default=False, space="protection", optimize=True)
    max_drawdown_lookback = IntParameter(100, 300, default=200, space="protection", optimize=True)
    max_drawdown_trade_limit = IntParameter(5, 15, default=10, space="protection", optimize=True)
    max_drawdown_stop_duration = IntParameter(1, 5, default=1, space="protection", optimize=True)
    max_allowed_drawdown = DecimalParameter(0.08, 0.25, default=0.15, decimals=2, space="protection", optimize=True)

    stoploss_guard_lookback = IntParameter(30, 80, default=50, space="protection", optimize=True)
    stoploss_guard_trade_limit = IntParameter(2, 6, default=3, space="protection", optimize=True)
    stoploss_guard_only_per_pair = BooleanParameter(default=True, space="protection", optimize=True)


    # Optional order type mapping.
    order_types = {
        'entry': 'market',
        'exit': 'market',
        'stoploss': 'limit',
        'stoploss_on_exchange': True
    }

    # Optional order time in force.
    order_time_in_force = {
        'entry': 'gtc',
        'exit': 'gtc'
    }

    plot_config = None

    def __init__(self, config: dict):
        # Signal Performance Tracking
        self.signal_performance_file = Path("user_data/strategies/AlexBandSniper_Signal_Performance.csv")
        self.signal_performance = self.load_signal_performance()
        self.signal_locks = {}
        
        # Configurable parameters - add these to your get_default_params()
        self.performance_window = 20        # More data points (was 15)
        self.min_trades_for_eval = 12      # More trades before evaluation (was 8)  
        self.min_win_rate = 0.25           # 25% win rate (was 35%)
        self.min_avg_profit = -0.025       # -2.5% average (was -1.5%)
        self.lock_duration_hours = 8       # Shorter lock period (was 12)
        super().__init__(config)
        print(ASCII_ART, flush=True)
        logger.info(f"🚀 [STRATEGY] Initializing {self.__class__.__name__}")  # AUTO-UPDATED: Klassenname wird automatisch angepasst
        
        # Initialize simple Optuna manager replacement
        try:
            self.optuna_manager = RealOptunaManager("AlexBandSniperV5AI")  # CHANGED: Strategy name angepasst
            logger.info(f"✅ [OPTUNA] Successfully initialized RealOptunaManager")
        except Exception as e:
            logger.error(f"❌ [OPTUNA] Failed to initialize RealOptunaManager: {e}")
            self.optuna_manager = None
        
        # Per-coin optimized parameters
        self.coin_params: Dict[str, Dict] = {}  # ADDED: Dictionary für coin-spezifische Parameter
        
        # Track performance for optimization
        self.coin_performance: Dict[str, float] = {}  # ADDED: Performance-Tracking pro Coin
        
        # Optimization settings - CHANGED: Enable proactive optimization
        self.enable_dynamic_optimization = True  # ADDED: Dynamische Optimierung aktiviert (deaktivieren für Backtest)
        self.optimization_min_trades = 0  # CHANGED: No minimum trades required for startup optimization
        
        logger.info(f"⚙️  [STRATEGY] Dynamic optimization: {'ENABLED' if self.enable_dynamic_optimization else 'DISABLED'}")
        logger.info(f"⚙️  [STRATEGY] Minimum trades for optimization: {self.optimization_min_trades}")
        logger.info(f"✅ [STRATEGY] {self.__class__.__name__} initialization completed")

    # ...existing code...
    
    def get_coin_params(self, pair: str) -> Dict[str, Any]:
        """Get optimized parameters for specific coin using real Optuna"""
        if pair not in self.coin_params:
            logger.debug(f"🔍 [OPTUNA] Loading parameters for new pair: {pair}")
            
            # Check if optimization is disabled first (for backtesting)
            if not self.enable_dynamic_optimization:
                logger.info(f"⏸️ [OPTUNA] Dynamic optimization DISABLED for {pair} - using default parameters")
                self.coin_params[pair] = self.get_default_params()
                return self.coin_params[pair]
            
            # Real Optuna optimization - only if enabled
            if self.enable_dynamic_optimization and self.optuna_manager:
                logger.info(f"🚀 [OPTUNA] Real Optuna will optimize {pair} based on actual trade results")
            
            if self.optuna_manager:
                best_params = self.optuna_manager.get_best_params(pair)
                
                if best_params:
                    self.coin_params[pair] = best_params
                    logger.info(f"✅ [OPTUNA] Loaded optimized parameters for {pair}: {len(best_params)} params")
                    logger.debug(f"📊 [OPTUNA] Parameters from study with {len(self.optuna_manager.studies.get(pair, {}).trials) if pair in self.optuna_manager.studies else 0} trials")
                else:
                    self.coin_params[pair] = self.get_default_params()
                    logger.info(f"🔧 [OPTUNA] Using default parameters for {pair} (no study found - will optimize after trades)")
                    
                    # ADD THIS SECTION: Force startup optimization for new pairs
                    if self.enable_dynamic_optimization and self.optuna_manager:
                        logger.info(f"🎯 [OPTUNA] Triggering STARTUP OPTIMIZATION for new pair: {pair}")
                        try:
                            # Call maybe_optimize_coin with force_startup=True
                            self.maybe_optimize_coin(pair, force_startup=True)
                            
                            # After optimization, try to reload the optimized parameters
                            optimized_params = self.optuna_manager.get_best_params(pair)
                            if optimized_params:
                                self.coin_params[pair] = optimized_params
                                logger.info(f"🎉 [OPTUNA] Successfully loaded STARTUP optimized parameters for {pair}")
                            else:
                                logger.warning(f"⚠️ [OPTUNA] Startup optimization completed but no parameters returned for {pair}")
                        except Exception as e:
                            logger.error(f"❌ [OPTUNA] Startup optimization failed for {pair}: {e}")
                            # Keep using default parameters if optimization fails
            else:
                self.coin_params[pair] = self.get_default_params()
                logger.warning(f"⚠️ [OPTUNA] RealOptunaManager not available, using default parameters for {pair}")
        
        # ENSURE ALL PARAMETERS ARE PRESENT
        default_params = self.get_default_params()
        current_params = self.coin_params.get(pair, {})
        
        # Merge defaults with current params (current params override defaults)
        complete_params = {**default_params, **current_params}
        self.coin_params[pair] = complete_params
        
        logger.debug(f"♻️ [OPTUNA] Using complete parameter set for {pair}")
        return self.coin_params[pair]
    
    def get_default_params(self) -> Dict[str, Any]:
        """Get enhanced default parameters including exit optimization"""
        return {
            # === EXISTING ENTRY PARAMETERS ===
            'min_divergence_count': 1,    # Default fallback only
            'min_signal_strength': 1,     # Default fallback only  
            'volume_threshold': 1.0,      # Default fallback only
            'adx_threshold': 15,          # Default fallback only
            'rsi_overbought': 80.0,       # Default fallback only
            'rsi_oversold': 15.0,         # Default fallback only
            'max_volatility': 0.025,      # Default fallback only
            'min_volatility': 0.005,      # Default fallback only
                    # === ADD THESE MISSING INDICATOR PARAMETERS ===

            # Technical Indicator Periods
            'volume_sma_period': 20,              # Volume SMA calculation
            'atr_period': 14,                     # ATR calculation
            'rsi_period': 14,                     # RSI period (if not already defined)
            'willr_period': 14,                   # Williams %R period
            'cci_period': 20,                     # CCI period
            'mom_period': 10,                     # Momentum period
            
            # Bollinger Bands Parameters  
            'bb_period': 20,                      # Bollinger Bands period
            'bb_std': 2.0,                        # Bollinger Bands standard deviation
            
            # EMA Parameters
            'ema_short_period': 8,                # Short EMA period
            'ema_long_period': 21,                # Long EMA period
            
            # MACD Parameters
            'macd_fast': 12,                      # MACD fast period
            'macd_slow': 26,                      # MACD slow period
            'macd_signal': 9,                     # MACD signal period
            
            # Stochastic Parameters
            'stoch_k': 14,                        # Stochastic K period
            'stoch_d': 3,                         # Stochastic D period
            
            # Market Structure Parameters
            'pivot_window': 10,
            'zigzag_period': 23,                   # Pivot points window
            'swing_period': 50,                   # Support/Resistance swing period
            'recent_high_low_period': 5,          # Recent highs/lows period
            'rolling_20_period': 20,              # 20-period rolling calculations
            'rolling_10_period': 10,              # 10-period rolling calculations
            
            # Market Analysis Parameters
            'chop_period': 14,                    # Choppiness index period
            'natr_period': 14,                    # NATR period
            'cmf_period': 20,                     # Chaikin Money Flow period
            # === EXISTING ENTRY TIMING PARAMETERS ===
            'rsi_recovery_periods': 3,
            'rsi_long_upper_tight': 65.0,
            'rsi_short_lower_tight': 35.0,
            'volume_ratio_min': 1.1,
            'volume_trend_multiplier': 1.5,
            'volume_momentum_multiplier': 1.2,
            'bb_percent_long_max': 0.25,
            'bb_percent_short_min': 0.75,
            'price_position_long_max': 0.3,
            'price_position_short_min': 0.7,
            'divergence_freshness_periods': 5,
            'min_strong_divergence_count': 2,
            'min_oversold_conditions': 3,
            'min_overbought_conditions': 3,
            'min_reversal_signals': 2,
            'min_trend_filters': 1,
            'chop_threshold': 61.8,
            'adx_trending_min': 25,
            'volume_breakout_multiplier': 2.5,
            'volume_breakout_rsi_min': 45.0,
            'volume_breakout_rsi_max': 60.0,
            'bb_oversold_threshold': 0.15,
            'bb_overbought_threshold': 0.85,
            'mean_reversion_rsi_oversold': 28.0,
            'mean_reversion_rsi_overbought': 72.0,
            'momentum_ema_periods': 9,
            'momentum_strength_min': 0.025,
            'momentum_pullback_max': 0.35,
            'momentum_pullback_min': 0.65,
            
            # === NEW EXIT PARAMETERS ===
            
            # Global Exit Settings
            'emergency_profit_limit': 0.25,
            'session_multiplier_overlap': 1.3,
            'session_multiplier_major': 1.15,
            'session_multiplier_quiet': 0.85,
            'volatility_sensitivity': 40,
            'high_signal_multiplier': 1.2,
            'low_signal_multiplier': 0.8,
            
            # Mean Reversion (MR1) Exit Parameters
            'mr1_emergency_stop': -0.015,           # NEW - Hard stop at -1.5%
            'mr1_max_hold_minutes': 20,             # CHANGED from 35
            'mr1_rsi_exit_long': 62.0,              # CHANGED from 68.0
            'mr1_rsi_exit_short': 38.0,             # CHANGED from 32.0
            'mr1_quick_profit_target': 0.015,       # CHANGED from 0.025
            'mr1_timeout_min_profit': 0.005,        # CHANGED from 0.008
            'mr1_bb_exit_long': 0.75,               # CHANGED from 0.7
            'mr1_bb_exit_short': 0.25,              # CHANGED from 0.3
            
            # Momentum Continuation (MC1) Exit Parameters
            'mc1_profit_target': 0.04,
            'mc1_momentum_break_threshold': 0.015,
            'mc1_max_hold_minutes': 45,
            'mc1_adx_exit_threshold': 22,
            'mc1_timeout_min_profit': 0.01,
            
            # Volume Breakout (VB1) Exit Parameters
            'vb1_volume_fade_threshold': 1.15,
            'vb1_profit_target': 0.03,
            'vb1_quick_profit': 0.02,
            'vb1_min_volatility': 0.012,
            
            # Reversal (RSV1) Exit Parameters
            'rsv1_rsi_recovery_threshold': 58.0,
            'rsv1_profit_target': 0.035,
            'rsv1_max_hold_minutes': 60,
            'rsv1_min_timeout_profit': 0.01,
            
            # Trend Following Exit Parameters
            'trend_profit_target': 0.045,
            'trend_ema_break_periods': 3,
            
            # Universal Exit Parameters
            'medium_term_base_target': 0.05,
            'reversal_min_profit_long': 0.008,
            'reversal_min_profit_short': 0.008,
            'rsi_exit_overbought': 78,
            'rsi_exit_oversold': 22,
            'rsi_extreme_min_profit': 0.015,
            'momentum_fade_min_profit': 0.012,
            'momentum_fade_threshold': 0.02,
            
            # Extended Duration Management
            'extended_2hr_target': 0.025,
            'extended_3hr_target': 0.015,
            'extended_5hr_target': 0.008,
            
            # Market Protection
            'friday_close_min_profit': 0.012,
            'overnight_min_profit': 0.008,
            'low_liquidity_min_profit': 0.01,
            'rsv1_rsi_oversold': 15,
            'rsv1_rsi_overbought': 80,
            'rsv1_willr_oversold': -80,
            'rsv1_willr_overbought': -20,
            'rsv1_cci_oversold': -100,
            'rsv1_cci_overbought': 100,
            'rsv1_bb_oversold': 0.1,
            'rsv1_bb_overbought': 0.9,
            'rsv1_volume_factor': 1.5,
            'rsv1_price_position_low': 0.2,
            'rsv1_price_position_high': 0.8,
            'rsv1_min_oversold_conditions': 3,
            'rsv1_min_overbought_conditions': 3,
            'rsv1_min_reversal_signals': 2,
            'rsv1_min_trend_filters': 1,
            # Add these to the existing get_default_params() method:

            # E1 (Primary) Parameters
            'e1_rsi_overbought_threshold': 75,
            'e1_rsi_overbought_buffer': 5,
            'e1_rsi_oversold_min': 25,
            'e1_rsi_oversold_threshold': 25,
            'e1_rsi_oversold_buffer': 5,
            'e1_rsi_overbought_max': 75,
            'volatility_min_multiplier': 0.5,
            'volatility_max_multiplier': 2.0,

            # Trend Parameters
            'trend_volume_window': 20,
            'trend_volume_multiplier': 1.2,
            'trend_rsi_min': 35,
            'trend_rsi_max': 70,
            'trend_close_lookback': 2,
            'trend_atr_window': 20,
            'trend_atr_multiplier': 0.8,
            'trend_adx_min': 20,

            # Momentum Parameters
            'momentum_rsi_lookback': 3,
            'momentum_rsi_pivot': 50,
            'momentum_rsi_trigger': 55,
            'momentum_rsi_trigger_short': 45,
            'momentum_high_lookback': 2,
            'momentum_low_lookback': 2,
            'momentum_volume_window': 10,
            'momentum_volume_multiplier': 1.2,
            'momentum_kc_upper_factor': 0.995,
            'momentum_kc_lower_factor': 1.005,
            'momentum_adx_min': 25,

            # Secondary Parameters
            'secondary_close_lookback': 2,
            'secondary_rsi_min': 35,
            'secondary_rsi_max': 70,
            'secondary_rsi_min_short': 30,
            'secondary_rsi_max_short': 80,

            # Tertiary Parameters
            'tertiary_rsi_min': 40,
            'tertiary_rsi_max': 65,
            'tertiary_rsi_min_short': 35,
            'tertiary_rsi_max_short': 75,
            'tertiary_volume_window': 10,
            'tertiary_volume_multiplier': 1.0,

            # Quaternary Parameters
            'quaternary_rsi_min': 45,
            'quaternary_rsi_max': 55,
            'quaternary_volume_window': 5,
            'quaternary_volume_multiplier': 1.2,

            # Fifth Parameters
            'fifth_rsi_min': 35,
            'fifth_rsi_max': 70,
            'fifth_rsi_min_short': 30,
            'fifth_rsi_max_short': 65,

            # Sixth Parameters
            'sixth_min_div_count': 2,
            'sixth_rsi_min': 30,
            'sixth_rsi_max': 75,
            'sixth_rsi_min_short': 25,
            'sixth_rsi_max_short': 70,

            # Seventh Parameters
            'seventh_volume_window': 20,
            'seventh_volume_multiplier': 1.2,
            'seventh_rsi_min': 30,
            'seventh_rsi_max': 70,
            'seventh_rsi_min_short': 30,
            'seventh_rsi_max_short': 70,
            'seventh_close_lookback': 2,
            'seventh_atr_window': 20,
            'seventh_atr_multiplier': 0.5,
            'seventh_adx_min': 15,

            # Divergence Parameters
            'div_rsi_min': 25,
            'div_rsi_max': 80,
            'div_rsi_min_short': 20,
            'div_rsi_max_short': 75,
            #SUPPORT
            'support_tolerance': 0.005,           # 0.5% tolerance for support touch
            'support_bounce_min': 0.002,          # 0.2% minimum bounce
            'resistance_tolerance': 0.005,        # 0.5% tolerance for resistance touch  
            'resistance_bounce_min': 0.002,       # 0.2% minimum rejection
            'sr_volume_window': 20,               # Volume average window
            'sr_volume_multiplier': 1.3,          # Volume surge multiplier
            'sr_rsi_min_long': 25,               # RSI minimum for long S/R
            'sr_rsi_max_long': 70,               # RSI maximum for long S/R
            'sr_rsi_min_short': 30,              # RSI minimum for short S/R
            'sr_rsi_max_short': 75,              # RSI maximum for short S/R
            'sr_price_position_max_long': 0.6,    # Max price position for long
            'sr_price_position_min_short': 0.4,   # Min price position for short
            'sr_bb_max_long': 0.7,               # Max BB% for long entries
            'sr_bb_min_short': 0.3,              # Min BB% for short entries
            'sr_momentum_lookback': 2,            # Momentum confirmation lookback
            'sr_adx_min': 15,                    # Minimum ADX for S/R

            # Strong S/R Parameters
            'bb_support_tolerance': 0.01,         # BB support tolerance
            'bb_resistance_tolerance': 0.01,      # BB resistance tolerance
            'kc_support_tolerance': 0.01,         # KC support tolerance
            'kc_resistance_tolerance': 0.01,      # KC resistance tolerance
            'ema_support_tolerance': 0.008,       # EMA support tolerance
            'ema_resistance_tolerance': 0.008,    # EMA resistance tolerance
            'strong_sr_volume_multiplier': 1.8,   # Strong S/R volume multiplier
            'strong_sr_rsi_min_long': 20,        # Strong S/R RSI min long
            'strong_sr_rsi_max_long': 50,        # Strong S/R RSI max long
            'strong_sr_rsi_min_short': 50,       # Strong S/R RSI min short
            'strong_sr_rsi_max_short': 80,       # Strong S/R RSI max short
            'min_distance_to_resistance': 0.02,   # Minimum distance to resistance
            'min_distance_to_support': 0.02,     # Minimum distance to support
            'strong_sr_adx_min': 18,             # Minimum ADX for strong S/R
            'breakdown_volume_threshold': 1.5,
            'breakdown_adx_threshold': 25,
            'breakdown_rsi_upper': 60,
            'breakdown_rsi_lower': 40,
            'breakdown_price_deviation': 1.0,
            'breakdown_volume_confirmation': 1.2,
            # Add to get_default_params():
            'breakdown_volume_multiplier': 1.3,
            'breakout_volume_multiplier': 1.3,
            'breakdown_rsi_min': 35,
            'breakdown_rsi_max': 60,
            'breakout_rsi_min': 40,
            'breakout_rsi_max': 75,
            # Signal Locking Parameters
            'signal_lock_enabled': True,           # Enable/disable signal locking
            'signal_performance_window': 15,       # Trades to evaluate over  
            'signal_lock_duration_hours': 12,      # Hours to lock signal
            'signal_min_win_rate': 0.35,          # Minimum win rate (35%)
            'signal_min_avg_profit': -0.015,      # Minimum avg profit (-1.5%)
            'signal_min_trades_for_eval': 3,      # Minimum trades before evaluation
            'ema_support_tolerance': 0.008,       # 0.8% tolerance for EMA support touch
            'ema_resistance_tolerance': 0.008,    # 0.8% tolerance for EMA resistance touch
            'bb_support_tolerance': 0.01,         # 1.0% tolerance for BB support touch
            'bb_resistance_tolerance': 0.01,      # 1.0% tolerance for BB resistance touch
            'kc_support_tolerance': 0.01,         # 1.0% tolerance for KC support touch
            'kc_resistance_tolerance': 0.01,      # 1.0% tolerance for KC resistance touch
        }

    def create_objective_function(self, pair: str):
        """Create enhanced objective function with exit parameter optimization"""
        def objective(trial):
            params = {
            # === EXISTING ENTRY PARAMETERS ===
            'min_divergence_count': trial.suggest_int('min_divergence_count', 1, 3),
            'min_signal_strength': trial.suggest_int('min_signal_strength', 1, 5),
            'volume_threshold': trial.suggest_float('volume_threshold', 1.0, 1.5),
            'adx_threshold': trial.suggest_int('adx_threshold', 15, 30),
            'rsi_overbought': trial.suggest_float('rsi_overbought', 65.0, 85.0),
            'rsi_oversold': trial.suggest_float('rsi_oversold', 15.0, 35.0),
            'max_volatility': trial.suggest_float('max_volatility', 0.015, 0.035),
            'min_volatility': trial.suggest_float('min_volatility', 0.003, 0.008),
            'volume_factor': trial.suggest_float('volume_factor', 1.2, 2.0),
            'atr_multiplier': trial.suggest_float('atr_multiplier', 2.0, 4.0),
            
            # === MISSING INDICATOR PARAMETERS ===
            'volume_sma_period': trial.suggest_int('volume_sma_period', 15, 25),
            'atr_period': trial.suggest_int('atr_period', 10, 20),
            'ema_short_period': trial.suggest_int('ema_short_period', 6, 12),
            'ema_long_period': trial.suggest_int('ema_long_period', 18, 25),
            'macd_fast': trial.suggest_int('macd_fast', 8, 16),
            'macd_slow': trial.suggest_int('macd_slow', 22, 30),
            'macd_signal': trial.suggest_int('macd_signal', 6, 12),
            'stoch_k': trial.suggest_int('stoch_k', 10, 18),
            'stoch_d': trial.suggest_int('stoch_d', 2, 5),
            'mom_period': trial.suggest_int('mom_period', 8, 15),
            'pivot_window': trial.suggest_int('pivot_window', 8, 15),
            'swing_period': trial.suggest_int('swing_period', 40, 60),
            'recent_high_low_period': trial.suggest_int('recent_high_low_period', 3, 8),
            'rolling_20_period': trial.suggest_int('rolling_20_period', 15, 25),
            'rolling_10_period': trial.suggest_int('rolling_10_period', 8, 15),
            'chop_period': trial.suggest_int('chop_period', 10, 18),
            'natr_period': trial.suggest_int('natr_period', 10, 18),
            'cmf_period': trial.suggest_int('cmf_period', 15, 25),
            'rsi_period': trial.suggest_int('rsi_period', 10, 20),
            'willr_period': trial.suggest_int('willr_period', 10, 20), 
            'cci_period': trial.suggest_int('cci_period', 15, 25),
            'bb_period': trial.suggest_int('bb_period', 15, 25),
            'bb_std': trial.suggest_float('bb_std', 1.8, 2.5),
            
            # === EXISTING ENTRY TIMING PARAMETERS ===
            'rsi_recovery_periods': trial.suggest_int('rsi_recovery_periods', 2, 5),
            'rsi_long_upper_tight': trial.suggest_float('rsi_long_upper_tight', 60.0, 70.0),
            'rsi_short_lower_tight': trial.suggest_float('rsi_short_lower_tight', 30.0, 40.0),
            'volume_ratio_min': trial.suggest_float('volume_ratio_min', 1.0, 1.4),
            'volume_trend_multiplier': trial.suggest_float('volume_trend_multiplier', 1.2, 2.0),
            'volume_momentum_multiplier': trial.suggest_float('volume_momentum_multiplier', 1.1, 1.5),
            'bb_percent_long_max': trial.suggest_float('bb_percent_long_max', 0.15, 0.35),
            'bb_percent_short_min': trial.suggest_float('bb_percent_short_min', 0.65, 0.85),
            'price_position_long_max': trial.suggest_float('price_position_long_max', 0.2, 0.4),
            'price_position_short_min': trial.suggest_float('price_position_short_min', 0.6, 0.8),
            'divergence_freshness_periods': trial.suggest_int('divergence_freshness_periods', 3, 8),
            'min_strong_divergence_count': trial.suggest_int('min_strong_divergence_count', 1, 3),
            'min_oversold_conditions': trial.suggest_int('min_oversold_conditions', 2, 4),
            'min_overbought_conditions': trial.suggest_int('min_overbought_conditions', 2, 4),
            'min_reversal_signals': trial.suggest_int('min_reversal_signals', 1, 3),
            'min_trend_filters': trial.suggest_int('min_trend_filters', 1, 2),
            'chop_threshold': trial.suggest_float('chop_threshold', 55.0, 65.0),
            'adx_trending_min': trial.suggest_int('adx_trending_min', 20, 30),
            'volume_breakout_multiplier': trial.suggest_float('volume_breakout_multiplier', 2.0, 4.0),
            'volume_breakout_rsi_min': trial.suggest_float('volume_breakout_rsi_min', 35.0, 55.0),
            'volume_breakout_rsi_max': trial.suggest_float('volume_breakout_rsi_max', 45.0, 65.0),
            'bb_oversold_threshold': trial.suggest_float('bb_oversold_threshold', 0.05, 0.25),
            'bb_overbought_threshold': trial.suggest_float('bb_overbought_threshold', 0.75, 0.95),
            'mean_reversion_rsi_oversold': trial.suggest_float('mean_reversion_rsi_oversold', 20.0, 35.0),
            'mean_reversion_rsi_overbought': trial.suggest_float('mean_reversion_rsi_overbought', 65.0, 80.0),
            'momentum_ema_periods': trial.suggest_int('momentum_ema_periods', 8, 21),
            'momentum_strength_min': trial.suggest_float('momentum_strength_min', 0.015, 0.035),
            'momentum_pullback_max': trial.suggest_float('momentum_pullback_max', 0.25, 0.45),
            'momentum_pullback_min': trial.suggest_float('momentum_pullback_min', 0.55, 0.75),
            
            # === NEW EXIT PARAMETER OPTIMIZATION ===
            
            # Global Exit Settings
            'session_multiplier_overlap': trial.suggest_float('session_multiplier_overlap', 1.1, 1.5),
            'session_multiplier_major': trial.suggest_float('session_multiplier_major', 1.0, 1.3),
            'volatility_sensitivity': trial.suggest_int('volatility_sensitivity', 25, 60),
            'high_signal_multiplier': trial.suggest_float('high_signal_multiplier', 1.0, 1.4),
            'low_signal_multiplier': trial.suggest_float('low_signal_multiplier', 0.6, 1.0),
            
            # Mean Reversion (MR1) Exit Optimization
            'mr1_max_hold_minutes': trial.suggest_int('mr1_max_hold_minutes', 15, 25),     # REDUCED RANGE
            'mr1_rsi_exit_long': trial.suggest_float('mr1_rsi_exit_long', 58.0, 65.0),      # REDUCED RANGE  
            'mr1_rsi_exit_short': trial.suggest_float('mr1_rsi_exit_short', 35.0, 42.0),    # INCREASED RANGE
            'mr1_quick_profit_target': trial.suggest_float('mr1_quick_profit_target', 0.01, 0.02), # REDUCED RANGE
            'mr1_timeout_min_profit': trial.suggest_float('mr1_timeout_min_profit', 0.003, 0.008), # REDUCED RANGE
            'mr1_bb_exit_long': trial.suggest_float('mr1_bb_exit_long', 0.6, 0.8),
            'mr1_bb_exit_short': trial.suggest_float('mr1_bb_exit_short', 0.2, 0.4),
            
            # Momentum Continuation (MC1) Exit Optimization
            'mc1_profit_target': trial.suggest_float('mc1_profit_target', 0.025, 0.055),
            'mc1_momentum_break_threshold': trial.suggest_float('mc1_momentum_break_threshold', 0.01, 0.025),
            'mc1_max_hold_minutes': trial.suggest_int('mc1_max_hold_minutes', 30, 60),
            'mc1_adx_exit_threshold': trial.suggest_int('mc1_adx_exit_threshold', 18, 28),
            'mc1_timeout_min_profit': trial.suggest_float('mc1_timeout_min_profit', 0.005, 0.02),
            
            # Volume Breakout (VB1) Exit Optimization
            'vb1_volume_fade_threshold': trial.suggest_float('vb1_volume_fade_threshold', 1.0, 1.3),
            'vb1_profit_target': trial.suggest_float('vb1_profit_target', 0.02, 0.045),
            'vb1_quick_profit': trial.suggest_float('vb1_quick_profit', 0.012, 0.03),
            'vb1_min_volatility': trial.suggest_float('vb1_min_volatility', 0.008, 0.018),
            
            # Reversal (RSV1) Exit Optimization
            'rsv1_rsi_recovery_threshold': trial.suggest_float('rsv1_rsi_recovery_threshold', 52.0, 65.0),
            'rsv1_profit_target': trial.suggest_float('rsv1_profit_target', 0.025, 0.05),
            'rsv1_max_hold_minutes': trial.suggest_int('rsv1_max_hold_minutes', 40, 80),
            'rsv1_min_timeout_profit': trial.suggest_float('rsv1_min_timeout_profit', 0.005, 0.02),
            
            # Trend Following Exit Optimization
            'trend_profit_target': trial.suggest_float('trend_profit_target', 0.03, 0.06),
            'trend_ema_break_periods': trial.suggest_int('trend_ema_break_periods', 2, 5),
            
            # Universal Exit Optimization
            'medium_term_base_target': trial.suggest_float('medium_term_base_target', 0.035, 0.065),
            'reversal_min_profit_long': trial.suggest_float('reversal_min_profit_long', 0.005, 0.015),
            'reversal_min_profit_short': trial.suggest_float('reversal_min_profit_short', 0.005, 0.015),
            'rsi_exit_overbought': trial.suggest_float('rsi_exit_overbought', 72, 85),
            'rsi_exit_oversold': trial.suggest_float('rsi_exit_oversold', 15, 28),
            'rsi_extreme_min_profit': trial.suggest_float('rsi_extreme_min_profit', 0.01, 0.025),
            'momentum_fade_min_profit': trial.suggest_float('momentum_fade_min_profit', 0.008, 0.02),
            'momentum_fade_threshold': trial.suggest_float('momentum_fade_threshold', 0.015, 0.03),
            
            # Extended Duration Management
            'extended_2hr_target': trial.suggest_float('extended_2hr_target', 0.015, 0.035),
            'extended_3hr_target': trial.suggest_float('extended_3hr_target', 0.008, 0.025),
            'extended_5hr_target': trial.suggest_float('extended_5hr_target', 0.005, 0.015),
            
            # Market Protection Optimization
            'friday_close_min_profit': trial.suggest_float('friday_close_min_profit', 0.008, 0.02),
            'overnight_min_profit': trial.suggest_float('overnight_min_profit', 0.005, 0.015),
            'low_liquidity_min_profit': trial.suggest_float('low_liquidity_min_profit', 0.006, 0.018),
            
            # === RSV1 REVERSAL OPTIMIZATION ===
            # RSI thresholds
            'rsv1_rsi_oversold': trial.suggest_float('rsv1_rsi_oversold', 20.0, 35.0),
            'rsv1_rsi_overbought': trial.suggest_float('rsv1_rsi_overbought', 65.0, 80.0),
            
            # Williams %R thresholds
            'rsv1_willr_oversold': trial.suggest_float('rsv1_willr_oversold', -90.0, -70.0),
            'rsv1_willr_overbought': trial.suggest_float('rsv1_willr_overbought', -30.0, -10.0),
            
            # CCI thresholds
            'rsv1_cci_oversold': trial.suggest_float('rsv1_cci_oversold', -120.0, -80.0),
            'rsv1_cci_overbought': trial.suggest_float('rsv1_cci_overbought', 80.0, 120.0),
            
            # Bollinger Band position
            'rsv1_bb_oversold': trial.suggest_float('rsv1_bb_oversold', 0.05, 0.15),
            'rsv1_bb_overbought': trial.suggest_float('rsv1_bb_overbought', 0.85, 0.95),
            
            # Volume and position filters
            'rsv1_volume_factor': trial.suggest_float('rsv1_volume_factor', 1.2, 2.0),
            'rsv1_price_position_low': trial.suggest_float('rsv1_price_position_low', 0.1, 0.3),
            'rsv1_price_position_high': trial.suggest_float('rsv1_price_position_high', 0.7, 0.9),
            
            # Condition count requirements
            'rsv1_min_oversold_conditions': trial.suggest_int('rsv1_min_oversold_conditions', 2, 4),
            'rsv1_min_overbought_conditions': trial.suggest_int('rsv1_min_overbought_conditions', 2, 4),
            'rsv1_min_reversal_signals': trial.suggest_int('rsv1_min_reversal_signals', 1, 3),
            'rsv1_min_trend_filters': trial.suggest_int('rsv1_min_trend_filters', 1, 2),

            # === E1 PARAMETERS ===
            'e1_rsi_overbought_threshold': trial.suggest_float('e1_rsi_overbought_threshold', 70.0, 80.0),
            'e1_rsi_overbought_buffer': trial.suggest_int('e1_rsi_overbought_buffer', 3, 8),
            'e1_rsi_oversold_min': trial.suggest_float('e1_rsi_oversold_min', 20.0, 30.0),
            'e1_rsi_oversold_threshold': trial.suggest_float('e1_rsi_oversold_threshold', 20.0, 30.0),
            'e1_rsi_oversold_buffer': trial.suggest_int('e1_rsi_oversold_buffer', 3, 8),
            'e1_rsi_overbought_max': trial.suggest_float('e1_rsi_overbought_max', 70.0, 80.0),
            'volatility_min_multiplier': trial.suggest_float('volatility_min_multiplier', 0.3, 1.0),
            'volatility_max_multiplier': trial.suggest_float('volatility_max_multiplier', 1.5, 3.0),

            # === TREND PARAMETERS ===
            'trend_volume_window': trial.suggest_int('trend_volume_window', 15, 30),
            'trend_volume_multiplier': trial.suggest_float('trend_volume_multiplier', 1.2, 2.0),
            'trend_rsi_min': trial.suggest_float('trend_rsi_min', 30.0, 40.0),
            'trend_rsi_max': trial.suggest_float('trend_rsi_max', 65.0, 70.0),
            'trend_close_lookback': trial.suggest_int('trend_close_lookback', 1, 4),
            'trend_atr_window': trial.suggest_int('trend_atr_window', 15, 25),
            'trend_atr_multiplier': trial.suggest_float('trend_atr_multiplier', 0.6, 1.0),
            'trend_adx_min': trial.suggest_int('trend_adx_min', 15, 25),

            # === MOMENTUM PARAMETERS ===
            'momentum_rsi_lookback': trial.suggest_int('momentum_rsi_lookback', 2, 5),
            'momentum_rsi_pivot': trial.suggest_float('momentum_rsi_pivot', 45.0, 55.0),
            'momentum_rsi_trigger': trial.suggest_float('momentum_rsi_trigger', 50.0, 60.0),
            'momentum_rsi_trigger_short': trial.suggest_float('momentum_rsi_trigger_short', 40.0, 50.0),
            'momentum_high_lookback': trial.suggest_int('momentum_high_lookback', 1, 4),
            'momentum_low_lookback': trial.suggest_int('momentum_low_lookback', 1, 4),
            'momentum_volume_window': trial.suggest_int('momentum_volume_window', 8, 15),
            'momentum_volume_multiplier': trial.suggest_float('momentum_volume_multiplier', 1.0, 1.5),
            'momentum_kc_upper_factor': trial.suggest_float('momentum_kc_upper_factor', 0.99, 1.0),
            'momentum_kc_lower_factor': trial.suggest_float('momentum_kc_lower_factor', 1.0, 1.01),
            'momentum_adx_min': trial.suggest_int('momentum_adx_min', 20, 30),

            # === SECONDARY PARAMETERS ===
            'secondary_close_lookback': trial.suggest_int('secondary_close_lookback', 1, 4),
            'secondary_rsi_min': trial.suggest_float('secondary_rsi_min', 30.0, 40.0),
            'secondary_rsi_max': trial.suggest_float('secondary_rsi_max', 65.0, 75.0),
            'secondary_rsi_min_short': trial.suggest_float('secondary_rsi_min_short', 25.0, 35.0),
            'secondary_rsi_max_short': trial.suggest_float('secondary_rsi_max_short', 75.0, 85.0),

            # === TERTIARY PARAMETERS ===
            'tertiary_rsi_min': trial.suggest_float('tertiary_rsi_min', 35.0, 45.0),
            'tertiary_rsi_max': trial.suggest_float('tertiary_rsi_max', 60.0, 70.0),
            'tertiary_rsi_min_short': trial.suggest_float('tertiary_rsi_min_short', 30.0, 40.0),
            'tertiary_rsi_max_short': trial.suggest_float('tertiary_rsi_max_short', 70.0, 80.0),
            'tertiary_volume_window': trial.suggest_int('tertiary_volume_window', 8, 15),
            'tertiary_volume_multiplier': trial.suggest_float('tertiary_volume_multiplier', 0.8, 1.3),

            # === QUATERNARY PARAMETERS ===
            'quaternary_rsi_min': trial.suggest_float('quaternary_rsi_min', 40.0, 50.0),
            'quaternary_rsi_max': trial.suggest_float('quaternary_rsi_max', 50.0, 60.0),
            'quaternary_volume_window': trial.suggest_int('quaternary_volume_window', 3, 8),
            'quaternary_volume_multiplier': trial.suggest_float('quaternary_volume_multiplier', 1.0, 1.5),

            # === FIFTH PARAMETERS ===
            'fifth_rsi_min': trial.suggest_float('fifth_rsi_min', 30.0, 40.0),
            'fifth_rsi_max': trial.suggest_float('fifth_rsi_max', 65.0, 75.0),
            'fifth_rsi_min_short': trial.suggest_float('fifth_rsi_min_short', 25.0, 35.0),
            'fifth_rsi_max_short': trial.suggest_float('fifth_rsi_max_short', 60.0, 70.0),

            # === SIXTH PARAMETERS ===
            'sixth_min_div_count': trial.suggest_int('sixth_min_div_count', 1, 3),
            'sixth_rsi_min': trial.suggest_float('sixth_rsi_min', 25.0, 35.0),
            'sixth_rsi_max': trial.suggest_float('sixth_rsi_max', 70.0, 80.0),
            'sixth_rsi_min_short': trial.suggest_float('sixth_rsi_min_short', 20.0, 30.0),
            'sixth_rsi_max_short': trial.suggest_float('sixth_rsi_max_short', 65.0, 75.0),

            # === SEVENTH PARAMETERS ===
            'seventh_volume_window': trial.suggest_int('seventh_volume_window', 15, 25),
            'seventh_volume_multiplier': trial.suggest_float('seventh_volume_multiplier', 1.0, 1.5),
            'seventh_rsi_min': trial.suggest_float('seventh_rsi_min', 25.0, 35.0),
            'seventh_rsi_max': trial.suggest_float('seventh_rsi_max', 65.0, 75.0),
            'seventh_rsi_min_short': trial.suggest_float('seventh_rsi_min_short', 25.0, 35.0),
            'seventh_rsi_max_short': trial.suggest_float('seventh_rsi_max_short', 65.0, 75.0),
            'seventh_close_lookback': trial.suggest_int('seventh_close_lookback', 1, 4),
            'seventh_atr_window': trial.suggest_int('seventh_atr_window', 15, 25),
            'seventh_atr_multiplier': trial.suggest_float('seventh_atr_multiplier', 0.3, 0.8),
            'seventh_adx_min': trial.suggest_int('seventh_adx_min', 10, 20),

            # === DIVERGENCE PARAMETERS ===
            'div_rsi_min': trial.suggest_float('div_rsi_min', 20.0, 30.0),
            'div_rsi_max': trial.suggest_float('div_rsi_max', 75.0, 85.0),
            'div_rsi_min_short': trial.suggest_float('div_rsi_min_short', 15.0, 25.0),
            'div_rsi_max_short': trial.suggest_float('div_rsi_max_short', 70.0, 80.0),
            
            # === SUPPORT/RESISTANCE PARAMETERS ===
            'support_tolerance': trial.suggest_float('support_tolerance', 0.003, 0.008),
            'support_bounce_min': trial.suggest_float('support_bounce_min', 0.001, 0.004),
            'resistance_tolerance': trial.suggest_float('resistance_tolerance', 0.003, 0.008),
            'resistance_bounce_min': trial.suggest_float('resistance_bounce_min', 0.001, 0.004),
            'sr_volume_multiplier': trial.suggest_float('sr_volume_multiplier', 1.1, 1.8),
            'sr_rsi_min_long': trial.suggest_float('sr_rsi_min_long', 20.0, 35.0),
            'sr_rsi_max_long': trial.suggest_float('sr_rsi_max_long', 60.0, 75.0),
            'sr_rsi_min_short': trial.suggest_float('sr_rsi_min_short', 25.0, 40.0),
            'sr_rsi_max_short': trial.suggest_float('sr_rsi_max_short', 65.0, 80.0),
            'strong_sr_volume_multiplier': trial.suggest_float('strong_sr_volume_multiplier', 1.5, 2.5),
            
            # === BREAKDOWN/BREAKOUT PARAMETERS ===
            'breakdown_volume_threshold': trial.suggest_float('breakdown_volume_threshold', 1.1, 3.0),
            'breakdown_adx_threshold': trial.suggest_int('breakdown_adx_threshold', 15, 45),
            'breakdown_rsi_upper': trial.suggest_int('breakdown_rsi_upper', 55, 70),
            'breakdown_rsi_lower': trial.suggest_int('breakdown_rsi_lower', 30, 45),
            'breakdown_price_deviation': trial.suggest_float('breakdown_price_deviation', 0.5, 3.0),
            'breakdown_volume_confirmation': trial.suggest_float('breakdown_volume_confirmation', 1.0, 2.5),
            'breakdown_volume_multiplier': trial.suggest_float('breakdown_volume_multiplier', 1.1, 2.0),
            'breakout_volume_multiplier': trial.suggest_float('breakout_volume_multiplier', 1.1, 2.0),
            'breakdown_rsi_min': trial.suggest_float('breakdown_rsi_min', 30.0, 45.0),
            'breakdown_rsi_max': trial.suggest_float('breakdown_rsi_max', 55.0, 70.0),
            'breakout_rsi_min': trial.suggest_float('breakout_rsi_min', 35.0, 50.0),
            'breakout_rsi_max': trial.suggest_float('breakout_rsi_max', 65.0, 80.0),
            
            # === ADDITIONAL TECHNICAL PARAMETERS ===
            'willr_oversold': trial.suggest_float('willr_oversold', -90, -70),
            'willr_overbought': trial.suggest_float('willr_overbought', -30, -10),
            'cci_oversold': trial.suggest_float('cci_oversold', -120, -80),
            'cci_overbought': trial.suggest_float('cci_overbought', 80, 120),
            'ema_support_tolerance': trial.suggest_float('ema_support_tolerance', 0.005, 0.015),
            'ema_resistance_tolerance': trial.suggest_float('ema_resistance_tolerance', 0.005, 0.015),
            'bb_support_tolerance': trial.suggest_float('bb_support_tolerance', 0.008, 0.020),
            'bb_resistance_tolerance': trial.suggest_float('bb_resistance_tolerance', 0.008, 0.020),
            'kc_support_tolerance': trial.suggest_float('kc_support_tolerance', 0.008, 0.020),
            'kc_resistance_tolerance': trial.suggest_float('kc_resistance_tolerance', 0.008, 0.020),
            }
            
            # Return performance for this coin
            recent_trades = self.get_recent_trade_performance(pair)
            if len(recent_trades) >= 1:  # Need at least 3 trades
                return sum(recent_trades) / len(recent_trades)
            else:
                return 0.0  # No optimization until we have trade data
        
        return objective

    def get_recent_trade_performance(self, pair: str) -> List[float]:
        """Get recent trade performance for optimization"""
        try:
            from freqtrade.persistence import Trade
            trades = Trade.get_trades_proxy(pair=pair)
            
            if not trades:
                return []
            
            # Get last 5 trades for this pair
            recent_trades = trades[-5:] if len(trades) >= 5 else trades
            
            # Calculate profit ratios
            performance = []
            for trade in recent_trades:
                if trade.close_date:  # Only closed trades
                    # FIX: Use the close_rate for calc_profit_ratio
                    performance.append(trade.calc_profit_ratio(trade.close_rate))
            
            logger.debug(f"📊 [OPTUNA] Found {len(performance)} completed trades for {pair}")
            return performance
            
        except Exception as e:
            logger.error(f"❌ [OPTUNA] Failed to get trade performance for {pair}: {e}")
            return []
    def daily_optimization_check(self):
        """Enhanced optimization check with smarter scheduling"""
        try:
            if not self.optuna_manager:
                return
            
            current_time = time.time()
            all_pairs = list(self.coin_params.keys())
            
            if not all_pairs:
                logger.info("No pairs available for optimization yet")
                return
            
            optimized_count = 0
            
            for pair in all_pairs:
                try:
                    if self.should_retrain_pair(pair, current_time):
                        logger.info(f"🔄 Optimizing {pair}")
                        self.optuna_manager.optimize_coin(pair, self.create_objective_function(pair), n_trials=30)
                        
                        # Track last optimization time
                        setattr(self, f'last_optimization_{pair.replace("/", "_").replace(":", "_")}', current_time)
                        optimized_count += 1
                        
                except Exception as e:
                    logger.error(f"❌ Optimization failed for {pair}: {e}")
            
            logger.info(f"✅ Optimization check completed: {optimized_count}/{len(all_pairs)} pairs optimized")
            
        except Exception as e:
            logger.error(f"❌ Daily optimization check failed: {e}")

    def should_retrain_pair(self, pair: str, current_time: float) -> bool:
        """Simplified retraining logic"""
        last_opt_time = getattr(self, f'last_optimization_{pair.replace("/", "_").replace(":", "_")}', 0)
        trades_since_last = self.get_trades_since_optimization(pair, last_opt_time)
        
        # Simplified: Just check if enough time passed OR enough trades
        time_threshold = 21600  # 6 hours (reduced from complex scheduling)
        trade_threshold = 2     # Only 2 trades needed (reduced from 5-10)
        
        return (current_time - last_opt_time > time_threshold) or (trades_since_last >= trade_threshold)

    def get_trades_since_optimization(self, pair: str, last_opt_time: float) -> int:
        """Count trades since last optimization"""
        try:
            from freqtrade.persistence import Trade
            from datetime import datetime
            
            last_opt_datetime = datetime.fromtimestamp(last_opt_time) if last_opt_time > 0 else datetime.min
            trades = Trade.get_trades_proxy(pair=pair)
            
            if not trades:
                return 0
            
            recent_trades = [t for t in trades if t.open_date_utc > last_opt_datetime]
            return len(recent_trades)
            
        except Exception as e:
            logger.debug(f"Failed to count recent trades for {pair}: {e}")
            return 0
    def load_signal_performance(self):
        """Load signal performance history from CSV file"""
        if os.path.exists(self.signal_performance_file):
            try:
                all_trades = []
                
                with open(self.signal_performance_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f, delimiter=';')
                    
                    for row in reader:
                        # Convert profit to float and win to boolean
                        trade_record = {
                            'pair': row['pair'],
                            'entry_tag': row['entry_tag'],
                            'profit': float(row['profit']),
                            'timestamp': row['timestamp'],
                            'win': row['win'].lower() == 'true'
                        }
                        all_trades.append(trade_record)
                
                # Store loaded trades
                self.all_signal_trades = all_trades
                
                # Rebuild the nested format for signal locking
                signal_perf = {}
                for trade in all_trades:
                    perf_key = f"{trade['pair']}_{trade['entry_tag']}"
                    if perf_key not in signal_perf:
                        signal_perf[perf_key] = []
                    
                    signal_perf[perf_key].append({
                        'profit': trade['profit'],
                        'timestamp': trade['timestamp'],
                        'win': trade['win']
                    })
                
                print(f"✅ Loaded {len(all_trades)} trades from CSV")
                return signal_perf
                
            except Exception as e:
                print(f"Failed to load CSV: {e}")
                return {}
        
        return {}
    def save_signal_performance(self):
        """Save signal performance to CSV file - Excel friendly"""
        try:
            os.makedirs(os.path.dirname(self.signal_performance_file), exist_ok=True)
            
            # Get all trades
            all_trades = getattr(self, 'all_signal_trades', [])
            
            if not all_trades:
                print("No trades to save")
                return
            
            # Write to CSV
            with open(self.signal_performance_file, 'w', newline='', encoding='utf-8') as f:
                # Define headers with semicolon separator
                fieldnames = ['pair', 'entry_tag', 'profit', 'timestamp', 'win']
                writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=';')
                
                # Write header
                writer.writeheader()
                
                # Write all trades
                for trade in all_trades:
                    writer.writerow(trade)
                    
            print(f"✅ Saved {len(all_trades)} trades to CSV")
            
        except Exception as e:
            print(f"Failed to save signal performance: {e}")
    def is_signal_locked(self, entry_tag, pair):
        """Check if a specific signal is locked for this pair"""
        lock_key = f"{pair}_{entry_tag}"
        
        if lock_key in self.signal_locks:
            if self.signal_locks[lock_key] > datetime.now():
                return True
            else:
                # Remove expired lock
                del self.signal_locks[lock_key]
                print(f"Signal lock expired for {entry_tag} on {pair}")
        
        return False
    
    def update_signal_performance(self, entry_tag, pair, profit_ratio):
        """Update performance tracking for a signal - Excel-friendly format + locking support"""
        
        # Create trade record with all info in one line (Excel-friendly)
        trade_record = {
            'pair': pair,
            'entry_tag': entry_tag, 
            'profit': profit_ratio,
            'timestamp': datetime.now().isoformat(),
            'win': profit_ratio > 0
        }
        
        # Store in flat list for Excel analysis
        if not hasattr(self, 'all_signal_trades'):
            self.all_signal_trades = []
        
        self.all_signal_trades.append(trade_record)
        
        # Keep only recent window
        if len(self.all_signal_trades) > 500:
            self.all_signal_trades = self.all_signal_trades[-500:]
        
        # Keep the nested format for the locking logic to work
        perf_key = f"{pair}_{entry_tag}"
        if perf_key not in self.signal_performance:
            self.signal_performance[perf_key] = []
        
        self.signal_performance[perf_key].append({
            'profit': profit_ratio,
            'timestamp': datetime.now().isoformat(),
            'win': profit_ratio > 0
        })
        
        if len(self.signal_performance[perf_key]) > self.performance_window:
            self.signal_performance[perf_key] = self.signal_performance[perf_key][-self.performance_window:]
        
        # Evaluate if signal should be locked
        self.evaluate_signal_lock(entry_tag, pair)
        
        # Save to file
        self.save_signal_performance()
    
    def evaluate_signal_lock(self, entry_tag, pair):
        """Evaluate if signal should be locked based on recent performance"""
        perf_key = f"{pair}_{entry_tag}"
        recent_trades = self.signal_performance.get(perf_key, [])
        
        # Need minimum trades for evaluation
        if len(recent_trades) < self.min_trades_for_eval:
            return
        
        # Calculate performance metrics
        profits = [trade['profit'] for trade in recent_trades]
        wins = sum(1 for profit in profits if profit > 0)
        win_rate = wins / len(profits)
        avg_profit = sum(profits) / len(profits)
        
        # Check if signal should be locked
        should_lock = (
            win_rate < self.min_win_rate or 
            avg_profit < self.min_avg_profit
        )
        
        if should_lock:
            lock_until = datetime.now() + timedelta(hours=self.lock_duration_hours)
            lock_key = f"{pair}_{entry_tag}"
            self.signal_locks[lock_key] = lock_until
            
            print(f"LOCKED SIGNAL: {entry_tag} for {pair}")
            print(f"  Win Rate: {win_rate:.1%} (min: {self.min_win_rate:.1%})")
            print(f"  Avg Profit: {avg_profit:.2%} (min: {self.min_avg_profit:.2%})")
            print(f"  Locked until: {lock_until.strftime('%Y-%m-%d %H:%M')}")
    
    def get_signal_status(self, pair):
        """Get status of all signals for a pair - for debugging"""
        status = {}
        
        # Complete list of entry tags from your trading script
        entry_tags = [
            # Primary conditions
            'Bull_E1', 'Bear_E1',
            
            # Support/Resistance bounce
            'Bull_SR_Bounce', 'Bear_SR_Bounce',
            
            # Strong Support/Resistance
            'Bull_Strong_Support', 'Bear_Strong_Resistance',
            
            # Trend breakout
            'Bull_Trend', 'Bear_Trend',
            
            # Momentum
            'Bull_Momentum', 'Bear_Momentum',
            
            # Momentum breakout
            'Bull_Momentum_Breakout', 'Bear_Momentum_Breakout',
            
            # Pullback momentum
            'Bull_Pullback_Momentum', 'Bear_Pullback_Momentum',
            
            # Secondary conditions
            'Bull_E2', 'Bear_E2',
            
            # Tertiary conditions (MISSING)
            'Bull_E3', 'Bear_E3',
            
            # Quaternary conditions (MISSING)
            'Bull_E4', 'Bear_E4',
            
            # Fifth conditions (MISSING)
            'Bull_E5', 'Bear_E5',
            
            # Sixth conditions (MISSING)
            'Bull_E6', 'Bear_E6',
            
            # Seventh condition
            'Bull_E7', 'Bear_E7',
            
            # Breakdown conditions
            'Bull_Breakdown', 'Bear_Breakdown',
            
            # Pure divergence
            'Bull_Div', 'Bear_Div',
            
            # High quality divergence
            'Bull_Div_HQ', 'Bear_Div_HQ',
            
            # Reversal (RSV1)
            'Bull_RSV1', 'Bear_RSV1',
            
            # Mean reversion (MR1)
            'Bull_MR1', 'Bear_MR1',
            
            # Momentum continuation (MC1)
            'Bull_MC1', 'Bear_MC1',
            
            # Simple breakout/breakdown
            'Bull_Simple_Breakout', 'Bear_Simple_Breakdown',
            
            # Fast Momentum (NEW)
            'Bull_Fast_Momentum', 'Bear_Fast_Momentum'
        ]
        
        for entry_tag in entry_tags:
            perf_key = f"{pair}_{entry_tag}"
            recent_trades = self.signal_performance.get(perf_key, [])
            
            is_locked = self.is_signal_locked(entry_tag, pair)
            
            if recent_trades:
                profits = [trade['profit'] for trade in recent_trades]
                wins = sum(1 for profit in profits if profit > 0)
                win_rate = wins / len(profits) if profits else 0
                avg_profit = sum(profits) / len(profits) if profits else 0
                
                status[entry_tag] = {
                    'locked': is_locked,
                    'trades_count': len(recent_trades),
                    'win_rate': win_rate,
                    'avg_profit': avg_profit,
                    'last_profit': profits[-1] if profits else None
                }
            else:
                status[entry_tag] = {
                    'locked': is_locked,
                    'trades_count': 0,
                    'win_rate': 0,
                    'avg_profit': 0,
                    'last_profit': None
                }
        
        return status

    def informative_pairs(self):
        """Define additional timeframes to download"""
        pairs = self.dp.current_whitelist()
        return [(pair, '1h') for pair in pairs]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Enhanced indicator population with multi-timeframe analysis and Optuna optimization
        ENHANCED: Now includes proactive Optuna optimization with parameterized indicators
        """
        
        pair = metadata['pair']
        
        logger.debug(f"📊 [INDICATORS] Processing {pair} with {len(dataframe)} candles")
        
        # ===== PROACTIVE OPTUNA OPTIMIZATION FIRST =====
        # ENHANCED: Get optimized parameters EARLY (triggers proactive optimization)
        try:
            coin_params = self.get_coin_params(pair)  # Get coin-specific parameters
            logger.debug(f"📈 [OPTUNA] Applied parameters for {pair}: min_div={coin_params.get('min_divergence_count')}, min_signal={coin_params.get('min_signal_strength')}")
        except Exception as e:
            logger.error(f"❌ [OPTUNA] Failed to get coin parameters for {pair}: {e}")
            coin_params = self.get_default_params()
        
        # ===== ENHANCED PARAMETER EXTRACTION =====
        # Extract all coin-specific parameters at once for better performance
        
        # Volume analysis parameters
        volume_sma_period = coin_params.get('volume_sma_period', 20)
        volume_threshold = coin_params.get('volume_threshold', 1.0)
        
        # ATR and volatility parameters  
        atr_period = coin_params.get('atr_period', 14)
        atr_multiplier = coin_params.get('atr_multiplier', 3.0)
        
        # Bollinger Bands parameters
        bb_period = coin_params.get('bb_period', getattr(self, 'bb_period', type('obj', (), {'value': 20})).value)
        bb_std = coin_params.get('bb_std', getattr(self, 'bb_std', type('obj', (), {'value': 2.0})).value)
        
        # EMA parameters
        ema_short_period = coin_params.get('ema_short_period', 8)
        ema_long_period = coin_params.get('ema_long_period', 21)
        
        # MACD parameters
        macd_fast = coin_params.get('macd_fast', 12)
        macd_slow = coin_params.get('macd_slow', 26) 
        macd_signal = coin_params.get('macd_signal', 9)
        
        # RSI and momentum parameters
        rsi_period = coin_params.get('rsi_period', 14)
        willr_period = coin_params.get('willr_period', 14) 
        cci_period = coin_params.get('cci_period', 20)
        mom_period = coin_params.get('mom_period', 10)
        stoch_k = coin_params.get('stoch_k', 14)
        stoch_d = coin_params.get('stoch_d', 3)
        
        # Pivot points window
        pivot_window = coin_params.get('pivot_window', getattr(self, 'window', type('obj', (), {'value': 10})).value)
        
        # Support/Resistance parameters
        swing_period = coin_params.get('swing_period', 50)
        recent_high_low_period = coin_params.get('recent_high_low_period', 5)
        
        # Rolling window parameters
        rolling_20_period = coin_params.get('rolling_20_period', 20)
        rolling_10_period = coin_params.get('rolling_10_period', 10)
        
        # Choppiness and other analysis parameters
        chop_period = coin_params.get('chop_period', 14)
        natr_period = coin_params.get('natr_period', 14)
        cmf_period = coin_params.get('cmf_period', 20)
        
        # Validate parameters to ensure they're within reasonable ranges
        rsi_period = max(2, min(50, rsi_period))
        willr_period = max(2, min(50, willr_period))
        cci_period = max(2, min(50, cci_period))
        
        # === MULTI-TIMEFRAME ANALYSIS ===
        # Get 1h timeframe for trend confirmation with improved error handling
        try:
            # Check if we're in backtesting mode or if pair supports 1h data
            if hasattr(self.dp, 'runmode') and self.dp.runmode.value in ['backtest', 'hyperopt']:
                # In backtesting, try to get 1h data but don't fail if unavailable
                informative_1h = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe='1h')
            else:
                # In live/dry run, be more cautious about data availability
                try:
                    informative_1h = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe='1h')
                except Exception:
                    informative_1h = None
            
            # Enhanced data validation
            if (informative_1h is not None and 
                len(informative_1h) > 50 and  # Reduced minimum requirement
                not informative_1h.empty and
                'close' in informative_1h.columns):
                
                try:
                    # 1h Trend indicators with additional error checking
                    informative_1h['ema50_1h'] = ta.EMA(informative_1h, timeperiod=50)
                    informative_1h['ema200_1h'] = ta.EMA(informative_1h, timeperiod=200)
                    informative_1h['trend_1h'] = ta.EMA(informative_1h, timeperiod=21)
                    informative_1h['trend_strength_1h'] = ta.ADX(informative_1h)
                    informative_1h['rsi_1h'] = ta.RSI(informative_1h)
                    
                    # Fill NaN values before merging
                    informative_1h = informative_1h.bfill().ffill()
                    
                    # Safe merge with additional error handling
                    dataframe = merge_informative_pair(dataframe, informative_1h, self.timeframe, '1h', ffill=True)
                    
                    
                except Exception as merge_error:
                    logger.warning(f"Failed to merge 1h data for {metadata['pair']}: {merge_error}")
                    self._add_dummy_1h_columns(dataframe)
            else:
                logger.info(f"Using fallback 1h indicators for {metadata['pair']} (insufficient data)")
                self._add_dummy_1h_columns(dataframe)
                
        except Exception as e:
            logger.warning(f"Error accessing 1h data for {metadata['pair']}: {e}")
            self._add_dummy_1h_columns(dataframe)
        
        # === 15M TIMEFRAME INDICATORS ===
        informative = dataframe.copy()
        
        # === VOLUME ANALYSIS WITH PARAMETERIZED VALUES ===
        try:
            informative['volume_sma'] = ta.SMA(informative['volume'], timeperiod=volume_sma_period)
            informative['volume_ratio'] = informative['volume'] / informative['volume_sma']
            informative['volume_ratio'] = informative['volume_ratio'].fillna(1.0)
        except:
            informative['volume_sma'] = informative['volume']
            informative['volume_ratio'] = 1.0
        
        # === VOLATILITY ANALYSIS WITH PARAMETERIZED VALUES ===
        try:
            informative['atr'] = ta.ATR(informative, timeperiod=atr_period)
            informative['volatility'] = informative['atr'] / informative['close']
            informative['volatility'] = informative['volatility'].fillna(0.01)
        except:
            informative['atr'] = informative['close'] * 0.02
            informative['volatility'] = 0.01
        
        # === MOMENTUM INDICATORS WITH PARAMETERIZED VALUES ===
        try:
            # Parameterized momentum indicators
            informative['rsi'] = ta.RSI(informative, timeperiod=rsi_period)
            informative['willr'] = ta.WILLR(informative, timeperiod=willr_period)
            informative['cci'] = ta.CCI(informative, timeperiod=cci_period)
            informative['mom'] = ta.MOM(informative, timeperiod=mom_period)
            
            # MACD with parameterized values
            macd = ta.MACD(informative, fastperiod=macd_fast, slowperiod=macd_slow, signalperiod=macd_signal)
            informative['macd'] = macd['macd']
            informative['macdsignal'] = macd['macdsignal']
            informative['macdhist'] = macd['macdhist']
            
            # Additional momentum indicators with parameters
            stoch = ta.STOCH(informative, fastk_period=stoch_k, slowk_period=stoch_d, slowd_period=stoch_d)
            informative['stoch'] = stoch['slowk']
            informative['roc'] = ta.ROC(informative)
            informative['uo'] = ta.ULTOSC(informative)
            informative['ao'] = qtpylib.awesome_oscillator(informative)
            informative['cmf'] = chaikin_money_flow(informative, cmf_period)
            informative['obv'] = ta.OBV(informative)
            informative['mfi'] = ta.MFI(informative)
            informative['adx'] = ta.ADX(informative)
            
            # Fill NaN values for all indicators
            indicator_columns = ['rsi', 'stoch', 'roc', 'uo', 'ao', 'macd', 'macdsignal', 'macdhist', 'cci', 'cmf', 'obv', 'mfi', 'adx', 'willr', 'mom']
            for col in indicator_columns:
                if col in informative.columns:
                    informative[col] = informative[col].bfill().fillna(50 if col in ['rsi', 'mfi'] else 0)
                    
        except Exception as e:
            logger.warning(f"Error calculating momentum indicators: {e}")
            # Provide fallback values
            informative['rsi'] = 50
            informative['stoch'] = 50
            informative['roc'] = 0
            informative['uo'] = 50
            informative['ao'] = 0
            informative['macd'] = 0
            informative['macdsignal'] = 0
            informative['macdhist'] = 0
            informative['cci'] = 0
            informative['cmf'] = 0
            informative['obv'] = informative['volume'].cumsum()
            informative['mfi'] = 50
            informative['adx'] = 25
            informative['willr'] = -50
            informative['mom'] = 0

        # === KELTNER CHANNEL ===
        try:
            keltner = emaKeltner(informative)
            informative["kc_upperband"] = keltner["upper"]
            informative["kc_middleband"] = keltner["mid"]
            informative["kc_lowerband"] = keltner["lower"]
        except:
            informative["kc_upperband"] = informative['close'] * 1.02
            informative["kc_middleband"] = informative['close']
            informative["kc_lowerband"] = informative['close'] * 0.98

        # === BOLLINGER BANDS WITH PARAMETERIZED VALUES ===
        try:
            # Parameterized Bollinger Bands
            bollinger = qtpylib.bollinger_bands(informative['close'], window=bb_period, stds=bb_std)
            informative['bb_lower'] = bollinger['lower']
            informative['bb_middle'] = bollinger['mid']
            informative['bb_upper'] = bollinger['upper']
            informative['bb_percent'] = (informative['close'] - informative['bb_lower']) / (informative['bb_upper'] - informative['bb_lower'])
            
            # Keep the original naming for compatibility
            informative['bollinger_upperband'] = bollinger['upper']
            informative['bollinger_lowerband'] = bollinger['lower']
        except:
            informative['bb_lower'] = informative['close'] * 0.98
            informative['bb_middle'] = informative['close']
            informative['bb_upper'] = informative['close'] * 1.02
            informative['bb_percent'] = 0.5
            informative['bollinger_upperband'] = informative['close'] * 1.02
            informative['bollinger_lowerband'] = informative['close'] * 0.98

        # === EMA WITH PARAMETERIZED VALUES ===
        try:
            # Parameterized EMAs
            informative['ema_short'] = ta.EMA(informative, timeperiod=ema_short_period)
            informative['ema_long'] = ta.EMA(informative, timeperiod=ema_long_period)
            
            # Standard EMAs
            informative['ema9'] = ta.EMA(informative, timeperiod=9)
            informative['ema20'] = ta.EMA(informative, timeperiod=20)
            informative['ema50'] = ta.EMA(informative, timeperiod=50)
            informative['ema200'] = ta.EMA(informative, timeperiod=200)
            informative['ema21'] = ta.EMA(informative, timeperiod=21)  # If not already exists
            informative['price_vs_ema21'] = (informative['close'] - informative['ema21']) / informative['ema21'] * 100
            informative['price_vs_ema50'] = (informative['close'] - informative['ema50']) / informative['ema50'] * 100
            informative['price_vs_ema200'] = (informative['close'] - informative['ema200']) / informative['ema200'] * 100
            # Fill NaN values for EMAs
            ema_columns = ['ema_short', 'ema_long', 'ema9', 'ema20', 'ema21', 'ema50', 'ema200']  # Added ema21
            for col in ema_columns:
                if col in informative.columns:
                    informative[col] = informative[col].bfill().fillna(informative['close'])
                    
            # ADD: Fill NaN values for price position indicators
            price_pos_columns = ['price_vs_ema21', 'price_vs_ema50', 'price_vs_ema200']
            for col in price_pos_columns:
                if col in informative.columns:
                    informative[col] = informative[col].bfill().fillna(0)
        except:
            informative['ema_short'] = informative['close']
            informative['ema_long'] = informative['close']
            informative['ema9'] = informative['close']
            informative['ema20'] = informative['close']
            informative['ema21'] = informative['close']  # ADD THIS
            informative['ema50'] = informative['close']
            informative['ema200'] = informative['close']
            # ADD THESE FALLBACK VALUES:
            informative['price_vs_ema21'] = 0
            informative['price_vs_ema50'] = 0
            informative['price_vs_ema200'] = 0

        # === PRICE POSITION ANALYSIS WITH PARAMETERIZED VALUES ===
        try:
            informative['high_20'] = informative['high'].rolling(window=rolling_20_period).max()
            informative['low_20'] = informative['low'].rolling(window=rolling_20_period).min()
            informative['price_position'] = (informative['close'] - informative['low_20']) / (informative['high_20'] - informative['low_20'])
        except:
            informative['high_20'] = informative['high']
            informative['low_20'] = informative['low']
            informative['price_position'] = 0.5

        # === PIVOT POINTS WITH PARAMETERIZED VALUES ===
        try:
            pivots = pivot_points(informative, pivot_window)
            informative['pivot_lows'] = pivots['pivot_lows']
            informative['pivot_highs'] = pivots['pivot_highs']
        except Exception as e:
            logger.warning(f"Error calculating pivot points: {e}")
            informative['pivot_lows'] = np.nan
            informative['pivot_highs'] = np.nan

        # === DIVERGENCE ANALYSIS ===
        try:
            self.initialize_divergences_lists(informative)
            (high_iterator, low_iterator) = self.get_iterators(informative)
            
            # Add divergences for multiple indicators (expanded list from first file)
            indicators = ['rsi', 'stoch', 'roc', 'uo', 'ao', 'macd', 'cci', 'cmf', 'obv', 'mfi']
            for indicator in indicators:
                try:
                    if indicator in informative.columns:
                        self.add_divergences(informative, indicator, high_iterator, low_iterator, pivot_window)
                except Exception as e:
                    logger.warning(f"Error adding divergences for {indicator}: {e}")
                    continue
        except Exception as e:
            logger.warning(f"Error in divergence analysis: {e}")
            # Initialize with empty divergence data
            informative["total_bullish_divergences"] = 0
            informative["total_bullish_divergences_count"] = 0
            informative["total_bullish_divergences_names"] = ''
            informative["total_bearish_divergences"] = 0
            informative["total_bearish_divergences_count"] = 0
            informative["total_bearish_divergences_names"] = ''
        
        # === SIGNAL STRENGTH CALCULATION ===
        try:
            informative['signal_strength'] = self.calculate_signal_strength(informative)
        except:
            informative['signal_strength'] = 0
        
        # === MERGE BACK TO DATAFRAME ===
        for col in informative.columns:
            if col not in dataframe.columns:
                dataframe[col] = informative[col]
            else:
                dataframe[col] = informative[col]

        # ===== APPLY COIN-SPECIFIC OPTUNA ADJUSTMENTS =====
        # Apply coin-specific adjustments AFTER all indicators are calculated
        dataframe['coin_min_divergence'] = coin_params.get('min_divergence_count', self.min_divergence_count.value)
        dataframe['coin_min_signal'] = coin_params.get('min_signal_strength', self.min_signal_strength.value)
        dataframe['coin_volume_threshold'] = coin_params.get('volume_threshold', self.volume_threshold.value)
        dataframe['coin_adx_threshold'] = coin_params.get('adx_threshold', self.adx_threshold.value)

        # === ADDITIONAL MARKET STRUCTURE ANALYSIS WITH PARAMETERIZED VALUES ===
        try:
            dataframe['chop'] = choppiness_index(dataframe['high'], dataframe['low'], dataframe['close'], window=chop_period)
            dataframe['natr'] = ta.NATR(dataframe['high'], dataframe['low'], dataframe['close'], timeperiod=natr_period)
            dataframe['natr_diff'] = dataframe['natr'] - dataframe['natr'].shift(1)
            dataframe['natr_direction_change'] = (dataframe['natr_diff'] * dataframe['natr_diff'].shift(1) < 0)
        except:
            dataframe['chop'] = 50
            dataframe['natr'] = 0.02
            dataframe['natr_diff'] = 0
            dataframe['natr_direction_change'] = False

        # === SUPPORT/RESISTANCE LEVELS WITH PARAMETERIZED VALUES ===
        try:
            dataframe['swing_high'] = dataframe['high'].rolling(window=swing_period, min_periods=1).max()
            dataframe['swing_low'] = dataframe['low'].rolling(window=swing_period, min_periods=1).min()
            dataframe['distance_to_resistance'] = (dataframe['swing_high'] - dataframe['close']) / dataframe['close']
            dataframe['distance_to_support'] = (dataframe['close'] - dataframe['swing_low']) / dataframe['close']
        except:
            dataframe['swing_high'] = dataframe['high']
            dataframe['swing_low'] = dataframe['low']
            dataframe['distance_to_resistance'] = 0.02
            dataframe['distance_to_support'] = 0.02
        
        try:
            # Calculate recent highs/lows for pattern recognition WITH PARAMETERIZED VALUES
            dataframe['recent_high_5'] = dataframe['high'].rolling(window=recent_high_low_period).max()
            dataframe['recent_low_5'] = dataframe['low'].rolling(window=recent_high_low_period).min()
            dataframe['recent_high_10'] = dataframe['high'].rolling(window=rolling_10_period).max()
            dataframe['recent_low_10'] = dataframe['low'].rolling(window=rolling_10_period).min()
            
            # MA alignment indicators for trend confirmation
            dataframe['ma_alignment_bull'] = (
                (dataframe['ema21'] > dataframe['ema50']) & 
                (dataframe['ema50'] > dataframe['ema200'])
            )
            dataframe['ma_alignment_bear'] = (
                (dataframe['ema21'] < dataframe['ema50']) & 
                (dataframe['ema50'] < dataframe['ema200'])
            )
            
            # Rejection pattern detection
            dataframe['rejected_at_ema21'] = (
                (dataframe['high'].shift(1) > dataframe['ema21'].shift(1)) & 
                (dataframe['close'].shift(1) < dataframe['ema21'].shift(1))
            )
            dataframe['rejected_at_ema50'] = (
                (dataframe['high'].shift(1) > dataframe['ema50'].shift(1)) & 
                (dataframe['close'].shift(1) < dataframe['ema50'].shift(1))
            )
            dataframe['bounced_from_ema21'] = (
                (dataframe['low'].shift(1) < dataframe['ema21'].shift(1)) & 
                (dataframe['close'].shift(1) > dataframe['ema21'].shift(1))
            )
            dataframe['bounced_from_ema50'] = (
                (dataframe['low'].shift(1) < dataframe['ema50'].shift(1)) & 
                (dataframe['close'].shift(1) > dataframe['ema50'].shift(1))
            )
            
            # Candle pattern indicators
            dataframe['red_candle'] = dataframe['close'] < dataframe['close'].shift(1)
            dataframe['green_candle'] = dataframe['close'] > dataframe['close'].shift(1)
            dataframe['strong_red'] = (dataframe['close'] < dataframe['close'].shift(1)) & (dataframe['volume_ratio'] > 1.2)
            dataframe['strong_green'] = (dataframe['close'] > dataframe['close'].shift(1)) & (dataframe['volume_ratio'] > 1.2)
            
            # Fill NaN values
            pattern_columns = [
                'recent_high_5', 'recent_low_5', 'recent_high_10', 'recent_low_10',
                'ma_alignment_bull', 'ma_alignment_bear', 'rejected_at_ema21', 'rejected_at_ema50',
                'bounced_from_ema21', 'bounced_from_ema50', 'red_candle', 'green_candle',
                'strong_red', 'strong_green'
            ]
            for col in pattern_columns:
                if col in dataframe.columns:
                    if col.startswith('recent_'):
                        dataframe[col] = dataframe[col].bfill().fillna(dataframe['high'] if 'high' in col else dataframe['low'])
                    else:
                        dataframe[col] = dataframe[col].fillna(False)
        except Exception as e:
            logger.warning(f"Error calculating breakdown patterns: {e}")
            # Fallback values
            dataframe['recent_high_5'] = dataframe['high']
            dataframe['recent_low_5'] = dataframe['low']
            dataframe['recent_high_10'] = dataframe['high']
            dataframe['recent_low_10'] = dataframe['low']
            dataframe['ma_alignment_bull'] = False
            dataframe['ma_alignment_bear'] = False
            dataframe['rejected_at_ema21'] = False
            dataframe['rejected_at_ema50'] = False
            dataframe['bounced_from_ema21'] = False
            dataframe['bounced_from_ema50'] = False
            dataframe['red_candle'] = False
            dataframe['green_candle'] = False
            dataframe['strong_red'] = False
            dataframe['strong_green'] = False

        # === PLOT CONFIGURATION ===
        try:
            self.plot_config = (
                PlotConfig()
                .add_total_divergences_in_config(dataframe)
                .config)
        except:
            self.plot_config = None

        logger.debug(f"✅ [INDICATORS] Completed processing for {pair}")
        # Run daily optimization check (only once per day)
        if not hasattr(self, 'last_daily_check'):
            self.last_daily_check = 0

        current_time = time.time() 
        if current_time - self.last_daily_check > 3600:  # Check every hour instead of daily
            self.daily_optimization_check()
            self.last_daily_check = current_time
        return dataframe
    def _add_dummy_1h_columns(self, dataframe):
        """Add dummy 1h columns when higher timeframe data is unavailable"""
        # Use current 15m data to simulate 1h trend
        try:
            dataframe['ema50_1h_1h'] = ta.EMA(dataframe, timeperiod=200)  # Use longer period on 15m
            dataframe['ema200_1h_1h'] = ta.EMA(dataframe, timeperiod=800)  # Use much longer period
            dataframe['trend_1h_1h'] = ta.EMA(dataframe, timeperiod=84)   # 21 * 4 (4x 15m = 1h)
            dataframe['trend_strength_1h_1h'] = ta.ADX(dataframe)
            dataframe['rsi_1h_1h'] = ta.RSI(dataframe, timeperiod=56)     # Adjusted for timeframe
            
            # Fill NaN values
            columns_1h = ['ema50_1h_1h', 'ema200_1h_1h', 'trend_1h_1h', 'trend_strength_1h_1h', 'rsi_1h_1h']
            for col in columns_1h:
                if col in dataframe.columns:
                    dataframe[col] = dataframe[col].bfill().fillna(
                        dataframe['close'] if 'ema' in col or 'trend' in col else 
                        25 if 'strength' in col else 50
                    )
        except Exception as e:
            logger.warning(f"Error creating dummy 1h columns: {e}")
            # Absolute fallback
            dataframe['ema50_1h_1h'] = dataframe['close']
            dataframe['ema200_1h_1h'] = dataframe['close']
            dataframe['trend_1h_1h'] = dataframe['close']
            dataframe['trend_strength_1h_1h'] = 25
            dataframe['rsi_1h_1h'] = 50

    def calculate_signal_strength(self, dataframe: DataFrame) -> Series:
        """
        Calculate overall signal strength based on multiple factors
        """
        strength = pd.Series(0, index=dataframe.index)
        
        # Divergence strength
        strength += dataframe['total_bullish_divergences_count'] * 2
        strength += dataframe['total_bearish_divergences_count'] * 2
        
        # Volume strength
        volume_strength = np.where(dataframe['volume_ratio'] > 1.5, 2, 
                                 np.where(dataframe['volume_ratio'] > 1.2, 1, 0))
        strength += volume_strength
        
        # Trend alignment strength
        ema_bullish = (dataframe['ema20'] > dataframe['ema50']) & (dataframe['ema50'] > dataframe['ema200'])
        ema_bearish = (dataframe['ema20'] < dataframe['ema50']) & (dataframe['ema50'] < dataframe['ema200'])
        strength += np.where(ema_bullish | ema_bearish, 1, 0)
        
        # ADX strength
        strength += np.where(dataframe['adx'] > 30, 1, 0)
        
        return strength

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        print(f"Processing entry signals for {metadata['pair']}")  # Debug line
        """
        Enhanced entry strategy with RealOptuna optimization
        Based on better-performing original logic with dynamic parameter tuning
        """
        if len(dataframe) > 0:
            last_row = dataframe.iloc[-1]
            print(f"  RSI: {last_row.get('rsi', 0):.1f}")
            print(f"  Volume: {last_row.get('volume', 0)}")
            print(f"  Bull Div: {last_row.get('total_bullish_divergences', 0)}")
            print(f"  Bear Div: {last_row.get('total_bearish_divergences', 0)}")
        # Initialize
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        dataframe["enter_tag"] = ""
        dataframe['entry_type'] = 0

        
        # Get optimized parameters for this pair
        coin_params = self.get_coin_params(metadata['pair'])
        pair = metadata['pair']
        signal_lock_enabled = coin_params.get('signal_lock_enabled', True)
        # === BASIC FILTERS ===
        has_volume = dataframe['volume'] > 0
        
        # === PRIMARY DIVERGENCE CONDITIONS (E1) - HIGHEST PRIORITY ===
        bullish_divergence = (
            (dataframe['total_bullish_divergences'].shift(1) > 0)
        )
        
        bearish_divergence = (
            (dataframe['total_bearish_divergences'].shift(1) > 0)
        )

        # Volatility filter with Optuna optimization
        volatility_ok = (
            (dataframe['volatility'].shift(1) >= coin_params.get('min_volatility', 0.003) * coin_params.get('volatility_min_multiplier', 0.5)) &
            (dataframe['volatility'].shift(1) <= coin_params.get('max_volatility', 0.035) * coin_params.get('volatility_max_multiplier', 2.0))
        )
        
        # KC Bands filter with optimization
        bands_long = (
            (dataframe['low'] <= dataframe['kc_lowerband']) |
            (dataframe['close'] <= dataframe['kc_lowerband'])
        )
        
        bands_short = (
            (dataframe['high'] >= dataframe['kc_upperband']) |
            (dataframe['close'] >= dataframe['kc_upperband'])
        )
        
        # RSI filters with optimization
        rsi_long_ok = (
            (dataframe['rsi'].shift(1) < coin_params.get('e1_rsi_overbought_threshold', 75) + coin_params.get('e1_rsi_overbought_buffer', 5)) &
            (dataframe['rsi'].shift(1) > coin_params.get('e1_rsi_oversold_min', 25))
        )
        
        rsi_short_ok = (
            (dataframe['rsi'].shift(1) > coin_params.get('e1_rsi_oversold_threshold', 25) - coin_params.get('e1_rsi_oversold_buffer', 5)) &
            (dataframe['rsi'].shift(1) < coin_params.get('e1_rsi_overbought_max', 75))
        )
        
        long_condition_primary = (
            bullish_divergence &
            volatility_ok &
            bands_long &
            rsi_long_ok &
            has_volume
        )
        
        short_condition_primary = (
            bearish_divergence &
            volatility_ok &
            bands_short &
            rsi_short_ok &
            has_volume
        )
        # Enhanced Support Bounce (Long Entry)
        long_condition_support_bounce = (
            # Multiple support level confluence - stronger than single swing level
            (
                # Primary: Recent low near swing support (existing)
                (dataframe['low'] <= dataframe['swing_low'] * (1 + coin_params.get('support_tolerance', 0.005))) |
                # Secondary: EMA20/50 support confluence
                (
                    (dataframe['low'] <= dataframe['ema_short'] * (1 - coin_params.get('ema_support_tolerance', 0.008))) &
                    (dataframe['ema_short'] > dataframe['ema_long']) # Bullish EMA alignment
                ) |
                # Tertiary: BB lower band support
                (dataframe['low'] <= dataframe['bb_lower'] * (1 + coin_params.get('bb_support_tolerance', 0.01)))
            ) &
            
            # Stronger bounce confirmation
            (dataframe['close'] > dataframe['swing_low'] * (1 + coin_params.get('support_bounce_min', 0.002))) &
            (dataframe['close'] > dataframe['open']) &  # Green candle
            (dataframe['close'] > dataframe['close'].shift(1)) &  # Higher close than previous
            
            # Enhanced volume confirmation
            (dataframe['volume'] > dataframe['volume'].rolling(coin_params.get('sr_volume_window', 20)).mean() * 
             coin_params.get('strong_sr_volume_multiplier', 1.8)) &  # Higher threshold
            
            # Better trend context
            (dataframe['adx'] > coin_params.get('strong_sr_adx_min', 18)) &  # Stronger momentum requirement
            (dataframe['ema20'] >= dataframe['ema50']) &  # Bullish bias
            
            # Improved RSI positioning
            (dataframe['rsi'] > coin_params.get('strong_sr_rsi_min_long', 20)) &
            (dataframe['rsi'] < coin_params.get('strong_sr_rsi_max_long', 50)) &
            (dataframe['rsi'] > dataframe['rsi'].shift(coin_params.get('sr_momentum_lookback', 2))) &  # RSI improving
            
            # Enhanced position and structure filters
            (dataframe['price_position'] < coin_params.get('sr_price_position_max_long', 0.6)) &
            (dataframe['bb_percent'] < coin_params.get('sr_bb_max_long', 0.7)) &
            
            # Distance filter - ensure room to move up
            (dataframe['close'] < dataframe['swing_high'] * (1 - coin_params.get('min_distance_to_resistance', 0.02))) &
            
            # Volatility filters
            (dataframe['volatility'] >= coin_params.get('min_volatility', 0.005)) &
            (dataframe['volatility'] <= coin_params.get('max_volatility', 0.025)) &
            
            # Momentum confirmation over multiple periods
            (dataframe['close'] > dataframe['close'].shift(coin_params.get('sr_momentum_lookback', 2))) &
            
            has_volume
        )

        # Enhanced Resistance Rejection (Short Entry)
        short_condition_resistance_bounce = (
            # Multiple resistance level confluence - stronger than single swing level
            (
                # Primary: Recent high near swing resistance (existing)
                (dataframe['high'] >= dataframe['swing_high'] * (1 - coin_params.get('resistance_tolerance', 0.005))) |
                # Secondary: EMA20/50 resistance confluence
                (
                    (dataframe['high'] >= dataframe['ema_short'] * (1 + coin_params.get('ema_resistance_tolerance', 0.008))) &
                    (dataframe['ema_short'] < dataframe['ema_long'])  # Bearish EMA alignment
                ) |
                # Tertiary: BB upper band resistance
                (dataframe['high'] >= dataframe['bb_upper'] * (1 - coin_params.get('bb_resistance_tolerance', 0.01)))
            ) &
            
            # Stronger rejection confirmation
            (dataframe['close'] < dataframe['swing_high'] * (1 - coin_params.get('resistance_bounce_min', 0.002))) &
            (dataframe['close'] < dataframe['open']) &  # Red candle
            (dataframe['close'] < dataframe['close'].shift(1)) &  # Lower close than previous
            
            # Enhanced volume confirmation
            (dataframe['volume'] > dataframe['volume'].rolling(coin_params.get('sr_volume_window', 20)).mean() * 
             coin_params.get('strong_sr_volume_multiplier', 1.8)) &  # Higher threshold
            
            # Better trend context
            (dataframe['adx'] > coin_params.get('strong_sr_adx_min', 18)) &  # Stronger momentum requirement
            (dataframe['ema_short'] <= dataframe['ema_long']) &  # Bearish bias
            
            # Improved RSI positioning
            (dataframe['rsi'] > coin_params.get('strong_sr_rsi_min_short', 50)) &
            (dataframe['rsi'] < coin_params.get('strong_sr_rsi_max_short', 80)) &
            (dataframe['rsi'] < dataframe['rsi'].shift(coin_params.get('sr_momentum_lookback', 2))) &  # RSI declining
            
            # Enhanced position and structure filters
            (dataframe['price_position'] > coin_params.get('sr_price_position_min_short', 0.4)) &
            (dataframe['bb_percent'] > coin_params.get('sr_bb_min_short', 0.3)) &
            
            # Distance filter - ensure room to move down
            (dataframe['close'] > dataframe['swing_low'] * (1 + coin_params.get('min_distance_to_support', 0.02))) &
            
            # Volatility filters
            (dataframe['volatility'] >= coin_params.get('min_volatility', 0.005)) &
            (dataframe['volatility'] <= coin_params.get('max_volatility', 0.025)) &
            
            # Momentum confirmation over multiple periods
            (dataframe['close'] < dataframe['close'].shift(coin_params.get('sr_momentum_lookback', 2))) &
            
            has_volume
        )

        # === ENHANCED S/R BOUNCE WITH CONFLUENCE ===
        # Stronger version that requires multiple confirmations

        # Strong Support Bounce (Higher Quality)
        long_condition_strong_support = (
            # Multiple support level confluence - at least one must be touched
            (
                # EMA50 support (existing)
                (dataframe['low'] <= dataframe['ema_long'] * (1 - coin_params.get('support_tolerance', 0.005))) |
                # BB lower band support
                (dataframe['low'] <= dataframe['bb_lower'] * (1 + coin_params.get('bb_support_tolerance', 0.01))) |
                # KC lower band support  
                (dataframe['low'] <= dataframe['kc_lowerband'] * (1 + coin_params.get('kc_support_tolerance', 0.01))) |
                # EMA20 support for shorter-term bounces
                (dataframe['low'] <= dataframe['ema_short'] * (1 - coin_params.get('ema_support_tolerance', 0.008)))
            ) &
            
            # Stronger bounce confirmation
            (dataframe['close'] > dataframe['low'] * (1 + coin_params.get('support_bounce_min', 0.002))) &
            (dataframe['close'] > dataframe['open']) &  # Green candle confirmation
            
            # Trend and momentum context
            (dataframe['adx'] > coin_params.get('strong_sr_adx_min', 18)) &  # Momentum strength
            
            # Enhanced volume requirement - more selective
            (dataframe['volume'] > dataframe['volume'].rolling(coin_params.get('sr_volume_window', 20)).mean() * 
             coin_params.get('strong_sr_volume_multiplier', 1.8)) &
            
            # Tighter RSI positioning for quality entries
            (dataframe['rsi'] > coin_params.get('strong_sr_rsi_min_long', 20)) &
            (dataframe['rsi'] < coin_params.get('strong_sr_rsi_max_long', 50)) &
            
            # Distance filter - ensure room to move up (avoid resistance squeeze)
            (dataframe['close'] < dataframe['swing_high'] * (1 - coin_params.get('min_distance_to_resistance', 0.02))) &
            
            # Basic volatility filter
            (dataframe['volatility'] >= coin_params.get('min_volatility', 0.005)) &
            (dataframe['volatility'] <= coin_params.get('max_volatility', 0.025)) &
            
            has_volume
        )

        # Enhanced Strong Resistance (Short Entry)
        short_condition_strong_resistance = (
            # Multiple resistance level confluence - at least one must be touched
            (
                # EMA50 resistance (existing)
                (dataframe['high'] >= dataframe['ema_long'] *  (1 + coin_params.get('resistance_tolerance', 0.005))) |
                # BB upper band resistance
                (dataframe['high'] >= dataframe['bb_upper'] * (1 - coin_params.get('bb_resistance_tolerance', 0.01))) |
                # KC upper band resistance
                (dataframe['high'] >= dataframe['kc_upperband'] * (1 - coin_params.get('kc_resistance_tolerance', 0.01))) |
                # EMA20 resistance for shorter-term rejections
                (dataframe['high'] >= dataframe['ema_short'] * (1 + coin_params.get('ema_resistance_tolerance', 0.008)))
            ) &
            
            # Stronger rejection confirmation
            (dataframe['close'] < dataframe['high'] * (1 - coin_params.get('resistance_bounce_min', 0.002))) &
            (dataframe['close'] < dataframe['open']) &  # Red candle confirmation
            
            # Trend and momentum context
            (dataframe['adx'] > coin_params.get('strong_sr_adx_min', 18)) &  # Momentum strength
            
            # Enhanced volume requirement - more selective
            (dataframe['volume'] > dataframe['volume'].rolling(coin_params.get('sr_volume_window', 20)).mean() * 
             coin_params.get('strong_sr_volume_multiplier', 1.8)) &
            
            # Tighter RSI positioning for quality entries
            (dataframe['rsi'] > coin_params.get('strong_sr_rsi_min_short', 50)) &
            (dataframe['rsi'] < coin_params.get('strong_sr_rsi_max_short', 80)) &
            
            # Distance filter - ensure room to move down (avoid support squeeze)
            (dataframe['close'] > dataframe['swing_low'] * (1 + coin_params.get('min_distance_to_support', 0.02))) &
            
            # Basic volatility filter
            (dataframe['volatility'] >= coin_params.get('min_volatility', 0.005)) &
            (dataframe['volatility'] <= coin_params.get('max_volatility', 0.025)) &
            
            has_volume
        )
        # === TREND BREAKOUT CONDITIONS ===
        long_condition_trend = (
            # 1. EARLY TREND DETECTION - Multiple Scenarios
            (
                # Scenario A: EMA crossover with momentum (early trend capture)
                (
                    (dataframe['ema9'] > dataframe['ema20']) &  # Fast EMA above slower
                    (dataframe['ema_short'] >= dataframe['ema_long'].shift(1)) &  # Trend alignment improving
                    (dataframe['close'] > dataframe['ema9']) &  # Price above fast EMA
                    (dataframe['rsi'] > dataframe['rsi'].shift(1)) &  # RSI improving
                    (dataframe['volume_ratio'] > coin_params.get('volume_ratio_min', 1.1))  # Volume confirmation
                ) |
                
                # Scenario B: Breakout above resistance with trend alignment
                (
                    (dataframe['close'] > dataframe['high'].rolling(coin_params.get('trend_close_lookback', 2)).max().shift(1)) &  # New high
                    (dataframe['ema9'] >= dataframe['ema20']) &  # Fast EMA alignment
                    (dataframe['rsi'] > coin_params.get('trend_rsi_min', 40)) &  # RSI above minimum
                    (dataframe['volume'] > dataframe['volume'].rolling(coin_params.get('trend_volume_window', 20)).mean() * 
                     coin_params.get('trend_volume_multiplier', 1.5))  # Volume breakout
                ) |
                
                # Scenario C: Pullback to EMA with bounce (re-entry)
                (
                    (dataframe['ema_short'] > dataframe['ema_long']) &  # Main trend up
                    (dataframe['low'] <= dataframe['ema20']) &  # Pullback to EMA20
                    (dataframe['close'] > dataframe['ema20']) &  # Bounce above EMA20
                    (dataframe['close'] > dataframe['open']) &  # Green candle
                    (dataframe['bb_percent'] > coin_params.get('bb_percent_long_max', 0.2)) &  # Not oversold
                    (dataframe['price_position'] > coin_params.get('price_position_long_max', 0.3))  # In upper range
                )
            ) &
            
            # 2. CORE FILTERS (using existing parameters)
            (dataframe['rsi'] > coin_params.get('trend_rsi_min', 40)) &  # RSI minimum
            (dataframe['rsi'] < coin_params.get('trend_rsi_max', 60)) &  # RSI maximum  
            (dataframe['adx'] > coin_params.get('trend_adx_min', 25)) &  # Momentum strength
            
            # 3. VOLUME & VOLATILITY (existing parameters)
            (dataframe['volume'] > dataframe['volume'].rolling(coin_params.get('momentum_volume_window', 10)).mean() * 
             coin_params.get('momentum_volume_multiplier', 1.2)) &  # Volume confirmation
            (dataframe['volatility'] >= coin_params.get('min_volatility', 0.005)) &  # Minimum volatility
            (dataframe['volatility'] <= coin_params.get('max_volatility', 0.025)) &  # Maximum volatility
            
            # 4. RISK MANAGEMENT
            (dataframe['close'] > dataframe['close'].shift(coin_params.get('momentum_rsi_lookback', 3))) &  # Recent momentum
            has_volume
        )

        # Enhanced Bear_Trend - Multiple entry scenarios with existing parameters
        short_condition_trend = (
            # 1. EARLY TREND DETECTION - Multiple Scenarios
            (
                # Scenario A: EMA crossover with momentum (early trend capture)
                (
                    (dataframe['ema9'] < dataframe['ema20']) &  # Fast EMA below slower
                    (dataframe['ema_short'] <= dataframe['ema_long'].shift(1)) &  # Trend alignment deteriorating
                    (dataframe['close'] < dataframe['ema9']) &  # Price below fast EMA
                    (dataframe['rsi'] < dataframe['rsi'].shift(1)) &  # RSI deteriorating
                    (dataframe['volume_ratio'] > coin_params.get('volume_ratio_min', 1.1))  # Volume confirmation
                ) |
                
                # Scenario B: Breakdown below support with trend alignment
                (
                    (dataframe['close'] < dataframe['low'].rolling(coin_params.get('trend_close_lookback', 2)).min().shift(1)) &  # New low
                    (dataframe['ema9'] <= dataframe['ema20']) &  # Fast EMA alignment
                    (dataframe['rsi'] < coin_params.get('secondary_rsi_max_short', 80)) &  # RSI below maximum
                    (dataframe['volume'] > dataframe['volume'].rolling(coin_params.get('trend_volume_window', 20)).mean() * 
                     coin_params.get('trend_volume_multiplier', 1.5))  # Volume breakdown
                ) |
                
                # Scenario C: Pullback to EMA with rejection (re-entry)
                (
                    (dataframe['ema_short'] < dataframe['ema_long']) &  # Main trend down
                    (dataframe['high'] >= dataframe['ema20']) &  # Pullback to EMA20
                    (dataframe['close'] < dataframe['ema20']) &  # Rejection below EMA20
                    (dataframe['close'] < dataframe['open']) &  # Red candle
                    (dataframe['bb_percent'] < coin_params.get('bb_percent_short_min', 0.8)) &  # Not overbought
                    (dataframe['price_position'] < coin_params.get('price_position_short_min', 0.7))  # In lower range
                )
            ) &
            
            # 2. CORE FILTERS (using existing parameters)
            (dataframe['rsi'] > coin_params.get('secondary_rsi_min_short', 30)) &  # RSI minimum
            (dataframe['rsi'] < coin_params.get('secondary_rsi_max_short', 80)) &  # RSI maximum
            (dataframe['adx'] > coin_params.get('trend_adx_min', 25)) &  # Momentum strength
            
            # 3. VOLUME & VOLATILITY (existing parameters)
            (dataframe['volume'] > dataframe['volume'].rolling(coin_params.get('momentum_volume_window', 10)).mean() * 
             coin_params.get('momentum_volume_multiplier', 1.2)) &  # Volume confirmation
            (dataframe['volatility'] >= coin_params.get('min_volatility', 0.005)) &  # Minimum volatility
            (dataframe['volatility'] <= coin_params.get('max_volatility', 0.025)) &  # Maximum volatility
            
            # 4. RISK MANAGEMENT
            (dataframe['close'] < dataframe['close'].shift(coin_params.get('momentum_rsi_lookback', 3))) &  # Recent momentum
            has_volume
        )

        # === ENHANCED MOMENTUM CONTINUATION CONDITIONS ===
        long_condition_momentum = (
            # Momentum detection (multiple scenarios)
            (
                # Scenario 1: Strong momentum continuation
                (
                    (dataframe['close'] > dataframe['ema_short']) &
                    (dataframe['rsi'] > dataframe['rsi'].shift(1)) &
                    (dataframe['rsi'].shift(1) > coin_params.get('momentum_rsi_trigger', 50)) &
                    (dataframe['close'] > dataframe['high'].shift(coin_params.get('momentum_high_lookback', 2)))
                ) |
                # Scenario 2: Momentum recovery after pullback
                (
                    (dataframe['rsi'].shift(3) < coin_params.get('momentum_rsi_pivot', 45)) &
                    (dataframe['rsi'] > coin_params.get('momentum_rsi_trigger', 52)) &
                    (dataframe['close'] > dataframe['ema_short']) &
                    (dataframe['volume'] > dataframe['volume'].shift(1))
                ) |
                # Scenario 3: Breaking out of consolidation with momentum
                (
                    (dataframe['close'] > dataframe['high'].rolling(5).max().shift(1)) &
                    (dataframe['rsi'] > 50) &
                    (dataframe['volume'] > dataframe['volume'].rolling(10).mean() * 1.5)
                )
            ) &
            # Basic trend filter (less restrictive)
            (dataframe['ema20'] >= dataframe['ema50']) &
            # Volume confirmation (flexible)
            (dataframe['volume'] > dataframe['volume'].rolling(coin_params.get('momentum_volume_window', 10)).mean() * coin_params.get('momentum_volume_multiplier', 1.0)) &
            # Not in extreme overbought
            (dataframe['close'] < dataframe['kc_upperband'] * coin_params.get('momentum_kc_upper_factor', 0.998)) &
            # Momentum strength
            (dataframe['adx'] > coin_params.get('momentum_adx_min', 20)) &
            has_volume
        )

        short_condition_momentum = (
            # Momentum detection (multiple scenarios)
            (
                # Scenario 1: Strong momentum continuation
                (
                    (dataframe['close'] < dataframe['ema_short']) &
                    (dataframe['rsi'] < dataframe['rsi'].shift(1)) &
                    (dataframe['rsi'].shift(1) < coin_params.get('momentum_rsi_trigger_short', 50)) &
                    (dataframe['close'] < dataframe['low'].shift(coin_params.get('momentum_low_lookback', 2)))
                ) |
                # Scenario 2: Momentum recovery after pullback (bearish)
                (
                    (dataframe['rsi'].shift(3) > coin_params.get('momentum_rsi_pivot', 55)) &
                    (dataframe['rsi'] < coin_params.get('momentum_rsi_trigger_short', 48)) &
                    (dataframe['close'] < dataframe['ema_short']) &
                    (dataframe['volume'] > dataframe['volume'].shift(1))
                ) |
                # Scenario 3: Breaking down from consolidation with momentum
                (
                    (dataframe['close'] < dataframe['low'].rolling(5).min().shift(1)) &
                    (dataframe['rsi'] < 50) &
                    (dataframe['volume'] > dataframe['volume'].rolling(10).mean() * 1.5)
                )
            ) &
            # Basic trend filter (less restrictive)
            (dataframe['ema_short'] <= dataframe['ema_long']) &
            # Volume confirmation (flexible)
            (dataframe['volume'] > dataframe['volume'].rolling(coin_params.get('momentum_volume_window', 10)).mean() * coin_params.get('momentum_volume_multiplier', 1.0)) &
            # Not in extreme oversold
            (dataframe['close'] > dataframe['kc_lowerband'] * coin_params.get('momentum_kc_lower_factor', 1.002)) &
            # Momentum strength
            (dataframe['adx'] > coin_params.get('momentum_adx_min', 20)) &
            has_volume
        )

        # === ADD NEW MOMENTUM BREAKOUT CONDITIONS ===
        # Momentum Breakout Long
        long_momentum_breakout = (
            # Trend context using existing trend parameters
            (dataframe['ema_short'] > dataframe['ema_long']) & # Overall bullish trend
            (dataframe['adx'] > coin_params.get('momentum_adx_min', 25)) & # Strong momentum
            # Enhanced breakout detection
            (dataframe['close'] > dataframe['high'].rolling(coin_params.get('trend_close_lookback', 2) * 10).max().shift(1)) & # Longer period breakout
            # Volume confirmation using existing parameters
            (dataframe['volume'] > dataframe['volume'].rolling(coin_params.get('momentum_volume_window', 10)).mean() *
             coin_params.get('volume_breakout_multiplier', 2.5)) & # Your existing volume breakout param
            # RSI positioning using existing parameters
            (dataframe['rsi'] > coin_params.get('volume_breakout_rsi_min', 45.0)) &
            (dataframe['rsi'] < coin_params.get('volume_breakout_rsi_max', 60.0)) &
            # Momentum strength filter
            (dataframe['close'] > dataframe['close'].shift(coin_params.get('momentum_high_lookback', 2))) &
            # ATR expansion for volatility
            (dataframe['atr'] > dataframe['atr'].rolling(coin_params.get('trend_atr_window', 20)).mean() *
             coin_params.get('trend_atr_multiplier', 1.0)) &
            (dataframe['volatility'] >= coin_params.get('min_volatility', 0.005)) &  # Added: Min volatility
            (dataframe['volatility'] <= coin_params.get('max_volatility', 0.025)) &  # Added: Max volatility
            (dataframe['rsi'] > dataframe['rsi'].shift(1)) &  # Added: RSI improving
            (dataframe['close'] < dataframe['swing_high'] * (1 - coin_params.get('min_distance_to_resistance', 0.02))) &  # Added: Room to move up
            has_volume
        )
        
        # Momentum Breakout Short
        short_momentum_breakout = (
            # Trend context using existing trend parameters
            (dataframe['ema_short'] < dataframe['ema_long']) & # Overall bearish trend
            (dataframe['adx'] > coin_params.get('momentum_adx_min', 25)) & # Strong momentum
            # Enhanced breakdown detection
            (dataframe['close'] < dataframe['low'].rolling(coin_params.get('trend_close_lookback', 2) * 10).min().shift(1)) & # Longer period breakdown
            # Volume confirmation using existing parameters
            (dataframe['volume'] > dataframe['volume'].rolling(coin_params.get('momentum_volume_window', 10)).mean() *
             coin_params.get('volume_breakout_multiplier', 2.5)) & # Your existing volume breakout param
            # RSI positioning - inverted for short
            (dataframe['rsi'] < coin_params.get('breakdown_rsi_upper', 60)) & # Using your breakdown params
            (dataframe['rsi'] > coin_params.get('breakdown_rsi_lower', 40)) & # Avoid oversold bounces
            # Momentum strength filter
            (dataframe['close'] < dataframe['close'].shift(coin_params.get('momentum_low_lookback', 2))) &
            # ATR expansion for volatility
            (dataframe['atr'] > dataframe['atr'].rolling(coin_params.get('trend_atr_window', 20)).mean() *
             coin_params.get('trend_atr_multiplier', 1.0)) &
            (dataframe['volatility'] >= coin_params.get('min_volatility', 0.005)) &  # Added: Min volatility
            (dataframe['volatility'] <= coin_params.get('max_volatility', 0.025)) &  # Added: Max volatility
            (dataframe['rsi'] < dataframe['rsi'].shift(1)) &  # Added: RSI declining
            (dataframe['close'] > dataframe['swing_low'] * (1 + coin_params.get('min_distance_to_support', 0.02))) &  # Added: Room to move down
            has_volume
        )
        
        # === ADD PULLBACK MOMENTUM CONDITIONS ===
        # Using existing optimized parameters for better confluence and filtering

        # Enhanced Long Pullback Momentum
        long_pullback_momentum = (
            # Stronger trend context - multiple confirmations
            (dataframe['ema_short'] > dataframe['ema_long']) &
            (dataframe['ema50'] > dataframe['ema200']) &  # Longer-term bullish context
            (dataframe['adx'] > coin_params.get('momentum_adx_min', 25)) &  # Strong trend momentum
            
            # Better pullback detection - more substantial pullback required
            (
                # Touched EMA20 support (existing)
                (dataframe['low'] <= dataframe['ema20']) &
                # Must have pulled back meaningfully from recent high
                (dataframe['low'] < dataframe['high'].rolling(coin_params.get('momentum_high_lookback', 2) * 3).max().shift(1) * 0.98) &
                # Price was above EMA20 recently (confirms this is a pullback, not breakdown)
                (dataframe['close'].shift(coin_params.get('momentum_rsi_lookback', 3)) > dataframe['ema20'].shift(coin_params.get('momentum_rsi_lookback', 3)))
            ) &
            
            # Stronger bounce confirmation
            (dataframe['close'] > dataframe['ema_short']) &  # Back above support
            (dataframe['close'] > dataframe['open']) &   # Green candle
            (dataframe['close'] > dataframe['close'].shift(1)) &  # Higher close than previous
            
            # Enhanced volume and momentum confirmation
            (dataframe['volume'] > dataframe['volume'].rolling(coin_params.get('momentum_volume_window', 10)).mean() * 
             coin_params.get('momentum_volume_multiplier', 1.2) * 1.5) &  # 1.8x total volume requirement
            
            # Better RSI positioning and momentum
            (dataframe['rsi'] > coin_params.get('momentum_rsi_trigger', 55)) &  # Above midpoint
            (dataframe['rsi'] < coin_params.get('volume_breakout_rsi_max', 60.0)) &  # Not overbought
            (dataframe['rsi'] > dataframe['rsi'].rolling(coin_params.get('momentum_rsi_lookback', 3)).mean()) &  # RSI improving over period
            
            # Price structure confirmation
            (dataframe['close'] > dataframe['kc_middleband']) &  # Above middle of range
            (dataframe['close'] < dataframe['kc_upperband'] * coin_params.get('momentum_kc_upper_factor', 0.995)) &  # Not at extreme top
            
            # Volatility and market conditions
            (dataframe['atr'] > dataframe['atr'].rolling(coin_params.get('trend_atr_window', 20)).mean() * 
             coin_params.get('trend_atr_multiplier', 1.0)) &  # Sufficient volatility
            
            has_volume
        )

        # Enhanced Short Pullback Momentum
        short_pullback_momentum = (
            # Stronger trend context - multiple confirmations
            (dataframe['ema_short'] < dataframe['ema_long']) &
            (dataframe['ema50'] < dataframe['ema200']) &  # Longer-term bearish context
            (dataframe['adx'] > coin_params.get('momentum_adx_min', 25)) &  # Strong trend momentum
            
            # Better pullback detection - more substantial pullback required
            (
                # Touched EMA20 resistance (existing)
                (dataframe['high'] >= dataframe['ema20']) &
                # Must have pulled back meaningfully from recent low
                (dataframe['high'] > dataframe['low'].rolling(coin_params.get('momentum_low_lookback', 2) * 3).min().shift(1) * 1.02) &
                # Price was below EMA20 recently (confirms this is a pullback, not breakout)
                (dataframe['close'].shift(coin_params.get('momentum_rsi_lookback', 3)) < dataframe['ema20'].shift(coin_params.get('momentum_rsi_lookback', 3)))
            ) &
            
            # Stronger rejection confirmation
            (dataframe['close'] < dataframe['ema_short']) &  # Back below resistance
            (dataframe['close'] < dataframe['open']) &   # Red candle
            (dataframe['close'] < dataframe['close'].shift(1)) &  # Lower close than previous
            
            # Enhanced volume and momentum confirmation
            (dataframe['volume'] > dataframe['volume'].rolling(coin_params.get('momentum_volume_window', 10)).mean() * 
             coin_params.get('momentum_volume_multiplier', 1.2) * 1.5) &  # 1.8x total volume requirement
            
            # Better RSI positioning and momentum
            (dataframe['rsi'] < coin_params.get('momentum_rsi_trigger_short', 45)) &  # Below midpoint
            (dataframe['rsi'] > coin_params.get('volume_breakout_rsi_min', 45.0) - 15) &  # Not oversold (30+)
            (dataframe['rsi'] < dataframe['rsi'].rolling(coin_params.get('momentum_rsi_lookback', 3)).mean()) &  # RSI declining over period
            
            # Price structure confirmation
            (dataframe['close'] < dataframe['kc_middleband']) &  # Below middle of range
            (dataframe['close'] > dataframe['kc_lowerband'] * coin_params.get('momentum_kc_lower_factor', 1.005)) &  # Not at extreme bottom
            
            # Volatility and market conditions
            (dataframe['atr'] > dataframe['atr'].rolling(coin_params.get('trend_atr_window', 20)).mean() * 
             coin_params.get('trend_atr_multiplier', 1.0)) &  # Sufficient volatility
            
            has_volume
        )
        
        # === SECONDARY CONDITIONS ===
        long_condition_secondary = (
            bullish_divergence &
            (dataframe['close'] > dataframe['close'].shift(coin_params.get('secondary_close_lookback', 2))) &
            (dataframe['rsi'].shift(1) > coin_params.get('secondary_rsi_min', 35)) &
            (dataframe['rsi'].shift(1) < coin_params.get('secondary_rsi_max', 70)) &
            has_volume
        )
        
        short_condition_secondary = (
            bearish_divergence &
            (dataframe['close'] < dataframe['close'].shift(coin_params.get('secondary_close_lookback', 2))) &
            (dataframe['rsi'].shift(1) > coin_params.get('secondary_rsi_min_short', 30)) &
            (dataframe['rsi'].shift(1) < coin_params.get('secondary_rsi_max_short', 80)) &
            has_volume
        )
        
        # === TERTIARY CONDITIONS ===
        long_condition_tertiary = (
            bullish_divergence &
            (dataframe['close'] > dataframe['ema_short']) &
            (dataframe['rsi'].shift(1) > coin_params.get('tertiary_rsi_min', 40)) &
            (dataframe['rsi'].shift(1) < coin_params.get('tertiary_rsi_max', 65)) &
            (dataframe['volume'] > dataframe['volume'].rolling(coin_params.get('tertiary_volume_window', 10)).mean() * coin_params.get('tertiary_volume_multiplier', 1.0)) &
            has_volume
        )
        
        short_condition_tertiary = (
            bearish_divergence &
            (dataframe['close'] < dataframe['ema_short']) &
            (dataframe['rsi'].shift(1) > coin_params.get('tertiary_rsi_min_short', 35)) &
            (dataframe['rsi'].shift(1) < coin_params.get('tertiary_rsi_max_short', 75)) &
            (dataframe['volume'] > dataframe['volume'].rolling(coin_params.get('tertiary_volume_window', 10)).mean() * coin_params.get('tertiary_volume_multiplier', 1.0)) &
            has_volume
        )
        
        # === QUATERNARY CONDITIONS (EMA CROSSOVER) ===
        long_condition_quaternary = (
            bullish_divergence &
            (dataframe['close'].shift(1) < dataframe['ema20'].shift(1)) &
            (dataframe['close'] > dataframe['ema_short']) &
            (dataframe['rsi'] > coin_params.get('quaternary_rsi_min', 45)) &
            (dataframe['volume'] > dataframe['volume'].rolling(coin_params.get('quaternary_volume_window', 5)).mean() * coin_params.get('quaternary_volume_multiplier', 1.2)) &
            has_volume
        )
        
        short_condition_quaternary = (
            bearish_divergence &
            (dataframe['close'].shift(1) > dataframe['ema20'].shift(1)) &
            (dataframe['close'] < dataframe['ema_short']) &
            (dataframe['rsi'] < coin_params.get('quaternary_rsi_max', 55)) &
            (dataframe['volume'] > dataframe['volume'].rolling(coin_params.get('quaternary_volume_window', 5)).mean() * coin_params.get('quaternary_volume_multiplier', 1.2)) &
            has_volume
        )
        
        # === FIFTH CONDITIONS (KC MIDDLE BAND) ===
        long_condition_fifth = (
            bullish_divergence &
            (dataframe['close'] > dataframe['kc_middleband']) &
            (dataframe['close'] < dataframe['kc_upperband']) &
            (dataframe['close'] > dataframe['close'].shift(1)) &
            (dataframe['rsi'].shift(1) > coin_params.get('fifth_rsi_min', 35)) &
            (dataframe['rsi'].shift(1) < coin_params.get('fifth_rsi_max', 70)) &
            has_volume
        )
        
        short_condition_fifth = (
            bearish_divergence &
            (dataframe['close'] < dataframe['kc_middleband']) &
            (dataframe['close'] > dataframe['kc_lowerband']) &
            (dataframe['close'] < dataframe['close'].shift(1)) &
            (dataframe['rsi'].shift(1) > coin_params.get('fifth_rsi_min_short', 30)) &
            (dataframe['rsi'].shift(1) < coin_params.get('fifth_rsi_max_short', 65)) &
            has_volume
        )
        
        # === SIXTH CONDITIONS - STRONG DIVERGENCE COUNT ===
        long_condition_sixth = (
            (dataframe['total_bullish_divergences_count'] >= coin_params.get('sixth_min_div_count', 2)) &
            bullish_divergence &
            (dataframe['rsi'].shift(1) > coin_params.get('sixth_rsi_min', 30)) &
            (dataframe['rsi'].shift(1) < coin_params.get('sixth_rsi_max', 75)) &
            has_volume
        )
        
        short_condition_sixth = (
            (dataframe['total_bearish_divergences_count'] >= coin_params.get('sixth_min_div_count', 2)) &
            bearish_divergence &
            (dataframe['rsi'].shift(1) > coin_params.get('sixth_rsi_min_short', 25)) &
            (dataframe['rsi'].shift(1) < coin_params.get('sixth_rsi_max_short', 70)) &
            has_volume
        )
        # Enhanced Seventh Condition - Uses Optuna Parameters
        long_condition_seventh = (
            # 1. Trend Context - Bullish bias
            (dataframe['ema_short'] > dataframe['ema_long']) &
            (dataframe['ema50'] > dataframe['ema200']) & # Add stronger trend filter
            # 2. Pullback and Recovery Pattern
            (
                # Was below EMA20 recently (pullback happened)
                (dataframe['close'].shift(coin_params.get('seventh_close_lookback', 2)) < dataframe['ema20'].shift(coin_params.get('seventh_close_lookback', 2))) &
                # Now recovering above EMA20 (breakout from pullback)
                (dataframe['close'] > dataframe['ema_short']) &
                # Clear break above with momentum
                (dataframe['close'] > dataframe['ema20'] * 1.002) # 0.2% above EMA20
            ) &
            # 3. Price Action Confirmation
            (
                # Current candle is green (bullish momentum)
                (dataframe['close'] > dataframe['open']) &
                # Recent momentum building
                (dataframe['close'] > dataframe['close'].shift(1)) &
                # Not too extended above EMA20
                (dataframe['close'] < dataframe['ema20'] * 1.02) # Within 2% of EMA20
            ) &
            # 4. Volume and Strength Filters
            (dataframe['volume'] > dataframe['volume'].rolling(coin_params.get('seventh_volume_window', 20)).mean() * coin_params.get('seventh_volume_multiplier', 1.2)) &
            (dataframe['rsi'].shift(1) > coin_params.get('seventh_rsi_min', 35)) & # Not oversold
            (dataframe['rsi'].shift(1) < coin_params.get('seventh_rsi_max', 65)) & # Not overbought
            (dataframe['rsi'] > dataframe['rsi'].shift(1)) & # RSI improving
            # 5. Market Structure
            (dataframe['atr'] > dataframe['atr'].rolling(coin_params.get('seventh_atr_window', 20)).mean() * coin_params.get('seventh_atr_multiplier', 0.7)) &
            (dataframe['close'] > dataframe['kc_middleband']) & # Above middle band
            (dataframe['adx'] > coin_params.get('seventh_adx_min', 18)) &
            # 6. Recent Context Validation
            (dataframe['low'] <= dataframe['ema20'].shift(1)) & # Recently touched EMA20 support
            (dataframe['close'] > dataframe['low'] * 1.005) & # Clear bounce from low
            (dataframe['volatility'] <= coin_params.get('max_volatility', 0.025)) &  # Added: Volatility cap
            has_volume
        )

        # Bear E7 - Improved Short Condition  
        short_condition_seventh = (
            # 1. Trend Context - Bearish bias
            (dataframe['ema_short'] < dataframe['ema_long']) &
            (dataframe['ema50'] < dataframe['ema200']) & # Strong bearish trend
            # 2. Pullback and Rejection Pattern
            (
                # Was above EMA20 recently (pullback happened)
                (dataframe['close'].shift(coin_params.get('seventh_close_lookback', 2)) > dataframe['ema20'].shift(coin_params.get('seventh_close_lookback', 2))) &
                # Now rejected below EMA20 (breakdown from pullback)
                (dataframe['close'] < dataframe['ema_short']) &
                # Clear break below with momentum
                (dataframe['close'] < dataframe['ema20'] * 0.998) # 0.2% below EMA20
            ) &
            # 3. Price Action Confirmation
            (
                # Current candle is red (bearish momentum)
                (dataframe['close'] < dataframe['open']) &
                # Recent momentum declining
                (dataframe['close'] < dataframe['close'].shift(1)) &
                # Actually touched resistance (not just any EMA20 cross)
                (dataframe['high'] >= dataframe['ema_short'] * 0.999) & # Very close touch
                # Strong rejection with upper wick
                (dataframe['high'] - dataframe['close']) > (dataframe['close'] - dataframe['low'])
            ) &
            # 4. Volume and Strength Filters
            (dataframe['volume'] > dataframe['volume'].rolling(coin_params.get('seventh_volume_window', 20)).mean() * coin_params.get('seventh_volume_multiplier', 1.2)) &
            (dataframe['rsi'].shift(1) > coin_params.get('seventh_rsi_min_short', 35)) & # Not oversold
            (dataframe['rsi'].shift(1) < coin_params.get('seventh_rsi_max_short', 65)) & # Not overbought
            (dataframe['rsi'] < dataframe['rsi'].shift(1)) & # RSI declining
            # 5. Market Structure
            (dataframe['atr'] > dataframe['atr'].rolling(coin_params.get('seventh_atr_window', 20)).mean() * coin_params.get('seventh_atr_multiplier', 0.7)) &
            (dataframe['close'] < dataframe['kc_middleband']) & # Below middle band
            (dataframe['adx'] > coin_params.get('seventh_adx_min', 18)) &
            # 6. Recent Context Validation
            (dataframe['high'] >= dataframe['ema20'].shift(1)) & # Recently touched EMA20 resistance
            (dataframe['close'] < dataframe['high'] * 0.995) & # Clear rejection from high
            # Was showing recent strength before rejection
            (dataframe['rsi'].shift(3) > 40) &
            (dataframe['volatility'] <= coin_params.get('max_volatility', 0.025)) &  # Added: Volatility cap
            has_volume
        )

        # === PURE DIVERGENCE CONDITIONS - FALLBACK ===
        long_condition_div = (
            bullish_divergence &
            (dataframe['rsi'].shift(1) > coin_params.get('div_rsi_min', 25)) &
            (dataframe['rsi'].shift(1) < coin_params.get('div_rsi_max', 80)) &
            has_volume
        )
        
        short_condition_div = (
            bearish_divergence &
            (dataframe['rsi'].shift(1) > coin_params.get('div_rsi_min_short', 20)) &
            (dataframe['rsi'].shift(1) < coin_params.get('div_rsi_max_short', 75)) &
            has_volume
        )
        strong_bear_breakdown = (
            (dataframe['close'] < dataframe['ema21']) &
            (dataframe['ma_alignment_bear']) &  # EMAs aligned bearish
            (dataframe['volume_ratio'] > coin_params.get('breakdown_volume_threshold', 1.5)) &  # Strong volume
            (dataframe['rsi'] < coin_params.get('breakdown_rsi_upper', 60)) &  # Not overbought
            (dataframe['adx'] > coin_params.get('breakdown_adx_threshold', 25)) &  # Strong trend
            (dataframe['strong_red']) &  # Volume-confirmed red candle
            (dataframe['price_vs_ema21'] < -coin_params.get('breakdown_price_deviation', 1.0)) &  # Below EMA21
            (dataframe['rejected_at_ema21']) &  # Previous rejection at EMA21
            (dataframe['close'] < dataframe['ema50'])  # Below intermediate MA
        )
        
        # Strong Bull Breakdown (Reversal) Pattern  
        strong_bull_breakdown = (
            (dataframe['close'] > dataframe['ema21']) &
            (dataframe['ma_alignment_bull']) &  # EMAs aligned bullish
            (dataframe['volume_ratio'] > coin_params.get('breakdown_volume_threshold', 1.5)) &  # Strong volume
            (dataframe['rsi'] > coin_params.get('breakdown_rsi_lower', 40)) &  # Not oversold
            (dataframe['adx'] > coin_params.get('breakdown_adx_threshold', 25)) &  # Strong trend
            (dataframe['strong_green']) &  # Volume-confirmed green candle
            (dataframe['price_vs_ema21'] > coin_params.get('breakdown_price_deviation', 1.0)) &  # Above EMA21
            (dataframe['bounced_from_ema21']) &  # Previous bounce from EMA21
            (dataframe['close'] > dataframe['ema50'])  # Above intermediate MA
        )
        
        # Resistance Rejection Pattern (Bears)
        resistance_rejection_bear = (
            (dataframe['rejected_at_ema50']) &  # Rejected at EMA50 resistance
            (dataframe['red_candle']) &  # Follow-through down
            (dataframe['volume_ratio'] > coin_params.get('breakdown_volume_confirmation', 1.2)) &  # Volume confirmation
            (dataframe['rsi'] < 65) &  # Not extremely overbought
            (dataframe['close'] < dataframe['ema21']) &  # Below shorter MA
            (dataframe['ma_alignment_bear'])  # Overall bearish alignment
        )
        
        # Support Rejection Pattern (Bulls)
        support_rejection_bull = (
            (dataframe['bounced_from_ema50']) &  # Bounced from EMA50 support
            (dataframe['green_candle']) &  # Follow-through up
            (dataframe['volume_ratio'] > coin_params.get('breakdown_volume_confirmation', 1.2)) &  # Volume confirmation
            (dataframe['rsi'] > 35) &  # Not extremely oversold
            (dataframe['close'] > dataframe['ema21']) &  # Above shorter MA
            (dataframe['ma_alignment_bull'])  # Overall bullish alignment
        )
        
        # Higher Timeframe Bias (using EMA200 as proxy)
        higher_tf_bearish = (
            (dataframe['close'] < dataframe['ema200']) &  # Below long-term trend
            (dataframe['ema50'] < dataframe['ema200']) &  # MAs aligned bearish
            (dataframe['ema21'] < dataframe['ema50'])   # Short-term also bearish
        )
        
        higher_tf_bullish = (
            (dataframe['close'] > dataframe['ema200']) &  # Above long-term trend
            (dataframe['ema50'] > dataframe['ema200']) &  # MAs aligned bullish
            (dataframe['ema21'] > dataframe['ema50'])   # Short-term also bullish
        )
        
        # Enhanced breakdown with multiple confirmation
        enhanced_bear_breakdown = (
            (strong_bear_breakdown | resistance_rejection_bear) & 
            higher_tf_bearish &
            (dataframe['price_vs_ema200'] < 0)  # Below long-term trend
        )
        
        enhanced_bull_breakdown = (
            (strong_bull_breakdown | support_rejection_bull) & 
            higher_tf_bullish &
            (dataframe['price_vs_ema200'] > 0)
        )
        simple_breakout_long = (
            (dataframe['close'] > dataframe['high'].rolling(
                coin_params.get('rolling_10_period', 10)
            ).max().shift(1)) &
            (dataframe['volume'] > dataframe['volume'].rolling(
                coin_params.get('volume_sma_period', 20)
            ).mean() * coin_params.get('breakout_volume_multiplier', 1.3)) &
            (dataframe['rsi'] > coin_params.get('breakout_rsi_min', 40)) &
            (dataframe['rsi'] < coin_params.get('breakout_rsi_max', 75)) &
            (dataframe['ema_short'] > dataframe['ema_long']) &
            (dataframe['adx'] > coin_params.get('breakdown_adx_threshold', 25)) &  # Added: Require stronger trend (symmetric to short)
            (dataframe['volatility'] >= coin_params.get('min_volatility', 0.005)) &  # Added: Min volatility
            (dataframe['volatility'] <= coin_params.get('max_volatility', 0.025)) &  # Added: Max volatility to avoid whipsaws
            (dataframe['rsi'] > dataframe['rsi'].shift(1)) &  # Added: RSI improving confirmation
            (dataframe['close'] < dataframe['swing_high'] * (1 - coin_params.get('min_distance_to_resistance', 0.02))) &  # Added: Room to move up
            (dataframe['atr'] > dataframe['atr'].rolling(coin_params.get('trend_atr_window', 20)).mean() * 
             coin_params.get('trend_atr_multiplier', 1.0))  # Added: ATR expansion for true breakouts
        )
        simple_breakdown_short = (
            (dataframe['close'] < dataframe['low'].rolling(
                coin_params.get('rolling_10_period', 10)
            ).min().shift(1)) &
            (dataframe['volume'] > dataframe['volume'].rolling(
                coin_params.get('volume_sma_period', 20)
            ).mean() * coin_params.get('breakdown_volume_multiplier', 1.3)) &
            (dataframe['rsi'] > coin_params.get('breakdown_rsi_min', 35)) &
            (dataframe['rsi'] < coin_params.get('breakdown_rsi_max', 60)) &
            (dataframe['ema_short'] < dataframe['ema_long']) &
            (dataframe['adx'] > coin_params.get('breakdown_adx_threshold', 25)) &  # Added: Require stronger trend
            (dataframe['volatility'] <= coin_params.get('max_volatility', 0.025)) &  # Added: Cap volatility
            (dataframe['rsi'] < dataframe['rsi'].shift(1)) &  # Added: RSI declining confirmation
            (dataframe['close'] > dataframe['swing_low'] * (1 + coin_params.get('min_distance_to_support', 0.02)))  # Added: Room to move down
        )
                
        # === OPTIMIZED RSV1 REVERSAL CONDITIONS ===
        # Replace your current reversal conditions with these optimized versions

        # Long reversal conditions - using coin-specific parameters
        oversold_conditions = [
            (dataframe['rsi'] < coin_params.get('rsv1_rsi_oversold', 25.0)),
            (dataframe['willr'] < coin_params.get('rsv1_willr_oversold', -80.0)),
            (dataframe['cci'] < coin_params.get('rsv1_cci_oversold', -100.0)),
            (dataframe['close'] <= dataframe['bb_lower']),
            (dataframe['bb_percent'] < coin_params.get('rsv1_bb_oversold', 0.1)),
        ]

        long_reversal_signals = [
            (dataframe['volume_ratio'] > coin_params.get('rsv1_volume_factor', 1.5)),
            (dataframe['close'] > dataframe['open']),
            (dataframe['macdhist'] > dataframe['macdhist'].shift(1)),
            (dataframe['price_position'] < coin_params.get('rsv1_price_position_low', 0.2)),
        ]

        long_trend_filter = [
            (dataframe['ema_short'] > dataframe['ema_short'].shift(coin_params.get('trend_close_lookback', 3))),
            (dataframe['close'] > dataframe['close'].shift(coin_params.get('secondary_close_lookback', 2))),
        ]

        long_condition_reversal = (
            (sum(oversold_conditions) >= coin_params.get('rsv1_min_oversold_conditions', 3)) & 
            (sum(long_reversal_signals) >= coin_params.get('rsv1_min_reversal_signals', 2)) & 
            (sum(long_trend_filter) >= coin_params.get('rsv1_min_trend_filters', 1)) & 
            (dataframe['volume'] > 0)
        )

        # Short reversal conditions - using coin-specific parameters
        overbought_conditions = [
            (dataframe['rsi'] > coin_params.get('rsv1_rsi_overbought', 75.0)),
            (dataframe['willr'] > coin_params.get('rsv1_willr_overbought', -20.0)),
            (dataframe['cci'] > coin_params.get('rsv1_cci_overbought', 100.0)),
            (dataframe['close'] >= dataframe['bb_upper']),
            (dataframe['bb_percent'] > coin_params.get('rsv1_bb_overbought', 0.9)),
        ]

        short_reversal_signals = [
            (dataframe['volume_ratio'] > coin_params.get('rsv1_volume_factor', 1.5)),
            (dataframe['close'] < dataframe['open']),
            (dataframe['macdhist'] < dataframe['macdhist'].shift(1)),
            (dataframe['price_position'] > coin_params.get('rsv1_price_position_high', 0.8)),
        ]

        short_trend_filter = [
            (dataframe['ema_short'] < dataframe['ema_short'].shift(coin_params.get('trend_close_lookback', 3))),
            (dataframe['close'] < dataframe['close'].shift(coin_params.get('secondary_close_lookback', 2))),
        ]

        short_condition_reversal = (
            (sum(overbought_conditions) >= coin_params.get('rsv1_min_overbought_conditions', 3)) & 
            (sum(short_reversal_signals) >= coin_params.get('rsv1_min_reversal_signals', 2)) & 
            (sum(short_trend_filter) >= coin_params.get('rsv1_min_trend_filters', 1)) & 
            (dataframe['volume'] > 0)
        )
        # === MEAN REVERSION CONDITIONS ===
        mean_reversion_long = (
            (dataframe['bb_percent'].shift(1) < coin_params.get('bb_oversold_threshold', 0.15)) &
            (dataframe['rsi'].shift(1) < coin_params.get('mean_reversion_rsi_oversold', 28)) &
            (dataframe['close'] > dataframe['low'].shift(1)) &  # Not continuing down
            (dataframe['volume_ratio'].shift(1) > coin_params.get('volume_ratio_min', 1.1)) &
            (dataframe['close'] > dataframe['open']) &  # Recovery candle
            has_volume
        )

        mean_reversion_short = (
            (dataframe['bb_percent'].shift(1) > coin_params.get('bb_overbought_threshold', 0.85)) &
            (dataframe['rsi'].shift(1) > coin_params.get('mean_reversion_rsi_overbought', 72)) &
            (dataframe['close'] < dataframe['high'].shift(1)) &  # Not continuing up
            (dataframe['volume_ratio'].shift(1) > coin_params.get('volume_ratio_min', 1.1)) &
            (dataframe['close'] < dataframe['open']) &  # Rejection candle
            has_volume
        )

        # === MOMENTUM CONTINUATION CONDITIONS ===
        momentum_continuation_long = (
            (dataframe['ema9'] > dataframe['ema20']) &
            (dataframe['ema_short'] > dataframe['ema_long']) &
            (dataframe['close'].shift(1) < dataframe['ema9'].shift(1)) &  # Pullback
            (dataframe['close'] > dataframe['ema9']) &  # Back above
            (dataframe['price_position'].shift(1) > coin_params.get('momentum_pullback_max', 0.35)) &
            (dataframe['adx'].shift(1) > coin_params.get('adx_trending_min', 25)) &
            (dataframe['volume_ratio'].shift(1) > coin_params.get('volume_ratio_min', 1.1)) &
            has_volume
        )

        momentum_continuation_short = (
            (dataframe['ema9'] < dataframe['ema20']) &
            (dataframe['ema_short'] < dataframe['ema_long']) &
            (dataframe['close'].shift(1) > dataframe['ema9'].shift(1)) &  # Pullback
            (dataframe['close'] < dataframe['ema9']) &  # Back below
            (dataframe['price_position'].shift(1) < coin_params.get('momentum_pullback_min', 0.65)) &
            (dataframe['adx'].shift(1) > coin_params.get('adx_trending_min', 25)) &
            (dataframe['volume_ratio'].shift(1) > coin_params.get('volume_ratio_min', 1.1)) &
            has_volume
        )
        # Fast Momentum Long - Catches early bullish momentum with volume confirmation
        
        fast_momentum_long = (
            # Quick volume spike detection - using optimized params
            (dataframe['volume'] > dataframe['volume'].rolling(coin_params.get('momentum_volume_window', 10)).mean() * 
             coin_params.get('momentum_volume_multiplier', 1.2) * 1.15) &  # Slightly reduced multiplier for sensitivity
            
            # Price momentum (multiple scenarios - ANY can trigger)
            (
                # Scenario 1: Breaking recent high with momentum (MAIN SCENARIO)
                (
                    (dataframe['close'] > dataframe['high'].rolling(coin_params.get('rolling_10_period', 10)).max().shift(1)) &
                    (dataframe['close'] > dataframe['open']) &  # Green candle
                    (dataframe['rsi'] > coin_params.get('volume_breakout_rsi_min', 45.0) - 5) &  # Slightly lower threshold
                    (dataframe['rsi'] < coin_params.get('breakout_rsi_max', 75)) &
                    (dataframe['close'] > dataframe['close'].shift(1))  # Momentum confirmation
                ) |
                
                # Scenario 2: Volume breakout with price surge
                (
                    (dataframe['volume_ratio'] > coin_params.get('volume_breakout_multiplier', 2.5) * 0.7) &  # Use existing param
                    (dataframe['close'] > dataframe['ema_short']) &
                    (dataframe['close'] > dataframe['close'].shift(1) * 1.003) &
                    (dataframe['rsi'] > coin_params.get('volume_breakout_rsi_min', 45.0) - 5) &
                    (dataframe['rsi'] < coin_params.get('volume_breakout_rsi_max', 60.0) + 15)  # Wider range
                ) |
                
                # Scenario 3: EMA bounce with strong volume
                (
                    (dataframe['low'] <= dataframe['ema20'] * 1.005) &
                    (dataframe['close'] > dataframe['ema20'] * 1.002) &
                    (dataframe['close'] > dataframe['close'].shift(1)) &
                    (dataframe['close'] > dataframe['open']) &
                    (dataframe['rsi'] > coin_params.get('momentum_rsi_trigger', 50) - 10) &
                    (dataframe['rsi'] < coin_params.get('volume_breakout_rsi_max', 60.0) + 10)
                )
            ) &
            
            # Relaxed trend filter - catch early alignment
            (
                (dataframe['ema_short'] >= dataframe['ema_long'] * 0.995) |
                (dataframe['ema9'] > dataframe['ema20'])
            ) &
            
            # Momentum confirmation using optimized lookback
            (dataframe['rsi'] > dataframe['rsi'].shift(coin_params.get('momentum_rsi_lookback', 3) - 1)) &
            
            # Not in extreme overbought
            (dataframe['close'] < dataframe['kc_upperband'] * coin_params.get('momentum_kc_upper_factor', 0.998) + 0.012) &
            
            # Volatility check using optimized params
            (dataframe['volatility'] >= coin_params.get('min_volatility', 0.005) * 0.7) &
            (dataframe['volatility'] <= coin_params.get('max_volatility', 0.025) * 1.3) &
            
            # ATR expansion check using optimized params
            (dataframe['atr'] > dataframe['atr'].rolling(coin_params.get('trend_atr_window', 20)).mean() * 
             coin_params.get('trend_atr_multiplier', 1.0) * 0.9) &  # Slightly relaxed
            
            has_volume
        )

        # Fast Momentum Short - Catches early bearish momentum with volume confirmation
        fast_momentum_short = (
            # Quick volume spike detection - using optimized params
            (dataframe['volume'] > dataframe['volume'].rolling(coin_params.get('momentum_volume_window', 10)).mean() * 
             coin_params.get('momentum_volume_multiplier', 1.2) * 1.15) &
            
            # Price momentum (multiple scenarios - ANY can trigger)
            (
                # Scenario 1: Breaking recent low with momentum
                (
                    (dataframe['close'] < dataframe['low'].rolling(coin_params.get('rolling_10_period', 10)).min().shift(1)) &
                    (dataframe['close'] < dataframe['open']) &  # Red candle
                    (dataframe['rsi'] < coin_params.get('breakdown_rsi_upper', 60) + 5) &  # Use optimized param
                    (dataframe['rsi'] > coin_params.get('breakdown_rsi_lower', 40) - 15) &  # Wider range
                    (dataframe['close'] < dataframe['close'].shift(1))
                ) |
                
                # Scenario 2: Volume breakdown with price drop
                (
                    (dataframe['volume_ratio'] > coin_params.get('breakdown_volume_multiplier', 1.3) * 1.3) &  # Use existing param
                    (dataframe['close'] < dataframe['ema_short']) &
                    (dataframe['close'] < dataframe['close'].shift(1) * 0.997) &
                    (dataframe['rsi'] < coin_params.get('breakdown_rsi_upper', 60) + 5) &
                    (dataframe['rsi'] > coin_params.get('breakdown_rsi_lower', 40) - 15)
                ) |
                
                # Scenario 3: EMA rejection with strong volume
                (
                    (dataframe['high'] >= dataframe['ema20'] * 0.995) &
                    (dataframe['close'] < dataframe['ema20'] * 0.998) &
                    (dataframe['close'] < dataframe['close'].shift(1)) &
                    (dataframe['close'] < dataframe['open']) &
                    (dataframe['rsi'] < coin_params.get('momentum_rsi_trigger_short', 50) + 10) &
                    (dataframe['rsi'] > coin_params.get('breakdown_rsi_lower', 40) - 10)
                )
            ) &
            
            # Relaxed trend filter
            (
                (dataframe['ema_short'] <= dataframe['ema_long'] * 1.005) |
                (dataframe['ema9'] < dataframe['ema20'])
            ) &
            
            # Momentum confirmation using optimized lookback
            (dataframe['rsi'] < dataframe['rsi'].shift(coin_params.get('momentum_rsi_lookback', 3) - 1)) &
            
            # Not in extreme oversold
            (dataframe['close'] > dataframe['kc_lowerband'] * coin_params.get('momentum_kc_lower_factor', 1.002) - 0.012) &
            
            # Volatility check using optimized params
            (dataframe['volatility'] >= coin_params.get('min_volatility', 0.005) * 0.7) &
            (dataframe['volatility'] <= coin_params.get('max_volatility', 0.025) * 1.3) &
            
            # ATR expansion check using optimized params
            (dataframe['atr'] > dataframe['atr'].rolling(coin_params.get('trend_atr_window', 20)).mean() * 
             coin_params.get('trend_atr_multiplier', 1.0) * 0.9) &
            
            has_volume
        )
        # === NEW RSI + WillR CROSS CONDITIONS ===
        long_condition_rsi_willr = (
            # RSI in oversold zone and recovering (not exact cross)
            (dataframe['rsi'] < coin_params.get('e1_rsi_oversold_min', 35)) &  # Raised from 30
            (dataframe['rsi'] > dataframe['rsi'].shift(1)) &  # RSI turning up
            (dataframe['rsi'] > dataframe['rsi'].shift(2)) &  # Confirmed upward momentum
            
            # WillR deeply oversold and turning up
            (dataframe['willr'] < coin_params.get('rsv1_willr_oversold', -80)) &  # Less extreme
            (dataframe['willr'] > dataframe['willr'].shift(1)) &  # WillR improving
            
            # Price action confirmation
            (dataframe['close'] > dataframe['open']) &  # Green candle
            
            # Relaxed volume (just needs to be present)
            (dataframe['volume'] > dataframe['volume'].rolling(20).mean() * 1.1) &  # Much lower: 1.1x instead of 1.5x
            
            # Basic volatility range
            (dataframe['volatility'] >= coin_params.get('min_volatility', 0.005)) &
            (dataframe['volatility'] <= coin_params.get('max_volatility', 0.03)) &  # Raised upper limit
            
            has_volume
        )

        # Short: RSI + WillR reversal from overbought
        short_condition_rsi_willr = (
            # RSI in overbought zone and declining (not exact cross)
            (dataframe['rsi'] > coin_params.get('e1_rsi_overbought_max', 65)) &  # Lowered from 70
            (dataframe['rsi'] < dataframe['rsi'].shift(1)) &  # RSI turning down
            (dataframe['rsi'] < dataframe['rsi'].shift(2)) &  # Confirmed downward momentum
            
            # WillR deeply overbought and turning down
            (dataframe['willr'] > coin_params.get('rsv1_willr_overbought', -20)) &  # Less extreme
            (dataframe['willr'] < dataframe['willr'].shift(1)) &  # WillR declining
            
            # Price action confirmation
            (dataframe['close'] < dataframe['open']) &  # Red candle
            
            # Relaxed volume
            (dataframe['volume'] > dataframe['volume'].rolling(20).mean() * 1.1) &
            
            # Basic volatility range
            (dataframe['volatility'] >= coin_params.get('min_volatility', 0.005)) &
            (dataframe['volatility'] <= coin_params.get('max_volatility', 0.03)) &
            
            has_volume
        )
        long_condition_rsi_willr_strict = (
            # Both indicators deeply oversold - using EXISTING Optuna params
            (dataframe['rsi'] < coin_params.get('e1_rsi_oversold_min', 30)) &  # Reusing existing param
            (dataframe['willr'] < coin_params.get('rsv1_willr_oversold', -85)) &  # Reusing existing param
            
            # Strong momentum shift (multiple candles)
            (dataframe['rsi'] > dataframe['rsi'].shift(1)) &
            (dataframe['rsi'].shift(1) > dataframe['rsi'].shift(2)) &  # 2 candles up
            (dataframe['willr'] > dataframe['willr'].shift(1)) &
            
            # Price near support (using BB)
            (dataframe['close'] < dataframe['bollinger_lowerband'] * 1.005) &  # Within 0.5% of lower BB
            
            # Volume spike (but not excessive) - using EXISTING param
            (dataframe['volume'] > dataframe['volume'].rolling(coin_params.get('sr_volume_window', 20)).mean() * 
             coin_params.get('strong_sr_volume_multiplier', 1.3)) &  # Reusing existing param
            (dataframe['volume'] < dataframe['volume'].rolling(coin_params.get('sr_volume_window', 20)).mean() * 3.0) &  # Not panic selling
            
            # Volatility in good range - using EXISTING params
            (dataframe['volatility'] >= coin_params.get('min_volatility', 0.008)) &
            (dataframe['volatility'] <= coin_params.get('max_volatility', 0.025)) &
            
            has_volume
        )

        # Short: Both indicators deeply overbought with strong momentum shift
        short_condition_rsi_willr_strict = (
            # Both indicators deeply overbought - using EXISTING Optuna params
            (dataframe['rsi'] > coin_params.get('e1_rsi_overbought_max', 70)) &  # Reusing existing param
            (dataframe['willr'] > coin_params.get('rsv1_willr_overbought', -15)) &  # Reusing existing param
            
            # Strong momentum shift
            (dataframe['rsi'] < dataframe['rsi'].shift(1)) &
            (dataframe['rsi'].shift(1) < dataframe['rsi'].shift(2)) &  # 2 candles down
            (dataframe['willr'] < dataframe['willr'].shift(1)) &
            
            # Price near resistance
            (dataframe['close'] > dataframe['bollinger_upperband'] * 0.995) &  # Within 0.5% of upper BB
            
            # Volume spike - using EXISTING params
            (dataframe['volume'] > dataframe['volume'].rolling(coin_params.get('sr_volume_window', 20)).mean() * 
             coin_params.get('strong_sr_volume_multiplier', 1.3)) &
            (dataframe['volume'] < dataframe['volume'].rolling(coin_params.get('sr_volume_window', 20)).mean() * 3.0) &
            
            # Volatility in good range - using EXISTING params
            (dataframe['volatility'] >= coin_params.get('min_volatility', 0.008)) &
            (dataframe['volatility'] <= coin_params.get('max_volatility', 0.025)) &
            
            has_volume
        )

        # Combine breakdown patterns
        long_condition_breakdown = enhanced_bull_breakdown
        short_condition_breakdown = enhanced_bear_breakdown
        long_condition_hqdiv = bullish_divergence
        short_condition_hqdiv = bearish_divergence

        # 1. PRIMARY CONDITIONS (E1) - Check lock before assignment
        if not signal_lock_enabled or not self.is_signal_locked('Bull_E1', pair):
            dataframe.loc[long_condition_primary, 'enter_long'] = 1
            dataframe.loc[long_condition_primary, 'enter_tag'] = 'Bull_E1'
        else:
            print(f"Signal LOCKED: Bull_E1 for {pair}")
        
        if not signal_lock_enabled or not self.is_signal_locked('Bear_E1', pair):
            dataframe.loc[short_condition_primary, 'enter_short'] = 1  
            dataframe.loc[short_condition_primary, 'enter_tag'] = 'Bear_E1'
        else:
            print(f"Signal LOCKED: Bear_E1 for {pair}")

        # 2. SUPPORT/RESISTANCE BOUNCE CONDITIONS
        if not signal_lock_enabled or not self.is_signal_locked('Bull_SR_Bounce', pair):
            sr_bounce_long = long_condition_support_bounce & (dataframe['enter_tag'] == "")
            dataframe.loc[sr_bounce_long, 'enter_long'] = 1
            dataframe.loc[sr_bounce_long, 'enter_tag'] = 'Bull_SR_Bounce'
        else:
            print(f"Signal LOCKED: Bull_SR_Bounce for {pair}")
        if not signal_lock_enabled or not self.is_signal_locked('Bear_SR_Bounce', pair):
            sr_bounce_short = short_condition_resistance_bounce & (dataframe['enter_tag'] == "")
            dataframe.loc[sr_bounce_short, 'enter_short'] = 1
            dataframe.loc[sr_bounce_short, 'enter_tag'] = 'Bear_SR_Bounce'
        else:
            print(f"Signal LOCKED: Bear_SR_Bounce for {pair}")
        # 3. STRONG SUPPORT/RESISTANCE CONDITIONS
        if not signal_lock_enabled or not self.is_signal_locked('Bull_Strong_Support', pair):
            strong_sr_long = long_condition_strong_support & (dataframe['enter_tag'] == "")
            dataframe.loc[strong_sr_long, 'enter_long'] = 1
            dataframe.loc[strong_sr_long, 'enter_tag'] = 'Bull_Strong_Support'
        else:
            print(f"Signal LOCKED: Bull_Strong_Support for {pair}")
        if not signal_lock_enabled or not self.is_signal_locked('Bear_Strong_Resistance', pair):
            strong_sr_short = short_condition_strong_resistance & (dataframe['enter_tag'] == "")
            dataframe.loc[strong_sr_short, 'enter_short'] = 1
            dataframe.loc[strong_sr_short, 'enter_tag'] = 'Bear_Strong_Resistance'
        else:
            print(f"Signal LOCKED: Bear_Strong_Resistancce for {pair}")
    
        #if not signal_lock_enabled or not self.is_signal_locked('Bull_Trend', pair):
        #    trend_long = long_condition_trend & (dataframe['enter_tag'] == "")
        #    dataframe.loc[trend_long, 'enter_long'] = 1
        #    dataframe.loc[trend_long, 'enter_tag'] = 'Bull_Trend'
        #else:
        #    print(f"Signal LOCKED: Bull_Trend for {pair}")
        
        if not signal_lock_enabled or not self.is_signal_locked('Bear_Trend', pair):
            trend_short = short_condition_trend & (dataframe['enter_tag'] == "")
            dataframe.loc[trend_short, 'enter_short'] = 1
            dataframe.loc[trend_short, 'enter_tag'] = 'Bear_Trend'
        else:
            print(f"Signal LOCKED: Bear_Trend for {pair}")

        # 5. MOMENTUM CONDITIONS
        if not signal_lock_enabled or not self.is_signal_locked('Bull_Momentum', pair):
            momentum_long = long_condition_momentum & (dataframe['enter_tag'] == "")
            dataframe.loc[momentum_long, 'enter_long'] = 1
            dataframe.loc[momentum_long, 'enter_tag'] = 'Bull_Momentum'
        else:
            print(f"Signal LOCKED: Bull_Momentum for {pair}")
        
        if not signal_lock_enabled or not self.is_signal_locked('Bear_Momentum', pair):
            momentum_short = short_condition_momentum & (dataframe['enter_tag'] == "")
            dataframe.loc[momentum_short, 'enter_short'] = 1
            dataframe.loc[momentum_short, 'enter_tag'] = 'Bear_Momentum'
        else:
            print(f"Signal LOCKED: Bear_Momentum for {pair}")

        # 6. MOMENTUM BREAKOUT CONDITIONS (Enhanced)
        if not signal_lock_enabled or not self.is_signal_locked('Bull_Momentum_Breakout', pair):
            breakout_long = long_momentum_breakout & (dataframe['enter_tag'] == "")
            dataframe.loc[breakout_long, 'enter_long'] = 1
            dataframe.loc[breakout_long, 'enter_tag'] = 'Bull_Momentum_Breakout'
        else:
            print(f"Signal LOCKED: Bull_Momentum_Breakout for {pair}")
        
        if not signal_lock_enabled or not self.is_signal_locked('Bear_Momentum_Breakout', pair):
            breakout_short = short_momentum_breakout & (dataframe['enter_tag'] == "")
            dataframe.loc[breakout_short, 'enter_short'] = 1
            dataframe.loc[breakout_short, 'enter_tag'] = 'Bear_Momentum_Breakout'
        else:
            print(f"Signal LOCKED: Bear_Momentum_Breakout for {pair}")

        # 7. PULLBACK MOMENTUM CONDITIONS (Enhanced)
        if not signal_lock_enabled or not self.is_signal_locked('Bull_Pullback_Momentum', pair):
            pullback_long = long_pullback_momentum & (dataframe['enter_tag'] == "")
            dataframe.loc[pullback_long, 'enter_long'] = 1
            dataframe.loc[pullback_long, 'enter_tag'] = 'Bull_Pullback_Momentum'
        else:
            print(f"Signal LOCKED: Bull_Pullback_Momentum for {pair}")
        
        if not signal_lock_enabled or not self.is_signal_locked('Bear_Pullback_Momentum', pair):
            pullback_short = short_pullback_momentum & (dataframe['enter_tag'] == "")
            dataframe.loc[pullback_short, 'enter_short'] = 1
            dataframe.loc[pullback_short, 'enter_tag'] = 'Bear_Pullback_Momentum'
        else:
            print(f"Signal LOCKED: Bear_Pullback_Momentum for {pair}")

        # 8. SECONDARY CONDITIONS (E2)
        if not signal_lock_enabled or not self.is_signal_locked('Bull_E2', pair):
            secondary_long = long_condition_secondary & (dataframe['enter_tag'] == "")
            dataframe.loc[secondary_long, 'enter_long'] = 1
            dataframe.loc[secondary_long, 'enter_tag'] = 'Bull_E2'
        else:
            print(f"Signal LOCKED: Bull_E2 for {pair}")
        
        if not signal_lock_enabled or not self.is_signal_locked('Bear_E2', pair):
            secondary_short = short_condition_secondary & (dataframe['enter_tag'] == "")
            dataframe.loc[secondary_short, 'enter_short'] = 1
            dataframe.loc[secondary_short, 'enter_tag'] = 'Bear_E2'
        else:
            print(f"Signal LOCKED: Bear_E2 for {pair}")

        # 9. BREAKDOWN CONDITIONS
        #if not signal_lock_enabled or not self.is_signal_locked('Bull_Breakdown', pair):
        #    breakdown_long = long_condition_breakdown & (dataframe['enter_tag'] == "")
        #    dataframe.loc[breakdown_long, 'enter_long'] = 1
        #    dataframe.loc[breakdown_long, 'enter_tag'] = 'Bull_Breakdown'
        #else:
        #    print(f"Signal LOCKED: Bull_Breakdown for {pair}")
        #if not signal_lock_enabled or not self.is_signal_locked('Bear_Breakdown', pair):
        #    breakdown_short = short_condition_breakdown & (dataframe['enter_tag'] == "")
        #    dataframe.loc[breakdown_short, 'enter_short'] = 1
        #    dataframe.loc[breakdown_short, 'enter_tag'] = 'Bear_Breakdown'
        #else:
        #    print(f"Signal LOCKED: Bear_Breakdown for {pair}")
        # 10. PURE DIVERGENCE CONDITIONS (Lower Priority)
        if not signal_lock_enabled or not self.is_signal_locked('Bull_Div', pair):
            div_long = long_condition_div & (dataframe['enter_tag'] == "")
            dataframe.loc[div_long, 'enter_long'] = 1
            dataframe.loc[div_long, 'enter_tag'] = 'Bull_Div'
        else:
            print(f"Signal LOCKED: Bull_Div for {pair}")
        
        if not signal_lock_enabled or not self.is_signal_locked('Bear_Div', pair):
            div_short = short_condition_div & (dataframe['enter_tag'] == "")
            dataframe.loc[div_short, 'enter_short'] = 1
            dataframe.loc[div_short, 'enter_tag'] = 'Bear_Div'
        else:
            print(f"Signal LOCKED: Bear_Div for {pair}")

        # 11. HIGH QUALITY DIVERGENCE CONDITIONS (Lowest Priority)
        if not signal_lock_enabled or not self.is_signal_locked('Bull_Div_HQ', pair):
            hqdiv_long = long_condition_hqdiv & (dataframe['enter_tag'] == "")
            dataframe.loc[hqdiv_long, 'enter_long'] = 1
            dataframe.loc[hqdiv_long, 'enter_tag'] = 'Bull_Div_HQ'
        else:
            print(f"Signal LOCKED: Bull_Div_HQ for {pair}")
        
        if not signal_lock_enabled or not self.is_signal_locked('Bear_Div_HQ', pair):
            hqdiv_short = short_condition_hqdiv & (dataframe['enter_tag'] == "")
            dataframe.loc[hqdiv_short, 'enter_short'] = 1
            dataframe.loc[hqdiv_short, 'enter_tag'] = 'Bear_Div_HQ'
        else:
            print(f"Signal LOCKED: Bear_Div_HQ for {pair}")

        # 12. RSV1
        #if not signal_lock_enabled or not self.is_signal_locked('Bull_RSV1', pair):
        #    rsv1_long = long_condition_reversal & (dataframe['enter_tag'] == "")
        #    dataframe.loc[rsv1_long, 'enter_long'] = 1
        #    dataframe.loc[rsv1_long, 'enter_tag'] = 'Bull_RSV1'
        
        #if not signal_lock_enabled or not self.is_signal_locked('Bear_RSV1', pair):
        #    rsv1_short = short_condition_hqdiv & (dataframe['enter_tag'] == "")
        #    dataframe.loc[rsv1_short, 'enter_short'] = 1
        #    dataframe.loc[rsv1_short, 'enter_tag'] = 'Bear_RSV1'

        # 12. MR1
        #if not signal_lock_enabled or not self.is_signal_locked('Bull_MR1', pair):
        #    mr1_long = mean_reversion_long & (dataframe['enter_tag'] == "")
        #    dataframe.loc[mr1_long, 'enter_long'] = 1
        #    dataframe.loc[mr1_long, 'enter_tag'] = 'Bull_MR1'
        
        #if not signal_lock_enabled or not self.is_signal_locked('Bear_MR1', pair):
        #    mr1_short = mean_reversion_short & (dataframe['enter_tag'] == "")
        #    dataframe.loc[mr1_short, 'enter_short'] = 1
        #    dataframe.loc[mr1_short, 'enter_tag'] = 'Bear_MR1'


        # 12. MC1
        if not signal_lock_enabled or not self.is_signal_locked('Bull_MC1', pair):
            mc1_long = momentum_continuation_long & (dataframe['enter_tag'] == "")
            dataframe.loc[mc1_long, 'enter_long'] = 1
            dataframe.loc[mc1_long, 'enter_tag'] = 'Bull_MC1'
        else:
            print(f"Signal LOCKED: Bull_MC1 for {pair}")  # Fixed: [pair} -> {pair}
        if not signal_lock_enabled or not self.is_signal_locked('Bear_MC1', pair):
            mc1_short = momentum_continuation_short & (dataframe['enter_tag'] == "")
            dataframe.loc[mc1_short, 'enter_short'] = 1
            dataframe.loc[mc1_short, 'enter_tag'] = 'Bear_MC1'
        else:
            print(f"Signal LOCKED: Bear_MC1 for {pair}")
        if not signal_lock_enabled or not self.is_signal_locked('Bull_E7', pair):
            seventh_long = long_condition_seventh & (dataframe['enter_tag'] == "")
            dataframe.loc[seventh_long, 'enter_long'] = 1
            dataframe.loc[seventh_long, 'enter_tag'] = 'Bull_E7'
        else:
            print(f"Signal LOCKED: Bull_E7 for {pair}")
        
        if not signal_lock_enabled or not self.is_signal_locked('Bear_E7', pair):
            seventh_short = short_condition_seventh & (dataframe['enter_tag'] == "")
            dataframe.loc[seventh_short, 'enter_short'] = 1  
            dataframe.loc[seventh_short, 'enter_tag'] = 'Bear_E7'
        else:
            print(f"Signal LOCKED: Bear_E7 for {pair}")

        # 12. SIMPLE BREAKOUT CONDITIONS (New)
        if not signal_lock_enabled or not self.is_signal_locked('Bull_Simple_Breakout', pair):
            simple_breakout = simple_breakout_long & (dataframe['enter_tag'] == "")
            dataframe.loc[simple_breakout, 'enter_long'] = 1
            dataframe.loc[simple_breakout, 'enter_tag'] = 'Bull_Simple_Breakout'
        else:
            print(f"Signal LOCKED: Bull_Simple_Breakout for {pair}")

        if not signal_lock_enabled or not self.is_signal_locked('Bear_Simple_Breakdown', pair):
            simple_breakdown = simple_breakdown_short & (dataframe['enter_tag'] == "")
            dataframe.loc[simple_breakdown, 'enter_short'] = 1
            dataframe.loc[simple_breakdown, 'enter_tag'] = 'Bear_Simple_Breakdown'
        else:
            print(f"Signal LOCKED: Bear_Simple_Breakout for {pair}")
            
        # TERTIARY CONDITIONS (E3) - Missing trigger
        if not signal_lock_enabled or not self.is_signal_locked('Bull_E3', pair):
            tertiary_long = long_condition_tertiary & (dataframe['enter_tag'] == "")
            dataframe.loc[tertiary_long, 'enter_long'] = 1
            dataframe.loc[tertiary_long, 'enter_tag'] = 'Bull_E3'
        else:
            print(f"Signal LOCKED: Bull_E3 for {pair}")
        
        if not signal_lock_enabled or not self.is_signal_locked('Bear_E3', pair):
            tertiary_short = short_condition_tertiary & (dataframe['enter_tag'] == "")
            dataframe.loc[tertiary_short, 'enter_short'] = 1
            dataframe.loc[tertiary_short, 'enter_tag'] = 'Bear_E3'
        else:
            print(f"Signal LOCKED: Bear_E3 for {pair}")

        # QUATERNARY CONDITIONS (E4) - Missing trigger
        if not signal_lock_enabled or not self.is_signal_locked('Bull_E4', pair):
            quaternary_long = long_condition_quaternary & (dataframe['enter_tag'] == "")
            dataframe.loc[quaternary_long, 'enter_long'] = 1
            dataframe.loc[quaternary_long, 'enter_tag'] = 'Bull_E4'
        else:
            print(f"Signal LOCKED: Bull_E4 for {pair}")
        
        if not signal_lock_enabled or not self.is_signal_locked('Bear_E4', pair):
            quaternary_short = short_condition_quaternary & (dataframe['enter_tag'] == "")
            dataframe.loc[quaternary_short, 'enter_short'] = 1
            dataframe.loc[quaternary_short, 'enter_tag'] = 'Bear_E4'
        else:
            print(f"Signal LOCKED: Bear_E4 for {pair}")

        # FIFTH CONDITIONS (E5) - Missing trigger
        if not signal_lock_enabled or not self.is_signal_locked('Bull_E5', pair):
            fifth_long = long_condition_fifth & (dataframe['enter_tag'] == "")
            dataframe.loc[fifth_long, 'enter_long'] = 1
            dataframe.loc[fifth_long, 'enter_tag'] = 'Bull_E5'
        else:
            print(f"Signal LOCKED: Bull_E5 for {pair}")
        
        if not signal_lock_enabled or not self.is_signal_locked('Bear_E5', pair):
            fifth_short = short_condition_fifth & (dataframe['enter_tag'] == "")
            dataframe.loc[fifth_short, 'enter_short'] = 1
            dataframe.loc[fifth_short, 'enter_tag'] = 'Bear_E5'
        else:
            print(f"Signal LOCKED: Bear_E5 for {pair}")

        # SIXTH CONDITIONS (E6) - Missing trigger
        if not signal_lock_enabled or not self.is_signal_locked('Bull_E6', pair):
            sixth_long = long_condition_sixth & (dataframe['enter_tag'] == "")
            dataframe.loc[sixth_long, 'enter_long'] = 1
            dataframe.loc[sixth_long, 'enter_tag'] = 'Bull_E6'
        else:
            print(f"Signal LOCKED: Bull_E6 for {pair}")
        
        if not signal_lock_enabled or not self.is_signal_locked('Bear_E6', pair):
            sixth_short = short_condition_sixth & (dataframe['enter_tag'] == "")
            dataframe.loc[sixth_short, 'enter_short'] = 1
            dataframe.loc[sixth_short, 'enter_tag'] = 'Bear_E6'
        else:
            print(f"Signal LOCKED: Bear_E6 for {pair}")
        # FAST MOMENTUM CONDITIONS
        #if not signal_lock_enabled or not self.is_signal_locked('Bull_Fast_Momentum', pair):
        #    fast_momentum_bull = fast_momentum_long & (dataframe['enter_tag'] == "")
        #    dataframe.loc[fast_momentum_bull, 'enter_long'] = 1
        #    dataframe.loc[fast_momentum_bull, 'enter_tag'] = 'Bull_Fast_Momentum'
        #else:
        #    print(f"Signal LOCKED: Bull_Fast_Momentum for {pair}")
        
        #if not signal_lock_enabled or not self.is_signal_locked('Bear_Fast_Momentum', pair):
        #    fast_momentum_bear = fast_momentum_short & (dataframe['enter_tag'] == "")
        #    dataframe.loc[fast_momentum_bear, 'enter_short'] = 1
        #    dataframe.loc[fast_momentum_bear, 'enter_tag'] = 'Bear_Fast_Momentum'
        #else:
        #    print(f"Signal LOCKED: Bear_Fast_Momentum for {pair}")
        #if not signal_lock_enabled or not self.is_signal_locked('Bull_RSI_WillR_Cross', pair):
        #    rsi_willr_long = long_condition_rsi_willr & (dataframe['enter_tag'] == "")
        #    dataframe.loc[rsi_willr_long, 'enter_long'] = 1
        #    dataframe.loc[rsi_willr_long, 'enter_tag'] = 'Bull_RSI_WillR_Cross'
        #else:
        #    print(f"Signal LOCKED: Bull_RSI_WillR_Cross for {pair}")

        #if not signal_lock_enabled or not self.is_signal_locked('Bear_RSI_WillR_Cross', pair):
        #    rsi_willr_short = short_condition_rsi_willr & (dataframe['enter_tag'] == "")
        #    dataframe.loc[rsi_willr_short, 'enter_short'] = 1
        #    dataframe.loc[rsi_willr_short, 'enter_tag'] = 'Bear_RSI_WillR_Cross'
        #else:
        #    print(f"Signal LOCKED: Bear_RSI_WillR_Cross for {pair}")
        if not signal_lock_enabled or not self.is_signal_locked('Bull_RSI_WillR_Strict', pair):
            rsi_willr_strict_long = long_condition_rsi_willr_strict & (dataframe['enter_tag'] == "")
            dataframe.loc[rsi_willr_strict_long, 'enter_long'] = 1
            dataframe.loc[rsi_willr_strict_long, 'enter_tag'] = 'Bull_RSI_WillR_Strict'
            
        else:
            print(f"Signal LOCKED: Bull_RSI_WillR_Strict for {pair}")

        # Short entry with locking
        if self.can_short:
            if not signal_lock_enabled or not self.is_signal_locked('Bear_RSI_WillR_Strict', pair):
                rsi_willr_strict_short = short_condition_rsi_willr_strict & (dataframe['enter_tag'] == "")
                dataframe.loc[rsi_willr_strict_short, 'enter_short'] = 1
                dataframe.loc[rsi_willr_strict_short, 'enter_tag'] = 'Bear_RSI_WillR_Strict'
            
            else:
                print(f"Signal LOCKED: Bear_RSI_WillR_Strict for {pair}")
        # === LOGGING ===
        if len(dataframe) > 0:
            last_row = dataframe.iloc[-1]
            if last_row.get('enter_long', 0) == 1 or last_row.get('enter_short', 0) == 1:
                logger.info(f"🚀 {metadata['pair']} ENTRY DETECTED!")
                logger.info(f"   🏷️ Tag: {last_row.get('enter_tag', '')}")
                logger.info(f"   📊 RSI: {last_row.get('rsi', 0):.1f}")
                logger.info(f"   💧 Volume Ratio: {last_row.get('volume_ratio', 0):.2f}")
                logger.info(f"   🎯 Bull Div Count: {last_row.get('total_bullish_divergences_count', 0)}")
                logger.info(f"   🎯 Bear Div Count: {last_row.get('total_bearish_divergences_count', 0)}")
                logger.info(f"   📈 Signal Strength: {last_row.get('signal_strength', 0)}")
                logger.info(f"   🎯 ADX: {last_row.get('adx', 0):.1f}")
        long_signals = (dataframe['enter_long'] == 1).sum()
        short_signals = (dataframe['enter_short'] == 1).sum()
        print(f"  Generated {long_signals} long signals, {short_signals} short signals")
        return dataframe

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                 side: str, **kwargs) -> float:
        """
        Adaptive leverage based on signal strength
        """
        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if len(dataframe) > 0:
                current_signal_strength = dataframe['signal_strength'].iloc[-1]
                
                # Reduce leverage for weaker signals
                if current_signal_strength >= 8:
                    return self.leverage_value  # Full leverage for strong signals
                elif current_signal_strength >= 6:
                    return self.leverage_value * 0.8  # 80% leverage
                elif current_signal_strength >= 4:
                    return self.leverage_value * 0.6  # 60% leverage
                else:
                    return self.leverage_value * 0.4  # 40% leverage for weak signals
        except:
            pass
        
        return self.leverage_value * 0.5  # Conservative fallback

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                        current_profit: float, after_fill: bool, **kwargs) -> float:
        """
        Enhanced dynamic stoploss with breakeven and trailing functionality
        Works seamlessly with your existing custom_exit
        """
        
        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if len(dataframe) < 1:
                return self.stoploss
            
            # Get market data
            last_row = dataframe.iloc[-1]
            atr = last_row.get('atr', 0)
            current_price = last_row['close']
            volatility = last_row.get('volatility', 0.02)
            signal_strength = last_row.get('signal_strength', 5)
            
            entry_tag = trade.enter_tag or ""
            trade_duration_minutes = (current_time - trade.open_date_utc).total_seconds() / 60
            
            # === BREAKEVEN LOGIC ===
            # Move to breakeven once we hit decent profit
            breakeven_trigger = 0.025  # 2.5% profit
            
            if any(tag in entry_tag for tag in ['MR1', 'E1', 'E2']):
                breakeven_trigger = 0.02   # Earlier breakeven for mean reversion
            elif any(tag in entry_tag for tag in ['Trend', 'SR_Bounce']):
                breakeven_trigger = 0.035  # Later breakeven for trend trades
            
            # Move to breakeven + small buffer
            if current_profit >= breakeven_trigger:
                breakeven_stop = 0.003  # 0.3% positive (covers fees)
                return stoploss_from_open(breakeven_stop, current_profit, is_short=trade.is_short)
            
            # === DYNAMIC TRAILING LOGIC ===
            # Start trailing after higher profit threshold  
            trailing_trigger = 0.045  # 4.5% profit
            
            if current_profit >= trailing_trigger:
                # Entry-specific trailing distances
                if any(tag in entry_tag for tag in ['MR1', 'E1', 'E2']):
                    trailing_distance = 0.015  # Tight trailing for mean reversion
                elif any(tag in entry_tag for tag in ['MC1', 'Momentum', 'Breakout']):
                    trailing_distance = 0.025  # Normal trailing for momentum
                elif any(tag in entry_tag for tag in ['Trend', 'SR_Bounce', 'Strong']):
                    trailing_distance = 0.035  # Wide trailing for trend
                else:
                    trailing_distance = 0.02   # Default trailing
                
                # Adjust for volatility
                volatility_adj = min(1.5, max(0.7, volatility / 0.02))
                trailing_distance *= volatility_adj
                
                # Adjust for signal strength
                if signal_strength >= 8:
                    trailing_distance *= 1.2   # Wider trailing for strong signals
                elif signal_strength <= 4:
                    trailing_distance *= 0.8   # Tighter trailing for weak signals
                
                return stoploss_from_open(trailing_distance, current_profit, is_short=trade.is_short)
            
            # === INITIAL DYNAMIC STOPLOSS (same as before) ===
            if signal_strength >= 8:
                atr_multiplier = 3.0
            elif signal_strength >= 6:
                atr_multiplier = 2.5
            elif signal_strength >= 4:
                atr_multiplier = 2.0
            else:
                atr_multiplier = 1.5
            
            # Adjust for volatility
            volatility_adjustment = min(1.5, max(0.7, volatility / 0.02))
            atr_multiplier *= volatility_adjustment
            
            # Calculate stop distance
            if atr > 0:
                stop_distance = (atr * atr_multiplier) / current_price
            else:
                stop_distance = 0.08 + (volatility * 2)
            
            # Entry type adjustments
            if any(tag in entry_tag for tag in ['MR1', 'E1', 'E2']):
                stop_distance *= 0.8
            elif any(tag in entry_tag for tag in ['MC1', 'Momentum', 'Breakout']):
                stop_distance *= 1.0
            elif any(tag in entry_tag for tag in ['Trend', 'SR_Bounce', 'Strong']):
                stop_distance *= 1.2
            elif any(tag in entry_tag for tag in ['Div', 'RSV1']):
                stop_distance *= 0.9
            
            # Safety limits
            min_stop = 0.04
            max_stop = 0.15
            stop_distance = max(min_stop, min(max_stop, stop_distance))
            
            dynamic_stoploss = -stop_distance
            
            # Time-based tightening for initial stops
            if trade_duration_minutes > 60:
                time_factor = min(0.2, (trade_duration_minutes - 60) / 300)
                dynamic_stoploss *= (1 - time_factor)
            
            # Never worse than original
            dynamic_stoploss = max(dynamic_stoploss, self.stoploss)
            
            # Minimal logging
            if hasattr(self, '_stoploss_log_counter'):
                self._stoploss_log_counter = getattr(self, '_stoploss_log_counter', 0) + 1
            else:
                self._stoploss_log_counter = 1
                
            if self._stoploss_log_counter % 30 == 0:
                if current_profit >= trailing_trigger:
                    logger.info(f"Trailing {pair}: Profit={current_profit:.3f}, Stop={dynamic_stoploss:.3f}, Tag={entry_tag}")
                elif current_profit >= breakeven_trigger:
                    logger.info(f"Breakeven {pair}: Profit={current_profit:.3f}, Stop={dynamic_stoploss:.3f}, Tag={entry_tag}")
                else:
                    logger.info(f"Dynamic SL {pair}: Signal={signal_strength}, Stop={dynamic_stoploss:.3f}, Tag={entry_tag}")
            
            return dynamic_stoploss
            
        except Exception as e:
            logger.error(f"Error in custom_stoploss for {pair}: {e}")
            return self.stoploss

    def initialize_divergences_lists(self, dataframe: DataFrame):
        """Initialize divergence tracking columns"""
        # Bullish Divergences
        dataframe["total_bullish_divergences"] = np.nan
        dataframe["total_bullish_divergences_count"] = 0
        dataframe["total_bullish_divergences_names"] = ''

        # Bearish Divergences
        dataframe["total_bearish_divergences"] = np.nan
        dataframe["total_bearish_divergences_count"] = 0
        dataframe["total_bearish_divergences_names"] = ''

    def get_iterators(self, dataframe):
        """Get pivot point iterators for divergence detection"""
        low_iterator = []
        high_iterator = []

        for index, row in enumerate(dataframe.itertuples(index=True, name='Pandas')):
            if np.isnan(row.pivot_lows):
                low_iterator.append(0 if len(low_iterator) == 0 else low_iterator[-1])
            else:
                low_iterator.append(index)
            if np.isnan(row.pivot_highs):
                high_iterator.append(0 if len(high_iterator) == 0 else high_iterator[-1])
            else:
                high_iterator.append(index)
        
        return high_iterator, low_iterator

    def add_divergences(self, dataframe: DataFrame, indicator: str, high_iterator, low_iterator, window):
        """Add divergence detection for a specific indicator"""
        (bearish_divergences, bearish_lines, bullish_divergences, bullish_lines) = self.divergence_finder_dataframe(
            dataframe, indicator, high_iterator, low_iterator, window)
        dataframe['bearish_divergence_' + indicator + '_occurence'] = bearish_divergences
        dataframe['bullish_divergence_' + indicator + '_occurence'] = bullish_divergences

    def divergence_finder_dataframe(self, dataframe: DataFrame, indicator_source: str, high_iterator, low_iterator, window) -> Tuple[pd.Series, pd.Series]:
        """Enhanced divergence finder with improved logic"""
        bearish_lines = [np.empty(len(dataframe['close'])) * np.nan]
        bearish_divergences = np.empty(len(dataframe['close'])) * np.nan
        bullish_lines = [np.empty(len(dataframe['close'])) * np.nan]
        bullish_divergences = np.empty(len(dataframe['close'])) * np.nan

        for index, row in enumerate(dataframe.itertuples(index=True, name='Pandas')):

            # Bearish divergence detection
            bearish_occurence = self.bearish_divergence_finder(
                dataframe, dataframe[indicator_source], high_iterator, index, window)

            if bearish_occurence is not None:
                (prev_pivot, current_pivot) = bearish_occurence
                bearish_prev_pivot = dataframe['close'][prev_pivot]
                bearish_current_pivot = dataframe['close'][current_pivot]
                bearish_ind_prev_pivot = dataframe[indicator_source][prev_pivot]
                bearish_ind_current_pivot = dataframe[indicator_source][current_pivot]
                
                # Enhanced validation for bearish divergence
                price_diff = abs(bearish_current_pivot - bearish_prev_pivot)
                indicator_diff = abs(bearish_ind_current_pivot - bearish_ind_prev_pivot)
                time_diff = current_pivot - prev_pivot
                
                # Only accept divergences with sufficient magnitude and time separation
                if (price_diff > dataframe['atr'][current_pivot] * 0.5 and 
                    indicator_diff > 5 and 
                    time_diff >= 5):
                    
                    bearish_divergences[index] = row.close
                    dataframe.loc[index, "total_bearish_divergences"] = row.close
                    dataframe.loc[index, "total_bearish_divergences_count"] += 1
                    dataframe.loc[index, "total_bearish_divergences_names"] += indicator_source.upper() + '<br>'

            # Bullish divergence detection
            bullish_occurence = self.bullish_divergence_finder(
                dataframe, dataframe[indicator_source], low_iterator, index, window)

            if bullish_occurence is not None:
                (prev_pivot, current_pivot) = bullish_occurence
                bullish_prev_pivot = dataframe['close'][prev_pivot]
                bullish_current_pivot = dataframe['close'][current_pivot]
                bullish_ind_prev_pivot = dataframe[indicator_source][prev_pivot]
                bullish_ind_current_pivot = dataframe[indicator_source][current_pivot]
                
                # Enhanced validation for bullish divergence
                price_diff = abs(bullish_current_pivot - bullish_prev_pivot)
                indicator_diff = abs(bullish_ind_current_pivot - bullish_ind_prev_pivot)
                time_diff = current_pivot - prev_pivot
                
                # Only accept divergences with sufficient magnitude and time separation
                if (price_diff > dataframe['atr'][current_pivot] * 0.5 and 
                    indicator_diff > 5 and 
                    time_diff >= 5):
                    
                    bullish_divergences[index] = row.close
                    dataframe.loc[index, "total_bullish_divergences"] = row.close
                    
                    # CORRECT - increment BULLISH counters for bullish divergence:
                    dataframe.loc[index, "total_bullish_divergences_count"] += 1
                    dataframe.loc[index, "total_bullish_divergences_names"] += indicator_source.upper() + '<br>'

        return (bearish_divergences, bearish_lines, bullish_divergences, bullish_lines)

    def bearish_divergence_finder(self, dataframe, indicator, high_iterator, index, window):
        """Enhanced bearish divergence detection"""
        try:
            if high_iterator[index] == index:
                current_pivot = high_iterator[index]
                occurences = list(dict.fromkeys(high_iterator))
                current_index = occurences.index(high_iterator[index])
                
                # Use passed window instead of self.window.value if possible, or adapt logic
                # The original code used self.window.value for the loop range. 
                # We should probably keep using self.window.value for the loop limit if it refers to max lookback pivots,
                # but we need 'window' (the pivot window) for the offset.
                # Let's assume self.window.value is correct for the loop limit (number of pivots to check back).
                
                for i in range(current_index-1, current_index - self.window.value - 1, -1):
                    if i < 0 or i >= len(occurences):
                        continue
                    prev_pivot = occurences[i]
                    if np.isnan(prev_pivot):
                        continue
                    
                    # Enhanced divergence validation
                    # Fix lookahead bias: offset indicator check by window
                    price_higher = dataframe['pivot_highs'][current_pivot] > dataframe['pivot_highs'][prev_pivot]
                    indicator_lower = indicator[current_pivot - window] < indicator[prev_pivot - window]
                    
                    price_lower = dataframe['pivot_highs'][current_pivot] < dataframe['pivot_highs'][prev_pivot]
                    indicator_higher = indicator[current_pivot - window] > indicator[prev_pivot - window]
                    
                    # Check for classic or hidden divergence
                    if (price_higher and indicator_lower) or (price_lower and indicator_higher):
                        # Additional validation: check trend consistency
                        if self.validate_divergence_trend(dataframe, prev_pivot, current_pivot, 'bearish'):
                            return (prev_pivot, current_pivot)
        except:
            pass
        return None

    def bullish_divergence_finder(self, dataframe, indicator, low_iterator, index, window):
        """Enhanced bullish divergence detection"""
        try:
            if low_iterator[index] == index:
                current_pivot = low_iterator[index]
                occurences = list(dict.fromkeys(low_iterator))
                current_index = occurences.index(low_iterator[index])
                
                for i in range(current_index-1, current_index - self.window.value - 1, -1):
                    if i <  0 or i >= len(occurences):
                        continue
                    prev_pivot = occurences[i]
                    if np.isnan(prev_pivot):
                        continue
                    
                    # Enhanced divergence validation
                    # Fix lookahead bias: offset indicator check by window
                    price_lower = dataframe['pivot_lows'][current_pivot] < dataframe['pivot_lows'][prev_pivot]
                    indicator_higher = indicator[current_pivot - window] > indicator[prev_pivot - window]
                    
                    price_higher = dataframe['pivot_lows'][current_pivot] > dataframe['pivot_lows'][prev_pivot]
                    indicator_lower = indicator[current_pivot - window] < indicator[prev_pivot - window]
                    
                    # Check for classic or hidden divergence
                    if (price_lower and indicator_higher) or (price_higher and indicator_lower):
                        # Additional validation: check trend consistency
                        if self.validate_divergence_trend(dataframe, prev_pivot, current_pivot, 'bullish'):
                            return (prev_pivot, current_pivot)
        except:
            pass
        return None

    def validate_divergence_trend(self, dataframe, prev_pivot, current_pivot, divergence_type):
        """Validate divergence by checking intermediate trend"""
        try:
            # Check if there's a clear trend between pivots
            mid_point = (prev_pivot + current_pivot) // 2
            
            if divergence_type == 'bearish':
                # For bearish divergence, expect uptrend in between
                return dataframe['ema20'][mid_point] > dataframe['ema20'][prev_pivot]
            else:
                # For bullish divergence, expect downtrend in between
                return dataframe['ema20'][mid_point] < dataframe['ema20'][prev_pivot]
        except:
            return True  # Default to accepting divergence if validation fails

    @property
    def protections(self):
        """Enhanced protection configuration"""
        prot = []
       
        if self.use_cooldown_protection.value:
            prot.append({
                "method": "CooldownPeriod",
                "stop_duration_candles": self.cooldown_lookback.value
            })
       
        if self.use_max_drawdown_protection.value:
            prot.append({
                "method": "MaxDrawdown",
                "lookback_period_candles": self.max_drawdown_lookback.value,
                "trade_limit": self.max_drawdown_trade_limit.value,
                "stop_duration_candles": self.max_drawdown_stop_duration.value,
                "max_allowed_drawdown": self.max_allowed_drawdown.value
            })
       
        if self.use_stop_protection.value:
            prot.append({
                "method": "StoplossGuard",
                "lookback_period_candles": self.stoploss_guard_lookback.value,
                "trade_limit": self.stoploss_guard_trade_limit.value,
                "stop_duration_candles": self.stop_duration.value,
                "only_per_pair": self.stoploss_guard_only_per_pair.value,
            })
       
        return prot
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Initialize
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0  
        dataframe['exit_tag'] = ''
        
        # === YOUR OTHER EXIT CONDITIONS FIRST ===
        # (Add any other exit conditions you have here)
        # any_long_exit = (some_other_condition)
        # any_short_exit = (some_other_condition)
        
        # === EXIT ON OPPOSITE SIGNALS (your previous approach) ===
        if 'enter_long' in dataframe.columns and 'enter_short' in dataframe.columns:
            # Exit longs on short signals
            reversal_long_exit = (dataframe['enter_short'] == 1)
            dataframe.loc[reversal_long_exit, 'exit_long'] = 1
            dataframe.loc[reversal_long_exit, 'exit_tag'] = 'Reversal_Short_Signal'
            
            # Exit shorts on long signals  
            if self.can_short:
                reversal_short_exit = (dataframe['enter_long'] == 1)
                dataframe.loc[reversal_short_exit, 'exit_short'] = 1
                dataframe.loc[reversal_short_exit, 'exit_tag'] = 'Reversal_Long_Signal'
        
        return dataframe

    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime',
                    current_rate: float, current_profit: float, **kwargs):
        """
        Simple and effective custom exit - just take profits!
        15m timeframe optimized - triggers exits reliably
        """
        from logging import getLogger
        logger = getLogger(__name__)

        # Get entry tag and trade duration
        entry_tag = getattr(trade, 'enter_tag', '')
        trade_duration_minutes = (current_time - trade.open_date_utc).total_seconds() / 60

        # === GET BASIC MARKET DATA FOR SAFETY EXITS ===
        rsi = 50
        momentum_score = 0
        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if not dataframe.empty:
                last_row = dataframe.iloc[-1]
                if 'rsi' in dataframe.columns:
                    rsi = last_row['rsi']
                
                # Simple momentum calculation
                if len(dataframe) >= 5:
                    momentum_score = (last_row['close'] - dataframe.iloc[-5]['close']) / dataframe.iloc[-5]['close']
        except Exception:
            pass

        # === ENHANCED LOG AND EXIT WITH SIGNAL TRACKING ===
        def log_and_exit(reason: str):
            # UPDATE SIGNAL PERFORMANCE TRACKING
            if hasattr(trade, 'enter_tag') and trade.enter_tag:
                try:
                    profit_ratio = trade.calc_profit_ratio(current_rate)
                    self.update_signal_performance(trade.enter_tag, pair, profit_ratio)
                    logger.debug(f"Updated signal performance: {trade.enter_tag} -> {profit_ratio:.4f}")
                except Exception as e:
                    logger.error(f"Failed to update signal performance: {e}")
            
            # LOG EXIT DETAILS
            logger.info(f"🔤 {pair} EXIT: {reason}")
            logger.info(f"   🕐 Time Held: {int(trade_duration_minutes)} min")
            logger.info(f"   💰 Profit: {current_profit * 100:.2f}%")
            logger.info(f"   📊 RSI: {rsi:.1f}")
            logger.info(f"   📈 Momentum: {momentum_score * 100:.2f}%")
            logger.info(f"   🏷️ Entry Tag: {entry_tag}")
            logger.info(f"   🔄 Exit Trigger: {reason}")
            return reason

        # === SIMPLE PROFIT TAKING - NO COMPLEX LOGIC ===
        # === SIMPLE PROFIT TAKING - NO COMPLEX LOGIC ===
        
        # Immediate exits for big profits
        if current_profit >= 0.12:  # 12%
            return log_and_exit("profit_12pct")
        elif current_profit >= 0.10:  # 10%
            return log_and_exit("profit_10pct")
        elif current_profit >= 0.08:  # 8%
            return log_and_exit("profit_8pct")
        elif current_profit >= 0.06:  # 6%
            return log_and_exit("profit_6pct")
        elif current_profit >= 0.05:  # 5%
            return log_and_exit("profit_5pct")
        elif current_profit >= 0.04:  # 4%
            return log_and_exit("profit_4pct")
        
        # Time-based exits - COVERS ALL TRADES UNDER 4% TOO
        elif current_profit >= 0.035 and trade_duration_minutes >= 15:  # 3.5% after 15min
            return log_and_exit("profit_35pct_15min")
        elif current_profit >= 0.03 and trade_duration_minutes >= 30:   # 3% after 30min
            return log_and_exit("profit_3pct_30min")
        elif current_profit >= 0.025 and trade_duration_minutes >= 45:  # 2.5% after 45min
            return log_and_exit("profit_25pct_45min")
        elif current_profit >= 0.02 and trade_duration_minutes >= 60:   # 2% after 1hr
            return log_and_exit("profit_2pct_1hr")
        elif current_profit >= 0.015 and trade_duration_minutes >= 90:  # 1.5% after 1.5hr
            return log_and_exit("profit_15pct_90min")
        elif current_profit >= 0.01 and trade_duration_minutes >= 120:  # 1% after 2hr
            return log_and_exit("profit_1pct_2hr")
        elif current_profit >= 0.008 and trade_duration_minutes >= 180: # 0.8% after 3hr
            return log_and_exit("profit_08pct_3hr")
        elif current_profit >= 0.005 and trade_duration_minutes >= 240: # 0.5% after 4hr
            return log_and_exit("force_exit_4hr")
        
        # === GET BASIC MARKET DATA FOR SAFETY EXITS ===
        rsi = 50
        momentum_score = 0
        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if not dataframe.empty:
                last_row = dataframe.iloc[-1]
                if 'rsi' in dataframe.columns:
                    rsi = last_row['rsi']
                
                # Simple momentum calculation
                if len(dataframe) >= 5:
                    momentum_score = (last_row['close'] - dataframe.iloc[-5]['close']) / dataframe.iloc[-5]['close']
        except Exception:
            pass

        # === SIMPLE RSI EXTREME EXITS ===
        if current_profit >= 0.02:  # Only if we have some profit
            if not trade.is_short and rsi >= 80:  # Very overbought
                return log_and_exit("rsi_extreme_80")
            elif trade.is_short and rsi <= 20:    # Very oversold
                return log_and_exit("rsi_extreme_20")
        
        # === SIMPLE RSI WARNING EXITS ===
        if current_profit >= 0.015:  # Only if we have decent profit
            if not trade.is_short and rsi >= 75:  # Overbought
                return log_and_exit("rsi_overbought_75")
            elif trade.is_short and rsi <= 25:    # Oversold
                return log_and_exit("rsi_oversold_25")

        # === EMERGENCY EXITS FOR HUGE PROFITS ===
        if current_profit >= 0.30:  # 30% - definitely exit!
            return log_and_exit("emergency_30pct")
        elif current_profit >= 0.25: # 25% - definitely exit!
            return log_and_exit("emergency_25pct")
        elif current_profit >= 0.20: # 20% - definitely exit!
            return log_and_exit("emergency_20pct")
        elif current_profit >= 0.15: # 15% - definitely exit!
            return log_and_exit("emergency_15pct")

        # === REVERSAL SIGNAL EXITS ===
        try:
            if dataframe is not None and not dataframe.empty:
                last_row = dataframe.iloc[-1]
                
                # Exit on opposite signals if we have any profit
                if current_profit >= 0.005:  # 0.5% minimum
                    if not trade.is_short and last_row.get('enter_short', 0) == 1:
                        return log_and_exit("reversal_short_signal")
                    elif trade.is_short and last_row.get('enter_long', 0) == 1:
                        return log_and_exit("reversal_long_signal")
        except Exception:
            pass

        # === WEEKEND SAFETY EXIT ===
        if current_time.weekday() == 4 and current_time.hour >= 20:  # Friday evening
            if current_profit >= 0.01:  # 1% minimum
                return log_and_exit("friday_close")

        # === FINAL SAFETY - EXIT ANYTHING AFTER 6 HOURS ===
        if trade_duration_minutes >= 360:  # 6 hours
            if current_profit >= 0.003:  # Even 0.3% profit
                return log_and_exit("final_safety_6hr")

        return None
    def maybe_optimize_coin(self, pair: str, force_startup: bool = False):
        """ENHANCED optimization trigger with better permanent optimization"""
        # ADDED: Komplette neue Methode für Optimierungslogik
        if not self.optuna_manager:
            logger.debug(f"🚫 [OPTUNA] OptunaManager not available for {pair}")
            return
        
        # ENHANCED: Force startup optimization or intelligent check
        if force_startup:
            logger.info(f"🚀 [OPTUNA] FORCED STARTUP OPTIMIZATION for {pair}")
            should_optimize = True
        else:
            should_optimize = self.optuna_manager.should_optimize(pair)
        
        if not should_optimize:
            return
        
        logger.info(f"🔄 [OPTUNA] Checking optimization conditions for {pair}")
        
        try:
            trades_count = self.get_coin_trades_count(pair)
            optimization_count = self.optuna_manager.optimization_trigger_count.get(pair, 0)
            
            logger.info(f"📊 [OPTUNA] {pair} has {trades_count} trades, {optimization_count} optimizations done")
            
            # ENHANCED: More flexible trade requirements
            min_trades_required = max(0, optimization_count * 5)  # Require more trades for subsequent optimizations
            
            if not force_startup and trades_count < min_trades_required:
                logger.info(f"⏳ [OPTUNA] Not enough trades for {pair} optimization ({trades_count}/{min_trades_required})")
                return
            
            # Track optimization type
            if force_startup:
                opt_type = "STARTUP"
            elif self.optuna_manager.should_optimize_based_on_performance(pair):
                opt_type = "PERFORMANCE-TRIGGERED"
            else:
                opt_type = "PERIODIC"
            
            logger.info(f"✨ [OPTUNA] {opt_type} OPTIMIZATION for {pair} (trades: {trades_count})")
            
            objective_func = self.create_objective_function(pair)
            
            # ENHANCED: More trials for performance-triggered optimizations
            n_trials = 30 if opt_type == "PERFORMANCE-TRIGGERED" else 15
            
            # Start optimization
            self.optuna_manager.optimize_coin(pair, objective_func, n_trials=n_trials)
            
            # Track optimization count
            self.optuna_manager.optimization_trigger_count[pair] = optimization_count + 1
            
            logger.info(f"✅ [OPTUNA] Completed {opt_type} optimization for {pair}")
            
        except Exception as e:
            logger.error(f"❌ [OPTUNA] Optimization failed for {pair}: {e}")
            import traceback
            logger.debug(f"🔍 [OPTUNA] Full traceback: {traceback.format_exc()}")
    
    def get_coin_trades_count(self, pair: str) -> int:
        """Get number of trades for specific coin"""
        # ADDED: Hilfsmethode für Trade-Anzahl pro Coin
        try:
            from freqtrade.persistence import Trade
            trades = Trade.get_trades_proxy(pair=pair)
            count = len(trades) if trades else 0
            logger.debug(f"📊 [OPTUNA] Trade count for {pair}: {count}")
            return count
        except Exception as e:
            logger.error(f"❌ [OPTUNA] Failed to get trade count for {pair}: {e}")
            return 0
    
    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                          rate: float, time_in_force: str, exit_reason: str,
                          current_time: datetime, **kwargs) -> bool:
        """Track trade exits and update performance - works in backtesting"""
        
        try:
            profit_ratio = trade.calc_profit_ratio(rate)
            
            # Update Optuna performance if available
            if self.optuna_manager:
                self.optuna_manager.update_performance(pair, profit_ratio)
                logger.debug(f"📈 [OPTUNA] Updated performance for {pair}: {profit_ratio:.4f}")
            
            if pair not in self.coin_performance:
                self.coin_performance[pair] = 0.0
            self.coin_performance[pair] += profit_ratio
            
            # ADD THIS: Update signal performance tracking for backtesting
            enter_tag = getattr(trade, 'enter_tag', None)
            if enter_tag:
                self.update_signal_performance(enter_tag, pair, profit_ratio)
                
                # Debug output
                result_type = "WIN" if profit_ratio > 0 else "LOSS"
                print(f"📊 Signal Performance Tracked: {pair} {enter_tag} - {result_type}")
                print(f"   Profit: {profit_ratio:.4f} ({profit_ratio*100:.2f}%)")
                print(f"   Exit reason: {exit_reason}")
            
            # Log significant performance updates
            if abs(profit_ratio) > 0.02:  # More than 2% gain/loss
                logger.info(f"💰 [OPTUNA] Significant trade result for {pair}: {profit_ratio:.2%} (cumulative: {self.coin_performance[pair]:.2%})")
            
        except Exception as e:
            logger.error(f"⛔ [OPTUNA] Failed to update performance for {pair}: {e}")
        
        return True

    def on_trade_close(self, trade: Trade, **kwargs) -> None:
        """Called when a trade is closed - captures ALL exits including stoploss"""
        try:
            entry_tag = getattr(trade, 'enter_tag', '')
            if entry_tag:  # Only track if we have an entry tag
                pair = trade.pair
                profit_ratio = trade.calc_profit_ratio()
                
                # Update signal performance for ALL exits (wins and losses)
                self.update_signal_performance(entry_tag, pair, profit_ratio)
                
                # Log the trade result
                result_type = "WIN" if profit_ratio > 0 else "LOSS"
                exit_reason = getattr(trade, 'exit_reason', 'unknown')
                
                print(f"Trade closed: {pair} {entry_tag} - {result_type}")
                print(f"  Profit: {profit_ratio:.4f} ({profit_ratio*100:.2f}%)")
                print(f"  Exit reason: {exit_reason}")
                
        except Exception as e:
            print(f"Error in on_trade_close: {e}")
def choppiness_index(high, low, close, window=14):
    """Calculate Choppiness Index"""
    natr = pd.Series(ta.NATR(high, low, close, window=window))
    high_max = high.rolling(window=window).max()
    low_min = low.rolling(window=window).min()
    
    choppiness = 100 * np.log10((natr.rolling(window=window).sum()) / (high_max - low_min)) / np.log10(window)
    return choppiness

def resample(indicator):
    """Resample function for compatibility"""
    return indicator

def two_bands_check_long(dataframe):
    """Allow long when price is near/at lower band (oversold area)"""
    return (
        (dataframe['low'] <= dataframe['kc_lowerband']) |
        (dataframe['close'] <= dataframe['kc_lowerband'])
    )

def two_bands_check_short(dataframe):
    """Allow short when price is near/at upper band (overbought area)"""
    return (
        (dataframe['high'] >= dataframe['kc_upperband']) |
        (dataframe['close'] >= dataframe['kc_upperband'])
    )
    
def green_candle(dataframe):
    """Check for green candle"""
    return dataframe[resample('open')] < dataframe[resample('close')]

def red_candle(dataframe):
    """Check for red candle"""
    return dataframe[resample('open')] > dataframe[resample('close')]

def pivot_points(dataframe: DataFrame, window: int = 5, pivot_source=None) -> DataFrame:
    """
    Williams Fractal / Delayed Pivot Implementation
    Detects pivots by checking if a candle 'window' periods ago was the max/min of the surrounding range.
    Signal is generated at the current candle (T), confirming the pivot at T-window.
    This is causal and has NO lookahead bias, but introduces a lag of 'window' periods.
    """
    df = dataframe.copy()
    
    # Define rolling window size: window * 2 + 1
    # We want to check if the candle at T-window is the max of [T-2*window, T]
    roll_window = window * 2 + 1
    
    # Calculate rolling max/min
    # Note: We use the default 'right' alignment (not center=True in the rolling call itself)
    # But effectively we are checking a centered window relative to the shifted candle.
    max_roll = df['high'].rolling(window=roll_window).max()
    min_roll = df['low'].rolling(window=roll_window).min()
    
    # Check if the candle at T-window equals the max of the rolling window ending at T
    # shift(window) aligns the candle at T-window with the current rolling result
    pivot_high_signal = df['high'].shift(window) == max_roll
    pivot_low_signal = df['low'].shift(window) == min_roll
    
    # Assign the pivot value at the confirmation time (T)
    # We use the value of the pivot (which is at T-window)
    pivot_highs = np.where(pivot_high_signal, df['high'].shift(window), np.nan)
    pivot_lows = np.where(pivot_low_signal, df['low'].shift(window), np.nan)
    
    return pd.DataFrame(index=dataframe.index, data={
        'pivot_lows': pivot_lows,
        'pivot_highs': pivot_highs
    })

def check_if_pivot_is_greater_or_less(current_value, high_source: str, low_source: str, left, right) -> Tuple[bool, bool]:
    """Helper function for pivot point validation"""
    is_greater = True
    is_less = True
    
    if (getattr(current_value, high_source) <= getattr(left, high_source) or
            getattr(current_value, high_source) <= getattr(right, high_source)):
        is_greater = False

    if (getattr(current_value, low_source) >= getattr(left, low_source) or
            getattr(current_value, low_source) >= getattr(right, low_source)):
        is_less = False
    
    return (is_greater, is_less)

def emaKeltner(dataframe):
    """Calculate EMA-based Keltner Channels"""
    keltner = {}
    atr = qtpylib.atr(dataframe, window=10)
    ema20 = ta.EMA(dataframe, timeperiod=20)
    keltner['upper'] = ema20 + atr
    keltner['mid'] = ema20
    keltner['lower'] = ema20 - atr
    return keltner

def chaikin_money_flow(dataframe, n=20, fillna=False) -> Series:
    """Calculate Chaikin Money Flow indicator"""
    df = dataframe.copy()
    mfv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
    mfv = mfv.fillna(0.0)
    mfv *= df['volume']
    cmf = (mfv.rolling(n, min_periods=0).sum() / df['volume'].rolling(n, min_periods=0).sum())
    if fillna:
        cmf = cmf.replace([np.inf, -np.inf], np.nan).fillna(0)
    return Series(cmf, name='cmf')