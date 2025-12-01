"""
SMC Simple FVG Trailing Strategy v2 - Enhanced with Frequency Improvements

Enhanced SMC implementation based on SMCSimpleFVGTrailing:
- Entry: CHoCH/BOS + FVG/OB retracement + relaxed filters
- Exit: Opposite CHoCH/BOS OR ROI with better risk management
- Added: Basic trend confirmation, volume filter, relaxed conditions
- Goal: Increase trade frequency from ~65/year to 15-30/month while maintaining quality

Base logic from SMCSimpleFVGTrailing with incremental improvements.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime
from pandas import DataFrame
from typing import Optional
import sys
import os

# Add smart-money-concepts to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'smart-money-concepts'))

logger = logging.getLogger(__name__)

from freqtrade.strategy import IStrategy
from freqtrade.exchange import timeframe_to_minutes
from freqtrade.data.dataprovider import DataProvider
from typing import Dict

# Import TA indicators for additional filtering
import talib.abstract as ta

# Import SMC library
import smartmoneyconcepts as smc
logger.info("smartmoneyconcepts library loaded successfully")


class SMCSimpleFVGTrailing_v2(IStrategy):
    """
    Enhanced SMC Simple FVG Trailing Strategy v2

    Based on SMCSimpleFVGTrailing with improvements:
    - Slightly relaxed entry conditions
    - Basic trend confirmation
    - Volume filter for quality
    - Better risk management
    - Target: 15-30 trades/month while keeping 60%+ win rate
    """

    INTERFACE_VERSION = 3

    # === IMPROVED PARAMETERS ===

    # ROI Table - More reasonable targets
    minimal_roi = {
        "0": 0.06,   # 6% target (reduced from 100%)
        "20": 0.04,  # Reduce to 4% after 20 min
        "40": 0.03,  # Reduce to 3% after 40 min
        "60": 0.02   # Reduce to 2% after 60 min
    }

    # Stop Loss - Reduced from 20% to 12% for better risk management
    stoploss = -0.12  # 12% = 1.2% with 10x leverage

    # Timeframe
    timeframe = '15m'

    # Trailing Stop - Earlier activation for better profits
    trailing_stop = True
    trailing_stop_positive = 0.02         # Start trailing at 2% (reduced from 34.3%)
    trailing_stop_positive_offset = 0.04    # 2% margin (total 4%)
    trailing_only_offset_is_reached = True

    # Order Types
    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'limit',
        'stoploss_on_exchange': False,
        'stoploss_on_exchange_interval': 60,
    }

    # Order Time in Force
    order_time_in_force = {
        'entry': 'GTC',
        'exit': 'GTC'
    }

    # Entry/Exit Settings
    use_exit_signal = True
    use_custom_exit = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    ignore_roi_if_exit_signal = False
    use_custom_stoploss = False  # Simplified - use standard stop loss

    # Position Settings
    position_adjustment_enable = True
    max_entry_position_adjustment = 1
    max_open_trades = 1

    # Startup Candle Count
    startup_candle_count: int = 60

    # === ENHANCED SMC PARAMETERS ===
    swing_length = 23  # Original value for good structure detection
    strength_sma_period = 20  # For trend strength calculation

    # === NEW FILTERING PARAMETERS ===

    # Trend filter for basic confirmation
    ema_fast = 20
    ema_slow = 50

    # Volume filter
    volume_period = 20
    volume_multiplier = 1.1  # Just above average volume

    # RSI filter (relaxed)
    rsi_period = 14
    rsi_lower = 30  # Lowered from 40
    rsi_upper = 70  # Raised from 60

    # === RETRACEMENT PARAMETERS (from original) ===
    fvg_touch_mode = "wick_full_zone"
    ob_touch_mode = "wick_full_zone"

    # === ENTRY MODE PARAMETERS (NEW) ===
    entry_mode = "hybrid"  # "strict_smc" or "hybrid"
    # hybrid = original SMC + additional relaxed conditions

    # Relaxed SMC parameters for hybrid mode
    relaxed_structure_lookback = 8  # Increased from 5
    relaxed_retracement_mode = True  # Allow partial zone touches

    def informative_pairs(self):
        """No additional informative pairs needed."""
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Calculate SMC indicators + basic technical filters.
        """
        logger.info(f"Calculating Enhanced SMC v2 indicators for {metadata['pair']}")

        # === ADDITIONAL TECHNICAL INDICATORS ===
        # Trend confirmation
        dataframe['ema_fast'] = ta.EMA(dataframe, timeperiod=self.ema_fast)
        dataframe['ema_slow'] = ta.EMA(dataframe, timeperiod=self.ema_slow)

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=self.rsi_period)

        # Volume
        dataframe['volume_avg'] = dataframe['volume'].rolling(window=self.volume_period).mean()
        dataframe['volume_ratio'] = dataframe['volume'] / dataframe['volume_avg']

        # === SMC INDICATORS (from original) ===
        return self._populate_indicators_smc_library(dataframe, metadata)

    def _populate_indicators_smc_library(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Calculate indicators using smartmoneyconcepts library (from original).
        """
        # Prepare OHLCV data for SMC library (expects lowercase columns + volume)
        ohlc = dataframe[['open', 'high', 'low', 'close', 'volume']].copy()

        # === 1. SWING HIGHS & LOWS (True Pivots) ===
        swing_data = smc.smc.swing_highs_lows(ohlc, swing_length=self.swing_length)

        # === 2. BOS & CHoCH ===
        bos_choch_data = smc.smc.bos_choch(ohlc, swing_data, close_break=True)

        # === 3. FAIR VALUE GAPS (FVG) ===
        fvg_data = smc.smc.fvg(ohlc, join_consecutive=False)

        # === 4. ORDER BLOCKS (OB) ===
        ob_data = smc.smc.ob(ohlc, swing_data, close_mitigation=False)

        # === 5. TREND STRENGTH (Optional) ===
        dataframe['trend_strength'] = (
            (dataframe['close'] / dataframe['close'].rolling(self.strength_sma_period).mean() - 1).abs()
        )

        # === CONVERT SMC DATA TO STRATEGY FORMAT ===

        # BOS signals
        dataframe['bos_bullish'] = (bos_choch_data['BOS'] == 1).astype(int)
        dataframe['bos_bearish'] = (bos_choch_data['BOS'] == -1).astype(int)

        # CHoCH signals
        dataframe['choch_bullish'] = (bos_choch_data['CHOCH'] == 1).astype(int)
        dataframe['choch_bearish'] = (bos_choch_data['CHOCH'] == -1).astype(int)

        # FVG signals
        dataframe['fvg_bullish'] = (fvg_data['FVG'] == 1).astype(int)
        dataframe['fvg_bearish'] = (fvg_data['FVG'] == -1).astype(int)

        # FVG levels
        dataframe['fvg_top_bullish'] = np.where(dataframe['fvg_bullish'] == 1, fvg_data['Top'], 0)
        dataframe['fvg_bottom_bullish'] = np.where(dataframe['fvg_bullish'] == 1, fvg_data['Bottom'], 0)
        dataframe['fvg_top_bearish'] = np.where(dataframe['fvg_bearish'] == 1, fvg_data['Top'], 0)
        dataframe['fvg_bottom_bearish'] = np.where(dataframe['fvg_bearish'] == 1, fvg_data['Bottom'], 0)

        # Order Block signals
        dataframe['ob_bullish'] = (ob_data['OB'] == 1).astype(int)
        dataframe['ob_bearish'] = (ob_data['OB'] == -1).astype(int)

        # Order Block levels
        dataframe['ob_top_bullish'] = np.where(dataframe['ob_bullish'] == 1, ob_data['Top'], 0)
        dataframe['ob_bottom_bullish'] = np.where(dataframe['ob_bullish'] == 1, ob_data['Bottom'], 0)
        dataframe['ob_top_bearish'] = np.where(dataframe['ob_bearish'] == 1, ob_data['Top'], 0)
        dataframe['ob_bottom_bearish'] = np.where(dataframe['ob_bearish'] == 1, ob_data['Bottom'], 0)

        # === ZONE MEMORY (from original) ===
        self._apply_zone_memory(dataframe)

        logger.info(f"Enhanced SMC v2 indicators calculated successfully for {metadata['pair']}")
        return dataframe

    def _apply_zone_memory(self, dataframe: DataFrame):
        """
        Apply zone memory using ffill() for performance (from original).
        """
        # FVG zones - Create Series first, then use ffill()
        fvg_bullish_top_series = pd.Series(np.where(
            dataframe['fvg_bullish'] == 1, dataframe['fvg_top_bullish'], np.nan
        ), index=dataframe.index)
        dataframe['recent_fvg_bullish_top'] = fvg_bullish_top_series.ffill().fillna(0)

        fvg_bullish_bottom_series = pd.Series(np.where(
            dataframe['fvg_bullish'] == 1, dataframe['fvg_bottom_bullish'], np.nan
        ), index=dataframe.index)
        dataframe['recent_fvg_bullish_bottom'] = fvg_bullish_bottom_series.ffill().fillna(0)

        fvg_bearish_top_series = pd.Series(np.where(
            dataframe['fvg_bearish'] == 1, dataframe['fvg_top_bearish'], np.nan
        ), index=dataframe.index)
        dataframe['recent_fvg_bearish_top'] = fvg_bearish_top_series.ffill().fillna(0)

        fvg_bearish_bottom_series = pd.Series(np.where(
            dataframe['fvg_bearish'] == 1, dataframe['fvg_bottom_bearish'], np.nan
        ), index=dataframe.index)
        dataframe['recent_fvg_bearish_bottom'] = fvg_bearish_bottom_series.ffill().fillna(0)

        # Order Block zones
        ob_bullish_top_series = pd.Series(np.where(
            dataframe['ob_bullish'] == 1, dataframe['ob_top_bullish'], np.nan
        ), index=dataframe.index)
        dataframe['recent_ob_bullish_top'] = ob_bullish_top_series.ffill().fillna(0)

        ob_bullish_bottom_series = pd.Series(np.where(
            dataframe['ob_bullish'] == 1, dataframe['ob_bottom_bullish'], np.nan
        ), index=dataframe.index)
        dataframe['recent_ob_bullish_bottom'] = ob_bullish_bottom_series.ffill().fillna(0)

        ob_bearish_top_series = pd.Series(np.where(
            dataframe['ob_bearish'] == 1, dataframe['ob_top_bearish'], np.nan
        ), index=dataframe.index)
        dataframe['recent_ob_bearish_top'] = ob_bearish_top_series.ffill().fillna(0)

        ob_bearish_bottom_series = pd.Series(np.where(
            dataframe['ob_bearish'] == 1, dataframe['ob_bottom_bearish'], np.nan
        ), index=dataframe.index)
        dataframe['recent_ob_bearish_bottom'] = ob_bearish_bottom_series.ffill().fillna(0)

    def check_retracement_criteria(self, dataframe: DataFrame, zone_type: str, direction: str) -> DataFrame:
        """
        Check if price retraces to FVG or Order Block zones (from original with modifications).
        """
        if zone_type == 'fvg':
            if direction == 'bullish':
                top_col = 'recent_fvg_bullish_top'
                bottom_col = 'recent_fvg_bullish_bottom'
            else:
                top_col = 'recent_fvg_bearish_top'
                bottom_col = 'recent_fvg_bearish_bottom'
        elif zone_type == 'ob':
            if direction == 'bullish':
                top_col = 'recent_ob_bullish_top'
                bottom_col = 'recent_ob_bullish_bottom'
            else:
                top_col = 'recent_ob_bearish_top'
                bottom_col = 'recent_ob_bearish_bottom'
        else:
            return pd.Series(False, index=dataframe.index)

        # Get parameters for this zone type
        if zone_type == 'fvg':
            touch_mode = self.fvg_touch_mode
        else:
            touch_mode = self.ob_touch_mode

        # Calculate 50% level of the zone
        zone_mid = (dataframe[top_col] + dataframe[bottom_col]) / 2

        # Check different retracement criteria
        if touch_mode == "full_zone":
            # Any part of the candle body touches the zone
            retracement = (
                (dataframe[top_col] > 0) &
                ((dataframe['close'] >= dataframe[bottom_col]) &
                 (dataframe['close'] <= dataframe[top_col]) |
                 (dataframe['open'] >= dataframe[bottom_col]) &
                 (dataframe['open'] <= dataframe[top_col]))
            )

        elif touch_mode == "body_50_percent":
            # Candle body touches the 50% level of the zone
            retracement = (
                (dataframe[top_col] > 0) &
                ((dataframe['close'] >= zone_mid) & (dataframe['open'] <= zone_mid) |
                 (dataframe['open'] >= zone_mid) & (dataframe['close'] <= zone_mid))
            )

        elif touch_mode == "wick_50_percent":
            # Candle wick touches the 50% level of the zone
            if direction == 'bullish':
                retracement = (
                    (dataframe[top_col] > 0) &
                    (dataframe['low'] <= zone_mid) &
                    (dataframe['high'] >= zone_mid)
                )
            else:
                retracement = (
                    (dataframe[top_col] > 0) &
                    (dataframe['high'] >= zone_mid) &
                    (dataframe['low'] <= zone_mid)
                )

        elif touch_mode == "wick_full_zone":
            # Candle wick touches any part of the zone
            if direction == 'bullish':
                retracement = (
                    (dataframe[top_col] > 0) &
                    (dataframe['low'] <= dataframe[top_col]) &
                    (dataframe['high'] >= dataframe[bottom_col])
                )
            else:
                retracement = (
                    (dataframe[top_col] > 0) &
                    (dataframe['high'] >= dataframe[bottom_col]) &
                    (dataframe['low'] <= dataframe[top_col])
                )
        else:
            retracement = pd.Series(False, index=dataframe.index)

        return retracement

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Generate enhanced entry signals with hybrid mode.

        Logic:
        - Mode 1: Strict SMC (original logic)
        - Mode 2: Hybrid (original + relaxed conditions)
        """
        logger.info(f"Generating Enhanced CHoCH + BOS + FVG/OB retracement signals for {metadata['pair']}")
        logger.info(f"Entry mode: {self.entry_mode}")
        logger.info(f"FVG touch mode: {self.fvg_touch_mode}, OB touch mode: {self.ob_touch_mode}")

        dataframe.loc[:, 'enter_long'] = 0
        dataframe.loc[:, 'enter_short'] = 0

        # === BASIC FILTERS ===
        # Trend confirmation
        uptrend = dataframe['ema_fast'] > dataframe['ema_slow']
        downtrend = dataframe['ema_fast'] < dataframe['ema_slow']

        # RSI filter
        rsi_ok_long = (dataframe['rsi'] >= self.rsi_lower) & (dataframe['rsi'] <= 80)
        rsi_ok_short = (dataframe['rsi'] >= 20) & (dataframe['rsi'] <= self.rsi_upper)

        # Volume filter
        volume_ok = dataframe['volume_ratio'] >= self.volume_multiplier

        # === ORIGINAL STRICT SMC LOGIC ===
        # Check if price retraces to FVG or Order Block zones using specific criteria

        # LONG RETRACEMENT: Price retraces to bullish FVG or bullish OB zone
        long_in_fvg = self.check_retracement_criteria(dataframe, 'fvg', 'bullish')
        long_in_ob = self.check_retracement_criteria(dataframe, 'ob', 'bullish')

        # SHORT RETRACEMENT: Price retraces to bearish FVG or bearish OB zone
        short_in_fvg = self.check_retracement_criteria(dataframe, 'fvg', 'bearish')
        short_in_ob = self.check_retracement_criteria(dataframe, 'ob', 'bearish')

        # === STRUCTURE BREAK CONDITIONS (Original) ===
        recent_structure_break_long = (
            ((dataframe['choch_bullish'].rolling(5).max() == 1) |
             (dataframe['bos_bullish'].rolling(5).max() == 1))
        )

        recent_structure_break_short = (
            ((dataframe['choch_bearish'].rolling(5).max() == 1) |
             (dataframe['bos_bearish'].rolling(5).max() == 1))
        )

        # === ORIGINAL STRICT ENTRY CONDITIONS ===
        # Structure break AND retracement to FVG/OB zone (Pure SMC)
        strict_buy_conditions = (
            recent_structure_break_long &               # Structure break happened
            (long_in_fvg | long_in_ob)                 # Price retraced to FVG or OB
        )

        strict_sell_conditions = (
            recent_structure_break_short &              # Structure break happened
            (short_in_fvg | short_in_ob)               # Price retraced to FVG or OB
        )

        # === RELAXED HYBRID CONDITIONS ===
        # Longer lookback for structure breaks
        relaxed_structure_break_long = (
            ((dataframe['choch_bullish'].rolling(self.relaxed_structure_lookback).max() == 1) |
             (dataframe['bos_bullish'].rolling(self.relaxed_structure_lookback).max() == 1))
        )

        relaxed_structure_break_short = (
            ((dataframe['choch_bearish'].rolling(self.relaxed_structure_lookback).max() == 1) |
             (dataframe['bos_bearish'].rolling(self.relaxed_structure_lookback).max() == 1))
        )

        # Relaxed entry with basic filters
        relaxed_buy_conditions = (
            uptrend &
            volume_ok &
            rsi_ok_long &
            relaxed_structure_break_long &
            (long_in_fvg | long_in_ob)
        )

        relaxed_sell_conditions = (
            downtrend &
            volume_ok &
            rsi_ok_short &
            relaxed_structure_break_short &
            (short_in_fvg | short_in_ob)
        )

        # === FINAL ENTRY CONDITIONS ===
        if self.entry_mode == "strict_smc":
            # Use original strict logic only
            buy_conditions = strict_buy_conditions
            sell_conditions = strict_sell_conditions
        else:  # hybrid mode
            # Combine strict and relaxed conditions
            buy_conditions = strict_buy_conditions | relaxed_buy_conditions
            sell_conditions = strict_sell_conditions | relaxed_sell_conditions

        # Set entry signals
        dataframe.loc[buy_conditions, 'enter_long'] = 1
        dataframe.loc[sell_conditions, 'enter_short'] = 1

        # Count and log signals
        buy_signals = dataframe['enter_long'].sum()
        sell_signals = dataframe['enter_short'].sum()

        # Count breakdown
        buy_in_fvg_count = long_in_fvg.sum()
        buy_in_ob_count = long_in_ob.sum()
        sell_in_fvg_count = short_in_fvg.sum()
        sell_in_ob_count = short_in_ob.sum()

        # Structure breakdown
        buy_choch = dataframe['choch_bullish'].sum()
        buy_bos = dataframe['bos_bullish'].sum()
        sell_choch = dataframe['choch_bearish'].sum()
        sell_bos = dataframe['bos_bearish'].sum()

        logger.info(f"Enhanced CHoCH + BOS + FVG/OB retracement entry signals generated for {metadata['pair']}")
        logger.info(f"  Buy: {buy_signals} (BOS: {buy_bos}, CHoCH: {buy_choch})")
        logger.info(f"    - In FVG: {buy_in_fvg_count}, In OB: {buy_in_ob_count}")
        logger.info(f"  Sell: {sell_signals} (BOS: {sell_bos}, CHoCH: {sell_choch})")
        logger.info(f"    - In FVG: {sell_in_fvg_count}, In OB: {sell_in_ob_count}")

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Generate exit signals: Opposite CHoCH/BOS OR ROI (from original).
        """
        # === OPPOSITE STRUCTURE EXIT CONDITIONS ===

        # Exit LONG on Bearish structure change (CHoCH or strong downtrend)
        exit_long_conditions = (
            (dataframe['choch_bearish'] == 1) |  # Bearish CHoCH
            (dataframe['bos_bearish'] == 1)      # Bearish BOS
        )

        # Exit SHORT on Bullish structure change (CHoCH or strong uptrend)
        exit_short_conditions = (
            (dataframe['choch_bullish'] == 1) |  # Bullish CHoCH
            (dataframe['bos_bullish'] == 1)      # Bullish BOS
        )

        dataframe.loc[exit_long_conditions, 'exit_long'] = 1
        dataframe.loc[exit_short_conditions, 'exit_short'] = 1

        return dataframe

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: str,
                 side: str, **kwargs) -> float:
        """
        Set leverage to 10x for all trades (from original).
        """
        return 10.0

    def custom_exit(self, pair: str, trade: 'Trade', current_time: datetime,
                   current_rate: float, current_profit: float, **kwargs) -> bool:
        """
        Simplified custom exit logic (from original).
        """
        # Track maximum profit achieved
        if not hasattr(trade, 'max_profit_reached'):
            trade.max_profit_reached = current_profit
        else:
            trade.max_profit_reached = max(trade.max_profit_reached, current_profit)

        # Log significant profit milestones for analysis
        if current_profit >= 0.01 and not hasattr(trade, 'profit_1_reached'):
            trade.profit_1_reached = True
            logger.info(f"PROFIT TRACKING - {pair} reached +1%: {current_profit:.3f}")

        if current_profit >= 0.05 and not hasattr(trade, 'profit_5_reached'):
            trade.profit_5_reached = True
            logger.info(f"PROFIT TRACKING - {pair} reached +5%: {current_profit:.3f}")

        if current_profit >= 0.10 and not hasattr(trade, 'profit_10_reached'):
            trade.profit_10_reached = True
            logger.info(f"PROFIT TRACKING - {pair} reached +10%: {current_profit:.3f}")

        # NO EXIT - let it run to opposite signal or stop loss
        return False

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float,
                           rate: float, time_in_force: str, current_time: datetime,
                           entry_tag: str, side: str, **kwargs) -> bool:
        """
        Enhanced SMC entry confirmation - log CHoCH/BOS signals with mode info.
        """
        if side == 'long':
            logger.info(f"🔧 ENHANCED SMC LONG entry confirmed for {pair} at {rate} (mode: {self.entry_mode})")
        else:
            logger.info(f"🔧 ENHANCED SMC SHORT entry confirmed for {pair} at {rate} (mode: {self.entry_mode})")

        return True

    def confirm_trade_exit(self, pair: str, trade: 'Trade', order_type: str, amount: float,
                           rate: float, time_in_force: str, exit_reason: str,
                           current_time: datetime, **kwargs) -> bool:
        """
        Enhanced exit logging with maximum profit analysis.
        """
        final_profit = trade.calc_profit_ratio(rate)

        # Log maximum profit achieved
        if hasattr(trade, 'max_profit_reached'):
            max_profit = trade.max_profit_reached
            logger.info(f"🔧 EXIT: {pair} - Reason: {exit_reason}")
            logger.info(f"  Final Profit: {final_profit:.3f}")
            logger.info(f"  Maximum Profit Reached: {max_profit:.3f}")
            logger.info(f"  Profit Degradation: {((max_profit - final_profit) / max_profit * 100):.1f}%")
        else:
            logger.info(f"🔧 EXIT: {pair} - Reason: {exit_reason}, Profit: {final_profit:.3f}")

        return True