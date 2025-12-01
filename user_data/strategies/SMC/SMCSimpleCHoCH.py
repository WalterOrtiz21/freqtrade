"""
SMC Simple CHoCH Strategy - Smart Money Concepts Library

Professional SMC implementation using smartmoneyconcepts library:
- Entry: CHoCH OR BOS + FVG/OB retracement
- Exit: Opposite CHoCH/BOS OR ROI
- True pivots detection (configurable swing length)
- Fair Value Gaps (FVG) and Order Blocks (OB)
- Retracement-based entries

Using smartmoneyconcepts library for professional SMC detection.
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

# Import SMC library
import smartmoneyconcepts as smc
logger.info("smartmoneyconcepts library loaded successfully")


class SMCSimpleCHoCH(IStrategy):
    """
    SMC Simple CHoCH Strategy - Smart Money Concepts Library

    Professional SMC implementation using smartmoneyconcepts library:
    - True pivot detection (configurable swing length)
    - CHoCH and BOS detection from library
    - FVG and Order Block detection
    - Retracement-based entries
    """

    INTERFACE_VERSION = 3

    # === FIXED PARAMETERS ===

    # ROI Table - VERY HIGH to see maximum potential (Exit on opposite SMC signal or stop loss)
    minimal_roi = {
        "0": 50.0   # Extremely high to avoid ROI exits completely
    }

    # Stop Loss - 20% = 2% with 10x leverage
    stoploss = -0.20

    # Timeframe
    timeframe = '15m'

    # Trailing Stop - ACTIVADO: Inicia al 5% con 2% margen
    trailing_stop = True
    trailing_stop_positive = 0.05          # Activar trailing al 5% de profit
    trailing_stop_positive_offset = 0.07   # Offset hasta 7% (5% + 2% margen)
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
    use_exit_signal = True  # REACTIVADO - Salir por señal SMC opuesta o trailing stop
    use_custom_exit = True  # Enable custom exit logic for partial profits
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    ignore_roi_if_exit_signal = False  # Take opposite signals even if profitable
    use_custom_stoploss = False  # Disabled - using fixed stoploss to avoid "trailing_stop_loss" exit reason confusion

    # Position Adjustment (for partial exits)
    position_adjustment_enable = True
    max_entry_position_adjustment = 1  # Allow scaling in (not needed but required)
    max_open_trades = 1

    # Startup Candle Count
    startup_candle_count: int = 50

    # === SMC LIBRARY PARAMETERS ===
    # Pivot detection (traditional SMC method)
    swing_length = 23         # Number of bars to look left and right for swing highs/lows

    # Trend detection periods
    strength_sma_period = 20 # SMA for trend strength (optional)

    # === RETRACEMENT PARAMETERS ===
    # FVG retracement criteria
    fvg_touch_mode = "wick_full_zone"  # Options: "full_zone", "body_50_percent", "wick_50_percent", "wick_full_zone"

    # OB retracement criteria
    ob_touch_mode = "wick_full_zone"    # Options: "full_zone", "body_50_percent", "wick_50_percent", "wick_full_zone"

    # Additional confirmation
    require_confirmation_candle = False  # Wait for next candle to confirm direction

    # === STRATEGY LOGIC ===

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Calculate SMC indicators using smartmoneyconcepts library.
        """
        logger.info(f"Calculating SMC indicators for {metadata['pair']}")
        logger.info("Using smartmoneyconcepts library")
        return self._populate_indicators_smc_library(dataframe, metadata)

    def _populate_indicators_smc_library(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Calculate indicators using smartmoneyconcepts library.
        """
        # Prepare OHLCV data for SMC library (expects lowercase columns + volume)
        ohlc = dataframe[['open', 'high', 'low', 'close', 'volume']].copy()

        # === 1. SWING HIGHS & LOWS (True Pivots) ===
        # Using configurable swing_length parameter
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

        # === ZONE MEMORY (Track recent zones) ===
        self._apply_zone_memory(dataframe)

        logger.info(f"SMC library indicators calculated successfully for {metadata['pair']}")
        return dataframe

    def _apply_zone_memory(self, dataframe: DataFrame):
        """
        Apply zone memory using ffill() for performance.
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
        Check if price retraces to FVG or Order Block zones based on criteria.

        Args:
            dataframe: DataFrame with all indicators
            zone_type: 'fvg' or 'ob'
            direction: 'bullish' or 'bearish'

        Returns:
            Boolean Series indicating valid retracement
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
        Generate entry signals based on CHoCH/BOS + FVG/OB retracement.

        Logic:
        1. CHoCH or BOS occurs (structure break)
        2. Wait for price to return to FVG or OB zone (retracement)
        3. Enter on retracement (better entry price)
        """
        logger.info(f"Generating CHoCH + BOS + FVG/OB retracement signals for {metadata['pair']}")
        logger.info(f"FVG touch mode: {self.fvg_touch_mode}, OB touch mode: {self.ob_touch_mode}")

        # === RETRACEMENT CONDITIONS ===
        # Check if price retraces to FVG or Order Block zones using specific criteria

        # LONG RETRACEMENT: Price retraces to bullish FVG or bullish OB zone
        long_in_fvg = self.check_retracement_criteria(dataframe, 'fvg', 'bullish')
        long_in_ob = self.check_retracement_criteria(dataframe, 'ob', 'bullish')

        # SHORT RETRACEMENT: Price retraces to bearish FVG or bearish OB zone
        short_in_fvg = self.check_retracement_criteria(dataframe, 'fvg', 'bearish')
        short_in_ob = self.check_retracement_criteria(dataframe, 'ob', 'bearish')

        # === STRUCTURE BREAK CONDITIONS ===
        # CHoCH or BOS must have occurred recently (within last 5 candles)
        recent_structure_break_long = (
            ((dataframe['choch_bullish'].rolling(5).max() == 1) |
             (dataframe['bos_bullish'].rolling(5).max() == 1))
        )

        recent_structure_break_short = (
            ((dataframe['choch_bearish'].rolling(5).max() == 1) |
             (dataframe['bos_bearish'].rolling(5).max() == 1))
        )

        # === FINAL ENTRY CONDITIONS ===
        # Structure break AND retracement to FVG/OB zone (Pure SMC)

        # LONG ENTRY: (CHoCH OR BOS occurred) AND (price in FVG OR OB)
        buy_conditions = (
            recent_structure_break_long &               # Structure break happened
            (long_in_fvg | long_in_ob)                 # Price retraced to FVG or OB
        )

        # SHORT ENTRY: (CHoCH OR BOS occurred) AND (price in FVG OR OB)
        sell_conditions = (
            recent_structure_break_short &              # Structure break happened
            (short_in_fvg | short_in_ob)               # Price retraced to FVG or OB
        )

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

        logger.info(f"CHoCH + BOS + FVG/OB retracement entry signals generated for {metadata['pair']}")
        logger.info(f"  Buy: {buy_signals} (BOS: {buy_bos}, CHoCH: {buy_choch})")
        logger.info(f"    - In FVG: {buy_in_fvg_count}, In OB: {buy_in_ob_count}")
        logger.info(f"  Sell: {sell_signals} (BOS: {sell_bos}, CHoCH: {sell_choch})")
        logger.info(f"    - In FVG: {sell_in_fvg_count}, In OB: {sell_in_ob_count}")

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Generate exit signals: Opposite CHoCH/BOS OR ROI hit.
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
        Set leverage to 10x for all trades.
        """
        return 10.0

    def informative_pairs(self):
        """
        No additional informative pairs needed.
        """
        return []

    # Custom stoploss disabled - using fixed stoploss instead
    # This avoids the confusion where exits are reported as "trailing_stop_loss"
    # even when using a fixed stoploss value
    # def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
    #                     current_rate: float, current_profit: float, **kwargs) -> float:
    #     """
    #     Custom stoploss logic with full exit at +15%:
    #     - Initial SL: -25%
    #     - At +15% profit: Complete exit via custom_exit (no trailing stops)
    #     - No partial exits, no trailing stops complications
    #     """
    #     # Since we take full exit at +15%, no need for complex BE logic
    #     # Just use initial stop loss throughout
    #     return -0.25  # -25% stop loss

    def custom_exit(self, pair: str, trade: 'Trade', current_time: datetime,
                   current_rate: float, current_profit: float, **kwargs) -> bool:
        """
        Track maximum profit potential - NO EXITS (let it run to opposite signal or stop loss).
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
        return False  # Don't exit normally

    # Position adjustment no longer needed - using full exits at +15%

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float,
                           rate: float, time_in_force: str, current_time: datetime,
                           entry_tag: str, side: str, **kwargs) -> bool:
        """
        SMC entry confirmation - log CHoCH/BOS signals.
        """
        if side == 'long':
            logger.info(f"SMC LONG entry confirmed for {pair} at {rate}")
        else:
            logger.info(f"SMC SHORT entry confirmed for {pair} at {rate}")

        return True

    def confirm_trade_exit(self, pair: str, trade: 'Trade', order_type: str, amount: float,
                           rate: float, time_in_force: str, exit_reason: str,
                           current_time: datetime, **kwargs) -> bool:
        """
        Log exit details with maximum profit analysis.
        """
        final_profit = trade.calc_profit_ratio(rate)

        # Log maximum profit achieved
        if hasattr(trade, 'max_profit_reached'):
            max_profit = trade.max_profit_reached
            logger.info(f"EXIT: {pair} - Reason: {exit_reason}")
            logger.info(f"  Final Profit: {final_profit:.3f}")
            logger.info(f"  Maximum Profit Reached: {max_profit:.3f}")
            logger.info(f"  Profit Degradation: {((max_profit - final_profit) / max_profit * 100):.1f}%")
        else:
            logger.info(f"EXIT: {pair} - Reason: {exit_reason}, Profit: {final_profit:.3f}")

        return True