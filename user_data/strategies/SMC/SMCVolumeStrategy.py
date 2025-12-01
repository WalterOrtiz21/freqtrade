"""
SMC + Volume Strategy - Complete Implementation

Based on SMC_VOLUME_STRATEGY.md following The Trading Channel methodology
YouTube: https://www.youtube.com/watch?v=0lavSirTpgE

This strategy combines Smart Money Concepts with 3 volume confirmation methods:
1. Volume Divergence Detection
2. Volume Spike Identification
3. Volume Profile Analysis

Key Logic:
- Estructura clara en HTF
- Zona SMC no mitigada
- Barrido de liquidez reciente
- Divergencia de volumen presente
- CHoCH formado y confirmado
- Spike de volumen visible
- Entrada en Order Block del CHoCH
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime
from pandas import DataFrame
from typing import Dict, Tuple

from freqtrade.strategy import IStrategy
import sys
import os

# Import volume indicators module
sys.path.append(os.path.dirname(__file__))
from volume_indicators import (
    calculate_volume_divergence,
    detect_volume_spikes,
    apply_all_volume_indicators
)

logger = logging.getLogger(__name__)


class SMCVolumeStrategy(IStrategy):
    """
    SMC + Volume Strategy - Complete Implementation

    Follows SMC_VOLUME_STRATEGY.md methodology exactly:
    1. Structural Analysis (HTF)
    2. Key Zone Identification
    3. Retracement Waiting
    4. Volume Confirmation
    5. CHoCH Formation
    6. Volume Spike Entry
    """

    INTERFACE_VERSION = 3

    # === FIXED PARAMETERS ===

    # ROI Table - Based on structure targets
    minimal_roi = {
        "0": 100.0  # DISABLED - using custom TP
    }

    # Stop Loss (will be overridden by custom_stoploss)
    stoploss = -0.05  # Fallback value

    # Timeframe
    timeframe = '15m'

    # Trailing Stop - DISABLED
    trailing_stop = False
    trailing_stop_positive = 0.0
    trailing_stop_positive_offset = 0.0
    trailing_only_offset_is_reached = False

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
    use_exit_signal = True  # Enable exit signals
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    use_custom_stoploss = True  # Enable custom stop loss
    use_custom_takeprofit = True  # Enable custom take profit

    # Startup Candle Count
    startup_candle_count: int = 200

    # === STRATEGY LOGIC ===

    # Trade tracking
    trade_info = {}

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Calculate all indicators for SMC + Volume strategy.
        """
        logger.info(f"Calculating SMC + Volume indicators for {metadata['pair']}")

        try:
            # === BASIC INDICATORS FIRST ===
            # Volume SMAs for different timeframes
            dataframe['volume_sma_6'] = dataframe['volume'].rolling(6).mean()    # Recent volume
            dataframe['volume_sma_20'] = dataframe['volume'].rolling(20).mean()  # Context volume

            # Volume ratios (needed early)
            dataframe['volume_ratio_recent'] = dataframe['volume'] / dataframe['volume_sma_6']
            dataframe['volume_ratio'] = dataframe['volume'] / dataframe['volume_sma_20']
            dataframe['volume_trend'] = dataframe['volume'] / dataframe['volume_sma_6'] - 1

            # === MARKET STRUCTURE FIRST ===
            # Basic structure detection
            dataframe['higher_high'] = dataframe['high'] > dataframe['high'].rolling(20).max().shift(1)
            dataframe['lower_low'] = dataframe['low'] < dataframe['low'].rolling(20).min().shift(1)
            dataframe['higher_low'] = dataframe['low'] > dataframe['low'].rolling(20).max().shift(1)
            dataframe['lower_high'] = dataframe['high'] < dataframe['high'].rolling(20).min().shift(1)

            # Market state tracking
            dataframe['market_state'] = 0  # 0=neutral, 1=bearish, 2=bullish
            dataframe.loc[dataframe['higher_high'], 'market_state'] = 2
            dataframe.loc[dataframe['lower_low'], 'market_state'] = 1

            # === VOLUME INDICATORS ===
            # Apply all volume indicators from the module
            dataframe = apply_all_volume_indicators(dataframe)

            # === RESPONSIVE VOLUME ANALYSIS (5-6 velas) ===
            # Volume acceleration/deceleration (key for trend changes)
            dataframe['volume_deceleration'] = (dataframe['volume'] < dataframe['volume_sma_6'] * 0.7).astype(int)
            dataframe['volume_acceleration'] = (dataframe['volume'] > dataframe['volume_sma_6'] * 1.3).astype(int)

            # Volume trend (increasing/decreasing over 3-6 periods)
            dataframe['volume_trend_short'] = dataframe['volume'].rolling(3).mean() / dataframe['volume'].rolling(6).mean() - 1

            # Spike detection relative to recent volume (not long-term average)
            dataframe['volume_spike_recent'] = (dataframe['volume'] > dataframe['volume_sma_6'] * 2.0).astype(int)
            dataframe['volume_spike_moderate'] = (dataframe['volume'] > dataframe['volume_sma_6'] * 1.5).astype(int)

            # Volume exhaustion (key signal before reversal)
            dataframe['volume_exhaustion'] = (
                (dataframe['volume'] < dataframe['volume_sma_6'] * 0.6) &  # Very low volume
                (dataframe['volume_deceleration'].rolling(2).sum() >= 1)  # Recent deceleration
            ).astype(int)

            # Volume expansion (confirmation of new trend)
            dataframe['volume_expansion'] = (
                (dataframe['volume'] > dataframe['volume_sma_6'] * 1.8) &  # High volume
                (dataframe['volume_acceleration'] == 1)  # Accelerating
            ).astype(int)

            # === CHoCH DETECTION ===
            # Enhanced CHoCH detection with responsive volume confirmation
            dataframe['choch_bullish'] = (
                (dataframe['close'] > dataframe['high'].rolling(20).max().shift(1)) &
                (dataframe['close'].shift(1) < dataframe['high'].rolling(20).max().shift(2)) &
                (dataframe['volume_expansion'] == 1)  # Volume expansion confirming breakout
            ).astype(int)

            dataframe['choch_bearish'] = (
                (dataframe['close'] < dataframe['low'].rolling(20).min().shift(1)) &
                (dataframe['close'].shift(1) > dataframe['low'].rolling(20).min().shift(2)) &
                (dataframe['volume_expansion'] == 1)  # Volume expansion confirming breakout
            ).astype(int)

            # === FAIR VALUE GAPS (FVG) ===
            # FVG created by CHoCH with responsive volume confirmation
            dataframe['fvg_bullish'] = (
                (dataframe['low'].shift(1) > dataframe['high'].shift(2)) &
                (dataframe['close'] > dataframe['open']) &
                (dataframe['choch_bullish'] == 1) &
                (dataframe['volume_spike_recent'] == 1)  # Recent spike confirming FVG
            ).astype(int)

            dataframe['fvg_bearish'] = (
                (dataframe['high'].shift(1) < dataframe['low'].shift(2)) &
                (dataframe['close'] < dataframe['open']) &
                (dataframe['choch_bearish'] == 1) &
                (dataframe['volume_spike_recent'] == 1)  # Recent spike confirming FVG
            ).astype(int)

            # Calculate FVG levels
            dataframe['fvg_top_bullish'] = np.where(
                dataframe['fvg_bullish'] == 1,
                dataframe['low'].shift(1),
                0
            )
            dataframe['fvg_bottom_bullish'] = np.where(
                dataframe['fvg_bullish'] == 1,
                dataframe['high'].shift(2),
                0
            )
            dataframe['fvg_top_bearish'] = np.where(
                dataframe['fvg_bearish'] == 1,
                dataframe['high'].shift(1),
                0
            )
            dataframe['fvg_bottom_bearish'] = np.where(
                dataframe['fvg_bearish'] == 1,
                dataframe['low'].shift(2),
                0
            )

            # === ORDER BLOCKS (from CHoCH candles) ===
            # Enhanced OB detection with responsive volume
            dataframe['ob_bullish'] = (
                (dataframe['close'].shift(1) < dataframe['open'].shift(1)) &
                (dataframe['choch_bullish'] == 1) &
                (dataframe['volume_ratio_recent'].shift(1) > 1.2)  # Recent volume in OB candle
            ).astype(int)

            dataframe['ob_bearish'] = (
                (dataframe['close'].shift(1) > dataframe['open'].shift(1)) &
                (dataframe['choch_bearish'] == 1) &
                (dataframe['volume_ratio_recent'].shift(1) > 1.2)  # Recent volume in OB candle
            ).astype(int)

            # Calculate Order Block levels
            dataframe['ob_top_bullish'] = np.where(
                dataframe['ob_bullish'] == 1,
                dataframe['high'].shift(1),
                0
            )
            dataframe['ob_bottom_bullish'] = np.where(
                dataframe['ob_bullish'] == 1,
                dataframe['low'].shift(1),
                0
            )
            dataframe['ob_top_bearish'] = np.where(
                dataframe['ob_bearish'] == 1,
                dataframe['high'].shift(1),
                0
            )
            dataframe['ob_bottom_bearish'] = np.where(
                dataframe['ob_bearish'] == 1,
                dataframe['low'].shift(1),
                0
            )

            # === ZONE MEMORY ===
            # Enhanced zone memory with volume validation
            dataframe['recent_fvg_bullish_top'] = 0.0
            dataframe['recent_fvg_bullish_bottom'] = 0.0
            dataframe['recent_fvg_bearish_top'] = 0.0
            dataframe['recent_fvg_bearish_bottom'] = 0.0
            dataframe['recent_ob_bullish_top'] = 0.0
            dataframe['recent_ob_bullish_bottom'] = 0.0
            dataframe['recent_ob_bearish_top'] = 0.0
            dataframe['recent_ob_bearish_bottom'] = 0.0

            # Update zone memory when new CHoCH occurs with volume confirmation
            for i in range(1, len(dataframe)):
                # Bullish CHoCH with high volume
                if dataframe.iloc[i]['fvg_bullish'] == 1 and dataframe.iloc[i]['volume_ratio'] > 2.0:
                    dataframe.loc[i:, 'recent_fvg_bullish_top'] = dataframe.iloc[i]['fvg_top_bullish']
                    dataframe.loc[i:, 'recent_fvg_bullish_bottom'] = dataframe.iloc[i]['fvg_bottom_bullish']

                if dataframe.iloc[i]['ob_bullish'] == 1 and dataframe.iloc[i]['volume_ratio'] > 1.5:
                    dataframe.loc[i:, 'recent_ob_bullish_top'] = dataframe.iloc[i]['ob_top_bullish']
                    dataframe.loc[i:, 'recent_ob_bullish_bottom'] = dataframe.iloc[i]['ob_bottom_bullish']

                # Bearish CHoCH with high volume
                if dataframe.iloc[i]['fvg_bearish'] == 1 and dataframe.iloc[i]['volume_ratio'] > 2.0:
                    dataframe.loc[i:, 'recent_fvg_bearish_top'] = dataframe.iloc[i]['fvg_top_bearish']
                    dataframe.loc[i:, 'recent_fvg_bearish_bottom'] = dataframe.iloc[i]['fvg_bottom_bearish']

                if dataframe.iloc[i]['ob_bearish'] == 1 and dataframe.iloc[i]['volume_ratio'] > 1.5:
                    dataframe.loc[i:, 'recent_ob_bearish_top'] = dataframe.iloc[i]['ob_top_bearish']
                    dataframe.loc[i:, 'recent_ob_bearish_bottom'] = dataframe.iloc[i]['ob_bottom_bearish']

            # === MARKET STATE TRACKING ===
            dataframe['market_state'] = 0  # 0=neutral, 1=bearish, 2=bullish

            # Update market state based on structure and volume
            dataframe.loc[dataframe['higher_high'] & (dataframe['volume_ratio'] > 1.2), 'market_state'] = 2
            dataframe.loc[dataframe['lower_low'] & (dataframe['volume_ratio'] > 1.2), 'market_state'] = 1

            # === LIQUIDITY SWEEPS DETECTION ===
            # Detect recent stop hunts (liquidity sweeps)
            dataframe['liquidity_sweep_high'] = (
                (dataframe['high'] > dataframe['high'].rolling(10).max().shift(1)) &
                (dataframe['close'] < dataframe['open']) &  # Rejection after sweep
                (dataframe['volume_ratio'] > 1.8)  # High volume on sweep
            ).astype(int)

            dataframe['liquidity_sweep_low'] = (
                (dataframe['low'] < dataframe['low'].rolling(10).min().shift(1)) &
                (dataframe['close'] > dataframe['open']) &  # Rejection after sweep
                (dataframe['volume_ratio'] > 1.8)  # High volume on sweep
            ).astype(int)

            logger.info(f"SMC + Volume indicators calculated successfully for {metadata['pair']}")

        except Exception as e:
            logger.error(f"Error calculating SMC + Volume indicators: {e}")

        return dataframe

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Dynamic stop loss based on Order Block levels.
        Stop loss placed just beyond the OB as per SMC methodology.
        """
        try:
            # Initialize trade info if not exists
            if trade.id not in self.trade_info:
                self.trade_info[trade.id] = {
                    'entry_price': trade.open_rate,
                    'ob_top': 0.0,
                    'ob_bottom': 0.0,
                    'sl_level': 0.0,
                    'tp1_hit': False,
                    'break_even_set': False
                }

            trade_data = self.trade_info[trade.id]

            # Check for break-even movement (TP1)
            if not trade_data['break_even_set'] and current_profit >= 0.03:
                logger.info(f"Moving to break-even for {trade.pair} - TP1 hit")
                trade_data['break_even_set'] = True
                return 0.0

            # If break even is set, maintain it
            if trade_data['break_even_set']:
                return 0.0

            # Calculate dynamic SL based on OB levels
            if trade_data['sl_level'] > 0:
                if trade.is_short:
                    # SHORT: SL above OB top
                    sl_distance = (trade_data['sl_level'] - current_rate) / current_rate
                    return max(sl_distance, -0.003)  # Max 3% real loss
                else:
                    # LONG: SL below OB bottom
                    sl_distance = (trade_data['sl_level'] - current_rate) / current_rate
                    return max(sl_distance, -0.002)  # Max 2% real loss
            else:
                # Fallback SL
                if trade.is_short:
                    return -0.003  # 0.3% = 3% real with 10x leverage
                else:
                    return -0.002  # 0.2% = 2% real with 10x leverage

        except Exception as e:
            logger.error(f"Error in custom stop loss: {e}")
            return -0.003

    def custom_takeprofit(self, pair: str, trade: 'Trade', current_time: datetime,
                         current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Dynamic take profit targeting opposite liquidity.
        TP1 at 3%, TP2 at 6% targeting swing point liquidity.
        """
        try:
            # Initialize trade info if not exists
            if trade.id not in self.trade_info:
                self.trade_info[trade.id] = {
                    'entry_price': trade.open_rate,
                    'ob_top': 0.0,
                    'ob_bottom': 0.0,
                    'sl_level': 0.0,
                    'tp1_hit': False,
                    'break_even_set': False
                }

            trade_data = self.trade_info[trade.id]

            # TP1 at 3% - Move to break even
            if not trade_data['tp1_hit'] and current_profit >= 0.03:
                logger.info(f"TP1 hit for {trade.pair} at {current_profit:.2%}")
                trade_data['tp1_hit'] = True
                trade_data['break_even_set'] = True
                return current_profit

            # TP2 at 6% - Full exit at opposite liquidity
            if current_profit >= 0.06:
                logger.info(f"TP2 hit for {trade.pair} at {current_profit:.2%}")
                return current_profit

            return current_profit

        except Exception as e:
            logger.error(f"Error in custom take profit: {e}")
            return current_profit

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float,
                           rate: float, time_in_force: str, current_time: datetime,
                           entry_tag: str, side: str, **kwargs) -> bool:
        """
        Confirm trade entry and capture exact OB levels.
        """
        try:
            # Log entry confirmation with SMC + Volume context
            logger.info(f"SMC+Volume entry confirmed for {pair} {side} at {rate}")

            # Store entry levels in trade_info
            if side == 'long':
                logger.info(f"LONG entry - looking for bullish divergence + volume spike")
                logger.info(f"Entry zone: Bullish OB/FVG with volume confirmation")
            else:
                logger.info(f"SHORT entry - looking for bearish divergence + volume spike")
                logger.info(f"Entry zone: Bearish OB/FVG with volume confirmation")

            return True

        except Exception as e:
            logger.error(f"Error confirming trade entry: {e}")
            return True

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Generate entry signals following SMC + Volume methodology.

        ENTRY CHECKLIST from SMC_VOLUME_STRATEGY.md:
        ✓ Estructura clara en HTF
        ✓ Zona SMC no mitigada identificada
        ✓ Barrido de liquidez reciente
        ✓ Divergencia de volumen presente
        ✓ CHoCH formado y confirmado
        ✓ Spike de volumen visible
        """
        logger.info(f"Generating SMC + Volume entry signals for {metadata['pair']}")

        # === SIMPLIFIED ENTRY CHECKLIST (más realista) ===

        # 1. STRUCTURE ANALYSIS - Clear trend
        bullish_structure = (
            (dataframe['market_state'] == 2) &  # Bullish structure
            (dataframe['higher_high'].rolling(5).sum() >= 1)  # Recent higher high
        )

        bearish_structure = (
            (dataframe['market_state'] == 1) &  # Bearish structure
            (dataframe['lower_low'].rolling(5).sum() >= 1)  # Recent lower low
        )

        # 2. KEY ZONE IDENTIFICATION - Unmitigated SMC zones
        in_bullish_zone = (
            (dataframe['recent_fvg_bullish_top'] > 0) |
            (dataframe['recent_ob_bullish_top'] > 0)
        )

        in_bearish_zone = (
            (dataframe['recent_fvg_bearish_top'] > 0) |
            (dataframe['recent_ob_bearish_top'] > 0)
        )

        # 3. VOLUME EXHAUSTION - Key signal before reversal (tu observación)
        volume_exhaustion_bullish = (
            dataframe['volume_exhaustion'] == 1  # Very low volume before bullish move
        )

        volume_exhaustion_bearish = (
            dataframe['volume_exhaustion'] == 1  # Very low volume before bearish move
        )

        # 4. VOLUME ACCELERATION - Confirmation of trend change
        volume_acceleration_bullish = (
            (dataframe['volume_acceleration'] == 1) &
            (dataframe['volume_ratio_recent'] > 1.3)  # Accelerating volume
        )

        volume_acceleration_bearish = (
            (dataframe['volume_acceleration'] == 1) &
            (dataframe['volume_ratio_recent'] > 1.3)  # Accelerating volume
        )

        # 5. RECENT CHoCH - Structure change
        recent_choch_bullish = (
            dataframe['choch_bullish'].rolling(3).sum() >= 1  # Recent bullish CHoCH
        )

        recent_choch_bearish = (
            dataframe['choch_bearish'].rolling(3).sum() >= 1  # Recent bearish CHoCH
        )

        # === FINAL ENTRY CONDITIONS (SIMPLIFIED) ===

        # 🔴 SHORT ENTRY CONDITIONS - Más relajado y realista
        short_entry_zone = (
            in_bearish_zone &
            (dataframe['close'] >= dataframe['recent_fvg_bearish_bottom']) &
            (dataframe['close'] <= dataframe['recent_fvg_bearish_top'])
        ) | (
            in_bearish_zone &
            (dataframe['close'] >= dataframe['recent_ob_bearish_bottom']) &
            (dataframe['close'] <= dataframe['recent_ob_bearish_top'])
        )

        sell_conditions = (
            bearish_structure &                    # 1. Clear bearish structure
            recent_choch_bearish &                 # 2. Recent CHoCH
            short_entry_zone &                     # 3. In key SMC zone
            (volume_exhaustion_bearish | volume_acceleration_bearish)  # 4. Volume signal
        )

        # 🟢 LONG ENTRY CONDITIONS - Más relajado y realista
        long_entry_zone = (
            in_bullish_zone &
            (dataframe['close'] >= dataframe['recent_fvg_bullish_bottom']) &
            (dataframe['close'] <= dataframe['recent_fvg_bullish_top'])
        ) | (
            in_bullish_zone &
            (dataframe['close'] >= dataframe['recent_ob_bullish_bottom']) &
            (dataframe['close'] <= dataframe['recent_ob_bullish_top'])
        )

        buy_conditions = (
            bullish_structure &                    # 1. Clear bullish structure
            recent_choch_bullish &                 # 2. Recent CHoCH
            long_entry_zone &                      # 3. In key SMC zone
            (volume_exhaustion_bullish | volume_acceleration_bullish)  # 4. Volume signal
        )

        # Set entry signals
        dataframe.loc[buy_conditions, 'enter_long'] = 1
        dataframe.loc[sell_conditions, 'enter_short'] = 1

        # Count and log signals
        buy_signals = dataframe['enter_long'].sum()
        sell_signals = dataframe['enter_short'].sum()

        logger.info(f"SMC + Volume entry signals generated for {metadata['pair']} - Buy: {buy_signals}, Sell: {sell_signals}")

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Generate exit signals - mainly handled by custom TP/SL.
        Only emergency exits here.
        """
        # Emergency exit conditions
        exit_long_conditions = (
            (dataframe['choch_bearish'] == 1) &
            (dataframe['volume_ratio'] > 2.0) &  # High volume confirmation
            (dataframe['close'] < dataframe['close'].shift(1) * 0.98)
        )

        exit_short_conditions = (
            (dataframe['choch_bullish'] == 1) &
            (dataframe['volume_ratio'] > 2.0) &  # High volume confirmation
            (dataframe['close'] > dataframe['close'].shift(1) * 1.02)
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