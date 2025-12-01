"""
SMC + Volume Futures Strategy - 10x Leverage

This strategy is optimized for futures trading:
1. Volume Analysis with confirmation
2. Simple price action signals
3. 10x leverage support
4. Tighter risk management for futures

Works without smartmoneyconcepts library.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime
from functools import reduce
from pandas import DataFrame

# Import our custom modules
import sys
import os
sys.path.append(os.path.dirname(__file__))
from volume_indicators import apply_all_volume_indicators

from freqtrade.strategy import IStrategy

logger = logging.getLogger(__name__)


class SMCVolumeFuturesStrategy(IStrategy):
    """
    Volume + Price Action Strategy for Futures Trading

    This strategy uses volume analysis and simple price action
    optimized for 10x leverage futures trading.
    """

    INTERFACE_VERSION = 3

    # === FIXED PARAMETERS ===

    # ROI Table (adjusted for 10x leverage - smaller targets)
    minimal_roi = {
        "0": 0.03,      # 3% for 10x = 0.3% profit
        "10": 0.02,     # 2% for 10x = 0.2% profit
        "20": 0.01,     # 1% for 10x = 0.1% profit
        "40": 0.005     # 0.5% for 10x = 0.05% profit
    }

    # Stop Loss (tighter for leverage)
    stoploss = -0.02  # 2% = 20% at 10x leverage

    # Timeframe
    timeframe = '15m'

    # Trailing Stop
    trailing_stop = True  # Enable trailing for futures
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02
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
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Startup Candle Count
    startup_candle_count: int = 50

    # === STRATEGY LOGIC ===

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Calculate all volume indicators and simple price action for futures.
        """
        logger.info(f"Calculating Volume indicators for {metadata['pair']} (Futures)")

        try:
            # Apply Volume Analysis
            dataframe = apply_all_volume_indicators(
                dataframe,
                divergence_length=5,
                spike_multiplier=2.0,
                require_spike_confirmation=True
            )

            # Add Simple Price Action indicators
            dataframe = self._populate_simple_indicators(dataframe, metadata)

            logger.info(f"Volume indicators calculated successfully for {metadata['pair']}")

        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            self._reset_all_signals(dataframe)

        return dataframe

    def _populate_simple_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Calculate simple price action indicators optimized for futures.
        """
        # Simple Moving Averages
        dataframe['sma_20'] = dataframe['close'].rolling(window=20).mean()
        dataframe['sma_50'] = dataframe['close'].rolling(window=50).mean()

        # Exponential Moving Averages (more responsive for futures)
        dataframe['ema_9'] = dataframe['close'].ewm(span=9).mean()
        dataframe['ema_21'] = dataframe['close'].ewm(span=21).mean()

        # RSI
        delta = dataframe['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        dataframe['rsi'] = 100 - (100 / (1 + rs))

        # ATR for volatility (important for futures)
        high_low = dataframe['high'] - dataframe['low']
        high_close = np.abs(dataframe['high'] - dataframe['close'].shift())
        low_close = np.abs(dataframe['low'] - dataframe['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        dataframe['atr'] = true_range.rolling(14).mean()

        # Simple structure analysis
        dataframe['higher_high'] = dataframe['high'] > dataframe['high'].shift(20)
        dataframe['lower_low'] = dataframe['low'] < dataframe['low'].shift(20)

        # Trend direction (use EMAs for more responsive signals)
        dataframe['trend_up'] = dataframe['ema_9'] > dataframe['ema_21']
        dataframe['trend_down'] = dataframe['ema_9'] < dataframe['ema_21']

        # Price position relative to MAs
        dataframe['above_ema9'] = dataframe['close'] > dataframe['ema_9']
        dataframe['below_ema9'] = dataframe['close'] < dataframe['ema_9']

        # Volatility filter (avoid high volatility for leverage)
        dataframe['volatility_ratio'] = dataframe['atr'] / dataframe['close']
        dataframe['normal_volatility'] = dataframe['volatility_ratio'] < 0.02  # Less than 2% volatility

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Generate entry signals optimized for futures trading.
        """
        logger.info(f"Generating entry signals for {metadata['pair']} (Futures)")

        # === BUY CONDITIONS (Long) ===

        # Volume-based signals (require stronger confirmation for futures)
        volume_buy_conditions = (
            (dataframe['bullish_divergence'] == 1) |
            (dataframe['strong_bullish_divergence'] == 1) |
            (dataframe['significant_spike'] == 1)  # Require significant volume
        )

        # Price action confirmation (stricter for futures)
        price_buy_conditions = (
            (dataframe['trend_up'] == True) &
            (dataframe['above_ema9'] == True) &
            (dataframe['close'] > dataframe['open']) &  # Bullish momentum
            (dataframe['rsi'] < 65) &  # More conservative RSI
            (dataframe['rsi'] > 35)    # Not oversold
        )

        # Volatility filter (critical for leverage)
        volatility_filter_buy = (
            dataframe['normal_volatility'] == True
        )

        # Additional momentum confirmation
        momentum_buy = (
            (dataframe['volume_change_pct'] > 30) &  # Higher volume requirement
            (dataframe['close'] > dataframe['close'].shift(1))  # Price momentum
        )

        # Combine all buy conditions (stricter for futures)
        buy_conditions = (
            volume_buy_conditions &
            price_buy_conditions &
            volatility_filter_buy &
            momentum_buy
        )

        # === SELL CONDITIONS (Short) ===

        # Volume-based signals
        volume_sell_conditions = (
            (dataframe['bearish_divergence'] == 1) |
            (dataframe['strong_bearish_divergence'] == 1) |
            (dataframe['significant_spike'] == 1)
        )

        # Price action confirmation
        price_sell_conditions = (
            (dataframe['trend_down'] == True) &
            (dataframe['below_ema9'] == True) &
            (dataframe['close'] < dataframe['open']) &  # Bearish momentum
            (dataframe['rsi'] < 65) &  # More conservative RSI
            (dataframe['rsi'] > 35)    # Not oversold
        )

        # Volatility filter
        volatility_filter_sell = (
            dataframe['normal_volatility'] == True
        )

        # Additional momentum confirmation
        momentum_sell = (
            (dataframe['volume_change_pct'] > 30) &  # Higher volume requirement
            (dataframe['close'] < dataframe['close'].shift(1))  # Price momentum
        )

        # Combine all sell conditions
        sell_conditions = (
            volume_sell_conditions &
            price_sell_conditions &
            volatility_filter_sell &
            momentum_sell
        )

        # === FINAL ENTRY SIGNALS ===

        # Set entry signals
        dataframe.loc[buy_conditions, 'enter_long'] = 1
        dataframe.loc[sell_conditions, 'enter_short'] = 1

        # Count and log signals
        buy_signals = dataframe['enter_long'].sum()
        sell_signals = dataframe['enter_short'].sum()

        logger.info(f"Futures entry signals generated for {metadata['pair']} - Buy: {buy_signals}, Sell: {sell_signals}")

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Generate exit signals optimized for futures trading.
        """
        # Exit long when trend reverses or RSI overbought (more conservative)
        exit_long_conditions = (
            (dataframe['close'] < dataframe['ema_9']) |
            (dataframe['rsi'] > 70) |
            (dataframe['trend_down'] == True)  # Trend reversal
        )

        # Exit short when trend reverses or RSI oversold
        exit_short_conditions = (
            (dataframe['close'] > dataframe['ema_9']) |
            (dataframe['rsi'] < 30) |
            (dataframe['trend_up'] == True)  # Trend reversal
        )

        dataframe.loc[exit_long_conditions, 'exit_long'] = 1
        dataframe.loc[exit_short_conditions, 'exit_short'] = 1

        return dataframe

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: str,
                 side: str, **kwargs) -> float:
        """
        Set leverage to 10x for all trades as requested.
        """
        return 10.0

    def _reset_all_signals(self, dataframe: DataFrame):
        """
        Reset all signals to 0 to avoid accidental trades.
        """
        signal_columns = [
            'enter_long', 'enter_short', 'exit_long', 'exit_short'
        ]

        for col in signal_columns:
            if col in dataframe.columns:
                dataframe[col] = 0

    def informative_pairs(self):
        """
        No additional informative pairs needed.
        """
        return []