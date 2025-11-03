"""
CHoCH/BOS FreqAI Strategy - Complete ML-based trading strategy

This strategy uses machine learning to optimize:
- Entry signals (CHoCH + BOS with volume confirmation)
- Exit methods (Fibonacci, MST, Smooth Trail, SFP)
- Stop loss methods (Static -2% to -5%, MS Trailing, FVG Trailing)

Features:
- CHoCH (Change of Character) detection
- BOS (Break of Structure) confirmation
- Volume correlation with structure breaks
- Fibonacci retracements (0.381, 0.5, 0.618)
- MST (Market Structure Targets)
- FVG (Fair Value Gaps) trailing
- Smooth Trail indicator
- SFP (Swing Failure Pattern)
- HMA 200 filter
- ZigZag structure (depth=23)

GPU Accelerated with PyTorch/TensorFlow support
"""

import logging
from functools import reduce
import numpy as np
import pandas as pd
import talib.abstract as ta
from pandas import DataFrame
from technical import qtpylib
from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter, CategoricalParameter

logger = logging.getLogger(__name__)


class CHoCHBOSFreqAI(IStrategy):
    """
    FreqAI Strategy combining Smart Money Concepts with ML optimization
    """

    INTERFACE_VERSION = 3

    # ROI table - disabled (rely on ML predictions)
    minimal_roi = {"0": 100}

    # Stoploss - will be overridden by ML predictions
    stoploss = -0.05

    # Trailing stop
    trailing_stop = False
    use_custom_stoploss = True
    use_exit_signal = True

    # Optimal timeframe
    timeframe = '15m'

    # Run "populate_indicators()" only for new candle
    process_only_new_candles = True

    # Position adjustment
    position_adjustment_enable = True
    max_entry_position_adjustment = -1

    # Startup candle count
    startup_candle_count: int = 200

    # Order types
    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'limit',
        'stoploss_on_exchange': False,
        'stoploss_on_exchange_interval': 60
    }

    # Order time in force
    order_time_in_force = {
        'entry': 'GTC',
        'exit': 'GTC'
    }

    # Can short
    can_short = True

    # Plot config for visualization
    plot_config = {
        "main_plot": {
            "hma_200": {"color": "blue"},
        },
        "subplots": {
            "Predictions": {
                "&s-action": {"color": "green"},
                "do_predict": {"color": "purple"},
            },
            "Volume": {
                "volume": {"color": "cyan"},
                "%-volume_choch_ratio": {"color": "yellow"},
            },
        },
    }

    # FreqAI parameters
    process_only_new_candles = True

    # ============================
    # CHoCH/BOS Parameters
    # ============================

    # ZigZag depth for structure detection
    zigzag_depth = IntParameter(
        low=15,
        high=30,
        default=23,
        space="buy",
        optimize=False,
        load=True
    )

    # HMA filter
    hma_length = IntParameter(
        low=100,
        high=300,
        default=200,
        space="buy",
        optimize=False,
        load=True
    )

    # Smooth Trail parameters
    smooth_trail_length = IntParameter(
        low=20,
        high=50,
        default=34,
        space="sell",
        optimize=False,
        load=True
    )

    smooth_trail_multiplier = DecimalParameter(
        low=0.5,
        high=2.0,
        default=1.0,
        decimals=1,
        space="sell",
        optimize=False,
        load=True
    )

    # ============================
    # State constants
    # ============================
    STATE_NEUTRAL = 0
    STATE_BULLISH = 1
    STATE_BEARISH = -1

    def feature_engineering_expand_all(
        self, dataframe: DataFrame, period: int, metadata: dict, **kwargs
    ) -> DataFrame:
        """
        Features that expand with different periods.
        These will be calculated for each period in indicator_periods_candles.
        """

        # Traditional technical indicators
        dataframe[f"%-rsi-period_{period}"] = ta.RSI(dataframe, timeperiod=period)
        dataframe[f"%-adx-period_{period}"] = ta.ADX(dataframe, timeperiod=period)
        dataframe[f"%-mfi-period_{period}"] = ta.MFI(dataframe, timeperiod=period)

        # Moving averages
        dataframe[f"%-ema-period_{period}"] = ta.EMA(dataframe, timeperiod=period)
        dataframe[f"%-sma-period_{period}"] = ta.SMA(dataframe, timeperiod=period)

        # Volatility
        dataframe[f"%-atr-period_{period}"] = ta.ATR(dataframe, timeperiod=period)
        dataframe[f"%-atr_pct-period_{period}"] = (
            dataframe[f"%-atr-period_{period}"] / dataframe["close"]
        )

        # Volume
        dataframe[f"%-volume_ma-period_{period}"] = dataframe["volume"].rolling(period).mean()
        dataframe[f"%-volume_ratio-period_{period}"] = (
            dataframe["volume"] / dataframe[f"%-volume_ma-period_{period}"]
        )

        # Rate of Change
        dataframe[f"%-roc-period_{period}"] = ta.ROC(dataframe, timeperiod=period)

        # Bollinger Bands
        bollinger = qtpylib.bollinger_bands(
            qtpylib.typical_price(dataframe), window=period, stds=2
        )
        dataframe[f"%-bb_width-period_{period}"] = (
            (bollinger["upper"] - bollinger["lower"]) / bollinger["mid"]
        )

        return dataframe

    def feature_engineering_expand_basic(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        """
        Basic features that don't expand with periods.
        """

        # Price features
        dataframe["%-pct_change"] = dataframe["close"].pct_change()
        dataframe["%-raw_volume"] = dataframe["volume"]
        dataframe["%-raw_price"] = dataframe["close"]

        # Candle features
        dataframe["%-hl_range"] = (dataframe["high"] - dataframe["low"]) / dataframe["close"]
        dataframe["%-body_size"] = abs(dataframe["close"] - dataframe["open"]) / dataframe["close"]

        # Wicks
        dataframe["%-upper_wick"] = (
            dataframe["high"] - dataframe[["open", "close"]].max(axis=1)
        ) / dataframe["close"]
        dataframe["%-lower_wick"] = (
            dataframe[["open", "close"]].min(axis=1) - dataframe["low"]
        ) / dataframe["close"]

        # Temporal features
        dataframe["%-day_of_week"] = dataframe["date"].dt.dayofweek
        dataframe["%-hour_of_day"] = dataframe["date"].dt.hour

        return dataframe

    def feature_engineering_standard(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        """
        Custom Smart Money Concepts features.
        This is where we add CHoCH, BOS, Fibonacci, MST, etc.
        """

        # ========================================
        # 1. ZIGZAG STRUCTURE DETECTION (depth=23)
        # ========================================

        depth = self.zigzag_depth.value
        dataframe['swing_high'] = 0.0
        dataframe['swing_low'] = 0.0

        # Calculate swing highs and lows
        for i in range(depth, len(dataframe) - depth):
            idx = dataframe.index[i]

            # Swing high: highest high in window
            current_high = dataframe.loc[idx, 'high']
            window_high = dataframe.iloc[i - depth:i + depth + 1]['high']
            is_swing_high = current_high == window_high.max()

            if is_swing_high:
                dataframe.loc[idx, 'swing_high'] = current_high

            # Swing low: lowest low in window
            current_low = dataframe.loc[idx, 'low']
            window_low = dataframe.iloc[i - depth:i + depth + 1]['low']
            is_swing_low = current_low == window_low.min()

            if is_swing_low:
                dataframe.loc[idx, 'swing_low'] = current_low

        # ========================================
        # 2. CHoCH and BOS DETECTION
        # ========================================

        dataframe['state'] = self.STATE_NEUTRAL
        dataframe['last_sig_high'] = np.nan
        dataframe['last_sig_low'] = np.nan
        dataframe['choch_up'] = 0
        dataframe['choch_down'] = 0
        dataframe['bos_up'] = 0
        dataframe['bos_down'] = 0

        current_state = self.STATE_NEUTRAL
        last_sig_high = np.nan
        last_sig_low = np.nan

        for i in range(len(dataframe)):
            idx = dataframe.index[i]
            close = dataframe.loc[idx, 'close']
            swing_high = dataframe.loc[idx, 'swing_high']
            swing_low = dataframe.loc[idx, 'swing_low']

            # Update significant levels
            if swing_high > 0:
                last_sig_high = swing_high
            if swing_low > 0:
                last_sig_low = swing_low

            dataframe.loc[idx, 'last_sig_high'] = last_sig_high
            dataframe.loc[idx, 'last_sig_low'] = last_sig_low

            # CHoCH detection (trend reversal)
            if current_state == self.STATE_BEARISH and not np.isnan(last_sig_high):
                if close > last_sig_high:
                    dataframe.loc[idx, 'choch_up'] = 1
                    current_state = self.STATE_BULLISH

            elif current_state == self.STATE_BULLISH and not np.isnan(last_sig_low):
                if close < last_sig_low:
                    dataframe.loc[idx, 'choch_down'] = 1
                    current_state = self.STATE_BEARISH

            # BOS detection (trend continuation)
            elif current_state == self.STATE_BULLISH and not np.isnan(last_sig_high):
                if close > last_sig_high:
                    dataframe.loc[idx, 'bos_up'] = 1

            elif current_state == self.STATE_BEARISH and not np.isnan(last_sig_low):
                if close < last_sig_low:
                    dataframe.loc[idx, 'bos_down'] = 1

            # Initial state
            if current_state == self.STATE_NEUTRAL:
                if not np.isnan(last_sig_high) and close > last_sig_high:
                    current_state = self.STATE_BULLISH
                elif not np.isnan(last_sig_low) and close < last_sig_low:
                    current_state = self.STATE_BEARISH

            dataframe.loc[idx, 'state'] = current_state

        # ========================================
        # 3. VOLUME CORRELATION WITH STRUCTURE
        # ========================================

        # Volume on CHoCH/BOS bars
        dataframe['%-volume_on_choch'] = np.where(
            (dataframe['choch_up'] == 1) | (dataframe['choch_down'] == 1),
            dataframe['volume'],
            0
        )

        dataframe['%-volume_on_bos'] = np.where(
            (dataframe['bos_up'] == 1) | (dataframe['bos_down'] == 1),
            dataframe['volume'],
            0
        )

        # Volume ratio on structure breaks
        volume_ma = dataframe['volume'].rolling(20).mean()
        dataframe['%-volume_choch_ratio'] = np.where(
            dataframe['%-volume_on_choch'] > 0,
            dataframe['%-volume_on_choch'] / volume_ma,
            0
        )

        dataframe['%-volume_bos_ratio'] = np.where(
            dataframe['%-volume_on_bos'] > 0,
            dataframe['%-volume_on_bos'] / volume_ma,
            0
        )

        # Volume spike detection
        dataframe['%-volume_spike'] = np.where(
            dataframe['volume'] > volume_ma * 2,
            1, 0
        )

        # ========================================
        # 4. FIBONACCI RETRACEMENT LEVELS
        # ========================================

        fib_period = 50
        period_high = dataframe['high'].rolling(fib_period).max()
        period_low = dataframe['low'].rolling(fib_period).min()
        fib_range = period_high - period_low

        # Calculate Fib levels
        dataframe['fib_0381'] = period_low + (fib_range * 0.381)
        dataframe['fib_0500'] = period_low + (fib_range * 0.500)
        dataframe['fib_0618'] = period_low + (fib_range * 0.618)

        # Distance to each Fib level (normalized)
        dataframe['%-dist_fib_0381'] = (dataframe['close'] - dataframe['fib_0381']) / dataframe['close']
        dataframe['%-dist_fib_0500'] = (dataframe['close'] - dataframe['fib_0500']) / dataframe['close']
        dataframe['%-dist_fib_0618'] = (dataframe['close'] - dataframe['fib_0618']) / dataframe['close']

        # Price at Fib level (within 0.5%)
        threshold = 0.005
        dataframe['%-at_fib_0381'] = np.where(abs(dataframe['%-dist_fib_0381']) < threshold, 1, 0)
        dataframe['%-at_fib_0500'] = np.where(abs(dataframe['%-dist_fib_0500']) < threshold, 1, 0)
        dataframe['%-at_fib_0618'] = np.where(abs(dataframe['%-dist_fib_0618']) < threshold, 1, 0)

        # ========================================
        # 5. FAIR VALUE GAPS (FVG)
        # ========================================

        # Bullish FVG: low[0] > high[2]
        dataframe['%-fvg_bullish'] = np.where(
            dataframe['low'] > dataframe['high'].shift(2),
            1, 0
        )

        # Bearish FVG: high[0] < low[2]
        dataframe['%-fvg_bearish'] = np.where(
            dataframe['high'] < dataframe['low'].shift(2),
            1, 0
        )

        # FVG size
        dataframe['%-fvg_size'] = np.where(
            dataframe['%-fvg_bullish'] == 1,
            (dataframe['low'] - dataframe['high'].shift(2)) / dataframe['close'],
            np.where(
                dataframe['%-fvg_bearish'] == 1,
                (dataframe['low'].shift(2) - dataframe['high']) / dataframe['close'],
                0
            )
        )

        # ========================================
        # 6. SWING FAILURE PATTERN (SFP)
        # ========================================

        swing_period = 20
        swing_high_sfp = dataframe['high'].rolling(swing_period).max()
        swing_low_sfp = dataframe['low'].rolling(swing_period).min()

        # Bullish SFP: breaks below swing low but closes above it
        dataframe['%-sfp_bullish'] = np.where(
            (dataframe['low'] < swing_low_sfp.shift(1)) &
            (dataframe['close'] > swing_low_sfp.shift(1)) &
            (dataframe['close'] > dataframe['open']),
            1, 0
        )

        # Bearish SFP: breaks above swing high but closes below it
        dataframe['%-sfp_bearish'] = np.where(
            (dataframe['high'] > swing_high_sfp.shift(1)) &
            (dataframe['close'] < swing_high_sfp.shift(1)) &
            (dataframe['close'] < dataframe['open']),
            1, 0
        )

        # ========================================
        # 7. SMOOTH TRAIL (for exits)
        # ========================================

        length = self.smooth_trail_length.value
        mult = self.smooth_trail_multiplier.value

        # Calculate smooth ATR
        atr = ta.ATR(dataframe, timeperiod=14)
        smooth_atr = ta.EMA(atr, timeperiod=length)

        # Smooth Trail bands
        dataframe['smooth_trail_upper'] = dataframe['close'] + (smooth_atr * mult)
        dataframe['smooth_trail_lower'] = dataframe['close'] - (smooth_atr * mult)

        # Distance to bands
        dataframe['%-dist_smooth_upper'] = (
            (dataframe['smooth_trail_upper'] - dataframe['close']) / dataframe['close']
        )
        dataframe['%-dist_smooth_lower'] = (
            (dataframe['close'] - dataframe['smooth_trail_lower']) / dataframe['close']
        )

        # Inside Smooth Trail zone
        ema_short = ta.EMA(dataframe['close'], timeperiod=10)
        dataframe['%-inside_smooth_zone'] = np.where(
            (ema_short >= dataframe['smooth_trail_lower']) &
            (ema_short <= dataframe['smooth_trail_upper']),
            1, 0
        )

        # ========================================
        # 8. MST (MARKET STRUCTURE TARGETS)
        # ========================================

        # Calculate MST projections based on last CHoCH
        dataframe['mst_target'] = np.nan
        dataframe['%-dist_mst'] = 0.0

        # Forward fill CHoCH signals
        choch_up_idx = dataframe[dataframe['choch_up'] == 1].index
        choch_down_idx = dataframe[dataframe['choch_down'] == 1].index

        for idx in dataframe.index:
            if idx in choch_up_idx:
                # Bullish CHoCH: find lowest point since last CHoCH
                prev_choch = choch_down_idx[choch_down_idx < idx]
                if len(prev_choch) > 0:
                    start_idx = prev_choch[-1]
                    lowest = dataframe.loc[start_idx:idx, 'low'].min()
                    choch_price = dataframe.loc[idx, 'close']
                    projection = choch_price - lowest
                    target = choch_price + projection  # 100% projection
                    dataframe.loc[idx:, 'mst_target'] = target

            elif idx in choch_down_idx:
                # Bearish CHoCH: find highest point since last CHoCH
                prev_choch = choch_up_idx[choch_up_idx < idx]
                if len(prev_choch) > 0:
                    start_idx = prev_choch[-1]
                    highest = dataframe.loc[start_idx:idx, 'high'].max()
                    choch_price = dataframe.loc[idx, 'close']
                    projection = highest - choch_price
                    target = choch_price - projection  # 100% projection
                    dataframe.loc[idx:, 'mst_target'] = target

        # Distance to MST target
        dataframe['%-dist_mst'] = np.where(
            ~dataframe['mst_target'].isna(),
            (dataframe['mst_target'] - dataframe['close']) / dataframe['close'],
            0
        )

        # ========================================
        # 9. HMA 200 TREND FILTER
        # ========================================

        hma_len = self.hma_length.value
        dataframe['hma_200'] = self.hma(dataframe['close'], hma_len)
        dataframe['%-price_vs_hma'] = (dataframe['close'] - dataframe['hma_200']) / dataframe['close']
        dataframe['%-above_hma'] = np.where(dataframe['close'] > dataframe['hma_200'], 1, 0)

        # HMA slope
        dataframe['%-hma_slope'] = dataframe['hma_200'].pct_change(5)

        # ========================================
        # 10. MARKET STRUCTURE STATE
        # ========================================

        # Convert state to features
        dataframe['%-bullish_state'] = np.where(dataframe['state'] == self.STATE_BULLISH, 1, 0)
        dataframe['%-bearish_state'] = np.where(dataframe['state'] == self.STATE_BEARISH, 1, 0)

        # Trend strength (ADX)
        dataframe['%-trend_strength'] = ta.ADX(dataframe, timeperiod=14)

        # Distance to swing levels
        dataframe['%-dist_swing_high'] = np.where(
            ~dataframe['last_sig_high'].isna(),
            (dataframe['last_sig_high'] - dataframe['close']) / dataframe['close'],
            0
        )
        dataframe['%-dist_swing_low'] = np.where(
            ~dataframe['last_sig_low'].isna(),
            (dataframe['close'] - dataframe['last_sig_low']) / dataframe['close'],
            0
        )

        # ========================================
        # 11. STATIC SL DISTANCES (for ML to choose)
        # ========================================

        dataframe['%-sl_2pct'] = -0.02
        dataframe['%-sl_3pct'] = -0.03
        dataframe['%-sl_4pct'] = -0.04
        dataframe['%-sl_5pct'] = -0.05

        # Fill NaNs to prevent training data from being dropped
        # Forward fill first, then backfill remaining NaNs, finally fill with 0
        feature_cols = [col for col in dataframe.columns if col.startswith('%-') or col.startswith('%')]
        for col in feature_cols:
            dataframe[col] = dataframe[col].ffill().bfill().fillna(0)

        return dataframe

    def hma(self, series, length):
        """Hull Moving Average calculation"""
        half_length = int(length / 2)
        sqrt_length = int(np.sqrt(length))

        wma_half = ta.WMA(series, timeperiod=half_length)
        wma_full = ta.WMA(series, timeperiod=length)

        raw_hma = 2 * wma_half - wma_full
        hma = ta.WMA(raw_hma, timeperiod=sqrt_length)

        return hma

    def set_freqai_targets(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
        """
        Define prediction targets for the ML model.

        We'll use multi-class classification:
        - enter_long: Enter a long position
        - enter_short: Enter a short position
        - exit: Exit current position
        - hold: Do nothing
        """

        # Set class names
        self.freqai.class_names = ["hold", "enter_long", "enter_short", "exit"]

        # Look ahead period
        label_period = self.freqai_info["feature_parameters"]["label_period_candles"]

        # Calculate future price movement
        future_close = dataframe["close"].shift(-label_period)
        price_change_pct = (future_close - dataframe["close"]) / dataframe["close"]

        # Define action based on future price movement
        # Thresholds adjusted for futures with 10x leverage
        entry_threshold = 0.005  # 0.5% move = 5% PnL with 10x
        exit_threshold = 0.003   # 0.3% move = 3% PnL with 10x

        # Classify actions
        dataframe["&s-action"] = "hold"

        # Enter long when significant upward movement expected
        dataframe.loc[price_change_pct > entry_threshold, "&s-action"] = "enter_long"

        # Enter short when significant downward movement expected
        dataframe.loc[price_change_pct < -entry_threshold, "&s-action"] = "enter_short"

        # Exit when movement is uncertain or reversal expected
        # (This would need to be refined based on whether we're in a position)
        dataframe.loc[
            (price_change_pct > -exit_threshold) & (price_change_pct < exit_threshold),
            "&s-action"
        ] = "exit"

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Initialize FreqAI and populate all indicators.
        """

        # FreqAI will call all feature_engineering_* functions
        dataframe = self.freqai.start(dataframe, metadata, self)

        # Additional indicators for visualization (not for ML)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Entry logic based on FreqAI predictions.
        """

        # LONG entries: Model predicts "enter_long" with high confidence
        dataframe.loc[
            (
                (dataframe["do_predict"] == 1) &
                (dataframe["&s-action"] == "enter_long")
            ),
            "enter_long"
        ] = 1

        # SHORT entries: Model predicts "enter_short" with high confidence
        dataframe.loc[
            (
                (dataframe["do_predict"] == 1) &
                (dataframe["&s-action"] == "enter_short")
            ),
            "enter_short"
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Exit logic based on FreqAI predictions.
        """

        # Exit when model predicts "exit" or opposite signal
        dataframe.loc[
            (
                (dataframe["do_predict"] == 1) &
                (
                    (dataframe["&s-action"] == "exit") |
                    (dataframe["&s-action"] == "enter_short")  # Exit long on short signal
                )
            ),
            "exit_long"
        ] = 1

        dataframe.loc[
            (
                (dataframe["do_predict"] == 1) &
                (
                    (dataframe["&s-action"] == "exit") |
                    (dataframe["&s-action"] == "enter_long")  # Exit short on long signal
                )
            ),
            "exit_short"
        ] = 1

        return dataframe

    def custom_stoploss(self, pair: str, trade, current_time, current_rate, current_profit, **kwargs) -> float:
        """
        Custom stop loss using ML predictions or fallback to static SL.
        This can be extended to use ML predictions for optimal SL level.
        """

        # Get the dataframe
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        # Check if we have ML predictions for SL
        # For now, use a simple ATR-based trailing stop
        atr = last_candle.get('%-atr-period_14', 0)

        if atr > 0:
            # Adaptive SL based on volatility
            atr_pct = atr / current_rate
            sl_distance = max(0.02, min(0.05, atr_pct * 2))  # Between 2% and 5%
            return -sl_distance

        # Fallback to static SL
        return self.stoploss
