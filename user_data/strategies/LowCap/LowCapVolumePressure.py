"""
LowCapVolumePressure Strategy V3 — Gaussian + VPI Hybrid
=========================================================

Architecture:
- GAUSSIAN BANDS (BigBeluga port): Trend context + dynamic bands
- VPI (Volume Pressure Index): Volume-confirmed entry timing

Entry Logic:
1. Gaussian trend_direction must agree (1=up → longs only, -1=down → shorts only)
2. Gaussian trend_score confirms strength (< 0.5 bullish, > 0.5 bearish)
3. VPI signal line cross (MACD-style) provides precise timing
4. RVOL confirms real volume spike (not noise)
5. Cumulative VPI confirms sustained pressure

Exit Logic:
- TP1 partial exit at fixed % (configurable)
- Dynamic TP: close touches opposite Gaussian band
- Break-Even at configurable price move
- Stoploss via bot_start leverage adjustment

Thesis: "Gaussian tells you WHERE the market is going. VPI tells you WHEN to enter."
"""

import logging
import numpy as np
from datetime import datetime, timedelta
from pandas import DataFrame
from typing import Optional

import talib.abstract as ta

from freqtrade.strategy import (
    IStrategy, IntParameter, DecimalParameter, BooleanParameter,
    stoploss_from_absolute
)
from freqtrade.persistence import Trade

# Import modules
try:
    from volume_pressure_oscillator import calculate_vpi
    from gaussian_bands import calculate_gaussian_bands
except ImportError:
    from user_data.strategies.LowCap.volume_pressure_oscillator import calculate_vpi
    from user_data.strategies.LowCap.gaussian_bands import calculate_gaussian_bands

logger = logging.getLogger(__name__)


class LowCapVolumePressure(IStrategy):
    """
    Low Market Cap Strategy V3: Gaussian Bands + VPI Hybrid
    """
    
    INTERFACE_VERSION = 3
    
    # ==========================================================================
    # CORE SETTINGS
    # ==========================================================================
    timeframe = '5m'
    process_only_new_candles = False
    can_short = True
    use_exit_signal = True
    use_custom_stoploss = True
    position_adjustment_enable = True
    startup_candle_count: int = 150  # Gaussian needs more history (volatility_period=100)
    
    minimal_roi = {"0": 1.0}
    stoploss = -0.99  # Overridden by bot_start()
    
    # ==========================================================================
    # GAUSSIAN BANDS PARAMETERS
    # ==========================================================================
    
    # Base Gaussian filter length (Pine default: 20)
    gauss_length = IntParameter(10, 40, default=20, space='buy', optimize=True)
    
    # Band width multiplier (Pine default: 1.0)
    gauss_distance = DecimalParameter(0.5, 3.0, default=1.0, decimals=1, space='buy', optimize=True)
    
    # Minimum trend score to confirm direction (0=fully bullish, 1=fully bearish)
    # For longs: score must be < (0.5 - min_score_margin) = strong bullish
    # For shorts: score must be > (0.5 + min_score_margin) = strong bearish
    min_score_margin = DecimalParameter(0.0, 0.3, default=0.1, decimals=2, space='buy', optimize=True)
    
    # Require price to be on the correct side of trend_line
    use_trendline_filter = BooleanParameter(default=True, space='buy', optimize=True)
    
    # ==========================================================================
    # VPI TRIGGER PARAMETERS
    # ==========================================================================
    
    rvol_period = IntParameter(10, 30, default=20, space='buy', optimize=True)
    vpi_smooth_period = IntParameter(3, 10, default=5, space='buy', optimize=True)
    vpi_signal_period = IntParameter(8, 21, default=13, space='buy', optimize=True)
    min_rvol = DecimalParameter(1.5, 5.0, default=2.0, decimals=1, space='buy', optimize=True)
    
    # Cumulative VPI
    cumulative_window = IntParameter(3, 10, default=5, space='buy', optimize=True)
    min_cumulative_vpi = DecimalParameter(0.5, 8.0, default=2.0, decimals=1, space='buy', optimize=True)
    
    # Volume persistence
    min_vol_persistence = IntParameter(1, 5, default=2, space='buy', optimize=True)
    
    # ==========================================================================
    # EXIT PARAMETERS
    # ==========================================================================
    
    # TP1: Partial exit at fixed price movement
    tp1_pct = DecimalParameter(0.005, 0.05, default=0.03, decimals=3, space='sell', optimize=True)
    tp1_enabled = BooleanParameter(default=True, space='sell', optimize=True)
    tp1_amount = DecimalParameter(30.0, 80.0, default=60.0, space='sell', optimize=True)
    
    # Dynamic TP: Exit when price reaches Gaussian band
    use_gaussian_tp = BooleanParameter(default=True, space='sell', optimize=True)
    
    # Break Even
    be_enabled = BooleanParameter(default=True, space='sell', optimize=True)
    be_trigger_pct = DecimalParameter(0.005, 0.03, default=0.02, decimals=3, space='sell', optimize=True)
    
    # Cooldown
    cooldown_candles = IntParameter(1, 10, default=5, space='buy', optimize=True)
    
    # Logging
    enable_logging = BooleanParameter(default=False, space='custom', optimize=False)
    
    # ==========================================================================
    # BOT START
    # ==========================================================================
    def bot_start(self, **kwargs) -> None:
        config_leverage = self.config.get('leverage', 1.0)
        raw_stoploss = self.config.get('stoploss', -0.03)
        self.stoploss = raw_stoploss * config_leverage
        
        logger.info(
            f"LowCapVolumePressure V3 (Gaussian+VPI):"
            f"\n  Leverage: {config_leverage}x"
            f"\n  Base SL (Price): {raw_stoploss:.2%}"
            f"\n  Effective SL (PnL): {self.stoploss:.2%}"
            f"\n  Gaussian: length={self.gauss_length.value}, "
            f"distance={self.gauss_distance.value}, "
            f"score_margin={self.min_score_margin.value}"
            f"\n  VPI: smooth={self.vpi_smooth_period.value}, "
            f"signal={self.vpi_signal_period.value}, "
            f"min_rvol={self.min_rvol.value}"
        )
    
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: str,
                 side: str, **kwargs) -> float:
        return self.config.get('leverage', 1.0)
    
    # ==========================================================================
    # INDICATORS
    # ==========================================================================
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata['pair']
        
        # --- GAUSSIAN BANDS ---
        gauss = calculate_gaussian_bands(
            close=dataframe['close'].values,
            high=dataframe['high'].values,
            low=dataframe['low'].values,
            length=self.gauss_length.value,
            sigma=10.0,  # Pine Script default
            distance=self.gauss_distance.value,
            volatility_period=100  # Pine Script default: SMA(high-low, 100)
        )
        
        dataframe['gauss_score'] = gauss['trend_score']
        dataframe['gauss_avg'] = gauss['avg_value']
        dataframe['gauss_upper'] = gauss['upper_band']
        dataframe['gauss_lower'] = gauss['lower_band']
        dataframe['gauss_trendline'] = gauss['trend_line']
        dataframe['gauss_trend'] = gauss['trend_direction']
        
        # --- VPI OSCILLATOR ---
        vpi_data = calculate_vpi(
            close=dataframe['close'].values,
            high=dataframe['high'].values,
            low=dataframe['low'].values,
            volume=dataframe['volume'].values,
            rvol_period=self.rvol_period.value,
            smooth_period=self.vpi_smooth_period.value,
            signal_period=self.vpi_signal_period.value,
            cumulative_window=self.cumulative_window.value,
            min_rvol_for_persist=self.min_rvol.value * 0.75
        )
        
        dataframe['vpi_smooth'] = vpi_data['vpi_smooth']
        dataframe['vpi_signal'] = vpi_data['vpi_signal']
        dataframe['vpi_histogram'] = vpi_data['vpi_histogram']
        dataframe['rvol'] = vpi_data['rvol']
        dataframe['cumulative_vpi'] = vpi_data['cumulative_vpi']
        dataframe['vol_persist'] = vpi_data['volume_persist']
        
        # Previous values for cross detection
        dataframe['vpi_smooth_prev'] = dataframe['vpi_smooth'].shift(1)
        dataframe['vpi_signal_prev'] = dataframe['vpi_signal'].shift(1)
        
        if self.enable_logging.value:
            last = dataframe.iloc[-1]
            logger.info(
                f"[{pair}] G_trend={last.get('gauss_trend', 0):.0f} "
                f"G_score={last.get('gauss_score', 0):.2f} "
                f"VPI={last.get('vpi_smooth', 0):.3f} "
                f"RVOL={last.get('rvol', 0):.2f} "
                f"cum={last.get('cumulative_vpi', 0):.2f}"
            )
        
        return dataframe
    
    # ==========================================================================
    # ENTRY SIGNALS — Gaussian Context + VPI Trigger
    # ==========================================================================
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        V3 Entry Logic:
        
        Layer 1 — GAUSSIAN CONTEXT (is there a trend?)
          - trend_direction == 1 (up) for longs, -1 (down) for shorts
          - trend_score < 0.4 for longs (bullish consensus), > 0.6 for shorts
          - Price above trend_line for longs, below for shorts
        
        Layer 2 — VPI TRIGGER (timing within the trend)
          - VPI signal line bullish/bearish cross
          - RVOL > min_rvol (real volume)
          - Cumulative VPI confirms sustained pressure
          - Volume persistence (consecutive elevated candles)
        """
        
        score_bull_thresh = 0.5 - self.min_score_margin.value
        score_bear_thresh = 0.5 + self.min_score_margin.value
        
        # ===== LAYER 1: GAUSSIAN CONTEXT =====
        
        gauss_bullish = (
            (dataframe['gauss_trend'] == 1) &
            (dataframe['gauss_score'] <= score_bull_thresh)
        )
        
        gauss_bearish = (
            (dataframe['gauss_trend'] == -1) &
            (dataframe['gauss_score'] >= score_bear_thresh)
        )
        
        # Trendline filter: price must be on correct side
        if self.use_trendline_filter.value:
            gauss_bullish = gauss_bullish & (dataframe['close'] > dataframe['gauss_trendline'])
            gauss_bearish = gauss_bearish & (dataframe['close'] < dataframe['gauss_trendline'])
        
        # ===== LAYER 2: VPI TRIGGER =====
        
        # Signal line cross
        vpi_bull_cross = (
            (dataframe['vpi_smooth'] > dataframe['vpi_signal']) &
            (dataframe['vpi_smooth_prev'] <= dataframe['vpi_signal_prev'])
        )
        
        vpi_bear_cross = (
            (dataframe['vpi_smooth'] < dataframe['vpi_signal']) &
            (dataframe['vpi_smooth_prev'] >= dataframe['vpi_signal_prev'])
        )
        
        # RVOL confirmation
        rvol_ok = dataframe['rvol'] >= self.min_rvol.value
        
        # Cumulative VPI (sustained pressure)
        cum_bullish = dataframe['cumulative_vpi'] >= self.min_cumulative_vpi.value
        cum_bearish = dataframe['cumulative_vpi'] <= -self.min_cumulative_vpi.value
        
        # Volume persistence
        vol_persist_ok = dataframe['vol_persist'] >= self.min_vol_persistence.value
        
        # ===== COMBINE: Gaussian Context AND VPI Trigger =====
        
        long_condition = (
            gauss_bullish &
            vpi_bull_cross &
            rvol_ok &
            cum_bullish &
            vol_persist_ok
        )
        
        short_condition = (
            gauss_bearish &
            vpi_bear_cross &
            rvol_ok &
            cum_bearish &
            vol_persist_ok
        )
        
        dataframe.loc[long_condition, 'enter_long'] = 1
        dataframe.loc[long_condition, 'enter_tag'] = 'gauss_vpi_long'
        
        dataframe.loc[short_condition, 'enter_short'] = 1
        dataframe.loc[short_condition, 'enter_tag'] = 'gauss_vpi_short'
        
        return dataframe
    
    # ==========================================================================
    # EXIT SIGNALS — Gaussian Band Touch + VPI Reversal
    # ==========================================================================
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Exit when:
        - Gaussian trend flips (strongest signal)
        - Price touches opposite Gaussian band (dynamic TP)
        """
        # Exit long: trend flips to -1 or price hits upper band
        exit_long = (dataframe['gauss_trend'] == -1) & (dataframe['gauss_trend'].shift(1) == 1)
        
        if self.use_gaussian_tp.value:
            exit_long = exit_long | (
                (dataframe['close'] >= dataframe['gauss_upper']) &
                (dataframe['gauss_upper'].notna())
            )
        
        dataframe.loc[exit_long, 'exit_long'] = 1
        
        # Exit short: trend flips to 1 or price hits lower band
        exit_short = (dataframe['gauss_trend'] == 1) & (dataframe['gauss_trend'].shift(1) == -1)
        
        if self.use_gaussian_tp.value:
            exit_short = exit_short | (
                (dataframe['close'] <= dataframe['gauss_lower']) &
                (dataframe['gauss_lower'].notna())
            )
        
        dataframe.loc[exit_short, 'exit_short'] = 1
        
        return dataframe
    
    # ==========================================================================
    # CONFIRM TRADE ENTRY — Cooldown + Duplicate Prevention
    # ==========================================================================
    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                           time_in_force: str, current_time: datetime, entry_tag: str | None,
                           side: str, **kwargs) -> bool:
        open_trades = Trade.get_trades_proxy(is_open=True, pair=pair)
        if open_trades:
            return False
        
        cooldown_minutes = self.cooldown_candles.value * 5
        recent_trades = Trade.get_trades_proxy(is_open=False, pair=pair)
        
        if recent_trades:
            last_trade = recent_trades[-1]
            if last_trade.close_date:
                time_since_close = current_time - last_trade.close_date
                if time_since_close < timedelta(minutes=cooldown_minutes):
                    return False
        
        logger.info(f"[{pair}] ENTRY: {side.upper()} tag={entry_tag} rate={rate:.6f}")
        return True
    
    # ==========================================================================
    # CUSTOM STOPLOSS — Break Even
    # ==========================================================================
    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        if not self.be_enabled.value:
            return 1
        
        be_activated = trade.get_custom_data('be_activated', default=False)
        
        if trade.is_short:
            extremum = trade.min_rate if trade.min_rate is not None else current_rate
            price_move_pct = (trade.open_rate - extremum) / trade.open_rate
        else:
            extremum = trade.max_rate if trade.max_rate is not None else current_rate
            price_move_pct = (extremum - trade.open_rate) / trade.open_rate
        
        if not be_activated and price_move_pct >= self.be_trigger_pct.value:
            fee_buffer_pct = 0.001
            if trade.is_short:
                be_stop_price = trade.open_rate * (1 - fee_buffer_pct)
            else:
                be_stop_price = trade.open_rate * (1 + fee_buffer_pct)
            
            trade.set_custom_data('be_activated', True)
            trade.set_custom_data('be_stop_price', be_stop_price)
            be_activated = True
            logger.info(f"[{pair}] BE at {price_move_pct:.2%}. Stop: {be_stop_price:.6f}")
        
        if be_activated:
            be_stop_price = trade.get_custom_data('be_stop_price', default=trade.open_rate)
            return stoploss_from_absolute(
                be_stop_price, current_rate,
                is_short=trade.is_short, leverage=trade.leverage
            )
        
        return 1
    
    # ==========================================================================
    # ADJUST TRADE POSITION — TP1 Partial Exit
    # ==========================================================================
    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float,
                              min_stake: float | None, max_stake: float,
                              current_entry_rate: float, current_exit_rate: float,
                              current_entry_profit: float, current_exit_profit: float,
                              **kwargs) -> float | None | tuple[float | None, str | None]:
        if not self.tp1_enabled.value:
            return None
        
        if trade.is_short:
            extremum = trade.min_rate if trade.min_rate else current_rate
            price_move = (trade.open_rate - extremum) / trade.open_rate
        else:
            extremum = trade.max_rate if trade.max_rate else current_rate
            price_move = (extremum - trade.open_rate) / trade.open_rate
        
        if price_move >= self.tp1_pct.value:
            tp1_taken = trade.get_custom_data('tp1_taken', default=False)
            if tp1_taken:
                return None
            
            trade.set_custom_data('tp1_taken', True)
            
            close_amount = trade.amount * (self.tp1_amount.value / 100.0)
            sell_value = close_amount * current_exit_rate
            stake_change = sell_value / trade.leverage
            
            logger.info(f"[{trade.pair}] TP1 at {price_move:.2%}. Closing {self.tp1_amount.value:.0f}%.")
            return -stake_change, "tp1"
        
        return None
