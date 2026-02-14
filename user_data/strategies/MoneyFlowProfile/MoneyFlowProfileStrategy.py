"""
Money Flow Profile Strategy
===========================

Standalone strategy based on LuxAlgo's Money Flow Profile indicator.

Thesis:
- Price is attracted to High Volume Nodes (HVN/POC) and moves quickly through 
  Low Volume Nodes (LVN)
- Combined with sentiment (bullish/bearish per level), we can anticipate direction

Entry Logic:
- LONG: Price in LVN + Bullish sentiment at level + Price below POC
- SHORT: Price in LVN + Bearish sentiment at level + Price above POC

Exit Logic:
- TP1: When price reaches POC
- TP2: When price reaches next HVN
- SL: Based on percentage or next LVN in opposite direction
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from pandas import DataFrame

from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy, DecimalParameter, IntParameter, BooleanParameter, CategoricalParameter

# Import profile calculator (same directory)
try:
    from money_flow_profile import MoneyFlowProfile
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from money_flow_profile import MoneyFlowProfile

logger = logging.getLogger(__name__)


class MoneyFlowProfileStrategy(IStrategy):
    """
    Money Flow Profile Strategy
    
    Uses volume distribution analysis to identify optimal entry points
    in Low Volume Nodes with sentiment confirmation.
    """
    
    INTERFACE_VERSION = 3
    
    # ROI - Let the strategy manage exits via TPs
    minimal_roi = {"0": 1.0}  # Disabled, using custom exits
    
    # Stoploss
    stoploss = -0.03  # 3% default, can be overridden in JSON
    
    # Timeframe
    timeframe = '1h'
    
    # Startup candles needed for profile calculation
    startup_candle_count = 210  # lookback + buffer
    
    # Trailing stop disabled - using custom exit logic
    trailing_stop = False
    
    # Can short
    can_short = True
    
    # Use custom stoploss
    use_custom_stoploss = True
    
    # =========================================================================
    # STRATEGY PARAMETERS (Hyperopt Optimizable)
    # =========================================================================
    
    # Profile Settings
    lookback_period = IntParameter(50, 300, default=200, space='buy', optimize=True)
    num_rows = IntParameter(15, 50, default=25, space='buy', optimize=True)
    
    # Node Classification Thresholds
    hvn_threshold = DecimalParameter(0.45, 0.75, default=0.53, decimals=2, space='buy', optimize=True)
    lvn_threshold = DecimalParameter(0.20, 0.45, default=0.37, decimals=2, space='buy', optimize=True)
    
    # Profile Type
    use_money_flow = BooleanParameter(default=False, space='buy', optimize=True)
    sentiment_method = CategoricalParameter(['polarity', 'pressure'], default='polarity', space='buy', optimize=True)
    
    # Entry Filters
    require_lvn = BooleanParameter(default=True, space='buy', optimize=False)
    require_sentiment_align = BooleanParameter(default=True, space='buy', optimize=True)
    require_poc_direction = BooleanParameter(default=True, space='buy', optimize=True)
    
    # Take Profit Settings (based on price movement, not leveraged profit)
    tp1_enabled = BooleanParameter(default=True, space='sell', optimize=False)
    tp1_pct = DecimalParameter(0.005, 0.02, default=0.01, decimals=3, space='sell', optimize=True)
    tp1_amount = DecimalParameter(20, 60, default=40, decimals=0, space='sell', optimize=True)
    
    tp2_pct = DecimalParameter(0.015, 0.05, default=0.025, decimals=3, space='sell', optimize=True)
    
    # Logging
    enable_logging = BooleanParameter(default=True, space='buy', optimize=False)
    
    # =========================================================================
    # INITIALIZATION
    # =========================================================================
    
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._profile_cache = {}
    
    def bot_start(self, **kwargs) -> None:
        """Strategy startup."""
        if self.enable_logging.value:
            logger.info(
                f"MoneyFlowProfile Strategy Started | "
                f"Lookback: {self.lookback_period.value}, Rows: {self.num_rows.value}, "
                f"HVN: {self.hvn_threshold.value}, LVN: {self.lvn_threshold.value}"
            )
    
    # =========================================================================
    # INDICATORS
    # =========================================================================
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Calculate Money Flow Profile indicators."""
        
        # Initialize profile calculator
        profile_calc = MoneyFlowProfile(
            num_rows=self.num_rows.value,
            hvn_threshold=self.hvn_threshold.value,
            lvn_threshold=self.lvn_threshold.value,
            use_money_flow=self.use_money_flow.value,
            sentiment_method=self.sentiment_method.value
        )
        
        # Calculate profile for dataframe
        dataframe = profile_calc.calculate_for_dataframe(
            dataframe, 
            lookback=self.lookback_period.value
        )
        
        if self.enable_logging.value:
            # Log some stats
            lvn_count = dataframe['mfp_in_lvn'].sum()
            hvn_count = dataframe['mfp_in_hvn'].sum()
            logger.info(
                f"MFP Indicators for {metadata['pair']}: "
                f"LVN candles: {lvn_count}, HVN candles: {hvn_count}"
            )
        
        return dataframe
    
    # =========================================================================
    # ENTRY SIGNALS
    # =========================================================================
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Generate entry signals based on Money Flow Profile."""
        
        # Initialize signal columns
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        
        # Base conditions
        has_data = dataframe['mfp_poc'].notna()
        
        # LVN condition (price is in low volume node)
        if self.require_lvn.value:
            in_lvn = dataframe['mfp_in_lvn'] == True
        else:
            in_lvn = pd.Series(True, index=dataframe.index)
        
        # Sentiment at current level
        bullish_sentiment = dataframe['mfp_level_sentiment'] > 0
        bearish_sentiment = dataframe['mfp_level_sentiment'] < 0
        
        if not self.require_sentiment_align.value:
            bullish_sentiment = pd.Series(True, index=dataframe.index)
            bearish_sentiment = pd.Series(True, index=dataframe.index)
        
        # Price vs POC
        below_poc = dataframe['mfp_price_vs_poc'] < 0
        above_poc = dataframe['mfp_price_vs_poc'] > 0
        
        if not self.require_poc_direction.value:
            below_poc = pd.Series(True, index=dataframe.index)
            above_poc = pd.Series(True, index=dataframe.index)
        
        # ===== LONG CONDITIONS =====
        # In LVN + Bullish sentiment + Below POC (target: move up to POC)
        long_condition = (
            has_data &
            in_lvn &
            bullish_sentiment &
            below_poc
        )
        
        # ===== SHORT CONDITIONS =====
        # In LVN + Bearish sentiment + Above POC (target: move down to POC)
        short_condition = (
            has_data &
            in_lvn &
            bearish_sentiment &
            above_poc
        )
        
        # Apply signals
        dataframe.loc[long_condition, 'enter_long'] = 1
        dataframe.loc[short_condition, 'enter_short'] = 1
        
        # Logging
        if self.enable_logging.value:
            logger.info(
                f"MFP Signals for {metadata['pair']}: "
                f"Long={long_condition.sum()}, Short={short_condition.sum()}"
            )
        
        return dataframe
    
    # =========================================================================
    # EXIT SIGNALS
    # =========================================================================
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Generate exit signals when price reaches HVN/POC."""
        
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0
        
        # Exit when price enters HVN (target reached)
        in_hvn = dataframe['mfp_in_hvn'] == True
        
        # For longs: exit when reaching HVN and was previously in LVN
        # For shorts: same logic
        # This is a simple version - more sophisticated would track entry point
        
        dataframe.loc[in_hvn, 'exit_long'] = 1
        dataframe.loc[in_hvn, 'exit_short'] = 1
        
        return dataframe
    
    # =========================================================================
    # TRADE MANAGEMENT
    # =========================================================================
    
    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                           time_in_force: str, current_time: datetime, entry_tag: str | None,
                           side: str, **kwargs) -> bool:
        """
        Prevent duplicate entries on same pair.
        """
        # Check for existing open trades on this pair
        open_trades = Trade.get_trades_proxy(pair=pair, is_open=True)
        if open_trades:
            if self.enable_logging.value:
                logger.info(f"Entry blocked for {pair}: Already have an open trade")
            return False
        
        return True
    
    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Custom stoploss with break-even after TP1.
        """
        # Check if TP1 was taken (break-even activated)
        tp1_taken = trade.get_custom_data('tp1_taken', default=False)
        
        if tp1_taken:
            # Move stop to break-even + small buffer
            fee_buffer = 0.001  # 0.1% buffer for fees
            
            if trade.is_short:
                be_price = trade.open_rate * (1 - fee_buffer)
            else:
                be_price = trade.open_rate * (1 + fee_buffer)
            
            # Calculate stoploss from absolute price
            if trade.is_short:
                sl = (be_price - current_rate) / current_rate
            else:
                sl = (current_rate - be_price) / current_rate
            
            return -abs(sl)
        
        return 1  # Use default stoploss
    
    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float,
                              min_stake: float | None, max_stake: float,
                              current_entry_rate: float, current_exit_rate: float,
                              current_entry_profit: float, current_exit_profit: float,
                              **kwargs) -> float | None | tuple[float | None, str | None]:
        """
        Partial Take Profit at TP1.
        """
        if not self.tp1_enabled.value:
            return None
        
        # Calculate price movement (not leveraged profit)
        if trade.is_short:
            price_movement = (trade.open_rate - current_rate) / trade.open_rate
        else:
            price_movement = (current_rate - trade.open_rate) / trade.open_rate
        
        # Check if TP1 already taken
        tp1_taken = trade.get_custom_data('tp1_taken', default=False)
        
        if not tp1_taken and price_movement > self.tp1_pct.value:
            # Mark as taken
            trade.set_custom_data('tp1_taken', True)
            
            # Calculate amount to close
            close_pct = self.tp1_amount.value / 100.0
            close_amount = trade.amount * close_pct
            sell_value = close_amount * current_rate
            stake_change = sell_value / trade.leverage
            
            if self.enable_logging.value:
                logger.info(
                    f"TP1 for {trade.pair}: price moved {price_movement:.2%}, "
                    f"closing {self.tp1_amount.value:.0f}%"
                )
            
            return (-stake_change, "TP1_partial")
        
        return None
    
    def custom_exit(self, pair: str, trade: Trade, current_time: datetime,
                   current_rate: float, current_profit: float, **kwargs) -> Optional[str]:
        """
        Full exit at TP2.
        """
        # Calculate price movement
        if trade.is_short:
            price_movement = (trade.open_rate - current_rate) / trade.open_rate
        else:
            price_movement = (current_rate - trade.open_rate) / trade.open_rate
        
        if price_movement >= self.tp2_pct.value:
            return "TP2_exit"
        
        return None
    
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: str,
                 side: str, **kwargs) -> float:
        """Get leverage from config."""
        return self.config.get('leverage', 5.0)
