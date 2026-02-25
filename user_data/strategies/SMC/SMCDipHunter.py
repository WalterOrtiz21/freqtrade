"""
SMCDipHunter Strategy
=====================

A Quant-driven Reversion Strategy ("Dip Hunter") that targets high-probability 
Order Blocks (OBs) backed by significant Volume Anomalies (RVOL).

Tesis:
Unlike standard SMC that waits for a Trend Change (CHoCH) confirmation, this strategy 
acts as a Liquidity Provider. It assumes that OBs formed with massive relative volume 
contain passive limit orders from institutions defending the zone. We "catch the knife" 
at the exact moment the price taps the OB, protected by a microscopic structural stop loss.

Key Features:
- STRICT VOLUMETRIC FILTER: Only OBs with > X times average volume are considered valid.
- NO CHoCH REQUIRED: Pure limit-hunting at the P/D array (OB boundary).
- AGGRESSIVE EXITS: Designed strictly for Mean-Reversion. Quick partial TPs and trailing.
- STRUCTURAL INVALIDATION: If a candle closes below the OB bottom, the thesis is dead.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pandas import DataFrame
from typing import Optional, Dict, List
import os

logger = logging.getLogger(__name__)

from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter, BooleanParameter, CategoricalParameter, merge_informative_pair
from freqtrade.persistence import Trade
import talib.abstract as ta
import pandas_ta as pta

# Import the modified Numba library (Must have the Volume outputs!)
try:
    from user_data.strategies.SMC.smc_luxalgo_numba import SMCLuxAlgoNumba as SMCLuxAlgo
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from smc_luxalgo_numba import SMCLuxAlgoNumba as SMCLuxAlgo


class SMCDipHunter(IStrategy):
    INTERFACE_VERSION = 3

    # ==========================================================================
    # FIXED PARAMETERS (Adjusted by leverage in bot_start)
    # ==========================================================================
    minimal_roi = {"0": 1.0}  # Overridden by custom exits
    stoploss = -0.02          # Base -2%, but custom_stoploss handles structural SL
    timeframe = '15m'
    
    trailing_stop = False
    use_exit_signal = True
    use_custom_stoploss = True
    position_adjustment_enable = True
    max_open_trades = 3
    startup_candle_count: int = 250
    can_short = True

    # ==========================================================================
    # STRATEGY PARAMETERS (Optimizables)
    # ==========================================================================
    
    # HTF TIMEFRAMES (Configurable filter)
    HTF_OPTIONS = ['none', '15m', '30m', '1h', '2h', '4h', '8h', '1d']
    htf_1 = CategoricalParameter(HTF_OPTIONS, default='1h', space='buy', optimize=True)
    htf_2 = CategoricalParameter(HTF_OPTIONS, default='4h', space='buy', optimize=True)
    
    # Enable/Disable HTF filtering strictly
    use_htf_filter = BooleanParameter(default=True, space='buy', optimize=True)
    
    # 1. SMC Structure
    internal_length = IntParameter(3, 10, default=5, space='buy', optimize=True)
    swing_length = IntParameter(20, 100, default=50, space='buy', optimize=True)
    
    # 2. Volumetric Anomaly Filter (RVOL)
    # The volume of the OB creation candle must be > X times the SMA of volume
    vol_sma_period = IntParameter(10, 50, default=20, space='buy', optimize=True)
    ob_rvol_threshold = DecimalParameter(1.2, 3.5, default=1.8, decimals=1, space='buy', optimize=True)
    
    # 3. Oversold/Overbought Context (Confirm the "Dip")
    # For a Long, price must be dropping aggressively into the OB. 
    rsi_oversold = IntParameter(20, 45, default=35, space='buy', optimize=True)
    rsi_overbought = IntParameter(55, 80, default=65, space='buy', optimize=True)
    
    # 4. Exits & Risk Management
    # Strict structural invalidation: Close trade if price closes beyond OB boundary
    strict_ob_invalidation = BooleanParameter(default=True, space='sell', optimize=True)
    
    # Adaptive Exit: Cut losses if market structure shifts against us after the bounce
    exit_on_internal_choch = BooleanParameter(default=True, space='sell', optimize=True)
    
    # Buffer outside the OB box before killing the trade (to survive wicks)
    sl_buffer_pct = DecimalParameter(0.001, 0.005, default=0.002, decimals=3, space='sell', optimize=True)
    
    # Aggressive Scalp/Mean-Reversion TP1
    enable_tp1 = BooleanParameter(default=True, space='sell', optimize=True)
    tp1_pct = DecimalParameter(0.005, 0.03, default=0.015, decimals=3, space='sell', optimize=True)
    tp1_amount = DecimalParameter(10.0, 100.0, default=50.0, space='sell', optimize=True)
    
    # Independent Break Even Logic
    enable_be = BooleanParameter(default=True, space='sell', optimize=True)
    be_activation_pct = DecimalParameter(0.01, 0.05, default=0.02, decimals=3, space='sell', optimize=True)
    be_buffer_pct = DecimalParameter(0.001, 0.005, default=0.002, decimals=3, space='sell', optimize=True)


    def __init__(self, config: dict) -> None:
        super().__init__(config)

    def bot_start(self, **kwargs) -> None:
        """Apply leverage to Risk metrics"""
        config_leverage = self.config.get('leverage', 1.0)
        self.stoploss = self.config.get('stoploss', -0.02) * config_leverage
        
        logger.info(
            f"SMCDipHunter Configured:"
            f"\n  Leverage: {config_leverage}x"
            f"\n  Base Stoploss: {self.config.get('stoploss', -0.02):.2%}"
            f"\n  RVOL OB Threshold: {self.ob_rvol_threshold.value}x"
        )

    def _get_active_htf_list(self) -> list:
        """Build list of active HTF timeframes from parameters."""
        htf_list = []
        if self.htf_1.value != 'none':
            htf_list.append(self.htf_1.value)
        if self.htf_2.value != 'none':
            htf_list.append(self.htf_2.value)
        return htf_list

    def informative_pairs(self):
        """Define pairs to load based on htf_1 and htf_2 parameters."""
        pairs = self.dp.current_whitelist()
        htf_list = self._get_active_htf_list()
        
        informative_pairs = []
        for tf in htf_list:
            informative_pairs += [(pair, tf) for pair in pairs]
        
        return informative_pairs

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: str,
                 side: str, **kwargs) -> float:
        """
        Require Freqtrade to explicitly set leverage on the exchange.
        Normally reads 'leverage' from config, defaults to 10.0 if not found.
        """
        return self.config.get('leverage', 10.0)

    # ==========================================================================
    # INDICATOR CALCULATION
    # ==========================================================================
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Calculate volumetric OBs and momentum features."""
        
        # 1. Market Baseline Volume
        dataframe['vol_sma'] = ta.SMA(dataframe['volume'], timeperiod=self.vol_sma_period.value)
        # Avoid division by zero
        dataframe['vol_sma'] = dataframe['vol_sma'].replace(0, np.nan)
        
        # 2. RSI for Momentum Context (Is it a real dip?)
        dataframe['rsi'] = ta.RSI(dataframe['close'], timeperiod=14)
        
        # 3. Numba SMC Engine (Extracting Volumetric Zones)
        smc = SMCLuxAlgo(
            dataframe,
            internal_length=self.internal_length.value,
            swing_length=self.swing_length.value
        )
        signals = smc.get_signals()
        
        # Merge purely the columns we need for Direct Limit Hunting
        # We need OB Top/Bottom and crucially, the OB VOLUME.
        # We We also need Internal CHoCH to cut turnaround losses.
        target_cols = [
            'active_bullish_ob_top', 'active_bullish_ob_bottom', 'active_bullish_ob_vol',
            'active_bearish_ob_top', 'active_bearish_ob_bottom', 'active_bearish_ob_vol',
            'swing_high', 'swing_low',
            'internal_choch_bullish', 'internal_choch_bearish' # For smart exits
        ]
        
        for col in target_cols:
            if col in signals.columns:
                # Shift zones to prevent Lookahead Bias when trading on the current candle
                dataframe[col] = signals[col].shift(1).fillna(0)
            else:
                logger.warning(f"CRITICAL: Volume column {col} missing from Numba backend!")
                dataframe[col] = 0

        # Calculate RVOL for active Order Blocks
        # RVOL = Volume del OB / Volumen Promedio del mercado HOY
        dataframe['bull_ob_rvol'] = dataframe['active_bullish_ob_vol'] / dataframe['vol_sma']
        dataframe['bear_ob_rvol'] = dataframe['active_bearish_ob_vol'] / dataframe['vol_sma']
        
        # --- 4. HTF Hull MA Calculations ---
        if self.dp and self.use_htf_filter.value:
            htf_list = self._get_active_htf_list()
            for htf in htf_list:
                try:
                    inf_htf = self.dp.get_pair_dataframe(metadata['pair'], htf)
                    if inf_htf.empty:
                        continue
                        
                    # Calculate Hull MA 200 on HTF
                    inf_htf['hma_200'] = pta.hma(inf_htf['close'], length=200)
                    
                    # Define Macro Trend based on price vs HMA
                    # 1 = Bullish Regime, -1 = Bearish Regime
                    trend_col = 'macro_trend'
                    inf_htf[trend_col] = np.where(inf_htf['close'] > inf_htf['hma_200'], 1, -1)
                    
                    # Merge using Freqtrade's secure logic to prevent Lookahead Bias
                    # This maps 4H data to the exact 15m candle where the 4H has FINISHED closing.
                    dataframe = merge_informative_pair(dataframe, inf_htf, self.timeframe, htf, ffill=True)
                    
                except Exception as e:
                    logger.error(f"Error processing HTF {htf}: {e}")
        
        return dataframe

    # ==========================================================================
    # ENTRY LOGIC
    # ==========================================================================
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        DIP HUNTING LOGIC:
        Enter instantly when price taps a High-Volume OB while oversold/overbought.
        No CHoCH required.
        """
        
        # ===== LONG ENTRY (Buying the Dip) =====
        
        # 1. Zone Tap: Low dips into the Bullish OB, but close stays above the invalidation line 
        in_bull_zone = (
            (dataframe['active_bullish_ob_top'] > 0) &
            (dataframe['low'] <= dataframe['active_bullish_ob_top']) &
            (dataframe['close'] >= dataframe['active_bullish_ob_bottom']) # Has not structurally broken yet
        )
        
        # 2. Volumetric Anomaly: The OB must have been created with massive volume
        high_vol_bull = dataframe['bull_ob_rvol'] >= self.ob_rvol_threshold.value
        
        # 3. Context: Market must actually be "dipping" (RSI Oversold)
        is_oversold = dataframe['rsi'] <= self.rsi_oversold.value
        
        # 4. Macro Trend Filter (Hull MA 200 on HTF)
        # We only catch bulls dips if the macro trend is green.
        macro_bullish = pd.Series(True, index=dataframe.index)
        if self.use_htf_filter.value:
            # Check all active HTFs - require them to be bullish (1) or unknown (0, e.g. startup)
            if self.htf_1.value != 'none':
                # Freqtrade appends the timeframe to the column during merge_informative_pair
                col_name = f'macro_trend_{self.htf_1.value}'
                if col_name in dataframe.columns:
                    macro_bullish &= (dataframe[col_name] >= 0)
            if self.htf_2.value != 'none':
                col_name = f'macro_trend_{self.htf_2.value}'
                if col_name in dataframe.columns:
                    macro_bullish &= (dataframe[col_name] >= 0)
        
        # Combined Long Signal
        dataframe.loc[
            in_bull_zone & high_vol_bull & is_oversold & macro_bullish,
            ['enter_long', 'enter_tag']
        ] = (1, 'dip_bull_ob')
        

        # ===== SHORT ENTRY (Selling the Rip) =====
        
        in_bear_zone = (
            (dataframe['active_bearish_ob_top'] > 0) &
            (dataframe['high'] >= dataframe['active_bearish_ob_bottom']) &
            (dataframe['close'] <= dataframe['active_bearish_ob_top']) # Has not structurally broken yet
        )
        
        high_vol_bear = dataframe['bear_ob_rvol'] >= self.ob_rvol_threshold.value
        is_overbought = dataframe['rsi'] >= self.rsi_overbought.value
        
        # Macro Trend Filter (Sell the Rip only if macro is bearish)
        macro_bearish = pd.Series(True, index=dataframe.index)
        if self.use_htf_filter.value:
            if self.htf_1.value != 'none':
                col_name = f'macro_trend_{self.htf_1.value}'
                if col_name in dataframe.columns:
                    macro_bearish &= (dataframe[col_name] <= 0)
            if self.htf_2.value != 'none':
                col_name = f'macro_trend_{self.htf_2.value}'
                if col_name in dataframe.columns:
                    macro_bearish &= (dataframe[col_name] <= 0)
        
        # Combined Short Signal
        dataframe.loc[
            in_bear_zone & high_vol_bear & is_overbought & macro_bearish,
            ['enter_short', 'enter_tag']
        ] = (1, 'rip_bear_ob')

        return dataframe


    # ==========================================================================
    # EXIT LOGIC & RISK MANAGEMENT
    # ==========================================================================
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Structural Exit Logic (Reaching opposite zones)."""
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0
        
        # Take Profit Estructural: 
        # Si un Long choca contra un OB Bajista y frena, salimos.
        touch_bear_zone = (
            (dataframe['active_bearish_ob_top'] > 0) &
            (dataframe['high'] >= dataframe['active_bearish_ob_bottom'])
        )
        dataframe.loc[touch_bear_zone, ['exit_long', 'exit_tag']] = (1, 'struct_bear_ob_tp')
        
        # Take Profit Estructural Short:
        touch_bull_zone = (
            (dataframe['active_bullish_ob_top'] > 0) &
            (dataframe['low'] <= dataframe['active_bullish_ob_top'])
        )
        dataframe.loc[touch_bull_zone, ['exit_short', 'exit_tag']] = (1, 'struct_bull_ob_tp')
        
        # Smart adaptive exit: Rebound failed, market structure shifted against us.
        if self.exit_on_internal_choch.value:
            # If we are LONG and the market makes a bearish internal change of character
            dataframe.loc[dataframe['internal_choch_bearish'] == 1, ['exit_long', 'exit_tag']] = (1, 'exit_int_choch_bear')
            
            # If we are SHORT and the market makes a bullish internal change of character
            dataframe.loc[dataframe['internal_choch_bullish'] == 1, ['exit_short', 'exit_tag']] = (1, 'exit_int_choch_bull')
        
        return dataframe

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        # --- Break Even Logic ---
        # If BE is enabled, check if the trade has reached the activation distance.
        if self.enable_be.value:
            # Calculate actual PRICE movement based on Freqtrade's ROE current_profit
            price_movement_pct = current_profit / trade.leverage
            
            # Record maximum observed profit to track high-water mark safely
            if not hasattr(self, 'trade_max_profit'):
                self.trade_max_profit = {}
                
            trade_key = f"{pair}_{trade.open_time_utc if hasattr(trade, 'open_time_utc') else trade.id}"
            max_profit = self.trade_max_profit.get(trade_key, 0.0)
            
            if price_movement_pct > max_profit:
                self.trade_max_profit[trade_key] = price_movement_pct
                max_profit = price_movement_pct

            # If we ever reached the BE activation threshold...
            if max_profit >= self.be_activation_pct.value:
                # We want to lock in a small buffer above exactly zero to cover fees
                # Break Even Price = Entry + (Entry * Buffer)
                if trade.is_short:
                    be_price = trade.open_rate * (1 - self.be_buffer_pct.value)
                    if current_rate >= be_price:
                        logger.info(f"{pair} Break Even Hit (Short) at {current_rate}")
                        return (trade.open_rate - current_rate) / trade.open_rate
                else:
                    be_price = trade.open_rate * (1 + self.be_buffer_pct.value)
                    if current_rate <= be_price:
                        logger.info(f"{pair} Break Even Hit (Long) at {current_rate}")
                        return (current_rate - trade.open_rate) / trade.open_rate

        # --- Micro-Structural Stop Loss (OB Defender) ---
        if not self.strict_ob_invalidation.value:
            return 1.0 # Deferred to standard SL
            
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1]
        
        buffer = self.sl_buffer_pct.value
        
        if trade.is_short:
            # We are short, defending the Bearish OB Top.
            ob_top = last_candle['active_bearish_ob_top']
            if ob_top > 0:
                invalidation_price = ob_top * (1 + buffer)
                # If current rate is above the invalidation price, trigger stop loss
                if current_rate > invalidation_price:
                    logger.info(f"{pair} Short Invalidated. Broken Bear OB Top at {current_rate}")
                    # Return exactly the negative profit representing this distance
                    return (trade.open_rate - current_rate) / trade.open_rate
        else:
            # We are long, defending the Bullish OB Bottom.
            ob_btm = last_candle['active_bullish_ob_bottom']
            if ob_btm > 0:
                invalidation_price = ob_btm * (1 - buffer)
                if current_rate < invalidation_price:
                    logger.info(f"{pair} Long Invalidated. Broken Bull OB Btm at {current_rate}")
                    return (current_rate - trade.open_rate) / trade.open_rate
                    
        return 1.0

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs):
        """Standard trailing / fallback logic."""
        return None

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float,
                              min_stake: Optional[float], max_stake: float,
                              current_entry_rate: float, current_exit_rate: float,
                              current_entry_profit: float, current_exit_profit: float,
                              **kwargs) -> Optional[float]:
        """
        Tiered Take Profit (Fast ROI).
        Since Mean-Reversion is risky, bag profits on the first strong bounce.
        """
        # Calculate actual PRICE movement based on Freqtrade's ROE current_profit
        # If current_profit = 20% (0.20) at 10x leverage, price_movement = 2% (0.02)
        price_movement_pct = current_profit / trade.leverage
        
        # Avoid multiple partials if we already did one
        if not self.enable_tp1.value or trade.nr_of_successful_exits > 0:
            return None

        # Take Profit 1 (Fast Scalp based on Price Movement)
        if price_movement_pct >= self.tp1_pct.value:
            amount_to_close = trade.stake_amount * (self.tp1_amount.value / 100.0)
            logger.info(f"{trade.pair} Reached DipHunter TP1 ({price_movement_pct:.2%}). Closing {self.tp1_amount.value}%")
            return -(amount_to_close)

        return None
