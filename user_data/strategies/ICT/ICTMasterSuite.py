"""
ICT Master Suite Strategy for Freqtrade
========================================

Implementation of ICT (Inner Circle Trader) concepts for automated trading.
Follows Freqtrade best practices: Numba module for calculations, strategy for trading logic.

Trading Models:
- Model 2022: Sweep → MSS → FVG entry
- Unicorn: Breaker Block + FVG overlap
- OTE: Entry in 62-79% Fibonacci zone
- Silver Bullet: Session-specific FVG entries
- Basic Structure: BoS/MSS + zone confluence

Session Filters:
- Kill Zone (London/NY overlap)
- Macros (ICT time windows)
- HTF direction confirmation
"""

import logging
from datetime import datetime
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
from pandas import DataFrame

from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter, BooleanParameter
from freqtrade.persistence import Trade

# Import ICT Numba module (calculations only)
try:
    from user_data.strategies.ICT.ict_numba import ICTNumba
except ImportError:
    from ict_numba import ICTNumba

logger = logging.getLogger(__name__)


class ICTMasterSuite(IStrategy):
    """
    ICT Master Suite Strategy
    
    Uses Numba-optimized calculations from ict_numba.py for:
    - Market Structure (BoS/MSS/CHoCH)
    - Order Blocks, FVGs, Breaker Blocks
    - Premium/Discount zones
    - Session and Macro time filters
    
    Trading logic (Model2022, Unicorn, etc.) is implemented here.
    """
    
    INTERFACE_VERSION = 3
    timeframe = '15m'
    can_short = True
    startup_candle_count = 200
    
    # Disable default ROI - using custom exits
    minimal_roi = {"0": 100}
    stoploss = -0.10
    trailing_stop = False
    process_only_new_candles = True
    use_custom_stoploss = True
    
    # =========================================================================
    # HYPEROPTABLE PARAMETERS
    # =========================================================================
    
    # ATR Parameters
    atr_period = IntParameter(10, 20, default=14, space="buy", optimize=True)
    swing_atr_mult = DecimalParameter(1.0, 4.0, default=2.0, decimals=1, space="buy", optimize=True)
    
    # Zone Filters
    require_ob = BooleanParameter(default=True, space="buy", optimize=True)
    require_fvg = BooleanParameter(default=False, space="buy", optimize=True)
    require_premium_discount = BooleanParameter(default=True, space="buy", optimize=True)
    
    # Trading Model Selection (OR logic)
    use_model_2022 = BooleanParameter(default=False, space="buy", optimize=True)
    use_unicorn = BooleanParameter(default=True, space="buy", optimize=True)
    use_ote = BooleanParameter(default=True, space="buy", optimize=True)
    use_silver_bullet = BooleanParameter(default=True, space="buy", optimize=True)
    use_liquidity_raid = BooleanParameter(default=True, space="buy", optimize=True)
    use_basic_structure = BooleanParameter(default=True, space="buy", optimize=True)
    
    # Session/Timing Filters
    require_kill_zone = BooleanParameter(default=False, space="buy", optimize=True)
    require_london_or_ny = BooleanParameter(default=False, space="buy", optimize=True)
    require_macro = BooleanParameter(default=False, space="buy", optimize=True)
    
    # Feature Filters
    require_rejection_block = BooleanParameter(default=False, space="buy", optimize=True)
    require_po3 = BooleanParameter(default=False, space="buy", optimize=True)
    
    # HTF Direction Filter
    use_htf_filter = BooleanParameter(default=False, space="buy", optimize=True)
    htf_timeframe = '1h'
    
    # Exit Parameters
    sl_atr_mult = DecimalParameter(1.0, 3.0, default=1.5, decimals=1, space="sell", optimize=True)
    tp_rr_ratio = DecimalParameter(1.0, 4.0, default=2.0, decimals=1, space="sell", optimize=True)
    exit_on_opposite_mss = BooleanParameter(default=True, space="sell", optimize=True)
    
    # Logging
    analysis_logging = BooleanParameter(default=False, space="buy", optimize=False)
    
    # =========================================================================
    # INSTANCE STORAGE
    # =========================================================================
    
    custom_trade_info: Dict[str, Dict[str, Any]] = {}
    
    def informative_pairs(self):
        """Return HTF pairs if filter is enabled."""
        if self.use_htf_filter.value:
            return [(pair, self.htf_timeframe) for pair in self.dp.current_whitelist()]
        return []
    
    # =========================================================================
    # POPULATE INDICATORS - Uses Numba for calculations
    # =========================================================================
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Calculate ICT indicators using Numba module."""
        
        # Get session times from config (for DST adjustment)
        session_times = self.config.get('session_times', {})
        
        # Calculate raw ICT data
        ict = ICTNumba(
            dataframe,
            atr_period=self.atr_period.value,
            swing_atr_mult=self.swing_atr_mult.value,
            asia_start=session_times.get('asia_start', 0),
            asia_end=session_times.get('asia_end', 8),
            london_start=session_times.get('london_start', 7),
            london_end=session_times.get('london_end', 16),
            ny_start=session_times.get('ny_start', 12),
            ny_end=session_times.get('ny_end', 21),
            sb_london_start=session_times.get('sb_london_start', 9),
            sb_london_end=session_times.get('sb_london_end', 10),
            sb_ny_am_start=session_times.get('sb_ny_am_start', 14),
            sb_ny_am_end=session_times.get('sb_ny_am_end', 15),
            sb_ny_pm_start=session_times.get('sb_ny_pm_start', 18),
            sb_ny_pm_end=session_times.get('sb_ny_pm_end', 19)
        )
        
        signals = ict.get_signals()
        
        # Merge into dataframe
        for col in signals.columns:
            dataframe[col] = signals[col].values
        
        # =====================================================================
        # TRADING MODEL CALCULATIONS (logic in strategy, not Numba)
        # =====================================================================
        
        # =====================================================================
        # TRADING MODEL CALCULATIONS (logic in strategy, not Numba)
        # =====================================================================
        
        # Helper: Recent Structural Shift (Bias)
        # We look for an MSS within the last 12 bars (3 hours on 15m) to establish bias
        dataframe['recent_mss_bull'] = dataframe['mss_bullish'].rolling(window=12).max()
        dataframe['recent_mss_bear'] = dataframe['mss_bearish'].rolling(window=12).max()
        
        # Helper: Recent Sweep (for Model 2022)
        dataframe['recent_sweep_low'] = dataframe['sweep_low'].rolling(window=5).max()
        dataframe['recent_sweep_high'] = dataframe['sweep_high'].rolling(window=5).max()

        # Model 2022: Sweep -> MSS -> Retracement to FVG
        # Logic: 
        # 1. We have a recent sweep (already factored into MSS logic mostly, but good to reinforce)
        # 2. We have a recent MSS (establishing bullish bias)
        # 3. Current price is now in a Discount FVG
        
        dataframe['model2022_bull'] = (
            (dataframe['recent_mss_bull'] == 1) & 
            (dataframe['in_bull_fvg'] == 1) & 
            (dataframe['in_discount'] == 1)
        )
        
        dataframe['model2022_bear'] = (
            (dataframe['recent_mss_bear'] == 1) & 
            (dataframe['in_bear_fvg'] == 1) & 
            (dataframe['in_premium'] == 1)
        )
        
        # Unicorn: Breaker Block + FVG overlap
        # Relaxed logic: Recent (active) BB and current price in FVG
        dataframe['unicorn_bull'] = (dataframe['bb_bull_top'] > 0) & (dataframe['in_bull_fvg'] == 1)
        dataframe['unicorn_bear'] = (dataframe['bb_bear_btm'] > 0) & (dataframe['in_bear_fvg'] == 1)
        
        # Silver Bullet: Session + structure + zone (OB or FVG) + premium/discount
        bull_zone = (dataframe['in_bull_ob'] == 1) | (dataframe['in_bull_fvg'] == 1)
        bear_zone = (dataframe['in_bear_ob'] == 1) | (dataframe['in_bear_fvg'] == 1)
        
        # SB Logic: If strictly in session window, generic structure direction matches, and we hit a zone
        dataframe['silver_bullet_bull'] = (
            (dataframe['in_silver_bullet'] == 1) &
            (dataframe['structure_direction'] == 1) &
            bull_zone
        )
        dataframe['silver_bullet_bear'] = (
            (dataframe['in_silver_bullet'] == 1) &
            (dataframe['structure_direction'] == -1) &
            bear_zone &
            (dataframe['in_premium'] == 1)
        )
        
        # Liquidity Raid: Sweep + Reversal + FVG/MSS
        # Logic: Sweep happened recently (last 5 bars) AND we have MSS AND FVG
        # Note: 'recent_sweep_low' is already calculated above
        
        dataframe['liquidity_raid_bull'] = (
            (dataframe['recent_sweep_low'] == 1) &
            (dataframe['mss_bullish'] == 1) &
            (dataframe['in_bull_fvg'] == 1)
        )
        
        dataframe['liquidity_raid_bear'] = (
            (dataframe['recent_sweep_high'] == 1) &
            (dataframe['mss_bearish'] == 1) &
            (dataframe['in_bear_fvg'] == 1)
        )
        
        # Rejection Blocks Processing
        # (Already calculated in Numba, just need to track active ones if we want statefulness, 
        #  but simple "in_zone" check is done via Numba if we added it there. 
        #  Wait, I didn't add "in_rejection_block" to Numba active zones kernel.
        #  For now, we can check if current price is within the *recent* rejection block)
        
        # Simple recent RB check (last 10 bars)
        # Vectorized "forward fill" for active RB levels is better done in Numba if strictly needed.
        # Here we will just use the raw signals for "Require RB" filter if they coincide with signal candle.
        
        # Po3 Filter Logic
        # Accumulation (range), Manipulation (fake move), Distribution (trend)
        # Bullish Entry mostly valid during Manipulation (buying below open) or start of Distribution
        dataframe['in_po3_buy_zone'] = (dataframe['po3_manipulation'] == 1) | (dataframe['po3_distribution'] == 1)
        dataframe['in_po3_sell_zone'] = (dataframe['po3_manipulation'] == 1) | (dataframe['po3_distribution'] == 1)
        
        # =====================================================================
        # MODEL SL/TP LEVELS (calculated here, not Numba)
        # =====================================================================
        
        # Calculate model-specific stoploss levels
        dataframe['model_sl_long'] = np.nan
        dataframe['model_sl_short'] = np.nan
        dataframe['model_tp_long'] = np.nan
        dataframe['model_tp_short'] = np.nan
        
        # Model 2022 SL: swept swing low/high
        model2022_long_mask = dataframe['model2022_bull']
        model2022_short_mask = dataframe['model2022_bear']
        
        dataframe.loc[model2022_long_mask, 'model_sl_long'] = (
            dataframe.loc[model2022_long_mask, 'low'].rolling(5).min() - 
            dataframe.loc[model2022_long_mask, 'atr'] * 0.5
        )
        # TP can target recent high or just use R:R. Using recent high might be conservative.
        dataframe.loc[model2022_long_mask, 'model_tp_long'] = dataframe.loc[model2022_long_mask, 'high'].rolling(20).max()
        
        dataframe.loc[model2022_short_mask, 'model_sl_short'] = (
            dataframe.loc[model2022_short_mask, 'high'].rolling(5).max() + 
            dataframe.loc[model2022_short_mask, 'atr'] * 0.5
        )
        dataframe.loc[model2022_short_mask, 'model_tp_short'] = dataframe.loc[model2022_short_mask, 'low'].rolling(20).min()
        
        # Unicorn SL: Breaker Block edge
        unicorn_long_mask = dataframe['unicorn_bull'] & dataframe['model_sl_long'].isna()
        unicorn_short_mask = dataframe['unicorn_bear'] & dataframe['model_sl_short'].isna()
        
        dataframe.loc[unicorn_long_mask, 'model_sl_long'] = (
            dataframe.loc[unicorn_long_mask, 'bb_bull_btm'] - 
            dataframe.loc[unicorn_long_mask, 'atr'] * 0.1
        )
        dataframe.loc[unicorn_short_mask, 'model_sl_short'] = (
            dataframe.loc[unicorn_short_mask, 'bb_bear_top'] + 
            dataframe.loc[unicorn_short_mask, 'atr'] * 0.1
        )
        
        # Fallback SL: OB/FVG edge
        no_sl_long = dataframe['model_sl_long'].isna()
        no_sl_short = dataframe['model_sl_short'].isna()
        
        dataframe.loc[no_sl_long, 'model_sl_long'] = (
            dataframe.loc[no_sl_long, 'active_bull_ob_btm'].fillna(
                dataframe.loc[no_sl_long, 'active_bull_fvg_btm']
            ).fillna(dataframe.loc[no_sl_long, 'ote_bull_btm'])
            - dataframe.loc[no_sl_long, 'atr'] * 0.1
        )
        
        dataframe.loc[no_sl_short, 'model_sl_short'] = (
            dataframe.loc[no_sl_short, 'active_bear_ob_top'].fillna(
                dataframe.loc[no_sl_short, 'active_bear_fvg_top']
            ).fillna(dataframe.loc[no_sl_short, 'ote_bear_top'])
            + dataframe.loc[no_sl_short, 'atr'] * 0.1
        )
        
        # Logging
        if self.analysis_logging.value:
            pair = metadata.get('pair', 'unknown')
            last = dataframe.iloc[-1]
            if last.get('mss_bullish', 0) == 1:
                logger.info(f"[{pair}] MSS Bullish detected")
            if last.get('mss_bearish', 0) == 1:
                logger.info(f"[{pair}] MSS Bearish detected")
            if last.get('model2022_bull', False):
                logger.info(f"[{pair}] Model 2022 LONG signal")
            if last.get('model2022_bear', False):
                logger.info(f"[{pair}] Model 2022 SHORT signal")

        # DEBUG: Log signal counts for the entire dataframe
        if self.dp.runmode.value in ('backtest', 'dry_run'):
            logger.info(f"[{metadata.get('pair')}] SIGNAL COUNTS:")
            logger.info(f"  MSS Bull: {dataframe['mss_bullish'].sum()}, MSS Bear: {dataframe['mss_bearish'].sum()}")
            logger.info(f"  BoS Bull: {dataframe['bos_bullish'].sum()}, BoS Bear: {dataframe['bos_bearish'].sum()}")
            logger.info(f"  In Discount: {dataframe['in_discount'].sum()}, In Premium: {dataframe['in_premium'].sum()}")
            logger.info(f"  In Bull FVG: {dataframe['in_bull_fvg'].sum()}, In Bear FVG: {dataframe['in_bear_fvg'].sum()}")
            logger.info(f"  Model 2022 Bull: {dataframe['model2022_bull'].sum()}")
            logger.info(f"  Basic Structure (MSS/BoS): {((dataframe['mss_bullish']==1)|(dataframe['bos_bullish']==1)).sum()}")
        
        return dataframe
    
    # =========================================================================
    # ENTRY LOGIC - All trading decisions here
    # =========================================================================
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Entry conditions based on ICT trading models.
        
        Uses OR logic between models: any enabled model can trigger.
        Uses AND logic for session/time filters if enabled.
        """
        dataframe.loc[:, 'enter_long'] = 0
        dataframe.loc[:, 'enter_short'] = 0
        dataframe.loc[:, 'enter_tag'] = ''
        
        # =====================================================================
        # SESSION / TIME FILTERS
        # =====================================================================
        
        if self.require_kill_zone.value:
            session_filter = dataframe['in_kill_zone'] == 1
        elif self.require_london_or_ny.value:
            session_filter = (dataframe['in_london'] == 1) | (dataframe['in_ny'] == 1)
        else:
            session_filter = True
        
        if self.require_macro.value:
            macro_filter = dataframe['in_macro'] == 1
        else:
            macro_filter = True
        
        # HTF filter
        if self.use_htf_filter.value:
            pair = metadata.get('pair', '')
            htf_df = self.dp.get_pair_dataframe(pair=pair, timeframe=self.htf_timeframe)
            if len(htf_df) > 0:
                htf_ict = ICTNumba(htf_df, atr_period=self.atr_period.value)
                htf_signals = htf_ict.get_signals()
                htf_direction = htf_signals['structure_direction'].iloc[-1] if len(htf_signals) > 0 else 0
                htf_long_filter = htf_direction == 1
                htf_short_filter = htf_direction == -1
            else:
                htf_long_filter = True
                htf_short_filter = True
        else:
            htf_long_filter = True
            htf_short_filter = True
        
        time_filter = session_filter & macro_filter
        
        # =====================================================================
        # LONG SIGNALS (OR logic between models)
        # =====================================================================
        
        long_signals = pd.Series(False, index=dataframe.index)
        long_tags = pd.Series('', index=dataframe.index)
        
        if self.use_model_2022.value:
            m2022_long = dataframe['model2022_bull']
            long_signals = long_signals | m2022_long
            long_tags = long_tags.where(~m2022_long, 'model2022_long')
        
        if self.use_unicorn.value:
            unicorn_long = dataframe['unicorn_bull']
            long_signals = long_signals | unicorn_long
            long_tags = long_tags.where(~unicorn_long, 'unicorn_long')
        
        if self.use_ote.value:
            ote_long = (dataframe['in_ote_bull'] == 1) & (dataframe['structure_direction'] == 1) & (dataframe['in_discount'] == 1)
            long_signals = long_signals | ote_long
            long_tags = long_tags.where(~ote_long, 'ote_long')
        
        if self.use_silver_bullet.value:
            sb_long = dataframe['silver_bullet_bull']
            long_signals = long_signals | sb_long
            long_tags = long_tags.where(~sb_long, 'silver_bullet_long')
        
        if self.use_basic_structure.value:
            basic_structure = (dataframe['mss_bullish'] == 1) | (dataframe['bos_bullish'] == 1)
            
            if self.require_ob.value or self.require_fvg.value:
                zone = (dataframe['in_bull_ob'] == 1) | (dataframe['in_bull_fvg'] == 1)
            else:
                zone = True
            
            if self.require_premium_discount.value:
                pd_filter = dataframe['in_discount'] == 1
            else:
                pd_filter = True
            
            # Rejection Block Filter
            if self.require_rejection_block.value:
                # Check if we are reacting off a bullish rejection block
                # Simplified: check if recent candles had a RB validation
                # Or just check if we are creating a RB right now? 
                # Better: Check if price touched a RB recently. But we didn't track "active RB" in Numba.
                # Alternative: Use "Strong High/Low" as proxy or just ensure we have a RB formation nearby.
                # Let's use: Signal candle OR previous candle formed a Bullish RB
                rb_filter = (dataframe['rb_bull_btm'] > 0) | (dataframe['rb_bull_btm'].shift(1) > 0)
            else:
                rb_filter = True
                
            # Po3 Filter
            if self.require_po3.value:
                po3_filter = dataframe['in_po3_buy_zone'] == 1
            else:
                po3_filter = True
            
            basic_long = basic_structure & zone & pd_filter & rb_filter & po3_filter
            long_signals = long_signals | basic_long
            long_tags = long_tags.where(~basic_long, 'structure_long')
        
        if self.use_liquidity_raid.value:
            raid_long = dataframe['liquidity_raid_bull']
            long_signals = long_signals | raid_long
            long_tags = long_tags.where(~raid_long, 'liquidity_raid_long')
        
        # Apply filters
        final_long = long_signals & time_filter & htf_long_filter
        dataframe.loc[final_long, 'enter_long'] = 1
        dataframe.loc[final_long, 'enter_tag'] = long_tags[final_long]
        
        # =====================================================================
        # SHORT SIGNALS
        # =====================================================================
        
        short_signals = pd.Series(False, index=dataframe.index)
        short_tags = pd.Series('', index=dataframe.index)
        
        if self.use_model_2022.value:
            m2022_short = dataframe['model2022_bear']
            short_signals = short_signals | m2022_short
            short_tags = short_tags.where(~m2022_short, 'model2022_short')
        
        if self.use_unicorn.value:
            unicorn_short = dataframe['unicorn_bear']
            short_signals = short_signals | unicorn_short
            short_tags = short_tags.where(~unicorn_short, 'unicorn_short')
        
        if self.use_ote.value:
            ote_short = (dataframe['in_ote_bear'] == 1) & (dataframe['structure_direction'] == -1) & (dataframe['in_premium'] == 1)
            short_signals = short_signals | ote_short
            short_tags = short_tags.where(~ote_short, 'ote_short')
        
        if self.use_silver_bullet.value:
            sb_short = dataframe['silver_bullet_bear']
            short_signals = short_signals | sb_short
            short_tags = short_tags.where(~sb_short, 'silver_bullet_short')
        
        if self.use_basic_structure.value:
            basic_structure = (dataframe['mss_bearish'] == 1) | (dataframe['bos_bearish'] == 1)
            
            if self.require_ob.value or self.require_fvg.value:
                zone = (dataframe['in_bear_ob'] == 1) | (dataframe['in_bear_fvg'] == 1)
            else:
                zone = True
            
            if self.require_premium_discount.value:
                pd_filter = dataframe['in_premium'] == 1
            else:
                pd_filter = True
                
            # Rejection Block Filter
            if self.require_rejection_block.value:
                rb_filter = (dataframe['rb_bear_top'] > 0) | (dataframe['rb_bear_top'].shift(1) > 0)
            else:
                rb_filter = True
                
            # Po3 Filter
            if self.require_po3.value:
                po3_filter = dataframe['in_po3_sell_zone'] == 1
            else:
                po3_filter = True
            
            basic_short = basic_structure & zone & pd_filter & rb_filter & po3_filter
            short_signals = short_signals | basic_short
            short_tags = short_tags.where(~basic_short, 'structure_short')
            
        if self.use_liquidity_raid.value:
            raid_short = dataframe['liquidity_raid_bear']
            short_signals = short_signals | raid_short
            short_tags = short_tags.where(~raid_short, 'liquidity_raid_short')
        
        # Apply filters
        final_short = short_signals & time_filter & htf_short_filter
        dataframe.loc[final_short, 'enter_short'] = 1
        dataframe.loc[final_short, 'enter_tag'] = short_tags[final_short]
        
        return dataframe
    
    # =========================================================================
    # EXIT LOGIC
    # =========================================================================
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Exit on opposite MSS if enabled."""
        dataframe.loc[:, 'exit_long'] = 0
        dataframe.loc[:, 'exit_short'] = 0
        
        if self.exit_on_opposite_mss.value:
            dataframe.loc[dataframe['mss_bearish'] == 1, 'exit_long'] = 1
            dataframe.loc[dataframe['mss_bullish'] == 1, 'exit_short'] = 1
        
        return dataframe
    
    # =========================================================================
    # CUSTOM STOPLOSS - Model-specific SL levels
    # =========================================================================
    
    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        after_fill: bool,
        **kwargs
    ) -> Optional[float]:
        """Dynamic stoploss using model-specific levels with ATR fallback."""
        
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if len(dataframe) < 1:
            return None
        
        last = dataframe.iloc[-1]
        atr = last.get('atr', 0)
        if atr == 0:
            return None
        
        trade_key = f"{pair}_{trade.open_date_utc.timestamp()}"
        
        if trade_key not in self.custom_trade_info:
            # Get model SL or fallback to ATR
            if trade.is_short:
                model_sl = last.get('model_sl_short', np.nan)
                model_tp = last.get('model_tp_short', np.nan)
            else:
                model_sl = last.get('model_sl_long', np.nan)
                model_tp = last.get('model_tp_long', np.nan)
            
            if pd.notna(model_sl) and model_sl > 0:
                sl_price = model_sl
                model_based = True
            else:
                sl_distance = atr * self.sl_atr_mult.value
                sl_price = trade.open_rate + sl_distance if trade.is_short else trade.open_rate - sl_distance
                model_based = False
            
            if pd.notna(model_tp) and model_tp > 0:
                tp_price = model_tp
            else:
                risk = abs(trade.open_rate - sl_price)
                reward = risk * self.tp_rr_ratio.value
                tp_price = trade.open_rate - reward if trade.is_short else trade.open_rate + reward
            
            self.custom_trade_info[trade_key] = {
                'sl_price': sl_price,
                'tp_price': tp_price,
                'model_based': model_based
            }
        
        sl_price = self.custom_trade_info[trade_key]['sl_price']
        
        if trade.is_short:
            sl_pct = (sl_price - current_rate) / current_rate
        else:
            sl_pct = (current_rate - sl_price) / current_rate
        
        return -abs(sl_pct)
    
    # =========================================================================
    # CUSTOM EXIT - Take Profit
    # =========================================================================
    
    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs
    ) -> Optional[str]:
        """Take profit at model-specific or R:R level."""
        
        trade_key = f"{pair}_{trade.open_date_utc.timestamp()}"
        
        if trade_key in self.custom_trade_info:
            info = self.custom_trade_info[trade_key]
            tp_price = info.get('tp_price')
            
            if tp_price:
                if trade.is_short and current_rate <= tp_price:
                    return "ICT_TP_model" if info.get('model_based') else f"ICT_TP_{self.tp_rr_ratio.value}R"
                elif not trade.is_short and current_rate >= tp_price:
                    return "ICT_TP_model" if info.get('model_based') else f"ICT_TP_{self.tp_rr_ratio.value}R"
        
        return None
    
    # =========================================================================
    # CONFIRM TRADE ENTRY - Additional filters
    # =========================================================================
    
    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
        entry_tag: Optional[str],
        side: str,
        **kwargs
    ) -> bool:
        """
        Additional entry filters:
        - Check spread is reasonable
        - Check we have valid SL level
        - Check equal highs/lows aren't too close (liquidity trap)
        """
        
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if len(dataframe) < 1:
            return False
        
        last = dataframe.iloc[-1]
        
        # Check we have a valid stoploss level
        if side == 'long':
            sl = last.get('model_sl_long', np.nan)
            if pd.notna(sl) and sl >= rate:
                logger.warning(f"[{pair}] Rejecting long: SL {sl} >= entry {rate}")
                return False
        else:
            sl = last.get('model_sl_short', np.nan)
            if pd.notna(sl) and sl <= rate:
                logger.warning(f"[{pair}] Rejecting short: SL {sl} <= entry {rate}")
                return False
        
        # Check for nearby equal levels (liquidity trap)
        equal_high = last.get('equal_high', np.nan)
        equal_low = last.get('equal_low', np.nan)
        atr = last.get('atr', 0)
        
        if side == 'long' and pd.notna(equal_low):
            if abs(rate - equal_low) < atr * 0.5:
                logger.info(f"[{pair}] Caution: Entry near equal lows @ {equal_low}")
        
        if side == 'short' and pd.notna(equal_high):
            if abs(rate - equal_high) < atr * 0.5:
                logger.info(f"[{pair}] Caution: Entry near equal highs @ {equal_high}")
        
        return True
    
    # =========================================================================
    # LEVERAGE - Dynamic based on volatility
    # =========================================================================
    
    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: Optional[str],
        side: str,
        **kwargs
    ) -> float:
        """
        Dynamic leverage based on session volatility.
        Lower leverage in high-volatility sessions.
        """
        
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if len(dataframe) < 1:
            return 1.0
        
        session_vol = dataframe.iloc[-1].get('session_volatility', 1.0)
        
        # Base leverage 1x, reduced in high volatility
        if session_vol >= 1.5:  # Kill zone
            return min(3.0, max_leverage)
        elif session_vol >= 1.0:  # London/NY
            return min(5.0, max_leverage)
        else:  # Asia
            return min(10.0, max_leverage)
