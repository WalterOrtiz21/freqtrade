
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from pandas import DataFrame
from typing import Optional, Tuple
from numba import njit
import talib.abstract as ta
import pandas_ta as pta

from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter, BooleanParameter, CategoricalParameter, stoploss_from_absolute, merge_informative_pair
from freqtrade.persistence import Trade

logger = logging.getLogger(__name__)

# =============================================================================
# NUMBA-OPTIMIZED FUNCTIONS
# =============================================================================

@njit(cache=False)
def _pivot_high_numba(high: np.ndarray, left: int, right: int) -> np.ndarray:
    """
    Calculate pivot highs with symmetric lookback.
    Matches value = ta.pivothigh(left, right)
    Returns array with pivot value at the index it is confirmed (or aligned), 
    but Pine's pivothigh returns the value at the 'pivot' bar but delayed?
    Actually ta.pivothigh(len, len) returns a value at the moment it is confirmed (bar_index), 
    but the value depends on offset.
    
    Wait, Pine's ta.pivothigh(left, right) returns values delayed by 'right' bars?
    No, in Pine:
    "This function returns the price of the pivot point. It returns 'NaN' if there was no pivot point."
    It returns the value on the bar where the pivot is unidentified? 
    Usually pivots are identified 'right' bars later.
    Freqtrade/Python implementations usually align it to the current bar if looking back.
    
    Let's check SFP implementation again.
    It returns result[i + right] = pivot_val. This means at 'future' index.
    So at index `k`, if `result[k]` is not NaN, it means a pivot happened at `k-right`.
    This logic matches Pine's behavior where `ph` is not NaN on the confirming bar.
    """
    n = len(high)
    result = np.full(n, np.nan)
    
    for i in range(left, n - right):
        is_pivot = True
        pivot_val = high[i]
        
        for j in range(1, left + 1):
            if high[i - j] >= pivot_val:
                is_pivot = False
                break
        
        if is_pivot:
            for j in range(1, right + 1):
                if high[i + j] >= pivot_val:
                    is_pivot = False
                    break
        
        if is_pivot:
            result[i + right] = pivot_val
    
    return result

@njit(cache=False)
def _pivot_low_numba(low: np.ndarray, left: int, right: int) -> np.ndarray:
    n = len(low)
    result = np.full(n, np.nan)
    
    for i in range(left, n - right):
        is_pivot = True
        pivot_val = low[i]
        
        for j in range(1, left + 1):
            if low[i - j] <= pivot_val:
                is_pivot = False
                break
        
        if is_pivot:
            for j in range(1, right + 1):
                if low[i + j] <= pivot_val:
                    is_pivot = False
                    break
        
        if is_pivot:
            result[i + right] = pivot_val
    
    return result

@njit(cache=False)
def _calculate_trendlines(
    close: np.ndarray,
    ph: np.ndarray,
    pl: np.ndarray,
    slopes: np.ndarray,
    length: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates upper/lower trendlines and breakout signals.
    """
    n = len(close)
    upper = np.zeros(n, dtype=np.float64)
    lower = np.zeros(n, dtype=np.float64)
    upos = np.zeros(n, dtype=np.int64)
    dnos = np.zeros(n, dtype=np.int64)
    
    # State variables
    # Initialize with first close or 0
    curr_upper = close[0]
    curr_lower = close[0]
    slope_ph = 0.0
    slope_pl = 0.0
    
    curr_upos = 0
    curr_dnos = 0
    
    # We need to handle the first few bars where we might not have data
    # Pine uses 'var', so values persist.
    
    for i in range(n):
        # 1. Update Slopes
        # slope_ph := ph ? slope : slope_ph
        if not np.isnan(ph[i]):
            slope_ph = slopes[i]
            
        # slope_pl := pl ? slope : slope_pl
        if not np.isnan(pl[i]):
            slope_pl = slopes[i]
            
        # 2. Update Trendlines
        # upper := ph ? ph : upper - slope_ph
        if not np.isnan(ph[i]):
            curr_upper = ph[i]
        else:
            curr_upper = curr_upper - slope_ph
            
        # lower := pl ? pl : lower + slope_pl
        if not np.isnan(pl[i]):
            curr_lower = pl[i]
        else:
            curr_lower = curr_lower + slope_pl
            
        # 3. Calculate Breakouts
        # upos := ph ? 0 : close > upper - slope_ph * length ? 1 : upos
        if not np.isnan(ph[i]):
            curr_upos = 0
        else:
            # Note: Pine 'close' is current close. 
            # 'upper' in pine at this line is already updated.
            # 'slope_ph' is current slope_ph.
            threshold = curr_upper - (slope_ph * length)
            if close[i] > threshold:
                curr_upos = 1
            # else keep previous value (upos)
        
        # dnos := pl ? 0 : close < lower + slope_pl * length ? 1 : dnos
        if not np.isnan(pl[i]):
            curr_dnos = 0
        else:
            threshold = curr_lower + (slope_pl * length)
            if close[i] < threshold:
                curr_dnos = 1
            # else keep previous
            
        upper[i] = curr_upper
        lower[i] = curr_lower
        upos[i] = curr_upos
        dnos[i] = curr_dnos
        
    return upper, lower, upos, dnos

# =============================================================================
# STRATEGY CLASS
# =============================================================================

class TrendlinesWithBreaks(IStrategy):
    """
    Trendlines with Breaks Strategy
    Ported from LuxAlgo Pine Script.
    Includes TP1/BE/SL logic based on price movement.
    """
    
    INTERFACE_VERSION = 3
    
    # Minimal ROI - We rely on signals and custom exit
    minimal_roi = {
        "0": 1.0  # High ROI to let custom expiry/signals handle it
    }
    
    # Stoploss - Set to something wide, we manage it dynamically
    stoploss = -0.15
    
    timeframe = '5m'
    
    # General Strategy Settings
    can_short = True
    use_exit_signal = True
    use_custom_stoploss = True
    position_adjustment_enable = True
    max_open_trades = 3
    startup_candle_count: int = 200
    
    # Trendline Parameters
    length = IntParameter(10, 50, default=14, space='buy', optimize=True)
    mult = DecimalParameter(0.5, 3.0, default=1.0, decimals=1, space='buy', optimize=True)
    # Calc Method: 0=Atr, 1=Stdev, 2=Linreg
    calc_method = CategoricalParameter(['Atr', 'Stdev', 'Linreg'], default='Atr', space='buy', optimize=True)

    # Exit Parameters (Price-Based)
    # TP1: Close a portion of position when price moves X%
    tp1_pct = DecimalParameter(0.005, 0.05, default=0.015, decimals=3, space='sell', optimize=True)
    tp1_enabled = BooleanParameter(default=True, space='sell', optimize=True)
    tp1_amount = DecimalParameter(10.0, 100.0, default=50.0, space='sell', optimize=True)
    
    # Break Even Parameters
    be_enabled = BooleanParameter(default=True, space='sell', optimize=True)
    be_trigger_pct = DecimalParameter(0.005, 0.05, default=0.015, decimals=3, space='sell', optimize=True)
    
    # Logging Control
    enable_logging = BooleanParameter(default=False, space='custom', optimize=False)

    # ==========================================================================
    # HTF TIMEFRAMES (Flexible selection - works with JSON buy params)
    # ==========================================================================
    # Set to 'none' to disable that slot
    # Common options: '5m', '15m', '30m', '1h', '2h', '4h', '8h', '12h', '1d'
    HTF_OPTIONS = ['none', '5m', '15m', '30m', '1h', '2h', '4h', '8h', '12h', '1d']
    
    htf_1 = CategoricalParameter(HTF_OPTIONS, default='1h', space='buy', optimize=False)
    htf_2 = CategoricalParameter(HTF_OPTIONS, default='4h', space='buy', optimize=False)
    
    # Toggle for HTF Trend Filter
    use_htf_filter = BooleanParameter(default=True, space='buy', optimize=True)

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.acc_stoploss = -0.05 # Default internal fallback
        
    def _get_active_htf_list(self) -> list:
        """Build list of active HTF timeframes from parameters."""
        htf_list = []
        if self.htf_1.value != 'none':
            htf_list.append(self.htf_1.value)
        if self.htf_2.value != 'none':
            htf_list.append(self.htf_2.value)
        return htf_list

    def informative_pairs(self):
        """
        Define pairs to load based on htf_1 and htf_2 parameters.
        """
        pairs = self.dp.current_whitelist()
        htf_list = self._get_active_htf_list()
        
        informative_pairs = []
        for tf in htf_list:
            informative_pairs += [(pair, tf) for pair in pairs]
        
        logger.info(f"TrendlinesWithBreaks: Informative pairs configured for timeframes: {htf_list}")
        return informative_pairs
        
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: str,
                 side: str, **kwargs) -> float:
        """
        Use leverage from config.
        """
        return self.config.get('leverage', 1.0)

        
    def bot_start(self, **kwargs) -> None:
        """
        Strategy startup.
        Applies leverage to risk parameters defined in JSON.
        """
        # 1. Get Leverage
        config_leverage = self.config.get('leverage', 1.0)
        
        # 2. Get Raw Values from Config (assumed to be Price Distance)
        # If 'stoploss' is in config (e.g. -0.05), we treat it as 5% PRICE distance.
        raw_stoploss = self.config.get('stoploss', -0.05)
        
        # 3. Apply Leverage to Calculate PnL thresholds for the HARD stop
        # Freqtrade uses self.stoploss as PnL stop. 
        # PnL = Price_Move * Leverage.
        # So effective PnL Stop = Raw_Stop * Leverage.
        self.stoploss = raw_stoploss * config_leverage
        
        logger.info(
            f"TrendlinesWithBreaks Configured:"
            f"\n  Leverage: {config_leverage}x"
            f"\n  Base Stoploss (Price): {raw_stoploss:.2%}"
            f"\n  Effective Stoploss (PnL): {self.stoploss:.2%}"
        )

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        length = self.length.value
        mult = self.mult.value
        method = self.calc_method.value
        
        # 1. Calculate Pivots
        high = dataframe['high'].values
        low = dataframe['low'].values
        close = dataframe['close'].values
        
        ph = _pivot_high_numba(high, length, length)
        pl = _pivot_low_numba(low, length, length)
        
        # 2. Calculate Slope based on method
        if method == 'Atr':
            # ta.atr(length) / length * mult
            atr = ta.ATR(dataframe, timeperiod=length)
            slope = (atr / length) * mult
            
        elif method == 'Stdev':
            # ta.stdev(src, length) / length * mult
            stdev = dataframe['close'].rolling(length).std()
            slope = (stdev / length) * mult
            
        else: # Linreg
            lr_slope = ta.LINEARREG_SLOPE(dataframe['close'], timeperiod=length)
            slope = lr_slope.abs() / 2 * mult
            
        # Fill NaNs in slope (first few bars)
        slope = slope.fillna(0.0).values
        
        # 3. Calculate Trendlines & Breaks (Numba)
        upper, lower, upos, dnos = _calculate_trendlines(close, ph, pl, slope, length)
        
        dataframe['tl_upper'] = upper
        dataframe['tl_lower'] = lower
        dataframe['tl_upos'] = upos
        dataframe['tl_dnos'] = dnos
        
        # Breakout Signals (Transition 0 -> 1)
        # In Pine: upos > upos[1]
        dataframe['break_up'] = (dataframe['tl_upos'] > dataframe['tl_upos'].shift(1)).astype(int)
        dataframe['break_down'] = (dataframe['tl_dnos'] > dataframe['tl_dnos'].shift(1)).astype(int)

        # --- 4. HTF Trendline Calculations ---
        if self.dp:
            htf_list = self._get_active_htf_list()
            for htf in htf_list:
                try:
                    inf_htf = self.dp.get_pair_dataframe(metadata['pair'], htf)
                    if inf_htf.empty:
                        logger.warning(f"No data for {metadata['pair']} {htf}")
                        continue
                    
                    # Calculate trendlines on HTF
                    htf_high = inf_htf['high'].values
                    htf_low = inf_htf['low'].values
                    htf_close = inf_htf['close'].values
                    
                    htf_ph = _pivot_high_numba(htf_high, length, length)
                    htf_pl = _pivot_low_numba(htf_low, length, length)
                    
                    # Calculate HTF slope
                    if method == 'Atr':
                        htf_atr = ta.ATR(inf_htf, timeperiod=length)
                        htf_slope = (htf_atr / length) * mult
                    elif method == 'Stdev':
                        htf_stdev = inf_htf['close'].rolling(length).std()
                        htf_slope = (htf_stdev / length) * mult
                    else:
                        htf_lr_slope = ta.LINEARREG_SLOPE(inf_htf['close'], timeperiod=length)
                        htf_slope = htf_lr_slope.abs() / 2 * mult
                    
                    htf_slope = htf_slope.fillna(0.0).values
                    
                    # Calculate HTF trendlines
                    htf_upper, htf_lower, _, _ = _calculate_trendlines(htf_close, htf_ph, htf_pl, htf_slope, length)
                    
                    # Determine HTF trend
                    # Bullish: close > upper (broken out above resistance)
                    # Bearish: close < lower (broken below support)
                    htf_trend_col = f'{htf}_trend'
                    inf_htf[htf_trend_col] = 0
                    inf_htf.loc[inf_htf['close'] > htf_upper, htf_trend_col] = 1
                    inf_htf.loc[inf_htf['close'] < htf_lower, htf_trend_col] = -1
                    
                    # Prepare for merge — merge_informative_pair shifts HTF dates forward
                    # by 1 HTF period so base candles only see CLOSED HTF candles.
                    # pd.merge on 'date' was lookahead: 5m candle at 00:00 was getting
                    # the 1h trend from the candle that opens at 00:00 but closes at 01:00.
                    inf_htf_merge = inf_htf[['date', htf_trend_col]].copy()
                    dataframe = merge_informative_pair(dataframe, inf_htf_merge, self.timeframe, htf, ffill=True)
                    merged_col = f'{htf_trend_col}_{htf}'
                    dataframe[htf_trend_col] = dataframe[merged_col].fillna(0)
                    dataframe.drop(columns=[merged_col, f'date_{htf}'], errors='ignore', inplace=True)
                    
                    logger.info(f"HTF {htf} trend calculated for {metadata['pair']}")
                    
                except Exception as e:
                    logger.error(f"Error processing HTF {htf}: {e}")

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        # Base signals from trendline breaks
        long_condition = dataframe['break_up'] == 1
        short_condition = dataframe['break_down'] == 1
        
        # ===== HTF TREND FILTER =====
        # HTF provides directional bias, breakout provides entry timing
        if self.use_htf_filter.value:
            htf_list = self._get_active_htf_list()
            
            # Start with True, apply AND for each HTF
            bullish_htf_ok = pd.Series(True, index=dataframe.index)
            bearish_htf_ok = pd.Series(True, index=dataframe.index)
            
            for htf in htf_list:
                trend_col = f'{htf}_trend'
                if trend_col in dataframe.columns:
                    htf_bull = dataframe[trend_col] == 1
                    htf_bear = dataframe[trend_col] == -1
                    
                    # AND logic: ALL HTFs must align
                    bullish_htf_ok &= htf_bull
                    bearish_htf_ok &= htf_bear
            
            long_condition = long_condition & bullish_htf_ok
            short_condition = short_condition & bearish_htf_ok
        
        # Apply signals
        dataframe.loc[long_condition, 'enter_long'] = 1
        dataframe.loc[short_condition, 'enter_short'] = 1
            
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Optional: Exit on opposite signal
        # If we are long (enter_long was 1), and now we have break_down (enter_short), we exit.
        
        dataframe.loc[
            (dataframe['break_down'] == 1),
            'exit_long'] = 1
            
        dataframe.loc[
            (dataframe['break_up'] == 1),
            'exit_short'] = 1
            
        return dataframe

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Break Even Logic with persistent state.
        Once BE is activated, the stoploss is fixed at entry price + small buffer for fees.
        """
        
        if not self.be_enabled.value:
            return 1  # Use default stoploss
        
        # Check if BE was already activated (persistent across ticks)
        be_activated = trade.get_custom_data('be_activated', default=False)
        
        # Calculate price movement to check if we should activate BE
        if trade.is_short:
            extremum = trade.min_rate if trade.min_rate is not None else current_rate
            price_move_pct = (trade.open_rate - extremum) / trade.open_rate
        else:
            extremum = trade.max_rate if trade.max_rate is not None else current_rate
            price_move_pct = (extremum - trade.open_rate) / trade.open_rate
        
        # Activate BE if price moved enough and not already activated
        if not be_activated and price_move_pct >= self.be_trigger_pct.value:
            # Calculate and STORE the BE stop price ONCE
            fee_buffer_pct = 0.001  # 0.1% price buffer
            
            if trade.is_short:
                 # For short, we want to exit slightly below entry (profit)
                 be_stop_price = trade.open_rate * (1 - fee_buffer_pct)
            else:
                 # For long, we want to exit slightly above entry (profit)
                 be_stop_price = trade.open_rate * (1 + fee_buffer_pct)

            trade.set_custom_data('be_activated', True)
            trade.set_custom_data('be_stop_price', be_stop_price)
            be_activated = True
            logger.info(f"Trendlines BE activated for {pair} at price move {price_move_pct:.2%}. Stop set at {be_stop_price:.4f}")
        
        # If BE is activated, return fixed stoploss using absolute helper
        if be_activated:
            be_stop_price = trade.get_custom_data('be_stop_price', default=trade.open_rate)
            return stoploss_from_absolute(be_stop_price, current_rate, is_short=trade.is_short, leverage=trade.leverage)
        
        return 1  # Use default stoploss

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float,
                              min_stake: float | None, max_stake: float,
                              current_entry_rate: float, current_exit_rate: float,
                              current_entry_profit: float, current_exit_profit: float,
                              **kwargs) -> float | None | tuple[float | None, str | None]:
        
        # Calculate price movement pct
        if trade.is_short:
            extremum = trade.min_rate if trade.min_rate else current_rate
            price_move = (trade.open_rate - extremum) / trade.open_rate
        else:
            extremum = trade.max_rate if trade.max_rate else current_rate
            price_move = (extremum - trade.open_rate) / trade.open_rate
            
        # TP1 Logic
        if self.tp1_enabled.value:
            if price_move >= self.tp1_pct.value:
                # Check if TP1 was already taken using persistent flag
                tp1_taken = trade.get_custom_data('tp1_taken', default=False)
                if tp1_taken:
                    return None

                # Mark TP1 as taken BEFORE placing the order (prevents duplicate execution)
                trade.set_custom_data('tp1_taken', True)

                # First exit (TP1)
                # Calculate amount to sell
                close_amount = trade.amount * (self.tp1_amount.value / 100.0)
                sell_value = close_amount * current_exit_rate
                
                # Divide by leverage to get the Margin Amount (Stake) to remove
                # (SFP/SMC Logic alignment)
                stake_change = sell_value / trade.leverage
                
                # We return negative value to reduce position
                return -stake_change, "tp1"

