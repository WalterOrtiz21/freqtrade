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
from freqtrade.exchange import timeframe_to_minutes
from freqtrade.data.dataprovider import DataProvider
from typing import Dict

# Import SMC library
import smartmoneyconcepts as smc
logger.info("smartmoneyconcepts library loaded successfully")


class SMCSimpleFVGTrailing(IStrategy):
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

    # ROI Table - Desactivado para usar TP1/TP2 en su lugar
    minimal_roi = {
        "0": 1.0    # 100% - muy alto para no interferir
    }

    # Stop Loss - 20% = 2% con 10x leverage
    stoploss = -0.20

    # Timeframe
    timeframe = '15m'

    # Trailing Stop - Parámetros optimizados (robustos) con datos completos 2020-2023
    trailing_stop = True
    trailing_stop_positive = 0.343         # Optimizado: Activar al 34.3%
    trailing_stop_positive_offset = 0.418    # Optimizado: Margen de 41.8%
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
    use_exit_signal = True  # REACTIVADO - Salir por señal SMC opuesta o trailing stop
    use_custom_exit = True  # Enable custom exit logic for partial profits
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    ignore_roi_if_exit_signal = False  # Take opposite signals even if profitable
    use_custom_stoploss = True  # ACTIVADO - Usaremos FVG-based trailing stop

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

    # === EXIT STRATEGY PARAMETERS (Hyperopt Optimizable) ===
    # Parámetros optimizables para diferentes estrategias de salida
    exit_strategy_type = "hybrid"  # "trailing", "tp1_tp2_be", "smc_only", "hybrid"

    # 1. Trailing Stop Parameters
    use_trailing = True
    trailing_start_profit = 0.05        # 5% profit para activar trailing
    trailing_stop_positive = 0.03       # 3% (trailing activation level)
    trailing_stop_positive_offset = 0.06  # 6% (activation + 3% margin)

    # 2. Take Profit 1 + Break Even + TP2 Parameters
    tp1_percentage = 0.05                # TP1 al 5%
    tp2_percentage = 0.15                # TP2 al 15%
    enable_be = True                     # Break Even activado
    be_trigger_tp1 = True                # BE cuando alcanza TP1

    # 3. SMC Signals (ya existe)
    use_smc_signals = True               # Usar señales contrarias SMC

    # 4. Hybrid Strategy Weights
    trailing_weight = 0.3                # Peso para trailing
    tp1_tp2_weight = 0.3                  # Peso para TP1+TP2
    smc_weight = 0.4                      # Peso para señales SMC

    # === FVG TRAILING STOP PARAMETERS (LuxAlgo inspired - DISABLED) ===
    fvg_trailing_len = 8          # Lookback para FVGs no mitigados
    fvg_smoothing_len = 12       # Longitud de suavizado
    fvg_trailing_enabled = False  # DESACTIVADO - Probando solo con señales SMC
    fvg_min_profit = 0.08        # Profit mínimo para activar trailing (8% - aumentado)

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

    def di_hyperopt(self) -> Dict:
        """
        Hyperopt space para optimización de estrategias de salida.
        """
        space = {}

        # Para simplificar, vamos a optimizar solo los parámetros principales de trailing
        # ya que Freqtrade tiene soporte nativo para esto

        return space

    def informative_pairs(self):
        """
        No additional informative pairs needed.
        """
        return []

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                    current_rate: float, current_profit: float, **kwargs) -> float:
        """
        FVG-based Trailing Stop (LuxAlgo inspired)
        - Initial SL: -20%
        - Dynamic trailing based on unmigrated FVGs
        - Uses FVG levels as support/resistance for trailing
        """
        if not self.fvg_trailing_enabled:
            return -1.0  # SMC PURE: -100% para nunca activar stop loss

        # Solo activar trailing si tenemos profit mínimo
        if current_profit < self.fvg_min_profit:
            return -1.0  # SMC PURE: Sin stop loss, solo señales SMC  # Mantener stop loss inicial hasta alcanzar profit mínimo

        # Guardar FVGs en el trade para tracking persistente
        if not hasattr(trade, 'fvg_trailing_level'):
            trade.fvg_trailing_level = None
            trade.fvg_trailing_side = None
            trade.initial_stop = -0.20

        # Obtener dataframe con kwargs (método compatible con Freqtrade)
        dataframe = kwargs.get('dataframe', None)
        if dataframe is None or dataframe.empty:
            return -1.0  # SMC PURE: Sin stop loss, solo señales SMC

        try:
            current_close = current_rate

            # Detectar side basado en la dirección de la operación
            if trade.open_rate < current_close:
                # Operación LONG
                if trade.fvg_trailing_side != 'long':
                    trade.fvg_trailing_side = 'long'
                    # Encontrar FVGs alcistas relevantes
                    fvg_bottoms = dataframe['recent_fvg_bullish_bottom'].iloc[-self.fvg_trailing_len:].values
                    # Solo FVGs por debajo del precio actual (no mitigados)
                    valid_fvgs = fvg_bottoms[fvg_bottoms < current_close]

                    if len(valid_fvgs) > 0:
                        trade.fvg_trailing_level = np.max(valid_fvgs)
                        logger.info(f"FVG Trailing para {pair}: LONG, nivel={trade.fvg_trailing_level:.2f}")

                # Calcular trailing stop actualizado
                if trade.fvg_trailing_level is not None:
                    # Verificar si el FVG sigue siendo válido
                    if trade.fvg_trailing_level < current_close:
                        stoploss_pct = (trade.fvg_trailing_level - trade.open_rate) / trade.open_rate
                        return max(stoploss_pct, -0.20)  # Nunca peor que -20%

            else:
                # Operación SHORT
                if trade.fvg_trailing_side != 'short':
                    trade.fvg_trailing_side = 'short'
                    # Encontrar FVGs bajistas relevantes
                    fvg_tops = dataframe['recent_fvg_bearish_top'].iloc[-self.fvg_trailing_len:].values
                    # Solo FVGs por encima del precio actual (no mitigados)
                    valid_fvgs = fvg_tops[fvg_tops > current_close]

                    if len(valid_fvgs) > 0:
                        trade.fvg_trailing_level = np.min(valid_fvgs)
                        logger.info(f"FVG Trailing para {pair}: SHORT, nivel={trade.fvg_trailing_level:.2f}")

                # Calcular trailing stop actualizado
                if trade.fvg_trailing_level is not None:
                    # Verificar si el FVG sigue siendo válido
                    if trade.fvg_trailing_level > current_close:
                        stoploss_pct = (trade.fvg_trailing_level - trade.open_rate) / trade.open_rate
                        return min(stoploss_pct, -0.20)  # Nunca peor que -20%

            # Si no hay FVGs válidos o fueron mitigados, usar stop loss inicial
            return -1.0  # SMC PURE: Sin stop loss, solo señales SMC

        except Exception as e:
            logger.warning(f"Error calculating FVG trailing stop for {pair}: {e}")
            return -1.0  # SMC PURE: Sin stop loss, solo señales SMC  # Fallback a stop loss fijo

    def custom_exit(self, pair: str, trade: 'Trade', current_time: datetime,
                   current_rate: float, current_profit: float, **kwargs) -> bool:
        """
        Multi-strategy exit logic simplified with main focus on:
        1) TP1 + BE + TP2 strategy
        2) Trailing via Freqtrade built-in
        3) SMC signals via exit_signal
        """
        # Initialize trade attributes
        if not hasattr(trade, 'tp1_hit'):
            trade.tp1_hit = False
            trade.be_set = False

        # Get configured parameters
        tp1 = getattr(self, 'tp1_percentage', 0.05)
        tp2 = getattr(self, 'tp2_percentage', 0.15)
        enable_be = getattr(self, 'enable_be', True)

        # ===== ESTRATEGIA: TP1 + BREAK EVEN + TP2 =====
        # Take Profit 1
        if not trade.tp1_hit and current_profit >= tp1:
            trade.tp1_hit = True
            logger.info(f"TP1 ALCANZADO - {pair} at {tp1:.1%} profit: {current_profit:.3f}")

            # Break Even logic (move stop to entry)
            if enable_be:
                trade.be_set = True
                logger.info(f"BREAK EVEN SET - {pair}")
                # Nota: BE se implementaría via custom_stoploss modificando el SL
                return False
            else:
                return False  # Keep position open for TP2

        # Take Profit 2 (exit 100%)
        if trade.tp1_hit and current_profit >= tp2:
            logger.info(f"TP2 ALCANZADO - {pair} at {tp2:.1%} profit: {current_profit:.3f}")
            return True

        # ===== TRACKING DE PROFIT MÁXIMO =====
        if not hasattr(trade, 'max_profit_reached'):
            trade.max_profit_reached = current_profit
        else:
            trade.max_profit_reached = max(trade.max_profit_reached, current_profit)

        # Log milestones
        if current_profit >= 0.01 and not hasattr(trade, 'profit_1_reached'):
            trade.profit_1_reached = True
            logger.info(f"PROFIT TRACKING - {pair} reached +1%: {current_profit:.3f}")

        if current_profit >= 0.05 and not hasattr(trade, 'profit_5_reached'):
            trade.profit_5_reached = True
            logger.info(f"PROFIT TRACKING - {pair} reached +5%: {current_profit:.3f}")

        if current_profit >= 0.10 and not hasattr(trade, 'profit_10_reached'):
            trade.profit_10_reached = True
            logger.info(f"PROFIT TRACKING - {pair} reached +10%: {current_profit:.3f}")

        # Default: No exit
        return False

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