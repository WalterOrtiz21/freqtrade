# Freqtrade: Multi Kernel Regression Strategy (Clean Version)
# Version: 5.0 (No Optuna - Minimal Comments)

import numpy as np
import pandas as pd
from pandas import DataFrame
import math
import logging
from freqtrade.strategy import (IStrategy, IntParameter, DecimalParameter, BooleanParameter, CategoricalParameter)
from freqtrade.persistence import Trade
import talib.abstract as ta
from datetime import datetime

logger = logging.getLogger(__name__)

class MultiKernelRegressionStrategy(IStrategy):
    INTERFACE_VERSION = 3

    # --- Parámetros de Estrategia ---
    bandwidth = IntParameter(5, 50, default=14, space='buy', optimize=True)
    kernel_type = CategoricalParameter(
        ["Triangular", "Gaussian", "Epanechnikov", "Logistic", "Log Logistic",
         "Cosine", "Sinc", "Laplace", "Quartic", "Parabolic", "Exponential",
         "Silverman", "Cauchy", "Tent", "Wave", "Power", "Morters"],
        default="Gaussian", space='buy', optimize=True
    )

    # --- Filtros ---
    use_rsi = BooleanParameter(default=False, space='buy', optimize=True)
    rsi_buy_min = IntParameter(30, 70, default=50, space='buy', optimize=True)
    rsi_sell_max = IntParameter(30, 70, default=50, space='sell', optimize=True)

    # Choppiness Index (Filtro de Tendencia)
    use_chop = BooleanParameter(default=True, space='buy', optimize=True)
    chop_length = IntParameter(10, 30, default=14, space='buy', optimize=True)
    chop_max = IntParameter(40, 60, default=50, space='buy', optimize=True)

    # EWO (Elliott Wave Oscillator) - Filtro de Momentum
    use_ewo = BooleanParameter(default=True, space='buy', optimize=True)
    ewo_high = DecimalParameter(2.0, 15.0, default=5.0, space='buy', optimize=True)
    ewo_low = DecimalParameter(-15.0, -2.0, default=-5.0, space='buy', optimize=True)

    use_adx = BooleanParameter(default=True, space='buy', optimize=True)
    adx_min = IntParameter(10, 50, default=25, space='buy', optimize=True)

    use_volatility_filter = BooleanParameter(default=False, space='buy', optimize=True)

    # --- Gestión de Riesgo (TPs Parciales & Breakeven) ---
    enable_partial_exits = BooleanParameter(default=True, space='sell', optimize=False)
    tp1_roi = DecimalParameter(0.5, 5.0, default=1.0, space='sell', optimize=True)
    tp2_roi = DecimalParameter(1.5, 10.0, default=2.0, space='sell', optimize=True)
    tp1_amount = DecimalParameter(10.0, 90.0, default=50.0, space='sell', optimize=True)
    tp2_amount = DecimalParameter(10.0, 90.0, default=30.0, space='sell', optimize=True)

    enable_breakeven = BooleanParameter(default=False, space='sell', optimize=False)
    breakeven_roi = DecimalParameter(0.1, 5.0, default=0.5, space='sell', optimize=True)

    # --- Configuración Base ---
    # ROI Table (Stepped)
    minimal_roi = {
        "0": 0.20,
        "30": 0.05,
        "60": 0.02,
        "120": 0.01
    }
    
    # Stoploss & Trailing
    stoploss = -0.05 # Stoploss fijo más amplio para dar aire
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.03
    trailing_only_offset_is_reached = True
    timeframe = '1h'
    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    can_short = True
    startup_candle_count: int = 100
    
    position_adjustment_enable = True
    use_custom_stoploss = True

    def bot_start(self, **kwargs) -> None:
        """ Ajuste dinámico de ROI/Stoploss por leverage """
        config_leverage = self.config.get('leverage', 1.0)
        self.stoploss = -0.05 * config_leverage
        self.minimal_roi = { 0: 0.99 * config_leverage }

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: str | None, side: str,
                 **kwargs) -> float:
        return self.config.get('leverage', 1.0)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 1. Calcular Kernel Regression
        dataframe['kernel_line'] = self.calculate_kernel_nrp(
            dataframe['close'], 
            self.bandwidth.value, 
            self.kernel_type.value
        )

        # 2. Determinar tendencia (1 = Alcista, 0 = Bajista)
        dataframe['trend_up'] = (dataframe['kernel_line'] > dataframe['kernel_line'].shift(1)).astype(int)

        # 3. Indicadores de confirmación
        dataframe['rsi'] = ta.RSI(dataframe)
        dataframe['adx'] = ta.ADX(dataframe)

        # Choppiness Index Calculation
        # 100 * LOG10( SUM(ATR(1), n) / ( MaxHi(n) - MinLo(n) ) ) / LOG10(n)
        if self.use_chop.value:
            chop_len = self.chop_length.value
            # ATR(1) es True Range
            dataframe['tr'] = ta.TRANGE(dataframe)
            dataframe['tr_sum'] = dataframe['tr'].rolling(window=chop_len).sum()
            dataframe['hh'] = dataframe['high'].rolling(window=chop_len).max()
            dataframe['ll'] = dataframe['low'].rolling(window=chop_len).min()
            
            # Evitar división por cero
            dataframe['chop_range'] = dataframe['hh'] - dataframe['ll']
            dataframe['chop_range'] = dataframe['chop_range'].replace(0, 0.0000001)
            
            dataframe['chop'] = 100 * np.log10(dataframe['tr_sum'] / dataframe['chop_range']) / np.log10(chop_len)
        else:
            dataframe['chop'] = 0 # Si está desactivado, asumimos 0 (tendencia fuerte) para no filtrar

        # EWO Calculation
        # SMA(5) - SMA(35)
        dataframe['ewo'] = ta.SMA(dataframe, timeperiod=5) - ta.SMA(dataframe, timeperiod=35)

        # 4. Volatility Filter (ATR(1) > ATR(10))
        if self.use_volatility_filter.value:
            atr1 = ta.ATR(dataframe, timeperiod=1)
            atr10 = ta.ATR(dataframe, timeperiod=10)
            dataframe['volatility_ok'] = atr1 > atr10
        else:
            dataframe['volatility_ok'] = True

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        volume_check = (dataframe['volume'] > 0)
        rsi_buy = (dataframe['rsi'] > self.rsi_buy_min.value) if self.use_rsi.value else True
        rsi_sell = (dataframe['rsi'] < self.rsi_sell_max.value) if self.use_rsi.value else True
        adx_ok = (dataframe['adx'] > self.adx_min.value) if self.use_adx.value else True
        chop_ok = (dataframe['chop'] < self.chop_max.value) if self.use_chop.value else True
        
        # EWO Filter
        ewo_buy = (dataframe['ewo'] > self.ewo_high.value) if self.use_ewo.value else True
        ewo_sell = (dataframe['ewo'] < self.ewo_low.value) if self.use_ewo.value else True
        
        vol_ok = dataframe['volatility_ok']

        # LONG: Cambio de tendencia 0 -> 1
        dataframe.loc[
            (
                (dataframe['trend_up'] == 1) &
                (dataframe['trend_up'].shift(1) == 0) &
                volume_check & rsi_buy & adx_ok & vol_ok & chop_ok & ewo_buy
            ),
            'enter_long'] = 1

        # SHORT: Cambio de tendencia 1 -> 0
        dataframe.loc[
            (
                (dataframe['trend_up'] == 0) &
                (dataframe['trend_up'].shift(1) == 1) &
                volume_check & rsi_sell & adx_ok & vol_ok & chop_ok & ewo_sell
            ),
            'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Exit Long: Tendencia cambia a bajista
        dataframe.loc[
            ((dataframe['trend_up'] == 0) & (dataframe['trend_up'].shift(1) == 1) & (dataframe['volume'] > 0)),
            'exit_long'] = 1

        # Exit Short: Tendencia cambia a alcista
        dataframe.loc[
            ((dataframe['trend_up'] == 1) & (dataframe['trend_up'].shift(1) == 0) & (dataframe['volume'] > 0)),
            'exit_short'] = 1

        return dataframe

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float,
                              min_stake: float | None, max_stake: float,
                              current_entry_rate: float, current_exit_rate: float,
                              current_entry_profit: float, current_exit_profit: float,
                              **kwargs) -> float | None | tuple[float | None, str | None]:
        """ Gestión de Take Profits Parciales """
        if not self.enable_partial_exits.value:
            return None

        current_price_profit = current_profit / trade.leverage
        tp1_target = self.tp1_roi.value / 100.0
        tp2_target = self.tp2_roi.value / 100.0

        if current_price_profit > tp1_target:
            filled_sells = [o for o in trade.orders if o.side == 'sell' and o.status == 'closed']
            count_sells = len(filled_sells)

            # TP 1
            if count_sells == 0:
                return -(trade.amount * (self.tp1_amount.value / 100.0) * current_rate)

            # TP 2
            if count_sells == 1 and current_price_profit > tp2_target:
                remaining = 1.0 - (self.tp1_amount.value / 100.0)
                if remaining > 0:
                    ratio = min((self.tp2_amount.value / 100.0) / remaining, 1.0)
                    return -(trade.amount * ratio * current_rate)

        return None

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        """ Gestión de Breakeven """
        if not self.enable_breakeven.value:
            return 1

        current_price_profit = current_profit / trade.leverage
        if current_price_profit > (self.breakeven_roi.value / 100.0):
            if trade.is_short:
                sl_relative = (current_rate - trade.open_rate) / current_rate
            else:
                sl_relative = (trade.open_rate - current_rate) / current_rate
            
            # Solo mover si mejora la posición (sl_relative negativo es profit locking)
            return sl_relative if sl_relative < 0 else 1

        return 1

    # --- Matemáticas Kernel (NRP) ---

    def calculate_kernel_nrp(self, series: pd.Series, bandwidth: int, kernel_name: str) -> pd.Series:
        weights = self.generate_weights(bandwidth, kernel_name)
        sum_weights = np.sum(weights)
        if sum_weights > 0: weights = weights / sum_weights
        
        src = series.values
        if len(src) < bandwidth: return pd.Series(np.nan, index=series.index)

        kernel_result = np.convolve(src, weights, mode='full')
        kernel_result = kernel_result[:len(src)]
        kernel_result[:bandwidth-1] = np.nan 
        
        return pd.Series(kernel_result, index=series.index)

    def generate_weights(self, bandwidth: int, kernel_name: str) -> np.array:
        weights = []
        for i in range(bandwidth):
            w = self.get_weight((i ** 2) / (bandwidth ** 2), 1, kernel_name)
            weights.append(w)
        return np.array(weights)

    def get_weight(self, source: float, bandwidth: float, style: str) -> float:
        x = source / bandwidth
        abs_x = abs(x)
        sq_x = x ** 2
        
        if style == "Gaussian": return math.exp(-sq_x / 2) / math.sqrt(2 * math.pi)
        elif style == "Triangular": return 1 - abs_x if abs_x <= 1 else 0.0
        elif style == "Epanechnikov": return (3/4) * (1 - sq_x) if abs_x <= 1 else 0.0
        elif style == "Quartic": return (15/16) * ((1 - sq_x) ** 2) if abs_x <= 1 else 0.0
        elif style == "Logistic": return 1 / (math.exp(x) + 2 + math.exp(-x))
        elif style == "Cosine": return (math.pi / 4) * math.cos((math.pi / 2) * x) if abs_x <= 1 else 0.0
        elif style == "Laplace": return (1 / (2 * bandwidth)) * math.exp(-abs_x)
        elif style == "Exponential": return (1 / bandwidth) * math.exp(-abs_x)
        elif style == "Silverman": return (0.5 * math.exp(-x/2) * math.sin(x/2 + math.pi/4)) if abs_x <= 0.5 else 0.0
        elif style == "Tent": return 1 - abs_x if abs_x <= 1 else 0.0
        elif style == "Cauchy": return 1 / (math.pi * bandwidth * (1 + sq_x))
        elif style == "Sinc": return (math.sin(math.pi * x) / (math.pi * x)) if source != 0 else 1.0
        elif style == "Wave": return ((1 - abs_x) * math.cos((math.pi * source) / bandwidth)) if abs_x <= 1 else 0.0
        elif style == "Parabolic": return 1 - sq_x if abs_x <= 1 else 0.0
        elif style == "Power": return (1 - (abs_x ** 3)) ** 3 if abs_x <= 1 else 0.0
        elif style == "Log Logistic": return 1 / ((1 + abs_x) ** 2)
        elif style == "Morters": return ((1 + math.cos(x)) / (2 * math.pi * bandwidth)) if abs_x <= math.pi else 0.0
        return 0.0