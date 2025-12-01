# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame
# --------------------------------

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
from freqtrade.strategy import (
    DecimalParameter,
    IStrategy,
    IntParameter,
    CategoricalParameter,
)


class ZeroLagMomentumStrategy15m_3of3(IStrategy):
    """
    Zero Lag Momentum Strategy - 15m TIMEFRAME with STRICT Multi-Timeframe Confirmations

    - Principal: 15m (ejecución de trades)
    - Confirmaciones: 1h, 4h, 1d (requiere 3 de 3 - MÁS ESTRICTO)
    - Momentum entries + trend-based exits
    - Risk Management para 10x leverage
    """

    INTERFACE_VERSION = 3

    # --- PARAMETERS ---
    timeframe = '15m'
    startup_candle_count: int = 200

    # Parámetros del indicador
    length = IntParameter(50, 80, default=66, space='buy', optimize=False)
    mult = DecimalParameter(1.8, 2.5, default=2.157, space='buy', optimize=False)

    # Multi-Timeframe Configuration (1h, 4h, 1d - need ALL 3 of 3)
    min_confirmations = IntParameter(3, 3, default=3, space='buy', optimize=False)

    # Risk Management
    stoploss = -0.02  # -2% = -20% real con 10x leverage
    minimal_roi = {}
    max_open_trades = 1

    # Order configuration
    unfilledtimeout = {'entry': 10, 'exit': 10}
    order_types = {
        'entry': 'market', 'exit': 'market', 'emergency_exit': 'market',
        'force_entry': 'market', 'force_exit': 'market', 'stoploss': 'market',
        'stoploss_on_exchange': False, 'stoploss_on_exchange_interval': 60,
    }
    order_time_in_force = {'entry': 'GTC', 'exit': 'GTC'}

    # --- METHODS ---

    def zero_lag_ema(self, dataframe: DataFrame, length: int) -> DataFrame:
        """Zero Lag EMA"""
        lag = int((length - 1) / 2)
        corrected_price = dataframe['close'] + (dataframe['close'] - dataframe['close'].shift(lag))
        zlema = ta.EMA(corrected_price, timeperiod=length)
        return zlema

    def volatility_bands(self, dataframe: DataFrame, length: int, multiplier: float) -> DataFrame:
        """Bandas de volatilidad"""
        atr = ta.ATR(dataframe, timeperiod=length)
        volatility = atr.rolling(length * 3).max() * multiplier
        return volatility

    def calculate_trend(self, dataframe: DataFrame, zlema: DataFrame, volatility: DataFrame) -> DataFrame:
        """
        Calcular tendencia usando la lógica original del indicador
        """
        trend = np.zeros(len(dataframe))

        # Cambios de tendencia por cruce de bandas
        crossover_up = qtpylib.crossed_above(dataframe['close'], zlema + volatility)
        crossover_down = qtpylib.crossed_below(dataframe['close'], zlema - volatility)

        trend[crossover_up] = 1
        trend[crossover_down] = -1

        # Propagar tendencia
        for i in range(1, len(trend)):
            if trend[i] == 0:  # Si no hay nueva señal, mantener anterior
                trend[i] = trend[i-1]

        return DataFrame(trend, index=dataframe.index, columns=['trend'])

    def momentum_entry_signals(self, dataframe: DataFrame, zlema: DataFrame, trend: DataFrame) -> tuple:
        """
        Entradas por momentum con tendencia confirmada
        """
        # Cruces del precio con ZLEMA
        bullish_cross_zlema = qtpylib.crossed_above(dataframe['close'], zlema)
        bearish_cross_zlema = qtpylib.crossed_below(dataframe['close'], zlema)

        # Tendencias establecidas (misma dirección por 2+ barras)
        bullish_trend = (trend['trend'] == 1) & (trend['trend'].shift(1) == 1)
        bearish_trend = (trend['trend'] == -1) & (trend['trend'].shift(1) == -1)

        # Entradas A FAVOR de tendencia (momentum)
        long_signal = bullish_cross_zlema & bullish_trend
        short_signal = bearish_cross_zlema & bearish_trend

        return long_signal, short_signal

    def calculate_higher_timeframe_trends(self, dataframe: DataFrame) -> DataFrame:
        """
        Calcular tendencias en timeframes superiores con diferentes períodos (1h, 4h, 1d)
        """
        # Timeframe 1h (4x 15m = 1h) - usando período más corto
        dataframe['zlema_1h'] = self.zero_lag_ema(dataframe, length=35)
        dataframe['volatility_1h'] = self.volatility_bands(dataframe, length=35, multiplier=2.0)
        trend_1h_df = self.calculate_trend(dataframe, dataframe['zlema_1h'], dataframe['volatility_1h'])
        dataframe['trend_1h'] = trend_1h_df['trend']

        # Timeframe 4h (16x 15m = 4h) - usando período medio
        dataframe['zlema_4h'] = self.zero_lag_ema(dataframe, length=50)
        dataframe['volatility_4h'] = self.volatility_bands(dataframe, length=50, multiplier=2.0)
        trend_4h_df = self.calculate_trend(dataframe, dataframe['zlema_4h'], dataframe['volatility_4h'])
        dataframe['trend_4h'] = trend_4h_df['trend']

        # Timeframe 1d (96x 15m ≈ 1d) - usando período más largo
        dataframe['zlema_1d'] = self.zero_lag_ema(dataframe, length=70)
        dataframe['volatility_1d'] = self.volatility_bands(dataframe, length=70, multiplier=2.0)
        trend_1d_df = self.calculate_trend(dataframe, dataframe['zlema_1d'], dataframe['volatility_1d'])
        dataframe['trend_1d'] = trend_1d_df['trend']

        return dataframe

    # --- STRATEGY CORE ---

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Calcular todos los indicadores para multiple timeframes"""

        # --- Timeframe Principal (15m) ---
        dataframe['zlema'] = self.zero_lag_ema(dataframe, self.length.value)
        dataframe['volatility'] = self.volatility_bands(dataframe, self.length.value, self.mult.value)

        # Tendencia principal
        main_trend = self.calculate_trend(dataframe, dataframe['zlema'], dataframe['volatility'])
        dataframe['trend_main'] = main_trend['trend']

        # --- Timeframes Superiores (1h, 4h, 1d) ---
        self.calculate_higher_timeframe_trends(dataframe)

        # Debug info
        dataframe['zlema_diff'] = (dataframe['close'] - dataframe['zlema']) / dataframe['zlema'] * 100
        dataframe['trend_score'] = (
            dataframe['trend_1h'].fillna(0) +
            dataframe['trend_4h'].fillna(0) +
            dataframe['trend_1d'].fillna(0)
        )

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Entradas con confirmación de 3 timeframes (1h + 4h + 1d) - requiere 3 de 3 (MÁS ESTRICTO)"""

        # Señales base de momentum en timeframe principal
        main_trend_df = DataFrame({'trend': dataframe['trend_main']})
        long_signal, short_signal = self.momentum_entry_signals(
            dataframe, dataframe['zlema'], main_trend_df
        )

        # Confirmación multi-timeframe (1h, 4h, 1d) - necesita TODOS 3 alineados
        # Para long: todas las tendencias deben ser > 0
        # Para short: todas las tendencias deben ser < 0
        bullishly_confirmed = (
            (dataframe['trend_1h'].fillna(0) > 0) &
            (dataframe['trend_4h'].fillna(0) > 0) &
            (dataframe['trend_1d'].fillna(0) > 0)
        )

        bearishly_confirmed = (
            (dataframe['trend_1h'].fillna(0) < 0) &
            (dataframe['trend_4h'].fillna(0) < 0) &
            (dataframe['trend_1d'].fillna(0) < 0)
        )

        # Combinar momentum + confirmación MTF muy estricta
        long_entries = long_signal & bullishly_confirmed
        short_entries = short_signal & bearishly_confirmed

        dataframe.loc[long_entries, 'enter_long'] = 1
        dataframe.loc[short_entries, 'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        SALIDAS SOLO por cambio de tendencia principal (sin ROI)
        """
        # Salir cuando cambia la tendencia principal
        trend_change_long = qtpylib.crossed_below(dataframe['trend_main'], 0)
        trend_change_short = qtpylib.crossed_above(dataframe['trend_main'], 0)

        dataframe.loc[trend_change_long, 'exit_long'] = 1
        dataframe.loc[trend_change_short, 'exit_short'] = 1

        return dataframe

    # --- HYPEROPT ---
    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return self.populate_entry_trend(dataframe, metadata)

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return self.populate_exit_trend(dataframe, metadata)