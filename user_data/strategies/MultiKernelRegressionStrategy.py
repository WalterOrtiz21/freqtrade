# Freqtrade: Multi Kernel Regression Strategy (Base Clean)
# Portado desde Pine Script: "Multi Kernel Regression Strategy [ChartPrime]"
# Autor del Port: Assistant (AlgoTrader)
# Versión: 5.0 (Base Clean - Leverage Ready)

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
    """
    Multi Kernel Regression Strategy (v5.0)

    Estrategia de seguimiento de tendencia basada en regresión de kernel no repintada (NRP).

    CARACTERÍSTICAS:
    - Leverage-agnostic: ROI y stoploss se ajustan automáticamente por leverage
    - TPs parciales opcionales (TP1/TP2) - Configurable con enable_partial_exits
    - Breakeven opcional e independiente - Configurable con enable_breakeven y breakeven_roi
    - Salida por señal opuesta del kernel

    PARÁMETROS:
    - enable_partial_exits: Habilitar/Deshabilitar TPs parciales (default: True)
    - tp1_roi: Objetivo TP1 en % de movimiento de precio (default: 1.0%)
    - tp2_roi: Objetivo TP2 en % de movimiento de precio (default: 2.0%)
    - tp1_amount: Cantidad a vender en TP1 (default: 50%)
    - tp2_amount: Cantidad a vender en TP2 (default: 30%)
    - enable_breakeven: Habilitar/Deshabilitar movimiento a breakeven (default: False)
    - breakeven_roi: Profit requerido para activar breakeven (default: 0.5%)

    IMPORTANTE:
    - Los valores ROI/Stoploss se ajustan automáticamente por leverage en bot_start()
    - Los TPs y breakeven se calculan sobre movimiento de precio, no ROE
    - El breakeven es INDEPENDIENTE de los TPs - se activa según breakeven_roi
    - Puedes usar breakeven sin TPs, o TPs sin breakeven, o ambos
    """

    INTERFACE_VERSION = 3

    # ==========================================
    # ─── PARÁMETROS OPTIMIZABLES ───
    # ==========================================
    
    # Ventana del Kernel (Bandwidth)
    bandwidth = IntParameter(5, 50, default=14, space='buy', optimize=True)
    
    # Tipo de Kernel
    kernel_type = CategoricalParameter(
        [
            "Triangular", "Gaussian", "Epanechnikov", "Logistic", "Log Logistic",
            "Cosine", "Sinc", "Laplace", "Quartic", "Parabolic", "Exponential",
            "Silverman", "Cauchy", "Tent", "Wave", "Power", "Morters"
        ],
        default="Gaussian",
        space='buy',
        optimize=True
    )

    # --- FILTROS (RSI / ADX) ---
    use_rsi = BooleanParameter(default=True, space='buy', optimize=True)
    rsi_buy_min = IntParameter(30, 70, default=50, space='buy', optimize=True)
    rsi_sell_max = IntParameter(30, 70, default=50, space='sell', optimize=True)

    use_adx = BooleanParameter(default=True, space='buy', optimize=True)
    adx_min = IntParameter(10, 50, default=25, space='buy', optimize=True)

    # ==========================================
    # ─── GESTIÓN DE RIESGO (TPs) ───
    # ==========================================

    # Habilitar/Deshabilitar Take Profits parciales
    enable_partial_exits = BooleanParameter(default=True, space='sell', optimize=False)

    # Objetivos de profit (1.0 = 1% de movimiento del precio)
    tp1_roi = DecimalParameter(0.5, 5.0, default=1.0, space='sell', optimize=True)
    tp2_roi = DecimalParameter(1.5, 10.0, default=2.0, space='sell', optimize=True)

    # Cantidades a vender (50.0 = 50% de la posición)
    tp1_amount = DecimalParameter(10.0, 90.0, default=50.0, space='sell', optimize=True)
    tp2_amount = DecimalParameter(10.0, 90.0, default=30.0, space='sell', optimize=True)

    # Habilitar/Deshabilitar Breakeven (independiente de TPs)
    enable_breakeven = BooleanParameter(default=False, space='sell', optimize=False)

    # Nivel de profit para activar breakeven (0.5 = 0.5% de movimiento del precio)
    breakeven_roi = DecimalParameter(0.1, 5.0, default=0.5, space='sell', optimize=True)

    # ==========================================
    # ─── CONFIGURACIÓN DE ESTRATEGIA ───
    # ==========================================

    # ROI mínimo alto para dejar que la estrategia gestione las salidas
    minimal_roi = {
        0: 0.99
    }

    # Stop Loss base (Hard Stop) - Se ajustará por leverage en bot_start()
    stoploss = -0.02

    timeframe = '1h'
    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    can_short = True
    startup_candle_count: int = 100

    # Habilitar ajuste de posición para TPs parciales
    position_adjustment_enable = True

    # Habilitar custom stoploss para breakeven
    use_custom_stoploss = True

    # ==========================================
    # ─── INICIALIZACIÓN DINÁMICA ───
    # ==========================================

    def bot_start(self, **kwargs) -> None:
        """
        Ajusta ROI y Stoploss por leverage al iniciar el bot.
        """
        config_leverage = self.config.get('leverage', 1.0)

        logger.info(f" >>> ESTRATEGIA INICIADA CON LEVERAGE: {config_leverage}x <<<")

        # Ajustar Stop Loss por leverage
        self.stoploss = -0.05 * config_leverage

        # Ajustar ROI por leverage
        self.minimal_roi = {
            0: 0.99 * config_leverage
        }

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: str | None, side: str,
                 **kwargs) -> float:
        return self.config.get('leverage', 1.0)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Calcula indicadores.
        """
        # Obtenemos los valores de los parámetros actuales
        bw = self.bandwidth.value
        k_type = self.kernel_type.value

        # 1. Calcular Kernel Regression
        dataframe['kernel_line'] = self.calculate_kernel_nrp(dataframe['close'], bw, k_type)

        # 2. Determinar tendencia (Derivada)
        # 1 = Alcista, 0 = Bajista
        dataframe['trend_up'] = (dataframe['kernel_line'] > dataframe['kernel_line'].shift(1)).astype(int)

        # 3. Indicadores de Confirmación
        dataframe['rsi'] = ta.RSI(dataframe)
        dataframe['adx'] = ta.ADX(dataframe)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Reglas de Entrada.
        """
        # Condiciones comunes
        volume_check = (dataframe['volume'] > 0)
        
        # Filtros RSI (si están activados)
        rsi_buy_cond = (dataframe['rsi'] > self.rsi_buy_min.value) if self.use_rsi.value else (dataframe['volume'] > 0)
        rsi_sell_cond = (dataframe['rsi'] < self.rsi_sell_max.value) if self.use_rsi.value else (dataframe['volume'] > 0)
        
        # Filtros ADX (si están activados)
        adx_cond = (dataframe['adx'] > self.adx_min.value) if self.use_adx.value else (dataframe['volume'] > 0)

        # LONG: Cambio de tendencia 0 -> 1
        dataframe.loc[
            (
                (dataframe['trend_up'] == 1) &
                (dataframe['trend_up'].shift(1) == 0) &
                volume_check &
                rsi_buy_cond &
                adx_cond
            ),
            'enter_long'] = 1

        # SHORT: Cambio de tendencia 1 -> 0
        dataframe.loc[
            (
                (dataframe['trend_up'] == 0) &
                (dataframe['trend_up'].shift(1) == 1) &
                volume_check &
                rsi_sell_cond &
                adx_cond
            ),
            'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Reglas de Salida (Cierre por señal opuesta).
        """
        # SALIDA LONG: Tendencia cambia a bajista
        dataframe.loc[
            (
                (dataframe['trend_up'] == 0) &
                (dataframe['trend_up'].shift(1) == 1) &
                (dataframe['volume'] > 0)
            ),
            'exit_long'] = 1

        # SALIDA SHORT: Tendencia cambia a alcista
        dataframe.loc[
            (
                (dataframe['trend_up'] == 1) &
                (dataframe['trend_up'].shift(1) == 0) &
                (dataframe['volume'] > 0)
            ),
            'exit_short'] = 1

        return dataframe

    # ==========================================
    # ─── GESTIÓN DE RIESGO Y SALIDAS ───
    # ==========================================

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float,
                              min_stake: float | None, max_stake: float,
                              current_entry_rate: float, current_exit_rate: float,
                              current_entry_profit: float, current_exit_profit: float,
                              **kwargs) -> float | None | tuple[float | None, str | None]:
        """
        Gestiona Take Profits parciales (TP1 y TP2).
        """
        # Si los partial exits están deshabilitados, no hacer nada
        if not self.enable_partial_exits.value:
            return None

        # NORMALIZAR PROFIT por leverage (convertir ROE a movimiento de precio)
        current_price_profit = current_profit / trade.leverage

        # Convertir objetivos de % a decimal (1.0 = 1% → 0.01)
        tp1_target_price = self.tp1_roi.value / 100.0
        tp2_target_price = self.tp2_roi.value / 100.0

        if current_price_profit > tp1_target_price:

            # Verificar cuántos TPs ya se ejecutaron
            filled_sells = [o for o in trade.orders if o.side == 'sell' and o.status == 'closed']
            count_sells = len(filled_sells)

            # --- TP 1 ---
            if count_sells == 0:
                sell_pct = self.tp1_amount.value / 100.0
                amount_to_sell = trade.amount * sell_pct
                return -(amount_to_sell * current_rate)

            # --- TP 2 ---
            if count_sells == 1 and current_price_profit > tp2_target_price:
                remaining_pct_total = 1.0 - (self.tp1_amount.value / 100.0)
                target_pct_total = self.tp2_amount.value / 100.0

                if remaining_pct_total > 0:
                    sell_pct_of_remaining = target_pct_total / remaining_pct_total
                    sell_pct_of_remaining = min(sell_pct_of_remaining, 1.0)
                    amount_to_sell = trade.amount * sell_pct_of_remaining
                    return -(amount_to_sell * current_rate)

        return None

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Mueve el stop loss a breakeven cuando el profit alcanza el nivel configurado.
        Independiente de los TPs - se activa según breakeven_roi.
        """
        # Si el breakeven está deshabilitado, no cambiar el stoploss
        # Retornar 1 = "no hacer ningún cambio al stoploss actual"
        if not self.enable_breakeven.value:
            return 1

        # NORMALIZAR PROFIT por leverage (convertir ROE a movimiento de precio)
        current_price_profit = current_profit / trade.leverage

        # Convertir objetivo de % a decimal (0.5 = 0.5% → 0.005)
        breakeven_target = self.breakeven_roi.value / 100.0

        # Si el profit alcanza el nivel configurado, mover a breakeven
        if current_price_profit > breakeven_target:
            # Calcular stoploss relativo para mover a breakeven
            if trade.is_short:
                # Para short: SL debe estar ARRIBA del precio actual
                # Formula: (current - entry) / current
                # Cuando el precio baja (ganando): current < entry → negativo (correcto)
                stoploss_relative = (current_rate - trade.open_rate) / current_rate
            else:
                # Para long: SL debe estar DEBAJO del precio actual
                # Formula: (entry - current) / current
                # Cuando el precio sube (ganando): current > entry → negativo (correcto)
                stoploss_relative = (trade.open_rate - current_rate) / current_rate

            # Si el stoploss calculado es positivo, no mover (esperamos que sea negativo)
            # Retornar 1 = no cambiar el stoploss
            if stoploss_relative > 0:
                return 1

            return stoploss_relative

        # Si no alcanzamos el nivel, no cambiar el stoploss
        return 1

    # ==========================================
    # ─── LÓGICA MATEMÁTICA DEL KERNEL ───
    # ==========================================

    def calculate_kernel_nrp(self, series: pd.Series, bandwidth: int, kernel_name: str) -> pd.Series:
        """
        Calcula la regresión del kernel NO repintada (NRP) usando convolución vectorizada.
        """
        # 1. Generar Pesos
        weights = self.generate_weights(bandwidth, kernel_name)
        
        # 2. Normalizar Pesos
        sum_weights = np.sum(weights)
        if sum_weights > 0:
            weights = weights / sum_weights
        
        # 3. Preparar datos
        src = series.values
        
        if len(src) < bandwidth:
            return pd.Series(np.nan, index=series.index)

        # 4. Convolución
        # Invertimos pesos para np.convolve (comportamiento estándar de procesamiento de señales)
        # Esto asegura que weights[0] se aplique al dato más reciente en la ventana deslizante
        kernel_result = np.convolve(src, weights, mode='full')
        
        # Recortar al tamaño original y ajustar NaNs iniciales
        kernel_result = kernel_result[:len(src)]
        kernel_result[:bandwidth-1] = np.nan 
        
        return pd.Series(kernel_result, index=series.index)

    def generate_weights(self, bandwidth: int, kernel_name: str) -> np.array:
        weights = []
        for i in range(bandwidth):
            j = (i ** 2) / (bandwidth ** 2)
            w = self.get_weight(j, 1, kernel_name)
            weights.append(w)
        return np.array(weights)

    def get_weight(self, source: float, bandwidth: float, style: str) -> float:
        # Implementación de funciones de peso del Kernel
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