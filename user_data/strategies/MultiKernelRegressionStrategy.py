# Freqtrade: Multi Kernel Regression Strategy
# Portado desde Pine Script: "Multi Kernel Regression Strategy [ChartPrime]"
# Autor del Port: Assistant
# Lógica: Usa Kernel Regression (NRP) para determinar la tendencia.

import numpy as np
import pandas as pd
from pandas import DataFrame
import math
import logging

from freqtrade.strategy import IStrategy, IntParameter, CategoricalParameter
import talib.abstract as ta
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

class MultiKernelRegressionStrategy(IStrategy):
    """
    Multi Kernel Regression Strategy
    Basada en ChartPrime Pine Script.
    Usa convolución matemática para aplicar 17 tipos de kernels diferentes.
    """

    INTERFACE_VERSION = 3

    # --- PARÁMETROS OPTIMIZABLES ---
    
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

    # Configuración de Estrategia
    minimal_roi = {
        "0": 0.20,
        "30": 0.10,
        "60": 0.05,
        "120": 0.01
    }

    stoploss = -0.10
    timeframe = '15m'
    
    # Trailing stop para asegurar ganancias
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = True

    # Solo operar en cambios de tendencia
    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    
    # Número de velas necesarias (bandwidth max + buffer)
    startup_candle_count: int = 2000

    leverage_value = 10.0

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                 side: str, **kwargs) -> float:
        return self.leverage_value

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Calcula el indicador Kernel Regression NRP usando convolución.
        """
        # Obtenemos los valores de los parámetros actuales
        bw = self.bandwidth.value
        k_type = self.kernel_type.value

        # Calcular la línea de regresión del kernel
        dataframe['kernel_line'] = self.calculate_kernel_nrp(dataframe['close'], bw, k_type)

        # Determinar tendencia
        # Trend Up: Valor actual > Valor anterior
        dataframe['trend_up'] = (dataframe['kernel_line'] > dataframe['kernel_line'].shift(1)).astype(int)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Genera señales de entrada.
        """
        # LONG: Cuando la tendencia cambia de bajista (0) a alcista (1)
        dataframe.loc[
            (
                (dataframe['trend_up'] == 1) &
                (dataframe['trend_up'].shift(1) == 0) &
                (dataframe['volume'] > 0)
            ),
            'enter_long'] = 1

        # SHORT: Cuando la tendencia cambia de alcista (1) a bajista (0)
        dataframe.loc[
            (
                (dataframe['trend_up'] == 0) &
                (dataframe['trend_up'].shift(1) == 1) &
                (dataframe['volume'] > 0)
            ),
            'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Genera señales de salida (Salida por señal opuesta).
        """
        # SALIDA LONG: Si la tendencia cambia a bajista
        dataframe.loc[
            (
                (dataframe['trend_up'] == 0) &
                (dataframe['trend_up'].shift(1) == 1) &
                (dataframe['volume'] > 0)
            ),
            'exit_long'] = 1

        # SALIDA SHORT: Si la tendencia cambia a alcista
        dataframe.loc[
            (
                (dataframe['trend_up'] == 1) &
                (dataframe['trend_up'].shift(1) == 0) &
                (dataframe['volume'] > 0)
            ),
            'exit_short'] = 1

        return dataframe

    # ==========================================
    # ─── LÓGICA MATEMÁTICA DEL KERNEL ───
    # ==========================================

    def calculate_kernel_nrp(self, series: pd.Series, bandwidth: int, kernel_name: str) -> pd.Series:
        """
        Calcula la regresión del kernel NO repintada (NRP).
        Usa np.convolve para velocidad extrema.
        """
        # 1. Generar los pesos según el tipo de kernel y el ancho de banda
        weights = self.generate_weights(bandwidth, kernel_name)
        
        # 2. Normalizar pesos (la suma debe ser 1)
        sum_weights = np.sum(weights)
        if sum_weights > 0:
            weights = weights / sum_weights
        
        # 3. Aplicar convolución
        # 'valid': La salida es más pequeña que la entrada (se pierden las primeras 'bandwidth' velas)
        # Invertimos los pesos ([::-1]) porque convolve es una operación de sliding dot product
        # Nota: La lógica del Pine itera i=0..bw-1. 
        # i=0 es la vela actual. i=bw-1 es la vela más vieja.
        # np.convolve alinea el final del kernel con el punto actual.
        
        # Convertimos a numpy para velocidad
        src = series.values
        
        # Si no hay suficientes datos, devolver serie vacía
        if len(src) < bandwidth:
            return pd.Series(np.nan, index=series.index)

        # Aplicar convolución weighted moving average
        # weights ya está en el orden correcto para convolve si consideramos weights[0] como el peso de la vela más reciente
        # np.convolve(a, v, mode='full')
        
        # Para replicar exactamente: nrp_sum += source[i] * weight
        # source[0] es actual, source[1] es anterior...
        # weights[0] es el peso para source[0] (actual)
        
        # np.convolve aplica el kernel deslizante.
        kernel_result = np.convolve(src, weights[::-1], mode='full')
        
        # Ajustar el resultado para que coincida con la longitud del dataframe (shift)
        # El modo 'full' agrega padding al principio y final.
        # Queremos los valores hasta el índice actual.
        kernel_result = kernel_result[:len(src)]
        
        # Los primeros 'bandwidth-1' valores son inexactos en convolución 'full' sin fill, 
        # pero Freqtrade maneja esto con startup_candle_count.
        
        # Desplazar para alinear correctamente (convolve introduce un lag natural basado en el tamaño de ventana)
        # Con weights[::-1] y tomando hasta len(src), el último valor es correcto.
        kernel_result[:bandwidth-1] = np.nan # Invalidar startups
        
        return pd.Series(kernel_result, index=series.index)

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: str | None, side: str,
                 **kwargs) -> float:
        """
        Customize leverage for each new trade. This method is only called in futures mode.
        """
        return self.config.get('leverage', 1.0)    

    def generate_weights(self, bandwidth: int, kernel_name: str) -> np.array:
        """
        Genera el array de pesos basado en las fórmulas del Pine Script.
        """
        weights = []
        
        # Replicamos el bucle NRP del Pine:
        # for i = 0 to bandwidth - 1
        #   j = math.pow(i, 2) / (math.pow(bandwidth, 2))
        #   weight = get_weight(j, 1, kernel_name)
        
        for i in range(bandwidth):
            # i representa el lag (0 = actual, 1 = ayer...)
            j = (i ** 2) / (bandwidth ** 2)
            w = self.get_weight(j, 1, kernel_name)
            weights.append(w)
            
        return np.array(weights)

    def get_weight(self, source: float, bandwidth: float, style: str) -> float:
        """
        Implementación Python de las 17 funciones matemáticas del Pine Script.
        source: input value (en el loop es 'j')
        bandwidth: siempre 1 en el loop NRP
        """
        x = source / bandwidth
        abs_x = abs(x)
        sq_x = x ** 2
        
        if style == "Gaussian":
            return math.exp(-sq_x / 2) / math.sqrt(2 * math.pi)
        
        elif style == "Triangular":
            return 1 - abs_x if abs_x <= 1 else 0.0
        
        elif style == "Epanechnikov":
            return (3/4) * (1 - sq_x) if abs_x <= 1 else 0.0
            
        elif style == "Quartic":
            return (15/16) * ((1 - sq_x) ** 2) if abs_x <= 1 else 0.0
            
        elif style == "Logistic":
            return 1 / (math.exp(x) + 2 + math.exp(-x))
            
        elif style == "Cosine":
            return (math.pi / 4) * math.cos((math.pi / 2) * x) if abs_x <= 1 else 0.0
            
        elif style == "Laplace":
            return (1 / (2 * bandwidth)) * math.exp(-abs_x)
            
        elif style == "Exponential":
            return (1 / bandwidth) * math.exp(-abs_x)
            
        elif style == "Silverman":
            if abs_x <= 0.5:
                return 0.5 * math.exp(-x/2) * math.sin(x/2 + math.pi/4)
            return 0.0
            
        elif style == "Tent":
            return 1 - abs_x if abs_x <= 1 else 0.0
            
        elif style == "Cauchy":
            return 1 / (math.pi * bandwidth * (1 + sq_x))
            
        elif style == "Sinc":
            if source == 0: return 1.0
            val = math.pi * x
            return math.sin(val) / val
            
        elif style == "Wave":
            if abs_x <= 1:
                return (1 - abs_x) * math.cos((math.pi * source) / bandwidth)
            return 0.0
            
        elif style == "Parabolic":
            return 1 - sq_x if abs_x <= 1 else 0.0
            
        elif style == "Power":
            if abs_x <= 1:
                return (1 - (abs_x ** 3)) ** 3
            return 0.0
            
        elif style == "Log Logistic":
            return 1 / ((1 + abs_x) ** 2)
            
        elif style == "Morters":
            if abs_x <= math.pi:
                return (1 + math.cos(x)) / (2 * math.pi * bandwidth)
            return 0.0
            
        return 0.0