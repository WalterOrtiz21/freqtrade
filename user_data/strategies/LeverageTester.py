# --- Do not remove these libs ---
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
from datetime import datetime
from freqtrade.persistence import Trade

class LeverageTester(IStrategy):
    """
    Estrategia Corregida: Leverage Agnostic.
    
    Esta versión OBLIGA a Freqtrade a usar custom_stoploss
    y calcula el movimiento del precio puro.
    """

    INTERFACE_VERSION = 3

    # ==========================================
    # CAMBIA ESTO PARA TUS PRUEBAS (1.0 o 10.0)
    # ==========================================
    MY_LEVERAGE = 1.0 
    # ==========================================

    timeframe = '5m'

    # Stoploss de Emergencia (2% de distancia de precio)
    # AJUSTADO POR LEVERAGE para que siempre sea -2% de movimiento de precio
    stoploss = -0.02 * MY_LEVERAGE

    # ROI (Take Profit) - AJUSTADO POR LEVERAGE
    # Con 1x: 4% de movimiento de precio
    # Con 10x: 4% de movimiento de precio = 40% ROE
    minimal_roi = { "0": 0.04 * MY_LEVERAGE }

    # --------------------------------------------------------
    # ¡ESTA ES LA LÍNEA QUE FALTABA!
    # Sin esto, Freqtrade ignora la función custom_stoploss
    # --------------------------------------------------------
    use_custom_stoploss = True

    plot_config = {
        'main_plot': {
            'rsi': {'color': 'orange'},
        },
    }

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: str, side: str,
                 **kwargs) -> float:
        return self.MY_LEVERAGE

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            ((dataframe['rsi'] < 30) & (dataframe['volume'] > 0)),
            'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    # -------------------------------------------------------------------------
    # LÓGICA PURA DE PRECIO (Ignorando Apalancamiento)
    # -------------------------------------------------------------------------
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        
        # 1. Calcular el cambio REAL del precio (Price Change)
        # Freqtrade nos da 'current_profit' ya apalancado (ROE).
        # Para saber cuánto se movió el gráfico, dividimos por el apalancamiento.
        
        # Ejemplo: Si vas 10x y current_profit es 0.10 (10%),
        # significa que el precio real se movió solo 0.01 (1%).
        
        real_price_change = current_profit / trade.leverage

        # 2. Configuración basada en GRÁFICO (No en dinero)
        # Queremos mover a BE si el precio sube un 1%
        PRICE_MOVE_TRIGGER = 0.01 
        
        # Queremos asegurar 0.2% de distancia
        PRICE_MOVE_SECURE = 0.002

        # 3. La condición ahora es idéntica para 1x y 10x
        if real_price_change > PRICE_MOVE_TRIGGER:
            
            # Calculamos el nuevo stoploss relativo al precio actual.
            # Queremos que el SL quede en (Entrada + 0.2%).
            # La distancia desde el precio actual hasta (Entrada + 0.2%) es:
            # -(Cambio_Precio_Actual - 0.2%)
            
            return - (real_price_change - PRICE_MOVE_SECURE)

        return 1