"""
FundingMeanReversion — contrarian directional on funding-rate extremes.

Tesis: extremos sostenidos de funding rate indican un lado crowded del book
pagando al otro. Históricamente estos extremos preceden reversión de precio
por (a) cierre/liquidación de los crowded, (b) erosión de convicción por carry.

V1 single-exchange (Binance USDT-M perps), single-timeframe (1h), 2 pares
(BTC, ETH). Señales:
  - funding sostenidamente positivo + z-score alto  →  SHORT
  - funding sostenidamente negativo + z-score bajo  →  LONG
"""
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd
import talib.abstract as ta
from pandas import DataFrame

from freqtrade.strategy import IStrategy
from freqtrade.persistence import Trade

logger = logging.getLogger(__name__)


class FundingMeanReversion(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "1h"
    can_short = True
    process_only_new_candles = True
    use_exit_signal = True
    use_custom_stoploss = True

    stoploss = -0.04
    minimal_roi = {"0": 0.025}

    # 7 días de history = 168 candles; le sumamos margen para ATR y rolling stats
    startup_candle_count: int = 200

    # Umbrales — calibrados sobre observación de funding Binance BTC/ETH
    # 2024-11 a 2026-04: percentil 75 = 0.0093%, percentil 90 ≈ 0.015%,
    # mínimo observado ≈ -0.015%. Usamos z-score como trigger principal
    # (se auto-calibra por par) y floors absolutos para evitar entradas
    # en regímenes donde la std colapsa cerca de 0.
    FUND_Z_SHORT = 1.0
    FUND_Z_LONG = -1.0
    FUND_ABS_FLOOR_SHORT = 0.00005   # 0.005%/8h mínimo (≈ p70)
    FUND_ABS_FLOOR_LONG = -0.00001   # -0.001%/8h (cualquier negativo moderado)
    FUND_24H_SHORT = 0.00008
    FUND_24H_LONG = -0.00003

    FUND_Z_EXIT_SHORT = 0.3
    FUND_Z_EXIT_LONG = -0.3

    # Filtros de seguridad
    ATR_PCT_CAP = 0.035              # no entrar con vol >3.5%/vela (noticias)
    PRICE_Z_CAP = 1.0                # no entrar si precio ya revirtió
    FUND_ANOMALY_CAP = 0.003         # |funding| > 0.3% = anomalía, skip

    # Exit por tiempo
    MAX_HOLD_HOURS = 48
    STAGNANT_PROFIT_BAND = 0.01      # ±1%

    def informative_pairs(self):
        # Declaramos funding_rate como pair informativo para que freqtrade lo
        # mantenga cacheado en live/dry-run.
        pairs = self.dp.current_whitelist() if self.dp else []
        return [(p, self.timeframe, "funding_rate") for p in pairs]

    # ------------------------------------------------------------------ INDICATORS
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata["pair"]

        # Carga funding a su TF nativo (8h en Binance). El dataprovider auto-fixea TF.
        try:
            funding = self.dp.get_pair_dataframe(pair=pair, candle_type="funding_rate")
        except Exception as e:
            logger.warning(f"[FundingMR] {pair} funding load error: {e}")
            funding = pd.DataFrame()

        if funding is None or len(funding) == 0:
            dataframe["funding_rate"] = np.nan
        else:
            f = funding[["date", "open"]].rename(columns={"open": "funding_rate"}).copy()
            f["date"] = pd.to_datetime(f["date"], utc=True)
            dataframe["date"] = pd.to_datetime(dataframe["date"], utc=True)
            # merge_asof: cada vela 1h recibe el último funding publicado
            dataframe = pd.merge_asof(
                dataframe.sort_values("date"),
                f.sort_values("date"),
                on="date",
                direction="backward",
            )

        dataframe["funding_rate"] = dataframe["funding_rate"].ffill().fillna(0.0)

        # Sostenido: suma funding últimas 24h = 3 pagos (en 1h TF cada pago se
        # repite 8 velas via ffill, por eso usamos diff para aislar eventos reales)
        # Más simple: rolling sum sobre el valor crudo — al estar ffilleado,
        # sum(24) en 1h TF equivale a 3× el promedio de los 3 pagos más recientes.
        # Usamos en su lugar el valor crudo multiplicado por 3 para estimar 24h.
        # (La señal robusta es el z-score sobre 7d.)
        dataframe["funding_24h_sum"] = dataframe["funding_rate"].rolling(24).mean() * 3

        # Baseline 7 días = 168 velas 1h
        dataframe["funding_7d_mean"] = dataframe["funding_rate"].rolling(168).mean()
        dataframe["funding_7d_std"] = dataframe["funding_rate"].rolling(168).std()
        dataframe["funding_zscore"] = (
            (dataframe["funding_rate"] - dataframe["funding_7d_mean"])
            / dataframe["funding_7d_std"].replace(0, np.nan)
        ).fillna(0.0)

        # Filtros de volatilidad y momentum
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)
        dataframe["atr_pct"] = dataframe["atr"] / dataframe["close"]

        close = dataframe["close"]
        dataframe["price_mean24"] = close.rolling(24).mean()
        dataframe["price_std24"] = close.rolling(24).std()
        dataframe["price_z24"] = (
            (close - dataframe["price_mean24"])
            / dataframe["price_std24"].replace(0, np.nan)
        ).fillna(0.0)

        return dataframe

    # ------------------------------------------------------------------ ENTRY
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # shift(1) para que cada vela use features de la vela anterior (evita lookahead)
        fr = dataframe["funding_rate"].shift(1)
        f24 = dataframe["funding_24h_sum"].shift(1)
        fz = dataframe["funding_zscore"].shift(1)
        atrp = dataframe["atr_pct"].shift(1)
        pz = dataframe["price_z24"].shift(1)

        base_ok = (
            atrp.notna()
            & (atrp < self.ATR_PCT_CAP)
            & (fr.abs() < self.FUND_ANOMALY_CAP)
            & (dataframe["funding_7d_std"].shift(1) > 0)
        )

        short_cond = (
            (fr >= self.FUND_ABS_FLOOR_SHORT)
            & (f24 >= self.FUND_24H_SHORT)
            & (fz >= self.FUND_Z_SHORT)
            & (pz > -self.PRICE_Z_CAP)
            & base_ok
        )

        long_cond = (
            (fr <= self.FUND_ABS_FLOOR_LONG)
            & (f24 <= self.FUND_24H_LONG)
            & (fz <= self.FUND_Z_LONG)
            & (pz < self.PRICE_Z_CAP)
            & base_ok
        )

        dataframe.loc[short_cond, ["enter_short", "enter_tag"]] = (1, "FUND_SHORT_EXT")
        dataframe.loc[long_cond, ["enter_long", "enter_tag"]] = (1, "FUND_LONG_EXT")
        return dataframe

    # ------------------------------------------------------------------ EXIT
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        fr = dataframe["funding_rate"].shift(1)
        fz = dataframe["funding_zscore"].shift(1)

        exit_short = (fz <= self.FUND_Z_EXIT_SHORT) | (fr <= 0.00002)
        exit_long = (fz >= self.FUND_Z_EXIT_LONG) | (fr >= -0.00001)

        dataframe.loc[exit_short, ["exit_short", "exit_tag"]] = (1, "FUND_REVERT")
        dataframe.loc[exit_long, ["exit_long", "exit_tag"]] = (1, "FUND_REVERT")
        return dataframe

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> Optional[str]:
        # Salida por tiempo si no capturamos el move esperado
        open_utc = trade.open_date_utc
        if open_utc is None:
            return None
        hold = current_time - open_utc
        if hold >= timedelta(hours=self.MAX_HOLD_HOURS):
            if abs(current_profit) <= self.STAGNANT_PROFIT_BAND:
                return "timeout_stagnant"
            if current_profit < 0:
                return "timeout_loss"
        return None

    # ------------------------------------------------------------------ STOPLOSS
    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        after_fill: bool,
        **kwargs,
    ) -> Optional[float]:
        # Break Even: al superar +1.2% profit, fijamos SL en +0.2% (cubre fees)
        if current_profit >= 0.012:
            return 0.002  # +0.2% profit como nuevo floor
        return None  # mantiene stoploss clase (-4%)

    # ------------------------------------------------------------------ LEVERAGE
    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: str,
        side: str,
        **kwargs,
    ) -> float:
        return self.config.get("leverage", 3.0)

    # ------------------------------------------------------------------ CONFIRM
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
        **kwargs,
    ) -> bool:
        # Blacklist funding anómalo (exchange glitch / launch event)
        df, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        if df is None or len(df) == 0:
            return False
        last = df.iloc[-1]
        fr = last.get("funding_rate", 0.0)
        if pd.isna(fr) or abs(fr) > self.FUND_ANOMALY_CAP:
            logger.info(f"[FundingMR] {pair} blocked — funding anomaly {fr}")
            return False
        return True
