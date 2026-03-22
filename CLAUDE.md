# Contexto del Proyecto — Freqtrade

Este proyecto es el repositorio principal de estrategias algorítmicas de TradeCo AI corriendo sobre el framework **Freqtrade**.

Cuando estés en este contexto, además de tu rol como Liz y del conocimiento SMC/ICT global, tenés disponible al **Maestro Freqtrade** para auditoría técnica de código, detección de lookahead bias, Hyperopt y FreqAI.

Adoptá el rol del Maestro Freqtrade internamente cuando Walter:
- Muestre código Python de una estrategia
- Pregunte sobre backtesting, Hyperopt, FreqAI, callbacks
- Pida auditar una estrategia existente
- Reporte comportamiento inesperado en backtest o live

## Estrategias activas

| Archivo | Estado | TF Entry | TF Contexto |
|---------|--------|----------|-------------|
| `SMCWithMLLuxAlgo.py` | En desarrollo | 15m | 1h / 4h |
| `SMCWithMLLuxAlgo5m.py` | En desarrollo | 5m | 1h / 4h |

**Librería SMC:** `smc_luxalgo_numba.py` — implementación NumPy/Numba de Smart Money Concepts alineada con LuxAlgo TradingView.

**Modelo ML:** `train_smc_model.py` — pipeline de entrenamiento externo (no FreqAI).

## Stack técnico del proyecto

- **Framework:** Freqtrade (INTERFACE_VERSION = 3)
- **Exchanges:** Binance Futures, Bitget, Hyperliquid, GRVT
- **Trading mode:** Futures / Isolated margin
- **Can short:** True
- **TF principal:** 5m / 15m
- **TF informativo:** 1h, 4h, 1D

@.agent/rules/freqtrade-expert.md
