# Freqtrade — Estrategias Algorítmicas

## Contexto del proyecto
Este repositorio contiene las estrategias algorítmicas de TradeCo AI corriendo sobre el framework **Freqtrade**. La estrategia `SMCWithMLLuxAlgo.py` está **en producción con capital real**.

## Equipo disponible

| Especialista | Cómo invocarlo | Cuándo |
|--------------|----------------|--------|
| Maestro Freqtrade | `/ft-audit [archivo]` | Auditoría de código, lookahead bias, Hyperopt, FreqAI |
| Arquitecto SMC/ICT | `/smc [idea]` | Validar lógica de mercado, OBs/FVGs/estructura/liquidez |
| Analista Forense | `/bt-forensics [estrategia]` | Walk-forward, Monte Carlo, detección de curve fitting |
| Axel (ML Engineer) | `Task` → `/home/wortiz/tradeco/agents/ml/CLAUDE.md` | Pipeline XGBoost, feature drift, threshold calibration |
| Sofia (Quant) | `Task` → `/home/wortiz/tradeco/agents/quant/CLAUDE.md` | Métricas de backtest, validación estadística |
| Dex (Risk) | `Task` → `/home/wortiz/tradeco/agents/risk/CLAUDE.md` | Stoploss, position sizing, circuit breakers |
| Rex (Live Ops) | `Task` → `/home/wortiz/tradeco/agents/live-ops/CLAUDE.md` | Monitoreo de producción, live vs backtest, alertas |

## Pipeline de validación antes de deployar

Toda estrategia nueva o modificación relevante debe pasar por este orden:

```
1. /ft-audit       → código limpio, sin lookahead
2. /smc            → lógica de mercado válida
3. /bt-forensics   → edge real, no curve fitting
4. Axel            → pipeline ML validado (si aplica)
5. Dex             → parámetros de riesgo aprobados
6. Sofia           → métricas finales documentadas
7. Dry-run 2 sem.  → Rex monitorea live vs backtest
8. Real            → Rex en alerta continua
```

## Estrategias activas

| Archivo | Estado | TF Entry | TF Contexto |
|---------|--------|----------|-------------|
| `SMCWithMLLuxAlgo.py` | **PRODUCCIÓN (real)** | 15m | 1h / 4h |

**Librería SMC:** `smc_luxalgo_numba.py` — implementación NumPy/Numba de Smart Money Concepts alineada con LuxAlgo TradingView.

**Modelo ML:** `train_smc_model.py` + `smc_xgboost_model.pkl` — XGBoost, actualmente con `use_ml_filter=False`.

## Stack técnico del proyecto

- **Framework:** Freqtrade (INTERFACE_VERSION = 3)
- **Exchanges:** Binance Futures, Bitget, Hyperliquid, GRVT
- **Trading mode:** Futures / Isolated margin
- **Can short:** True
- **TF principal:** 15m
- **TF informativo:** 1h, 4h
