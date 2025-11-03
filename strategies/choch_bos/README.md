# CHoCH/BOS Trading Strategy

Estrategia de trading basada en Smart Money Concepts: Change of Character (CHoCH) y Break of Structure (BOS).

## 📊 Rendimiento (Backtest)

**Periodo de prueba:** Jul-Oct 2025 (4 meses)
**Timeframe:** 15m (optimizado)

| Grupo | Profit | Win Rate | Trades | Mejor Par |
|-------|--------|----------|--------|-----------|
| **Altcoins** | +20.54% | 71.3% | 143 | SOL +10.59% |
| **DeFi** | +6.89% | 70.8% | 96 | ETH +4.90% |

**Pares recomendados:** SOL, ADA, DOT, AVAX, LINK, AAVE, BNB
**Evitar:** BTC (-5.89% en backtest)

---

## 🚀 Quick Start

### Opción 1: Script Interactivo (Recomendado)
```bash
cd D:/Scripts/freqtrade/strategies/choch_bos/scripts
./start_trading.sh
```

### Opción 2: Comando Directo
```bash
cd D:/Scripts/freqtrade
/c/Users/walte/AppData/Local/Programs/Python/Python312/Scripts/freqtrade.exe trade \
    --strategy CHoCHBOSStrategy \
    --config strategies/choch_bos/configs/config_choch_bos_live.json \
    --timeframe 15m
```

---

## 📁 Estructura de Archivos

```
strategies/choch_bos/
│
├── README.md                          # Este archivo
├── CHoCHBOSStrategy.py               # Código de la estrategia (copia)
├── CHoCHBOSStrategy.json             # Parámetros optimizados (copia)
│
├── configs/                          # Configuraciones
│   ├── config_choch_bos_live.json   # Config principal (Binance Futures)
│   ├── config.exchange.binance_spot.json    # Override para Spot
│   ├── config.exchange.hyperliquid.json     # Override para Hyperliquid
│   └── backtest/                     # Configs de backtest/hyperopt
│       ├── config_5m_hyperopt.json
│       ├── config_basic_hyperopt_longterm.json
│       ├── config_eth.json
│       ├── config_multi_alt.json
│       ├── config_multi_defi.json
│       └── config_multi_major.json
│
├── docs/                             # Documentación
│   ├── START_TRADING_QUICK_GUIDE.md # Guía rápida (5 min)
│   ├── LIVE_TRADING_GUIDE.md        # Guía completa detallada
│   └── CHOCH_BOS_STRATEGY_ANALYSIS.md # Análisis técnico
│
└── scripts/                          # Scripts utilitarios
    └── start_trading.sh             # Launcher de trading

NOTA: La estrategia original sigue en:
  - user_data/strategies/CHoCHBOSStrategy.py
  - user_data/strategies/CHoCHBOSStrategy.json
```

---

## ⚙️ Configuración Rápida

### 1. Editar Config Principal
```bash
nano configs/config_choch_bos_live.json
```

Cambiar:
- `"key"`: Tu API key de Binance
- `"secret"`: Tu API secret de Binance
- `"dry_run": false` cuando estés listo para trading real

### 2. Cambiar Exchange (Opcional)

**Binance Futures (por defecto):** Ya está configurado

**Binance Spot:**
```bash
./scripts/start_trading.sh binance_spot 15m dry
```

**Hyperliquid:**
```bash
./scripts/start_trading.sh hyperliquid 15m dry
```

O edita `configs/config_choch_bos_live.json` directamente.

---

## 📈 Parámetros Optimizados

Configurados en `CHoCHBOSStrategy.json`:

```json
{
  "params": {
    "buy": {
      "zigzag_depth": 35,
      "zigzag_deviation": 10
    },
    "sell": {
      "use_bos_tps": false,
      "move_to_be_after_tp1": false
    }
  }
}
```

**Optimizados para:**
- Timeframe: 15m
- Periodo: 2020-2024 (4.5 años de entrenamiento)
- Win rate: 70-72%
- Sharpe ratio: 2.72 (Altcoins)

**Para 5m:** Hyperopt en progreso, parámetros pendientes.

---

## 📚 Documentación

- **[Guía Rápida](docs/START_TRADING_QUICK_GUIDE.md)** - Empezar en 5 minutos
- **[Guía Completa](docs/LIVE_TRADING_GUIDE.md)** - Documentación detallada
- **[Análisis de Estrategia](docs/CHOCH_BOS_STRATEGY_ANALYSIS.md)** - Detalles técnicos

---

## 🔧 Comandos Útiles

### Trading en Vivo
```bash
# Dry run (papel)
./scripts/start_trading.sh binance_futures 15m dry

# LIVE trading (dinero real!)
./scripts/start_trading.sh binance_futures 15m live
```

### Backtest
```bash
cd ../../..  # Volver a raíz de freqtrade

freqtrade backtesting \
    --strategy CHoCHBOSStrategy \
    --config strategies/choch_bos/configs/backtest/config_multi_alt.json \
    --timeframe 15m \
    --timerange 20250701-20251027
```

### Hyperopt (Optimización)
```bash
freqtrade hyperopt \
    --strategy CHoCHBOSStrategy \
    --config strategies/choch_bos/configs/backtest/config_5m_hyperopt.json \
    --timeframe 5m \
    --timerange 20200101-20240630 \
    --hyperopt-loss SharpeHyperOptLoss \
    --spaces buy sell \
    --epochs 200 \
    -j -1
```

### Ver Status
```bash
freqtrade status \
    --config strategies/choch_bos/configs/config_choch_bos_live.json
```

### Ver Logs
```bash
tail -f ../../user_data/logs/freqtrade.log
```

---

## 🌐 FreqUI (Interfaz Web)

**URL:** http://127.0.0.1:8080

**Credenciales por defecto:**
- Usuario: `freqtrader`
- Password: `CHANGE_THIS_PASSWORD`

⚠️ **Cambiar password en config antes de producción!**

---

## ⚠️ Disclaimer

**Esta estrategia involucra trading de criptomonedas con riesgo significativo.**

- Resultados pasados no garantizan rendimiento futuro
- Usa solo dinero que puedas perder
- Prueba SIEMPRE en dry-run antes de live trading
- La estrategia fue optimizada con datos históricos (2020-2024)
- Condiciones de mercado pueden cambiar

---

## 📝 Changelog

**v1.0.0 - Oct 30, 2025**
- Estrategia CHoCH/BOS optimizada para 15m
- Backtest: +20.54% (Altcoins), +6.89% (DeFi) en 4 meses
- Sistema multi-exchange (Binance, Hyperliquid)
- Scripts de automatización
- Documentación completa

**En desarrollo:**
- Optimización para 5m timeframe
- Walk-forward optimization
- Soporte para más exchanges
