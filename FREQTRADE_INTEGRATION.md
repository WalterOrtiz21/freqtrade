# Freqtrade Integration - DYNAMIC Strategy

## Overview

Este documento explica cómo usar Freqtrade para optimización automática de la estrategia DYNAMIC-Aggressive High TP.

**¿Por qué Freqtrade?**
- Optimización automática (Hyperopt) con validación cruzada
- Detección de overfitting incorporada (walk-forward analysis)
- Backtesting robusto con múltiples timeframes
- Búsqueda de parámetros por fuerza bruta sin overfitting

## Estado Actual

✅ **Estrategia creada**: `user_data/strategies/DynamicAggressiveHighTP.py`
❌ **Freqtrade no instalado**: Requiere Python 3.11+ (tienes 3.8)

## Instalación

### Opción 1: Actualizar Python (RECOMENDADO)

```bash
# Instalar Python 3.11+ desde python.org
# Luego:
cd /d/trading/freqtrade
./setup.sh -i
```

### Opción 2: Usar Docker (si tienes Docker instalado)

```bash
cd /d/trading/freqtrade
docker-compose up -d
```

## Uso

### 1. Configuración Inicial

Crear `user_data/config.json`:

```json
{
  "max_open_trades": 5,
  "stake_currency": "USDT",
  "stake_amount": "unlimited",
  "tradable_balance_ratio": 0.05,
  "fiat_display_currency": "USD",
  "timeframe": "5m",
  "dry_run": true,
  "cancel_open_orders_on_exit": false,
  "trading_mode": "futures",
  "margin_mode": "isolated",
  "unfilledtimeout": {
    "entry": 10,
    "exit": 10,
    "exit_timeout_count": 0,
    "unit": "minutes"
  },
  "entry_pricing": {
    "price_side": "same",
    "use_order_book": true,
    "order_book_top": 1,
    "price_last_balance": 0.0,
    "check_depth_of_market": {
      "enabled": false,
      "bids_to_ask_delta": 1
    }
  },
  "exit_pricing": {
    "price_side": "same",
    "use_order_book": true,
    "order_book_top": 1
  },
  "exchange": {
    "name": "binance",
    "key": "your_api_key",
    "secret": "your_secret",
    "ccxt_config": {},
    "ccxt_async_config": {},
    "pair_whitelist": [
      "BTC/USDT",
      "ETH/USDT",
      "SOL/USDT",
      "BNB/USDT",
      "SUI/USDT",
      "LTC/USDT"
    ],
    "pair_blacklist": []
  },
  "pairlists": [
    {
      "method": "StaticPairList"
    }
  ],
  "telegram": {
    "enabled": false
  },
  "api_server": {
    "enabled": true,
    "listen_ip_address": "127.0.0.1",
    "listen_port": 8080,
    "verbosity": "info",
    "enable_openapi": true,
    "jwt_secret_key": "your_secret_key",
    "CORS_origins": [],
    "username": "freqtrader",
    "password": "SuperSecurePassword"
  },
  "bot_name": "freqtrade_dynamic",
  "initial_state": "running",
  "force_entry_enable": false,
  "internals": {
    "process_throttle_secs": 5
  }
}
```

### 2. Descargar Datos Históricos

```bash
# Descargar 365 días de datos en 5m
freqtrade download-data \
  --exchange binance \
  --pairs BTC/USDT ETH/USDT SOL/USDT BNB/USDT SUI/USDT LTC/USDT \
  --timeframes 5m \
  --days 365
```

### 3. Backtesting (Validación)

```bash
# Backtest básico (365 días)
freqtrade backtesting \
  --strategy DynamicAggressiveHighTP \
  --timerange 20240101-20241231 \
  --config user_data/config.json

# Backtest con análisis detallado
freqtrade backtesting \
  --strategy DynamicAggressiveHighTP \
  --timerange 20240101-20241231 \
  --breakdown month \
  --export trades
```

**Interpretar resultados:**
- **Total Return**: Debe ser cercano a +11.71% promedio
- **Win Rate**: Esperado ~57%
- **Max Drawdown**: Monitorear drawdowns grandes (overfitting signal)
- **Sharpe Ratio**: Mayor = mejor risk-adjusted return

### 4. Hyperopt (Optimización Automática)

**ESTO ES LO QUE NECESITAS** - Búsqueda automática de mejores parámetros:

```bash
# Optimizar parámetros de entrada (RSI threshold, momentum, etc.)
freqtrade hyperopt \
  --strategy DynamicAggressiveHighTP \
  --hyperopt-loss SharpeHyperOptLoss \
  --spaces buy \
  --epochs 1000 \
  --timerange 20240101-20241231

# Optimizar trailing stop distances
freqtrade hyperopt \
  --strategy DynamicAggressiveHighTP \
  --hyperopt-loss SharpeHyperOptLoss \
  --spaces roi stoploss trailing \
  --epochs 1000 \
  --timerange 20240101-20241231

# Optimización completa (CUIDADO: puede overfittear)
freqtrade hyperopt \
  --strategy DynamicAggressiveHighTP \
  --hyperopt-loss SharpeHyperOptLoss \
  --spaces all \
  --epochs 500 \
  --timerange 20240101-20241231
```

**Funciones de pérdida disponibles:**
- `SharpeHyperOptLoss`: Maximiza Sharpe ratio (RECOMENDADO)
- `SortinoHyperOptLoss`: Similar a Sharpe pero solo penaliza downside
- `CalmarHyperOptLoss`: Return / Max Drawdown
- `MaxDrawDownRelativeHyperOptLoss`: Minimiza drawdown
- `ProfitDrawDownHyperOptLoss`: Balance profit vs drawdown

**Prevenir overfitting:**
```bash
# 1. Usa walk-forward analysis (divide datos en train/test)
freqtrade hyperopt \
  --strategy DynamicAggressiveHighTP \
  --hyperopt-loss SharpeHyperOptLoss \
  --spaces buy sell \
  --epochs 1000 \
  --timerange 20240101-20240930  # Solo primeros 9 meses

# 2. Valida en período fuera de muestra
freqtrade backtesting \
  --strategy DynamicAggressiveHighTP \
  --timerange 20241001-20241231  # Últimos 3 meses (out-of-sample)
```

### 5. Análisis de Resultados

```bash
# Ver trades detallados
freqtrade backtesting-analysis \
  --analysis-groups 0 1 2 3 4 5

# Generar gráficos
freqtrade plot-dataframe \
  --strategy DynamicAggressiveHighTP \
  --pairs BTC/USDT \
  --indicators1 rsi ma5 \
  --indicators2 momentum

# Plot de profit
freqtrade plot-profit \
  --strategy DynamicAggressiveHighTP \
  --timerange 20240101-20241231
```

## Workflow de Optimización

### Proceso Completo (Anti-Overfitting)

```bash
# 1. Download data (365 días)
freqtrade download-data --exchange binance --pairs BTC/USDT ETH/USDT SOL/USDT BNB/USDT SUI/USDT LTC/USDT --timeframes 5m --days 365

# 2. Divide datos: 270 días train, 95 días test
TRAIN_PERIOD="20240101-20240927"  # ~9 meses
TEST_PERIOD="20240928-20241231"   # ~3 meses

# 3. Hyperopt en período train ONLY
freqtrade hyperopt \
  --strategy DynamicAggressiveHighTP \
  --hyperopt-loss SharpeHyperOptLoss \
  --spaces buy sell roi stoploss trailing \
  --epochs 1000 \
  --timerange $TRAIN_PERIOD

# 4. Freqtrade te dará mejores parámetros - cópialos a la estrategia
# Ejemplo output:
#   Best result:
#     RSI threshold: 28
#     Momentum threshold: -0.025
#     Trailing activation: 0.025
#     etc...

# 5. Valida en período TEST (out-of-sample)
freqtrade backtesting \
  --strategy DynamicAggressiveHighTP \
  --timerange $TEST_PERIOD

# 6. Compara resultados:
#    - Si test >> train: Lucky (repite experimento)
#    - Si test ≈ train: GOOD! Parámetros robustos
#    - Si test << train: OVERFITTING (reduce epochs o simplifica)

# 7. Si resultados son consistentes, backtest completo
freqtrade backtesting \
  --strategy DynamicAggressiveHighTP \
  --timerange 20240101-20241231 \
  --breakdown month
```

### Espacios de Búsqueda (Hyperopt Spaces)

**`--spaces buy`**: Optimiza condiciones de entrada
- RSI threshold
- Momentum threshold
- MA periods

**`--spaces sell`**: Optimiza condiciones de salida (si usas)

**`--spaces roi`**: Optimiza take profit levels

**`--spaces stoploss`**: Optimiza stop loss level

**`--spaces trailing`**: Optimiza trailing stop parameters

**`--spaces all`**: Optimiza todo (PELIGRO: overfitting fácil)

## Limitaciones con Pacifica Exchange

**Freqtrade NO soporta Pacifica directamente.**

**Opciones:**

1. **Usar Freqtrade solo para backtesting/hyperopt offline:**
   - Optimiza parámetros en Binance/otro exchange con data similar
   - Valida parámetros encontrados
   - Implementa parámetros validados en tu bot Pacifica actual

2. **Crear custom exchange adapter** (avanzado):
   - Implementar CCXT-compatible exchange class
   - Requiere desarrollo significativo

3. **Usar datos de Pacifica en Freqtrade** (MEJOR OPCIÓN):
   - Exporta tus JSONs históricos de Pacifica a formato Freqtrade
   - Corre hyperopt offline con esos datos
   - Implementa parámetros en tu bot

## Convertir Datos Pacifica → Freqtrade

Crear script `convert_pacifica_to_freqtrade.py`:

```python
import json
import pandas as pd
from pathlib import Path

def convert_pacifica_to_freqtrade(input_file, output_dir, symbol):
    """Convert Pacifica JSON format to Freqtrade JSON format."""

    # Load Pacifica data
    with open(input_file, 'r') as f:
        data = json.load(f)

    # Extract data (adjust based on your JSON structure)
    if isinstance(data, dict) and symbol in data:
        candles = data[symbol]
    elif 'data' in data:
        candles = data['data']
    else:
        candles = data

    # Convert to DataFrame
    df = pd.DataFrame(candles)

    # Ensure correct columns (Freqtrade expects: timestamp, open, high, low, close, volume)
    if 'price' in df.columns:
        df['close'] = df['price']

    # Ensure timestamp is in milliseconds
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp']).astype(int) // 10**6

    # Select required columns
    freqtrade_data = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].to_dict('records')

    # Save in Freqtrade format
    output_file = Path(output_dir) / f"{symbol.lower()}_usdt-5m.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(freqtrade_data, f)

    print(f"Converted {symbol}: {len(freqtrade_data)} candles -> {output_file}")

# Convert all symbols
symbols = ['BTC', 'ETH', 'SOL', 'BNB', 'SUI', 'LTC']
for symbol in symbols:
    convert_pacifica_to_freqtrade(
        f'../pacifica-python-sdk/data/historical/{symbol.lower()}_usdt_365days_5m.json',
        'user_data/data/binance',  # Freqtrade data location
        symbol
    )
```

## Próximos Pasos

1. **Actualizar Python a 3.11+**
2. **Instalar Freqtrade**: `cd /d/trading/freqtrade && ./setup.sh -i`
3. **Convertir tus datos históricos** a formato Freqtrade
4. **Correr hyperopt** para encontrar mejores parámetros
5. **Validar con walk-forward** para evitar overfitting
6. **Implementar parámetros validados** en tu bot `trading_bot_ws.py`

## Recursos

- [Freqtrade Docs](https://www.freqtrade.io/en/stable/)
- [Hyperopt Guide](https://www.freqtrade.io/en/stable/hyperopt/)
- [Strategy Development](https://www.freqtrade.io/en/stable/strategy-customization/)
- [Backtesting](https://www.freqtrade.io/en/stable/backtesting/)

## Preguntas Frecuentes

**Q: ¿Puedo usar Freqtrade para trading live en Pacifica?**
A: No directamente. Usa Freqtrade para optimización offline, luego implementa parámetros en tu bot actual.

**Q: ¿Cómo evito overfitting?**
A: Usa walk-forward analysis (train en 9 meses, test en 3 meses out-of-sample).

**Q: ¿Cuántos epochs debo usar?**
A: 500-1000 es bueno. Más epochs = más riesgo de overfitting.

**Q: ¿Qué hyperopt loss usar?**
A: `SharpeHyperOptLoss` es más robusto. `CalmarHyperOptLoss` si te preocupa drawdown.

**Q: ¿Puedo optimizar todos los parámetros a la vez?**
A: Sí, pero incrementa riesgo de overfitting. Mejor optimiza por partes (primero entry, luego exit).
