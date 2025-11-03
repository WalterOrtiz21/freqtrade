# CHoCH/BOS - Guía Rápida para Empezar

## Configuración Rápida (5 minutos)

### 1. Editar Config Base

Edita `config_choch_bos_live.json`:

```json
{
    "dry_run": true,  // Cambiar a false para trading real
    "exchange": {
        "name": "binance",  // Por defecto: Binance Futures
        "key": "TU_API_KEY",
        "secret": "TU_API_SECRET",
        "pair_whitelist": [
            "SOL/USDT:USDT",   // Los mejores pares del backtest
            "ADA/USDT:USDT",
            "LINK/USDT:USDT",
            "AAVE/USDT:USDT"
        ]
    }
}
```

### 2. Ejecutar Bot

**Opción A: Script interactivo (recomendado)**
```bash
./start_trading.sh
```

**Opción B: Línea de comandos**
```bash
# Binance Futures (por defecto) - 15m
./start_trading.sh binance_futures 15m dry

# Binance Spot - 15m
./start_trading.sh binance_spot 15m dry

# Hyperliquid - 15m
./start_trading.sh hyperliquid 15m dry
```

**Opción C: Comando directo**
```bash
cd /d/Scripts/freqtrade
/c/Users/walte/AppData/Local/Programs/Python/Python312/Scripts/freqtrade.exe trade \
    --strategy CHoCHBOSStrategy \
    --config config_choch_bos_live.json \
    --timeframe 15m
```

---

## Cambiar Exchange

### Por Defecto: Binance Futures
Ya está configurado en `config_choch_bos_live.json`

### Cambiar a Binance Spot:

**Método 1: Usar config override**
```bash
./start_trading.sh binance_spot 15m dry
```

**Método 2: Editar config base**
Edita `config_choch_bos_live.json`:
```json
{
    "trading_mode": "spot",        // Cambiar de "futures" a "spot"
    "margin_mode": "",             // Dejar vacío
    "stake_amount": 100,           // Cambiar de "unlimited" a cantidad fija
    "exchange": {
        "pair_whitelist": [
            "SOL/USDT",            // Quitar ":USDT" del final
            "ADA/USDT",
            "LINK/USDT"
        ]
    }
}
```

### Cambiar a Hyperliquid:

**Método 1: Usar config override**
```bash
./start_trading.sh hyperliquid 15m dry
```

**Método 2: Editar config base**
Edita `config_choch_bos_live.json`:
```json
{
    "stake_currency": "USDC",      // Cambiar de "USDT" a "USDC"
    "margin_mode": "cross",        // Cambiar de "isolated" a "cross"
    "exchange": {
        "name": "hyperliquid",     // Cambiar de "binance" a "hyperliquid"
        "key": "TU_WALLET_ADDRESS",
        "secret": "TU_PRIVATE_KEY",
        "pair_whitelist": [
            "SOL/USDC:USDC",       // Usar USDC en vez de USDT
            "ADA/USDC:USDC"
        ]
    }
}
```

---

## Cambiar Timeframe

### 15m (Recomendado - Optimizado)
```bash
./start_trading.sh binance_futures 15m dry
```

**Rendimiento esperado (backtest):**
- +20.54% en 4 meses (Altcoins)
- Win rate: 71.3%
- 1-2 trades por día

### 5m (Experimental - Pendiente optimización)
```bash
./start_trading.sh binance_futures 5m dry
```

**Rendimiento esperado (sin optimizar):**
- +10.55% en 4 meses (Altcoins)
- Win rate: 67.4%
- 3-4 trades por día

**Nota:** Espera resultados de hyperopt para parámetros óptimos en 5m.

---

## Cambiar Estrategia

Edita `start_trading.sh` línea 13:
```bash
STRATEGY="CHoCHBOSStrategy"  # Cambiar al nombre de tu estrategia
```

O usa comando directo:
```bash
freqtrade.exe trade \
    --strategy TuEstrategiaNueva \
    --config config_choch_bos_live.json \
    --timeframe 15m
```

---

## Parámetros de la Estrategia

Los parámetros optimizados están en `user_data/strategies/CHoCHBOSStrategy.json`:

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

**Estos son óptimos para 15m.** Para 5m, espera hyperopt.

---

## Monitoreo

### FreqUI (Interfaz Web)
```
http://127.0.0.1:8080
```

**Credenciales:**
- Usuario: `freqtrader`
- Contraseña: `CHANGE_THIS_PASSWORD` (cámbialo en el config!)

### Ver Logs
```bash
tail -f user_data/logs/freqtrade.log
```

### Verificar Status
```bash
freqtrade.exe status --config config_choch_bos_live.json
```

---

## Parar el Bot

**En la terminal donde corre:**
```bash
CTRL+C
```

**O forzar cierre:**
```bash
pkill -f freqtrade
```

**Cerrar todos los trades:**
```bash
freqtrade.exe forceexit all --config config_choch_bos_live.json
```

---

## Checklist Antes de LIVE Trading

- [ ] Probado en dry-run por al menos 1 semana
- [ ] API key configurado correctamente
- [ ] API key con permisos de trading (NO retiro)
- [ ] `"dry_run": false` en config
- [ ] Balance suficiente en cuenta
- [ ] Timeframe correcto (15m recomendado)
- [ ] Pares correctos (evitar BTC)
- [ ] Telegram configurado (opcional pero recomendado)
- [ ] Contraseña de FreqUI cambiada

---

## Estructura de Archivos

```
config_choch_bos_live.json              # Config base (Binance Futures por defecto)
config.exchange.binance_spot.json       # Override para Binance Spot
config.exchange.hyperliquid.json        # Override para Hyperliquid
start_trading.sh                        # Script launcher
LIVE_TRADING_GUIDE.md                   # Guía completa detallada
START_TRADING_QUICK_GUIDE.md            # Esta guía rápida
```

---

## Solución Rápida de Problemas

### "Invalid API key"
- Verifica key/secret en config
- Checa permisos del API key en Binance

### "No buy signals"
- Normal si no hay CHoCH/BOS patterns
- Espera, el bot encontrará señales

### "Insufficient balance"
- Reduce `max_open_trades` o `stake_amount`
- Asegura tener balance suficiente

### Bot muy lento
- Usa 15m en vez de 5m
- Reduce número de pares

---

## Soporte

**Documentación completa:** `LIVE_TRADING_GUIDE.md`

**Análisis de estrategia:** `CHOCH_BOS_STRATEGY_ANALYSIS.md`

**Freqtrade docs:** https://www.freqtrade.io/
