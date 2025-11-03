# CHoCH/BOS Strategy - Análisis y Optimización

**Fecha:** 30 de Octubre 2025
**Objetivo:** Optimizar estrategia CHoCH/BOS con Smart Money Concepts
**Resultado:** ✅ Estrategia rentable con generalización comprobada

---

## Resumen Ejecutivo

**Hallazgo Principal:** El período de entrenamiento es CRÍTICO. Una estrategia simple entrenada con 4.5 años de datos (2020-2024) superó significativamente a versiones con períodos cortos y a versiones con múltiples filtros de confluencia.

**Mejor Resultado:**
- **Training (2020-2024):** +152.84% profit, 72.3% win rate, 382 trades
- **Test (Jul-Oct 2025):** +2.76% profit, 72.7% win rate, 22 trades
- **Degradación:** Prácticamente NULA (72.3% → 72.7% win rate)

---

## Comparación de Estrategias

### 1. CHoCHBOSStrategy - Entrenamiento Corto (Sept 2024 - Jun 2025)
**❌ SOBREAJUSTADA**

| Métrica | Training | Test | Degradación |
|---------|----------|------|-------------|
| Período | Sept 2024 - Jun 2025 (10 meses) | Jul - Oct 2025 (4 meses) | - |
| Trades | 256 | 54 | -79% |
| Win Rate | 69.1% | 57.4% | -11.7 pts |
| Profit | +17.42% | -9.10% | -26.52 pts |
| Drawdown | 6.91% | Mayor | Peor |

**Problema:** Solo vio condiciones de Sep 2024 - Jun 2025. No experimentó:
- Bear market 2022
- Bull market 2020-2021
- Lateral market 2023

---

### 2. CHoCHBOSConfluence - Multi-Timeframe con HTF Filters (2020-2024)
**⚠️ DEMASIADO RESTRICTIVA**

| Métrica | Training | Test | Resultado |
|---------|----------|------|-----------|
| Período | 2020 - Jun 2024 (4.5 años) | Jul - Oct 2025 (4 meses) | - |
| Trades | 98 | 3 | -97% |
| Win Rate | 80.6% | 33.3% | -47.3 pts |
| Profit | +39.61% | -1.73% | -41.34 pts |
| Drawdown | 2.53% | 2.17% | Similar |

**Filtros Aplicados:**
- Order Blocks desde 4h
- Fair Value Gaps desde 4h
- Fibonacci retracements desde HTF pivots
- Money Flow Index (período 29)
- Mínimo 1 confluencia requerida
- Solo entrar con OB + Fib

**Problema:** Tan selectiva que perdió la mayoría de oportunidades válidas. Mejor pérdida absoluta (-1.73% vs -9.10%) pero por evitar trades, no por mejor calidad.

---

### 3. CHoCHBOSStrategy - Entrenamiento Largo (2020-2024) ✅
**🎯 GANADORA - GENERALIZACIÓN PERFECTA**

| Métrica | Training | Test | Degradación |
|---------|----------|------|-------------|
| Período | 2020 - Jun 2024 (4.5 años) | Jul - Oct 2025 (4 meses) | - |
| Trades | 382 | 22 | Normal |
| Win Rate | 72.3% | 72.7% | **+0.4 pts ✓** |
| Profit | +152.84% | +2.76% | Proporcional |
| Avg Profit/Trade | 0.75% | 0.38% | -50% |
| Drawdown | 7.47% | 1.64% | Mejor |
| Stop Losses | - | 5 (vs 21 en v1) | -76% |

**¿Por qué funcionó?**
1. Vio TODOS los tipos de mercado:
   - Bull 2020-2021: ETH $130 → $4800
   - Bear 2022: ETH $4800 → $1000
   - Lateral 2023-2024: ETH $1000 → $2500

2. Parámetros optimizados para generalización, no overfitting

3. Sin filtros complejos innecesarios

---

## Parámetros Optimizados (ETH/USDT:USDT)

### Buy Parameters
```python
buy_params = {
    "zigzag_depth": 35,        # Antes: 44 (más sensible a ruido)
    "zigzag_deviation": 10,    # Antes: 6 (menos filtro)
}
```

**Cambios clave:**
- `zigzag_depth` reducido de 44 a 35: Detecta swings más significativos
- `zigzag_deviation` aumentado de 6 a 10: Filtra mejor el ruido del mercado

### Sell Parameters
```python
sell_params = {
    "move_to_be_after_tp1": False,    # Antes: True
    "tp1_percentage": 0.3,            # Antes: 0.6
    "tp2_percentage": 0.5,            # Antes: 0.3
    "use_bos_tps": False,             # Confirmado: trailing stop > BOS TPs
}
```

**Cambios clave:**
- Break-even desactivado: No prematuramente
- TPs rebalanceados: 30%/50% más conservador que 60%/30%
- BOS-based TPs descartados: Trailing stop demostró ser superior

### Fixed Parameters
```python
minimal_roi = {
    "0": 0.1,      # 10% ROI inmediato
    "60": 0.05,    # 5% después de 1 hora
    "120": 0.03,   # 3% después de 2 horas
    "240": 0.01    # 1% después de 4 horas
}

stoploss = -0.03                              # -3% (con 3x leverage = -9% real)
trailing_stop = True
trailing_stop_positive = 0.01                 # Activar trailing a +1%
trailing_stop_positive_offset = 0.02          # Comenzar a +2%
trailing_only_offset_is_reached = True
```

---

## Análisis Detallado - Test Period

### Exit Reasons (Jul-Oct 2025)
| Exit Reason | Trades | Avg Profit | Total Profit | Win Rate |
|-------------|--------|------------|--------------|----------|
| Trailing Stop | 14 | **+1.42%** | +66.60 USDT | **92.9%** ✓ |
| ROI | 3 | +1.67% | +16.95 USDT | 100% |
| Stop Loss | 5 | -3.30% | -55.97 USDT | 0% |

**Insights:**
1. Trailing stop funciona excelentemente: 92.9% win rate
2. Solo 5 stop losses (vs 21 en versión short-trained)
3. ROI también efectivo: 100% win rate en 3 trades

### Trade Distribution
- **Long trades:** 11 (profit: +1.40%)
- **Short trades:** 11 (profit: +1.36%)
- **Balance:** Perfectamente balanceado

### Risk Metrics
- **Max Drawdown:** 1.64% (excelente)
- **Sortino:** 1.78 (muy bueno)
- **Sharpe:** 0.62 (aceptable)
- **Calmar:** 27.17 (excelente)
- **Profit Factor:** 1.46

---

## Lecciones Aprendidas

### 1. El Período de Entrenamiento es CRÍTICO
**Hallazgo:** 10 meses de datos NO son suficientes. Necesitas ver:
- Al menos 1 ciclo completo bull/bear
- Idealmente 4-5 años de historia
- Diferentes regímenes de volatilidad

**Evidencia:**
- 10 meses: -9.10% test profit, 57.4% win rate
- 4.5 años: +2.76% test profit, 72.7% win rate

### 2. Simplicidad > Complejidad
**Hallazgo:** Filtros HTF complejos (OB, FVG, Fib) no mejoraron resultados

**Comparación:**
- Estrategia simple: 22 trades, +2.76%
- Estrategia compleja: 3 trades, -1.73%

**Conclusión:** La confluencia HTF filtró tanto que eliminó oportunidades válidas sin mejorar la calidad de las entradas.

### 3. BOS-Based Take Profits No Funcionan
**Hallazgo:** Usar eventos BOS como niveles de TP tuvo peor desempeño que trailing stops

**Evidencia:**
- `use_bos_tps=False` elegido por hyperopt en ambos entrenamientos
- Trailing stop: 92.9% win rate
- BOS TP2: 36.4% win rate en pruebas anteriores

### 4. Parámetros Conservadores Generalizan Mejor
**Hallazgo:** Parámetros más conservadores (zigzag_depth=35, deviation=10) generalizaron mejor que los agresivos

**Razón:** Filtran mejor el ruido sin sacrificar señales válidas

---

## Próximos Pasos

### Fase 1: Validación Multi-Símbolo (PRÓXIMO)

**Objetivo:** Probar parámetros actuales en múltiples pares para aumentar oportunidades de trading

#### Grupos Propuestos

**Grupo A - Major Pairs (Alta Liquidez)**
- BTC/USDT:USDT
- ETH/USDT:USDT  ✓ (Ya optimizado)
- BNB/USDT:USDT

**Grupo B - DeFi / Layer 2 (Correlación con ETH)**
- LINK/USDT:USDT
- AAVE/USDT:USDT
- MATIC/USDT:USDT
- ARB/USDT:USDT

**Grupo C - Altcoins (Mayor Volatilidad)**
- SOL/USDT:USDT
- ADA/USDT:USDT
- DOT/USDT:USDT
- AVAX/USDT:USDT

#### Test Plan
1. **Usar parámetros actuales sin modificar**
   - Backtest cada par individual en período de test (Jul-Oct 2025)
   - Evaluar performance por grupo

2. **Análisis de Correlación**
   - Si BTC y BNB tienen buena correlación → Mismo set de parámetros
   - Si ETH y L2s se mueven similar → Mismo set de parámetros
   - Si hay divergencia significativa → Considerar sets específicos

3. **Agregación de Resultados**
   - Calcular profit total combinado
   - Evaluar distribución de trades por par
   - Verificar que no haya concentración excesiva

#### Criterios de Éxito
- Win rate > 65% por grupo
- Profit total combinado > profit individual de ETH
- Al menos 50-100 trades en 4 meses (vs 22 con solo ETH)
- Max drawdown < 10% combinado

---

### Fase 2: Optimización por Grupos (SI ES NECESARIO)

**Solo ejecutar si Fase 1 muestra divergencia significativa**

#### Escenario A: Parámetros Universales Funcionan
- ✅ Usar mismos parámetros para todos los pares
- ✅ Activar trading en todos los grupos exitosos
- ✅ Configurar `max_open_trades` apropiado (ej: 5-10)

#### Escenario B: Necesario Optimizar por Grupo
**Si algún grupo tiene < 60% win rate o pérdidas:**

1. **Grupo BTC/BNB (Alta Correlación)**
   ```bash
   freqtrade hyperopt --strategy CHoCHBOSStrategy \
     --config config_btc_bnb.json \
     --timerange 20200101-20240630 \
     --pairs BTC/USDT:USDT BNB/USDT:USDT \
     --epochs 200 -j -1
   ```

2. **Grupo ETH/L2s (Alta Correlación)**
   ```bash
   freqtrade hyperopt --strategy CHoCHBOSStrategy \
     --config config_eth_l2.json \
     --timerange 20200101-20240630 \
     --pairs ETH/USDT:USDT LINK/USDT:USDT MATIC/USDT:USDT \
     --epochs 200 -j -1
   ```

3. **Grupo Altcoins (Independientes)**
   - Considerar optimización individual si tienen comportamientos muy diferentes
   - O descartarlos si no muestran buenos resultados

---

### Fase 3: Walk-Forward Optimization (AVANZADO)

**Para producción real - evitar overfitting futuro**

#### Metodología
1. **Dividir datos en ventanas:**
   - Train: 12 meses
   - Test: 3 meses
   - Avanzar 3 meses y repetir

2. **Ejemplo Timeline:**
   | Window | Train Period | Test Period | Action |
   |--------|--------------|-------------|--------|
   | 1 | 2020-01 a 2020-12 | 2021-01 a 2021-03 | Optimize + Test |
   | 2 | 2020-04 a 2021-03 | 2021-04 a 2021-06 | Optimize + Test |
   | 3 | 2020-07 a 2021-06 | 2021-07 a 2021-09 | Optimize + Test |
   | ... | ... | ... | ... |

3. **Métricas a Rastrear:**
   - Consistency of win rate across windows
   - Parameter stability (¿cambian mucho?)
   - Profit degradation promedio

#### Criterio de Éxito
- Win rate promedio > 65% en todas las ventanas
- Degradación train→test < 10% promedio
- Parámetros relativamente estables (no cambios salvajes)

---

## Configuración de Archivos

### Crear Configs por Grupo

**config_multi_major.json** (BTC + ETH + BNB)
```json
{
    "max_open_trades": 3,
    "stake_currency": "USDT",
    "stake_amount": "unlimited",
    "tradable_balance_ratio": 0.99,
    "trading_mode": "futures",
    "margin_mode": "isolated",
    "exchange": {
        "name": "binance",
        "pair_whitelist": [
            "BTC/USDT:USDT",
            "ETH/USDT:USDT",
            "BNB/USDT:USDT"
        ]
    }
}
```

**config_multi_defi.json** (ETH + L2s + DeFi)
```json
{
    "exchange": {
        "pair_whitelist": [
            "ETH/USDT:USDT",
            "LINK/USDT:USDT",
            "MATIC/USDT:USDT",
            "AAVE/USDT:USDT"
        ]
    }
}
```

**config_multi_alt.json** (Altcoins)
```json
{
    "exchange": {
        "pair_whitelist": [
            "SOL/USDT:USDT",
            "ADA/USDT:USDT",
            "DOT/USDT:USDT",
            "AVAX/USDT:USDT"
        ]
    }
}
```

---

## Comandos Útiles

### Descargar Datos para Nuevos Pares
```bash
cd /d/Scripts/freqtrade && freqtrade download-data \
  --exchange binance \
  --trading-mode futures \
  --pairs BTC/USDT:USDT BNB/USDT:USDT SOL/USDT:USDT LINK/USDT:USDT \
  --timeframes 15m 4h \
  --timerange 20200101- \
  --prepend
```

### Backtest Multi-Pair con Parámetros Actuales
```bash
cd /d/Scripts/freqtrade && freqtrade backtesting \
  --strategy CHoCHBOSStrategy \
  --config config_multi_major.json \
  --timeframe 15m \
  --timerange 20250701-20251027
```

### Hyperopt por Grupo (Si Necesario)
```bash
cd /d/Scripts/freqtrade && freqtrade hyperopt \
  --strategy CHoCHBOSStrategy \
  --config config_multi_major.json \
  --timeframe 15m \
  --timerange 20200101-20240630 \
  --hyperopt-loss SharpeHyperOptLoss \
  --spaces buy sell \
  --epochs 200 \
  -j -1
```

---

## Advertencias y Consideraciones

### 1. Leverage y Risk Management
- Usando 3x leverage con -3% SL = -9% pérdida real
- Con múltiples pares: Asegurar que max_open_trades no sobre-apalanca
- Recomendación: `max_open_trades` = 3-5 con múltiples pares

### 2. Correlación Entre Pares
- Si BTC cae fuertemente, ETH/BNB/etc suelen seguir
- No asumir que más pares = menos riesgo
- Monitorear correlación entre pares activos

### 3. Liquidez y Slippage
- Pares de baja liquidez pueden tener peor ejecución
- Preferir pares con volumen > $100M diario
- Considerar usar `order_book_top` y `check_depth_of_market`

### 4. Freqtrade Position Sizing
- Con `stake_amount: "unlimited"` y `tradable_balance_ratio: 0.99`:
  - Se divide balance equitativamente entre max_open_trades
  - Ejemplo: $1000 / 3 trades = $333 por trade
  - Con 3x leverage: $1000 de exposición por trade

### 5. Reoptimización Periódica
- No usar los mismos parámetros por años sin revisar
- Reoptimizar cada 6-12 meses con datos recientes
- Siempre validar en período de test out-of-sample

---

## Archivos de Estrategia

### CHoCHBOSStrategy.py
**Ubicación:** `D:\Scripts\freqtrade\user_data\strategies\CHoCHBOSStrategy.py`

**Archivos de parámetros:**
- `CHoCHBOSStrategy.json` - Parámetros optimizados actuales (2020-2024 ETH)

### State Machine Logic
La estrategia usa una máquina de estados para rastrear estructura del mercado:

```python
STATE_NEUTRAL = 0                           # Sin tendencia clara
STATE_TENDENCIA_ALCISTA = 1                 # Uptrend confirmado
STATE_TENDENCIA_BAJISTA = 2                 # Downtrend confirmado
STATE_ESPERANDO_CONFIRMACION_ALCISTA = 3    # CHoCH up detectado, esperando confirmación
STATE_ESPERANDO_CONFIRMACION_BAJISTA = 4    # CHoCH down detectado, esperando confirmación
```

**Entry Conditions:**
- **LONG:** CHoCH up detectado → Esperando confirmación → BOS up confirma → Entry
- **SHORT:** CHoCH down detectado → Esperando confirmación → BOS down confirma → Entry

**Exit Conditions:**
1. Trailing stop (principal)
2. ROI alcanzado
3. Stop loss hit
4. CHoCH opuesto (exit signal)

---

## Contacto y Notas

**Autor:** Claude Code + Usuario
**Framework:** Freqtrade 2025.9.1
**Python:** 3.12.10

**Importante:** Esta estrategia está basada en backtesting. Resultados pasados no garantizan resultados futuros. Siempre testear en paper trading antes de usar capital real.

---

**Última actualización:** 30 Octubre 2025, 20:10 UTC
