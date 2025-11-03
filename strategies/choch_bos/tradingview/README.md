# CHoCH/BOS Strategy - TradingView Pine Script v6

**Implementación correcta** de la estrategia CHoCH/BOS siguiendo las reglas exactas de Smart Money Concepts.

---

## 🎯 LA REGLA DE ORO

**CHoCH DEBE ser confirmado OBLIGATORIAMENTE por BOS para entrar**

1. **CHoCH** (Change of Character) = Señal de PREPARACIÓN
2. **BOS** (Break of Structure) = Señal de CONFIRMACIÓN
3. **ENTRY** = Solo cuando CHoCH + BOS en misma dirección

---

## 📋 Quick Start

### 1. Importar a TradingView

1. Abre TradingView: https://www.tradingview.com/
2. Click en "Pine Editor" (abajo)
3. Copia todo el contenido de `CHoCH_BOS_Strategy.pine`
4. Pega en Pine Editor
5. Click "Add to Chart"

---

## 🔑 Conceptos Clave

### CHoCH (Change of Character)

- **Qué es:** Ruptura de estructura GRANDE (50+ velas por defecto)
- **Qué significa:** Cambio potencial de tendencia
- **Acción:** NO entrar inmediatamente, ESPERAR confirmación
- **Visual:** 🟢 Círculo grande verde (alcista) o 🔴 Círculo grande rojo (bajista)

### BOS (Break of Structure)

- **Qué es:** Ruptura de estructura PEQUEÑA (5+ velas por defecto)
- **Qué significa:** Continuación de tendencia
- **Acción:** Si sigue a CHoCH en misma dirección → CONFIRMA la entrada
- **Visual:** 💚 Diamante pequeño verde (alcista) o 🟠 Diamante pequeño naranja (bajista)

---

## 📊 Flujo de Trading

### Para LONG:

```
1. Aparece CHoCH Bullish (🟢) → "Preparándose para posible long"
   └─ Fondo VERDE CLARO = "Esperando confirmación BOS"

2. Aparece BOS Bullish (💚) → "Confirmación recibida"
   └─ Etiqueta "LONG (CHoCH+BOS)" = ENTRAR

3. Mantenerse en long mientras:
   ├─ Aparecen más BOS bullish → Tendencia continúa fuerte
   └─ NO aparece CHoCH Bearish

4. Aparece CHoCH Bearish (🔴) → SALIR INMEDIATAMENTE
```

### Para SHORT:

```
1. Aparece CHoCH Bearish (🔴) → "Preparándose para posible short"
   └─ Fondo NARANJA CLARO = "Esperando confirmación BOS"

2. Aparece BOS Bearish (🟠) → "Confirmación recibida"
   └─ Etiqueta "SHORT (CHoCH+BOS)" = ENTRAR

3. Mantenerse en short mientras:
   ├─ Aparecen más BOS bearish → Tendencia continúa fuerte
   └─ NO aparece CHoCH Bullish

4. Aparece CHoCH Bullish (🟢) → SALIR INMEDIATAMENTE
```

---

## ⚙️ Parámetros Configurables

### Detección de Pivotes (ZigZag)

**Pivot Lookback: 5 (estándar)**
- Velas hacia atrás y adelante para confirmar pivote
- Estándar = 5+5+1 (punto 11)
- Aumentar = menos pivotes, más confiables
- Disminuir = más pivotes, más señales

### Tipos de Estructura

**Internal Structure: 5 velas**
- Mínimo de velas para que ruptura sea BOS
- Estructuras pequeñas = continuación de tendencia

**Swing Structure: 50 velas**
- Mínimo de velas para que ruptura sea CHoCH
- Estructuras grandes = cambio de carácter

**¿Cómo funciona la clasificación?**
```
Si precio rompe un nivel y han pasado:
  >= 50 velas desde ese nivel → CHoCH (cambio de carácter)
  >= 5 velas pero < 50 → BOS (continuación)
  < 5 velas → No se marca (ruido)
```

### Risk Management

- **Stop Loss:** -5.8% (ajustable)
- **Trailing Stop:** Activado por defecto
- **Trailing Positive:** 9.9%
- **Trailing Offset:** 19.5%

---

## 🎨 Elementos Visuales

### En el Gráfico

| Símbolo | Significado | Acción |
|---------|-------------|--------|
| 🔺 Triángulo rojo (arriba) | Pivot High confirmado | Máximo significativo |
| 🔻 Triángulo verde (abajo) | Pivot Low confirmado | Mínimo significativo |
| ━━━ Línea roja | Último High | Nivel de resistencia actual |
| ━━━ Línea verde | Último Low | Nivel de soporte actual |
| 🟢 Círculo grande verde | CHoCH Bullish | Prepárate para long, espera BOS |
| 🔴 Círculo grande rojo | CHoCH Bearish | Prepárate para short, espera BOS |
| 💚 Diamante pequeño verde | BOS Bullish | Continuación alcista / Confirma CHoCH |
| 🟠 Diamante pequeño naranja | BOS Bearish | Continuación bajista / Confirma CHoCH |
| 🟩 "LONG (CHoCH+BOS)" | Señal de entrada long | ENTRAR EN LONG |
| 🟥 "SHORT (CHoCH+BOS)" | Señal de entrada short | ENTRAR EN SHORT |
| ❌ "EXIT" | Señal de salida | CERRAR POSICIÓN |

### Fondos de Color

| Color | Estado | Descripción |
|-------|--------|-------------|
| Verde claro (transparente) | Esperando BOS long | CHoCH bullish detectado, esperando confirmación |
| Naranja claro (transparente) | Esperando BOS short | CHoCH bearish detectado, esperando confirmación |
| Sin color | Neutral | No hay CHoCH pendiente de confirmación |

### Tabla de Información (Esquina superior derecha)

**Status:**
- `NEUTRAL` - Sin posición, sin CHoCH pendiente
- `WAIT LONG BOS` - CHoCH bullish detectado, esperando BOS
- `WAIT SHORT BOS` - CHoCH bearish detectado, esperando BOS
- `IN LONG` - En posición long activa
- `IN SHORT` - En posición short activa

**Otros datos:**
- Last High/Low: Niveles actuales
- Pivot Lookback: Configuración de detección
- Internal/Swing Struct: Umbrales de clasificación
- Stop Loss: % de SL configurado
- Position: Estado actual de posición

---

## 📖 Ejemplos de Uso

### Ejemplo 1: Entrada Long Perfecta

```
Barra 100: Precio rompe máximo de hace 55 velas
           → CHoCH Bullish 🟢 (>50 velas = estructura grande)
           → Fondo verde claro = "Esperando BOS"

Barra 107: Precio rompe máximo de hace 8 velas
           → BOS Bullish 💚 (8 velas = estructura pequeña)
           → Label "LONG (CHoCH+BOS)" = ENTRAR LONG ✅

Barra 120: BOS Bullish 💚 (continuación)
           → Mantener posición

Barra 135: BOS Bullish 💚 (continuación)
           → Mantener posición

Barra 150: CHoCH Bearish 🔴
           → EXIT ❌ → SALIR DE LONG
```

### Ejemplo 2: CHoCH sin Confirmación (NO ENTRY)

```
Barra 200: CHoCH Bullish 🟢
           → Fondo verde claro = "Esperando BOS"

Barra 210: Precio baja, NO rompe ningún máximo
           → No hay BOS

Barra 220: CHoCH Bearish 🔴
           → Invalida la espera anterior
           → NO se entró en long (correcto)
```

### Ejemplo 3: Múltiples BOS en Tendencia

```
Barra 50:  CHoCH Bearish 🔴
Barra 55:  BOS Bearish 🟠 → ENTRY SHORT ✅

Barra 65:  BOS Bearish 🟠 (continuación)
Barra 75:  BOS Bearish 🟠 (continuación)
Barra 85:  BOS Bearish 🟠 (continuación)
           → Trend fuerte, mantener short

Barra 95:  CHoCH Bullish 🟢 → EXIT ❌
```

---

## ⚠️ Reglas Importantes

### ✅ HACER

1. **Esperar siempre la confirmación BOS** después de CHoCH
2. **Salir inmediatamente** cuando aparece CHoCH opuesto
3. **Mantener posición** mientras aparecen BOS en tu dirección
4. **Usar timeframe 15m** (mejor performance)
5. **Probar primero en paper trading**

### ❌ NO HACER

1. **NO entrar solo con CHoCH** sin esperar BOS
2. **NO ignorar señales de salida** (CHoCH opuesto)
3. **NO cerrar posición** solo porque hay mucho profit si no hay CHoCH opuesto
4. **NO usar en timeframes muy pequeños** (<5m) - mucho ruido
5. **NO usar con apalancamiento alto** sin experiencia

---

## 🛠️ Optimización de Parámetros

### Para Timeframes Diferentes

**5 minutos (scalping):**
```
Pivot Lookback: 3
Internal Structure: 3 velas
Swing Structure: 20 velas
```

**15 minutos (recomendado):**
```
Pivot Lookback: 5
Internal Structure: 5 velas
Swing Structure: 50 velas
```

**1 hora (swing trading):**
```
Pivot Lookback: 7
Internal Structure: 7 velas
Swing Structure: 70 velas
```

### Para Mayor/Menor Agresividad

**Más conservador (menos trades, más confiables):**
```
Pivot Lookback: 7-10
Internal Structure: 7 velas
Swing Structure: 70 velas
```

**Más agresivo (más trades, menos confiables):**
```
Pivot Lookback: 3
Internal Structure: 3 velas
Swing Structure: 30 velas
```

---

## 🔔 Alertas Configurables

El script incluye 8 tipos de alertas:

1. **CHoCH Bullish** - Prepárate para posible long
2. **CHoCH Bearish** - Prepárate para posible short
3. **BOS Bullish** - Continuación alcista
4. **BOS Bearish** - Continuación bajista
5. **Long Entry** - ¡ENTRAR LONG! (CHoCH + BOS confirmado)
6. **Short Entry** - ¡ENTRAR SHORT! (CHoCH + BOS confirmado)
7. **Long Exit** - ¡SALIR DE LONG! (CHoCH bearish)
8. **Short Exit** - ¡SALIR DE SHORT! (CHoCH bullish)

### Cómo Configurar

1. Click en el ícono "⏰" en el indicador
2. Selecciona el tipo de alerta
3. Configura notificación (popup, email, webhook)
4. Click "Create"

---

## 📈 Diferencias con Implementación Anterior

### Versión Anterior (Incorrecta)

- ❌ Trataba todos los quiebres igual (no diferenciaba CHoCH vs BOS)
- ❌ Entraba inmediatamente al detectar cambio de estructura
- ❌ Usaba máquina de estados complicada
- ❌ No esperaba confirmación

### Versión Actual (Correcta)

- ✅ Diferencia CHoCH (grande) vs BOS (pequeño) por número de velas
- ✅ Implementa LA REGLA DE ORO: CHoCH → esperar BOS → entrar
- ✅ Lógica clara basada en confirmación
- ✅ Señales más limpias y precisas

---

## 🎯 Mejores Prácticas

### 1. Gestión de Riesgo

- Nunca arriesgar más del 1-2% del capital por trade
- Con 12 USDT y 5x leverage, -5.8% SL = pérdida de ~3.48 USDT
- Ajustar position size según capital y riesgo

### 2. Selección de Pares

**Mejores resultados (backtesting):**
- SOL/USDT
- ADA/USDT
- LINK/USDT
- AAVE/USDT
- AVAX/USDT

**Evitar:**
- BTC/USDT (menor performance con esta estrategia)

### 3. Confirmaciones Adicionales

Para mejorar win rate, combinar con:
- Fair Value Gaps (FVG)
- Order Blocks
- Zonas de Premium/Discount
- Niveles de liquidez

### 4. Backtesting

Antes de usar en vivo:
1. Activar Strategy Tester en TradingView
2. Probar en datos históricos (mínimo 6 meses)
3. Analizar win rate, drawdown, profit factor
4. Ajustar parámetros si es necesario

---

## ⚠️ Advertencias Importantes

### 1. Repainting

Los pivotes se confirman **5 velas después** (lookback), lo que significa:
- Las señales aparecen con retraso
- Los pivotes pueden "desaparecer" si se invalidan
- **Solución:** Esperar cierre de vela antes de actuar

### 2. Leverage

Con 5x leverage:
- -5.8% SL = **-29% pérdida real**
- +20% profit = **+100% ganancia real**
- **Usa leverage con precaución**

### 3. Market Conditions

La estrategia funciona mejor en:
- ✅ Mercados con tendencia clara
- ✅ Timeframes medianos (15m-1h)
- ❌ Mercados laterales/rangos
- ❌ Alta volatilidad extrema

---

## 📝 Version History

**v2.0 (2025-10-31) - REESCRITURA COMPLETA**
- ✅ Implementación correcta de las reglas CHoCH/BOS
- ✅ Diferenciación entre estructuras internas (BOS) y swing (CHoCH)
- ✅ Lógica de confirmación: CHoCH → BOS → Entry
- ✅ Detección de pivotes con ZigZag estándar (5+5+1)
- ✅ Estados de espera de confirmación visualizados
- ✅ Clasificación automática de rupturas por tamaño de estructura
- ✅ Salida inmediata por CHoCH opuesto

**v1.2 (2025-10-31)**
- Fixed BOS signal noise
- BOS solo en nuevos niveles

**v1.1 (2025-10-31)**
- Updated to Pine Script v6
- Fixed syntax errors

**v1.0 (2025-10-31)**
- Initial conversion (implementación incorrecta)

---

## 💡 Tips Finales

1. **Paciencia es clave** - Espera siempre la confirmación BOS
2. **No te adelantes** - CHoCH solo prepara, BOS confirma
3. **Respeta las salidas** - CHoCH opuesto = salir sin dudas
4. **Practica primero** - Paper trading mínimo 2 semanas
5. **Ajusta a tu estilo** - Los parámetros son solo punto de partida
6. **Combina con otros conceptos** - FVG, Order Blocks, etc.
7. **Monitorea el performance** - Lleva registro de todos tus trades

---

**Happy Trading! 📊💹**

*Recuerda: Esta estrategia es una herramienta. El éxito depende de tu disciplina, gestión de riesgo y práctica.*

Para más información sobre la estrategia en Freqtrade, consulta: `../README.md`
