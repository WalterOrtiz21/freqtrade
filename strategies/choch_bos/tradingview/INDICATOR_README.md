# CHoCH/BOS Indicadores para TradingView

Indicadores basados en la estrategia CHoCH/BOS (Change of Character + Break of Structure) que muestran señales claras de entrada y salida.

---

## 📁 Archivos Disponibles

### `CHoCH_BOS_Indicator_Advanced.pine` - Versión Estándar
**Indicador profesional con características básicas**

- ✅ **Flechas verdes/rojas** para señales de entrada
- ✅ **"X" rojas/naranjas** para señales de salida y SL
- ✅ **Líneas horizontales** para niveles de Stop Loss
- ✅ **Detección completa de CHoCH/BOS** con ZigZag
- ✅ **FVG (Fair Value Gap) detection**
- ✅ **Filtro opcional de FVG retest** para mayor precisión
- ✅ **Zonas de FVG visualizadas** con cajas coloreadas
- ✅ **Tabla de información en tiempo real** extendida
- ✅ **Alertas configurables** para todas las señales

### `CHoCH_BOS_Indicator_Complete.pine` - Versión Premium ⭐
**Indicador completo con sistema dual de Take Profit (matching Python)**

- ✅ **Todas las características de la versión Advanced**
- 🎯 **Sistema Dual de Take Profit**: ROI por tiempo + BOS Counting
- 💎 **Señales de TP1/TP2** con diamantes coloreados
- 📊 **Contador de BOS** visible en tabla
- 🟦 **Señales de ROI** con cuadrados azules
- ⚙️ **Parámetros completos de TP** configurables
- 🔄 **Move to BE** después de TP1 (opcional)
- 📈 **12 tipos de alertas** específicas para cada evento

---

## 🎯 Sistema Dual de Take Profit (Versión Complete)

### 📊 **ROI por Tiempo (matching Python):**
```
ROI 0min: 10%    (inmediato)
ROI 60min: 5%     (después de 1 hora)
ROI 120min: 3%    (después de 2 horas)
ROI 240min: 1%    (después de 4 horas)
```

### 🔢 **BOS Counting System (matching Python):**
```
TP1: Primer BOS después de entrada → Cerrar tp1_percentage (default 50%)
TP2: Segundo BOS después de entrada → Cerrar tp2_percentage (default 30%)
Resto: Posición restante corre hasta CHoCH opuesto o ROI
Move to BE: SL se mueve a breakeven después de TP1 (opcional)
```

### 🎯 **Sistemas de Salida Combinados:**
1. **ROI System**: Cierra por objetivos de tiempo
2. **BOS Counting**: Cierra parcialmente en hitos de BOS
3. **CHoCH Opposite**: Siempre cierra posición completa
4. **Stop Loss**: Protección máxima contra pérdidas

---

## 🚀 Cómo Usar los Indicadores

### Instalación en TradingView

1. **Abre TradingView**: https://www.tradingview.com/
2. **Ve al Pine Editor**: Click en "Pine Editor" en la parte inferior
3. **Copia el código**: Selecciona y copia todo el contenido del indicador deseado
4. **Pega en el editor**: Reemplaza cualquier código existente
5. **Añade al gráfico**: Click en "Add to Chart"
6. **Configura parámetros**: Click en el ícono de engranaje ⚙️ del indicador

### Configuración Recomendada

#### Para TradingView Web (Versión Standard)
```
ZigZag Length: 23
Stop Loss %: 13.0
Require FVG Retest: ON (recomendado para mayor precisión)
FVG Lookback Period: 20
```

#### Para TradingView Web (Versión Complete - Sistema Dual TP)
```
ZigZag Length: 23
Stop Loss %: 13.0
Require FVG Retest: ON

// Take Profit System
Enable ROI System: true
ROI 0min: 10.0%
ROI 60min: 5.0%
ROI 120min: 3.0%
ROI 240min: 1.0%

Enable BOS Counting: true
TP1 at 1st BOS: 50.0%
TP2 at 2nd BOS: 30.0%
Move to BE after TP1: true
```

#### Para Timeframes Específicos

**5 minutos (Scalping):**
```
ZigZag Length: 15
Stop Loss %: 8.0
Require FVG Retest: OFF (más señales)
```

**15 minutos (Recomendado):**
```
ZigZag Length: 23
Stop Loss %: 13.0
Require FVG Retest: ON (mejor precisión)
```

**1 hora (Swing Trading):**
```
ZigZag Length: 34
Stop Loss %: 15.0
Require FVG Retest: ON (confirmación adicional)
```

---

## 🎨 Elementos Visuales

### Señales de Entrada (Ambas Versiones)

| Símbolo | Dirección | Significado | Acción |
|---------|-----------|-------------|--------|
| 🟢 **FLECHA ARRIBA** | LONG | CHoCH + BOS alcista | **COMPRAR** |
| 🔴 **FLECHA ABAJO** | SHORT | CHoCH + BOS bajista | **VENDER** |

### Señales de Salida (Versión Standard)

| Símbolo | Dirección | Significado | Acción |
|---------|-----------|-------------|--------|
| ❌ **X ROJA** | LONG | CHoCH bajista detectado | **CERRAR LONG** |
| ❌ **X ROJA** | SHORT | CHoCH alcista detectado | **CERRAR SHORT** |

### Señales de Take Profit (Versión Complete)

| Símbolo | Dirección | Significado | Acción |
|---------|-----------|-------------|--------|
| 💎 **DIAMANTE VERDE** | LONG | TP1 - Primer BOS | **Cerrar 50%** |
| 💎 **DIAMANTE VERDE CLARO** | LONG | TP2 - Segundo BOS | **Cerrar 30%** |
| 💎 **DIAMANTE ROJO OSCURO** | SHORT | TP1 - Primer BOS | **Cerrar 50%** |
| 💎 **DIAMANTE ROJO CLARO** | SHORT | TP2 - Segundo BOS | **Cerrar 30%** |
| 🟦 **CUADRADO AZUL** | AMBOS | Salida ROI | **Cerrar por tiempo** |

### Stop Loss (Ambas Versiones)

| Símbolo | Significado | Acción |
|---------|-------------|--------|
| 📍 **LÍNEA ROJA HORIZONTAL** | Nivel de Stop Loss actual | **Monitorear** |
| 🟠 **X NARANJA** | Stop Loss alcanzado | **Salida automática** |

### Estructura de Mercado (Ambas Versiones)

| Símbolo | Significado | Importancia |
|---------|-------------|-------------|
| 🟢 **CHoCH** | Cambio de carácter alcista | Preparación para posible long |
| 🔴 **CHoCH** | Cambio de carácter bajista | Preparación para posible short |
| 🟢 **BOS** | Ruptura de estructura alcista | Continuación o confirmación |
| 🔴 **BOS** | Ruptura de estructura bajista | Continuación o confirmación |

---

## 📊 Lógica de Trading

### Regla de Oro

**CHoCH + BOS = ENTRADA**

1. **CHoCH** (Change of Character) = Señal de preparación
2. **BOS** (Break of Structure) = Señal de confirmación
3. **SOLO ENTRAR** cuando ambos ocurren en la misma dirección

### Flujo Completo (Versión Standard)

#### Para LONG:
```
1. Aparece CHoCH alcista 🟢
   → Prepararse para posible entrada
   
2. Aparece BOS alcista 🟢 (dentro de 20 velas)
   → Confirmación recibida
   
3. Aparece FLECHA VERDE ⬆️
   → ENTRAR EN LONG
   
4. Mantener posición hasta:
   - Aparece CHoCH bajista 🔴 → Salir con ❌
   - Se alcanza Stop Loss → Salir automático
```

#### Para SHORT:
```
1. Aparece CHoCH bajista 🔴
   → Prepararse para posible entrada
   
2. Aparece BOS bajista 🔴 (dentro de 20 velas)
   → Confirmación recibida
   
3. Aparece FLECHA ROJA ⬇️
   → ENTRAR EN SHORT
   
4. Mantener posición hasta:
   - Aparece CHoCH alcista 🟢 → Salir con ❌
   - Se alcanza Stop Loss → Salir automático
```

### Flujo Completo (Versión Complete - Sistema Dual TP)

#### Para LONG:
```
1. CHoCH Bullish 🟢 → "Prepararse para posible long"
2. BOS Bullish 💚 → "Confirmación recibida"
3. Entrada LONG 🟢 → "CHoCH+BOS confirmado"
4. TP1: Primer BOS después de entrada 💎 → "Cerrar 50%"
5. TP2: Segundo BOS después de entrada 💎 → "Cerrar 30%"
6. Resto: Corre hasta CHoCH opuesto ❌ o ROI 🟦
```

#### Para SHORT:
```
1. CHoCH Bearish 🔴 → "Prepararse para posible short"
2. BOS Bearish 🟠 → "Confirmación recibida"
3. Entrada SHORT 🔴 → "CHoCH+BOS confirmado"
4. TP1: Primer BOS después de entrada 💎 → "Cerrar 50%"
5. TP2: Segundo BOS después de entrada 💎 → "Cerrar 30%"
6. Resto: Corre hasta CHoCH opuesto ❌ o ROI 🟦
```

---

## ⚙️ Parámetros Detallados

### Structure Detection (Ambas Versiones)
- **ZigZag Length**: Período para detección de pivotes (5-50)
  - Más alto = Menos señales, más confiables
  - Más bajo = Más señales, más ruido

### Entry Filters (Ambas Versiones)
- **Require FVG Retest**: Exigir retest de FVG para entrada
  - ON: Menos trades, mayor win rate
  - OFF: Más trades, menor win rate
- **FVG Lookback Period**: Velas hacia atrás para buscar FVG (10-50)

### Take Profit System (Solo Versión Complete)
- **Enable ROI System**: Activar ROI por tiempo (default true)
- **ROI 0min/60min/120min/240min**: Porcentajes configurables
- **Enable BOS Counting TP**: Activar TP por BOS (default true)
- **TP1 at 1st BOS (%)**: Porcentaje a cerrar en TP1 (default 50%)
- **TP2 at 2nd BOS (%)**: Porcentaje a cerrar en TP2 (default 30%)
- **Move to BE after TP1**: Mover SL a breakeven después de TP1 (default true)

### Risk Management (Ambas Versiones)
- **Stop Loss %**: Porcentaje de stop loss (5-25%)
  - Ajustar según tu tolerancia al riesgo

### Visual Settings (Ambas Versiones)
- **Show Entry Signals**: Mostrar flechas de entrada
- **Show Exit Signals**: Mostrar "X" de salida
- **Show TP1/TP2 Signals**: Mostrar diamantes de TP (solo Complete)
- **Show Price Labels**: Etiquetas de precio fuera del gráfico
- **Show Connection Lines**: Líneas punteadas de conexión
- **Show Stop Loss Levels**: Mostrar líneas de SL
- **Show Structure**: Mostrar CHoCH/BOS markers
- **Show FVG Zones**: Mostrar zonas de FVG
- **Show Pivots**: Mostrar triángulos de pivotes

---

## 🔔 Configuración de Alertas

### Alertas Básicas (Ambas Versiones)
1. **Long Entry**: CHoCH + BOS alcista → Entrar long
2. **Short Entry**: CHoCH + BOS bajista → Entrar short
3. **Long Exit**: Salida de posición long
4. **Short Exit**: Salida de posición short

### Alertas de Take Profit (Solo Versión Complete)
5. **Long TP1 (1st BOS)**: Primer BOS en LONG
6. **Long TP2 (2nd BOS)**: Segundo BOS en LONG
7. **Short TP1 (1st BOS)**: Primer BOS en SHORT
8. **Short TP2 (2nd BOS)**: Segundo BOS en SHORT
9. **Long ROI Exit**: Salida ROI en LONG
10. **Short ROI Exit**: Salida ROI en SHORT

### Alertas Adicionales (Ambas Versiones)
11. **Bullish FVG**: Nueva zona FVG alcista detectada
12. **Bearish FVG**: Nueva zona FVG bajista detectada

### Cómo Configurar Alertas
1. Click en el ícono de campana 🔔 en el indicador
2. Selecciona el tipo de alerta
3. Configura notificación (popup, email, webhook)
4. Click "Create"

---

## 📈 Mejores Prácticas

### ✅ Qué Hacer

1. **Esperar siempre la confirmación BOS** después de CHoCH
2. **Respetar los niveles de Stop Loss** configurados
3. **Usar el timeframe recomendado** (15m para mejores resultados)
4. **Combinar con análisis adicional** (volumen, niveles de soporte/resistencia)
5. **Probar en paper trading** antes de usar capital real

### ❌ Qué Evitar

1. **NO entrar solo con CHoCH** sin esperar BOS
2. **NO ignorar señales de salida** (CHoCH opuesto)
3. **NO mover el Stop Loss** emocionalmente
4. **NO usar en timeframes muy pequeños** (<5m) - mucho ruido
5. **NO arriesgar más del 1-2%** del capital por trade

---

## 🎯 Optimización por Timeframe

### Scalping (1-5m)
```
ZigZag Length: 10-15
Stop Loss: 8-10%
FVG Filter: OFF
```

### Day Trading (15m-1h)
```
ZigZag Length: 20-30
Stop Loss: 12-15%
FVG Filter: ON (opcional)
```

### Swing Trading (4h-1d)
```
ZigZag Length: 30-50
Stop Loss: 15-20%
FVG Filter: ON (recomendado)
```

---

## 📊 Tabla de Información

Los indicadores incluyen una tabla en la esquina superior derecha que muestra:

### Versión Standard:
- **Trend**: Tendencia actual (BULLISH/BEARISH/NEUTRAL)
- **Position**: Posición actual (LONG/SHORT/FLAT)
- **FVG Filter**: Estado del filtro FVG
- **Stop Loss**: Porcentaje configurado
- **Last High/Low**: Últimos pivotes significativos

### Versión Complete:
- **Trend**: Tendencia actual (BULLISH/BEARISH/NEUTRAL)
- **Position**: Posición actual (LONG/SHORT/FLAT)
- **BOS Count**: Contador de BOS después de entrada (0, 1, 2...)
- **ROI Target**: Objetivo ROI actual según tiempo en posición
- **FVG Filter**: Estado del filtro FVG (WAITING/RETESTED/DISABLED)
- **Stop Loss**: Porcentaje configurado
- **Last High/Low**: Últimos pivotes significativos

---

## 🚀 Diferencias vs Estrategia

| Característica | Estrategia | Indicador Standard | Indicador Complete |
|---------------|-------------|-------------------|-------------------|
| **Ejecución** | Automática | Manual | Manual |
| **Gestión** | Completa | Visual | Visual Avanzada |
| **Flexibilidad** | Fija | Total | Total Premium |
| **Take Profit** | ROI + BOS Counting | Básico | **ROI + BOS Counting** |
| **Backtesting** | Integrado | Manual | Manual |
| **Alertas** | Básicas | Completas | **Premium (12 tipos)** |

### Ventajas del Indicador Complete:
- ✅ **Misma lógica Python** exacta
- ✅ **Sistema dual de TP** completo
- ✅ **Control total** sobre decisiones
- ✅ **Flexibilidad** para combinar con otros análisis
- ✅ **Alertas premium** personalizadas
- ✅ **Visualización clara** de todos los eventos
- ✅ **Contador de BOS** en tiempo real
- ✅ **ROI dinámico** según tiempo

---

## 📞 Soporte y Preguntas

Para dudas o sugerencias:

1. **Revisa este README** - Contiene la mayoría de las respuestas
2. **Prueba en paper trading** - Familiarízate antes de usar capital real
3. **Ajusta parámetros** - Cada par y timeframe puede necesitar ajustes
4. **Combina con otros indicadores** - Mejora la precisión

---

## 🎯 ¿Qué Versión Elegir?

### **CHoCH_BOS_Indicator_Advanced.pine** - Para:
- Traders principiantes
- Quienes prefieren simplicidad
- Trading manual básico
- Menos parámetros que configurar

### **CHoCH_BOS_Indicator_Complete.pine** - Para:
- Traders avanzados ⭐
- Quienes quieren **replicar exactamente** la estrategia Python
- Trading con gestión sofisticada de TP
- Máximo control y flexibilidad
- Sistema dual de Take Profit profesional

---

**¡Happy Trading! 📊💹**

*Recuerda: Estos indicadores son herramientas de análisis. El éxito depende de tu disciplina, gestión de riesgo y práctica constante.*