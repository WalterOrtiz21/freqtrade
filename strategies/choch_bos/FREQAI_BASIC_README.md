# CHoCH/BOS FreqAI - Versión Básica

## 📋 Descripción

Estrategia FreqAI **simple pero efectiva** que combina Smart Money Concepts con Machine Learning para predecir trades ganadores.

### Diferencias con FreqAISmartMoneyStrategy

| Aspecto | FreqAISmartMoneyStrategy | CHoCHBOSFreqAIBasic |
|---------|-------------------------|---------------------|
| **Features** | ~100+ (con expansión) | ~25-30 (esencial) |
| **Complejidad** | Alta | Baja |
| **Target** | Multiclasse (bullish/neutral/bearish) | Binario (ganador/perdedor) |
| **Objetivo** | Predecir dirección del mercado | Predecir éxito del trade |
| **Uso** | Producción avanzada | Aprendizaje y testing |

---

## 🎯 Objetivo de la IA

**Target Específico:** ¿Este trade será ganador?

```python
# Simula un trade real con:
# - Take Profit: +5%
# - Stop Loss: -2%
# - Periodo: 15 velas

# Resultado:
# 1 = Trade alcanza +5% antes que -2% ✅
# 0 = Trade alcanza -2% primero o no alcanza ninguno ❌
```

**¿Por qué este objetivo?**
- Es exactamente lo que queremos saber antes de entrar
- Simula nuestra estrategia real de TP/SL
- Más fácil de interpretar que "dirección del mercado"

---

## 🔧 Features (~25-30 totales)

### 1. Indicadores Expandidos (con múltiples periodos: 10, 20, 30)

- RSI (Momentum)
- MFI (Money Flow)
- ADX (Trend Strength)
- ATR (Volatility)
- EMA (Moving Average)
- ROC (Rate of Change)
- Volumen Relativo

**Total con expansión:** ~21 features (7 × 3 periodos)

### 2. Price Action Básico

- Cambio de precio (%)
- Tamaño del cuerpo de vela
- Upper wick (sombra superior)
- Lower wick (sombra inferior)
- High-Low range
- Vela alcista/bajista

**Total:** 6 features

### 3. Smart Money Concepts (Simplificados)

- CHoCH Bullish (reversión alcista)
- CHoCH Bearish (reversión bajista)
- BOS Bullish (continuación alcista)
- BOS Bearish (continuación bajista)

**Total:** 4 features

### 4. Patrones de Precio

- Bullish Engulfing (envolvente alcista)
- Bearish Engulfing (envolvente bajista)

**Total:** 2 features

### 5. Market Structure

- Higher High (máximos más altos)
- Lower Low (mínimos más bajos)

**Total:** 2 features

### 6. Indicadores Tradicionales

- MACD
- MACD Signal
- MACD Histogram
- EMA 20/50 Crossover (alcista/bajista)
- Distancia a EMA 20
- Distancia a EMA 50

**Total:** 6 features

---

## 📊 Arquitectura del Modelo

**Modelo Recomendado:** LightGBMClassifier

**¿Por qué LightGBM?**
- ✅ Rápido en entrenamiento
- ✅ Maneja bien features categóricos
- ✅ Menos propenso a overfitting que XGBoost
- ✅ Bueno con datasets pequeños/medianos

**Configuración:**
```json
"model_training_parameters": {
    "n_estimators": 1000,      // Número de árboles
    "learning_rate": 0.02,     // Tasa de aprendizaje (conservadora)
    "max_depth": 6,            // Profundidad máxima (previene overfitting)
    "subsample": 0.8,          // 80% de datos por árbol
    "colsample_bytree": 0.8,   // 80% de features por árbol
    "random_state": 42         // Reproducibilidad
}
```

---

## 🚀 Uso

### 1. Descargar Datos

```bash
freqtrade download-data \
    --exchange bybit \
    --pairs BTC/USDT:USDT ETH/USDT:USDT \
    --timeframes 5m 15m 1h \
    --timerange 20240101-
```

### 2. Backtesting

```bash
# Backtesting con entrenamiento de modelo
freqtrade backtesting \
    --strategy CHoCHBOSFreqAIBasic \
    --config config_freqai_basic.json \
    --timerange 20240101-20241101 \
    --freqaimodel LightGBMClassifier
```

**Importante:** El primer backtesting tardará más porque:
1. Entrena el modelo con los primeros 30 días
2. Backtest en los siguientes 7 días
3. Repite este proceso hasta cubrir todo el rango

### 3. Ver Feature Importance

Después del backtesting, revisa qué features son más importantes:

```bash
# Los resultados se guardan en:
# user_data/models/CHoCHBOSBasic/
```

Busca archivos como `feature_importance_*.png` para ver gráficamente qué features usa más el modelo.

### 4. Dry Run (Paper Trading)

```bash
freqtrade trade \
    --strategy CHoCHBOSFreqAIBasic \
    --config config_freqai_basic.json \
    --freqaimodel LightGBMClassifier
```

---

## ⚙️ Configuración

### FreqAI Settings (config_freqai_basic.json)

```json
"freqai": {
    "train_period_days": 30,        // Datos para entrenar
    "backtest_period_days": 7,      // Datos para predecir
    "live_retrain_hours": 0,        // No reentrenar en live (0 = solo al inicio)

    "feature_parameters": {
        "label_period_candles": 15,  // Velas hacia adelante para el target

        "include_timeframes": [      // Timeframes adicionales
            "5m",
            "15m",
            "1h"
        ],

        "indicator_periods_candles": [  // Periodos para features expandidos
            10,
            20,
            30
        ],

        "include_shifted_candles": 2,  // Features de velas anteriores (t-1, t-2)

        "DI_threshold": 0.9,           // Threshold de disimilitud (0.9 = conservador)
        "use_SVM_to_remove_outliers": true  // Remover outliers
    }
}
```

**Parámetros Clave:**

- **label_period_candles: 15**
  - Simula trades de 15 velas (15m × 15 = 3.75 horas)
  - Ajusta según tu horizonte de trading
  - Menor = trades más cortos, Mayor = trades más largos

- **train_period_days: 30**
  - Entrena con 30 días de datos
  - Más días = mejor contexto pero modelo más lento
  - Menos días = modelo más rápido pero menos contexto

- **DI_threshold: 0.9**
  - Filtro de confianza: solo predice si los datos son similares al entrenamiento
  - 0.9 = muy conservador (pocas predicciones pero más confiables)
  - 0.5 = liberal (más predicciones pero menos confiables)

---

## 📈 Lógica de Entry/Exit

### Entry LONG

```python
Condiciones:
1. do_predict == 1              # Modelo tiene confianza
2. &-target_profitable > 0.6    # >60% probabilidad de éxito
3. RSI entre 25-75              # No extremos
4. Volumen > 0                  # Hay actividad
```

### Entry SHORT

```python
Condiciones:
1. do_predict == 1              # Modelo tiene confianza
2. &-target_profitable < 0.4    # <40% probabilidad (bajista)
3. RSI entre 25-75              # No extremos
4. Volumen > 0                  # Hay actividad
```

### Exit LONG

```python
Condiciones:
1. do_predict == 1
2. &-target_profitable < 0.3    # Predicción cambió a bajista
```

### Exit SHORT

```python
Condiciones:
1. do_predict == 1
2. &-target_profitable > 0.7    # Predicción cambió a alcista
```

---

## 🔍 Análisis del Modelo

### Ver Feature Importance

Después del backtesting, analiza qué features son más importantes:

```python
# Los archivos se guardan en:
user_data/models/CHoCHBOSBasic/sub-train-*_TIMESTAMP/

# Archivos importantes:
- feature_importance.png           # Gráfico de importancia
- model_metrics.json               # Métricas del modelo
- do_predict.pkl                   # Modelo entrenado
```

### Métricas a Revisar

```json
{
    "accuracy": 0.65,        // Precisión general (>60% es bueno)
    "precision": 0.70,       // De los que predice 1, cuántos son correctos
    "recall": 0.60,          // De los verdaderos 1, cuántos detecta
    "f1_score": 0.65,        // Balance entre precision y recall
    "roc_auc": 0.72          // Capacidad de discriminación (>0.7 es bueno)
}
```

**¿Qué buscar?**
- **Accuracy > 0.60:** Modelo mejor que azar (50%)
- **Precision alta:** Menos falsos positivos (menos trades malos)
- **Recall alto:** Captura más oportunidades
- **ROC AUC > 0.70:** Buena separación entre clases

---

## 🎓 Mejoras Incrementales

### Paso 1: Baseline (Actual)
- ~25-30 features
- Target binario simple
- LightGBM básico

### Paso 2: Optimizar Threshold
```python
# En populate_entry_trend, experimenta con:
dataframe["&-target_profitable"] > 0.7  # Más conservador (menos trades)
dataframe["&-target_profitable"] > 0.5  # Más agresivo (más trades)
```

### Paso 3: Agregar Features Selectivos
- Solo agrega features que el modelo use (revisa feature importance)
- Evita correlación (no agregues RSI 14 si ya tienes RSI 10, 20, 30)

### Paso 4: Ajustar Target
```python
# Diferentes objetivos:
profit_target = 0.03  # 3% (más alcanzable)
profit_target = 0.10  # 10% (más ambicioso)

label_period = 10     # Trades más cortos
label_period = 30     # Trades más largos
```

### Paso 5: Probar Otros Modelos
```bash
# XGBoost (más lento pero a veces mejor)
--freqaimodel XGBoostClassifier

# CatBoost (bueno con categorías)
--freqaimodel CatboostClassifier

# PyTorch (neural network, requiere más datos)
--freqaimodel PyTorchMLPClassifier
```

---

## ⚠️ Advertencias y Consejos

### Do's ✅

1. **Empieza con dry run:** Prueba en papel antes de dinero real
2. **Revisa feature importance:** Elimina features inútiles
3. **Valida en datos nuevos:** Backtest en periodo diferente al entrenamiento
4. **Monitorea `do_predict`:** Si siempre es 0, ajusta DI_threshold
5. **Revisa balanceo de clases:** Si target es 90% ceros, ajusta TP/SL

### Don'ts ❌

1. **No agregues 400 features:** Más features ≠ mejor modelo
2. **No optimices sobre el mismo periodo:** Causa overfitting
3. **No ignores `do_predict == 0`:** Significa datos desconocidos
4. **No uses modelos sin entender:** Lee la documentación
5. **No confíes ciegamente en el backtest:** Forward testing es clave

---

## 📚 Recursos

### FreqAI Docs
- [FreqAI Overview](https://www.freqtrade.io/en/stable/freqai/)
- [Feature Engineering](https://www.freqtrade.io/en/stable/freqai-feature-engineering/)
- [Parameter Table](https://www.freqtrade.io/en/stable/freqai-parameter-table/)

### Machine Learning Concepts
- [Precision vs Recall](https://en.wikipedia.org/wiki/Precision_and_recall)
- [ROC AUC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)
- [Overfitting](https://en.wikipedia.org/wiki/Overfitting)

---

## 🐛 Troubleshooting

### "No predictions being made" (do_predict siempre 0)

**Causa:** DI_threshold muy alto o datos muy diferentes

**Solución:**
```json
"DI_threshold": 0.5,  // Reducir de 0.9 a 0.5
```

### "Model accuracy very low" (<50%)

**Causa:** Target muy difícil de predecir o features irrelevantes

**Solución:**
1. Ajusta TP/SL a valores más realistas
2. Aumenta `label_period_candles`
3. Revisa feature importance y elimina features inútiles

### "Backtest muy lento"

**Causa:** Demasiados timeframes o periodos

**Solución:**
```json
"include_timeframes": ["15m"],  // Solo timeframe base
"indicator_periods_candles": [20]  // Solo un periodo
```

### "Out of memory"

**Causa:** Demasiados datos o features

**Solución:**
1. Reduce `train_period_days`
2. Reduce cantidad de features
3. Reduce `include_shifted_candles`

---

## 📊 Ejemplo de Resultados Esperados

### Backtesting (30 días train, 7 días test)

```
Resultados esperados (optimistas pero realistas):

Win Rate: 55-60%
Profit Factor: 1.2-1.5
Total Trades: 20-40 (en 7 días, 2 pares)
Average Profit: 2-3%
Max Drawdown: 5-8%
```

**Nota:** Estos son ejemplos. Tus resultados dependerán de:
- Calidad de datos
- Periodo seleccionado
- Configuración de TP/SL
- Condiciones del mercado

---

## 🎯 Próximos Pasos

1. **Backtest inicial:**
   ```bash
   freqtrade backtesting --strategy CHoCHBOSFreqAIBasic \
       --config config_freqai_basic.json \
       --timerange 20240601-20240701 \
       --freqaimodel LightGBMClassifier
   ```

2. **Analiza resultados:**
   - Revisa métricas del modelo
   - Mira feature importance
   - Valida en periodo diferente

3. **Ajusta si es necesario:**
   - Cambia TP/SL
   - Ajusta threshold de entry
   - Modifica DI_threshold

4. **Dry run:**
   ```bash
   freqtrade trade --strategy CHoCHBOSFreqAIBasic \
       --config config_freqai_basic.json
   ```

5. **Live trading (cuando estés listo):**
   - Empieza con capital mínimo
   - Monitorea constantemente
   - Ajusta basado en resultados reales

---

**¿Preguntas? Revisa la documentación de FreqAI o consulta el CLAUDE.md principal.**
