# Hyperopt 5000 Epochs - Guía Completa

Esta guía explica cómo usar el sistema de hyperopt con **5000 epochs**, **resume automático** y **tracking de parámetros probados**.

## 🎯 Archivos Incluidos

```
freqtrade/
├── hyperopt_full_5000.py           # Script principal (5000 epochs con resume)
├── check_hyperopt_completion.py    # Verificador de progreso
├── hyperopt_progress_5000.json     # Progreso guardado (auto-generado)
├── hyperopt_results_5000.json      # Resultados finales (auto-generado)
└── hyperopt_5000.log               # Log completo (auto-generado)
```

## 🚀 Inicio Rápido

### 1. Primera ejecución (0 → 5000 epochs)

```bash
# Activar entorno virtual
source .venv/bin/activate  # Linux/Mac
# o
.venv\Scripts\activate     # Windows

# Ejecutar hyperopt
python hyperopt_full_5000.py
```

**Tiempo estimado:**
- VPS potente (4+ cores): 1-2 horas
- VPS básico (2 cores): 3-5 horas
- PC local: 2-4 horas

### 2. Si se interrumpe (presionas Ctrl+C)

```bash
# El script guarda automáticamente el progreso
# Para continuar desde donde quedó:
python hyperopt_full_5000.py
```

**El script detecta automáticamente:**
- ✅ Cuántos epochs ya completaste
- ✅ Qué parámetros ya probaste
- ✅ Continúa solo con los epochs restantes

### 3. Verificar progreso en cualquier momento

```bash
# Ver estado actual y análisis de cobertura
python check_hyperopt_completion.py
```

**Output:**
```
📊 PROGRESO GENERAL
Epochs completados: 1,234 / 5,000
Progreso: 24.7%
Epochs restantes: 3,766

📝 HISTORIAL DE EJECUCIONES
Total de ejecuciones: 3

✅ Run 1: 500 epochs
   Start: 2025-10-06T10:15:30
   End:   2025-10-06T11:30:45

❌ Run 2: 0 epochs
   Error: Interrupted by user

✅ Run 3: 734 epochs
   Start: 2025-10-06T12:00:00
   End:   2025-10-06T13:15:20
```

## 📊 ¿Cómo funciona el Resume?

### Sistema de Tracking Automático

**Archivo: `hyperopt_progress_5000.json`**

```json
{
  "epochs_completed": 1234,
  "last_run": "2025-10-06T13:15:20",
  "runs": [
    {
      "start": "2025-10-06T10:15:30",
      "end": "2025-10-06T11:30:45",
      "epochs_completed": 500,
      "success": true
    },
    {
      "start": "2025-10-06T12:00:00",
      "end": "2025-10-06T13:15:20",
      "epochs_completed": 734,
      "success": true
    }
  ]
}
```

### Lógica de Resume

1. **Inicio**: Script lee `hyperopt_progress_5000.json`
2. **Cálculo**: `epochs_restantes = 5000 - epochs_completed`
3. **Ejecución**: Corre solo los epochs restantes
4. **Guardado**: Actualiza progreso al finalizar/interrumpir

### Escenarios

**Escenario 1: Primera ejecución**
```bash
$ python hyperopt_full_5000.py
> Progreso cargado: 0/5000 epochs completados
> Epochs restantes: 5000
> Iniciando optimización de 5000 epochs...
```

**Escenario 2: Continuación después de interrupción**
```bash
$ python hyperopt_full_5000.py
> Progreso cargado: 1234/5000 epochs completados
> Epochs restantes: 3766
> Iniciando optimización de 3766 epochs...
```

**Escenario 3: Ya completado**
```bash
$ python hyperopt_full_5000.py
> Progreso cargado: 5000/5000 epochs completados
> ✅ COMPLETADO: Todas las 5000 iteraciones han sido ejecutadas
```

## 🔍 ¿Se prueban TODAS las combinaciones?

### TL;DR: NO (y no es necesario)

**Espacio total de parámetros: ~2.4 TRILLION combinaciones**

```python
Entry parameters:
  - RSI threshold: 36 valores
  - Momentum threshold: 96 valores
  - RSI period: 11 valores
  - Momentum period: 8 valores
  - MA period: 13 valores

Exit parameters:
  - Emergency SL: 11 valores
  - Stop Loss: 15 valores
  - Take Profit: 19 valores
  - Trailing activation: 20 valores
  - Trailing low/mid/high: 10 * 9 * 16 valores

Total = 36 * 96 * 11 * 8 * 13 * 11 * 15 * 19 * 20 * 10 * 9 * 16
      = ~2.4 trillion
```

### ¿Cómo funciona entonces?

Freqtrade usa **optimización Bayesiana**, NO fuerza bruta:

1. **Inicio**: Prueba combinaciones aleatorias
2. **Aprendizaje**: Identifica qué parámetros funcionan mejor
3. **Optimización**: Prueba variaciones de los mejores parámetros
4. **Convergencia**: Encuentra óptimo local en 1000-2000 epochs típicamente

**Analogía**: Como un científico que no mezcla todos los químicos posibles, sino que prueba variaciones inteligentes basadas en resultados previos.

### ¿Cómo saber si ya encontró el óptimo?

El script `check_hyperopt_completion.py` detecta convergencia:

```python
# Si últimas 3 ejecuciones NO mejoraron:
⚠️  ADVERTENCIA: Últimas 3 ejecuciones sin mejora
    Posiblemente se han probado todas las combinaciones útiles
```

**Interpretación:**
- ✅ **Convergió**: Hyperopt encontró el óptimo local
- ⚠️ **Continuar**: Puedes continuar buscando, pero mejoras serán mínimas
- ❌ **Overfitting**: Si sigues optimizando, riesgo de overfitting aumenta

## 📈 Monitoreo de Progreso

### Opción 1: Check manual

```bash
python check_hyperopt_completion.py
```

### Opción 2: Seguir log en tiempo real

```bash
# Linux/Mac
tail -f hyperopt_5000.log

# Windows PowerShell
Get-Content hyperopt_5000.log -Wait
```

### Opción 3: Ver archivo de progreso

```bash
cat hyperopt_progress_5000.json
```

## 🎯 Workflow Completo

### Paso 1: Ejecutar hyperopt (5000 epochs)

```bash
python hyperopt_full_5000.py
```

**Duración:** 1-5 horas dependiendo de hardware

### Paso 2: Revisar mejores parámetros

```bash
# Los mejores parámetros se muestran al final del log
cat hyperopt_5000.log | grep -A 20 "Best result"
```

### Paso 3: Validar en datos de TEST

```bash
# IMPORTANTE: Usar datos OUT-OF-SAMPLE (Jul-Oct 2025)
.venv/Scripts/freqtrade backtesting \
  --strategy DynamicAggressiveHighTP \
  --timerange 20250707-20251006
```

### Paso 4: Comparar Train vs Test

```
Train (Oct 2024 - Jul 2025): +41.28%
Test  (Jul 2025 - Oct 2025): +7.64%  → +30% annualized

Análisis:
✅ Train ≈ Test (dentro del mismo orden de magnitud)
✅ Test positivo (+30% anual)
✅ Parámetros ROBUSTOS → Implementar
```

### Paso 5: Implementar parámetros validados

Editar `pacifica-python-sdk/trading_bot_ws.py`:

```python
# OLD (baseline)
self.rsi_threshold_t3 = 30
self.momentum_threshold_t3 = -0.02
self.stop_loss_t3 = -0.05
self.take_profit_t3 = 0.80

# NEW (optimized from hyperopt)
self.rsi_threshold_t3 = 21      # Del hyperopt
self.momentum_threshold_t3 = -0.038  # Del hyperopt
self.stop_loss_t3 = -0.01       # Del hyperopt
self.take_profit_t3 = 1.50      # Del hyperopt
```

## ⚙️ Configuración Avanzada

### Cambiar número de epochs

Editar `hyperopt_full_5000.py`:

```python
TOTAL_EPOCHS = 5000  # Cambiar a 10000, 2000, etc.
```

### Cambiar período de training

```python
TRAIN_TIMERANGE = "20241006-20250707"  # Ajustar fechas
TEST_TIMERANGE = "20250707-20251006"   # Ajustar fechas
```

### Cambiar spaces de optimización

```python
cmd_hyperopt = [
    ...
    "--spaces", "buy", "sell",  # Agregar: "roi", "stoploss", "trailing"
    ...
]
```

**Opciones de spaces:**
- `buy`: Parámetros de entrada
- `sell`: Parámetros de salida
- `roi`: ROI table (minimal_roi)
- `stoploss`: Stoploss
- `trailing`: Trailing stop

## 🐛 Troubleshooting

### Error: "freqtrade no encontrado"

```bash
# Verificar que venv esté activo
which freqtrade  # Linux/Mac
where freqtrade  # Windows

# Reinstalar si es necesario
pip install -e .
```

### Error: "No se encontró archivo de progreso"

**Normal en primera ejecución**. El archivo `hyperopt_progress_5000.json` se crea automáticamente.

### Hyperopt muy lento

**Opciones:**

1. **Reducir epochs**: Cambiar `TOTAL_EPOCHS = 2000`
2. **Reducir período**: Usar solo 6 meses de datos
3. **Usar VPS más potente**: 4+ cores recomendado
4. **Reducir pares**: Optimizar solo para BTC/ETH/SOL

### Resultados inconsistentes

**Causas comunes:**

1. **Overfitting**: Train >> Test
   - **Solución**: Reducir complejidad, usar menos parámetros

2. **Pocos trades**: < 50 trades en train/test
   - **Solución**: Ampliar período o reducir filtros

3. **Mala suerte**: Train << Test
   - **Solución**: Repetir con diferentes períodos

## 📊 Ejemplos de Output

### Inicio de hyperopt

```
================================================================================
HYPEROPT COMPLETO - 5000 iteraciones con RESUME
================================================================================
Strategy: DynamicAggressiveHighTP
Total epochs: 5000
Train period: Oct 2024 - Jul 2025 (9 meses)
Test period: Jul 2025 - Oct 2025 (3 meses)
================================================================================

[2025-10-06 10:15:30] Progreso cargado: 0/5000 epochs completados
[2025-10-06 10:15:30] Epochs restantes: 5000
[2025-10-06 10:15:30] Iniciando optimización de 5000 epochs...
```

### Progreso durante ejecución

```
Epoch 1/5000: Best: +15.32% (Sharpe: 0.85)
Epoch 2/5000: Best: +15.32% (Sharpe: 0.85)
Epoch 3/5000: Best: +18.45% (Sharpe: 0.92) ← Mejora!
...
Epoch 1234/5000: Best: +41.28% (Sharpe: 1.15)
```

### Finalización

```
================================================================================
✅ OPTIMIZACIÓN COMPLETADA
================================================================================
Total epochs: 5000
Archivo de progreso: hyperopt_progress_5000.json
Log completo: hyperopt_5000.log

SIGUIENTE PASO: VALIDACIÓN
================================================================================

Ejecuta validación en datos de TEST:
  .venv/Scripts/freqtrade backtesting --strategy DynamicAggressiveHighTP --timerange 20250707-20251006

Compara resultados Train vs Test:
  ✅ Train ≈ Test = ROBUSTO (parámetros buenos)
  ⚠️  Train >> Test = OVERFITTING (rechazar)
  ⚠️  Train << Test = SUERTE (repetir)
```

## 🎓 Mejores Prácticas

### 1. Validación Walk-Forward

```bash
# SIEMPRE dividir datos:
Train: 75% de los datos (9 meses)
Test:  25% de los datos (3 meses)

# NUNCA optimizar en datos de test
```

### 2. Epochs Suficientes

```
Mínimo: 1000 epochs
Recomendado: 2000-5000 epochs
Excesivo: >10000 epochs (riesgo de overfitting)
```

### 3. Monitorear Convergencia

```bash
# Si no hay mejoras por 1000+ epochs → convergió
# Continuar es pérdida de tiempo y riesgo de overfitting
```

### 4. Validación Multi-período

```bash
# Después de hyperopt, validar en diferentes períodos:
Q1 2024: Backtest
Q2 2024: Backtest
Q3 2024: Backtest
Q4 2024: Backtest

# Si 3/4 positivos → parámetros robustos
```

## 📚 Referencias

- [Freqtrade Hyperopt Docs](https://www.freqtrade.io/en/stable/hyperopt/)
- [Anti-Overfitting Guide](https://www.freqtrade.io/en/stable/strategy-advanced/)
- [Parameter Optimization](https://www.freqtrade.io/en/stable/backtesting/)

---

**¿Preguntas?** Revisa el log: `hyperopt_5000.log`
