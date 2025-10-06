# Hyperopt - Elige tu Estrategia de Datos

## ✅ Datos Disponibles
- **2023-10-06 a 2025-10-06** (2 años completos)

## 🎯 Opción 1: Últimos 365 Días (RECOMENDADO para trading actual)

```powershell
python hyperopt_365days.py
```

**Características:**
- Training: Oct 2024 - Jul 2025 (9 meses)
- Test: Jul 2025 - Oct 2025 (3 meses)
- ✅ Datos MÁS RECIENTES
- ✅ Condiciones de mercado actuales
- ✅ Relevante para trading en 2025
- ⏱️ Tiempo: ~20-40 minutos

**Validación:**
```powershell
python validate_365days.py
```

---

## 📊 Opción 2: 2 Años Completos (MÁS ROBUSTO estadísticamente)

```powershell
python hyperopt_2years.py
```

**Características:**
- Training: Oct 2023 - Apr 2025 (18 meses)
- Test: May 2025 - Oct 2025 (6 meses)
- ✅ MÁS DATOS = mejor validación
- ✅ Diferentes condiciones de mercado (bear + bull)
- ✅ Reduce overfitting
- ⏱️ Tiempo: ~30-60 minutos

**Validación:**
```powershell
python validate_2years.py
```

---

## 🤔 ¿Cuál Elegir?

### Elige **365 días** si:
- ✅ Quieres parámetros para trading ACTUAL (2025)
- ✅ El mercado ha cambiado significativamente vs 2023
- ✅ Prefieres optimización rápida

### Elige **2 años** si:
- ✅ Quieres parámetros MÁS ROBUSTOS
- ✅ Necesitas validación estadística fuerte
- ✅ Buscas parámetros que funcionen en múltiples condiciones

---

## 📈 Mi Recomendación

**Ejecuta AMBOS** y compara:

```powershell
# 1. Hyperopt con 365 días (más rápido)
python hyperopt_365days.py
python validate_365days.py

# 2. Hyperopt con 2 años (más robusto)
python hyperopt_2years.py
python validate_2years.py
```

**Criterio de decisión:**
- Si **ambos** dan resultados positivos → Usa parámetros de **2 años** (más robusto)
- Si **solo 365 días** funciona → El mercado cambió, usa esos parámetros
- Si **ninguno** funciona → La estrategia no es viable

---

## 🚨 Nota Importante

El hyperopt anterior (2024 solamente) falló porque:
- ❌ Probó solo 9 meses de datos (insuficiente)
- ❌ No incluyó datos de 2025 (condiciones actuales)
- ❌ No validó en out-of-sample

**Ahora tenemos:**
- ✅ Datos hasta HOY (2025-10-06)
- ✅ 2 años completos para validación
- ✅ Split train/test correcto (75/25)
