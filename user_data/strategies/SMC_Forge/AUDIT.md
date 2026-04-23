# SMC_Forge — Auditoría Inicial

**Fecha:** 2026-04-22
**Auditor:** Liz (CEO) + skill `/smc`
**Base:** copia íntegra de `user_data/strategies/SMC/` al 2026-04-22
**Objetivo:** diagnosticar qué tenemos, qué falta, y definir base sólida antes de iterar estrategia.

---

## 0. TL;DR

El sistema es **serio, grande y con varios aciertos estructurales** (mitigación sistemática de lookahead, multi-zone tracking, macro bias, dynamic TP1, LLM/ML opcionales). Pero tiene **tres huecos que hoy lo mantienen por debajo del canon SMC**:

1. **No hay IDM (Inducement)** enforced — el concepto central de SMC no está en el engine ni en la estrategia. Solo se computa post-hoc en `analyze_features.py` para estudio, nunca como gate de entrada.
2. **El sweep está mal usado** — `use_sweep_bypass` (L1893-1911) trata al sweep como señal extra para entrar sin zona. El canon dice lo opuesto: el sweep es el paso *previo* a esperar CHoCH + displacement y entrar en OB/FVG. La configuración por default lo deja en False, pero el concepto está al revés.
3. **No hay displacement validation** — OBs y FVGs se crean estructuralmente sin validar que hubo impulso mínimo (ATR×N) después. Resultado: OBs retail-grade entran al mismo pool que OBs institucionales.

Sumado: varios faltantes de segundo orden (EQH/EQL, Strong/Weak highs-lows, OTE, CE, BPR, data forward-looking crypto como funding/OI/liquidations). Y dos bugs puntuales en el pipeline ML (threshold leak + split multi-pair roto).

**Veredicto:** base útil, no botarla. Forjar sobre ella: cerrar los 3 huecos críticos, agregar data forward-looking, y hacer el ML/LLM opt-in solo después de que la estrategia reglas-puras tenga edge probado.

---

## 1. Inventario y mapa de dependencias

```
SMC_Forge/
├── smc_engine.py               28 KB  ← Numba kernel (BoS/CHoCH/OB/FVG/Sweeps/Breakers)
├── SMCWithMLLuxAlgo.py        141 KB  ← Estrategia Freqtrade (~3000 líneas)
├── SMCWithMLLuxAlgo.json        4 KB  ← Hyperopt params activos
├── train_smc_model.py          27 KB  ← Pipeline ML (XGBoost + Optuna)
├── llm_filter.py               19 KB  ← Filtro LLM via OpenRouter
├── test_llm_filter.py          10 KB  ← Tests del filtro
├── analyze_features.py         20 KB  ← Post-hoc feature analysis
├── analyze_backtest.py         21 KB  ← Post-hoc backtest analysis
├── feature_analysis.csv        ~     ← Output de análisis
├── feature_analysis_by_subsetup.csv
├── models/                            ← Modelos XGBoost serializados
└── analisis/                          ← Parquets exportados por `export_indicator_data`
```

**Dependencias internas:**

```
SMCWithMLLuxAlgo.py
  ├── smc_engine.SMCEngine               (kernel SMC, llamado 3 veces: MTF 15m + HTFs + BTC 4h)
  ├── llm_filter.LLMConfluenceFilter     (opt-in en confirm_trade_entry)
  └── train_smc_model.train_model        (opt-in con enable_auto_training)

train_smc_model.py
  └── smc_engine.SMCEngine               (mismo kernel, consistencia signals train vs live)

llm_filter.py
  └── OpenRouter API                     (z-ai/glm-5-turbo default, fallback a free models)
```

**Deps externas:** numba, xgboost, optuna, scikit-learn, pandas, talib, requests, freqtrade.

---

## 2. Auditoría — `smc_engine.py`

### 2.1 Lo que hace bien (canon SMC cumplido)

| Concepto | Línea | Validación |
|---|---|---|
| BoS / CHoCH por **body close** (no wick) | 194, 212, 245, 263 | ✅ Correcto según canon |
| Dual structure (internal length=5 + swing length=50) | 114-171 | ✅ LuxAlgo-style |
| OB = lowest-low / highest-high entre pivote y break | 204, 222, 255, 273 | ✅ "última vela contraria al impulso" |
| FVG detection 3-bar (`low[i] > high[i-2]`) | 294-300 | ✅ Correcto |
| Sweep separado de BoS (wick + close back) | 232, 237, 283, 288 | ✅ Definición canónica |
| Mitigation por **close**, no por wick | 387, 401, 427, 453 | ✅ Correcto |
| Breaker flip (OB roto → breaker opuesto) | 389, 403 | ✅ Canon |
| **Inverse FVG** (FVG roto → breaker opuesto) | 427-429, 453-455 | ✅ Avanzado, bien implementado |
| Raw event arrays para multi-zone downstream | 703-725 | ✅ Buena separación kernel/strategy |

### 2.2 Faltantes vs canon SMC

| Concepto | Importancia | Notas |
|---|---|---|
| **IDM (Inducement)** | 🔴 Crítica | Ni detectado ni expuesto. El corazón de SMC. |
| **EQH / EQL** | 🔴 Alta | Pools de stops obvios no identificados |
| **Displacement filter** | 🔴 Alta | OB se crea sin validar impulso post-break (1-3 velas, ATR×N) |
| **Strong / Weak Highs/Lows** | 🟡 Media | BigBeluga las tiene. No distingue swings "defendidos" de "próximos a barrerse" |
| **CE (Consequent Encroachment) de FVG** | 🟡 Media | Entry precision 50% del FVG no marcado |
| **BPR (Balanced Price Range)** | 🟡 Media | Dos FVGs opuestos superpuestos — zona alta probabilidad |
| **Volume Imbalance** | 🟢 Baja | Gap body→wick menor al FVG |
| **Mitigation Block vs Breaker distinction** | 🟢 Baja | Engine trata igual al OB que falló que al que sostuvo y después rompió |
| **Propulsion Block** | 🟢 Baja | OB ya mitigado ofreciendo re-entry |
| **Ranking de calidad de OB** | 🟡 Media | Fresh vs touched, HTF vs LTF, con displacement, alto vol — el engine expone raw; la strategy compensa parcialmente en multi-zone tracker |

### 2.3 Riesgos menores en el código

- **L20, L25:** `min_val = 1e15`, `max_val = -1.0` hardcoded. Frágil si price sale del rango asumido. OK para crypto actual.
- **L145-151, L177-183:** State machine de pivots correcta pero compleja. Se podría cubrir con tests unitarios con ondas conocidas.
- **L703-725:** export `*_raw` arrays sustituye NaN por 0. Downstream depende de `> 0` checks; si algún precio real fuera 0 (no en crypto) rompería. No-issue práctico.

### 2.4 Conclusión engine

El kernel **está bien hecho en lo que hace**, pero **hace menos que el canon SMC**. Tres features críticas (IDM, EQH/EQL, displacement) no están. Incluso BigBeluga — de donde se inspiró `require_volumetric_ob` y `fvg_atr_threshold` — expone Strong/Weak highs-lows y EQH/EQL que no se portaron.

---

## 3. Auditoría — `SMCWithMLLuxAlgo.py`

### 3.1 Aciertos estructurales

| Feature | Línea | Por qué es bueno |
|---|---|---|
| `shift(1)` sistemático en HTF antes del merge | 722-723 | Previene lookahead de vela HTF no cerrada |
| `shift(1)` en `active_*` zone columns | 649-650 | Mitigation zones reflejan estado al cierre de vela previa |
| `shift(1)` en macro bias BTC 1D y 4H | 1288, 1304 | Macro nunca usa vela en formación |
| `merge_asof(direction="backward")` | 1319, 1324 | Alignment HTF→MTF sin leak |
| Multi-zone tracker con shift final | 1022-1244 | Tracks N OBs + N FVGs + N Breakers con metadata (rvol, age, touches, filled_pct) |
| Macro bias = **consensus** 1D EMA200 + 4H swing trend | 1334-1337 | Solo sesga si ambos TFs coinciden |
| HTF zone requirement real (price-in-zone check) | 1978-2014 | 15m CHoCH = trigger; HTF zone = "why" (presencia institucional) |
| Dynamic TP1 según opposing zone proximity | 2016-2092 | Evita TP detrás de resistencia obvia |
| Circuit breaker market panic | 2723-2728 | Bloquea nuevos trades durante volatilidad extrema |
| Cooldown post-close | 2738-2769 | Evita re-entry immediate en mismo signal |
| LLM en `confirm_trade_entry` (1×trade) | 2802-2836 | Eficiente, no per-candle |
| Per-symbol ML models + fallback general | 426-518 | Mejor ajuste por par |
| Shadow mode LLM | 2802, llm_filter:109 | Testing sin bloquear producción |

### 3.2 Red flags — severidad ALTA

#### 3.2.1 🔴 Sweep bypass invertido (L1893-1911)

```python
# Sweep bypasses zone requirement (INCLUDING volumetric filter)
bullish_zone |= recent_bull_sweep
bearish_zone |= recent_bear_sweep
```

**Problema:** trata al sweep como señal extra para entrar *sin zona*. El canon SMC dice lo opuesto — el sweep (especialmente el del IDM) es la **condición previa**, no la señal de entrada. Flujo correcto:

```
1. Identificar pool de liquidez (swing high/low, EQH/EQL, IDM)
2. Esperar sweep (wick barre el pool sin body close)
3. Esperar CHoCH/MSS con displacement post-sweep
4. Entrar en OB o FVG válido que quedó en el impulso
```

La implementación actual equivale a "ya barrió, entremos sin esperar confluencia". El sweep alone + CHoCH **puede** funcionar si el CHoCH incluye displacement, pero el código no valida displacement — solo revisa que el sweep ocurrió dentro de `sweep_lookback` (default 3 velas).

**Mitigante:** `default=False`. Pero sigue siendo hyperopt-able, puede activarse y producir trades que institucionalmente son trampas.

**Recomendación:** invertir la lógica → el sweep es **prerequisito** del entry (AND), no bypass (OR). Y validar displacement post-sweep (body moves ≥ N × ATR en M velas).

#### 3.2.2 🔴 Filtros hardcoded desde backtests previos (L2774-2797 + `llm_filter.py` L31-80)

```python
if self.filter_block_kz_ldn.value and "KZ_LDN" in tag:
    return False
if self.filter_block_sweep.value and "+SWEEP" in tag:
    return False
```

Y peor, el LLM system prompt:

```
[HTF_ALIGN subsetup — baseline 50.6%, n=154]
  + bars_since_swing_choch 0-3  → 64% WR  (+14pp)
  - SHORT + btc_macro_bias = -1 → 35% WR  (-15pp)
  - is_monday = True            → 36% WR  (-15pp)
```

**Problema:** son **curve-fit empírico convertido en regla**. El LLM no está razonando, está regurgitando WRs de un backtest específico (n=154). En cuanto el régimen cambie, estas reglas dejan de valer y el LLM va a seguir aplicándolas.

Peor aún, la sección `FEATURES WITH NO PROVEN SIGNAL (IGNORE)` le enseña al LLM a ignorar **conceptos SMC válidos** (OTE, equilibrium, inducement sweep, displacement) porque el backtest no los encontró discriminantes. Están descartados a priori por un N=358 trades.

**Mitigante:** todos los filtros default `False`. LLM default `False`.

**Recomendación:** tratar estos filtros como "configs de shadow mode", nunca activar en producción sin un **out-of-sample fresco ≥ n=200**. El prompt del LLM debe ser instructivo (canon SMC), no empírico (WRs pasados).

#### 3.2.3 🔴 `enter_tag = "NoZone"` permitido (L2185, L2199)

Cuando `required_zone == "none"`, la strategy permite entrar sin confluencia de OB/FVG/Breaker. Solo basa la decisión en CHoCH + trend. El canon SMC requiere POI (OB/FVG) para entry. Sin zona = entry a mercado en la nada.

**Recomendación:** deprecar el valor `"none"` del enum, o al menos forzar confluencia mínima (ej. premium/discount + BTC bias alineado).

### 3.3 Red flags — severidad MEDIA

#### 3.3.1 🟡 `ewm(span=200, min_periods=1)` en macro bias (L1282)

EMA200 empieza desde la barra 1, convergiendo a EMA200 real recién ~200 barras después. Durante warmup, el macro bias es ruido. `startup_candle_count=400` cubre 15m, pero BTC 1D con 400 velas ya son 400 días — suficiente. Bien en la práctica, pero `min_periods=200` sería más strict para evitar bias espurio al inicio de backtests cortos.

#### 3.3.2 🟡 Macro bias asume correlación BTC (L1276, L1291)

Todo el filtro macro se basa en BTC. Para memes (DOGE, PEPE, WIF, FLOKI en training pairs) o alts con narrative (TAO, JTO), la correlación con BTC se rompe en catalysts específicos. Puede bloquear trades válidos o permitir malos.

**Recomendación:** por par, detectar régimen de correlación rolling. Si corr(30d) con BTC < 0.5, desactivar macro filter para ese par.

#### 3.3.3 🟡 HTF zone check usa `dataframe["low"]` y `dataframe["close"]` actuales (L2002-2003, L2011-2012)

Las HTF zones ya están shift(1)'d antes del merge (L722-723), pero `low/high/close` del MTF son del bar actual. Para entry signal esto es correcto (vela cerrada por `process_only_new_candles=True`), pero conviene dejar un comentario explícito porque es fácil romperlo con cambios futuros.

### 3.4 Red flags — severidad BAJA

- **L1397-1399:** `above_pdh` / `below_pdl` computados pero no usados como gate de entrada. PDH/PDL son pools de liquidez obvios — entrar sin considerarlos es desaprovechar contexto.
- **L1509-1510, L1538-1539:** Fib levels calculados pero no usados en entry logic. Existe la infraestructura para OTE pero no se usa.
- **L740, L1476-1541:** `premium_discount` computado pero **no enforced** como gate. El canon dice "longs solo en discount, shorts en premium". Hoy es feature informativa, no filtro.

### 3.5 Conclusión estrategia

La estrategia está **ingenierilmente bien construida** (lookahead mitigation excelente, arquitectura clara). Pero **la política de trading es laxa vs canon SMC**:

- Permite entrar sin zona (`required_zone="none"`).
- Usa el sweep al revés (bypass en vez de prerequisito).
- No filtra por premium/discount aunque lo calcula.
- No requiere IDM swept antes del entry.
- No valida displacement post-break.

Resultado: el setup A+ ("IDM swept + CHoCH con displacement + OB fresh en discount + BTC aligned") no es distinguible del setup B ("CHoCH + cualquier OB + tendencia ok"). Todos entran al mismo pool.

---

## 4. Auditoría — `train_smc_model.py` (ML pipeline)

### 4.1 Aciertos

| Feature | Línea | Comentario |
|---|---|---|
| `TimeSeriesSplit` en CV | 576 | Correcto para series temporales |
| `scale_pos_weight` derivado de data | 535-540 | Correcto para imbalance |
| `eval_metric="aucpr"` | 567 | PR-AUC mejor que ROC para minority-class |
| Hyperopt via Optuna | 551-594 | 100 trials, objective = mean CV score |
| Optimal threshold F1-maximizado | 632-646 | Razonable — **pero ver 4.2.1** |

### 4.2 Bugs / red flags ALTOS

#### 4.2.1 🔴 Threshold leak en test set (L632-646)

```python
precision, recall, thresholds = precision_recall_curve(y_test, probs)
f1_scores = 2 * (precision * recall) / (precision + recall)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
```

El threshold óptimo se calcula en el **mismo test set** usado para reportar AUC y PR-AUC. Luego se guarda a JSON y se usa en producción. Clásico data leakage del threshold.

**Impacto:** el F1 reportado es **optimistic por construcción**. En live, la performance va a ser ≤ lo reportado, sistemáticamente.

**Fix:** partir `X_train_full` en train(60%) / val(20%) / test(20%). Tunear threshold en val, reportar y usar en test. O mejor, walk-forward con threshold recalculado por fold.

#### 4.2.2 🔴 Split temporal roto en multi-pair (L528-531 + L391)

```python
# load_data: pd.concat(all_data, ignore_index=True)
# ...
split = int(len(X) * 0.8)
X_train_full, X_test = X.iloc[:split], X.iloc[split:]
```

Los pares se concatenan secuencialmente. Después `iloc[:80%]` corta donde sea — probablemente deja al último par casi entero en test. **No es un split temporal limpio.** Además `TimeSeriesSplit` en el CV asume X ordenado temporalmente — con multi-pair concat, esto NO se cumple.

**Impacto:** los folds de CV pueden tener distribuciones de par desequilibradas, inflando o desinflando AUC según el par que domine el fold.

**Fix:** hacer split por par primero (cada par split 80/20 por su propia fecha), después concatenar los train sets juntos y test sets juntos. O usar `GroupTimeSeriesSplit`.

#### 4.2.3 🔴 `future_window=96` @ 15m = 24h (L422, L112)

Label: "TP hit before SL en las próximas 96 velas". 24h es mucho — un setup SMC de 15m típicamente se resuelve en 4-12h. Con 24h, muchos trades ganan por **drift general del mercado**, no por el setup. El modelo aprende ruido macro, no edge SMC.

**Fix:** reducir a 16-32 velas (4-8h). O hacer label dinámico: "TP antes que SL, condicional a que swing opuesto no se forme" (invalidación estructural antes de tiempo).

### 4.3 Red flags MEDIOS

#### 4.3.1 🟡 Direction como feature + OOD en inference (L523, L2112, L2118)

Training: `direction` columna = 1 para longs, -1 para shorts, solo donde hay signal.
Inference: strategy arma `X_long` con direction=1 y `X_short` con direction=-1 para **cada fila**, aunque esa fila solo tenga signal_long o solo signal_short.

El model nunca vio "score long sobre un candle que originalmente era short signal" durante train. La predicción downstream funciona porque se AND'ea con la condición SMC, pero el score cruzado es OOD.

**Fix:** entrenar dos modelos (uno long, uno short) con features limpias por dirección. Evita el OOD completamente.

#### 4.3.2 🟡 Mezcla large caps + memes en mismo dataset (L62-98)

Training pairs incluye BTC/ETH/BNB (large caps) junto con PEPE/FLOKI/1000BONK (memes). Régimenes distintos. El modelo aprende un promedio que no es óptimo para ninguno.

**Fix:** o bien entrenar per-symbol (ya soportado con `per_symbol_models=True`), o bien entrenar por clusters (large cap / mid / meme) y cargar el cluster correcto por par.

---

## 5. Auditoría — `llm_filter.py`

### 5.1 Aciertos

- Cache por market-state (L438-451) — mismo estado hit cache, ahorra API calls.
- Shadow mode para testing no-destructivo (L109-111, L151-153, L210-213).
- Fallback model en 402/403 (L380-386) — tolerancia a modelos discontinuados.
- Rate limit handling exponencial (L375-378).
- JSON parsing robusto que extrae del **último** bloque JSON (L405-425) — funciona con reasoning models que piensan en voz alta.

### 5.2 Red flags

- **System prompt es curve-fit**: toda la sección EMPIRICAL RULES (L40-69) son WRs de un backtest de N=358. El LLM no razona, aplica. Ver 3.2.2.
- **"FEATURES WITH NO PROVEN SIGNAL (IGNORE)"** (L64-69): descarta OTE, equilibrium, inducement sweep. Anti-canon.
- **Cache key no incluye volatilidad** (L438-451): mismo "market state" en régimen de alta vol vs baja vol hit mismo cache. Confidence de régimen pasado se aplica al presente.
- **Fail-open en API errors** (L233-238): fino por ahora, pero debería fail-closed tras N fallos consecutivos para no dejar la estrategia ciega silenciosamente.

### 5.3 Conclusión LLM filter

Bien ingenierilmente (cache, retry, shadow, parsing robusto), pero **epistemológicamente cuestionable** en el estado actual. El prompt memoriza un backtest en lugar de aplicar principios SMC. Si se usa, debería ser en shadow mode mientras se reemplaza el prompt por algo instructivo.

---

## 6. Análisis de brecha vs canon SMC

Leyenda: ✅ presente y correcto · 🟡 presente pero débil · ❌ ausente

| Concepto SMC | En engine | En strategy | En ML/LLM | Gap |
|---|---|---|---|---|
| BoS / CHoCH | ✅ | ✅ | ✅ (feature) | — |
| MSS (CHoCH + displacement) | ❌ | ❌ | ❌ | **Falta displacement filter** |
| Order Block | ✅ | ✅ multi-zone | ✅ feature | — |
| OB con displacement post-break | ❌ | ❌ | ❌ | **Falta** |
| Ranking calidad OB (fresh, RVOL, HTF) | ❌ engine | 🟡 RVOL + age en multi-zone | ✅ features | Falta ranking compuesto |
| Strong / Weak Highs/Lows | ❌ | ❌ | ❌ | **Falta (BigBeluga lo tiene)** |
| FVG / BISI / SIBI | ✅ | ✅ | ✅ | — |
| CE (50% FVG) | ❌ | ❌ | ❌ | Falta entry precision |
| IFVG (FVG roto → breaker) | ✅ | ✅ | ❌ | OK |
| BPR (2 FVGs opuestos) | ❌ | ❌ | ❌ | Falta |
| Volume Imbalance | ❌ | ❌ | ❌ | Falta |
| Breaker Block | ✅ | ✅ | ✅ | — |
| Mitigation Block (vs Breaker) | ❌ | ❌ | ❌ | Falta distinción |
| Propulsion Block | ❌ | ❌ | ❌ | Falta |
| **IDM (Inducement)** | ❌ | ❌ enforced, 🟡 en analyze_features | ❌ | 🔴 **Crítico** |
| Liquidity Sweep | ✅ | 🟡 mal usado (bypass) | ✅ feature | 🔴 **Invertir lógica** |
| **EQH / EQL** | ❌ | ❌ | ❌ | 🔴 **Alta** |
| SFP (Swing Failure Pattern) | ❌ | ❌ | ❌ | Media (BigBeluga lo tiene) |
| PDH / PDL / PWH / PWL | ❌ engine | ✅ strategy | ✅ feature | OK |
| Premium / Discount | ❌ engine | ✅ calculado, ❌ enforced | ✅ feature | 🟡 No usado como gate |
| Equilibrium (50%) | ❌ engine | ✅ calculado | ✅ feature | 🟡 No usado |
| OTE (62-79%) | ❌ | ❌ | ❌ | Media |
| Sesión / Killzones | ❌ engine | ✅ strategy | ✅ feature | OK |
| BTC bias (macro) | ❌ engine | ✅ consensus | ✅ feature | OK |
| **Funding rate** (crypto) | ❌ | ❌ | ❌ | 🔴 **Falta — freqtrade lo soporta nativo** |
| **Open Interest** (crypto) | ❌ | ❌ | ❌ | 🔴 **Falta — requiere integración externa** |
| **Liquidation heatmap** (crypto) | ❌ | ❌ | ❌ | 🔴 **Falta — Coinglass/Hyblock API** |
| Basis (spot vs perp) | ❌ | ❌ | ❌ | Media |
| FVG raid detection | ❌ | ❌ | ❌ | BigBeluga lo tiene |

---

## 7. Bibliografía externa consultada

**Librerías:**
- [`joshyattridge/smart-money-concepts`](https://github.com/joshyattridge/smart-money-concepts) — Python SMC lib (FVG, BoS/CHoCH, OB, Liquidity, Previous H/L, Sessions, Retracements). **Confirma los gaps:** la lib estándar TAMPOCO tiene IDM, EQH/EQL, premium/discount, OTE, breaker distincto, displacement filter. Nuestro engine está al nivel de lo público.

**Indicadores de referencia (TradingView):**
- [LuxAlgo — Smart Money Concepts (SMC)](https://www.luxalgo.com/library/indicator/smart-money-concepts-smc/) — el Pine original que inspira `SMCWithMLLuxAlgo.py`.
- [LuxAlgo — Market Structure with Inducements & Sweeps](https://www.luxalgo.com/library/indicator/market-structure-with-inducements-sweeps/) — **versión más reciente que SÍ incluye IDM**. Relevante para portar.
- [BigBeluga — Price Action Smart Money Concepts](https://www.tradingview.com/script/jvNJYfbL-Price-Action-Smart-Money-Concepts-BigBeluga/) — origen de `require_volumetric_ob` y `fvg_atr_threshold`. **Tiene adicionalmente**: Strong/Weak H-L, EQH/EQL, SFP, FVG raid, liquidity prints, deviation areas — **NINGUNO portado**.

**Artículos / framework:**
- [ACY — Confirmation Model: OB + FVG + Liquidity Sweep](https://acy.com/en/market-news/education/confirmation-model-ob-fvg-liquidity-sweep-j-o-20251112-094218/) — define la **secuencia canónica de 3 actos**: (1) sweep primero → (2) displacement + FVG → (3) retrace a OB-FVG overlap. Esto es lo contrario de como usamos el sweep hoy (bypass). Modelo a internalizar.
- [TradingFinder — Inducement en SMC/ICT](https://tradingfinder.com/education/forex/inducement/) — definición y mecánica IDM.
- [Trade The Pool — SMC Core Principles](https://tradethepool.com/mental-skill/understanding-smart-money-concepts-core-principles/) — referencia conceptual.
- [LiquidityFinder — OB+FVG+Sweep Confirmation](https://liquidityfinder.com/news/the-confirmation-model-ob-fvg-liquidity-sweep-smart-money-concepts-50fe3) — replicación del mismo modelo.

**Freqtrade (futures data):**
- [Freqtrade — Data Downloading](https://www.freqtrade.io/en/stable/data-download/) — desde v2025.12 soporta funding rate dinámico 1h nativo. **Disponible, no usado**.
- [Issue #11082 — Order Book / OI / CVD integration](https://github.com/freqtrade/freqtrade/issues/11082) — OI no es nativo, hay que integrarlo vía CCXT manual o Coinglass API.
- [Issue #7302 — Funding rate usage patterns](https://github.com/freqtrade/freqtrade/issues/7302) — patrones documentados para leer funding en strategies.

---

## 8. Roadmap priorizado

### Fase 1 — Cerrar huecos críticos del canon (ALTA prioridad)

1. **IDM Detector**: función que dado un swing point grande identifica el último pullback menor (IDM candidate) y expone `idm_high_level`, `idm_low_level`, `idm_swept` por bar. Sin esto, nada de SMC real se puede implementar.
2. **Invertir lógica del sweep**: reemplazar `use_sweep_bypass` por `require_idm_swept`. El entry debería requerir: IDM swept ∧ CHoCH/MSS con displacement ∧ OB/FVG fresh.
3. **Displacement filter**: en el engine, validar que tras un BoS/CHoCH hubo `N` velas con body movement ≥ `K × ATR`. Marcar `bos_with_displacement` / `choch_with_displacement`. Solo OBs con displacement se flaguean como "A-grade".
4. **EQH/EQL detection**: marcar niveles donde 2+ highs o lows están dentro de `ε × ATR`. Exponer `eqh_level`, `eql_level` como magnets candidatos para pools de liquidez.
5. **Premium/Discount como gate**: hoy se calcula pero no filtra. Agregar param `require_pd_alignment` que bloquee longs en premium y shorts en discount por default.

### Fase 2 — Data forward-looking crypto (ALTA prioridad)

6. **Funding rate filter**: usar `dp.get_pair_dataframe(pair, timeframe, candle_type="funding_rate")` (nativo freqtrade). Bloquear longs cuando funding > +0.1%/8h sostenido 3+ bars. Inverso para shorts.
7. **Open Interest delta**: integrar CCXT `fetch_open_interest` via informative_pairs custom hook. Detectar divergencias OI-precio como señal de reversión.
8. **Liquidation clusters**: integrar Coinglass API (free tier) para traer niveles de liquidation clusters como zonas magnéticas. Combinar con EQH/EQL y HTF pools.

### Fase 3 — Precision entry (MEDIA prioridad)

9. **CE (50% FVG)**: marcar el punto medio de cada FVG activo. Usar como entry preciso en vez de "price-in-FVG".
10. **OTE zone (62-79% + sweet spot 70.5%)**: en el rango de swing relevante, flaguear zona OTE y usar en combinación con OB/FVG para A+ entries.
11. **Strong / Weak Highs-Lows** (BigBeluga-style): distinguir swings "defendidos" (post-BoS opuesto) de swings "próximos a barrerse". Solo los weak son targets de liquidez.

### Fase 4 — ML/LLM sano (MEDIA prioridad, opt-in)

12. **Fix threshold leak**: train / val / test split. Tunear threshold en val, reportar en test.
13. **Fix multi-pair split**: split per-pair primero, después concatenar. O `GroupTimeSeriesSplit`.
14. **Reducir `future_window` a 32 velas** (8h @ 15m) o label condicional por invalidación estructural.
15. **Reemplazar LLM prompt empírico por instructivo**: canon SMC (sweep → displacement → OB retrace) en vez de WRs memorizados.
16. **Dos modelos ML separados** (long + short) en vez de direction-as-feature.

### Fase 5 — Baseline rules-puras primero (GATE)

**Antes de Fase 4**: validar que la estrategia **sin ML ni LLM**, con solo Fases 1+2+3, tiene edge OOS positivo en walk-forward. Si no lo tiene, el ML/LLM no lo va a "arreglar" — solo va a curve-fittear más.

Criterio de go/no-go: PF ≥ 1.3, WR ≥ 45% con RR ≥ 1.5, max DD ≤ 15%, en ≥ 2 regimes distintos (bull + bear + choppy).

---

## 9. Archivos producidos por esta auditoría

- `SMC_Forge/` — copia íntegra de SMC/ al 2026-04-22, intocada
- `SMC_Forge/AUDIT.md` — este documento

**Próximo paso sugerido:** discutir Fase 1 con Walter, priorizar 1-2 items concretos para el primer sprint. IDM detector es el mejor candidato porque desbloquea todo lo demás.
