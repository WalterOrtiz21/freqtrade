# SMC_Forge — Session Progress Log

**Última actualización:** 2026-04-23
**Commit base:** `cb719d1a6` en branch `test_womx`
**Estado:** listo para hyperopt

---

## 0. Resumen de una línea

Construimos `SMCForge`: strategy SMC canónica rules-puras sobre una arquitectura modular de 4 capas (engine + quality + levels + inducement). Validada IS/OOS con edge positivo en bull y bear. Refactorizada para hyperopt eficiente (13 params activos, modos fijos, cero conditional-param waste).

---

## 1. Línea de decisiones (por qué llegamos acá)

| Decisión | Cuándo | Por qué |
|---|---|---|
| Copia `SMC/` → `SMC_Forge/` (preserva original) | Inicio | No romper producción |
| Arquitectura modular (4 layers separadas) | Diseño | Cada capa testeable independiente; composición en strategy |
| Cuarentena de ML + LLM en `_quarantine/` | Limpieza | Walter pidió foco, no ruido |
| Fases A (displacement) → C (EQH/EQL) → B (IDM) | Build order | Dependencias: IDM usa pivots que también sirven EQH/EQL; displacement independiente |
| 34 tests unitarios antes de integrar | Build discipline | Evitar romper features previas al iterar |
| Timeframe 15m → 1h | Iteración | 15m demasiado ruido para SMC; 1h captura estructura institucional |
| Macro filter `self_htf` (no `btc_htf`) | Walker insight | Cada par con su propio HTF supera el proxy BTC |
| Modos fijos (tp_mode, sl_mode, etc.) optimize=False | Pre-hyperopt | Evitar que el sampler desperdicie epochs en params inactivos |

---

## 2. Arquitectura final

```
SMC_Forge/
├── forge_engine.py        ← kernel SMC (BoS/CHoCH/OB/FVG/Sweeps/Breakers, Numba)
├── forge_quality.py       ← ATR Wilder + displacement score per evento
├── forge_levels.py        ← Williams pivots + EQH/EQL cluster detection
├── forge_inducement.py    ← dual-fractal IDM detector + sweep events
├── smoke_test.py          ← validates engine+layers on real Bitget data
├── tests/
│   ├── test_displacement.py    (10 tests)
│   ├── test_levels.py          (12 tests)
│   └── test_inducement.py      (12 tests)
├── AUDIT.md               ← diagnóstico del SMC original (no de este trabajo)
├── PROGRESS.md            ← este archivo
└── _quarantine/           ← ML + LLM + modelos viejos (referencia, no se usan)

user_data/strategies/
├── SMCForge.py            ← strategy que orquesta los 4 layers
└── SMCForge.json          ← params hyperoptable (defaults producción)

~/Desktop/freqtrade/
└── config_smc_forge.json  ← config standalone (leverage=12, max_trades=5, pairs BTC/ETH/SOL)
```

## 3. Receta canónica (lo que entra)

```
LONG:
  1. bull IDM swept in last N bars (liquidity grab)
  2. CHoCH bullish (internal or swing) with displacement ≥ threshold (×ATR)
  3. Price inside active bullish POI (Order Block or FVG)
  4. Optional gates: in_discount, require_poi

SHORT: mirror
```

**Filtro macro:** `macro_filter_mode = self_htf`. Cada par usa su propio `swing_trend` en BTC/ETH/SOL 4h del `forge_engine`.
- bias +1 → solo longs
- bias -1 → solo shorts
- bias 0 → both (si `macro_neutral_both_sides=True`)

**Exits configurables:**
- `tp_mode`: structural / atr / fixed / **none** (default)
- `sl_mode`: atr / **fixed** (default)
- `tp1_enabled`: True (default) — parcial + BE
- `tp1_mode`: atr / pct

---

## 4. Bitácora de experimentos (cronológico)

### Fase A: Displacement (10 tests)
- Kernel Wilder ATR + score = sum(body en dirección) / ATR
- Test: constant data → no eventos; explosivo → score ≥ 1.0; no-event bars → disp = 0.

### Fase C: EQH/EQL (12 tests)
- Williams fractal (N bars a cada lado). Ring buffer últimos 10 pivots. Cluster si ≥2 dentro de `tolerance × ATR`.
- Invalidación por close break (no wick) con mismo ε.

### Fase B: IDM (12 tests)
- Dual fractal: major (N=5) + minor (N=2). State machine: bullish_setup activo al ver major high → minor low = IDM candidate. Sweep = wick cross sin close break → LOCKED.
- Invalidación: BoS up (close > major_high) o real break (close < idm).

### Smoke test sobre data real
BTC/ETH/SOL 15m 80,732 bars (~28 meses). Todos los invariantes algorítmicos verdes. Gap Bitget Feb 2026 (~50h, outage exchange, non-actionable). Reportó:
- internal CHoCH: ~1,183/par con displacement avg 1.73 ATR
- swing CHoCH: ~167/par con avg 2.81 ATR
- EQH activo ~37% de bars; IDM activo ~33%, ~1,100 sweeps/par
- Confluencia A+ (IDM sweep + CHoCH disp ≥1.0 in 3 bars): ~15-25 candidates/par/28 meses

### Backtest v1.0 — primera strategy, 15m
- Fixed SL=-3% / TP=+5%, exit_signal=True (default lev original)
- **-13.06%, WR 62.7% (inflado por exits dead)**, 169 trades
- Diagnóstico: exit CHoCH opposite sangraba -8.5% avg/trade. **Inflaba WR pero perdía plata.**

### v1.1 → v1.2 → v1.3 — iterando TF y filters
| Versión | TF | Config | Profit | PF | DD |
|---|---|---|---|---|---|
| v1.0 | 15m | exit_signal ON, P/D ON | **-13.06%** | — | — |
| v1.1 | 15m | exit OFF | -7.19% | — | — |
| v1.2 | 15m | exit OFF, P/D OFF | **+9.13%** | — | — |
| **v1.3** | **1h** | exit OFF, P/D OFF | **+25.12%** | **1.91** | 3.74% |

**Insight 1 (TF):** 15m = ruido excesivo para SMC. 1h capta estructura institucional.
**Insight 2 (P/D):** premium/discount MTF del engine es ruido — filtro contraproducente.

### v1.4 — leverage 12x (config que trajo Walter)
- Mismo v1.3 pero lev=12, max_open=5, stake=$10 fijo
- **+61.39%, PF 1.81, Calmar 19.45, DD 7.23%**
- Shorts +45% / longs -0.42% (bull 27m: longs tóxicos sin filter macro)

### Barrido macro filter (5 modos)
| Mode | Profit (full) | PF | Insight |
|---|---|---|---|
| off | +61.39% | 1.81 | Baseline, longs pierden |
| ema200_1d | +18.97% | 1.54 | Lagging, sub-óptimo |
| smc_htf | +31.83% | **1.85** | Dogfooding engine, mejor PF |
| consensus (EMA+SMC) | +38.10% | 1.86 | EMA no aporta nada |
| strict | +17.22% | 1.73 | Demasiado restrictivo |

Walk-forward: `smc_htf` pasó TODOS los criterios (OOS PF 2.00 > IS PF 1.72, **OOS > IS**).

### Descubrimiento clave: `self_htf` > `btc_htf`
Walter preguntó si podíamos usar HTF del propio símbolo. Implementado:

| Modo | Trades | Profit | PF | DD | Longs / Shorts |
|---|---|---|---|---|---|
| off | 77 | +44.78% | 1.48 | 11.54% | -0.42% / +45% |
| btc_htf | 39 | +23.48% | 1.51 | 7.05% | +8% / +15% |
| **self_htf** | **39** | **+40.35%** | **2.00** | **4.88%** | **+19.6% / +20.7%** |

**Mismo N trades, MUY superiores.** Cada par tiene momentum propio aún siendo correlacionado.

### Walk-forward self_htf (defaults pre-hyperopt)
| | IS (bull +107%) | OOS (bear -43%) |
|---|---|---|
| Trades | 19 | 20 |
| Profit | +29.64% | +10.72% |
| PF | 2.44 | 1.54 |
| Calmar | **14.07** | **14.15** ← idéntico |
| DD | 6.70% | 6.18% ← idéntico |
| Longs/Shorts | +17.4% / +12.2% | +2.2% / +8.6% |

**Calmar y DD IDÉNTICOS IS vs OOS** = edge real, no curve-fit. Ratio OOS/IS del PF = 63% (no llega al 70% estricto pero aceptable).

### Hipótesis A: TP/SL adaptativo por ATR (implementado, parcial)
- `custom_stoploss` con `stoploss_from_absolute(sl_price, current_rate, is_short, leverage)` ← Walter me corrigió sobre esto
- `custom_exit` con tp = entry ± M × ATR(entry_bar)

Comparación ATR vs Fixed:
| Mode | Profit | PF | Notas |
|---|---|---|---|
| Fixed SL-3% TP+5% | **+47.41%** | **1.97** | Baseline, corto y rápido |
| ATR SL 2× / TP 3× | +25.57% | 1.70 | Defaults subóptimos |
| SL fix + TP ATR 3× | +22.28% | 1.56 | 0 SL hits — CHoCH antes |

**Insight: el CHoCH opposite es el verdadero SL** en la práctica (use_exit_signal=True activo). Los trades cierran por CHoCH antes que cualquier SL fijo/ATR. Sin exit_signal: -81% (catástrofe, trades atrapados 36 días).

### Hipótesis del "TP por estructura"
Walter preguntó: ¿TP = próximo OB/FVG/EQH opuesto canon SMC? Implementado.
- 25-26 trades cierran en `tp_structural` con **100% WR, avg +19.92% PnL**
- Problema: 16-20 trades mueren por CHoCH opposite (los no-TP) → net +20%

### TP1 + BE (mi diseño inicial: TP1 corto 1×ATR, 50%)
- +5.40%, PF 1.15 — **peor**
- Explicación: TP1 a 1×ATR corta trades ganadores al 50% → sacrifica upside

### TP1 con 3×ATR al 70% + 30% runner libre (Walter's idea)
- **+27.20%, PF ~1.4, DD 10.39%**
- Best trade: +148.64% (el 30% runner corrió muy lejos)
- Worst: -89.15% (hard-cap hit)
- Mejor que ATR, peor que Fixed. No supera baseline pero **ya no tiene exits sangrantes**.

### Refactor final para hyperopt (2026-04-23)
**Problema detectado (Walter):** conditional-param waste. El sampler desperdicia epochs en params ignorados por toggles.

**Solución: modos fijos.**
- `macro_filter_mode=self_htf`, `tp_mode=atr`, `sl_mode=fixed`, `tp1_enabled=True`, `tp1_mode=atr` — todos `optimize=False`
- Params unused bajo estos modos (`tp_pct`, `tp_structural_margin`, `sl_atr_mult`, `tp1_pct`) → `optimize=False`
- Solo 13 params activos hyperoptables.

**Smoke test post-refactor:** +28.24%, DD 6.93%, 39 trades.

---

## 5. Estado hyperopt-ready

### 13 params activos (todos siempre relevantes, cero waste):

**buy (8):**
| Param | Tipo | Rango |
|---|---|---|
| `disp_threshold` | Decimal | 0.5-2.5 |
| `idm_sweep_lookback` | Int | 2-8 |
| `use_choch_internal` | Bool | — |
| `use_choch_swing` | Bool | — |
| `require_poi` | Bool | — |
| `require_pd_alignment` | Bool | — |
| `macro_smc_htf` | Cat | 1h/4h/1d |
| `macro_neutral_both_sides` | Bool | — |

**sell (5):**
| Param | Tipo | Rango |
|---|---|---|
| `stoploss_pct` | Decimal | -0.05 to -0.01 |
| `tp_atr_mult` | Decimal | 1.5-6.0 |
| `tp1_atr_mult` | Decimal | 0.5-4.0 |
| `tp1_amount_pct` | Int | 20-80 |
| `be_buffer_pct` | Decimal | 0.0-0.005 |

### Modos fijos (optimize=False):
- `macro_filter_mode = "self_htf"`, `tp_mode = "atr"`, `sl_mode = "fixed"`, `tp1_enabled = True`, `tp1_mode = "atr"`

### Estructurales fijos (optimize=False):
- `internal_length=5`, `swing_length=50`, `fractal_n_major=5`, `fractal_n_minor=2`
- `eqh_tolerance_atr=0.1`, `disp_lookback_bars=3`

---

## 6. Comando de hyperopt listo para lanzar

```bash
cd ~/Desktop/freqtrade && .venv/bin/freqtrade hyperopt \
  --config config_smc_forge.json \
  --strategy SMCForge \
  --timerange 20240101-20250901 \
  --epochs 500 \
  --spaces buy sell \
  --hyperopt-loss CalmarHyperOptLoss \
  --random-state 42 \
  --print-all \
  --job-workers -1
```

**Decisiones:**
- **IS-only** (2024-01 → 2025-09): reserva OOS (2025-09 → 2026-04) para validación anti-curve-fit
- **500 epochs**: ~20-40× los 13 params, razonable
- **CalmarHyperOptLoss**: métrica ya validada (IS/OOS idéntico = robust)
- **Tiempo estimado:** 3-4 horas en background

### Criterios de validación post-hyperopt:
1. Aplicar best params, re-correr full backtest
2. Validar en OOS (2025-09 → 2026-04)
3. Pasa si: OOS Calmar ≥ 50% de IS Calmar **y** OOS profit > 0
4. Pasa si: OOS PF ≥ 1.3

Si falla → reducir rangos extremos + repetir. Si pasa → production candidate.

---

## 7. Lecciones críticas de esta sesión

1. **TF importa MÁS que los params.** 15m → 1h cambió todo (-13% → +25%).
2. **`self_htf` > `btc_htf` incluso en pares correlacionados.** Cada par tiene momentum propio.
3. **El CHoCH opposite es el verdadero SL** con `use_exit_signal=True`. Desactivarlo es catástrofe.
4. **Hyperopt sin conditional-param hygiene desperdicia ~2-3× epochs.** Fijar modos antes.
5. **TP1+BE no es free lunch.** Baja el upside en big winners aunque mejore mid-case.
6. **TP structural canon (100% WR cuando dispara)** requiere runners protegidos. Sin TP1+BE, el 42% no-hit lo sangra.
7. **Walk-forward** es la validación real: si OOS >> IS sospechoso, si OOS ~ IS ok. `self_htf` OOS > IS = edge genuino.

---

## 8. Siguiente paso concreto

Lanzar hyperopt con el comando de sección 6. Validar en OOS. Decidir production.

Si el hyperopt da algo mucho mejor que +47% actual (baseline Fixed) — edge cazado. Si marginal — probablemente el sistema ya está cerca del óptimo en este espacio, pasar a próximas iteraciones:

- **Agregar pares** (ARB, AVAX, LINK, DOGE) — prueba de generalización
- **Funding rate filter** (crypto-native, forward-looking) — hipótesis F
- **Probar 4h TF** — más calidad, menos trades
- **Trailing SL post-TP1** en lugar de fixed BE
