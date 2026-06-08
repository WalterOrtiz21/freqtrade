# ICTSilverBullet — Reescritura fiel del modelo ICT Silver Bullet

- **Fecha:** 2026-06-08
- **Estado:** Diseño aprobado (Walter) + review Sofia ✅ (⚠️ luz verde para codear, gates duros para concluir edge — incorporados a §4/§5) — pendiente review final de Walter del spec antes de pasar a plan de implementación
- **Autor:** Liz + experto SMC/ICT + ft-audit + Sofia (quant)
- **Reemplaza conceptualmente a:** `user_data/strategies/SMCForgeSilverBullet.py` (queda intacto para comparación)

---

## 1. Motivación

`SMCForgeSilverBullet.py` está **técnicamente sano** (ft-audit: sin lookahead, callbacks correctos, `atr_14` existe) pero **NO es un ICT Silver Bullet fiel**. El review SMC encontró 4 divergencias del canon que invalidan el backtest como test del modelo SB:

1. **🔴 Bug DST.** `_KZ` (L192-197) hardcodea `EST+5=UTC` todo el año. US opera DST (`EDT=UTC-4`) ~8 meses → la ventana NYAM real (10-11 ET) queda corrida ~1h medio año. Crítico para stocks (subyacente anclado al reloj US con DST).
2. **🔴 Killzone ≠ ventana SB.** Usa sesiones amplias (NYAM 09:30-11:00, LON 02:00-05:00, NYPM 13:30-16:00) en vez de las 3 ventanas canónicas de **1 hora** (London 03-04, AM 10-11, PM 14-15 ET). Diluye el time-boxing que ES el edge del SB.
3. **⚠️ Secuencia no causalmente ordenada.** `rolling(lookback).max()` independiente para sweep y CHoCH → no garantiza `sweep < MSS`, ni que el FVG nazca del displacement. Permite OB/breakers (SB es FVG-only).
4. **🔴 A 1h/4h el SB no existe.** Los resultados positivos (PF 1.08-1.18) tenían holds de 15-22h = swing overnight, no scalp de sesión. Validaban algo que no es SB.

**Backtest de contexto (14 stocks tokenizados líquidos, lev1, sep25-may26):**

| TF | trades | WR | Profit | PF | Sharpe |
|----|--------|-----|--------|------|--------|
| 5m | 431 | 39.4% | −3.14% | 0.71 | −4.32 |
| 15m | 732 | 38.1% | −4.09% | 0.83 | −3.63 |
| 1h | 322 | 42.9% | +1.19% | 1.08 | 0.71 |
| 4h | 77 | 35.1% | +1.52% | 1.18 | 0.47 |

Eje dominante = frecuencia/fees. Pero todos corrieron con las ventanas DST-corridas y la secuencia laxa → ninguno es un test limpio del SB.

---

## 2. Objetivos / No-objetivos

**Objetivos**
- Implementar el ICT Silver Bullet **canónico** sobre acciones tokenizadas Bitget, ejecutable y backtesteable sin ambigüedad.
- Corregir los 4 puntos de fidelidad ICT.
- Reusar el motor `SMC_Forge` validado (cero re-derivación de primitivas, cero riesgo de lookahead nuevo).

**No-objetivos**
- No tocar `SMCForgeSilverBullet.py` ni el motor `SMC_Forge/`.
- No optimizar (hyperopt) en esta fase — primero edge crudo a defaults sanos.
- No deployar a live. Solo backtest + forense.

---

## 3. Diseño

### 3.1 Archivo y reuso del motor
- Nuevo archivo `user_data/strategies/ICTSilverBullet.py`, clase `ICTSilverBullet`, `INTERFACE_VERSION = 3`, `can_short = True`.
- Importa de `SMC_Forge/` (mismo patrón `sys.path.insert` que el viejo): `SMCEngine` (swings, CHoCH, displacement, sweep events), `annotate_displacement` (`atr_14` + `*_disp`), `annotate_eqh_eql` (pools eqh/eql + swings para targets).
- **NO** se re-implementa detección de FVG/sweep/displacement.

### 3.2 Ventanas SB — DST-aware (fix central)
Reemplaza el `_KZ` UTC hardcodeado por conversión real a hora del Este:
```python
from zoneinfo import ZoneInfo
ET = ZoneInfo("America/New_York")
et = dataframe['date'].dt.tz_convert(ET)        # date es tz-aware UTC
hour, minute = et.dt.hour, et.dt.minute
in_am = (hour == 10)                            # 10:00–10:59 ET
in_pm = (hour == 14)                            # 14:00–14:59 ET
in_sb = in_am | in_pm
```
- DST se maneja solo (zoneinfo conoce las transiciones). Ventanas: **AM 10:00-11:00 ET**, **PM 14:00-15:00 ET**.
- Columna `sb_window_id` ∈ {`AM`, `PM`, `none`} y `sb_window_open` (True en la primera vela de cada ventana por día → resetea el state machine).
- London (03-04 ET) **excluido** (stocks ilíquidos, mercado US cerrado).

### 3.3 Secuencia de entrada — state machine causal (FVG-only)
Por par, loop sobre las barras (estilo kernels del motor) que **fuerza el orden** dentro de cada ventana. Long (short = espejo):

```
estado por ventana: {swept, sweep_low, mss_seen, fvg_top, fvg_bottom}
en sb_window_open[i]:  reset estado
si in_sb[i]:
  # 1) SWEEP sell-side (toma liquidez debajo) — el grab
  si (internal_sweep_bullish[i] o swing_sweep_bullish[i]):
      swept=True; sweep_low=low[i]
  # 2) MSS = CHoCH alcista + displacement >= disp_threshold, DESPUÉS del sweep
  si swept y (internal_choch_bullish[i] o swing_choch_bullish[i])
            y (choch_bull_disp[i] >= disp_threshold):
      mss_seen=True
      fvg_top=active_bullish_fvg_top[i]; fvg_bottom=active_bullish_fvg_bottom[i]   # el FVG del displacement
  # 3) ENTRY: retrace al FVG, DESPUÉS del MSS
  si mss_seen y fvg_top>0 y (low[i] <= fvg_top) y (close[i] >= fvg_bottom):
      enter_long[i]=1
      sb_sl_price[i] = sweep_low                 # SL detrás del sweep (3.4)
      reset estado                                # un setup por ventana
no in_sb[i]: estado queda inactivo (no entries fuera de ventana)
```
- Garantiza `sweep_bar < mss_bar ≤ entry_bar` y que el FVG sea **el del displacement** (capturado en el paso 2), no uno activo cualquiera.
- **FVG-only** (sin OB ni breakers).
- Emite columnas: `enter_long/enter_short`, `enter_tag` (`SB_AM`/`SB_PM`), `sb_sl_price`.
- **Filtro direccional opcional** (`use_macro_filter`, default True): `macro_bias` HTF 1h del `SMCEngine` (entra a favor del draw on liquidity). Bloquea long si bias<0, short si bias>0.

### 3.4 Salidas — híbrido
- **SL (custom_stoploss):** lee `sb_sl_price` de la fila de entrada → SL detrás del sweep + buffer `ATR×0.5`. Convertir con `stoploss_from_absolute(sl_price, current_rate, is_short, leverage=trade.leverage)`. Tras TP1 → **break-even** (+ buffer).
- **TP1 parcial (adjust_trade_position):** a `R = |entry − sl|` × `tp1_r_mult` (default 1.0R) o `ATR×tp1_atr_mult`; cierra `tp1_amount_pct` (default 50%); set `tp1_taken`.
- **TP2/final (custom_exit):** pool de **liquidez opuesta más cercano** por encima (long) / debajo (short) del entry — reusa columnas del motor (`eqh_level`/`eql_level`, `swing_high`/`swing_low`, OB opuesto), patrón `_structural_tp_price` del viejo.
- **Time-stop (custom_exit):** cierre forzado al **fin de la sesión US (16:00 ET)** del día de entrada. **NUNCA overnight.** (Mata el problema de holds 15-22h.)
- **Hard stop backstop:** `stoploss = -0.10` FIJO (sin escalar por leverage — corrige el `-0.10*lev` del viejo). El stop operativo es el custom.

### 3.5 Universo / TF / leverage / warmup
- **Universo:** 14 stocks tokenizados líquidos: MSTR, TSLA, NVDA, CRCL, AAPL, COIN, PLTR, META, GOOGL, HOOD, AMZN, MCD, BABA, MSFT (`/USDT:USDT`). Ampliable.
- **TF:** backtest en **5m y 15m** (comparar). En 5m la ventana = 12 velas; en 15m = 4 velas (justo).
- **Leverage:** configurable, **default 1** (backtest limpio).
- **startup_candle_count:** ~1000 (5m) para cubrir el `swing_length=50` del engine 1h en live.

---

## 4. Plan de validación (post-implementación) — gates de Sofia (quant)

**Test primario declarado A PRIORI:** `5m AM+PM`. `AM-only` y `15m` son **secundarios** (con corrección de múltiples comparaciones). `15m` se corre **solo como sanity de la state machine**, NO como test de edge (4 velas/ventana → muestra inviable).

1. **OOS reservado intocado:** apartar los **últimos ~2 meses** (o sep25) que NO entran en ninguna decisión hasta el veredicto final. Todo el diseño/tuning se hace sobre el resto.
2. **Backtest** 5m y 15m sobre los 14 stocks, ventanas DST-aware, lev1, rango (sin OOS).
3. **Umbral de avance (gate duro):** `≥150 trades` en 5m **Y** `PF>1 con borde inferior del CI >1.0` (no PF puntual). `<150 trades` ⇒ resultado **"no concluyente"**, NO "no viable" — no quemar la tesis por baja potencia.
4. Si pasa el gate → **ronda forense (bt-forensics):**
   - **Walk-forward** sobre el período no-OOS.
   - **Monte Carlo por bloques temporales** (los 14 pares NO son independientes — beta común cripto/macro → N efectivo < nominal; samplear bloques, no trades individuales).
   - **Ablación** SB-windows on/off + AM-only vs AM+PM (juzga si el edge viene del time-boxing o es incidental / si la tesis de sesión-en-tokenizado se sostiene).
   - **Corrección por múltiples comparaciones:** Deflated Sharpe / Bonferroni sobre las 4 celdas (TF × window-config). Sin esto, probar 4 configs infla el falso positivo.
5. **Veredicto final sobre el OOS reservado** (recién acá se toca). Solo si sobrevive → leverage modesto / hyperopt acotado / dry-run.

---

## 5. Riesgos
- **🔴 Sample size bajo (predecible, no "se ve después"):** time-boxing a 2h/día + secuencia estricta de 3 eventos ⇒ Sofia estima **~60-150 trades en 5m** y **~25-60 en 15m** sobre 7.5 meses. Mínimo para conclusión separable de PF=1 con WR~40%: **≥150-200 trades**. El **15m nace muerto estadísticamente** (solo sanity), el 5m queda borderline. Gate del §4.3 lo gobierna.
- **🔴 Data snooping / selección por backtest:** elegir **stocks DESPUÉS** de ver que crypto fallaba, + universo "14 líquidos" elegido a posteriori del exploratorio, es selección por backtest. Mitigación: **OOS reservado intocado** (§4.1) + **corrección de múltiples comparaciones** (§4.4). Sin esos dos, el resultado no es creíble por más PF>1 que dé.
- **🔴 Tesis de sesión en tokenizados — no demostrada:** el perp tokenizado Bitget **NO es el subyacente** — tiene su propio order flow, sin los MM/dark-pools que generan el sweep institucional del modelo ICT. Plausible pero sin demostrar. La ablación SB-on/off (§4.4) es quien la juzga, no el PF agregado.
- **15m con 4 velas/ventana:** la secuencia de 3 eventos casi no cabe → pocos trades. Por eso es sanity-only, no test de edge.

## 6. Preguntas abiertas
Ninguna — todas las decisiones de diseño resueltas con Walter (archivo nuevo + motor, AM+PM, 5m+15m, salidas híbridas).
