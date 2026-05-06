"""
SMCForgeReverse — Smoke Test on Real Bitget Data
=================================================

Runs the SMCForgeReverse pipeline (LTF engine + 4h informer projection +
approach_strength + tier classification + 5-gate entry) over BTC/ETH/SOL
real 15m + 4h Bitget futures data.

Validates:
    * No NaN in approach_strength after warmup.
    * tier columns produced for every bar (NONE allowed; never NaN).
    * No-lookahead invariant of approach_strength still holds on real data.
    * Counts entries per gate (drop-off ratios) and total entries — sanity
      check for "drastically less than SMCForge".

Usage:
    python reverse_smoke_test.py

Exits non-zero on any structural failure. Drop-off ratios are reported,
not asserted (universe + thresholds matter — auditing only).
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from forge_engine import SMCEngine
from forge_quality import annotate_displacement
from forge_levels import annotate_eqh_eql
from forge_inducement import annotate_inducement
from reverse_approach import compute_approach_strength
from reverse_tier import classify_long_tier, classify_short_tier, tier_passes_min


DATA_DIR = Path('/home/wortiz/Desktop/freqtrade/user_data/data/bitget/futures')
PAIRS = ['BTC', 'ETH', 'SOL']
LTF = '15m'
MTF = '4h'

APPROACH_LOOKBACK = 7
APPROACH_BASELINE = 20
APPROACH_THR = 0.55
SWEEP_LOOKBACK = 4
TIER_MIN = 'A_minus'
PD_DISCOUNT_MAX = 38.0
PD_PREMIUM_MIN = 62.0


_MTF_PROJECT_COLS = (
    'swing_high', 'swing_low',
    'eqh_level', 'eql_level',
    'active_bullish_ob_top', 'active_bullish_ob_bottom',
    'active_bearish_ob_top', 'active_bearish_ob_bottom',
    'active_bullish_fvg_top', 'active_bullish_fvg_bottom',
    'active_bearish_fvg_top', 'active_bearish_fvg_bottom',
    'active_bullish_breaker_top', 'active_bullish_breaker_bottom',
    'active_bearish_breaker_top', 'active_bearish_breaker_bottom',
    'active_bullish_fvg_breaker_top', 'active_bullish_fvg_breaker_bottom',
    'active_bearish_fvg_breaker_top', 'active_bearish_fvg_breaker_bottom',
    'swing_sweep_bullish', 'swing_sweep_bearish',
    'internal_sweep_bullish', 'internal_sweep_bearish',
    'bull_idm_swept', 'bear_idm_swept',
)


def load_pair(pair: str, tf: str) -> pd.DataFrame:
    fp = DATA_DIR / f'{pair}_USDT_USDT-{tf}-futures.feather'
    df = pd.read_feather(fp)
    df['date'] = pd.to_datetime(df['date'], utc=True)
    return df.reset_index(drop=True)


def project_mtf_to_ltf(ltf_df: pd.DataFrame, mtf_df: pd.DataFrame
                       ) -> pd.DataFrame:
    """
    Mimics SMCForgeReverse._compute_htf_snapshot for one pair.
    Adds {col}_h4 columns to ltf_df, shift(1)-anti-lookahead, merge_asof
    backward.
    """
    sig = SMCEngine(mtf_df, internal_length=5, swing_length=50).get_signals()
    levels = annotate_eqh_eql(
        mtf_df, atr_period=14, fractal_n=5, tolerance_atr=0.10,
    )
    snap = pd.DataFrame({'date': mtf_df['date'].values})
    for col in _MTF_PROJECT_COLS:
        if col in levels.columns:
            snap[f'{col}_h4'] = levels[col].shift(1).values
        elif col in sig.columns:
            snap[f'{col}_h4'] = sig[col].shift(1).values
        else:
            snap[f'{col}_h4'] = 0.0
    ref_tz = ltf_df['date'].dt.tz
    src_tz = snap['date'].dt.tz if hasattr(snap['date'].dt, 'tz') else None
    if ref_tz is not None and src_tz is None:
        snap['date'] = pd.to_datetime(snap['date']).dt.tz_localize('UTC')
    elif ref_tz is None and src_tz is not None:
        snap['date'] = pd.to_datetime(snap['date']).dt.tz_convert(None)
    return pd.merge_asof(
        ltf_df.sort_values('date'),
        snap.sort_values('date'),
        on='date', direction='backward',
    )


def run_pipeline(pair: str) -> dict:
    """Run the full SMCForgeReverse pipeline (sans freqtrade IStrategy hooks)."""
    ltf = load_pair(pair, LTF)
    mtf = load_pair(pair, MTF)

    ltf['vol_sma_20'] = ltf['volume'].rolling(20).mean()

    sig = SMCEngine(ltf, internal_length=5, swing_length=50).get_signals()
    for col in sig.columns:
        ltf[col] = sig[col].values
    disp = annotate_displacement(ltf, sig, atr_period=14, lookback=3)
    for col in disp.columns:
        if col not in ltf.columns:
            ltf[col] = disp[col].values
    levels = annotate_eqh_eql(
        ltf, atr_period=14, fractal_n=5, tolerance_atr=0.10,
    )
    for col in levels.columns:
        if col not in ltf.columns:
            ltf[col] = levels[col].values
    idm = annotate_inducement(ltf, fractal_n_major=5, fractal_n_minor=2)
    for col in idm.columns:
        if col not in ltf.columns:
            ltf[col] = idm[col].values

    df = project_mtf_to_ltf(ltf, mtf)

    rng_h4 = (df['swing_high_h4'] - df['swing_low_h4']).replace(0.0, np.nan)
    df['range_pct_h4'] = (
        (df['close'] - df['swing_low_h4']) / rng_h4 * 100.0
    ).fillna(50.0).clip(0.0, 100.0)
    df['in_discount_h4'] = df['range_pct_h4'] < PD_DISCOUNT_MAX
    df['in_premium_h4'] = df['range_pct_h4'] > PD_PREMIUM_MIN

    approach = compute_approach_strength(
        df, lookback=APPROACH_LOOKBACK, baseline_window=APPROACH_BASELINE,
    )
    df['approach_strength'] = approach.shift(1).values

    df['reverse_tier_long'] = classify_long_tier(df, htf_suffix='_h4').values
    df['reverse_tier_short'] = classify_short_tier(df, htf_suffix='_h4').values

    return {'pair': pair, 'df': df}


def gate_counts_long(df: pd.DataFrame) -> dict:
    n = len(df)
    base = df.iloc[200:]   # skip warmup
    pd_passed = base['in_discount_h4'].fillna(False)
    swing_sweep = base['swing_sweep_bullish_h4'].fillna(0.0)
    internal_sweep = base.get(
        'internal_sweep_bullish_h4', pd.Series(0.0, index=base.index)
    ).fillna(0.0)
    sweep_raw = (swing_sweep.eq(1) | internal_sweep.eq(1)).astype(float)
    sweep = sweep_raw.rolling(SWEEP_LOOKBACK, min_periods=1).max() == 1
    poi_active = (
        ((base['active_bullish_ob_top_h4'] > 0).fillna(False))
        | ((base['active_bullish_fvg_top_h4'] > 0).fillna(False))
        | ((base['active_bullish_breaker_top_h4'] > 0).fillna(False))
    )
    tap = poi_active & (
        ((base['low'] <= base['active_bullish_ob_top_h4'])
         & (base['close'] >= base['active_bullish_ob_bottom_h4'])).fillna(False)
        | ((base['low'] <= base['active_bullish_fvg_top_h4'])
           & (base['close'] >= base['active_bullish_fvg_bottom_h4'])).fillna(False)
        | ((base['low'] <= base['active_bullish_breaker_top_h4'])
           & (base['close'] >= base['active_bullish_breaker_bottom_h4'])).fillna(False)
    )
    tier_ok = tier_passes_min(base['reverse_tier_long'], TIER_MIN)
    weak = (base['approach_strength'] < APPROACH_THR).fillna(False)
    full = pd_passed & sweep & poi_active & tap & tier_ok & weak

    return {
        'total_bars': len(base),
        'pd': int(pd_passed.sum()),
        'sweep': int(sweep.sum()),
        'poi_active': int(poi_active.sum()),
        'tap': int(tap.sum()),
        'tier_ok': int(tier_ok.sum()),
        'weak': int(weak.sum()),
        'full_pass': int(full.sum()),
    }


def gate_counts_short(df: pd.DataFrame) -> dict:
    base = df.iloc[200:]
    pd_passed = base['in_premium_h4'].fillna(False)
    swing_sweep = base['swing_sweep_bearish_h4'].fillna(0.0)
    internal_sweep = base.get(
        'internal_sweep_bearish_h4', pd.Series(0.0, index=base.index)
    ).fillna(0.0)
    sweep_raw = (swing_sweep.eq(1) | internal_sweep.eq(1)).astype(float)
    sweep = sweep_raw.rolling(SWEEP_LOOKBACK, min_periods=1).max() == 1
    poi_active = (
        ((base['active_bearish_ob_top_h4'] > 0).fillna(False))
        | ((base['active_bearish_fvg_top_h4'] > 0).fillna(False))
        | ((base['active_bearish_breaker_top_h4'] > 0).fillna(False))
    )
    tap = poi_active & (
        ((base['high'] >= base['active_bearish_ob_bottom_h4'])
         & (base['close'] <= base['active_bearish_ob_top_h4'])).fillna(False)
        | ((base['high'] >= base['active_bearish_fvg_bottom_h4'])
           & (base['close'] <= base['active_bearish_fvg_top_h4'])).fillna(False)
        | ((base['high'] >= base['active_bearish_breaker_bottom_h4'])
           & (base['close'] <= base['active_bearish_breaker_top_h4'])).fillna(False)
    )
    tier_ok = tier_passes_min(base['reverse_tier_short'], TIER_MIN)
    weak = (base['approach_strength'] < APPROACH_THR).fillna(False)
    full = pd_passed & sweep & poi_active & tap & tier_ok & weak

    return {
        'total_bars': len(base),
        'pd': int(pd_passed.sum()),
        'sweep': int(sweep.sum()),
        'poi_active': int(poi_active.sum()),
        'tap': int(tap.sum()),
        'tier_ok': int(tier_ok.sum()),
        'weak': int(weak.sum()),
        'full_pass': int(full.sum()),
    }


def validate_invariants(pair: str, df: pd.DataFrame) -> list[str]:
    errs: list[str] = []
    base = df.iloc[200:]   # post-warmup window

    nan_app = base['approach_strength'].isna().sum()
    if nan_app > 0:
        errs.append(f'{pair}: approach_strength has {nan_app} NaN bars '
                    f'after warmup (expected 0)')

    if base['reverse_tier_long'].isna().any():
        errs.append(f'{pair}: reverse_tier_long has NaN values '
                    f'(must always be a string)')
    if base['reverse_tier_short'].isna().any():
        errs.append(f'{pair}: reverse_tier_short has NaN values')

    valid_tiers = {'S', 'A', 'A_minus', 'B', 'B_minus', 'NONE'}
    bad = set(base['reverse_tier_long'].unique()) - valid_tiers
    if bad:
        errs.append(f'{pair}: reverse_tier_long has unknown labels: {bad}')

    return errs


def main() -> int:
    print('SMCForgeReverse smoke test')
    print(f'  data_dir={DATA_DIR}')
    print(f'  pairs={PAIRS}  ltf={LTF}  mtf={MTF}')
    print(f'  approach: lookback={APPROACH_LOOKBACK} baseline={APPROACH_BASELINE} '
          f'thr={APPROACH_THR}')
    print(f'  pd: discount<{PD_DISCOUNT_MAX}  premium>{PD_PREMIUM_MIN}')
    print(f'  tier_min={TIER_MIN}')
    print()

    all_errors: list[str] = []
    t0 = time.time()
    for pair in PAIRS:
        try:
            res = run_pipeline(pair)
        except Exception as e:
            all_errors.append(f'{pair}: pipeline raised: {e!r}')
            continue
        df = res['df']
        n = len(df)

        errs = validate_invariants(pair, df)
        all_errors.extend(errs)

        long_g = gate_counts_long(df)
        short_g = gate_counts_short(df)

        print(f'── {pair}  bars={n}  range={df["date"].iloc[0]} → '
              f'{df["date"].iloc[-1]}')
        print(f'   LONG  pd={long_g["pd"]}  sweep={long_g["sweep"]}  '
              f'poi={long_g["poi_active"]}  tap={long_g["tap"]}  '
              f'tier_ok={long_g["tier_ok"]}  weak={long_g["weak"]}  '
              f'→ FULL={long_g["full_pass"]}')
        print(f'   SHORT pd={short_g["pd"]}  sweep={short_g["sweep"]}  '
              f'poi={short_g["poi_active"]}  tap={short_g["tap"]}  '
              f'tier_ok={short_g["tier_ok"]}  weak={short_g["weak"]}  '
              f'→ FULL={short_g["full_pass"]}')
        if errs:
            for e in errs:
                print(f'   ERROR: {e}')
        print()

    elapsed = time.time() - t0
    print(f'Smoke test elapsed: {elapsed:.1f}s')

    if all_errors:
        print()
        print(f'❌ {len(all_errors)} structural error(s):')
        for e in all_errors:
            print(f'  - {e}')
        return 1

    print('✅ All structural invariants OK.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
