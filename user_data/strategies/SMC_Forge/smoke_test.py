"""
SMC_Forge — Smoke Test on Real Bitget Data
==========================================

Runs all four forge layers (engine + quality + levels + inducement) over
freshly-downloaded BTC, ETH, SOL 15m futures data and validates structural
invariants that MUST hold on any real market.

Usage:
    python smoke_test.py

Exits non-zero on any invariant violation.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Allow imports from this directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

from forge_engine import SMCEngine
from forge_quality import annotate_displacement
from forge_levels import annotate_eqh_eql
from forge_inducement import annotate_inducement


DATA_DIR = Path('/home/wortiz/Desktop/freqtrade/user_data/data/bitget/futures')
PAIRS = ['BTC', 'ETH', 'SOL']
TIMEFRAME = '15m'

# Reasonable upper bound for ANY invariant violation — if we exceed,
# something is wrong with the algo, not the market.
MAX_VIOLATIONS = 0


# =============================================================================
# Loaders
# =============================================================================

def load_pair(pair: str) -> pd.DataFrame:
    fp = DATA_DIR / f'{pair}_USDT_USDT-{TIMEFRAME}-futures.feather'
    df = pd.read_feather(fp)
    df['date'] = pd.to_datetime(df['date'], utc=True)
    return df.reset_index(drop=True)


# =============================================================================
# Invariant checks
# =============================================================================

def check_data_continuity(pair: str, df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Returns (errors, warnings). Exchange-side outages (gaps > 1h) are
    warnings, not errors — they're real and not actionable from our side.
    Sub-hour gaps would suggest a download bug.
    """
    errs: list[str] = []
    warns: list[str] = []
    diffs = df['date'].diff().dropna()
    expected = pd.Timedelta('15min')

    # Sub-hour gaps = local issue (download / parse bug)
    suspect = diffs[(diffs > expected * 2) & (diffs <= pd.Timedelta('1h'))]
    if len(suspect) > 0:
        errs.append(f'{pair}: {len(suspect)} small gaps 30min-1h '
                     f'(largest = {suspect.max()})')

    # Multi-hour gaps = exchange outage, log but don't fail
    big = diffs[diffs > pd.Timedelta('1h')]
    if len(big) > 0:
        when = df.loc[big.index, 'date'].tolist()
        warns.append(
            f'{pair}: {len(big)} exchange-outage gap(s) '
            f'(largest = {big.max()}, at {when[0]})'
        )
    return errs, warns


def check_no_critical_nans(pair: str, df: pd.DataFrame) -> list[str]:
    errs = []
    for col in ['open', 'high', 'low', 'close', 'volume']:
        n = df[col].isna().sum()
        if n > 0:
            errs.append(f'{pair}: {col} has {n} NaN values')
    return errs


def check_engine_outputs_aligned(pair: str, df: pd.DataFrame,
                                  signals: pd.DataFrame) -> list[str]:
    errs = []
    if len(signals) != len(df):
        errs.append(f'{pair}: engine output len mismatch '
                     f'({len(signals)} vs {len(df)})')
    return errs


def check_displacement_invariants(pair: str, df: pd.DataFrame,
                                   signals: pd.DataFrame,
                                   disp: pd.DataFrame) -> list[str]:
    """
    Displacement scores must be:
      - non-negative
      - 0 on bars without the corresponding event
      - finite
    """
    errs = []
    disp_cols = [c for c in disp.columns if c.endswith('_disp')]

    for col in disp_cols:
        vals = disp[col]
        if (vals < 0).any():
            errs.append(f'{pair}: {col} has negative values '
                         f'({(vals < 0).sum()} bars)')
        if not np.isfinite(vals.fillna(0)).all():
            errs.append(f'{pair}: {col} has inf values')

    # Check that disp is 0 where the matching event flag is 0
    pairs_to_check = [
        ('internal_bos_bull_disp', 'internal_bos_bullish'),
        ('internal_bos_bear_disp', 'internal_bos_bearish'),
        ('internal_choch_bull_disp', 'internal_choch_bullish'),
        ('internal_choch_bear_disp', 'internal_choch_bearish'),
        ('swing_bos_bull_disp', 'swing_bos_bullish'),
        ('swing_bos_bear_disp', 'swing_bos_bearish'),
        ('swing_choch_bull_disp', 'swing_choch_bullish'),
        ('swing_choch_bear_disp', 'swing_choch_bearish'),
    ]
    for disp_col, event_col in pairs_to_check:
        no_event = signals[event_col] == 0
        nonzero_disp_no_event = (disp[disp_col] > 0) & no_event
        n = nonzero_disp_no_event.sum()
        if n > 0:
            errs.append(f'{pair}: {disp_col} > 0 on {n} bars without {event_col}')
    return errs


def check_levels_invariants(pair: str, df: pd.DataFrame,
                             levels: pd.DataFrame) -> list[str]:
    """
    EQH/EQL levels must:
      - lie within the historical price range
      - eqh_count == 0 ⟺ eqh_level is NaN
    """
    errs = []
    lo, hi = df['low'].min(), df['high'].max()

    for level_col, count_col in [('eqh_level', 'eqh_count'),
                                  ('eql_level', 'eql_count')]:
        levels_present = levels[level_col].dropna()
        out_of_range = ((levels_present < lo * 0.5) |
                        (levels_present > hi * 1.5))
        n = out_of_range.sum()
        if n > 0:
            errs.append(f'{pair}: {level_col} has {n} values outside '
                         f'sane price range [{lo*0.5:.2f}, {hi*1.5:.2f}]')

        # Coherence: count > 0 ⟺ level not NaN
        count_pos_no_level = ((levels[count_col] > 0) &
                              levels[level_col].isna()).sum()
        level_no_count = ((levels[count_col] == 0) &
                          levels[level_col].notna()).sum()
        if count_pos_no_level > 0:
            errs.append(f'{pair}: {count_col} > 0 with NaN {level_col} '
                         f'({count_pos_no_level} bars)')
        if level_no_count > 0:
            errs.append(f'{pair}: {count_col} == 0 with non-NaN {level_col} '
                         f'({level_no_count} bars)')

    # When count is set, must be >= 2 (single pivots aren't EQH)
    bad_eqh_count = ((levels['eqh_count'] > 0) & (levels['eqh_count'] < 2)).sum()
    bad_eql_count = ((levels['eql_count'] > 0) & (levels['eql_count'] < 2)).sum()
    if bad_eqh_count > 0:
        errs.append(f'{pair}: eqh_count == 1 on {bad_eqh_count} bars '
                     f'(should be 0 or >= 2)')
    if bad_eql_count > 0:
        errs.append(f'{pair}: eql_count == 1 on {bad_eql_count} bars')

    return errs


def check_inducement_invariants(pair: str, df: pd.DataFrame,
                                 idm: pd.DataFrame) -> list[str]:
    """
    IDM invariants:
      - bull_idm_swept[i] == 1  ⟹  low[i] < bull_idm_level[i] AND close[i] >= bull_idm_level[i]
      - bear_idm_swept[i] == 1  ⟹  high[i] > bear_idm_level[i] AND close[i] <= bear_idm_level[i]
      - sweep cannot fire when level is NaN
      - levels within sane price range
    """
    errs = []
    lo, hi = df['low'].min(), df['high'].max()

    # Sweep semantic checks
    bull_sw = idm['bull_idm_swept'] == 1
    if bull_sw.any():
        idx = bull_sw[bull_sw].index
        for i in idx:
            level = idm['bull_idm_level'].iloc[i]
            l = df['low'].iloc[i]
            c = df['close'].iloc[i]
            if pd.isna(level):
                errs.append(f'{pair}: bull_idm_swept=1 with NaN level at bar {i}')
                break
            if not (l < level and c >= level):
                errs.append(f'{pair}: bull_idm_swept invariant fail at bar {i} '
                             f'(low={l:.4f}, close={c:.4f}, level={level:.4f})')
                break

    bear_sw = idm['bear_idm_swept'] == 1
    if bear_sw.any():
        idx = bear_sw[bear_sw].index
        for i in idx:
            level = idm['bear_idm_level'].iloc[i]
            h = df['high'].iloc[i]
            c = df['close'].iloc[i]
            if pd.isna(level):
                errs.append(f'{pair}: bear_idm_swept=1 with NaN level at bar {i}')
                break
            if not (h > level and c <= level):
                errs.append(f'{pair}: bear_idm_swept invariant fail at bar {i} '
                             f'(high={h:.4f}, close={c:.4f}, level={level:.4f})')
                break

    # Sane range
    for col in ['bull_idm_level', 'bear_idm_level']:
        present = idm[col].dropna()
        out = ((present < lo * 0.5) | (present > hi * 1.5)).sum()
        if out > 0:
            errs.append(f'{pair}: {col} out of sane range ({out} bars)')

    return errs


# =============================================================================
# Reporting
# =============================================================================

def summarize(pair: str, df: pd.DataFrame, signals: pd.DataFrame,
              disp: pd.DataFrame, levels: pd.DataFrame,
              idm: pd.DataFrame, elapsed_s: float) -> str:
    lines = [f'\n=== {pair} ({len(df):,} bars, {elapsed_s:.2f}s) ===']
    lines.append(
        f'  range : {df["date"].iloc[0]}  →  {df["date"].iloc[-1]}'
    )
    lines.append(
        f'  price : low={df["low"].min():.2f}  high={df["high"].max():.2f}'
    )

    # Engine event counts
    lines.append('  engine events:')
    for col in ['internal_choch_bullish', 'internal_choch_bearish',
                'swing_choch_bullish', 'swing_choch_bearish',
                'internal_bos_bullish', 'internal_bos_bearish',
                'swing_bos_bullish', 'swing_bos_bearish']:
        lines.append(f'    {col:32s} = {int(signals[col].sum()):6d}')

    # Displacement averages on event bars
    lines.append('  displacement (mean score | events):')
    for evt, disp_col in [
        ('internal_choch_bull', 'internal_choch_bull_disp'),
        ('swing_choch_bull', 'swing_choch_bull_disp'),
        ('internal_choch_bear', 'internal_choch_bear_disp'),
        ('swing_choch_bear', 'swing_choch_bear_disp'),
    ]:
        nonzero = disp[disp_col][disp[disp_col] > 0]
        if len(nonzero) > 0:
            lines.append(f'    {evt:25s} mean={nonzero.mean():5.2f}  '
                          f'med={nonzero.median():5.2f}  '
                          f'p90={nonzero.quantile(0.90):5.2f}  '
                          f'n={len(nonzero)}')

    # Levels
    n_eqh_active = (levels['eqh_count'] >= 2).sum()
    n_eql_active = (levels['eql_count'] >= 2).sum()
    lines.append(f'  levels: eqh active on {n_eqh_active:6d} bars  |  '
                 f'eql active on {n_eql_active:6d} bars')

    # IDM
    bull_active = idm['bull_idm_level'].notna().sum()
    bear_active = idm['bear_idm_level'].notna().sum()
    bull_sweeps = int(idm['bull_idm_swept'].sum())
    bear_sweeps = int(idm['bear_idm_swept'].sum())
    lines.append(f'  IDM: bull active {bull_active:6d} bars  '
                 f'({bull_sweeps} sweeps) | '
                 f'bear active {bear_active:6d} bars '
                 f'({bear_sweeps} sweeps)')

    # Confluence preview: IDM swept + CHoCH bull with displacement >= 1.0 within 3 bars
    lookback = 3
    bull_recent_sweep = idm['bull_idm_swept'].rolling(lookback).max() == 1
    strong_choch = (
        (signals['internal_choch_bullish'] == 1) &
        (disp['internal_choch_bull_disp'] >= 1.0)
    )
    confluences_long = (bull_recent_sweep & strong_choch).sum()

    bear_recent_sweep = idm['bear_idm_swept'].rolling(lookback).max() == 1
    strong_choch_bear = (
        (signals['internal_choch_bearish'] == 1) &
        (disp['internal_choch_bear_disp'] >= 1.0)
    )
    confluences_short = (bear_recent_sweep & strong_choch_bear).sum()

    lines.append(f'  CONFLUENCE A+ (IDM sweep + CHoCH+disp ≥1.0 within {lookback} bars):')
    lines.append(f'    long candidates  = {confluences_long}')
    lines.append(f'    short candidates = {confluences_short}')

    return '\n'.join(lines)


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    print('SMC_Forge smoke test — real Bitget 15m data')
    print('=' * 72)

    all_errors: list[str] = []
    all_warnings: list[str] = []

    for pair in PAIRS:
        print(f'\n[{pair}] loading...', flush=True)
        df = load_pair(pair)

        errs, warns = check_data_continuity(pair, df)
        all_errors += errs
        all_warnings += warns
        all_errors += check_no_critical_nans(pair, df)

        t0 = time.time()
        eng = SMCEngine(df, internal_length=5, swing_length=50)
        signals = eng.get_signals()
        disp = annotate_displacement(df, signals)
        levels = annotate_eqh_eql(df, fractal_n=5, tolerance_atr=0.10)
        idm = annotate_inducement(df, fractal_n_major=5, fractal_n_minor=2)
        elapsed = time.time() - t0

        all_errors += check_engine_outputs_aligned(pair, df, signals)
        all_errors += check_displacement_invariants(pair, df, signals, disp)
        all_errors += check_levels_invariants(pair, df, levels)
        all_errors += check_inducement_invariants(pair, df, idm)

        print(summarize(pair, df, signals, disp, levels, idm, elapsed))

    print('\n' + '=' * 72)
    if all_warnings:
        print(f'!  {len(all_warnings)} warnings (non-fatal):')
        for w in all_warnings:
            print(f'   - {w}')

    if all_errors:
        print(f'\nFAIL  {len(all_errors)} INVARIANT VIOLATIONS:')
        for e in all_errors:
            print(f'   - {e}')
        return 1

    print('\nOK  all algorithmic invariants hold across BTC/ETH/SOL')
    return 0


if __name__ == '__main__':
    sys.exit(main())
