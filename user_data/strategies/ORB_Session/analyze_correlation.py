"""
Diversification check: correlate ORBSession equity curve vs SMC_Forge equity curve.

Usage:
    python user_data/strategies/ORB_Session/analyze_correlation.py \\
        --orb user_data/backtest_results/orb_april2026.json \\
        --smc user_data/backtest_results/smc_april2026.json

Verdict (spec §9):
    |corr| < 0.4         -> GREEN: diversification holds
    0.4 <= |corr| < 0.6  -> YELLOW: weak diversification
    |corr| >= 0.6        -> RED: diversification thesis fails
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def equity_curve_from_backtest(json_path: Path, strategy_name: str) -> pd.Series:
    """Daily equity curve from a freqtrade backtest result JSON."""
    data = json.loads(json_path.read_text())
    trades = data['strategy'][strategy_name]['trades']
    if not trades:
        raise SystemExit(f'no trades in {json_path}')

    df = pd.DataFrame(trades)
    df['close_date'] = pd.to_datetime(df['close_date'])
    df = df.sort_values('close_date')
    df['date'] = df['close_date'].dt.tz_localize(None).dt.normalize()
    return df.groupby('date')['profit_abs'].sum().cumsum()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--orb', required=True, type=Path)
    ap.add_argument('--smc', required=True, type=Path)
    ap.add_argument('--orb-strategy-name', default='ORBSession')
    ap.add_argument('--smc-strategy-name', default='SMCForge')
    args = ap.parse_args()

    orb = equity_curve_from_backtest(args.orb, args.orb_strategy_name)
    smc = equity_curve_from_backtest(args.smc, args.smc_strategy_name)

    aligned = pd.DataFrame({'orb': orb, 'smc': smc}).fillna(method='ffill').dropna()
    returns = aligned.diff().dropna()

    if len(returns) < 5:
        raise SystemExit(f'not enough overlapping days ({len(returns)})')

    corr = returns['orb'].corr(returns['smc'])
    abs_corr = abs(corr)

    print(f'Overlapping days: {len(returns)}')
    print(f'Pearson correlation (daily returns): {corr:+.3f}')
    print(f'|corr| = {abs_corr:.3f}')
    if abs_corr < 0.4:
        verdict = 'GREEN -- diversification holds (spec §9 target)'
    elif abs_corr < 0.6:
        verdict = 'YELLOW -- weak diversification, evaluate operational complexity'
    else:
        verdict = 'RED -- diversification thesis fails (spec §9)'
    print(f'Verdict: {verdict}')


if __name__ == '__main__':
    main()
