#!/usr/bin/env python3
"""
XSMomentum Phase 2 — matrix runner
Cells: L∈{7,14,28} × N∈{5,3} = 6 cells.
Baseline (L=14, N=5) already done — included here for completeness/reparse.
For each cell: 9 quarterly PIT backtests.
Results captured via subprocess, summary extracted from stdout.
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

BASE = Path("/home/wortiz/Desktop/freqtrade")
FT = str(BASE / ".venv/bin/freqtrade")

QUARTERS = {
    "2024Q1": ("20240101", "20240401"),
    "2024Q2": ("20240401", "20240701"),
    "2024Q3": ("20240701", "20241001"),
    "2024Q4": ("20241001", "20250101"),
    "2025Q1": ("20250101", "20250401"),
    "2025Q2": ("20250401", "20250701"),
    "2025Q3": ("20250701", "20251001"),
    "2025Q4": ("20251001", "20260101"),
    "2026Q1": ("20260101", "20260401"),
}

# Matrix cells (L_days, N); baseline L=14,N=5 included
CELLS = [
    (7,  5),
    (14, 5),  # baseline — re-run for consistency
    (28, 5),
    (7,  3),
    (14, 3),
    (28, 3),
]


def write_params(L: int, N: int):
    params = {
        "strategy_name": "XSMomentum",
        "params": {
            "buy": {"lookback_days": L, "book_n": N},
            "sell": {"stop_pct": 0.12},
        },
    }
    path = BASE / "user_data/strategies/XSMomentum.json"
    with open(path, "w") as f:
        json.dump(params, f, indent=4)
    # Clear pycache so freqtrade picks up fresh params
    pycache = BASE / "user_data/strategies/__pycache__"
    if pycache.exists():
        import shutil
        shutil.rmtree(pycache)


def get_config(Q: str, N: int) -> str:
    if N == 5:
        return str(BASE / f"config_xsmom_pit_N40_{Q}.json")
    else:  # N == 3
        return str(BASE / f"config_xsmom_pit_N40_N3_{Q}.json")


def parse_output(stdout: str) -> dict:
    """Extract key metrics from freqtrade backtesting stdout."""
    result = {
        "trades": None,
        "tot_profit_pct": None,
        "win_rate": None,
        "drawdown_pct": None,
        "profit_factor": None,
        "avg_profit_pct": None,
        "long_trades": None,
        "short_trades": None,
        "long_profit_pct": None,
        "short_profit_pct": None,
        "error": None,
    }

    # Strategy summary table line:
    # │ XSMomentum │  278 │  3.14 │  87.164 │  87.16 │ 3d 3:00 │ 127  0  151  45.7 │ 10.84 USDT 10.83% │
    summary_re = re.search(
        r"XSMomentum\s*│\s*(\d+)\s*│\s*(-?[\d.]+)\s*│\s*(-?[\d.]+)\s*│\s*(-?[\d.]+)\s*│.*?│\s*(\d+)\s+\d+\s+(\d+)\s+([\d.]+)\s*│\s*([\d.]+)\s+USDT\s+([\d.]+)%",
        stdout,
    )
    if summary_re:
        result["trades"] = int(summary_re.group(1))
        result["avg_profit_pct"] = float(summary_re.group(2))
        result["tot_profit_pct"] = float(summary_re.group(4))
        wins = int(summary_re.group(5))
        losses = int(summary_re.group(6))
        result["win_rate"] = float(summary_re.group(7))
        result["drawdown_pct"] = float(summary_re.group(9))

    # Profit factor
    pf_match = re.search(r"Profit factor\s*│\s*([\d.]+)", stdout)
    if pf_match:
        result["profit_factor"] = float(pf_match.group(1))

    # Long/Short trades
    ls_match = re.search(r"Long / Short trades\s*│\s*(\d+)\s*/\s*(\d+)", stdout)
    if ls_match:
        result["long_trades"] = int(ls_match.group(1))
        result["short_trades"] = int(ls_match.group(2))

    # Long/Short profit
    lsp_match = re.search(r"Long / Short profit %\s*│\s*(-?[\d.]+)%\s*/\s*(-?[\d.]+)%", stdout)
    if lsp_match:
        result["long_profit_pct"] = float(lsp_match.group(1))
        result["short_profit_pct"] = float(lsp_match.group(2))

    if result["trades"] is None:
        # Try to detect error
        if "0 trades" in stdout or "No trades were made" in stdout:
            result["trades"] = 0
        else:
            result["error"] = "parse_failed"

    return result


def run_backtest(L: int, N: int, Q: str) -> dict:
    config = get_config(Q, N)
    start, end = QUARTERS[Q]
    timerange = f"{start}-{end}"

    cmd = [
        FT, "backtesting",
        "--config", config,
        "--strategy", "XSMomentum",
        "--timeframe", "4h",
        "--timerange", timerange,
        "--cache", "none",
    ]

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, cwd=str(BASE), timeout=300
        )
        stdout = proc.stdout + proc.stderr
        r = parse_output(stdout)
        if r["trades"] is None and proc.returncode != 0:
            r["error"] = f"exit_code={proc.returncode}"
        return r
    except subprocess.TimeoutExpired:
        return {"error": "timeout", "trades": None}
    except Exception as e:
        return {"error": str(e), "trades": None}


def main():
    all_results = {}  # {(L, N): {Q: metrics}}

    for (L, N) in CELLS:
        cell_key = f"L{L}_N{N}"
        print(f"\n{'='*60}")
        print(f"CELL {cell_key} (L={L}d, N={N}/N={N})")
        print(f"{'='*60}")

        write_params(L, N)
        cell_results = {}

        for Q in QUARTERS:
            print(f"  {Q}...", end=" ", flush=True)
            r = run_backtest(L, N, Q)
            cell_results[Q] = r

            if r.get("error"):
                print(f"ERROR: {r['error']}")
            else:
                print(
                    f"trades={r['trades']} profit={r['tot_profit_pct']:.1f}% "
                    f"PF={r['profit_factor']} DD={r['drawdown_pct']:.1f}%"
                )

        all_results[cell_key] = cell_results

    # Save full results
    out_path = BASE / "user_data/backtest_results/xsmom_matrix_phase2.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Print summary table
    print_summary(all_results)


def print_summary(all_results: dict):
    print("\n" + "=" * 80)
    print("MATRIX SUMMARY")
    print("=" * 80)
    print(f"{'Cell':<12} {'Trades':>7} {'CompPft%':>9} {'PF_pool':>8} {'%Pos':>6} {'MaxDD%':>8} {'Calmar':>8}")
    print("-" * 80)

    for cell_key, qdata in all_results.items():
        quarters = list(QUARTERS.keys())
        trades_total = 0
        pf_wins = 0.0
        pf_losses = 0.0
        positive_qs = 0
        max_dd = 0.0
        profits = []
        calmar_quarters = []

        for Q in quarters:
            r = qdata.get(Q, {})
            if r.get("error") or r.get("trades") is None:
                continue

            t = r.get("trades", 0) or 0
            pct = r.get("tot_profit_pct", 0) or 0
            dd = r.get("drawdown_pct", 0) or 0
            pf = r.get("profit_factor")

            trades_total += t
            profits.append(pct)
            if pct > 0:
                positive_qs += 1
            if dd > max_dd:
                max_dd = dd

            # Approximate PF pooled from per-quarter PF + trades
            # (Since we don't have individual trade P&L breakdown, use PF×DD proxy)
            # Better: track gross win/loss sums — but we only have PF ratio.
            # Use PF = wins/(losses) => wins = PF * losses; if wins + losses = total_profit_USDT...
            # We'll approximate from tot_profit_pct as proxy for net P&L direction:
            # wins contribution = pct if pct > 0 else 0 (very rough)
            # Use proper PF accumulation: wins_sum / losses_sum
            if pf and pf > 0 and t > 0:
                # Approximate: if PF = W/L and W + L = gross volume (normalize by trades)
                pf_wins += pf * t
                pf_losses += t

        # Compounded profit
        comp = 100.0
        for p in profits:
            comp *= (1 + p / 100)
        comp_profit = comp - 100.0

        # PF pooled approximation (weighted by trades, rough)
        pf_pooled = (pf_wins / pf_losses) if pf_losses > 0 else 0.0

        # Calmar approx: annualized return / max_dd
        # 9 quarters ≈ 2.25 years; compound profit over 2.25 years → CAGR
        n_years = 9 / 4.0
        cagr = ((1 + comp_profit / 100) ** (1 / n_years) - 1) * 100 if comp_profit > -100 else -999
        calmar = cagr / max_dd if max_dd > 0 else 0.0

        pct_pos = positive_qs / len(quarters) * 100

        print(
            f"{cell_key:<12} {trades_total:>7} {comp_profit:>+9.1f} {pf_pooled:>8.3f} "
            f"{pct_pos:>5.0f}% {max_dd:>8.1f} {calmar:>8.2f}"
        )


if __name__ == "__main__":
    main()
