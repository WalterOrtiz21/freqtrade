"""
Test LLM confluence filter against backtest cases.

Usage:
    python test_llm_filter.py [--model MODEL] [--cases N] [--zip PATH]

Defaults:
    model  = z-ai/glm-5-turbo
    cases  = 50 (25 winners + 25 losers, stratified)
    zip    = latest in user_data/backtest_results/

Outputs:
    - Accuracy global
    - Precision / Recall / F1 per class
    - Accuracy by subsetup (HTF_ALIGN, HTF_CONF, OTHER)
    - Profit preserved (sum of profit_pct of allowed trades vs total)
"""
import argparse
import io
import json
import os
import sys
import zipfile
from pathlib import Path

import joblib
import pandas as pd

# Load .env from freqtrade root
_env_path = Path(__file__).parents[3] / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

# Allow importing from parent dirs
sys.path.insert(0, str(Path(__file__).parents[3]))  # freqtrade root

from user_data.strategies.SMC.llm_filter import LLMConfluenceFilter


def load_latest_zip() -> Path:
    results_dir = Path(__file__).parents[3] / "user_data/backtest_results"
    zips = sorted(results_dir.glob("backtest-result-*.zip"))
    if not zips:
        raise FileNotFoundError("No backtest zip found in user_data/backtest_results/")
    return zips[-1]


def load_data(zip_path: Path):
    z = zipfile.ZipFile(zip_path)

    # Signals (entry candles with all indicators)
    pkl_files = [f for f in z.namelist() if "_signals.pkl" in f]
    if not pkl_files:
        raise FileNotFoundError("No _signals.pkl in zip")
    raw = z.read(pkl_files[0])
    signals_raw = joblib.load(io.BytesIO(raw))
    strategy = list(signals_raw.keys())[0]

    dfs = []
    for pair, df in signals_raw[strategy].items():
        df2 = df.copy()
        df2["pair"] = pair
        dfs.append(df2)
    signals = pd.concat(dfs, ignore_index=True)

    # Trade outcomes from JSON
    json_files = [f for f in z.namelist() if f.endswith(".json")
                  and "_meta" not in f and "_config" not in f
                  and "_SMC" not in f]
    stats = json.loads(z.read(json_files[0]))
    trades_raw = stats["strategy"][strategy].get("trades", [])
    trades = pd.DataFrame(trades_raw)[["pair", "open_date", "profit_ratio", "exit_reason", "enter_tag", "is_short"]]
    trades["open_date"] = pd.to_datetime(trades["open_date"], utc=True)
    trades["win"] = trades["profit_ratio"] > 0

    return signals, trades, strategy


def join_outcomes(signals: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    sig = signals.copy()
    sig["date"] = pd.to_datetime(sig["date"], utc=True)
    # Freqtrade opens the trade on the candle AFTER the signal candle.
    # Signals use 15m timeframe → signal date + 15min = trade open_date.
    sig["trade_open_date"] = sig["date"] + pd.Timedelta(minutes=15)
    merged = sig.merge(trades, left_on=["pair", "trade_open_date"], right_on=["pair", "open_date"], how="inner")
    return merged


def sample_cases(df: pd.DataFrame, n_winners: int, n_losers: int, seed: int = 42):
    winners = df[df["win"]].sample(min(n_winners, df["win"].sum()), random_state=seed)
    losers = df[~df["win"]].sample(min(n_losers, (~df["win"]).sum()), random_state=seed)
    return pd.concat([winners, losers]).sort_index()


def run(model: str, n_cases: int, zip_path: Path, verbose: bool = False, threshold: float = 0.6):
    print(f"\n{'='*70}")
    print(f"  LLM Filter Test — model={model}, cases={n_cases}")
    print(f"  Zip: {zip_path.name}")
    print(f"{'='*70}\n")

    signals, trades, strategy = load_data(zip_path)
    merged = join_outcomes(signals, trades)
    print(f"Joined {len(merged)} trades with signals\n")

    half = n_cases // 2
    cases = sample_cases(merged, n_winners=half, n_losers=n_cases - half)

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set")
    llm = LLMConfluenceFilter(api_key=api_key, model=model, confidence_threshold=threshold)

    results = []
    correct = 0
    for idx, (_, row) in enumerate(cases.iterrows(), 1):
        pair = row["pair"]
        direction = "SHORT" if row.get("is_short", False) else "LONG"
        actual_win = row["win"]
        profit_pct = row["profit_ratio"] * 100
        tag = row.get("enter_tag", "N/A")

        if verbose:
            print(f"\n{'─'*60}")
            print(f"  CASE {idx}/{len(cases)} — {pair} {direction}")
            print(f"  Actual: {'WIN' if actual_win else 'LOSS'} ({profit_pct:+.2f}%)")
            print(f"  Tag: {tag}")
            print(f"  Exit: {row.get('exit_reason', 'N/A')}")
            print(f"{'─'*60}")
            prompt = llm._build_prompt(pair, row, direction)
            print("\n--- PROMPT ---")
            print(prompt)

        confidence, reason = llm._evaluate_single(pair, row, direction)
        llm_conf = confidence if confidence is not None else 0.0
        llm_allow = llm_conf >= llm._threshold
        llm_reason = reason

        # Correct if: allow=True and win, OR allow=False and loss
        case_correct = (llm_allow == actual_win)
        correct += case_correct

        # Compact per-case line (always printed)
        verdict = "ALLOW" if llm_allow else "BLOCK"
        outcome = "WIN " if actual_win else "LOSS"
        mark = "✓" if case_correct else "✗"
        print(f"[{idx:>2}/{len(cases)}] {mark} {pair:<16} {direction:<5} "
              f"{outcome} {profit_pct:+6.2f}%  tag={tag:<22} "
              f"LLM={verdict} conf={llm_conf:.2f}")
        if verbose:
            print(f"       reason: {llm_reason}")

        results.append({
            "pair": pair,
            "direction": direction,
            "tag": row.get("enter_tag", ""),
            "actual_win": actual_win,
            "profit_pct": round(profit_pct, 2),
            "llm_allow": llm_allow,
            "llm_conf": round(llm_conf, 2),
            "correct": case_correct,
        })

    summary = pd.DataFrame(results)

    # Global accuracy
    n_total = len(summary)
    accuracy = correct / n_total if n_total else 0.0

    # Confusion matrix: treat WIN as positive class
    tp = int(((summary["llm_allow"] == True) & (summary["actual_win"] == True)).sum())
    fp = int(((summary["llm_allow"] == True) & (summary["actual_win"] == False)).sum())
    tn = int(((summary["llm_allow"] == False) & (summary["actual_win"] == False)).sum())
    fn = int(((summary["llm_allow"] == False) & (summary["actual_win"] == True)).sum())

    def _f1(p: float, r: float) -> float:
        return (2 * p * r) / (p + r) if (p + r) > 0 else 0.0

    win_prec = tp / (tp + fp) if (tp + fp) else 0.0
    win_rec = tp / (tp + fn) if (tp + fn) else 0.0
    win_f1 = _f1(win_prec, win_rec)

    loss_prec = tn / (tn + fn) if (tn + fn) else 0.0
    loss_rec = tn / (tn + fp) if (tn + fp) else 0.0
    loss_f1 = _f1(loss_prec, loss_rec)

    # Profit preserved — filter point of view
    total_profit = summary["profit_pct"].sum()
    allowed_profit = summary.loc[summary["llm_allow"], "profit_pct"].sum()
    blocked_profit = summary.loc[~summary["llm_allow"], "profit_pct"].sum()

    # Accuracy by subsetup
    def _subsetup(tag: str) -> str:
        if "HTF_ALIGN" in tag:
            return "HTF_ALIGN"
        if "HTF_CONF" in tag:
            return "HTF_CONF"
        return "OTHER"

    summary["subsetup"] = summary["tag"].astype(str).map(_subsetup)
    by_sub = summary.groupby("subsetup").agg(
        n=("correct", "size"),
        acc=("correct", "mean"),
        wr=("actual_win", "mean"),
        allowed=("llm_allow", "sum"),
        profit_total=("profit_pct", "sum"),
    ).round(3)

    print(f"\n{'='*70}")
    print(f"  GLOBAL SUMMARY")
    print(f"{'='*70}")
    print(f"  Accuracy:        {correct}/{n_total} = {accuracy*100:.1f}%")
    print(f"  Baseline (50%):  {'BEAT' if accuracy > 0.50 else 'MISSED'} by {(accuracy-0.5)*100:+.1f}pp")
    print(f"  Target (60%):    {'BEAT' if accuracy >= 0.60 else 'MISSED'} by {(accuracy-0.6)*100:+.1f}pp")
    print()
    print(f"  Confusion matrix (WIN as positive):")
    print(f"              predicted_WIN   predicted_LOSS")
    print(f"  actual_WIN      {tp:3d}             {fn:3d}")
    print(f"  actual_LOSS     {fp:3d}             {tn:3d}")
    print()
    print(f"  WIN  class: precision={win_prec:.2f}  recall={win_rec:.2f}  F1={win_f1:.2f}")
    print(f"  LOSS class: precision={loss_prec:.2f}  recall={loss_rec:.2f}  F1={loss_f1:.2f}")
    print()
    print(f"  Profit total:    {total_profit:+.2f}%")
    print(f"  Profit allowed:  {allowed_profit:+.2f}%  ({len(summary[summary['llm_allow']])} trades)")
    print(f"  Profit blocked:  {blocked_profit:+.2f}%  ({len(summary[~summary['llm_allow']])} trades)")
    print()
    print(f"  BY SUBSETUP:")
    print(by_sub.to_string())
    print(f"{'='*70}\n")

    print(summary[["pair", "direction", "tag", "actual_win", "profit_pct",
                   "llm_allow", "llm_conf", "correct"]].to_string(index=False))

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="z-ai/glm-5-turbo")
    parser.add_argument("--cases", type=int, default=50)
    parser.add_argument("--zip", default=None)
    parser.add_argument("--verbose", action="store_true", help="Print full prompt per case")
    parser.add_argument("--threshold", type=float, default=0.6, help="LLM confidence threshold (default 0.6)")
    args = parser.parse_args()

    zip_path = Path(args.zip) if args.zip else load_latest_zip()
    run(model=args.model, n_cases=args.cases, zip_path=zip_path, verbose=args.verbose, threshold=args.threshold)


if __name__ == "__main__":
    main()
