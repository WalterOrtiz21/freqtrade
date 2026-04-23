"""
Backtest Deep Analysis Script
==============================
Reads the latest (or specified) Freqtrade backtest ZIP and produces:
  - Per-pair breakdown (win rate, profit, avg duration)
  - Entry tag breakdown
  - Long vs Short comparison
  - Exit reason breakdown
  - Day-of-week patterns
  - Consecutive loss analysis
  - Pair category classification (BTC/ETH/large/mid/meme)

Usage:
    python analyze_backtest.py [path/to/backtest.zip]
    python analyze_backtest.py  # uses .last_result.json automatically
"""

import json
import sys
import zipfile
from pathlib import Path

import pandas as pd

pd.set_option("display.max_rows", 100)
pd.set_option("display.width", 120)
pd.set_option("display.float_format", "{:.2f}".format)

RESULTS_DIR = Path(__file__).parent.parent.parent / "backtest_results"

# --- Coin category mapping ---
LARGE_CAPS = {"BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE", "AVAX", "DOT", "LINK", "MATIC", "LTC"}
MID_CAPS = {"SUI", "APT", "ARB", "OP", "INJ", "TIA", "SEI", "JUP", "WIF", "PENDLE", "RUNE", "STX"}
MEMES = {"PEPE", "FLOKI", "BONK", "WIF", "POPCAT", "MOODENG", "NEIRO", "TURBO", "BRETT", "PIPPIN",
         "VIRTUAL", "MOG", "MEME", "BOME", "SLERF", "PNUT", "ACT", "GOAT", "AI16Z", "FARTCOIN"}


def get_coin(pair: str) -> str:
    return pair.split("/")[0]


def categorize(coin: str) -> str:
    if coin in LARGE_CAPS:
        return "Large Cap"
    if coin in MID_CAPS:
        return "Mid Cap"
    if coin in MEMES:
        return "Meme"
    return "Other"


def load_latest_zip() -> dict:
    last = RESULTS_DIR / ".last_result.json"
    if last.exists():
        name = json.loads(last.read_text())["latest_backtest"]
        zip_path = RESULTS_DIR / name
    else:
        zips = sorted(RESULTS_DIR.glob("*.zip"))
        if not zips:
            raise FileNotFoundError("No backtest zip found")
        zip_path = zips[-1]
    print(f"Loading: {zip_path.name}\n")
    return load_zip(zip_path)


def load_zip(path: Path) -> dict:
    with zipfile.ZipFile(path) as z:
        json_files = [f for f in z.namelist() if f.endswith(".json") and "config" not in f]
        with z.open(json_files[0]) as f:
            return json.load(f)


INDICATOR_COLS = [
    "bull_ob_0_rvol", "bear_ob_0_rvol",
    "bull_fvg_0_rvol", "bear_fvg_0_rvol",
    "bull_fvg_0_filled_pct", "bear_fvg_0_filled_pct",
    "bull_ob_0_touches", "bear_ob_0_touches",
    "n_active_bull_obs", "n_active_bear_obs",
    "htf_range_position_pct", "daily_range_position_pct",
]

ANALISIS_DIR = Path(__file__).parent / "analisis"
TF_MINUTES   = 15  # strategy timeframe


def _load_indicator_parquets() -> dict[str, pd.DataFrame]:
    """Load per-pair indicator parquets saved by export_indicator_data=True.
    Returns {pair: df} where df is indexed by date (UTC)."""
    result = {}
    for p in ANALISIS_DIR.glob("indicators_*.parquet"):
        # reverse safe_pair naming: _ back to / and :
        raw = p.stem.replace("indicators_", "")
        # e.g. BTC_USDT_USDT -> BTC/USDT:USDT
        parts = raw.split("_")
        if len(parts) == 3:
            pair = f"{parts[0]}/{parts[1]}:{parts[2]}"
        elif len(parts) == 2:
            pair = f"{parts[0]}/{parts[1]}"
        else:
            continue
        try:
            ind = pd.read_parquet(p)
            ind["date"] = pd.to_datetime(ind["date"], utc=True)
            ind = ind.set_index("date")
            result[pair] = ind
        except Exception:
            pass
    return result


def _merge_indicators(df: pd.DataFrame, ind_by_pair: dict) -> pd.DataFrame:
    """Join indicator snapshot onto trades.
    Trade open_date = candle i+1 (entry bar), so signal candle = open_date - TF.
    """
    if not ind_by_pair:
        return df
    tf_delta = pd.Timedelta(minutes=TF_MINUTES)
    new_cols = {c: [] for c in INDICATOR_COLS}
    open_dates = pd.to_datetime(df["open_date"], utc=True)
    for i, row in df.iterrows():
        pair    = row["pair"]
        sig_ts  = open_dates[i] - tf_delta
        ind_df  = ind_by_pair.get(pair)
        if ind_df is not None and sig_ts in ind_df.index:
            src = ind_df.loc[sig_ts]
            for c in INDICATOR_COLS:
                new_cols[c].append(src.get(c, float("nan")))
        else:
            for c in INDICATOR_COLS:
                new_cols[c].append(float("nan"))
    for c, vals in new_cols.items():
        df[c] = vals
    return df


def build_df(data: dict) -> tuple[pd.DataFrame, str]:
    strategy = list(data["strategy"].keys())[0]
    trades = data["strategy"][strategy].get("trades", [])
    df = pd.DataFrame(trades)
    df["open_date"] = pd.to_datetime(df["open_date"])
    df["close_date"] = pd.to_datetime(df["close_date"])
    df["profit_pct"] = df["profit_ratio"] * 100
    df["win"] = df["profit_ratio"] > 0
    df["coin"] = df["pair"].apply(get_coin)
    df["category"] = df["coin"].apply(categorize)
    df["direction"] = df["is_short"].apply(lambda x: "Short" if x else "Long")
    df["duration_h"] = df["trade_duration"] / 60
    df["weekday"] = df["open_date"].dt.day_name()

    # Merge per-pair indicator parquets (only if export_indicator_data was enabled)
    ind_by_pair = _load_indicator_parquets()
    if ind_by_pair:
        df = _merge_indicators(df, ind_by_pair)
        print(f"[indicators] loaded parquets for {len(ind_by_pair)} pairs")

    return df, strategy


def section(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def summary_table(df: pd.DataFrame, groupby: str, label: str = None):
    g = df.groupby(groupby).agg(
        Trades=("profit_ratio", "count"),
        Wins=("win", "sum"),
        Win_pct=("win", lambda x: round(x.mean() * 100, 1)),
        Avg_profit=("profit_pct", "mean"),
        Total_profit=("profit_pct", "sum"),
        Avg_duration_h=("duration_h", "mean"),
    ).sort_values("Total_profit", ascending=False)
    g["Loss_pct"] = 100 - g["Win_pct"]
    print(g.to_string())


class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

# -----------------------------------------------------------------------
def main():
    analisis_dir = Path(__file__).parent / "analisis"
    analisis_dir.mkdir(exist_ok=True)
    out_file = analisis_dir / "resultados_backtest.txt"
    sys.stdout = Tee(sys.stdout, open(out_file, "w", encoding="utf-8"))

    if len(sys.argv) > 1:
        data = load_zip(Path(sys.argv[1]))
    else:
        data = load_latest_zip()

    df, strategy = build_df(data)
    print(f"Strategy : {strategy}")
    print(f"Trades   : {len(df)}  |  Winners: {df['win'].sum()}  |  Losers: {(~df['win']).sum()}")
    print(f"Win rate : {df['win'].mean()*100:.1f}%")
    print(f"Avg profit/trade: {df['profit_pct'].mean():.2f}%")
    print(f"Total profit: {df['profit_pct'].sum():.2f}%")

    # ---- 1. Long vs Short ----
    section("1. LONG vs SHORT")
    summary_table(df, "direction")

    # ---- 2. By category ----
    section("2. BY COIN CATEGORY")
    summary_table(df, "category")

    # ---- 3. Per pair ----
    section("3. PER PAIR (sorted by total profit)")
    g = df.groupby("pair").agg(
        Trades=("profit_ratio", "count"),
        Win_pct=("win", lambda x: round(x.mean() * 100, 1)),
        Avg_profit=("profit_pct", "mean"),
        Total_profit=("profit_pct", "sum"),
        Best=("profit_pct", "max"),
        Worst=("profit_pct", "min"),
    ).sort_values("Total_profit", ascending=False)
    print(g.to_string())

    # ---- 4. Entry tag ----
    section("4. ENTRY TAG BREAKDOWN")
    if "enter_tag" in df.columns and df["enter_tag"].notna().any():
        summary_table(df, "enter_tag")
    else:
        print("No enter_tag data found.")

    # ---- 5. Exit reason ----
    section("5. EXIT REASON")
    g = df.groupby("exit_reason").agg(
        Trades=("profit_ratio", "count"),
        Win_pct=("win", lambda x: round(x.mean() * 100, 1)),
        Avg_profit=("profit_pct", "mean"),
        Total_profit=("profit_pct", "sum"),
    ).sort_values("Trades", ascending=False)
    print(g.to_string())

    # ---- 6. Day of week ----
    section("6. DAY OF WEEK (entry day)")
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    g = df.groupby("weekday").agg(
        Trades=("profit_ratio", "count"),
        Win_pct=("win", lambda x: round(x.mean() * 100, 1)),
        Avg_profit=("profit_pct", "mean"),
        Total_profit=("profit_pct", "sum"),
    ).reindex([d for d in order if d in df["weekday"].unique()])
    print(g.to_string())

    # ---- 7. Worst pairs detail ----
    section("7. WORST 10 PAIRS")
    worst = df[~df["win"]].groupby("pair").agg(
        Losses=("profit_ratio", "count"),
        Total_loss=("profit_pct", "sum"),
        Avg_loss=("profit_pct", "mean"),
        Worst=("profit_pct", "min"),
    ).sort_values("Total_loss").head(10)
    print(worst.to_string())

    # ---- 8. Best pairs detail ----
    section("8. BEST 10 PAIRS")
    best = df[df["win"]].groupby("pair").agg(
        Wins=("profit_ratio", "count"),
        Total_profit=("profit_pct", "sum"),
        Avg_profit=("profit_pct", "mean"),
        Best=("profit_pct", "max"),
    ).sort_values("Total_profit", ascending=False).head(10)
    print(best.to_string())

    # ---- 9. Consecutive losses ----
    section("9. CONSECUTIVE LOSS STREAKS")
    df_sorted = df.sort_values("open_date").reset_index(drop=True)
    streaks = []
    current = 0
    for win in df_sorted["win"]:
        if not win:
            current += 1
        else:
            if current > 0:
                streaks.append(current)
            current = 0
    if current > 0:
        streaks.append(current)
    if streaks:
        streaks_s = pd.Series(streaks)
        print(f"Max consecutive losses : {streaks_s.max()}")
        print(f"Avg streak length      : {streaks_s.mean():.1f}")
        print(f"Streaks >= 5 losses    : {(streaks_s >= 5).sum()}")
        print(f"Streaks >= 10 losses   : {(streaks_s >= 10).sum()}")
        print(f"\nStreak distribution:")
        print(streaks_s.value_counts().sort_index().to_string())

    # ---- 10. Category + direction cross ----
    section("10. CATEGORY x DIRECTION")
    g = df.groupby(["category", "direction"]).agg(
        Trades=("profit_ratio", "count"),
        Win_pct=("win", lambda x: round(x.mean() * 100, 1)),
        Total_profit=("profit_pct", "sum"),
    ).sort_values("Total_profit", ascending=False)
    print(g.to_string())

    # ---- 11. Duration analysis winners vs losers ----
    section("11. TRADE DURATION: Winners vs Losers")
    dur = df.groupby("win")["duration_h"].describe()
    dur.index = ["Losers", "Winners"]
    print(dur[["count", "mean", "min", "50%", "max"]].to_string())

    # ---- 12. Component breakdown ----
    section("12. ENTRY TAG COMPONENT ANALYSIS")

    COMPONENTS = {
        "zone_type":  ["OB", "FVG", "BRK", "FVGBRK"],
        "ob_quality": ["OB_FRESH", "OB_TOUCHED", "OB_MITIGATED"],
        "pd_zone":    ["PD_DEEP_DISC", "PD_DISC", "PD_NEUTRAL", "PD_PREM", "PD_DEEP_PREM"],
        "session":    ["KZ_LDN", "KZ_NY", "SESSION", "OFF"],
        "fvg_qual":   ["FVG_CLEAN", "FVG_PARTIAL", "FVG_STALE"],
        "structure":  ["SWEEP", "HTF_ALIGN", "HTF_PART", "HTF_CONF"],
        "weekday":    ["MON", "FRI"],
    }

    if "enter_tag" in df.columns and df["enter_tag"].notna().any():
        for group_name, components in COMPONENTS.items():
            print(f"\n--- {group_name.upper()} ---")
            rows = []
            for comp in components:
                mask = df["enter_tag"].str.contains(comp, na=False)
                subset = df[mask]
                complement = df[~mask]
                if len(subset) == 0:
                    continue
                rows.append({
                    "Component":      comp,
                    "Trades":         len(subset),
                    "Win%":           round(subset["win"].mean() * 100, 1),
                    "Avg_profit":     round(subset["profit_pct"].mean(), 2),
                    "Total_profit":   round(subset["profit_pct"].sum(), 2),
                    "Avg_duration_h": round(subset["duration_h"].mean(), 1),
                    "Win%_without":   round(complement["win"].mean() * 100, 1) if len(complement) > 0 else None,
                    "AvgP_without":   round(complement["profit_pct"].mean(), 2) if len(complement) > 0 else None,
                })
            if rows:
                print(pd.DataFrame(rows).set_index("Component").to_string())
            else:
                print("  (no trades matched any component)")
    else:
        print("No enter_tag data found.")

    # ---- 13. FVG_STALE deep dive ----
    section("13. FVG_STALE DEEP DIVE")

    if "enter_tag" in df.columns and df["enter_tag"].notna().any():

        stale_mask = df["enter_tag"].str.contains("FVG_STALE", na=False)
        stale = df[stale_mask].copy()
        not_stale = df[~stale_mask].copy()

        print(f"\nFVG_STALE trades : {len(stale)}")
        print(f"Other trades     : {len(not_stale)}")
        if len(stale) > 0 and len(not_stale) > 0:
            print(f"\nFVG_STALE  → Win% {stale['win'].mean()*100:.1f}%  Avg profit {stale['profit_pct'].mean():.2f}%")
            print(f"Others     → Win% {not_stale['win'].mean()*100:.1f}%  Avg profit {not_stale['profit_pct'].mean():.2f}%")

        if not stale.empty:
            # --- By pair ---
            print("\n-- FVG_STALE by pair --")
            g = stale.groupby("pair").agg(
                Trades=("profit_ratio", "count"),
                Win_pct=("win", lambda x: round(x.mean() * 100, 1)),
                Avg_profit=("profit_pct", "mean"),
                Total_profit=("profit_pct", "sum"),
                Best=("profit_pct", "max"),
                Worst=("profit_pct", "min"),
            ).sort_values("Total_profit", ascending=False)
            print(g.to_string())

            # --- By direction ---
            print("\n-- FVG_STALE by direction --")
            g = stale.groupby("direction").agg(
                Trades=("profit_ratio", "count"),
                Win_pct=("win", lambda x: round(x.mean() * 100, 1)),
                Avg_profit=("profit_pct", "mean"),
                Total_profit=("profit_pct", "sum"),
            )
            print(g.to_string())

            # --- By session component ---
            print("\n-- FVG_STALE by session component --")
            session_components = ["KZ_LDN", "KZ_NY", "SESSION", "OFF"]
            rows = []
            for comp in session_components:
                mask = stale["enter_tag"].str.contains(comp, na=False)
                subset = stale[mask]
                if len(subset) == 0:
                    continue
                rows.append({
                    "Session":       comp,
                    "Trades":        len(subset),
                    "Win%":          round(subset["win"].mean() * 100, 1),
                    "Avg_profit":    round(subset["profit_pct"].mean(), 2),
                    "Total_profit":  round(subset["profit_pct"].sum(), 2),
                    "Avg_duration_h":round(subset["duration_h"].mean(), 1),
                })
            if rows:
                print(pd.DataFrame(rows).set_index("Session").to_string())

            # --- By HTF component ---
            print("\n-- FVG_STALE by HTF component --")
            htf_components = ["HTF_ALIGN", "HTF_PART", "HTF_CONF"]
            rows = []
            for comp in htf_components:
                mask = stale["enter_tag"].str.contains(comp, na=False)
                subset = stale[mask]
                if len(subset) == 0:
                    continue
                rows.append({
                    "HTF":           comp,
                    "Trades":        len(subset),
                    "Win%":          round(subset["win"].mean() * 100, 1),
                    "Avg_profit":    round(subset["profit_pct"].mean(), 2),
                    "Total_profit":  round(subset["profit_pct"].sum(), 2),
                    "Avg_duration_h":round(subset["duration_h"].mean(), 1),
                })
            if rows:
                print(pd.DataFrame(rows).set_index("HTF").to_string())

            # --- By PD zone ---
            print("\n-- FVG_STALE by Premium/Discount zone --")
            pd_components = ["PD_DEEP_DISC", "PD_DISC", "PD_NEUTRAL", "PD_PREM", "PD_DEEP_PREM"]
            rows = []
            for comp in pd_components:
                mask = stale["enter_tag"].str.contains(comp, na=False)
                subset = stale[mask]
                if len(subset) == 0:
                    continue
                rows.append({
                    "PD_Zone":       comp,
                    "Trades":        len(subset),
                    "Win%":          round(subset["win"].mean() * 100, 1),
                    "Avg_profit":    round(subset["profit_pct"].mean(), 2),
                    "Total_profit":  round(subset["profit_pct"].sum(), 2),
                })
            if rows:
                print(pd.DataFrame(rows).set_index("PD_Zone").to_string())

            # --- By zone type (what else is in the tag) ---
            print("\n-- FVG_STALE by zone type combination --")
            zone_components = ["OB", "FVG", "BRK", "FVGBRK"]
            rows = []
            for comp in zone_components:
                mask = stale["enter_tag"].str.contains(comp, na=False)
                subset = stale[mask]
                if len(subset) == 0:
                    continue
                rows.append({
                    "Zone":          comp,
                    "Trades":        len(subset),
                    "Win%":          round(subset["win"].mean() * 100, 1),
                    "Avg_profit":    round(subset["profit_pct"].mean(), 2),
                    "Total_profit":  round(subset["profit_pct"].sum(), 2),
                })
            if rows:
                print(pd.DataFrame(rows).set_index("Zone").to_string())

            # --- By weekday ---
            print("\n-- FVG_STALE by weekday --")
            order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            g = stale.groupby("weekday").agg(
                Trades=("profit_ratio", "count"),
                Win_pct=("win", lambda x: round(x.mean() * 100, 1)),
                Avg_profit=("profit_pct", "mean"),
                Total_profit=("profit_pct", "sum"),
            ).reindex([d for d in order if d in stale["weekday"].unique()])
            print(g.to_string())

            # --- Exit reason ---
            print("\n-- FVG_STALE exit reason --")
            g = stale.groupby("exit_reason").agg(
                Trades=("profit_ratio", "count"),
                Win_pct=("win", lambda x: round(x.mean() * 100, 1)),
                Avg_profit=("profit_pct", "mean"),
                Total_profit=("profit_pct", "sum"),
            ).sort_values("Trades", ascending=False)
            print(g.to_string())

            # --- Duration distribution ---
            print("\n-- FVG_STALE duration: winners vs losers --")
            dur = stale.groupby("win")["duration_h"].describe()
            dur.index = ["Losers", "Winners"][:len(dur)]
            print(dur[["count", "mean", "min", "50%", "max"]].to_string())

            # --- Concentration check: top 5 pairs contribution ---
            print("\n-- FVG_STALE concentration: top 5 pairs vs rest --")
            pair_profit = stale.groupby("pair")["profit_pct"].sum().sort_values(ascending=False)
            top5 = pair_profit.head(5)
            rest = pair_profit.iloc[5:]
            print(f"Top 5 pairs total profit : {top5.sum():.2f}%  ({len(top5)} pairs)")
            print(f"Remaining pairs total    : {rest.sum():.2f}%  ({len(rest)} pairs)")
            print(f"\nTop 5 pairs:")
            print(top5.to_string())

    # ---- 14. Indicator quantile analysis (requires export_indicator_data=True) ----
    ind_cols_present = [c for c in INDICATOR_COLS if c in df.columns and df[c].notna().any()]
    if ind_cols_present:
        section("14. INDICATOR QUANTILE ANALYSIS (export_indicator_data)")
        print("Columns with data:", ind_cols_present)

        for col in ind_cols_present:
            sub = df[df[col].notna() & (df[col] > 0)].copy()
            if len(sub) < 10:
                continue
            sub["q"] = pd.qcut(sub[col], q=4, duplicates="drop", labels=False)
            g = sub.groupby("q").agg(
                Trades=("profit_ratio", "count"),
                Win_pct=("win",        lambda x: round(x.mean() * 100, 1)),
                Avg_profit=("profit_pct", "mean"),
            )
            g.index = [f"Q{i+1}" for i in g.index]
            print(f"\n-- {col} (Q1=low … Q4=high) --")
            print(g.to_string())
    else:
        section("14. INDICATOR QUANTILE ANALYSIS (export_indicator_data)")
        print("No indicator parquet data found.")
        print("Set export_indicator_data=true in SMCWithMLLuxAlgo.json and re-run backtest.")

    print("\n" + "=" * 70)
    print("  Done.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
