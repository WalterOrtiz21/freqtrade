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


# -----------------------------------------------------------------------
def main():
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

    print("\n" + "=" * 70)
    print("  Done.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
