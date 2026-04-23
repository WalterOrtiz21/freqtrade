"""
Statistical feature analysis for SMC backtest trades.

Joins backtest trades with signal-candle features, computes additional
derived features from OHLC history (entry candle structure, CHoCH
displacement, inducement sweeps, zone rejection, draw to opposite zone),
and ranks every feature by its ability to discriminate winners from losers.

Usage:
    python analyze_features.py [--zip PATH] [--output CSV]
"""
import argparse
import io
import json
import os
import sys
import warnings
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_classif

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Allow importing from parent dirs (same pattern as test_llm_filter.py)
sys.path.insert(0, str(Path(__file__).parents[3]))


# ==============================================================================
#  Data loading
# ==============================================================================

def load_latest_zip() -> Path:
    results_dir = Path(__file__).parents[3] / "user_data/backtest_results"
    zips = sorted(results_dir.glob("backtest-result-*.zip"))
    if not zips:
        raise FileNotFoundError("No backtest zip found")
    return zips[-1]


def load_data(zip_path: Path):
    """Return (signals_by_pair, trades_df, strategy_name)."""
    z = zipfile.ZipFile(zip_path)

    pkl_files = [f for f in z.namelist() if "_signals.pkl" in f]
    if not pkl_files:
        raise FileNotFoundError("No _signals.pkl in zip")
    signals_raw = joblib.load(io.BytesIO(z.read(pkl_files[0])))
    strategy = list(signals_raw.keys())[0]
    signals_by_pair = signals_raw[strategy]

    json_files = [f for f in z.namelist() if f.endswith(".json")
                  and "_meta" not in f and "_config" not in f
                  and "_SMC" not in f]
    stats_data = json.loads(z.read(json_files[0]))
    trades_raw = stats_data["strategy"][strategy].get("trades", [])
    trades = pd.DataFrame(trades_raw)[
        ["pair", "open_date", "profit_ratio", "exit_reason", "enter_tag", "is_short"]
    ]
    trades["open_date"] = pd.to_datetime(trades["open_date"], utc=True)
    trades["win"] = trades["profit_ratio"] > 0

    return signals_by_pair, trades, strategy


# ==============================================================================
#  Derived features (computed post-hoc from OHLC + existing columns)
# ==============================================================================

def _safe_div(a, b, default=0.0):
    return a / b if b not in (0, 0.0) else default


def compute_entry_candle_features(row: pd.Series) -> dict:
    o, h, l, c = row["open"], row["high"], row["low"], row["close"]
    rng = max(h - l, 1e-12)
    body = abs(c - o)
    upper = h - max(o, c)
    lower = min(o, c) - l
    return {
        "entry_body_pct": body / rng,
        "entry_upper_wick_pct": upper / rng,
        "entry_lower_wick_pct": lower / rng,
        "entry_is_bullish": int(c > o),
    }


def compute_choch_displacement(pair_df: pd.DataFrame, signal_idx: int,
                                is_short: bool) -> dict:
    """
    From bars_since_{swing,internal}_choch_{bull,bear} walk back to the CHoCH
    bar and measure how far price moved in setup direction between the CHoCH
    and the extreme reached before the pullback.
    """
    if is_short:
        bs_col = "bars_since_internal_choch_bear"
    else:
        bs_col = "bars_since_internal_choch_bull"

    if bs_col not in pair_df.columns:
        return {"choch_displacement_pct": 0.0, "choch_displacement_atr": 0.0}

    bs = int(pair_df.iloc[signal_idx].get(bs_col, 999))
    if bs >= 400 or bs < 1:
        return {"choch_displacement_pct": 0.0, "choch_displacement_atr": 0.0}

    choch_bar = signal_idx - bs
    if choch_bar < 0:
        return {"choch_displacement_pct": 0.0, "choch_displacement_atr": 0.0}

    segment = pair_df.iloc[choch_bar:signal_idx + 1]
    choch_price = pair_df.iloc[choch_bar]["close"]

    if is_short:
        extreme = segment["low"].min()
        displacement = (choch_price - extreme) / choch_price
    else:
        extreme = segment["high"].max()
        displacement = (extreme - choch_price) / choch_price

    # Normalize by ATR at CHoCH time (use a small rolling proxy on the segment)
    hl = (segment["high"] - segment["low"]).mean()
    atr_norm = _safe_div(displacement, hl / choch_price, 0.0)

    return {
        "choch_displacement_pct": max(displacement, 0.0),
        "choch_displacement_atr": max(atr_norm, 0.0),
    }


def compute_inducement_sweep(pair_df: pd.DataFrame, signal_idx: int,
                              is_short: bool, lookback_zone: int = 6,
                              lookback_pivot: int = 20) -> dict:
    """
    Inducement = in the last `lookback_zone` bars, did price briefly poke above
    (for SHORT) a local high of the previous `lookback_pivot` bars and close back
    below it within 2 bars?  (For LONG: mirror image of local low.)
    """
    if signal_idx < lookback_zone + lookback_pivot:
        return {"inducement_swept_recent": 0, "inducement_magnitude_pct": 0.0}

    zone = pair_df.iloc[signal_idx - lookback_zone:signal_idx + 1]
    magnitude = 0.0
    swept = 0
    for i in range(len(zone) - 2):
        bar = zone.iloc[i]
        pivot_window = pair_df.iloc[
            signal_idx - lookback_zone - lookback_pivot + i:signal_idx - lookback_zone + i
        ]
        if pivot_window.empty:
            continue
        if is_short:
            prior_high = pivot_window["high"].max()
            if bar["high"] > prior_high > 0:
                nxt = zone.iloc[i + 1:i + 3]
                if (nxt["close"] < prior_high).any():
                    swept = 1
                    magnitude = max(magnitude,
                                    (bar["high"] - prior_high) / prior_high)
        else:
            prior_low = pivot_window["low"].min()
            if bar["low"] < prior_low and prior_low > 0:
                nxt = zone.iloc[i + 1:i + 3]
                if (nxt["close"] > prior_low).any():
                    swept = 1
                    magnitude = max(magnitude,
                                    (prior_low - bar["low"]) / prior_low)
    return {"inducement_swept_recent": swept,
            "inducement_magnitude_pct": magnitude}


def compute_zone_rejection(pair_df: pd.DataFrame, signal_idx: int,
                            is_short: bool) -> dict:
    """
    For the last 1-3 bars inside the active zone, measure how much of the range
    is wick on the rejection side.  Uses the entry bar and up to 2 prior bars.
    """
    bars = pair_df.iloc[max(0, signal_idx - 2):signal_idx + 1]
    wicks = []
    for _, bar in bars.iterrows():
        rng = max(bar["high"] - bar["low"], 1e-12)
        if is_short:
            w = (bar["high"] - max(bar["open"], bar["close"])) / rng
        else:
            w = (min(bar["open"], bar["close"]) - bar["low"]) / rng
        wicks.append(w)
    return {"zone_rejection_strength": float(np.mean(wicks))} if wicks else \
        {"zone_rejection_strength": 0.0}


def compute_draw_to_opposite(row: pd.Series, is_short: bool) -> dict:
    """
    Distance from close to nearest opposite-direction active zone.
    For a SHORT: look for nearest bullish OB/FVG/Breaker BELOW (target).
    For a LONG: nearest bearish zone ABOVE (target).
    """
    close = row.get("close", 0.0)
    if close <= 0:
        return {"draw_pct_to_opposite_zone": 0.0}

    if is_short:
        candidates = [
            row.get("active_bullish_ob_top", 0.0),
            row.get("active_bullish_fvg_top", 0.0),
            row.get("active_bullish_breaker_top", 0.0),
            row.get("active_bullish_fvg_breaker_top", 0.0),
        ]
        # target is below price → level < close, want largest (closest from below)
        below = [lvl for lvl in candidates if 0 < lvl < close]
        if not below:
            return {"draw_pct_to_opposite_zone": 0.0}
        nearest = max(below)
        return {"draw_pct_to_opposite_zone": (close - nearest) / close}
    else:
        candidates = [
            row.get("active_bearish_ob_bottom", 0.0),
            row.get("active_bearish_fvg_bottom", 0.0),
            row.get("active_bearish_breaker_bottom", 0.0),
            row.get("active_bearish_fvg_breaker_bottom", 0.0),
        ]
        above = [lvl for lvl in candidates if lvl > close > 0]
        if not above:
            return {"draw_pct_to_opposite_zone": 0.0}
        nearest = min(above)
        return {"draw_pct_to_opposite_zone": (nearest - close) / close}


def compute_bars_since_sweep(pair_df: pd.DataFrame, signal_idx: int,
                              is_short: bool, lookback: int = 40) -> dict:
    """Bars since last internal_sweep in setup direction."""
    col = "internal_sweep_bearish" if is_short else "internal_sweep_bullish"
    if col not in pair_df.columns:
        return {"bars_since_last_sweep_same_dir": 999}
    start = max(0, signal_idx - lookback)
    window = pair_df.iloc[start:signal_idx + 1][col].values
    if window.sum() == 0:
        return {"bars_since_last_sweep_same_dir": 999}
    last_pos = np.where(window == 1)[0][-1]
    return {"bars_since_last_sweep_same_dir": len(window) - 1 - last_pos}


def compute_volume_spike(pair_df: pd.DataFrame, signal_idx: int,
                          is_short: bool) -> dict:
    """Volume at CHoCH bar relative to 20-bar SMA preceding it."""
    bs_col = "bars_since_internal_choch_bear" if is_short \
        else "bars_since_internal_choch_bull"
    if bs_col not in pair_df.columns:
        return {"volume_spike_at_choch": 1.0}
    bs = int(pair_df.iloc[signal_idx].get(bs_col, 999))
    if bs >= 400 or bs < 1 or signal_idx - bs < 20:
        return {"volume_spike_at_choch": 1.0}
    choch_bar = signal_idx - bs
    sma = pair_df.iloc[choch_bar - 20:choch_bar]["volume"].mean()
    if sma <= 0:
        return {"volume_spike_at_choch": 1.0}
    return {"volume_spike_at_choch": float(pair_df.iloc[choch_bar]["volume"] / sma)}


# ==============================================================================
#  Build enriched trade dataframe
# ==============================================================================

def enrich_trades(signals_by_pair: dict, trades: pd.DataFrame) -> pd.DataFrame:
    """
    For each trade, locate its signal candle in the pair's dataframe and
    compute derived features from surrounding OHLC.
    """
    rows = []
    missing = 0

    for pair, pdf in signals_by_pair.items():
        if pdf.empty or "date" not in pdf.columns:
            continue
        pdf = pdf.copy().reset_index(drop=True)
        pdf["date"] = pd.to_datetime(pdf["date"], utc=True)
        # Trade opens on candle AFTER signal → signal_date + 15m = open_date
        pdf["trade_open_date"] = pdf["date"] + pd.Timedelta(minutes=15)
        idx_by_open = {t.to_pydatetime(): i
                       for i, t in enumerate(pdf["trade_open_date"])}

        pair_trades = trades[trades["pair"] == pair]
        for _, trade in pair_trades.iterrows():
            od = trade["open_date"].to_pydatetime()
            sig_idx = idx_by_open.get(od)
            if sig_idx is None:
                missing += 1
                continue
            sig = pdf.iloc[sig_idx]
            is_short = bool(trade.get("is_short", False))

            feat = dict(sig)
            feat.update(compute_entry_candle_features(sig))
            feat.update(compute_choch_displacement(pdf, sig_idx, is_short))
            feat.update(compute_inducement_sweep(pdf, sig_idx, is_short))
            feat.update(compute_zone_rejection(pdf, sig_idx, is_short))
            feat.update(compute_draw_to_opposite(sig, is_short))
            feat.update(compute_bars_since_sweep(pdf, sig_idx, is_short))
            feat.update(compute_volume_spike(pdf, sig_idx, is_short))
            feat["pair"] = pair
            feat["is_short"] = is_short
            feat["win"] = bool(trade["win"])
            feat["profit_pct"] = float(trade["profit_ratio"]) * 100
            feat["enter_tag"] = str(trade.get("enter_tag", ""))
            feat["exit_reason"] = str(trade.get("exit_reason", ""))
            # Direction-aware bars_since_choch (the one relevant for THIS trade)
            if is_short:
                feat["bars_since_int_choch_dir"] = int(
                    sig.get("bars_since_internal_choch_bear", 999))
                feat["bars_since_sw_choch_dir"] = int(
                    sig.get("bars_since_swing_choch_bear", 999))
            else:
                feat["bars_since_int_choch_dir"] = int(
                    sig.get("bars_since_internal_choch_bull", 999))
                feat["bars_since_sw_choch_dir"] = int(
                    sig.get("bars_since_swing_choch_bull", 999))
            rows.append(feat)

    if missing:
        print(f"WARNING: {missing} trades could not be matched to a signal bar")
    return pd.DataFrame(rows)


# ==============================================================================
#  Statistical ranking
# ==============================================================================

CANDIDATE_FEATURES = [
    # Derived (new)
    "entry_body_pct", "entry_upper_wick_pct", "entry_lower_wick_pct",
    "entry_is_bullish",
    "choch_displacement_pct", "choch_displacement_atr",
    "inducement_swept_recent", "inducement_magnitude_pct",
    "zone_rejection_strength",
    "draw_pct_to_opposite_zone",
    "bars_since_last_sweep_same_dir",
    "volume_spike_at_choch",
    "bars_since_int_choch_dir", "bars_since_sw_choch_dir",
    # Pre-existing numeric
    "atr_pct_rank",
    "swing_range_position_pct", "htf_range_position_pct",
    "daily_range_position_pct", "weekly_range_position_pct",
    "bull_ob_0_rvol", "bear_ob_0_rvol",
    "bull_ob_0_age", "bear_ob_0_age",
    "bull_ob_0_touches", "bear_ob_0_touches",
    "bull_ob_0_mitigated", "bear_ob_0_mitigated",
    "bull_fvg_0_rvol", "bear_fvg_0_rvol",
    "bull_fvg_0_filled_pct", "bear_fvg_0_filled_pct",
    "n_active_bull_obs", "n_active_bear_obs",
    "n_active_bull_fvgs", "n_active_bear_fvgs",
    # Pre-existing booleans
    "above_pdh", "below_pdl",
    "in_kill_zone", "is_monday", "is_friday",
    "internal_sweep_bullish", "internal_sweep_bearish",
    "swing_sweep_bullish", "swing_sweep_bearish",
    "btc_macro_bias",
]


def rank_features(df: pd.DataFrame, out_csv: Path,
                   features: list[str] | None = None) -> pd.DataFrame:
    features = features or CANDIDATE_FEATURES
    available = [f for f in features if f in df.columns]
    missing = [f for f in features if f not in df.columns]
    if missing:
        print(f"  (skipping missing: {missing})")

    X = df[available].astype(float).fillna(0).values
    y = df["win"].astype(int).values

    mi = mutual_info_classif(X, y, random_state=42)

    rows = []
    for i, feat in enumerate(available):
        vals = df[feat].astype(float).values
        unique = np.unique(vals)
        if len(unique) <= 4:
            # categorical: winrate per value
            sub = df.groupby(feat)["win"].agg(["mean", "count"])
            lift_top = sub["mean"].max() / max(y.mean(), 1e-9)
            lift_bot = sub["mean"].min() / max(y.mean(), 1e-9)
            try:
                tbl = pd.crosstab(df[feat], df["win"])
                _, pval, _, _ = stats.chi2_contingency(tbl)
            except Exception:
                pval = 1.0
            n_top = int(sub["count"].max())
            n_bot = int(sub["count"].min())
        else:
            try:
                quintiles = pd.qcut(vals, 5, labels=False, duplicates="drop")
            except ValueError:
                continue
            sub = pd.DataFrame({"q": quintiles, "win": y})
            agg = sub.groupby("q")["win"].agg(["mean", "count"])
            if agg.empty:
                continue
            lift_top = agg["mean"].max() / max(y.mean(), 1e-9)
            lift_bot = agg["mean"].min() / max(y.mean(), 1e-9)
            try:
                _, pval = stats.ttest_ind(
                    vals[y == 1], vals[y == 0], equal_var=False
                )
            except Exception:
                pval = 1.0
            n_top = int(agg["count"].iloc[-1])
            n_bot = int(agg["count"].iloc[0])

        rows.append({
            "feature": feat,
            "mutual_info": round(mi[i], 4),
            "lift_top": round(float(lift_top), 3),
            "lift_bot": round(float(lift_bot), 3),
            "lift_spread": round(float(abs(lift_top - lift_bot)), 3),
            "p_value": round(float(pval), 4),
            "n_top_bin": n_top,
            "n_bot_bin": n_bot,
        })

    out = pd.DataFrame(rows).sort_values("mutual_info", ascending=False)
    out.to_csv(out_csv, index=False)
    return out


def rank_by_subsetup(df: pd.DataFrame, out_csv: Path) -> pd.DataFrame:
    """Rank features separately within HTF_ALIGN vs HTF_CONF and long vs short."""
    subsets = {
        "HTF_ALIGN": df[df["enter_tag"].str.contains("HTF_ALIGN", na=False)],
        "HTF_CONF":  df[df["enter_tag"].str.contains("HTF_CONF", na=False)],
        "LONG":      df[~df["is_short"]],
        "SHORT":     df[df["is_short"]],
        "FVGBRK":    df[df["enter_tag"].str.contains("FVGBRK", na=False)],
        "OB_FRESH":  df[df["enter_tag"].str.contains("OB_FRESH", na=False)],
    }

    all_rows = []
    for name, sub in subsets.items():
        if len(sub) < 20:
            continue
        ranking = rank_features(
            sub, Path("/tmp/_ignore.csv"), features=CANDIDATE_FEATURES
        )
        ranking["subsetup"] = name
        ranking["n"] = len(sub)
        ranking["winrate"] = round(sub["win"].mean(), 3)
        all_rows.append(ranking)

    combined = pd.concat(all_rows, ignore_index=True)
    combined = combined[["subsetup", "feature", "n", "winrate",
                         "mutual_info", "lift_top", "lift_bot",
                         "lift_spread", "p_value"]]
    combined.to_csv(out_csv, index=False)
    return combined


# ==============================================================================
#  Main
# ==============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=None)
    ap.add_argument("--output", default="user_data/strategies/SMC/feature_analysis.csv")
    ap.add_argument("--subsetup-output",
                    default="user_data/strategies/SMC/feature_analysis_by_subsetup.csv")
    args = ap.parse_args()

    zip_path = Path(args.zip) if args.zip else load_latest_zip()
    print(f"Loading: {zip_path.name}")
    signals_by_pair, trades, strategy = load_data(zip_path)
    print(f"  strategy={strategy} | pairs={len(signals_by_pair)} | trades={len(trades)}")

    print("\nEnriching trades with derived OHLC features...")
    df = enrich_trades(signals_by_pair, trades)
    print(f"  enriched rows: {len(df)} | winrate: {df['win'].mean():.2%}")

    out_path = Path(args.output)
    sub_path = Path(args.subsetup_output)

    print(f"\nRanking features globally → {out_path}")
    global_ranking = rank_features(df, out_path)
    print(f"\nTop 15 by mutual information:")
    print(global_ranking.head(15).to_string(index=False))

    print(f"\nRanking features per subsetup → {sub_path}")
    sub_ranking = rank_by_subsetup(df, sub_path)
    print(f"  subsets analyzed: {sub_ranking['subsetup'].nunique()}")
    print(f"  top features per subsetup (MI >=0.01):")
    sig = sub_ranking[sub_ranking["mutual_info"] >= 0.01]
    print(sig.head(30).to_string(index=False))

    # Filter summary: features that pass gating criteria
    print(f"\n{'='*70}")
    print(f"  FEATURES PASSING FILTER (MI>=0.01 AND lift_spread>=0.25)")
    print(f"{'='*70}")
    passers = global_ranking[
        (global_ranking["mutual_info"] >= 0.01)
        & (global_ranking["lift_spread"] >= 0.25)
    ]
    print(passers.to_string(index=False))


if __name__ == "__main__":
    main()
