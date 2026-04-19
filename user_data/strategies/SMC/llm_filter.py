"""
LLM Confluence Filter for SMC Strategy via OpenRouter API.

Evaluates SMC entry signals using an external LLM as a confluence filter.
Only activates in dry-run and live modes (disabled for backtest/hyperopt).

Usage:
    from llm_filter import LLMConfluenceFilter

    filter = LLMConfluenceFilter(
        api_key=os.environ["OPENROUTER_API_KEY"],
        model="deepseek/deepseek-chat-v3-0324:free",
        confidence_threshold=0.6,
    )
    long_ok, short_ok, stats = filter.evaluate_entries(pair, dataframe, long_mask, short_mask)
"""

import json
import logging
import os
import time
from datetime import datetime, timezone

import pandas as pd
import requests

logger = logging.getLogger(__name__)

API_URL = "https://openrouter.ai/api/v1/chat/completions"

SYSTEM_PROMPT = (
    "You are an SMC filter calibrated on 358 backtested trades (baseline winrate 50.3%).\n"
    "Score each signal 0.0-1.0 using ONLY the empirical rules below, derived from that backtest.\n"
    "\n"
    "SETUP:\n"
    "Continuation pullback after CHoCH. Entry fires on the retracement candle (1-10 bars after CHoCH), not on CHoCH itself.\n"
    "CHoCH flags will read 0 at entry — this is expected. The enter_tag already confirms CHoCH occurred.\n"
    "\n"
    "EMPIRICAL RULES (WR = winrate in that subset):\n"
    "\n"
    "[HTF_ALIGN subsetup — baseline 50.6%, n=154]\n"
    "  + bars_since_swing_choch 0-3  → 64% WR  (prime freshness, +14pp)\n"
    "  - bars_since_swing_choch 4-7  → 40% WR  (edge decayed, -11pp)\n"
    "  + bars_since_internal_choch 0-2 → 70% WR  (fresh internal break)\n"
    "  - bars_since_internal_choch ≥6  → 36% WR  (stale internal, -15pp)\n"
    "  - SHORT + btc_macro_bias = -1  → 35% WR  (BTC already bearish, reversion risk, -15pp)\n"
    "  - is_monday = True             → 36% WR  (-15pp vs Tue-Sun 55%)\n"
    "\n"
    "[HTF_CONF subsetup — baseline 48.2%, n=199]\n"
    "  ++ in_kill_zone = True          → 73% WR  (strongest single edge, n=15, +25pp)\n"
    "  +  weekly_range_position 20-40%  → 62% WR  (+14pp)\n"
    "  -  weekly_range_position <20%    → 30% WR  (-18pp)\n"
    "  -  SHORT + bear_fvg_filled >20%  → 31% WR  (stale FVG, -17pp)\n"
    "  +  SHORT + bear_fvg_filled <20%  → 52% WR\n"
    "  -  SHORT + bear_ob_age >80 bars  → 35% WR  (stale OB, -13pp)\n"
    "\n"
    "[GLOBAL — all subsetups]\n"
    "  -- is_monday + SHORT            → 26% WR  (worst cell, -24pp)\n"
    "  -- is_monday + FVGBRK tag       → 26% WR  (-22pp)\n"
    "  ++ in_kill_zone + FVGBRK tag    → 79% WR  (+31pp)\n"
    "  +  weekly_range_position 20-40% → 62% WR  (+12pp global)\n"
    "  -  weekly_range_position <20%   → 31% WR  (-19pp global)\n"
    "\n"
    "FEATURES WITH NO PROVEN SIGNAL (IGNORE if cited in the input):\n"
    "  atr_pct_rank, OTE/Fibonacci, equilibrium levels, FVG-in-OB flag,\n"
    "  zone density counts, volume_spike_at_choch, wick/body ratios,\n"
    "  draw_to_opposite_zone, inducement sweep, daily/HTF/swing P/D labels,\n"
    "  PDH/PDL/PWH/PWL distances, choch_displacement.\n"
    "Do NOT invent justifications from these. If a rule above does not apply, stay near baseline 0.50.\n"
    "\n"
    "SCORING GUIDE:\n"
    "  0.80-1.00 : 2+ strong positive rules hit (e.g. HTF_CONF + in_kill_zone, or HTF_ALIGN + bars_since_choch ≤3 with no adverse BTC/Monday)\n"
    "  0.60-0.79 : 1 strong positive rule hit, no severe negatives\n"
    "  0.40-0.59 : no clear rule applies, or mixed (1 positive + 1 negative) — baseline territory\n"
    "  0.20-0.39 : 1 severe negative rule (stale CHoCH, Monday short, weekly Q1, stale FVG/OB)\n"
    "  0.00-0.19 : 2+ severe negatives stacked (e.g. Monday + bear BTC + stale CHoCH)\n"
    "\n"
    'Respond ONLY with valid JSON: {"confidence": <0.0-1.0>, "reason": "<which empirical rules drove the score>"}\n'
    "No markdown fences, no other text."
)


class LLMConfluenceFilter:
    """LLM-based confluence filter for SMC entries via OpenRouter API."""

    FALLBACK_MODEL = "openai/gpt-oss-20b:free"

    def __init__(
        self,
        api_key: str,
        model: str = "z-ai/glm-5-turbo",
        confidence_threshold: float = 0.6,
        timeout: int = 25,
        max_retries: int = 2,
        cache_ttl: int = 900,
        shadow_mode: bool = False,
        log_dir: str = "",
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._threshold = confidence_threshold
        self._timeout = timeout
        self._max_retries = max_retries
        self._cache_ttl = cache_ttl
        self._cache: dict[str, tuple[float, float]] = {}
        self._shadow_mode = shadow_mode
        self._log_path = ""
        if shadow_mode and log_dir:
            os.makedirs(log_dir, exist_ok=True)
            date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
            self._log_path = os.path.join(log_dir, f"llm_shadow_{date_str}.jsonl")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate_entries(
        self,
        pair: str,
        dataframe: pd.DataFrame,
        long_mask: pd.Series,
        short_mask: pd.Series,
    ) -> tuple[pd.Series, pd.Series, dict]:
        """
        Evaluate SMC entry signals with LLM confluence.

        In shadow mode: logs LLM verdict but never blocks entries.
        In filter mode: blocks entries below threshold.

        Returns
        -------
        long_ok, short_ok : pd.Series[bool]
            True where LLM approves the entry (or on error — never blocks).
        stats : dict
            Counts of blocked/bypassed/cached/shadow calls.
        """
        long_ok = pd.Series(True, index=dataframe.index)
        short_ok = pd.Series(True, index=dataframe.index)
        stats = {"blocked_long": 0, "blocked_short": 0, "errors": 0, "cached": 0,
                 "shadow_logged": 0}

        for idx in dataframe.index[long_mask]:
            row = dataframe.loc[idx]
            result, reason = self._evaluate_single(pair, row, "LONG")
            if result is None:
                stats["errors"] += 1
                continue
            would_block = result < self._threshold
            if would_block:
                stats["blocked_long"] += 1
            if self._shadow_mode:
                self._log_shadow(pair, row, "LONG", result, reason, would_block)
                stats["shadow_logged"] += 1
            elif would_block:
                long_ok.at[idx] = False

        for idx in dataframe.index[short_mask]:
            row = dataframe.loc[idx]
            result, reason = self._evaluate_single(pair, row, "SHORT")
            if result is None:
                stats["errors"] += 1
                continue
            would_block = result < self._threshold
            if would_block:
                stats["blocked_short"] += 1
            if self._shadow_mode:
                self._log_shadow(pair, row, "SHORT", result, reason, would_block)
                stats["shadow_logged"] += 1
            elif would_block:
                short_ok.at[idx] = False

        return long_ok, short_ok, stats

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _evaluate_single(self, pair: str, row: pd.Series, direction: str) -> tuple[float | None, str]:
        """Returns (confidence 0-1, reason) or (None, '') on error."""
        cache_key = self._get_cache_key(pair, row, direction)
        cached = self._check_cache(cache_key)
        if cached is not None:
            return cached, ""

        prompt = self._build_prompt(pair, row, direction)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        response = self._call_api(messages)
        if response is None:
            return None, ""

        parsed = self._parse_response(response)
        if parsed is None:
            return None, ""

        self._store_cache(cache_key, parsed["confidence"])
        logger.info(
            f"LLM ({pair} {direction}): confidence={parsed['confidence']:.2f} "
            f"| {parsed['reason'][:100]}"
        )
        return parsed["confidence"], parsed["reason"]

    def _log_shadow(
        self, pair: str, row: pd.Series, direction: str,
        confidence: float, reason: str, would_block: bool,
    ) -> None:
        """Append LLM evaluation to JSONL log (shadow mode)."""
        if not self._log_path:
            return
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "pair": pair,
            "direction": direction,
            "price": self._safe(row, "close", 0.0),
            "confidence": round(confidence, 3),
            "reason": reason[:200],
            "would_block": would_block,
            "enter_tag": str(row.get("enter_tag", "")),
            "model": self._model,
            "threshold": self._threshold,
        }
        try:
            with open(self._log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.warning(f"LLM shadow log write error: {e}")

    def _build_prompt(self, pair: str, row: pd.Series, direction: str) -> str:
        """Build compact prompt with only statistically-validated features."""
        enter_tag = str(row.get("enter_tag", ""))

        # Subsetup classification (drives which rule block applies)
        if "HTF_ALIGN" in enter_tag:
            subsetup = "HTF_ALIGN"
        elif "HTF_CONF" in enter_tag:
            subsetup = "HTF_CONF"
        else:
            subsetup = "OTHER"

        # Freshness — bars since CHoCH in setup direction
        if direction == "LONG":
            bs_int_choch = int(self._safe(row, "bars_since_internal_choch_bull", 999))
            bs_sw_choch = int(self._safe(row, "bars_since_swing_choch_bull", 999))
        else:
            bs_int_choch = int(self._safe(row, "bars_since_internal_choch_bear", 999))
            bs_sw_choch = int(self._safe(row, "bars_since_swing_choch_bear", 999))

        # Weekly range position (only HTF timeframe with proven signal)
        weekly_range_pct = self._safe(row, "weekly_range_position_pct", 50.0)

        # Context
        in_kz = bool(self._safe(row, "in_kill_zone", 0.0))
        btc_macro = int(self._safe(row, "btc_macro_bias", 0))
        session = str(row.get("session", ""))

        # is_monday — compute from row date if available
        is_monday = False
        date_val = row.get("date", None)
        if date_val is None and hasattr(row, "name"):
            date_val = row.name
        try:
            ts = pd.Timestamp(date_val) if date_val is not None else None
            if ts is not None and not pd.isna(ts):
                is_monday = ts.weekday() == 0
        except (ValueError, TypeError):
            pass

        # Zone staleness — only validated features
        bear_ob_age = self._safe(row, "bear_ob_0_age", 0.0)
        bull_ob_age = self._safe(row, "bull_ob_0_age", 0.0)
        bear_fvg_filled = self._safe(row, "bear_fvg_0_filled_pct", 0.0)
        bull_fvg_filled = self._safe(row, "bull_fvg_0_filled_pct", 0.0)

        # Only emit the zone metrics relevant for the direction
        if direction == "SHORT":
            zone_line = (
                f"  bear_ob_age: {bear_ob_age:.0f} bars | "
                f"bear_fvg_filled: {bear_fvg_filled:.0%}\n"
            )
        else:
            zone_line = (
                f"  bull_ob_age: {bull_ob_age:.0f} bars | "
                f"bull_fvg_filled: {bull_fvg_filled:.0%}\n"
            )

        return (
            f"Pair: {pair} | Direction: {direction}\n"
            f"enter_tag: {enter_tag}\n"
            f"subsetup: {subsetup}\n"
            f"\nFRESHNESS (bars since CHoCH in setup direction):\n"
            f"  internal: {bs_int_choch} | swing: {bs_sw_choch}\n"
            f"\nHTF BIAS:\n"
            f"  weekly_range_position: {weekly_range_pct:.0f}%\n"
            f"  btc_macro_bias: {btc_macro:+d}\n"
            f"\nSESSION:\n"
            f"  session: {session} | in_kill_zone: {in_kz} | is_monday: {is_monday}\n"
            f"\nZONE STALENESS:\n"
            + zone_line
        )

    def _call_api(self, messages: list[dict]) -> dict | None:
        """Call OpenRouter API with retry and rate-limit handling."""
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/freqtrade",
            "X-Title": "SMC-LLM-Filter",
        }
        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 2000,
        }

        for attempt in range(self._max_retries):
            try:
                resp = requests.post(
                    API_URL, headers=headers, json=payload,
                    timeout=self._timeout,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    message = data["choices"][0]["message"]
                    content = message.get("content") or ""
                    reasoning = message.get("reasoning") or ""
                    # Prefer content; fall back to reasoning (DeepSeek-R1, GLM thinking models)
                    # Combine both so _parse_response can find the JSON wherever it lands
                    combined = (content + "\n" + reasoning).strip()
                    return combined if combined else None
                elif resp.status_code == 429:
                    wait = min(2 ** attempt * 2, 30)
                    logger.warning(f"LLM rate limited, retrying in {wait}s")
                    time.sleep(wait)
                else:
                    if resp.status_code in (402, 403) and self._model != self.FALLBACK_MODEL:
                        logger.warning(
                            f"LLM model {self._model} unavailable ({resp.status_code}), "
                            f"falling back to {self.FALLBACK_MODEL}"
                        )
                        self._model = self.FALLBACK_MODEL
                        payload["model"] = self._model
                        continue
                    logger.error(f"LLM API error {resp.status_code}: {resp.text[:200]}")
                    return None
            except requests.Timeout:
                logger.warning(f"LLM timeout (attempt {attempt + 1}/{self._max_retries})")
            except Exception as e:
                logger.error(f"LLM API exception: {e}")
                return None
        return None

    def _parse_response(self, raw_content: str) -> dict | None:
        """Extract JSON from LLM response, handling markdown fences."""
        text = raw_content.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.startswith("```")]
            text = "\n".join(lines).strip()

        # Find the LAST valid JSON block — reasoning models put JSON at the end
        last_end = text.rfind("}")
        if last_end == -1:
            logger.warning(f"LLM response has no JSON: {text[:100]}")
            return None

        # Walk backwards from last_end to find matching opening brace
        depth = 0
        start = -1
        for i in range(last_end, -1, -1):
            if text[i] == "}":
                depth += 1
            elif text[i] == "{":
                depth -= 1
                if depth == 0:
                    start = i
                    break

        if start == -1:
            logger.warning(f"LLM response has no valid JSON block: {text[:100]}")
            return None

        try:
            parsed = json.loads(text[start:last_end + 1])
            confidence = float(parsed.get("confidence", 0))
            return {
                "confidence": max(0.0, min(1.0, confidence)),
                "reason": str(parsed.get("reason", ""))[:200],
            }
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.warning(f"LLM JSON parse error: {e} | raw: {text[start:last_end+1][:100]}")
            return None

    def _get_cache_key(self, pair: str, row: pd.Series, direction: str) -> str:
        """Deterministic fingerprint of the market context."""
        parts = [
            pair,
            direction,
            str(int(self._safe(row, "swing_trend", 0))),
            str(row.get("enter_tag", "")),
            f"{self._safe(row, 'bull_ob_0_rvol', 0):.1f}",
            f"{self._safe(row, 'bear_ob_0_rvol', 0):.1f}",
            str(row.get("session", "")),
            str(int(self._safe(row, "btc_macro_bias", 0))),
            str(row.get("daily_prem_disc_label", "")),
        ]
        return "|".join(parts)

    def _check_cache(self, key: str) -> float | None:
        if key in self._cache:
            conf, ts = self._cache[key]
            if time.time() - ts < self._cache_ttl:
                return conf
            del self._cache[key]
        return None

    def _store_cache(self, key: str, confidence: float) -> None:
        self._cache[key] = (confidence, time.time())
        if len(self._cache) > 200:
            oldest = sorted(self._cache.items(), key=lambda x: x[1][1])[:50]
            for k, _ in oldest:
                del self._cache[k]

    @staticmethod
    def _safe(row: pd.Series, col: str, default=0.0):
        """Safe column access with default for missing/NaN values."""
        val = row.get(col, default)
        if isinstance(val, float) and val != val:  # NaN check
            return default
        return val
