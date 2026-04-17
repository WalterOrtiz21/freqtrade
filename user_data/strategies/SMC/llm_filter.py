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
    "You are an institutional SMC/ICT trading analyst with these empirical rules from live backtesting:\n"
    "\n"
    "ENTRY QUALITY HIERARCHY (most to least important):\n"
    "1. HTF TREND ALIGNMENT — if ANY higher timeframe (1H, 4H) conflicts with entry direction, "
    "confidence drops DRAMATICALLY. A long against 4H bear is low-probability regardless of LTF structure.\n"
    "2. PRICE LOCATION — entries at deep premium (swing_range >80%) against BTC macro bias are traps. "
    "Perfect structure means nothing if price is at a distribution extreme against the dominant flow.\n"
    "3. ZONE QUALITY — fresh OB (touches=0, mitigated=0, rvol>2.0, age<15) is strong. "
    "Mitigated OB (mitigated=1), touched OB (touches>1), stale FVG (filled>50%) are weak zones.\n"
    "4. STRUCTURAL CONFIRMATION — CHoCH must be present. A bounce without CHoCH is NOT a reversal, "
    "just a retracement. Sweep without CHoCH = continuation trap.\n"
    "\n"
    "EMPIRICAL RULES:\n"
    "- SWEEP entries: 38.3% win rate — consistently below average. Treat sweep-only setups skeptically.\n"
    "- OFF-session entries (22:00-00:00 UTC): lower win rate vs session entries.\n"
    "- FVG_STALE + HTF_PART combo: 17 trades avg +0.15% — not worth risk.\n"
    "- Deep premium longs (swing_range >85%) + BTC bearish bias = strong trap signal "
    "even with perfect LTF structure.\n"
    "- An entry can have OB_FRESH + CHoCH + Kill Zone and STILL be bad "
    "if the higher timeframe context is against it.\n"
    "\n"
    "SCORING GUIDELINES:\n"
    "- 0.8-1.0: All timeframes aligned, fresh zone, discount/premium aligned, CHoCH present, kill zone active.\n"
    "- 0.5-0.7: Good LTF structure but ONE conflicting signal "
    "(e.g., one HTF against, or premium but BTC neutral).\n"
    "- 0.2-0.5: Mixed signals, ambiguous zone, or conflicting HTF context. Borderline.\n"
    "- 0.0-0.2: Trap setup. Structure looks OK but context is wrong "
    "(against HTF, deep premium with bearish BTC, no CHoCH, sweep-only).\n"
    "\n"
    'Respond ONLY with valid JSON: {"confidence": <0.0-1.0>, "reason": "<one line>"}\n'
    "No other text, no markdown fences."
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
        """Build compact prompt from SMC context for a single entry signal."""
        close = self._safe(row, "close", 0.0)

        # Zone data — determine which zone the price is in
        zone_parts = []
        zone_type = "NoZone"
        zone_bottom = 0.0
        zone_top = 0.0

        for prefix, label in [("active_bullish_ob", "OB"), ("active_bullish_breaker", "BRK"),
                              ("active_bullish_fvg", "FVG"), ("active_bull_fvg_breaker", "FVGBRK")]:
            top = self._safe(row, f"{prefix}_top", 0.0)
            btm = self._safe(row, f"{prefix}_bottom", 0.0)
            if top > 0 and btm > 0 and direction == "LONG":
                zone_type = label
                zone_top = top
                zone_bottom = btm
                break

        for prefix, label in [("active_bearish_ob", "OB"), ("active_bearish_breaker", "BRK"),
                              ("active_bearish_fvg", "FVG"), ("active_bear_fvg_breaker", "FVGBRK")]:
            top = self._safe(row, f"{prefix}_top", 0.0)
            btm = self._safe(row, f"{prefix}_bottom", 0.0)
            if top > 0 and btm > 0 and direction == "SHORT":
                zone_type = label
                zone_top = top
                zone_bottom = btm
                break

        # Zone quality
        ob_prefix = "bull" if direction == "LONG" else "bear"
        ob_rvol = self._safe(row, f"{ob_prefix}_ob_0_rvol", 0.0)
        ob_age = self._safe(row, f"{ob_prefix}_ob_0_age", 99.0)
        ob_touches = self._safe(row, f"{ob_prefix}_ob_0_touches", 0.0)
        ob_mitigated = int(self._safe(row, f"{ob_prefix}_ob_0_mitigated", 0.0))

        fvg_filled = self._safe(row, f"{ob_prefix}_fvg_0_filled_pct", 0.0)
        fvg_rvol = self._safe(row, f"{ob_prefix}_fvg_0_rvol", 0.0)

        # Trends
        int_trend = int(self._safe(row, "internal_trend", 0))
        swing_trend = int(self._safe(row, "swing_trend", 0))
        htf_1h = int(self._safe(row, "1h_swing_trend", 0))
        htf_4h = int(self._safe(row, "4h_swing_trend", 0))

        trend_labels = {1: "Bull", -1: "Bear", 0: "Neutral"}

        # Context
        enter_tag = str(row.get("enter_tag", ""))
        session = str(row.get("session", ""))
        in_kz = bool(self._safe(row, "in_kill_zone", 0.0))
        btc_macro = int(self._safe(row, "btc_macro_bias", 0))
        pd_label = str(row.get("daily_prem_disc_label", ""))
        swing_range_pct = self._safe(row, "swing_range_position_pct", 50.0)

        # Structure
        int_choch_bull = int(self._safe(row, "internal_choch_bullish", 0))
        int_choch_bear = int(self._safe(row, "internal_choch_bearish", 0))
        sw_choch_bull = int(self._safe(row, "swing_choch_bullish", 0))
        sw_choch_bear = int(self._safe(row, "swing_choch_bearish", 0))
        int_sweep_bull = int(self._safe(row, "internal_sweep_bullish", 0))
        int_sweep_bear = int(self._safe(row, "internal_sweep_bearish", 0))
        sw_sweep_bull = int(self._safe(row, "swing_sweep_bullish", 0))
        sw_sweep_bear = int(self._safe(row, "swing_sweep_bearish", 0))

        # Distance from zone
        zone_dist = 0.0
        if zone_top > 0 and close > 0:
            zone_dist = ((close - zone_top) / close) * 100

        return (
            f"Pair: {pair} | Direction: {direction}\n"
            f"Price: {close:.8f}\n"
            f"\nTREND:\n"
            f"  Internal: {trend_labels.get(int_trend, '?')} | Swing: {trend_labels.get(swing_trend, '?')}\n"
            f"  HTF 1H: {trend_labels.get(htf_1h, '?')} | HTF 4H: {trend_labels.get(htf_4h, '?')}\n"
            f"\nZONE:\n"
            f"  Type: {zone_type} [{zone_bottom:.8f} - {zone_top:.8f}] (dist: {zone_dist:+.2f}%)\n"
            f"  OB RVol: {ob_rvol:.2f} | Age: {ob_age:.0f} bars | Touches: {ob_touches:.0f} | Mitigated: {ob_mitigated}\n"
            f"  FVG Fill: {fvg_filled:.0%} | FVG RVol: {fvg_rvol:.2f}\n"
            f"\nCONTEXT:\n"
            f"  Confluence: {enter_tag}\n"
            f"  Session: {session} | Kill Zone: {in_kz}\n"
            f"  BTC Bias: {btc_macro:+d} | P/D: {pd_label}\n"
            f"  Swing Range Position: {swing_range_pct:.1f}%\n"
            f"\nSTRUCTURE:\n"
            f"  Int CHoCH Bull: {int_choch_bull} | Bear: {int_choch_bear}\n"
            f"  Sw CHoCH Bull: {sw_choch_bull} | Bear: {sw_choch_bear}\n"
            f"  Int Sweep Bull: {int_sweep_bull} | Bear: {int_sweep_bear}\n"
            f"  Sw Sweep Bull: {sw_sweep_bull} | Bear: {sw_sweep_bear}\n"
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
                    # Some models (GLM, DeepSeek-R1) put output in "reasoning" field
                    content = message.get("content")
                    if not content:
                        reasoning = message.get("reasoning", "")
                        if reasoning:
                            # Try to extract JSON from reasoning text
                            return reasoning
                    return content
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

        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            logger.warning(f"LLM response has no JSON: {text[:100]}")
            return None

        try:
            parsed = json.loads(text[start:end + 1])
            confidence = float(parsed.get("confidence", 0))
            return {
                "confidence": max(0.0, min(1.0, confidence)),
                "reason": str(parsed.get("reason", ""))[:200],
            }
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.warning(f"LLM JSON parse error: {e} | raw: {text[:100]}")
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
