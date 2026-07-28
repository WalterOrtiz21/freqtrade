"""
Ablation harness for LorentzianSuperTrend.

Goal: isolate how much of the strategy's edge comes from the Lorentzian KNN
classifier versus the surrounding trend/kernel filters.

Everything is inherited from the production strategy. The ONLY thing this
subclass changes is the source of `dataframe['signal']` — the latching function,
the `filter_all` gate, entry/exit logic, stoploss and ROI are untouched, so any
difference in results is attributable to the classifier alone.

Modes (env var ABLATION_MODE):
  knn    - baseline passthrough, must reproduce LorentzianSuperTrend exactly
  trend  - prediction replaced by the position EMA direction (close vs EMA200).
           Note: the ml_ema/ml_sma filters are disabled in the deployed params
           (use_ml_ema_filter=False, use_ml_sma_filter=False), so the position
           EMA is the only trend filter actually active in the strategy.
  kernel - prediction replaced by the Nadaraya-Watson kernel direction
  random - prediction replaced by a random walk whose sign-flip rate matches
           the real classifier's, per pair. Destroys the information while
           preserving trade pacing. Seed comes from ABLATION_SEED.
"""

import hashlib
import importlib.util
import logging
import os
from pathlib import Path

import numpy as np
from pandas import DataFrame

logger = logging.getLogger(__name__)

# Load the production strategy by path. Importing by module name is unsafe here
# because the package dir and the module file share the name LorentzianSuperTrend.
_BASE_PATH = (
    Path(__file__).resolve().parent.parent
    / "LorentzianSuperTrend"
    / "LorentzianSuperTrend.py"
)
_spec = importlib.util.spec_from_file_location("_lst_base", _BASE_PATH)
_base_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base_mod)

LorentzianSuperTrend = _base_mod.LorentzianSuperTrend
numba_calculate_latched_signal = _base_mod.numba_calculate_latched_signal


class LorentzianAblation(LorentzianSuperTrend):

    # The base strategy now defaults to the EMA signal; the harness needs the
    # classifier computed so that 'knn' mode stays a real control arm.
    use_knn_signal = True

    ablation_mode = os.environ.get("ABLATION_MODE", "knn").lower()
    ablation_seed = int(os.environ.get("ABLATION_SEED", "0"))

    def _flip_rate(self, predictions: np.ndarray) -> float:
        """Per-bar probability that the classifier's sign changes."""
        signs = np.sign(predictions)
        nz = signs[signs != 0]
        if len(nz) < 2:
            return 0.05
        return float(np.mean(nz[1:] != nz[:-1]))

    def _random_predictions(self, predictions: np.ndarray, pair: str) -> np.ndarray:
        """Random walk with the same sign-flip rate as the real classifier."""
        p_flip = self._flip_rate(predictions)
        # Deterministic per (pair, seed) so runs are reproducible.
        digest = hashlib.sha256(f"{pair}:{self.ablation_seed}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))

        n = len(predictions)
        flips = rng.random(n) < p_flip
        out = np.empty(n, dtype=np.float64)
        current = 1.0 if rng.random() < 0.5 else -1.0
        for i in range(n):
            if flips[i]:
                current = -current
            out[i] = current
        return out

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_indicators(dataframe, metadata)

        mode = self.ablation_mode
        if mode == "knn":
            return dataframe

        predictions = dataframe["prediction"].fillna(0.0).values.astype(np.float64)

        if mode == "trend":
            up = dataframe["bullish"].fillna(False).values
            down = dataframe["bearish"].fillna(False).values
            ablated = np.where(up, 1.0, np.where(down, -1.0, 0.0))
        elif mode == "kernel":
            up = dataframe["is_bullish_kernel"].fillna(False).values
            down = dataframe["is_bearish_kernel"].fillna(False).values
            ablated = np.where(up, 1.0, np.where(down, -1.0, 0.0))
        elif mode == "random":
            ablated = self._random_predictions(predictions, metadata["pair"])
        else:
            raise ValueError(f"unknown ABLATION_MODE: {mode}")

        dataframe["prediction"] = ablated
        filter_all = dataframe["filter_all"].fillna(False).values.astype(bool)
        dataframe["signal"] = numba_calculate_latched_signal(ablated, filter_all)
        return dataframe
