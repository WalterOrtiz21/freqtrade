"""
Volume Pressure Oscillator (VPI) V2 — Numba Optimized
======================================================

Core indicator module for Low Market Cap strategy.

V2 Improvements:
- Signal Line: Slower EMA of VPI for precise cross timing (like MACD)
- Cumulative VPI: Rolling sum over N candles for sustained pressure detection
- Volume Persistence: Counts consecutive elevated RVOL candles
- RVOL Acceleration: Detects increasing relative volume

Thesis: "In low-cap, volume precedes price."

Components:
- RVOL (Relative Volume): volume / SMA(volume, period)
- Net Pressure: (close - low - (high - close)) / (high - low) ∈ [-1, +1]
- VPI: Net_Pressure * RVOL → Volume-weighted directional pressure
- VPI Signal: Slower EMA of VPI → cross-based timing
- VPI Histogram: VPI_smooth - VPI_signal → momentum of pressure
- Cumulative VPI: Rolling sum → detects SUSTAINED pressure
- Volume Persistence: # consecutive candles with RVOL > threshold
"""

import numpy as np
from numba import njit


# =============================================================================
# CORE CALCULATIONS (Numba)
# =============================================================================

@njit(cache=False)
def _calculate_rvol(volume: np.ndarray, period: int) -> np.ndarray:
    """
    Relative Volume = volume / SMA(volume, period).
    
    RVOL > 2.0 = significant volume spike.
    RVOL > 5.0 = extreme (whale entry or wash trade).
    """
    n = len(volume)
    rvol = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        total = 0.0
        for j in range(period):
            total += volume[i - j]
        avg = total / period
        
        if avg > 0:
            rvol[i] = volume[i] / avg
        else:
            rvol[i] = 0.0
    
    return rvol


@njit(cache=False)
def _calculate_net_pressure(close: np.ndarray, high: np.ndarray, low: np.ndarray) -> np.ndarray:
    """
    Net Buy/Sell Pressure per candle.
    
      +1.0 = Full bullish candle (close == high)
      -1.0 = Full bearish candle (close == low)
       0.0 = Doji (equal pressure)
    """
    n = len(close)
    pressure = np.zeros(n)
    
    for i in range(n):
        candle_range = high[i] - low[i]
        if candle_range > 0:
            bp = close[i] - low[i]
            sp = high[i] - close[i]
            pressure[i] = (bp - sp) / candle_range
        else:
            pressure[i] = 0.0
    
    return pressure


@njit(cache=False)
def _ema_numba(data: np.ndarray, period: int) -> np.ndarray:
    """
    Exponential Moving Average (EMA).
    Uses SMA for the initial seed value.
    """
    n = len(data)
    ema = np.full(n, np.nan)
    alpha = 2.0 / (period + 1)
    
    valid_count = 0
    total = 0.0
    seed_idx = -1
    
    for i in range(n):
        if not np.isnan(data[i]):
            total += data[i]
            valid_count += 1
            if valid_count == period:
                seed_idx = i
                break
    
    if seed_idx < 0:
        return ema
    
    ema[seed_idx] = total / period
    
    for i in range(seed_idx + 1, n):
        if np.isnan(data[i]):
            ema[i] = ema[i - 1]
        else:
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
    
    return ema


@njit(cache=False)
def _cumulative_vpi(vpi_raw: np.ndarray, window: int) -> np.ndarray:
    """
    Rolling sum of VPI over a window.
    
    Detects SUSTAINED pressure:
    - A single spike of VPI=+5 is noise
    - 5 candles of VPI=+1.0 is accumulation (cumulative=+5)
    
    The cumulative VPI filters out single-candle spikes and rewards
    consistent directional pressure over multiple candles.
    """
    n = len(vpi_raw)
    cum_vpi = np.full(n, np.nan)
    
    for i in range(window - 1, n):
        total = 0.0
        valid_count = 0
        for j in range(window):
            val = vpi_raw[i - j]
            if not np.isnan(val):
                total += val
                valid_count += 1
        if valid_count > 0:
            cum_vpi[i] = total
    
    return cum_vpi


@njit(cache=False)
def _volume_persistence(rvol: np.ndarray, min_rvol: float) -> np.ndarray:
    """
    Count consecutive candles where RVOL > min_rvol.
    
    Resets to 0 when RVOL drops below threshold.
    High persistence = sustained institutional activity.
    Low persistence = random spike, likely noise.
    """
    n = len(rvol)
    persist = np.zeros(n, dtype=np.int64)
    
    for i in range(n):
        if np.isnan(rvol[i]):
            persist[i] = 0
        elif rvol[i] >= min_rvol:
            if i > 0:
                persist[i] = persist[i - 1] + 1
            else:
                persist[i] = 1
        else:
            persist[i] = 0
    
    return persist


@njit(cache=False)
def _candle_body_position(close: np.ndarray, high: np.ndarray, low: np.ndarray) -> np.ndarray:
    """
    Where is the close relative to the candle range? [0, 1]
    
    0.0 = close at the low (full bearish)
    1.0 = close at the high (full bullish)
    0.5 = close at midpoint (doji)
    
    More useful than simple "above/below mid" because 
    we can require close to be in top/bottom quartile.
    """
    n = len(close)
    pos = np.full(n, 0.5)
    
    for i in range(n):
        candle_range = high[i] - low[i]
        if candle_range > 0:
            pos[i] = (close[i] - low[i]) / candle_range
    
    return pos


@njit(cache=False)
def _calculate_vpi_full_v2(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    volume: np.ndarray,
    rvol_period: int,
    smooth_period: int,
    signal_period: int,
    cumulative_window: int,
    min_rvol_for_persist: float
) -> tuple:
    """
    Full VPI V2 calculation pipeline.
    
    Returns:
        (vpi_raw, vpi_smooth, vpi_signal, vpi_histogram,
         rvol, net_pressure, cumulative_vpi, volume_persist, body_position)
    """
    # 1. RVOL
    rvol = _calculate_rvol(volume, rvol_period)
    
    # 2. Net Pressure
    pressure = _calculate_net_pressure(close, high, low)
    
    # 3. VPI Raw = Pressure * RVOL
    n = len(close)
    vpi_raw = np.full(n, np.nan)
    for i in range(n):
        if not np.isnan(rvol[i]):
            vpi_raw[i] = pressure[i] * rvol[i]
    
    # 4. VPI Smoothed = EMA(VPI_raw, smooth_period)
    vpi_smooth = _ema_numba(vpi_raw, smooth_period)
    
    # 5. VPI Signal Line = EMA(VPI_smooth, signal_period) — slower, like MACD signal
    vpi_signal = _ema_numba(vpi_smooth, signal_period)
    
    # 6. VPI Histogram = VPI_smooth - VPI_signal — momentum of pressure
    vpi_histogram = np.full(n, np.nan)
    for i in range(n):
        if not np.isnan(vpi_smooth[i]) and not np.isnan(vpi_signal[i]):
            vpi_histogram[i] = vpi_smooth[i] - vpi_signal[i]
    
    # 7. Cumulative VPI = Rolling sum of VPI_raw
    cum_vpi = _cumulative_vpi(vpi_raw, cumulative_window)
    
    # 8. Volume Persistence = Consecutive candles with RVOL > threshold
    vol_persist = _volume_persistence(rvol, min_rvol_for_persist)
    
    # 9. Body Position = Close position within candle [0, 1]
    body_pos = _candle_body_position(close, high, low)
    
    return (vpi_raw, vpi_smooth, vpi_signal, vpi_histogram,
            rvol, pressure, cum_vpi, vol_persist, body_pos)


# =============================================================================
# PUBLIC API
# =============================================================================

def calculate_vpi(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    volume: np.ndarray,
    rvol_period: int = 20,
    smooth_period: int = 5,
    signal_period: int = 13,
    cumulative_window: int = 5,
    min_rvol_for_persist: float = 1.5
) -> dict:
    """
    Calculate the Volume Pressure Index (VPI) V2.
    
    Args:
        close, high, low, volume: OHLCV arrays
        rvol_period: Lookback for RVOL baseline SMA (default: 20)
        smooth_period: Fast EMA period for VPI (default: 5)
        signal_period: Slow EMA period for signal line (default: 13)
        cumulative_window: Rolling window for cumulative VPI (default: 5)
        min_rvol_for_persist: Min RVOL to count as "elevated" volume (default: 1.5)
    
    Returns:
        dict with all indicator arrays
    """
    results = _calculate_vpi_full_v2(
        close.astype(np.float64),
        high.astype(np.float64),
        low.astype(np.float64),
        volume.astype(np.float64),
        rvol_period,
        smooth_period,
        signal_period,
        cumulative_window,
        min_rvol_for_persist
    )
    
    return {
        'vpi_raw': results[0],
        'vpi_smooth': results[1],        # Fast line
        'vpi_signal': results[2],        # Slow line (cross timing)
        'vpi_histogram': results[3],      # Momentum of pressure
        'rvol': results[4],
        'net_pressure': results[5],
        'cumulative_vpi': results[6],     # Sustained pressure
        'volume_persist': results[7],     # Consecutive elevated RVOL candles
        'body_position': results[8]       # Close position in candle [0,1]
    }
