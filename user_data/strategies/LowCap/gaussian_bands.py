"""
Gaussian Bands — Numba Optimized
=================================

Ported from BigBeluga's "Volatility Gaussian Bands [BigBeluga]" Pine Script.

Components:
- Gaussian Filter: Weighted moving average using Gaussian kernel
- Multi-Trend Score: 21 variations of the filter, 0-1 score of trend agreement
- Volatility Bands: avg ± SMA(high-low, volatility_period) * distance
- Trend Detection: Flip on price crossing upper/lower band

Usage in strategy:
- trend_score > 0.5 → bearish consensus, < 0.5 → bullish consensus
- trend (bool) → current direction (True = up, False = down)
- trend_line → dynamic support/resistance level
- upper_band / lower_band → TP target zones
"""

import numpy as np
from numba import njit


@njit(cache=False)
def _gaussian_filter(src: np.ndarray, length: int, sigma: float, start_idx: int) -> float:
    """
    Single-point Gaussian filter calculation.
    
    Applies Gaussian-weighted smoothing to the source series at position start_idx.
    
    Args:
        src: Source price series
        length: Filter length (number of points to consider)
        sigma: Standard deviation for the Gaussian kernel
        start_idx: Current position in the series
    
    Returns:
        Gaussian-filtered value at start_idx
    """
    if start_idx < length - 1:
        return np.nan
    
    pi = 3.141592653589793
    total_weight = 0.0
    weighted_sum = 0.0
    
    for i in range(length):
        # Gaussian kernel weight
        x = (i - length / 2.0) / sigma
        weight = np.exp(-0.5 * x * x) / np.sqrt(sigma * 2.0 * pi)
        
        weighted_sum += src[start_idx - i] * weight
        total_weight += weight
    
    if total_weight > 0:
        return weighted_sum / total_weight
    return np.nan


@njit(cache=False)
def _calculate_gaussian_bands(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    length: int,
    sigma: float,
    distance: float,
    volatility_period: int,
    mode: int  # 0=AVG, 1=MEDIAN, 2=MODE (we'll use AVG for simplicity & speed)
) -> tuple:
    """
    Full Gaussian Bands calculation.
    
    Ported from BigBeluga Pine Script:
    1. Calculate Gaussian filter at 21 different periods (length to length+20)
    2. Score = how many longer-period filters are above the shortest → trend strength
    3. Value = average of all 21 Gaussian filters
    4. Bands = value ± volatility * distance
    5. Trend flips on price crossing bands
    
    Args:
        close: Close prices
        high: High prices
        low: Low prices
        length: Base Gaussian filter length (Pine default: 20)
        sigma: Gaussian sigma (Pine default: 10)
        distance: Band width multiplier (Pine default: 1.0)
        volatility_period: SMA period for volatility (range-based ATR)
        mode: Aggregation mode (0=AVG)
    
    Returns:
        (trend_score, avg_value, upper_band, lower_band, trend_line, trend_direction)
    """
    n = len(close)
    num_steps = 21  # Pine: for step = 0 to 20
    
    # Output arrays
    trend_score = np.full(n, np.nan)
    avg_value = np.full(n, np.nan)
    upper_band = np.full(n, np.nan)
    lower_band = np.full(n, np.nan)
    trend_line = np.full(n, np.nan)
    trend_dir = np.zeros(n, dtype=np.int8)  # 1=up, -1=down, 0=undefined
    
    # Pre-calculate volatility: SMA(high - low, volatility_period)
    volatility = np.full(n, np.nan)
    for i in range(volatility_period - 1, n):
        total = 0.0
        for j in range(volatility_period):
            total += (high[i - j] - low[i - j])
        volatility[i] = total / volatility_period
    
    # Minimum lookback needed
    min_lookback = length + num_steps - 1
    if min_lookback > n:
        return (trend_score, avg_value, upper_band, lower_band, trend_line, trend_dir)
    
    # Track previous trend for persistence
    prev_trend = 0  # 0=undefined
    
    for i in range(min_lookback, n):
        if np.isnan(volatility[i]):
            continue
        
        # Calculate 21 Gaussian filter values at different periods
        g_values = np.empty(num_steps)
        for step in range(num_steps):
            period = length + step
            g_values[step] = _gaussian_filter(close, period, sigma, i)
        
        # Check for NaN in g_values
        has_nan = False
        for step in range(num_steps):
            if np.isnan(g_values[step]):
                has_nan = True
                break
        if has_nan:
            continue
        
        # Score: count how many longer-period filters are above the first (shortest)
        # Pine: if g_f > g_value.first() then score += coeff
        coeff = 0.05
        score = 0.0
        first_val = g_values[0]
        for step in range(num_steps):
            if g_values[step] > first_val:
                score += coeff
        
        trend_score[i] = score
        
        # Value = average of all Gaussian filter values
        total = 0.0
        for step in range(num_steps):
            total += g_values[step]
        value = total / num_steps
        avg_value[i] = value
        
        # Bands
        vol = volatility[i]
        lb = value - vol * distance
        ub = value + vol * distance
        lower_band[i] = lb
        upper_band[i] = ub
        
        # Trend detection: crossover/crossunder with bands
        current_trend = prev_trend
        
        # Crossover upper band → bullish
        if i > 0:
            prev_close = close[i - 1]
            if not np.isnan(upper_band[i - 1]):
                if close[i] > ub and prev_close <= upper_band[i - 1]:
                    current_trend = 1
            if not np.isnan(lower_band[i - 1]):
                if close[i] < lb and prev_close >= lower_band[i - 1]:
                    current_trend = -1
        
        trend_dir[i] = current_trend
        prev_trend = current_trend
        
        # Trend line: show opposite band as support/resistance
        if current_trend == 1:
            trend_line[i] = lb  # In uptrend, lower band is support
        elif current_trend == -1:
            trend_line[i] = ub  # In downtrend, upper band is resistance
    
    return (trend_score, avg_value, upper_band, lower_band, trend_line, trend_dir)


def calculate_gaussian_bands(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    length: int = 20,
    sigma: float = 10.0,
    distance: float = 1.0,
    volatility_period: int = 100
) -> dict:
    """
    Calculate Gaussian Bands (BigBeluga port).
    
    Args:
        close, high, low: Price arrays
        length: Base Gaussian filter period (default: 20)
        sigma: Gaussian smoothing sigma (default: 10.0)
        distance: Band width multiplier (default: 1.0)
        volatility_period: Lookback for volatility SMA (default: 100)
    
    Returns:
        dict with:
        - trend_score: 0.0 (fully bullish) to 1.0 (fully bearish)
        - avg_value: Gaussian-smoothed price (midline)
        - upper_band: avg + volatility * distance
        - lower_band: avg - volatility * distance
        - trend_line: Dynamic support (uptrend) or resistance (downtrend)
        - trend_direction: 1 (up), -1 (down), 0 (undefined)
    """
    results = _calculate_gaussian_bands(
        close.astype(np.float64),
        high.astype(np.float64),
        low.astype(np.float64),
        length,
        sigma,
        distance,
        volatility_period,
        0  # AVG mode
    )
    
    return {
        'trend_score': results[0],
        'avg_value': results[1],
        'upper_band': results[2],
        'lower_band': results[3],
        'trend_line': results[4],
        'trend_direction': results[5]
    }
