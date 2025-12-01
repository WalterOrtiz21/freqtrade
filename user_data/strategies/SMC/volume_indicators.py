"""
Volume Indicators Module for SMC + Volume Trading Strategy
Based on Volume Trading Strategies + Smart Money Concepts video

This module provides volume analysis functions including:
1. Volume Divergence Detection
2. Volume Spike Identification
3. Volume Trend Analysis
4. Volume Profile Analysis (basic)

Author: Adapted from The Trading Channel methodology
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


def calculate_volume_divergence(
    dataframe: pd.DataFrame,
    length: int = 5,
    min_divergence_strength: float = 0.1
) -> pd.DataFrame:
    """
    Calculate volume divergence signals based on price vs volume behavior.

    Args:
        dataframe: OHLCV DataFrame
        length: Lookback period for divergence detection
        min_divergence_strength: Minimum divergence strength (0-1)

    Returns:
        DataFrame with divergence indicators added
    """
    df = dataframe.copy()

    # Calculate volume moving average for trend
    df['volume_sma'] = df['volume'].rolling(window=length).mean()
    df['volume_ema'] = df['volume'].ewm(span=length).mean()

    # Calculate volume trend (relative to moving average)
    df['volume_trend'] = df['volume'] / df['volume_sma'] - 1

    # Calculate price trend (for comparison)
    df['price_trend'] = (df['close'] / df['close'].shift(length) - 1)

    # Detect divergences
    # Bearish divergence: Price trending up, volume trending down
    df['bearish_divergence'] = (
        (df['price_trend'] > min_divergence_strength) &  # Price moving up
        (df['volume_trend'] < -min_divergence_strength)  # Volume moving down
    ).astype(int)

    # Bullish divergence: Price trending down, volume trending up
    df['bullish_divergence'] = (
        (df['price_trend'] < -min_divergence_strength) &  # Price moving down
        (df['volume_trend'] > min_divergence_strength)    # Volume moving up
    ).astype(int)

    # Detect divergences over multiple periods
    df['price_higher_high'] = (
        df['high'] > df['high'].rolling(window=length).max().shift(1)
    )
    df['price_lower_low'] = (
        df['low'] < df['low'].rolling(window=length).min().shift(1)
    )

    df['volume_lower_high'] = (
        df['volume'] < df['volume'].rolling(window=length).max().shift(1)
    )
    df['volume_higher_low'] = (
        df['volume'] > df['volume'].rolling(window=length).min().shift(1)
    )

    # Strong bearish divergence: Higher highs in price, lower highs in volume
    df['strong_bearish_divergence'] = (
        df['price_higher_high'] & df['volume_lower_high']
    ).astype(int)

    # Strong bullish divergence: Lower lows in price, higher lows in volume
    df['strong_bullish_divergence'] = (
        df['price_lower_low'] & df['volume_higher_low']
    ).astype(int)

    logger.debug(f"Volume divergence calculations completed. Bearish: {df['bearish_divergence'].sum()}, Bullish: {df['bullish_divergence'].sum()}")

    return df


def detect_volume_spikes(
    dataframe: pd.DataFrame,
    spike_multiplier: float = 2.0,
    lookback_period: int = 20
) -> pd.DataFrame:
    """
    Detect significant volume spikes that confirm market moves.

    Args:
        dataframe: OHLCV DataFrame
        spike_multiplier: Multiplier above average volume to consider a spike
        lookback_period: Period to calculate average volume

    Returns:
        DataFrame with spike indicators added
    """
    df = dataframe.copy()

    # Calculate volume averages
    df['volume_avg'] = df['volume'].rolling(window=lookback_period).mean()
    df['volume_std'] = df['volume'].rolling(window=lookback_period).std()

    # Detect volume spikes
    df['volume_spike'] = (df['volume'] > df['volume_avg'] * spike_multiplier).astype(int)

    # Calculate spike strength (how much above average)
    df['spike_strength'] = df['volume'] / df['volume_avg']

    # Detect significant spikes (using standard deviation)
    df['significant_spike'] = (
        df['volume'] > (df['volume_avg'] + df['volume_std'] * 2)
    ).astype(int)

    # Volume increase/decrease percentage
    df['volume_change_pct'] = df['volume'].pct_change() * 100

    # Detect accelerating volume (volume increasing over time)
    df['volume_acceleration'] = (
        df['volume'] > df['volume'].shift(1).rolling(window=3).mean()
    ).astype(int)

    logger.debug(f"Volume spike detection completed. Spikes: {df['volume_spike'].sum()}, Significant: {df['significant_spike'].sum()}")

    return df


def calculate_volume_trend(
    dataframe: pd.DataFrame,
    short_period: int = 10,
    long_period: int = 20
) -> pd.DataFrame:
    """
    Calculate volume trend indicators.

    Args:
        dataframe: OHLCV DataFrame
        short_period: Short period EMA
        long_period: Long period EMA

    Returns:
        DataFrame with volume trend indicators added
    """
    df = dataframe.copy()

    # Calculate volume EMAs
    df['volume_ema_short'] = df['volume'].ewm(span=short_period).mean()
    df['volume_ema_long'] = df['volume'].ewm(span=long_period).mean()

    # Volume trend direction
    df['volume_trend_up'] = (df['volume_ema_short'] > df['volume_ema_long']).astype(int)
    df['volume_trend_down'] = (df['volume_ema_short'] < df['volume_ema_long']).astype(int)

    # Volume trend strength (ratio)
    df['volume_trend_strength'] = df['volume_ema_short'] / df['volume_ema_long'] - 1

    # Volume crossovers (trend changes)
    df['volume_bull_cross'] = (
        (df['volume_ema_short'] > df['volume_ema_long']) &
        (df['volume_ema_short'].shift(1) <= df['volume_ema_long'].shift(1))
    ).astype(int)

    df['volume_bear_cross'] = (
        (df['volume_ema_short'] < df['volume_ema_long']) &
        (df['volume_ema_short'].shift(1) >= df['volume_ema_long'].shift(1))
    ).astype(int)

    # Relative volume (current vs historical)
    df['relative_volume'] = df['volume'] / df['volume'].rolling(window=50).mean()

    # High volume threshold (top 20%)
    df['high_volume_threshold'] = df['volume'].rolling(window=50).quantile(0.8)
    df['is_high_volume'] = (df['volume'] > df['high_volume_threshold']).astype(int)

    logger.debug(f"Volume trend calculations completed")

    return df


def calculate_volume_profile(
    dataframe: pd.DataFrame,
    bins: int = 20
) -> pd.DataFrame:
    """
    Calculate basic volume profile for identifying high-volume price levels.

    Args:
        dataframe: OHLCV DataFrame
        bins: Number of price bins for volume profile

    Returns:
        DataFrame with volume profile indicators added
    """
    df = dataframe.copy()

    # Create price bins
    price_min = df['low'].min()
    price_max = df['high'].max()
    price_bins = np.linspace(price_min, price_max, bins + 1)

    # Calculate volume at each price level
    volume_profile = []

    for i in range(len(df)):
        current_price = df.iloc[i]['close']
        current_volume = df.iloc[i]['volume']

        # Find which price bin this belongs to
        bin_idx = np.digitize(current_price, price_bins) - 1
        bin_idx = np.clip(bin_idx, 0, bins - 1)

        volume_profile.append({
            'price': current_price,
            'volume': current_volume,
            'price_bin': bin_idx
        })

    volume_df = pd.DataFrame(volume_profile)

    # Calculate volume at each price level
    level_volume = volume_df.groupby('price_bin')['volume'].sum()

    # Identify high-volume levels (top 30%)
    high_volume_threshold = level_volume.quantile(0.7)
    high_volume_bins = level_volume[level_volume >= high_volume_threshold].index

    # Add volume profile data to main dataframe
    for i in range(len(df)):
        current_price = df.iloc[i]['close']
        bin_idx = np.digitize(current_price, price_bins) - 1
        bin_idx = np.clip(bin_idx, 0, bins - 1)

        df.loc[df.index[i], 'volume_at_level'] = level_volume.iloc[bin_idx] if bin_idx < len(level_volume) else 0
        df.loc[df.index[i], 'is_high_volume_level'] = int(bin_idx in high_volume_bins)

    # Calculate volume-weighted average price (VWAP)
    df['vwap'] = (df['close'] * df['volume']).rolling(window=20).sum() / df['volume'].rolling(window=20).sum()

    # Distance from VWAP
    df['vwap_distance'] = (df['close'] - df['vwap']) / df['vwap'] * 100

    logger.debug(f"Volume profile calculations completed. High volume levels identified")

    return df


def calculate_on_balance_volume(
    dataframe: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate On-Balance Volume (OBV) indicator.

    Args:
        dataframe: OHLCV DataFrame

    Returns:
        DataFrame with OBV indicators added
    """
    df = dataframe.copy()

    # Calculate OBV
    price_change = df['close'].diff()
    obv = [0]  # Start with 0

    for i in range(1, len(df)):
        if price_change.iloc[i] > 0:
            obv.append(obv[-1] + df['volume'].iloc[i])
        elif price_change.iloc[i] < 0:
            obv.append(obv[-1] - df['volume'].iloc[i])
        else:
            obv.append(obv[-1])

    df['obv'] = obv

    # OBV moving averages
    df['obv_sma'] = df['obv'].rolling(window=10).mean()
    df['obv_ema'] = df['obv'].ewm(span=10).mean()

    # OBV trend
    df['obv_trend_up'] = (df['obv'] > df['obv_ema']).astype(int)
    df['obv_trend_down'] = (df['obv'] < df['obv_ema']).astype(int)

    # OBV divergences
    df['obv_divergence_strength'] = (df['obv'] - df['obv_ema']) / df['obv_ema']

    logger.debug(f"OBV calculations completed")

    return df


def generate_volume_signals(
    dataframe: pd.DataFrame,
    min_divergence_bars: int = 3,
    spike_confirmation_required: bool = True
) -> pd.DataFrame:
    """
    Generate final volume-based trading signals.

    Args:
        dataframe: DataFrame with all volume indicators
        min_divergence_bars: Minimum bars of divergence before signal
        spike_confirmation_required: Whether to require volume spike confirmation

    Returns:
        DataFrame with final volume signals
    """
    df = dataframe.copy()

    # Initialize signals
    df['volume_buy_signal'] = 0
    df['volume_sell_signal'] = 0

    # Track divergence streaks
    df['bullish_divergence_streak'] = df['bullish_divergence'].groupby((df['bullish_divergence'] == 0).cumsum()).cumsum()
    df['bearish_divergence_streak'] = df['bearish_divergence'].groupby((df['bearish_divergence'] == 0).cumsum()).cumsum()

    # Strong divergence conditions
    strong_bullish = (
        (df['strong_bullish_divergence'] == 1) |
        (df['bullish_divergence_streak'] >= min_divergence_bars)
    )

    strong_bearish = (
        (df['strong_bearish_divergence'] == 1) |
        (df['bearish_divergence_streak'] >= min_divergence_bars)
    )

    # Volume confirmation
    volume_confirmed_bullish = strong_bullish & (df['volume_trend_up'] == 1)
    volume_confirmed_bearish = strong_bearish & (df['volume_trend_down'] == 1)

    # Spike confirmation (if required)
    if spike_confirmation_required:
        spike_confirmed_bullish = volume_confirmed_bullish & (df['volume_spike'] == 1)
        spike_confirmed_bearish = volume_confirmed_bearish & (df['volume_spike'] == 1)
    else:
        spike_confirmed_bullish = volume_confirmed_bullish
        spike_confirmed_bearish = volume_confirmed_bearish

    # Final signals
    df.loc[spike_confirmed_bullish, 'volume_buy_signal'] = 1
    df.loc[spike_confirmed_bearish, 'volume_sell_signal'] = 1

    # Signal strength (combination of factors)
    df['buy_signal_strength'] = (
        df['bullish_divergence'] * 0.3 +
        df['volume_trend_up'] * 0.3 +
        df['volume_spike'] * 0.2 +
        df['significant_spike'] * 0.2
    )

    df['sell_signal_strength'] = (
        df['bearish_divergence'] * 0.3 +
        df['volume_trend_down'] * 0.3 +
        df['volume_spike'] * 0.2 +
        df['significant_spike'] * 0.2
    )

    # Count signals
    buy_signals = df['volume_buy_signal'].sum()
    sell_signals = df['volume_sell_signal'].sum()

    logger.info(f"Volume signals generated - Buy: {buy_signals}, Sell: {sell_signals}")

    return df


def apply_all_volume_indicators(
    dataframe: pd.DataFrame,
    divergence_length: int = 5,
    spike_multiplier: float = 2.0,
    require_spike_confirmation: bool = True
) -> pd.DataFrame:
    """
    Apply all volume indicators to DataFrame in sequence.

    Args:
        dataframe: OHLCV DataFrame
        divergence_length: Lookback for divergence detection
        spike_multiplier: Multiplier for spike detection
        require_spike_confirmation: Whether spikes are required for signals

    Returns:
        DataFrame with all volume indicators and signals
    """
    logger.info("Applying volume indicators...")

    # Apply all indicators
    df = calculate_volume_divergence(dataframe, length=divergence_length)
    df = detect_volume_spikes(df, spike_multiplier=spike_multiplier)
    df = calculate_volume_trend(df)
    df = calculate_volume_profile(df)
    df = calculate_on_balance_volume(df)

    # Generate final signals
    df = generate_volume_signals(df, spike_confirmation_required=require_spike_confirmation)

    logger.info("Volume indicators applied successfully")

    return df


# Utility function for testing
def test_volume_indicators():
    """
    Test function to verify volume indicators work correctly.
    """
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='H')

    # Simulate price with trend and some volatility
    price_trend = np.linspace(100, 120, 100)
    price_noise = np.random.normal(0, 1, 100)
    close_prices = price_trend + price_noise

    # Create OHLC
    high_prices = close_prices + np.random.uniform(0, 2, 100)
    low_prices = close_prices - np.random.uniform(0, 2, 100)
    open_prices = close_prices + np.random.uniform(-1, 1, 100)

    # Simulate volume (higher during big moves)
    volume_base = 1000000
    volume_volatility = np.random.uniform(0.5, 2.0, 100)
    volume = volume_base * volume_volatility

    # Create DataFrame
    test_df = pd.DataFrame({
        'date': dates,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume
    })

    # Apply indicators
    result_df = apply_all_volume_indicators(test_df)

    # Check results
    print("Test Results:")
    print(f"Total rows: {len(result_df)}")
    print(f"Buy signals: {result_df['volume_buy_signal'].sum()}")
    print(f"Sell signals: {result_df['volume_sell_signal'].sum()}")
    print(f"Volume spikes: {result_df['volume_spike'].sum()}")
    print(f"Bullish divergences: {result_df['bullish_divergence'].sum()}")
    print(f"Bearish divergences: {result_df['bearish_divergence'].sum()}")

    return result_df


if __name__ == "__main__":
    # Run test
    test_volume_indicators()