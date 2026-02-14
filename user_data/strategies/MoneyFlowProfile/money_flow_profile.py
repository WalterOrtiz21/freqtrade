"""
Money Flow Profile Calculator
=============================

Port of LuxAlgo's "Money Flow Profile" indicator to Python.
Calculates volume/money flow distribution by price levels.

Key Concepts:
- POC (Point of Control): Price level with highest volume
- HVN (High Volume Node): Levels with >53% of max volume (consolidation zones)
- LVN (Low Volume Node): Levels with <37% of max volume (liquidity zones)
- Sentiment: Bullish/Bearish bias per level based on candle polarity
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class ProfileLevel:
    """Single level in the volume profile."""
    price_low: float
    price_high: float
    price_mid: float
    total_volume: float
    bullish_volume: float
    bearish_volume: float
    volume_ratio: float  # ratio to max volume (0-1)
    sentiment: float  # positive = bullish, negative = bearish
    node_type: str  # 'HVN', 'LVN', 'AVN' (average), 'POC'


@dataclass
class MoneyFlowProfileResult:
    """Complete profile calculation result."""
    levels: List[ProfileLevel]
    poc_price: float
    poc_level_idx: int
    profile_high: float
    profile_low: float
    total_volume: float
    dominant_sentiment: str  # 'bullish' or 'bearish'
    
    def get_current_level(self, price: float) -> Optional[ProfileLevel]:
        """Find which level contains the given price."""
        for level in self.levels:
            if level.price_low <= price <= level.price_high:
                return level
        return None
    
    def get_hvn_levels(self) -> List[ProfileLevel]:
        """Get all High Volume Nodes."""
        return [l for l in self.levels if l.node_type in ('HVN', 'POC')]
    
    def get_lvn_levels(self) -> List[ProfileLevel]:
        """Get all Low Volume Nodes."""
        return [l for l in self.levels if l.node_type == 'LVN']
    
    def is_price_in_lvn(self, price: float) -> bool:
        """Check if price is in a Low Volume Node."""
        level = self.get_current_level(price)
        return level is not None and level.node_type == 'LVN'
    
    def is_price_in_hvn(self, price: float) -> bool:
        """Check if price is in a High Volume Node."""
        level = self.get_current_level(price)
        return level is not None and level.node_type in ('HVN', 'POC')
    
    def get_level_sentiment(self, price: float) -> Optional[str]:
        """Get sentiment at price level: 'bullish', 'bearish', or None."""
        level = self.get_current_level(price)
        if level is None:
            return None
        return 'bullish' if level.sentiment > 0 else 'bearish'


class MoneyFlowProfile:
    """
    Calculate Volume/Money Flow Profile from OHLCV data.
    
    Ported from LuxAlgo Pine Script indicator.
    """
    
    def __init__(
        self,
        num_rows: int = 25,
        hvn_threshold: float = 0.53,
        lvn_threshold: float = 0.37,
        use_money_flow: bool = False,
        sentiment_method: str = 'polarity'  # 'polarity' or 'pressure'
    ):
        """
        Initialize profile calculator.
        
        Args:
            num_rows: Number of price levels to divide the range into
            hvn_threshold: Ratio threshold for High Volume Nodes (default 0.53 = 53%)
            lvn_threshold: Ratio threshold for Low Volume Nodes (default 0.37 = 37%)
            use_money_flow: If True, weight volume by price (money flow)
            sentiment_method: 'polarity' (close > open) or 'pressure' (buying pressure)
        """
        self.num_rows = num_rows
        self.hvn_threshold = hvn_threshold
        self.lvn_threshold = lvn_threshold
        self.use_money_flow = use_money_flow
        self.sentiment_method = sentiment_method
    
    def calculate(self, df: pd.DataFrame) -> MoneyFlowProfileResult:
        """
        Calculate the money flow profile for given OHLCV data.
        
        Args:
            df: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
            
        Returns:
            MoneyFlowProfileResult with all profile data
        """
        # Get price range
        profile_low = df['low'].min()
        profile_high = df['high'].max()
        
        if profile_high <= profile_low:
            raise ValueError("Invalid price range: high <= low")
        
        # Calculate step size for each row
        step = (profile_high - profile_low) / self.num_rows
        
        # Initialize volume arrays
        total_volume = np.zeros(self.num_rows)
        bullish_volume = np.zeros(self.num_rows)
        
        # Determine if each bar is bullish
        if self.sentiment_method == 'polarity':
            is_bullish = (df['close'] > df['open']).values
        else:  # pressure
            is_bullish = ((df['close'] - df['low']) > (df['high'] - df['close'])).values
        
        # Process each bar
        for i in range(len(df)):
            bar_high = df['high'].iloc[i]
            bar_low = df['low'].iloc[i]
            bar_volume = df['volume'].iloc[i]
            bar_range = bar_high - bar_low
            
            if bar_range <= 0 or bar_volume <= 0:
                continue
            
            # For each price level, calculate what portion of the bar is in it
            for level_idx in range(self.num_rows):
                level_low = profile_low + level_idx * step
                level_high = level_low + step
                level_mid = level_low + step * 0.5
                
                # Check if bar touches this level
                if bar_high >= level_low and bar_low < level_high:
                    # Calculate volume portion in this level
                    # (Ported from Pine Script lines 189-196)
                    if bar_low >= level_low and bar_high > level_high:
                        # Bar starts in level, extends above
                        vol_portion = (level_high - bar_low) / bar_range
                    elif bar_high <= level_high and bar_low < level_low:
                        # Bar ends in level, extends below
                        vol_portion = (bar_high - level_low) / bar_range
                    elif bar_low >= level_low and bar_high <= level_high:
                        # Bar completely within level
                        vol_portion = 1.0
                    else:
                        # Bar spans entire level
                        vol_portion = step / bar_range
                    
                    # Apply volume (optionally weighted by price for money flow)
                    if self.use_money_flow:
                        vol_value = bar_volume * vol_portion * level_mid
                    else:
                        vol_value = bar_volume * vol_portion
                    
                    total_volume[level_idx] += vol_value
                    
                    if is_bullish[i]:
                        bullish_volume[level_idx] += vol_value
        
        # Calculate bearish volume and sentiment
        bearish_volume = total_volume - bullish_volume
        
        # Sentiment: difference between bull and bear (positive = bullish)
        # Pine Script line 226: bbp = 2 * rpVSB.get(l) - rpVST.get(l)
        sentiment = 2 * bullish_volume - total_volume
        
        # Find max volume for ratio calculations
        max_volume = total_volume.max() if total_volume.max() > 0 else 1.0
        volume_ratios = total_volume / max_volume
        
        # Find POC (level with max volume)
        poc_idx = int(np.argmax(total_volume))
        poc_price = profile_low + (poc_idx + 0.5) * step
        
        # Build level objects
        levels = []
        for idx in range(self.num_rows):
            level_low = profile_low + idx * step
            level_high = level_low + step
            level_mid = level_low + step * 0.5
            
            ratio = volume_ratios[idx]
            
            # Classify node type
            if idx == poc_idx:
                node_type = 'POC'
            elif ratio > self.hvn_threshold:
                node_type = 'HVN'
            elif ratio < self.lvn_threshold:
                node_type = 'LVN'
            else:
                node_type = 'AVN'
            
            levels.append(ProfileLevel(
                price_low=level_low,
                price_high=level_high,
                price_mid=level_mid,
                total_volume=total_volume[idx],
                bullish_volume=bullish_volume[idx],
                bearish_volume=bearish_volume[idx],
                volume_ratio=ratio,
                sentiment=sentiment[idx],
                node_type=node_type
            ))
        
        # Determine overall dominant sentiment
        total_bull = bullish_volume.sum()
        total_bear = bearish_volume.sum()
        dominant_sentiment = 'bullish' if total_bull > total_bear else 'bearish'
        
        return MoneyFlowProfileResult(
            levels=levels,
            poc_price=poc_price,
            poc_level_idx=poc_idx,
            profile_high=profile_high,
            profile_low=profile_low,
            total_volume=total_volume.sum(),
            dominant_sentiment=dominant_sentiment
        )
    
    def calculate_for_dataframe(
        self, 
        df: pd.DataFrame, 
        lookback: int = 200
    ) -> pd.DataFrame:
        """
        Add profile columns to dataframe for each row (rolling calculation).
        
        This is optimized for Freqtrade - calculates profile for the lookback
        window ending at each row.
        
        Args:
            df: Full OHLCV dataframe
            lookback: Number of bars to use for profile calculation
            
        Returns:
            DataFrame with added columns:
            - mfp_poc: POC price
            - mfp_in_lvn: Boolean, price is in LVN
            - mfp_in_hvn: Boolean, price is in HVN  
            - mfp_level_sentiment: Sentiment at current price level
            - mfp_price_vs_poc: 1 if price > POC, -1 if price < POC
            - mfp_dominant_sentiment: Overall profile sentiment
        """
        n = len(df)
        
        # Initialize output columns
        poc_prices = np.full(n, np.nan)
        in_lvn = np.zeros(n, dtype=bool)
        in_hvn = np.zeros(n, dtype=bool)
        level_sentiment = np.full(n, np.nan)
        price_vs_poc = np.zeros(n)
        dominant_sentiment = np.zeros(n)  # 1 = bullish, -1 = bearish
        
        # Calculate profile for each position (sliding window)
        for i in range(lookback, n):
            try:
                window_df = df.iloc[i - lookback:i].copy()
                profile = self.calculate(window_df)
                
                current_price = df['close'].iloc[i]
                
                poc_prices[i] = profile.poc_price
                in_lvn[i] = profile.is_price_in_lvn(current_price)
                in_hvn[i] = profile.is_price_in_hvn(current_price)
                
                sent = profile.get_level_sentiment(current_price)
                level_sentiment[i] = 1 if sent == 'bullish' else (-1 if sent == 'bearish' else 0)
                
                price_vs_poc[i] = 1 if current_price > profile.poc_price else -1
                dominant_sentiment[i] = 1 if profile.dominant_sentiment == 'bullish' else -1
                
            except Exception as e:
                logger.warning(f"Profile calculation failed at index {i}: {e}")
                continue
        
        # Add columns to dataframe
        result_df = df.copy()
        result_df['mfp_poc'] = poc_prices
        result_df['mfp_in_lvn'] = in_lvn
        result_df['mfp_in_hvn'] = in_hvn
        result_df['mfp_level_sentiment'] = level_sentiment
        result_df['mfp_price_vs_poc'] = price_vs_poc
        result_df['mfp_dominant_sentiment'] = dominant_sentiment
        
        return result_df
