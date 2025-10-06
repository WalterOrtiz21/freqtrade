"""
Convert Pacifica historical data to Freqtrade format.

This script converts JSON data from Pacifica SDK to Freqtrade-compatible format
for offline backtesting and hyperopt optimization.

Usage:
    python convert_pacifica_to_freqtrade.py
"""

import json
import pandas as pd
from pathlib import Path
import sys


def convert_pacifica_to_freqtrade(input_file, output_dir, symbol, timeframe='5m'):
    """
    Convert Pacifica JSON format to Freqtrade JSON format.

    Args:
        input_file: Path to Pacifica JSON file
        output_dir: Output directory for Freqtrade data
        symbol: Symbol name (e.g., 'BTC')
        timeframe: Timeframe (default: '5m')
    """
    print(f"Converting {symbol}...")

    # Load Pacifica data
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"  ERROR: File not found: {input_file}")
        return False
    except json.JSONDecodeError as e:
        print(f"  ERROR: Invalid JSON in {input_file}: {e}")
        return False

    # Extract candle data (handle different JSON structures)
    candles = None

    # Try structure 1: {symbol: [...]}
    if isinstance(data, dict) and symbol.upper() in data:
        candles = data[symbol.upper()]

    # Try structure 2: {data: [...]}
    elif isinstance(data, dict) and 'data' in data:
        candles = data['data']

    # Try structure 3: [...]
    elif isinstance(data, list):
        candles = data

    if candles is None:
        print(f"  ERROR: Could not find candle data in {input_file}")
        print(f"  JSON keys: {list(data.keys()) if isinstance(data, dict) else 'list'}")
        return False

    # Convert to DataFrame
    df = pd.DataFrame(candles)

    # Verify required columns exist
    required_base = ['timestamp']
    if not all(col in df.columns for col in required_base):
        print(f"  ERROR: Missing required columns. Found: {list(df.columns)}")
        return False

    # Handle different column naming conventions
    # Pacifica might use 'price' instead of 'close'
    if 'price' in df.columns and 'close' not in df.columns:
        df['close'] = df['price']

    # If OHLC columns don't exist, create them from price
    if 'open' not in df.columns:
        df['open'] = df['close']
    if 'high' not in df.columns:
        df['high'] = df['close']
    if 'low' not in df.columns:
        df['low'] = df['close']
    if 'volume' not in df.columns:
        df['volume'] = 0  # Default if volume not available

    # Convert timestamp to milliseconds (Freqtrade format)
    # Check if timestamp is already in milliseconds or seconds
    sample_ts = df['timestamp'].iloc[0]

    if isinstance(sample_ts, str):
        # Parse datetime string
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['timestamp'] = (df['timestamp'].astype(int) // 10**6)
    elif sample_ts > 2000000000:  # Already in milliseconds
        df['timestamp'] = df['timestamp'].astype(int)
    else:  # In seconds, convert to milliseconds
        df['timestamp'] = (df['timestamp'] * 1000).astype(int)

    # Convert numeric columns to float
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)

    # Select and order columns for Freqtrade
    # Freqtrade expects: [timestamp, open, high, low, close, volume]
    df_freqtrade = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()

    # Sort by timestamp
    df_freqtrade = df_freqtrade.sort_values('timestamp')

    # Remove duplicates
    df_freqtrade = df_freqtrade.drop_duplicates(subset=['timestamp'])

    # Convert to list of lists (Freqtrade format)
    freqtrade_data = df_freqtrade.values.tolist()

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Freqtrade filename format: SYMBOL_PAIR-TIMEFRAME.json
    output_file = output_path / f"{symbol.upper()}_USDT-{timeframe}.json"

    # Save in Freqtrade format
    with open(output_file, 'w') as f:
        json.dump(freqtrade_data, f, indent=2)

    print(f"  SUCCESS: {len(freqtrade_data)} candles -> {output_file}")
    return True


def main():
    """Convert all Pacifica historical data to Freqtrade format."""

    print("=" * 60)
    print("Pacifica to Freqtrade Data Converter")
    print("=" * 60)

    # Configuration
    pacifica_data_dir = Path('../pacifica-python-sdk/data/historical')
    freqtrade_data_dir = Path('user_data/data/binance')  # Freqtrade data location
    symbols = ['BTC', 'ETH', 'SOL', 'BNB', 'SUI', 'LTC']
    timeframe = '5m'

    # Check if Pacifica data directory exists
    if not pacifica_data_dir.exists():
        print(f"\nERROR: Pacifica data directory not found: {pacifica_data_dir}")
        print("Please ensure you're running this from /trading/freqtrade directory")
        return 1

    # Convert each symbol
    success_count = 0
    fail_count = 0

    for symbol in symbols:
        # Try different filename patterns
        patterns = [
            pacifica_data_dir / f"{symbol.lower()}_usdt_365days_5m.json",
            pacifica_data_dir / f"{symbol.lower()}_usdt_730days_5m.json",
            pacifica_data_dir / f"{symbol.upper()}_USDT_365days_5m.json",
        ]

        converted = False
        for pattern in patterns:
            if pattern.exists():
                if convert_pacifica_to_freqtrade(pattern, freqtrade_data_dir, symbol, timeframe):
                    success_count += 1
                    converted = True
                else:
                    fail_count += 1
                break

        if not converted:
            print(f"Converting {symbol}...")
            print(f"  ERROR: No data file found. Tried:")
            for pattern in patterns:
                print(f"    - {pattern}")
            fail_count += 1

    # Summary
    print("\n" + "=" * 60)
    print("Conversion Summary")
    print("=" * 60)
    print(f"Success: {success_count}/{len(symbols)}")
    print(f"Failed:  {fail_count}/{len(symbols)}")

    if success_count > 0:
        print(f"\nConverted data saved to: {freqtrade_data_dir}")
        print("\nNext steps:")
        print("1. Install Freqtrade (requires Python 3.11+):")
        print("   ./setup.sh -i")
        print("\n2. Run backtest:")
        print("   freqtrade backtesting --strategy DynamicAggressiveHighTP --timerange 20240101-20241231")
        print("\n3. Run hyperopt (parameter optimization):")
        print("   freqtrade hyperopt --strategy DynamicAggressiveHighTP --hyperopt-loss SharpeHyperOptLoss --epochs 1000")

    return 0 if fail_count == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
