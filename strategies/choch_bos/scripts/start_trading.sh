#!/bin/bash

# CHoCH/BOS Strategy Live Trading Launcher
# Usage: ./start_trading.sh [exchange] [timeframe] [dry_run]
#
# Examples:
#   ./start_trading.sh binance_futures 15m dry    # Dry run with Binance Futures on 15m
#   ./start_trading.sh binance_spot 5m dry         # Dry run with Binance Spot on 5m
#   ./start_trading.sh hyperliquid 15m live        # LIVE trading with Hyperliquid on 15m
#   ./start_trading.sh                             # Interactive mode

FREQTRADE_PATH="/c/Users/walte/AppData/Local/Programs/Python/Python312/Scripts/freqtrade.exe"
STRATEGY="CHoCHBOSStrategy"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
CONFIG_DIR="$SCRIPT_DIR/../configs"
BASE_CONFIG="$CONFIG_DIR/config_choch_bos_live.json"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to display header
show_header() {
    echo -e "${BLUE}============================================${NC}"
    echo -e "${BLUE}  CHoCH/BOS Strategy Live Trading Launcher${NC}"
    echo -e "${BLUE}============================================${NC}"
    echo ""
}

# Function to select exchange
select_exchange() {
    echo -e "${YELLOW}Select Exchange:${NC}"
    echo "1) Binance Futures (recommended for CHoCH/BOS)"
    echo "2) Binance Spot"
    echo "3) Hyperliquid (requires setup)"
    echo ""
    read -p "Enter choice [1-3]: " exchange_choice

    case $exchange_choice in
        1) EXCHANGE="binance_futures" ;;
        2) EXCHANGE="binance_spot" ;;
        3) EXCHANGE="hyperliquid" ;;
        *) echo -e "${RED}Invalid choice${NC}"; exit 1 ;;
    esac
}

# Function to select timeframe
select_timeframe() {
    echo ""
    echo -e "${YELLOW}Select Timeframe:${NC}"
    echo "1) 15m (recommended - optimized parameters)"
    echo "2) 5m (more trades, requires optimization)"
    echo ""
    read -p "Enter choice [1-2]: " tf_choice

    case $tf_choice in
        1) TIMEFRAME="15m" ;;
        2) TIMEFRAME="5m" ;;
        *) echo -e "${RED}Invalid choice${NC}"; exit 1 ;;
    esac
}

# Function to select mode
select_mode() {
    echo ""
    echo -e "${YELLOW}Select Mode:${NC}"
    echo "1) Dry Run (paper trading - recommended for testing)"
    echo "2) LIVE Trading (real money - USE WITH CAUTION!)"
    echo ""
    read -p "Enter choice [1-2]: " mode_choice

    case $mode_choice in
        1) MODE="dry" ;;
        2) MODE="live"
           echo ""
           echo -e "${RED}⚠️  WARNING: You are about to start LIVE trading!${NC}"
           echo -e "${RED}⚠️  Real money will be at risk!${NC}"
           echo ""
           read -p "Type 'YES' to confirm LIVE trading: " confirm
           if [ "$confirm" != "YES" ]; then
               echo "Cancelled."
               exit 0
           fi
           ;;
        *) echo -e "${RED}Invalid choice${NC}"; exit 1 ;;
    esac
}

# Parse command line arguments or interactive mode
show_header

if [ $# -eq 0 ]; then
    # Interactive mode
    select_exchange
    select_timeframe
    select_mode
else
    # Command line mode
    EXCHANGE=$1
    TIMEFRAME=${2:-15m}
    MODE=${3:-dry}
fi

# Set config files based on exchange
case $EXCHANGE in
    binance_futures)
        CONFIG_ARGS="--config $BASE_CONFIG"
        ;;
    binance_spot)
        CONFIG_ARGS="--config $BASE_CONFIG --config $CONFIG_DIR/config.exchange.binance_spot.json"
        ;;
    hyperliquid)
        CONFIG_ARGS="--config $BASE_CONFIG --config $CONFIG_DIR/config.exchange.hyperliquid.json"
        ;;
    *)
        echo -e "${RED}Invalid exchange: $EXCHANGE${NC}"
        echo "Valid options: binance_futures, binance_spot, hyperliquid"
        exit 1
        ;;
esac

# Check if base config exists
if [ ! -f "$BASE_CONFIG" ]; then
    echo -e "${RED}Base config not found: $BASE_CONFIG${NC}"
    exit 1
fi

# Update dry_run setting if needed
if [ "$MODE" == "live" ]; then
    echo -e "${YELLOW}Updating config for LIVE trading...${NC}"
    # Create a temporary modified config
    sed 's/"dry_run": true/"dry_run": false/' "$BASE_CONFIG" > "${BASE_CONFIG}.tmp"
    CONFIG_ARGS="--config ${BASE_CONFIG}.tmp $(echo $CONFIG_ARGS | cut -d' ' -f3-)"
fi

# Display configuration summary
echo ""
echo -e "${GREEN}Configuration:${NC}"
echo "  Exchange:  $EXCHANGE"
echo "  Timeframe: $TIMEFRAME"
echo "  Strategy:  $STRATEGY"
echo "  Mode:      $([ "$MODE" == "live" ] && echo "LIVE TRADING" || echo "DRY RUN")"
echo "  Config:    $CONFIG_ARGS"
echo ""

# Final confirmation
read -p "Press ENTER to start trading or CTRL+C to cancel..."

# Start freqtrade
echo ""
echo -e "${GREEN}Starting Freqtrade...${NC}"
echo ""

cd "$PROJECT_ROOT"

"$FREQTRADE_PATH" trade \
    --strategy "$STRATEGY" \
    $CONFIG_ARGS \
    --timeframe "$TIMEFRAME" \
    --logfile "user_data/logs/freqtrade_${EXCHANGE}_${TIMEFRAME}_$(date +%Y%m%d_%H%M%S).log"

# Cleanup temporary config if created
if [ -f "${BASE_CONFIG}.tmp" ]; then
    rm "${BASE_CONFIG}.tmp"
fi
