#!/bin/bash
# Script para subir datos históricos a VPS
# Uso: ./upload_data_to_vps.sh USER@VPS_IP

if [ -z "$1" ]; then
    echo "Error: Proporciona usuario@ip de la VPS"
    echo "Uso: ./upload_data_to_vps.sh user@123.456.789.0"
    exit 1
fi

VPS=$1

echo "=== Comprimiendo datos históricos ==="
cd "$(dirname "$0")"
tar -czf historical_data.tar.gz user_data/data/binance/

echo "=== Subiendo a VPS ==="
scp historical_data.tar.gz $VPS:~/freqtrade/

echo "=== Descomprimiendo en VPS ==="
ssh $VPS "cd ~/freqtrade && tar -xzf historical_data.tar.gz && rm historical_data.tar.gz"

echo "=== Limpiando archivo local ==="
rm historical_data.tar.gz

echo "=== Listo! ==="
echo "Datos históricos disponibles en VPS"
echo "Para verificar, ejecuta en VPS:"
echo "  ls -lh ~/freqtrade/user_data/data/binance/futures/"
