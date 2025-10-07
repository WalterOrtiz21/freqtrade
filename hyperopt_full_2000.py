#!/usr/bin/env python3
"""
Hyperopt COMPLETO - 2000 iteraciones
Optimiza TODOS los parámetros de DYNAMIC-Aggressive High TP

Parámetros optimizables:
- Entry: RSI threshold, momentum threshold, RSI period, momentum period, MA period
- Exit: Emergency SL, Stop Loss, Take Profit, Trailing activation
- Trailing: Low/Mid/High distances

Train period: Oct 2024 - Jul 2025 (9 meses)
Test period: Jul 2025 - Oct 2025 (3 meses)
"""

import subprocess
import sys
from pathlib import Path

STRATEGY = "DynamicAggressiveHighTP"
TRAIN_TIMERANGE = "20241006-20250707"  # 9 meses train
TEST_TIMERANGE = "20250707-20251006"   # 3 meses test
EPOCHS = 2000

print("=" * 80)
print("HYPEROPT COMPLETO - Optimización Total de DYNAMIC")
print("=" * 80)
print(f"Strategy: {STRATEGY}")
print(f"Epochs: {EPOCHS}")
print(f"Train period: Oct 2024 - Jul 2025 (9 meses)")
print(f"Test period: Jul 2025 - Oct 2025 (3 meses)")
print()
print("Parámetros optimizables:")
print("  ENTRY (buy space):")
print("    - RSI threshold: 15-50")
print("    - Momentum threshold: -10% a -0.5%")
print("    - RSI period: 10-20")
print("    - Momentum period: 3-10")
print("    - MA period: 3-15")
print()
print("  EXIT (sell space):")
print("    - Emergency SL: -15% a -5%")
print("    - Stop Loss: -15% a -1%")
print("    - Take Profit: +20% a +200%")
print("    - Trailing activation: 0.5% a 10%")
print("    - Trailing distances (Low/Mid/High): 0.1% a 20%")
print("=" * 80)
print()

freqtrade_path = Path(__file__).parent / ".venv" / "Scripts" / "freqtrade.exe"

print("FASE 1: OPTIMIZACIÓN (Train period)")
print("-" * 80)

cmd_hyperopt = [
    str(freqtrade_path),
    "hyperopt",
    "--strategy", STRATEGY,
    "--hyperopt-loss", "SharpeHyperOptLoss",
    "--spaces", "buy", "sell",
    "--epochs", str(EPOCHS),
    "--timerange", TRAIN_TIMERANGE,
    "--min-trades", "50",  # Requiere al menos 50 trades para ser válido
]

print("Comando hyperopt:")
print(" ".join(cmd_hyperopt))
print()

try:
    result = subprocess.run(cmd_hyperopt, cwd=Path(__file__).parent)

    if result.returncode != 0:
        print("\n\nERROR: Hyperopt falló")
        sys.exit(1)

    print()
    print("=" * 80)
    print("FASE 2: VALIDACIÓN (Test period)")
    print("=" * 80)
    print()
    print("IMPORTANTE: Revisa los parámetros óptimos arriba y luego ejecuta:")
    print(f"  .venv/Scripts/freqtrade.exe backtesting --strategy {STRATEGY} --timerange {TEST_TIMERANGE}")
    print()
    print("Compara resultados:")
    print("  - Train ≈ Test = ROBUSTO (buenos parámetros)")
    print("  - Train >> Test = OVERFITTING (rechazar)")
    print("  - Train << Test = SUERTE (repetir experimento)")
    print()
    print("=" * 80)

except KeyboardInterrupt:
    print("\n\nOptimización interrumpida")
    sys.exit(1)
except Exception as e:
    print(f"\n\nError: {e}")
    sys.exit(1)
