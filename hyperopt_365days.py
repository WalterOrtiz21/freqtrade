#!/usr/bin/env python3
"""
Hyperopt - Últimos 365 días (más reciente y relevante)
Training: 2024-10-06 a 2025-07-06 (9 meses)
Test: 2025-07-07 a 2025-10-06 (3 meses)
"""

import subprocess
import sys
from pathlib import Path

# Configuration
STRATEGY = "DynamicAggressiveHighTP"
LOSS_FUNCTION = "SharpeHyperOptLoss"
SPACES = ["buy", "sell"]
EPOCHS = 1000
TIMERANGE = "20241006-20250706"  # Training: últimos 12 meses menos 3 para test
MIN_TRADES = 100

print("=" * 60)
print("Hyperopt - Últimos 365 Días (DATOS RECIENTES)")
print("=" * 60)
print(f"Strategy: {STRATEGY}")
print(f"Training period: Oct 2024 - Jul 2025 (9 meses)")
print(f"Test period: Jul 2025 - Oct 2025 (3 meses)")
print(f"Epochs: {EPOCHS}")
print("=" * 60)
print("Ventajas:")
print("  ✓ Datos MÁS RECIENTES (condiciones actuales)")
print("  ✓ Relevante para trading actual")
print("  ✓ Incluye datos de 2025")
print("=" * 60)
print(f"Estimated time: 20-40 minutes")
print("=" * 60)
print()

# Build command
freqtrade_path = Path(__file__).parent / ".venv" / "Scripts" / "freqtrade.exe"

cmd = [
    str(freqtrade_path),
    "hyperopt",
    "--strategy", STRATEGY,
    "--hyperopt-loss", LOSS_FUNCTION,
    "--spaces", *SPACES,
    "--epochs", str(EPOCHS),
    "--timerange", TIMERANGE,
    "--min-trades", str(MIN_TRADES),
]

print(f"Running command:")
print(" ".join(cmd))
print()

# Run hyperopt
try:
    result = subprocess.run(cmd, cwd=Path(__file__).parent)

    print()
    print("=" * 60)
    print("Hyperopt Completado!")
    print("=" * 60)
    print("Siguiente paso: Validar en datos out-of-sample")
    print("  python validate_365days.py")
    print("=" * 60)

    sys.exit(result.returncode)
except KeyboardInterrupt:
    print("\n\nHyperopt interrupted by user")
    sys.exit(1)
except Exception as e:
    print(f"\n\nError running hyperopt: {e}")
    sys.exit(1)
