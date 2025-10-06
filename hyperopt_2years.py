#!/usr/bin/env python3
"""
Hyperopt - 2 años completos (más robusto, diferentes condiciones de mercado)
Training: 2023-10-06 a 2025-04-30 (18 meses)
Test: 2025-05-01 a 2025-10-06 (6 meses)
"""

import subprocess
import sys
from pathlib import Path

# Configuration
STRATEGY = "DynamicAggressiveHighTP"
LOSS_FUNCTION = "SharpeHyperOptLoss"
SPACES = ["buy", "sell"]
EPOCHS = 1000
TIMERANGE = "20231006-20250430"  # Training: 18 meses (75% de 2 años)
MIN_TRADES = 100

print("=" * 60)
print("Hyperopt - 2 Años Completos (MÁS ROBUSTO)")
print("=" * 60)
print(f"Strategy: {STRATEGY}")
print(f"Training period: Oct 2023 - Apr 2025 (18 meses)")
print(f"Test period: May 2025 - Oct 2025 (6 meses)")
print(f"Epochs: {EPOCHS}")
print("=" * 60)
print("Ventajas:")
print("  ✓ MÁS DATOS = mejor validación estadística")
print("  ✓ Incluye diferentes condiciones de mercado")
print("  ✓ Reduce riesgo de overfitting")
print("  ✓ Abarca tanto bear como bull markets")
print("=" * 60)
print(f"Estimated time: 30-60 minutes")
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
    print("  python validate_2years.py")
    print("=" * 60)

    sys.exit(result.returncode)
except KeyboardInterrupt:
    print("\n\nHyperopt interrupted by user")
    sys.exit(1)
except Exception as e:
    print(f"\n\nError running hyperopt: {e}")
    sys.exit(1)
