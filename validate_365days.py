#!/usr/bin/env python3
"""
Validate hyperopt results - Últimos 365 días
Test period: Jul 2025 - Oct 2025 (3 meses out-of-sample)
"""

import subprocess
import sys
from pathlib import Path

STRATEGY = "DynamicAggressiveHighTP"
TIMERANGE = "20250707-20251006"  # 3 meses de test

print("=" * 60)
print("Validation - Test Period (365 días)")
print("=" * 60)
print(f"Strategy: {STRATEGY}")
print(f"Test period: Jul 2025 - Oct 2025 (3 meses)")
print("=" * 60)
print("Comparar con training period (Oct 2024 - Jul 2025)")
print("Si resultados son similares -> ROBUSTO")
print("Si test << train -> OVERFITTING")
print("=" * 60)
print()

freqtrade_path = Path(__file__).parent / ".venv" / "Scripts" / "freqtrade.exe"

cmd = [
    str(freqtrade_path),
    "backtesting",
    "--strategy", STRATEGY,
    "--timerange", TIMERANGE,
]

print(f"Running command:")
print(" ".join(cmd))
print()

try:
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    print()
    print("=" * 60)
    print("Validation Complete!")
    print("=" * 60)
    sys.exit(result.returncode)
except KeyboardInterrupt:
    print("\n\nValidation interrupted")
    sys.exit(1)
except Exception as e:
    print(f"\n\nError: {e}")
    sys.exit(1)
