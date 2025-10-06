#!/usr/bin/env python3
"""
Validate hyperopt results - 2 años
Test period: May 2025 - Oct 2025 (6 meses out-of-sample)
"""

import subprocess
import sys
from pathlib import Path

STRATEGY = "DynamicAggressiveHighTP"
TIMERANGE = "20250501-20251006"  # 6 meses de test

print("=" * 60)
print("Validation - Test Period (2 años)")
print("=" * 60)
print(f"Strategy: {STRATEGY}")
print(f"Test period: May 2025 - Oct 2025 (6 meses)")
print("=" * 60)
print("Comparar con training period (Oct 2023 - Apr 2025)")
print("Si resultados son similares → ROBUSTO ✓")
print("Si test << train → OVERFITTING ✗")
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
