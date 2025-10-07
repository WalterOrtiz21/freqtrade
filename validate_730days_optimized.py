#!/usr/bin/env python3
"""
Validación de parámetros optimizados - 730 días completos
Test en TODO el dataset histórico (Oct 2023 - Oct 2025)

Parámetros a validar:
- RSI threshold: 21 (vs 37 original)
- Momentum threshold: -3.8% (vs -2.8% original)
- Stop Loss: -1% (vs -3% original)
- Take Profit: +150% (vs +70% original)
- Trailing activation: 0.6% (vs 2% original)
"""

import subprocess
import sys
from pathlib import Path

STRATEGY = "DynamicAggressiveHighTP"
TIMERANGE = "20231006-20251006"  # 730 días (2 años completos)

print("=" * 80)
print("VALIDACIÓN - Parámetros Optimizados en 730 Días")
print("=" * 80)
print(f"Strategy: {STRATEGY}")
print(f"Period: Oct 2023 - Oct 2025 (730 días / 2 años)")
print()
print("Parámetros optimizados activos:")
print("  ENTRY:")
print("    - RSI threshold: 21 (vs 37 original)")
print("    - Momentum threshold: -3.8% (vs -2.8% original)")
print("    - RSI period: 12 (vs 14 original)")
print("    - Momentum period: 7 (vs 5 original)")
print("    - MA period: 8 (vs 5 original)")
print()
print("  EXIT:")
print("    - Emergency SL: -8% (sin cambio)")
print("    - Stop Loss: -1% (vs -3%/-5% original)")
print("    - Take Profit: +150% (vs +70% original)")
print("    - Trailing activation: 0.6% (vs 2% original)")
print("    - Trailing distances: 0.7%, 1.3%, 5%")
print("=" * 80)
print()

freqtrade_path = Path(__file__).parent / ".venv" / "Scripts" / "freqtrade.exe"

cmd = [
    str(freqtrade_path),
    "backtesting",
    "--strategy", STRATEGY,
    "--timerange", TIMERANGE,
]

print("Ejecutando backtest en 730 días...")
print(" ".join(cmd))
print()

try:
    result = subprocess.run(cmd, cwd=Path(__file__).parent)

    print()
    print("=" * 80)
    print("Validación 730 días completada!")
    print("=" * 80)
    print()
    print("IMPORTANTE: Compara estos resultados con:")
    print("  - Training period (9 meses): +41.28%")
    print("  - Test period (3 meses): +7.64% (+30% anualizado)")
    print()
    print("Esperamos ver:")
    print("  - Retorno anualizado entre 20-35%")
    print("  - Win rate ~60-70%")
    print("  - Max drawdown <10%")
    print("=" * 80)

    sys.exit(result.returncode)

except KeyboardInterrupt:
    print("\n\nValidación interrumpida")
    sys.exit(1)
except Exception as e:
    print(f"\n\nError: {e}")
    sys.exit(1)
