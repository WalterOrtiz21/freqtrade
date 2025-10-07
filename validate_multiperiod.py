#!/usr/bin/env python3
"""
Validación Multi-Período - Walk-Forward Analysis
Prueba los parámetros optimizados en 8 períodos diferentes de 3 meses

Esto detecta:
- Overfitting (si solo funciona en ciertos períodos)
- Robustez (si funciona en todos los períodos)
- Adaptabilidad a diferentes condiciones de mercado
"""

import subprocess
import sys
from pathlib import Path

STRATEGY = "DynamicAggressiveHighTP"

# Períodos de 3 meses cada uno (8 períodos = 24 meses)
PERIODS = [
    ("Q4 2023", "20231006-20240106"),
    ("Q1 2024", "20240106-20240406"),
    ("Q2 2024", "20240406-20240706"),
    ("Q3 2024", "20240706-20241006"),
    ("Q4 2024", "20241006-20250106"),
    ("Q1 2025", "20250106-20250406"),
    ("Q2 2025", "20250406-20250706"),
    ("Q3 2025", "20250706-20251006"),
]

print("=" * 80)
print("VALIDACIÓN MULTI-PERÍODO - Walk-Forward Analysis")
print("=" * 80)
print(f"Strategy: {STRATEGY}")
print(f"Períodos: 8 x 3 meses (Oct 2023 - Oct 2025)")
print()
print("Objetivo: Verificar robustez en diferentes condiciones de mercado")
print("=" * 80)
print()

freqtrade_path = Path(__file__).parent / ".venv" / "Scripts" / "freqtrade.exe"

results = []

for period_name, timerange in PERIODS:
    print("=" * 80)
    print(f"PERÍODO: {period_name}")
    print(f"Timerange: {timerange}")
    print("=" * 80)

    cmd = [
        str(freqtrade_path),
        "backtesting",
        "--strategy", STRATEGY,
        "--timerange", timerange,
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True
        )

        # Extraer métricas clave del output
        output = result.stdout

        # Buscar Total profit %
        profit_line = [line for line in output.split('\n') if 'Total profit %' in line]
        if profit_line:
            profit = profit_line[0].split('|')[-2].strip()
            results.append((period_name, timerange, profit))
            print(f"[OK] Completado - Profit: {profit}")
        else:
            results.append((period_name, timerange, "N/A"))
            print(f"[OK] Completado - No se pudo extraer profit")

    except Exception as e:
        print(f"[ERROR] {e}")
        results.append((period_name, timerange, "ERROR"))

    print()

# Resumen de resultados
print("=" * 80)
print("RESUMEN DE RESULTADOS - Multi-Período")
print("=" * 80)
print()
print(f"{'Período':<15} {'Timerange':<25} {'Profit %':>12}")
print("-" * 80)

for period_name, timerange, profit in results:
    print(f"{period_name:<15} {timerange:<25} {profit:>12}")

print()
print("=" * 80)
print("ANÁLISIS:")
print()
print("ROBUSTO si:")
print("  - Mayoría de períodos son positivos (>60%)")
print("  - No hay caídas extremas en ningún período")
print("  - Resultados relativamente consistentes")
print()
print("OVERFITTING si:")
print("  - Solo 1-2 períodos son muy positivos")
print("  - Resto de períodos son negativos")
print("  - Alta variabilidad entre períodos")
print("=" * 80)

print()
input("Presiona Enter para salir...")
