#!/usr/bin/env python3
"""
Script para verificar el progreso de hyperopt y detectar si se han probado
todas las combinaciones de parámetros.

USO:
    python check_hyperopt_completion.py
"""

import json
from pathlib import Path
from datetime import datetime

PROGRESS_FILE = Path("hyperopt_progress_5000.json")
RESULTS_DIR = Path("user_data/hyperopt_results")

def load_progress():
    """Load progress from JSON file"""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, "r") as f:
            return json.load(f)
    return None

def analyze_hyperopt_results():
    """Analyze hyperopt results from Freqtrade's output"""
    if not RESULTS_DIR.exists():
        return None

    # Find latest results file
    result_files = sorted(RESULTS_DIR.glob("*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)

    if not result_files:
        return None

    # Freqtrade stores results in pickle format, we'll just count files
    return len(result_files)

def estimate_parameter_space():
    """
    Estimate total parameter space for DYNAMIC strategy.

    ENTRY (buy):
    - RSI threshold: 15-50 (integers) = 36 values
    - Momentum threshold: -10.0 to -0.5 (step 0.1) = 96 values
    - RSI period: 10-20 (integers) = 11 values
    - Momentum period: 3-10 (integers) = 8 values
    - MA period: 3-15 (integers) = 13 values

    EXIT (sell):
    - Emergency SL: -0.15 to -0.05 (step 0.01) = 11 values
    - Stop Loss: -0.15 to -0.01 (step 0.01) = 15 values
    - Take Profit: 0.20 to 2.00 (step 0.10) = 19 values
    - Trailing activation: 0.005 to 0.10 (step 0.005) = 20 values
    - Trailing dist low: 0.001 to 0.01 (step 0.001) = 10 values
    - Trailing dist mid: 0.01 to 0.05 (step 0.005) = 9 values
    - Trailing dist high: 0.05 to 0.20 (step 0.01) = 16 values

    Total combinations = 36 * 96 * 11 * 8 * 13 * 11 * 15 * 19 * 20 * 10 * 9 * 16
                      = ~2.4 trillion combinations

    In practice, Freqtrade uses intelligent search (Bayesian optimization),
    not brute force. It typically finds optimal parameters in 1000-5000 epochs.
    """

    param_space = {
        "entry": {
            "rsi_threshold": 36,      # 15-50
            "momentum_threshold": 96,  # -10.0 to -0.5, step 0.1
            "rsi_period": 11,         # 10-20
            "momentum_period": 8,     # 3-10
            "ma_period": 13,          # 3-15
        },
        "exit": {
            "emergency_sl": 11,       # -0.15 to -0.05
            "stop_loss": 15,          # -0.15 to -0.01
            "take_profit": 19,        # 0.20 to 2.00
            "trailing_activation": 20, # 0.005 to 0.10
            "trailing_low": 10,       # 0.001 to 0.01
            "trailing_mid": 9,        # 0.01 to 0.05
            "trailing_high": 16,      # 0.05 to 0.20
        }
    }

    # Calculate total combinations
    entry_combinations = 1
    for param, values in param_space["entry"].items():
        entry_combinations *= values

    exit_combinations = 1
    for param, values in param_space["exit"].items():
        exit_combinations *= values

    total_combinations = entry_combinations * exit_combinations

    return {
        "entry_combinations": entry_combinations,
        "exit_combinations": exit_combinations,
        "total_combinations": total_combinations,
        "param_space": param_space
    }

def format_number(n):
    """Format large numbers with thousands separators"""
    if n >= 1_000_000_000_000:
        return f"{n/1_000_000_000_000:.2f} trillion"
    elif n >= 1_000_000_000:
        return f"{n/1_000_000_000:.2f} billion"
    elif n >= 1_000_000:
        return f"{n/1_000_000:.2f} million"
    elif n >= 1_000:
        return f"{n/1_000:.2f}k"
    return str(n)

def main():
    print()
    print("=" * 80)
    print("ANÁLISIS DE PROGRESO HYPEROPT - 5000 epochs")
    print("=" * 80)
    print()

    # Load progress
    progress = load_progress()

    if not progress:
        print("❌ No se encontró archivo de progreso")
        print(f"   Archivo esperado: {PROGRESS_FILE}")
        print()
        print("Ejecuta primero: python hyperopt_full_5000.py")
        return

    # Display progress
    epochs_completed = progress.get("epochs_completed", 0)
    total_epochs = 5000

    print(f"📊 PROGRESO GENERAL")
    print("-" * 80)
    print(f"Epochs completados: {epochs_completed:,} / {total_epochs:,}")
    print(f"Progreso: {epochs_completed/total_epochs*100:.1f}%")
    print(f"Epochs restantes: {total_epochs - epochs_completed:,}")
    print()

    # Display run history
    runs = progress.get("runs", [])
    if runs:
        print(f"📝 HISTORIAL DE EJECUCIONES")
        print("-" * 80)
        print(f"Total de ejecuciones: {len(runs)}")
        print()

        for i, run in enumerate(runs, 1):
            success = "✅" if run.get("success") else "❌"
            epochs = run.get("epochs_completed", run.get("epochs_attempted", 0))
            start = run.get("start", "N/A")
            end = run.get("end", "N/A")

            print(f"{success} Run {i}: {epochs} epochs")
            print(f"   Start: {start}")
            print(f"   End:   {end}")

            if not run.get("success"):
                print(f"   Error: {run.get('error', 'Unknown')}")

            print()

    # Estimate parameter space
    space = estimate_parameter_space()

    print(f"🔢 ESPACIO DE PARÁMETROS")
    print("-" * 80)
    print(f"Combinaciones de entrada: {format_number(space['entry_combinations'])}")
    print(f"Combinaciones de salida:  {format_number(space['exit_combinations'])}")
    print(f"Total combinaciones:      {format_number(space['total_combinations'])}")
    print()

    # Coverage analysis
    coverage = (epochs_completed / space['total_combinations']) * 100

    print(f"📈 ANÁLISIS DE COBERTURA")
    print("-" * 80)
    print(f"Epochs completados: {epochs_completed:,}")
    print(f"Espacio total:      {format_number(space['total_combinations'])}")
    print(f"Cobertura:          {coverage:.8f}%")
    print()

    # Interpretation
    print(f"💡 INTERPRETACIÓN")
    print("-" * 80)

    if coverage < 0.000001:
        print("Estado: EXPLORANDO")
        print()
        print("El espacio de parámetros es ENORME (~2.4 trillion combinaciones).")
        print("Hyperopt usa optimización Bayesiana, NO fuerza bruta.")
        print()
        print("Significado:")
        print("  - Freqtrade NO probará todas las combinaciones (imposible)")
        print("  - Buscará las mejores usando búsqueda inteligente")
        print("  - 5000 epochs es suficiente para encontrar parámetros óptimos")
        print("  - Típicamente converge en 1000-2000 epochs")
        print()
        print("Recomendación:")
        print("  ✅ Continuar hasta 5000 epochs")
        print("  ✅ Monitorear si hay mejoras después de epoch 2000")
        print("  ✅ Si no hay mejoras por 1000 epochs → probablemente convergió")

    print()
    print("=" * 80)
    print()

    # Next steps
    if epochs_completed < total_epochs:
        print("SIGUIENTE ACCIÓN:")
        print(f"  python hyperopt_full_5000.py  # Continuar desde epoch {epochs_completed}")
    else:
        print("✅ OPTIMIZACIÓN COMPLETADA")
        print()
        print("SIGUIENTE ACCIÓN:")
        print("  1. Revisar mejores parámetros en logs")
        print("  2. Validar en datos de TEST:")
        print("     freqtrade backtesting --strategy DynamicAggressiveHighTP --timerange 20250707-20251006")
        print("  3. Implementar parámetros validados en trading_bot_ws.py")

    print()

if __name__ == "__main__":
    main()
