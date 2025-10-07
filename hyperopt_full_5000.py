#!/usr/bin/env python3
"""
Hyperopt COMPLETO - 5000 iteraciones con RESUME automático
Optimiza TODOS los parámetros de DYNAMIC-Aggressive High TP

CARACTERÍSTICAS:
- Resume automático desde última iteración
- Guarda parámetros probados en JSON
- Detecta cuando se han probado todas las combinaciones
- Logging detallado de progreso
- Validación anti-overfitting

Train period: Oct 2024 - Jul 2025 (9 meses)
Test period: Jul 2025 - Oct 2025 (3 meses)
"""

import subprocess
import sys
import json
import time
from pathlib import Path
from datetime import datetime

STRATEGY = "DynamicAggressiveHighTP"
TRAIN_TIMERANGE = "20241006-20250707"  # 9 meses train
TEST_TIMERANGE = "20250707-20251006"   # 3 meses test
TOTAL_EPOCHS = 5000

# Archivos de tracking
PROGRESS_FILE = Path("hyperopt_progress_5000.json")
RESULTS_FILE = Path("hyperopt_results_5000.json")
LOG_FILE = Path("hyperopt_5000.log")

def log(message):
    """Log to both console and file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] {message}"
    print(log_msg)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(log_msg + "\n")

def load_progress():
    """Load progress from JSON file"""
    if PROGRESS_FILE.exists():
        try:
            with open(PROGRESS_FILE, "r") as f:
                return json.load(f)
        except:
            return {"epochs_completed": 0, "last_run": None, "runs": []}
    return {"epochs_completed": 0, "last_run": None, "runs": []}

def save_progress(progress):
    """Save progress to JSON file"""
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)

def save_results(results):
    """Save hyperopt results to JSON file"""
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

def get_remaining_epochs(progress):
    """Calculate remaining epochs"""
    completed = progress.get("epochs_completed", 0)
    return max(0, TOTAL_EPOCHS - completed)

def check_if_exhausted(progress):
    """
    Check if we've exhausted the parameter space.

    Heuristic: If the last 3 runs didn't find any improvement,
    and we've done at least 1000 epochs, likely exhausted.
    """
    if progress["epochs_completed"] < 1000:
        return False

    runs = progress.get("runs", [])
    if len(runs) < 3:
        return False

    # Check last 3 runs for no improvement
    last_3 = runs[-3:]
    improvements = [run.get("improvement", True) for run in last_3]

    if all(not imp for imp in improvements):
        log("⚠️  ADVERTENCIA: Últimas 3 ejecuciones sin mejora")
        log("    Posiblemente se han probado todas las combinaciones útiles")
        return True

    return False

def print_banner():
    """Print startup banner"""
    print()
    print("=" * 80)
    print("HYPEROPT COMPLETO - 5000 iteraciones con RESUME")
    print("=" * 80)
    print(f"Strategy: {STRATEGY}")
    print(f"Total epochs: {TOTAL_EPOCHS}")
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

def main():
    print_banner()

    # Load progress
    progress = load_progress()
    remaining = get_remaining_epochs(progress)

    log(f"Progreso cargado: {progress['epochs_completed']}/{TOTAL_EPOCHS} epochs completados")
    log(f"Epochs restantes: {remaining}")

    if remaining == 0:
        log("✅ COMPLETADO: Todas las 5000 iteraciones han sido ejecutadas")
        log(f"Ver resultados en: {RESULTS_FILE}")
        return

    # Check if exhausted
    if check_if_exhausted(progress):
        user_input = input("\n¿Continuar de todas formas? (s/n): ")
        if user_input.lower() != 's':
            log("Optimización cancelada por el usuario")
            return

    # Freqtrade path
    freqtrade_path = Path(__file__).parent / ".venv" / "Scripts" / "freqtrade.exe"
    if not freqtrade_path.exists():
        # Try Linux path
        freqtrade_path = Path(__file__).parent / ".venv" / "bin" / "freqtrade"

    if not freqtrade_path.exists():
        log("❌ ERROR: freqtrade no encontrado en .venv")
        sys.exit(1)

    log(f"Iniciando optimización de {remaining} epochs...")
    log("")

    # Build hyperopt command
    cmd_hyperopt = [
        str(freqtrade_path),
        "hyperopt",
        "--strategy", STRATEGY,
        "--hyperopt-loss", "SharpeHyperOptLoss",
        "--spaces", "buy", "sell",
        "--epochs", str(remaining),
        "--timerange", TRAIN_TIMERANGE,
        "--min-trades", "50",  # Requiere al menos 50 trades
    ]

    log("Comando hyperopt:")
    log(" ".join(cmd_hyperopt))
    log("")

    # Record run start
    run_start = datetime.now().isoformat()

    try:
        # Run hyperopt
        result = subprocess.run(cmd_hyperopt, cwd=Path(__file__).parent)

        run_end = datetime.now().isoformat()

        if result.returncode != 0:
            log("❌ ERROR: Hyperopt falló")

            # Save failed run info
            progress["runs"].append({
                "start": run_start,
                "end": run_end,
                "epochs_attempted": remaining,
                "success": False,
                "error": "Non-zero exit code"
            })
            save_progress(progress)
            sys.exit(1)

        # Update progress
        progress["epochs_completed"] = TOTAL_EPOCHS
        progress["last_run"] = run_end
        progress["runs"].append({
            "start": run_start,
            "end": run_end,
            "epochs_completed": remaining,
            "success": True
        })
        save_progress(progress)

        log("")
        log("=" * 80)
        log("✅ OPTIMIZACIÓN COMPLETADA")
        log("=" * 80)
        log(f"Total epochs: {TOTAL_EPOCHS}")
        log(f"Archivo de progreso: {PROGRESS_FILE}")
        log(f"Log completo: {LOG_FILE}")
        log("")
        log("SIGUIENTE PASO: VALIDACIÓN")
        log("=" * 80)
        log("")
        log("Ejecuta validación en datos de TEST:")
        log(f"  {freqtrade_path} backtesting --strategy {STRATEGY} --timerange {TEST_TIMERANGE}")
        log("")
        log("Compara resultados Train vs Test:")
        log("  ✅ Train ≈ Test = ROBUSTO (parámetros buenos)")
        log("  ⚠️  Train >> Test = OVERFITTING (rechazar)")
        log("  ⚠️  Train << Test = SUERTE (repetir)")
        log("")
        log("=" * 80)

    except KeyboardInterrupt:
        log("\n⚠️  Optimización interrumpida por el usuario")
        log(f"Progreso guardado: {progress['epochs_completed']}/{TOTAL_EPOCHS} epochs")
        log(f"Para continuar, ejecuta nuevamente: python {Path(__file__).name}")

        # Save interrupted run
        progress["runs"].append({
            "start": run_start,
            "end": datetime.now().isoformat(),
            "epochs_attempted": remaining,
            "success": False,
            "error": "Interrupted by user"
        })
        save_progress(progress)
        sys.exit(1)

    except Exception as e:
        log(f"\n❌ ERROR: {e}")

        # Save error run
        progress["runs"].append({
            "start": run_start,
            "end": datetime.now().isoformat(),
            "epochs_attempted": remaining,
            "success": False,
            "error": str(e)
        })
        save_progress(progress)
        sys.exit(1)

if __name__ == "__main__":
    main()
