import subprocess
import argparse
import sys
from datetime import datetime, timedelta

# ================= CONFIGURACIÓN POR DEFECTO =================
STRATEGY_NAME = "MultiKernelRegressionStrategy"
DEFAULT_CONFIG = "config.json"
DEFAULT_START = "2024-01-01"
DEFAULT_END = "2024-05-01"
DEFAULT_WINDOW = 30 # Días por segmento
# =============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Ejecuta un Backtest Walk-Forward para simular re-entrenamiento periódico.")
    
    parser.add_argument("--start", type=str, default=DEFAULT_START, 
                        help=f"Fecha de inicio (YYYY-MM-DD). Default: {DEFAULT_START}")
    
    parser.add_argument("--end", type=str, default=DEFAULT_END, 
                        help=f"Fecha de fin (YYYY-MM-DD). Default: {DEFAULT_END}")
    
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW, 
                        help=f"Tamaño de la ventana en días (frecuencia de re-calibración). Default: {DEFAULT_WINDOW}")
    
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG, 
                        help=f"Archivo de configuración de Freqtrade. Default: {DEFAULT_CONFIG}")

    return parser.parse_args()

def run_segment(start_date, end_date, config_file):
    """Ejecuta una instancia de backtest para un periodo específico"""
    
    # Formato Freqtrade: YYYYMMDD-YYYYMMDD
    timerange = f"{start_date.strftime('%Y%m%d')}-{end_date.strftime('%Y%m%d')}"
    
    print(f"\n>>> 🚀 INICIANDO SEGMENTO WALK-FORWARD: {timerange} <<<")
    
    # Comando de Freqtrade
    # NOTA: Se eliminó '--quiet' ya que causaba error en algunas versiones
    cmd = [
        "freqtrade", "backtesting",
        "--strategy", STRATEGY_NAME,
        "--config", config_file,
        "--timerange", timerange,
        "--cache", "none" # IMPORTANTE: No usar caché para forzar que la estrategia recalibre
    ]
    
    try:
        # Ejecutamos y esperamos
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Error ejecutando segmento {timerange}")
        return False
    except KeyboardInterrupt:
        print("\n🛑 Interrupción de usuario detectada.")
        sys.exit(0)

def main():
    args = parse_args()
    
    try:
        current_start = datetime.strptime(args.start, "%Y-%m-%d")
        final_end = datetime.strptime(args.end, "%Y-%m-%d")
    except ValueError:
        print("❌ Error: Formato de fecha incorrecto. Asegúrate de usar YYYY-MM-DD (ej: 2024-01-31).")
        sys.exit(1)

    if current_start >= final_end:
        print("❌ Error: La fecha de inicio debe ser anterior a la fecha de fin.")
        sys.exit(1)

    print(f"🔧 Configuración WF: Inicio={args.start}, Fin={args.end}, Ventana={args.window} días, Config={args.config}")

    total_segments = 0
    
    while current_start < final_end:
        # Calcular fin del segmento actual
        current_end = current_start + timedelta(days=args.window)
        
        # No pasarse de la fecha final global
        if current_end > final_end:
            current_end = final_end
            
        # Ejecutar segmento
        success = run_segment(current_start, current_end, args.config)
        
        if not success:
            print("⚠️ El segmento falló, deteniendo Walk-Forward.")
            break
            
        total_segments += 1
        
        # Mover la ventana: El inicio del siguiente es el fin del actual
        # (Así simulamos continuidad perfecta)
        current_start = current_end

    print(f"\n✅ WALK-FORWARD COMPLETADO ({total_segments} segmentos).")
    print("ℹ️  Para ver el PnL total, revisa los logs en 'user_data/backtest_results' o suma los reportes individuales.")

if __name__ == "__main__":
    main()