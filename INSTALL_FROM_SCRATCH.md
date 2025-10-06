# Instalación Desde Cero - Freqtrade con Pacifica Integration

Esta guía te permite instalar todo el setup en una nueva máquina (VPS, servidor, otra laptop).

## Requisitos Previos

### Sistema Operativo
- **Linux** (Ubuntu 20.04+ / Debian 11+) - RECOMENDADO para VPS
- **Windows** (10/11 con WSL2 o nativo)
- **macOS** (Monterey+)

### Software
- **Python 3.11 o superior** (REQUERIDO)
- **Git** 2.x+
- **8 GB RAM** mínimo (16 GB recomendado para hyperopt)
- **10 GB espacio libre** (datos históricos)

---

## 🚀 Instalación Rápida (Linux/Ubuntu)

```bash
# 1. Actualizar sistema
sudo apt update && sudo apt upgrade -y

# 2. Instalar dependencias
sudo apt install -y python3.11 python3.11-venv python3-pip git curl build-essential

# 3. Verificar Python
python3.11 --version  # Debe mostrar 3.11.x

# 4. Clonar tu fork (REEMPLAZA CON TU URL)
cd ~
git clone https://github.com/WalterOrtiz21/freqtrade.git
cd freqtrade

# 5. Checkout branch de Pacifica
git checkout pacifica-integration

# 6. Crear virtual environment
python3.11 -m venv .venv

# 7. Activar venv
source .venv/bin/activate

# 8. Actualizar pip
pip install --upgrade pip

# 9. Instalar Freqtrade
pip install -e .

# 10. Instalar dependencias de hyperopt
pip install -r requirements-hyperopt.txt

# 11. Verificar instalación
freqtrade --version
```

**Tiempo estimado: 10-15 minutos**

---

## 🪟 Instalación en Windows

```powershell
# 1. Verificar Python 3.11
py -3.11 --version

# Si no está instalado:
# Descargar de https://www.python.org/downloads/
# Marcar "Add to PATH" durante instalación

# 2. Clonar repo
cd C:\
git clone https://github.com/WalterOrtiz21/freqtrade.git
cd freqtrade

# 3. Checkout branch
git checkout pacifica-integration

# 4. Crear venv
py -3.11 -m venv .venv

# 5. Activar venv
.venv\Scripts\Activate.ps1

# 6. Instalar
pip install --upgrade pip
pip install -e .
pip install -r requirements-hyperopt.txt

# 7. Verificar
.venv\Scripts\freqtrade.exe --version
```

---

## 📊 Descargar Datos Históricos

```bash
# Activar venv (si no está activo)
source .venv/bin/activate  # Linux/Mac
# o
.venv\Scripts\activate  # Windows

# Descargar 2 años de datos (2023-2025)
freqtrade download-data \
  --exchange binance \
  --pairs BTC/USDT:USDT ETH/USDT:USDT SOL/USDT:USDT BNB/USDT:USDT SUI/USDT:USDT LTC/USDT:USDT \
  --timerange 20231006-20251006 \
  --timeframe 5m \
  --trading-mode futures

# Verificar datos descargados
ls -lh user_data/data/binance/futures/
```

**Tiempo estimado: 2-5 minutos (depende de conexión)**

---

## ⚡ Ejecutar Hyperopt

### Opción 1: Últimos 365 días (más rápido)
```bash
python hyperopt_365days.py
```

### Opción 2: 2 años completos (más robusto)
```bash
python hyperopt_2years.py
```

**Tiempo en VPS (depende de CPU):**
- VPS básico (2 cores): 20-40 minutos
- VPS potente (4+ cores): 10-20 minutos
- Local Windows: 30-60 minutos

---

## 🔧 Configuración (Opcional)

### Si usas diferente exchange o pares:

Edita `config.json`:
```json
{
  "exchange": {
    "name": "binance",  // Cambiar si usas otro exchange
    "pair_whitelist": [
      "BTC/USDT:USDT",
      "ETH/USDT:USDT"
      // Agregar/quitar pares aquí
    ]
  },
  "max_open_trades": 5  // Ajustar según tu capital
}
```

### Variables de Entorno (si necesitas API keys para trading):

```bash
# Crear .env (solo para trading real, NO para hyperopt)
cat > .env << EOF
API_ENVIRONMENT=testnet
PUBLIC_KEY=your_public_key
API_AGENT_KEY=your_agent_key
AGENT_PUBLIC_KEY=your_agent_public_key
EOF
```

---

## 📈 Comparación de Performance: Local vs VPS

### Tu PC Windows (actual)
```
CPU: ~8 cores @ 2-3 GHz
RAM: 16-32 GB
Hyperopt 1000 epochs: 30-60 minutos
Costo: $0 (ya lo tienes)
```

### VPS Linux (opciones recomendadas)

#### DigitalOcean - CPU-Optimized Droplet
```
CPU: 4 cores @ 3.7 GHz
RAM: 8 GB
Hyperopt 1000 epochs: 15-25 minutos
Costo: $48/mes (o $0.07/hora = ~$0.20 por hyperopt)
Link: https://www.digitalocean.com/
```

#### Hetzner - CPX31
```
CPU: 4 cores @ 2.7 GHz
RAM: 8 GB
Hyperopt 1000 epochs: 20-30 minutos
Costo: €14.90/mes (~$16/mes)
Link: https://www.hetzner.com/cloud
```

#### Google Cloud - e2-standard-4
```
CPU: 4 cores @ 2.8 GHz
RAM: 16 GB
Hyperopt 1000 epochs: 15-25 minutos
Costo: ~$121/mes (o $0.17/hora = ~$0.50 por hyperopt)
Créditos: $300 gratis para nuevos usuarios
Link: https://cloud.google.com/
```

#### AWS EC2 - c6i.xlarge (CPU-optimized)
```
CPU: 4 cores @ 3.5 GHz
RAM: 8 GB
Hyperopt 1000 epochs: 12-20 minutos
Costo: ~$145/mes (o $0.20/hora = ~$0.60 por hyperopt)
Link: https://aws.amazon.com/ec2/
```

### Recomendación: **Hetzner CPX31** 🏆
- ✅ Mejor precio/performance ($16/mes)
- ✅ Europa (buena conexión)
- ✅ Ubuntu pre-instalado
- ✅ 4 cores suficientes

---

## 🚀 Setup Rápido en VPS (Script Automatizado)

```bash
#!/bin/bash
# save as setup_freqtrade.sh

echo "=== Freqtrade Setup Script ==="

# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install -y python3.11 python3.11-venv python3-pip git curl build-essential

# Clone repo (REPLACE WITH YOUR FORK URL)
cd ~
git clone https://github.com/WalterOrtiz21/freqtrade.git
cd freqtrade
git checkout pacifica-integration

# Create venv
python3.11 -m venv .venv
source .venv/bin/activate

# Install Freqtrade
pip install --upgrade pip
pip install -e .
pip install -r requirements-hyperopt.txt

# Download data
freqtrade download-data \
  --exchange binance \
  --pairs BTC/USDT:USDT ETH/USDT:USDT SOL/USDT:USDT BNB/USDT:USDT SUI/USDT:USDT LTC/USDT:USDT \
  --timerange 20231006-20251006 \
  --timeframe 5m \
  --trading-mode futures

echo "=== Setup Complete! ==="
echo "Run: source .venv/bin/activate"
echo "Then: python hyperopt_365days.py"
```

**Uso:**
```bash
# En tu VPS:
wget https://raw.githubusercontent.com/TU_USUARIO/freqtrade/pacifica-integration/setup_freqtrade.sh
chmod +x setup_freqtrade.sh
./setup_freqtrade.sh
```

---

## ⚙️ Ejecutar Hyperopt en Background (VPS)

Si quieres cerrar SSH y que siga corriendo:

```bash
# Método 1: screen (recomendado)
sudo apt install screen
screen -S hyperopt
source .venv/bin/activate
python hyperopt_365days.py

# Presiona Ctrl+A, luego D para detach
# Reconectar: screen -r hyperopt

# Método 2: nohup
nohup python hyperopt_365days.py > hyperopt.log 2>&1 &

# Ver progreso: tail -f hyperopt.log
```

---

## 📦 Archivos Incluidos en el Repo

```
freqtrade/
├── config.json                          # Configuración de trading
├── hyperopt_365days.py                  # Hyperopt últimos 365 días
├── hyperopt_2years.py                   # Hyperopt 2 años completos
├── validate_365days.py                  # Validación 365 días
├── validate_2years.py                   # Validación 2 años
├── convert_pacifica_to_freqtrade.py     # Conversor de datos
├── user_data/strategies/
│   └── DynamicAggressiveHighTP.py       # Estrategia implementada
├── SETUP_COMPLETE.md                    # Guía de setup
├── HYPEROPT_GUIDE.md                    # Guía completa de hyperopt
├── QUICK_START_HYPEROPT.md              # Referencia rápida
├── README_HYPEROPT.md                   # Selección de datos
├── FREQTRADE_INTEGRATION.md             # Integración con Pacifica
└── INSTALL_FROM_SCRATCH.md              # Esta guía
```

---

## 🐛 Troubleshooting

### Error: "No module named 'freqtrade'"
```bash
# Asegúrate de estar en el venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Reinstalar
pip install -e .
```

### Error: "Python 3.11 not found"
```bash
# Ubuntu/Debian
sudo apt install python3.11 python3.11-venv

# Windows: Descargar de python.org
```

### Error: "TA-Lib installation failed"
```bash
# Ubuntu/Debian
sudo apt install build-essential
pip install TA-Lib

# Si sigue fallando:
sudo apt install libta-lib-dev
```

### Hyperopt muy lento
```bash
# Verificar CPU usage
htop

# Reducir epochs si es necesario
# Editar hyperopt_365days.py: EPOCHS = 500
```

---

## 📊 Benchmark de Velocidad

Puedes testear la velocidad de tu VPS con:

```bash
# Test rápido (10 epochs)
time freqtrade hyperopt \
  --strategy DynamicAggressiveHighTP \
  --epochs 10 \
  --timerange 20250901-20251006

# Si tarda más de 2 minutos → VPS lento
# Óptimo: < 1 minuto para 10 epochs
```

---

## ✅ Checklist de Instalación

- [ ] Python 3.11+ instalado (`python3.11 --version`)
- [ ] Git instalado (`git --version`)
- [ ] Repo clonado y branch correcto (`git branch`)
- [ ] Venv creado y activo (prompt muestra `(.venv)`)
- [ ] Freqtrade instalado (`freqtrade --version`)
- [ ] Datos descargados (`ls user_data/data/binance/futures/`)
- [ ] Config verificado (`cat config.json`)
- [ ] Hyperopt ready (`python hyperopt_365days.py --help`)

---

## 📞 Soporte

Si algo falla:
1. Revisa los logs: `tail -f logs/freqtrade.log`
2. Verifica versión Python: `python --version`
3. Verifica instalación: `pip list | grep freqtrade`
4. Consulta docs oficiales: https://www.freqtrade.io/

---

**Listo para instalar en VPS!** 🚀

Tiempo total estimado: 15-20 minutos
