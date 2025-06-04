# 📦 Installation Guide

## Complete Installation Instructions for All Environments

This guide covers installation for development, testing, and production environments across different platforms and deployment scenarios.

---

## 🔧 **Prerequisites**

### **System Requirements**

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| **Python** | 3.9+ | 3.11+ |
| **RAM** | 4GB | 8GB+ |
| **Storage** | 2GB free | 10GB+ SSD |
| **CPU** | 2 cores | 4+ cores |
| **Network** | Internet connection | Stable broadband |

### **Required Software**

=== "macOS"

    ```bash
    # Install Homebrew (if not already installed)
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    
    # Install Python 3.11
    brew install python@3.11
    
    # Install Git
    brew install git
    
    # Install TA-Lib (for technical analysis)
    brew install ta-lib
    ```

=== "Ubuntu/Debian"

    ```bash
    # Update package list
    sudo apt update
    
    # Install Python 3.11 and pip
    sudo apt install python3.11 python3.11-pip python3.11-venv
    
    # Install Git
    sudo apt install git
    
    # Install TA-Lib dependencies
    sudo apt install build-essential wget
    
    # Install TA-Lib
    wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
    tar -xzf ta-lib-0.4.0-src.tar.gz
    cd ta-lib/
    ./configure --prefix=/usr
    make
    sudo make install
    cd ..
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz
    ```

=== "CentOS/RHEL"

    ```bash
    # Install EPEL repository
    sudo dnf install epel-release
    
    # Install Python 3.11
    sudo dnf install python3.11 python3.11-pip
    
    # Install Git
    sudo dnf install git
    
    # Install TA-Lib dependencies
    sudo dnf groupinstall "Development Tools"
    sudo dnf install wget
    
    # Install TA-Lib (same as Ubuntu steps)
    wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
    tar -xzf ta-lib-0.4.0-src.tar.gz
    cd ta-lib/
    ./configure --prefix=/usr
    make
    sudo make install
    cd ..
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz
    ```

=== "Windows"

    ```powershell
    # Install Python 3.11 from python.org or use winget
    winget install Python.Python.3.11
    
    # Install Git
    winget install Git.Git
    
    # Add Python and pip to PATH (usually done automatically)
    # Verify installation
    python --version
    pip --version
    
    # For TA-Lib on Windows, we'll use pip installation later
    # No additional system packages needed
    ```

---

## 🏠 **Development Installation**

### **Option 1: Standard Installation (Recommended)**

```bash
# 1. Clone the repository
git clone https://github.com/trading-optimizer/hyperopt-strat.git
cd hyperopt-strat

# 2. Create virtual environment
python3.11 -m venv venv

# 3. Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# 4. Upgrade pip
pip install --upgrade pip

# 5. Install dependencies
pip install -r requirements.txt

# 6. Verify installation
python -c "import pandas, numpy, hyperopt, fastapi; print('✅ Core dependencies installed')"
```

### **Option 2: Development with Testing Tools**

```bash
# Follow steps 1-5 from above, then:

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests to verify everything works
pytest tests/ -v

# Start development server
cd src/api
python main.py
```

### **Option 3: Poetry Installation**

```bash
# Install Poetry (if not already installed)
curl -sSL https://install.python-poetry.org | python3 -

# Clone and install
git clone https://github.com/trading-optimizer/hyperopt-strat.git
cd hyperopt-strat

# Install dependencies with Poetry
poetry install

# Activate Poetry shell
poetry shell

# Verify installation
python -c "import pandas, numpy, hyperopt, fastapi; print('✅ Dependencies installed via Poetry')"
```

---

## 🚀 **Production Installation**

### **Option 1: Direct Server Installation**

```bash
# 1. Create production user (recommended)
sudo useradd -m -s /bin/bash trading-optimizer
sudo usermod -aG sudo trading-optimizer
sudo su - trading-optimizer

# 2. Clone repository
git clone https://github.com/trading-optimizer/hyperopt-strat.git
cd hyperopt-strat

# 3. Create production virtual environment
python3.11 -m venv prod-venv
source prod-venv/bin/activate

# 4. Install production dependencies only
pip install --upgrade pip
pip install -r requirements.txt --no-dev

# 5. Configure environment
cp .env.example .env
# Edit .env with production settings

# 6. Set up systemd service (optional)
sudo cp scripts/trading-optimizer.service /etc/systemd/system/
sudo systemctl enable trading-optimizer
sudo systemctl start trading-optimizer
```

### **Option 2: Docker Production**

```bash
# 1. Clone repository
git clone https://github.com/trading-optimizer/hyperopt-strat.git
cd hyperopt-strat

# 2. Configure environment
cp .env.example .env
# Edit .env with production settings

# 3. Build and start production containers
docker-compose -f docker-compose.prod.yml up -d

# 4. Verify deployment
docker-compose -f docker-compose.prod.yml logs api
curl http://localhost:8000/api/v1/health
```

---

## 🐳 **Docker Installation**

### **Development with Docker**

```bash
# 1. Clone repository
git clone https://github.com/trading-optimizer/hyperopt-strat.git
cd hyperopt-strat

# 2. Build development image
docker-compose -f docker-compose.dev.yml build

# 3. Start development stack
docker-compose -f docker-compose.dev.yml up -d

# 4. Access development server
curl http://localhost:8000/api/v1/health

# 5. View logs
docker-compose -f docker-compose.dev.yml logs -f api

# 6. Stop when done
docker-compose -f docker-compose.dev.yml down
```

### **Production with Docker Swarm**

```bash
# 1. Initialize Docker Swarm
docker swarm init

# 2. Clone repository
git clone https://github.com/trading-optimizer/hyperopt-strat.git
cd hyperopt-strat

# 3. Configure production environment
cp .env.example .env
# Edit .env with production settings

# 4. Deploy stack
docker stack deploy -c docker-compose.prod.yml trading-optimizer

# 5. Monitor deployment
docker service ls
docker service logs trading-optimizer_api
```

---

## ☸️ **Kubernetes Installation**

### **Prerequisites for Kubernetes**

```bash
# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# Install Helm (optional, for easier management)
curl https://get.helm.sh/helm-v3.12.0-linux-amd64.tar.gz | tar xz
sudo mv linux-amd64/helm /usr/local/bin/
```

### **Kubernetes Deployment**

```bash
# 1. Clone repository
git clone https://github.com/trading-optimizer/hyperopt-strat.git
cd hyperopt-strat

# 2. Create namespace
kubectl create namespace trading-optimizer

# 3. Create configuration secret
kubectl create secret generic trading-optimizer-config \
  --from-env-file=.env \
  --namespace=trading-optimizer

# 4. Apply Kubernetes manifests
kubectl apply -f k8s/ --namespace=trading-optimizer

# 5. Verify deployment
kubectl get pods --namespace=trading-optimizer
kubectl get services --namespace=trading-optimizer

# 6. Access the service
kubectl port-forward service/trading-optimizer-api 8000:8000 --namespace=trading-optimizer
```

### **Helm Installation (Alternative)**

```bash
# 1. Add Helm repository (if available)
helm repo add trading-optimizer https://charts.trading-optimizer.com
helm repo update

# 2. Install with Helm
helm install trading-optimizer trading-optimizer/hyperopt-strat \
  --namespace=trading-optimizer \
  --create-namespace \
  --values=values.prod.yml

# 3. Verify installation
helm status trading-optimizer --namespace=trading-optimizer
```

---

## 🔧 **Configuration Setup**

### **Environment Variables**

Create and configure your `.env` file:

```bash
# Copy example configuration
cp .env.example .env

# Edit configuration
nano .env  # or vim, code, etc.
```

**Required Configuration:**

```env
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Security
SECRET_KEY=your-super-secret-key-here
API_KEYS=dev-key-12345,prod-key-67890

# Database (optional)
DATABASE_URL=sqlite:///./trading_optimizer.db

# External APIs (optional)
ALPHA_VANTAGE_API_KEY=your-alpha-vantage-key
POLYGON_API_KEY=your-polygon-key

# Monitoring
ENABLE_METRICS=true
PROMETHEUS_PORT=8001

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
```

### **Database Setup (Optional)**

If using a database for persistent storage:

```bash
# PostgreSQL example
sudo apt install postgresql postgresql-contrib
sudo -u postgres createdb trading_optimizer
sudo -u postgres createuser trading_optimizer
sudo -u postgres psql -c "ALTER USER trading_optimizer PASSWORD 'secure_password';"

# Update .env
echo "DATABASE_URL=postgresql://trading_optimizer:secure_password@localhost/trading_optimizer" >> .env
```

### **Monitoring Setup**

```bash
# Start monitoring stack
cd monitoring/
docker-compose up -d

# Access Grafana
# URL: http://localhost:3001
# Username: admin
# Password: trading_api_2024
```

---

## ✅ **Installation Verification**

### **Basic Verification**

```bash
# 1. Activate environment
source venv/bin/activate  # or poetry shell

# 2. Test imports
python -c "
import sys
print(f'Python version: {sys.version}')

# Test core dependencies
import pandas as pd
import numpy as np
import hyperopt
import fastapi
import talib
print('✅ All core dependencies imported successfully')

# Test specific modules
from src.strategies.base_strategy import BaseStrategy
from src.optimization.hyperopt_optimizer import HyperoptOptimizer
print('✅ Core modules imported successfully')
"

# 3. Test API server
cd src/api
python -c "
from main import app
print('✅ FastAPI application created successfully')
"
```

### **API Server Verification**

```bash
# 1. Start server in background
cd src/api
python main.py &
SERVER_PID=$!

# 2. Wait for startup
sleep 5

# 3. Test endpoints
echo "Testing health endpoint..."
curl -s http://localhost:8000/api/v1/health | jq '.'

echo "Testing strategies endpoint..."
curl -s -H "X-API-Key: dev-key-12345" http://localhost:8000/api/v1/strategies/list | jq '.strategies | length'

echo "Testing metrics endpoint..."
curl -s http://localhost:8000/metrics | head -5

# 4. Stop server
kill $SERVER_PID
echo "✅ API server verification complete"
```

### **Optimization Test**

```bash
# Run a quick optimization test
python -c "
import sys
sys.path.append('src')

from strategies.moving_average_strategies import MovingAverageCrossover
from optimization.hyperopt_optimizer import HyperoptOptimizer
import pandas as pd
import numpy as np

# Create sample data
dates = pd.date_range('2023-01-01', periods=100, freq='1D')
data = pd.DataFrame({
    'close': 100 + np.cumsum(np.random.randn(100) * 0.01),
    'volume': np.random.randint(1000, 10000, 100)
}, index=dates)

# Test strategy
strategy = MovingAverageCrossover()
print('✅ Strategy created successfully')

# Test optimization (quick)
optimizer = HyperoptOptimizer(strategy, data)
print('✅ Optimizer created successfully')
print('🎉 Installation verification complete!')
"
```

---

## 🚨 **Troubleshooting**

### **Common Issues**

#### **TA-Lib Installation Issues**

=== "macOS"

    ```bash
    # If TA-Lib installation fails
    brew install ta-lib
    
    # If still failing, try:
    pip install --upgrade setuptools
    pip install TA-Lib
    
    # Alternative: use conda
    conda install -c conda-forge ta-lib
    ```

=== "Ubuntu/Linux"

    ```bash
    # If TA-Lib compilation fails
    sudo apt install build-essential
    sudo apt install libpython3-dev
    
    # Reinstall TA-Lib from source
    wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
    tar -xzf ta-lib-0.4.0-src.tar.gz
    cd ta-lib/
    ./configure --prefix=/usr
    make
    sudo make install
    sudo ldconfig
    ```

=== "Windows"

    ```powershell
    # Use precompiled wheels
    pip install --upgrade pip
    pip install TA-Lib
    
    # If that fails, download wheel from:
    # https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
    # Then install: pip install TA_Lib-0.4.24-cp311-cp311-win_amd64.whl
    ```

#### **Permission Issues**

```bash
# Fix common permission issues
sudo chown -R $USER:$USER /path/to/hyperopt-strat
chmod +x scripts/*.sh

# For Docker issues
sudo usermod -aG docker $USER
newgrp docker  # or logout/login
```

#### **Memory Issues**

```bash
# Increase swap for low-memory systems
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Make permanent
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

#### **Port Conflicts**

```bash
# Find what's using port 8000
sudo netstat -tulpn | grep :8000
sudo lsof -i :8000

# Kill conflicting process
sudo kill -9 PID_HERE

# Or use different port
export API_PORT=8080
```

### **Performance Optimization**

```bash
# Install performance-optimized numpy
pip uninstall numpy
pip install numpy[mkl]  # Intel MKL optimized

# For AMD CPUs
pip install numpy[openblas]

# Enable multiprocessing optimizations
export OMP_NUM_THREADS=4
export NUMBA_NUM_THREADS=4
```

---

## 🔄 **Updating the System**

### **Development Updates**

```bash
# 1. Backup current installation (if needed)
cp -r hyperopt-strat hyperopt-strat.backup

# 2. Update code
cd hyperopt-strat
git pull origin main

# 3. Update dependencies
source venv/bin/activate
pip install --upgrade -r requirements.txt

# 4. Run tests
pytest tests/

# 5. Restart services
cd src/api
python main.py
```

### **Production Updates**

```bash
# 1. Download new version
wget https://github.com/trading-optimizer/hyperopt-strat/archive/main.zip
unzip main.zip

# 2. Stop services
sudo systemctl stop trading-optimizer

# 3. Backup and replace
sudo cp -r /opt/trading-optimizer /opt/trading-optimizer.backup
sudo cp -r hyperopt-strat-main/* /opt/trading-optimizer/

# 4. Update dependencies
cd /opt/trading-optimizer
source prod-venv/bin/activate
pip install --upgrade -r requirements.txt

# 5. Restart services
sudo systemctl start trading-optimizer
sudo systemctl status trading-optimizer
```

---

## 📞 **Getting Help**

### **Installation Support**

If you encounter issues during installation:

1. **📖 Check Documentation**: [Troubleshooting Guide](../examples/troubleshooting.md)
2. **🔍 Search Issues**: [GitHub Issues](https://github.com/trading-optimizer/hyperopt-strat/issues)
3. **💬 Ask Community**: [GitHub Discussions](https://github.com/trading-optimizer/hyperopt-strat/discussions)
4. **📧 Enterprise Support**: [contact@trading-optimizer.com](mailto:contact@trading-optimizer.com)

### **System Information for Support**

When reporting issues, include:

```bash
# Generate system info
python -c "
import sys, platform, pkg_resources
print(f'Python: {sys.version}')
print(f'Platform: {platform.platform()}')
print(f'Architecture: {platform.architecture()}')

# Installed packages
installed = [d for d in pkg_resources.working_set]
for package in sorted(installed, key=lambda x: x.project_name.lower()):
    if any(pkg in package.project_name.lower() for pkg in ['pandas', 'numpy', 'fastapi', 'hyperopt', 'talib']):
        print(f'{package.project_name}: {package.version}')
"
```

---

**🎉 Installation complete! You're ready to start optimizing trading strategies!**

**Next Steps:**
- 📖 [Quick Start Guide](quick-start.md) - Run your first optimization
- 🏗️ [System Architecture](../architecture/system-overview.md) - Understand the system
- 📡 [API Reference](../api/overview.md) - Explore the API endpoints 