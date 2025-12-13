#!/usr/bin/env python3
"""
AI Trading Bot - Automated Setup Script
Python 3.11+ Virtual Environment Setup
"""

import os
import sys
import subprocess
import json
from pathlib import Path

# Check Python version
if sys.version_info < (3, 11):
    print("❌ Error: Python 3.11 or higher is required")
    print(f"Current version: {sys.version}")
    sys.exit(1)

print("=" * 60)
print("AI Trading Bot - Automated Setup")
print("=" * 60)
print(f"Python version: {sys.version.split()[0]} ✓")
print()

# Project structure
PROJECT_ROOT = Path.cwd()
VENV_DIR = PROJECT_ROOT / "venv"
MODULES_DIR = PROJECT_ROOT / "modules"
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
CONFIGS_DIR = PROJECT_ROOT / "configs"

# Create directory structure
directories = [MODULES_DIR, MODELS_DIR, DATA_DIR, LOGS_DIR, CONFIGS_DIR]

print("📁 Creating directory structure...")
for directory in directories:
    directory.mkdir(exist_ok=True)
    print(f"  ✓ {directory}")
print()

# Requirements
REQUIREMENTS = """# Core Dependencies
ibapi==9.81.1.post1
pandas==2.1.4
numpy==1.26.2
scikit-learn==1.3.2
requests==2.31.0

# Data Processing
python-dateutil==2.8.2
pytz==2023.3

# Optional: Deep Learning (uncomment if needed)
# tensorflow==2.15.0
# torch==2.1.2
# keras==2.15.0

# Optional: Sentiment Analysis (uncomment if needed)
# newsapi-python==0.2.7
# tweepy==4.14.0
# textblob==0.17.1
# transformers==4.36.2

# Optional: Backtesting (uncomment if needed)
# backtrader==1.9.78.123
# pyfolio-reloaded==0.9.5

# Optional: Web API (uncomment if needed)
# flask==3.0.0
# flask-cors==4.0.0
# fastapi==0.108.0
# uvicorn==0.25.0
# websockets==12.0

# Optional: Database (uncomment if needed)
# sqlalchemy==2.0.23
# psycopg2-binary==2.9.9

# Optional: Alerts (uncomment if needed)
# twilio==8.11.0
# discord.py==2.3.2

# Development Tools
pytest==7.4.3
black==23.12.1
flake8==6.1.0
"""

# Configuration template
CONFIG_TEMPLATE = {
    "account_type": "paper",
    "ibkr": {
        "host": "127.0.0.1",
        "port": 7497,
        "client_id": 1,
        "paper_port": 7497,
        "live_port": 7496
    },
    "trading": {
        "watchlist": ["AAPL", "TSLA", "NVDA", "MSFT", "AMD", "GOOGL"],
        "max_position_size": 10000,
        "daily_loss_limit": 2000,
        "stop_loss_percent": 1.5,
        "target_gain_percent": 2.5,
        "min_volume": 1000000,
        "price_range_min": 5,
        "price_range_max": 500
    },
    "ai_strategy": {
        "confidence_threshold": 0.6,
        "use_lstm": True,
        "use_alpha_fusion": True,
        "ensemble_weights": {
            "lstm": 0.4,
            "alpha_fusion": 0.35,
            "technical": 0.25
        },
        "learning_rate": 0.001,
        "sequence_length": 60
    },
    "warrior_trading": {
        "enabled": True,
        "min_gap_percent": 3.0,
        "momentum_strategies": ["bull_flag", "vwap_hold", "reversal"],
        "min_volume_multiplier": 3.0
    },
    "risk_management": {
        "max_positions": 10,
        "position_concentration_limit": 0.2,
        "use_trailing_stops": True,
        "trailing_stop_percent": 1.0
    },
    "logging": {
        "level": "INFO",
        "log_trades": True,
        "log_signals": True,
        "log_file": "logs/trading.log"
    },
    "data_sources": {
        "newsapi_key": "YOUR_NEWSAPI_KEY_HERE",
        "twitter_enabled": False,
        "twitter_bearer_token": "YOUR_TWITTER_TOKEN_HERE"
    }
}

# Run command helper
def run_command(command, description):
    """Execute shell command with status output"""
    print(f"⏳ {description}...")
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        print(f"  ✓ {description} complete")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ❌ Error: {e}")
        print(f"  Output: {e.output}")
        return False

# Main setup
def main():
    print("🚀 Starting setup process...\n")
    
    # Step 1: Create virtual environment
    if not VENV_DIR.exists():
        print("📦 Creating Python 3.11 virtual environment...")
        if not run_command(
            f"python3.11 -m venv {VENV_DIR}",
            "Creating venv"
        ):
            print("Trying with 'python' command...")
            run_command(f"python -m venv {VENV_DIR}", "Creating venv")
    else:
        print("✓ Virtual environment already exists\n")
    
    # Determine pip path based on OS
    if sys.platform == "win32":
        pip_path = VENV_DIR / "Scripts" / "pip"
        python_path = VENV_DIR / "Scripts" / "python"
        activate_cmd = str(VENV_DIR / "Scripts" / "activate.bat")
    else:
        pip_path = VENV_DIR / "bin" / "pip"
        python_path = VENV_DIR / "bin" / "python"
        activate_cmd = f"source {VENV_DIR / 'bin' / 'activate'}"
    
    # Step 2: Upgrade pip
    run_command(
        f"{python_path} -m pip install --upgrade pip",
        "Upgrading pip"
    )
    
    # Step 3: Create requirements.txt
    requirements_file = PROJECT_ROOT / "requirements.txt"
    print("\n📝 Creating requirements.txt...")
    with open(requirements_file, 'w') as f:
        f.write(REQUIREMENTS)
    print(f"  ✓ Created {requirements_file}")
    
    # Step 4: Install dependencies
    print("\n📚 Installing core dependencies...")
    run_command(
        f"{pip_path} install -r {requirements_file}",
        "Installing packages"
    )
    
    # Step 5: Create configuration
    config_file = CONFIGS_DIR / "config.json"
    if not config_file.exists():
        print("\n⚙️  Creating configuration file...")
        with open(config_file, 'w') as f:
            json.dump(CONFIG_TEMPLATE, f, indent=2)
        print(f"  ✓ Created {config_file}")
    else:
        print("\n✓ Configuration file already exists")
    
    # Step 6: Create .env template
    env_file = PROJECT_ROOT / ".env.template"
    env_content = """# IBKR Configuration
IBKR_HOST=127.0.0.1
IBKR_PORT=7497
IBKR_CLIENT_ID=1

# API Keys (replace with your actual keys)
NEWSAPI_KEY=your_newsapi_key_here
TWITTER_BEARER_TOKEN=your_twitter_token_here

# Trading Configuration
ACCOUNT_TYPE=paper
MAX_POSITION_SIZE=10000
DAILY_LOSS_LIMIT=2000

# Logging
LOG_LEVEL=INFO
"""
    print("\n🔐 Creating .env template...")
    with open(env_file, 'w') as f:
        f.write(env_content)
    print(f"  ✓ Created {env_file}")
    print("  ⚠️  Copy .env.template to .env and add your API keys")
    
    # Step 7: Create .gitignore
    gitignore_file = PROJECT_ROOT / ".gitignore"
    gitignore_content = """# Virtual Environment
venv/
env/
.env

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python

# Trading Data
data/
logs/
*.log
*.db
*.sqlite

# Models
models/*.h5
models/*.pkl
models/*.pth

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# API Keys
.env
config.json
"""
    print("\n📄 Creating .gitignore...")
    with open(gitignore_file, 'w') as f:
        f.write(gitignore_content)
    print(f"  ✓ Created {gitignore_file}")
    
    # Step 8: Create startup script
    if sys.platform == "win32":
        startup_script = PROJECT_ROOT / "start_bot.bat"
        startup_content = f"""@echo off
echo Starting AI Trading Bot...
call {VENV_DIR}\\Scripts\\activate.bat
python ibkr_trading_backend.py
pause
"""
    else:
        startup_script = PROJECT_ROOT / "start_bot.sh"
        startup_content = f"""#!/bin/bash
echo "Starting AI Trading Bot..."
source {VENV_DIR}/bin/activate
python ibkr_trading_backend.py
"""
    
    print("\n🚀 Creating startup script...")
    with open(startup_script, 'w') as f:
        f.write(startup_content)
    
    if sys.platform != "win32":
        os.chmod(startup_script, 0o755)
    
    print(f"  ✓ Created {startup_script}")
    
    # Step 9: Create README
    readme_file = PROJECT_ROOT / "README.md"
    readme_content = """# AI Trading Bot

Advanced AI-powered algorithmic trading system with IBKR integration.

## Quick Start

### 1. Activate Virtual Environment

**Windows:**
```bash
venv\\Scripts\\activate
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

### 2. Configure Settings

Edit `configs/config.json` with your settings.

### 3. Run the Bot

**Using startup script:**
```bash
# Windows
start_bot.bat

# Linux/Mac
./start_bot.sh
```

**Manual start:**
```bash
python ibkr_trading_backend.py
```

## Project Structure

```
ai_trading_bot/
├── venv/                          # Virtual environment
├── modules/                       # Custom trading modules
├── models/                        # Saved AI models
├── data/                         # Historical data
├── logs/                         # Log files
├── configs/                      # Configuration files
├── ibkr_trading_backend.py       # Main trading bot
├── lstm_neural_network.py        # AI/ML models
├── modular_dashboard_config.py   # Module system
├── requirements.txt              # Dependencies
└── README.md                     # This file
```

## Prerequisites

- Python 3.11+
- IBKR TWS or IB Gateway installed
- Paper or Live trading account

## Configuration

1. Copy `.env.template` to `.env`
2. Add your API keys
3. Configure `configs/config.json`

## Features

- ✅ IBKR TWS integration
- ✅ AI-powered trading strategies
- ✅ Risk management
- ✅ Real-time market data
- ✅ Modular architecture

## Documentation

See `setup_guide.md` for detailed setup instructions.
"""
    
    print("\n📖 Creating README...")
    with open(readme_file, 'w') as f:
        f.write(readme_content)
    print(f"  ✓ Created {readme_file}")
    
    # Final summary
    print("\n" + "=" * 60)
    print("✅ Setup Complete!")
    print("=" * 60)
    print("\n📂 Project Structure Created:")
    print(f"  • Virtual environment: {VENV_DIR}")
    print(f"  • Modules directory: {MODULES_DIR}")
    print(f"  • Models directory: {MODELS_DIR}")
    print(f"  • Data directory: {DATA_DIR}")
    print(f"  • Logs directory: {LOGS_DIR}")
    print(f"  • Config directory: {CONFIGS_DIR}")
    
    print("\n📦 Files Created:")
    print("  • requirements.txt")
    print("  • .env.template")
    print("  • .gitignore")
    print(f"  • {startup_script.name}")
    print("  • README.md")
    print("  • configs/config.json")
    
    print("\n🚀 Next Steps:")
    print("\n1. Activate virtual environment:")
    print(f"   {activate_cmd}")
    print("\n2. Copy .env template and add API keys:")
    print("   cp .env.template .env")
    print("\n3. Edit configuration:")
    print("   nano configs/config.json")
    print("\n4. Place the Python bot files in this directory:")
    print("   • ibkr_trading_backend.py")
    print("   • lstm_neural_network.py")
    print("   • modular_dashboard_config.py")
    print("\n5. Start IBKR TWS (Paper Trading)")
    print("\n6. Run the bot:")
    if sys.platform == "win32":
        print("   start_bot.bat")
    else:
        print("   ./start_bot.sh")
    
    print("\n" + "=" * 60)
    print("Happy Trading! 📈")
    print("=" * 60)

if __name__ == "__main__":
    main()
