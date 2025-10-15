"""
Install required packages for LSTM trading bot
Save as: install_packages.py in C:\IBKR_Algo_BOT
Run with: python install_packages.py
"""
import subprocess
import sys
import time

def install_package(package):
    """Install a single package"""
    print(f"\n{'='*60}")
    print(f"Installing: {package}")
    print('='*60)
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✓ {package} installed successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to install {package}: {e}")
        return False

# List of required packages with versions
packages = [
    "tensorflow==2.15.0",
    "scikit-learn==1.3.0", 
    "pandas==2.0.3",
    "numpy==1.24.3",
    "yfinance==0.2.28",
    "matplotlib==3.7.2",
    "seaborn==0.12.2",
    "joblib==1.3.2"
]

print("="*60)
print("LSTM TRADING BOT - PACKAGE INSTALLER")
print("="*60)
print(f"\nThis will install {len(packages)} packages:")
for pkg in packages:
    print(f"  • {pkg}")

print("\nEstimated time: 5-10 minutes")
print("Press Ctrl+C to cancel, or wait 3 seconds to start...")

time.sleep(3)

# First, upgrade pip itself
print("\n" + "="*60)
print("Upgrading pip to latest version...")
print("="*60)
try:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    print("✓ pip upgraded successfully")
except:
    print("⚠ Could not upgrade pip, continuing anyway...")

# Install each package
successful = 0
failed = 0
failed_packages = []

for package in packages:
    if install_package(package):
        successful += 1
    else:
        failed += 1
        failed_packages.append(package)
    time.sleep(1)

# Summary
print("\n" + "="*60)
print("INSTALLATION COMPLETE")
print("="*60)
print(f"✓ Successfully installed: {successful}/{len(packages)}")
if failed > 0:
    print(f"✗ Failed: {failed}/{len(packages)}")
    print(f"\nFailed packages:")
    for pkg in failed_packages:
        print(f"  • {pkg}")
print("="*60)

# Verify critical installations
print("\n" + "="*60)
print("VERIFYING INSTALLATIONS")
print("="*60)

all_working = True

# Test TensorFlow
try:
    import tensorflow as tf
    print(f"✓ TensorFlow: {tf.__version__}")
except Exception as e:
    print(f"✗ TensorFlow: {e}")
    all_working = False

# Test Pandas
try:
    import pandas as pd
    print(f"✓ Pandas: {pd.__version__}")
except Exception as e:
    print(f"✗ Pandas: {e}")
    all_working = False

# Test NumPy
try:
    import numpy as np
    print(f"✓ NumPy: {np.__version__}")
except Exception as e:
    print(f"✗ NumPy: {e}")
    all_working = False

# Test scikit-learn
try:
    import sklearn
    print(f"✓ scikit-learn: {sklearn.__version__}")
except Exception as e:
    print(f"✗ scikit-learn: {e}")
    all_working = False

# Test yfinance
try:
    import yfinance
    print(f"✓ yfinance: {yfinance.__version__}")
except Exception as e:
    print(f"✗ yfinance: {e}")
    all_working = False

# Test matplotlib
try:
    import matplotlib
    print(f"✓ matplotlib: {matplotlib.__version__}")
except Exception as e:
    print(f"✗ matplotlib: {e}")
    all_working = False

# Test seaborn
try:
    import seaborn
    print(f"✓ seaborn: {seaborn.__version__}")
except Exception as e:
    print(f"✗ seaborn: {e}")
    all_working = False

# Test joblib
try:
    import joblib
    print(f"✓ joblib: {joblib.__version__}")
except Exception as e:
    print(f"✗ joblib: {e}")
    all_working = False

print("="*60)

if all_working:
    print("\n✅ SUCCESS! All packages installed and working!")
    print("\n📁 Next steps:")
    print("  1. Create folders: mkdir models\\lstm_pipeline models\\lstm_trading data logs")
    print("  2. Run test: python test_lstm.py")
    print("  3. Train models: python train_real_stocks.py")
else:
    print("\n⚠ Some packages failed to install or import")
    print("\n🔧 Troubleshooting:")
    print("  1. Try running this script again")
    print("  2. Try: python -m pip install --upgrade pip")
    print("  3. Check your Python version: python --version")
    print("  4. Make sure you have Python 3.8 or newer")

print("="*60)
