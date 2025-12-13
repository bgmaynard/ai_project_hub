"""
Diagnose Python and pip installation
Save as: check_python.py in C:\IBKR_Algo_BOT
Run with: python check_python.py
"""
import sys
import subprocess
import os

print("="*60)
print("PYTHON ENVIRONMENT DIAGNOSTIC TOOL")
print("="*60)

# 1. Python version and location
print("\n1. PYTHON VERSION:")
print(f"   Version: {sys.version}")
print(f"   Executable: {sys.executable}")

# Check if 64-bit
is_64bit = sys.maxsize > 2**32
print(f"   Architecture: {'64-bit' if is_64bit else '32-bit'}")

# 2. Python in PATH
print("\n2. PYTHON IN SYSTEM PATH:")
python_path = os.environ.get('PATH', '').split(';')
python_dirs = [p for p in python_path if 'python' in p.lower()]
if python_dirs:
    for p in python_dirs[:5]:
        print(f"   {p}")
else:
    print("   ⚠ No Python directories found in PATH")

# 3. pip module test
print("\n3. PIP MODULE CHECK:")
try:
    import pip
    print("   ✓ pip module found")
    print(f"   Version: {pip.__version__}")
except ImportError:
    print("   ✗ pip module not found")

# 4. pip command test
print("\n4. PIP COMMAND TEST:")
try:
    result = subprocess.run([sys.executable, "-m", "pip", "--version"], 
                          capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        print("   ✓ pip command works!")
        print(f"   {result.stdout.strip()}")
    else:
        print("   ✗ pip command failed")
        print(f"   Error: {result.stderr}")
except subprocess.TimeoutExpired:
    print("   ✗ pip command timed out (may be frozen)")
except Exception as e:
    print(f"   ✗ Error running pip: {e}")

# 5. Check installed packages
print("\n5. REQUIRED PACKAGES CHECK:")
required = {
    'tensorflow': 'TensorFlow',
    'pandas': 'Pandas',
    'numpy': 'NumPy',
    'sklearn': 'scikit-learn',
    'yfinance': 'yfinance',
    'matplotlib': 'Matplotlib',
    'seaborn': 'Seaborn',
    'joblib': 'joblib'
}

installed = []
missing = []

for module, name in required.items():
    try:
        mod = __import__(module)
        version = getattr(mod, '__version__', 'unknown')
        installed.append(name)
        print(f"   ✓ {name:20} {version}")
    except ImportError:
        missing.append(name)
        print(f"   ✗ {name:20} NOT INSTALLED")

# 6. Check disk space
print("\n6. SYSTEM INFORMATION:")
try:
    import shutil
    total, used, free = shutil.disk_usage("C:\\")
    print(f"   Disk Space (C:): {free // (2**30)} GB free")
except:
    print("   Could not check disk space")

# 7. Check internet connectivity
print("\n7. INTERNET CONNECTIVITY:")
try:
    import urllib.request
    urllib.request.urlopen('https://pypi.org', timeout=3)
    print("   ✓ Can reach PyPI (package repository)")
except:
    print("   ✗ Cannot reach PyPI - check internet connection")

# Summary
print("\n" + "="*60)
print("DIAGNOSTIC SUMMARY")
print("="*60)
print(f"Python Version: {sys.version.split()[0]}")
print(f"Packages Installed: {len(installed)}/{len(required)}")
print(f"Packages Missing:   {len(missing)}/{len(required)}")

# Recommendations
print("\n" + "="*60)
print("RECOMMENDATIONS")
print("="*60)

if len(missing) == 0:
    print("✅ All required packages are installed!")
    print("\n📁 Next steps:")
    print("   1. Create folders: mkdir models\\lstm_pipeline models\\lstm_trading data logs")
    print("   2. Run test: python test_lstm.py")
    print("   3. Train models: python train_real_stocks.py")

elif len(missing) < len(required):
    print(f"⚠ {len(missing)} package(s) still need to be installed")
    print("\n📝 To install missing packages:")
    missing_modules = [k for k, v in required.items() if v in missing]
    print(f"   python -m pip install {' '.join(missing_modules)}")

else:
    print("❌ No packages are installed yet")
    print("\n📝 To install all required packages:")
    print("   Method 1 (Recommended):")
    print("      python install_packages.py")
    print("\n   Method 2 (Manual):")
    print("      python -m pip install tensorflow scikit-learn pandas numpy yfinance matplotlib seaborn joblib")

# Check for common issues
print("\n" + "="*60)
print("COMMON ISSUES CHECK")
print("="*60)

# Check Python version compatibility
version_parts = sys.version.split()[0].split('.')
major = int(version_parts[0])
minor = int(version_parts[1])

if major < 3 or (major == 3 and minor < 8):
    print("❌ Python version too old!")
    print(f"   Current: {major}.{minor}")
    print("   Required: 3.8 or newer")
    print("   Please upgrade Python from: https://www.python.org/downloads/")
elif major == 3 and minor > 11:
    print("⚠ Python version very new")
    print(f"   Current: {major}.{minor}")
    print("   TensorFlow may not be fully compatible yet")
    print("   Consider using Python 3.11 if you have issues")
else:
    print(f"✓ Python version compatible: {major}.{minor}")

# Check for virtual environment
if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
    print("ℹ Running in virtual environment")
else:
    print("ℹ Running in system Python (not virtual environment)")

print("="*60)
print("\n💡 TIP: If pip doesn't work, always use: python -m pip install [package]")
print("="*60)
