import subprocess
import sys
import os
from pathlib import Path

def main():
    print("🚀 Starting IBKR Trading Bot...")
    
    # Load environment
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("✅ Environment loaded")
    except ImportError:
        print("❌ Installing python-dotenv...")
        subprocess.run([sys.executable, "-m", "pip", "install", "python-dotenv"])
        from dotenv import load_dotenv
        load_dotenv()
    
    # Check TWS connection
    import socket
    host = os.getenv("TWS_HOST", "127.0.0.1")
    port = int(os.getenv("TWS_PORT", "7497"))
    
    try:
        with socket.create_connection((host, port), timeout=3):
            print(f"✅ TWS detected on {host}:{port}")
    except:
        print(f"❌ TWS not detected on {host}:{port}")
        print("💡 Start TWS/Gateway and enable API settings")
    
    # Start API
    api_path = Path("store/code/IBKR_Algo_BOT/dashboard_api.py")
    if api_path.exists():
        print("🚀 Starting dashboard API...")
        os.chdir("store/code/IBKR_Algo_BOT")
        subprocess.run([sys.executable, "dashboard_api.py"])
    else:
        print("❌ dashboard_api.py not found")

if __name__ == "__main__":
    main()
