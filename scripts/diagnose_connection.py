import os
import socket
from pathlib import Path

def main():
    print("🔍 IBKR Connection Diagnostics")
    print("=" * 40)
    
    # Load environment
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("✅ Environment loaded")
    except ImportError:
        print("❌ python-dotenv not installed")
        return
    
    # Check environment variables
    print("\n🔧 Environment Variables:")
    for var in ["TWS_HOST", "TWS_PORT", "TWS_CLIENT_ID", "LOCAL_API_KEY"]:
        value = os.getenv(var)
        display = "***" if "KEY" in var else value
        status = "✅" if value else "❌"
        print(f"   {status} {var}={display}")
    
    # Test TWS connection
    host = os.getenv("TWS_HOST", "127.0.0.1")
    port = int(os.getenv("TWS_PORT", "7497"))
    
    print(f"\n🔗 Testing connection to {host}:{port}")
    try:
        with socket.create_connection((host, port), timeout=5):
            print("✅ TWS is listening and accepting connections")
    except ConnectionRefusedError:
        print("❌ Connection refused - TWS not listening")
        print("💡 Start TWS/Gateway and enable API")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
    
    # Test imports
    print("\n📦 Testing imports:")
    try:
        import ib_insync
        print("✅ ib_insync available")
    except ImportError:
        print("❌ ib_insync not installed")
    
    try:
        import fastapi
        print("✅ fastapi available")
    except ImportError:
        print("❌ fastapi not installed")

if __name__ == "__main__":
    main()
