import os
import asyncio
import logging
from datetime import datetime
from pathlib import Path

# Load environment first
try:
    from dotenv import load_dotenv
    project_root = Path(__file__).parent.parent.parent
    load_dotenv(project_root / '.env')
    print("✅ Environment loaded")
except ImportError:
    print("⚠️ python-dotenv not installed")

# FastAPI imports
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    import uvicorn
    print("✅ FastAPI imported")
except ImportError:
    print("❌ FastAPI not installed - run: pip install fastapi uvicorn")
    exit(1)

# Try to import IBKR adapter
try:
    from bridge.ib_adapter import IBAdapter, IBConfig
    print("✅ IB Adapter imported")
    IBKR_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ IB Adapter import failed: {e}")
    print("⚠️ Starting without IBKR connection")
    IBKR_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global adapter instance
ib_adapter = None

# Initialize FastAPI app
app = FastAPI(
    title="IBKR Trading Bot API",
    description="IBKR Trading Bot with Connection Diagnostics",
    version="1.0.0"
)

@app.on_event("startup")
async def startup():
    global ib_adapter
    
    print("🚀 Starting IBKR Dashboard API...")
    
    # Environment validation
    required_vars = ["LOCAL_API_KEY", "TWS_HOST", "TWS_PORT", "TWS_CLIENT_ID"]
    for var in required_vars:
        value = os.getenv(var)
        if value:
            display_value = "***" if "KEY" in var else value
            print(f"✅ {var}={display_value}")
        else:
            print(f"❌ Missing {var} in .env")
    
    # Create IBKR adapter if available
    if IBKR_AVAILABLE:
        try:
            config = IBConfig()
            ib_adapter = IBAdapter(config)
            print("✅ IB Adapter created")
        except Exception as e:
            print(f"❌ Error creating IB Adapter: {e}")
            ib_adapter = None
    
    print("✅ Dashboard API startup complete")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "ibkr_available": IBKR_AVAILABLE,
        "adapter_created": ib_adapter is not None
    }

@app.get("/api/status")
async def get_status():
    if not ib_adapter:
        return {
            "ib_connection": False,
            "state": "NOT_AVAILABLE",
            "error": "IB Adapter not available",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    # Get status from adapter
    try:
        status = ib_adapter.get_connection_status()
        return status
    except Exception as e:
        return {
            "ib_connection": False,
            "state": "ERROR",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

@app.get("/api/debug")
async def debug_info():
    import socket
    
    # Environment check
    env_vars = {}
    for var in ["TWS_HOST", "TWS_PORT", "TWS_CLIENT_ID", "LOCAL_API_KEY"]:
        value = os.getenv(var)
        env_vars[var] = {
            "set": bool(value),
            "value": "***" if "KEY" in var else value
        }
    
    # Socket test
    host = os.getenv("TWS_HOST", "127.0.0.1")
    port = int(os.getenv("TWS_PORT", "7497"))
    
    socket_test = {"success": False, "error": None}
    try:
        with socket.create_connection((host, port), timeout=5):
            socket_test["success"] = True
    except Exception as e:
        socket_test["error"] = str(e)
    
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "environment": env_vars,
        "socket_test": {
            "host": host,
            "port": port,
            "success": socket_test["success"],
            "error": socket_test.get("error")
        },
        "ibkr_available": IBKR_AVAILABLE,
        "adapter_status": "created" if ib_adapter else "not_created"
    }

@app.get("/")
async def root():
    return {
        "message": "IBKR Trading Bot API",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "status": "/api/status", 
            "debug": "/api/debug"
        }
    }

if __name__ == "__main__":
    print("🚀 Starting IBKR Dashboard API Server...")
    print("🌐 API will be available at: http://127.0.0.1:9101")
    print("🔍 Debug info at: http://127.0.0.1:9101/api/debug")
    print("❤️ Health check at: http://127.0.0.1:9101/health")
    print()
    
    uvicorn.run(
        "dashboard_api:app",
        host=os.getenv("API_HOST", "127.0.0.1"),
        port=int(os.getenv("API_PORT", "9101")),
        reload=False
    )
