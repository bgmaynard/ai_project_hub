"""
IBKR Trading Bot Dashboard API - FINAL WORKING VERSION
Uses synchronous IBKR connection
"""
import os
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Try to import IB components
ADAPTER_OK = False
ib_adapter = None
startup_error: Optional[str] = None

try:
    from store.code.IBKR_Algo_BOT.bridge.ib_adapter import IBConfig, IBAdapter
    ADAPTER_OK = True
    print("✅ IB Adapter import OK")
except Exception as e:
    print(f"⚠️ IB Adapter import failed: {e}")

app = FastAPI(title="IBKR Trading Bot API", version="2.0.0")

# Helper to create adapter
def _make_adapter():
    global startup_error
    try:
        cfg = IBConfig()
        adp = IBAdapter(cfg)
        adp.connect()  # Synchronous connect
        return adp
    except Exception as e:
        startup_error = str(e)
        return None

# Startup event
@app.on_event("startup")
def _on_startup():
    global ib_adapter
    print("🚀 Starting IBKR Dashboard API...")
    ib_adapter = _make_adapter()
    print("✅ Startup complete")

# Models
class OrderIn(BaseModel):
    symbol: str
    side: str
    qty: int
    limitPrice: float | None = None

# API Key protection
def require_api_key(x_api_key: str = Header(None)):
    expected = os.getenv("LOCAL_API_KEY")
    if not expected:
        raise HTTPException(500, "LOCAL_API_KEY not configured")
    if x_api_key != expected:
        raise HTTPException(401, "Invalid or missing X-API-Key")
    return True

# Health endpoint
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    }

# Status endpoint
@app.get("/api/status")
def status():
    out = {
        "timestamp": datetime.utcnow().isoformat(),
        "ib_connection": False,
        "ai_connection": True,
        "current_client_id": None,
        "state": "UNKNOWN",
        "api_key_configured": bool(os.getenv("LOCAL_API_KEY")),
        "mode": "simulation"
    }
    
    if ib_adapter:
        try:
            status_data = ib_adapter.get_status()
            out.update({
                "ib_connection": ib_adapter.is_connected(),
                "current_client_id": status_data.get("current_client_id"),
                "state": status_data.get("state"),
                "host": status_data.get("host"),
                "port": status_data.get("port"),
                "mode": "live" if ib_adapter.is_connected() else "simulation"
            })
        except Exception as e:
            out["error"] = str(e)
    else:
        out["state"] = "NOT_INITIALIZED"
        out["error"] = startup_error or "Adapter not created"
    
    return out

# Order endpoints (protected)
@app.post("/api/order/preview")
def order_preview(body: OrderIn, _=Depends(require_api_key)):
    if not ib_adapter:
        raise HTTPException(503, "Adapter not available")
    
    return {
        "status": "preview_ready",
        "order": body.dict(),
        "timestamp": datetime.utcnow().isoformat()
    }

# AI prediction endpoint
@app.post("/api/ai/predict")
async def ai_predict(data: dict):
    symbol = data.get("symbol", "UNKNOWN")
    return {
        "symbol": symbol,
        "prob_up": 0.65,
        "confidence": 0.75,
        "timestamp": datetime.utcnow().isoformat()
    }

# UI file serving
@app.get("/ui/{file_path:path}")
async def serve_ui(file_path: str):
    from pathlib import Path
    ui_path = Path("ui") / file_path
    if ui_path.exists() and ui_path.is_file():
        return FileResponse(ui_path)
    raise HTTPException(404, "File not found")

# Root endpoint
@app.get("/")
def root():
    return {
        "message": "IBKR Trading Bot API",
        "version": "2.0.0",
        "status_endpoint": "/api/status"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "dashboard_api:app",
        host=os.getenv("API_HOST", "127.0.0.1"),
        port=int(os.getenv("API_PORT", "9101")),
        reload=True
    )
