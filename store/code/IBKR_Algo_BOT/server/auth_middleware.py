from fastapi import Header, HTTPException, status
from typing import Optional
from .config import get_settings

def require_api_key(x_api_key: Optional[str] = Header(None)):
    settings = get_settings()
    expected = (settings.api_key or "").strip()
    if not expected:
        # If no API key configured, allow (local dev). Consider tightening to localhost only.
        return True
    if (x_api_key or "").strip() != expected:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key")
    return True
