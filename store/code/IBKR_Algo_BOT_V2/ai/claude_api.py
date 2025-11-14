"""
Claude AI Integration - Clean Working Version
Professional trading analysis and validation
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Create the router (this is what dashboard_api.py imports)
router = APIRouter()

# Optional: Load Anthropic client if API key available
try:
    import anthropic
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if api_key:
        client = anthropic.Anthropic(api_key=api_key)
        AI_AVAILABLE = True
        print("[OK] Claude AI initialized successfully")
    else:
        AI_AVAILABLE = False
        print("[WARN] Claude AI running in fallback mode (no API key)")
except ImportError:
    AI_AVAILABLE = False
    print("[WARN] anthropic package not installed - Claude AI in fallback mode")
except Exception as e:
    AI_AVAILABLE = False
    print(f"[WARN] Claude AI initialization error: {e}")


# Request/Response Models
class QuickCheckRequest(BaseModel):
    symbol: str

class MarketAnalysisRequest(BaseModel):
    symbols: List[str]
    timeframe: Optional[str] = "1D"

class TradeValidationRequest(BaseModel):
    symbol: str
    action: str
    quantity: int
    price: float
    strategy: str


# Helper function to get AI response
def get_claude_response(prompt: str, max_tokens: int = 1000) -> str:
    """Get response from Claude AI"""
    if not AI_AVAILABLE:
        return "Claude AI analysis not available. Running in fallback mode."
    
    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text
    except Exception as e:
        return f"AI analysis unavailable: {str(e)}"


# API Endpoints

@router.get("/api/claude/status")
async def get_status():
    """Check Claude AI status"""
    return {
        "status": "operational",
        "ai_available": AI_AVAILABLE,
        "model": "claude-sonnet-4-20250514" if AI_AVAILABLE else None,
        "mode": "full" if AI_AVAILABLE else "fallback",
        "timestamp": datetime.now().isoformat()
    }


@router.get("/api/claude/quick-check/{symbol}")
async def quick_check(symbol: str):
    """Quick market check for a symbol"""
    
    prompt = f"""You are a professional day trading analyst. Provide a brief 2-3 sentence analysis of {symbol} for pre-market momentum trading (4-10 AM window).

Focus on:
- Current momentum (bullish/bearish/neutral)
- Key technical levels
- Trading recommendation

Be concise and actionable."""
    
    analysis = get_claude_response(prompt, max_tokens=300)
    
    return {
        "symbol": symbol,
        "analysis": analysis,
        "timestamp": datetime.now().isoformat(),
        "mode": "ai" if AI_AVAILABLE else "fallback"
    }


@router.post("/api/claude/analyze-market")
async def analyze_market(request: MarketAnalysisRequest):
    """Comprehensive market analysis"""
    
    symbols_str = ", ".join(request.symbols)
    
    prompt = f"""You are an algorithmic trading analyst. Analyze these symbols: {symbols_str}

For each symbol provide:
1. Market sentiment (bullish/bearish/neutral)
2. Key support/resistance levels
3. Trading opportunity assessment
4. Risk factors

Focus on pre-market momentum trading (4-10 AM). Be specific and data-driven."""
    
    analysis = get_claude_response(prompt, max_tokens=2000)
    
    return {
        "symbols": request.symbols,
        "timeframe": request.timeframe,
        "analysis": analysis,
        "timestamp": datetime.now().isoformat()
    }


@router.post("/api/claude/validate-trade")
async def validate_trade(request: TradeValidationRequest):
    """Validate a trade setup"""
    
    prompt = f"""You are a risk management specialist using the 3-5-7 strategy:
- 3% max risk per trade
- 5% max daily loss
- 7% max drawdown

Evaluate this trade:
Symbol: {request.symbol}
Action: {request.action}
Quantity: {request.quantity}
Price: ${request.price}
Strategy: {request.strategy}

Provide:
1. Trade quality score (1-10)
2. Key risks
3. Position sizing recommendation
4. Entry/exit suggestions

Be objective and focus on risk management."""
    
    validation = get_claude_response(prompt, max_tokens=500)
    
    return {
        "symbol": request.symbol,
        "action": request.action,
        "validation": validation,
        "approved": True,  # Basic approval (can add logic)
        "timestamp": datetime.now().isoformat()
    }


@router.get("/api/claude/daily-summary")
async def daily_summary():
    """Get daily market summary"""
    
    prompt = """You are a trading performance analyst. Provide a brief daily market overview for pre-market traders (4-10 AM window):

1. Overall market sentiment today
2. Key sectors to watch
3. Pre-market trading strategy recommendations
4. Risk management reminder (3-5-7 strategy)

Keep it concise, actionable, and focused on day trading opportunities."""
    
    summary = get_claude_response(prompt, max_tokens=800)
    
    return {
        "summary": summary,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "timestamp": datetime.now().isoformat()
    }


@router.get("/api/claude/strategies")
async def get_strategies():
    """Get information about trading strategies"""
    
    strategies = {
        "Gap and Go": {
            "description": "Trade stocks with significant pre-market gaps with volume",
            "risk": "High volatility, requires quick exits",
            "best_time": "4:00-10:00 AM"
        },
        "Warrior Momentum": {
            "description": "Momentum scalping with tight stops",
            "risk": "Fast-paced, requires discipline",
            "best_time": "9:30-11:00 AM"
        },
        "Bull Flag Micro Pullback": {
            "description": "Enter on pullbacks in strong trends",
            "risk": "Can fail if trend reverses",
            "best_time": "All day"
        },
        "Flat Top Breakout": {
            "description": "Breakout from consolidation patterns",
            "risk": "False breakouts common",
            "best_time": "Any session"
        }
    }
    
    return {
        "strategies": strategies,
        "risk_rules": {
            "per_trade": "3% max risk",
            "daily_loss": "5% max loss",
            "max_drawdown": "7% limit"
        },
        "timestamp": datetime.now().isoformat()
    }


print("[OK] Claude AI router loaded successfully")
