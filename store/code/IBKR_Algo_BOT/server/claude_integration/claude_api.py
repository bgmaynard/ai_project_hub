"""
FastAPI Integration for Claude AI Trading Modules
Connects AI analysis tools to your existing IBKR trading system

This file shows how to add AI-powered endpoints to your dashboard_api.py
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

# Import our AI modules
from market_analyst import MarketAnalyst, simple_market_check
from trade_validator import TradeValidator, quick_trade_check

# Initialize AI components
market_analyst = MarketAnalyst()
trade_validator = TradeValidator()

# Create FastAPI app (in production, this would be added to your existing dashboard_api.py)
app = FastAPI(title="Claude AI Trading Assistant")


# ============================================================================
# DATA MODELS (for API requests/responses)
# ============================================================================

class MarketAnalysisRequest(BaseModel):
    """Request model for market analysis"""
    symbols: List[str]
    include_news: bool = False

class TradeValidationRequest(BaseModel):
    """Request model for trade validation"""
    symbol: str
    action: str  # BUY, SELL, HOLD
    quantity: int
    entry_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    reason: str = ""

class PortfolioRiskRequest(BaseModel):
    """Request model for portfolio risk analysis"""
    positions: List[dict]
    portfolio_value: float


# ============================================================================
# AI-POWERED API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "Claude AI Trading Assistant",
        "version": "2.0",
        "timestamp": datetime.now().isoformat()
    }


@app.post("/api/claude/analyze-market")
async def analyze_market(request: MarketAnalysisRequest):
    """
    Get AI-powered market analysis for specified symbols
    
    Example usage:
        POST /api/claude/analyze-market
        {
            "symbols": ["AAPL", "MSFT", "GOOGL"],
            "include_news": true
        }
    """
    try:
        # For now, returns simplified analysis
        # In production, this would call real market data from IBKR
        result = await simple_market_check(request.symbols)
        
        return {
            "success": True,
            "analysis": result,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/claude/validate-trade")
async def validate_trade(request: TradeValidationRequest):
    """
    Validate a trade before execution
    
    Example usage:
        POST /api/claude/validate-trade
        {
            "symbol": "AAPL",
            "action": "BUY",
            "quantity": 100,
            "entry_price": 150.00,
            "stop_loss": 145.00,
            "take_profit": 160.00,
            "reason": "Breakout above resistance"
        }
    """
    try:
        validation_result = await trade_validator.validate_trade(
            symbol=request.symbol,
            action=request.action,
            quantity=request.quantity,
            entry_price=request.entry_price,
            stop_loss=request.stop_loss,
            take_profit=request.take_profit,
            reason=request.reason
        )
        
        return {
            "success": True,
            "validation": validation_result,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/claude/quick-check/{symbol}")
async def quick_stock_check(symbol: str):
    """
    Quick check on a single stock
    
    Example usage:
        GET /api/claude/quick-check/AAPL
    """
    try:
        analysis = await market_analyst.analyze_single_stock(symbol)
        
        return {
            "success": True,
            "symbol": symbol,
            "analysis": analysis,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/claude/portfolio-risk")
async def analyze_portfolio_risk(request: PortfolioRiskRequest):
    """
    Analyze portfolio risk and concentration
    
    Example usage:
        POST /api/claude/portfolio-risk
        {
            "positions": [
                {"symbol": "AAPL", "quantity": 100, "avg_price": 150},
                {"symbol": "MSFT", "quantity": 50, "avg_price": 300}
            ],
            "portfolio_value": 50000
        }
    """
    try:
        # Calculate risk metrics
        risk_analysis = {
            "portfolio_value": request.portfolio_value,
            "number_of_positions": len(request.positions),
            "concentration_warnings": [],
            "diversification_score": 0,
            "risk_level": "MODERATE",
            "recommendations": []
        }
        
        # Check for concentration issues
        for position in request.positions:
            position_value = position["quantity"] * position["avg_price"]
            concentration = (position_value / request.portfolio_value) * 100
            
            if concentration > 20:
                risk_analysis["concentration_warnings"].append({
                    "symbol": position["symbol"],
                    "concentration_percent": round(concentration, 2),
                    "message": f"⚠️  {position['symbol']} represents {concentration:.1f}% of portfolio (high concentration)"
                })
        
        # Generate recommendations
        if len(request.positions) < 5:
            risk_analysis["recommendations"].append(
                "Consider adding more positions for better diversification"
            )
        
        if risk_analysis["concentration_warnings"]:
            risk_analysis["risk_level"] = "HIGH"
            risk_analysis["recommendations"].append(
                "Consider reducing concentration in overweight positions"
            )
        
        return {
            "success": True,
            "risk_analysis": risk_analysis,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/claude/daily-summary")
async def get_daily_summary():
    """
    Get daily market summary and action items
    
    Example usage:
        GET /api/claude/daily-summary
    """
    try:
        summary = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "market_status": "OPEN",  # Would check actual market hours
            "key_events": [
                "Market opened higher on positive earnings",
                "Tech sector showing strength",
                "Watch for Fed announcement at 2PM"
            ],
            "your_portfolio": {
                "performance_today": "+1.2%",
                "alerts": [],
                "action_items": []
            },
            "opportunities": [
                "AAPL showing bullish momentum",
                "Consider taking profits on TSLA position"
            ]
        }
        
        return {
            "success": True,
            "summary": summary,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/claude/validation-stats")
async def get_validation_statistics(days: int = 7):
    """
    Get trade validation statistics
    
    Example usage:
        GET /api/claude/validation-stats?days=7
    """
    try:
        stats = trade_validator.get_validation_summary(days=days)
        
        return {
            "success": True,
            "statistics": stats,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# INTEGRATION GUIDE FOR EXISTING SYSTEM
# ============================================================================

"""
TO INTEGRATE WITH YOUR EXISTING IBKR SYSTEM:

1. Copy these endpoint functions to your dashboard_api.py file

2. Import the AI modules at the top of dashboard_api.py:
   
   from market_analyst import MarketAnalyst
   from trade_validator import TradeValidator

3. Initialize them in your app startup:
   
   market_analyst = MarketAnalyst()
   trade_validator = TradeValidator()

4. Connect to your IBKR data:
   
   Instead of mock data, replace with actual IBKR API calls:
   
   @app.post("/api/claude/analyze-market")
   async def analyze_market(request: MarketAnalysisRequest):
       # Get real market data from IBKR
       ib = get_ib_connection()
       market_data = {}
       
       for symbol in request.symbols:
           contract = Stock(symbol, 'SMART', 'USD')
           ticker = ib.reqMktData(contract)
           ib.sleep(1)  # Wait for data
           
           market_data[symbol] = {
               "price": ticker.last,
               "change": ticker.change,
               "volume": ticker.volume
           }
       
       # Now pass real data to analyst
       analysis = await market_analyst.analyze_market(
           request.symbols, 
           market_data
       )
       
       return {"success": True, "analysis": analysis}

5. Add to your UI:
   
   In platform.html, add buttons to call these endpoints:
   
   <button onclick="analyzeMarket()">Analyze Market</button>
   <button onclick="validateTrade()">Validate Trade</button>
   
   <script>
   async function analyzeMarket() {
       const response = await fetch('/api/claude/analyze-market', {
           method: 'POST',
           headers: {'Content-Type': 'application/json'},
           body: JSON.stringify({
               symbols: ['AAPL', 'MSFT', 'GOOGL'],
               include_news: true
           })
       });
       const data = await response.json();
       console.log(data);
       // Display in UI
   }
   </script>

6. Test the integration:
   
   python -m pytest tests/test_ai_integration.py

"""


# ============================================================================
# STARTUP MESSAGE
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Display startup information"""
    print("\n" + "="*70)
    print("🤖 CLAUDE AI TRADING ASSISTANT - STARTING UP")
    print("="*70)
    print("\n✅ AI Modules Loaded:")
    print("   • Market Analyst")
    print("   • Trade Validator")
    print("\n📊 Available Endpoints:")
    print("   • POST /api/claude/analyze-market")
    print("   • POST /api/claude/validate-trade")
    print("   • POST /api/claude/portfolio-risk")
    print("   • GET  /api/claude/daily-summary")
    print("   • GET  /api/claude/quick-check/{symbol}")
    print("   • GET  /api/claude/validation-stats")
    print("\n💡 Quick Start:")
    print("   1. Open browser to http://localhost:8000/docs")
    print("   2. Test endpoints with interactive API documentation")
    print("   3. Integrate with your existing IBKR system")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    import uvicorn
    
    # Run the server
    print("\n🚀 Starting Claude AI Trading Assistant...")
    print("📍 Server will be available at: http://localhost:8000")
    print("📖 API Documentation at: http://localhost:8000/docs\n")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )
