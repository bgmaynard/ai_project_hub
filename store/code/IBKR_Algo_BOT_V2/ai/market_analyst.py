"""
Claude AI Market Analyst Module - REAL INTEGRATION
Provides real-time market analysis using Anthropic's Claude API
"""

from typing import List, Dict, Optional
import json
import os
from datetime import datetime
from anthropic import Anthropic

class MarketAnalyst:
    """
    Analyzes market conditions and provides AI-powered insights using Claude
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the market analyst with Claude API"""
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        
        if not self.api_key:
            print("[WARN] WARNING: No Anthropic API key found!")
            print("   Set ANTHROPIC_API_KEY environment variable or pass to constructor")
            self.client = None
        else:
            self.client = Anthropic(api_key=self.api_key)
            print("[OK] Claude AI initialized successfully")
        
        self.last_analysis_time = None
        self.analysis_history = []
    
    async def analyze_single_stock(
        self, 
        symbol: str,
        timeframe: str = "1D",
        price: Optional[float] = None,
        volume: Optional[int] = None,
        news: Optional[List[str]] = None
    ) -> Dict:
        """
        Deep dive analysis on a single stock using Claude AI
        
        Args:
            symbol: Stock ticker symbol
            timeframe: Analysis timeframe (1D, 1W, 1M)
            price: Current price
            volume: Current volume
            news: Recent news headlines
            
        Returns:
            Detailed stock analysis from Claude
        """
        
        if not self.client:
            # Return empty structure if no API key
            return self._empty_analysis(symbol, timeframe)
        
        try:
            # Build the prompt for Claude
            prompt = self._build_stock_analysis_prompt(
                symbol, timeframe, price, volume, news
            )
            
            # Call Claude API
            message = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2000,
                temperature=0.7,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            
            # Extract Claude's response
            claude_response = message.content[0].text
            
            # Parse response into structured format
            analysis = self._parse_analysis_response(
                symbol, timeframe, claude_response
            )
            
            # Store in history
            self.analysis_history.append(analysis)
            self.last_analysis_time = datetime.now()
            
            return analysis
            
        except Exception as e:
            print(f"[ERROR] Claude API error: {e}")
            # Return error analysis
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "error": str(e),
                "analysis_text": f"Unable to analyze {symbol}: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    def _build_stock_analysis_prompt(
        self,
        symbol: str,
        timeframe: str,
        price: Optional[float],
        volume: Optional[int],
        news: Optional[List[str]]
    ) -> str:
        """Build Claude prompt for stock analysis"""
        
        prompt = f"""You are an expert day trader analyzing momentum stocks. Provide a quick analysis of {symbol}.

**Current Data:**
- Symbol: {symbol}
- Timeframe: {timeframe}
"""
        
        if price:
            prompt += f"- Current Price: ${price:.2f}\n"
        
        if volume:
            prompt += f"- Volume: {volume:,}\n"
        
        if news and len(news) > 0:
            prompt += f"\n**Recent News:**\n"
            for headline in news[:3]:
                prompt += f"- {headline}\n"
        
        prompt += """

**Analysis Required:**

1. **Quick Take** (2-3 sentences): What's your immediate assessment of this stock right now?

2. **Momentum Assessment**: 
   - Is this stock showing strong momentum?
   - Any bullish or bearish signals?
   
3. **Key Levels**:
   - What price levels should traders watch?
   
4. **Trade Idea**:
   - Would you buy, sell, or avoid this stock right now?
   - If buying, what entry point and stop loss?
   
5. **Risk Rating**: LOW, MEDIUM, or HIGH - and why?

Keep it practical and actionable for day traders. Be direct and concise."""
        
        return prompt
    
    def _parse_analysis_response(
        self, 
        symbol: str, 
        timeframe: str, 
        claude_response: str
    ) -> Dict:
        """Parse Claude's text response into structured format"""
        
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "analysis_text": claude_response,
            "timestamp": datetime.now().isoformat(),
            "source": "Claude Sonnet 4.5"
        }
    
    def _empty_analysis(self, symbol: str, timeframe: str) -> Dict:
        """Return empty analysis structure when API key not available"""
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "analysis_text": "⚠️ Claude API key not configured. Set ANTHROPIC_API_KEY environment variable.",
            "timestamp": datetime.now().isoformat(),
            "error": "No API key"
        }
    
    async def analyze_market(
        self, 
        symbols: List[str],
        market_data: Dict,
        news_context: Optional[str] = None
    ) -> Dict:
        """
        Analyze market conditions for multiple symbols
        
        Args:
            symbols: List of ticker symbols
            market_data: Current price data
            news_context: Optional news context
            
        Returns:
            Market analysis from Claude
        """
        
        if not self.client:
            return {
                "error": "No API key configured",
                "symbols": symbols,
                "timestamp": datetime.now().isoformat()
            }
        
        try:
            # Build prompt
            prompt = f"""You are a market analyst. Analyze these stocks: {', '.join(symbols)}

**Market Data:**
{json.dumps(market_data, indent=2)}
"""
            
            if news_context:
                prompt += f"\n**News Context:**\n{news_context}\n"
            
            prompt += """
Provide:
1. Overall market sentiment
2. Top opportunities
3. Stocks to avoid
4. Key risks

Be concise and actionable."""
            
            # Call Claude
            message = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1500,
                messages=[{"role": "user", "content": prompt}]
            )
            
            analysis = {
                "timestamp": datetime.now().isoformat(),
                "symbols": symbols,
                "analysis": message.content[0].text,
                "source": "Claude Sonnet 4.5"
            }
            
            self.analysis_history.append(analysis)
            return analysis
            
        except Exception as e:
            return {
                "error": str(e),
                "symbols": symbols,
                "timestamp": datetime.now().isoformat()
            }
    
    def get_analysis_history(self, limit: int = 10) -> List[Dict]:
        """Retrieve recent analysis history"""
        return self.analysis_history[-limit:]


# Simplified function for quick use
async def simple_market_check(symbols: List[str]) -> str:
    """
    Quick market check using Claude AI
    
    Usage:
        result = await simple_market_check(['AAPL', 'MSFT'])
        print(result)
    """
    
    analyst = MarketAnalyst()
    
    if not analyst.client:
        return "⚠️ Claude API not configured. Set ANTHROPIC_API_KEY environment variable."
    
    # Mock data - would come from IBKR in production
    market_data = {
        symbol: {"price": 0.0, "change": 0.0, "volume": 0}
        for symbol in symbols
    }
    
    analysis = await analyst.analyze_market(symbols, market_data)
    
    if "error" in analysis:
        return f"❌ Error: {analysis['error']}"
    
    return analysis.get("analysis", "No analysis available")


if __name__ == "__main__":
    # Test the module
    import asyncio
    
    async def test():
        print("Testing Claude AI Market Analyst...\n")
        
        # Test single stock
        analyst = MarketAnalyst()
        result = await analyst.analyze_single_stock(
            "AAPL", 
            price=150.25, 
            volume=50000000,
            news=["Apple announces new iPhone", "Strong earnings beat"]
        )
        
        print("Analysis Result:")
        print(json.dumps(result, indent=2))
    
    asyncio.run(test())
