"""
Claude Conversational AI Module
==============================
Full-featured conversational AI with Claude-like reasoning capabilities.
Provides unrestricted intelligent conversation with trading context awareness.

Features:
- Full conversational ability like Claude.ai
- Persistent conversation memory
- Deep reasoning and analysis
- Trading context awareness
- Tool use for real-time data
- Multi-turn conversations with full context

Author: AI Trading Bot Team
Version: 1.0
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class ClaudeConversationalAI:
    """
    Full-featured conversational AI providing Claude-like reasoning and interaction.
    This is designed to give the same quality of conversation as claude.ai.
    """

    def __init__(self):
        """Initialize the conversational AI"""
        self.api_key = os.getenv("ANTHROPIC_API_KEY", "")
        self.client = None
        self.ai_available = False

        # Initialize Anthropic client
        try:
            import anthropic
            if self.api_key:
                self.client = anthropic.Anthropic(api_key=self.api_key)
                self.ai_available = True
                logger.info("Claude Conversational AI initialized successfully")
            else:
                logger.warning("No ANTHROPIC_API_KEY found - AI features disabled")
        except ImportError:
            logger.warning("anthropic package not installed - run: pip install anthropic")
        except Exception as e:
            logger.error(f"Failed to initialize Claude: {e}")

        # Use the most capable model for full reasoning
        self.model = "claude-sonnet-4-5-20250929"
        self.max_tokens = 8192  # Allow longer responses for detailed reasoning

        # Conversation state - persisted across messages
        self.conversations: Dict[str, List[Dict]] = {}  # session_id -> messages
        self.default_session = "default"

        # Storage
        self.data_path = Path("store/conversations")
        self.data_path.mkdir(parents=True, exist_ok=True)

        # Load existing conversations
        self._load_conversations()

        # Define available tools for real-time data
        self.tools = self._define_tools()

    def _load_conversations(self):
        """Load saved conversation history"""
        try:
            conv_file = self.data_path / "conversation_history.json"
            if conv_file.exists():
                with open(conv_file, "r") as f:
                    self.conversations = json.load(f)
                logger.info(f"Loaded {len(self.conversations)} conversation sessions")
        except Exception as e:
            logger.warning(f"Could not load conversations: {e}")
            self.conversations = {}

    def _save_conversations(self):
        """Save conversation history"""
        try:
            conv_file = self.data_path / "conversation_history.json"
            # Keep only last 50 messages per session
            trimmed = {}
            for session_id, messages in self.conversations.items():
                trimmed[session_id] = messages[-50:]
            with open(conv_file, "w") as f:
                json.dump(trimmed, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save conversations: {e}")

    def _define_tools(self) -> List[Dict]:
        """Define tools for real-time data access"""
        return [
            {
                "name": "get_account_info",
                "description": "Get current trading account information including equity, buying power, and positions",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "get_positions",
                "description": "Get all current open positions with P&L",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "get_stock_quote",
                "description": "Get current quote/price for a stock symbol",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock symbol (e.g., AAPL, TSLA, SPY)"
                        }
                    },
                    "required": ["symbol"]
                }
            },
            {
                "name": "get_stock_bars",
                "description": "Get historical price bars for technical analysis",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock symbol"
                        },
                        "timeframe": {
                            "type": "string",
                            "description": "Bar timeframe (1Min, 5Min, 15Min, 1Hour, 1Day)",
                            "default": "1Day"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Number of bars to fetch",
                            "default": 20
                        }
                    },
                    "required": ["symbol"]
                }
            },
            {
                "name": "get_market_status",
                "description": "Check if the market is open and get market clock info",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "get_bot_status",
                "description": "Get the trading bot's current status, configuration, and recent activity",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "get_ai_prediction",
                "description": "Get AI prediction for a stock symbol",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock symbol to analyze"
                        }
                    },
                    "required": ["symbol"]
                }
            },
            {
                "name": "get_regime_analysis",
                "description": "Get market regime analysis (trending, ranging, volatile, etc.)",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        ]

    def _execute_tool(self, tool_name: str, tool_input: Dict) -> Dict:
        """Execute a tool and return results"""
        try:
            if tool_name == "get_account_info":
                from alpaca_integration import get_alpaca_connector
                connector = get_alpaca_connector()
                return connector.get_account()

            elif tool_name == "get_positions":
                from alpaca_integration import get_alpaca_connector
                connector = get_alpaca_connector()
                positions = connector.get_positions()
                return {"positions": positions, "count": len(positions)}

            elif tool_name == "get_stock_quote":
                from alpaca_market_data import get_alpaca_market_data
                market_data = get_alpaca_market_data()
                symbol = tool_input.get("symbol", "SPY").upper()
                quote = market_data.get_latest_quote(symbol)
                snapshot = market_data.get_snapshot(symbol)
                return {
                    "symbol": symbol,
                    "quote": quote,
                    "snapshot": snapshot
                }

            elif tool_name == "get_stock_bars":
                from alpaca_market_data import get_alpaca_market_data
                market_data = get_alpaca_market_data()
                symbol = tool_input.get("symbol", "SPY").upper()
                timeframe = tool_input.get("timeframe", "1Day")
                limit = tool_input.get("limit", 20)
                bars = market_data.get_bars(symbol, timeframe=timeframe, limit=limit)
                return {"symbol": symbol, "bars": bars, "count": len(bars) if bars else 0}

            elif tool_name == "get_market_status":
                from alpaca_integration import get_alpaca_connector
                connector = get_alpaca_connector()
                clock = connector.get_clock()
                return clock

            elif tool_name == "get_bot_status":
                from bot_manager import get_bot_manager
                bot = get_bot_manager()
                return bot.get_status()

            elif tool_name == "get_ai_prediction":
                from ai.alpaca_ai_predictor import get_alpaca_predictor
                predictor = get_alpaca_predictor()
                symbol = tool_input.get("symbol", "SPY").upper()
                return predictor.predict(symbol)

            elif tool_name == "get_regime_analysis":
                from bot_manager import get_bot_manager
                bot = get_bot_manager()
                return bot.classify_market_regime()

            else:
                return {"error": f"Unknown tool: {tool_name}"}

        except Exception as e:
            logger.error(f"Tool {tool_name} error: {e}")
            return {"error": str(e)}

    def _get_system_prompt(self) -> str:
        """Get the system prompt for full conversational AI"""
        return """You are Claude, a helpful AI assistant integrated into a trading bot platform. You have full conversational capabilities - you can discuss any topic, provide detailed analysis, explain concepts, and engage in natural dialogue just like the Claude.ai web interface.

ABOUT YOUR ENVIRONMENT:
You are embedded in an algorithmic trading platform that trades stocks via Alpaca. You have access to real-time market data, account information, positions, and the AI predictor's signals.

YOUR CAPABILITIES:
1. FULL CONVERSATION - You can discuss anything: trading strategies, market analysis, general knowledge, coding help, explanations, creative tasks, etc.
2. REAL-TIME DATA - Use your tools to fetch live market data, account status, positions when relevant
3. TRADING ANALYSIS - Provide deep analysis of stocks, market conditions, and trading strategies
4. REASONING - Show your thinking process, explain your logic, provide nuanced perspectives
5. MEMORY - You remember the conversation history and can reference earlier messages

GUIDELINES:
- Be genuinely helpful and conversational, not robotic
- When discussing trading, use real data from your tools
- Show your reasoning process for complex questions
- Admit uncertainty when appropriate
- Provide balanced perspectives on market views
- For trading questions, always consider risk management
- You can ask clarifying questions if needed

You are NOT limited to just trading topics. You can help with:
- General questions and discussions
- Coding and technical help
- Explanations and education
- Creative tasks
- Analysis and research
- Any topic the user wants to discuss

Be conversational, thoughtful, and genuinely helpful. This should feel like chatting with Claude.ai."""

    def chat(self, message: str, session_id: str = None,
             use_tools: bool = True) -> Dict:
        """
        Full conversational chat with Claude-like reasoning.

        Args:
            message: User's message
            session_id: Optional session ID for conversation continuity
            use_tools: Whether to enable real-time data tools

        Returns:
            Response with full reasoning
        """
        session_id = session_id or self.default_session

        # Initialize session if needed
        if session_id not in self.conversations:
            self.conversations[session_id] = []

        # Add user message to history
        self.conversations[session_id].append({
            "role": "user",
            "content": message,
            "timestamp": datetime.now().isoformat()
        })

        if not self.ai_available:
            response = self._get_fallback_response(message)
            self.conversations[session_id].append({
                "role": "assistant",
                "content": response,
                "timestamp": datetime.now().isoformat()
            })
            self._save_conversations()
            return {
                "response": response,
                "session_id": session_id,
                "ai_available": False,
                "timestamp": datetime.now().isoformat()
            }

        try:
            # Build messages for API call (include conversation history)
            api_messages = []
            for msg in self.conversations[session_id][-20:]:  # Last 20 messages for context
                api_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })

            # Make API call with tools
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=self._get_system_prompt(),
                tools=self.tools if use_tools else [],
                messages=api_messages
            )

            # Handle tool use loop
            max_iterations = 5
            iteration = 0

            while response.stop_reason == "tool_use" and iteration < max_iterations:
                iteration += 1

                # Execute all tool calls
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        result = self._execute_tool(block.name, block.input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(result, default=str)
                        })
                        logger.info(f"Executed tool: {block.name}")

                # Add assistant response and tool results
                api_messages.append({"role": "assistant", "content": response.content})
                api_messages.append({"role": "user", "content": tool_results})

                # Get next response
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    system=self._get_system_prompt(),
                    tools=self.tools if use_tools else [],
                    messages=api_messages
                )

            # Extract final text response
            final_text = ""
            for block in response.content:
                if hasattr(block, 'text'):
                    final_text += block.text

            # Add assistant response to history
            self.conversations[session_id].append({
                "role": "assistant",
                "content": final_text,
                "timestamp": datetime.now().isoformat()
            })

            self._save_conversations()

            return {
                "response": final_text,
                "session_id": session_id,
                "ai_available": True,
                "model": self.model,
                "tools_used": iteration > 0,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Chat error: {e}")
            error_response = f"I encountered an error: {str(e)}. Please try again."

            self.conversations[session_id].append({
                "role": "assistant",
                "content": error_response,
                "timestamp": datetime.now().isoformat()
            })
            self._save_conversations()

            return {
                "response": error_response,
                "session_id": session_id,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def _get_fallback_response(self, message: str) -> str:
        """Fallback when AI is not available"""
        return f"""I apologize, but my AI capabilities are currently unavailable (missing API key or connection issue).

To enable full conversational AI:
1. Ensure you have an ANTHROPIC_API_KEY in your .env file
2. Install the anthropic package: pip install anthropic
3. Restart the server

Your message was: "{message[:100]}..."

In the meantime, you can still use the trading platform's other features."""

    def get_conversation_history(self, session_id: str = None) -> List[Dict]:
        """Get conversation history for a session"""
        session_id = session_id or self.default_session
        return self.conversations.get(session_id, [])

    def clear_conversation(self, session_id: str = None) -> Dict:
        """Clear conversation history for a session"""
        session_id = session_id or self.default_session
        self.conversations[session_id] = []
        self._save_conversations()
        return {"success": True, "message": f"Conversation {session_id} cleared"}

    def list_sessions(self) -> List[str]:
        """List all conversation sessions"""
        return list(self.conversations.keys())

    def get_status(self) -> Dict:
        """Get AI status"""
        return {
            "ai_available": self.ai_available,
            "model": self.model if self.ai_available else None,
            "max_tokens": self.max_tokens,
            "sessions": len(self.conversations),
            "tools_available": len(self.tools),
            "api_key_configured": bool(self.api_key)
        }


# Singleton instance
_conversational_ai: Optional[ClaudeConversationalAI] = None


def get_conversational_ai() -> ClaudeConversationalAI:
    """Get or create the conversational AI singleton"""
    global _conversational_ai
    if _conversational_ai is None:
        _conversational_ai = ClaudeConversationalAI()
    return _conversational_ai


# Quick test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    ai = get_conversational_ai()
    print(f"\nAI Status: {ai.get_status()}")

    # Test conversation
    response = ai.chat("Hello! What can you help me with?")
    print(f"\nResponse: {response['response'][:500]}...")
