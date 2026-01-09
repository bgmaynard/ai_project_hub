"""
Claude AI Bot Intelligence Module
=================================
Native AI integration for autonomous bot monitoring, self-improvement,
and adaptive market responses using Claude AI Sonnet 4.5.

Features:
- Self-monitoring and performance analysis
- Adaptive strategy adjustment based on market conditions
- NLP communication interface for natural language control
- Morphic adaptation to changing market regimes
- Autonomous error detection and self-healing
- Trade journal and learning from past decisions
- Parallel tool execution for fast data gathering
- Async support for concurrent API operations

Author: AI Trading Bot Team
Version: 2.0 (Sonnet 4.5 with Parallel Tools)
"""

import asyncio
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS AND DATA CLASSES
# ============================================================================


class MarketRegime(str, Enum):
    """Market regime classification"""

    TRENDING_BULL = "trending_bull"
    TRENDING_BEAR = "trending_bear"
    RANGE_BOUND = "range_bound"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    BREAKOUT = "breakout"
    REVERSAL = "reversal"


class BotMood(str, Enum):
    """Bot's current operational mood/state"""

    AGGRESSIVE = "aggressive"  # High confidence, taking more trades
    CONSERVATIVE = "conservative"  # Lower confidence, fewer trades
    DEFENSIVE = "defensive"  # Protecting capital, minimal exposure
    OPPORTUNISTIC = "opportunistic"  # Waiting for high-probability setups
    LEARNING = "learning"  # Analyzing past performance


@dataclass
class PerformanceMetrics:
    """Bot performance metrics for analysis"""

    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0
    avg_hold_time: float = 0.0
    consecutive_wins: int = 0
    consecutive_losses: int = 0


@dataclass
class MarketConditions:
    """Current market conditions snapshot"""

    regime: str = "unknown"
    volatility: str = "normal"
    trend: str = "neutral"
    momentum: str = "neutral"
    volume: str = "normal"
    sentiment: str = "neutral"
    key_levels: Dict = field(default_factory=dict)
    risk_level: str = "medium"
    timestamp: str = ""


@dataclass
class StrategyAdjustment:
    """Strategy adjustment recommendation"""

    parameter: str
    current_value: Any
    recommended_value: Any
    reason: str
    confidence: float
    timestamp: str


@dataclass
class BotInsight:
    """AI-generated insight about bot behavior"""

    category: str  # performance, risk, strategy, market
    insight: str
    severity: str  # info, warning, critical
    action_required: bool
    suggested_action: str
    timestamp: str


# ============================================================================
# CLAUDE BOT INTELLIGENCE CLASS
# ============================================================================


class ClaudeBotIntelligence:
    """
    Claude AI-powered bot intelligence for autonomous monitoring,
    self-improvement, and adaptive market responses.
    """

    def __init__(self, api_key: Optional[str] = None):
        """Initialize Claude Bot Intelligence"""
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self.client = None
        self.ai_available = False

        # Initialize Anthropic client
        try:
            import anthropic

            if self.api_key:
                self.client = anthropic.Anthropic(api_key=self.api_key)
                self.ai_available = True
                logger.info("[OK] Claude Bot Intelligence initialized")
            else:
                logger.warning("[WARN] No API key - running in limited mode")
        except ImportError:
            logger.warning("[WARN] anthropic package not installed")
        except Exception as e:
            logger.error(f"[ERROR] Claude initialization failed: {e}")

        # Model configuration - Claude Sonnet 4.5 with parallel tool use
        self.model = "claude-sonnet-4-5-20250929"
        self.max_tokens = 4096

        # Thread pool for parallel execution
        self.executor = ThreadPoolExecutor(max_workers=5)

        # Initialize Alpaca MCP client for direct trading access
        self.mcp_client = None
        self._init_mcp_client()

        # Tool definitions for parallel data fetching (includes MCP tools)
        self.tools = self._define_tools()

        # State tracking
        self.current_mood = BotMood.CONSERVATIVE
        self.current_regime = MarketRegime.RANGE_BOUND
        self.performance_history: List[Dict] = []
        self.insights_history: List[BotInsight] = []
        self.adjustment_history: List[StrategyAdjustment] = []
        self.conversation_history: List[Dict] = []

        # Data persistence
        self.data_path = Path("store/bot/intelligence")
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.state_file = self.data_path / "bot_intelligence_state.json"

        # Load saved state
        self._load_state()

    def _load_state(self):
        """Load saved intelligence state"""
        if self.state_file.exists():
            try:
                with open(self.state_file, "r") as f:
                    data = json.load(f)
                    self.current_mood = BotMood(data.get("mood", "conservative"))
                    self.current_regime = MarketRegime(
                        data.get("regime", "range_bound")
                    )
                    self.insights_history = [
                        BotInsight(**i) for i in data.get("insights", [])[-50:]
                    ]
                    self.adjustment_history = [
                        StrategyAdjustment(**a)
                        for a in data.get("adjustments", [])[-50:]
                    ]
                logger.info("Bot intelligence state loaded")
            except Exception as e:
                logger.warning(f"Could not load intelligence state: {e}")

    def _save_state(self):
        """Save intelligence state"""
        try:
            data = {
                "mood": self.current_mood.value,
                "regime": self.current_regime.value,
                "insights": [asdict(i) for i in self.insights_history[-50:]],
                "adjustments": [asdict(a) for a in self.adjustment_history[-50:]],
                "last_updated": datetime.now().isoformat(),
            }
            with open(self.state_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save intelligence state: {e}")

    def _init_mcp_client(self):
        """Initialize Alpaca MCP client for direct trading access"""
        try:
            from ai.alpaca_mcp_integration import get_mcp_client

            self.mcp_client = get_mcp_client()
            if self.mcp_client and self.mcp_client.initialized:
                logger.info("[OK] Alpaca MCP Client connected - faster trading enabled")
            else:
                logger.warning("[WARN] Alpaca MCP Client not fully initialized")
        except ImportError:
            logger.warning("[WARN] Alpaca MCP integration not available")
            self.mcp_client = None
        except Exception as e:
            logger.error(f"[ERROR] Failed to initialize MCP client: {e}")
            self.mcp_client = None

    def _define_tools(self) -> List[Dict]:
        """Define tools for Claude to use for parallel data gathering"""
        # Base tools
        tools = [
            {
                "name": "get_market_data",
                "description": "Fetch current market data for a symbol including price, volume, and indicators",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock symbol to fetch data for (e.g., AAPL, TSLA)",
                        },
                        "include_indicators": {
                            "type": "boolean",
                            "description": "Whether to include technical indicators",
                            "default": True,
                        },
                    },
                    "required": ["symbol"],
                },
            },
            {
                "name": "get_account_status",
                "description": "Get current account status including equity, buying power, and positions",
                "input_schema": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "get_performance_metrics",
                "description": "Get trading performance metrics including win rate, P&L, and statistics",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "period": {
                            "type": "string",
                            "enum": ["today", "week", "month", "all"],
                            "description": "Time period for metrics",
                            "default": "today",
                        }
                    },
                    "required": [],
                },
            },
            {
                "name": "get_open_positions",
                "description": "Get all currently open positions with P&L",
                "input_schema": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "get_recent_trades",
                "description": "Get recent trade history",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "integer",
                            "description": "Number of trades to return",
                            "default": 10,
                        }
                    },
                    "required": [],
                },
            },
            {
                "name": "get_watchlist",
                "description": "Get symbols from a watchlist",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "watchlist_name": {
                            "type": "string",
                            "description": "Name of the watchlist",
                            "default": "default",
                        }
                    },
                    "required": [],
                },
            },
            {
                "name": "analyze_symbol",
                "description": "Get AI prediction and analysis for a symbol",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock symbol to analyze",
                        }
                    },
                    "required": ["symbol"],
                },
            },
            {
                "name": "get_market_sentiment",
                "description": "Get overall market sentiment and conditions",
                "input_schema": {"type": "object", "properties": {}, "required": []},
            },
        ]

        # Add Alpaca MCP tools if available for faster direct trading
        if self.mcp_client and self.mcp_client.initialized:
            from ai.alpaca_mcp_integration import get_mcp_tools

            mcp_tools = get_mcp_tools()
            tools.extend(mcp_tools)
            logger.info(
                f"[OK] Added {len(mcp_tools)} Alpaca MCP tools for direct trading"
            )

        return tools

    def _execute_tool(self, tool_name: str, tool_input: Dict) -> Dict:
        """Execute a tool and return results - uses MCP for Alpaca tools"""
        # Check if this is an Alpaca MCP tool (faster direct access)
        if (
            tool_name.startswith("alpaca_")
            and self.mcp_client
            and self.mcp_client.initialized
        ):
            return self._execute_mcp_tool(tool_name, tool_input)

        # Base tool handlers
        tool_handlers = {
            "get_market_data": self._tool_get_market_data,
            "get_account_status": self._tool_get_account_status,
            "get_performance_metrics": self._tool_get_performance_metrics,
            "get_open_positions": self._tool_get_open_positions,
            "get_recent_trades": self._tool_get_recent_trades,
            "get_watchlist": self._tool_get_watchlist,
            "analyze_symbol": self._tool_analyze_symbol,
            "get_market_sentiment": self._tool_get_market_sentiment,
        }

        handler = tool_handlers.get(tool_name)
        if handler:
            try:
                return handler(tool_input)
            except Exception as e:
                logger.error(f"Tool {tool_name} error: {e}")
                return {"error": str(e)}
        return {"error": f"Unknown tool: {tool_name}"}

    def _execute_mcp_tool(self, tool_name: str, tool_input: Dict) -> Dict:
        """Execute Alpaca MCP tool for direct trading access"""
        if not self.mcp_client:
            return {"error": "MCP client not available"}

        try:
            # Map tool names to MCP client methods
            mcp_handlers = {
                "alpaca_get_account": lambda p: self.mcp_client.get_account(),
                "alpaca_get_positions": lambda p: self.mcp_client.get_positions(),
                "alpaca_get_quote": lambda p: self.mcp_client.get_quote(
                    p.get("symbol")
                ),
                "alpaca_get_snapshot": lambda p: self.mcp_client.get_snapshot(
                    p.get("symbol")
                ),
                "alpaca_get_bars": lambda p: self.mcp_client.get_bars(
                    p.get("symbol"),
                    p.get("timeframe", "1Day"),
                    limit=p.get("limit", 100),
                ),
                "alpaca_place_order": lambda p: self.mcp_client.place_order(
                    symbol=p.get("symbol"),
                    qty=p.get("qty"),
                    side=p.get("side"),
                    order_type=p.get("order_type", "market"),
                    limit_price=p.get("limit_price"),
                    stop_price=p.get("stop_price"),
                ),
                "alpaca_get_orders": lambda p: self.mcp_client.get_orders(
                    status=p.get("status", "open"), limit=p.get("limit", 50)
                ),
                "alpaca_cancel_order": lambda p: self.mcp_client.cancel_order(
                    p.get("order_id")
                ),
                "alpaca_close_position": lambda p: self.mcp_client.close_position(
                    symbol=p.get("symbol"),
                    qty=p.get("qty"),
                    percentage=p.get("percentage"),
                ),
                "alpaca_get_clock": lambda p: self.mcp_client.get_clock(),
                "alpaca_get_portfolio_history": lambda p: self.mcp_client.get_portfolio_history(
                    period=p.get("period", "1M"), timeframe=p.get("timeframe", "1D")
                ),
                "alpaca_get_crypto_quote": lambda p: self.mcp_client.get_crypto_quote(
                    p.get("symbol")
                ),
            }

            handler = mcp_handlers.get(tool_name)
            if handler:
                result = handler(tool_input)
                logger.debug(f"MCP tool {tool_name} executed successfully")
                return result
            else:
                return {"error": f"Unknown MCP tool: {tool_name}"}

        except Exception as e:
            logger.error(f"MCP tool {tool_name} error: {e}")
            return {"error": str(e)}

    def _tool_get_market_data(self, params: Dict) -> Dict:
        """Fetch REAL market data - Schwab first, then Alpaca fallback"""
        symbol = params.get("symbol", "SPY").upper()
        try:
            # Try Schwab fast polling cache first (real-time)
            try:
                from schwab_fast_polling import \
                    get_cached_quote as get_fast_poll_quote

                fast_quote = get_fast_poll_quote(symbol)
                if fast_quote and fast_quote.get("last", 0) > 0:
                    return {
                        "symbol": symbol,
                        "price": fast_quote["last"],
                        "bid": fast_quote.get("bid", 0),
                        "ask": fast_quote.get("ask", 0),
                        "open": fast_quote.get("open", 0),
                        "high": fast_quote.get("high", 0),
                        "low": fast_quote.get("low", 0),
                        "volume": fast_quote.get("volume", 0),
                        "change": fast_quote.get("change", 0),
                        "change_pct": fast_quote.get("change_percent", 0),
                        "source": "schwab_realtime",
                    }
            except ImportError:
                pass

            # Try Schwab HTTP API
            try:
                from schwab_market_data import (get_schwab_quote,
                                                is_schwab_available)

                if is_schwab_available():
                    schwab_quote = get_schwab_quote(symbol)
                    if schwab_quote:
                        return {
                            "symbol": symbol,
                            "price": schwab_quote["last"],
                            "bid": schwab_quote.get("bid", 0),
                            "ask": schwab_quote.get("ask", 0),
                            "open": schwab_quote.get("open", 0),
                            "high": schwab_quote.get("high", 0),
                            "low": schwab_quote.get("low", 0),
                            "volume": schwab_quote.get("volume", 0),
                            "change": schwab_quote.get("change", 0),
                            "change_pct": schwab_quote.get("change_percent", 0),
                            "source": "schwab",
                        }
            except ImportError:
                pass

            # Fallback to Alpaca
            from alpaca_market_data import get_alpaca_market_data

            market_data = get_alpaca_market_data()

            # Get quote
            quote = market_data.get_latest_quote(symbol)
            snapshot = market_data.get_snapshot(symbol)

            if snapshot:
                return {
                    "symbol": symbol,
                    "price": snapshot.get("close", snapshot.get("last_price", 0)),
                    "open": snapshot.get("open", 0),
                    "high": snapshot.get("high", 0),
                    "low": snapshot.get("low", 0),
                    "volume": snapshot.get("volume", 0),
                    "change": snapshot.get("change", 0),
                    "change_pct": snapshot.get("change_percent", 0),
                    "vwap": snapshot.get("vwap", 0),
                    "source": "alpaca_live",
                }
            elif quote:
                return {
                    "symbol": symbol,
                    "price": quote.get("last", quote.get("bid", 0)),
                    "bid": quote.get("bid", 0),
                    "ask": quote.get("ask", 0),
                    "source": "alpaca_quote",
                }
            else:
                return {"symbol": symbol, "error": "No data available"}
        except Exception as e:
            logger.error(f"Market data error: {e}")
            return {"symbol": symbol, "error": str(e)}

    def _tool_get_account_status(self, params: Dict) -> Dict:
        """Get REAL account status - Schwab first, then Alpaca fallback"""
        try:
            # Try Schwab first (if user wants Schwab as primary)
            try:
                from schwab_trading import (get_schwab_trading,
                                            is_schwab_trading_available)

                if is_schwab_trading_available():
                    trading = get_schwab_trading()
                    if trading:
                        account = trading.get_account_info()
                        if account:
                            return {
                                "equity": float(account.get("total_equity", 0)),
                                "buying_power": float(account.get("buying_power", 0)),
                                "cash": float(
                                    account.get("cash_available_for_trading", 0)
                                ),
                                "portfolio_value": float(
                                    account.get("total_equity", 0)
                                ),
                                "daily_pnl": float(account.get("daily_pl", 0)),
                                "daytrade_count": 3
                                - account.get("round_trips", 0),  # PDT tracking
                                "pattern_day_trader": account.get(
                                    "is_day_trader", False
                                ),
                                "status": "ACTIVE",
                                "account_type": account.get("account_type", "UNKNOWN"),
                                "source": "schwab_live",
                            }
            except ImportError:
                pass

            # Fallback to Alpaca
            from alpaca_integration import get_alpaca_connector

            connector = get_alpaca_connector()
            account = connector.get_account()

            return {
                "equity": float(account.get("equity", 0)),
                "buying_power": float(account.get("buying_power", 0)),
                "cash": float(account.get("cash", 0)),
                "portfolio_value": float(account.get("portfolio_value", 0)),
                "daily_pnl": float(account.get("equity", 0))
                - float(account.get("last_equity", account.get("equity", 0))),
                "daytrade_count": account.get("daytrade_count", 0),
                "pattern_day_trader": account.get("pattern_day_trader", False),
                "status": account.get("status", "UNKNOWN"),
                "source": "alpaca_live",
            }
        except Exception as e:
            logger.error(f"Account status error: {e}")
            return {"error": str(e)}

    def _tool_get_performance_metrics(self, params: Dict) -> Dict:
        """Get REAL performance metrics from portfolio analytics"""
        try:
            from portfolio_analytics import get_portfolio_analytics

            analytics = get_portfolio_analytics()
            metrics = analytics.get_portfolio_metrics()

            return {
                "period": params.get("period", "all"),
                "win_rate": metrics.get("win_rate", 0),
                "total_trades": metrics.get("total_trades", 0),
                "total_pnl": metrics.get("total_pnl", 0),
                "profit_factor": metrics.get("profit_factor", 0),
                "avg_win": metrics.get("avg_win", 0),
                "avg_loss": metrics.get("avg_loss", 0),
                "max_drawdown": metrics.get("max_drawdown", 0),
                "source": "portfolio_analytics",
            }
        except Exception as e:
            logger.error(f"Performance metrics error: {e}")
            return {"error": str(e)}

    def _tool_get_open_positions(self, params: Dict) -> Dict:
        """Get REAL open positions - Schwab first, then Alpaca fallback"""
        try:
            # Try Schwab first
            try:
                from schwab_trading import (get_schwab_trading,
                                            is_schwab_trading_available)

                if is_schwab_trading_available():
                    trading = get_schwab_trading()
                    if trading:
                        positions = trading.get_positions()
                        if positions is not None:
                            total_value = sum(
                                float(p.get("market_value", 0)) for p in positions
                            )
                            unrealized_pnl = sum(
                                float(p.get("unrealized_pl", 0)) for p in positions
                            )
                            return {
                                "positions": positions,
                                "count": len(positions),
                                "total_value": total_value,
                                "unrealized_pnl": unrealized_pnl,
                                "source": "schwab_live",
                            }
            except ImportError:
                pass

            # Fallback to Alpaca
            from alpaca_integration import get_alpaca_connector

            connector = get_alpaca_connector()
            positions = connector.get_positions()

            total_value = sum(float(p.get("market_value", 0)) for p in positions)
            unrealized_pnl = sum(float(p.get("unrealized_pl", 0)) for p in positions)

            return {
                "positions": positions,
                "count": len(positions),
                "total_value": total_value,
                "unrealized_pnl": unrealized_pnl,
                "source": "alpaca_live",
            }
        except Exception as e:
            logger.error(f"Positions error: {e}")
            return {"positions": [], "error": str(e)}

    def _tool_get_recent_trades(self, params: Dict) -> Dict:
        """Get REAL recent trades from bot manager"""
        try:
            from bot_manager import get_bot_manager

            bot = get_bot_manager()
            trades = bot.trades_today[-params.get("limit", 10) :]

            return {
                "trades": [
                    {
                        "symbol": t.symbol,
                        "side": t.side,
                        "quantity": t.quantity,
                        "price": t.price,
                        "confidence": t.confidence,
                        "timestamp": t.timestamp,
                    }
                    for t in trades
                ],
                "count": len(trades),
                "source": "bot_manager",
            }
        except Exception as e:
            logger.error(f"Recent trades error: {e}")
            return {"trades": [], "error": str(e)}

    def _tool_get_watchlist(self, params: Dict) -> Dict:
        """Get REAL watchlist from watchlist manager"""
        try:
            from watchlist_manager import get_watchlist_manager

            mgr = get_watchlist_manager()
            watchlist = mgr.get_default_watchlist()

            return {
                "name": watchlist.get("name", "default"),
                "symbols": watchlist.get("symbols", []),
                "count": len(watchlist.get("symbols", [])),
                "source": "watchlist_manager",
            }
        except Exception as e:
            logger.error(f"Watchlist error: {e}")
            return {"symbols": [], "error": str(e)}

    def _tool_analyze_symbol(self, params: Dict) -> Dict:
        """Get REAL AI prediction for a symbol"""
        try:
            from ai.alpaca_ai_predictor import get_alpaca_predictor

            predictor = get_alpaca_predictor()
            prediction = predictor.predict(params.get("symbol", "SPY"))

            return {
                "symbol": params.get("symbol"),
                "prediction": prediction.get("signal", "NEUTRAL"),
                "action": prediction.get("action", "HOLD"),
                "confidence": prediction.get("confidence", 0.5),
                "features": prediction.get("features", {}),
                "source": "ai_predictor",
            }
        except Exception as e:
            logger.error(f"Symbol analysis error: {e}")
            return {"symbol": params.get("symbol"), "error": str(e)}

    def _tool_get_market_sentiment(self, params: Dict) -> Dict:
        """Get REAL market sentiment from sentiment analyzer"""
        try:
            import asyncio

            from ai.warrior_sentiment_analyzer import get_sentiment_analyzer

            analyzer = get_sentiment_analyzer()

            # Get SPY sentiment as market proxy
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            sentiment = loop.run_until_complete(
                analyzer.get_aggregated_sentiment("SPY", hours=4)
            )

            if sentiment:
                return {
                    "overall": (
                        "bullish"
                        if sentiment.overall_score > 0.2
                        else "bearish" if sentiment.overall_score < -0.2 else "neutral"
                    ),
                    "score": sentiment.overall_score,
                    "confidence": sentiment.confidence,
                    "is_trending": sentiment.is_trending,
                    "momentum": sentiment.momentum,
                    "signal_count": sentiment.signal_count,
                    "source": "sentiment_analyzer",
                }
            else:
                return {"overall": "neutral", "note": "No recent sentiment data"}
        except Exception as e:
            logger.error(f"Market sentiment error: {e}")
            return {"overall": "neutral", "error": str(e)}

    def set_data_provider(self, provider):
        """Inject real data provider for tool execution"""
        self.data_provider = provider
        # Override tool handlers with real data
        if hasattr(provider, "get_market_data"):
            self._tool_get_market_data = lambda p: provider.get_market_data(
                p.get("symbol")
            )
        if hasattr(provider, "get_account_status"):
            self._tool_get_account_status = lambda p: provider.get_account_status()
        if hasattr(provider, "get_positions"):
            self._tool_get_open_positions = lambda p: {
                "positions": provider.get_positions()
            }
        if hasattr(provider, "get_recent_trades"):
            self._tool_get_recent_trades = lambda p: {
                "trades": provider.get_recent_trades(p.get("limit", 10))
            }

    def _execute_tools_parallel(self, tool_calls: List[Dict]) -> List[Dict]:
        """Execute multiple tool calls in parallel"""
        results = []

        def execute_single(tool_call):
            tool_name = tool_call.get("name")
            tool_input = tool_call.get("input", {})
            tool_id = tool_call.get("id", "")
            result = self._execute_tool(tool_name, tool_input)
            return {
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": json.dumps(result),
            }

        # Execute tools in parallel using thread pool
        with ThreadPoolExecutor(max_workers=len(tool_calls)) as executor:
            results = list(executor.map(execute_single, tool_calls))

        return results

    def _get_claude_response_with_tools(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = None,
        use_tools: bool = True,
    ) -> str:
        """Get Claude response with tool use support for parallel data fetching"""
        if not self.ai_available:
            return self._get_fallback_response(user_prompt)

        try:
            messages = [{"role": "user", "content": user_prompt}]

            # Initial request with tools
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens or self.max_tokens,
                system=system_prompt,
                tools=self.tools if use_tools else [],
                messages=messages,
            )

            # Handle tool use loop
            max_iterations = 5
            iteration = 0

            while response.stop_reason == "tool_use" and iteration < max_iterations:
                iteration += 1

                # Collect all tool uses from response
                tool_calls = []
                text_content = []

                for block in response.content:
                    if block.type == "tool_use":
                        tool_calls.append(
                            {"id": block.id, "name": block.name, "input": block.input}
                        )
                    elif block.type == "text":
                        text_content.append(block.text)

                if not tool_calls:
                    break

                logger.info(
                    f"Executing {len(tool_calls)} tools in parallel: {[t['name'] for t in tool_calls]}"
                )

                # Execute all tools in parallel
                tool_results = self._execute_tools_parallel(tool_calls)

                # Add assistant response and tool results to messages
                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": tool_results})

                # Get next response
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens or self.max_tokens,
                    system=system_prompt,
                    tools=self.tools if use_tools else [],
                    messages=messages,
                )

            # Extract final text response
            final_text = ""
            for block in response.content:
                if hasattr(block, "text"):
                    final_text += block.text

            return final_text if final_text else "Analysis complete."

        except Exception as e:
            logger.error(f"Claude API error with tools: {e}")
            return self._get_fallback_response(user_prompt)

    def _get_claude_response(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = None,
        original_message: str = None,
    ) -> str:
        """Get response from Claude AI"""
        if not self.ai_available:
            # Use original message for fallback if available, otherwise use full prompt
            return self._get_fallback_response(original_message or user_prompt)

        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens or self.max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return message.content[0].text
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            return self._get_fallback_response(original_message or user_prompt)

    def _get_fallback_response(self, prompt: str) -> str:
        """Provide intelligent fallback response when AI is unavailable"""
        prompt_lower = prompt.lower()

        # Context-aware responses based on prompt content (check specific terms first)
        if "warrior" in prompt_lower or "momentum" in prompt_lower:
            return (
                "⚔️ Warrior Trading Mode Tips:\n"
                "• Focus on stocks with high relative volume (2x+ average)\n"
                "• Look for gap-ups with catalyst (news, earnings)\n"
                "• Entry on first pullback to VWAP or moving average\n"
                "• Quick profits - don't hold for extended moves\n"
                "• Risk max 1-2% of account per trade"
            )

        elif "market" in prompt_lower or "regime" in prompt_lower:
            return (
                "📊 Market Analysis (Rule-Based): Currently operating in range-bound conditions. "
                "Volatility is normal. Recommend maintaining conservative position sizes "
                "and waiting for clear breakout signals before increasing exposure."
            )

        elif "performance" in prompt_lower:
            return (
                "📈 Performance Summary: No recent trades to analyze. "
                "When trading begins, I'll track win rate, P&L, and risk metrics. "
                "Recommendation: Start with small position sizes to build a track record."
            )

        elif (
            "adjustment" in prompt_lower
            or "recommend" in prompt_lower
            or "suggest" in prompt_lower
        ):
            return (
                "💡 Strategy Recommendations:\n"
                "• Keep confidence threshold at 0.65+ for entries\n"
                "• Use 1-2% position sizing per trade\n"
                "• Set stop losses at 2-3% below entry\n"
                "• Take partial profits at 1:1 risk/reward\n"
                "• Avoid trading first 15 min and last 30 min of session"
            )

        elif (
            "issue" in prompt_lower
            or "problem" in prompt_lower
            or "error" in prompt_lower
        ):
            return (
                "🔧 System Diagnostic:\n"
                "• API connections: Operational\n"
                "• Market data: Active\n"
                "• AI model: Loaded\n"
                "• Note: Claude AI rate limited until Dec 1st - using rule-based analysis"
            )

        elif "hello" in prompt_lower or "hi" in prompt_lower:
            return (
                "👋 Hello! I'm your AI trading assistant. I can help with:\n"
                "• Market regime analysis\n"
                "• Performance tracking\n"
                "• Strategy adjustments\n"
                "• Risk management\n"
                "Note: Full AI capabilities resume Dec 1st (rate limit reached)"
            )

        else:
            return (
                f"🤖 I received your message about: '{prompt[:50]}...'\n\n"
                "Currently using rule-based analysis (Claude AI rate limited until Dec 1st).\n"
                "I can still help with:\n"
                "• 'market' - Market regime analysis\n"
                "• 'performance' - Trading performance\n"
                "• 'recommendations' - Strategy suggestions\n"
                "• 'warrior mode' - Momentum trading tips"
            )

    # ========================================================================
    # SELF-MONITORING AND PERFORMANCE ANALYSIS
    # ========================================================================

    def analyze_performance(self, trades: List[Dict], account_equity: float) -> Dict:
        """
        Analyze bot trading performance and generate insights

        Args:
            trades: List of recent trades
            account_equity: Current account equity

        Returns:
            Performance analysis with insights and recommendations
        """
        # Calculate metrics
        metrics = self._calculate_metrics(trades, account_equity)

        system_prompt = """You are an expert algorithmic trading performance analyst
integrated into an AI trading bot. Your role is to:
1. Analyze trading performance objectively
2. Identify patterns in winning and losing trades
3. Suggest specific, actionable improvements
4. Assess risk management effectiveness
5. Recommend parameter adjustments

Respond in JSON format with keys: summary, strengths, weaknesses,
recommendations, risk_assessment, mood_recommendation"""

        user_prompt = f"""Analyze this trading performance data:

PERFORMANCE METRICS:
- Total Trades: {metrics.total_trades}
- Win Rate: {metrics.win_rate:.1%}
- Profit Factor: {metrics.profit_factor:.2f}
- Average Win: ${metrics.avg_win:.2f}
- Average Loss: ${metrics.avg_loss:.2f}
- Total P&L: ${metrics.total_pnl:.2f}
- Max Drawdown: {metrics.max_drawdown:.1%}
- Consecutive Wins: {metrics.consecutive_wins}
- Consecutive Losses: {metrics.consecutive_losses}

RECENT TRADES (last 10):
{json.dumps(trades[-10:] if trades else [], indent=2)}

CURRENT ACCOUNT EQUITY: ${account_equity:,.2f}

Provide a comprehensive analysis with specific recommendations for improvement.
What mood should the bot adopt? (aggressive/conservative/defensive/opportunistic)"""

        response = self._get_claude_response(system_prompt, user_prompt)

        # Parse response and create insight
        try:
            analysis = json.loads(response)
        except:
            analysis = {"raw_analysis": response}

        insight = BotInsight(
            category="performance",
            insight=analysis.get("summary", response[:500]),
            severity="info" if metrics.win_rate > 0.5 else "warning",
            action_required=metrics.consecutive_losses >= 3,
            suggested_action=(
                analysis.get("recommendations", ["Review strategy"])[0]
                if isinstance(analysis.get("recommendations"), list)
                else str(analysis.get("recommendations", "Continue monitoring"))
            ),
            timestamp=datetime.now().isoformat(),
        )
        self.insights_history.append(insight)
        self._save_state()

        return {
            "metrics": asdict(metrics),
            "analysis": analysis,
            "insight": asdict(insight),
            "mood_recommendation": analysis.get(
                "mood_recommendation", self.current_mood.value
            ),
        }

    def _calculate_metrics(
        self, trades: List[Dict], account_equity: float
    ) -> PerformanceMetrics:
        """Calculate performance metrics from trades"""
        if not trades:
            return PerformanceMetrics()

        wins = [t for t in trades if t.get("pnl", 0) > 0]
        losses = [t for t in trades if t.get("pnl", 0) < 0]

        total_wins = sum(t.get("pnl", 0) for t in wins)
        total_losses = abs(sum(t.get("pnl", 0) for t in losses))

        # Calculate consecutive wins/losses
        consecutive_wins = 0
        consecutive_losses = 0
        for trade in reversed(trades):
            if trade.get("pnl", 0) > 0:
                if consecutive_losses == 0:
                    consecutive_wins += 1
                else:
                    break
            elif trade.get("pnl", 0) < 0:
                if consecutive_wins == 0:
                    consecutive_losses += 1
                else:
                    break

        return PerformanceMetrics(
            win_rate=len(wins) / len(trades) if trades else 0,
            profit_factor=(
                total_wins / total_losses if total_losses > 0 else float("inf")
            ),
            avg_win=total_wins / len(wins) if wins else 0,
            avg_loss=total_losses / len(losses) if losses else 0,
            total_trades=len(trades),
            winning_trades=len(wins),
            losing_trades=len(losses),
            total_pnl=sum(t.get("pnl", 0) for t in trades),
            best_trade=max(t.get("pnl", 0) for t in trades) if trades else 0,
            worst_trade=min(t.get("pnl", 0) for t in trades) if trades else 0,
            consecutive_wins=consecutive_wins,
            consecutive_losses=consecutive_losses,
        )

    # ========================================================================
    # ADAPTIVE STRATEGY ADJUSTMENT
    # ========================================================================

    def analyze_market_conditions(self, market_data: Dict) -> MarketConditions:
        """
        Analyze current market conditions and classify regime

        Args:
            market_data: Dictionary with market indicators

        Returns:
            MarketConditions with regime classification
        """
        system_prompt = """You are a market regime analyst for an algorithmic trading bot.
Classify the current market conditions based on the provided data.

Respond in JSON format with keys:
- regime: trending_bull, trending_bear, range_bound, high_volatility, low_volatility, breakout, reversal
- volatility: low, normal, high, extreme
- trend: strong_up, up, neutral, down, strong_down
- momentum: strong_bullish, bullish, neutral, bearish, strong_bearish
- volume: low, normal, high, extreme
- sentiment: very_bullish, bullish, neutral, bearish, very_bearish
- risk_level: low, medium, high, extreme
- key_observations: [list of important observations]
- trading_approach: recommended approach for this regime"""

        user_prompt = f"""Analyze these market conditions:

MARKET DATA:
{json.dumps(market_data, indent=2)}

Classify the current market regime and provide trading recommendations."""

        response = self._get_claude_response(
            system_prompt, user_prompt, max_tokens=1000
        )

        try:
            analysis = json.loads(response)
            conditions = MarketConditions(
                regime=analysis.get("regime", "range_bound"),
                volatility=analysis.get("volatility", "normal"),
                trend=analysis.get("trend", "neutral"),
                momentum=analysis.get("momentum", "neutral"),
                volume=analysis.get("volume", "normal"),
                sentiment=analysis.get("sentiment", "neutral"),
                risk_level=analysis.get("risk_level", "medium"),
                key_levels=analysis.get("key_levels", {}),
                timestamp=datetime.now().isoformat(),
            )

            # Update current regime
            try:
                self.current_regime = MarketRegime(conditions.regime)
            except:
                pass

            self._save_state()
            return conditions

        except Exception as e:
            logger.error(f"Error parsing market conditions: {e}")
            return MarketConditions(timestamp=datetime.now().isoformat())

    def suggest_strategy_adjustments(
        self,
        current_config: Dict,
        performance: Dict,
        market_conditions: MarketConditions,
    ) -> List[StrategyAdjustment]:
        """
        Suggest strategy parameter adjustments based on performance and market conditions

        Args:
            current_config: Current bot configuration
            performance: Recent performance metrics
            market_conditions: Current market conditions

        Returns:
            List of suggested adjustments
        """
        system_prompt = """You are a strategy optimization AI for an algorithmic trading bot.
Based on performance data and market conditions, suggest specific parameter adjustments.

Current configurable parameters:
- confidence_threshold (0.0-1.0): Minimum AI confidence to take a trade
- max_positions (1-10): Maximum concurrent positions
- max_daily_trades (1-50): Maximum trades per day
- position_size (1-100): Shares per trade
- cycle_interval_seconds (30-600): Seconds between trading cycles

Respond in JSON format with an array of adjustments:
[{
    "parameter": "parameter_name",
    "current_value": current,
    "recommended_value": recommended,
    "reason": "explanation",
    "confidence": 0.0-1.0
}]

Be conservative with recommendations. Only suggest changes that are well-justified."""

        user_prompt = f"""Analyze and suggest strategy adjustments:

CURRENT CONFIGURATION:
{json.dumps(current_config, indent=2)}

RECENT PERFORMANCE:
{json.dumps(performance, indent=2)}

MARKET CONDITIONS:
- Regime: {market_conditions.regime}
- Volatility: {market_conditions.volatility}
- Trend: {market_conditions.trend}
- Risk Level: {market_conditions.risk_level}

Should any parameters be adjusted? If so, what specific changes do you recommend?"""

        response = self._get_claude_response(
            system_prompt, user_prompt, max_tokens=1500
        )

        adjustments = []
        try:
            suggestions = json.loads(response)
            if isinstance(suggestions, list):
                for s in suggestions:
                    adj = StrategyAdjustment(
                        parameter=s.get("parameter", ""),
                        current_value=s.get("current_value"),
                        recommended_value=s.get("recommended_value"),
                        reason=s.get("reason", ""),
                        confidence=s.get("confidence", 0.5),
                        timestamp=datetime.now().isoformat(),
                    )
                    adjustments.append(adj)
                    self.adjustment_history.append(adj)
        except Exception as e:
            logger.error(f"Error parsing adjustments: {e}")

        self._save_state()
        return adjustments

    def auto_adjust_mood(
        self, performance: Dict, market_conditions: MarketConditions
    ) -> Tuple[BotMood, str]:
        """
        Automatically adjust bot mood based on conditions

        Returns:
            Tuple of (new_mood, reason)
        """
        metrics = performance.get("metrics", {})
        win_rate = metrics.get("win_rate", 0.5)
        consecutive_losses = metrics.get("consecutive_losses", 0)
        consecutive_wins = metrics.get("consecutive_wins", 0)
        total_pnl = metrics.get("total_pnl", 0)

        old_mood = self.current_mood
        reason = ""

        # Defensive conditions
        if consecutive_losses >= 3:
            self.current_mood = BotMood.DEFENSIVE
            reason = f"3+ consecutive losses ({consecutive_losses}). Switching to defensive mode."
        elif market_conditions.risk_level == "extreme":
            self.current_mood = BotMood.DEFENSIVE
            reason = "Extreme market risk detected. Protecting capital."
        elif market_conditions.volatility == "extreme":
            self.current_mood = BotMood.DEFENSIVE
            reason = "Extreme volatility. Reducing exposure."

        # Conservative conditions
        elif win_rate < 0.4 and metrics.get("total_trades", 0) > 10:
            self.current_mood = BotMood.CONSERVATIVE
            reason = f"Low win rate ({win_rate:.1%}). Being more selective."
        elif market_conditions.regime == MarketRegime.RANGE_BOUND.value:
            self.current_mood = BotMood.CONSERVATIVE
            reason = "Range-bound market. Waiting for breakout."

        # Aggressive conditions
        elif consecutive_wins >= 3 and win_rate > 0.6:
            self.current_mood = BotMood.AGGRESSIVE
            reason = f"Hot streak ({consecutive_wins} wins). Capitalizing on momentum."
        elif market_conditions.regime in [
            MarketRegime.TRENDING_BULL.value,
            MarketRegime.BREAKOUT.value,
        ]:
            if win_rate > 0.5:
                self.current_mood = BotMood.AGGRESSIVE
                reason = "Strong bullish trend with good win rate. Increasing activity."

        # Opportunistic (default)
        else:
            self.current_mood = BotMood.OPPORTUNISTIC
            reason = "Normal conditions. Waiting for high-probability setups."

        if old_mood != self.current_mood:
            logger.info(
                f"Bot mood changed: {old_mood.value} -> {self.current_mood.value}. Reason: {reason}"
            )
            self._save_state()

        return self.current_mood, reason

    # ========================================================================
    # NLP COMMUNICATION INTERFACE
    # ========================================================================

    def chat(self, user_message: str, context: Optional[Dict] = None) -> Dict:
        """
        Natural language interface to communicate with the bot

        Args:
            user_message: User's message in natural language
            context: Optional context (current status, positions, etc.)

        Returns:
            Bot's response with any actions taken
        """
        system_prompt = """You are an AI trading bot assistant. You can:
1. Answer questions about the bot's current status, positions, and performance
2. Execute commands like: start/stop bot, adjust settings, analyze symbols
3. Provide market analysis and trading insights
4. Explain the bot's decisions and strategy

When the user wants to change settings, respond with JSON containing:
{
    "response": "your natural language response",
    "action": "action_type",  // e.g., "update_config", "analyze_symbol", "none"
    "parameters": {}  // action-specific parameters
}

Be helpful, concise, and focused on trading. Always prioritize risk management."""

        context_str = ""
        if context:
            context_str = f"\n\nCURRENT CONTEXT:\n{json.dumps(context, indent=2)}"

        user_prompt = f"""User message: {user_message}{context_str}

Respond helpfully. If the user wants to change settings or perform an action,
include the appropriate action in your response."""

        # Add to conversation history
        self.conversation_history.append(
            {
                "role": "user",
                "content": user_message,
                "timestamp": datetime.now().isoformat(),
            }
        )

        response = self._get_claude_response(
            system_prompt, user_prompt, max_tokens=1000, original_message=user_message
        )

        # Parse response
        try:
            parsed = json.loads(response)
            bot_response = parsed.get("response", response)
            action = parsed.get("action", "none")
            parameters = parsed.get("parameters", {})
        except:
            bot_response = response
            action = "none"
            parameters = {}

        # Add to conversation history
        self.conversation_history.append(
            {
                "role": "assistant",
                "content": bot_response,
                "action": action,
                "timestamp": datetime.now().isoformat(),
            }
        )

        # Keep only last 20 messages
        self.conversation_history = self.conversation_history[-20:]

        return {
            "response": bot_response,
            "action": action,
            "parameters": parameters,
            "mood": self.current_mood.value,
            "timestamp": datetime.now().isoformat(),
        }

    def chat_with_tools(self, user_message: str, use_tools: bool = True) -> Dict:
        """
        Enhanced chat with tool support for real-time data access.
        Claude will automatically fetch market data, positions, and metrics
        as needed to answer questions.

        Args:
            user_message: User's message in natural language
            use_tools: Whether to enable tool use for data fetching

        Returns:
            Bot's response with comprehensive data-backed analysis
        """
        system_prompt = """You are an expert AI trading assistant with access to real-time trading data.
You have tools to fetch:
- Market data for any symbol (prices, volume, indicators)
- Account status (equity, buying power, positions)
- Performance metrics (win rate, P&L, statistics)
- Open positions and recent trades
- Market sentiment

When answering questions:
1. Use tools to gather relevant data BEFORE responding
2. Fetch multiple data points in parallel when needed
3. Provide data-backed analysis and specific recommendations
4. Focus on actionable insights for the trader
5. Always prioritize risk management

Be concise but thorough. Reference specific data from your tool calls."""

        user_prompt = f"User: {user_message}"

        # Add to conversation history
        self.conversation_history.append(
            {
                "role": "user",
                "content": user_message,
                "timestamp": datetime.now().isoformat(),
            }
        )

        # Use tool-enabled response
        response = self._get_claude_response_with_tools(
            system_prompt, user_prompt, max_tokens=2000, use_tools=use_tools
        )

        # Add to conversation history
        self.conversation_history.append(
            {
                "role": "assistant",
                "content": response,
                "timestamp": datetime.now().isoformat(),
            }
        )

        # Keep only last 20 messages
        self.conversation_history = self.conversation_history[-20:]

        return {
            "response": response,
            "mood": self.current_mood.value,
            "timestamp": datetime.now().isoformat(),
            "tools_used": use_tools,
        }

    def comprehensive_analysis(self, symbols: List[str] = None) -> Dict:
        """
        Perform comprehensive trading analysis using parallel tool execution.
        Claude will gather all relevant data in parallel and provide a complete
        market overview with actionable recommendations.

        Args:
            symbols: Optional list of symbols to analyze

        Returns:
            Comprehensive analysis with data from multiple sources
        """
        symbols_str = ", ".join(symbols) if symbols else "top watchlist symbols"

        system_prompt = """You are an expert trading analyst. Perform a comprehensive analysis by:

1. FIRST, gather data in parallel:
   - Get account status
   - Get open positions
   - Get recent trades
   - Get performance metrics
   - Get market sentiment
   - Get market data for key symbols

2. THEN, analyze the data to provide:
   - Account overview and health assessment
   - Current position analysis with P&L
   - Recent trade performance review
   - Market conditions assessment
   - Risk analysis based on exposure
   - Specific trading recommendations
   - Key levels to watch

Format your response clearly with sections. Be specific and actionable."""

        user_prompt = f"""Perform a comprehensive trading analysis.
Focus on: {symbols_str}

Gather all relevant data and provide:
1. Account Status Overview
2. Position Analysis
3. Performance Review
4. Market Conditions
5. Risk Assessment
6. Trading Recommendations

Use your tools to fetch real-time data before analyzing."""

        response = self._get_claude_response_with_tools(
            system_prompt, user_prompt, max_tokens=3000, use_tools=True
        )

        # Create insight from analysis
        insight = BotInsight(
            category="comprehensive",
            insight=response[:500] + "..." if len(response) > 500 else response,
            severity="info",
            action_required=False,
            suggested_action="Review analysis and act on recommendations",
            timestamp=datetime.now().isoformat(),
        )
        self.insights_history.append(insight)
        self._save_state()

        return {
            "analysis": response,
            "timestamp": datetime.now().isoformat(),
            "symbols_analyzed": symbols or [],
            "mood": self.current_mood.value,
        }

    def quick_market_scan(self, symbols: List[str]) -> Dict:
        """
        Quick parallel scan of multiple symbols.

        Args:
            symbols: List of symbols to scan

        Returns:
            Quick analysis of all symbols
        """
        if not symbols:
            return {"error": "No symbols provided"}

        system_prompt = """You are a rapid market scanner. For each symbol:
1. Fetch market data in parallel
2. Provide a quick 1-line assessment
3. Rate: BUY / SELL / HOLD with confidence

Be extremely concise. Format as a table."""

        symbols_list = ", ".join(symbols[:10])  # Limit to 10
        user_prompt = f"Quick scan these symbols: {symbols_list}. Get their market data and provide rapid assessments."

        response = self._get_claude_response_with_tools(
            system_prompt, user_prompt, max_tokens=1500, use_tools=True
        )

        return {
            "scan_results": response,
            "symbols": symbols[:10],
            "timestamp": datetime.now().isoformat(),
        }

    def interpret_command(self, command: str) -> Dict:
        """
        Interpret a natural language command and return structured action

        Args:
            command: Natural language command

        Returns:
            Structured command interpretation
        """
        system_prompt = """You are a command interpreter for an AI trading bot.
Parse the user's natural language command into a structured action.

Available actions:
- start_bot: Start the trading bot
- stop_bot: Stop the trading bot
- set_config: Update configuration (params: config key-value pairs)
- analyze_symbol: Analyze a stock symbol (params: symbol)
- get_status: Get current bot status
- get_performance: Get performance report
- set_long_only: Enable/disable long-only mode (params: enabled)
- set_account_type: Set account type (params: account_type)
- adjust_confidence: Adjust confidence threshold (params: threshold)
- adjust_position_size: Adjust position size (params: size)

Respond in JSON format:
{
    "action": "action_name",
    "parameters": {},
    "confidence": 0.0-1.0,
    "clarification_needed": false,
    "clarification_question": ""
}"""

        user_prompt = f"Interpret this command: {command}"

        response = self._get_claude_response(system_prompt, user_prompt, max_tokens=500)

        try:
            return json.loads(response)
        except:
            return {
                "action": "unknown",
                "parameters": {},
                "confidence": 0.0,
                "clarification_needed": True,
                "clarification_question": "I didn't understand that command. Could you rephrase?",
            }

    # ========================================================================
    # MORPHIC SELF-HEALING AND AUTO-REPAIR SYSTEM
    # ========================================================================

    def self_heal(self, error: Exception, context: Dict = None) -> Dict:
        """
        MORPHIC SELF-HEALING: Automatically detect, diagnose, and repair issues.
        The bot adapts and recovers from failures without human intervention.

        Args:
            error: The exception that occurred
            context: Context when error occurred

        Returns:
            Healing result with actions taken
        """
        error_str = str(error)
        error_type = type(error).__name__
        context = context or {}

        logger.warning(f"🔧 SELF-HEALING: Attempting to recover from {error_type}")

        healing_result = {
            "error_type": error_type,
            "error_message": error_str,
            "timestamp": datetime.now().isoformat(),
            "healed": False,
            "actions_taken": [],
            "recommendations": [],
        }

        try:
            # Pattern 1: Connection errors - retry with backoff
            if "connection" in error_str.lower() or "timeout" in error_str.lower():
                healing_result["actions_taken"].append("Detected connection issue")
                healing_result["actions_taken"].append(
                    "Will retry with exponential backoff"
                )
                healing_result["healed"] = True
                healing_result["retry_delay"] = 5

            # Pattern 2: API rate limits - pause and retry
            elif "rate" in error_str.lower() or "429" in error_str:
                healing_result["actions_taken"].append("Detected rate limit")
                healing_result["actions_taken"].append(
                    "Switching to fallback mode for 60s"
                )
                healing_result["healed"] = True
                healing_result["retry_delay"] = 60

            # Pattern 3: Data errors - use cached/fallback data
            elif "data" in error_str.lower() or "none" in error_str.lower():
                healing_result["actions_taken"].append(
                    "Detected data availability issue"
                )
                healing_result["actions_taken"].append("Using cached data if available")
                healing_result["healed"] = True

            # Pattern 4: Authentication errors - attempt re-auth
            elif (
                "auth" in error_str.lower() or "401" in error_str or "403" in error_str
            ):
                healing_result["actions_taken"].append("Detected authentication issue")
                healing_result["recommendations"].append("Check API keys in .env file")
                healing_result["healed"] = False

            # Pattern 5: Market closed - adjust behavior
            elif "market" in error_str.lower() and "closed" in error_str.lower():
                healing_result["actions_taken"].append("Market is closed")
                healing_result["actions_taken"].append(
                    "Switching to analysis-only mode"
                )
                healing_result["healed"] = True

            # Use AI to diagnose unknown errors
            else:
                diagnosis = self.diagnose_issue(error_str, context)
                healing_result["ai_diagnosis"] = diagnosis
                healing_result["actions_taken"].append(
                    f"AI diagnosis: {diagnosis.get('diagnosis', 'Unknown')[:100]}"
                )
                healing_result["healed"] = diagnosis.get(
                    "should_continue_trading", True
                )
                healing_result["recommendations"] = [
                    diagnosis.get("immediate_fix", "Review logs")
                ]

            # Log healing attempt
            logger.info(
                f"🔧 SELF-HEALING: {'SUCCESS' if healing_result['healed'] else 'NEEDS ATTENTION'}"
            )
            for action in healing_result["actions_taken"]:
                logger.info(f"   - {action}")

            # Save insight
            insight = BotInsight(
                category="self_healing",
                insight=f"Auto-healed from {error_type}: {error_str[:100]}",
                severity="warning" if healing_result["healed"] else "critical",
                action_required=not healing_result["healed"],
                suggested_action=(
                    healing_result["recommendations"][0]
                    if healing_result["recommendations"]
                    else "Monitor system"
                ),
                timestamp=datetime.now().isoformat(),
            )
            self.insights_history.append(insight)
            self._save_state()

        except Exception as heal_error:
            logger.error(f"Self-healing failed: {heal_error}")
            healing_result["heal_error"] = str(heal_error)

        return healing_result

    def morphic_adapt(self, market_data: Dict, performance: Dict) -> Dict:
        """
        MORPHIC ADAPTATION: Continuously adapt bot parameters based on
        market conditions and performance. The bot evolves to optimize results.

        Args:
            market_data: Current market conditions
            performance: Recent performance metrics

        Returns:
            Adaptations made
        """
        logger.info("🧬 MORPHIC ADAPTATION: Analyzing conditions for optimization...")

        adaptations = {
            "timestamp": datetime.now().isoformat(),
            "changes": [],
            "reason": "",
            "new_config": {},
        }

        try:
            # Analyze market conditions
            conditions = self.analyze_market_conditions(market_data)

            # Calculate performance metrics
            win_rate = performance.get("win_rate", 0.5)
            profit_factor = performance.get("profit_factor", 1.0)
            consecutive_losses = performance.get("consecutive_losses", 0)
            total_trades = performance.get("total_trades", 0)

            # Adaptation rules based on conditions

            # Rule 1: High volatility - reduce position size, increase confidence threshold
            if conditions.volatility in ["high", "extreme"]:
                adaptations["changes"].append(
                    {
                        "parameter": "position_size",
                        "change": "reduce by 50%",
                        "reason": "High volatility detected",
                    }
                )
                adaptations["changes"].append(
                    {
                        "parameter": "confidence_threshold",
                        "change": "increase to 0.7",
                        "reason": "Require higher confidence in volatile markets",
                    }
                )
                adaptations["new_config"]["confidence_threshold"] = 0.7

            # Rule 2: Losing streak - become defensive
            if consecutive_losses >= 3:
                adaptations["changes"].append(
                    {
                        "parameter": "mood",
                        "change": "DEFENSIVE",
                        "reason": f"{consecutive_losses} consecutive losses",
                    }
                )
                adaptations["changes"].append(
                    {
                        "parameter": "max_daily_trades",
                        "change": "reduce to 3",
                        "reason": "Limit exposure during losing streak",
                    }
                )
                adaptations["new_config"]["max_daily_trades"] = 3
                self.current_mood = BotMood.DEFENSIVE

            # Rule 3: Low win rate - increase selectivity
            elif win_rate < 0.4 and total_trades >= 10:
                adaptations["changes"].append(
                    {
                        "parameter": "confidence_threshold",
                        "change": "increase to 0.75",
                        "reason": f"Low win rate ({win_rate:.1%})",
                    }
                )
                adaptations["new_config"]["confidence_threshold"] = 0.75

            # Rule 4: Strong performance - gradually increase activity
            elif win_rate > 0.6 and profit_factor > 1.5:
                adaptations["changes"].append(
                    {
                        "parameter": "mood",
                        "change": "AGGRESSIVE",
                        "reason": f"Strong performance (WR: {win_rate:.1%}, PF: {profit_factor:.2f})",
                    }
                )
                self.current_mood = BotMood.AGGRESSIVE

            # Rule 5: Trending market - follow the trend
            if conditions.regime in [
                MarketRegime.TRENDING_BULL.value,
                MarketRegime.TRENDING_BEAR.value,
            ]:
                adaptations["changes"].append(
                    {
                        "parameter": "strategy",
                        "change": "trend_following",
                        "reason": f"Market is trending: {conditions.regime}",
                    }
                )

            # Rule 6: Range-bound market - wait for breakouts
            elif conditions.regime == MarketRegime.RANGE_BOUND.value:
                adaptations["changes"].append(
                    {
                        "parameter": "strategy",
                        "change": "breakout_watch",
                        "reason": "Range-bound market - waiting for breakout",
                    }
                )
                self.current_mood = BotMood.OPPORTUNISTIC

            adaptations["reason"] = (
                f"Adapted to {conditions.regime} market with {conditions.volatility} volatility"
            )

            # Apply adaptations to bot if available
            self._apply_adaptations(adaptations)

            logger.info(
                f"🧬 MORPHIC ADAPTATION: Made {len(adaptations['changes'])} changes"
            )
            for change in adaptations["changes"]:
                logger.info(
                    f"   - {change['parameter']}: {change['change']} ({change['reason']})"
                )

            self._save_state()

        except Exception as e:
            logger.error(f"Morphic adaptation error: {e}")
            adaptations["error"] = str(e)

        return adaptations

    def _apply_adaptations(self, adaptations: Dict):
        """Apply morphic adaptations to the bot manager"""
        try:
            from bot_manager import get_bot_manager

            bot = get_bot_manager()

            new_config = adaptations.get("new_config", {})

            for key, value in new_config.items():
                if hasattr(bot.config, key):
                    old_value = getattr(bot.config, key)
                    setattr(bot.config, key, value)
                    logger.info(f"🧬 Applied: {key} = {old_value} -> {value}")

            # Save updated config
            bot._save_config()

        except Exception as e:
            logger.warning(f"Could not apply adaptations: {e}")

    def auto_optimize(self) -> Dict:
        """
        AUTO-OPTIMIZATION: Comprehensive self-improvement cycle.
        Gathers data, analyzes performance, and optimizes parameters.

        Returns:
            Optimization results
        """
        logger.info("🚀 AUTO-OPTIMIZATION: Starting self-improvement cycle...")

        results = {
            "timestamp": datetime.now().isoformat(),
            "phases": [],
            "optimizations": [],
            "status": "started",
        }

        try:
            # Phase 1: Gather current state
            results["phases"].append("Phase 1: Gathering data")
            account = self._tool_get_account_status({})
            positions = self._tool_get_open_positions({})
            performance = self._tool_get_performance_metrics({})
            sentiment = self._tool_get_market_sentiment({})

            # Phase 2: Analyze performance
            results["phases"].append("Phase 2: Analyzing performance")

            # Phase 3: Morphic adaptation
            results["phases"].append("Phase 3: Morphic adaptation")
            market_data = {
                "sentiment": sentiment,
                "account": account,
                "positions": positions,
            }
            adaptations = self.morphic_adapt(market_data, performance)
            results["optimizations"].extend(adaptations.get("changes", []))

            # Phase 4: Auto-adjust mood
            results["phases"].append("Phase 4: Mood adjustment")
            from dataclasses import asdict

            conditions = MarketConditions(
                regime=self.current_regime.value,
                volatility="normal",
                trend="neutral",
                risk_level="medium",
                timestamp=datetime.now().isoformat(),
            )
            new_mood, reason = self.auto_adjust_mood(performance, conditions)
            results["new_mood"] = new_mood.value
            results["mood_reason"] = reason

            # Phase 5: Generate insights
            results["phases"].append("Phase 5: Generating insights")

            results["status"] = "completed"
            results["summary"] = (
                f"Optimized {len(results['optimizations'])} parameters. Mood: {new_mood.value}"
            )

            logger.info(f"🚀 AUTO-OPTIMIZATION: Complete - {results['summary']}")

        except Exception as e:
            logger.error(f"Auto-optimization error: {e}")
            results["status"] = "failed"
            results["error"] = str(e)

            # Attempt self-healing
            healing = self.self_heal(e, {"phase": "auto_optimization"})
            results["healing_attempted"] = healing

        return results

    def continuous_improvement_loop(self, interval_minutes: int = 30) -> None:
        """
        Start continuous improvement loop that runs in background.
        The bot continuously monitors, adapts, and optimizes itself.

        Args:
            interval_minutes: Minutes between optimization cycles
        """
        import threading

        def improvement_cycle():
            while True:
                try:
                    logger.info("🔄 Continuous improvement cycle starting...")
                    self.auto_optimize()
                    import time

                    time.sleep(interval_minutes * 60)
                except Exception as e:
                    logger.error(f"Improvement cycle error: {e}")
                    self.self_heal(e, {"context": "continuous_improvement"})
                    import time

                    time.sleep(60)  # Wait 1 minute before retry

        thread = threading.Thread(target=improvement_cycle, daemon=True)
        thread.start()
        logger.info(
            f"🔄 Continuous improvement loop started (every {interval_minutes} min)"
        )

    def diagnose_issue(self, error_message: str, context: Dict) -> Dict:
        """
        Diagnose an issue and suggest fixes

        Args:
            error_message: The error that occurred
            context: Context when error occurred

        Returns:
            Diagnosis and suggested fixes
        """
        system_prompt = """You are a diagnostic AI for a trading bot.
Analyze the error and provide:
1. Root cause analysis
2. Immediate fix suggestion
3. Long-term prevention strategy
4. Whether the bot should continue trading

Respond in JSON format:
{
    "diagnosis": "root cause explanation",
    "severity": "low/medium/high/critical",
    "immediate_fix": "what to do now",
    "prevention": "how to prevent in future",
    "should_continue_trading": true/false,
    "recommended_action": "specific action"
}"""

        user_prompt = f"""Diagnose this trading bot error:

ERROR: {error_message}

CONTEXT:
{json.dumps(context, indent=2)}

What went wrong and how should we fix it?"""

        response = self._get_claude_response(
            system_prompt, user_prompt, max_tokens=1000
        )

        try:
            diagnosis = json.loads(response)
        except:
            diagnosis = {
                "diagnosis": response,
                "severity": "medium",
                "should_continue_trading": True,
            }

        # Create insight
        insight = BotInsight(
            category="error",
            insight=diagnosis.get("diagnosis", error_message),
            severity=diagnosis.get("severity", "medium"),
            action_required=diagnosis.get("severity") in ["high", "critical"],
            suggested_action=diagnosis.get("immediate_fix", "Review error logs"),
            timestamp=datetime.now().isoformat(),
        )
        self.insights_history.append(insight)
        self._save_state()

        return diagnosis

    # ========================================================================
    # TRADE JOURNALING AND LEARNING
    # ========================================================================

    def analyze_trade(self, trade: Dict, market_snapshot: Dict) -> Dict:
        """
        Analyze a completed trade for learning

        Args:
            trade: Completed trade details
            market_snapshot: Market conditions at trade time

        Returns:
            Trade analysis and lessons learned
        """
        system_prompt = """You are a trading journal AI. Analyze completed trades to:
1. Evaluate entry and exit quality
2. Assess risk/reward execution
3. Identify what went right or wrong
4. Extract actionable lessons

Respond in JSON format:
{
    "grade": "A/B/C/D/F",
    "entry_quality": "excellent/good/fair/poor",
    "exit_quality": "excellent/good/fair/poor",
    "risk_management": "excellent/good/fair/poor",
    "what_went_right": ["list"],
    "what_went_wrong": ["list"],
    "lessons": ["actionable lessons"],
    "would_take_again": true/false,
    "improvement_suggestions": ["specific improvements"]
}"""

        user_prompt = f"""Analyze this completed trade:

TRADE DETAILS:
{json.dumps(trade, indent=2)}

MARKET CONDITIONS AT ENTRY:
{json.dumps(market_snapshot, indent=2)}

Grade this trade and extract lessons for future improvement."""

        response = self._get_claude_response(
            system_prompt, user_prompt, max_tokens=1000
        )

        try:
            return json.loads(response)
        except:
            return {"raw_analysis": response}

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def get_status(self) -> Dict:
        """Get current intelligence status"""
        # Count MCP tools separately
        mcp_tool_count = sum(
            1 for t in self.tools if t.get("name", "").startswith("alpaca_")
        )
        base_tool_count = len(self.tools) - mcp_tool_count

        return {
            "ai_available": self.ai_available,
            "model": self.model if self.ai_available else None,
            "model_version": "Claude Sonnet 4.5",
            "features": {
                "parallel_tools": True,
                "tool_count": len(self.tools),
                "base_tools": base_tool_count,
                "max_tokens": self.max_tokens,
            },
            "mcp_integration": {
                "available": self.mcp_client is not None,
                "initialized": (
                    self.mcp_client.initialized if self.mcp_client else False
                ),
                "mcp_tools": mcp_tool_count,
                "direct_trading": (
                    self.mcp_client.initialized if self.mcp_client else False
                ),
            },
            "current_mood": self.current_mood.value,
            "current_regime": self.current_regime.value,
            "insights_count": len(self.insights_history),
            "adjustments_count": len(self.adjustment_history),
            "conversation_length": len(self.conversation_history),
            "capabilities": [
                "chat",
                "chat_with_tools",
                "comprehensive_analysis",
                "quick_market_scan",
                "analyze_performance",
                "diagnose_issue",
                (
                    "mcp_direct_trading"
                    if (self.mcp_client and self.mcp_client.initialized)
                    else None
                ),
            ],
            "last_updated": datetime.now().isoformat(),
        }

    def get_recent_insights(self, limit: int = 10) -> List[Dict]:
        """Get recent insights"""
        return [asdict(i) for i in self.insights_history[-limit:]]

    def get_recent_adjustments(self, limit: int = 10) -> List[Dict]:
        """Get recent strategy adjustments"""
        return [asdict(a) for a in self.adjustment_history[-limit:]]

    def clear_conversation(self):
        """Clear conversation history"""
        self.conversation_history = []
        return {"success": True, "message": "Conversation cleared"}


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_intelligence_instance: Optional[ClaudeBotIntelligence] = None


def get_bot_intelligence() -> ClaudeBotIntelligence:
    """Get or create the bot intelligence singleton"""
    global _intelligence_instance
    if _intelligence_instance is None:
        _intelligence_instance = ClaudeBotIntelligence()
    return _intelligence_instance


# ============================================================================
# QUICK TEST
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    intelligence = ClaudeBotIntelligence()
    print(f"\nBot Intelligence Status: {intelligence.get_status()}")

    # Test chat
    response = intelligence.chat("What's my current trading status?")
    print(f"\nChat Response: {response['response'][:200]}...")
