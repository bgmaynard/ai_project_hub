"""
Bot Manager Module
Manages the AI auto-trading bot lifecycle and provides status tracking
Includes PDT compliance, account type rules, trading restrictions,
and Claude AI integration for adaptive self-improvement.
"""
import asyncio
import logging
import threading
from datetime import datetime, date, timedelta
from typing import Dict, Optional, List
from dataclasses import dataclass, asdict, field
from enum import Enum
import json
from pathlib import Path

# Import trading dependencies at module level for faster execution
from alpaca_integration import get_alpaca_connector
from ai.alpaca_ai_predictor import get_alpaca_predictor
from watchlist_manager import get_watchlist_manager
from portfolio_analytics import get_portfolio_analytics
from trade_execution import get_execution_tracker
from alpaca_market_data import get_alpaca_market_data

# Unified broker for live trading (Schwab primary)
try:
    from unified_broker import get_unified_broker
    HAS_UNIFIED_BROKER = True
except ImportError:
    HAS_UNIFIED_BROKER = False

# Claude AI Intelligence for adaptive behavior
from ai.claude_bot_intelligence import get_bot_intelligence, BotMood

# Next-Gen AI Logic Blueprint Modules
from ai.regime_classifier import get_regime_classifier, MarketRegime
from ai.trade_narrative import get_narrative_generator
from ai.kelly_sizer import get_kelly_sizer
from ai.brier_tracker import get_brier_tracker

logger = logging.getLogger(__name__)


class AccountType(str, Enum):
    """Supported account types with different trading rules"""
    CASH = "cash"           # No margin, T+2 settlement, no PDT concerns
    MARGIN = "margin"       # Margin available, PDT rules apply if < $25k
    IRA = "ira"             # No shorting, no margin, tax-advantaged
    ROTH_IRA = "roth_ira"   # No shorting, no margin, tax-free growth


@dataclass
class PDTStatus:
    """Pattern Day Trader status tracking"""
    day_trades_count: int = 0           # Day trades in rolling 5-day window
    day_trades_remaining: int = 3       # Remaining before PDT flag
    is_pdt_flagged: bool = False        # Currently flagged as PDT
    equity: float = 0.0                 # Current account equity
    pdt_threshold: float = 25000.0      # PDT equity threshold
    is_pdt_restricted: bool = False     # Trading restricted due to PDT
    last_updated: str = ""

    # Track day trades with timestamps for rolling window
    day_trade_history: List[Dict] = field(default_factory=list)


@dataclass
class BotTrade:
    """Record of a bot trade"""
    symbol: str
    side: str
    quantity: int
    price: float
    confidence: float
    signal: str
    order_id: str
    timestamp: str
    status: str = "filled"


@dataclass
class BotConfig:
    """Bot configuration with compliance settings and AI integration"""
    # Trading parameters
    confidence_threshold: float = 0.15
    max_positions: int = 3
    max_daily_trades: int = 10
    position_size: int = 1
    trading_enabled: bool = True
    cycle_interval_seconds: int = 120

    # Account type and compliance
    account_type: str = "margin"        # cash, margin, ira, roth_ira
    long_only: bool = False             # If True, no shorting allowed
    enforce_pdt_rules: bool = True      # Enforce PDT compliance
    max_day_trades: int = 3             # Max day trades in 5-day window (PDT limit)

    # Settlement tracking (for cash accounts)
    respect_settlement: bool = True     # Wait for T+2 settlement in cash accounts
    settled_cash_only: bool = False     # Only use settled funds (cash accounts)

    # NEWS CATALYST TRADING (Ross Cameron's Warrior Trading)
    news_trigger_enabled: bool = True   # Trigger trades on breaking news
    news_min_severity: float = 0.5      # Minimum news severity to trigger (0.0-1.0)
    news_boost_confidence: float = 0.2  # Boost to AI confidence when news present
    news_scan_interval: int = 60        # Check for news every N seconds

    # RISK MANAGEMENT - TRAILING STOPS
    trailing_stop_enabled: bool = True   # Enable trailing stop protection
    trailing_stop_percent: float = 3.0   # Trailing stop percentage (e.g., 3.0 = 3%)
    initial_stop_percent: float = 5.0    # Initial stop loss percentage
    take_profit_percent: float = 8.0     # Take profit threshold
    lock_in_profit_at: float = 2.0       # Lock in profit once gain reaches this %
    lock_in_trail_percent: float = 1.5   # Tighter trailing stop after locking profit

    # END-OF-DAY LIQUIDATION (Small Cap Momentum Strategy)
    # Close all positions before market close to avoid overnight gap risk
    eod_liquidation_enabled: bool = True  # Enable end-of-day position closing
    eod_liquidation_time: str = "15:45"   # Time to start liquidation (ET) - 15 min before close
    no_new_entries_after: str = "15:30"   # Stop opening new positions after this time (ET)
    friday_early_close: bool = True       # Close even earlier on Fridays (avoid weekend risk)
    friday_liquidation_time: str = "15:30" # Friday liquidation time (ET)

    # ORDER RATE LIMITING
    min_order_interval_seconds: int = 5  # Minimum seconds between orders for same symbol
    max_orders_per_symbol: int = 2       # Maximum pending orders per symbol

    # Claude AI Integration
    ai_enabled: bool = True             # Enable Claude AI intelligence
    ai_adaptive_mode: bool = True       # Allow AI to adjust parameters
    ai_mood_control: bool = True        # Let AI control trading mood
    ai_self_healing: bool = True        # Enable AI error recovery
    ai_analysis_interval: int = 10      # Analyze performance every N trades
    ai_market_analysis: bool = True     # Analyze market conditions


class BotManager:
    """
    Singleton manager for the AI trading bot
    Handles bot lifecycle, configuration, status tracking, compliance,
    and Claude AI integration for adaptive self-improvement.
    """

    def __init__(self):
        self.config = BotConfig()
        self.running = False
        self.enabled = False
        self.initialized = False

        # Trading state
        self.trades_today: List[BotTrade] = []
        self.pnl_today: float = 0.0
        self.last_cycle_time: Optional[str] = None
        self.current_positions: Dict[str, dict] = {}
        self.last_signals: Dict[str, dict] = {}

        # Position tracking for trailing stops
        self.position_high_watermarks: Dict[str, float] = {}  # symbol -> highest price seen
        self.position_entry_prices: Dict[str, float] = {}     # symbol -> entry price
        self.profit_locked: Dict[str, bool] = {}              # symbol -> has profit been locked
        self.last_order_time: Dict[str, datetime] = {}        # symbol -> last order timestamp
        self.pending_orders_count: Dict[str, int] = {}        # symbol -> pending order count

        # PDT and compliance tracking
        self.pdt_status = PDTStatus()
        self.compliance_warnings: List[str] = []

        # Claude AI Intelligence
        self.ai_intelligence = None
        self.ai_mood = BotMood.CONSERVATIVE
        self.ai_last_analysis: Optional[str] = None
        self.trades_since_analysis = 0

        # Next-Gen AI Logic Modules
        self.regime_classifier = None
        self.narrative_generator = None
        self.kelly_sizer = None
        self.brier_tracker = None
        self.current_regime = MarketRegime.UNKNOWN

        # Bot thread
        self._bot_task: Optional[asyncio.Task] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # Data persistence
        self.data_path = Path("store/bot")
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.trades_file = self.data_path / "bot_trades.json"
        self.config_file = self.data_path / "bot_config.json"
        self.pdt_file = self.data_path / "pdt_status.json"

        # Load saved state
        self._load_config()
        self._load_trades()
        self._load_pdt_status()

        # Initialize Claude AI Intelligence
        self._init_ai_intelligence()

        # Initialize Sentiment Analyzer for news triggers
        self.sentiment_analyzer = None
        self._init_sentiment_analyzer()

        # Initialize Next-Gen AI Logic modules
        self._init_nextgen_ai_modules()

        logger.info("BotManager initialized with account type: %s, AI enabled: %s, News triggers: %s",
                   self.config.account_type, self.config.ai_enabled, self.config.news_trigger_enabled)

        # Start continuous improvement if AI is enabled
        if self.config.ai_enabled and self.config.ai_adaptive_mode and self.ai_intelligence:
            try:
                self.ai_intelligence.continuous_improvement_loop(interval_minutes=30)
                logger.info("🧬 Morphic self-improvement loop started")
            except Exception as e:
                logger.warning(f"Could not start improvement loop: {e}")

    def _load_config(self):
        """Load bot configuration from file"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                    self.config = BotConfig(**data)
                logger.info("Bot config loaded from file")
            except Exception as e:
                logger.warning(f"Could not load bot config: {e}")

    def _save_config(self):
        """Save bot configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(asdict(self.config), f, indent=2)
        except Exception as e:
            logger.error(f"Could not save bot config: {e}")

    def _load_trades(self):
        """Load today's trades from file"""
        if self.trades_file.exists():
            try:
                with open(self.trades_file, 'r') as f:
                    data = json.load(f)

                # Only load today's trades
                today = date.today().isoformat()
                self.trades_today = []
                for trade_data in data.get("trades", []):
                    trade_date = trade_data.get("timestamp", "")[:10]
                    if trade_date == today:
                        self.trades_today.append(BotTrade(**trade_data))

                self.pnl_today = sum(t.price * t.quantity * (1 if t.side == "SELL" else -1)
                                     for t in self.trades_today)
                logger.info(f"Loaded {len(self.trades_today)} trades for today")
            except Exception as e:
                logger.warning(f"Could not load trades: {e}")

    def _load_pdt_status(self):
        """Load PDT status from file"""
        if self.pdt_file.exists():
            try:
                with open(self.pdt_file, 'r') as f:
                    data = json.load(f)
                    # Filter day trade history to last 5 trading days
                    history = data.get("day_trade_history", [])
                    cutoff = (datetime.now() - timedelta(days=5)).isoformat()
                    filtered_history = [t for t in history if t.get("timestamp", "") > cutoff]
                    data["day_trade_history"] = filtered_history
                    data["day_trades_count"] = len(filtered_history)
                    self.pdt_status = PDTStatus(**data)
                logger.info(f"PDT status loaded: {self.pdt_status.day_trades_count} day trades in window")
            except Exception as e:
                logger.warning(f"Could not load PDT status: {e}")

    def _save_pdt_status(self):
        """Save PDT status to file"""
        try:
            with open(self.pdt_file, 'w') as f:
                json.dump(asdict(self.pdt_status), f, indent=2)
        except Exception as e:
            logger.error(f"Could not save PDT status: {e}")

    # ========================================================================
    # CLAUDE AI INTELLIGENCE INTEGRATION
    # ========================================================================

    def _init_ai_intelligence(self):
        """Initialize Claude AI Intelligence"""
        if self.config.ai_enabled:
            try:
                self.ai_intelligence = get_bot_intelligence()
                self.ai_mood = self.ai_intelligence.current_mood
                logger.info(f"Claude AI Intelligence initialized. Mood: {self.ai_mood.value}")
            except Exception as e:
                logger.warning(f"Could not initialize AI Intelligence: {e}")
                self.ai_intelligence = None

    def _init_sentiment_analyzer(self):
        """Initialize Sentiment Analyzer for news catalyst triggers"""
        if self.config.news_trigger_enabled:
            try:
                from ai.warrior_sentiment_analyzer import get_sentiment_analyzer
                self.sentiment_analyzer = get_sentiment_analyzer()
                logger.info("Sentiment Analyzer initialized for news triggers")
            except ImportError:
                logger.warning("Sentiment analyzer not available - news triggers disabled")
                self.sentiment_analyzer = None
            except Exception as e:
                logger.warning(f"Could not initialize Sentiment Analyzer: {e}")
                self.sentiment_analyzer = None

    def _init_nextgen_ai_modules(self):
        """Initialize Next-Gen AI Logic Blueprint modules"""
        try:
            # Initialize regime classifier
            self.regime_classifier = get_regime_classifier()
            logger.info("Regime Classifier initialized")

            # Initialize narrative generator
            self.narrative_generator = get_narrative_generator()
            logger.info("Trade Narrative Generator initialized")

            # Initialize Kelly sizer (will update account value on first trade)
            self.kelly_sizer = get_kelly_sizer()
            logger.info("Kelly Sizer initialized")

            # Initialize Brier tracker
            self.brier_tracker = get_brier_tracker()
            logger.info("Brier Tracker initialized")

            logger.info("Next-Gen AI modules ready")
        except Exception as e:
            logger.warning(f"Could not initialize some Next-Gen AI modules: {e}")

    def classify_market_regime(self, symbol: str = "SPY") -> Dict:
        """
        Classify current market regime using price data.

        Returns regime analysis with adjustments for trading parameters.
        """
        if not self.regime_classifier:
            return {"regime": "unknown", "adjustments": {}}

        try:
            market_data = get_alpaca_market_data()
            bars = market_data.get_bars(symbol, timeframe="5Min", limit=50)

            if not bars:
                return {"regime": "unknown", "adjustments": {}}

            prices = [b.get("close", 0) for b in bars]
            highs = [b.get("high", 0) for b in bars]
            lows = [b.get("low", 0) for b in bars]
            volumes = [b.get("volume", 0) for b in bars]

            analysis = self.regime_classifier.classify(
                prices=prices,
                highs=highs,
                lows=lows,
                volumes=volumes,
                timestamp=datetime.now()
            )

            self.current_regime = analysis.regime
            adjustments = self.regime_classifier.get_regime_adjustments(analysis.regime)

            return {
                "regime": analysis.regime.value,
                "session": analysis.session.value,
                "confidence": analysis.confidence,
                "trend_strength": analysis.trend_strength,
                "volatility_percentile": analysis.volatility_percentile,
                "recommendation": analysis.recommendation,
                "adjustments": adjustments,
                "details": analysis.details
            }

        except Exception as e:
            logger.warning(f"Regime classification failed: {e}")
            return {"regime": "unknown", "adjustments": {}}

    def generate_trade_narrative(self, symbol: str, action: str, price: float,
                                 factors: Dict) -> Dict:
        """Generate a plain-English narrative for a trade decision"""
        if not self.narrative_generator:
            return {"narrative": "Narrative generation unavailable"}

        try:
            regime_info = self.classify_market_regime()

            narrative = self.narrative_generator.generate(
                symbol=symbol,
                action=action,
                price=price,
                factors=factors,
                regime=regime_info.get("regime", "unknown"),
                session=regime_info.get("session", "unknown")
            )

            return narrative.to_dict()

        except Exception as e:
            logger.warning(f"Narrative generation failed: {e}")
            return {"narrative": f"Error: {e}"}

    def calculate_position_size(self, symbol: str, price: float,
                                confidence: float = 0.5,
                                stop_loss_percent: float = None) -> Dict:
        """
        Calculate optimal position size using Kelly criterion.

        Returns sizing recommendation with reasoning.
        """
        if not self.kelly_sizer:
            return {
                "recommended_shares": self.config.position_size,
                "reasoning": "Kelly sizer unavailable, using default"
            }

        try:
            # Update account value
            connector = get_alpaca_connector()
            account = connector.get_account()
            equity = float(account.get("equity", 100000))
            self.kelly_sizer.update_account_value(equity)

            # Get regime adjustments
            regime_info = self.classify_market_regime()
            regime_multiplier = regime_info.get("adjustments", {}).get("position_size_multiplier", 1.0)

            # Calculate size
            stop_pct = stop_loss_percent or self.config.initial_stop_percent

            result = self.kelly_sizer.calculate_position_size(
                symbol=symbol,
                price=price,
                confidence=confidence,
                stop_loss_percent=stop_pct,
                volatility_factor=regime_info.get("volatility_percentile", 50) / 50,  # Normalize around 1.0
                regime_multiplier=regime_multiplier
            )

            return {
                "recommended_shares": result.recommended_shares,
                "recommended_value": result.recommended_value,
                "kelly_fraction": result.kelly_fraction,
                "position_percent": result.position_percent,
                "reasoning": result.reasoning,
                "expected_value": result.expected_value
            }

        except Exception as e:
            logger.warning(f"Position sizing failed: {e}")
            return {
                "recommended_shares": self.config.position_size,
                "reasoning": f"Error: {e}, using default"
            }

    def record_prediction_outcome(self, symbol: str, entry_price: float,
                                   exit_price: float, model_name: str = "default"):
        """Record a prediction outcome for Brier score tracking"""
        if not self.brier_tracker:
            return

        try:
            # Resolve any pending predictions for this symbol
            resolved = self.brier_tracker.resolve_by_symbol(
                symbol=symbol,
                exit_price=exit_price,
                model_name=model_name
            )

            if resolved:
                logger.info(f"Resolved {len(resolved)} predictions for {symbol}")

                # Also record in Kelly sizer for win rate tracking
                if self.kelly_sizer:
                    for pred in resolved:
                        side = "long" if pred.prediction_direction == "up" else "short"
                        self.kelly_sizer.record_trade(
                            symbol=symbol,
                            entry_price=entry_price,
                            exit_price=exit_price,
                            quantity=1,  # Normalized
                            side=side
                        )

        except Exception as e:
            logger.warning(f"Failed to record prediction outcome: {e}")

    def get_model_health(self, model_name: str = None) -> Dict:
        """Get health status for AI prediction models"""
        if not self.brier_tracker:
            return {"status": "tracker_unavailable"}

        try:
            if model_name:
                health = self.brier_tracker.get_model_health(model_name)
                if health:
                    return {
                        "model": model_name,
                        "brier_score": health.brier_score,
                        "accuracy": health.accuracy,
                        "is_degraded": health.is_degraded,
                        "degradation_reason": health.degradation_reason,
                        "total_predictions": health.total_predictions
                    }
            else:
                summary = self.brier_tracker.get_summary()
                return summary

        except Exception as e:
            logger.warning(f"Failed to get model health: {e}")
            return {"error": str(e)}

    async def check_breaking_news(self, symbols: List[str]) -> List[Dict]:
        """
        Check for breaking news on symbols that could trigger trades

        Args:
            symbols: List of symbols to check

        Returns:
            List of news alerts with trading signals
        """
        if not self.sentiment_analyzer or not self.config.news_trigger_enabled:
            return []

        news_signals = []

        try:
            for symbol in symbols:
                # Check for breaking news
                if hasattr(self.sentiment_analyzer, 'detect_breaking_news'):
                    alert = await self.sentiment_analyzer.detect_breaking_news(symbol)

                    if alert and alert.severity >= self.config.news_min_severity:
                        # Determine trade direction based on sentiment
                        if alert.sentiment_score > 0.3:
                            action = "BUY"
                        elif alert.sentiment_score < -0.3:
                            action = "SELL"
                        else:
                            continue  # Skip neutral news

                        news_signals.append({
                            "symbol": symbol,
                            "action": action,
                            "confidence": min(0.95, alert.confidence + self.config.news_boost_confidence),
                            "signal": f"NEWS_{alert.alert_type.upper()}",
                            "news_headline": alert.headline,
                            "news_severity": alert.severity,
                            "news_sentiment": alert.sentiment_score,
                            "trigger": "breaking_news"
                        })
                        logger.info(f"📰 NEWS TRIGGER: {symbol} - {action} (Severity: {alert.severity:.2f}, Sentiment: {alert.sentiment_score:+.2f})")

        except Exception as e:
            logger.error(f"Error checking breaking news: {e}")

        return news_signals

    def ai_analyze_performance(self) -> Dict:
        """Have Claude AI analyze bot performance"""
        if not self.ai_intelligence or not self.config.ai_enabled:
            return {"status": "ai_disabled"}

        try:
            # Get account info
            connector = get_alpaca_connector()
            account = connector.get_account()
            equity = account.get("equity", 0)

            # Get trades with PnL
            trades_with_pnl = []
            for trade in self.trades_today:
                trade_dict = asdict(trade)
                # Simple PnL calculation (would need exit price for accuracy)
                trade_dict["pnl"] = 0  # Placeholder
                trades_with_pnl.append(trade_dict)

            # Run analysis
            analysis = self.ai_intelligence.analyze_performance(trades_with_pnl, equity)

            # Update mood if AI mood control is enabled
            if self.config.ai_mood_control:
                mood_rec = analysis.get("mood_recommendation", self.ai_mood.value)
                try:
                    self.ai_mood = BotMood(mood_rec)
                    self.ai_intelligence.current_mood = self.ai_mood
                except:
                    pass

            self.ai_last_analysis = datetime.now().isoformat()
            self.trades_since_analysis = 0

            return analysis

        except Exception as e:
            logger.error(f"AI performance analysis failed: {e}")
            return {"error": str(e)}

    def ai_analyze_market(self, symbols: List[str] = None) -> Dict:
        """Have Claude AI analyze market conditions"""
        if not self.ai_intelligence or not self.config.ai_market_analysis:
            return {"status": "ai_disabled"}

        try:
            market_data = get_alpaca_market_data()
            symbols = symbols or ["SPY", "QQQ", "VIX"]

            # Gather market data - convert to JSON-serializable format
            market_snapshot = {}
            for symbol in symbols:
                try:
                    quote = market_data.get_latest_quote(symbol)
                    if quote:
                        # Convert quote dict values to JSON-serializable types
                        serializable_quote = {}
                        for k, v in quote.items():
                            if hasattr(v, 'isoformat'):  # datetime objects
                                serializable_quote[k] = v.isoformat()
                            elif hasattr(v, '__dict__'):  # objects
                                serializable_quote[k] = str(v)
                            else:
                                serializable_quote[k] = v
                        market_snapshot[symbol] = serializable_quote
                except:
                    pass

            # Analyze conditions
            conditions = self.ai_intelligence.analyze_market_conditions(market_snapshot)

            # Convert conditions to dict, handling datetime fields
            if hasattr(conditions, '__dict__'):
                conditions_dict = asdict(conditions)
            elif isinstance(conditions, dict):
                conditions_dict = conditions
            else:
                conditions_dict = {"raw": str(conditions)}

            return {
                "conditions": conditions_dict,
                "regime": self.ai_intelligence.current_regime.value,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"AI market analysis failed: {e}")
            return {"error": str(e)}

    def ai_suggest_adjustments(self) -> List[Dict]:
        """Get AI-suggested strategy adjustments"""
        if not self.ai_intelligence or not self.config.ai_adaptive_mode:
            return []

        try:
            # Get current performance
            performance = self.ai_analyze_performance()

            # Get market conditions
            market = self.ai_analyze_market()
            from ai.claude_bot_intelligence import MarketConditions
            conditions = MarketConditions(**market.get("conditions", {}))

            # Get suggestions
            adjustments = self.ai_intelligence.suggest_strategy_adjustments(
                current_config=asdict(self.config),
                performance=performance,
                market_conditions=conditions
            )

            return [asdict(a) for a in adjustments]

        except Exception as e:
            logger.error(f"AI adjustment suggestions failed: {e}")
            return []

    def ai_apply_adjustments(self, adjustments: List[Dict] = None) -> Dict:
        """Apply AI-suggested adjustments (with user approval or auto)"""
        if not adjustments:
            adjustments = self.ai_suggest_adjustments()

        applied = []
        for adj in adjustments:
            if adj.get("confidence", 0) >= 0.7:  # Only apply high-confidence adjustments
                param = adj.get("parameter")
                new_value = adj.get("recommended_value")

                if param and new_value is not None:
                    if hasattr(self.config, param):
                        setattr(self.config, param, new_value)
                        applied.append(adj)
                        logger.info(f"AI adjusted {param}: {adj.get('current_value')} -> {new_value}")

        if applied:
            self._save_config()

        return {
            "applied": applied,
            "skipped": len(adjustments) - len(applied),
            "timestamp": datetime.now().isoformat()
        }

    def ai_chat(self, message: str) -> Dict:
        """Chat with the bot via natural language"""
        if not self.ai_intelligence:
            return {"response": "AI is not available", "action": "none"}

        # Provide context
        context = {
            "status": "running" if self.running else "stopped",
            "enabled": self.enabled,
            "mood": self.ai_mood.value,
            "trades_today": len(self.trades_today),
            "pnl_today": self.pnl_today,
            "positions": len(self.current_positions),
            "config": asdict(self.config)
        }

        response = self.ai_intelligence.chat(message, context)

        # Execute any actions from the response
        action = response.get("action", "none")
        params = response.get("parameters", {})

        if action == "start_bot":
            self.start()
        elif action == "stop_bot":
            self.stop()
        elif action == "set_config":
            self.update_config(params)
        elif action == "set_long_only":
            self.config.long_only = params.get("enabled", True)
            self._save_config()

        return response

    def ai_diagnose_error(self, error: str, context: Dict = None) -> Dict:
        """Have AI diagnose an error"""
        if not self.ai_intelligence or not self.config.ai_self_healing:
            return {"diagnosis": "AI unavailable"}

        context = context or {
            "running": self.running,
            "positions": len(self.current_positions),
            "last_trade": asdict(self.trades_today[-1]) if self.trades_today else None
        }

        return self.ai_intelligence.diagnose_issue(error, context)

    def ai_get_mood_adjustments(self) -> Dict:
        """Get trading parameter adjustments based on AI mood"""
        mood_adjustments = {
            BotMood.AGGRESSIVE: {
                "confidence_multiplier": 0.8,  # Lower threshold
                "position_multiplier": 1.5,
                "max_positions_multiplier": 1.5
            },
            BotMood.CONSERVATIVE: {
                "confidence_multiplier": 1.2,  # Higher threshold
                "position_multiplier": 0.7,
                "max_positions_multiplier": 0.7
            },
            BotMood.DEFENSIVE: {
                "confidence_multiplier": 1.5,  # Much higher threshold
                "position_multiplier": 0.5,
                "max_positions_multiplier": 0.5
            },
            BotMood.OPPORTUNISTIC: {
                "confidence_multiplier": 1.0,
                "position_multiplier": 1.0,
                "max_positions_multiplier": 1.0
            },
            BotMood.LEARNING: {
                "confidence_multiplier": 1.3,
                "position_multiplier": 0.5,
                "max_positions_multiplier": 0.5
            }
        }

        adjustments = mood_adjustments.get(self.ai_mood, mood_adjustments[BotMood.CONSERVATIVE])

        return {
            "mood": self.ai_mood.value,
            "effective_confidence_threshold": self.config.confidence_threshold * adjustments["confidence_multiplier"],
            "effective_position_size": max(1, int(self.config.position_size * adjustments["position_multiplier"])),
            "effective_max_positions": max(1, int(self.config.max_positions * adjustments["max_positions_multiplier"])),
            "adjustments": adjustments
        }

    # ========================================================================
    # TRAILING STOP & RISK MANAGEMENT
    # ========================================================================

    def _can_place_order(self, symbol: str) -> bool:
        """Check if we can place an order for this symbol (rate limiting)"""
        now = datetime.now()

        # Check order interval
        last_time = self.last_order_time.get(symbol)
        if last_time:
            elapsed = (now - last_time).total_seconds()
            if elapsed < self.config.min_order_interval_seconds:
                logger.debug(f"Order rate limited for {symbol}: {elapsed:.1f}s < {self.config.min_order_interval_seconds}s")
                return False

        # CRITICAL: Check ACTUAL open orders from Alpaca, not stale counter
        # The pending_orders_count was never being decremented, causing orders to get stuck
        try:
            open_orders = self.connector.get_orders(status="open", limit=50)
            actual_pending = sum(1 for o in open_orders if o.get("symbol") == symbol)

            # Sync our counter with reality
            if actual_pending != self.pending_orders_count.get(symbol, 0):
                logger.debug(f"Syncing pending orders for {symbol}: counter={self.pending_orders_count.get(symbol, 0)} actual={actual_pending}")
                self.pending_orders_count[symbol] = actual_pending

            if actual_pending >= self.config.max_orders_per_symbol:
                logger.debug(f"Max pending orders for {symbol}: {actual_pending} >= {self.config.max_orders_per_symbol}")
                return False
        except Exception as e:
            logger.warning(f"Could not check open orders for {symbol}: {e}")
            # Fall back to counter if API fails
            pending = self.pending_orders_count.get(symbol, 0)
            if pending >= self.config.max_orders_per_symbol:
                return False

        return True

    def _record_order_placed(self, symbol: str):
        """Record that an order was placed for rate limiting"""
        self.last_order_time[symbol] = datetime.now()
        self.pending_orders_count[symbol] = self.pending_orders_count.get(symbol, 0) + 1

    def _record_order_filled(self, symbol: str):
        """Record that an order was filled"""
        self.pending_orders_count[symbol] = max(0, self.pending_orders_count.get(symbol, 0) - 1)

    # ========================================================================
    # END-OF-DAY LIQUIDATION (Small Cap Momentum - No Overnight Risk)
    # ========================================================================

    def _get_current_et_time(self):
        """Get current time in Eastern timezone"""
        import pytz
        et_tz = pytz.timezone('US/Eastern')
        return datetime.now(et_tz)

    def _is_eod_liquidation_time(self) -> bool:
        """
        Check if it's time to liquidate all positions.
        Returns True if we should close everything.
        """
        if not self.config.eod_liquidation_enabled:
            return False

        now_et = self._get_current_et_time()
        current_time = now_et.strftime("%H:%M")
        day_of_week = now_et.weekday()  # Monday=0, Friday=4

        # Friday: use earlier liquidation time
        if day_of_week == 4 and self.config.friday_early_close:
            liquidation_time = self.config.friday_liquidation_time
        else:
            liquidation_time = self.config.eod_liquidation_time

        return current_time >= liquidation_time

    def _should_block_new_entries(self) -> bool:
        """
        Check if we should stop opening new positions.
        Returns True if too late in the day for new entries.
        """
        if not self.config.eod_liquidation_enabled:
            return False

        now_et = self._get_current_et_time()
        current_time = now_et.strftime("%H:%M")

        return current_time >= self.config.no_new_entries_after

    async def _liquidate_all_positions(self, connector, reason: str = "eod_liquidation") -> List[Dict]:
        """
        Liquidate all positions for end-of-day.
        Uses EMERGENCY sell (market order during regular hours) for fast execution.
        """
        results = []
        positions = connector.get_positions()

        if not positions:
            logger.info(f"📤 EOD Liquidation: No positions to close")
            return results

        logger.warning(f"🌙 EOD LIQUIDATION TRIGGERED: Closing {len(positions)} positions - {reason}")

        for position in positions:
            symbol = position.get("symbol")
            quantity = int(abs(float(position.get("qty") or position.get("quantity", 0))))

            if quantity <= 0:
                continue

            try:
                # Use EMERGENCY=True for fastest execution (market order during regular hours)
                order = connector.place_smart_order(
                    symbol=symbol,
                    quantity=quantity,
                    side="SELL",
                    emergency=True  # Fast execution - use market order if possible
                )

                if order and order.get("order_id"):
                    result = {
                        "symbol": symbol,
                        "quantity": quantity,
                        "order_id": order.get("order_id"),
                        "status": "submitted",
                        "reason": reason,
                        "order_type": order.get("order_type", "unknown")
                    }
                    results.append(result)
                    logger.info(f"🌙 EOD SELL: {symbol} x{quantity} - {order.get('order_type', 'unknown')} order")

                    # Cleanup tracking
                    self._cleanup_position_tracking(symbol)

            except Exception as e:
                logger.error(f"EOD Liquidation failed for {symbol}: {e}")
                results.append({
                    "symbol": symbol,
                    "quantity": quantity,
                    "status": "error",
                    "error": str(e)
                })

        return results

    def _update_position_tracking(self, symbol: str, current_price: float, entry_price: float):
        """Update position tracking for trailing stops"""
        # Store entry price if not already stored
        if symbol not in self.position_entry_prices:
            self.position_entry_prices[symbol] = entry_price
            self.position_high_watermarks[symbol] = current_price
            self.profit_locked[symbol] = False
            logger.info(f"📊 Position tracking started for {symbol}: entry=${entry_price:.2f}")

        # Update high watermark
        if current_price > self.position_high_watermarks.get(symbol, 0):
            self.position_high_watermarks[symbol] = current_price
            logger.debug(f"📈 New high watermark for {symbol}: ${current_price:.2f}")

        # Check if profit should be locked
        entry = self.position_entry_prices[symbol]
        gain_pct = ((current_price - entry) / entry) * 100

        if gain_pct >= self.config.lock_in_profit_at and not self.profit_locked.get(symbol):
            self.profit_locked[symbol] = True
            logger.info(f"🔒 Profit LOCKED for {symbol}: +{gain_pct:.1f}% (threshold: {self.config.lock_in_profit_at}%)")

    def _check_trailing_stop(self, symbol: str, current_price: float) -> Dict:
        """
        Check if trailing stop has been triggered for a position.

        TRAILING STOP LOGIC (based on % from HIGH, not entry):
        - Always trails from the highest price reached (high watermark)
        - As gains increase, the trail tightens to protect profits
        - Example: Stock bought at $100, runs to $110 (+10%), then drops:
          - At $108.50 (1.5% drop from high) -> SELL to lock in ~8.5% gain
          - NOT waiting until it drops below $100 entry!

        Returns action recommendation with reason.
        """
        if not self.config.trailing_stop_enabled:
            return {"action": "HOLD", "reason": "trailing_stop_disabled"}

        if symbol not in self.position_entry_prices:
            return {"action": "HOLD", "reason": "no_position_tracking"}

        entry_price = self.position_entry_prices[symbol]
        high_watermark = self.position_high_watermarks.get(symbol, entry_price)

        # Calculate percentages
        current_gain_pct = ((current_price - entry_price) / entry_price) * 100
        gain_from_entry_at_high = ((high_watermark - entry_price) / entry_price) * 100 if entry_price > 0 else 0
        drop_from_high_pct = ((high_watermark - current_price) / high_watermark) * 100 if high_watermark > 0 else 0

        # 1. CHECK INITIAL STOP LOSS (from entry price - only safety net)
        if current_gain_pct <= -self.config.initial_stop_percent:
            return {
                "action": "SELL",
                "reason": "initial_stop_loss",
                "details": f"Loss of {current_gain_pct:.1f}% exceeds initial stop of -{self.config.initial_stop_percent}%",
                "urgency": "HIGH"
            }

        # 2. CHECK TAKE PROFIT (optional auto-exit at target)
        if current_gain_pct >= self.config.take_profit_percent:
            return {
                "action": "SELL",
                "reason": "take_profit",
                "details": f"Profit of +{current_gain_pct:.1f}% reached take profit of +{self.config.take_profit_percent}%",
                "urgency": "MEDIUM"
            }

        # 3. DYNAMIC TRAILING STOP (% from HIGH WATERMARK)
        # Trail tightens as profits grow - protect gains aggressively!

        # Determine trailing % based on how much we're up from HIGH
        if gain_from_entry_at_high >= 5.0:
            # Big winner (+5%+): Very tight 1% trail to protect gains
            trail_pct = 1.0
            trail_mode = "tight_winner"
        elif gain_from_entry_at_high >= 3.0:
            # Good profit (+3-5%): Tight 1.5% trail
            trail_pct = 1.5
            trail_mode = "profit_protection"
        elif gain_from_entry_at_high >= 1.5:
            # Small profit (+1.5-3%): Medium 2% trail
            trail_pct = 2.0
            trail_mode = "lock_profit"
        elif gain_from_entry_at_high >= 0:
            # Break-even to small gain: Standard trail
            trail_pct = self.config.trailing_stop_percent  # 2.5%
            trail_mode = "standard"
        else:
            # Underwater (high was still below entry): Wider trail, let it recover
            trail_pct = self.config.trailing_stop_percent + 1.0  # 3.5%
            trail_mode = "recovery"

        # CHECK IF TRAILING STOP TRIGGERED
        if drop_from_high_pct >= trail_pct:
            # Calculate what we're locking in
            profit_locked_in = current_gain_pct

            return {
                "action": "SELL",
                "reason": "trailing_stop",
                "details": f"Price dropped {drop_from_high_pct:.1f}% from high ${high_watermark:.2f}. Trail: {trail_pct}% ({trail_mode}). Locking in {profit_locked_in:+.1f}%",
                "urgency": "HIGH",
                "trail_mode": trail_mode,
                "profit_at_exit": profit_locked_in,
                "high_watermark": high_watermark
            }

        return {
            "action": "HOLD",
            "reason": "within_limits",
            "current_gain_pct": current_gain_pct,
            "gain_at_high": gain_from_entry_at_high,
            "drop_from_high_pct": drop_from_high_pct,
            "trail_pct": trail_pct,
            "trail_mode": trail_mode,
            "high_watermark": high_watermark
        }

    def _detect_reversal(self, symbol: str, current_price: float, bars: List[Dict] = None) -> Dict:
        """
        Detect potential reversal patterns for quick exit.
        Uses price action and momentum to identify reversals.
        """
        if not bars or len(bars) < 5:
            return {"reversal_detected": False, "confidence": 0}

        # Get recent price action
        recent_closes = [b.get("close", 0) for b in bars[-5:]]
        recent_highs = [b.get("high", 0) for b in bars[-5:]]
        recent_lows = [b.get("low", 0) for b in bars[-5:]]

        if not all(recent_closes) or 0 in recent_closes:
            return {"reversal_detected": False, "confidence": 0}

        reversal_signals = []
        confidence = 0

        # 1. BEARISH ENGULFING: Current candle engulfs previous green candle
        if len(bars) >= 2:
            prev_bar = bars[-2]
            curr_bar = bars[-1]
            prev_body = prev_bar.get("close", 0) - prev_bar.get("open", 0)
            curr_body = curr_bar.get("close", 0) - curr_bar.get("open", 0)

            # Previous was green (close > open), current is red and engulfs
            if prev_body > 0 and curr_body < 0:
                if curr_bar.get("open", 0) >= prev_bar.get("close", 0) and curr_bar.get("close", 0) <= prev_bar.get("open", 0):
                    reversal_signals.append("bearish_engulfing")
                    confidence += 30

        # 2. THREE CONSECUTIVE RED CANDLES
        if len(bars) >= 3:
            last_three_red = all(
                bars[i].get("close", 0) < bars[i].get("open", 0)
                for i in range(-3, 0)
            )
            if last_three_red:
                reversal_signals.append("three_red_candles")
                confidence += 25

        # 3. LOWER HIGHS (downtrend starting)
        if recent_highs[-1] < recent_highs[-2] < recent_highs[-3]:
            reversal_signals.append("lower_highs")
            confidence += 20

        # 4. PRICE BELOW ENTRY WITH MOMENTUM DOWN
        entry_price = self.position_entry_prices.get(symbol)
        if entry_price and current_price < entry_price:
            # Check momentum (average of last 3 candles vs previous 3)
            if len(recent_closes) >= 5:
                recent_avg = sum(recent_closes[-3:]) / 3
                prev_avg = sum(recent_closes[-5:-2]) / 3
                if recent_avg < prev_avg:
                    reversal_signals.append("momentum_down_below_entry")
                    confidence += 25

        # 5. SHARP DROP (>2% in single bar)
        if len(bars) >= 1:
            last_bar = bars[-1]
            bar_change = ((last_bar.get("close", 0) - last_bar.get("open", 0)) / last_bar.get("open", 1)) * 100
            if bar_change <= -2.0:
                reversal_signals.append("sharp_drop")
                confidence += 20

        reversal_detected = confidence >= 40

        return {
            "reversal_detected": reversal_detected,
            "confidence": min(100, confidence),
            "signals": reversal_signals,
            "recommendation": "EXIT" if reversal_detected else "HOLD"
        }

    def _cleanup_position_tracking(self, symbol: str):
        """Clean up tracking data when position is closed"""
        if symbol in self.position_entry_prices:
            del self.position_entry_prices[symbol]
        if symbol in self.position_high_watermarks:
            del self.position_high_watermarks[symbol]
        if symbol in self.profit_locked:
            del self.profit_locked[symbol]
        logger.info(f"🧹 Position tracking cleared for {symbol}")

    async def _check_positions_for_exits(self, connector, market_data) -> List[Dict]:
        """
        Check all positions for trailing stop or reversal exits.
        Returns list of sell signals.
        """
        exit_signals = []

        # USE CENTRALIZED DATA BUS FOR ALL MARKET DATA (ORDER INTEGRITY!)
        from market_data_bus import get_market_data_bus
        data_bus = get_market_data_bus()

        for symbol, position in self.current_positions.items():
            try:
                # Get current price from centralized data bus
                quote = data_bus.get_quote(symbol, require_bid_ask=True)

                if not quote:
                    logger.warning(f"No valid quote from data bus for {symbol} - skipping exit check")
                    continue

                # Use bid price for exits (what buyers are offering)
                current_price = quote.get("bid") or quote.get("ask", 0)
                if not current_price or current_price <= 0:
                    continue

                # Get entry price from position data - use avg_price (Alpaca field name)
                entry_price = float(position.get("avg_price") or position.get("avg_entry_price") or current_price)

                # Update tracking (this will initialize if needed)
                self._update_position_tracking(symbol, current_price, entry_price)

                # EMERGENCY CHECK: If already down more than initial_stop_percent, EXIT IMMEDIATELY!
                loss_pct = ((current_price - entry_price) / entry_price) * 100
                if loss_pct <= -self.config.initial_stop_percent:
                    logger.error(f"🚨 EMERGENCY STOP for {symbol}: Loss of {loss_pct:.1f}% exceeds -{self.config.initial_stop_percent}%!")
                    exit_signals.append({
                        "symbol": symbol,
                        "action": "SELL",
                        "reason": "emergency_stop",
                        "details": f"Loss of {loss_pct:.1f}% exceeds initial stop of -{self.config.initial_stop_percent}%",
                        "urgency": "HIGH",
                        "current_price": current_price,
                        "entry_price": entry_price,
                        "quantity": int(position.get("qty") or position.get("quantity", 0))
                    })
                    continue

                # Check trailing stop
                stop_check = self._check_trailing_stop(symbol, current_price)

                if stop_check["action"] == "SELL":
                    logger.warning(f"🛑 TRAILING STOP TRIGGERED for {symbol}: {stop_check['reason']} - {stop_check.get('details', '')}")
                    exit_signals.append({
                        "symbol": symbol,
                        "action": "SELL",
                        "reason": stop_check["reason"],
                        "details": stop_check.get("details", ""),
                        "urgency": stop_check.get("urgency", "MEDIUM"),
                        "current_price": current_price,
                        "entry_price": entry_price,
                        "quantity": int(position.get("qty") or position.get("quantity", 0))
                    })
                    continue

                # Check for reversal (if we have bars)
                try:
                    bars = market_data.get_bars(symbol, timeframe="1Min", limit=10)
                    if bars:
                        reversal = self._detect_reversal(symbol, current_price, bars)
                        if reversal["reversal_detected"] and reversal["confidence"] >= 60:
                            logger.warning(f"⚠️ REVERSAL DETECTED for {symbol}: {reversal['signals']} (confidence: {reversal['confidence']}%)")

                            # Only exit on reversal if we're in profit or near break-even
                            gain_pct = ((current_price - entry_price) / entry_price) * 100
                            if gain_pct >= -1.0:  # Only exit if not already at big loss
                                exit_signals.append({
                                    "symbol": symbol,
                                    "action": "SELL",
                                    "reason": "reversal_detected",
                                    "details": f"Reversal signals: {reversal['signals']}",
                                    "urgency": "MEDIUM",
                                    "current_price": current_price,
                                    "entry_price": entry_price,
                                    "quantity": int(position.get("qty", 0)),
                                    "reversal_confidence": reversal["confidence"]
                                })
                except Exception as e:
                    logger.debug(f"Could not check reversal for {symbol}: {e}")

            except Exception as e:
                logger.error(f"Error checking position {symbol}: {e}")

        return exit_signals

    def _save_trades(self):
        """Save trades to file"""
        try:
            # Load existing trades first
            all_trades = []
            if self.trades_file.exists():
                with open(self.trades_file, 'r') as f:
                    data = json.load(f)
                    all_trades = data.get("trades", [])

            # Add new trades (avoiding duplicates)
            existing_ids = {t.get("order_id") for t in all_trades}
            for trade in self.trades_today:
                if trade.order_id not in existing_ids:
                    all_trades.append(asdict(trade))

            with open(self.trades_file, 'w') as f:
                json.dump({"trades": all_trades}, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save trades: {e}")

    def _update_pdt_status(self):
        """Update PDT status from account data"""
        try:
            connector = get_alpaca_connector()
            account = connector.get_account()

            self.pdt_status.equity = account.get("equity", 0.0)
            self.pdt_status.is_pdt_flagged = account.get("pattern_day_trader", False)
            self.pdt_status.last_updated = datetime.now().isoformat()

            # Clean up old day trades (older than 5 trading days)
            cutoff = (datetime.now() - timedelta(days=5)).isoformat()
            self.pdt_status.day_trade_history = [
                t for t in self.pdt_status.day_trade_history
                if t.get("timestamp", "") > cutoff
            ]
            self.pdt_status.day_trades_count = len(self.pdt_status.day_trade_history)
            self.pdt_status.day_trades_remaining = max(0, self.config.max_day_trades - self.pdt_status.day_trades_count)

            # Check if PDT restricted (flagged but under $25k)
            if self.pdt_status.is_pdt_flagged and self.pdt_status.equity < self.pdt_status.pdt_threshold:
                self.pdt_status.is_pdt_restricted = True
            else:
                self.pdt_status.is_pdt_restricted = False

            self._save_pdt_status()

        except Exception as e:
            logger.error(f"Error updating PDT status: {e}")

    def _record_day_trade(self, symbol: str, side: str):
        """Record a day trade for PDT tracking"""
        self.pdt_status.day_trade_history.append({
            "symbol": symbol,
            "side": side,
            "timestamp": datetime.now().isoformat()
        })
        self.pdt_status.day_trades_count = len(self.pdt_status.day_trade_history)
        self.pdt_status.day_trades_remaining = max(0, self.config.max_day_trades - self.pdt_status.day_trades_count)
        self._save_pdt_status()
        logger.info(f"Day trade recorded: {symbol}. Remaining: {self.pdt_status.day_trades_remaining}")

    def _is_day_trade(self, symbol: str, side: str) -> bool:
        """
        Check if this trade would constitute a day trade
        A day trade is buying and selling (or shorting and covering) the same security on the same day
        """
        today = date.today().isoformat()

        # Check today's trades for opposite side on same symbol
        for trade in self.trades_today:
            if trade.symbol == symbol:
                trade_date = trade.timestamp[:10]
                if trade_date == today:
                    # If we bought today and now selling, it's a day trade
                    if trade.side == "BUY" and side == "SELL":
                        return True
                    # If we sold/shorted today and now buying, it's a day trade
                    if trade.side == "SELL" and side == "BUY":
                        return True

        return False

    def check_compliance(self, symbol: str, side: str, quantity: int) -> Dict:
        """
        Check if a trade complies with account rules and regulations

        Returns:
            Dict with 'allowed' bool and 'reason' if blocked
        """
        self.compliance_warnings = []
        account_type = AccountType(self.config.account_type)

        # 1. Check long-only restriction
        if self.config.long_only and side.upper() == "SELL":
            # Check if this is closing a position or opening a short
            if symbol not in self.current_positions:
                return {
                    "allowed": False,
                    "reason": "Long-only mode enabled. Short selling is not permitted.",
                    "rule": "LONG_ONLY"
                }

        # 2. Check IRA/Roth IRA restrictions (no shorting)
        if account_type in [AccountType.IRA, AccountType.ROTH_IRA]:
            if side.upper() == "SELL" and symbol not in self.current_positions:
                return {
                    "allowed": False,
                    "reason": f"{account_type.value.upper()} accounts cannot short sell.",
                    "rule": "IRA_NO_SHORT"
                }

        # 3. Check PDT rules for margin accounts
        if account_type == AccountType.MARGIN and self.config.enforce_pdt_rules:
            self._update_pdt_status()

            # If already PDT restricted, block all day trades
            if self.pdt_status.is_pdt_restricted:
                return {
                    "allowed": False,
                    "reason": "Account is PDT restricted. Equity must be >= $25,000 to day trade.",
                    "rule": "PDT_RESTRICTED"
                }

            # Check if this would be a day trade
            if self._is_day_trade(symbol, side):
                # If under $25k, check remaining day trades
                if self.pdt_status.equity < self.pdt_status.pdt_threshold:
                    if self.pdt_status.day_trades_remaining <= 0:
                        return {
                            "allowed": False,
                            "reason": f"PDT limit reached. {self.pdt_status.day_trades_count}/3 day trades used in 5-day window. Equity: ${self.pdt_status.equity:,.2f} (need $25,000+)",
                            "rule": "PDT_LIMIT"
                        }
                    else:
                        self.compliance_warnings.append(
                            f"Warning: This is a day trade. {self.pdt_status.day_trades_remaining - 1} remaining after this trade."
                        )

        # 4. Cash account settlement rules
        if account_type == AccountType.CASH and self.config.respect_settlement:
            # In cash accounts, cannot trade with unsettled funds (T+2)
            # This would require tracking settled vs unsettled cash
            self.compliance_warnings.append(
                "Cash account: Ensure funds are settled (T+2) before trading."
            )

        return {
            "allowed": True,
            "warnings": self.compliance_warnings,
            "pdt_status": {
                "day_trades_used": self.pdt_status.day_trades_count,
                "day_trades_remaining": self.pdt_status.day_trades_remaining,
                "equity": self.pdt_status.equity,
                "is_day_trade": self._is_day_trade(symbol, side)
            }
        }

    def get_account_rules(self) -> Dict:
        """Get current account type rules and restrictions"""
        account_type = AccountType(self.config.account_type)

        rules = {
            "account_type": account_type.value,
            "long_only": self.config.long_only,
            "can_short": not self.config.long_only and account_type not in [AccountType.IRA, AccountType.ROTH_IRA],
            "can_use_margin": account_type == AccountType.MARGIN,
            "pdt_applies": account_type == AccountType.MARGIN and self.config.enforce_pdt_rules,
            "settlement_rules": account_type == AccountType.CASH,
            "restrictions": []
        }

        if self.config.long_only:
            rules["restrictions"].append("Long positions only (no shorting)")

        if account_type in [AccountType.IRA, AccountType.ROTH_IRA]:
            rules["restrictions"].append("IRA account - no short selling or margin")

        if account_type == AccountType.CASH:
            rules["restrictions"].append("Cash account - T+2 settlement applies")

        if rules["pdt_applies"]:
            rules["restrictions"].append(f"PDT rules enforced - max {self.config.max_day_trades} day trades per 5 days if equity < $25k")

        return rules

    def initialize(self, config: Optional[dict] = None) -> dict:
        """Initialize the bot with optional configuration"""
        if config:
            # Trading parameters
            self.config.confidence_threshold = config.get("confidence_threshold", self.config.confidence_threshold)
            self.config.max_positions = config.get("max_positions", self.config.max_positions)
            self.config.max_daily_trades = config.get("max_daily_trades", self.config.max_daily_trades)
            self.config.position_size = config.get("position_size", self.config.position_size)
            self.config.cycle_interval_seconds = config.get("cycle_interval", self.config.cycle_interval_seconds)

            # Account type and compliance settings
            if "account_type" in config:
                self.config.account_type = config["account_type"]
            if "long_only" in config:
                self.config.long_only = bool(config["long_only"])
            if "enforce_pdt_rules" in config:
                self.config.enforce_pdt_rules = bool(config["enforce_pdt_rules"])
            if "max_day_trades" in config:
                self.config.max_day_trades = int(config["max_day_trades"])
            if "respect_settlement" in config:
                self.config.respect_settlement = bool(config["respect_settlement"])

            self._save_config()

        # Update PDT status on initialize
        self._update_pdt_status()

        self.initialized = True
        logger.info("Bot initialized with config: %s", asdict(self.config))

        return {
            "success": True,
            "message": "Bot initialized",
            "config": asdict(self.config),
            "account_rules": self.get_account_rules(),
            "pdt_status": asdict(self.pdt_status)
        }

    async def _trading_loop(self):
        """Main trading loop - runs when bot is started"""
        # Dependencies imported at module level for faster execution
        connector = get_alpaca_connector()
        predictor = get_alpaca_predictor()
        watchlist_mgr = get_watchlist_manager()
        analytics = get_portfolio_analytics()
        tracker = get_execution_tracker()

        logger.info("Bot trading loop started")

        while self.running and self.enabled:
            try:
                self.last_cycle_time = datetime.now().isoformat()

                # Check daily trade limit
                if len(self.trades_today) >= self.config.max_daily_trades:
                    logger.info(f"Daily trade limit reached: {len(self.trades_today)}/{self.config.max_daily_trades}")
                    await asyncio.sleep(self.config.cycle_interval_seconds)
                    continue

                # Get current positions
                positions = connector.get_positions()
                self.current_positions = {p["symbol"]: p for p in positions}

                # ============================================================
                # END-OF-DAY LIQUIDATION CHECK (Small Cap Momentum Strategy)
                # Close all positions before market close to avoid overnight gap risk
                # ============================================================
                if self._is_eod_liquidation_time() and positions:
                    logger.warning("🌙 EOD LIQUIDATION TIME - Closing all positions to avoid overnight risk")
                    liquidation_results = await self._liquidate_all_positions(connector, reason="eod_close")
                    if liquidation_results:
                        logger.info(f"EOD Liquidation: {len(liquidation_results)} positions closed")
                        for result in liquidation_results:
                            logger.info(f"  - {result['symbol']}: {result['status']}")
                    # After liquidation, continue loop to let orders fill
                    await asyncio.sleep(self.config.cycle_interval_seconds)
                    continue

                # ============================================================
                # CHECK POSITIONS FOR TRAILING STOP / REVERSAL EXITS (PRIORITY)
                # ============================================================
                if self.config.trailing_stop_enabled and positions:
                    market_data = get_alpaca_market_data()
                    exit_signals = await self._check_positions_for_exits(connector, market_data)

                    for exit_sig in exit_signals:
                        symbol = exit_sig["symbol"]

                        # Check rate limiting
                        if not self._can_place_order(symbol):
                            logger.warning(f"Exit order rate-limited for {symbol}")
                            continue

                        try:
                            # ============================================================
                            # USE SMART ORDER WITH EMERGENCY FLAG FOR STOP LOSS EXITS
                            # - HIGH urgency (emergency stop): emergency=True (use BID, market during regular hours)
                            # - Normal urgency (take profit, trailing stop): emergency=False (use ASK for better exit)
                            # - All orders are LIMIT with extended_hours=True (never get stuck)
                            # ============================================================
                            is_emergency = exit_sig["urgency"] == "HIGH"
                            order = connector.place_smart_order(
                                symbol=symbol,
                                quantity=exit_sig["quantity"],
                                side="SELL",
                                emergency=is_emergency
                            )

                            if order and order.get("order_id"):
                                self._record_order_placed(symbol)
                                logger.info(f"🛑 EXIT ORDER PLACED: SELL {exit_sig['quantity']} {symbol} - {exit_sig['reason']}")

                                # Record trade
                                trade = BotTrade(
                                    symbol=symbol,
                                    side="SELL",
                                    quantity=exit_sig["quantity"],
                                    price=exit_sig["current_price"],
                                    confidence=0.99,
                                    signal=f"EXIT_{exit_sig['reason'].upper()}",
                                    order_id=order["order_id"],
                                    timestamp=datetime.now().isoformat()
                                )
                                self.trades_today.append(trade)
                                self._save_trades()

                                # Cleanup tracking
                                self._cleanup_position_tracking(symbol)

                        except Exception as e:
                            logger.error(f"Exit order failed for {symbol}: {e}")

                if len(positions) >= self.config.max_positions:
                    logger.info(f"Max positions reached: {len(positions)}/{self.config.max_positions}")
                    await asyncio.sleep(self.config.cycle_interval_seconds)
                    continue

                # Get watchlist symbols
                try:
                    watchlist = watchlist_mgr.get_default_watchlist()
                    symbols = watchlist.get("symbols", [])
                except:
                    symbols = ["SPY", "QQQ", "AAPL", "MSFT", "GOOGL"]

                # Filter out symbols we already hold
                held_symbols = set(self.current_positions.keys())
                available_symbols = [s for s in symbols if s not in held_symbols]

                if not available_symbols:
                    logger.info("All watchlist symbols already held")
                    await asyncio.sleep(self.config.cycle_interval_seconds)
                    continue

                # ============================================================
                # CHECK FOR BREAKING NEWS TRIGGERS (Warrior Trading Catalyst)
                # ============================================================
                news_signals = []
                if self.config.news_trigger_enabled and self.sentiment_analyzer:
                    try:
                        news_signals = await self.check_breaking_news(available_symbols)
                        for ns in news_signals:
                            # Check compliance for news-triggered trades
                            compliance = self.check_compliance(ns["symbol"], ns["action"], self.config.position_size)
                            if compliance["allowed"]:
                                ns["compliance"] = compliance
                                logger.info(f"📰 NEWS SIGNAL: {ns['symbol']} - {ns['action']} ({ns['confidence']*100:.1f}%)")
                            else:
                                logger.warning(f"News signal blocked for {ns['symbol']}: {compliance.get('reason')}")
                                news_signals.remove(ns)
                    except Exception as e:
                        logger.warning(f"News trigger check failed: {e}")

                # ============================================================
                # GET AI PREDICTIONS FOR SYMBOLS
                # ============================================================
                signals = []
                for symbol in available_symbols:
                    try:
                        prediction = predictor.predict(symbol)
                        confidence = prediction.get("confidence", 0)
                        action = prediction.get("action", "HOLD")
                        signal = prediction.get("signal", "NEUTRAL")

                        self.last_signals[symbol] = {
                            "signal": signal,
                            "action": action,
                            "confidence": confidence,
                            "timestamp": datetime.now().isoformat()
                        }

                        if confidence >= self.config.confidence_threshold and action in ["BUY", "SELL"]:
                            # Check compliance before adding to signals
                            compliance = self.check_compliance(symbol, action, self.config.position_size)
                            if compliance["allowed"]:
                                signals.append({
                                    "symbol": symbol,
                                    "confidence": confidence,
                                    "action": action,
                                    "signal": signal,
                                    "compliance": compliance,
                                    "trigger": "ai_prediction"
                                })
                                logger.info(f"Signal: {symbol} - {signal} ({confidence*100:.1f}% confidence)")
                                if compliance.get("warnings"):
                                    for warning in compliance["warnings"]:
                                        logger.warning(warning)
                            else:
                                logger.warning(f"Signal blocked for {symbol}: {compliance.get('reason')} [Rule: {compliance.get('rule')}]")
                    except Exception as e:
                        logger.warning(f"Prediction failed for {symbol}: {e}")

                # ============================================================
                # BLOCK NEW ENTRIES NEAR MARKET CLOSE (Small Cap Momentum)
                # Stop opening new positions late in day to avoid overnight risk
                # ============================================================
                if self._should_block_new_entries():
                    now_et = self._get_current_et_time()
                    logger.info(f"⏰ Blocking new entries after {self.config.no_new_entries_after} ET (current: {now_et.strftime('%H:%M')})")
                    # Still process exit signals (trailing stops, take profit) but skip new entries
                    await asyncio.sleep(self.config.cycle_interval_seconds)
                    continue

                # ============================================================
                # COMBINE AI SIGNALS + NEWS TRIGGERS (News gets priority)
                # ============================================================
                all_signals = news_signals + signals

                if not all_signals:
                    logger.info("No actionable signals found")
                    await asyncio.sleep(self.config.cycle_interval_seconds)
                    continue

                # Sort by confidence and take the best (news signals typically have boosted confidence)
                all_signals.sort(key=lambda x: x["confidence"], reverse=True)
                best = all_signals[0]

                trigger_type = best.get("trigger", "unknown")
                if trigger_type == "breaking_news":
                    logger.info(f"📰 BEST SIGNAL (NEWS): {best['symbol']} - {best['action']} ({best['confidence']*100:.1f}%)")
                    logger.info(f"   Headline: {best.get('news_headline', 'N/A')[:60]}...")
                else:
                    logger.info(f"Best signal (AI): {best['symbol']} - {best['action']} ({best['confidence']*100:.1f}%)")

                # Execute the trade (with rate limiting)
                try:
                    # Check rate limiting before placing order
                    if not self._can_place_order(best["symbol"]):
                        logger.info(f"Order rate-limited for {best['symbol']}, skipping this cycle")
                        await asyncio.sleep(self.config.cycle_interval_seconds)
                        continue

                    # ============================================================
                    # USE SMART ORDER - MOMENTUM STRATEGY
                    # - ALL orders are LIMIT with extended_hours=True
                    # - BUY: Uses ASK price (hit the ask, ride momentum)
                    # - SELL: Uses ASK price (better exit while momentum carries)
                    # - Pricing handled inside place_smart_order from Schwab data bus
                    # ============================================================
                    from market_data_bus import get_market_data_bus
                    data_bus = get_market_data_bus()

                    # Get quote for logging/tracking (order uses quote internally)
                    quote = data_bus.get_quote(best["symbol"])
                    if not quote:
                        logger.error(f"No valid quote from data bus for {best['symbol']} - SKIPPING ORDER")
                        await asyncio.sleep(self.config.cycle_interval_seconds)
                        continue

                    market_price = quote.get("ask") if best["action"].upper() == "BUY" else quote.get("bid")
                    if not market_price or market_price < 0.50:
                        logger.error(f"Invalid price ${market_price} for {best['symbol']} - SKIPPING ORDER")
                        await asyncio.sleep(self.config.cycle_interval_seconds)
                        continue

                    logger.info(f"Entry order for {best['symbol']}: bid=${quote.get('bid', 0):.2f} ask=${quote.get('ask', 0):.2f} (source: {quote.get('source', 'schwab')})")

                    # Place smart order - handles pricing strategy internally
                    order = connector.place_smart_order(
                        symbol=best["symbol"],
                        quantity=self.config.position_size,
                        side=best["action"].upper(),
                        emergency=False  # Normal entry, not emergency
                    )

                    # Get limit price from order response for tracking
                    limit_price = order.get("limit_price", market_price)

                    if order and order.get("order_id"):
                        order_id = order["order_id"]

                        # Record order for rate limiting
                        self._record_order_placed(best["symbol"])

                        # Track execution
                        tracker.start_order_tracking(
                            order_id=order_id,
                            symbol=best["symbol"],
                            side=best["action"],
                            quantity=self.config.position_size,
                            order_type="LMT",
                            market_price=market_price,
                            limit_price=limit_price
                        )

                        # Record trade
                        trade = BotTrade(
                            symbol=best["symbol"],
                            side=best["action"],
                            quantity=self.config.position_size,
                            price=market_price,
                            confidence=best["confidence"],
                            signal=best["signal"],
                            order_id=order_id,
                            timestamp=datetime.now().isoformat()
                        )
                        self.trades_today.append(trade)
                        self._save_trades()

                        # Record in analytics
                        analytics.record_trade_entry(
                            symbol=best["symbol"],
                            side=best["action"],
                            quantity=self.config.position_size,
                            price=market_price,
                            order_id=order_id,
                            ai_signal=best["signal"],
                            ai_confidence=best["confidence"]
                        )

                        # Record day trade if applicable (for PDT tracking)
                        if best.get("compliance", {}).get("pdt_status", {}).get("is_day_trade"):
                            self._record_day_trade(best["symbol"], best["action"])

                        logger.info(f"Trade executed: {best['action']} {self.config.position_size} {best['symbol']} @ ${limit_price:.2f} LIMIT (Order ID: {order_id})")

                except Exception as e:
                    logger.error(f"Trade execution failed: {e}")

                # Wait for next cycle
                await asyncio.sleep(self.config.cycle_interval_seconds)

            except asyncio.CancelledError:
                logger.info("Trading loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")

                # MORPHIC SELF-HEALING: Attempt to recover from errors
                if self.config.ai_self_healing and self.ai_intelligence:
                    try:
                        healing_result = self.ai_intelligence.self_heal(e, {
                            "phase": "trading_loop",
                            "last_cycle": self.last_cycle_time,
                            "trades_today": len(self.trades_today)
                        })

                        if healing_result.get("healed"):
                            retry_delay = healing_result.get("retry_delay", 30)
                            logger.info(f"🔧 Self-healed. Retrying in {retry_delay}s")
                            await asyncio.sleep(retry_delay)
                            continue
                        else:
                            logger.warning(f"🔧 Self-healing needs attention: {healing_result.get('recommendations')}")
                    except Exception as heal_error:
                        logger.error(f"Self-healing failed: {heal_error}")

                await asyncio.sleep(60)  # Wait before retry

        logger.info("Bot trading loop stopped")

    def start(self) -> dict:
        """Start the trading bot"""
        if not self.initialized:
            self.initialize()

        if self.running:
            return {"success": False, "message": "Bot is already running"}

        self.running = True
        self.enabled = True

        # Start the trading loop in background
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        self._loop = loop

        # Create task for trading loop
        if loop.is_running():
            self._bot_task = asyncio.create_task(self._trading_loop())
        else:
            # Start in a thread if loop isn't running
            def run_bot():
                asyncio.run(self._trading_loop())

            self._bot_thread = threading.Thread(target=run_bot, daemon=True)
            self._bot_thread.start()

        logger.info("Bot started")

        return {
            "success": True,
            "message": "Bot started",
            "status": "running"
        }

    def stop(self) -> dict:
        """Stop the trading bot"""
        if not self.running:
            return {"success": False, "message": "Bot is not running"}

        self.running = False
        self.enabled = False

        if self._bot_task:
            self._bot_task.cancel()

        logger.info("Bot stopped")

        return {
            "success": True,
            "message": "Bot stopped",
            "status": "stopped"
        }

    def enable(self) -> dict:
        """Enable trading (bot must be started)"""
        self.enabled = True
        self.config.trading_enabled = True
        self._save_config()

        return {"success": True, "message": "Trading enabled"}

    def disable(self) -> dict:
        """Disable trading (bot keeps running but doesn't trade)"""
        self.enabled = False
        self.config.trading_enabled = False
        self._save_config()

        return {"success": True, "message": "Trading disabled"}

    def get_status(self) -> dict:
        """Get current bot status including compliance info"""
        # Update PDT status before returning
        self._update_pdt_status()

        # Get Next-Gen AI module status
        nextgen_ai_status = {
            "regime_classifier": self.regime_classifier is not None,
            "narrative_generator": self.narrative_generator is not None,
            "kelly_sizer": self.kelly_sizer is not None,
            "brier_tracker": self.brier_tracker is not None,
            "current_regime": self.current_regime.value if self.current_regime else "unknown"
        }

        # Get model health summary if available
        model_health = self.get_model_health() if self.brier_tracker else {}

        return {
            "initialized": self.initialized,
            "running": self.running,
            "enabled": self.enabled,
            "status": "running" if self.running else "stopped",
            "trades_today": len(self.trades_today),
            "max_daily_trades": self.config.max_daily_trades,
            "pnl_today": self.pnl_today,
            "positions_count": len(self.current_positions),
            "max_positions": self.config.max_positions,
            "last_cycle": self.last_cycle_time,
            "config": asdict(self.config),
            # Compliance info
            "account_rules": self.get_account_rules(),
            "pdt_status": {
                "day_trades_count": self.pdt_status.day_trades_count,
                "day_trades_remaining": self.pdt_status.day_trades_remaining,
                "is_pdt_flagged": self.pdt_status.is_pdt_flagged,
                "is_pdt_restricted": self.pdt_status.is_pdt_restricted,
                "equity": self.pdt_status.equity,
                "pdt_threshold": self.pdt_status.pdt_threshold,
                "last_updated": self.pdt_status.last_updated
            },
            "compliance_warnings": self.compliance_warnings,
            # Next-Gen AI Logic Blueprint
            "nextgen_ai": nextgen_ai_status,
            "model_health": model_health
        }

    def get_trades(self) -> List[dict]:
        """Get today's bot trades"""
        return [asdict(t) for t in self.trades_today]

    def get_signals(self) -> dict:
        """Get last signals for all scanned symbols"""
        return self.last_signals

    def update_config(self, config: dict) -> dict:
        """Update bot configuration including compliance settings"""
        # Trading parameters
        if "confidence_threshold" in config:
            self.config.confidence_threshold = float(config["confidence_threshold"])
        if "max_positions" in config:
            self.config.max_positions = int(config["max_positions"])
        if "max_daily_trades" in config:
            self.config.max_daily_trades = int(config["max_daily_trades"])
        if "position_size" in config:
            self.config.position_size = int(config["position_size"])
        if "cycle_interval" in config:
            self.config.cycle_interval_seconds = int(config["cycle_interval"])

        # Account type and compliance settings
        if "account_type" in config:
            valid_types = ["cash", "margin", "ira", "roth_ira"]
            if config["account_type"] in valid_types:
                self.config.account_type = config["account_type"]
            else:
                return {"success": False, "message": f"Invalid account_type. Must be one of: {valid_types}"}
        if "long_only" in config:
            self.config.long_only = bool(config["long_only"])
        if "enforce_pdt_rules" in config:
            self.config.enforce_pdt_rules = bool(config["enforce_pdt_rules"])
        if "max_day_trades" in config:
            self.config.max_day_trades = int(config["max_day_trades"])
        if "respect_settlement" in config:
            self.config.respect_settlement = bool(config["respect_settlement"])
        if "settled_cash_only" in config:
            self.config.settled_cash_only = bool(config["settled_cash_only"])

        self._save_config()

        return {
            "success": True,
            "message": "Configuration updated",
            "config": asdict(self.config),
            "account_rules": self.get_account_rules()
        }


# Singleton instance
_bot_manager: Optional[BotManager] = None


def get_bot_manager() -> BotManager:
    """Get or create the bot manager singleton"""
    global _bot_manager
    if _bot_manager is None:
        _bot_manager = BotManager()
    return _bot_manager
