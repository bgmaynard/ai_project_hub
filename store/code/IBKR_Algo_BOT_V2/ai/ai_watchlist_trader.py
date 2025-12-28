"""
AI Watchlist Trader - Automated Analysis and Trade Execution
Monitors watchlist symbols, analyzes with AI, queues opportunities, executes trades
"""
import logging
import json
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum
import os

logger = logging.getLogger(__name__)


class TradeSignal(Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


class OpportunityStatus(Enum):
    PENDING = "pending"           # Waiting for better entry
    READY = "ready"               # Ready to execute
    EXECUTED = "executed"         # Trade placed
    EXPIRED = "expired"           # Opportunity expired
    CANCELLED = "cancelled"       # Manually cancelled


@dataclass
class TradeOpportunity:
    """Represents a potential trade opportunity"""
    id: str
    symbol: str
    signal: str
    confidence: float
    entry_price: float
    target_price: float
    stop_loss: float
    position_size: int
    strategy: str
    reason: str
    created_at: str
    expires_at: str
    status: str = "pending"
    executed_at: Optional[str] = None
    order_id: Optional[str] = None
    priority: int = 5  # 1-10, higher = more urgent


@dataclass
class SymbolAnalysis:
    """Complete AI analysis for a symbol"""
    symbol: str
    timestamp: str
    ai_signal: str
    ai_confidence: float
    ai_action: str
    current_price: float
    predicted_direction: str
    momentum_score: float
    volume_score: float
    technical_score: float
    news_sentiment: float
    overall_score: float
    strategies_triggered: List[str]
    trade_recommendation: Optional[TradeOpportunity]


class AIWatchlistTrader:
    """
    AI-powered watchlist monitoring and trade execution system

    Features:
    - Monitors watchlist for new symbols
    - Runs comprehensive AI analysis on each symbol
    - Queues trade opportunities based on strategy rules
    - Executes trades when conditions are optimal
    - Tracks performance and learns from outcomes
    """

    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        self.opportunity_queue: List[TradeOpportunity] = []
        self.analyzed_symbols: Dict[str, SymbolAnalysis] = {}
        self.execution_log: List[Dict] = []
        self._running = False
        self._monitor_thread = None
        self._last_watchlist_hash = None
        self._opportunity_counter = 0

        # Paths
        self.data_dir = Path("store/ai_trader")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.queue_file = self.data_dir / "opportunity_queue.json"
        self.analysis_file = self.data_dir / "symbol_analysis.json"
        self.execution_file = self.data_dir / "execution_log.json"

        # Load saved state
        self._load_state()

        logger.info(f"AI Watchlist Trader initialized: {self.config}")

    def _default_config(self) -> Dict:
        """Default trading configuration"""
        return {
            # Trading enabled
            "enabled": os.getenv("AI_TRADER_ENABLED", "false").lower() == "true",
            "paper_mode": os.getenv("AI_TRADER_PAPER", "true").lower() == "true",

            # AI Thresholds
            "min_confidence": float(os.getenv("AI_TRADER_MIN_CONFIDENCE", "0.60")),
            "strong_signal_threshold": float(os.getenv("AI_TRADER_STRONG_THRESHOLD", "0.75")),

            # Position Sizing
            "max_position_value": float(os.getenv("AI_TRADER_MAX_POSITION", "1000")),
            "position_size_pct": float(os.getenv("AI_TRADER_POSITION_PCT", "0.05")),  # 5% of capital

            # Risk Management
            "stop_loss_pct": float(os.getenv("AI_TRADER_STOP_LOSS", "0.03")),  # 3%
            "take_profit_pct": float(os.getenv("AI_TRADER_TAKE_PROFIT", "0.06")),  # 6%
            "max_daily_trades": int(os.getenv("AI_TRADER_MAX_TRADES", "10")),
            "max_daily_loss": float(os.getenv("AI_TRADER_MAX_LOSS", "500")),
            "max_open_positions": int(os.getenv("AI_TRADER_MAX_POSITIONS", "5")),

            # Timing
            "opportunity_expiry_minutes": 60,
            "analysis_interval_seconds": 30,
            "monitor_interval_seconds": 5,

            # Strategies to use
            "active_strategies": ["momentum", "ai_signal", "breakout", "mean_reversion"],
        }

    def _load_state(self):
        """Load saved state from files"""
        try:
            if self.queue_file.exists():
                with open(self.queue_file, 'r') as f:
                    data = json.load(f)
                    self.opportunity_queue = [
                        TradeOpportunity(**opp) for opp in data
                        if opp.get('status') in ['pending', 'ready']
                    ]
        except Exception as e:
            logger.warning(f"Could not load opportunity queue: {e}")

        try:
            if self.execution_file.exists():
                with open(self.execution_file, 'r') as f:
                    self.execution_log = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load execution log: {e}")

    def _save_state(self):
        """Save current state to files"""
        try:
            with open(self.queue_file, 'w') as f:
                json.dump([asdict(opp) for opp in self.opportunity_queue], f, indent=2)
        except Exception as e:
            logger.error(f"Could not save opportunity queue: {e}")

        try:
            with open(self.execution_file, 'w') as f:
                json.dump(self.execution_log[-1000:], f, indent=2)  # Keep last 1000
        except Exception as e:
            logger.error(f"Could not save execution log: {e}")

    def analyze_symbol(self, symbol: str) -> SymbolAnalysis:
        """
        Run comprehensive AI analysis on a symbol

        Returns complete analysis with trade recommendation if applicable
        """
        symbol = symbol.upper()
        logger.info(f"[AI TRADER] Analyzing {symbol}...")

        try:
            # Get AI prediction
            from ai.ai_predictor import get_predictor
            predictor = get_predictor()

            ai_result = {}
            try:
                if predictor.model is not None:
                    ai_result = predictor.predict(symbol)
                else:
                    ai_result = {"signal": "NEUTRAL", "confidence": 0, "action": "HOLD"}
            except Exception as e:
                logger.warning(f"AI prediction failed for {symbol}: {e}")
                ai_result = {"signal": "NEUTRAL", "confidence": 0, "action": "HOLD", "error": str(e)}

            # Get current price
            current_price = ai_result.get("current_price", 0)
            if current_price == 0:
                try:
                    import yfinance as yf
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="1d")
                    if not hist.empty:
                        current_price = float(hist['Close'].iloc[-1])
                except:
                    pass

            # Calculate component scores
            momentum_score = self._calculate_momentum_score(symbol)
            volume_score = self._calculate_volume_score(symbol)
            technical_score = self._calculate_technical_score(symbol)
            news_sentiment = self._get_news_sentiment(symbol)

            # Overall weighted score
            ai_conf = ai_result.get("confidence", 0)
            overall_score = (
                ai_conf * 0.35 +
                momentum_score * 0.25 +
                technical_score * 0.20 +
                volume_score * 0.10 +
                news_sentiment * 0.10
            )

            # Determine which strategies are triggered
            strategies_triggered = self._check_strategies(
                symbol, ai_result, momentum_score, technical_score, volume_score
            )

            # Create trade recommendation if criteria met
            trade_rec = None
            if strategies_triggered and overall_score >= self.config["min_confidence"]:
                trade_rec = self._create_opportunity(
                    symbol, ai_result, current_price, overall_score, strategies_triggered
                )

            # Determine direction
            ai_signal = ai_result.get("signal", "NEUTRAL")
            if "BULLISH" in ai_signal or ai_result.get("action") == "BUY":
                direction = "UP"
            elif "BEARISH" in ai_signal or ai_result.get("action") == "SELL":
                direction = "DOWN"
            else:
                direction = "NEUTRAL"

            analysis = SymbolAnalysis(
                symbol=symbol,
                timestamp=datetime.now().isoformat(),
                ai_signal=ai_signal,
                ai_confidence=ai_conf,
                ai_action=ai_result.get("action", "HOLD"),
                current_price=current_price,
                predicted_direction=direction,
                momentum_score=momentum_score,
                volume_score=volume_score,
                technical_score=technical_score,
                news_sentiment=news_sentiment,
                overall_score=overall_score,
                strategies_triggered=strategies_triggered,
                trade_recommendation=trade_rec
            )

            # Cache analysis
            self.analyzed_symbols[symbol] = analysis

            logger.info(f"[AI TRADER] {symbol}: Score={overall_score:.2f}, Signal={ai_signal}, Strategies={strategies_triggered}")

            return analysis

        except Exception as e:
            logger.error(f"[AI TRADER] Analysis failed for {symbol}: {e}")
            return SymbolAnalysis(
                symbol=symbol,
                timestamp=datetime.now().isoformat(),
                ai_signal="ERROR",
                ai_confidence=0,
                ai_action="HOLD",
                current_price=0,
                predicted_direction="NEUTRAL",
                momentum_score=0,
                volume_score=0,
                technical_score=0,
                news_sentiment=0,
                overall_score=0,
                strategies_triggered=[],
                trade_recommendation=None
            )

    def _calculate_momentum_score(self, symbol: str) -> float:
        """Calculate momentum score (0-1)"""
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1mo")
            if hist.empty or len(hist) < 5:
                return 0.5

            # Price momentum
            returns = hist['Close'].pct_change()
            recent_return = returns.iloc[-5:].mean()

            # Normalize to 0-1 scale
            score = 0.5 + (recent_return * 10)  # Scale factor
            return max(0, min(1, score))
        except:
            return 0.5

    def _calculate_volume_score(self, symbol: str) -> float:
        """Calculate volume score (0-1)"""
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1mo")
            if hist.empty or len(hist) < 10:
                return 0.5

            # Volume relative to average
            avg_volume = hist['Volume'].iloc[:-5].mean()
            recent_volume = hist['Volume'].iloc[-5:].mean()

            if avg_volume > 0:
                volume_ratio = recent_volume / avg_volume
                score = min(1, volume_ratio / 2)  # Cap at 2x average = 1.0
                return score
            return 0.5
        except:
            return 0.5

    def _calculate_technical_score(self, symbol: str) -> float:
        """Calculate technical indicators score (0-1)"""
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="3mo")
            if hist.empty or len(hist) < 50:
                return 0.5

            close = hist['Close']

            # Simple technical signals
            signals = []

            # Price above 20-day MA
            ma20 = close.rolling(20).mean()
            if close.iloc[-1] > ma20.iloc[-1]:
                signals.append(1)
            else:
                signals.append(0)

            # Price above 50-day MA
            ma50 = close.rolling(50).mean()
            if len(ma50.dropna()) > 0 and close.iloc[-1] > ma50.iloc[-1]:
                signals.append(1)
            else:
                signals.append(0)

            # RSI not overbought/oversold
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1]

            if 30 < current_rsi < 70:
                signals.append(1)
            elif current_rsi <= 30:  # Oversold = potential buy
                signals.append(0.8)
            else:  # Overbought
                signals.append(0.2)

            return sum(signals) / len(signals)
        except:
            return 0.5

    def _get_news_sentiment(self, symbol: str) -> float:
        """Get news sentiment score (0-1, 0.5=neutral)"""
        try:
            # Try to get from news feed monitor
            from ai.news_feed_monitor import get_news_monitor
            monitor = get_news_monitor()
            sentiment = monitor.get_symbol_sentiment(symbol)
            if sentiment:
                # Convert -1 to 1 scale to 0-1
                return (sentiment + 1) / 2
        except:
            pass
        return 0.5  # Neutral default

    def _check_strategies(self, symbol: str, ai_result: Dict,
                         momentum: float, technical: float, volume: float) -> List[str]:
        """Check which trading strategies are triggered"""
        triggered = []
        ai_signal = ai_result.get("signal", "NEUTRAL")
        ai_conf = ai_result.get("confidence", 0)
        action = ai_result.get("action", "HOLD")

        # AI Signal Strategy
        if "ai_signal" in self.config["active_strategies"]:
            if ai_conf >= self.config["min_confidence"] and action in ["BUY", "SELL"]:
                triggered.append(f"ai_signal:{action}")

        # Momentum Strategy
        if "momentum" in self.config["active_strategies"]:
            if momentum >= 0.65 and action == "BUY":
                triggered.append("momentum:bullish")
            elif momentum <= 0.35 and action == "SELL":
                triggered.append("momentum:bearish")

        # Breakout Strategy
        if "breakout" in self.config["active_strategies"]:
            if volume >= 0.7 and momentum >= 0.6 and action == "BUY":
                triggered.append("breakout:volume_surge")

        # Mean Reversion Strategy
        if "mean_reversion" in self.config["active_strategies"]:
            if technical <= 0.3 and ai_conf >= 0.5:  # Oversold
                triggered.append("mean_reversion:oversold")

        # Strong Signal (multiple confirmations)
        if len(triggered) >= 2 and ai_conf >= self.config["strong_signal_threshold"]:
            triggered.append("multi_confirm:strong")

        return triggered

    def _create_opportunity(self, symbol: str, ai_result: Dict,
                           current_price: float, score: float,
                           strategies: List[str]) -> TradeOpportunity:
        """Create a trade opportunity from analysis"""
        self._opportunity_counter += 1
        opp_id = f"OPP-{datetime.now().strftime('%Y%m%d%H%M%S')}-{self._opportunity_counter}"

        action = ai_result.get("action", "HOLD")
        signal = "STRONG_BUY" if score >= self.config["strong_signal_threshold"] else "BUY"
        if action == "SELL":
            signal = "STRONG_SELL" if score >= self.config["strong_signal_threshold"] else "SELL"

        # Calculate position size
        position_value = min(
            self.config["max_position_value"],
            10000 * self.config["position_size_pct"]  # Assume $10k capital
        )
        position_size = max(1, int(position_value / current_price)) if current_price > 0 else 1

        # Calculate targets
        if action == "BUY":
            target_price = current_price * (1 + self.config["take_profit_pct"])
            stop_loss = current_price * (1 - self.config["stop_loss_pct"])
        else:  # SELL/SHORT
            target_price = current_price * (1 - self.config["take_profit_pct"])
            stop_loss = current_price * (1 + self.config["stop_loss_pct"])

        expiry = datetime.now() + timedelta(minutes=self.config["opportunity_expiry_minutes"])

        # Priority based on score
        priority = min(10, max(1, int(score * 10)))

        opportunity = TradeOpportunity(
            id=opp_id,
            symbol=symbol,
            signal=signal,
            confidence=score,
            entry_price=current_price,
            target_price=round(target_price, 2),
            stop_loss=round(stop_loss, 2),
            position_size=position_size,
            strategy=", ".join(strategies),
            reason=f"AI Score: {score:.2f}, Strategies: {strategies}",
            created_at=datetime.now().isoformat(),
            expires_at=expiry.isoformat(),
            status="pending",
            priority=priority
        )

        return opportunity

    def add_to_queue(self, opportunity: TradeOpportunity):
        """Add opportunity to the trade queue"""
        # Check for duplicates
        existing = [o for o in self.opportunity_queue
                   if o.symbol == opportunity.symbol and o.status == "pending"]

        if existing:
            # Update existing if new one has higher priority
            if opportunity.priority > existing[0].priority:
                self.opportunity_queue.remove(existing[0])
            else:
                logger.info(f"[AI TRADER] {opportunity.symbol} already in queue")
                return

        self.opportunity_queue.append(opportunity)
        self.opportunity_queue.sort(key=lambda x: -x.priority)  # Sort by priority desc
        self._save_state()

        logger.info(f"[AI TRADER] Added {opportunity.symbol} to queue: {opportunity.signal} @ ${opportunity.entry_price}")

    def process_queue(self) -> List[Dict]:
        """Process the opportunity queue and execute ready trades"""
        executed = []
        now = datetime.now()

        for opp in self.opportunity_queue[:]:
            # Check expiry
            expires_at = datetime.fromisoformat(opp.expires_at)
            if now > expires_at:
                opp.status = "expired"
                logger.info(f"[AI TRADER] Opportunity expired: {opp.symbol}")
                continue

            if opp.status != "pending":
                continue

            # Check if we should execute
            if self._should_execute(opp):
                result = self._execute_trade(opp)
                if result.get("success"):
                    opp.status = "executed"
                    opp.executed_at = now.isoformat()
                    opp.order_id = result.get("order_id")
                    executed.append(result)

        # Clean up old opportunities
        self.opportunity_queue = [
            o for o in self.opportunity_queue
            if o.status in ["pending", "ready"] or
            (o.status == "executed" and
             datetime.fromisoformat(o.executed_at) > now - timedelta(hours=24))
        ]

        self._save_state()
        return executed

    def _should_execute(self, opp: TradeOpportunity) -> bool:
        """Determine if an opportunity should be executed now"""
        if not self.config["enabled"]:
            return False

        # Check daily limits
        today_trades = [
            e for e in self.execution_log
            if datetime.fromisoformat(e['timestamp']).date() == datetime.now().date()
        ]

        if len(today_trades) >= self.config["max_daily_trades"]:
            logger.warning("[AI TRADER] Daily trade limit reached")
            return False

        # Check daily loss
        today_pnl = sum(e.get('pnl', 0) for e in today_trades)
        if today_pnl < -self.config["max_daily_loss"]:
            logger.warning("[AI TRADER] Daily loss limit reached")
            return False

        # High priority opportunities execute immediately
        if opp.priority >= 8:
            return True

        # Medium priority - check for favorable entry
        if opp.priority >= 5:
            # TODO: Add price check against entry_price
            return True

        return False

    def _execute_trade(self, opp: TradeOpportunity) -> Dict:
        """
        Execute a trade from an opportunity.

        GATING ENFORCEMENT: All trades MUST go through Signal Gating Engine.
        """
        logger.info(f"[AI TRADER] Executing: {opp.signal} {opp.symbol} x{opp.position_size}")

        # GATING ENFORCEMENT: Route through Signal Gating Engine first
        try:
            from ai.gated_trading import get_gated_trading_manager
            manager = get_gated_trading_manager()

            approved, exec_request, reason = manager.gate_trade_attempt(
                symbol=opp.symbol,
                trigger_type="ai_watchlist",
                quote={"price": opp.entry_price}
            )

            if not approved:
                logger.info(f"[AI TRADER] GATING VETOED: {opp.symbol} - {reason}")
                result = {
                    "success": False,
                    "gating_vetoed": True,
                    "reason": reason,
                    "symbol": opp.symbol,
                    "timestamp": datetime.now().isoformat()
                }
                self.execution_log.append(result)
                self._save_state()
                return result

            gating_token = f"GATED_{opp.symbol}_{datetime.now().strftime('%H%M%S')}"
            logger.info(f"[AI TRADER] GATING APPROVED: {opp.symbol} - token={gating_token}")

        except Exception as e:
            # Fail-closed: on gating error, reject trade
            logger.error(f"[AI TRADER] GATING ERROR (fail-closed): {opp.symbol} - {e}")
            result = {
                "success": False,
                "gating_error": True,
                "error": str(e),
                "symbol": opp.symbol,
                "timestamp": datetime.now().isoformat()
            }
            self.execution_log.append(result)
            self._save_state()
            return result

        # Gating approved - proceed with execution
        if self.config["paper_mode"]:
            # Paper trading - simulate execution
            result = {
                "success": True,
                "paper_trade": True,
                "gating_token": gating_token,
                "symbol": opp.symbol,
                "action": "BUY" if "BUY" in opp.signal else "SELL",
                "quantity": opp.position_size,
                "price": opp.entry_price,
                "order_id": f"PAPER-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                "timestamp": datetime.now().isoformat(),
                "pnl": 0
            }
        else:
            # Real trading
            try:
                from unified_broker import get_broker
                broker = get_broker()

                action = "BUY" if "BUY" in opp.signal else "SELL"
                order_result = broker.place_limit_order(
                    symbol=opp.symbol,
                    qty=opp.position_size,
                    limit_price=opp.entry_price,
                    side=action
                )

                result = {
                    "success": True,
                    "paper_trade": False,
                    "gating_token": gating_token,
                    "symbol": opp.symbol,
                    "action": action,
                    "quantity": opp.position_size,
                    "price": opp.entry_price,
                    "order_id": order_result.get("order_id"),
                    "timestamp": datetime.now().isoformat(),
                    "pnl": 0
                }
            except Exception as e:
                logger.error(f"[AI TRADER] Execution failed: {e}")
                result = {
                    "success": False,
                    "gating_token": gating_token,
                    "error": str(e),
                    "symbol": opp.symbol,
                    "timestamp": datetime.now().isoformat()
                }

        self.execution_log.append(result)
        self._save_state()

        # Log to Trade Journal
        self._log_to_journal(opp, result)

        return result

    def _log_to_journal(self, opp: TradeOpportunity, result: Dict):
        """Log completed trade to the Trade Journal"""
        try:
            from ai.trade_journal import get_trade_journal, TradeEntry
            journal = get_trade_journal()

            # Get the analysis data if available
            analysis = self.analyzed_symbols.get(opp.symbol)

            trade = TradeEntry(
                id=result.get("order_id", f"{opp.symbol}-{datetime.now().strftime('%Y%m%d%H%M%S')}"),
                symbol=opp.symbol,
                trade_date=datetime.now().strftime('%Y-%m-%d'),
                entry_time=result.get("timestamp", datetime.now().isoformat()),
                entry_price=result.get("price", opp.entry_price),
                entry_reason=opp.reason,
                quantity=opp.position_size,
                position_size_usd=opp.entry_price * opp.position_size,
                direction="LONG" if "BUY" in opp.signal else "SHORT",
                strategy=opp.strategy,
                pattern=analysis.strategies_triggered[0] if analysis and analysis.strategies_triggered else "",
                setup_quality="A" if opp.confidence >= 0.75 else "B" if opp.confidence >= 0.60 else "C",
                ai_signal=analysis.ai_signal if analysis else opp.signal,
                ai_confidence=opp.confidence,
                ai_prediction=analysis.predicted_direction if analysis else "",
                gap_percent=0,  # TODO: Add from analysis
                relative_volume=0,  # TODO: Add from analysis
                stop_loss=opp.stop_loss,
                take_profit=opp.target_price,
                risk_reward_ratio=opp.target_price / opp.stop_loss if opp.stop_loss else 0,
                notes=f"Auto-traded via AI Watchlist Trader. Signal: {opp.signal}",
                tags=f"{opp.strategy},{opp.signal}",
                paper_trade=result.get("paper_trade", True)
            )

            journal.log_trade(trade)
            logger.info(f"[AI TRADER] Trade logged to journal: {trade.id}")

        except Exception as e:
            logger.warning(f"[AI TRADER] Failed to log to journal: {e}")

    def analyze_watchlist(self, symbols: List[str] = None) -> Dict:
        """Analyze all symbols in watchlist"""
        if symbols is None:
            try:
                from watchlist_manager import get_watchlist_manager
                wm = get_watchlist_manager()
                symbols = wm.get_all_symbols()
            except:
                symbols = []

        results = {
            "analyzed": 0,
            "opportunities": 0,
            "symbols": []
        }

        for symbol in symbols:
            analysis = self.analyze_symbol(symbol)
            results["analyzed"] += 1

            if analysis.trade_recommendation:
                self.add_to_queue(analysis.trade_recommendation)
                results["opportunities"] += 1

            results["symbols"].append({
                "symbol": symbol,
                "score": analysis.overall_score,
                "signal": analysis.ai_signal,
                "action": analysis.ai_action,
                "has_opportunity": analysis.trade_recommendation is not None
            })

        logger.info(f"[AI TRADER] Analyzed {results['analyzed']} symbols, {results['opportunities']} opportunities")
        return results

    def on_symbol_added(self, symbol: str):
        """Called when a new symbol is added to watchlist"""
        logger.info(f"[AI TRADER] New symbol detected: {symbol}")

        # Analyze the symbol
        analysis = self.analyze_symbol(symbol)

        # Queue if opportunity found
        if analysis.trade_recommendation:
            self.add_to_queue(analysis.trade_recommendation)
            logger.info(f"[AI TRADER] {symbol} queued for trading: {analysis.trade_recommendation.signal}")
        else:
            logger.info(f"[AI TRADER] {symbol} analyzed but no opportunity (score: {analysis.overall_score:.2f})")

        return analysis

    def get_queue_status(self) -> Dict:
        """Get current queue status"""
        pending = [o for o in self.opportunity_queue if o.status == "pending"]
        ready = [o for o in self.opportunity_queue if o.status == "ready"]

        return {
            "enabled": self.config["enabled"],
            "paper_mode": self.config["paper_mode"],
            "pending_opportunities": len(pending),
            "ready_to_execute": len(ready),
            "today_trades": len([
                e for e in self.execution_log
                if datetime.fromisoformat(e['timestamp']).date() == datetime.now().date()
            ]),
            "max_daily_trades": self.config["max_daily_trades"],
            "queue": [asdict(o) for o in pending[:10]],  # Top 10
            "analyzed_symbols": len(self.analyzed_symbols),
            "last_analysis": max(
                (a.timestamp for a in self.analyzed_symbols.values()),
                default=None
            )
        }

    def start_monitoring(self):
        """Start background monitoring of watchlist"""
        if self._running:
            return

        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("[AI TRADER] Monitoring started")

    def stop_monitoring(self):
        """Stop background monitoring"""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("[AI TRADER] Monitoring stopped")

    def _monitor_loop(self):
        """Background monitoring loop"""
        last_full_analysis = datetime.min

        while self._running:
            try:
                # Check for watchlist changes
                current_hash = self._get_watchlist_hash()
                if current_hash != self._last_watchlist_hash:
                    new_symbols = self._detect_new_symbols()
                    for symbol in new_symbols:
                        self.on_symbol_added(symbol)
                    self._last_watchlist_hash = current_hash

                # Full analysis periodically
                if datetime.now() - last_full_analysis > timedelta(seconds=self.config["analysis_interval_seconds"]):
                    self.analyze_watchlist()
                    last_full_analysis = datetime.now()

                # Process queue
                self.process_queue()

            except Exception as e:
                logger.error(f"[AI TRADER] Monitor error: {e}")

            time.sleep(self.config["monitor_interval_seconds"])

    def _get_watchlist_hash(self) -> str:
        """Get hash of current watchlist for change detection"""
        try:
            from watchlist_manager import get_watchlist_manager
            wm = get_watchlist_manager()
            symbols = sorted(wm.get_all_symbols())
            return hash(tuple(symbols))
        except:
            return ""

    def _detect_new_symbols(self) -> List[str]:
        """Detect newly added symbols"""
        try:
            from watchlist_manager import get_watchlist_manager
            wm = get_watchlist_manager()
            current_symbols = set(wm.get_all_symbols())
            analyzed = set(self.analyzed_symbols.keys())
            return list(current_symbols - analyzed)
        except:
            return []

    def update_config(self, new_config: Dict):
        """Update trader configuration"""
        self.config.update(new_config)
        logger.info(f"[AI TRADER] Config updated: {self.config}")


# Singleton instance
_ai_watchlist_trader: Optional[AIWatchlistTrader] = None


def get_ai_watchlist_trader() -> AIWatchlistTrader:
    """Get singleton AI watchlist trader"""
    global _ai_watchlist_trader
    if _ai_watchlist_trader is None:
        _ai_watchlist_trader = AIWatchlistTrader()
    return _ai_watchlist_trader
