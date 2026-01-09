"""
Phase 2 Continuation Manager
============================
Monitors for Phase 2 entry conditions after Phase 1 exits.

Integrates with:
- two_phase_strategy.py: Core Phase 1/Phase 2 logic
- macd_analyzer.py: MACD-based entry/exit signals
- hft_scalper.py: Trade execution

Workflow:
1. When HFT Scalper exits a profitable position, register for Phase 2 watching
2. Monitor MACD for bullish crossover/momentum expansion
3. Check volume for pickup
4. Execute Phase 2 entry via scalper when conditions met
"""

import asyncio
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class Phase2Candidate:
    """Candidate for Phase 2 continuation trade"""

    symbol: str
    phase1_exit_price: float
    phase1_exit_time: datetime
    phase1_pnl_pct: float
    current_price: float = 0.0
    last_check_time: Optional[datetime] = None
    check_count: int = 0
    macd_ready: bool = False
    volume_ready: bool = False
    conditions_met: List[str] = field(default_factory=list)
    status: str = "watching"  # watching, ready, executed, expired, rejected
    expiry_time: Optional[datetime] = None

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["phase1_exit_time"] = (
            self.phase1_exit_time.isoformat() if self.phase1_exit_time else None
        )
        d["last_check_time"] = (
            self.last_check_time.isoformat() if self.last_check_time else None
        )
        d["expiry_time"] = self.expiry_time.isoformat() if self.expiry_time else None
        return d


class Phase2Manager:
    """
    Manages Phase 2 continuation entries.

    After a profitable Phase 1 exit, watches for:
    - MACD bullish crossover or momentum expansion
    - Volume pickup (1.5x+ average)
    - Price consolidation then new push

    Executes via HFT Scalper when conditions met.
    """

    def __init__(self):
        self.candidates: Dict[str, Phase2Candidate] = {}
        self.executed_trades: List[Dict] = []
        self.running = False
        self._monitor_task = None

        # Configuration
        self.config = {
            "enabled": True,
            "min_phase1_pnl_pct": 2.0,  # Only track if Phase 1 was +2%+
            "watch_duration_minutes": 30,  # How long to watch for Phase 2
            "check_interval_seconds": 10,  # How often to check conditions
            "min_pullback_pct": 1.0,  # Min pullback from Phase 1 exit before entry
            "max_pullback_pct": 5.0,  # Max pullback before rejecting
            "require_macd": True,  # Require MACD bullish
            "require_volume": True,  # Require volume pickup
            "min_volume_ratio": 1.5,  # Volume vs average
            "use_scalper_for_execution": True,
            "paper_mode": True,
        }

        self._load_config()

    def _load_config(self):
        """Load config from file"""
        config_path = os.path.join(os.path.dirname(__file__), "phase2_config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    loaded = json.load(f)
                    self.config.update(loaded)
                    logger.info("Phase 2 config loaded")
            except Exception as e:
                logger.error(f"Error loading Phase 2 config: {e}")

    def _save_config(self):
        """Save config to file"""
        config_path = os.path.join(os.path.dirname(__file__), "phase2_config.json")
        try:
            with open(config_path, "w") as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving Phase 2 config: {e}")

    def register_phase1_exit(
        self, symbol: str, exit_price: float, pnl_pct: float
    ) -> Dict:
        """
        Register a Phase 1 exit for Phase 2 monitoring.

        Called by HFT Scalper when a position is closed profitably.
        """
        if not self.config["enabled"]:
            return {"action": "SKIP", "reason": "Phase 2 disabled"}

        # Only track profitable Phase 1 exits
        if pnl_pct < self.config["min_phase1_pnl_pct"]:
            return {
                "action": "SKIP",
                "reason": f"Phase 1 P&L {pnl_pct:.1f}% below threshold",
            }

        # Check if already tracking
        if symbol in self.candidates:
            return {"action": "SKIP", "reason": "Already tracking for Phase 2"}

        now = datetime.now()
        expiry = now + timedelta(minutes=self.config["watch_duration_minutes"])

        candidate = Phase2Candidate(
            symbol=symbol,
            phase1_exit_price=exit_price,
            phase1_exit_time=now,
            phase1_pnl_pct=pnl_pct,
            current_price=exit_price,
            expiry_time=expiry,
        )

        self.candidates[symbol] = candidate

        # Also register with two_phase_strategy
        try:
            from .two_phase_strategy import (TradePhase, TwoPhasePosition,
                                             get_two_phase_strategy)

            strategy = get_two_phase_strategy()

            if symbol not in strategy.positions:
                strategy.positions[symbol] = TwoPhasePosition(
                    symbol=symbol,
                    phase=TradePhase.PHASE2_WATCHING,
                    entry_price=exit_price,  # Use as reference
                    high_since_entry=exit_price,
                )
        except Exception as e:
            logger.warning(f"Could not register with two_phase_strategy: {e}")

        logger.info(
            f"[PHASE2] Registered {symbol} for Phase 2 watching (Phase 1: +{pnl_pct:.1f}%)"
        )

        return {
            "action": "WATCHING",
            "symbol": symbol,
            "exit_price": exit_price,
            "pnl_pct": pnl_pct,
            "expiry": expiry.isoformat(),
        }

    async def check_conditions(self, symbol: str) -> Dict:
        """
        Check if Phase 2 entry conditions are met.

        Returns detailed analysis of conditions.
        """
        if symbol not in self.candidates:
            return {"ready": False, "reason": "not_tracking"}

        candidate = self.candidates[symbol]
        now = datetime.now()

        # Check expiry
        if candidate.expiry_time and now > candidate.expiry_time:
            candidate.status = "expired"
            return {"ready": False, "reason": "watch_period_expired"}

        conditions_met = []
        conditions_failed = []

        # Get current price
        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            current_price = ticker.fast_info.get("lastPrice", 0)
            if not current_price:
                hist = ticker.history(period="1d", interval="1m")
                if not hist.empty:
                    current_price = hist["Close"].iloc[-1]

            if current_price:
                candidate.current_price = current_price
        except Exception as e:
            logger.debug(f"Could not get price for {symbol}: {e}")
            current_price = candidate.phase1_exit_price

        # Check pullback
        pullback_pct = (
            (candidate.phase1_exit_price - current_price) / candidate.phase1_exit_price
        ) * 100

        if pullback_pct > self.config["max_pullback_pct"]:
            candidate.status = "rejected"
            return {"ready": False, "reason": f"pullback_too_deep_{pullback_pct:.1f}%"}

        if pullback_pct >= self.config["min_pullback_pct"]:
            conditions_met.append(f"Pullback {pullback_pct:.1f}%")
        else:
            conditions_failed.append(
                f"Waiting for pullback (currently {pullback_pct:.1f}%)"
            )

        # Check MACD
        macd_ready = False
        if self.config["require_macd"]:
            try:
                from .macd_analyzer import get_macd_analyzer

                analyzer = get_macd_analyzer()
                should_enter, reason, details = analyzer.check_phase2_entry(symbol)

                if should_enter:
                    macd_ready = True
                    conditions_met.append(f"MACD: {reason}")
                else:
                    conditions_failed.append(f"MACD: {reason}")
            except Exception as e:
                logger.debug(f"MACD check failed for {symbol}: {e}")
                conditions_failed.append("MACD: unavailable")
        else:
            macd_ready = True

        # Check volume
        volume_ready = False
        if self.config["require_volume"]:
            try:
                import yfinance as yf

                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="5d", interval="1d")
                if not hist.empty:
                    avg_volume = (
                        hist["Volume"].iloc[:-1].mean()
                    )  # Average of previous days
                    current_volume = hist["Volume"].iloc[-1]
                    volume_ratio = (
                        current_volume / avg_volume if avg_volume > 0 else 1.0
                    )

                    if volume_ratio >= self.config["min_volume_ratio"]:
                        volume_ready = True
                        conditions_met.append(f"Volume {volume_ratio:.1f}x")
                    else:
                        conditions_failed.append(
                            f"Volume {volume_ratio:.1f}x (need {self.config['min_volume_ratio']}x)"
                        )
            except Exception as e:
                logger.debug(f"Volume check failed for {symbol}: {e}")
                conditions_failed.append("Volume: unavailable")
        else:
            volume_ready = True

        # Update candidate
        candidate.macd_ready = macd_ready
        candidate.volume_ready = volume_ready
        candidate.conditions_met = conditions_met
        candidate.last_check_time = now
        candidate.check_count += 1

        # Determine if ready
        required_conditions = 1  # At least pullback
        if self.config["require_macd"]:
            required_conditions += 1
        if self.config["require_volume"]:
            required_conditions += 1

        ready = len(conditions_met) >= required_conditions

        if ready:
            candidate.status = "ready"

        return {
            "ready": ready,
            "symbol": symbol,
            "current_price": current_price,
            "pullback_pct": pullback_pct,
            "conditions_met": conditions_met,
            "conditions_failed": conditions_failed,
            "macd_ready": macd_ready,
            "volume_ready": volume_ready,
            "check_count": candidate.check_count,
        }

    async def execute_phase2_entry(self, symbol: str) -> Dict:
        """
        Execute Phase 2 entry via HFT Scalper.
        """
        if symbol not in self.candidates:
            return {"success": False, "reason": "not_tracking"}

        candidate = self.candidates[symbol]

        if candidate.status != "ready":
            return {"success": False, "reason": f"not_ready: {candidate.status}"}

        try:
            if self.config["use_scalper_for_execution"]:
                from .hft_scalper import get_hft_scalper

                scalper = get_hft_scalper()

                # Add to priority queue for immediate execution
                if symbol not in scalper.priority_symbols:
                    scalper.priority_symbols.append(symbol)

                logger.info(f"[PHASE2 ENTRY] {symbol} added to scalper priority queue")

                candidate.status = "executed"

                trade_record = {
                    "symbol": symbol,
                    "phase": "PHASE2",
                    "entry_price": candidate.current_price,
                    "phase1_exit_price": candidate.phase1_exit_price,
                    "phase1_pnl_pct": candidate.phase1_pnl_pct,
                    "conditions_met": candidate.conditions_met,
                    "timestamp": datetime.now().isoformat(),
                }

                self.executed_trades.append(trade_record)

                return {
                    "success": True,
                    "action": "SCALPER_QUEUE",
                    "trade": trade_record,
                }
            else:
                # Log-only mode
                logger.info(
                    f"[PHASE2 ENTRY SIGNAL] {symbol} @ ${candidate.current_price:.2f}"
                )
                candidate.status = "executed"

                return {
                    "success": True,
                    "action": "SIGNAL_ONLY",
                    "symbol": symbol,
                    "price": candidate.current_price,
                }

        except Exception as e:
            logger.error(f"Phase 2 execution error for {symbol}: {e}")
            return {"success": False, "error": str(e)}

    async def _monitor_loop(self):
        """Background monitoring loop"""
        while self.running:
            try:
                # Check all candidates
                symbols_to_remove = []

                for symbol, candidate in self.candidates.items():
                    if candidate.status in ["executed", "expired", "rejected"]:
                        symbols_to_remove.append(symbol)
                        continue

                    result = await self.check_conditions(symbol)

                    if result.get("ready"):
                        exec_result = await self.execute_phase2_entry(symbol)
                        if exec_result.get("success"):
                            logger.info(f"[PHASE2] Executed entry for {symbol}")

                # Clean up
                for symbol in symbols_to_remove:
                    del self.candidates[symbol]

            except Exception as e:
                logger.error(f"Phase 2 monitor error: {e}")

            await asyncio.sleep(self.config["check_interval_seconds"])

    def start_monitoring(self):
        """Start the monitoring loop"""
        if self.running:
            return {"status": "already_running"}

        self.running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("[PHASE2] Monitoring started")
        return {"status": "started"}

    def stop_monitoring(self):
        """Stop the monitoring loop"""
        self.running = False
        if self._monitor_task:
            self._monitor_task.cancel()
        logger.info("[PHASE2] Monitoring stopped")
        return {"status": "stopped"}

    def get_status(self) -> Dict:
        """Get current status"""
        return {
            "running": self.running,
            "enabled": self.config["enabled"],
            "paper_mode": self.config["paper_mode"],
            "candidates_count": len(self.candidates),
            "candidates": [c.to_dict() for c in self.candidates.values()],
            "executed_today": len(
                [
                    t
                    for t in self.executed_trades
                    if datetime.fromisoformat(t["timestamp"]).date()
                    == datetime.now().date()
                ]
            ),
        }

    def get_config(self) -> Dict:
        """Get current configuration"""
        return self.config.copy()

    def update_config(self, updates: Dict) -> Dict:
        """Update configuration"""
        for key, value in updates.items():
            if key in self.config:
                self.config[key] = value

        self._save_config()
        return self.config


# Singleton instance
_manager: Optional[Phase2Manager] = None


def get_phase2_manager() -> Phase2Manager:
    """Get or create the Phase 2 manager instance"""
    global _manager
    if _manager is None:
        _manager = Phase2Manager()
    return _manager


def register_phase1_exit(symbol: str, exit_price: float, pnl_pct: float) -> Dict:
    """Quick helper to register a Phase 1 exit"""
    manager = get_phase2_manager()
    return manager.register_phase1_exit(symbol, exit_price, pnl_pct)


if __name__ == "__main__":
    import asyncio

    logging.basicConfig(level=logging.INFO)

    async def test():
        manager = get_phase2_manager()

        # Simulate a Phase 1 exit
        result = manager.register_phase1_exit("AAPL", 195.50, 3.5)
        print(f"Registered: {result}")

        # Check conditions
        conditions = await manager.check_conditions("AAPL")
        print(f"Conditions: {conditions}")

        # Status
        print(f"Status: {manager.get_status()}")

    asyncio.run(test())
