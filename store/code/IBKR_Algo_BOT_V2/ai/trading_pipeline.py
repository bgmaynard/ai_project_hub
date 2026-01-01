"""
Trading Pipeline - Wires Warrior Scanner → Watchlist → AI Trader → Execution

This module connects all the components of the Morpheus Trading Bot:
1. Warrior Scanner finds momentum stocks meeting criteria
2. Qualifying stocks are auto-added to the watchlist
3. AI Trader analyzes watchlist stocks
4. High-confidence opportunities are queued for execution
5. Execution happens in paper mode (or live when enabled)

Author: Claude Code
"""

import asyncio
import logging
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Global pipeline instance
_pipeline_instance = None


@dataclass
class PipelineConfig:
    """Configuration for the trading pipeline"""

    # Auto-add to watchlist settings
    auto_add_to_watchlist: bool = True
    min_scanner_score: float = 60.0  # Min confidence to add to watchlist
    max_watchlist_size: int = 20

    # AI Analysis settings
    auto_analyze_on_add: bool = True
    min_ai_confidence: float = 0.60  # 60% confidence to queue

    # Execution settings
    auto_execute: bool = False  # Must be explicitly enabled
    paper_mode: bool = True  # Safety first

    # Pipeline intervals (seconds)
    scanner_interval: int = 60  # How often to run scanner
    analysis_interval: int = 30  # How often to analyze watchlist
    execution_interval: int = 10  # How often to process queue


class TradingPipeline:
    """
    Orchestrates the full trading flow:
    Scanner → Watchlist → AI Analysis → Trade Queue → Execution
    """

    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        self.is_running = False
        self._scanner_thread = None
        self._analysis_thread = None
        self._execution_thread = None

        # Statistics
        self.stats = {
            "scanner_runs": 0,
            "stocks_scanned": 0,
            "stocks_added_to_watchlist": 0,
            "stocks_analyzed": 0,
            "opportunities_found": 0,
            "trades_executed": 0,
            "last_scanner_run": None,
            "last_analysis_run": None,
            "last_execution_run": None,
        }

        # Callbacks for events
        self._on_stock_found_callbacks: List[Callable] = []
        self._on_opportunity_callbacks: List[Callable] = []
        self._on_trade_callbacks: List[Callable] = []

        logger.info("[PIPELINE] Trading Pipeline initialized")

    def on_stock_found(self, callback: Callable):
        """Register callback for when scanner finds a stock"""
        self._on_stock_found_callbacks.append(callback)

    def on_opportunity(self, callback: Callable):
        """Register callback for when AI finds opportunity"""
        self._on_opportunity_callbacks.append(callback)

    def on_trade(self, callback: Callable):
        """Register callback for when trade is executed"""
        self._on_trade_callbacks.append(callback)

    # =========================================================================
    # STEP 1: WARRIOR SCANNER → WATCHLIST
    # =========================================================================

    def run_scanner(self) -> Dict:
        """Run Warrior Scanner and add qualifying stocks to watchlist"""
        results = {"scanned": 0, "qualified": 0, "added_to_watchlist": 0, "setups": []}

        try:
            # Import scanner
            from ai.warrior_scanner import WarriorScanner

            scanner = WarriorScanner()

            # Run scan
            candidates = scanner.scan_premarket()
            results["scanned"] = len(candidates) if candidates else 0

            if not candidates:
                logger.info("[PIPELINE] Scanner found no candidates")
                return results

            # Filter by minimum score
            qualified = [
                c
                for c in candidates
                if c.confidence_score >= self.config.min_scanner_score
            ]
            results["qualified"] = len(qualified)

            if not qualified:
                logger.info(
                    f"[PIPELINE] No stocks above {self.config.min_scanner_score} score threshold"
                )
                return results

            # Add to watchlist if enabled
            if self.config.auto_add_to_watchlist:
                added = self._add_to_watchlist([c.symbol for c in qualified])
                results["added_to_watchlist"] = added

            # Store setups
            results["setups"] = [c.to_dict() for c in qualified[:10]]

            # Trigger callbacks
            for callback in self._on_stock_found_callbacks:
                try:
                    callback(qualified)
                except Exception as e:
                    logger.error(f"[PIPELINE] Callback error: {e}")

            self.stats["scanner_runs"] += 1
            self.stats["stocks_scanned"] += results["scanned"]
            self.stats["stocks_added_to_watchlist"] += results["added_to_watchlist"]
            self.stats["last_scanner_run"] = datetime.now().isoformat()

            logger.info(
                f"[PIPELINE] Scanner: {results['scanned']} scanned, {results['qualified']} qualified, {results['added_to_watchlist']} added"
            )

        except Exception as e:
            logger.error(f"[PIPELINE] Scanner error: {e}")
            results["error"] = str(e)

        return results

    def _add_to_watchlist(self, symbols: List[str]) -> int:
        """Add symbols to watchlist"""
        added = 0
        try:
            from watchlist_manager import get_watchlist_manager

            mgr = get_watchlist_manager()

            if not mgr:
                logger.warning("[PIPELINE] Watchlist manager not available")
                return 0

            # Get current watchlist
            watchlist = mgr.get_default_watchlist()
            current_symbols = set(watchlist.get("symbols", []))
            watchlist_id = watchlist.get("watchlist_id")

            # Filter new symbols
            new_symbols = [s for s in symbols if s not in current_symbols]

            # Respect max size
            available_slots = self.config.max_watchlist_size - len(current_symbols)
            symbols_to_add = new_symbols[: max(0, available_slots)]

            if symbols_to_add and watchlist_id:
                mgr.add_symbols(watchlist_id, symbols_to_add)
                added = len(symbols_to_add)
                logger.info(
                    f"[PIPELINE] Added {added} symbols to watchlist: {symbols_to_add}"
                )

                # Trigger AI analysis for new symbols
                if self.config.auto_analyze_on_add:
                    self._analyze_symbols(symbols_to_add)

        except Exception as e:
            logger.error(f"[PIPELINE] Watchlist error: {e}")

        return added

    # =========================================================================
    # STEP 2: WATCHLIST → AI TRADER ANALYSIS
    # =========================================================================

    def analyze_watchlist(self) -> Dict:
        """Analyze all watchlist symbols with AI Trader"""
        results = {"analyzed": 0, "opportunities_found": 0, "symbols": []}

        try:
            from ai.ai_watchlist_trader import get_ai_watchlist_trader

            trader = get_ai_watchlist_trader()

            # Get watchlist symbols
            from watchlist_manager import get_watchlist_manager

            mgr = get_watchlist_manager()
            watchlist = mgr.get_default_watchlist() if mgr else {}
            symbols = watchlist.get("symbols", [])

            if not symbols:
                return results

            for symbol in symbols:
                try:
                    analysis = trader.analyze_symbol(symbol)
                    results["analyzed"] += 1

                    symbol_result = {
                        "symbol": symbol,
                        "score": analysis.overall_score,
                        "signal": analysis.ai_signal,
                        "confidence": analysis.ai_confidence,
                        "has_opportunity": analysis.trade_recommendation is not None,
                    }
                    results["symbols"].append(symbol_result)

                    if analysis.trade_recommendation:
                        results["opportunities_found"] += 1
                        # Trigger callbacks
                        for callback in self._on_opportunity_callbacks:
                            try:
                                callback(analysis)
                            except:
                                pass

                except Exception as e:
                    logger.warning(f"[PIPELINE] Analysis error for {symbol}: {e}")

            self.stats["stocks_analyzed"] += results["analyzed"]
            self.stats["opportunities_found"] += results["opportunities_found"]
            self.stats["last_analysis_run"] = datetime.now().isoformat()

            logger.info(
                f"[PIPELINE] Analysis: {results['analyzed']} analyzed, {results['opportunities_found']} opportunities"
            )

        except Exception as e:
            logger.error(f"[PIPELINE] Analysis error: {e}")
            results["error"] = str(e)

        return results

    def _analyze_symbols(self, symbols: List[str]):
        """Analyze specific symbols (called when added to watchlist)"""
        try:
            from ai.ai_watchlist_trader import get_ai_watchlist_trader

            trader = get_ai_watchlist_trader()

            for symbol in symbols:
                try:
                    analysis = trader.on_symbol_added(symbol)
                    logger.info(
                        f"[PIPELINE] Analyzed {symbol}: score={analysis.overall_score:.2f}, signal={analysis.ai_signal}"
                    )
                except Exception as e:
                    logger.warning(f"[PIPELINE] Analysis error for {symbol}: {e}")

        except Exception as e:
            logger.error(f"[PIPELINE] Analyze symbols error: {e}")

    # =========================================================================
    # STEP 3: AI TRADER → EXECUTION
    # =========================================================================

    def process_execution_queue(self) -> Dict:
        """Process the trade opportunity queue"""
        results = {"processed": 0, "executed": 0, "trades": []}

        if not self.config.auto_execute:
            logger.debug("[PIPELINE] Auto-execute disabled, skipping queue processing")
            return results

        try:
            from ai.ai_watchlist_trader import get_ai_watchlist_trader

            trader = get_ai_watchlist_trader()

            # Enable temporarily if in auto mode
            was_enabled = trader.config["enabled"]
            if self.config.auto_execute:
                trader.config["enabled"] = True

            # Process queue
            executed = trader.process_queue()
            results["executed"] = len(executed)
            results["trades"] = executed

            # Restore state
            trader.config["enabled"] = was_enabled

            # Trigger callbacks
            for trade in executed:
                for callback in self._on_trade_callbacks:
                    try:
                        callback(trade)
                    except:
                        pass

            self.stats["trades_executed"] += results["executed"]
            self.stats["last_execution_run"] = datetime.now().isoformat()

            if results["executed"] > 0:
                logger.info(f"[PIPELINE] Executed {results['executed']} trades")

        except Exception as e:
            logger.error(f"[PIPELINE] Execution error: {e}")
            results["error"] = str(e)

        return results

    # =========================================================================
    # PIPELINE CONTROL
    # =========================================================================

    def start(self):
        """Start the automated pipeline"""
        if self.is_running:
            logger.warning("[PIPELINE] Already running")
            return

        self.is_running = True
        logger.info("[PIPELINE] Starting automated pipeline...")

        # Start background threads
        self._scanner_thread = threading.Thread(target=self._scanner_loop, daemon=True)
        self._analysis_thread = threading.Thread(
            target=self._analysis_loop, daemon=True
        )
        self._execution_thread = threading.Thread(
            target=self._execution_loop, daemon=True
        )

        self._scanner_thread.start()
        self._analysis_thread.start()
        self._execution_thread.start()

        logger.info("[PIPELINE] All pipeline threads started")

    def stop(self):
        """Stop the automated pipeline"""
        self.is_running = False
        logger.info("[PIPELINE] Stopping pipeline...")

        # Threads will exit on next iteration
        time.sleep(2)
        logger.info("[PIPELINE] Pipeline stopped")

    def _scanner_loop(self):
        """Background scanner loop"""
        while self.is_running:
            try:
                self.run_scanner()
            except Exception as e:
                logger.error(f"[PIPELINE] Scanner loop error: {e}")
            time.sleep(self.config.scanner_interval)

    def _analysis_loop(self):
        """Background analysis loop"""
        while self.is_running:
            try:
                self.analyze_watchlist()
            except Exception as e:
                logger.error(f"[PIPELINE] Analysis loop error: {e}")
            time.sleep(self.config.analysis_interval)

    def _execution_loop(self):
        """Background execution loop"""
        while self.is_running:
            try:
                self.process_execution_queue()
            except Exception as e:
                logger.error(f"[PIPELINE] Execution loop error: {e}")
            time.sleep(self.config.execution_interval)

    def get_status(self) -> Dict:
        """Get pipeline status"""
        return {
            "is_running": self.is_running,
            "config": asdict(self.config),
            "stats": self.stats,
            "threads": {
                "scanner": (
                    self._scanner_thread.is_alive() if self._scanner_thread else False
                ),
                "analysis": (
                    self._analysis_thread.is_alive() if self._analysis_thread else False
                ),
                "execution": (
                    self._execution_thread.is_alive()
                    if self._execution_thread
                    else False
                ),
            },
        }


def get_trading_pipeline() -> TradingPipeline:
    """Get or create the trading pipeline instance"""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = TradingPipeline()
    return _pipeline_instance
