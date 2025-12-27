"""
Warrior Setup Detector - Integration Module
============================================
Integrates all Ross Cameron trading components:
- Pattern Detector (Bull Flag, ABCD, Micro Pullback, HOD Break)
- Tape Analyzer (Green/Red Flow, Seller Thinning, Flush Detection)
- Setup Classifier (A/B/C Grading, Position Sizing)
- Halt Detector (LULD Bands, Countdown, Resume Prediction)

This module provides a unified interface for the HFT Scalper
to get complete trading signals based on Warrior Trading methodology.
"""

import logging
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import asyncio
import httpx

logger = logging.getLogger(__name__)


@dataclass
class WarriorSignal:
    """Complete trading signal from Warrior methodology"""
    symbol: str
    timestamp: datetime

    # Signal type
    signal_type: str = "NO_SIGNAL"  # ENTRY, EXIT, WARNING, NO_SIGNAL
    action: str = "WAIT"  # BUY, SELL, HOLD, WAIT

    # Setup analysis
    setup_type: str = ""  # Bull Flag, ABCD, Micro PB, etc.
    setup_grade: str = "C"  # A, B, C
    setup_confidence: float = 0.0

    # Trade parameters
    entry_price: float = 0.0
    stop_loss: float = 0.0
    target_price: float = 0.0
    position_size_pct: float = 0.0

    # Component signals
    pattern_signal: str = ""
    tape_signal: str = ""
    halt_warning: bool = False

    # Reasons
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        risk = self.entry_price - self.stop_loss if self.stop_loss else 0
        reward = self.target_price - self.entry_price if self.target_price else 0
        rr = reward / risk if risk > 0 else 0

        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'signal_type': self.signal_type,
            'action': self.action,
            'setup': {
                'type': self.setup_type,
                'grade': self.setup_grade,
                'confidence': round(self.setup_confidence, 3)
            },
            'trade': {
                'entry': self.entry_price,
                'stop': self.stop_loss,
                'target': self.target_price,
                'risk_reward': round(rr, 2),
                'position_size_pct': self.position_size_pct
            },
            'components': {
                'pattern': self.pattern_signal,
                'tape': self.tape_signal,
                'halt_warning': self.halt_warning
            },
            'reasons': self.reasons
        }


class WarriorSetupDetector:
    """
    Unified setup detection using all Warrior Trading components.

    Usage:
        detector = get_warrior_detector()
        signal = await detector.analyze(symbol, current_price, quote_data)

        if signal.action == "BUY":
            # Execute trade with signal.entry_price, signal.stop_loss, signal.target_price
    """

    def __init__(self):
        # Lazy load components
        self._pattern_detector = None
        self._tape_analyzer = None
        self._setup_classifier = None
        self._halt_detector = None

        # Configuration
        self.config = {
            "min_setup_grade": "B",  # Minimum grade to trade
            "min_confidence": 0.6,
            "min_risk_reward": 1.5,
            "require_tape_confirmation": True,
            "avoid_approaching_halts": True
        }

        logger.info("WarriorSetupDetector initialized")

    def _load_components(self):
        """Lazy load all components"""
        if self._pattern_detector is None:
            try:
                from ai.pattern_detector import get_pattern_detector
                self._pattern_detector = get_pattern_detector()
            except Exception as e:
                logger.warning(f"Pattern detector not available: {e}")

        if self._tape_analyzer is None:
            try:
                from ai.tape_analyzer import get_tape_analyzer
                self._tape_analyzer = get_tape_analyzer()
            except Exception as e:
                logger.warning(f"Tape analyzer not available: {e}")

        if self._setup_classifier is None:
            try:
                from ai.setup_classifier import get_setup_classifier
                self._setup_classifier = get_setup_classifier()
            except Exception as e:
                logger.warning(f"Setup classifier not available: {e}")

        if self._halt_detector is None:
            try:
                from ai.halt_detector import get_halt_detector
                self._halt_detector = get_halt_detector()
            except Exception as e:
                logger.warning(f"Halt detector not available: {e}")

    async def analyze(
        self,
        symbol: str,
        current_price: float,
        quote: Dict = None,
        stock_info: Dict = None
    ) -> WarriorSignal:
        """
        Analyze a symbol for trading setup.

        Args:
            symbol: Stock symbol
            current_price: Current price
            quote: Quote data with bid/ask
            stock_info: Stock info with float, volume, change%, etc.

        Returns:
            WarriorSignal with complete analysis
        """
        self._load_components()

        signal = WarriorSignal(
            symbol=symbol,
            timestamp=datetime.now()
        )

        reasons = []

        # Get stock info if not provided
        if stock_info is None:
            stock_info = await self._get_stock_info(symbol)

        # ===== 1. CHECK FOR HALT WARNING =====
        if self._halt_detector and quote:
            halt_status = await self._check_halt_status(symbol, quote)
            if halt_status.get('warning'):
                signal.halt_warning = True
                if self.config["avoid_approaching_halts"]:
                    reasons.append(f"Approaching LULD band: {halt_status.get('approaching')}")
                    signal.signal_type = "WARNING"
                    signal.action = "WAIT"
                    signal.reasons = reasons
                    return signal

        # ===== 2. PATTERN DETECTION =====
        pattern_results = {}
        if self._pattern_detector:
            pattern_results = self._pattern_detector.detect_all_patterns(symbol)

            # Find best pattern
            detected = [
                (name, p) for name, p in pattern_results.items()
                if p.detected
            ]
            if detected:
                best_name, best_pattern = max(detected, key=lambda x: x[1].confidence)
                signal.pattern_signal = best_pattern.pattern_type
                signal.entry_price = best_pattern.entry_price
                signal.stop_loss = best_pattern.stop_loss
                signal.target_price = best_pattern.target_price
                reasons.append(f"Pattern: {best_pattern.pattern_type} ({best_pattern.confidence:.0%})")

        # ===== 3. TAPE ANALYSIS =====
        if self._tape_analyzer:
            tape_signal = self._tape_analyzer.get_entry_signal(symbol)
            signal.tape_signal = tape_signal.get('signal', 'NO_SIGNAL')

            if signal.tape_signal != 'NO_SIGNAL':
                reasons.append(f"Tape: {signal.tape_signal} ({tape_signal.get('confidence', 0):.0%})")

                # If tape shows dip buy setup, use current price as entry
                if signal.tape_signal == 'IRRATIONAL_FLUSH':
                    signal.entry_price = current_price
                    signal.stop_loss = current_price * 0.97
                    signal.target_price = current_price * 1.05

        # ===== 4. SETUP CLASSIFICATION =====
        if self._setup_classifier and stock_info:
            from ai.setup_classifier import StockCriteria

            criteria = StockCriteria(
                symbol=symbol,
                has_news=stock_info.get('has_news', False),
                float_under_10m=0 < stock_info.get('float', 0) < 10_000_000,
                price_in_range=1.0 <= current_price <= 20.0,
                change_over_10pct=stock_info.get('change_pct', 0) >= 10.0,
                rvol_over_5x=(stock_info.get('volume', 0) /
                              stock_info.get('avg_volume', 1) >= 5.0)
                             if stock_info.get('avg_volume', 0) > 0 else False,
                float_shares=stock_info.get('float', 0),
                price=current_price,
                change_pct=stock_info.get('change_pct', 0),
                relative_volume=(stock_info.get('volume', 0) /
                                 stock_info.get('avg_volume', 1))
                                if stock_info.get('avg_volume', 0) > 0 else 0
            )

            classification = self._setup_classifier.classify(
                symbol=symbol,
                stock_criteria=criteria,
                pattern_results=pattern_results,
                tape_analysis={'signal': signal.tape_signal},
                current_price=current_price
            )

            signal.setup_type = classification.setup_type.value
            signal.setup_grade = classification.setup_grade.value
            signal.setup_confidence = classification.confidence
            signal.position_size_pct = classification.position_size_pct
            signal.action = classification.action

            reasons.extend(classification.reasons)

        # ===== 5. DETERMINE FINAL SIGNAL =====
        grade_order = {"A+": 0, "A": 1, "B": 2, "C": 3, "F": 4}
        min_grade_order = grade_order.get(self.config["min_setup_grade"], 2)
        current_grade_order = grade_order.get(signal.setup_grade, 4)

        if current_grade_order <= min_grade_order:
            if signal.setup_confidence >= self.config["min_confidence"]:
                # Check risk/reward
                if signal.entry_price and signal.stop_loss and signal.target_price:
                    risk = signal.entry_price - signal.stop_loss
                    reward = signal.target_price - signal.entry_price
                    if risk > 0 and reward / risk >= self.config["min_risk_reward"]:
                        signal.signal_type = "ENTRY"
                        if signal.action == "WAIT":
                            signal.action = "BUY"
                    else:
                        reasons.append(f"R:R below minimum ({reward/risk if risk > 0 else 0:.1f}:1)")
                elif signal.tape_signal in ["FIRST_GREEN_PRINT", "SELLER_THINNING"]:
                    # Tape signal without full setup - still valid for scalp
                    signal.signal_type = "ENTRY"
                    signal.action = "SCALP"
                    signal.entry_price = current_price
                    signal.stop_loss = current_price * 0.97
                    signal.target_price = current_price * 1.03
        else:
            if signal.setup_grade != "F":
                reasons.append(f"Grade {signal.setup_grade} below minimum {self.config['min_setup_grade']}")

        signal.reasons = reasons
        return signal

    async def _check_halt_status(self, symbol: str, quote: Dict) -> Dict:
        """Check if approaching LULD halt"""
        if not self._halt_detector:
            return {'warning': False}

        # Check for halt
        halt_info = await self._halt_detector.check_for_halt(symbol, quote)
        if halt_info:
            return {'warning': True, 'halted': True}

        # Check band proximity
        price = quote.get('price') or quote.get('last', 0)
        if price and symbol in self._halt_detector.luld_bands:
            proximity = self._halt_detector.check_band_proximity(symbol, price)
            return proximity

        return {'warning': False}

    async def _get_stock_info(self, symbol: str) -> Dict:
        """Get stock info from API"""
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"http://localhost:9100/api/worklist",
                    timeout=5.0
                )
                if resp.status_code == 200:
                    data = resp.json()
                    for stock in data.get('worklist', []):
                        if stock.get('symbol') == symbol:
                            return {
                                'float': stock.get('float', 0),
                                'volume': stock.get('volume', 0),
                                'avg_volume': stock.get('avg_volume', 0),
                                'change_pct': stock.get('change_pct', 0),
                                'has_news': stock.get('has_news', False)
                            }
        except Exception as e:
            logger.debug(f"Failed to get stock info for {symbol}: {e}")

        return {}

    def feed_tape_data(self, symbol: str, trades: List[Dict]):
        """Feed trade data to tape analyzer"""
        self._load_components()
        if self._tape_analyzer:
            self._tape_analyzer.add_trades_batch(symbol, trades)

    def feed_candle_data(self, symbol: str, candles: List[Dict]):
        """Feed candle data to pattern detector"""
        self._load_components()
        if self._pattern_detector:
            self._pattern_detector.add_candles_batch(symbol, candles)

    def calculate_luld_bands(self, symbol: str, reference_price: float, tier: int = 2):
        """Calculate LULD bands for a symbol"""
        self._load_components()
        if self._halt_detector:
            return self._halt_detector.calculate_luld_bands(symbol, reference_price, tier)
        return None

    def get_status(self) -> Dict:
        """Get status of all components"""
        self._load_components()

        return {
            'pattern_detector': {
                'available': self._pattern_detector is not None,
                'symbols': list(self._pattern_detector.candle_history.keys())
                          if self._pattern_detector else []
            },
            'tape_analyzer': {
                'available': self._tape_analyzer is not None,
                'symbols': list(self._tape_analyzer.tape_history.keys())
                          if self._tape_analyzer else []
            },
            'setup_classifier': {
                'available': self._setup_classifier is not None
            },
            'halt_detector': {
                'available': self._halt_detector is not None,
                'halted': list(self._halt_detector.halted_stocks.keys())
                         if self._halt_detector else [],
                'luld_tracking': list(self._halt_detector.luld_bands.keys())
                                if self._halt_detector else []
            },
            'config': self.config
        }


# Singleton instance
_warrior_detector: Optional[WarriorSetupDetector] = None


def get_warrior_detector() -> WarriorSetupDetector:
    """Get or create Warrior Setup Detector instance"""
    global _warrior_detector
    if _warrior_detector is None:
        _warrior_detector = WarriorSetupDetector()
    return _warrior_detector


# Convenience functions
async def analyze_setup(symbol: str, price: float, quote: Dict = None) -> WarriorSignal:
    """Quick setup analysis"""
    detector = get_warrior_detector()
    return await detector.analyze(symbol, price, quote)


async def get_trading_signal(symbol: str) -> Dict:
    """Get trading signal for API response"""
    detector = get_warrior_detector()

    # Get current price
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"http://localhost:9100/api/price/{symbol}",
                timeout=3.0
            )
            if resp.status_code == 200:
                quote = resp.json()
                price = quote.get('price') or quote.get('last', 0)
                signal = await detector.analyze(symbol, price, quote)
                return signal.to_dict()
    except Exception as e:
        logger.error(f"Failed to get trading signal for {symbol}: {e}")

    return {'error': 'Failed to analyze', 'symbol': symbol}
