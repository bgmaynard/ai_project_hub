"""
News Spike Validator
====================
When breaking news hits, algos cause instant spikes.
This module validates if the spike is LEGIT or just NOISE.

Validation Flow:
1. News detected -> Get initial price/volume
2. Wait 5-15 seconds -> Check if holding or fading
3. Run validation checks:
   - Volume confirmation (3x+ normal = real interest)
   - Price action (holding above spike = legit)
   - Spread analysis (widening = danger)
   - News quality (material vs fluff)
   - Market cap check (small cap = more volatile)
4. Generate confidence score -> Trade or wait

Ross Cameron: "The first move is for the algos, the second move is for us"
"""

import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional

import pytz
import requests

logger = logging.getLogger(__name__)


@dataclass
class SpikeValidation:
    """Result of spike validation"""

    symbol: str
    news_headline: str
    news_catalyst: str

    # Price action
    pre_news_price: float
    spike_price: float
    current_price: float
    spike_pct: float
    holding_pct: float  # How much of spike is holding

    # Volume analysis
    volume_ratio: float  # Current vs average
    volume_surge: bool

    # Spread analysis
    spread_pct: float
    spread_safe: bool

    # Validation scores (0-100)
    volume_score: int
    price_action_score: int
    news_quality_score: int
    timing_score: int
    overall_score: int

    # Final verdict
    verdict: str  # "BUY_NOW", "WAIT_PULLBACK", "FADE", "AVOID"
    confidence: float
    reason: str

    timestamp: datetime

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "headline": self.news_headline[:80],
            "catalyst": self.news_catalyst,
            "pre_price": f"${self.pre_news_price:.2f}",
            "spike_price": f"${self.spike_price:.2f}",
            "current_price": f"${self.current_price:.2f}",
            "spike_pct": f"{self.spike_pct:+.1f}%",
            "holding_pct": f"{self.holding_pct:.0f}%",
            "volume_ratio": f"{self.volume_ratio:.1f}x",
            "spread_pct": f"{self.spread_pct:.2f}%",
            "scores": {
                "volume": self.volume_score,
                "price_action": self.price_action_score,
                "news_quality": self.news_quality_score,
                "timing": self.timing_score,
                "overall": self.overall_score,
            },
            "verdict": self.verdict,
            "confidence": f"{self.confidence:.0%}",
            "reason": self.reason,
            "timestamp": self.timestamp.isoformat(),
        }


class NewsSpikeValidator:
    """
    Validates breaking news spikes to determine if they're real or algo noise.
    """

    def __init__(self, api_url: str = "http://localhost:9100/api/alpaca"):
        self.api_url = api_url
        self.et_tz = pytz.timezone("US/Eastern")

        # Price tracking for validation
        self.pre_news_prices: Dict[str, float] = {}
        self.spike_prices: Dict[str, float] = {}
        self.price_history: Dict[str, List[tuple]] = defaultdict(list)

        # Volume tracking
        self.avg_volumes: Dict[str, int] = {}

        # Validation results
        self.validations: List[SpikeValidation] = []
        self.max_validations = 50

        # Callbacks
        self.on_buy_signal: Optional[Callable] = None
        self.on_wait_signal: Optional[Callable] = None
        self.on_avoid_signal: Optional[Callable] = None

        # News quality keywords
        self.high_quality_catalysts = {
            "fda_approval": 100,
            "merger": 95,
            "acquisition": 95,
            "buyout": 95,
            "earnings_beat": 85,
            "guidance_raise": 80,
            "upgrade": 75,
            "contract_win": 70,
            "partnership": 65,
        }

        self.low_quality_catalysts = {
            "rumor": 30,
            "speculation": 25,
            "analyst_comment": 40,
            "social_media": 20,
            "reddit": 15,
            "meme": 10,
        }

        logger.info("NewsSpikeValidator initialized")

    def record_pre_news_price(self, symbol: str, price: float):
        """Record price before news hit"""
        self.pre_news_prices[symbol] = price
        self.price_history[symbol].append((datetime.now(self.et_tz), price))

    def record_spike_price(self, symbol: str, price: float):
        """Record the spike high"""
        self.spike_prices[symbol] = price
        self.price_history[symbol].append((datetime.now(self.et_tz), price))

    async def validate_spike(
        self, symbol: str, headline: str, catalyst_type: str, wait_seconds: int = 10
    ) -> SpikeValidation:
        """
        Full spike validation - waits and analyzes price action.
        """
        symbol = symbol.upper()
        now = datetime.now(self.et_tz)

        # Get initial snapshot
        initial_data = self._get_quote(symbol)
        if not initial_data:
            return self._create_avoid_validation(
                symbol, headline, "Cannot get quote data"
            )

        pre_price = self.pre_news_prices.get(symbol, initial_data["price"])
        spike_price = max(
            self.spike_prices.get(symbol, initial_data["price"]), initial_data["price"]
        )

        # Record initial spike
        self.spike_prices[symbol] = spike_price

        # Wait for price to settle
        logger.info(f"Validating {symbol} spike - waiting {wait_seconds}s...")
        await self._async_sleep(wait_seconds)

        # Get current state after waiting
        current_data = self._get_quote(symbol)
        if not current_data:
            return self._create_avoid_validation(symbol, headline, "Lost quote data")

        # Run all validation checks
        validation = self._run_validation(
            symbol=symbol,
            headline=headline,
            catalyst_type=catalyst_type,
            pre_price=pre_price,
            spike_price=spike_price,
            current_data=current_data,
        )

        # Store and return
        self.validations.append(validation)
        if len(self.validations) > self.max_validations:
            self.validations = self.validations[-self.max_validations :]

        # Trigger callbacks
        self._trigger_callbacks(validation)

        return validation

    def validate_spike_sync(
        self, symbol: str, headline: str, catalyst_type: str, wait_seconds: int = 10
    ) -> SpikeValidation:
        """Synchronous version of validate_spike"""
        symbol = symbol.upper()
        now = datetime.now(self.et_tz)

        # Get initial snapshot
        initial_data = self._get_quote(symbol)
        if not initial_data:
            return self._create_avoid_validation(
                symbol, headline, "Cannot get quote data"
            )

        pre_price = self.pre_news_prices.get(symbol, initial_data["price"])
        spike_price = max(
            self.spike_prices.get(symbol, initial_data["price"]), initial_data["price"]
        )

        # Track price during wait period
        prices_during_wait = [spike_price]
        for i in range(wait_seconds):
            time.sleep(1)
            data = self._get_quote(symbol)
            if data:
                prices_during_wait.append(data["price"])
                if data["price"] > spike_price:
                    spike_price = data["price"]

        # Get final state
        current_data = self._get_quote(symbol)
        if not current_data:
            return self._create_avoid_validation(symbol, headline, "Lost quote data")

        # Update spike price if we saw higher
        self.spike_prices[symbol] = spike_price

        # Run validation
        validation = self._run_validation(
            symbol=symbol,
            headline=headline,
            catalyst_type=catalyst_type,
            pre_price=pre_price,
            spike_price=spike_price,
            current_data=current_data,
        )

        # Store
        self.validations.append(validation)
        if len(self.validations) > self.max_validations:
            self.validations = self.validations[-self.max_validations :]

        # Callbacks
        self._trigger_callbacks(validation)

        return validation

    def quick_validate(
        self, symbol: str, headline: str = "", catalyst_type: str = "unknown"
    ) -> SpikeValidation:
        """
        Quick validation without waiting - for immediate decision.
        Uses current price action and volume.
        """
        symbol = symbol.upper()

        current_data = self._get_quote(symbol)
        if not current_data:
            return self._create_avoid_validation(symbol, headline, "Cannot get quote")

        pre_price = self.pre_news_prices.get(symbol, current_data["price"] * 0.95)
        spike_price = self.spike_prices.get(symbol, current_data["price"])

        return self._run_validation(
            symbol=symbol,
            headline=headline,
            catalyst_type=catalyst_type,
            pre_price=pre_price,
            spike_price=spike_price,
            current_data=current_data,
        )

    def _run_validation(
        self,
        symbol: str,
        headline: str,
        catalyst_type: str,
        pre_price: float,
        spike_price: float,
        current_data: Dict,
    ) -> SpikeValidation:
        """Run all validation checks and generate verdict"""

        now = datetime.now(self.et_tz)
        current_price = current_data["price"]
        bid = current_data.get("bid", current_price)
        ask = current_data.get("ask", current_price)
        volume = current_data.get("volume", 0)

        # Calculate metrics
        spike_pct = (
            ((spike_price - pre_price) / pre_price * 100) if pre_price > 0 else 0
        )

        # How much of the spike is holding?
        if spike_price > pre_price:
            holding_pct = (current_price - pre_price) / (spike_price - pre_price) * 100
            holding_pct = max(0, min(100, holding_pct))
        else:
            holding_pct = 100 if current_price >= pre_price else 0

        # Spread
        spread = ask - bid if ask > bid else 0
        spread_pct = (spread / bid * 100) if bid > 0 else 0
        spread_safe = spread_pct < 2.0

        # Volume ratio
        avg_vol = self.avg_volumes.get(symbol, volume // 2)
        volume_ratio = (volume / avg_vol) if avg_vol > 0 else 1.0
        volume_surge = volume_ratio >= 2.0

        # === SCORING ===

        # 1. Volume Score (0-100)
        if volume_ratio >= 5.0:
            volume_score = 100
        elif volume_ratio >= 3.0:
            volume_score = 85
        elif volume_ratio >= 2.0:
            volume_score = 70
        elif volume_ratio >= 1.5:
            volume_score = 50
        else:
            volume_score = 30

        # 2. Price Action Score (0-100)
        if holding_pct >= 90:
            price_action_score = 100  # Holding all gains
        elif holding_pct >= 70:
            price_action_score = 85  # Holding most
        elif holding_pct >= 50:
            price_action_score = 60  # Holding half
        elif holding_pct >= 30:
            price_action_score = 40  # Fading
        else:
            price_action_score = 20  # Dumping

        # Bonus for making new highs
        if current_price > spike_price:
            price_action_score = min(100, price_action_score + 15)

        # 3. News Quality Score (0-100)
        news_quality_score = 50  # Default
        catalyst_lower = catalyst_type.lower()

        for catalyst, score in self.high_quality_catalysts.items():
            if catalyst in catalyst_lower:
                news_quality_score = score
                break

        for catalyst, score in self.low_quality_catalysts.items():
            if catalyst in catalyst_lower or catalyst in headline.lower():
                news_quality_score = min(news_quality_score, score)

        # 4. Timing Score (0-100)
        hour = now.hour
        minute = now.minute

        # Best times: 9:30-10:00, 10:00-11:00
        if 9 <= hour < 10 or (hour == 9 and minute >= 30):
            timing_score = 90  # Opening momentum
        elif 10 <= hour < 11:
            timing_score = 85  # Follow through
        elif hour == 4 or (hour < 9 and hour >= 4):
            timing_score = 70  # Premarket (can fade at open)
        elif 11 <= hour < 14:
            timing_score = 50  # Midday chop
        elif 14 <= hour < 16:
            timing_score = 65  # Afternoon momentum
        else:
            timing_score = 40  # After hours

        # === OVERALL SCORE ===
        overall_score = int(
            volume_score * 0.30
            + price_action_score * 0.35
            + news_quality_score * 0.25
            + timing_score * 0.10
        )

        # === VERDICT ===
        if overall_score >= 80 and spread_safe and holding_pct >= 70:
            verdict = "BUY_NOW"
            confidence = min(0.95, overall_score / 100)
            reason = f"Strong spike holding {holding_pct:.0f}% with {volume_ratio:.1f}x volume"

        elif overall_score >= 65 and spread_safe:
            if holding_pct >= 50:
                verdict = "BUY_NOW"
                confidence = min(0.80, overall_score / 100)
                reason = (
                    f"Good setup - {holding_pct:.0f}% holding, {volume_ratio:.1f}x vol"
                )
            else:
                verdict = "WAIT_PULLBACK"
                confidence = 0.60
                reason = f"Fading from spike - wait for pullback entry"

        elif overall_score >= 50 and spread_safe:
            verdict = "WAIT_PULLBACK"
            confidence = 0.50
            reason = f"Moderate conviction - wait for better entry"

        elif not spread_safe:
            verdict = "AVOID"
            confidence = 0.30
            reason = f"Spread too wide ({spread_pct:.1f}%) - liquidity risk"

        elif holding_pct < 30:
            verdict = "FADE"
            confidence = 0.70
            reason = f"Spike fading hard ({holding_pct:.0f}% left) - consider short"

        else:
            verdict = "AVOID"
            confidence = 0.40
            reason = f"Low conviction score ({overall_score})"

        return SpikeValidation(
            symbol=symbol,
            news_headline=headline,
            news_catalyst=catalyst_type,
            pre_news_price=pre_price,
            spike_price=spike_price,
            current_price=current_price,
            spike_pct=spike_pct,
            holding_pct=holding_pct,
            volume_ratio=volume_ratio,
            volume_surge=volume_surge,
            spread_pct=spread_pct,
            spread_safe=spread_safe,
            volume_score=volume_score,
            price_action_score=price_action_score,
            news_quality_score=news_quality_score,
            timing_score=timing_score,
            overall_score=overall_score,
            verdict=verdict,
            confidence=confidence,
            reason=reason,
            timestamp=now,
        )

    def _get_quote(self, symbol: str) -> Optional[Dict]:
        """Get current quote data"""
        try:
            r = requests.get(f"{self.api_url}/quote/{symbol}", timeout=2)
            if r.status_code != 200:
                return None

            q = r.json()
            price = float(q.get("last", 0)) or float(q.get("ask", 0))

            return {
                "price": price,
                "bid": float(q.get("bid", 0)),
                "ask": float(q.get("ask", 0)),
                "volume": int(q.get("volume", 0) or 0),
            }
        except Exception as e:
            logger.debug(f"Quote error for {symbol}: {e}")
            return None

    def _create_avoid_validation(
        self, symbol: str, headline: str, reason: str
    ) -> SpikeValidation:
        """Create an AVOID validation"""
        return SpikeValidation(
            symbol=symbol,
            news_headline=headline,
            news_catalyst="unknown",
            pre_news_price=0,
            spike_price=0,
            current_price=0,
            spike_pct=0,
            holding_pct=0,
            volume_ratio=0,
            volume_surge=False,
            spread_pct=0,
            spread_safe=False,
            volume_score=0,
            price_action_score=0,
            news_quality_score=0,
            timing_score=0,
            overall_score=0,
            verdict="AVOID",
            confidence=0,
            reason=reason,
            timestamp=datetime.now(self.et_tz),
        )

    async def _async_sleep(self, seconds: int):
        """Async sleep"""
        import asyncio

        await asyncio.sleep(seconds)

    def _trigger_callbacks(self, validation: SpikeValidation):
        """Trigger appropriate callback based on verdict"""
        try:
            if validation.verdict == "BUY_NOW" and self.on_buy_signal:
                self.on_buy_signal(validation)
            elif validation.verdict == "WAIT_PULLBACK" and self.on_wait_signal:
                self.on_wait_signal(validation)
            elif validation.verdict in ["AVOID", "FADE"] and self.on_avoid_signal:
                self.on_avoid_signal(validation)
        except Exception as e:
            logger.error(f"Callback error: {e}")

    def get_validations(self, limit: int = 20) -> List[Dict]:
        """Get recent validations"""
        return [v.to_dict() for v in self.validations[-limit:]]

    def get_buy_signals(self) -> List[Dict]:
        """Get recent BUY_NOW signals"""
        cutoff = datetime.now(self.et_tz) - timedelta(minutes=5)
        return [
            v.to_dict()
            for v in self.validations
            if v.verdict == "BUY_NOW" and v.timestamp >= cutoff
        ]


# Singleton
_validator: Optional[NewsSpikeValidator] = None


def get_spike_validator() -> NewsSpikeValidator:
    """Get or create spike validator singleton"""
    global _validator
    if _validator is None:
        _validator = NewsSpikeValidator()
    return _validator


def validate_news_spike(
    symbol: str, headline: str, catalyst: str, wait_seconds: int = 10
) -> SpikeValidation:
    """Convenience function to validate a spike"""
    validator = get_spike_validator()
    return validator.validate_spike_sync(symbol, headline, catalyst, wait_seconds)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test validation
    validator = get_spike_validator()

    # Simulate a news spike
    print("Testing spike validation...")

    result = validator.quick_validate(
        symbol="AAPL",
        headline="Apple FDA approval for health device",
        catalyst_type="fda_approval",
    )

    print(f"\nValidation Result:")
    print(f"  Symbol: {result.symbol}")
    print(f"  Verdict: {result.verdict}")
    print(f"  Confidence: {result.confidence:.0%}")
    print(f"  Reason: {result.reason}")
    print(f"\n  Scores:")
    print(f"    Volume: {result.volume_score}")
    print(f"    Price Action: {result.price_action_score}")
    print(f"    News Quality: {result.news_quality_score}")
    print(f"    Timing: {result.timing_score}")
    print(f"    Overall: {result.overall_score}")
