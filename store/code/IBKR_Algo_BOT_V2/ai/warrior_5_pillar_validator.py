"""
Warrior Trading 5-Pillar Validation System
===========================================
Every symbol must pass ALL 5 pillars before entering the worklist.

PILLARS:
1. PRICE - $1.00 to $20.00
2. VOLUME - Rel Vol >= 3.0x OR Absolute >= 1M
3. FLOAT - <= 20M shares (preferred <= 10M)
4. CATALYST - Gap >= 5% OR news OR unusual volume
5. TECHNICAL - Near HOD, VWAP reclaim, or pre-market high break

Selection quality > signal quantity
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class PillarResult(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    DEGRADED = "DEGRADED"  # Pass with warning
    UNKNOWN = "UNKNOWN"    # Data missing, allow provisionally


@dataclass
class PillarValidation:
    """Result of a single pillar check"""
    pillar: str
    passed: bool
    value: float = 0.0
    threshold: float = 0.0
    reason: str = ""
    data_quality: str = "OK"


@dataclass
class ValidationResult:
    """Complete 5-pillar validation result"""
    symbol: str
    passed: bool
    pillars: Dict[str, PillarValidation] = field(default_factory=dict)
    fail_reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    timestamp: str = ""
    data_quality: str = "OK"

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "passed": self.passed,
            "pillars": {
                k: {
                    "passed": v.passed,
                    "value": v.value,
                    "threshold": v.threshold,
                    "reason": v.reason,
                    "data_quality": v.data_quality
                } for k, v in self.pillars.items()
            },
            "fail_reasons": self.fail_reasons,
            "warnings": self.warnings,
            "timestamp": self.timestamp,
            "data_quality": self.data_quality
        }


class Warrior5PillarValidator:
    """
    Validates symbols against Ross Cameron's 5 Warrior Trading pillars.

    NO symbol enters the worklist without passing ALL 5 pillars.
    This is discipline, not optimization.
    """

    def __init__(self):
        # Pillar 1: Price thresholds
        self.min_price = 1.00
        self.max_price = 20.00

        # Pillar 2: Volume thresholds
        self.min_rel_vol = 3.0
        self.min_abs_volume = 1_000_000

        # Pillar 3: Float thresholds
        self.max_float = 20_000_000
        self.preferred_float = 10_000_000

        # Pillar 4: Catalyst thresholds
        self.min_gap_percent = 5.0

        # Pillar 5: Technical thresholds
        self.max_hod_distance_pct = 5.0  # Within 5% of HOD

        # Validation log
        self.validation_log: List[ValidationResult] = []

        logger.info("Warrior 5-Pillar Validator initialized")

    def validate(
        self,
        symbol: str,
        price: float,
        volume: int,
        avg_volume: int = 0,
        float_shares: int = 0,
        gap_percent: float = 0.0,
        has_news: bool = False,
        hod: float = 0.0,
        vwap: float = 0.0,
        premarket_high: float = 0.0
    ) -> ValidationResult:
        """
        Validate a symbol against all 5 pillars.

        Returns ValidationResult with pass/fail and reasons.
        """
        result = ValidationResult(
            symbol=symbol.upper(),
            passed=True,
            timestamp=datetime.now().isoformat()
        )

        # PILLAR 1: PRICE
        p1 = self._validate_price(price)
        result.pillars["price"] = p1
        if not p1.passed:
            result.passed = False
            result.fail_reasons.append(f"PILLAR_FAIL_PRICE: ${price:.2f} outside $1-$20 range")

        # PILLAR 2: VOLUME
        p2 = self._validate_volume(volume, avg_volume)
        result.pillars["volume"] = p2
        if not p2.passed:
            result.passed = False
            result.fail_reasons.append(f"PILLAR_FAIL_VOLUME: Vol {volume:,}, RelVol {p2.value:.1f}x < 3.0x")
        if p2.data_quality == "DEGRADED":
            result.warnings.append("Volume data degraded - using estimates")
            result.data_quality = "DEGRADED"

        # PILLAR 3: FLOAT
        p3 = self._validate_float(float_shares)
        result.pillars["float"] = p3
        if not p3.passed:
            result.passed = False
            result.fail_reasons.append(f"PILLAR_FAIL_FLOAT: {float_shares/1e6:.1f}M > 20M max")
        if p3.data_quality == "UNKNOWN":
            result.warnings.append("Float unknown - provisional entry")

        # PILLAR 4: CATALYST
        p4 = self._validate_catalyst(gap_percent, has_news, volume, avg_volume)
        result.pillars["catalyst"] = p4
        if not p4.passed:
            result.passed = False
            result.fail_reasons.append(f"PILLAR_FAIL_CATALYST: Gap {gap_percent:.1f}% < 5%, no news")

        # PILLAR 5: TECHNICAL
        p5 = self._validate_technical(price, hod, vwap, premarket_high)
        result.pillars["technical"] = p5
        if not p5.passed:
            result.passed = False
            result.fail_reasons.append(f"PILLAR_FAIL_TECHNICAL: Not near HOD/VWAP/PM high")

        # Log result
        self.validation_log.append(result)
        if len(self.validation_log) > 500:
            self.validation_log = self.validation_log[-500:]

        # Emit log
        if result.passed:
            logger.info(f"SCANNER_VALIDATION: {symbol} PASSED all 5 pillars")
        else:
            logger.warning(f"SCANNER_VALIDATION: {symbol} FAILED - {result.fail_reasons}")

        return result

    def _validate_price(self, price: float) -> PillarValidation:
        """Pillar 1: Price must be $1-$20"""
        passed = self.min_price <= price <= self.max_price
        return PillarValidation(
            pillar="price",
            passed=passed,
            value=price,
            threshold=self.max_price,
            reason="" if passed else f"Price ${price:.2f} outside range"
        )

    def _validate_volume(self, volume: int, avg_volume: int) -> PillarValidation:
        """Pillar 2: Relative Volume >= 3x OR Absolute >= 1M"""

        # Calculate relative volume
        if avg_volume > 0:
            rel_vol = volume / avg_volume
            data_quality = "OK"
        else:
            # Fallback: estimate avg_volume as volume/2 (degraded)
            rel_vol = 2.0  # Assume moderate if unknown
            data_quality = "DEGRADED"

        # Pass if rel_vol >= 3x OR absolute >= 1M
        passed = rel_vol >= self.min_rel_vol or volume >= self.min_abs_volume

        return PillarValidation(
            pillar="volume",
            passed=passed,
            value=rel_vol,
            threshold=self.min_rel_vol,
            reason="" if passed else f"RelVol {rel_vol:.1f}x < 3.0x, Vol {volume:,} < 1M",
            data_quality=data_quality
        )

    def _validate_float(self, float_shares: int) -> PillarValidation:
        """Pillar 3: Float <= 20M (preferred <= 10M)"""

        if float_shares == 0:
            # Unknown float - allow provisional entry
            return PillarValidation(
                pillar="float",
                passed=True,  # Provisional pass
                value=0,
                threshold=self.max_float,
                reason="Float unknown - provisional",
                data_quality="UNKNOWN"
            )

        passed = float_shares <= self.max_float

        return PillarValidation(
            pillar="float",
            passed=passed,
            value=float_shares / 1e6,  # In millions
            threshold=self.max_float / 1e6,
            reason="" if passed else f"Float {float_shares/1e6:.1f}M > 20M max"
        )

    def _validate_catalyst(
        self,
        gap_percent: float,
        has_news: bool,
        volume: int,
        avg_volume: int
    ) -> PillarValidation:
        """Pillar 4: Must have gap >= 5% OR news OR unusual volume"""

        has_gap = gap_percent >= self.min_gap_percent
        has_unusual_volume = (volume / avg_volume >= 5.0) if avg_volume > 0 else False

        passed = has_gap or has_news or has_unusual_volume

        reasons = []
        if has_gap:
            reasons.append(f"Gap {gap_percent:.1f}%")
        if has_news:
            reasons.append("News catalyst")
        if has_unusual_volume:
            reasons.append("Unusual volume")

        return PillarValidation(
            pillar="catalyst",
            passed=passed,
            value=gap_percent,
            threshold=self.min_gap_percent,
            reason=", ".join(reasons) if passed else "No catalyst detected"
        )

    def _validate_technical(
        self,
        price: float,
        hod: float,
        vwap: float,
        premarket_high: float
    ) -> PillarValidation:
        """
        Pillar 5: Technical context - not dead
        Must have at least one:
        - Near HOD (within 5%)
        - Above VWAP
        - Break of pre-market high
        """

        near_hod = False
        above_vwap = False
        pm_break = False

        if hod > 0:
            distance_from_hod = ((hod - price) / hod) * 100
            near_hod = distance_from_hod <= self.max_hod_distance_pct

        if vwap > 0:
            above_vwap = price >= vwap

        if premarket_high > 0:
            pm_break = price >= premarket_high

        # In pre-market, be lenient - if we have gap that's enough
        # Technical context is hard to judge pre-market
        # Pass if ANY condition met, or if we're missing data
        passed = near_hod or above_vwap or pm_break

        # Pre-market leniency: if no HOD/VWAP data, pass provisionally
        if hod == 0 and vwap == 0 and premarket_high == 0:
            passed = True
            reason = "Pre-market - technical provisional"
        else:
            reasons = []
            if near_hod:
                reasons.append("Near HOD")
            if above_vwap:
                reasons.append("Above VWAP")
            if pm_break:
                reasons.append("PM high break")
            reason = ", ".join(reasons) if passed else "Not near HOD/VWAP/PM high"

        return PillarValidation(
            pillar="technical",
            passed=passed,
            value=price,
            threshold=hod if hod > 0 else 0,
            reason=reason
        )

    def get_validation_log(self, limit: int = 50) -> List[dict]:
        """Get recent validation results"""
        return [v.to_dict() for v in self.validation_log[-limit:]]

    def get_failed_symbols(self) -> List[dict]:
        """Get symbols that failed validation"""
        return [v.to_dict() for v in self.validation_log if not v.passed]

    def get_passed_symbols(self) -> List[dict]:
        """Get symbols that passed validation"""
        return [v.to_dict() for v in self.validation_log if v.passed]

    def clear_log(self):
        """Clear validation log (call at session reset)"""
        self.validation_log = []
        logger.info("Validation log cleared")


# Singleton instance
_validator: Optional[Warrior5PillarValidator] = None


def get_5_pillar_validator() -> Warrior5PillarValidator:
    """Get the singleton validator instance"""
    global _validator
    if _validator is None:
        _validator = Warrior5PillarValidator()
    return _validator


async def validate_symbol_for_worklist(symbol: str) -> Tuple[bool, ValidationResult]:
    """
    Validate a symbol before adding to worklist.

    Fetches required data and runs 5-pillar validation.
    Returns (passed, result).
    """
    import httpx
    import yfinance as yf

    validator = get_5_pillar_validator()
    symbol = symbol.upper()

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Get price data
            try:
                price_resp = await client.get(f"http://localhost:9100/api/price/{symbol}")
                if price_resp.status_code != 200:
                    return False, ValidationResult(
                        symbol=symbol,
                        passed=False,
                        fail_reasons=["Could not fetch price data"],
                        timestamp=datetime.now().isoformat()
                    )
                quote = price_resp.json()
            except Exception as e:
                logger.warning(f"Price API failed for {symbol}, using yfinance: {e}")
                # Fallback to yfinance
                ticker = yf.Ticker(symbol)
                info = ticker.fast_info
                quote = {
                    "price": getattr(info, 'last_price', 0) or 0,
                    "volume": getattr(info, 'last_volume', 0) or 0,
                    "change_percent": 0  # Will be estimated
                }

            price = quote.get("price", 0)
            volume = quote.get("volume", 0)
            change_pct = quote.get("change_percent", 0)

            # Get avg_volume and float from yfinance (more reliable)
            avg_volume = 0
            float_shares = 0
            prev_close = 0
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info

                # Try info dict first
                avg_volume = info.get("averageVolume", 0) or info.get("averageDailyVolume10Day", 0) or 0
                float_shares = info.get("floatShares", 0) or 0
                prev_close = info.get("previousClose", 0) or info.get("regularMarketPreviousClose", 0)

                # If avg_volume not in info, calculate from history (small caps often missing)
                if avg_volume == 0:
                    try:
                        hist = ticker.history(period="5d")
                        if len(hist) > 0 and "Volume" in hist.columns:
                            avg_volume = int(hist["Volume"].mean())
                            logger.info(f"Calculated avg_volume from 5d history for {symbol}: {avg_volume:,}")
                    except Exception as he:
                        logger.warning(f"History lookup failed for {symbol}: {he}")

                # If prev_close not in info, get from history
                if prev_close == 0:
                    try:
                        hist = ticker.history(period="2d")
                        if len(hist) >= 2:
                            prev_close = hist["Close"].iloc[-2]  # Previous day's close
                        elif len(hist) == 1:
                            prev_close = hist["Close"].iloc[-1]
                    except:
                        pass

                # Calculate change_pct from prev_close if we didn't get it from API
                if change_pct == 0 and prev_close > 0 and price > 0:
                    change_pct = ((price - prev_close) / prev_close) * 100

                logger.info(f"yfinance data for {symbol}: avg_vol={avg_volume:,}, float={float_shares:,}, prev_close={prev_close:.2f}")
            except Exception as e:
                logger.warning(f"yfinance lookup failed for {symbol}: {e}")

            # Check for news via API
            has_news = False
            try:
                news_resp = await client.get(f"http://localhost:9100/api/news-log/symbol/{symbol}")
                if news_resp.status_code == 200:
                    news_data = news_resp.json()
                    has_news = len(news_data.get("news", [])) > 0
            except:
                pass

            # Run validation
            result = validator.validate(
                symbol=symbol,
                price=price,
                volume=volume,
                avg_volume=avg_volume,
                float_shares=float_shares,
                gap_percent=change_pct,
                has_news=has_news,
                hod=0,  # Pre-market - no HOD yet
                vwap=0,  # Pre-market - no VWAP yet
                premarket_high=0
            )

            return result.passed, result

    except Exception as e:
        logger.error(f"Validation error for {symbol}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False, ValidationResult(
            symbol=symbol,
            passed=False,
            fail_reasons=[str(e)],
            timestamp=datetime.now().isoformat()
        )
