"""
Trade Narrative Module
Generates plain-English reasoning for each trade decision.
Improves transparency, debugging, and trader trust.

Part of the Next-Gen AI Logic Blueprint.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class SignalType(str, Enum):
    """Types of signals that can trigger trades"""
    BREAKOUT = "breakout"
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    VWAP_CROSS = "vwap_cross"
    NEWS_CATALYST = "news_catalyst"
    AI_PREDICTION = "ai_prediction"
    TRAILING_STOP = "trailing_stop"
    TAKE_PROFIT = "take_profit"
    STOP_LOSS = "stop_loss"
    REVERSAL = "reversal"
    UNKNOWN = "unknown"


class ConfidenceLevel(str, Enum):
    """Human-readable confidence levels"""
    VERY_HIGH = "very_high"    # 90%+
    HIGH = "high"              # 75-90%
    MODERATE = "moderate"      # 50-75%
    LOW = "low"                # 25-50%
    VERY_LOW = "very_low"      # <25%


@dataclass
class TradeNarrative:
    """Complete narrative for a trade decision"""
    symbol: str
    action: str                     # BUY, SELL, HOLD
    signal_type: SignalType
    confidence: float               # 0-1
    confidence_level: ConfidenceLevel
    narrative: str                  # Plain-English explanation
    supporting_factors: List[str]   # Reasons FOR the trade
    opposing_factors: List[str]     # Reasons AGAINST (risks)
    regime: str                     # Market regime
    session: str                    # Market session
    price: float
    target_price: Optional[float]
    stop_price: Optional[float]
    risk_reward: Optional[float]
    timestamp: str

    def to_dict(self) -> Dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class TradeNarrativeGenerator:
    """
    Generates human-readable narratives for trade decisions.
    Combines multiple signals and factors into coherent explanations.
    """

    def __init__(self):
        self.log_path = Path("store/trade_narratives")
        self.log_path.mkdir(parents=True, exist_ok=True)
        self.narratives: List[TradeNarrative] = []
        self.max_history = 500

        logger.info("TradeNarrativeGenerator initialized")

    def _get_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """Convert numeric confidence to human-readable level"""
        if confidence >= 0.90:
            return ConfidenceLevel.VERY_HIGH
        elif confidence >= 0.75:
            return ConfidenceLevel.HIGH
        elif confidence >= 0.50:
            return ConfidenceLevel.MODERATE
        elif confidence >= 0.25:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW

    def _determine_signal_type(self, factors: Dict) -> SignalType:
        """Determine the primary signal type from factors"""
        if factors.get("is_trailing_stop"):
            return SignalType.TRAILING_STOP
        if factors.get("is_take_profit"):
            return SignalType.TAKE_PROFIT
        if factors.get("is_stop_loss"):
            return SignalType.STOP_LOSS
        if factors.get("is_reversal"):
            return SignalType.REVERSAL
        if factors.get("news_trigger"):
            return SignalType.NEWS_CATALYST
        if factors.get("vwap_signal"):
            return SignalType.VWAP_CROSS
        if factors.get("breakout"):
            return SignalType.BREAKOUT
        if factors.get("momentum_signal"):
            return SignalType.MOMENTUM
        if factors.get("mean_reversion"):
            return SignalType.MEAN_REVERSION
        if factors.get("ai_prediction"):
            return SignalType.AI_PREDICTION
        return SignalType.UNKNOWN

    def _build_narrative(self, symbol: str, action: str, factors: Dict,
                        signal_type: SignalType) -> str:
        """Build plain-English narrative from factors"""
        parts = []

        # Opening statement
        confidence = factors.get("confidence", 0)
        conf_word = "strong" if confidence >= 0.75 else "moderate" if confidence >= 0.5 else "weak"

        if action == "BUY":
            parts.append(f"Initiating LONG position in {symbol} based on {conf_word} {signal_type.value} signal.")
        elif action == "SELL":
            if factors.get("is_exit", False):
                parts.append(f"Closing position in {symbol} due to {signal_type.value}.")
            else:
                parts.append(f"Initiating SHORT position in {symbol} based on {conf_word} {signal_type.value} signal.")
        else:
            parts.append(f"HOLD on {symbol} - no actionable signal detected.")

        # Technical factors
        if factors.get("above_vwap"):
            parts.append("Price is trading above VWAP, indicating bullish intraday strength.")
        elif factors.get("below_vwap"):
            parts.append("Price is trading below VWAP, indicating bearish intraday pressure.")

        if factors.get("rsi"):
            rsi = factors["rsi"]
            if rsi >= 70:
                parts.append(f"RSI at {rsi:.0f} signals overbought conditions (caution on longs).")
            elif rsi <= 30:
                parts.append(f"RSI at {rsi:.0f} signals oversold conditions (potential bounce).")
            else:
                parts.append(f"RSI at {rsi:.0f} is in neutral territory.")

        if factors.get("volume_surge"):
            parts.append("Volume surge detected, confirming the move.")

        if factors.get("breakout"):
            parts.append(f"Breakout above key resistance at ${factors.get('breakout_level', 0):.2f}.")

        if factors.get("breakdown"):
            parts.append(f"Breakdown below key support at ${factors.get('breakdown_level', 0):.2f}.")

        # Sentiment/News
        if factors.get("news_headline"):
            parts.append(f"News catalyst: \"{factors['news_headline'][:50]}...\"")

        if factors.get("sentiment_score"):
            sent = factors["sentiment_score"]
            sent_word = "bullish" if sent > 0.3 else "bearish" if sent < -0.3 else "neutral"
            parts.append(f"Sentiment analysis shows {sent_word} bias ({sent:+.2f}).")

        # Regime context
        if factors.get("regime"):
            parts.append(f"Market regime: {factors['regime']}.")

        # Exit-specific narratives
        if factors.get("is_trailing_stop"):
            drop_pct = factors.get("drop_from_high", 0)
            high = factors.get("high_watermark", 0)
            parts.append(f"Trailing stop triggered after {drop_pct:.1f}% drop from high of ${high:.2f}.")

        if factors.get("is_take_profit"):
            gain = factors.get("gain_percent", 0)
            parts.append(f"Take profit triggered at +{gain:.1f}% gain.")

        if factors.get("is_stop_loss"):
            loss = factors.get("loss_percent", 0)
            parts.append(f"Stop loss triggered at -{abs(loss):.1f}% loss to protect capital.")

        if factors.get("is_reversal"):
            signals = factors.get("reversal_signals", [])
            parts.append(f"Reversal detected: {', '.join(signals)}.")

        return " ".join(parts)

    def _extract_supporting_factors(self, factors: Dict) -> List[str]:
        """Extract list of factors supporting the trade"""
        supporting = []

        if factors.get("trend_aligned"):
            supporting.append("Trade aligned with overall trend")
        if factors.get("above_vwap") and factors.get("action") == "BUY":
            supporting.append("Price above VWAP (bullish)")
        if factors.get("volume_surge"):
            supporting.append("High volume confirms move")
        if factors.get("momentum_positive") and factors.get("action") == "BUY":
            supporting.append("Positive momentum")
        if factors.get("sentiment_positive") and factors.get("action") == "BUY":
            supporting.append("Positive sentiment")
        if factors.get("breakout"):
            supporting.append("Technical breakout")
        if factors.get("ai_confidence", 0) >= 0.7:
            supporting.append(f"High AI confidence ({factors['ai_confidence']*100:.0f}%)")
        if factors.get("regime") in ["trending_up"] and factors.get("action") == "BUY":
            supporting.append("Favorable market regime")
        if factors.get("profit_locked"):
            supporting.append("Profit already locked in")

        return supporting

    def _extract_opposing_factors(self, factors: Dict) -> List[str]:
        """Extract list of factors against the trade (risks)"""
        opposing = []

        if factors.get("rsi", 50) >= 70 and factors.get("action") == "BUY":
            opposing.append("RSI overbought - pullback risk")
        if factors.get("rsi", 50) <= 30 and factors.get("action") == "SELL":
            opposing.append("RSI oversold - bounce risk")
        if factors.get("regime") == "volatile":
            opposing.append("Volatile market conditions")
        if factors.get("regime") == "ranging":
            opposing.append("Choppy/ranging market")
        if factors.get("low_volume"):
            opposing.append("Low volume - weak confirmation")
        if factors.get("counter_trend"):
            opposing.append("Trade against overall trend")
        if factors.get("wide_spread"):
            opposing.append("Wide bid-ask spread")
        if factors.get("sentiment_negative") and factors.get("action") == "BUY":
            opposing.append("Negative sentiment headwind")
        if factors.get("near_resistance") and factors.get("action") == "BUY":
            opposing.append("Near resistance level")
        if factors.get("near_support") and factors.get("action") == "SELL":
            opposing.append("Near support level")

        return opposing

    def generate(self, symbol: str, action: str, price: float,
                factors: Dict, regime: str = "unknown",
                session: str = "unknown") -> TradeNarrative:
        """
        Generate a complete trade narrative.

        Args:
            symbol: Stock symbol
            action: BUY, SELL, or HOLD
            price: Current/execution price
            factors: Dict of all relevant factors
            regime: Market regime
            session: Market session

        Returns:
            TradeNarrative object
        """
        confidence = factors.get("confidence", factors.get("ai_confidence", 0.5))
        signal_type = self._determine_signal_type(factors)

        # Build narrative
        narrative = self._build_narrative(symbol, action, factors, signal_type)
        supporting = self._extract_supporting_factors(factors)
        opposing = self._extract_opposing_factors(factors)

        # Calculate risk/reward if possible
        target = factors.get("target_price")
        stop = factors.get("stop_price")
        risk_reward = None

        if target and stop and price:
            if action == "BUY" and target > price and stop < price:
                reward = target - price
                risk = price - stop
                risk_reward = reward / risk if risk > 0 else 0
            elif action == "SELL" and target < price and stop > price:
                reward = price - target
                risk = stop - price
                risk_reward = reward / risk if risk > 0 else 0

        trade_narrative = TradeNarrative(
            symbol=symbol,
            action=action,
            signal_type=signal_type,
            confidence=confidence,
            confidence_level=self._get_confidence_level(confidence),
            narrative=narrative,
            supporting_factors=supporting,
            opposing_factors=opposing,
            regime=regime,
            session=session,
            price=price,
            target_price=target,
            stop_price=stop,
            risk_reward=round(risk_reward, 2) if risk_reward else None,
            timestamp=datetime.now().isoformat()
        )

        # Store and log
        self._store_narrative(trade_narrative)

        return trade_narrative

    def _store_narrative(self, narrative: TradeNarrative):
        """Store narrative in memory and optionally to file"""
        self.narratives.append(narrative)

        # Trim history
        if len(self.narratives) > self.max_history:
            self.narratives = self.narratives[-self.max_history:]

        # Log to file
        try:
            log_file = self.log_path / f"narratives_{datetime.now().strftime('%Y%m%d')}.jsonl"
            with open(log_file, "a") as f:
                f.write(narrative.to_json() + "\n")
        except Exception as e:
            logger.warning(f"Could not write narrative to file: {e}")

    def get_recent_narratives(self, symbol: str = None,
                             limit: int = 10) -> List[TradeNarrative]:
        """Get recent narratives, optionally filtered by symbol"""
        if symbol:
            filtered = [n for n in self.narratives if n.symbol == symbol]
        else:
            filtered = self.narratives

        return filtered[-limit:]

    def generate_exit_narrative(self, symbol: str, entry_price: float,
                               exit_price: float, quantity: int,
                               exit_reason: str, factors: Dict) -> TradeNarrative:
        """
        Generate narrative specifically for exits.

        Args:
            symbol: Stock symbol
            entry_price: Original entry price
            exit_price: Exit/current price
            quantity: Position size
            exit_reason: Reason for exit
            factors: Additional factors
        """
        pnl = (exit_price - entry_price) * quantity
        pnl_pct = ((exit_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0

        # Enrich factors for exit
        factors.update({
            "is_exit": True,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl": pnl,
            "pnl_percent": pnl_pct,
            "exit_reason": exit_reason
        })

        if "trailing_stop" in exit_reason.lower():
            factors["is_trailing_stop"] = True
        if "take_profit" in exit_reason.lower():
            factors["is_take_profit"] = True
        if "stop_loss" in exit_reason.lower():
            factors["is_stop_loss"] = True
        if "reversal" in exit_reason.lower():
            factors["is_reversal"] = True

        return self.generate(
            symbol=symbol,
            action="SELL",
            price=exit_price,
            factors=factors,
            regime=factors.get("regime", "unknown"),
            session=factors.get("session", "unknown")
        )


# Singleton instance
_narrative_generator: Optional[TradeNarrativeGenerator] = None


def get_narrative_generator() -> TradeNarrativeGenerator:
    """Get or create the narrative generator singleton"""
    global _narrative_generator
    if _narrative_generator is None:
        _narrative_generator = TradeNarrativeGenerator()
    return _narrative_generator
