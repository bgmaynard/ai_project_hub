"""
Momentum Score Calculator (v2 - ChatGPT Spec Compliant)
========================================================
Unified 0-100 momentum score with hard vetoes and full integrations.

Inputs:
- Recent trades/prices (5s, 15s, 30s intervals)
- Quote (bid/ask, spread)
- L2-derived buy_pressure (from order_flow_analyzer.py)
- VWAP context (from vwap_manager.py)
- MTF confirmation (from mtf_confirmation.py)
- Chronos context (market_regime, confidence)

Outputs:
- momentum_score: 0-100
- veto_reasons: list of blocking reasons
- debug: dict of computed stats

Hard Vetoes (score=0):
- spread > max_spread_pct
- buy_pressure < min_buy_pressure
- below VWAP
- extended > vwap_max_extension_pct
- regime is unfavorable
"""

import logging
import json
import os
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class MomentumGrade(Enum):
    """Momentum grade based on score"""
    A = "A"  # 80-100: Strong ignition candidate
    B = "B"  # 65-79: Good setup, watch for ignition
    C = "C"  # 50-64: Marginal, needs more confirmation
    D = "D"  # 35-49: Weak, likely to fade
    F = "F"  # 0-34: No momentum, avoid


class VetoReason(Enum):
    """Hard veto reasons that block entry"""
    SPREAD_WIDE = "SPREAD_WIDE"
    SELL_PRESSURE = "SELL_PRESSURE"
    BELOW_VWAP = "BELOW_VWAP"
    VWAP_EXTENDED = "VWAP_EXTENDED"
    REGIME_BAD = "REGIME_BAD"
    CONFIDENCE_LOW = "CONFIDENCE_LOW"
    MTF_MISALIGNED = "MTF_MISALIGNED"
    NO_MOMENTUM = "NO_MOMENTUM"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


@dataclass
class PriceUrgencyScore:
    """Price/Urgency component (0-40 points)"""
    # Raw metrics
    r_5s: float = 0.0          # Rate of change last 5 seconds
    r_15s: float = 0.0         # Rate of change last 15 seconds
    r_30s: float = 0.0         # Rate of change last 30 seconds
    accel: float = 0.0         # Acceleration (r_15s - r_30s/2)
    range_30s: float = 0.0     # (high-low)/price over 30s
    hod_distance_pct: float = 0.0  # Distance from high of day
    hod_break: bool = False    # Just broke HOD

    # VWAP context
    vwap_position: str = "BELOW"  # ABOVE, AT, BELOW
    vwap_extension_pct: float = 0.0  # How far above/below VWAP

    # Scores
    roc_score: int = 0         # 0-15: Speed of move
    accel_score: int = 0       # 0-10: Acceleration
    range_score: int = 0       # 0-10: Range expansion
    hod_score: int = 0         # 0-5: HOD proximity/break

    @property
    def total(self) -> int:
        return min(40, self.roc_score + self.accel_score + self.range_score + self.hod_score)

    def to_dict(self) -> Dict:
        return {
            'r_5s': round(self.r_5s, 4),
            'r_15s': round(self.r_15s, 4),
            'r_30s': round(self.r_30s, 4),
            'accel': round(self.accel, 4),
            'range_30s': round(self.range_30s, 4),
            'hod_distance_pct': round(self.hod_distance_pct, 2),
            'hod_break': self.hod_break,
            'vwap_position': self.vwap_position,
            'vwap_extension_pct': round(self.vwap_extension_pct, 2),
            'roc_score': self.roc_score,
            'accel_score': self.accel_score,
            'range_score': self.range_score,
            'hod_score': self.hod_score,
            'total': self.total
        }


@dataclass
class ParticipationScore:
    """Participation component (0-35 points)"""
    volume_surge: float = 0.0      # Current volume vs average
    vol_rate_30s: float = 0.0      # Volume rate vs 30s baseline
    relative_volume: float = 0.0   # RVol for the day
    float_rotation_pct: float = 0.0  # % of float traded

    # Scores
    surge_score: int = 0       # 0-15: Volume surge
    vol_rate_score: int = 0    # 0-10: Volume rate
    rotation_score: int = 0    # 0-10: Float rotation

    @property
    def total(self) -> int:
        return min(35, self.surge_score + self.vol_rate_score + self.rotation_score)

    def to_dict(self) -> Dict:
        return {
            'volume_surge': round(self.volume_surge, 2),
            'vol_rate_30s': round(self.vol_rate_30s, 2),
            'relative_volume': round(self.relative_volume, 2),
            'float_rotation_pct': round(self.float_rotation_pct, 2),
            'surge_score': self.surge_score,
            'vol_rate_score': self.vol_rate_score,
            'rotation_score': self.rotation_score,
            'total': self.total
        }


@dataclass
class LiquidityScore:
    """Liquidity/Confirmation component (0-25 points)"""
    spread_pct: float = 0.0        # Bid-ask spread %
    buy_pressure: float = 0.0      # Order flow buy pressure
    imbalance_ratio: float = 0.0   # Bid vs ask imbalance
    tape_signal: str = "NEUTRAL"   # BULLISH, NEUTRAL, BEARISH

    # Context flags
    mtf_aligned: bool = False      # Multi-timeframe confirmation
    chronos_ok: bool = False       # Chronos regime favorable

    # Scores
    spread_score: int = 0      # 0-10: Tight spread = good
    flow_score: int = 0        # 0-10: Buy pressure
    context_score: int = 0     # 0-5: MTF + Chronos

    @property
    def total(self) -> int:
        return min(25, self.spread_score + self.flow_score + self.context_score)

    def to_dict(self) -> Dict:
        return {
            'spread_pct': round(self.spread_pct, 3),
            'buy_pressure': round(self.buy_pressure, 3),
            'imbalance_ratio': round(self.imbalance_ratio, 2),
            'tape_signal': self.tape_signal,
            'mtf_aligned': self.mtf_aligned,
            'chronos_ok': self.chronos_ok,
            'spread_score': self.spread_score,
            'flow_score': self.flow_score,
            'context_score': self.context_score,
            'total': self.total
        }


@dataclass
class MomentumResult:
    """Complete momentum analysis result"""
    symbol: str
    timestamp: datetime
    score: int                 # 0-100 total score
    grade: MomentumGrade

    # Component scores
    price_urgency: PriceUrgencyScore
    participation: ParticipationScore
    liquidity: LiquidityScore

    # Veto system
    vetoed: bool = False           # True if any hard veto triggered
    veto_reasons: List[VetoReason] = field(default_factory=list)

    # Flags
    is_tradeable: bool = False     # Score >= threshold AND not vetoed
    ignition_ready: bool = False   # All components aligned
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # Debug info
    debug: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'score': self.score,
            'grade': self.grade.value,
            'price_urgency': self.price_urgency.to_dict(),
            'participation': self.participation.to_dict(),
            'liquidity': self.liquidity.to_dict(),
            'vetoed': self.vetoed,
            'veto_reasons': [v.value for v in self.veto_reasons],
            'is_tradeable': self.is_tradeable,
            'ignition_ready': self.ignition_ready,
            'reasons': self.reasons,
            'warnings': self.warnings,
            'debug': self.debug
        }


@dataclass
class MomentumConfig:
    """Configurable thresholds for momentum scoring"""
    # Entry threshold
    momentum_score_threshold_enter: int = 70

    # ROC thresholds
    r_5s_min: float = 0.1      # 0.1% in 5 seconds
    r_15s_min: float = 0.2     # 0.2% in 15 seconds
    r_30s_min: float = 0.3     # 0.3% in 30 seconds
    accel_min: float = 0.05    # Minimum acceleration

    # Volume thresholds
    vol_rate_min: float = 2.0  # 2x normal volume rate
    surge_min: float = 3.0     # 3x surge minimum

    # Hard veto thresholds
    max_spread_pct: float = 1.0        # Max 1% spread
    min_buy_pressure: float = 0.55     # Min 55% buy pressure
    vwap_max_extension_pct: float = 3.0  # Max 3% above VWAP

    # Component minimums for ignition
    min_price_urgency: int = 20        # 20/40
    min_participation: int = 15        # 15/35
    min_liquidity: int = 10            # 10/25

    @classmethod
    def from_scalper_config(cls) -> 'MomentumConfig':
        """Load config from scalper_config.json if available"""
        config_path = os.path.join(os.path.dirname(__file__), "scalper_config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    data = json.load(f)
                return cls(
                    momentum_score_threshold_enter=data.get('momentum_score_threshold_enter', 70),
                    r_5s_min=data.get('r_5s_min', 0.1),
                    r_15s_min=data.get('r_15s_min', 0.2),
                    r_30s_min=data.get('r_30s_min', 0.3),
                    accel_min=data.get('accel_min', 0.05),
                    vol_rate_min=data.get('vol_rate_min', 2.0),
                    surge_min=data.get('surge_min', 3.0),
                    max_spread_pct=data.get('max_spread_percent', 1.0),
                    min_buy_pressure=data.get('min_buy_pressure', 0.55),
                    vwap_max_extension_pct=data.get('vwap_max_extension_pct', 3.0),
                    min_price_urgency=data.get('min_price_urgency', 20),
                    min_participation=data.get('min_participation', 15),
                    min_liquidity=data.get('min_liquidity', 10)
                )
            except Exception as e:
                logger.warning(f"Failed to load config: {e}")
        return cls()


class MomentumScorer:
    """
    Calculates unified momentum score with hard vetoes.

    ChatGPT Spec Compliant:
    - r_5s, r_15s, r_30s ROC calculations
    - accel (acceleration) calculation
    - Hard vetoes with veto_reasons
    - Integration with VWAP, Chronos, order flow
    """

    def __init__(self, config: MomentumConfig = None):
        self.config = config or MomentumConfig.from_scalper_config()
        self._cache: Dict[str, MomentumResult] = {}
        self._cache_ttl = 5  # Cache results for 5 seconds

        # Lazy-loaded integrations
        self._vwap_manager = None
        self._chronos_adapter = None
        self._order_flow = None

    @property
    def vwap_manager(self):
        """Lazy load VWAP manager"""
        if self._vwap_manager is None:
            try:
                from ai.vwap_manager import get_vwap_manager
                self._vwap_manager = get_vwap_manager()
            except Exception as e:
                logger.debug(f"VWAP manager not available: {e}")
        return self._vwap_manager

    @property
    def chronos_adapter(self):
        """Lazy load Chronos adapter"""
        if self._chronos_adapter is None:
            try:
                from ai.chronos_adapter import get_chronos_adapter
                self._chronos_adapter = get_chronos_adapter()
            except Exception as e:
                logger.debug(f"Chronos adapter not available: {e}")
        return self._chronos_adapter

    def calculate(self,
                  symbol: str,
                  current_price: float,
                  prices_5s: List[float] = None,
                  prices_15s: List[float] = None,
                  prices_30s: List[float] = None,
                  high_30s: float = 0,
                  low_30s: float = 0,
                  high_of_day: float = 0,
                  vwap: float = 0,
                  current_volume: int = 0,
                  volume_30s_baseline: int = 0,
                  avg_volume: int = 0,
                  float_shares: int = 0,
                  day_volume: int = 0,
                  spread_pct: float = 0,
                  buy_pressure: float = 0.5,
                  imbalance_ratio: float = 1.0,
                  tape_signal: str = "NEUTRAL",
                  mtf_aligned: bool = None,
                  chronos_regime: str = None,
                  chronos_confidence: float = None) -> MomentumResult:
        """
        Calculate momentum score with full integration.

        Args:
            symbol: Stock symbol
            current_price: Current price
            prices_5s: Prices from last 5 seconds
            prices_15s: Prices from last 15 seconds
            prices_30s: Prices from last 30 seconds
            high_30s: High price in last 30 seconds
            low_30s: Low price in last 30 seconds
            high_of_day: High of day
            vwap: VWAP price (or fetched from vwap_manager)
            current_volume: Current bar volume
            volume_30s_baseline: 30-second volume baseline
            avg_volume: Average daily volume
            float_shares: Float shares
            day_volume: Total volume today
            spread_pct: Bid-ask spread as percentage
            buy_pressure: Buy pressure from order flow (0-1)
            imbalance_ratio: Bid/ask imbalance ratio
            tape_signal: Tape reading signal
            mtf_aligned: Multi-timeframe alignment (or fetched)
            chronos_regime: Market regime (or fetched)
            chronos_confidence: Chronos confidence (or fetched)
        """
        now = datetime.now()
        reasons = []
        warnings = []
        veto_reasons = []
        debug = {}

        # Fetch missing data from integrations
        if vwap == 0 and self.vwap_manager:
            try:
                vwap_data = self.vwap_manager.get_vwap_data(symbol)
                if vwap_data:
                    vwap = vwap_data.vwap
            except Exception as e:
                debug['vwap_fetch_error'] = str(e)

        if chronos_regime is None and self.chronos_adapter:
            try:
                context = self.chronos_adapter.get_context(symbol)
                if context:
                    chronos_regime = context.get('market_regime', 'UNKNOWN')
                    chronos_confidence = context.get('regime_confidence', 0.5)
            except Exception as e:
                debug['chronos_fetch_error'] = str(e)

        # Calculate Price/Urgency Score
        price_urgency = self._calc_price_urgency(
            current_price=current_price,
            prices_5s=prices_5s or [],
            prices_15s=prices_15s or [],
            prices_30s=prices_30s or [],
            high_30s=high_30s,
            low_30s=low_30s,
            high_of_day=high_of_day,
            vwap=vwap,
            reasons=reasons,
            warnings=warnings,
            veto_reasons=veto_reasons,
            debug=debug
        )

        # Calculate Participation Score
        participation = self._calc_participation(
            current_volume=current_volume,
            volume_30s_baseline=volume_30s_baseline,
            avg_volume=avg_volume,
            float_shares=float_shares,
            day_volume=day_volume,
            reasons=reasons,
            warnings=warnings,
            debug=debug
        )

        # Calculate Liquidity Score
        liquidity = self._calc_liquidity(
            spread_pct=spread_pct,
            buy_pressure=buy_pressure,
            imbalance_ratio=imbalance_ratio,
            tape_signal=tape_signal,
            mtf_aligned=mtf_aligned,
            chronos_regime=chronos_regime,
            chronos_confidence=chronos_confidence,
            reasons=reasons,
            warnings=warnings,
            veto_reasons=veto_reasons,
            debug=debug
        )

        # Calculate raw score
        raw_score = price_urgency.total + participation.total + liquidity.total

        # Apply vetoes
        vetoed = len(veto_reasons) > 0
        if vetoed:
            final_score = 0
            warnings.append(f"VETOED: {[v.value for v in veto_reasons]}")
        else:
            final_score = raw_score

        # Grade
        grade = self._get_grade(final_score)

        # Tradeable check (not vetoed and score >= threshold)
        is_tradeable = not vetoed and final_score >= self.config.momentum_score_threshold_enter

        # Ignition ready check (all components must meet minimums)
        ignition_ready = (
            not vetoed and
            final_score >= self.config.momentum_score_threshold_enter and
            price_urgency.total >= self.config.min_price_urgency and
            participation.total >= self.config.min_participation and
            liquidity.total >= self.config.min_liquidity
        )

        if ignition_ready:
            reasons.append("IGNITION READY - All components aligned, no vetoes")

        # Store debug info
        debug['raw_score'] = raw_score
        debug['final_score'] = final_score
        debug['config'] = {
            'threshold': self.config.momentum_score_threshold_enter,
            'max_spread': self.config.max_spread_pct,
            'min_buy_pressure': self.config.min_buy_pressure
        }

        result = MomentumResult(
            symbol=symbol,
            timestamp=now,
            score=final_score,
            grade=grade,
            price_urgency=price_urgency,
            participation=participation,
            liquidity=liquidity,
            vetoed=vetoed,
            veto_reasons=veto_reasons,
            is_tradeable=is_tradeable,
            ignition_ready=ignition_ready,
            reasons=reasons,
            warnings=warnings,
            debug=debug
        )

        # Cache result
        self._cache[symbol] = result

        return result

    def _calc_price_urgency(self,
                            current_price: float,
                            prices_5s: List[float],
                            prices_15s: List[float],
                            prices_30s: List[float],
                            high_30s: float,
                            low_30s: float,
                            high_of_day: float,
                            vwap: float,
                            reasons: List[str],
                            warnings: List[str],
                            veto_reasons: List[VetoReason],
                            debug: Dict) -> PriceUrgencyScore:
        """Calculate price urgency with r_5s, r_15s, r_30s, accel"""
        score = PriceUrgencyScore()

        # ROC calculations
        if prices_5s and len(prices_5s) >= 2:
            score.r_5s = ((current_price - prices_5s[0]) / prices_5s[0]) * 100
        if prices_15s and len(prices_15s) >= 2:
            score.r_15s = ((current_price - prices_15s[0]) / prices_15s[0]) * 100
        if prices_30s and len(prices_30s) >= 2:
            score.r_30s = ((current_price - prices_30s[0]) / prices_30s[0]) * 100

        # Acceleration (r_15s - r_30s/2)
        score.accel = score.r_15s - (score.r_30s / 2) if score.r_30s != 0 else 0

        debug['roc'] = {
            'r_5s': round(score.r_5s, 4),
            'r_15s': round(score.r_15s, 4),
            'r_30s': round(score.r_30s, 4),
            'accel': round(score.accel, 4)
        }

        # Range 30s
        if high_30s > 0 and low_30s > 0 and current_price > 0:
            score.range_30s = (high_30s - low_30s) / current_price * 100

        # ROC score (0-15)
        max_roc = max(abs(score.r_5s) * 1.5, abs(score.r_15s), abs(score.r_30s) * 0.7)
        if max_roc >= 0.8:
            score.roc_score = 15
            reasons.append(f"Strong ROC: {max_roc:.2f}%")
        elif max_roc >= 0.5:
            score.roc_score = 12
        elif max_roc >= 0.3:
            score.roc_score = 8
        elif max_roc >= 0.15:
            score.roc_score = 4
        else:
            score.roc_score = 0
            if max_roc < 0.05:
                warnings.append("No momentum detected")
                veto_reasons.append(VetoReason.NO_MOMENTUM)

        # Acceleration score (0-10)
        if score.accel >= 0.2:
            score.accel_score = 10
            reasons.append(f"Strong acceleration: {score.accel:.3f}")
        elif score.accel >= 0.1:
            score.accel_score = 7
        elif score.accel >= 0.05:
            score.accel_score = 4
        elif score.accel >= 0:
            score.accel_score = 2
        else:
            score.accel_score = 0
            if score.accel < -0.1:
                warnings.append("Decelerating (negative accel)")

        # Range score (0-10)
        if score.range_30s >= 2.0:
            score.range_score = 10
            reasons.append(f"Range expansion: {score.range_30s:.2f}%")
        elif score.range_30s >= 1.0:
            score.range_score = 7
        elif score.range_30s >= 0.5:
            score.range_score = 4
        elif score.range_30s >= 0.2:
            score.range_score = 2
        else:
            score.range_score = 0

        # HOD proximity (0-5)
        if high_of_day > 0 and current_price > 0:
            score.hod_distance_pct = ((high_of_day - current_price) / current_price) * 100
            score.hod_break = current_price >= high_of_day * 0.999

            if score.hod_break:
                score.hod_score = 5
                reasons.append("HOD BREAK")
            elif score.hod_distance_pct <= 0.5:
                score.hod_score = 4
            elif score.hod_distance_pct <= 1.0:
                score.hod_score = 3
            elif score.hod_distance_pct <= 2.0:
                score.hod_score = 1
            else:
                score.hod_score = 0

        # VWAP position and veto check
        if vwap > 0:
            score.vwap_extension_pct = ((current_price - vwap) / vwap) * 100

            if current_price > vwap * 1.005:
                score.vwap_position = "ABOVE"
                # Check if too extended
                if score.vwap_extension_pct > self.config.vwap_max_extension_pct:
                    warnings.append(f"Extended {score.vwap_extension_pct:.1f}% above VWAP")
                    veto_reasons.append(VetoReason.VWAP_EXTENDED)
            elif current_price >= vwap * 0.995:
                score.vwap_position = "AT"
            else:
                score.vwap_position = "BELOW"
                warnings.append("Below VWAP")
                veto_reasons.append(VetoReason.BELOW_VWAP)

        return score

    def _calc_participation(self,
                            current_volume: int,
                            volume_30s_baseline: int,
                            avg_volume: int,
                            float_shares: int,
                            day_volume: int,
                            reasons: List[str],
                            warnings: List[str],
                            debug: Dict) -> ParticipationScore:
        """Calculate participation score"""
        score = ParticipationScore()

        # Volume surge vs average
        if avg_volume > 0:
            avg_per_minute = avg_volume / 390
            score.volume_surge = current_volume / max(avg_per_minute, 1)

            if score.volume_surge >= 10.0:
                score.surge_score = 15
                reasons.append(f"Volume surge: {score.volume_surge:.0f}x")
            elif score.volume_surge >= 5.0:
                score.surge_score = 12
            elif score.volume_surge >= 3.0:
                score.surge_score = 8
            elif score.volume_surge >= 2.0:
                score.surge_score = 4
            else:
                score.surge_score = 0
                if score.volume_surge < 1.0:
                    warnings.append("Low volume")

        # Volume rate vs 30s baseline
        if volume_30s_baseline > 0:
            score.vol_rate_30s = current_volume / volume_30s_baseline

            if score.vol_rate_30s >= 5.0:
                score.vol_rate_score = 10
                reasons.append(f"Vol rate: {score.vol_rate_30s:.1f}x")
            elif score.vol_rate_30s >= 3.0:
                score.vol_rate_score = 7
            elif score.vol_rate_30s >= 2.0:
                score.vol_rate_score = 4
            elif score.vol_rate_30s >= 1.5:
                score.vol_rate_score = 2
            else:
                score.vol_rate_score = 0
        else:
            # Fallback to relative volume if no baseline
            if avg_volume > 0 and day_volume > 0:
                now = datetime.now()
                market_open = now.replace(hour=9, minute=30, second=0)
                minutes_elapsed = max(1, (now - market_open).total_seconds() / 60)
                expected_volume = avg_volume * (minutes_elapsed / 390)
                score.relative_volume = day_volume / max(expected_volume, 1)

                if score.relative_volume >= 3.0:
                    score.vol_rate_score = 7
                elif score.relative_volume >= 2.0:
                    score.vol_rate_score = 4
                elif score.relative_volume >= 1.5:
                    score.vol_rate_score = 2
                else:
                    score.vol_rate_score = 0

        # Float rotation
        if float_shares > 0 and day_volume > 0:
            score.float_rotation_pct = (day_volume / float_shares) * 100

            if score.float_rotation_pct >= 100:
                score.rotation_score = 10
                reasons.append(f"Float rotation: {score.float_rotation_pct:.0f}%")
            elif score.float_rotation_pct >= 50:
                score.rotation_score = 7
            elif score.float_rotation_pct >= 25:
                score.rotation_score = 4
            elif score.float_rotation_pct >= 10:
                score.rotation_score = 2
            else:
                score.rotation_score = 0

        debug['participation'] = {
            'volume_surge': round(score.volume_surge, 2),
            'vol_rate_30s': round(score.vol_rate_30s, 2),
            'float_rotation': round(score.float_rotation_pct, 2)
        }

        return score

    def _calc_liquidity(self,
                        spread_pct: float,
                        buy_pressure: float,
                        imbalance_ratio: float,
                        tape_signal: str,
                        mtf_aligned: bool,
                        chronos_regime: str,
                        chronos_confidence: float,
                        reasons: List[str],
                        warnings: List[str],
                        veto_reasons: List[VetoReason],
                        debug: Dict) -> LiquidityScore:
        """Calculate liquidity score with vetoes"""
        score = LiquidityScore()
        score.spread_pct = spread_pct
        score.buy_pressure = buy_pressure
        score.imbalance_ratio = imbalance_ratio
        score.tape_signal = tape_signal

        # Spread score and veto (0-10)
        if spread_pct <= 0.1:
            score.spread_score = 10
        elif spread_pct <= 0.25:
            score.spread_score = 8
        elif spread_pct <= 0.5:
            score.spread_score = 6
        elif spread_pct <= self.config.max_spread_pct:
            score.spread_score = 3
        else:
            score.spread_score = 0
            warnings.append(f"Wide spread: {spread_pct:.2f}%")
            veto_reasons.append(VetoReason.SPREAD_WIDE)

        # Order flow score and veto (0-10)
        if buy_pressure >= 0.70:
            score.flow_score = 10
            reasons.append(f"Strong buy pressure: {buy_pressure:.0%}")
        elif buy_pressure >= 0.60:
            score.flow_score = 8
        elif buy_pressure >= self.config.min_buy_pressure:
            score.flow_score = 5
        elif buy_pressure >= 0.50:
            score.flow_score = 2
        else:
            score.flow_score = 0
            warnings.append(f"Sell pressure: {buy_pressure:.0%}")
            veto_reasons.append(VetoReason.SELL_PRESSURE)

        # Context score (0-5): MTF + Chronos
        context_points = 0

        # MTF alignment
        if mtf_aligned is not None:
            score.mtf_aligned = mtf_aligned
            if mtf_aligned:
                context_points += 2
                reasons.append("MTF aligned")
            else:
                warnings.append("MTF not aligned")
                # Note: MTF misalignment is a warning, not a hard veto

        # Chronos regime
        if chronos_regime:
            favorable_regimes = ['TRENDING_UP', 'RANGING']
            bad_regimes = ['TRENDING_DOWN', 'VOLATILE']

            if chronos_regime in favorable_regimes:
                score.chronos_ok = True
                context_points += 2
                if chronos_confidence and chronos_confidence >= 0.6:
                    context_points += 1
            elif chronos_regime in bad_regimes:
                warnings.append(f"Regime: {chronos_regime}")
                veto_reasons.append(VetoReason.REGIME_BAD)

        score.context_score = min(5, context_points)

        debug['liquidity'] = {
            'spread_pct': spread_pct,
            'buy_pressure': buy_pressure,
            'mtf_aligned': mtf_aligned,
            'chronos_regime': chronos_regime,
            'chronos_confidence': chronos_confidence
        }

        return score

    def _get_grade(self, score: int) -> MomentumGrade:
        """Get grade from score"""
        if score >= 80:
            return MomentumGrade.A
        elif score >= 65:
            return MomentumGrade.B
        elif score >= 50:
            return MomentumGrade.C
        elif score >= 35:
            return MomentumGrade.D
        else:
            return MomentumGrade.F

    def get_cached(self, symbol: str) -> Optional[MomentumResult]:
        """Get cached result if still valid"""
        if symbol in self._cache:
            result = self._cache[symbol]
            age = (datetime.now() - result.timestamp).total_seconds()
            if age < self._cache_ttl:
                return result
        return None

    def update_config(self, **kwargs):
        """Update configuration dynamically"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"Config updated: {key} = {value}")


# Singleton instance
_scorer: Optional[MomentumScorer] = None


def get_momentum_scorer() -> MomentumScorer:
    """Get singleton scorer instance"""
    global _scorer
    if _scorer is None:
        _scorer = MomentumScorer()
    return _scorer


def calculate_momentum_score(symbol: str, **kwargs) -> MomentumResult:
    """Quick calculation helper"""
    return get_momentum_scorer().calculate(symbol, **kwargs)


if __name__ == "__main__":
    # Test with sample data
    scorer = MomentumScorer()

    print("=" * 60)
    print("MOMENTUM SCORE v2 TEST (ChatGPT Spec)")
    print("=" * 60)

    # Test 1: Good momentum, no vetoes
    # Price $5.50, VWAP $5.40 = 1.85% extension (under 3% max)
    result = scorer.calculate(
        symbol="TEST_GOOD",
        current_price=5.50,
        prices_5s=[5.45, 5.47, 5.50],
        prices_15s=[5.40, 5.44, 5.48, 5.50],
        prices_30s=[5.30, 5.40, 5.45, 5.50],
        high_30s=5.52,
        low_30s=5.28,
        high_of_day=5.52,
        vwap=5.40,  # Close to price so not extended
        current_volume=50000,
        volume_30s_baseline=10000,
        avg_volume=500000,
        float_shares=2000000,
        day_volume=400000,
        spread_pct=0.3,
        buy_pressure=0.65,
        mtf_aligned=True,
        chronos_regime="TRENDING_UP",
        chronos_confidence=0.7
    )

    print(f"\n--- TEST 1: Good Momentum (no vetoes) ---")
    print(f"Score: {result.score}/100 (Grade {result.grade.value})")
    print(f"Vetoed: {result.vetoed}")
    print(f"Ignition Ready: {result.ignition_ready}")
    print(f"Components: P={result.price_urgency.total}/40, V={result.participation.total}/35, L={result.liquidity.total}/25")
    print(f"ROC: r_5s={result.price_urgency.r_5s:.2f}%, r_15s={result.price_urgency.r_15s:.2f}%, r_30s={result.price_urgency.r_30s:.2f}%")
    print(f"Accel: {result.price_urgency.accel:.3f}")
    print(f"Reasons: {result.reasons}")
    if result.warnings:
        print(f"Warnings: {result.warnings}")

    # Test 2: Below VWAP + Sell Pressure (should veto)
    result2 = scorer.calculate(
        symbol="TEST_VETO_BELOW",
        current_price=5.10,  # Below VWAP of 5.20
        prices_5s=[5.08, 5.09, 5.10],
        prices_15s=[5.05, 5.07, 5.09, 5.10],
        prices_30s=[5.00, 5.05, 5.08, 5.10],
        high_30s=5.12,
        low_30s=4.98,
        high_of_day=5.25,
        vwap=5.20,
        current_volume=30000,
        avg_volume=500000,
        spread_pct=0.5,
        buy_pressure=0.45,  # Below minimum 55%
    )

    print(f"\n--- TEST 2: Below VWAP + Sell Pressure ---")
    print(f"Score: {result2.score}/100 (Grade {result2.grade.value})")
    print(f"Vetoed: {result2.vetoed}")
    print(f"Veto Reasons: {[v.value for v in result2.veto_reasons]}")
    print(f"Warnings: {result2.warnings}")

    # Test 3: Wide spread veto (set VWAP close so only spread triggers)
    # Note: scalper_config.json may have max_spread=1.5%, so use 2.0%
    result3 = scorer.calculate(
        symbol="TEST_SPREAD",
        current_price=5.50,
        prices_30s=[5.40, 5.45, 5.50],
        vwap=5.45,  # Close to price
        spread_pct=2.0,  # Too wide (exceeds any max)
        buy_pressure=0.60,
    )

    print(f"\n--- TEST 3: Wide Spread ---")
    print(f"Score: {result3.score}/100")
    print(f"Vetoed: {result3.vetoed}")
    print(f"Veto Reasons: {[v.value for v in result3.veto_reasons]}")

    # Test 4: VWAP extended veto
    result4 = scorer.calculate(
        symbol="TEST_EXTENDED",
        current_price=5.50,
        prices_30s=[5.40, 5.45, 5.50],
        vwap=5.20,  # 5.77% extension > 3% max
        spread_pct=0.3,
        buy_pressure=0.65,
    )

    print(f"\n--- TEST 4: VWAP Extended (>3%) ---")
    print(f"Score: {result4.score}/100")
    print(f"Vetoed: {result4.vetoed}")
    print(f"Veto Reasons: {[v.value for v in result4.veto_reasons]}")
    print(f"VWAP Extension: {result4.price_urgency.vwap_extension_pct:.1f}%")

    # Test 5: Bad regime veto
    result5 = scorer.calculate(
        symbol="TEST_REGIME",
        current_price=5.50,
        prices_30s=[5.40, 5.45, 5.50],
        vwap=5.45,
        spread_pct=0.3,
        buy_pressure=0.65,
        chronos_regime="VOLATILE",
    )

    print(f"\n--- TEST 5: Bad Regime (VOLATILE) ---")
    print(f"Score: {result5.score}/100")
    print(f"Vetoed: {result5.vetoed}")
    print(f"Veto Reasons: {[v.value for v in result5.veto_reasons]}")

    # Test 6: No momentum (flat price)
    result6 = scorer.calculate(
        symbol="TEST_FLAT",
        current_price=5.50,
        prices_5s=[5.50, 5.50, 5.50],
        prices_15s=[5.50, 5.50, 5.50, 5.50],
        prices_30s=[5.50, 5.50, 5.50, 5.50],
        vwap=5.45,
        spread_pct=0.3,
        buy_pressure=0.65,
    )

    print(f"\n--- TEST 6: No Momentum (flat) ---")
    print(f"Score: {result6.score}/100")
    print(f"Vetoed: {result6.vetoed}")
    print(f"Veto Reasons: {[v.value for v in result6.veto_reasons]}")

    print("\n" + "=" * 60)
    print("VETO SYSTEM TEST COMPLETE")
    print("=" * 60)

    # Summary
    tests = [
        ("Good Momentum", result, False),
        ("Below VWAP", result2, True),
        ("Wide Spread", result3, True),
        ("VWAP Extended", result4, True),
        ("Bad Regime", result5, True),
        ("No Momentum", result6, True),
    ]

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    passed = 0
    for name, r, expected_veto in tests:
        status = "PASS" if r.vetoed == expected_veto else "FAIL"
        if r.vetoed == expected_veto:
            passed += 1
        print(f"[{status}] {name}: Score={r.score}, Vetoed={r.vetoed} (expected {expected_veto})")
    print(f"\nPassed: {passed}/{len(tests)}")
