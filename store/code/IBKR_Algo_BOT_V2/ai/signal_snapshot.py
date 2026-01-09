"""
Signal Snapshot - Canonical Symbol Data Schema
===============================================

This is the SINGLE source of truth for symbol data across all pipeline tasks.
Every task (R1-R15) MUST use this schema. No silent defaults that hide failures.

Contract Rules:
1. If data unavailable, set field to None - NEVER use 0 or 1 as fallback
2. All data quality issues MUST be flagged in data_quality dict
3. Timestamps are always UTC
4. Enums enforce valid states only

Usage:
    snapshot = SignalSnapshot.from_quote(symbol, quote_data, float_data)
    if not snapshot.is_tradeable:
        print(f"Cannot trade: {snapshot.data_quality}")
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class HODStatus(Enum):
    """High of Day status - strict enum, no silent failures"""

    AT_HOD = "AT_HOD"  # Within 0.5% of HOD
    NEAR_HOD = "NEAR_HOD"  # Within 2% of HOD
    PULLBACK = "PULLBACK"  # 2-5% from HOD
    FAIL = "FAIL"  # >5% from HOD or broke down
    UNKNOWN = "UNKNOWN"  # Data unavailable


class DataQualityFlag(Enum):
    """Data quality issues that must be tracked"""

    FLOAT_MISSING = "FLOAT_MISSING"
    FLOAT_ZERO = "FLOAT_ZERO"
    AVG_VOLUME_MISSING = "AVG_VOLUME_MISSING"
    AVG_VOLUME_ZERO = "AVG_VOLUME_ZERO"
    HOD_INCONSISTENT = "HOD_INCONSISTENT"  # hod_price < current price
    REL_VOL_ABSURD = "REL_VOL_ABSURD"  # rel_vol > 5000
    PRICE_STALE = "PRICE_STALE"  # timestamp too old
    SPREAD_WIDE = "SPREAD_WIDE"  # spread > 3%
    DATA_INCOMPLETE = "DATA_INCOMPLETE"  # critical fields missing


class ModelSource(Enum):
    """Source of model predictions"""

    REAL = "REAL"  # Trained model with validation
    SIMULATED = "SIMULATED"  # Placeholder/mock values
    UNAVAILABLE = "UNAVAILABLE"  # Model not loaded


@dataclass
class DataQuality:
    """
    Tracks all data quality issues for a snapshot.
    Gate MUST check this before approving trades.
    """

    flags: List[DataQualityFlag] = field(default_factory=list)
    float_missing: bool = False
    avg_volume_missing: bool = False
    hod_inconsistent: bool = False
    rel_vol_absurd: bool = False
    price_stale: bool = False
    spread_wide: bool = False
    stale_seconds: Optional[float] = None

    @property
    def is_clean(self) -> bool:
        """Returns True only if NO data quality issues"""
        return len(self.flags) == 0

    @property
    def has_critical_issues(self) -> bool:
        """Critical issues that MUST block trading"""
        critical = {
            DataQualityFlag.PRICE_STALE,
            DataQualityFlag.DATA_INCOMPLETE,
            DataQualityFlag.HOD_INCONSISTENT,
        }
        return bool(set(self.flags) & critical)

    def add_flag(self, flag: DataQualityFlag):
        if flag not in self.flags:
            self.flags.append(flag)

    def to_dict(self) -> Dict:
        return {
            "flags": [f.value for f in self.flags],
            "float_missing": self.float_missing,
            "avg_volume_missing": self.avg_volume_missing,
            "hod_inconsistent": self.hod_inconsistent,
            "rel_vol_absurd": self.rel_vol_absurd,
            "price_stale": self.price_stale,
            "spread_wide": self.spread_wide,
            "stale_seconds": self.stale_seconds,
            "is_clean": self.is_clean,
            "has_critical_issues": self.has_critical_issues,
        }


@dataclass
class SignalSnapshot:
    """
    Canonical symbol snapshot used by ALL pipeline tasks.

    NO SILENT DEFAULTS - if data is missing, it's None and flagged.
    """

    # Identity
    symbol: str
    timestamp_utc: datetime

    # Price Data (required)
    price: float
    prev_close: Optional[float] = None
    gap_pct: Optional[float] = None

    # Volume Data
    volume_today: Optional[int] = None
    avg_daily_volume_30d: Optional[int] = None  # None if unknown, NOT 1
    rel_vol_daily: Optional[float] = None  # None if can't compute

    # 5-minute volume
    vol_5m: Optional[int] = None
    avg_5m_volume_30d: Optional[int] = None
    rel_vol_5m: Optional[float] = None

    # Float/Short Data
    float_shares: Optional[int] = None  # None if unknown, NOT 0
    short_interest_shares: Optional[int] = None
    short_interest_pct: Optional[float] = None

    # HOD Tracking (strict enum)
    hod_price: Optional[float] = None
    pct_from_hod: Optional[float] = None
    hod_status: HODStatus = HODStatus.UNKNOWN
    hod_breaks: int = 0
    failed_breaks: int = 0

    # Spread/Liquidity
    bid: Optional[float] = None
    ask: Optional[float] = None
    spread: Optional[float] = None
    spread_pct: Optional[float] = None

    # Data Quality
    data_quality: DataQuality = field(default_factory=DataQuality)

    # Model Predictions (with source tracking)
    qlib_score: Optional[float] = None
    qlib_source: ModelSource = ModelSource.UNAVAILABLE
    chronos_score: Optional[float] = None
    chronos_source: ModelSource = ModelSource.UNAVAILABLE
    entry_score: Optional[float] = None

    # Metadata
    priority: str = "NORMAL"  # HIGH, NORMAL, LOW

    def __post_init__(self):
        """Validate and flag data quality issues"""
        self._validate_and_flag()

    def _validate_and_flag(self):
        """Run all validation checks and set flags"""
        # Float validation
        if self.float_shares is None:
            self.data_quality.float_missing = True
            self.data_quality.add_flag(DataQualityFlag.FLOAT_MISSING)
        elif self.float_shares == 0:
            self.data_quality.add_flag(DataQualityFlag.FLOAT_ZERO)

        # Avg volume validation
        if self.avg_daily_volume_30d is None:
            self.data_quality.avg_volume_missing = True
            self.data_quality.add_flag(DataQualityFlag.AVG_VOLUME_MISSING)
        elif self.avg_daily_volume_30d == 0:
            self.data_quality.add_flag(DataQualityFlag.AVG_VOLUME_ZERO)

        # HOD consistency check
        if self.hod_price is not None and self.price is not None:
            if self.hod_price < self.price * 0.99:  # Allow 1% tolerance for timing
                self.data_quality.hod_inconsistent = True
                self.data_quality.add_flag(DataQualityFlag.HOD_INCONSISTENT)

        # Rel vol sanity check
        if self.rel_vol_daily is not None and self.rel_vol_daily > 5000:
            self.data_quality.rel_vol_absurd = True
            self.data_quality.add_flag(DataQualityFlag.REL_VOL_ABSURD)

        if self.rel_vol_5m is not None and self.rel_vol_5m > 5000:
            self.data_quality.rel_vol_absurd = True
            self.data_quality.add_flag(DataQualityFlag.REL_VOL_ABSURD)

        # Spread check
        if self.spread_pct is not None and self.spread_pct > 3.0:
            self.data_quality.spread_wide = True
            self.data_quality.add_flag(DataQualityFlag.SPREAD_WIDE)

    def mark_stale(self, stale_seconds: float):
        """Mark snapshot as stale (called by connection manager)"""
        self.data_quality.price_stale = True
        self.data_quality.stale_seconds = stale_seconds
        self.data_quality.add_flag(DataQualityFlag.PRICE_STALE)

    @property
    def is_tradeable(self) -> bool:
        """Can this snapshot be used for trade decisions?"""
        return not self.data_quality.has_critical_issues

    @property
    def hod_status_str(self) -> str:
        """String representation for reports"""
        return self.hod_status.value if self.hod_status else "UNKNOWN"

    def compute_rel_volumes(self):
        """Compute relative volumes if raw data available"""
        if self.volume_today and self.avg_daily_volume_30d:
            self.rel_vol_daily = round(self.volume_today / self.avg_daily_volume_30d, 2)

        if self.vol_5m and self.avg_5m_volume_30d:
            self.rel_vol_5m = round(self.vol_5m / self.avg_5m_volume_30d, 2)

    def compute_hod_status(self):
        """Compute HOD status from price and hod_price"""
        if self.hod_price is None or self.price is None:
            self.hod_status = HODStatus.UNKNOWN
            return

        if self.hod_price <= 0:
            self.hod_status = HODStatus.UNKNOWN
            return

        self.pct_from_hod = round(
            ((self.price - self.hod_price) / self.hod_price) * 100, 2
        )

        # Classify HOD status
        abs_pct = abs(self.pct_from_hod)
        if abs_pct <= 0.5:
            self.hod_status = HODStatus.AT_HOD
        elif abs_pct <= 2.0:
            self.hod_status = HODStatus.NEAR_HOD
        elif abs_pct <= 5.0:
            self.hod_status = HODStatus.PULLBACK
        else:
            self.hod_status = HODStatus.FAIL

    def compute_spread(self):
        """Compute spread from bid/ask"""
        if self.bid and self.ask and self.bid > 0:
            self.spread = round(self.ask - self.bid, 4)
            self.spread_pct = round((self.spread / self.bid) * 100, 2)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for JSON reports"""
        return {
            "symbol": self.symbol,
            "timestamp_utc": (
                self.timestamp_utc.isoformat() if self.timestamp_utc else None
            ),
            "price": self.price,
            "prev_close": self.prev_close,
            "gap_pct": self.gap_pct,
            "volume_today": self.volume_today,
            "avg_daily_volume_30d": self.avg_daily_volume_30d,
            "rel_vol_daily": self.rel_vol_daily,
            "vol_5m": self.vol_5m,
            "avg_5m_volume_30d": self.avg_5m_volume_30d,
            "rel_vol_5m": self.rel_vol_5m,
            "float_shares": self.float_shares,
            "short_interest_shares": self.short_interest_shares,
            "short_interest_pct": self.short_interest_pct,
            "hod_price": self.hod_price,
            "pct_from_hod": self.pct_from_hod,
            "hod_status": self.hod_status.value if self.hod_status else None,
            "hod_breaks": self.hod_breaks,
            "failed_breaks": self.failed_breaks,
            "bid": self.bid,
            "ask": self.ask,
            "spread": self.spread,
            "spread_pct": self.spread_pct,
            "data_quality": self.data_quality.to_dict(),
            "qlib_score": self.qlib_score,
            "qlib_source": self.qlib_source.value if self.qlib_source else None,
            "chronos_score": self.chronos_score,
            "chronos_source": (
                self.chronos_source.value if self.chronos_source else None
            ),
            "entry_score": self.entry_score,
            "priority": self.priority,
            "is_tradeable": self.is_tradeable,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SignalSnapshot":
        """Deserialize from dict"""
        # Parse timestamp
        ts = data.get("timestamp_utc")
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        elif ts is None:
            ts = datetime.now(timezone.utc)

        # Parse enums
        hod_status = HODStatus.UNKNOWN
        if data.get("hod_status"):
            try:
                hod_status = HODStatus(data["hod_status"])
            except ValueError:
                hod_status = HODStatus.UNKNOWN

        qlib_source = ModelSource.UNAVAILABLE
        if data.get("qlib_source"):
            try:
                qlib_source = ModelSource(data["qlib_source"])
            except ValueError:
                pass

        chronos_source = ModelSource.UNAVAILABLE
        if data.get("chronos_source"):
            try:
                chronos_source = ModelSource(data["chronos_source"])
            except ValueError:
                pass

        # Handle None-as-explicit-value: .get() returns None if key exists with None value
        # Use `or` to provide defaults for required numeric fields
        snapshot = cls(
            symbol=data.get("symbol") or "",
            timestamp_utc=ts,
            price=data.get("price") or 0,  # Required - must be numeric
            prev_close=data.get("prev_close"),
            gap_pct=data.get("gap_pct"),
            volume_today=data.get("volume_today"),
            avg_daily_volume_30d=data.get("avg_daily_volume_30d"),
            rel_vol_daily=data.get("rel_vol_daily"),
            vol_5m=data.get("vol_5m"),
            avg_5m_volume_30d=data.get("avg_5m_volume_30d"),
            rel_vol_5m=data.get("rel_vol_5m"),
            float_shares=data.get("float_shares"),
            short_interest_shares=data.get("short_interest_shares"),
            short_interest_pct=data.get("short_interest_pct"),
            hod_price=data.get("hod_price"),
            pct_from_hod=data.get("pct_from_hod"),
            hod_status=hod_status,
            hod_breaks=data.get("hod_breaks") or 0,  # Required int - defaults to 0
            failed_breaks=data.get("failed_breaks")
            or 0,  # Required int - defaults to 0
            bid=data.get("bid"),
            ask=data.get("ask"),
            spread=data.get("spread"),
            spread_pct=data.get("spread_pct"),
            qlib_score=data.get("qlib_score"),
            qlib_source=qlib_source,
            chronos_score=data.get("chronos_score"),
            chronos_source=chronos_source,
            entry_score=data.get("entry_score"),
            priority=data.get("priority") or "NORMAL",
        )

        return snapshot

    @classmethod
    def from_quote(
        cls, symbol: str, quote: Dict, float_data: Optional[Dict] = None
    ) -> "SignalSnapshot":
        """
        Create snapshot from Schwab quote data.
        This is the PRIMARY factory method.
        """
        now = datetime.now(timezone.utc)

        # Extract price
        price = quote.get("lastPrice") or quote.get("price") or 0
        prev_close = (
            quote.get("closePrice")
            or quote.get("previousClose")
            or quote.get("prev_close")
        )

        # Compute gap if we have prev_close
        gap_pct = None
        if prev_close and prev_close > 0 and price:
            gap_pct = round(((price - prev_close) / prev_close) * 100, 2)

        # Volume
        volume_today = quote.get("totalVolume") or quote.get("volume")
        avg_volume = (
            quote.get("averageVolume")
            or quote.get("avgVolume")
            or quote.get("averageVolume10Day")
        )

        # Important: Don't use 1 as fallback - use None
        if avg_volume == 0 or avg_volume == 1:
            avg_volume = None

        # HOD
        hod_price = quote.get("highPrice") or quote.get("dayHigh") or quote.get("high")

        # Bid/Ask
        bid = quote.get("bidPrice") or quote.get("bid")
        ask = quote.get("askPrice") or quote.get("ask")

        # Float data
        float_shares = None
        short_interest = None
        if float_data:
            if float_data.get("status") == "success":
                fs = float_data.get("float_shares")
                if fs and fs > 0:
                    float_shares = int(fs)
                si = float_data.get("short_interest")
                if si and si > 0:
                    short_interest = int(si)

        snapshot = cls(
            symbol=symbol,
            timestamp_utc=now,
            price=price,
            prev_close=prev_close,
            gap_pct=gap_pct,
            volume_today=volume_today,
            avg_daily_volume_30d=avg_volume,
            float_shares=float_shares,
            short_interest_shares=short_interest,
            hod_price=hod_price,
            bid=bid,
            ask=ask,
        )

        # Compute derived fields
        snapshot.compute_rel_volumes()
        snapshot.compute_hod_status()
        snapshot.compute_spread()

        return snapshot


def create_snapshot_from_worklist_item(item: Dict) -> SignalSnapshot:
    """
    Create snapshot from worklist API response item.
    Maps worklist field names to canonical schema.
    """
    now = datetime.now(timezone.utc)

    # Extract data with proper field mapping
    price = item.get("price", 0)
    prev_close = None  # Worklist doesn't provide this directly

    # Compute gap from change_percent
    gap_pct = (
        item.get("change_percent") or item.get("change_pct") or item.get("changePct")
    )

    # Volume
    volume_today = item.get("volume")
    avg_volume = item.get("avg_volume") or item.get("avgVolume")
    if avg_volume == 0 or avg_volume == 1:
        avg_volume = None

    # Float - worklist has it
    float_raw = item.get("float")
    float_shares = None
    if float_raw and float_raw > 0:
        float_shares = int(float_raw)

    # HOD
    hod_price = item.get("high") or item.get("dayHigh")
    pct_from_hod = item.get("percent_from_hod")

    # Determine HOD status from percent_from_hod
    hod_status = HODStatus.UNKNOWN
    if pct_from_hod is not None:
        abs_pct = abs(pct_from_hod)
        if abs_pct <= 0.5:
            hod_status = HODStatus.AT_HOD
        elif abs_pct <= 2.0:
            hod_status = HODStatus.NEAR_HOD
        elif abs_pct <= 5.0:
            hod_status = HODStatus.PULLBACK
        else:
            hod_status = HODStatus.FAIL

    # Bid/Ask
    bid = item.get("bid")
    ask = item.get("ask")

    # Rel vol from worklist
    rel_vol = item.get("rel_volume")

    snapshot = SignalSnapshot(
        symbol=item.get("symbol", ""),
        timestamp_utc=now,
        price=price,
        gap_pct=gap_pct,
        volume_today=volume_today,
        avg_daily_volume_30d=avg_volume,
        rel_vol_daily=rel_vol,
        float_shares=float_shares,
        hod_price=hod_price,
        pct_from_hod=pct_from_hod,
        hod_status=hod_status,
        bid=bid,
        ask=ask,
    )

    # Compute spread
    snapshot.compute_spread()

    return snapshot
