"""
Position Synchronizer & Slippage Combat Module
==============================================
Real-time synchronization with Schwab broker and aggressive slippage detection/combat.

This module:
1. Periodically syncs positions/orders with broker (every 5 seconds)
2. Detects when orders are stale or stuck
3. Detects slippage during fills
4. Takes AGGRESSIVE action to combat slippage
5. Ensures bot state matches live broker state

SLIPPAGE COMBAT STRATEGIES:
- Unfilled orders > 30 sec: Cancel and resubmit at market
- Partial fills with price moving away: Complete at market
- Limit orders not filling: Adjust price aggressively
- Stale orders in pre-market: Convert to extended hours
"""

import logging
import asyncio
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import pytz

logger = logging.getLogger(__name__)


@dataclass
class OrderTracker:
    """Track order for slippage detection"""
    order_id: str
    symbol: str
    side: str
    quantity: float
    order_type: str
    limit_price: Optional[float]
    submitted_at: datetime
    last_checked: datetime
    status: str
    filled_qty: float = 0
    filled_avg_price: float = 0
    expected_price: float = 0  # Price at submission time
    slippage_detected: bool = False
    slippage_amount: float = 0
    action_taken: str = ""


@dataclass
class PositionDrift:
    """Tracks drift between expected and actual position"""
    symbol: str
    expected_qty: float
    actual_qty: float
    expected_cost: float
    actual_cost: float
    drift_qty: float
    drift_percent: float
    detected_at: datetime
    reconciled: bool = False


@dataclass
class SyncState:
    """Current sync state"""
    last_sync: datetime = None
    positions: List[Dict] = field(default_factory=list)
    open_orders: List[Dict] = field(default_factory=list)
    order_trackers: Dict[str, OrderTracker] = field(default_factory=dict)
    sync_errors: int = 0
    slippage_events: List[Dict] = field(default_factory=list)
    # Drift detection
    expected_positions: Dict[str, Dict] = field(default_factory=dict)
    drift_events: List[PositionDrift] = field(default_factory=list)
    last_drift_check: datetime = None


class PositionSynchronizer:
    """
    Real-time position and order synchronization with slippage combat.
    """

    def __init__(self):
        self.et_tz = pytz.timezone('US/Eastern')
        self.state = SyncState()

        # Sync settings
        self.sync_interval = 5  # seconds
        self.order_stale_threshold = 30  # seconds - order considered stale
        self.slippage_threshold_pct = 0.5  # 0.5% slippage triggers action

        # Combat settings
        self.max_resubmit_attempts = 3
        self.price_chase_increment = 0.01  # 1 cent chase for limit orders

        # Background sync
        self._running = False
        self._sync_thread = None

        logger.info("PositionSynchronizer initialized")

    def start_background_sync(self):
        """Start background synchronization thread"""
        if self._running:
            return

        self._running = True
        self._sync_thread = threading.Thread(target=self._sync_loop, daemon=True)
        self._sync_thread.start()
        logger.info("Background sync started")

    def stop_background_sync(self):
        """Stop background synchronization"""
        self._running = False
        if self._sync_thread:
            self._sync_thread.join(timeout=2)
        logger.info("Background sync stopped")

    def _sync_loop(self):
        """Background sync loop"""
        while self._running:
            try:
                self.sync_now()
            except Exception as e:
                logger.error(f"Sync error: {e}")
                self.state.sync_errors += 1

            # Sleep for sync interval
            import time
            time.sleep(self.sync_interval)

    def sync_now(self) -> Dict:
        """
        Perform immediate sync with broker.
        Returns sync results including any slippage alerts.
        """
        from unified_broker import get_unified_broker
        broker = get_unified_broker()

        results = {
            "timestamp": datetime.now().isoformat(),
            "positions_synced": 0,
            "orders_synced": 0,
            "stale_orders": [],
            "slippage_alerts": [],
            "actions_taken": []
        }

        try:
            # Sync positions
            positions = broker.get_positions()
            self.state.positions = positions
            results["positions_synced"] = len(positions)

            # Sync orders
            orders = broker.get_orders(status="open")
            self.state.open_orders = orders
            results["orders_synced"] = len(orders)

            # Track and analyze orders
            for order in orders:
                order_id = order.get("order_id")

                # Create tracker if new
                if order_id not in self.state.order_trackers:
                    self._create_order_tracker(order)

                # Update existing tracker
                tracker = self.state.order_trackers.get(order_id)
                if tracker:
                    tracker.status = order.get("status", "unknown")
                    tracker.filled_qty = float(order.get("filled_qty", 0))
                    tracker.filled_avg_price = float(order.get("filled_avg_price", 0) or 0)
                    tracker.last_checked = datetime.now(self.et_tz)

                    # Check for stale orders
                    if self._is_order_stale(tracker):
                        results["stale_orders"].append({
                            "order_id": order_id,
                            "symbol": tracker.symbol,
                            "age_seconds": self._get_order_age(tracker),
                            "status": tracker.status
                        })

                        # Take action on stale orders
                        action = self._combat_stale_order(tracker, broker)
                        if action:
                            results["actions_taken"].append(action)

                    # Check for slippage
                    slippage = self._detect_slippage(tracker, broker)
                    if slippage:
                        results["slippage_alerts"].append(slippage)

                        # Take action on slippage
                        action = self._combat_slippage(tracker, slippage, broker)
                        if action:
                            results["actions_taken"].append(action)

            # Clean up filled/cancelled order trackers
            self._cleanup_trackers(orders)

            self.state.last_sync = datetime.now(self.et_tz)

        except Exception as e:
            logger.error(f"Sync failed: {e}")
            results["error"] = str(e)
            self.state.sync_errors += 1

        return results

    def _create_order_tracker(self, order: Dict):
        """Create a new order tracker"""
        order_id = order.get("order_id")

        # Get expected price from current quote
        try:
            from unified_market_data import get_unified_market_data
            market_data = get_unified_market_data()
            quote = market_data.get_quote(order.get("symbol", ""))

            side = order.get("side", "").lower()
            if side == "buy":
                expected_price = quote.get("ask", 0) if quote else 0
            else:
                expected_price = quote.get("bid", 0) if quote else 0
        except:
            expected_price = float(order.get("limit_price", 0) or 0)

        # Parse actual submission time from order
        submitted_at = datetime.now(self.et_tz)
        if order.get("submitted_at"):
            try:
                from dateutil import parser
                submitted_at = parser.parse(str(order.get("submitted_at")))
                if submitted_at.tzinfo is None:
                    submitted_at = self.et_tz.localize(submitted_at)
                else:
                    submitted_at = submitted_at.astimezone(self.et_tz)
            except Exception as e:
                logger.warning(f"Could not parse submitted_at: {e}")

        tracker = OrderTracker(
            order_id=order_id,
            symbol=order.get("symbol", ""),
            side=order.get("side", ""),
            quantity=float(order.get("quantity", 0)),
            order_type=order.get("order_type", ""),
            limit_price=float(order.get("limit_price", 0) or 0) if order.get("limit_price") else None,
            submitted_at=submitted_at,
            last_checked=datetime.now(self.et_tz),
            status=order.get("status", "unknown"),
            expected_price=expected_price
        )

        self.state.order_trackers[order_id] = tracker
        logger.debug(f"Created tracker for order {order_id}")

    def _is_order_stale(self, tracker: OrderTracker) -> bool:
        """Check if order is stale (unfilled for too long)"""
        if tracker.status in ["filled", "cancelled", "expired"]:
            return False

        age = self._get_order_age(tracker)
        return age > self.order_stale_threshold

    def _get_order_age(self, tracker: OrderTracker) -> float:
        """Get order age in seconds"""
        now = datetime.now(self.et_tz)
        submitted = tracker.submitted_at
        if submitted.tzinfo is None:
            submitted = self.et_tz.localize(submitted)
        return (now - submitted).total_seconds()

    def _detect_slippage(self, tracker: OrderTracker, connector) -> Optional[Dict]:
        """Detect slippage for an order"""
        if tracker.filled_qty == 0:
            return None  # No fill yet

        if tracker.expected_price == 0:
            return None  # No expected price

        # Calculate slippage
        if tracker.side.lower() == "buy":
            # For buys, slippage is paying more than expected
            slippage_pct = ((tracker.filled_avg_price - tracker.expected_price) / tracker.expected_price) * 100
        else:
            # For sells, slippage is receiving less than expected
            slippage_pct = ((tracker.expected_price - tracker.filled_avg_price) / tracker.expected_price) * 100

        if slippage_pct > self.slippage_threshold_pct:
            slippage_info = {
                "order_id": tracker.order_id,
                "symbol": tracker.symbol,
                "side": tracker.side,
                "expected_price": tracker.expected_price,
                "filled_price": tracker.filled_avg_price,
                "slippage_pct": round(slippage_pct, 2),
                "slippage_amount": round(abs(tracker.filled_avg_price - tracker.expected_price) * tracker.filled_qty, 2),
                "severity": "HIGH" if slippage_pct > 1.0 else "MEDIUM"
            }

            tracker.slippage_detected = True
            tracker.slippage_amount = slippage_info["slippage_amount"]

            self.state.slippage_events.append(slippage_info)
            logger.warning(f"SLIPPAGE DETECTED: {tracker.symbol} {slippage_pct:.2f}%")

            return slippage_info

        return None

    def _combat_stale_order(self, tracker: OrderTracker, connector) -> Optional[Dict]:
        """Take action on stale orders"""
        age = self._get_order_age(tracker)
        action_result = None

        # Check market session
        session = connector.get_market_session()

        if tracker.order_type == "market" and session.get("is_extended_hours"):
            # Market order in extended hours - convert to limit
            try:
                connector.cancel_order(tracker.order_id)

                # Resubmit as extended hours limit order
                result = connector.place_smart_order(
                    symbol=tracker.symbol,
                    quantity=int(tracker.quantity),
                    side=tracker.side.upper()
                )

                action_result = {
                    "action": "CONVERTED_TO_EXTENDED_HOURS",
                    "order_id": tracker.order_id,
                    "symbol": tracker.symbol,
                    "new_order_id": result.get("order_id"),
                    "reason": f"Market order stale ({age:.0f}s) in extended hours"
                }

                tracker.action_taken = "converted_extended"
                logger.info(f"Converted stale market order to extended hours: {tracker.symbol}")

            except Exception as e:
                logger.error(f"Failed to convert stale order: {e}")

        elif tracker.order_type == "limit" and age > 60:
            # Limit order stale for over 1 minute - chase price
            try:
                # ============================================================
                # USE CENTRALIZED DATA BUS FOR ALL MARKET DATA (ORDER INTEGRITY!)
                # ============================================================
                from market_data_bus import get_market_data_bus
                data_bus = get_market_data_bus()

                # Get order price from centralized bus (Schwab only)
                base_price = data_bus.get_price_for_order(tracker.symbol, tracker.side)

                if not base_price:
                    logger.warning(f"Skipping price chase for {tracker.symbol} - no valid data from data bus")
                    return None

                # Apply price chase increment
                if tracker.side.lower() == "buy":
                    new_price = base_price + self.price_chase_increment
                else:
                    new_price = base_price - self.price_chase_increment

                # SAFETY CHECK: Ensure new price is valid (minimum $0.50)
                if new_price < 0.50:
                    logger.warning(f"Skipping price chase for {tracker.symbol} - calculated price too low (${new_price:.2f})")
                    return None

                # Log the price chase
                quote = data_bus.get_quote(tracker.symbol)
                logger.info(f"Price chase for {tracker.symbol}: ${new_price:.2f} (source: {quote.get('source', 'bus') if quote else 'bus'})")

                # Cancel and resubmit at new price
                connector.cancel_order(tracker.order_id)

                if session.get("is_extended_hours"):
                    result = connector.place_smart_order(
                        symbol=tracker.symbol,
                        quantity=int(tracker.quantity),
                        side=tracker.side.upper(),
                        limit_price=new_price
                    )
                else:
                    # Regular hours - just use market order for speed
                    result = connector.place_smart_order(
                        symbol=tracker.symbol,
                        quantity=int(tracker.quantity),
                        side=tracker.side.upper()
                    )

                action_result = {
                    "action": "PRICE_CHASE",
                    "order_id": tracker.order_id,
                    "symbol": tracker.symbol,
                    "old_price": tracker.limit_price,
                    "new_price": new_price if session.get("is_extended_hours") else "market",
                    "new_order_id": result.get("order_id"),
                    "reason": f"Limit order stale ({age:.0f}s) - chasing price"
                }

                tracker.action_taken = "price_chased"
                logger.info(f"Chased price on stale limit order: {tracker.symbol}")

            except Exception as e:
                logger.error(f"Failed to chase price: {e}")

        return action_result

    def _combat_slippage(self, tracker: OrderTracker, slippage: Dict, connector) -> Optional[Dict]:
        """Take action to combat slippage"""
        # If partial fill with high slippage, complete at market
        if tracker.filled_qty > 0 and tracker.filled_qty < tracker.quantity:
            remaining = int(tracker.quantity - tracker.filled_qty)

            if slippage.get("severity") == "HIGH":
                try:
                    # Cancel remaining and complete at market
                    connector.cancel_order(tracker.order_id)

                    result = connector.place_smart_order(
                        symbol=tracker.symbol,
                        quantity=remaining,
                        side=tracker.side.upper()
                    )

                    action_result = {
                        "action": "COMPLETE_AT_MARKET",
                        "order_id": tracker.order_id,
                        "symbol": tracker.symbol,
                        "remaining_qty": remaining,
                        "new_order_id": result.get("order_id"),
                        "reason": f"High slippage ({slippage.get('slippage_pct'):.2f}%) - completing at market"
                    }

                    tracker.action_taken = "completed_at_market"
                    logger.warning(f"Completed order at market due to slippage: {tracker.symbol}")

                    return action_result

                except Exception as e:
                    logger.error(f"Failed to complete at market: {e}")

        return None

    def _cleanup_trackers(self, current_orders: List[Dict]):
        """Remove trackers for filled/cancelled orders"""
        current_ids = {o.get("order_id") for o in current_orders}

        to_remove = []
        for order_id, tracker in self.state.order_trackers.items():
            if order_id not in current_ids:
                to_remove.append(order_id)

        for order_id in to_remove:
            del self.state.order_trackers[order_id]

    # ============================================================================
    # DRIFT DETECTION
    # ============================================================================

    def register_expected_position(self, symbol: str, quantity: float, cost_basis: float):
        """
        Register an expected position from a trade.
        Call this when placing orders to track expected vs actual.
        """
        symbol = symbol.upper()
        self.state.expected_positions[symbol] = {
            "quantity": quantity,
            "cost_basis": cost_basis,
            "registered_at": datetime.now(self.et_tz).isoformat()
        }
        logger.debug(f"Registered expected position: {symbol} x {quantity} @ ${cost_basis:.2f}")

    def update_expected_position(self, symbol: str, qty_change: float, price: float):
        """
        Update expected position after a fill (add or remove shares).
        """
        symbol = symbol.upper()

        if symbol not in self.state.expected_positions:
            self.state.expected_positions[symbol] = {"quantity": 0, "cost_basis": 0}

        current = self.state.expected_positions[symbol]
        old_qty = current["quantity"]
        old_cost = current["cost_basis"]

        new_qty = old_qty + qty_change
        if new_qty > 0:
            # Weighted average cost basis
            total_cost = (old_qty * old_cost) + (qty_change * price)
            new_cost = total_cost / new_qty
        else:
            new_cost = 0

        self.state.expected_positions[symbol] = {
            "quantity": new_qty,
            "cost_basis": new_cost,
            "updated_at": datetime.now(self.et_tz).isoformat()
        }

        # Remove if position closed
        if abs(new_qty) < 0.01:
            del self.state.expected_positions[symbol]
            logger.debug(f"Position closed: {symbol}")

    def detect_drift(self) -> List[PositionDrift]:
        """
        Compare expected positions with actual broker positions.
        Returns list of drift events.
        """
        drifts = []
        now = datetime.now(self.et_tz)

        # Build actual positions map
        actual_positions = {}
        for pos in self.state.positions:
            symbol = pos.get("symbol", "").upper()
            actual_positions[symbol] = {
                "quantity": float(pos.get("qty", pos.get("quantity", 0))),
                "cost_basis": float(pos.get("cost_basis", pos.get("avg_entry_price", 0)))
            }

        # Check expected vs actual
        all_symbols = set(self.state.expected_positions.keys()) | set(actual_positions.keys())

        for symbol in all_symbols:
            expected = self.state.expected_positions.get(symbol, {"quantity": 0, "cost_basis": 0})
            actual = actual_positions.get(symbol, {"quantity": 0, "cost_basis": 0})

            expected_qty = expected["quantity"]
            actual_qty = actual["quantity"]

            # Calculate drift
            drift_qty = actual_qty - expected_qty
            drift_percent = 0

            if expected_qty != 0:
                drift_percent = abs(drift_qty / expected_qty) * 100
            elif actual_qty != 0:
                drift_percent = 100  # Unexpected position

            # Report significant drift (more than 1 share or 5%)
            if abs(drift_qty) >= 1 or drift_percent >= 5:
                drift = PositionDrift(
                    symbol=symbol,
                    expected_qty=expected_qty,
                    actual_qty=actual_qty,
                    expected_cost=expected["cost_basis"],
                    actual_cost=actual["cost_basis"],
                    drift_qty=drift_qty,
                    drift_percent=drift_percent,
                    detected_at=now
                )
                drifts.append(drift)
                self.state.drift_events.append(drift)

                if actual_qty > expected_qty:
                    logger.warning(f"DRIFT: {symbol} has MORE shares than expected: {actual_qty} vs {expected_qty}")
                elif actual_qty < expected_qty:
                    logger.warning(f"DRIFT: {symbol} has FEWER shares than expected: {actual_qty} vs {expected_qty}")
                else:
                    logger.warning(f"DRIFT: Unexpected position in {symbol}: {actual_qty} shares")

        self.state.last_drift_check = now
        return drifts

    def reconcile_positions(self):
        """
        Reconcile expected positions with actual broker positions.
        Updates expected to match actual (assumes broker is source of truth).
        """
        reconciled_count = 0

        for pos in self.state.positions:
            symbol = pos.get("symbol", "").upper()
            actual_qty = float(pos.get("qty", pos.get("quantity", 0)))
            actual_cost = float(pos.get("cost_basis", pos.get("avg_entry_price", 0)))

            # Update expected to match actual
            if actual_qty > 0:
                self.state.expected_positions[symbol] = {
                    "quantity": actual_qty,
                    "cost_basis": actual_cost,
                    "reconciled_at": datetime.now(self.et_tz).isoformat()
                }
                reconciled_count += 1

        # Remove expected positions that don't exist in broker
        actual_symbols = {pos.get("symbol", "").upper() for pos in self.state.positions}
        to_remove = [s for s in self.state.expected_positions if s not in actual_symbols]
        for symbol in to_remove:
            del self.state.expected_positions[symbol]

        logger.info(f"Reconciled {reconciled_count} positions, removed {len(to_remove)} stale expected positions")

        # Mark all drift events as reconciled
        for drift in self.state.drift_events:
            if not drift.reconciled:
                drift.reconciled = True

        return {"reconciled": reconciled_count, "removed": len(to_remove)}

    def get_drift_summary(self) -> Dict:
        """Get summary of drift detection status"""
        recent_drifts = [d for d in self.state.drift_events if not d.reconciled]

        return {
            "expected_positions": len(self.state.expected_positions),
            "actual_positions": len(self.state.positions),
            "unreconciled_drifts": len(recent_drifts),
            "total_drift_events": len(self.state.drift_events),
            "last_drift_check": self.state.last_drift_check.isoformat() if self.state.last_drift_check else None,
            "drift_details": [
                {
                    "symbol": d.symbol,
                    "expected": d.expected_qty,
                    "actual": d.actual_qty,
                    "drift": d.drift_qty,
                    "percent": round(d.drift_percent, 2),
                    "detected_at": d.detected_at.isoformat()
                }
                for d in recent_drifts
            ]
        }

    def get_sync_status(self) -> Dict:
        """Get current sync status with drift detection info"""
        drift_summary = self.get_drift_summary()

        return {
            "last_sync": self.state.last_sync.isoformat() if self.state.last_sync else None,
            "positions_count": len(self.state.positions),
            "open_orders_count": len(self.state.open_orders),
            "tracked_orders": len(self.state.order_trackers),
            "sync_errors": self.state.sync_errors,
            "recent_slippage_events": self.state.slippage_events[-10:],
            "is_running": self._running,
            "drift_detection": {
                "expected_positions": drift_summary["expected_positions"],
                "actual_positions": drift_summary["actual_positions"],
                "unreconciled_drifts": drift_summary["unreconciled_drifts"],
                "last_check": drift_summary["last_drift_check"]
            }
        }

    def get_positions(self) -> List[Dict]:
        """Get current synced positions"""
        return self.state.positions

    def get_orders(self) -> List[Dict]:
        """Get current synced orders"""
        return self.state.open_orders

    def force_sync(self) -> Dict:
        """Force immediate sync"""
        return self.sync_now()


# Singleton instance
_position_sync: Optional[PositionSynchronizer] = None


def get_position_sync() -> PositionSynchronizer:
    """Get or create the position synchronizer singleton"""
    global _position_sync
    if _position_sync is None:
        _position_sync = PositionSynchronizer()
    return _position_sync


def start_sync():
    """Start background position sync"""
    sync = get_position_sync()
    sync.start_background_sync()
    return sync


def stop_sync():
    """Stop background position sync"""
    sync = get_position_sync()
    sync.stop_background_sync()
