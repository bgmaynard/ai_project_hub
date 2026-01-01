"""
Real-Time Hub - WebSocket Manager for Live Updates
===================================================
Centralized hub for pushing real-time updates to connected clients.

Channels:
- trading: Bot status, trades, positions, orders
- ai: Predictions, signals, brain status, drift alerts
- risk: VAR, drawdown, circuit breaker, portfolio guard
- market: Prices, quotes (handled separately by stream manager)

Usage:
    from ai.realtime_hub import get_realtime_hub, broadcast

    # Broadcast update to all clients
    await broadcast("trading", {"type": "trade", "symbol": "AAPL", "pnl": 150})

    # Or use the hub directly
    hub = get_realtime_hub()
    await hub.broadcast("risk", {"type": "drawdown_alert", "pct": 5.2})

Created: December 2025
"""

import asyncio
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set

import pytz

logger = logging.getLogger(__name__)


class Channel(str, Enum):
    """WebSocket channels"""

    TRADING = "trading"  # Bot, trades, positions
    AI = "ai"  # Predictions, signals, brain
    RISK = "risk"  # VAR, drawdown, alerts
    SYSTEM = "system"  # Server status, errors
    ALL = "all"  # Broadcast to all channels


@dataclass
class RealtimeMessage:
    """Standard message format for real-time updates"""

    channel: str
    type: str
    data: Dict[str, Any]
    timestamp: str

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    def to_dict(self) -> Dict:
        return asdict(self)


class RealtimeHub:
    """
    Central hub for managing WebSocket connections and broadcasting updates.

    Features:
    - Multiple channels (trading, ai, risk, system)
    - Client subscription management
    - Message queuing per client
    - Automatic cleanup on disconnect
    - Rate limiting (optional)
    """

    def __init__(self):
        self.et_tz = pytz.timezone("US/Eastern")

        # Client management
        # client_id -> {queue, channels}
        self._clients: Dict[str, Dict] = {}

        # Channel subscriptions
        # channel -> set of client_ids
        self._channel_subscriptions: Dict[str, Set[str]] = {
            Channel.TRADING.value: set(),
            Channel.AI.value: set(),
            Channel.RISK.value: set(),
            Channel.SYSTEM.value: set(),
        }

        # Message history (last N messages per channel)
        self._history: Dict[str, List[RealtimeMessage]] = {
            Channel.TRADING.value: [],
            Channel.AI.value: [],
            Channel.RISK.value: [],
            Channel.SYSTEM.value: [],
        }
        self._history_limit = 50

        # Stats
        self._messages_sent = 0
        self._messages_by_channel: Dict[str, int] = {
            c.value: 0 for c in Channel if c != Channel.ALL
        }

        logger.info("RealtimeHub initialized")

    def register_client(
        self, client_id: str, channels: List[str] = None
    ) -> asyncio.Queue:
        """
        Register a new WebSocket client.

        Args:
            client_id: Unique client identifier
            channels: List of channels to subscribe to (default: all)

        Returns:
            asyncio.Queue for receiving messages
        """
        if channels is None:
            channels = [
                Channel.TRADING.value,
                Channel.AI.value,
                Channel.RISK.value,
                Channel.SYSTEM.value,
            ]

        queue = asyncio.Queue()

        self._clients[client_id] = {
            "queue": queue,
            "channels": set(channels),
            "connected_at": datetime.now(self.et_tz).isoformat(),
        }

        # Add to channel subscriptions
        for channel in channels:
            if channel in self._channel_subscriptions:
                self._channel_subscriptions[channel].add(client_id)

        logger.info(f"Client {client_id} registered for channels: {channels}")
        return queue

    def unregister_client(self, client_id: str):
        """Remove a client and clean up subscriptions"""
        if client_id in self._clients:
            client = self._clients.pop(client_id)

            # Remove from all channel subscriptions
            for channel in client.get("channels", []):
                if channel in self._channel_subscriptions:
                    self._channel_subscriptions[channel].discard(client_id)

            logger.info(f"Client {client_id} unregistered")

    def subscribe(self, client_id: str, channels: List[str]):
        """Subscribe a client to additional channels"""
        if client_id not in self._clients:
            return

        for channel in channels:
            if channel in self._channel_subscriptions:
                self._channel_subscriptions[channel].add(client_id)
                self._clients[client_id]["channels"].add(channel)

    def unsubscribe(self, client_id: str, channels: List[str]):
        """Unsubscribe a client from channels"""
        if client_id not in self._clients:
            return

        for channel in channels:
            if channel in self._channel_subscriptions:
                self._channel_subscriptions[channel].discard(client_id)
                self._clients[client_id]["channels"].discard(channel)

    async def broadcast(self, channel: str, message_type: str, data: Dict[str, Any]):
        """
        Broadcast a message to all clients subscribed to a channel.

        Args:
            channel: Channel name (trading, ai, risk, system, all)
            message_type: Type of message (e.g., "trade", "prediction", "alert")
            data: Message payload
        """
        timestamp = datetime.now(self.et_tz).isoformat()

        message = RealtimeMessage(
            channel=channel, type=message_type, data=data, timestamp=timestamp
        )

        message_json = message.to_json()

        # Determine target clients
        if channel == Channel.ALL.value:
            target_clients = set(self._clients.keys())
        else:
            target_clients = self._channel_subscriptions.get(channel, set())

        # Add to history
        if channel in self._history:
            self._history[channel].append(message)
            if len(self._history[channel]) > self._history_limit:
                self._history[channel] = self._history[channel][-self._history_limit :]

        # Send to all target clients
        for client_id in target_clients:
            if client_id in self._clients:
                try:
                    await self._clients[client_id]["queue"].put(message_json)
                    self._messages_sent += 1
                    if channel in self._messages_by_channel:
                        self._messages_by_channel[channel] += 1
                except Exception as e:
                    logger.error(f"Error sending to client {client_id}: {e}")

    async def send_to_client(
        self, client_id: str, message_type: str, data: Dict[str, Any]
    ):
        """Send a message to a specific client"""
        if client_id not in self._clients:
            return

        timestamp = datetime.now(self.et_tz).isoformat()

        message = RealtimeMessage(
            channel="direct", type=message_type, data=data, timestamp=timestamp
        )

        try:
            await self._clients[client_id]["queue"].put(message.to_json())
        except Exception as e:
            logger.error(f"Error sending to client {client_id}: {e}")

    def get_history(self, channel: str, limit: int = 20) -> List[Dict]:
        """Get recent message history for a channel"""
        if channel not in self._history:
            return []

        messages = self._history[channel][-limit:]
        return [m.to_dict() for m in messages]

    def get_status(self) -> Dict:
        """Get hub status and stats"""
        return {
            "connected_clients": len(self._clients),
            "channel_subscribers": {
                c: len(s) for c, s in self._channel_subscriptions.items()
            },
            "messages_sent": self._messages_sent,
            "messages_by_channel": self._messages_by_channel,
            "clients": [
                {
                    "id": cid,
                    "channels": list(c.get("channels", [])),
                    "connected_at": c.get("connected_at"),
                }
                for cid, c in self._clients.items()
            ],
        }


# Singleton instance
_realtime_hub: Optional[RealtimeHub] = None


def get_realtime_hub() -> RealtimeHub:
    """Get or create the realtime hub singleton"""
    global _realtime_hub
    if _realtime_hub is None:
        _realtime_hub = RealtimeHub()
    return _realtime_hub


async def broadcast(channel: str, data: Dict[str, Any], message_type: str = "update"):
    """Convenience function to broadcast a message"""
    hub = get_realtime_hub()
    await hub.broadcast(channel, message_type, data)


# ═══════════════════════════════════════════════════════════════════════════════
#                    PRE-BUILT BROADCAST FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════


async def broadcast_trade(
    symbol: str, side: str, quantity: int, price: float, pnl: float = None
):
    """Broadcast a trade execution"""
    data = {
        "symbol": symbol,
        "side": side,
        "quantity": quantity,
        "price": price,
        "pnl": pnl,
    }
    await broadcast(Channel.TRADING.value, data, "trade")


async def broadcast_position_update(positions: List[Dict]):
    """Broadcast position updates"""
    await broadcast(Channel.TRADING.value, {"positions": positions}, "positions")


async def broadcast_bot_status(status: str, message: str = None):
    """Broadcast bot status change"""
    data = {"status": status, "message": message}
    await broadcast(Channel.TRADING.value, data, "bot_status")


async def broadcast_prediction(
    symbol: str, signal: str, confidence: float, price: float
):
    """Broadcast AI prediction"""
    data = {
        "symbol": symbol,
        "signal": signal,
        "confidence": confidence,
        "price": price,
    }
    await broadcast(Channel.AI.value, data, "prediction")


async def broadcast_brain_status(metrics: Dict):
    """Broadcast brain metrics update"""
    await broadcast(Channel.AI.value, metrics, "brain_status")


async def broadcast_drift_alert(drift_pct: float, recommendation: str):
    """Broadcast prediction drift alert"""
    data = {"drift_pct": drift_pct, "recommendation": recommendation}
    await broadcast(Channel.AI.value, data, "drift_alert")


async def broadcast_var_update(var_95_pct: float, var_95_dollars: float):
    """Broadcast VAR update"""
    data = {"var_95_pct": var_95_pct, "var_95_dollars": var_95_dollars}
    await broadcast(Channel.RISK.value, data, "var")


async def broadcast_drawdown_update(drawdown_pct: float, max_drawdown_pct: float):
    """Broadcast drawdown update"""
    data = {"drawdown_pct": drawdown_pct, "max_drawdown_pct": max_drawdown_pct}
    await broadcast(Channel.RISK.value, data, "drawdown")


async def broadcast_circuit_breaker(level: str, reason: str, can_trade: bool):
    """Broadcast circuit breaker status"""
    data = {"level": level, "reason": reason, "can_trade": can_trade}
    await broadcast(Channel.RISK.value, data, "circuit_breaker")


async def broadcast_risk_alert(
    alert_type: str, message: str, severity: str = "warning"
):
    """Broadcast risk alert"""
    data = {"alert_type": alert_type, "message": message, "severity": severity}
    await broadcast(Channel.RISK.value, data, "alert")


async def broadcast_system_status(status: str, details: Dict = None):
    """Broadcast system status"""
    data = {"status": status, "details": details or {}}
    await broadcast(Channel.SYSTEM.value, data, "status")


async def broadcast_error(error: str, component: str = None):
    """Broadcast system error"""
    data = {"error": error, "component": component}
    await broadcast(Channel.SYSTEM.value, data, "error")
