import logging, time
from typing import Optional, List
try:
    from .broker_if import BrokerIF, Order, Fill, BrokerError
except ImportError:  # fallback when run outside package
    from broker_if import BrokerIF, Order, Fill, BrokerError
from .risk_manager import RiskManager

log = logging.getLogger(__name__)

class OrderRouter:
    def __init__(self, broker: BrokerIF, risk: RiskManager):
        self.broker = broker
        self.risk = risk

    def ensure_connected(self):
        if not self.broker.is_connected():
            self.broker.connect()

    def submit(self, order: Order, mark_price: Optional[float]=None, atr: Optional[float]=None) -> Optional[str]:
        self.ensure_connected()
        px = mark_price or order.limit_price or 0.0
        ok, why = self.risk.can_open(order.symbol, px, order.qty)
        if not ok:
            log.warning("BLOCKED %s: %s", order, why)
            return None
        oid = self.broker.submit_order(order)
        log.info("submitted %s -> oid=%s", order, oid)
        return oid

    def pump_events(self) -> List[object]:
        evts = self.broker.poll_events()
        for e in evts:
            if isinstance(e, Fill):
                # positive for sells if short; keep simple for now
                sign = 1.0 if e.side == "SELL" else -1.0
                self.risk.state.update_pnl(sign * e.avg_price * e.qty * 0.0)  # placeholder
            elif isinstance(e, BrokerError):
                log.error("broker error: %s", e)
        return evts

