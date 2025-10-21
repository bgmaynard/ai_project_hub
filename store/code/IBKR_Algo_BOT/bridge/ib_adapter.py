
from dataclasses import dataclass
from typing import Optional, Dict, Any
import time

try:
    from ib_insync import IB, Stock, MarketOrder, LimitOrder
except Exception:
    IB = None

@dataclass
class IBConfig:
    host: str = "127.0.0.1"
    port: int = 7497
    client_id: int = 1
    read_only: bool = True

class IBAdapter:
    def __init__(self, cfg: IBConfig):
        self.cfg = cfg
        self.connected = False
        self.ib: Optional['IB'] = None

    def connect(self) -> bool:
        if IB is None:
            self.connected = False
            return False
        self.ib = IB()
        try:
            self.ib.connect(self.cfg.host, self.cfg.port, clientId=self.cfg.client_id)
            self.connected = self.ib.isConnected()
            return self.connected
        except Exception:
            self.connected = False
            return False

    def _contract(self, symbol: str):
        return Stock(symbol.upper(), 'SMART', 'USD')

    def preview_order(self, symbol: str, side: str, qty: int, limitPrice: float = None) -> Dict[str, Any]:
        if not self.connected or self.ib is None:
            raise RuntimeError("IB not connected")
        contract = self._contract(symbol)
        ot = 'BUY' if side.upper()=='BUY' else 'SELL'
        order = LimitOrder(ot, int(qty), limitPrice) if limitPrice is not None else MarketOrder(ot, int(qty))
        try:
            preview = self.ib.whatIfOrder(contract, order)
            return {
                "ok": True,
                "symbol": symbol.upper(),
                "side": ot,
                "qty": int(qty),
                "estimatedCommission": getattr(preview, 'commission', None),
                "initialMarginChange": getattr(preview, 'initMarginChange', None),
                "maintenanceMarginChange": getattr(preview, 'maintMarginChange', None),
                "equityWithLoanChange": getattr(preview, 'equityWithLoanChange', None),
                "warningText": getattr(preview, 'warningText', None)
            }
        except Exception as e:
            return { "ok": False, "error": str(e) }

    def place_order(self, symbol: str, side: str, qty: int, limitPrice: float = None) -> Dict[str, Any]:
        if self.cfg.read_only:
            raise PermissionError("Adapter in read_only mode; refusing to place live order.")
        if not self.connected or self.ib is None:
            raise RuntimeError("IB not connected")
        contract = self._contract(symbol)
        ot = 'BUY' if side.upper()=='BUY' else 'SELL'
        order = LimitOrder(ot, int(qty), limitPrice) if limitPrice is not None else MarketOrder(ot, int(qty))
        trade = self.ib.placeOrder(contract, order)
        t0 = time.time()
        while trade.order.orderId is None and time.time()-t0 < 2.0:
            self.ib.waitOnUpdate(timeout=0.2)
        return {
            "orderId": trade.order.orderId,
            "status": trade.orderStatus.status if trade.orderStatus else "submitted",
            "symbol": symbol.upper(),
            "side": ot,
            "qty": int(qty),
            "limitPrice": limitPrice
        }
