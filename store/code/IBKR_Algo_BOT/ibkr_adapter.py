import threading, time, queue, logging
from typing import List, Union, Dict
from dataclasses import dataclass
try:
    from .broker_if import BrokerIF, Order, Fill, BrokerError, Position
except ImportError:
    from broker_if import BrokerIF, Order, Fill, BrokerError, Position

# Lazy import to avoid hard dependency during tests
try:
    from ibapi.client import EClient
    from ibapi.wrapper import EWrapper
    from ibapi.contract import Contract
    from ibapi.order import Order as IBOrder
except Exception:  # pragma: no cover
    EClient = object; EWrapper = object; Contract = object; IBOrder = object  # type: ignore

log = logging.getLogger(__name__)

class _IBWrapper(EWrapper):
    INFO_CODES = {2104, 2106, 2158}  # farm/HMDS OK messages

    def __init__(self, evt_q: "queue.Queue[Union[Fill,BrokerError,Dict]]"):
        super().__init__()
        self.evt_q = evt_qdef error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        self.evt_q.put(BrokerError(code=errorCode, message=str(errorString), ctx={"reqId": reqId}))

    def execDetails(self, reqId, contract, execution):
        self.evt_q.put(Fill(
            order_id=str(execution.orderId),
            symbol=contract.symbol,
            side="BUY" if execution.side.upper().startswith("B") else "SELL",
            avg_price=execution.avgPrice,
            qty=execution.shares,
            ts=time.time(),
        ))

class _IBClient(EClient):
    def __init__(self, wrapper):
        EClient.__init__(self, wrapper)

@dataclass
class IBKRConfig:
    host: str = "127.0.0.1"
    port: int = 7497      # TWS paper=7497, live=7496
    client_id: int = 1
    paper: bool = True

def _contract_for_symbol(sym: str) -> "Contract":
    c = Contract()
    c.symbol = sym
    c.secType = "STK"
    c.currency = "USD"
    c.exchange = "SMART"
    return c

def _ib_order(o: Order) -> "IBOrder":
    ib = IBOrder()
    ib.action = "BUY" if o.side == "BUY" else "SELL"
    ib.totalQuantity = o.qty
    ib.orderType = {"MKT":"MKT","LMT":"LMT","STP":"STP","STP_LMT":"STP LMT"}[o.type]
    if o.limit_price is not None: ib.lmtPrice = float(o.limit_price)
    if o.stop_price  is not None: ib.auxPrice = float(o.stop_price)
    ib.tif = o.tif
    return ib

class IBKRAdapter(BrokerIF):
    def __init__(self, cfg: IBKRConfig):
        self.cfg = cfg
        self._evt_q: "queue.Queue[Union[Fill,BrokerError,Dict]]" = queue.Queue()
        self._wrapper = _IBWrapper(self._evt_q)
        self._client = _IBClient(self._wrapper)
        self._thread: threading.Thread | None = None
        self._req_id = 1
        self._connected = False

    def connect(self) -> None:
        if self._connected: return
        log.info("Connecting IBKR %s:%s client_id=%s", self.cfg.host, self.cfg.port, self.cfg.client_id)
        self._client.connect(self.cfg.host, self.cfg.port, self.cfg.client_id)
        self._thread = threading.Thread(target=self._client.run, daemon=True)
        self._thread.start()
        # crude wait; for production, use callbacks/flags
        time.sleep(1.0)
        self._connected = True

    def disconnect(self) -> None:
        try:
            self._client.disconnect()
        finally:
            self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def submit_order(self, order: Order) -> str:
        if not self._connected: raise RuntimeError("IBKR not connected")
        oid = self._next_id()
        self._client.placeOrder(oid, _contract_for_symbol(order.symbol), _ib_order(order))
        return str(oid)

    def cancel_order(self, order_id: str) -> bool:
        if not self._connected: return False
        try:
            self._client.cancelOrder(int(order_id))
            return True
        except Exception:
            return False

    def get_positions(self) -> List[Position]:
        # simple polling via account updates could be added; return empty for now
        return []

    def subscribe_market_data(self, symbols: List[str]) -> None:
        # minimal: snapshot; extend to streaming if needed
        for sym in symbols:
            req_id = self._next_id()
            try:
                self._client.reqMktData(req_id, _contract_for_symbol(sym), "", False, False, [])
            except Exception as e:
                self._evt_q.put(BrokerError(code=-1, message=f"mktdata {sym}: {e}"))

    def poll_events(self):
        evts = []
        while True:
            try:
                evts.append(self._evt_q.get_nowait())
            except queue.Empty:
                break
        return evts

    def _next_id(self) -> int:
        self._req_id += 1
        return self._req_id



