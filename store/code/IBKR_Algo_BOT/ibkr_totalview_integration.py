from __future__ import annotations
import threading
import time
import queue
from typing import Dict, Any, Optional

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract

class TotalViewBook:
    def __init__(self, levels: int = 10):
        self.levels = levels
        self._bids: Dict[float, float] = {}
        self._asks: Dict[float, float] = {}
        self._lock = threading.Lock()

    def _sorted(self, side: str):
        if side == "bids":
            items = sorted(self._bids.items(), key=lambda x: (-x[0],))
        else:
            items = sorted(self._asks.items(), key=lambda x: (x[0],))
        return [{"price": float(p), "size": float(sz)} for p, sz in items if sz > 0][: self.levels]

    def snapshot(self):
        with self._lock:
            return {"bids": self._sorted("bids"), "asks": self._sorted("asks")}

    def apply_l2(self, side: int, operation: int, price: float, size: float):
        book = self._bids if side == 1 else self._asks
        with self._lock:
            if operation == 2:  # remove
                if price in book:
                    del book[price]
            else:               # insert/update
                book[price] = size


class _TVWrapper(EWrapper):
    def __init__(self, evt_q: "queue.Queue[Dict[str, Any]]", id_to_symbol: Dict[int, str],
                 books: Dict[str, TotalViewBook], connected_evt: threading.Event):
        super().__init__()
        self.evt_q = evt_q
        self.id_to_symbol = id_to_symbol
        self.books = books
        self._connected_evt = connected_evt

    # connection lifecycle
    def nextValidId(self, orderId: int):
        self._connected_evt.set()

    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        code = int(errorCode)
        kind = "info" if code in {2104, 2106, 2158} else "error"
        self.evt_q.put({"type": kind, "code": code, "msg": str(errorString), "reqId": int(reqId)})

    # depth handlers
    def updateMktDepthL2(self, reqId, position, marketMaker, operation, side, price, size, isSmartDepth):
        sym = self.id_to_symbol.get(int(reqId))
        if not sym:
            return
        book = self.books.setdefault(sym, TotalViewBook(levels=10))
        book.apply_l2(int(side), int(operation), float(price), float(size))
        self.evt_q.put({"type": "depth", "symbol": sym, "reqId": int(reqId),
                        "position": int(position), "mm": str(marketMaker),
                        "op": int(operation), "side": int(side),
                        "price": float(price), "size": float(size),
                        "smart": bool(isSmartDepth)})

    def updateMktDepth(self, reqId, position, operation, side, price, size):
        sym = self.id_to_symbol.get(int(reqId))
        if not sym:
            return
        book = self.books.setdefault(sym, TotalViewBook(levels=10))
        book.apply_l2(int(side), int(operation), float(price), float(size))
        self.evt_q.put({"type": "depth", "symbol": sym, "reqId": int(reqId),
                        "position": int(position), "mm": "",
                        "op": int(operation), "side": int(side),
                        "price": float(price), "size": float(size),
                        "smart": False})


class TotalViewDataHandler(EClient, _TVWrapper):
    def __init__(self, host="127.0.0.1", port=7497, clientId=42,
                 rows=10, smart_depth=True, market_data_type=1, exchange: Optional[str]=None):
        self._evt_q: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self._id_to_symbol: Dict[int, str] = {}
        self._books: Dict[str, TotalViewBook] = {}
        self._connected_evt = threading.Event()
        _TVWrapper.__init__(self, self._evt_q, self._id_to_symbol, self._books, self._connected_evt)
        EClient.__init__(self, wrapper=self)

        self.host, self.port, self.clientId = host, int(port), int(clientId)
        self.rows, self.smart_depth = int(rows), bool(smart_depth)
        self.market_data_type = int(market_data_type)  # 1=live, 3=delayed
        self.exchange = (exchange or "").upper() or None  # NYSE, ARCA, ISLAND, BATS, IEX

        self._nextDepthId = 7000
        self._thread: Optional[threading.Thread] = None

    def connect(self):
        EClient.connect(self, self.host, self.port, self.clientId)
        self._thread = threading.Thread(target=self.run, name="IBKR-Reader", daemon=True)
        self._thread.start()

    def wait_connected(self, timeout: float = 7.0) -> bool:
        if self.isConnected():
            return True
        return self._connected_evt.wait(timeout)

    def after_connected_setup(self):
        # choose market data type
        try:
            self.reqMarketDataType(self.market_data_type)
        except Exception:
            pass
        # tiny settle delay helps avoid 504 immediately after nextValidId
        time.sleep(0.5)

    def close(self):
        try:
            EClient.disconnect(self)
        except Exception:
            pass

    def _new_req_id(self) -> int:
        rid = self._nextDepthId
        self._nextDepthId += 1
        return rid

    def _stock_contract(self, symbol: str, smart_depth: bool) -> Contract:
        c = Contract()
        c.symbol = symbol.upper()
        c.secType = "STK"
        c.currency = "USD"

        if self.exchange:
            # explicit venue (e.g., NYSE/ARCA/ISLAND/BATS/IEX)
            c.exchange = self.exchange
            c.primaryExchange = "NASDAQ" if self.exchange in {"ISLAND"} else self.exchange
        else:
            # SMART vs venue default (ISLAND for NASDAQ TotalView)
            if smart_depth:
                c.exchange = "SMART"
                c.primaryExchange = "NASDAQ"
            else:
                c.exchange = "ISLAND"
                c.primaryExchange = "NASDAQ"
        return c

    def subscribe_depth(self, symbol: str, rows=None, smart_depth=None) -> int:
        rows = self.rows if rows is None else int(rows)
        smart_depth = self.smart_depth if smart_depth is None else bool(smart_depth)
        c = self._stock_contract(symbol, smart_depth)
        req_id = self._new_req_id()
        self._id_to_symbol[req_id] = symbol.upper()
        print(f"Requesting depth: symbol={symbol} rows={rows} smart={smart_depth} exchange={c.exchange}")
        # robust: try up to 3 times if 504 occurs shortly after connect
        for attempt in range(3):
            try:
                self.reqMktDepth(req_id, c, int(rows), bool(smart_depth), [])
                break
            except Exception:
                time.sleep(0.5)
        return req_id

    def unsubscribe_depth(self, req_id: int):
        try:
            self.cancelMktDepth(int(req_id), True)
        except Exception:
            pass
        self._id_to_symbol.pop(int(req_id), None)

    def process_messages(self, timeout: float = 0.5):
        out = []
        t0 = time.time()
        while (time.time() - t0) < timeout:
            try:
                ev = self._evt_q.get(timeout=0.25)
                if ev.get("type") in ("info", "error"):
                    print(f"[{ev.get('type').upper()}] code={ev.get('code')} msg={ev.get('msg')}")
                out.append(ev)
            except queue.Empty:
                break
        return out

    def get_book(self, symbol: str):
        return self._books.get(symbol.upper(), TotalViewBook()).snapshot()
