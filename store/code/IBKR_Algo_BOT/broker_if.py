from dataclasses import dataclass
from typing import Optional, List, Literal, Dict, Union

Side = Literal["BUY", "SELL"]
OrderType = Literal["MKT", "LMT", "STP", "STP_LMT"]
TimeInForce = Literal["DAY", "GTC", "IOC", "FOK"]

@dataclass
class Order:
    symbol: str
    side: Side
    qty: float
    type: OrderType = "MKT"
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    tif: TimeInForce = "DAY"
    meta: Optional[Dict] = None  # strategy tags, correlation id, etc.

@dataclass
class Fill:
    order_id: str
    symbol: str
    side: Side
    avg_price: float
    qty: float
    ts: float

@dataclass
class Position:
    symbol: str
    qty: float
    avg_price: float
    unrealized_pnl: float

@dataclass
class BrokerError:
    code: int
    message: str
    ctx: Optional[Dict] = None

class BrokerIF:
    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    def is_connected(self) -> bool: ...
    def submit_order(self, order: Order) -> str: ...
    def cancel_order(self, order_id: str) -> bool: ...
    def get_positions(self) -> List[Position]: ...
    def subscribe_market_data(self, symbols: List[str]) -> None: ...
    def poll_events(self) -> List[Union[Fill, BrokerError, Dict]]: ...
