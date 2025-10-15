from IBKR_Algo_BOT.broker_if import Order
from IBKR_Algo_BOT.order_router import OrderRouter
from IBKR_Algo_BOT.risk_manager import RiskManager, RiskConfig

class DummyBroker:
    def __init__(self):
        self._connected=False
        self.submitted=[]
    def connect(self):
        self._connected=True
    def is_connected(self):
        return self._connected
    def submit_order(self, order):
        self.submitted.append(order)
        return "OID1"
    def cancel_order(self, oid):
        return True
    def get_positions(self):
        return []
    def subscribe_market_data(self, symbols):
        pass
    def poll_events(self):
        return []

def test_blocks_huge_notional():
    risk = RiskManager(RiskConfig(max_position_usd=100, per_trade_risk_usd=10))
    brk  = DummyBroker()
    rt   = OrderRouter(brk, risk)
    oid  = rt.submit(Order(symbol="AAPL", side="BUY", qty=10, type="MKT"), mark_price=50.0)
    assert oid is None   # 10*50 = 500 > max_position_usd

def test_allows_small_trade():
    risk = RiskManager(RiskConfig(max_position_usd=1000, per_trade_risk_usd=50))
    brk  = DummyBroker()
    rt   = OrderRouter(brk, risk)
    oid  = rt.submit(Order(symbol="AAPL", side="BUY", qty=1, type="MKT"), mark_price=50.0)
    assert oid == "OID1"
    assert len(brk.submitted) == 1
