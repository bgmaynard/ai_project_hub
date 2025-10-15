from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import Order
import threading
import time

class TestWrapper(EWrapper):
    def __init__(self):
        super().__init__()
        self.connected_event = threading.Event()
        self.next_order_id = None
    
    def connectAck(self):
        print("✅ Connected to TWS!")
        self.connected_event.set()
    
    def nextValidId(self, orderId):
        self.next_order_id = orderId
        print(f"✅ Next order ID: {orderId}")
    
    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=''):
        print(f"TWS Message [{errorCode}]: {errorString}")

class TestClient(EClient):
    def __init__(self, wrapper):
        super().__init__(wrapper)

print("="*60)
print("Testing IBKR Connection")
print("="*60)

wrapper = TestWrapper()
client = TestClient(wrapper)

print("\nConnecting to TWS at 127.0.0.1:7497...")
client.connect('127.0.0.1', 7497, 1)

def run():
    client.run()

thread = threading.Thread(target=run, daemon=True)
thread.start()

print("Waiting for connection...")
if wrapper.connected_event.wait(timeout=5):
    print("\n✅ CONNECTION SUCCESSFUL!")
    print(f"Next Order ID: {wrapper.next_order_id}")
    
    # Try to place test order
    print("\nPlacing test order...")
    contract = Contract()
    contract.symbol = "AAPL"
    contract.secType = "STK"
    contract.exchange = "SMART"
    contract.currency = "USD"
    
    order = Order()
    order.action = "BUY"
    order.totalQuantity = 1
    order.orderType = "MKT"
    
    if wrapper.next_order_id:
        client.placeOrder(wrapper.next_order_id, contract, order)
        print(f"✅ Order placed with ID: {wrapper.next_order_id}")
        print("Check TWS to see the order!")
    
    time.sleep(5)
    client.disconnect()
else:
    print("\n❌ CONNECTION TIMEOUT")
    print("Make sure:")
    print("  1. TWS is running")
    print("  2. API is enabled (File → Global Configuration → API → Settings)")
    print("  3. Port is 7497")

print("\n" + "="*60)