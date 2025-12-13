"""
Test NASDAQ TotalView Connection
Save as: totalview_test.py
Run with: python totalview_test.py

This will verify your TotalView subscription and show live Level 2 data
"""

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from datetime import datetime
from collections import defaultdict
import threading
import time

class TotalViewTest(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.connected = False
        self.next_req_id = 1000
        self.market_depth = defaultdict(lambda: {'bids': {}, 'asks': {}})
        self.update_count = 0
        
    def nextValidId(self, orderId: int):
        super().nextValidId(orderId)
        self.next_req_id = orderId
        self.connected = True
        print("✓ Connected to IBKR TWS/Gateway")
        print(f"   Next Order ID: {orderId}")
        
    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        if errorCode == 2104 or errorCode == 2106 or errorCode == 2158:
            # Market data farm connection messages (informational)
            print(f"ℹ Market Data: {errorString}")
        elif errorCode == 10167:
            print(f"⚠ {errorString}")
            print("   Make sure TotalView is enabled in TWS Market Data subscriptions")
        else:
            print(f"⚠ Error {errorCode}: {errorString}")
    
    def updateMktDepth(self, reqId, position, operation, side, price, size):
        """Level 2 update received"""
        self.update_count += 1
        
        # Get symbol name from reqId (simplified for test)
        symbol = "AAPL"  # We know we're testing AAPL
        
        book = self.market_depth[symbol]
        side_name = "ASK" if side == 0 else "BID"
        side_dict = book['asks'] if side == 0 else book['bids']
        
        # Update book
        if operation == 0 or operation == 1:  # Insert or Update
            side_dict[price] = size
        elif operation == 2:  # Delete
            side_dict.pop(price, None)
        
        # Print first few updates to show it's working
        if self.update_count <= 10:
            op_name = {0: "INSERT", 1: "UPDATE", 2: "DELETE"}[operation]
            print(f"   Update #{self.update_count}: {side_name} {op_name} @ ${price:.2f} x {size}")
    
    def updateMktDepthL2(self, reqId, position, marketMaker, operation, side, price, size, isSmartDepth):
        """TotalView Level 2 with market maker info"""
        self.updateMktDepth(reqId, position, operation, side, price, size)
    
    def display_order_book(self):
        """Display current order book"""
        symbol = "AAPL"
        book = self.market_depth[symbol]
        
        if not book['bids'] or not book['asks']:
            print("   Waiting for order book data...")
            return
        
        bids = sorted(book['bids'].items(), reverse=True)[:10]
        asks = sorted(book['asks'].items())[:10]
        
        print(f"\n{'='*60}")
        print(f"ORDER BOOK SNAPSHOT - {symbol} @ {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*60}")
        print(f"{'ASK PRICE':<12} {'SIZE':<12} | {'BID PRICE':<12} {'SIZE':<12}")
        print(f"{'-'*60}")
        
        for i in range(max(len(asks), len(bids))):
            ask_price = f"${asks[i][0]:.2f}" if i < len(asks) else ""
            ask_size = f"{asks[i][1]:,}" if i < len(asks) else ""
            bid_price = f"${bids[i][0]:.2f}" if i < len(bids) else ""
            bid_size = f"{bids[i][1]:,}" if i < len(bids) else ""
            
            print(f"{ask_price:<12} {ask_size:<12} | {bid_price:<12} {bid_size:<12}")
        
        # Calculate metrics
        best_bid = bids[0][0]
        best_ask = asks[0][0]
        spread = best_ask - best_bid
        mid = (best_bid + best_ask) / 2
        spread_bps = (spread / mid) * 10000
        
        bid_vol = sum(size for _, size in bids[:5])
        ask_vol = sum(size for _, size in asks[:5])
        imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol) if (bid_vol + ask_vol) > 0 else 0
        
        print(f"{'-'*60}")
        print(f"Mid Price: ${mid:.2f}")
        print(f"Spread: ${spread:.2f} ({spread_bps:.1f} bps)")
        print(f"Order Imbalance: {imbalance:+.3f} {'(BUY PRESSURE)' if imbalance > 0 else '(SELL PRESSURE)'}")
        print(f"Top 5 Bid Volume: {bid_vol:,} shares")
        print(f"Top 5 Ask Volume: {ask_vol:,} shares")
        print(f"Total Updates Received: {self.update_count}")
        print(f"{'='*60}\n")


def run_test():
    """Run TotalView test"""
    print("="*60)
    print("NASDAQ TOTALVIEW CONNECTION TEST")
    print("="*60)
    
    # Create client
    app = TotalViewTest()
    
    print("\n1. Connecting to IBKR...")
    print("   Make sure TWS or IB Gateway is running!")
    
    # Try common ports
    ports = [
        (7497, "TWS Paper Trading"),
        (7496, "TWS Live Trading"),
        (4002, "IB Gateway Paper"),
        (4001, "IB Gateway Live")
    ]
    
    connected = False
    for port, name in ports:
        try:
            print(f"   Trying {name} (port {port})...", end=" ")
            app.connect("127.0.0.1", port, clientId=1)
            
            # Start message processing thread
            api_thread = threading.Thread(target=app.run, daemon=True)
            api_thread.start()
            
            # Wait for connection
            time.sleep(2)
            
            if app.connected:
                print("✓ Connected!")
                connected = True
                break
            else:
                print("✗ No response")
                app.disconnect()
        except Exception as e:
            print(f"✗ Failed: {e}")
    
    if not connected:
        print("\n❌ Could not connect to IBKR")
        print("\nTroubleshooting:")
        print("  1. Make sure TWS or IB Gateway is running")
        print("  2. Check API settings in TWS:")
        print("     - File → Global Configuration → API → Settings")
        print("     - Enable ActiveX and Socket Clients: ✓")
        print("     - Socket port should match (7497 for paper, 7496 for live)")
        print("     - Trusted IP: 127.0.0.1")
        return
    
    print("\n2. Subscribing to AAPL Level 2 Market Depth...")
    
    # Create contract
    contract = Contract()
    contract.symbol = "AAPL"
    contract.secType = "STK"
    contract.exchange = "SMART"
    contract.currency = "USD"
    
    # Subscribe to market depth
    # num_rows=20 gets more depth levels (TotalView supports up to 50+)
    req_id = app.next_req_id
    app.reqMktDepth(req_id, contract, numRows=20, isSmartDepth=True, mktDepthOptions=[])
    
    print("   Waiting for Level 2 data...")
    print("   (First 10 updates will be displayed)\n")
    
    time.sleep(3)
    
    if app.update_count == 0:
        print("\n⚠ No Level 2 updates received")
        print("\nPossible issues:")
        print("  1. TotalView subscription not active")
        print("     → Check TWS: Global Config → Market Data → Subscriptions")
        print("  2. Market is closed")
        print("     → Try during market hours (9:30 AM - 4:00 PM ET)")
        print("  3. Symbol has no market depth data")
        print("     → AAPL should always have data during market hours")
        return
    
    print(f"\n✓ Receiving Level 2 data! ({app.update_count} updates so far)")
    
    # Display order book snapshots every 5 seconds
    print("\n3. Displaying live order book (press Ctrl+C to stop)...")
    
    try:
        for i in range(6):  # Show 6 snapshots (30 seconds)
            time.sleep(5)
            app.display_order_book()
    except KeyboardInterrupt:
        print("\n\nStopped by user")
    
    # Disconnect
    print("\n4. Disconnecting...")
    app.disconnect()
    
    print("\n" + "="*60)
    print("✓ TOTALVIEW TEST COMPLETE!")
    print("="*60)
    
    if app.update_count > 0:
        print("\n✅ TotalView is working correctly!")
        print(f"   Received {app.update_count} Level 2 updates")
        print("\nNext steps:")
        print("  1. Let basic LSTM training finish")
        print("  2. Run: python train_with_totalview.py")
        print("  3. Enhanced LSTM will use Level 2 features")
    else:
        print("\n⚠ TotalView not receiving data")
        print("   Check subscription status in TWS")


if __name__ == "__main__":
    run_test()
