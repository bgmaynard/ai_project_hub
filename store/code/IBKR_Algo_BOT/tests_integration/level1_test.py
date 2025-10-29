"""
Test Basic Level 1 Market Data (No TotalView needed)
Save as: level1_test.py
Run with: python level1_test.py

Tests what data you CAN access with basic IBKR subscription
"""

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from datetime import datetime
import threading
import time

class Level1Test(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.connected = False
        self.next_req_id = 1000
        self.tick_data = {}
        
    def nextValidId(self, orderId: int):
        super().nextValidId(orderId)
        self.next_req_id = orderId
        self.connected = True
        print("✓ Connected to IBKR")
        
    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        if errorCode == 2104 or errorCode == 2106 or errorCode == 2158:
            pass  # Connection messages
        else:
            print(f"⚠ Error {errorCode}: {errorString}")
    
    def tickPrice(self, reqId, tickType, price, attrib):
        """Real-time price updates"""
        symbol = "AAPL"  # We know we're testing AAPL
        
        if symbol not in self.tick_data:
            self.tick_data[symbol] = {}
        
        tick_names = {
            1: "BID",
            2: "ASK", 
            4: "LAST",
            6: "HIGH",
            7: "LOW",
            9: "CLOSE"
        }
        
        if tickType in tick_names:
            self.tick_data[symbol][tick_names[tickType]] = price
            print(f"   {tick_names[tickType]}: ${price:.2f}")
    
    def tickSize(self, reqId, tickType, size):
        """Real-time size updates"""
        symbol = "AAPL"
        
        if symbol not in self.tick_data:
            self.tick_data[symbol] = {}
        
        size_names = {
            0: "BID_SIZE",
            3: "ASK_SIZE",
            5: "LAST_SIZE",
            8: "VOLUME"
        }
        
        if tickType in size_names:
            self.tick_data[symbol][size_names[tickType]] = size
            if tickType != 8:  # Don't print volume constantly
                print(f"   {size_names[tickType]}: {size:,}")
    
    def tickString(self, reqId, tickType, value):
        """String-based tick data"""
        if tickType == 45:  # LAST_TIMESTAMP
            symbol = "AAPL"
            if symbol not in self.tick_data:
                self.tick_data[symbol] = {}
            self.tick_data[symbol]['TIMESTAMP'] = value
    
    def display_summary(self):
        """Display current market data"""
        symbol = "AAPL"
        
        if symbol not in self.tick_data or len(self.tick_data[symbol]) == 0:
            print("\n⚠ No data received yet...")
            return
        
        data = self.tick_data[symbol]
        
        print(f"\n{'='*60}")
        print(f"LEVEL 1 MARKET DATA - {symbol} @ {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*60}")
        
        if 'BID' in data and 'ASK' in data:
            bid = data['BID']
            ask = data['ASK']
            mid = (bid + ask) / 2
            spread = ask - bid
            spread_bps = (spread / mid) * 10000
            
            print(f"Bid: ${bid:.2f} x {data.get('BID_SIZE', 0):,}")
            print(f"Ask: ${ask:.2f} x {data.get('ASK_SIZE', 0):,}")
            print(f"Mid: ${mid:.2f}")
            print(f"Spread: ${spread:.2f} ({spread_bps:.1f} bps)")
        
        if 'LAST' in data:
            print(f"Last: ${data['LAST']:.2f} x {data.get('LAST_SIZE', 0):,}")
        
        if 'HIGH' in data:
            print(f"High: ${data['HIGH']:.2f}")
        
        if 'LOW' in data:
            print(f"Low: ${data['LOW']:.2f}")
        
        if 'CLOSE' in data:
            print(f"Prev Close: ${data['CLOSE']:.2f}")
        
        if 'VOLUME' in data:
            print(f"Volume: {data['VOLUME']:,}")
        
        print(f"{'='*60}\n")


def run_test():
    """Run Level 1 data test"""
    print("="*60)
    print("LEVEL 1 MARKET DATA TEST")
    print("="*60)
    print("This tests basic market data (no TotalView needed)")
    
    app = Level1Test()
    
    print("\n1. Connecting to IBKR...")
    
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
            
            api_thread = threading.Thread(target=app.run, daemon=True)
            api_thread.start()
            
            time.sleep(2)
            
            if app.connected:
                print("✓")
                connected = True
                break
            else:
                print("✗")
                app.disconnect()
        except:
            print("✗")
    
    if not connected:
        print("\n❌ Could not connect")
        return
    
    print("\n2. Subscribing to AAPL Level 1 market data...")
    
    # Create contract
    contract = Contract()
    contract.symbol = "AAPL"
    contract.secType = "STK"
    contract.exchange = "SMART"
    contract.currency = "USD"
    
    # Request market data (Level 1 only)
    req_id = app.next_req_id
    app.reqMktData(req_id, contract, "", False, False, [])
    
    print("   Waiting for data...\n")
    time.sleep(3)
    
    if not app.tick_data:
        print("\n⚠ No data received")
        print("\nPossible reasons:")
        print("  1. Market is closed (try 9:30 AM - 4:00 PM ET)")
        print("  2. No market data subscription")
        print("  3. Symbol not found")
        app.disconnect()
        return
    
    print("\n✓ Receiving market data!")
    
    # Display snapshots
    print("\n3. Live market data (press Ctrl+C to stop)...")
    
    try:
        for i in range(6):
            time.sleep(5)
            app.display_summary()
    except KeyboardInterrupt:
        print("\n\nStopped by user")
    
    app.disconnect()
    
    print("\n" + "="*60)
    print("✓ LEVEL 1 TEST COMPLETE!")
    print("="*60)
    
    if app.tick_data:
        print("\n✅ You have access to Level 1 market data!")
        print("\nThis includes:")
        print("  ✓ Best bid/ask prices")
        print("  ✓ Best bid/ask sizes")
        print("  ✓ Last trade price/size")
        print("  ✓ Daily high/low/close")
        print("  ✓ Volume")
        
        print("\n📊 You can still build a great LSTM bot with this data!")
        print("\nYour LSTM will use:")
        print("  • Price action (OHLC)")
        print("  • Volume patterns")
        print("  • Technical indicators (RSI, MACD, etc.)")
        print("  • Bid/Ask spread")
        print("  • Price momentum")
        
        print("\n💡 To get Level 2 (TotalView):")
        print("  1. Check TWS: Account → Market Data Subscriptions")
        print("  2. Ensure NASDAQ TotalView is active")
        print("  3. Contact IBKR support if needed")
    
    print("="*60)


if __name__ == "__main__":
    run_test()
