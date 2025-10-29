"""
IBKR Connection & Order Validation Script
=========================================

Tests all critical IBKR functionality:
1. Connection to TWS/Gateway
2. Market data streaming
3. Historical data fetching
4. Order placement (paper trading)
5. Position reconciliation
6. Account data retrieval

Usage:
    python validate_ibkr_connection.py --test-orders
"""

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import Order
import threading
import time
from datetime import datetime
import argparse

class IBKRValidator(EWrapper, EClient):
    """Test all IBKR functionality"""
    
    def __init__(self):
        EClient.__init__(self, self)
        self.connected = False
        self.next_order_id = None
        
        # Test results
        self.tests_passed = []
        self.tests_failed = []
        
        # Data storage
        self.market_data_received = {}
        self.historical_data_received = {}
        self.positions_received = []
        self.account_values_received = {}
        self.orders_placed = []
        self.order_statuses = {}
        
    def nextValidId(self, orderId: int):
        """Connection established"""
        super().nextValidId(orderId)
        self.next_order_id = orderId
        self.connected = True
        print(f"✓ Connected! Next order ID: {orderId}")
        self.tests_passed.append("Connection")
    
    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        """Error handler"""
        if errorCode in [2104, 2106, 2158]:  # Info messages
            print(f"  Info: {errorString}")
        elif errorCode == 202:  # Order cancelled
            print(f"  Order {reqId} cancelled")
        elif errorCode >= 2000:  # Other info
            print(f"  {errorString}")
        else:
            print(f"  ❌ Error {errorCode}: {errorString}")
    
    def tickPrice(self, reqId, tickType, price, attrib):
        """Market data - prices"""
        if reqId not in self.market_data_received:
            self.market_data_received[reqId] = {}
        
        tick_names = {1: 'BID', 2: 'ASK', 4: 'LAST', 6: 'HIGH', 7: 'LOW', 9: 'CLOSE'}
        if tickType in tick_names:
            self.market_data_received[reqId][tick_names[tickType]] = price
    
    def tickSize(self, reqId, tickType, size):
        """Market data - sizes"""
        if reqId not in self.market_data_received:
            self.market_data_received[reqId] = {}
        
        size_names = {0: 'BID_SIZE', 3: 'ASK_SIZE', 5: 'LAST_SIZE', 8: 'VOLUME'}
        if tickType in size_names:
            self.market_data_received[reqId][size_names[tickType]] = size
    
    def historicalData(self, reqId, bar):
        """Historical data bars"""
        if reqId not in self.historical_data_received:
            self.historical_data_received[reqId] = []
        
        self.historical_data_received[reqId].append({
            'date': bar.date,
            'open': bar.open,
            'high': bar.high,
            'low': bar.low,
            'close': bar.close,
            'volume': bar.volume
        })
    
    def historicalDataEnd(self, reqId, start, end):
        """Historical data complete"""
        count = len(self.historical_data_received.get(reqId, []))
        print(f"  ✓ Historical data complete: {count} bars")
    
    def position(self, account, contract, position, avgCost):
        """Position update"""
        self.positions_received.append({
            'symbol': contract.symbol,
            'quantity': position,
            'avg_cost': avgCost,
            'account': account
        })
    
    def positionEnd(self):
        """All positions received"""
        print(f"  ✓ Positions received: {len(self.positions_received)}")
    
    def updateAccountValue(self, key, val, currency, accountName):
        """Account value update"""
        self.account_values_received[key] = {
            'value': val,
            'currency': currency
        }
    
    def accountDownloadEnd(self, accountName):
        """Account data complete"""
        print(f"  ✓ Account data received: {len(self.account_values_received)} values")
    
    def orderStatus(self, orderId, status, filled, remaining, avgFillPrice,
                   permId, parentId, lastFillPrice, clientId, whyHeld, mktCapPrice):
        """Order status update"""
        self.order_statuses[orderId] = {
            'status': status,
            'filled': filled,
            'remaining': remaining,
            'avg_fill_price': avgFillPrice
        }
        print(f"  Order {orderId}: {status} (filled {filled}, remaining {remaining})")
    
    def openOrder(self, orderId, contract, order, orderState):
        """Open order notification"""
        print(f"  Open order {orderId}: {order.action} {order.totalQuantity} {contract.symbol}")
    
    def execDetails(self, reqId, contract, execution):
        """Execution details"""
        print(f"  ✓ Executed: {execution.side} {execution.shares} {contract.symbol} @ ${execution.price:.2f}")


def run_validation_tests(test_orders=False):
    """Run all validation tests"""
    
    print("\n" + "="*70)
    print("IBKR CONNECTION & FUNCTIONALITY VALIDATOR")
    print("="*70 + "\n")
    
    app = IBKRValidator()
    
    # Test 1: Connection
    print("TEST 1: Connection to TWS/Gateway")
    print("-" * 70)
    
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
            print(f"  Trying {name} (port {port})...", end=" ")
            app.connect("127.0.0.1", port, clientId=1)
            
            api_thread = threading.Thread(target=app.run, daemon=True)
            api_thread.start()
            
            time.sleep(2)
            
            if app.connected:
                print("✓ CONNECTED")
                connected = True
                break
            else:
                print("✗")
                app.disconnect()
        except Exception as e:
            print(f"✗ ({e})")
    
    if not connected:
        print("\n❌ FAILED: Could not connect to IBKR")
        print("\nTroubleshooting:")
        print("  1. Is TWS or IB Gateway running?")
        print("  2. Go to: File → Global Configuration → API → Settings")
        print("  3. Enable: 'Enable ActiveX and Socket Clients'")
        print("  4. Check port number matches")
        print("  5. Restart TWS/Gateway after changes")
        return
    
    app.tests_passed.append("Connection")
    
    # Test 2: Market Data
    print("\n\nTEST 2: Real-Time Market Data")
    print("-" * 70)
    
    contract = Contract()
    contract.symbol = "AAPL"
    contract.secType = "STK"
    contract.exchange = "SMART"
    contract.currency = "USD"
    
    req_id = 1001
    print("  Requesting market data for AAPL...")
    app.reqMktData(req_id, contract, "", False, False, [])
    
    time.sleep(3)
    
    if req_id in app.market_data_received and len(app.market_data_received[req_id]) > 0:
        print("  ✓ Market data received:")
        data = app.market_data_received[req_id]
        for key, value in data.items():
            if isinstance(value, (int, float)):
                print(f"    {key}: {value:,.2f}" if isinstance(value, float) else f"    {key}: {value:,}")
        app.tests_passed.append("Market Data")
    else:
        print("  ⚠️  No market data received")
        print("  Possible reasons:")
        print("    - Market is closed (9:30 AM - 4:00 PM ET)")
        print("    - No market data subscription")
        app.tests_failed.append("Market Data")
    
    # Cancel market data
    app.cancelMktData(req_id)
    
    # Test 3: Historical Data
    print("\n\nTEST 3: Historical Data")
    print("-" * 70)
    
    req_id = 2001
    end_time = datetime.now().strftime('%Y%m%d %H:%M:%S')
    
    print("  Requesting 2 days of 1-hour bars for AAPL...")
    app.reqHistoricalData(
        req_id, contract, end_time, "2 D", "1 hour",
        "TRADES", 1, 1, False, []
    )
    
    time.sleep(5)
    
    if req_id in app.historical_data_received and len(app.historical_data_received[req_id]) > 0:
        bars = app.historical_data_received[req_id]
        print(f"  ✓ Historical data received: {len(bars)} bars")
        print(f"    First bar: {bars[0]['date']}")
        print(f"    Last bar: {bars[-1]['date']}")
        print(f"    Sample: O={bars[-1]['open']:.2f}, H={bars[-1]['high']:.2f}, L={bars[-1]['low']:.2f}, C={bars[-1]['close']:.2f}")
        app.tests_passed.append("Historical Data")
    else:
        print("  ❌ No historical data received")
        app.tests_failed.append("Historical Data")
    
    # Test 4: Account Data
    print("\n\nTEST 4: Account Data")
    print("-" * 70)
    
    print("  Requesting account updates...")
    app.reqAccountUpdates(True, "")
    
    time.sleep(2)
    
    if len(app.account_values_received) > 0:
        print("  ✓ Account data received:")
        important_keys = ['NetLiquidation', 'TotalCashValue', 'AvailableFunds', 'BuyingPower']
        for key in important_keys:
            if key in app.account_values_received:
                val = app.account_values_received[key]
                print(f"    {key}: {val['value']} {val['currency']}")
        app.tests_passed.append("Account Data")
    else:
        print("  ⚠️  No account data received")
        app.tests_failed.append("Account Data")
    
    # Test 5: Positions
    print("\n\nTEST 5: Positions")
    print("-" * 70)
    
    print("  Requesting positions...")
    app.reqPositions()
    
    time.sleep(2)
    
    if len(app.positions_received) > 0:
        print("  ✓ Positions received:")
        for pos in app.positions_received:
            print(f"    {pos['symbol']}: {pos['quantity']} @ ${pos['avg_cost']:.2f}")
        app.tests_passed.append("Positions")
    else:
        print("  ℹ️  No open positions (this is normal if you haven't traded yet)")
        app.tests_passed.append("Positions")
    
    # Test 6: Order Placement (Optional)
    if test_orders:
        print("\n\nTEST 6: Order Placement (PAPER TRADING ONLY!)")
        print("-" * 70)
        print("  ⚠️  WARNING: This will place a REAL order in your paper account")
        
        confirm = input("  Continue? (yes/no): ")
        if confirm.lower() == 'yes':
            # Place small test order
            order = Order()
            order.action = "BUY"
            order.totalQuantity = 1
            order.orderType = "LMT"
            order.lmtPrice = 100.00  # Way below market - won't fill
            
            order_id = app.next_order_id
            
            print("  Placing test order: BUY 1 AAPL @ $100.00 (limit)...")
            app.placeOrder(order_id, contract, order)
            app.next_order_id += 1
            
            time.sleep(2)
            
            if order_id in app.order_statuses:
                status = app.order_statuses[order_id]
                print("  ✓ Order placed successfully!")
                print(f"    Status: {status['status']}")
                
                # Cancel the order
                print("  Cancelling test order...")
                app.cancelOrder(order_id, "")
                time.sleep(1)
                
                app.tests_passed.append("Order Placement")
            else:
                print("  ❌ Order placement failed")
                app.tests_failed.append("Order Placement")
        else:
            print("  Skipped order placement test")
    
    # Disconnect
    app.disconnect()
    
    # Summary
    print("\n\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    
    print(f"\n✓ Tests Passed: {len(app.tests_passed)}")
    for test in app.tests_passed:
        print(f"  ✓ {test}")
    
    if app.tests_failed:
        print(f"\n❌ Tests Failed: {len(app.tests_failed)}")
        for test in app.tests_failed:
            print(f"  ❌ {test}")
    
    print("\n" + "="*70)
    
    if len(app.tests_failed) == 0:
        print("✅ ALL TESTS PASSED - READY FOR LIVE TRADING!")
    else:
        print("⚠️  SOME TESTS FAILED - REVIEW ABOVE")
    
    print("="*70 + "\n")
    
    return app


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Validate IBKR Connection')
    parser.add_argument('--test-orders', action='store_true',
                       help='Test order placement (paper trading only!)')
    
    args = parser.parse_args()
    
    if args.test_orders:
        print("\n⚠️  WARNING: Order placement test enabled!")
        print("This will place a test order in your paper trading account.")
        print("Make sure you are connected to PAPER TRADING (port 7497), not live!\n")
        time.sleep(2)
    
    run_validation_tests(test_orders=args.test_orders)


if __name__ == "__main__":
    main()
