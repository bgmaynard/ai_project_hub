"""
Order & Position Reconciliation Tool
====================================

Monitors and reconciles:
- Open orders
- Filled orders
- Current positions
- P&L calculations
- Order history

Usage:
    python reconcile_orders_positions.py
    python reconcile_orders_positions.py --export trades.csv
"""

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
import threading
import time
from datetime import datetime
import pandas as pd
import argparse
from collections import defaultdict


class OrderReconciler(EWrapper, EClient):
    """Tracks and reconciles all orders and positions"""
    
    def __init__(self):
        EClient.__init__(self, self)
        
        self.connected = False
        self.next_order_id = None
        
        # Order tracking
        self.orders = {}  # {order_id: order_details}
        self.executions = []  # List of all executions
        self.open_orders = {}  # Currently open orders
        
        # Position tracking
        self.positions = {}  # {symbol: position_details}
        
        # Account data
        self.account_summary = {}
        
        # Completion flags
        self.orders_complete = False
        self.positions_complete = False
        self.executions_complete = False
    
    def nextValidId(self, orderId: int):
        """Connection established"""
        super().nextValidId(orderId)
        self.next_order_id = orderId
        self.connected = True
        print(f"✓ Connected. Next order ID: {orderId}")
    
    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        """Error handler"""
        if errorCode in [2104, 2106, 2158, 202]:
            pass  # Ignore info messages
        elif errorCode >= 2000:
            print(f"  Info: {errorString}")
        else:
            print(f"  Error {errorCode}: {errorString}")
    
    # ========================================
    # Order Tracking
    # ========================================
    
    def orderStatus(self, orderId, status, filled, remaining, avgFillPrice,
                   permId, parentId, lastFillPrice, clientId, whyHeld, mktCapPrice):
        """Order status updates"""
        
        if orderId not in self.orders:
            self.orders[orderId] = {}
        
        self.orders[orderId].update({
            'order_id': orderId,
            'status': status,
            'filled': filled,
            'remaining': remaining,
            'avg_fill_price': avgFillPrice,
            'last_fill_price': lastFillPrice,
            'perm_id': permId,
            'parent_id': parentId,
            'client_id': clientId,
            'why_held': whyHeld,
            'timestamp': datetime.now()
        })
    
    def openOrder(self, orderId, contract, order, orderState):
        """Open order details"""
        
        if orderId not in self.orders:
            self.orders[orderId] = {}
        
        self.orders[orderId].update({
            'symbol': contract.symbol,
            'sec_type': contract.secType,
            'exchange': contract.exchange,
            'action': order.action,
            'order_type': order.orderType,
            'total_quantity': order.totalQuantity,
            'limit_price': order.lmtPrice,
            'aux_price': order.auxPrice,
            'tif': order.tif,
            'order_state': orderState.status,
            'commission': orderState.commission,
            'min_commission': orderState.minCommission,
            'max_commission': orderState.maxCommission,
        })
        
        # Track open orders separately
        if orderState.status in ['PreSubmitted', 'Submitted', 'PendingSubmit']:
            self.open_orders[orderId] = self.orders[orderId]
        elif orderId in self.open_orders:
            del self.open_orders[orderId]
    
    def openOrderEnd(self):
        """All open orders received"""
        self.orders_complete = True
        print(f"  ✓ Orders received: {len(self.orders)} total, {len(self.open_orders)} open")
    
    def execDetails(self, reqId, contract, execution):
        """Execution details"""
        exec_detail = {
            'exec_id': execution.execId,
            'order_id': execution.orderId,
            'symbol': contract.symbol,
            'sec_type': contract.secType,
            'side': execution.side,
            'shares': execution.shares,
            'price': execution.price,
            'perm_id': execution.permId,
            'client_id': execution.clientId,
            'order_ref': execution.orderRef,
            'time': execution.time,
            'account': execution.acctNumber,
            'exchange': execution.exchange,
            'cum_qty': execution.cumQty,
            'avg_price': execution.avgPrice
        }
        
        self.executions.append(exec_detail)
    
    def execDetailsEnd(self, reqId):
        """All executions received"""
        self.executions_complete = True
        print(f"  ✓ Executions received: {len(self.executions)}")
    
    # ========================================
    # Position Tracking
    # ========================================
    
    def position(self, account, contract, position, avgCost):
        """Position updates"""
        symbol = contract.symbol
        
        self.positions[symbol] = {
            'symbol': symbol,
            'sec_type': contract.secType,
            'exchange': contract.exchange,
            'position': position,
            'avg_cost': avgCost,
            'account': account,
            'market_value': 0,  # Will be updated with market data
            'unrealized_pnl': 0
        }
    
    def positionEnd(self):
        """All positions received"""
        self.positions_complete = True
        print(f"  ✓ Positions received: {len(self.positions)}")
    
    # ========================================
    # Account Data
    # ========================================
    
    def updateAccountValue(self, key, val, currency, accountName):
        """Account value updates"""
        try:
            self.account_summary[key] = float(val)
        except:
            self.account_summary[key] = val
    
    def accountDownloadEnd(self, accountName):
        """Account data complete"""
        print(f"  ✓ Account data received for {accountName}")


def display_summary(app):
    """Display comprehensive trading summary"""
    
    print("\n" + "="*80)
    print("TRADING SUMMARY")
    print("="*80)
    
    # Account Overview
    print("\n📊 ACCOUNT OVERVIEW")
    print("-" * 80)
    
    important_values = [
        ('NetLiquidation', 'Net Liquidation'),
        ('TotalCashValue', 'Total Cash'),
        ('StockMarketValue', 'Stock Value'),
        ('UnrealizedPnL', 'Unrealized P&L'),
        ('RealizedPnL', 'Realized P&L'),
        ('AvailableFunds', 'Available Funds'),
        ('BuyingPower', 'Buying Power')
    ]
    
    for key, label in important_values:
        if key in app.account_summary:
            value = app.account_summary[key]
            if isinstance(value, (int, float)):
                print(f"  {label:.<30} ${value:>15,.2f}")
            else:
                print(f"  {label:.<30} {value:>15}")
    
    # Current Positions
    print("\n📦 CURRENT POSITIONS")
    print("-" * 80)
    
    if not app.positions:
        print("  No open positions")
    else:
        print(f"  {'Symbol':<10} {'Quantity':>10} {'Avg Cost':>12} {'Market Value':>15} {'Unrealized P&L':>15}")
        print("  " + "-"*76)
        
        total_value = 0
        total_pnl = 0
        
        for symbol, pos in app.positions.items():
            qty = pos['position']
            avg_cost = pos['avg_cost']
            
            # Get current market price if available
            # For now, use avg_cost (would need real-time data)
            current_price = avg_cost
            market_value = qty * current_price
            unrealized_pnl = (current_price - avg_cost) * qty
            
            total_value += market_value
            total_pnl += unrealized_pnl
            
            print(f"  {symbol:<10} {qty:>10} ${avg_cost:>11.2f} ${market_value:>14.2f} ${unrealized_pnl:>14.2f}")
        
        print("  " + "-"*76)
        print(f"  {'TOTAL':<10} {'':<10} {'':<12} ${total_value:>14.2f} ${total_pnl:>14.2f}")
    
    # Open Orders
    print("\n📝 OPEN ORDERS")
    print("-" * 80)
    
    if not app.open_orders:
        print("  No open orders")
    else:
        print(f"  {'ID':<8} {'Symbol':<10} {'Action':<8} {'Type':<10} {'Qty':>8} {'Price':>10} {'Status':<15}")
        print("  " + "-"*76)
        
        for order_id, order in app.open_orders.items():
            symbol = order.get('symbol', 'N/A')
            action = order.get('action', 'N/A')
            order_type = order.get('order_type', 'N/A')
            qty = order.get('total_quantity', 0)
            price = order.get('limit_price', 0)
            status = order.get('order_state', 'N/A')
            
            price_str = f"${price:.2f}" if price > 0 else "MKT"
            print(f"  {order_id:<8} {symbol:<10} {action:<8} {order_type:<10} {qty:>8} {price_str:>10} {status:<15}")
    
    # Recent Executions
    print("\n✅ RECENT EXECUTIONS (Last 10)")
    print("-" * 80)
    
    if not app.executions:
        print("  No executions found")
    else:
        print(f"  {'Time':<20} {'Symbol':<10} {'Side':<8} {'Shares':>8} {'Price':>10} {'Order ID':<10}")
        print("  " + "-"*76)
        
        # Show last 10 executions
        recent_execs = sorted(app.executions, key=lambda x: x['time'], reverse=True)[:10]
        
        for exec in recent_execs:
            time_str = exec['time']
            symbol = exec['symbol']
            side = exec['side']
            shares = exec['shares']
            price = exec['price']
            order_id = exec['order_id']
            
            print(f"  {time_str:<20} {symbol:<10} {side:<8} {shares:>8} ${price:>9.2f} {order_id:<10}")
    
    # Order Summary Statistics
    print("\n📈 ORDER STATISTICS")
    print("-" * 80)
    
    if app.orders:
        total_orders = len(app.orders)
        
        # Count by status
        status_counts = defaultdict(int)
        for order in app.orders.values():
            status = order.get('status', 'Unknown')
            status_counts[status] += 1
        
        print(f"  Total Orders: {total_orders}")
        print("\n  By Status:")
        for status, count in sorted(status_counts.items()):
            print(f"    {status:.<20} {count:>5}")
    
    # Execution Summary
    if app.executions:
        print(f"\n  Total Executions: {len(app.executions)}")
        
        # Calculate totals by side
        buy_shares = sum(e['shares'] for e in app.executions if e['side'] == 'BOT')
        sell_shares = sum(e['shares'] for e in app.executions if e['side'] == 'SLD')
        
        buy_value = sum(e['shares'] * e['price'] for e in app.executions if e['side'] == 'BOT')
        sell_value = sum(e['shares'] * e['price'] for e in app.executions if e['side'] == 'SLD')
        
        print("\n  Buy Side:")
        print(f"    Shares: {buy_shares:,}")
        print(f"    Value:  ${buy_value:,.2f}")
        
        print("\n  Sell Side:")
        print(f"    Shares: {sell_shares:,}")
        print(f"    Value:  ${sell_value:,.2f}")
        
        if sell_value > 0 and buy_value > 0:
            gross_pnl = sell_value - buy_value
            print(f"\n  Gross P&L: ${gross_pnl:+,.2f}")
    
    print("\n" + "="*80 + "\n")


def export_to_csv(app, filename):
    """Export trading data to CSV files"""
    
    print("\n📁 Exporting data...")
    
    # Export positions
    if app.positions:
        pos_df = pd.DataFrame([
            {
                'symbol': pos['symbol'],
                'position': pos['position'],
                'avg_cost': pos['avg_cost'],
                'account': pos['account']
            }
            for pos in app.positions.values()
        ])
        
        pos_file = filename.replace('.csv', '_positions.csv')
        pos_df.to_csv(pos_file, index=False)
        print(f"  ✓ Positions exported to: {pos_file}")
    
    # Export orders
    if app.orders:
        orders_df = pd.DataFrame([
            {
                'order_id': order.get('order_id'),
                'symbol': order.get('symbol'),
                'action': order.get('action'),
                'quantity': order.get('total_quantity'),
                'order_type': order.get('order_type'),
                'status': order.get('status'),
                'filled': order.get('filled', 0),
                'avg_fill_price': order.get('avg_fill_price', 0),
                'timestamp': order.get('timestamp')
            }
            for order in app.orders.values()
        ])
        
        orders_file = filename.replace('.csv', '_orders.csv')
        orders_df.to_csv(orders_file, index=False)
        print(f"  ✓ Orders exported to: {orders_file}")
    
    # Export executions
    if app.executions:
        exec_df = pd.DataFrame(app.executions)
        
        exec_file = filename.replace('.csv', '_executions.csv')
        exec_df.to_csv(exec_file, index=False)
        print(f"  ✓ Executions exported to: {exec_file}")
    
    print()


def run_reconciliation(export_file=None):
    """Run complete order and position reconciliation"""
    
    print("\n" + "="*80)
    print("ORDER & POSITION RECONCILIATION")
    print("="*80 + "\n")
    
    app = OrderReconciler()
    
    # Connect to TWS
    print("Connecting to IBKR...")
    
    ports = [(7497, "Paper"), (7496, "Live"), (4002, "Gateway Paper"), (4001, "Gateway Live")]
    
    connected = False
    for port, name in ports:
        try:
            print(f"  Trying {name} (port {port})...", end=" ")
            app.connect("127.0.0.1", port, clientId=2)
            
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
        print("\n❌ Could not connect to IBKR")
        return None
    
    print("\nRequesting data...")
    
    # Request all data
    app.reqOpenOrders()
    app.reqAllOpenOrders()
    app.reqAutoOpenOrders(True)
    
    time.sleep(1)
    
    app.reqPositions()
    
    time.sleep(1)
    
    app.reqAccountUpdates(True, "")
    
    time.sleep(1)
    
    # Request executions for today
    from ibapi.execution import ExecutionFilter
    exec_filter = ExecutionFilter()
    exec_filter.clientId = 0  # All clients
    app.reqExecutions(1, exec_filter)
    
    # Wait for data
    print("\nWaiting for data...")
    timeout = 10
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        if (app.orders_complete and app.positions_complete and app.executions_complete):
            break
        time.sleep(0.5)
    
    # Display summary
    display_summary(app)
    
    # Export if requested
    if export_file:
        export_to_csv(app, export_file)
    
    # Disconnect
    app.disconnect()
    
    return app


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Reconcile Orders and Positions')
    parser.add_argument('--export', type=str, help='Export data to CSV file')
    parser.add_argument('--watch', action='store_true', help='Continuous monitoring mode')
    parser.add_argument('--interval', type=int, default=30, help='Update interval in seconds (for watch mode)')
    
    args = parser.parse_args()
    
    if args.watch:
        print("\n🔄 Continuous monitoring mode (Ctrl+C to stop)")
        print(f"   Update interval: {args.interval} seconds\n")
        
        try:
            iteration = 0
            while True:
                iteration += 1
                print(f"\n{'='*80}")
                print(f"UPDATE #{iteration} @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"{'='*80}")
                
                run_reconciliation(args.export)
                
                print(f"\nNext update in {args.interval} seconds...")
                time.sleep(args.interval)
                
        except KeyboardInterrupt:
            print("\n\nStopped by user")
    
    else:
        # Single run
        run_reconciliation(args.export)


if __name__ == "__main__":
    main()
