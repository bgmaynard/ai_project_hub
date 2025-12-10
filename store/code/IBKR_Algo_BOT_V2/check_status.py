"""Quick status check"""
from dotenv import load_dotenv
import os
from alpaca.trading.client import TradingClient

load_dotenv()
client = TradingClient(os.getenv('ALPACA_API_KEY'), os.getenv('ALPACA_SECRET_KEY'), paper=True)

acct = client.get_account()
print(f"Portfolio: ${float(acct.portfolio_value):,.2f}")
print(f"Buying Power: ${float(acct.buying_power):,.2f}")
print(f"Cash: ${float(acct.cash):,.2f}")
print()

positions = client.get_all_positions()
if positions:
    print("POSITIONS:")
    for p in positions:
        pnl = float(p.unrealized_pl)
        pnl_pct = float(p.unrealized_plpc) * 100
        print(f"  {p.symbol}: {p.qty} @ ${float(p.avg_entry_price):.2f} | P/L: ${pnl:.2f} ({pnl_pct:+.1f}%)")
else:
    print("No positions - FLAT")
