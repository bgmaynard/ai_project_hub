"""Check both Schwab accounts"""
from schwab_trading import get_schwab_trading

schwab = get_schwab_trading()

# Get all accounts
accounts = schwab.get_accounts()
print('Accounts:', accounts)

# Get data for each account
for acc in accounts:
    acc_num = acc['account_number']
    print(f'\n{"="*50}')
    print(f'Account {acc_num}')
    print("="*50)

    # Select this account
    schwab.select_account(acc_num)

    # Get account info
    info = schwab.get_account_info()
    if info:
        print(f"Type: {info.get('type', 'N/A')}")
        print(f"Cash: ${info.get('cash', 0):,.2f}")
        print(f"Market Value: ${info.get('market_value', 0):,.2f}")
        print(f"Daily P/L: ${info.get('daily_pl', 0):,.2f} ({info.get('daily_pl_pct', 0):.2f}%)")
        print(f"Positions Count: {info.get('positions_count', 0)}")
        print(f"Buying Power: ${info.get('buying_power', 0):,.2f}")

    # Get positions
    positions = schwab.get_positions()
    if positions:
        print(f'\nOpen Positions ({len(positions)}):')
        for p in positions:
            sym = p.get('symbol', 'N/A')
            qty = p.get('quantity', 0)
            avg = p.get('avg_price', 0)
            current = p.get('current_price', 0)
            pnl = p.get('unrealized_pl', 0)
            pnl_pct = p.get('unrealized_pl_pct', 0)
            print(f"  {sym}: {qty} shares @ ${avg:.2f} | Current: ${current:.2f} | P/L: ${pnl:.2f} ({pnl_pct:.2f}%)")
    else:
        print('\nNo open positions')
