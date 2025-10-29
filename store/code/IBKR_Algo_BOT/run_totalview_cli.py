import argparse
import time
import sys
from IBKR_Algo_BOT.ibkr_totalview_integration import TotalViewDataHandler

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="AAPL,MSFT")
    ap.add_argument("--rows", type=int, default=10)
    ap.add_argument("--seconds", type=int, default=60)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7497)  # 7497=paper, 7496=live
    ap.add_argument("--client-id", type=int, default=42)
    ap.add_argument("--smart-depth", action="store_true")
    ap.add_argument("--exchange", default="")          # NYSE / ARCA / ISLAND / BATS / IEX
    ap.add_argument("--delayed", action="store_true")  # use delayed market data
    args = ap.parse_args()

    syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    mdt = 3 if args.delayed else 1

    tv = TotalViewDataHandler(host=args.host, port=args.port, clientId=args.client_id,
                              rows=args.rows, smart_depth=args.smart_depth,
                              market_data_type=mdt, exchange=args.exchange)
    tv.connect()
    if not tv.wait_connected(timeout=7.0):
        print("❌ Could not establish API connection (no nextValidId). Check TWS/IBG is running, API enabled, and port.", file=sys.stderr)
        sys.exit(3)

    tv.after_connected_setup()
    time.sleep(0.5)  # tiny cushion

    for s in syms:
        tv.subscribe_depth(s)

    t0 = time.time()
    while time.time() - t0 < args.seconds:
        _ = tv.process_messages(timeout=0.3)

    for s in syms:
        book = tv.get_book(s)
        bids, asks = book.get("bids", []), book.get("asks", [])
        print(f"\n[{s}] bids:{len(bids)} asks:{len(asks)}")
        if bids: print("  best bid:", bids[0])
        if asks: print("  best ask:", asks[0])

    tv.close()

if __name__ == "__main__":
    main()
