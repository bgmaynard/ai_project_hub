"""Connection Monitor - Detects when Polygon stream fails"""
import time
import requests
from datetime import datetime

def check_connection():
    try:
        r = requests.get("http://localhost:9100/api/polygon/stream/status", timeout=5)
        data = r.json()
        return {
            "connected": data.get("connected", False),
            "healthy": data.get("healthy", False),
            "trades": data.get("trades_received", 0),
            "quotes": data.get("quotes_received", 0),
            "last_message": data.get("last_message", ""),
            "reconnects": data.get("reconnects", 0)
        }
    except Exception as e:
        return {"error": str(e)}

def main():
    print(f"=== Connection Monitor Started at {datetime.now().strftime('%H:%M:%S')} ===")
    print("Checking every 10 seconds for 10 minutes...")
    print("")

    last_trades = 0
    last_status = None
    stale_count = 0

    for i in range(60):  # 10 minutes
        status = check_connection()
        now = datetime.now().strftime("%H:%M:%S")

        if "error" in status:
            print(f"[{now}] [ERROR] API ERROR: {status['error']}")
            last_status = "error"
            stale_count = 0
        elif not status["connected"] or not status["healthy"]:
            print(f"[{now}] [WARN] DISCONNECTED! connected={status['connected']} healthy={status['healthy']} reconnects={status['reconnects']}")
            last_status = "disconnected"
            stale_count = 0
        else:
            trades = status["trades"]
            if trades == last_trades and last_trades > 0:
                stale_count += 1
                if stale_count >= 3:  # 30 seconds of no new trades
                    print(f"[{now}] [WARN] STALE DATA - No new trades for {stale_count * 10}s (trades={trades})")
            else:
                if last_status != "ok":
                    print(f"[{now}] [OK] CONNECTED - trades={trades} quotes={status['quotes']} reconnects={status['reconnects']}")
                stale_count = 0
                last_status = "ok"
            last_trades = trades

        time.sleep(10)

    print(f"\n=== Monitor finished at {datetime.now().strftime('%H:%M:%S')} ===")

if __name__ == "__main__":
    main()
