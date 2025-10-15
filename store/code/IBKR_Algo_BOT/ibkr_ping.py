# C:\ai_project_hub\store\code\IBKR_Algo_BOT\ibkr_ping.py
import time, threading, argparse
from ibapi.client import EClient
from ibapi.wrapper import EWrapper

class PingWrap(EWrapper):
    def __init__(self):
        super().__init__()
        self.got_id = threading.Event()
        self.got_time = threading.Event()
    def nextValidId(self, orderId: int):
        print("nextValidId:", orderId)
        self.got_id.set()
    def currentTime(self, time_):
        print("currentTime:", time_)
        self.got_time.set()
    def error(self, reqId, code, msg, advancedOrderRejectJson=""):
        print(f"[ERROR] code={code} msg={msg}")

class PingClient(EClient, PingWrap):
    def __init__(self):
        PingWrap.__init__(self)
        EClient.__init__(self, self)

def main(host="127.0.0.1", port=7497, client_id=123):
    c = PingClient()
    c.connect(host, port, client_id)
    t = threading.Thread(target=c.run, daemon=True)
    t.start()

    if not c.got_id.wait(5):
        print("❌ no nextValidId (not connected)")
        return 3

    c.reqCurrentTime()
    c.got_time.wait(5)
    c.disconnect()
    print("✅ ping ok")
    return 0

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7497)  # 7497=paper, 7496=live
    ap.add_argument("--client-id", type=int, default=123)
    a = ap.parse_args()
    raise SystemExit(main(a.host, a.port, a.client_id))
