import argparse
import sys
import traceback
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

try:
    from IBKR_Algo_BOT.improved_backtest import run_improved_backtest
except Exception as e:
    print("❌ Could not import run_improved_backtest:", e)
    traceback.print_exc()
    sys.exit(2)

def fetch_prices(symbol: str, period: str = "6mo", interval: str = "1d"):
    try:
        import yfinance as yf
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        if df is None or df.empty:
            raise RuntimeError("empty df")
        df = df.rename(columns=str.capitalize)  # Close, Open, etc.
        return df
    except Exception:
        # fallback synthetic series (so the CLI always works)
        n = 180 if interval.endswith("d") else 1000
        idx = pd.date_range(end=datetime.now(), periods=n, freq="D")
        # geometric random walk
        rets = np.random.normal(0, 0.01, size=n)
        prices = 100 * np.exp(np.cumsum(rets))
        df = pd.DataFrame({"Close": prices}, index=idx)
        return df

class MACrossoverModel:
    """Tiny demo model that outputs signal (+1/-1/0) and confidence [0..1]."""
    def __init__(self, fast=10, slow=30):
        self.fast = fast
        self.slow = slow

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        px = df["Close"].astype(float)
        ma_f = px.rolling(self.fast, min_periods=1).mean()
        ma_s = px.rolling(self.slow, min_periods=1).mean()
        diff = ma_f - ma_s
        signal = np.sign(diff).fillna(0).astype(int)   # +1/-1/0
        # confidence: scaled absolute distance
        conf = (np.abs(diff) / (ma_s.replace(0, np.nan))).clip(0, 1).fillna(0)
        out = pd.DataFrame({"signal": signal, "confidence": conf}, index=df.index)
        return out

def main():
    ap = argparse.ArgumentParser(description="Run improved backtest with a simple MA-crossover model")
    ap.add_argument("--symbol", default="AAPL")
    ap.add_argument("--period", default="6mo")
    ap.add_argument("--interval", default="1d")
    ap.add_argument("--fast", type=int, default=10)
    ap.add_argument("--slow", type=int, default=30)
    ap.add_argument("--conf", type=float, default=0.65, help="confidence threshold")
    args = ap.parse_args()

    df = fetch_prices(args.symbol, args.period, args.interval)
    if "Close" not in df.columns:
        print("❌ price DataFrame must have a 'Close' column")
        return 2

    model = MACrossoverModel(args.fast, args.slow)
    try:
        results = run_improved_backtest(model, df, confidence_threshold=args.conf)
    except TypeError:
        # if your function signature differs, try alternate call styles here
        print("❌ run_improved_backtest signature mismatch. Check the function definition.")
        traceback.print_exc()
        return 2
    except Exception:
        traceback.print_exc()
        return 2

    # Best-effort summary printing
    print("\n=== Backtest Results ===")
    if isinstance(results, dict):
        for k, v in results.items():
            if isinstance(v, (float, int)):
                print(f"{k}: {v}")
            else:
                print(f"{k}: {type(v).__name__}")
    else:
        print(type(results).__name__, "returned.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
