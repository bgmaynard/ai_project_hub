"""Pre-market spike scanner - runs until 9:30 AM ET"""

import sys

sys.path.insert(0, ".")
from dotenv import load_dotenv

load_dotenv(".env")
import warnings

warnings.filterwarnings("ignore")

import time
from datetime import datetime

import pytz
import yfinance as yf

# Penny stocks to monitor for spikes
WATCHLIST = [
    "LHAI",
    "YCBD",
    "OCGN",
    "MIRA",
    "WULF",
    "RIOT",
    "AMST",
    "DRUG",
    "GEVO",
    "BITF",
    "MARA",
    "SOUN",
    "KULR",
    "LCID",
    "CLSK",
    "BTDR",
    "ATHA",
    "NVAX",
    "VCNX",
    "SLRX",
    "NISN",
    "CABA",
    "BHAT",
    "CANG",
    "SPCB",
    "COSM",
    "FGEN",
    "XTIA",
    "CNEY",
    "GRPN",
    "WKEY",
]

et = pytz.timezone("US/Eastern")
CUTOFF_HOUR = 9
CUTOFF_MIN = 30

print("=" * 60)
print("SPIKE SCANNER ACTIVE - Monitoring for breakouts")
print("=" * 60)
print(f"Watching {len(WATCHLIST)} penny stocks")
print("Alert criteria: >10% move with volume surge")
print("Cutoff: 9:30 AM ET")
print("=" * 60)
print()

scan_count = 0
while True:
    now = datetime.now(et)

    # Check cutoff
    if now.hour > CUTOFF_HOUR or (now.hour == CUTOFF_HOUR and now.minute >= CUTOFF_MIN):
        print(
            f"\n[{now.strftime('%H:%M:%S')}] 9:30 AM CUTOFF - Scanner stopped. No trade day."
        )
        break

    scan_count += 1
    alerts = []

    for sym in WATCHLIST:
        try:
            ticker = yf.Ticker(sym)
            hist = ticker.history(period="2d")
            if hist.empty or len(hist) < 2:
                continue

            price = hist["Close"].iloc[-1]
            prev = hist["Close"].iloc[-2]
            volume = hist["Volume"].iloc[-1]
            avg_vol = hist["Volume"].mean()

            chg_pct = ((price - prev) / prev) * 100
            rvol = volume / avg_vol if avg_vol > 0 else 0

            # Alert on 10%+ spike with 2x+ volume
            if chg_pct >= 10 and rvol >= 2.0 and price >= 0.50 and price <= 20:
                alerts.append(
                    {"symbol": sym, "price": price, "change": chg_pct, "rvol": rvol}
                )
        except:
            pass

    if alerts:
        print(f"\n{'!'*60}")
        print(f"[{now.strftime('%H:%M:%S')}] SPIKE ALERT!")
        for a in alerts:
            print(
                f"  >> {a['symbol']} ${a['price']:.2f} +{a['change']:.1f}% RVol:{a['rvol']:.1f}x"
            )
        print(f"{'!'*60}\n")
    else:
        if scan_count % 6 == 0:  # Status every ~3 min
            print(
                f"[{now.strftime('%H:%M:%S')}] Scan #{scan_count} - No spikes. Market flat."
            )

    time.sleep(30)  # Scan every 30 seconds
