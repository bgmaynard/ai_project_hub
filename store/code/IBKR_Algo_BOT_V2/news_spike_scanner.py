"""Combined News + Spike Scanner - Pre-market until 9:30 AM"""
import sys
sys.path.insert(0, '.')
from dotenv import load_dotenv
load_dotenv('.env')
import warnings
warnings.filterwarnings('ignore')

import time
import requests
from datetime import datetime
import pytz

API = "http://localhost:9100"
et = pytz.timezone('US/Eastern')

# Track what we've already alerted
alerted = set()

print("=" * 60)
print("NEWS + SPIKE SCANNER - Pre-Market Monitor")
print("=" * 60)
print("Scanning for:")
print("  - Breaking news on penny stocks ($1-$20)")
print("  - Price spikes >10% with volume")
print("Auto-adds qualifying stocks to watchlist")
print("Cutoff: 9:30 AM ET")
print("=" * 60)

scan_num = 0
while True:
    now = datetime.now(et)
    
    if now.hour > 9 or (now.hour == 9 and now.minute >= 30):
        print(f"\n[{now.strftime('%H:%M:%S')}] 9:30 CUTOFF - Done for today.")
        break
    
    scan_num += 1
    
    # 1. Check news feed
    try:
        resp = requests.get(f"{API}/api/news/fetch?limit=30", timeout=10)
        news = resp.json().get('news', [])
        
        for item in news:
            symbols = item.get('symbols', [])
            headline = item.get('headline', item.get('title', ''))
            
            for sym in symbols:
                if sym in alerted or not sym.isalpha():
                    continue
                    
                # Check if penny stock
                try:
                    price_resp = requests.get(f"{API}/api/price/{sym}", timeout=5)
                    pdata = price_resp.json()
                    price = pdata.get('price', 0)
                    chg = pdata.get('change_percent', 0)
                    
                    # Penny stock with news + movement
                    if 0.50 <= price <= 20 and abs(chg) >= 5:
                        print(f"\n{'!'*60}")
                        print(f"[{now.strftime('%H:%M:%S')}] NEWS ALERT: {sym}")
                        print(f"  Price: ${price:.2f} ({chg:+.1f}%)")
                        print(f"  News: {headline[:60]}")
                        
                        # Add to watchlist
                        requests.post(f"{API}/api/watchlist-ai/add", json={'symbol': sym})
                        print(f"  >> ADDED TO WATCHLIST")
                        print(f"{'!'*60}\n")
                        alerted.add(sym)
                except:
                    pass
    except Exception as e:
        pass
    
    # 2. Quick scan current watchlist for spikes
    try:
        resp = requests.get(f"{API}/api/worklist", timeout=10)
        stocks = resp.json().get('data', [])
        
        for s in stocks:
            sym = s.get('symbol')
            chg = s.get('change_percent', 0)
            price = s.get('price', 0)
            
            if sym not in alerted and chg >= 15:
                print(f"\n{'!'*60}")
                print(f"[{now.strftime('%H:%M:%S')}] SPIKE: {sym} ${price:.2f} +{chg:.1f}%")
                print(f"{'!'*60}\n")
                alerted.add(sym)
    except:
        pass
    
    # Status update every ~2 min
    if scan_num % 4 == 0:
        print(f"[{now.strftime('%H:%M:%S')}] Scan #{scan_num} - Monitoring news + prices...")
    
    time.sleep(30)

print("\nScanner complete. No more pre-market trading today.")
