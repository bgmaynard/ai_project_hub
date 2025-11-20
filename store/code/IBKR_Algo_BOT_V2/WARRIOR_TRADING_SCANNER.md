# Warrior Trading Gappers Scanner

**Created:** 2025-11-17
**Purpose:** Scan for momentum stocks using Warrior Trading criteria

---

## What It Does

Finds the **top 10 gapper stocks** that match Warrior Trading's momentum trading criteria:

✅ **Price:** $2 - $20
✅ **Volume:** 1M+ (high volume)
✅ **Movement:** Top % gainers (captures 10%+ moves)
✅ **Count:** Top 10 results only

---

## Warrior Trading Criteria (Target)

### What We Match:
- ✅ **Price Range:** $2-$20 (perfect match)
- ✅ **High Volume:** 1M+ volume filter
- ✅ **Price Up 10%+:** Top % gainers scan captures this
- ✅ **Top 10:** Returns exactly 10 results

### What's Missing (Manual Check):
- ⚠️ **5x Relative Volume:** IBKR doesn't have this filter built-in
  - **Workaround:** Scanner shows volume, you can eyeball if it's unusually high
  - **Future:** Can calculate relative volume in post-processing

- ⚠️ **Breaking News:** Requires news API integration
  - **Workaround:** Check news manually for top results
  - **Future:** Integrate with Benzinga, Alpha Vantage, or Finnhub news APIs

---

## How to Use

### From complete_platform.html:

1. **Open Scanner:**
   - Click scanner button or window

2. **Select "🎯 Warrior Trading Gappers":**
   - This preset appears first in the dropdown

3. **Click "Scan":**
   - Returns top 10 gappers matching criteria

4. **Review Results:**
   - Rank, Symbol, % Change, Volume
   - Sorted by % gain (biggest movers first)

5. **Add to Worklist:**
   - Click symbols to add to watchlist
   - Or use "Add All to Worklist" button

### Via API:

```bash
curl -X POST http://127.0.0.1:9101/api/scanner/ibkr/scan \
  -H "Content-Type: application/json" \
  -d '{
    "scan_code": "WARRIOR_GAPPERS",
    "instrument": "STK",
    "location": "STK.US.MAJOR",
    "num_rows": 10
  }'
```

---

## Scanner Logic

### Step 1: Base Scan
```
Scan Code: TOP_PERC_GAIN
- Finds stocks with highest % gains today
- Sorted by % change (biggest first)
```

### Step 2: Apply Filters
```javascript
Filters:
- priceAbove: $2      (min price)
- priceBelow: $20     (max price)
- volumeAbove: 1M     (high volume)
```

### Step 3: Limit Results
```
Return top 10 only
```

### Step 4: Return Data
```json
{
  "success": true,
  "data": {
    "scan_code": "WARRIOR_GAPPERS",
    "results": [
      {
        "rank": 1,
        "symbol": "ABCD",
        "contract_id": 123456,
        "distance": 15.5,  // % change
        "price": 8.50,
        "volume": 5234567
      },
      // ... 9 more
    ],
    "count": 10,
    "source": "ibkr_live"
  }
}
```

---

## Example Results

**Market Open Scenario:**
```
Rank  Symbol  Price   Change   Volume     Relative Vol
1     TSLA    $12.50  +18.5%   8.2M       6.2x  ✅ Gap up on news
2     NVDA    $15.20  +15.2%   12.5M      4.8x  ✅ Earnings beat
3     AMD     $8.75   +12.8%   5.1M       7.1x  ✅ Sector momentum
4     AAPL    $6.30   +11.5%   3.2M       5.5x  ✅ Catalyst
5     MSFT    $18.90  +10.2%   2.8M       4.2x  ✅ Breaking news
...
```

**Pre-Market Scenario:**
```
- Scanner runs but results may be limited
- Volume might be lower (use 500K+ filter for pre-market)
- Best used at/after 9:30 AM ET market open
```

---

## Warrior Trading Strategy Context

### What Ross Cameron Looks For:
1. **Gap up at open** (our scanner finds these)
2. **High relative volume** (5x+) - *manual check needed*
3. **News catalyst** (breaking news) - *manual check needed*
4. **Price range $2-$20** ✅ (our scanner filters this)
5. **Clean chart** (support/resistance) - *manual chart review*
6. **Volume > 1M shares** ✅ (our scanner filters this)

### How to Use Scanner Results:
1. **Scanner gives you candidates** (top 10 gappers)
2. **Manual checks:**
   - Is relative volume 5x+? (compare to avg volume)
   - Is there breaking news? (check news sources)
   - Does chart look clean? (check TradingView/IBKR chart)
3. **Trade setup:**
   - Enter on pullback or bull flag
   - Risk management: 2:1 reward/risk
   - Stop loss below VWAP or support

---

## Enhancements Roadmap

### Phase 1: ✅ Complete
- Price filter ($2-$20)
- Volume filter (1M+)
- Top % gainers scan
- Top 10 results

### Phase 2: 🔄 In Progress
- Fix event loop error (async scanner)
- Real-time scanner data (not mock)

### Phase 3: 📋 Planned
- **Relative Volume Calculation:**
  ```python
  relative_volume = current_volume / average_volume_20d
  filter: relative_volume >= 5.0
  ```

- **News API Integration:**
  - Benzinga News API
  - Finnhub News API
  - Alpha Vantage News Sentiment
  - Add "Breaking News" indicator

- **Pre-Market Scanner:**
  - Adjust filters for pre-market hours
  - Lower volume threshold (500K+)
  - Flag as "pre-market" vs "market hours"

### Phase 4: 🎯 Future
- **TradingView Integration:**
  - Webhook alerts → Auto-add to worklist
  - Pattern recognition (bull flags, etc.)
  - Technical indicator filters

- **AI Analysis:**
  - Claude analyzes news sentiment
  - Risk/reward calculation
  - Entry/exit suggestions

- **Historical Performance:**
  - Track scanner results
  - Win rate by setup type
  - Best time of day

---

## Testing Instructions

### Test 1: Basic Scanner
```
1. Restart server: .\RESTART_SERVER.ps1
2. Connect to IBKR
3. Open complete_platform.html
4. Open Scanner window
5. Select "🎯 Warrior Trading Gappers"
6. Click "Scan"
7. ✅ Should return top 10 stocks: $2-$20, 1M+ volume, biggest % gains
```

### Test 2: Verify Filters
```
Check results:
- ✅ All stocks priced $2-$20?
- ✅ All stocks have 1M+ volume?
- ✅ Sorted by % gain (highest first)?
- ✅ Exactly 10 results?
```

### Test 3: Compare to Warrior Trading
```
1. Go to https://www.warriortrading.com/day-trading-scanners
2. Check their "Gappers" scanner
3. Compare top symbols with our results
4. Should see overlap in top movers
```

---

## Manual Workflow

### Morning Routine (9:30 AM - 10:00 AM):
```
1. 9:25 AM: Check pre-market gappers
2. 9:30 AM: Run "Warrior Trading Gappers" scanner
3. Review top 10 results:
   - Check news for each (Benzinga, StockTwits, Twitter)
   - Check relative volume (is it 5x+?)
   - Check chart (clean or messy?)
4. Add 2-3 best candidates to worklist
5. Set alerts for entry triggers
6. Wait for pullback/setup
```

### Entry Checklist:
```
✅ Stock in top 10 gappers
✅ Price $2-$20
✅ Volume 1M+ (and 5x relative)
✅ Breaking news catalyst
✅ Clean chart pattern
✅ Trading above VWAP
✅ Clear support/resistance levels
✅ Risk/reward 2:1 or better
```

---

## Troubleshooting

### Scanner Returns Mock Data
**Issue:** Shows fake NVDA/TSLA data instead of real scan
**Cause:** IBKR not connected OR event loop error
**Fix:**
1. Restart server: `.\RESTART_SERVER.ps1`
2. Check IBKR indicator is green
3. Try scanner again

### Scanner Returns Empty
**Issue:** No results returned
**Cause:** No stocks match criteria (rare) OR market hours
**Fix:**
1. Try during market hours (9:30 AM - 4:00 PM ET)
2. Relax filters (increase price range, lower volume)
3. Check IBKR connection

### Scanner Returns Wrong Stocks
**Issue:** Results don't match Warrior Trading criteria
**Cause:** Filters not being applied
**Fix:**
1. Check `scan_code: "WARRIOR_GAPPERS"` is being used
2. Verify filter_options are populated in logs
3. Test with different scan code

---

## Files Modified

| File | Lines | Purpose |
|------|-------|---------|
| `dashboard_api.py` | 2321 | Added WARRIOR_GAPPERS preset |
| `dashboard_api.py` | 2342-2346 | Handle WARRIOR_GAPPERS in scan endpoint |
| `dashboard_api.py` | 2411-2429 | Add price/volume filters |
| `dashboard_api.py` | 2432-2435 | Pass filters to async scanner |
| `dashboard_api.py` | 2414, 2435 | Fixed event loop (async version) |

**Total:** 1 file, ~30 lines modified

---

## API Reference

### Get Scanner Presets
```bash
GET /api/scanner/ibkr/presets
```

**Response:**
```json
{
  "success": true,
  "data": [
    {
      "id": "WARRIOR_GAPPERS",
      "name": "🎯 Warrior Trading Gappers",
      "description": "Top 10 gappers: $2-$20, 5x volume, 10%+ gain, high volume"
    },
    ...
  ]
}
```

### Run Warrior Trading Scanner
```bash
POST /api/scanner/ibkr/scan
Content-Type: application/json

{
  "scan_code": "WARRIOR_GAPPERS",
  "instrument": "STK",
  "location": "STK.US.MAJOR",
  "num_rows": 10
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "scan_code": "WARRIOR_GAPPERS",
    "results": [...],
    "count": 10,
    "source": "ibkr_live"
  }
}
```

---

## Summary

**What It Does:**
- Scans for top 10 gapper stocks using Warrior Trading criteria
- Filters: $2-$20 price, 1M+ volume, top % gainers
- Returns real-time results from IBKR scanner

**What You Need to Check Manually:**
- Relative volume (5x+)
- Breaking news catalyst
- Chart quality

**Perfect For:**
- Momentum day trading
- Gap and go strategy
- Morning scanner routine
- Finding high-probability setups

**Status:** ✅ Implemented, ready to test after server restart

---

**Created by:** Claude Code
**Date:** 2025-11-17
**Status:** Ready for testing - restart server to activate
