# Worklist Synchronization Guide

**Created:** 2025-11-17
**Issue:** Worklist not syncing across different UI pages

---

## Problem

User ran Warrior Trading scanner in one UI, added results to worklist, then opened a different UI and the worklist wasn't showing up.

---

## Root Cause

**Not all UIs have worklist functionality!**

Only **complete_platform.html** has full worklist support. Other UIs use watchlist (a different feature) or have no list functionality at all.

---

## UI Comparison

### ✅ complete_platform.html (RECOMMENDED)
**Location:** `http://127.0.0.1:9101/ui/complete_platform.html`

**Features:**
- ✅ Worklist with auto-sync (every 2 seconds)
- ✅ Warrior Trading Scanner
- ✅ Add scanner results to worklist
- ✅ AI predictions for worklist symbols
- ✅ Market data updates
- ✅ Order entry
- ✅ Positions/orders
- ✅ TradingView charts

**Worklist Support:** FULL - Uses `/api/worklist` API with auto-refresh

---

### ❌ platform.html (LEGACY)
**Location:** `http://127.0.0.1:9101/ui/platform.html`

**Features:**
- ❌ No worklist
- ✅ Watchlist (localStorage only, not synced)
- ✅ Order entry
- ✅ Basic charts

**Worklist Support:** NONE - This UI doesn't call the worklist API at all

---

### Other UIs

**trading_platform_complete.html:** Unknown worklist support
**professional_platform.html:** Unknown worklist support
**live_dashboard.html:** Unknown worklist support
**monitor.html:** Unknown worklist support

---

## How Worklist Works

### Server Side (dashboard_api.py)

```python
# Lines 2128-2129
worklist = []  # Shared in-memory list
worklist_predictions = {}
```

**Endpoints:**
- `GET /api/worklist` - Get all worklist items
- `POST /api/worklist/add` - Add single symbol
- `POST /api/scanner/ibkr/add-to-worklist` - Add scanner results (bulk)
- `DELETE /api/worklist/{symbol}` - Remove symbol

**Storage:** In-memory only (resets on server restart)

### Client Side (complete_platform.html)

```javascript
// Line 2378-2391: Load worklist from server
async function loadWorklist() {
    const response = await fetch(API_BASE_URL + '/api/worklist');
    const data = await response.json();
    worklistData = data.data;
    updateWorklistDisplay();
}

// Line 3433: Auto-refresh every 2 seconds
setInterval(() => {
    loadWorklist();  // Syncs with server
}, 2000);
```

**Storage:** None (always fetches from server)

---

## Solution

### ✅ To See Synchronized Worklist

**Use complete_platform.html ONLY:**

```
http://127.0.0.1:9101/ui/complete_platform.html
```

1. Open complete_platform.html in multiple browser tabs/windows
2. All tabs will sync automatically every 2 seconds
3. Add symbols in one tab → appears in all tabs within 2 seconds

---

### ❌ Don't Use platform.html for Worklist

platform.html has **no worklist functionality**. It has a watchlist instead, which is stored in localStorage and doesn't sync across tabs or with the server.

If you need worklist features:
- Scanner integration
- AI predictions
- Synchronized across tabs
- Market data updates

→ **Use complete_platform.html**

---

## Testing Sync

### Test 1: Single UI Multiple Tabs

```
1. Open complete_platform.html in Chrome tab 1
2. Open complete_platform.html in Chrome tab 2
3. In tab 1: Run Warrior Trading scanner
4. In tab 1: Add results to worklist
5. Wait 2 seconds
6. ✅ Tab 2 should show same worklist (auto-refreshed)
```

### Test 2: Verify API

```bash
# Check current worklist
curl http://127.0.0.1:9101/api/worklist

# Should return JSON with all worklist symbols
```

### Test 3: Add Symbol via API

```bash
# Add a symbol directly
curl -X POST http://127.0.0.1:9101/api/worklist/add \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL", "exchange": "SMART"}'

# Refresh complete_platform.html → should see AAPL within 2 seconds
```

---

## Current Worklist Status

As of testing at 2025-11-17 12:49:20, the worklist has **10 symbols:**

1. PACS - $15.77
2. ANVS - $3.29
3. SSP - $4.09
4. DGNX - $13.72
5. SGML - $7.92
6. SGBX - $2.73
7. RPTX - $2.12
8. VTGN - $4.46
9. IVVD - $2.83
10. DGXX - $3.83

✅ All symbols have live market data
✅ All symbols have AI predictions
✅ Worklist API is working correctly

---

## Limitations

### ⚠️ In-Memory Storage

The worklist is stored in-memory only:

```python
worklist = []  # Resets on server restart
```

**Impact:**
- Server restart → worklist cleared
- No persistence across sessions

**Future Enhancement:**
- Save to JSON file
- Or use SQLite database
- Auto-save on changes

### ⚠️ UI Support

Only 1 of 6+ UIs supports worklist:
- ✅ complete_platform.html
- ❌ platform.html
- ❓ Other UIs unknown

**Recommendation:** Standardize on complete_platform.html for all trading

---

## Troubleshooting

### Issue: Worklist not showing in UI

**Cause:** You're using platform.html or another UI without worklist support
**Fix:** Switch to complete_platform.html

### Issue: Worklist empty after server restart

**Cause:** Worklist is stored in-memory only
**Fix:** Expected behavior - add symbols again or implement persistence

### Issue: Worklist not syncing between tabs

**Cause 1:** Using different UIs (complete_platform.html vs platform.html)
**Fix:** Use complete_platform.html in all tabs

**Cause 2:** Not waiting 2 seconds for auto-refresh
**Fix:** Wait up to 2 seconds for sync

**Cause 3:** Browser cache
**Fix:** Hard refresh (Ctrl+Shift+R)

### Issue: Scanner results not adding to worklist

**Cause:** Checkboxes not selected
**Fix:** Check boxes next to symbols before clicking "Add All to Worklist"

---

## API Reference

### Get Worklist

```bash
GET /api/worklist
```

**Response:**
```json
{
  "success": true,
  "data": [
    {
      "symbol": "AAPL",
      "exchange": "SMART",
      "current_price": 150.25,
      "bid": 150.24,
      "ask": 150.26,
      "volume": 1234567,
      "prediction": "UP",
      "confidence": 0.85,
      "predicted_change": 2.5,
      "analysis": "Strong momentum...",
      "added_at": "2025-11-17T12:00:00",
      "has_live_data": true
    }
  ]
}
```

### Add to Worklist (Single Symbol)

```bash
POST /api/worklist/add
Content-Type: application/json

{
  "symbol": "AAPL",
  "exchange": "SMART",
  "notes": "Optional notes"
}
```

### Add Scanner Results (Bulk)

```bash
POST /api/scanner/ibkr/add-to-worklist
Content-Type: application/json

{
  "symbols": ["AAPL", "TSLA", "NVDA"]
}
```

### Remove from Worklist

```bash
DELETE /api/worklist/{symbol}
```

---

## Summary

**Problem:** User tried to view worklist in multiple UIs, but only complete_platform.html supports it.

**Solution:** Use complete_platform.html for all worklist functionality.

**Why:** Only complete_platform.html has the worklist API integration and auto-sync code.

**Recommendation:**
1. Bookmark: `http://127.0.0.1:9101/ui/complete_platform.html`
2. Use this UI for all trading (has all features)
3. Ignore platform.html (legacy UI with limited features)

---

**Status:** ✅ Worklist works correctly in complete_platform.html
**Date:** 2025-11-17
**Next Steps:** Use complete_platform.html exclusively for worklist features
