# UI Isolation Guide

## Overview

The Warrior Trading Bot now supports **UI isolation**, allowing multiple dashboard UIs to run simultaneously without interfering with each other's settings and layouts.

---

## Available Dashboards

### 1. Monitor Dashboard (Current)
- **URL**: `http://localhost:8000/ui/monitor.html`
- **UI Type**: `monitor`
- **Purpose**: Comprehensive monitoring with Trading, AI/ML, and Monitoring widgets

### 2. Trading Platform
- **URL**: `http://127.0.0.1:9101/ui/platform.html`
- **UI Type**: `platform`
- **Purpose**: Main trading interface

### 3. Complete Platform
- **URL**: `http://127.0.0.1:9101/ui/complete_platform.html`
- **UI Type**: `complete_platform`
- **Purpose**: Full-featured trading platform

---

## How UI Isolation Works

### 1. Separate Storage Namespaces

Each UI type has its own isolated storage:

**localStorage Keys:**
- Monitor: `monitor_dashboard-layout`
- Platform: `platform_dashboard-layout`
- Complete Platform: `complete_platform_dashboard-layout`

**Database Filtering:**
- Layouts are filtered by `ui_type` column
- Each UI only sees its own saved layouts

### 2. Independent Settings

Each dashboard maintains:
- ✅ Widget arrangements
- ✅ Custom layouts
- ✅ Default layout preferences
- ✅ Widget sizes and positions

### 3. Zero Interference

- Saving a layout in Monitor Dashboard won't affect Platform Dashboard
- Each UI loads only its own saved configurations
- Settings are completely isolated from each other

---

## Switching Between UIs

### Method 1: Menu Navigation (Recommended)

In Monitor Dashboard:
1. Click **"Switch UI"** in the menu bar
2. Select your desired dashboard:
   - Monitor Dashboard (Current)
   - Trading Platform
   - Complete Platform

### Method 2: Direct URLs

Navigate directly to:
- `http://localhost:8000/ui/monitor.html`
- `http://127.0.0.1:9101/ui/platform.html`
- `http://127.0.0.1:9101/ui/complete_platform.html`

---

## Usage Examples

### Example 1: Multi-Monitor Setup

**Monitor 1 (Primary):**
```
Open: http://localhost:8000/ui/monitor.html
Add: Scanner Results, Active Trades, Risk Manager
Save as: "Primary Trading Monitor"
```

**Monitor 2 (Secondary):**
```
Open: http://127.0.0.1:9101/ui/platform.html
Add: Charts, Order Book, Performance Metrics
Save as: "Secondary Platform Monitor"
```

**Monitor 3 (Analytics):**
```
Open: http://127.0.0.1:9101/ui/complete_platform.html
Add: P&L Charts, ML Predictions, Sentiment Analysis
Save as: "Analytics Dashboard"
```

### Example 2: Workflow-Based Layouts

**Pre-Market Scan Layout (Monitor Dashboard):**
- Scanner Results (large)
- Watch List (medium)
- ML Pattern Detection (medium)
- Sentiment Analysis (medium)

**Live Trading Layout (Platform Dashboard):**
- Active Trades (large)
- Active Orders (large)
- Risk Manager (medium)
- Slippage Monitor (small)

**Post-Market Analysis (Complete Platform):**
- Trade History (large)
- Performance Metrics (large)
- P&L Chart (large)
- Error Log (medium)

---

## Technical Implementation

### Frontend (JavaScript)

**UI Type Constant:**
```javascript
// In monitor-dragdrop.js
const UI_TYPE = 'monitor';  // Changes based on UI
```

**localStorage Operations:**
```javascript
// Save layout
const storageKey = `${UI_TYPE}_dashboard-layout`;
localStorage.setItem(storageKey, JSON.stringify(layout));

// Load layout
const stored = localStorage.getItem(storageKey);
```

### Backend (Python)

**Database Schema:**
```sql
CREATE TABLE user_layouts (
    ...
    ui_type TEXT DEFAULT 'monitor',
    ...
);
```

**API Filtering:**
```python
# Get layouts for specific UI type
layouts = db.get_layouts(ui_type='monitor')

# Save layout with UI type
layout_id = db.save_layout(
    layout_name='My Layout',
    layout_config=config,
    ui_type='monitor'
)
```

---

## Migration

If you have an existing database, the UI isolation system includes automatic migration:

### Running the Migration

```bash
cd C:\ai_project_hub\store\code\IBKR_Algo_BOT_V2
python database/migrate_ui_type.py
```

### What the Migration Does

1. **Adds `ui_type` column** to `user_layouts` table
2. **Sets existing layouts** to `ui_type='monitor'`
3. **Preserves all data** - no layouts are lost

**Output:**
```
============================================================
Database Migration: UI Type Isolation
============================================================
Migrating database: C:\...\warrior_trading.db
[OK] ui_type column already exists - no migration needed
============================================================
```

---

## API Endpoints

### Save Layout (with UI Type)

**Endpoint:** `POST /api/monitoring/layouts/save`

**Request:**
```json
{
  "layout_name": "My Trading Setup",
  "layout_config": {...},
  "is_default": false,
  "ui_type": "monitor"
}
```

### Get Layouts (filtered by UI Type)

**Endpoint:** `GET /api/monitoring/layouts?ui_type=monitor`

**Response:**
```json
{
  "success": true,
  "count": 3,
  "ui_type": "monitor",
  "layouts": [...]
}
```

### Get Default Layout (for UI Type)

**Endpoint:** `GET /api/monitoring/layouts/default?ui_type=monitor`

**Response:**
```json
{
  "success": true,
  "ui_type": "monitor",
  "layout": {...}
}
```

---

## Troubleshooting

### Issue: Layouts Not Loading

**Check:**
1. Verify UI type matches: `console.log(UI_TYPE)`
2. Check localStorage: `localStorage.getItem('monitor_dashboard-layout')`
3. Verify database migration: `python database/migrate_ui_type.py`

### Issue: Settings Bleeding Between UIs

**Solution:**
- Clear browser cache and reload
- Verify each UI has correct `UI_TYPE` constant
- Check that localStorage keys are different

### Issue: Cannot Save Layouts

**Check:**
1. Database migration completed successfully
2. Backend API is running
3. Network tab shows successful POST request
4. Check browser console for errors

---

## Best Practices

### 1. Naming Conventions

Use descriptive layout names that indicate UI type:
- ✅ "Monitor - Pre-Market Scan"
- ✅ "Platform - Live Trading"
- ✅ "Complete - Post-Market Analysis"
- ❌ "My Layout" (too generic)

### 2. Widget Organization

**Monitor Dashboard:**
- Focus on monitoring and oversight
- Scanner Results, Risk Manager, Alerts

**Platform Dashboard:**
- Focus on active trading
- Active Orders, Active Trades, Quick Entry

**Complete Platform:**
- Focus on analysis and performance
- Charts, Metrics, History

### 3. Multi-Monitor Strategy

- Primary Monitor: Active trading (Platform)
- Secondary Monitor: Monitoring & alerts (Monitor)
- Third Monitor: Analysis & research (Complete)

---

## Future Enhancements

The UI isolation system is designed to support:

- [ ] Cross-UI widget sharing
- [ ] Layout templates marketplace
- [ ] Cloud-synced layouts
- [ ] Mobile-optimized UI types
- [ ] Custom UI type creation

---

## Summary

✅ **3 Independent Dashboards**
- Monitor Dashboard
- Trading Platform
- Complete Platform

✅ **Isolated Storage**
- Separate localStorage namespaces
- Database filtering by UI type
- Zero interference

✅ **Easy Navigation**
- "Switch UI" menu
- Direct URL access
- Bookmark support

✅ **Backward Compatible**
- Automatic migration
- Existing layouts preserved
- No manual intervention needed

---

For more information, see:
- [Monitoring Dashboard Guide](MONITORING_DASHBOARD_GUIDE.md)
- [Monitoring System Implementation](MONITORING_SYSTEM_IMPLEMENTATION.md)
