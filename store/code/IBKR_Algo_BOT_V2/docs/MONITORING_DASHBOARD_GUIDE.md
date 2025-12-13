# Warrior Trading Monitor - User Guide

## Overview

The Warrior Trading Monitor is a comprehensive multi-widget dashboard system for real-time trade tracking, error monitoring, and performance analysis. Built with a modular architecture, it supports custom layouts, drag-and-drop widgets, multi-monitor setups, and real-time WebSocket updates.

---

## Features

### Core Capabilities

- **Real-Time Trade Monitoring**: Track active positions with live P&L updates
- **Trade History**: Complete trade log with filtering and sorting
- **Performance Analytics**: Win rate, profit factor, R multiples, and more
- **Error Tracking**: System error logs with severity levels and resolution tracking
- **Slippage Monitoring**: Execution quality tracking with cost analysis
- **Custom Layouts**: Save and load personalized dashboard configurations
- **Multi-Monitor Support**: Span widgets across multiple screens
- **WebSocket Updates**: Real-time data streaming without page refresh
- **Drag-and-Drop**: Reposition and resize widgets freely
- **Keyboard Shortcuts**: Fast navigation and control

---

## Getting Started

### 1. Start the API Server

```bash
cd C:\ai_project_hub\store\code\IBKR_Algo_BOT_V2
python dashboard_api.py
```

The server will start on `http://localhost:8000`

### 2. Open the Monitor

Navigate to: `http://localhost:8000/ui/monitor.html`

### 3. Initial Setup

On first load, you'll see a default layout with:
- Active Trades widget
- Performance Metrics widget
- P&L Chart widget
- Error Log widget

---

## Widget Types

### 1. Active Trades
**Icon**: 📋 List
**Purpose**: Monitor currently open positions
**Data**: Symbol, Side, Shares, Entry Price, Current Price, P&L, Pattern
**Actions**: Exit trade button
**Refresh**: 2 seconds

### 2. Trade History
**Icon**: 🕐 History
**Purpose**: Review closed trades
**Data**: Time, Symbol, Side, Entry, Exit, P&L, R Multiple, Status
**Features**: Filter by symbol and status
**Refresh**: 5 seconds

### 3. Performance Metrics
**Icon**: 📊 Chart Bar
**Purpose**: Key performance indicators
**Metrics**:
- Win Rate (% of winning trades)
- Total P&L (cumulative profit/loss)
- Profit Factor (gross profit / gross loss)
- Average R Multiple (risk-reward ratio)
- Total Trades
- Best Trade (largest winner)
**Refresh**: 10 seconds

### 4. P&L Chart
**Icon**: 📈 Chart Area
**Purpose**: Visual profit/loss over time
**Display**: Cumulative P&L line chart (30 days)
**Refresh**: 10 seconds

### 5. Error Log
**Icon**: ⚠️ Exclamation Triangle
**Purpose**: System error tracking
**Data**: Timestamp, Severity, Module, Error Message, Status
**Features**: Resolve errors, filter by severity
**Refresh**: 5 seconds

### 6. Slippage Monitor
**Icon**: 🏎️ Tachometer
**Purpose**: Execution quality analysis
**Metrics**:
- Total Executions
- Average Slippage
- Acceptable Count (≤0.1%)
- Warning Count (0.1-0.25%)
- Critical Count (>0.25%)
- Total Slippage Cost
**Refresh**: 5 seconds

### 7. Watch List
**Icon**: 👁️ Eye
**Purpose**: Monitor specific symbols
**Refresh**: 3 seconds

### 8. Alerts
**Icon**: 🔔 Bell
**Purpose**: System notifications and alerts
**Types**: Info, Warning, Critical
**Refresh**: 3 seconds

---

## Using the Menu System

### File Menu

**New Dashboard** (Ctrl+N)
Clear current layout and start fresh

**Load Layout** (Ctrl+O)
Load a previously saved layout configuration

**Save Layout** (Ctrl+S)
Save current widget arrangement with a custom name

**Export Data**
Export trades, errors, or performance data to CSV/JSON

### Widgets Menu

Click any widget type to add it to the dashboard:
- Active Trades
- Trade History
- Performance Metrics
- P&L Chart
- Error Log
- Slippage Monitor
- Watch List
- Alerts

When adding a widget, you'll be prompted for size:
- `sm` - Small (3x3 grid cells)
- `md` - Medium (4x4 grid cells)
- `lg` - Large (6x6 grid cells)
- `xl` - Extra Large (8x8 grid cells)
- `full` - Full Screen (12x12 grid cells)

### View Menu

**Fullscreen** (F11)
Toggle fullscreen mode for distraction-free monitoring

**Reset Layout** (Ctrl+R)
Return to default widget arrangement

**Toggle Grid**
Show/hide grid lines for alignment

### Tools Menu

**Custom Monitor Builder**
Visual interface for creating custom dashboard layouts (Coming Soon)

**Settings**
Configure refresh intervals, API endpoints, themes

**Manage Layouts**
View, rename, delete, and set default layouts

### Help Menu

**Documentation**
Open this guide

**Keyboard Shortcuts**
View all available shortcuts

**About**
Version and feature information

---

## Widget Controls

### Header Controls

Every widget has three control buttons in the top-right:

1. **Refresh** (🔄): Manually refresh widget data
2. **Minimize** (➖): Collapse widget to header only
3. **Close** (✖️): Remove widget from dashboard

### Moving Widgets

1. Click and hold the widget **header**
2. Drag to desired position
3. Drop to place widget
4. Layout auto-saves to localStorage

### Resizing Widgets

1. Hover over **bottom-right corner** of widget
2. Click and drag resize handle
3. Release to set new size
4. Layout auto-saves

---

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| Ctrl+N | New Dashboard |
| Ctrl+S | Save Layout |
| Ctrl+O | Load Layout |
| Ctrl+R | Reset Layout |
| F11 | Toggle Fullscreen |

---

## Multi-Monitor Setup

### Setting Up Multi-Monitor Dashboards

1. **Save Primary Monitor Layout**
   - Arrange widgets for your primary screen
   - File > Save Layout > "Primary Monitor"

2. **Create Secondary Monitor Layout**
   - Open monitor in new browser window
   - Move window to second monitor
   - Add different widgets (e.g., charts on secondary)
   - File > Save Layout > "Secondary Monitor"

3. **Load Layouts on Startup**
   - Open two browser windows/tabs
   - In window 1: File > Load Layout > "Primary Monitor"
   - In window 2: File > Load Layout > "Secondary Monitor"
   - Move window 2 to second monitor

### Example Multi-Monitor Setups

**Day Trading Setup**:
- **Monitor 1**: Active Trades, Slippage Monitor, Alerts
- **Monitor 2**: P&L Chart, Performance Metrics
- **Monitor 3**: Trade History, Error Log

**Risk Management Setup**:
- **Monitor 1**: Active Trades, Slippage Monitor
- **Monitor 2**: Performance Metrics, Error Log
- **Monitor 3**: Trade History, P&L Chart

---

## Filtering and Searching

### Trade History Filters

1. **By Symbol**: Enter ticker (e.g., "AAPL")
2. **By Status**: Select from dropdown (Closed, Stopped, Target)
3. Click **Filter** button to apply

### Error Log Filters

1. **By Severity**: Critical, Error, Warning
2. **Show Resolved**: Toggle checkbox
3. Auto-refreshes on change

---

## Real-Time Updates

### WebSocket Connection

The monitor maintains a WebSocket connection to the backend for real-time updates:

**Connection Status Indicators**:
- 🟢 **Connected**: Real-time updates active
- 🟡 **Connecting**: Reconnecting to server
- 🔴 **Disconnected**: No connection (check server)

**Real-Time Events**:
- New trade executions → Active Trades updates
- Trade exits → Trade History updates
- System errors → Error Log alerts
- Performance calculations → Metrics refresh

### Manual Refresh

If real-time updates aren't working:
1. Click refresh button (🔄) on individual widgets
2. Or wait for auto-refresh interval

---

## Layout Management

### Saving Layouts

1. Arrange widgets as desired
2. File > Save Layout
3. Enter layout name (e.g., "Morning Trading")
4. Choose "Set as default" if desired
5. Click OK

Layouts are saved to:
- **Server**: Accessible from any device
- **localStorage**: Quick access, browser-specific

### Loading Layouts

1. File > Load Layout
2. Select from list of saved layouts
3. Click OK

Dashboard will clear and recreate saved configuration.

### Default Layout

Set a layout as default to auto-load on startup:
1. Save layout with "Set as default" checked
2. On next page load, default layout loads automatically

---

## Troubleshooting

### Dashboard Not Loading

**Check**:
1. API server is running: `python dashboard_api.py`
2. Navigate to correct URL: `http://localhost:8000/ui/monitor.html`
3. Check browser console for errors (F12)

### Widgets Not Updating

**Check**:
1. Connection status (top-right corner)
2. WebSocket connection (should show "Connected")
3. API health endpoint: `http://localhost:8000/api/monitoring/health`

### Data Not Displaying

**Check**:
1. Database exists: `database/warrior_trading.db`
2. Tables initialized: Run schema.sql
3. Trade data logged: Check trade history endpoint

### Layout Not Saving

**Check**:
1. Browser localStorage enabled
2. No browser privacy/incognito mode
3. Check browser console for errors

---

## API Endpoints

The monitor uses these backend endpoints:

### Trades
- `GET /api/monitoring/trades` - Get trade history
- `GET /api/monitoring/active-trades` - Get open positions
- `GET /api/monitoring/trades/summary` - Get statistics

### Errors
- `GET /api/monitoring/errors` - Get error logs
- `POST /api/monitoring/errors/resolve` - Mark error resolved
- `GET /api/monitoring/errors/stats` - Get error statistics

### Performance
- `GET /api/monitoring/performance/daily` - Get daily metrics
- `POST /api/monitoring/performance/calculate` - Calculate metrics
- `GET /api/monitoring/slippage/stats` - Get slippage stats

### Layouts
- `POST /api/monitoring/layouts/save` - Save layout
- `GET /api/monitoring/layouts` - Get all layouts
- `GET /api/monitoring/layouts/default` - Get default layout

### WebSocket
- `WS /api/monitoring/stream` - Real-time updates

---

## Performance Tips

### Optimize for Speed

1. **Reduce Refresh Intervals**: Edit in `monitor-widgets.js`
2. **Limit Widget Count**: More widgets = more API calls
3. **Use Filters**: Reduce data returned by API
4. **Close Unused Widgets**: Free up resources

### Large Datasets

If you have thousands of trades:
1. Use date range filters
2. Increase pagination limits carefully
3. Consider data archiving for old trades

---

## Customization

### Adding Custom Widgets

1. Create widget template in `monitor.html`:
```html
<template id="my-widget-template">
    <div class="widget-content">
        <!-- Your custom content -->
    </div>
</template>
```

2. Add widget type in `monitor-widgets.js`:
```javascript
case 'my-widget':
    await this.updateMyWidget(widgetId);
    break;
```

3. Add update function:
```javascript
async updateMyWidget(widgetId) {
    const data = await MonitorAPI.get('/api/my-endpoint');
    // Update DOM
}
```

4. Add to Widgets menu in HTML

### Changing Themes

Edit CSS variables in `monitor.css`:
```css
:root {
    --bg-primary: #0a0e1a;    /* Change background */
    --accent-primary: #3b82f6; /* Change accent color */
}
```

---

## Best Practices

### For Day Trading
1. Use Active Trades + Slippage Monitor on primary screen
2. Keep Error Log visible for instant alerts
3. Set aggressive refresh intervals (2-3 seconds)
4. Use fullscreen mode during market hours

### For Performance Review
1. Focus on Performance Metrics + P&L Chart
2. Use Trade History with date filters
3. Longer refresh intervals (10-30 seconds)
4. Save layouts for weekly/monthly reviews

### For Multi-Session Trading
1. Save layouts for different strategies
2. Use custom names ("Scalping", "Swing", "Options")
3. Load appropriate layout per strategy
4. Track performance separately per layout

---

## Support

For issues, feature requests, or questions:
- Check browser console for errors
- Review API logs: `dashboard_api.py` output
- Check database: `database/warrior_trading.db`
- GitHub Issues: https://github.com/bgmaynard/ai_project_hub

---

## Version History

**v1.0.0** - Initial Release
- Multi-widget dashboard
- 8 widget types
- Drag-and-drop interface
- Layout management
- Real-time WebSocket updates
- Multi-monitor support
