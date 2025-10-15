# 🚀 Trading Dashboard - Quick Reference Card

## ⚡ Quick Start (3 Steps)

```bash
# 1. Run setup script (one-time)
python setup_dashboard.py

# 2. Start dashboard
start_dashboard.bat

# 3. Open browser
http://localhost:3000
```

---

## 📁 File Structure

```
C:\IBKR_Algo_BOT\
├── dashboard_api.py          ← Backend API server
├── database.py                ← Database management
├── setup_dashboard.py         ← Installation script
├── start_dashboard.bat        ← Start everything
├── start_backend.bat          ← Start API only
├── start_frontend.bat         ← Start UI only
│
├── dashboard_data/            ← Data storage
│   └── trading_bot.db        ← SQLite database
│
└── frontend/                  ← React application
    ├── package.json
    ├── vite.config.js
    └── src/
        └── App.jsx           ← Main dashboard UI
```

---

## 🔧 Commands Cheat Sheet

### Starting/Stopping

```bash
# Start everything (recommended)
start_dashboard.bat

# Start backend only
python dashboard_api.py

# Start frontend only
cd frontend
npm run dev

# Stop all (Ctrl+C in terminals)
```

### Testing

```bash
# Test backend health
curl http://localhost:5000/api/health

# Test status endpoint
curl http://localhost:5000/api/status

# Start MTF bot via API
curl -X POST http://localhost:5000/api/mtf/start

# Stop MTF bot via API
curl -X POST http://localhost:5000/api/mtf/stop
```

### Database

```bash
# Backup database
copy dashboard_data\trading_bot.db backups\backup_YYYYMMDD.db

# Reset database (delete and restart backend)
del dashboard_data\trading_bot.db
python dashboard_api.py
```

---

## 🌐 URLs

| Service | URL | Purpose |
|---------|-----|---------|
| Dashboard | http://localhost:3000 | Main UI |
| API | http://localhost:5000 | Backend API |
| Health Check | http://localhost:5000/api/health | Test backend |
| WebSocket | ws://localhost:5000/socket.io/ | Real-time updates |

---

## 📊 API Endpoints

### System Control

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/status` | Get all system status |
| POST | `/api/mtf/start` | Start MTF bot |
| POST | `/api/mtf/stop` | Stop MTF bot |
| POST | `/api/warrior/start` | Start Warrior scanner |
| POST | `/api/warrior/stop` | Stop Warrior scanner |
| POST | `/api/emergency-stop` | Stop all bots |

### Data Retrieval

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/positions` | Get current positions |
| GET | `/api/trades` | Get trade history |
| GET | `/api/logs` | Get activity logs |
| GET | `/api/pnl` | Get P&L summary |

---

## 🎨 UI Components

### Main Dashboard Sections

1. **System Status Cards** (Top Row)
   - MTF Swing Trading
   - Warrior Momentum
   - IBKR Connection Status

2. **Total P&L Banner**
   - Combined profit/loss
   - Updates every 5 seconds

3. **Current Positions Table**
   - All open positions
   - Real-time P&L
   - Duration tracking

4. **Recent Trades**
   - Last 10 trades
   - Buy/Sell indicators
   - Profit/loss for each

5. **Live Activity Log**
   - Real-time messages
   - Color-coded severity
   - Auto-scroll

---

## 🔴 Status Indicators

| Color | Meaning |
|-------|---------|
| 🟢 Green | Running / Success |
| 🔴 Red | Stopped / Error |
| ⚪ Gray | Inactive |
| 🟡 Yellow | Warning |
| 🔵 Blue | Info |

---

## 🐛 Quick Troubleshooting

### Dashboard Won't Load

```bash
# Check if backend is running
curl http://localhost:5000/api/health

# If not, start it
python dashboard_api.py
```

### "Cannot connect to API" Error

1. Verify backend is running on port 5000
2. Check browser console (F12) for errors
3. Ensure no firewall blocking localhost

### WebSocket Not Connecting

```bash
# Reinstall socket dependencies
pip install python-socketio flask-socketio --force-reinstall
```

### Database Errors

```bash
# Delete and recreate
del dashboard_data\trading_bot.db
python dashboard_api.py
```

---

## 💡 Pro Tips

### Speed Up Updates

Edit React component polling interval:

```javascript
// Faster updates (2 seconds)
const interval = setInterval(fetchAllData, 2000);
```

### Reduce CPU Usage

```javascript
// Slower updates (10 seconds)
const interval = setInterval(fetchAllData, 10000);
```

### View More Logs

In browser, change limit:

```javascript
const logs = await api.getLogs({ limit: 200 }); // Default: 100
```

### Export Logs

Add to React component:

```javascript
const exportLogs = () => {
  const csv = logs.map(l => 
    `${l.timestamp},${l.level},${l.source},"${l.message}"`
  ).join('\n');
  
  const blob = new Blob([csv], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'logs.csv';
  a.click();
};
```

---

## 🔒 Security Checklist

**For Local Use (Current)**
- ✅ Running on localhost only
- ✅ No external access
- ✅ Safe for development

**Before External Deployment**
- ⚠️ Add authentication
- ⚠️ Enable HTTPS
- ⚠️ Use API keys
- ⚠️ Add rate limiting
- ⚠️ Configure firewall

---

## 📱 Keyboard Shortcuts

| Key | Action |
|-----|--------|
| F5 | Refresh dashboard |
| F12 | Open DevTools (debugging) |
| Ctrl+R | Reload page |
| Ctrl+Shift+I | Open console |

---

## 🎯 Common Tasks

### Add New Symbol to Track

1. **Via Warrior Scanner**: Automatically found and added
2. **Via API**:
```python
import requests
requests.post('http://localhost:5000/api/watchlist/add', 
              json={'symbol': 'NVDA'})
```

### Start Trading Session

```bash
# 1. Ensure TWS is running (port 7497)
# 2. Start dashboard
start_dashboard.bat

# 3. In dashboard UI:
#    - Click START on MTF card (for swing trading)
#    - Click START on Warrior card (for momentum trading)
```

### End Trading Session

```bash
# 1. In dashboard: Click STOP on each bot
# 2. Or use emergency stop for all
# 3. Close dashboard (Ctrl+C in terminals)
```

### View Historical Performance

```bash
# Query database directly
sqlite3 dashboard_data/trading_bot.db

# Example queries:
SELECT * FROM trades WHERE date(timestamp) = date('now');
SELECT symbol, SUM(pnl) as total_pnl FROM trades GROUP BY symbol;
SELECT strategy, AVG(pnl) as avg_pnl FROM trades GROUP BY strategy;
```

---

## 📊 Database Tables

### Available Tables

| Table | Purpose |
|-------|---------|
| `trades` | Completed trades |
| `positions` | Current open positions |
| `activity_logs` | System activity |
| `watchlists` | Saved watchlists |
| `watchlist_symbols` | Symbols in watchlists |
| `training_history` | Model training records |
| `performance_metrics` | Daily performance stats |

### Quick Queries

```sql
-- Today's P&L
SELECT SUM(pnl) FROM trades 
WHERE date(timestamp) = date('now');

-- Win rate by strategy
SELECT strategy, 
       COUNT(*) as total,
       SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
       ROUND(SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as win_rate
FROM trades 
GROUP BY strategy;

-- Best performing symbols
SELECT symbol, 
       COUNT(*) as trades, 
       SUM(pnl) as total_pnl,
       AVG(pnl) as avg_pnl
FROM trades 
GROUP BY symbol 
ORDER BY total_pnl DESC 
LIMIT 10;
```

---

## 🔄 Update Flow

### How Data Flows

```
Bot executes trade
    ↓
Calls update_dashboard_trade()
    ↓
POST to /api/trade/add
    ↓
Backend stores in database
    ↓
WebSocket broadcasts update
    ↓
React frontend receives
    ↓
UI updates instantly
```

### Polling Schedule

- **Status**: Every 5 seconds
- **Positions**: Every 5 seconds
- **Trades**: Every 5 seconds
- **Logs**: Via WebSocket (instant)
- **P&L**: Every 5 seconds

---

## 🎨 Customization Examples

### Change Dashboard Colors

Edit React component:

```javascript
// Change P&L banner gradient
className="bg-gradient-to-r from-green-600 to-blue-600"

// Change status colors
const colors = {
  running: 'bg-blue-500',  // Change to 'bg-purple-500'
  stopped: 'bg-gray-500',
  error: 'bg-red-500'
};
```

### Add Sound Notifications

```javascript
// In React component
const playSound = (type) => {
  const audio = new Audio(type === 'win' ? '/win.mp3' : '/loss.mp3');
  audio.play();
};

// When trade completes
if (trade.pnl > 0) {
  playSound('win');
}
```

### Add Email Alerts

```python
# In dashboard_api.py
import smtplib
from email.mime.text import MIMEText

def send_alert(subject, message):
    msg = MIMEText(message)
    msg['Subject'] = subject
    msg['From'] = 'your-email@gmail.com'
    msg['To'] = 'your-email@gmail.com'
    
    with smtplib.SMTP('smtp.gmail.com', 587) as server:
        server.starttls()
        server.login('your-email@gmail.com', 'your-app-password')
        server.send_message(msg)

# Call when big win
if pnl > 500:
    send_alert('Big Win!', f'Made ${pnl} on {symbol}')
```

---

## 📈 Performance Monitoring

### Key Metrics to Watch

1. **Win Rate**: Should be >55% for profitability
2. **Profit Factor**: Avg win / Avg loss (target >1.5)
3. **Max Drawdown**: Largest loss streak (keep <10%)
4. **Sharpe Ratio**: Risk-adjusted return (target >2.0)

### Dashboard Shows

- ✅ Real-time P&L
- ✅ Win/loss count
- ✅ Active positions
- ✅ Recent trade history

### Need to Add (Future)

- 📊 P&L chart over time
- 📊 Win rate by symbol
- 📊 Average hold time
- 📊 Best trading hours

---

## 🔧 Integration with Existing Bots

### MTF Bot Integration

Add to `ibkr_live_trading_connector.py`:

```python
import requests

def notify_dashboard(event, data):
    """Send updates to dashboard"""
    try:
        requests.post(
            'http://localhost:5000/api/event',
            json={'event': event, 'data': data},
            timeout=1
        )
    except:
        pass  # Don't break bot if dashboard is down

# When opening position
notify_dashboard('position_opened', {
    'symbol': symbol,
    'quantity': qty,
    'entry_price': price,
    'strategy': 'MTF'
})

# When closing position
notify_dashboard('position_closed', {
    'symbol': symbol,
    'pnl': pnl,
    'strategy': 'MTF'
})
```

### Warrior Scanner Integration

Add to `warrior_momentum_scanner.py`:

```python
def send_gappers_to_dashboard(gappers):
    """Send found gappers to dashboard"""
    try:
        requests.post(
            'http://localhost:5000/api/warrior/gappers',
            json={'gappers': gappers},
            timeout=1
        )
    except:
        pass
```

---

## 🎓 Learning Resources

### Understanding the Stack

- **Flask**: Backend web framework (Python)
- **React**: Frontend UI framework (JavaScript)
- **SQLite**: Database for storing data
- **WebSocket**: Real-time bidirectional communication
- **Tailwind CSS**: Styling framework

### Recommended Reading

1. Flask docs: https://flask.palletsprojects.com/
2. React tutorial: https://react.dev/learn
3. Tailwind: https://tailwindcss.com/docs
4. WebSocket guide: https://socket.io/docs/

---

## 💾 Backup Strategy

### What to Backup

```bash
# Database (critical)
copy dashboard_data\trading_bot.db backups\

# Configuration files
copy dashboard_api.py backups\
copy database.py backups\

# Frontend code
xcopy frontend\src backups\frontend\src /E /I
```

### Automated Daily Backup

Create scheduled task (Windows):

```batch
@echo off
set TIMESTAMP=%date:~-4,4%%date:~-10,2%%date:~-7,2%
xcopy dashboard_data backups\%TIMESTAMP%\ /E /I /Y
echo Backup completed: %TIMESTAMP%
```

---

## 🚨 Emergency Procedures

### Bot Going Wild

```bash
# 1. Emergency stop via dashboard UI
Click "EMERGENCY STOP" button

# 2. Or via API
curl -X POST http://localhost:5000/api/emergency-stop

# 3. Or kill processes
taskkill /F /IM python.exe
```

### Data Corruption

```bash
# 1. Stop all services
# 2. Restore from backup
copy backups\latest\trading_bot.db dashboard_data\

# 3. Restart services
start_dashboard.bat
```

### Can't Access Dashboard

```bash
# Check what's running on ports
netstat -ano | findstr :5000
netstat -ano | findstr :3000

# Kill process if needed
taskkill /F /PID <process_id>
```

---

## 📞 Quick Help Commands

```bash
# Check Python version
python --version

# Check Node version
node --version
npm --version

# Check installed Python packages
pip list | findstr flask

# Check frontend dependencies
cd frontend
npm list

# Test backend is responding
curl http://localhost:5000/api/health
```

---

## 🎯 Development Roadmap

### Phase 1 (Current) ✅
- Basic dashboard UI
- System control
- Real-time monitoring
- Activity logging

### Phase 2 (Next)
- [ ] Watchlist manager
- [ ] Training interface
- [ ] Backtest viewer
- [ ] Performance charts

### Phase 3 (Future)
- [ ] Advanced analytics
- [ ] Strategy optimizer
- [ ] Risk calculator
- [ ] Paper trading mode

### Phase 4 (Advanced)
- [ ] Multi-account support
- [ ] Cloud deployment
- [ ] Mobile app
- [ ] Alert system

---

## ✅ Daily Checklist

### Before Trading

- [ ] TWS/Gateway running (port 7497)
- [ ] Dashboard backend started
- [ ] Dashboard frontend accessible
- [ ] IBKR connection verified (green status)
- [ ] Models loaded for symbols trading
- [ ] Risk limits configured

### During Trading

- [ ] Monitor dashboard for errors
- [ ] Check positions match expectations
- [ ] Watch P&L vs. daily limits
- [ ] Review activity log periodically
- [ ] Verify trades executing correctly

### After Trading

- [ ] Review day's trades
- [ ] Check win rate
- [ ] Export logs if needed
- [ ] Backup database
- [ ] Stop all services
- [ ] Document any issues

---

## 🎉 Success Metrics

### Dashboard is Working When:

✅ UI loads at localhost:3000
✅ Backend API responds at localhost:5000
✅ WebSocket shows "connected" in DevTools
✅ Status cards update every 5 seconds
✅ Start/Stop buttons work
✅ Activity log shows messages
✅ Positions table displays correctly
✅ Can see IBKR connection status

### Bots are Integrated When:

✅ Dashboard shows "Running" when bot starts
✅ Positions appear in dashboard
✅ Trades logged to Recent Trades table
✅ Activity log shows bot messages
✅ P&L updates in real-time
✅ Can start/stop from dashboard

---

## 📌 Important Notes

⚠️ **Always run in paper trading first!**
⚠️ **Test with small position sizes**
⚠️ **Monitor the first few days closely**
⚠️ **Backup database regularly**
⚠️ **Don't expose dashboard to internet without auth**

---

## 🏆 You're All Set!

This quick reference has everything you need. Save it and refer back as you use the dashboard!

**Main Commands to Remember:**

```bash
# Setup (once)
python setup_dashboard.py

# Start dashboard
start_dashboard.bat

# Access UI
http://localhost:3000
```

Happy trading! 📈🚀💰