# 🚀 Trading Bot Dashboard - Complete Setup Guide

## 📋 What You're Getting

A professional web-based dashboard to monitor and control your trading systems:

✅ **Real-time monitoring** of MTF and Warrior bots  
✅ **Live P&L tracking** and position management  
✅ **Activity logging** with color-coded messages  
✅ **Start/Stop controls** for each trading system  
✅ **Professional UI** with modern design  
✅ **WebSocket integration** for instant updates  

---

## 🏗️ Architecture Overview

```
┌─────────────────┐
│  React Frontend │ ← What you see in browser
│  (Port 3000)    │
└────────┬────────┘
         │
         ↓ HTTP/WebSocket
┌─────────────────┐
│  Flask Backend  │ ← API Server
│  (Port 5000)    │
└────────┬────────┘
         │
    ┌────┴────┬─────────┐
    ↓         ↓         ↓
┌───────┐ ┌──────┐ ┌────────┐
│ MTF   │ │Warrior│ │Database│
│ Bot   │ │ Bot   │ │SQLite  │
└───────┘ └──────┘ └────────┘
```

---

## 📦 Installation Steps

### Step 1: Install Required Python Packages

```bash
# Navigate to your project directory
cd C:\IBKR_Algo_BOT

# Install backend dependencies
pip install flask flask-cors flask-socketio python-socketio
```

### Step 2: Create Project Structure

Create these directories:

```
C:\IBKR_Algo_BOT\
├── dashboard_api.py          (Backend API - provided above)
├── database.py                (Database module - provided above)
├── dashboard_data/            (Will be auto-created)
│   └── trading_bot.db        (SQLite database)
└── frontend/                  (React app)
    ├── package.json
    ├── vite.config.js
    └── src/
        └── App.jsx           (Dashboard UI - provided above)
```

### Step 3: Set Up React Frontend

#### Option A: Quick Setup (Recommended)

1. **Install Node.js** (if not installed):
   - Download from https://nodejs.org/ (LTS version)
   - Verify installation: `node --version`

2. **Create React app**:

```bash
# In C:\IBKR_Algo_BOT\frontend
npm create vite@latest . -- --template react
npm install
npm install lucide-react
```

3. **Replace src/App.jsx** with the React dashboard code provided above

4. **Update vite.config.js**:

```javascript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://localhost:5000',
        changeOrigin: true
      }
    }
  }
})
```

5. **Update package.json** to include Tailwind:

```bash
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p
```

6. **Create tailwind.config.js**:

```javascript
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}
```

7. **Update src/index.css**:

```css
@tailwind base;
@tailwind components;
@tailwind utilities;
```

#### Option B: Manual Setup

If you prefer, you can copy the React code into an HTML artifact and run it standalone (but you'll lose hot-reload benefits).

---

## 🚦 Running the Dashboard

### Step 1: Start the Backend API

```bash
# In C:\IBKR_Algo_BOT\
python dashboard_api.py
```

You should see:
```
INFO: Starting Trading Bot Dashboard API Server...
INFO: Background status monitor started
INFO: Server running on http://localhost:5000
```

### Step 2: Start the Frontend

```bash
# In C:\IBKR_Algo_BOT\frontend\
npm run dev
```

You should see:
```
VITE v5.x.x  ready in xxx ms

➜  Local:   http://localhost:3000/
```

### Step 3: Open Dashboard

Open your browser and go to: **http://localhost:3000**

---

## 🎯 Using the Dashboard

### Main Features

#### 1. **System Control Cards**
- **MTF Swing Trading** - Control your LSTM-based swing trader
- **Warrior Momentum** - Control your gap scanner
- **IBKR Status** - Connection status and account info

Each card shows:
- Current status (Running/Stopped/Error)
- Active positions count
- Current P&L
- Start/Stop button

#### 2. **Total P&L Banner**
- Shows combined profit/loss from both systems
- Updates in real-time

#### 3. **Current Positions Table**
- All open positions from both bots
- Shows entry price, current price, P&L
- Duration of each position
- Which bot opened it

#### 4. **Recent Trades**
- Last 10 trades executed
- Buy/Sell indicators
- P&L for each trade
- Timestamp and strategy

#### 5. **Live Activity Log**
- Real-time system messages
- Color-coded by severity:
  - 🔵 Blue = Info
  - 🟢 Green = Success
  - 🟡 Yellow = Warning
  - 🔴 Red = Error
- Shows which bot generated each message

---

## 🔧 Integrating Your Existing Bots

### Connecting the MTF Bot

Edit `dashboard_api.py` in the `MTFBotController.start()` method:

```python
def start(self, symbols=None):
    # Uncomment and modify:
    from ibkr_live_trading_connector import MTFTradingBot
    
    self.bot_instance = MTFTradingBot()
    self.thread = threading.Thread(
        target=self.bot_instance.run,
        args=(symbols or ['AAPL', 'TSLA'],)
    )
    self.thread.daemon = True
    self.thread.start()
```

### Connecting the Warrior Scanner

Edit `dashboard_api.py` in the `WarriorBotController.start()` method:

```python
def start(self):
    from warrior_momentum_scanner import WarriorScanner
    
    self.scanner_instance = WarriorScanner()
    self.thread = threading.Thread(target=self.scanner_instance.scan_loop)
    self.thread.daemon = True
    self.thread.start()
```

### Updating Bot State from Your Bots

Add these callback functions to your existing bots to update the dashboard:

```python
# In your bot code (ibkr_live_trading_connector.py or warrior_momentum_scanner.py)

def update_dashboard_position(symbol, quantity, entry_price, current_price, pnl, strategy):
    """Send position update to dashboard"""
    try:
        import requests
        data = {
            'symbol': symbol,
            'qty': quantity,
            'entry': entry_price,
            'current': current_price,
            'pnl': pnl,
            'source': strategy
        }
        requests.post('http://localhost:5000/api/position/update', json=data)
    except:
        pass

def update_dashboard_trade(symbol, action, quantity, price, pnl, strategy):
    """Send trade update to dashboard"""
    try:
        import requests
        data = {
            'time': datetime.now().isoformat(),
            'symbol': symbol,
            'action': action,
            'qty': quantity,
            'price': price,
            'pnl': pnl,
            'strategy': strategy
        }
        requests.post('http://localhost:5000/api/trade/add', json=data)
    except:
        pass
```

---

## 📊 Data Flow

### How Updates Work

1. **Bot executes trade** → Calls `update_dashboard_trade()`
2. **Backend receives update** → Stores in database
3. **Backend broadcasts via WebSocket** → Frontend receives instantly
4. **Dashboard updates UI** → You see the change

### Polling vs WebSocket

- **Polling** (every 5 seconds): Status, positions, P&L
- **WebSocket** (instant): Activity logs, trade notifications

---

## 🎨 Customization

### Adding More Bots

To add a third bot (e.g., "Alpha Fusion"):

1. **Add controller in `dashboard_api.py`**:

```python
class AlphaFusionController:
    def __init__(self):
        self.running = False
        
    def start(self):
        # Your start logic
        bot_state.alpha_fusion = {
            'status': 'running',
            'active_positions': [],
            'pnl': 0.0
        }
        return {'success': True}
    
    def stop(self):
        # Your stop logic
        return {'success': True}
```

2. **Add to global state**:

```python
bot_state.alpha_fusion = {
    'status': 'stopped',
    'active_positions': [],
    'pnl': 0.0
}
```

3. **Add API endpoints**:

```python
@app.route('/api/alpha/start', methods=['POST'])
def start_alpha_fusion():
    return jsonify(alpha_controller.start())

@app.route('/api/alpha/stop', methods=['POST'])
def stop_alpha_fusion():
    return jsonify(alpha_controller.stop())
```

4. **Add card to React UI** - Copy existing SystemCard component

### Changing Colors

Edit the React component's Tailwind classes:

```javascript
// Change from blue/purple to green/orange
className="bg-gradient-to-r from-green-600 to-orange-600"
```

### Adding Notifications

Install and use browser notifications:

```javascript
// In your React component
useEffect(() => {
  if (Notification.permission === "default") {
    Notification.requestPermission();
  }
}, []);

// When trade completes
if (trade.pnl > 100) {
  new Notification("Big Win!", {
    body: `${trade.symbol}: +${trade.pnl}`,
    icon: "/trophy.png"
  });
}
```

---

## 🔒 Security Considerations

### For Development (Current Setup)
✅ Running on localhost only
✅ No external access
✅ Safe for testing

### For Production (If Deploying)

**⚠️ IMPORTANT: Do NOT expose to internet without:**

1. **Authentication** - Add login system
2. **HTTPS** - Use SSL certificates
3. **API Keys** - Protect endpoints
4. **Rate Limiting** - Prevent abuse
5. **Firewall Rules** - Restrict access

Example auth middleware:

```python
from functools import wraps
from flask import request, jsonify

def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if token != 'your-secret-token':
            return jsonify({'error': 'Unauthorized'}), 401
        return f(*args, **kwargs)
    return decorated

@app.route('/api/mtf/start', methods=['POST'])
@require_auth
def start_mtf_bot():
    # Protected endpoint
    pass
```

---

## 🐛 Troubleshooting

### Problem: Backend won't start

**Error**: `ImportError: No module named 'flask'`

**Solution**:
```bash
pip install flask flask-cors flask-socketio
```

### Problem: Frontend shows "Cannot connect to API"

**Checklist**:
1. Is backend running? Check `http://localhost:5000/api/health`
2. Check browser console for CORS errors
3. Verify proxy in `vite.config.js`
4. Make sure ports 3000 and 5000 are not blocked

### Problem: WebSocket not connecting

**Solution**: Check that `flask-socketio` is installed:
```bash
pip install python-socketio flask-socketio
```

### Problem: Database errors

**Solution**: Delete and recreate database:
```bash
# Windows
del dashboard_data\trading_bot.db

# Then restart backend - it will recreate
python dashboard_api.py
```

### Problem: Real-time updates not working

**Check**:
1. Open browser DevTools → Network tab → WS (WebSocket)
2. Should see connection to `ws://localhost:5000/socket.io/`
3. If not connected, restart both backend and frontend

---

## 📈 Performance Tips

### For Faster Updates

Reduce polling interval in React:

```javascript
// Change from 5000ms to 2000ms (2 seconds)
const interval = setInterval(fetchAllData, 2000);
```

### For Less CPU Usage

Increase polling interval:

```javascript
// Change to 10 seconds
const interval = setInterval(fetchAllData, 10000);
```

### Database Optimization

Add indexes for faster queries:

```python
# In database.py init_database()
cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)')
cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp)')
```

---

## 🎯 Next Steps - Phase 2 Features

Once basic dashboard is working, add:

### 1. Watchlist Manager
- Drag-and-drop symbols
- Multiple watchlists
- Import/export CSV
- Auto-add from Warrior scanner

### 2. Training Interface
- Select symbols for training
- Configure LSTM parameters
- View training progress
- Load/save models

### 3. Analytics Page
- P&L charts over time
- Win rate by strategy
- Best/worst performers
- Sharpe ratio tracking

### 4. Settings Page
- Configure risk limits
- Set position sizes
- Daily loss limits
- Trading hours

---

## 💾 Backup Your Data

### Manual Backup

```bash
# Backup database
copy dashboard_data\trading_bot.db dashboard_data\backup_YYYYMMDD.db

# Backup configuration
copy dashboard_api.py dashboard_api.py.backup
```

### Automated Backup Script

Create `backup_dashboard.py`:

```python
import shutil
from datetime import datetime
from pathlib import Path

def backup_dashboard():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_dir = Path(f'backups/dashboard_{timestamp}')
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    # Backup database
    shutil.copy2('dashboard_data/trading_bot.db', 
                 backup_dir / 'trading_bot.db')
    
    # Backup code
    for file in ['dashboard_api.py', 'database.py']:
        shutil.copy2(file, backup_dir / file)
    
    print(f'✅ Backup created: {backup_dir}')

if __name__ == '__main__':
    backup_dashboard()
```

Run daily:
```bash
python backup_dashboard.py
```

---

## 📞 Support & Resources

### Getting Help

1. **Check logs** - Look at backend console for errors
2. **Browser DevTools** - Check console and network tabs
3. **Test endpoints** - Use `curl` or Postman
4. **Database queries** - Use DB Browser for SQLite

### Useful Commands

```bash
# Test backend health
curl http://localhost:5000/api/health

# Test getting status
curl http://localhost:5000/api/status

# Start MTF bot via API
curl -X POST http://localhost:5000/api/mtf/start
```

### Documentation Links

- Flask: https://flask.palletsprojects.com/
- React: https://react.dev/
- Vite: https://vitejs.dev/
- Tailwind CSS: https://tailwindcss.com/
- Socket.IO: https://socket.io/

---

## ✅ Quick Start Checklist

- [ ] Python packages installed (`flask`, `flask-cors`, `flask-socketio`)
- [ ] Node.js installed (v18+)
- [ ] Frontend directory created with React app
- [ ] `dashboard_api.py` saved in project root
- [ ] `database.py` saved in project root
- [ ] Backend started (`python dashboard_api.py`)
- [ ] Frontend started (`npm run dev`)
- [ ] Dashboard accessible at http://localhost:3000
- [ ] API responding at http://localhost:5000/api/health
- [ ] WebSocket connected (check browser DevTools)

---

## 🎉 You're Ready!

Your trading dashboard is now set up! You should see:

✅ Clean, professional UI  
✅ Real-time position tracking  
✅ Live activity logs  
✅ Start/Stop controls for each bot  
✅ P&L monitoring  

**Next**: Integrate your existing MTF and Warrior bots, then start trading!

---

**Questions?** Save this guide and reference it as you build out more features!