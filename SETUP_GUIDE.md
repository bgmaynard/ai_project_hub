# 🤖 IBKR Trading Bot - Complete Setup Guide

> **Version 2.1** | Fixed ConnectionRefusedError(22) | Enhanced Trading Capabilities

A comprehensive Interactive Brokers (IBKR) trading bot with a modern web interface, real-time market data, and automated trading capabilities. This version specifically addresses connection issues and provides a robust trading platform.

## 🚀 Quick Start

### 1. Prerequisites
- **Python 3.8+** installed
- **TWS (Trader Workstation)** or **IB Gateway** running
- **IBKR Paper Trading Account** (recommended for testing)

### 2. Download & Setup
```bash
# 1. Download the files to your project directory
# 2. Open PowerShell as Administrator
# 3. Navigate to your project directory
cd C:\path\to\your\ibkr-trading-bot

# 4. Run the bootstrap script
powershell -ExecutionPolicy Bypass -File bootstrap.ps1
```

### 3. Configure TWS/IB Gateway
1. **Start TWS or IB Gateway**
2. **Enable API access:**
   - Go to: **Global Configuration → API → Settings**
   - Check: **"Enable ActiveX and Socket Clients"**
   - Socket Port: **7497** (Paper Trading) or **7496** (Live Trading)
   - Add **127.0.0.1** to Trusted IPs
3. **Restart TWS** after changes

### 4. Configure Your Settings
Edit the `.env` file that was created:
```env
# CRITICAL: Change this API key!
LOCAL_API_KEY=Your_Super_Strong_API_Key_123

# TWS Connection (7497 = Paper, 7496 = Live)
TWS_HOST=127.0.0.1
TWS_PORT=7497
TWS_CLIENT_ID=6001

# API Server
API_HOST=127.0.0.1
API_PORT=9101
```

### 5. Access the Interface
- **Status Dashboard:** http://127.0.0.1:9101/ui/status.html
- **Trading Interface:** http://127.0.0.1:9101/ui/trading.html
- **API Documentation:** http://127.0.0.1:9101

## 📁 File Structure

```
ibkr-trading-bot/
├── ib_adapter.py          # Enhanced IBKR connection adapter
├── dashboard_api.py       # FastAPI trading server
├── status.html           # Real-time status dashboard
├── trading.html          # Trading interface
├── bootstrap.ps1         # Windows setup script
├── requirements.txt      # Python dependencies
├── .env.example         # Configuration template
└── README.md           # This file
```

## 🔧 Configuration Options

### Environment Variables (.env)
| Variable | Description | Default | Notes |
|----------|-------------|---------|-------|
| `LOCAL_API_KEY` | API security key | ⚠️ **REQUIRED** | Change the default! |
| `TWS_HOST` | TWS/Gateway host | `127.0.0.1` | Usually localhost |
| `TWS_PORT` | TWS/Gateway port | `7497` | 7497=Paper, 7496=Live |
| `TWS_CLIENT_ID` | Unique client ID | `6001` | Must be unique per connection |
| `API_HOST` | API server host | `127.0.0.1` | Web interface host |
| `API_PORT` | API server port | `9101` | Web interface port |
| `IB_HEARTBEAT_SEC` | Connection check interval | `3.0` | How often to check connection |
| `IB_CONNECT_TIMEOUT_SEC` | Connection timeout | `15.0` | Max time to wait for connection |

### TWS Port Reference
| TWS Type | Paper Trading | Live Trading |
|----------|---------------|--------------|
| **TWS** | 7497 | 7496 |
| **IB Gateway** | 4001 | 4000 |

## 🚨 Troubleshooting Connection Issues

### ❌ ConnectionRefusedError(22) - Solutions

1. **TWS Not Running**
   ```
   ✅ Solution: Start TWS or IB Gateway
   ```

2. **API Not Enabled**
   ```
   ✅ Solution: 
   - Global Configuration → API → Settings
   - Check "Enable ActiveX and Socket Clients"
   - Restart TWS
   ```

3. **Wrong Port**
   ```
   ✅ Solution: Verify .env TWS_PORT matches TWS Socket Port setting
   ```

4. **Client ID Conflict**
   ```
   ✅ Solution: Change TWS_CLIENT_ID in .env (each connection needs unique ID)
   ```

5. **Firewall/Security**
   ```
   ✅ Solution: 
   - Add 127.0.0.1 to TWS Trusted IPs
   - Check Windows Firewall
   - Disable antivirus temporarily for testing
   ```

### 🔍 Diagnostic Tools

**Built-in Diagnostics:**
- Status Dashboard: http://127.0.0.1:9101/ui/status.html
- Debug Endpoint: http://127.0.0.1:9101/api/debug/ibkr
- Connection Test: Run `bootstrap.ps1` (includes pre-flight checks)

**Manual Connection Test:**
```powershell
# Test if TWS is listening
Test-NetConnection -ComputerName 127.0.0.1 -Port 7497
```

## 📊 API Endpoints

### Public Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/status` | GET | Connection status |
| `/api/debug/ibkr` | GET | Diagnostic information |
| `/api/tws/ping` | GET | Simple TWS test |
| `/api/reconnect` | POST | Manual reconnection |

### Protected Endpoints (Require X-API-Key header)
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/account` | GET | Account information |
| `/api/positions` | GET | Current positions |
| `/api/orders` | GET | Open orders |
| `/api/order/preview` | POST | Preview order |
| `/api/order/place` | POST | Place order |
| `/api/order/{id}` | DELETE | Cancel order |
| `/api/market/data` | POST | Market data |

### Example API Usage
```python
import requests

headers = {
    'X-API-Key': 'Your_API_Key_From_Env',
    'Content-Type': 'application/json'
}

# Get account info
response = requests.get('http://127.0.0.1:9101/api/account', headers=headers)
print(response.json())

# Place market order
order_data = {
    'symbol': 'AAPL',
    'quantity': 100,  # Positive = Buy, Negative = Sell
    'order_type': 'MKT',
    'exchange': 'SMART'
}

response = requests.post(
    'http://127.0.0.1:9101/api/order/place', 
    headers=headers, 
    json=order_data
)
print(response.json())
```

## 🛡️ Security Best Practices

1. **Strong API Key**
   ```
   ❌ LOCAL_API_KEY=Your_Super_Strong_API_Key_123
   ✅ LOCAL_API_KEY=MyComplex_API_Key_2024_987654321
   ```

2. **Paper Trading First**
   ```
   ✅ Always test with Paper Trading (port 7497) before Live Trading
   ```

3. **Network Security**
   ```
   ✅ Only bind to localhost (127.0.0.1) unless needed
   ✅ Use firewall rules to restrict access
   ```

4. **Environment File**
   ```
   ⚠️  Never commit .env files to version control
   ✅ Keep .env file secure and backed up
   ```

## 🔄 Advanced Usage

### Custom Trading Strategies
```python
# Example: Custom strategy integration
async def my_trading_strategy():
    # Get market data
    market_data = await ib_adapter.get_market_data('AAPL')
    
    # Your analysis logic here
    if market_data['last'] < some_threshold:
        # Place buy order
        result = await ib_adapter.place_order('AAPL', 100, 'MKT')
        print(f"Order placed: {result}")
```

### Automated Monitoring
```python
# Example: Position monitoring
async def monitor_positions():
    positions = await ib_adapter.get_positions()
    for pos in positions['positions']:
        if pos['unrealized_pnl'] < -1000:  # $1000 loss
            print(f"ALERT: Large loss in {pos['symbol']}")
```

### WebSocket Integration (Future)
```python
# Planned: Real-time data streaming
async def stream_market_data():
    # Future feature for real-time price feeds
    pass
```

## 🆘 Support & Troubleshooting

### Common Issues

**"ib_adapter not found"**
```
✅ Ensure ib_adapter.py is in the same directory as dashboard_api.py
```

**"API key required"**
```
✅ Set LOCAL_API_KEY in .env file
✅ Use X-API-Key header in requests
```

**"IBKR not connected"**
```
✅ Check TWS is running and API enabled
✅ Visit /api/debug/ibkr for diagnostics
✅ Try manual reconnect: POST /api/reconnect
```

**UI not loading**
```
✅ Check API server is running on correct port
✅ Visit http://127.0.0.1:9101 directly
```

### Getting Help

1. **Check Status Dashboard** first: http://127.0.0.1:9101/ui/status.html
2. **Review Debug Info**: http://127.0.0.1:9101/api/debug/ibkr
3. **Check TWS Logs** in TWS application
4. **Review Python Console** output for errors

### Performance Tips

1. **Connection Stability**
   - Keep TWS running continuously
   - Use dedicated TWS instance for API
   - Monitor connection status regularly

2. **Order Management**
   - Preview orders before placing
   - Monitor open orders regularly
   - Set appropriate timeouts

3. **Resource Usage**
   - Close unused market data subscriptions
   - Limit concurrent API calls
   - Monitor memory usage

## 📈 Features

### ✅ Current Features
- [x] Robust IBKR connection with auto-reconnect
- [x] Real-time status monitoring
- [x] Market data retrieval
- [x] Order placement (Market & Limit orders)
- [x] Position and account monitoring
- [x] Order management (cancel, modify)
- [x] Web-based trading interface
- [x] Comprehensive error handling
- [x] Connection diagnostics

### 🚧 Planned Features
- [ ] Real-time data streaming via WebSocket
- [ ] Advanced order types (Stop, Stop-Limit, etc.)
- [ ] Trading strategy backtesting
- [ ] Risk management rules
- [ ] Trade logging and reporting
- [ ] Multi-account support
- [ ] Mobile-responsive interface
- [ ] AI-powered market analysis

## 📜 License

This project is for educational and development purposes. Use at your own risk with paper trading accounts first. Always test thoroughly before using with live trading accounts.

## ⚠️ Disclaimer

**Important:** This software is provided for educational purposes. Trading involves significant risk of loss. Always:
- Start with paper trading accounts
- Test thoroughly before live trading
- Understand the risks involved
- Comply with your local regulations
- Monitor your positions actively

The authors are not responsible for any trading losses or damages resulting from the use of this software.

---

**Happy Trading! 🚀📊**

*For additional support, check the status dashboard and debug endpoints first.*
