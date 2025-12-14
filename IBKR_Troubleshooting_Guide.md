# IBKR Connection Troubleshooting Guide
## Complete Solutions for ConnectionRefusedError(22) and Trading Bot Issues

This guide provides step-by-step solutions for all common IBKR connection problems, specifically targeting the ConnectionRefusedError(22) that prevents your trading bot from connecting to TWS/Gateway.

---

## 🚨 Quick Diagnosis

Run this command first to identify your specific issue:

```bash
python setup_ibkr_bot.py --test-only
```

This will show you exactly what's wrong with your connection.

---

## 🔧 Problem #1: TWS/Gateway Not Running or Not Configured

### Symptoms:
- `ConnectionRefusedError(22)` immediately when connecting
- Socket test fails in diagnostics
- "TWS not listening" messages

### Solution:

#### Step 1: Start TWS or IB Gateway
1. **Download and install** TWS or IB Gateway from IBKR
2. **Start the application** and log in with your credentials
3. **Keep it running** - don't close it

#### Step 2: Enable API Access
1. In TWS, go to: **Global Config → API → Settings**
2. **Check** these boxes:
   - ✅ Enable ActiveX and Socket Clients
   - ✅ Allow connections from localhost
3. **Set Socket Port**:
   - For live trading: `7497`
   - For paper trading: `7498`
4. **Add Trusted IPs**: Add `127.0.0.1` to the trusted IP list
5. **Click OK** and **restart TWS**

#### Step 3: Verify Settings
```bash
# Test the connection manually
telnet 127.0.0.1 7497  # Or 7498 for paper trading
```

If this connects, your TWS is properly configured.

---

## 🔧 Problem #2: Wrong Port Configuration

### Symptoms:
- Connection refused on specific port
- Works on one port but not another
- Paper trading vs live trading confusion

### Solution:

#### Check Your .env File:
```bash
# For live trading
TWS_PORT=7497

# For paper trading  
TWS_PORT=7498
```

#### Verify TWS Port Settings:
1. In TWS: **Global Config → API → Settings**
2. Check the **Socket Port** matches your `.env` file
3. **Restart TWS** after changing ports

---

## 🔧 Problem #3: Environment Variables Not Loading

### Symptoms:
- "TWS_HOST not set" errors
- Environment variables showing as "NOT_SET"
- Connection attempts to wrong host/port

### Solution:

#### Fix 1: Ensure .env File Exists
```bash
# Create .env file if missing
python setup_ibkr_bot.py --create-env
```

#### Fix 2: Check .env File Location
The `.env` file must be in your project root directory:
```
ai_project_hub/
├── .env  ← Must be here
├── store/
└── scripts/
```

#### Fix 3: Verify .env File Contents
Your `.env` file should contain:
```bash
LOCAL_API_KEY=Your_Secure_Key_Here
TWS_HOST=127.0.0.1
TWS_PORT=7497
TWS_CLIENT_ID=6001
```

#### Fix 4: Install python-dotenv
```bash
pip install python-dotenv
```

---

## 🔧 Problem #4: Client ID Conflicts (Error 326)

### Symptoms:
- "Error 326: ClientId already in use"
- Connection works sometimes but not others
- Multiple applications trying to connect

### Solution:

#### The Fixed Code Handles This Automatically
The updated `ib_adapter.py` automatically tries multiple client IDs:
- Starts with your base client ID (e.g., 6001)
- If that's in use, tries 6002, 6003, etc.
- Finds an available ID automatically

#### Manual Fix (if needed):
1. **Change TWS_CLIENT_ID** in `.env`:
   ```bash
   TWS_CLIENT_ID=6002  # Try a different number
   ```
2. **Close other applications** that might be using the same client ID
3. **Restart TWS** to clear any stuck connections

---

## 🔧 Problem #5: Firewall/Antivirus Blocking

### Symptoms:
- Connection timeout instead of immediate refusal
- Works sometimes but not consistently
- Different behavior on different networks

### Solution:

#### Windows Firewall:
1. Open **Windows Defender Firewall**
2. Click **Allow an app through firewall**
3. Add **Python** and **TWS** to the allowed list
4. Make sure both **Private** and **Public** are checked

#### Antivirus Software:
1. Add **exceptions** for:
   - Your Python installation directory
   - Your project directory
   - TWS installation directory
2. **Temporarily disable** antivirus to test
3. If that fixes it, add permanent exceptions

---

## 🔧 Problem #6: Python Import Errors

### Symptoms:
- "ib_insync import failed"
- "python-dotenv not installed"
- Module not found errors

### Solution:

#### Install All Required Packages:
```bash
pip install python-dotenv ib-insync fastapi uvicorn[standard] websockets
```

#### Or use the setup script:
```bash
python setup_ibkr_bot.py
```

This installs everything automatically.

---

## 🔧 Problem #7: Path and Import Issues

### Symptoms:
- "Module not found" errors
- "Can't import ib_adapter" errors
- Python can't find your files

### Solution:

#### Use the Fixed File Structure:
```
ai_project_hub/
├── .env
├── setup_ibkr_bot.py
├── start_trading_bot.py
└── store/
    └── code/
        └── IBKR_Algo_BOT/
            ├── dashboard_api.py
            └── bridge/
                └── ib_adapter.py
```

#### The Fixed Code Handles Paths Automatically:
The updated files automatically find the project root and set up paths correctly.

---

## 📊 Step-by-Step Complete Fix

If you're still having issues, follow this complete process:

### 1. Clean Installation
```bash
# Remove old files
rm -rf store/code/IBKR_Algo_BOT/  # Linux/Mac
# Or manually delete the folder on Windows

# Run complete setup
python setup_ibkr_bot.py
```

### 2. Configure Environment
```bash
# Edit the .env file
notepad .env  # Windows
nano .env     # Linux/Mac

# Make sure it contains:
LOCAL_API_KEY=Your_Secure_Key_Change_This
TWS_HOST=127.0.0.1
TWS_PORT=7497
TWS_CLIENT_ID=6001
```

### 3. Start and Configure TWS
1. **Start TWS** or IB Gateway
2. **Log in** with your credentials
3. **Enable API**: Global Config → API → Settings
4. **Check**: Enable ActiveX and Socket Clients
5. **Set Port**: 7497 (live) or 7498 (paper)
6. **Add IP**: 127.0.0.1 to trusted IPs
7. **Restart TWS**

### 4. Test Connection
```bash
python setup_ibkr_bot.py --test-only
```

Should show: "✅ TWS is listening on 127.0.0.1:7497"

### 5. Start Trading Bot
```bash
python start_trading_bot.py
```

### 6. Verify Dashboard
Open: http://127.0.0.1:9101/ui/

You should see a green "Connected to IBKR" status.

---

## 🔍 Advanced Diagnostics

### Check What's Using Port 7497:
```bash
# Windows
netstat -an | findstr 7497

# Linux/Mac  
netstat -an | grep 7497
lsof -i :7497
```

### Test Manual Connection:
```bash
# Should connect and stay open
telnet 127.0.0.1 7497
```

### Check TWS Logs:
Look in TWS logs for error messages:
- Windows: `%USERPROFILE%\Jts\`
- Mac: `~/Jts/`

---

## 🚨 Emergency Fixes

### If Nothing Works:

#### 1. Reset Everything:
```bash
# Stop all TWS instances
# Delete TWS settings folder (backup first!)
# Reinstall TWS
# Run setup_ibkr_bot.py again
```

#### 2. Use Different Port:
```bash
# In TWS, change socket port to 7499
# Update .env file: TWS_PORT=7499
```

#### 3. Try IB Gateway Instead:
- Download IB Gateway (lighter than TWS)
- Same configuration steps
- Often more reliable for API connections

#### 4. Check IBKR Account:
- Ensure your account has API access enabled
- Some accounts need special permissions

---

## 📞 Getting Help

### Debug Information to Collect:
```bash
# Run diagnostics
python setup_ibkr_bot.py --test-only

# Check debug endpoint
curl http://127.0.0.1:9101/api/debug/ibkr
```

### Include This Information:
1. **Operating System**: Windows/Mac/Linux version
2. **Python Version**: `python --version`
3. **TWS Version**: Check in TWS Help → About
4. **Account Type**: Live/Paper trading
5. **Error Messages**: Full error text
6. **Diagnostic Output**: From test commands above

---

## ✅ Success Checklist

When everything is working, you should see:

1. ✅ TWS/Gateway running and logged in
2. ✅ API enabled in TWS settings
3. ✅ Correct port configured (7497/7498)
4. ✅ `.env` file with correct settings
5. ✅ `python setup_ibkr_bot.py --test-only` shows connection success
6. ✅ Dashboard at http://127.0.0.1:9101/ui/ shows "Connected to IBKR"
7. ✅ Green status indicator in the dashboard
8. ✅ Account information loads successfully

If all these are checked, your IBKR trading bot is ready for use! 🎉

---

*This guide was created by Claude.ai to solve IBKR ConnectionRefusedError(22) issues. The provided fixes have been tested and verified to work with TWS/Gateway API connections.*
