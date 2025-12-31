# Session Update - December 31, 2024

## Summary
Fixed UI issues where scanner panel showed no data and account selector only displayed one account.

---

## Changes Made

### 1. Scanner Panel - Finviz Fallback

**Problem:** Scanner panel was empty all day because:
- UI wasn't rebuilt after code changes
- Schwab movers API returns empty on holidays/market closed
- No fallback data source

**Solution:** Added Finviz as automatic fallback when Schwab is empty.

**Files Modified:**

#### `ui/trading/src/services/api.ts`
Added Finviz API methods:
```typescript
async getFinvizGainers(minChange = 5, maxPrice = 20) {
  return this.fetch(`/scanner/finviz/gainers?min_change=${minChange}&max_price=${maxPrice}`)
}

async getFinvizLowFloat(maxFloat = 20) {
  return this.fetch(`/scanner/finviz/low-float?max_float=${maxFloat}`)
}

async getFinvizBreakouts() {
  return this.fetch('/scanner/finviz/breakouts')
}

async getFinvizAll() {
  return this.fetch('/scanner/finviz/scan-all')
}

async syncFinvizToWatchlist(minChange = 5, maxCount = 10) {
  return this.fetch(`/scanner/finviz/sync-to-watchlist?min_change=${minChange}&max_count=${maxCount}`, {
    method: 'POST'
  })
}
```

#### `ui/trading/src/components/ScannerPanel.tsx`
Updated `fetchScannerData` to fallback to Finviz:
```typescript
// Try Schwab scanners first
switch (activeScanner) {
  case 'hod': data = await api.getHODScanner(); break;
  case 'gappers': data = await api.getGappersScanner(); break;
  case 'gainers': data = await api.getGainersScanner(); break;
  case 'premarket': data = await api.getPreMarketScanner(); break;
}

// Check if Schwab returned empty - fall back to Finviz
const schwabItems = data?.candidates || data?.results || [];
if (schwabItems.length === 0) {
  useFinviz = true;
  switch (activeScanner) {
    case 'gainers':
    case 'gappers':
      data = await api.getFinvizGainers(5, 50);
      break;
    case 'hod':
      data = await api.getFinvizBreakouts();
      break;
    case 'premarket':
      data = await api.getFinvizLowFloat(20);
      break;
  }
}
```

**Result:** Scanner panel now shows 77+ gainers from Finviz when Schwab is empty.

---

### 2. Account Selector Dropdown

**Problem:** Only one Schwab account visible (IRA). No way to switch accounts.

**Solution:** Added multi-account support with dropdown selector.

**Files Modified:**

#### `morpheus_trading_api.py`
Fixed `/api/accounts` to return ALL accounts:
```python
@app.get("/api/accounts")
async def get_accounts_list():
    """Get list of ALL available Schwab accounts"""
    accounts = []
    selected_account = None

    if HAS_SCHWAB_TRADING:
        schwab = get_schwab_trading()
        if schwab:
            all_accounts = schwab.get_accounts()
            selected_account = schwab.get_selected_account()

            for acc in all_accounts:
                acc_num = acc.get("account_number")
                accounts.append({
                    "accountNumber": acc_num,
                    "accountType": acc.get("type", "UNKNOWN"),
                    "selected": acc_num == selected_account
                })

    return {"accounts": accounts, "selected": selected_account}
```

Added new endpoint to select account:
```python
@app.post("/api/accounts/select/{account_number}")
async def select_account(account_number: str):
    """Select a Schwab account for trading"""
    schwab = get_schwab_trading()
    success = schwab.select_account(account_number)
    if success:
        account_info = schwab.get_account_info()
        return {
            "success": True,
            "selected": account_number,
            "account_type": account_info.get("type")
        }
    return {"success": False, "error": f"Account {account_number} not found"}
```

#### `schwab_trading.py`
Updated `get_accounts()` to fetch account types:
```python
def get_accounts(self) -> List[Dict]:
    """Get list of available accounts with their types"""
    if not self._accounts:
        self._load_accounts()

    result = []
    for acc in self._accounts:
        acc_num = acc.get('accountNumber')
        acc_hash = acc.get('hashValue')
        acc_type = "UNKNOWN"

        # Fetch account type from Schwab API
        try:
            data = _make_trading_request("GET", f"/accounts/{acc_hash}")
            if data:
                securities_account = data.get('securitiesAccount', {})
                acc_type = securities_account.get('type', 'UNKNOWN')
        except Exception:
            pass

        result.append({
            "account_number": acc_num,
            "type": acc_type,
            "selected": acc_num == self._selected_account
        })

    return result
```

#### `ui/trading/src/services/api.ts`
Added account selection method:
```typescript
async getAccounts(): Promise<{
  accounts: Array<{ accountNumber: string; accountType: string; selected?: boolean }>;
  selected?: string
}> {
  return this.fetch('/accounts')
}

async selectAccount(accountNumber: string): Promise<{ success: boolean; selected?: string; error?: string }> {
  return this.fetch(`/accounts/select/${accountNumber}`, { method: 'POST' })
}
```

#### `ui/trading/src/components/Account.tsx`
Added account dropdown selector:
```typescript
const [accounts, setAccounts] = useState<AccountInfo[]>([])
const [isChanging, setIsChanging] = useState(false)

// Fetch list of all accounts
useEffect(() => {
  const fetchAccounts = async () => {
    const data = await api.getAccounts()
    if (data.accounts) setAccounts(data.accounts)
  }
  fetchAccounts()
  const interval = setInterval(fetchAccounts, 30000)
  return () => clearInterval(interval)
}, [])

const handleAccountChange = async (accountNumber: string) => {
  if (accountNumber === account?.accountId) return
  setIsChanging(true)
  const result = await api.selectAccount(accountNumber)
  if (result.success) {
    // Refresh account data after switch
    const data = await api.getAccount()
    setAccount({...})
  }
  setIsChanging(false)
}

// In render - dropdown when multiple accounts:
{accounts.length > 1 ? (
  <select
    value={account?.accountId || ''}
    onChange={(e) => handleAccountChange(e.target.value)}
    disabled={isChanging}
    className="bg-sterling-bg text-sterling-text text-xxs px-1 py-0.5 rounded border"
  >
    {accounts.map((acc) => (
      <option key={acc.accountNumber} value={acc.accountNumber}>
        {acc.accountNumber.slice(-4)} ({acc.accountType})
      </option>
    ))}
  </select>
) : (
  <span className="text-sterling-muted text-xxs">
    {account?.accountId ? `...${account.accountId.slice(-4)}` : '--'}
  </span>
)}
```

**Result:** Account panel now shows dropdown with both accounts:
- `...7852 (CASH)` - Main trading account
- `...3923 (CASH)` - Second account (IRA)

---

## API Endpoints

### New Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/accounts/select/{account_number}` | Switch active trading account |

### Updated Endpoints
| Method | Endpoint | Change |
|--------|----------|--------|
| GET | `/api/accounts` | Now returns ALL accounts with `selected` flag |

### Finviz Endpoints (Already Existed)
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/scanner/finviz/gainers` | Top percentage gainers |
| GET | `/api/scanner/finviz/low-float` | Low float movers |
| GET | `/api/scanner/finviz/breakouts` | High volume breakouts |
| GET | `/api/scanner/finviz/scan-all` | All Finviz scans combined |
| POST | `/api/scanner/finviz/sync-to-watchlist` | Add Finviz picks to watchlist |

---

## Data Flow

### Scanner Panel Data Flow
```
User opens Scanner Panel
    ↓
fetchScannerData() called
    ↓
Try Schwab scanner API first
    ↓
If Schwab empty → Fallback to Finviz
    ↓
Map results to ScannerResult format
    ↓
Display in table with sortable columns
```

### Account Switch Flow
```
User selects account from dropdown
    ↓
POST /api/accounts/select/{account_number}
    ↓
Backend: schwab.select_account(account_number)
    ↓
Frontend: Refresh account data + accounts list
    ↓
UI updates with new account info
```

---

## Testing

```bash
# Test scanner with Finviz data
curl http://localhost:9100/api/scanner/finviz/gainers?min_change=5&max_price=50

# Test accounts list
curl http://localhost:9100/api/accounts

# Test account switch
curl -X POST http://localhost:9100/api/accounts/select/70083923

# Verify switch worked
curl http://localhost:9100/api/account
```

---

## Notes

1. **Schwab Account Types**: Both accounts return "CASH" from Schwab API. IRA status may be in a different field not exposed by the API.

2. **Finviz Rate Limits**: ~1 request/second recommended. Data is 15-20 minutes delayed (free tier).

3. **UI Rebuild Required**: Any TypeScript/React changes require `npm run build` in `ui/trading/` directory.

4. **Server Restart Required**: Any Python changes require server restart to take effect.
