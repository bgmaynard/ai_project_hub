import { useEffect, useState } from 'react'
import { usePortfolioStore } from '../stores/portfolioStore'
import api from '../services/api'

interface AccountInfo {
  accountNumber: string
  accountType: string
  selected?: boolean
}

export default function Account() {
  const { account, setAccount, positions } = usePortfolioStore()
  const [accounts, setAccounts] = useState<AccountInfo[]>([])
  const [isChanging, setIsChanging] = useState(false)

  // Fetch list of all accounts
  useEffect(() => {
    const fetchAccounts = async () => {
      try {
        const data = await api.getAccounts()
        if (data.accounts) {
          setAccounts(data.accounts)
        }
      } catch (err) {
        console.error('Failed to load accounts list:', err)
      }
    }
    fetchAccounts()
    // Refresh accounts list every 30s
    const interval = setInterval(fetchAccounts, 30000)
    return () => clearInterval(interval)
  }, [])

  // Fetch current account details
  useEffect(() => {
    const fetchAccount = async () => {
      try {
        const data = await api.getAccount() as any
        // Schwab returns data at both top level and in summary
        const s = data?.summary || {}
        const d = data || {}

        setAccount({
          accountId: s.account_id || d.account_number || d.accountNumber || '',
          accountType: s.account_type || d.type || d.accountType || 'Unknown',
          netLiquidation: s.net_liquidation || d.market_value || d.equity || d.netLiquidation || 0,
          buyingPower: s.buying_power || d.buying_power || d.buyingPower || 0,
          cashBalance: s.total_cash || d.cash || d.totalCash || 0,
          settledCash: d.cash_available_for_trading || d.cashAvailableForTrading ||
                       s.settled_cash || d.available_funds || s.total_cash || d.cash || 0,
          marginUsed: d.margin_balance || s.maintenance_margin ||
                      ((s.net_liquidation || d.market_value || 0) - (s.total_cash || d.cash || 0)),
          dayPnl: d.daily_pl || d.dailyPnL || s.daily_pnl || 0,
          totalPnl: d.unrealized_pnl || d.unrealizedPnL || s.unrealized_pnl || 0,
        })
      } catch (err) {
        console.error('Failed to load account:', err)
      }
    }

    fetchAccount()
    const interval = setInterval(fetchAccount, 5000) // Faster refresh
    return () => clearInterval(interval)
  }, [setAccount])

  const handleAccountChange = async (accountNumber: string) => {
    if (accountNumber === account?.accountId) return
    setIsChanging(true)
    try {
      const result = await api.selectAccount(accountNumber)
      if (result.success) {
        // Refresh account data after switch
        const data = await api.getAccount() as any
        const s = data?.summary || {}
        const d = data || {}
        setAccount({
          accountId: s.account_id || d.account_number || d.accountNumber || '',
          accountType: s.account_type || d.type || d.accountType || 'Unknown',
          netLiquidation: s.net_liquidation || d.market_value || d.equity || d.netLiquidation || 0,
          buyingPower: s.buying_power || d.buying_power || d.buyingPower || 0,
          cashBalance: s.total_cash || d.cash || d.totalCash || 0,
          settledCash: d.cash_available_for_trading || d.cashAvailableForTrading ||
                       s.settled_cash || d.available_funds || s.total_cash || d.cash || 0,
          marginUsed: d.margin_balance || s.maintenance_margin ||
                      ((s.net_liquidation || d.market_value || 0) - (s.total_cash || d.cash || 0)),
          dayPnl: d.daily_pl || d.dailyPnL || s.daily_pnl || 0,
          totalPnl: d.unrealized_pnl || d.unrealizedPnL || s.unrealized_pnl || 0,
        })
        // Refresh accounts list to update selected state
        const accountsData = await api.getAccounts()
        if (accountsData.accounts) {
          setAccounts(accountsData.accounts)
        }
      }
    } catch (err) {
      console.error('Failed to switch account:', err)
    }
    setIsChanging(false)
  }

  const formatCurrency = (value: number | undefined) =>
    `$${(value ?? 0).toLocaleString('en-US', {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    })}`

  const formatCompact = (value: number | undefined) => {
    const v = value ?? 0
    if (Math.abs(v) >= 1000000) return `$${(v / 1000000).toFixed(1)}M`
    if (Math.abs(v) >= 1000) return `$${(v / 1000).toFixed(1)}K`
    return `$${v.toFixed(0)}`
  }

  const dayPnl = account?.dayPnl ?? account?.dayPnL ?? 0
  const accountType = (account?.accountType ?? '').toUpperCase()
  const isMarginAccount = accountType.includes('MARGIN')

  return (
    <div className="h-full flex flex-col bg-sterling-panel text-xs">
      {/* Header */}
      <div className="flex items-center justify-between px-2 py-0.5 bg-sterling-header border-b border-sterling-border">
        <span className="font-bold text-sterling-text text-xxs">ACCOUNT</span>
        <div className="flex items-center gap-2">
          <span className={`text-xxs font-bold px-1 rounded ${isMarginAccount ? 'bg-[#1e3a5f] text-accent-primary' : 'bg-[#134e4a] text-up'}`}>
            {isMarginAccount ? 'MARGIN' : accountType.includes('IRA') ? 'IRA' : 'CASH'}
          </span>
          {/* Account Selector Dropdown */}
          {accounts.length > 1 ? (
            <select
              value={account?.accountId || ''}
              onChange={(e) => handleAccountChange(e.target.value)}
              disabled={isChanging}
              className="bg-sterling-bg text-sterling-text text-xxs px-1 py-0.5 rounded border border-sterling-border cursor-pointer hover:bg-sterling-highlight disabled:opacity-50"
              title="Switch Account"
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
        </div>
      </div>

      {/* Compact Stats - 3 columns */}
      <div className="p-1.5 grid grid-cols-3 gap-1">
        {/* Row 1 */}
        <div className="text-center">
          <div className="text-[9px] text-sterling-muted">EQUITY</div>
          <div className="text-[11px] font-bold text-white">
            {formatCompact(account?.netLiquidation ?? account?.equity)}
          </div>
        </div>
        <div className="text-center">
          <div className="text-[9px] text-sterling-muted">BP</div>
          <div className="text-[11px] font-bold text-accent-primary">
            {formatCompact(account?.buyingPower)}
          </div>
        </div>
        <div className="text-center">
          <div className="text-[9px] text-sterling-muted">DAY P&L</div>
          <div className={`text-[11px] font-bold ${dayPnl >= 0 ? 'text-up' : 'text-down'}`}>
            {dayPnl >= 0 ? '+' : ''}{formatCurrency(dayPnl)}
          </div>
        </div>

        {/* Row 2 */}
        <div className="text-center">
          <div className="text-[9px] text-sterling-muted">CASH</div>
          <div className="text-[11px] font-bold text-sterling-text">
            {formatCompact(account?.cashBalance)}
          </div>
        </div>
        <div className="text-center">
          <div className="text-[9px] text-sterling-muted">
            {isMarginAccount ? 'MARGIN' : 'SETTLED'}
          </div>
          <div className={`text-[11px] font-bold ${isMarginAccount ? 'text-warning' : 'text-up'}`}>
            {isMarginAccount
              ? formatCompact(account?.marginUsed)
              : formatCompact(account?.settledCash ?? account?.cashBalance)
            }
          </div>
        </div>
        <div className="text-center">
          <div className="text-[9px] text-sterling-muted">POS</div>
          <div className="text-[11px] font-bold text-sterling-text">
            {positions.length}
          </div>
        </div>
      </div>
    </div>
  )
}
