import { useEffect, useState, useCallback } from 'react'
import { usePortfolioStore, Position } from '../stores/portfolioStore'
import { useSymbolStore } from '../stores/symbolStore'
import api from '../services/api'

export default function Positions() {
  const { positions, setPositions, setLoading } = usePortfolioStore()
  const { setActiveSymbol } = useSymbolStore()
  const [aiMonitored, setAiMonitored] = useState<Set<string>>(new Set())
  const [isRefreshing, setIsRefreshing] = useState(false)
  const [lastSync, setLastSync] = useState<Date | null>(null)

  const fetchPositions = useCallback(async (manual = false) => {
    if (manual) setIsRefreshing(true)
    setLoading(true)
    try {
      // Try dedicated positions endpoint first
      let positionsData: any[] = []

      try {
        const posResponse = await api.getPositions() as any
        if (posResponse?.positions && Array.isArray(posResponse.positions)) {
          positionsData = posResponse.positions
        }
      } catch {
        // Fallback to account endpoint
      }

      // If no positions from dedicated endpoint, try account endpoint
      if (positionsData.length === 0) {
        const accountData = await api.getAccount() as any
        if (accountData?.positions && Array.isArray(accountData.positions)) {
          positionsData = accountData.positions
        }
      }

      const mapped: Position[] = positionsData.map((p: any) => {
        const qty = p.quantity || p.longQuantity || p.shortQuantity || 0
        const avgCost = p.avg_cost || p.averagePrice || p.average_price || 0
        const marketValue = p.market_value || p.marketValue || p.currentValue || 0
        const currentPrice = p.current_price || p.lastPrice || (qty !== 0 ? marketValue / Math.abs(qty) : 0)
        const unrealizedPnl = p.unrealized_pnl || p.unrealizedPnL || p.currentDayProfitLoss || 0

        return {
          symbol: p.symbol || p.instrument?.symbol || '',
          quantity: qty,
          avgCost: avgCost,
          currentPrice: currentPrice,
          marketValue: marketValue,
          unrealizedPnl: unrealizedPnl,
          unrealizedPnlPercent: avgCost && qty
            ? ((currentPrice - avgCost) / avgCost) * 100
            : 0,
        }
      })

      setPositions(mapped)
      setLastSync(new Date())
    } catch (err) {
      console.error('Failed to load positions:', err)
    }
    setLoading(false)
    if (manual) setIsRefreshing(false)
  }, [setPositions, setLoading])

  useEffect(() => {
    fetchPositions()
    const interval = setInterval(() => fetchPositions(false), 3000)
    return () => clearInterval(interval)
  }, [fetchPositions])

  const toggleAiTakeover = async (symbol: string, enabled: boolean) => {
    try {
      if (enabled) {
        await fetch(`/api/scalp/takeover/${symbol}`, { method: 'POST' })
        setAiMonitored((prev) => new Set(prev).add(symbol))
      } else {
        await fetch(`/api/scalp/release/${symbol}`, { method: 'POST' })
        setAiMonitored((prev) => {
          const next = new Set(prev)
          next.delete(symbol)
          return next
        })
      }
    } catch (err) {
      console.error('AI takeover toggle failed:', err)
    }
  }

  const handleRowClick = (symbol: string) => {
    setActiveSymbol(symbol)
  }

  const totalPnl = positions.reduce((sum, p) => sum + (p.unrealizedPnl ?? p.unrealizedPnL ?? 0), 0)

  return (
    <div className="h-full flex flex-col bg-sterling-panel text-xs">
      {/* Header */}
      <div className="flex items-center justify-between px-2 py-1 bg-sterling-header border-b border-sterling-border">
        <div className="flex items-center gap-2">
          <span className="font-bold text-sterling-text">POSITIONS</span>
          <button
            onClick={() => fetchPositions(true)}
            disabled={isRefreshing}
            className={`px-1.5 py-0.5 bg-[#1e3a5f] text-accent-primary text-xxs rounded-sm hover:brightness-110 disabled:opacity-50 ${
              isRefreshing ? 'animate-pulse' : ''
            }`}
            title="Sync from Schwab"
          >
            {isRefreshing ? '...' : '🔄'}
          </button>
          {lastSync && (
            <span className="text-sterling-muted text-xxs">
              {lastSync.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
            </span>
          )}
        </div>
        <span className={`font-bold ${totalPnl >= 0 ? 'text-up' : 'text-down'}`}>
          {totalPnl >= 0 ? '+' : ''}${totalPnl.toFixed(2)}
        </span>
      </div>

      {/* Table */}
      <div className="flex-1 overflow-auto">
        <table className="w-full">
          <thead className="sticky top-0 bg-sterling-header">
            <tr className="text-sterling-muted">
              <th className="px-1 py-1 text-center w-6">AI</th>
              <th className="px-1 py-1 text-left">Sym</th>
              <th className="px-1 py-1 text-center">Side</th>
              <th className="px-1 py-1 text-right">Qty</th>
              <th className="px-1 py-1 text-right">Avg</th>
              <th className="px-1 py-1 text-right">Last</th>
              <th className="px-1 py-1 text-right">P&L</th>
              <th className="px-1 py-1 text-right">%</th>
            </tr>
          </thead>
          <tbody>
            {positions.length === 0 ? (
              <tr>
                <td colSpan={8} className="text-center py-4 text-sterling-muted">
                  No positions
                </td>
              </tr>
            ) : (
              positions.map((pos) => {
                const side = pos.quantity > 0 ? 'LONG' : 'SHORT'
                const lastPrice = (pos.marketValue ?? 0) / Math.abs(pos.quantity)
                const pnl = pos.unrealizedPnl ?? pos.unrealizedPnL ?? 0
                const pnlPercent = pos.unrealizedPnlPercent ?? pos.unrealizedPnLPercent ?? 0
                const avgCost = pos.avgCost ?? pos.avgPrice ?? 0
                const isMonitored = aiMonitored.has(pos.symbol)

                return (
                  <tr
                    key={pos.symbol}
                    className="hover:bg-sterling-highlight cursor-pointer border-b border-sterling-border"
                    onClick={() => handleRowClick(pos.symbol)}
                  >
                    <td className="px-1 py-1 text-center">
                      <input
                        type="checkbox"
                        checked={isMonitored}
                        onChange={(e) => {
                          e.stopPropagation()
                          toggleAiTakeover(pos.symbol, e.target.checked)
                        }}
                        className="w-3 h-3 cursor-pointer"
                        title="Enable AI auto-exit"
                      />
                    </td>
                    <td className="px-1 py-1 font-bold text-accent-primary">
                      {pos.symbol}
                    </td>
                    <td className="px-1 py-1 text-center">
                      <span
                        className={`px-1 rounded text-xxs font-bold ${
                          side === 'LONG'
                            ? 'bg-buy text-white'
                            : 'bg-sell text-white'
                        }`}
                      >
                        {side}
                      </span>
                    </td>
                    <td className="px-1 py-1 text-right">
                      {Math.abs(pos.quantity)}
                    </td>
                    <td className="px-1 py-1 text-right">
                      ${avgCost.toFixed(2)}
                    </td>
                    <td className="px-1 py-1 text-right">
                      ${lastPrice.toFixed(2)}
                    </td>
                    <td
                      className={`px-1 py-1 text-right font-bold ${
                        pnl >= 0 ? 'text-up' : 'text-down'
                      }`}
                    >
                      {pnl >= 0 ? '+' : ''}$
                      {pnl.toFixed(2)}
                    </td>
                    <td
                      className={`px-1 py-1 text-right ${
                        pnlPercent >= 0 ? 'text-up' : 'text-down'
                      }`}
                    >
                      {pnlPercent >= 0 ? '+' : ''}
                      {pnlPercent.toFixed(1)}%
                    </td>
                  </tr>
                )
              })
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}
