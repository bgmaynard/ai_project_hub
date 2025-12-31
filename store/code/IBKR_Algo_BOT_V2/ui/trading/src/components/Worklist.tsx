import { useEffect, useState, useCallback, useMemo } from 'react'
import { useWatchlistStore, WatchlistItem } from '../stores/watchlistStore'
import { useSymbolStore } from '../stores/symbolStore'
import api from '../services/api'

type SortField = 'symbol' | 'price' | 'changePercent' | 'rvol' | 'float' | 'aiScore' | 'volume'
type SortDir = 'asc' | 'desc'

interface WatchlistStatus {
  session_date: string
  cycle_count: number
  active_count: number
  total_candidates: number
  config: {
    current_rel_vol_floor: number
    max_active_symbols: number
  }
  active_watchlist: Array<{
    symbol: string
    rank: number
    dominance_score: number
    gap_pct: number
    rel_vol_daily: number
    price: number
  }>
}

export default function Worklist() {
  const { items, setItems, removeItem, setLoading } = useWatchlistStore()
  const { setActiveSymbol, setNewsFilter } = useSymbolStore()
  const [newSymbol, setNewSymbol] = useState('')
  const [isAdding, setIsAdding] = useState(false)
  const [error, setError] = useState('')
  const [sortField, setSortField] = useState<SortField>('changePercent')
  const [sortDir, setSortDir] = useState<SortDir>('desc')

  // Momentum watchlist state
  const [watchlistStatus, setWatchlistStatus] = useState<WatchlistStatus | null>(null)
  const [isRefreshing, setIsRefreshing] = useState(false)
  const [isPurging, setIsPurging] = useState(false)
  const [isRunningDiscovery, setIsRunningDiscovery] = useState(false)

  const fetchWorklist = useCallback(async () => {
    setLoading(true)
    try {
      const response = await fetch('/api/worklist')
      const json = await response.json()

      // API returns { success: true, data: [...], count: N }
      const items = json?.data || json?.symbols || (Array.isArray(json) ? json : [])

      if (Array.isArray(items)) {
        const mapped: WatchlistItem[] = items.map((s: any) => ({
          symbol: s.symbol || s,
          price: s.price || 0,
          change: s.change || 0,
          changePercent: s.change_percent || s.changePercent || 0,
          volume: s.volume || 0,
          rvol: s.rel_volume || s.rvol || 1.0,
          float: s.float || 0,
          momentum: s.momentum || (s.change_percent > 0 ? 'UP' : s.change_percent < 0 ? 'DOWN' : 'FLAT'),
          aiScore: s.ai_confidence || s.ai_score || s.ensemble_score || 50,
          hasNews: s.has_news || false,
          lastNews: s.last_news || '',
          fsmState: s.fsm_state || '',
        }))
        setItems(mapped)
      }
    } catch (err) {
      console.error('Failed to load worklist:', err)
    }
    setLoading(false)
  }, [setItems, setLoading])

  // Fetch momentum watchlist status
  const fetchWatchlistStatus = useCallback(async () => {
    try {
      const status = await api.getWatchlistStatus() as WatchlistStatus
      setWatchlistStatus(status)
      console.log('[Worklist] Fetched watchlist status:', status)
    } catch (err) {
      console.error('[Worklist] Failed to fetch watchlist status:', err)
    }
  }, [])

  // Refresh momentum watchlist
  const handleRefreshWatchlist = async () => {
    setIsRefreshing(true)
    setError('')
    try {
      const result = await api.refreshWatchlist()
      console.log('[Worklist] Refresh result:', result)
      await fetchWatchlistStatus()
      await fetchWorklist()
    } catch (err) {
      console.error('[Worklist] Refresh failed:', err)
      setError('Refresh failed')
      setTimeout(() => setError(''), 3000)
    }
    setIsRefreshing(false)
  }

  // Purge momentum watchlist
  const handlePurgeWatchlist = async () => {
    if (!confirm('Purge ALL symbols from momentum watchlist?')) return
    setIsPurging(true)
    setError('')
    try {
      const result = await api.purgeWatchlist()
      console.log('[Worklist] Purge result:', result)
      await fetchWatchlistStatus()
      await fetchWorklist()
    } catch (err) {
      console.error('[Worklist] Purge failed:', err)
      setError('Purge failed')
      setTimeout(() => setError(''), 3000)
    }
    setIsPurging(false)
  }

  // Run discovery pipeline
  const handleRunDiscovery = async () => {
    setIsRunningDiscovery(true)
    setError('')
    try {
      const result = await api.runDiscovery()
      console.log('[Worklist] Discovery result:', result)
      // Wait a moment for pipeline to process
      setTimeout(async () => {
        await fetchWatchlistStatus()
        await fetchWorklist()
        setIsRunningDiscovery(false)
      }, 2000)
    } catch (err) {
      console.error('[Worklist] Discovery failed:', err)
      setError('Discovery failed')
      setTimeout(() => setError(''), 3000)
      setIsRunningDiscovery(false)
    }
  }

  // Delete symbol from momentum watchlist
  const handleDeleteFromMomentumWatchlist = async (symbol: string, e: React.MouseEvent) => {
    e.stopPropagation()
    try {
      await api.deleteWatchlistSymbol(symbol)
      console.log(`[Worklist] Deleted ${symbol} from momentum watchlist`)
      await fetchWatchlistStatus()
    } catch (err) {
      console.error(`[Worklist] Failed to delete ${symbol}:`, err)
    }
  }

  useEffect(() => {
    fetchWorklist()
    fetchWatchlistStatus()
    const interval = setInterval(() => {
      fetchWorklist()
      fetchWatchlistStatus()
    }, 5000)  // Poll every 5s
    return () => clearInterval(interval)
  }, [fetchWorklist, fetchWatchlistStatus])

  // Sort handler
  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDir(sortDir === 'asc' ? 'desc' : 'asc')
    } else {
      setSortField(field)
      setSortDir('desc')
    }
  }

  // Sorted items
  const sortedItems = useMemo(() => {
    return [...items].sort((a, b) => {
      let aVal = a[sortField]
      let bVal = b[sortField]

      // Handle string comparison for symbol
      if (sortField === 'symbol') {
        return sortDir === 'asc'
          ? String(aVal).localeCompare(String(bVal))
          : String(bVal).localeCompare(String(aVal))
      }

      // Numeric comparison
      const aNum = Number(aVal) || 0
      const bNum = Number(bVal) || 0
      return sortDir === 'asc' ? aNum - bNum : bNum - aNum
    })
  }, [items, sortField, sortDir])

  // Sort indicator
  const SortIndicator = ({ field }: { field: SortField }) => {
    if (sortField !== field) return <span className="text-sterling-muted/30 ml-0.5">↕</span>
    return <span className="text-accent-primary ml-0.5">{sortDir === 'asc' ? '↑' : '↓'}</span>
  }

  const handleAddSymbol = async (e: React.FormEvent) => {
    e.preventDefault()
    const symbol = newSymbol.trim().toUpperCase()
    if (!symbol) return

    setIsAdding(true)
    setError('')
    try {
      const response = await fetch('/api/worklist/add', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symbol, skip_screening: true }),
      })
      const result = await response.json()

      if (result.success) {
        setNewSymbol('')
        // Refresh the list
        await fetchWorklist()
      } else {
        setError(result.message || 'Failed to add symbol')
        // Clear error after 3 seconds
        setTimeout(() => setError(''), 3000)
      }
    } catch (err) {
      console.error('Failed to add symbol:', err)
      setError('Network error')
      setTimeout(() => setError(''), 3000)
    }
    setIsAdding(false)
  }

  const handleRemove = async (symbol: string, e: React.MouseEvent) => {
    e.stopPropagation()
    try {
      await fetch(`/api/worklist/${symbol}`, { method: 'DELETE' })
      removeItem(symbol)
    } catch (err) {
      console.error('Failed to remove:', err)
    }
  }

  const getMomentumArrow = (momentum: string) => {
    switch (momentum) {
      case 'UP':
        return <span className="text-up font-bold">▲</span>
      case 'DOWN':
        return <span className="text-down font-bold">▼</span>
      default:
        return <span className="text-sterling-muted">―</span>
    }
  }

  const getAIScoreColor = (score: number) => {
    if (score >= 70) return 'text-up'
    if (score >= 50) return 'text-warning'
    return 'text-down'
  }

  return (
    <div className="h-full flex flex-col bg-sterling-panel text-xs">
      {/* Header with Add Symbol */}
      <div className="flex items-center justify-between px-2 py-1 bg-sterling-header border-b border-sterling-border gap-2">
        <span className="font-bold text-sterling-text whitespace-nowrap">WORKLIST</span>

        {/* Add Symbol Form */}
        <form onSubmit={handleAddSymbol} className="flex gap-1 flex-1 max-w-[200px]">
          <input
            type="text"
            value={newSymbol}
            onChange={(e) => setNewSymbol(e.target.value.toUpperCase())}
            placeholder="Add symbol..."
            className="flex-1 px-2 py-0.5 bg-[#252525] border border-[#404040] text-white text-xs rounded-sm focus:border-accent-primary focus:outline-none min-w-0"
          />
          <button
            type="submit"
            disabled={isAdding || !newSymbol.trim()}
            className="px-2 py-0.5 bg-[#134e4a] text-up text-xs rounded-sm hover:brightness-110 disabled:opacity-50"
          >
            +
          </button>
        </form>

        <span className="text-sterling-muted text-xxs whitespace-nowrap">{items.length}</span>
      </div>

      {/* Momentum Watchlist Controls */}
      <div className="flex items-center justify-between px-2 py-1 bg-[#1a1a2e] border-b border-sterling-border gap-1">
        {/* Control Buttons */}
        <div className="flex items-center gap-1">
          <button
            onClick={handleRunDiscovery}
            disabled={isRunningDiscovery}
            className={`px-2 py-0.5 text-xs rounded-sm hover:brightness-110 disabled:opacity-50 ${
              isRunningDiscovery ? 'bg-warning text-black animate-pulse' : 'bg-[#166534] text-up'
            }`}
            title="Run Discovery Pipeline (POST /api/task-queue/run)"
          >
            {isRunningDiscovery ? '⏳' : '🚀'} Discovery
          </button>
          <button
            onClick={handleRefreshWatchlist}
            disabled={isRefreshing}
            className={`px-2 py-0.5 text-xs rounded-sm hover:brightness-110 disabled:opacity-50 ${
              isRefreshing ? 'bg-accent-primary text-black animate-pulse' : 'bg-[#1e3a5f] text-accent-primary'
            }`}
            title="Refresh Watchlist (POST /api/watchlist/refresh)"
          >
            {isRefreshing ? '⏳' : '🔄'} Refresh
          </button>
          <button
            onClick={handlePurgeWatchlist}
            disabled={isPurging}
            className={`px-2 py-0.5 text-xs rounded-sm hover:brightness-110 disabled:opacity-50 ${
              isPurging ? 'bg-down text-white animate-pulse' : 'bg-[#7f1d1d] text-down'
            }`}
            title="Purge ALL Watchlist (POST /api/watchlist/purge)"
          >
            {isPurging ? '⏳' : '🗑️'} Purge
          </button>
        </div>

        {/* Status Display */}
        {watchlistStatus && (
          <div className="flex items-center gap-2 text-xxs text-sterling-muted">
            <span title="Session Date">📅 {watchlistStatus.session_date}</span>
            <span title="Current Rel Vol Floor">RVol≥{watchlistStatus.config?.current_rel_vol_floor?.toFixed(1) || '?'}</span>
            <span title="Active Symbols" className="text-up">{watchlistStatus.active_count || 0} active</span>
            <span title="Cycle Count">C{watchlistStatus.cycle_count || 0}</span>
          </div>
        )}
      </div>

      {/* Error display */}
      {error && (
        <div className="px-2 py-1 bg-[#7f1d1d] text-down text-xs border-b border-sterling-border">
          {error}
        </div>
      )}

      {/* Table */}
      <div className="flex-1 overflow-auto">
        <table className="w-full">
          <thead className="sticky top-0 bg-sterling-header">
            <tr className="text-sterling-muted text-xxs">
              <th
                className="px-1 py-1 text-left cursor-pointer hover:text-white select-none"
                onClick={() => handleSort('symbol')}
              >
                Sym<SortIndicator field="symbol" />
              </th>
              <th
                className="px-1 py-1 text-right cursor-pointer hover:text-white select-none"
                onClick={() => handleSort('price')}
              >
                Price<SortIndicator field="price" />
              </th>
              <th
                className="px-1 py-1 text-right cursor-pointer hover:text-white select-none"
                onClick={() => handleSort('changePercent')}
              >
                %Chg<SortIndicator field="changePercent" />
              </th>
              <th
                className="px-1 py-1 text-right cursor-pointer hover:text-white select-none"
                onClick={() => handleSort('rvol')}
              >
                RVol<SortIndicator field="rvol" />
              </th>
              <th
                className="px-1 py-1 text-right cursor-pointer hover:text-white select-none"
                onClick={() => handleSort('float')}
              >
                Float<SortIndicator field="float" />
              </th>
              <th className="px-1 py-1 text-center">Mom</th>
              <th
                className="px-1 py-1 text-center cursor-pointer hover:text-white select-none"
                onClick={() => handleSort('aiScore')}
              >
                AI<SortIndicator field="aiScore" />
              </th>
              <th className="px-1 py-1 text-center">News</th>
              <th className="px-1 py-1 text-center w-12" title="Delete from worklist / momentum watchlist">Del</th>
            </tr>
          </thead>
          <tbody>
            {sortedItems.length === 0 ? (
              <tr>
                <td colSpan={9} className="text-center py-8 text-sterling-muted">
                  <div className="mb-2">No symbols in worklist</div>
                  <div className="text-[10px]">Add symbols using the input above</div>
                </td>
              </tr>
            ) : (
              sortedItems.map((item) => (
                <tr
                  key={item.symbol}
                  className="hover:bg-sterling-highlight cursor-pointer border-b border-sterling-border"
                  onClick={() => setActiveSymbol(item.symbol)}
                >
                  <td className="px-1 py-1 font-bold text-accent-primary">
                    {item.symbol}
                  </td>
                  <td className="px-1 py-1 text-right text-white">
                    ${item.price.toFixed(2)}
                  </td>
                  <td
                    className={`px-1 py-1 text-right font-bold ${
                      item.changePercent >= 0 ? 'text-up' : 'text-down'
                    }`}
                  >
                    {item.changePercent >= 0 ? '+' : ''}
                    {item.changePercent.toFixed(1)}%
                  </td>
                  <td
                    className={`px-1 py-1 text-right ${
                      item.rvol >= 2 ? 'text-warning font-bold' : 'text-sterling-text'
                    }`}
                  >
                    {item.rvol.toFixed(1)}x
                  </td>
                  <td className="px-1 py-1 text-right text-sterling-text">
                    {item.float > 0
                      ? item.float >= 1000
                        ? `${(item.float / 1000).toFixed(0)}B`
                        : `${item.float.toFixed(0)}M`
                      : '--'}
                  </td>
                  <td className="px-1 py-1 text-center">
                    {getMomentumArrow(item.momentum)}
                  </td>
                  <td
                    className={`px-1 py-1 text-center font-bold ${getAIScoreColor(
                      item.aiScore
                    )}`}
                  >
                    {item.aiScore}
                  </td>
                  <td className="px-1 py-1 text-center">
                    {item.hasNews ? (
                      <span
                        className="text-warning cursor-pointer hover:text-white"
                        title={`Click to filter news: ${item.lastNews}`}
                        onClick={(e) => {
                          e.stopPropagation()
                          setNewsFilter(item.symbol)
                        }}
                      >
                        📰
                      </span>
                    ) : (
                      <span className="text-sterling-muted">-</span>
                    )}
                  </td>
                  <td className="px-1 py-1 text-center">
                    <div className="flex items-center justify-center gap-1">
                      <button
                        onClick={(e) => handleRemove(item.symbol, e)}
                        className="text-down hover:text-white text-xxs"
                        title="Remove from worklist (DELETE /api/worklist/{symbol})"
                      >
                        ✕
                      </button>
                      <button
                        onClick={(e) => handleDeleteFromMomentumWatchlist(item.symbol, e)}
                        className="text-warning hover:text-white text-xxs"
                        title="Delete from momentum watchlist (DELETE /api/watchlist/{symbol})"
                      >
                        ⊘
                      </button>
                    </div>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}
