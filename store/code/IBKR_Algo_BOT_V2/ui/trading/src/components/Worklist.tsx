import { useEffect, useState, useCallback } from 'react'
import { useWatchlistStore, WatchlistItem } from '../stores/watchlistStore'
import { useSymbolStore } from '../stores/symbolStore'

export default function Worklist() {
  const { items, setItems, removeItem, setLoading } = useWatchlistStore()
  const { setActiveSymbol } = useSymbolStore()
  const [newSymbol, setNewSymbol] = useState('')
  const [isAdding, setIsAdding] = useState(false)
  const [error, setError] = useState('')

  const fetchWorklist = useCallback(async () => {
    setLoading(true)
    try {
      const response = await fetch('/api/worklist')
      const data = await response.json()

      if (data?.symbols && Array.isArray(data.symbols)) {
        const mapped: WatchlistItem[] = data.symbols.map((s: any) => ({
          symbol: s.symbol,
          price: s.price || 0,
          change: s.change || 0,
          changePercent: s.change_percent || 0,
          volume: s.volume || 0,
          rvol: s.rvol || 1.0,
          float: s.float || 0,
          momentum: s.momentum || 'FLAT',
          aiScore: s.ai_score || s.ensemble_score || 50,
          hasNews: s.has_news || false,
          lastNews: s.last_news || '',
          fsmState: s.fsm_state || '',
        }))
        setItems(mapped)
      } else if (Array.isArray(data)) {
        // Handle case where API returns array directly
        const mapped: WatchlistItem[] = data.map((s: any) => ({
          symbol: s.symbol || s,
          price: s.price || 0,
          change: s.change || 0,
          changePercent: s.change_percent || s.changePercent || 0,
          volume: s.volume || 0,
          rvol: s.rvol || 1.0,
          float: s.float || 0,
          momentum: s.momentum || 'FLAT',
          aiScore: s.ai_score || s.ensemble_score || 50,
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

  useEffect(() => {
    fetchWorklist()
    const interval = setInterval(fetchWorklist, 5000)
    return () => clearInterval(interval)
  }, [fetchWorklist])

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
      {/* Header with Add Symbol and Refresh */}
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

        {/* Refresh and Count */}
        <div className="flex items-center gap-2">
          <button
            onClick={() => fetchWorklist()}
            className="px-2 py-0.5 bg-[#1e3a5f] text-accent-primary text-xs rounded-sm hover:brightness-110"
            title="Refresh worklist"
          >
            🔄
          </button>
          <span className="text-sterling-muted text-xxs whitespace-nowrap">{items.length}</span>
        </div>
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
              <th className="px-1 py-1 text-left">Sym</th>
              <th className="px-1 py-1 text-right">Price</th>
              <th className="px-1 py-1 text-right">%Chg</th>
              <th className="px-1 py-1 text-right">RVol</th>
              <th className="px-1 py-1 text-right">Float</th>
              <th className="px-1 py-1 text-center">Mom</th>
              <th className="px-1 py-1 text-center">AI</th>
              <th className="px-1 py-1 text-center">News</th>
              <th className="px-1 py-1 text-center w-6">X</th>
            </tr>
          </thead>
          <tbody>
            {items.length === 0 ? (
              <tr>
                <td colSpan={9} className="text-center py-8 text-sterling-muted">
                  <div className="mb-2">No symbols in worklist</div>
                  <div className="text-[10px]">Add symbols using the input above</div>
                </td>
              </tr>
            ) : (
              items.map((item) => (
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
                        className="text-warning cursor-pointer"
                        title={item.lastNews}
                      >
                        📰
                      </span>
                    ) : (
                      <span className="text-sterling-muted">-</span>
                    )}
                  </td>
                  <td className="px-1 py-1 text-center">
                    <button
                      onClick={(e) => handleRemove(item.symbol, e)}
                      className="text-down hover:text-white text-xxs"
                      title="Remove from worklist"
                    >
                      ✕
                    </button>
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
