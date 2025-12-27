import { useEffect, useState, useCallback } from 'react'
import { useScannerStore, ScannerResult, ScannerType } from '../stores/scannerStore'
import { useSymbolStore } from '../stores/symbolStore'
import api from '../services/api'

export default function ScannerPanel() {
  const { activeScanner, results, setActiveScanner, setResults, setLoading } =
    useScannerStore()
  const { setActiveSymbol } = useSymbolStore()
  const [isRefreshing, setIsRefreshing] = useState(false)
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null)

  const fetchScannerData = useCallback(async (manual = false) => {
    if (manual) setIsRefreshing(true)
    setLoading(activeScanner, true)
    try {
      let data: any

      switch (activeScanner) {
        case 'hod':
          data = await api.getHODScanner()
          break
        case 'gappers':
          data = await api.getGappersScanner()
          break
        case 'gainers':
          data = await api.getGainersScanner()
          break
      }

      if (data) {
        const items = data.alerts || data.results || data.movers || []
        const mapped: ScannerResult[] = items.slice(0, 20).map((item: any) => ({
          symbol: item.symbol,
          price: item.price || item.current_price || 0,
          changePercent: item.change_percent || item.pct_change || 0,
          direction:
            item.direction ||
            (item.change_percent > 0 ? 'UP' : 'DOWN'),
          volume: item.volume || 0,
          float: item.float || 0,
          lastNews: item.last_news || item.catalyst || '',
          grade: item.grade,
          score: item.score,
        }))
        setResults(activeScanner, mapped)
        setLastUpdate(new Date())
      }
    } catch (err) {
      console.error(`Failed to load ${activeScanner} scanner:`, err)
    }
    setLoading(activeScanner, false)
    if (manual) setIsRefreshing(false)
  }, [activeScanner, setResults, setLoading])

  useEffect(() => {
    fetchScannerData()
    const interval = setInterval(() => fetchScannerData(false), 5000)
    return () => clearInterval(interval)
  }, [fetchScannerData])

  const currentResults = results[activeScanner]

  const getDirectionBadge = (direction: string) => {
    switch (direction) {
      case 'BREAKING':
        return (
          <span className="px-1 py-0.5 bg-up text-black text-xxs font-bold rounded">
            BREAK
          </span>
        )
      case 'TESTING':
        return (
          <span className="px-1 py-0.5 bg-warning text-black text-xxs font-bold rounded">
            TEST
          </span>
        )
      case 'REJECTING':
        return (
          <span className="px-1 py-0.5 bg-down text-white text-xxs font-bold rounded">
            REJ
          </span>
        )
      case 'UP':
        return <span className="text-up font-bold">▲</span>
      case 'DOWN':
        return <span className="text-down font-bold">▼</span>
      default:
        return <span className="text-sterling-muted">―</span>
    }
  }

  return (
    <div className="h-full flex flex-col bg-sterling-panel text-xs">
      {/* Header with Tabs */}
      <div className="flex items-center gap-1 px-2 py-1 bg-sterling-header border-b border-sterling-border">
        {(['hod', 'gappers', 'gainers'] as ScannerType[]).map((scanner) => (
          <button
            key={scanner}
            onClick={() => setActiveScanner(scanner)}
            className={`px-2 py-0.5 rounded text-xxs font-bold uppercase ${
              activeScanner === scanner
                ? 'bg-accent-primary text-white'
                : 'bg-sterling-bg text-sterling-muted hover:bg-sterling-highlight'
            }`}
          >
            {scanner === 'hod' ? 'HOD' : scanner === 'gappers' ? 'Gappers' : 'Gainers'}
          </button>
        ))}
        <div className="ml-auto flex items-center gap-2">
          <button
            onClick={() => fetchScannerData(true)}
            disabled={isRefreshing}
            className={`px-2 py-0.5 bg-[#1e3a5f] text-accent-primary text-xs rounded-sm hover:brightness-110 disabled:opacity-50 ${
              isRefreshing ? 'animate-pulse' : ''
            }`}
            title="Refresh scanner"
          >
            {isRefreshing ? '...' : '🔄'}
          </button>
          <span className="text-sterling-muted text-xxs">
            {currentResults.length}
          </span>
          {lastUpdate && (
            <span className="text-sterling-muted text-xxs" title={lastUpdate.toLocaleTimeString()}>
              {lastUpdate.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
            </span>
          )}
        </div>
      </div>

      {/* Table */}
      <div className="flex-1 overflow-auto">
        <table className="w-full">
          <thead className="sticky top-0 bg-sterling-header">
            <tr className="text-sterling-muted text-xxs">
              <th className="px-1 py-1 text-left">Sym</th>
              <th className="px-1 py-1 text-right">Price</th>
              <th className="px-1 py-1 text-right">%Chg</th>
              <th className="px-1 py-1 text-center">Dir</th>
              <th className="px-1 py-1 text-left">Catalyst</th>
            </tr>
          </thead>
          <tbody>
            {currentResults.length === 0 ? (
              <tr>
                <td colSpan={5} className="text-center py-4 text-sterling-muted">
                  No scanner results
                </td>
              </tr>
            ) : (
              currentResults.map((item) => (
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
                  <td className="px-1 py-1 text-center">
                    {getDirectionBadge(item.direction)}
                  </td>
                  <td className="px-1 py-1 text-sterling-text truncate max-w-[150px]">
                    {item.lastNews || '--'}
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
