import { useEffect, useState, useCallback, useMemo } from 'react'
import { useScannerStore, ScannerResult, ScannerType } from '../stores/scannerStore'
import { useSymbolStore } from '../stores/symbolStore'
import api from '../services/api'

interface ScannerStatusInfo {
  current_time_et: string
  active_scanners: string[]
  past_cutoff: boolean
}

type SortColumn = 'symbol' | 'price' | 'changePercent' | 'relVol' | 'volume'
type SortDirection = 'asc' | 'desc'

export default function ScannerPanel() {
  const { activeScanner, results, setActiveScanner, setResults, setLoading } =
    useScannerStore()
  const { setActiveSymbol } = useSymbolStore()
  const [isRefreshing, setIsRefreshing] = useState(false)
  const [isDiscovering, setIsDiscovering] = useState(false)
  const [scannerStatus, setScannerStatus] = useState<ScannerStatusInfo | null>(null)
  const [sortColumn, setSortColumn] = useState<SortColumn>('changePercent')
  const [sortDirection, setSortDirection] = useState<SortDirection>('desc')

  // Fetch scanner status to show which are active
  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const status = await api.getScannerStatus() as any
        setScannerStatus({
          current_time_et: status.current_time_et,
          active_scanners: status.active_scanners || [],
          past_cutoff: status.past_cutoff || false,
        })
      } catch (err) {
        console.error('Failed to fetch scanner status:', err)
      }
    }
    fetchStatus()
    const interval = setInterval(fetchStatus, 30000) // Update status every 30s
    return () => clearInterval(interval)
  }, [])

  const fetchScannerData = useCallback(async (manual = false) => {
    if (manual) setIsRefreshing(true)
    setLoading(activeScanner, true)
    try {
      let data: any
      let useFinviz = false

      // Try Schwab scanners first
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
        case 'premarket':
          data = await api.getPreMarketScanner()
          break
      }

      // Check if Schwab returned empty - fall back to Finviz
      const schwabItems = data?.candidates || data?.results || data?.alerts || data?.movers || data?.watchlist || []
      if (schwabItems.length === 0) {
        useFinviz = true
        // Use Finviz as fallback based on scanner type
        switch (activeScanner) {
          case 'gainers':
          case 'gappers':
            data = await api.getFinvizGainers(5, 50)
            break
          case 'hod':
            data = await api.getFinvizBreakouts()
            break
          case 'premarket':
            data = await api.getFinvizLowFloat(20)
            break
        }
      }

      if (data) {
        // Handle both Warrior scanner format and Finviz format
        const items = data.candidates || data.results || data.alerts || data.movers || data.watchlist || []
        const mapped: ScannerResult[] = items.slice(0, 25).map((item: any) => ({
          symbol: item.symbol,
          price: item.price || item.current_price || 0,
          changePercent: item.pct_change || item.change_percent || item.change_pct || item.gap_pct || 0,
          direction:
            item.direction ||
            ((item.pct_change || item.change_pct || item.gap_pct || 0) > 0 ? 'UP' : 'DOWN'),
          volume: item.volume || 0,
          relVol: item.rel_vol || 0,
          dayHigh: item.day_high || 0,
          gapPct: item.gap_pct || 0,
          float: item.float || 0,
          lastNews: item.last_news || item.catalyst || item.headline || '',
          grade: item.grade,
          score: item.score,
          scanner: useFinviz ? 'FINVIZ' : item.scanner,
        }))
        setResults(activeScanner, mapped)
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

  // Auto-discover from Schwab movers
  const handleDiscover = useCallback(async () => {
    setIsDiscovering(true)
    try {
      const data = await api.discoverCandidates() as any
      if (data.raw_movers?.length > 0) {
        // Map raw movers to scanner format
        const mapped: ScannerResult[] = data.raw_movers.map((item: any) => ({
          symbol: item.symbol,
          price: item.price || 0,
          changePercent: item.change_pct || 0,
          direction: (item.change_pct || 0) > 0 ? 'UP' : 'DOWN',
          volume: item.volume || 0,
          relVol: 0,
          scanner: 'SCHWAB',
        }))
        // Set results for current scanner tab
        setResults(activeScanner, mapped)
      }
    } catch (err) {
      console.error('Discovery failed:', err)
    }
    setIsDiscovering(false)
  }, [activeScanner, setResults])

  const currentResults = results[activeScanner]

  // Handle column sort click
  const handleSort = (column: SortColumn) => {
    if (sortColumn === column) {
      // Toggle direction if same column
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc')
    } else {
      // New column - default to desc for numbers, asc for symbol
      setSortColumn(column)
      setSortDirection(column === 'symbol' ? 'asc' : 'desc')
    }
  }

  // Sort results
  const sortedResults = useMemo(() => {
    if (!currentResults.length) return currentResults

    return [...currentResults].sort((a, b) => {
      let aVal: string | number
      let bVal: string | number

      switch (sortColumn) {
        case 'symbol':
          aVal = a.symbol
          bVal = b.symbol
          break
        case 'price':
          aVal = a.price
          bVal = b.price
          break
        case 'changePercent':
          aVal = a.changePercent
          bVal = b.changePercent
          break
        case 'relVol':
          aVal = a.relVol || 0
          bVal = b.relVol || 0
          break
        case 'volume':
          aVal = a.volume || 0
          bVal = b.volume || 0
          break
        default:
          return 0
      }

      if (typeof aVal === 'string' && typeof bVal === 'string') {
        return sortDirection === 'asc'
          ? aVal.localeCompare(bVal)
          : bVal.localeCompare(aVal)
      }

      return sortDirection === 'asc'
        ? (aVal as number) - (bVal as number)
        : (bVal as number) - (aVal as number)
    })
  }, [currentResults, sortColumn, sortDirection])

  // Sort indicator component
  const SortIndicator = ({ column }: { column: SortColumn }) => {
    if (sortColumn !== column) return <span className="text-sterling-muted/30 ml-0.5">⇅</span>
    return <span className="text-accent-primary ml-0.5">{sortDirection === 'asc' ? '↑' : '↓'}</span>
  }

  // Check if a scanner is currently active based on time
  const isScannerActive = (scanner: ScannerType): boolean => {
    if (!scannerStatus) return true // Assume active if no status
    const scannerMap: Record<ScannerType, string> = {
      hod: 'HOD',
      gappers: 'GAPPER',
      gainers: 'GAINER',
      premarket: 'GAPPER', // Pre-market uses gap scanner
    }
    return scannerStatus.active_scanners.includes(scannerMap[scanner])
  }

  return (
    <div className="h-full flex flex-col bg-sterling-panel text-xs">
      {/* Header with Tabs */}
      <div className="flex items-center gap-1 px-2 py-1 bg-sterling-header border-b border-sterling-border">
        {(['hod', 'gappers', 'gainers', 'premarket'] as ScannerType[]).map((scanner) => {
          const isActive = isScannerActive(scanner)
          return (
            <button
              key={scanner}
              onClick={() => setActiveScanner(scanner)}
              className={`px-2 py-0.5 rounded text-xxs font-bold uppercase ${
                activeScanner === scanner
                  ? 'bg-accent-primary text-white'
                  : isActive
                    ? 'bg-sterling-bg text-sterling-text hover:bg-sterling-highlight'
                    : 'bg-sterling-bg text-sterling-muted opacity-50 hover:bg-sterling-highlight'
              }`}
              title={isActive ? 'Active' : 'Outside time window'}
            >
              {scanner === 'hod' ? 'HOD' : scanner === 'gappers' ? 'Gap' : scanner === 'premarket' ? 'Pre' : 'Gain'}
              {isActive && <span className="ml-1 text-up">*</span>}
            </button>
          )
        })}
        <div className="ml-auto flex items-center gap-2">
          {scannerStatus && (
            <span className="text-sterling-muted text-xxs" title="ET Time">
              {scannerStatus.current_time_et} ET
            </span>
          )}
          <button
            onClick={handleDiscover}
            disabled={isDiscovering}
            className={`px-2 py-0.5 bg-up/20 text-up text-xxs font-bold rounded-sm hover:bg-up/30 disabled:opacity-50 ${
              isDiscovering ? 'animate-pulse' : ''
            }`}
            title="Auto-discover from Schwab movers"
          >
            {isDiscovering ? '...' : 'SCAN'}
          </button>
          <button
            onClick={() => fetchScannerData(true)}
            disabled={isRefreshing}
            className={`px-2 py-0.5 bg-[#1e3a5f] text-accent-primary text-xs rounded-sm hover:brightness-110 disabled:opacity-50 ${
              isRefreshing ? 'animate-pulse' : ''
            }`}
            title="Refresh scanner"
          >
            {isRefreshing ? '...' : '↻'}
          </button>
          <span className="text-sterling-muted text-xxs">
            {currentResults.length}
          </span>
        </div>
      </div>

      {/* Table */}
      <div className="flex-1 overflow-auto">
        <table className="w-full">
          <thead className="sticky top-0 bg-sterling-header">
            <tr className="text-sterling-muted text-xxs">
              <th
                className="px-1 py-1 text-left cursor-pointer hover:text-sterling-text select-none"
                onClick={() => handleSort('symbol')}
              >
                Sym<SortIndicator column="symbol" />
              </th>
              <th
                className="px-1 py-1 text-right cursor-pointer hover:text-sterling-text select-none"
                onClick={() => handleSort('price')}
              >
                Price<SortIndicator column="price" />
              </th>
              <th
                className="px-1 py-1 text-right cursor-pointer hover:text-sterling-text select-none"
                onClick={() => handleSort('changePercent')}
              >
                %Chg<SortIndicator column="changePercent" />
              </th>
              <th
                className="px-1 py-1 text-right cursor-pointer hover:text-sterling-text select-none"
                onClick={() => handleSort('relVol')}
              >
                RVol<SortIndicator column="relVol" />
              </th>
              <th
                className="px-1 py-1 text-right cursor-pointer hover:text-sterling-text select-none"
                onClick={() => handleSort('volume')}
              >
                Vol<SortIndicator column="volume" />
              </th>
            </tr>
          </thead>
          <tbody>
            {sortedResults.length === 0 ? (
              <tr>
                <td colSpan={5} className="text-center py-4 text-sterling-muted">
                  {!isScannerActive(activeScanner)
                    ? `${activeScanner.toUpperCase()} scanner outside active window`
                    : 'No scanner results'}
                </td>
              </tr>
            ) : (
              sortedResults.map((item) => (
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
                  <td className={`px-1 py-1 text-right ${
                    (item.relVol || 0) >= 3 ? 'text-up font-bold' :
                    (item.relVol || 0) >= 2 ? 'text-warning' : 'text-sterling-muted'
                  }`}>
                    {item.relVol ? `${item.relVol.toFixed(1)}x` : '--'}
                  </td>
                  <td className="px-1 py-1 text-right text-sterling-text">
                    {item.volume ? `${(item.volume / 1000).toFixed(0)}K` : '--'}
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {/* Footer with scanner info */}
      {scannerStatus?.past_cutoff && (
        <div className="px-2 py-1 bg-warning/20 text-warning text-xxs text-center">
          Past 10:30 ET cutoff - No new candidates
        </div>
      )}
    </div>
  )
}
