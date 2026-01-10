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
  const [isAddingAll, setIsAddingAll] = useState(false)
  const [addedSymbols, setAddedSymbols] = useState<Set<string>>(new Set())
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
      // Check raw_movers OR scanner_results for candidates
      const rawMovers = data.raw_movers || []
      const scannerResults = data.scanner_results || {}

      // First try raw movers (Schwab gainers)
      if (rawMovers.length > 0) {
        const mapped: ScannerResult[] = rawMovers.map((item: any) => ({
          symbol: item.symbol,
          price: item.price || item.lastPrice || 0,
          changePercent: item.change_pct || item.percentChange || 0,
          direction: (item.change_pct || item.percentChange || 0) > 0 ? 'UP' : 'DOWN',
          volume: item.volume || item.totalVolume || 0,
          relVol: 0,
          scanner: 'SCHWAB',
        }))
        setResults(activeScanner, mapped)
      }
      // Fallback: try scanner_results
      else if (Object.keys(scannerResults).length > 0) {
        const allCandidates: ScannerResult[] = []
        for (const [scannerName, candidates] of Object.entries(scannerResults)) {
          if (Array.isArray(candidates)) {
            candidates.forEach((item: any) => {
              allCandidates.push({
                symbol: item.symbol,
                price: item.price || 0,
                changePercent: item.change_pct || item.gap_pct || 0,
                direction: (item.change_pct || item.gap_pct || 0) > 0 ? 'UP' : 'DOWN',
                volume: item.volume || 0,
                relVol: item.rel_vol || 0,
                scanner: scannerName as 'GAPPER' | 'GAINER' | 'HOD',
              })
            })
          }
        }
        if (allCandidates.length > 0) {
          setResults(activeScanner, allCandidates)
        }
      }
      // If still empty, try Finviz as backup
      else {
        console.log('[Scanner] No Schwab movers, trying Finviz backup...')
        const finvizData = await api.getFinvizGainers(5, 50) as any
        const finvizItems = finvizData?.results || finvizData?.candidates || []
        if (finvizItems.length > 0) {
          const mapped: ScannerResult[] = finvizItems.slice(0, 25).map((item: any) => ({
            symbol: item.symbol,
            price: item.price || 0,
            changePercent: item.pct_change || item.change_pct || 0,
            direction: (item.pct_change || 0) > 0 ? 'UP' : 'DOWN',
            volume: item.volume || 0,
            relVol: item.rel_vol || 0,
            scanner: 'FINVIZ',
          }))
          setResults(activeScanner, mapped)
        }
      }
    } catch (err) {
      console.error('Discovery failed:', err)
    }
    setIsDiscovering(false)
  }, [activeScanner, setResults])

  const currentResults = results[activeScanner]

  // Enrich scanner results with setup/exec status (Task E)
  const enrichWithSetupExec = useCallback(async (items: ScannerResult[]): Promise<ScannerResult[]> => {
    if (items.length === 0) return items
    try {
      const symbols = items.map(i => i.symbol)
      const response = await api.getBatchSetupExecStatus(symbols)
      if (response.results) {
        return items.map(item => {
          const status = response.results[item.symbol]
          if (status) {
            return {
              ...item,
              grade: status.grade || item.grade,
              score: status.score || item.score,
              execStatus: status.exec_status || undefined,
              execReason: status.exec_reason || undefined,
            }
          }
          return item
        })
      }
    } catch (err) {
      console.debug('Setup/exec enrichment failed:', err)
    }
    return items
  }, [])

  // Auto-enrich current results when they change (Task E)
  useEffect(() => {
    const enrichResults = async () => {
      if (currentResults.length > 0 && !currentResults[0].execStatus) {
        const enriched = await enrichWithSetupExec(currentResults)
        if (enriched !== currentResults) {
          setResults(activeScanner, enriched)
        }
      }
    }
    enrichResults()
  }, [currentResults, activeScanner, setResults, enrichWithSetupExec])

  // Add single symbol to worklist
  const handleAddToWorklist = useCallback(async (symbol: string, e: React.MouseEvent) => {
    e.stopPropagation()
    try {
      await fetch('/api/worklist/add', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symbol, skip_screening: true })
      })
      setAddedSymbols(prev => new Set(prev).add(symbol))
      // Clear the added indicator after 3 seconds
      setTimeout(() => {
        setAddedSymbols(prev => {
          const newSet = new Set(prev)
          newSet.delete(symbol)
          return newSet
        })
      }, 3000)
    } catch (err) {
      console.error(`Failed to add ${symbol} to worklist:`, err)
    }
  }, [])

  // Add all scanner results to worklist
  const handleAddAllToWorklist = useCallback(async () => {
    setIsAddingAll(true)
    const symbols = currentResults.map(r => r.symbol)
    try {
      // Add symbols in parallel (batch of 5 at a time)
      for (let i = 0; i < symbols.length; i += 5) {
        const batch = symbols.slice(i, i + 5)
        await Promise.all(batch.map(symbol =>
          fetch('/api/worklist/add', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ symbol, skip_screening: true })
          })
        ))
      }
      // Mark all as added
      setAddedSymbols(new Set(symbols))
      setTimeout(() => setAddedSymbols(new Set()), 3000)
    } catch (err) {
      console.error('Failed to add all to worklist:', err)
    }
    setIsAddingAll(false)
  }, [currentResults])

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
            onClick={handleAddAllToWorklist}
            disabled={isAddingAll || currentResults.length === 0}
            className={`px-2 py-0.5 bg-accent-primary/20 text-accent-primary text-xxs font-bold rounded-sm hover:bg-accent-primary/30 disabled:opacity-50 ${
              isAddingAll ? 'animate-pulse' : ''
            }`}
            title="Add all scanner results to worklist"
          >
            {isAddingAll ? '...' : '+ALL'}
          </button>
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
              <th className="px-1 py-1 text-center" title="Setup Quality (A/B/C grade)">Setup</th>
              <th className="px-1 py-1 text-center" title="Execution Permission (gating pass/fail)">Exec</th>
              <th className="px-1 py-1 text-center w-8" title="Add to Worklist">+</th>
            </tr>
          </thead>
          <tbody>
            {sortedResults.length === 0 ? (
              <tr>
                <td colSpan={8} className="text-center py-4 text-sterling-muted">
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
                  {/* Setup Grade Column (Task E) */}
                  <td className="px-1 py-1 text-center">
                    {item.grade ? (
                      <span
                        className={`px-1.5 py-0.5 rounded text-xxs font-bold ${
                          item.grade === 'A' ? 'bg-up/30 text-up' :
                          item.grade === 'B' ? 'bg-warning/30 text-warning' :
                          'bg-sterling-muted/30 text-sterling-muted'
                        }`}
                        title={item.score ? `Score: ${item.score}%` : ''}
                      >
                        {item.grade}
                        {item.score && <span className="ml-0.5 text-xxs opacity-70">({item.score})</span>}
                      </span>
                    ) : (
                      <span className="text-sterling-muted">--</span>
                    )}
                  </td>
                  {/* Exec Permission Column (Task E) */}
                  <td className="px-1 py-1 text-center">
                    {item.execStatus ? (
                      <span
                        className={`px-1.5 py-0.5 rounded text-xxs font-bold ${
                          item.execStatus === 'YES' ? 'bg-up/30 text-up' : 'bg-down/30 text-down'
                        }`}
                        title={item.execReason || ''}
                      >
                        {item.execStatus}
                      </span>
                    ) : (
                      <span className="text-sterling-muted">--</span>
                    )}
                  </td>
                  <td className="px-1 py-1 text-center">
                    <button
                      onClick={(e) => handleAddToWorklist(item.symbol, e)}
                      className={`w-5 h-5 rounded-sm text-xxs font-bold transition-all ${
                        addedSymbols.has(item.symbol)
                          ? 'bg-up/30 text-up'
                          : 'bg-accent-primary/20 text-accent-primary hover:bg-accent-primary/40'
                      }`}
                      title={addedSymbols.has(item.symbol) ? 'Added!' : `Add ${item.symbol} to worklist`}
                    >
                      {addedSymbols.has(item.symbol) ? '✓' : '+'}
                    </button>
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
