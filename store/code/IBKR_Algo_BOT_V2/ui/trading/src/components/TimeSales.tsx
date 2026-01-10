import { useEffect, useState, useRef, useCallback } from 'react'
import { useSymbolStore } from '../stores/symbolStore'

interface Trade {
  time: string
  price: number
  size: number
  side: 'B' | 'S' | 'N' // Buy, Sell, Neutral
}

export default function TimeSales() {
  const { activeSymbol } = useSymbolStore()
  const [trades, setTrades] = useState<Trade[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [lastSymbol, setLastSymbol] = useState('')
  const prevPriceRef = useRef<number>(0)
  const containerRef = useRef<HTMLDivElement>(null)

  const fetchTrades = useCallback(async () => {
    if (!activeSymbol) return
    setIsLoading(true)

    try {
      // Use timesales endpoint which has Schwab fallback
      const response = await fetch(`/api/timesales/${activeSymbol}`)
      const data = await response.json()

      if (data?.trades || Array.isArray(data)) {
        const tradesArray = data.trades || data
        const mapped: Trade[] = tradesArray.slice(0, 100).map((t: any) => {
          const price = t.price || t.p || 0
          const side: 'B' | 'S' | 'N' =
            t.side ||
            (price > prevPriceRef.current ? 'B' : price < prevPriceRef.current ? 'S' : 'N')

          // Format time
          let timeStr = ''
          if (t.time) {
            timeStr = t.time
          } else if (t.timestamp) {
            const date = new Date(t.timestamp)
            timeStr = date.toLocaleTimeString('en-US', {
              hour12: false,
              hour: '2-digit',
              minute: '2-digit',
              second: '2-digit'
            })
          } else {
            timeStr = new Date().toLocaleTimeString('en-US', {
              hour12: false,
              hour: '2-digit',
              minute: '2-digit',
              second: '2-digit'
            })
          }

          return {
            time: timeStr,
            price,
            size: t.size || t.s || t.volume || 0,
            side,
          }
        })

        if (mapped.length > 0) {
          prevPriceRef.current = mapped[0].price
        }
        setTrades(mapped)
      }
    } catch (err) {
      console.error('Failed to load T&S:', err)
    }
    setIsLoading(false)
  }, [activeSymbol])

  useEffect(() => {
    if (!activeSymbol) return

    // Clear trades when symbol changes
    if (activeSymbol !== lastSymbol) {
      setTrades([])
      setLastSymbol(activeSymbol)
      prevPriceRef.current = 0
    }

    fetchTrades()
    const interval = setInterval(fetchTrades, 3000) // Refresh every 1 second
    return () => clearInterval(interval)
  }, [activeSymbol, fetchTrades, lastSymbol])

  const isLargePrint = (size: number) => size >= 1000

  const getRowStyle = (trade: Trade) => {
    if (isLargePrint(trade.size)) {
      return { background: 'rgba(255, 255, 0, 0.15)' }
    }
    if (trade.side === 'B') {
      return { background: 'rgba(0, 170, 0, 0.15)' }
    }
    if (trade.side === 'S') {
      return { background: 'rgba(204, 51, 51, 0.15)' }
    }
    return {}
  }

  return (
    <div className="h-full flex flex-col bg-black font-mono text-xs">
      {/* Header */}
      <div className="flex items-center justify-between px-2 py-1 bg-sterling-header border-b border-sterling-border flex-shrink-0">
        <span className="font-bold text-sterling-text">TIME & SALES</span>
        <span className="font-bold text-accent-primary">{activeSymbol}</span>
      </div>

      {/* Column Headers - Sticky */}
      <div
        className="grid grid-cols-3 px-1 py-0.5 text-[11px] font-bold text-[#888] border-b border-[#333] flex-shrink-0"
        style={{
          background: '#2d2d2d',
          gridTemplateColumns: '72px 82px 60px'
        }}
      >
        <div>TIME</div>
        <div className="text-right">PRICE</div>
        <div className="text-right">SIZE</div>
      </div>

      {/* Trades - Scrollable */}
      <div ref={containerRef} className="flex-1 overflow-y-auto">
        {trades.length === 0 ? (
          <div className="text-center py-8 text-[#555] italic text-[15px]">
            {isLoading ? (
              <div>Loading {activeSymbol}...</div>
            ) : (
              <>
                <div>No T&S data available</div>
                <div className="text-[12px] mt-2">Waiting for market data...</div>
              </>
            )}
          </div>
        ) : (
          trades.map((trade, i) => (
            <div
              key={i}
              className="grid px-1 py-0.5 border-b border-[#0a0a0a] hover:bg-[#1a2a3a] transition-colors"
              style={{
                ...getRowStyle(trade),
                gridTemplateColumns: '72px 82px 60px',
              }}
            >
              <div className="text-[#666] text-[11px]">{trade.time}</div>
              <div
                className={`text-right font-bold ${
                  trade.side === 'B'
                    ? 'text-[#00cc00]'
                    : trade.side === 'S'
                    ? 'text-[#ff3333]'
                    : 'text-white'
                }`}
              >
                {trade.price.toFixed(2)}
              </div>
              <div
                className={`text-right ${
                  isLargePrint(trade.size)
                    ? 'text-yellow-400 font-bold'
                    : 'text-[#aaa]'
                }`}
              >
                {trade.size.toLocaleString()}
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  )
}
