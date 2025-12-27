import { useEffect, useState } from 'react'
import { useSymbolStore } from '../stores/symbolStore'
import { useMarketDataStore } from '../stores/marketDataStore'
import api from '../services/api'

interface Level2Entry {
  price: number
  size: number
  exchange?: string
  orders?: number
}

// Sterling-style level colors
const bidLevelColors = [
  'rgba(0, 170, 0, 0.25)',
  'rgba(0, 170, 0, 0.18)',
  'rgba(0, 170, 0, 0.12)',
  'rgba(0, 170, 0, 0.08)',
  'rgba(0, 170, 0, 0.04)',
]

const askLevelColors = [
  'rgba(204, 51, 51, 0.25)',
  'rgba(204, 51, 51, 0.18)',
  'rgba(204, 51, 51, 0.12)',
  'rgba(204, 51, 51, 0.08)',
  'rgba(204, 51, 51, 0.04)',
]

export default function Level2() {
  const { activeSymbol } = useSymbolStore()
  const { quotes, setQuote } = useMarketDataStore()
  const quote = quotes[activeSymbol]

  const [bids, setBids] = useState<Level2Entry[]>([])
  const [asks, setAsks] = useState<Level2Entry[]>([])
  const [luldUpper, setLuldUpper] = useState<number | null>(null)
  const [luldLower, setLuldLower] = useState<number | null>(null)

  useEffect(() => {
    if (!activeSymbol) return

    const fetchLevel2 = async () => {
      try {
        const data = (await api.getLevel2(activeSymbol)) as any
        if (data) {
          if (data.bids) {
            setBids(
              data.bids.slice(0, 15).map((b: any) => ({
                price: b.price || b.p || 0,
                size: b.size || b.s || 0,
                exchange: b.exchange || b.x || b.mmid || 'NSDQ',
                orders: b.orders || b.count || 1,
              }))
            )
          }
          if (data.asks) {
            setAsks(
              data.asks.slice(0, 15).map((a: any) => ({
                price: a.price || a.p || 0,
                size: a.size || a.s || 0,
                exchange: a.exchange || a.x || a.mmid || 'NSDQ',
                orders: a.orders || a.count || 1,
              }))
            )
          }

          // LULD bands if available
          if (data.luld) {
            setLuldUpper(data.luld.upper)
            setLuldLower(data.luld.lower)
          }

          // Update quote
          if (data.bid && data.ask) {
            setQuote(activeSymbol, {
              symbol: activeSymbol,
              bid: data.bid,
              ask: data.ask,
              last: data.last || quote?.last || 0,
              volume: data.volume || quote?.volume || 0,
              change: data.change || quote?.change || 0,
              changePercent: data.changePercent || quote?.changePercent || 0,
            })
          }
        }
      } catch (err) {
        console.error('Failed to load L2:', err)
      }
    }

    fetchLevel2()
    const interval = setInterval(fetchLevel2, 2000)
    return () => clearInterval(interval)
  }, [activeSymbol, setQuote, quote])

  const setOrderPrice = (price: number, _side: 'BUY' | 'SELL') => {
    // This would communicate with OrderEntry component
    // For now, just log
    console.log(`Set order price: ${price}`)
  }

  const isLargeSize = (size: number) => size >= 1000

  return (
    <div className="h-full flex flex-col bg-black font-mono text-xs">
      {/* Header */}
      <div className="flex items-center justify-between px-2 py-1 bg-sterling-header border-b border-sterling-border flex-shrink-0">
        <span className="font-bold text-sterling-text">LEVEL II</span>
        <span className="font-bold text-accent-primary">{activeSymbol}</span>
      </div>

      {/* LULD Indicator */}
      {(luldUpper || luldLower) && (
        <div className="px-2 py-1 bg-[#1a1a1a] border-b border-[#333] text-[11px] flex gap-4">
          <span className="text-sterling-muted">LULD:</span>
          <span className="text-up">▲ {luldUpper?.toFixed(2) || '--'}</span>
          <span className="text-down">▼ {luldLower?.toFixed(2) || '--'}</span>
        </div>
      )}

      {/* L2 Grid */}
      <div className="flex-1 flex overflow-hidden">
        {/* Bids Column */}
        <div className="flex-1 flex flex-col border-r border-[#1a1a1a] overflow-hidden">
          {/* Bid Header */}
          <div
            className="grid grid-cols-4 px-0.5 py-0.5 text-[10px] font-bold uppercase tracking-wide text-[#888] border-b border-[#333] flex-shrink-0"
            style={{ background: 'linear-gradient(180deg, #1a1a1a 0%, #0d0d0d 100%)' }}
          >
            <div className="text-center">MM</div>
            <div className="text-center">SIZE</div>
            <div className="text-center">BID</div>
            <div className="text-center">ORD</div>
          </div>
          {/* Bid Rows */}
          <div className="flex-1 overflow-y-auto">
            {bids.length === 0 ? (
              <div className="text-center py-4 text-[#555] italic">Level 2 data unavailable</div>
            ) : (
              bids.map((bid, i) => (
                <div
                  key={i}
                  onClick={() => setOrderPrice(bid.price, 'BUY')}
                  className="grid grid-cols-4 px-0.5 py-0.5 text-[12px] border-b border-[#0a0a0a] cursor-pointer hover:bg-[#1a2a3a]"
                  style={{ background: bidLevelColors[Math.min(i, 4)] }}
                >
                  <div className="text-center text-[#00aaff] text-[11px] font-semibold">
                    {bid.exchange?.slice(0, 4) || 'NSDQ'}
                  </div>
                  <div className={`text-center font-semibold ${isLargeSize(bid.size) ? 'text-yellow-400' : 'text-white'}`}>
                    {bid.size.toLocaleString()}
                  </div>
                  <div className="text-center font-bold text-[#00ff00]">
                    {bid.price.toFixed(2)}
                  </div>
                  <div className="text-center text-[#666] text-[10px]">
                    {bid.orders || ''}
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

        {/* Asks Column */}
        <div className="flex-1 flex flex-col border-l border-[#1a1a1a] overflow-hidden">
          {/* Ask Header */}
          <div
            className="grid grid-cols-4 px-0.5 py-0.5 text-[10px] font-bold uppercase tracking-wide text-[#888] border-b border-[#333] flex-shrink-0"
            style={{ background: 'linear-gradient(180deg, #1a1a1a 0%, #0d0d0d 100%)' }}
          >
            <div className="text-center">ORD</div>
            <div className="text-center">ASK</div>
            <div className="text-center">SIZE</div>
            <div className="text-center">MM</div>
          </div>
          {/* Ask Rows */}
          <div className="flex-1 overflow-y-auto">
            {asks.length === 0 ? (
              <div className="text-center py-4 text-[#555] italic">Level 2 data unavailable</div>
            ) : (
              asks.map((ask, i) => (
                <div
                  key={i}
                  onClick={() => setOrderPrice(ask.price, 'SELL')}
                  className="grid grid-cols-4 px-0.5 py-0.5 text-[12px] border-b border-[#0a0a0a] cursor-pointer hover:bg-[#1a2a3a]"
                  style={{ background: askLevelColors[Math.min(i, 4)] }}
                >
                  <div className="text-center text-[#666] text-[10px]">
                    {ask.orders || ''}
                  </div>
                  <div className="text-center font-bold text-[#ff3333]">
                    {ask.price.toFixed(2)}
                  </div>
                  <div className={`text-center font-semibold ${isLargeSize(ask.size) ? 'text-yellow-400' : 'text-white'}`}>
                    {ask.size.toLocaleString()}
                  </div>
                  <div className="text-center text-[#00aaff] text-[11px] font-semibold">
                    {ask.exchange?.slice(0, 4) || 'NSDQ'}
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
