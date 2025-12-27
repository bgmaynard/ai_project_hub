import { useEffect, useState } from 'react'
import { useSymbolStore } from '../stores/symbolStore'
import { useMarketDataStore } from '../stores/marketDataStore'
import api from '../services/api'

interface HaltInfo {
  isHalted: boolean
  haltTime: string | null
  reason: string | null
}

interface AssetStatus {
  shortable: boolean
  etb: boolean  // Easy to borrow
  htb: boolean  // Hard to borrow
  marginable: boolean
}

export default function Quote() {
  const { activeSymbol } = useSymbolStore()
  const { quotes, setQuote } = useMarketDataStore()
  const quote = quotes[activeSymbol]

  const [haltInfo, setHaltInfo] = useState<HaltInfo>({ isHalted: false, haltTime: null, reason: null })
  const [assetStatus, setAssetStatus] = useState<AssetStatus>({ shortable: true, etb: true, htb: false, marginable: true })
  const [extendedData, setExtendedData] = useState<{
    high: number
    low: number
    open: number
    volume: number
    bidSize: number
    askSize: number
  }>({ high: 0, low: 0, open: 0, volume: 0, bidSize: 0, askSize: 0 })

  // Fetch quote and halt status
  useEffect(() => {
    if (!activeSymbol) return

    const fetchData = async () => {
      try {
        // Fetch price data
        const priceData = await api.getQuote(activeSymbol) as any
        if (priceData) {
          setQuote(activeSymbol, {
            symbol: activeSymbol,
            bid: priceData.bid || 0,
            ask: priceData.ask || 0,
            last: priceData.last || priceData.price || 0,
            volume: priceData.volume || 0,
            change: priceData.change || 0,
            changePercent: priceData.changePercent || priceData.change_percent || 0,
          })

          setExtendedData({
            high: priceData.high || priceData.dayHigh || 0,
            low: priceData.low || priceData.dayLow || 0,
            open: priceData.open || priceData.openPrice || 0,
            volume: priceData.volume || 0,
            bidSize: priceData.bidSize || priceData.bid_size || 0,
            askSize: priceData.askSize || priceData.ask_size || 0,
          })

          // Detect halt conditions
          const isHalted = (priceData.bid <= 0 && priceData.ask <= 0) ||
                          priceData.halted === true ||
                          priceData.status === 'HALTED'

          if (isHalted && !haltInfo.isHalted) {
            setHaltInfo({
              isHalted: true,
              haltTime: new Date().toLocaleTimeString(),
              reason: priceData.haltReason || 'Trading Halt Detected'
            })
          } else if (!isHalted && haltInfo.isHalted) {
            setHaltInfo({ isHalted: false, haltTime: null, reason: null })
          }
        }

        // Fetch borrow status
        try {
          const borrowData = await fetch(`/api/scanner/borrow-status/${activeSymbol}`).then(r => r.json()) as any
          if (borrowData) {
            setAssetStatus({
              shortable: borrowData.shortable !== false,
              etb: borrowData.status === 'ETB' || borrowData.etb === true,
              htb: borrowData.status === 'HTB' || borrowData.htb === true,
              marginable: borrowData.marginable !== false,
            })
          }
        } catch {
          // Borrow status not available
        }
      } catch (err) {
        console.error('Failed to fetch quote:', err)
      }
    }

    fetchData()
    const interval = setInterval(fetchData, 1000)
    return () => clearInterval(interval)
  }, [activeSymbol, setQuote, haltInfo.isHalted])

  const formatVolume = (vol: number) => {
    if (vol >= 1000000) return (vol / 1000000).toFixed(1) + 'M'
    if (vol >= 1000) return (vol / 1000).toFixed(1) + 'K'
    return vol.toString()
  }

  const changeColor = (quote?.change || 0) >= 0 ? 'text-up' : 'text-down'
  const changeSign = (quote?.change || 0) >= 0 ? '+' : ''

  return (
    <div className="h-full flex flex-col bg-sterling-panel font-mono text-xs">
      {/* Header */}
      <div className="flex items-center justify-between px-2 py-1 bg-sterling-header border-b border-sterling-border flex-shrink-0">
        <span className="font-bold text-sterling-text">QUOTE</span>
        <span className="font-bold text-accent-primary">{activeSymbol}</span>
      </div>

      {/* Main Quote Display - Bid/Last/Ask */}
      <div className="grid grid-cols-3 gap-1.5 p-2">
        {/* Bid */}
        <div className="text-center">
          <div className="text-[9px] text-[#666] mb-0.5">BID</div>
          <div className="text-[13px] font-bold text-down">
            {(quote?.bid || 0).toFixed(2)}
          </div>
          <div className="text-[9px] text-[#666]">
            {extendedData.bidSize.toLocaleString()}
          </div>
        </div>

        {/* Last */}
        <div className="text-center">
          <div className="text-[9px] text-[#666] mb-0.5">LAST</div>
          <div className="text-[13px] font-bold text-accent-primary">
            {(quote?.last || quote?.price || 0).toFixed(2)}
          </div>
          <div className={`text-[9px] ${changeColor}`}>
            {changeSign}{(quote?.change || 0).toFixed(2)} ({changeSign}{(quote?.changePercent || 0).toFixed(2)}%)
          </div>
        </div>

        {/* Ask */}
        <div className="text-center">
          <div className="text-[9px] text-[#666] mb-0.5">ASK</div>
          <div className="text-[13px] font-bold text-up">
            {(quote?.ask || 0).toFixed(2)}
          </div>
          <div className="text-[9px] text-[#666]">
            {extendedData.askSize.toLocaleString()}
          </div>
        </div>
      </div>

      {/* Secondary Stats - Vol/High/Low/Open */}
      <div className="grid grid-cols-4 gap-1 px-2 pb-2 border-t border-[#333] pt-2">
        <div className="text-center">
          <div className="text-[10px] text-[#666]">VOL</div>
          <div className="text-[12px] font-semibold text-[#aaa]">
            {formatVolume(extendedData.volume)}
          </div>
        </div>
        <div className="text-center">
          <div className="text-[10px] text-[#666]">HIGH</div>
          <div className="text-[12px] font-semibold text-[#aaa]">
            {extendedData.high.toFixed(2)}
          </div>
        </div>
        <div className="text-center">
          <div className="text-[10px] text-[#666]">LOW</div>
          <div className="text-[12px] font-semibold text-[#aaa]">
            {extendedData.low.toFixed(2)}
          </div>
        </div>
        <div className="text-center">
          <div className="text-[10px] text-[#666]">OPEN</div>
          <div className="text-[12px] font-semibold text-[#aaa]">
            {extendedData.open.toFixed(2)}
          </div>
        </div>
      </div>

      {/* Halt Indicator */}
      {haltInfo.isHalted && (
        <div className="mx-2 mb-2 p-1 rounded text-center bg-[#dc2626] text-white text-[11px]">
          <span className="font-bold">🛑 HALT</span>
          {haltInfo.haltTime && (
            <span className="text-[9px] ml-1.5">{haltInfo.haltTime}</span>
          )}
          {haltInfo.reason && (
            <span className="text-[9px] ml-1.5">- {haltInfo.reason}</span>
          )}
        </div>
      )}

      {/* Asset Status Indicators */}
      <div className="flex gap-1.5 px-2 pb-2 flex-wrap">
        {assetStatus.shortable && (
          <div className="px-1.5 py-0.5 rounded text-[10px] font-semibold bg-[#134e4a] text-up">
            SHORTABLE
          </div>
        )}
        {assetStatus.etb && (
          <div className="px-1.5 py-0.5 rounded text-[10px] font-semibold bg-[#1e3a5f] text-accent-primary">
            ETB
          </div>
        )}
        {assetStatus.htb && (
          <div className="px-1.5 py-0.5 rounded text-[10px] font-semibold bg-[#7f1d1d] text-down">
            HTB
          </div>
        )}
        {assetStatus.marginable && (
          <div className="px-1.5 py-0.5 rounded text-[10px] font-semibold bg-[#3f3f46] text-[#a1a1aa]">
            MARGIN
          </div>
        )}
      </div>

      {/* Spread Display */}
      <div className="px-2 pb-2 text-center">
        <span className="text-[10px] text-[#666]">SPREAD: </span>
        <span className="text-[11px] font-semibold text-warning">
          ${((quote?.ask || 0) - (quote?.bid || 0)).toFixed(2)}
          {quote?.bid && quote?.ask && quote.bid > 0 && (
            <span className="text-[9px] ml-1">
              ({(((quote.ask - quote.bid) / quote.bid) * 100).toFixed(2)}%)
            </span>
          )}
        </span>
      </div>
    </div>
  )
}
