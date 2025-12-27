import { useEffect, useRef, useState } from 'react'
import { createChart, IChartApi, ISeriesApi, CandlestickData, Time } from 'lightweight-charts'
import { useSymbolStore } from '../stores/symbolStore'

export default function Chart() {
  const { activeSymbol } = useSymbolStore()
  const containerRef = useRef<HTMLDivElement>(null)
  const chartContainerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<IChartApi | null>(null)
  const seriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null)
  const [timeframe, setTimeframe] = useState<'1' | '5' | '15' | '60'>('1')

  // Initialize chart
  useEffect(() => {
    if (!chartContainerRef.current) return

    const chart = createChart(chartContainerRef.current, {
      width: chartContainerRef.current.clientWidth,
      height: chartContainerRef.current.clientHeight,
      layout: {
        background: { color: '#000000' },
        textColor: '#c0c0c0',
      },
      grid: {
        vertLines: { color: '#1a1a1a' },
        horzLines: { color: '#1a1a1a' },
      },
      crosshair: {
        mode: 1,
        vertLine: { color: '#555', width: 1, style: 2 },
        horzLine: { color: '#555', width: 1, style: 2 },
      },
      rightPriceScale: {
        borderColor: '#2a2a2a',
        scaleMargins: { top: 0.1, bottom: 0.1 },
        visible: true,
      },
      timeScale: {
        borderColor: '#2a2a2a',
        timeVisible: true,
        secondsVisible: false,
        visible: true,
      },
      handleScale: { axisPressedMouseMove: true },
      handleScroll: { vertTouchDrag: true },
    })

    const candleSeries = chart.addCandlestickSeries({
      upColor: '#00cc00',
      downColor: '#ff3333',
      borderUpColor: '#00cc00',
      borderDownColor: '#ff3333',
      wickUpColor: '#00cc00',
      wickDownColor: '#ff3333',
    })

    chartRef.current = chart
    seriesRef.current = candleSeries

    // Use ResizeObserver for container resize detection (works with Golden Layout)
    const resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect
        if (chartRef.current && width > 0 && height > 0) {
          chartRef.current.applyOptions({ width, height })
        }
      }
    })

    resizeObserver.observe(chartContainerRef.current)

    return () => {
      resizeObserver.disconnect()
      chart.remove()
    }
  }, [])

  // Fetch and update data when symbol or timeframe changes
  useEffect(() => {
    if (!activeSymbol || !seriesRef.current) return

    const fetchData = async () => {
      try {
        const response = await fetch(
          `/api/polygon/bars/${activeSymbol}?timeframe=${timeframe}&limit=200`
        )
        const data = await response.json()

        if (data?.bars && Array.isArray(data.bars)) {
          const candles: CandlestickData<Time>[] = data.bars.map((bar: any) => ({
            time: (new Date(bar.t || bar.timestamp).getTime() / 1000) as Time,
            open: bar.o || bar.open,
            high: bar.h || bar.high,
            low: bar.l || bar.low,
            close: bar.c || bar.close,
          }))

          if (candles.length > 0) {
            seriesRef.current?.setData(candles)
            chartRef.current?.timeScale().fitContent()
          }
        }
      } catch (err) {
        console.error('Failed to load chart data:', err)
      }
    }

    fetchData()
    const interval = setInterval(fetchData, 60000) // Refresh every minute
    return () => clearInterval(interval)
  }, [activeSymbol, timeframe])

  return (
    <div ref={containerRef} className="h-full w-full flex flex-col bg-sterling-panel overflow-hidden">
      {/* Header with timeframe buttons */}
      <div className="flex items-center justify-between px-2 py-1 bg-sterling-header border-b border-sterling-border flex-shrink-0">
        <div className="flex items-center gap-2">
          <span className="font-bold text-xs text-sterling-text">CHART</span>
          <span className="font-bold text-xs text-accent-primary">{activeSymbol}</span>
        </div>
        <div className="flex gap-1">
          {(['1', '5', '15', '60'] as const).map((tf) => (
            <button
              key={tf}
              onClick={() => setTimeframe(tf)}
              className={`px-2 py-0.5 rounded text-xxs font-bold ${
                timeframe === tf
                  ? 'bg-accent-primary text-white'
                  : 'bg-sterling-bg text-sterling-muted hover:bg-sterling-highlight'
              }`}
            >
              {tf === '60' ? '1H' : `${tf}M`}
            </button>
          ))}
        </div>
      </div>

      {/* Chart container - fills remaining space */}
      <div
        ref={chartContainerRef}
        className="flex-1 w-full min-h-0"
        style={{ position: 'relative' }}
      />
    </div>
  )
}
