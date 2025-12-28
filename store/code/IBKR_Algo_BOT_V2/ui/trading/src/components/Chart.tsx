import { useEffect, useRef, useState } from 'react'
import { createChart, IChartApi, ISeriesApi, CandlestickData, Time, LineData, HistogramData } from 'lightweight-charts'
import { useSymbolStore } from '../stores/symbolStore'

// Calculate EMA
function calculateEMA(data: number[], period: number): number[] {
  const ema: number[] = []
  const multiplier = 2 / (period + 1)

  // Start with SMA for first value
  let sum = 0
  for (let i = 0; i < period && i < data.length; i++) {
    sum += data[i]
  }
  ema[period - 1] = sum / period

  // Calculate EMA for rest
  for (let i = period; i < data.length; i++) {
    ema[i] = (data[i] - ema[i - 1]) * multiplier + ema[i - 1]
  }

  return ema
}

// Calculate MACD
function calculateMACD(closes: number[]): { macd: number[]; signal: number[]; histogram: number[] } {
  const ema12 = calculateEMA(closes, 12)
  const ema26 = calculateEMA(closes, 26)

  const macdLine: number[] = []
  for (let i = 0; i < closes.length; i++) {
    if (ema12[i] !== undefined && ema26[i] !== undefined) {
      macdLine[i] = ema12[i] - ema26[i]
    }
  }

  // Signal line is 9-period EMA of MACD
  const validMacd = macdLine.filter(v => v !== undefined)
  const signalEma = calculateEMA(validMacd, 9)

  const signal: number[] = []
  const histogram: number[] = []
  let signalIdx = 0

  for (let i = 0; i < closes.length; i++) {
    if (macdLine[i] !== undefined) {
      if (signalIdx >= 8) {
        signal[i] = signalEma[signalIdx]
        histogram[i] = macdLine[i] - signal[i]
      }
      signalIdx++
    }
  }

  return { macd: macdLine, signal, histogram }
}

// Calculate VWAP
function calculateVWAP(bars: any[]): number[] {
  const vwap: number[] = []
  let cumulativeTPV = 0
  let cumulativeVolume = 0

  for (let i = 0; i < bars.length; i++) {
    const typicalPrice = (bars[i].high + bars[i].low + bars[i].close) / 3
    const volume = bars[i].volume || 0
    cumulativeTPV += typicalPrice * volume
    cumulativeVolume += volume
    vwap[i] = cumulativeVolume > 0 ? cumulativeTPV / cumulativeVolume : typicalPrice
  }

  return vwap
}

interface BarData {
  time: Time
  open: number
  high: number
  low: number
  close: number
  volume: number
}

export default function Chart() {
  const { activeSymbol } = useSymbolStore()
  const containerRef = useRef<HTMLDivElement>(null)
  const chartContainerRef = useRef<HTMLDivElement>(null)
  const macdContainerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<IChartApi | null>(null)
  const macdChartRef = useRef<IChartApi | null>(null)
  const seriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null)
  const volumeSeriesRef = useRef<ISeriesApi<'Histogram'> | null>(null)
  const ema9SeriesRef = useRef<ISeriesApi<'Line'> | null>(null)
  const ema20SeriesRef = useRef<ISeriesApi<'Line'> | null>(null)
  const vwapSeriesRef = useRef<ISeriesApi<'Line'> | null>(null)
  const macdLineRef = useRef<ISeriesApi<'Line'> | null>(null)
  const signalLineRef = useRef<ISeriesApi<'Line'> | null>(null)
  const macdHistRef = useRef<ISeriesApi<'Histogram'> | null>(null)
  const isSyncingRef = useRef(false) // Prevent infinite sync loops
  const [timeframe, setTimeframe] = useState<'1' | '5' | '15' | '60'>('1')
  const [showIndicators, setShowIndicators] = useState({ ema9: true, ema20: true, vwap: true, macd: true })

  // Initialize main chart
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
        scaleMargins: { top: 0.05, bottom: 0.25 }, // Leave room for volume
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

    // Candlestick series
    const candleSeries = chart.addCandlestickSeries({
      upColor: '#00cc00',
      downColor: '#ff3333',
      borderUpColor: '#00cc00',
      borderDownColor: '#ff3333',
      wickUpColor: '#00cc00',
      wickDownColor: '#ff3333',
    })

    // Volume histogram (at bottom of price chart)
    const volumeSeries = chart.addHistogramSeries({
      color: '#26a69a',
      priceFormat: { type: 'volume' },
      priceScaleId: 'volume',
    })
    chart.priceScale('volume').applyOptions({
      scaleMargins: { top: 0.85, bottom: 0 },
    })

    // 9 EMA (blue)
    const ema9Series = chart.addLineSeries({
      color: '#2196f3',
      lineWidth: 1,
      priceLineVisible: false,
      lastValueVisible: false,
    })

    // 20 EMA (cyan)
    const ema20Series = chart.addLineSeries({
      color: '#00bcd4',
      lineWidth: 1,
      priceLineVisible: false,
      lastValueVisible: false,
    })

    // VWAP (yellow)
    const vwapSeries = chart.addLineSeries({
      color: '#ffeb3b',
      lineWidth: 2,
      priceLineVisible: false,
      lastValueVisible: false,
    })

    chartRef.current = chart
    seriesRef.current = candleSeries
    volumeSeriesRef.current = volumeSeries
    ema9SeriesRef.current = ema9Series
    ema20SeriesRef.current = ema20Series
    vwapSeriesRef.current = vwapSeries

    // ResizeObserver for container resize
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

  // Initialize MACD chart
  useEffect(() => {
    if (!macdContainerRef.current || !showIndicators.macd) return

    const macdChart = createChart(macdContainerRef.current, {
      width: macdContainerRef.current.clientWidth,
      height: macdContainerRef.current.clientHeight,
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
        visible: true,
      },
      timeScale: {
        borderColor: '#2a2a2a',
        timeVisible: false,
        visible: false, // Hide - synced with main chart
      },
    })

    // MACD histogram
    const macdHist = macdChart.addHistogramSeries({
      color: '#26a69a',
      priceFormat: { type: 'price', precision: 4, minMove: 0.0001 },
    })

    // MACD line (red)
    const macdLine = macdChart.addLineSeries({
      color: '#ef5350',
      lineWidth: 1,
      priceLineVisible: false,
      lastValueVisible: false,
    })

    // Signal line (orange)
    const signalLine = macdChart.addLineSeries({
      color: '#ff9800',
      lineWidth: 1,
      priceLineVisible: false,
      lastValueVisible: false,
    })

    macdChartRef.current = macdChart
    macdHistRef.current = macdHist
    macdLineRef.current = macdLine
    signalLineRef.current = signalLine

    // ResizeObserver
    const resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect
        if (macdChartRef.current && width > 0 && height > 0) {
          macdChartRef.current.applyOptions({ width, height })
        }
      }
    })

    resizeObserver.observe(macdContainerRef.current)

    // Sync time scales between main chart and MACD chart using TIME range (not logical range)
    const syncTimeScales = () => {
      if (!chartRef.current || !macdChartRef.current) return

      // Main chart controls MACD chart - sync by actual time
      chartRef.current.timeScale().subscribeVisibleTimeRangeChange((timeRange) => {
        if (isSyncingRef.current || !timeRange || !macdChartRef.current) return
        isSyncingRef.current = true
        try {
          macdChartRef.current.timeScale().setVisibleRange(timeRange)
        } catch { /* ignore if range is invalid */ }
        isSyncingRef.current = false
      })

      // MACD chart controls main chart (bidirectional sync)
      macdChart.timeScale().subscribeVisibleTimeRangeChange((timeRange) => {
        if (isSyncingRef.current || !timeRange || !chartRef.current) return
        isSyncingRef.current = true
        try {
          chartRef.current.timeScale().setVisibleRange(timeRange)
        } catch { /* ignore if range is invalid */ }
        isSyncingRef.current = false
      })
    }

    // Small delay to ensure both charts are ready
    setTimeout(syncTimeScales, 100)

    return () => {
      resizeObserver.disconnect()
      macdChart.remove()
      macdChartRef.current = null
    }
  }, [showIndicators.macd])

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
          // Parse bars
          const bars: BarData[] = data.bars.map((bar: any) => ({
            time: (new Date(bar.t || bar.timestamp).getTime() / 1000) as Time,
            open: bar.o || bar.open,
            high: bar.h || bar.high,
            low: bar.l || bar.low,
            close: bar.c || bar.close,
            volume: bar.v || bar.volume || 0,
          }))

          if (bars.length === 0) return

          // Set candlestick data
          const candles: CandlestickData<Time>[] = bars.map(b => ({
            time: b.time,
            open: b.open,
            high: b.high,
            low: b.low,
            close: b.close,
          }))
          seriesRef.current?.setData(candles)

          // Set volume data with up/down colors
          const volumeData: HistogramData<Time>[] = bars.map((b) => ({
            time: b.time,
            value: b.volume,
            color: b.close >= b.open ? '#26a69a80' : '#ef535080', // Green/Red with transparency
          }))
          volumeSeriesRef.current?.setData(volumeData)

          // Calculate and set EMAs
          const closes = bars.map(b => b.close)
          const times = bars.map(b => b.time)

          // 9 EMA
          if (showIndicators.ema9) {
            const ema9 = calculateEMA(closes, 9)
            const ema9Data: LineData<Time>[] = []
            for (let i = 0; i < bars.length; i++) {
              if (ema9[i] !== undefined) {
                ema9Data.push({ time: times[i], value: ema9[i] })
              }
            }
            ema9SeriesRef.current?.setData(ema9Data)
          }

          // 20 EMA
          if (showIndicators.ema20) {
            const ema20 = calculateEMA(closes, 20)
            const ema20Data: LineData<Time>[] = []
            for (let i = 0; i < bars.length; i++) {
              if (ema20[i] !== undefined) {
                ema20Data.push({ time: times[i], value: ema20[i] })
              }
            }
            ema20SeriesRef.current?.setData(ema20Data)
          }

          // VWAP
          if (showIndicators.vwap) {
            const vwapValues = calculateVWAP(bars)
            const vwapData: LineData<Time>[] = bars.map((b, i) => ({
              time: b.time,
              value: vwapValues[i],
            }))
            vwapSeriesRef.current?.setData(vwapData)
          }

          // MACD
          if (showIndicators.macd && macdChartRef.current) {
            const { macd, signal, histogram } = calculateMACD(closes)

            const macdLineData: LineData<Time>[] = []
            const signalLineData: LineData<Time>[] = []
            const histData: HistogramData<Time>[] = []

            for (let i = 0; i < bars.length; i++) {
              if (macd[i] !== undefined) {
                macdLineData.push({ time: times[i], value: macd[i] })
              }
              if (signal[i] !== undefined) {
                signalLineData.push({ time: times[i], value: signal[i] })
              }
              if (histogram[i] !== undefined) {
                histData.push({
                  time: times[i],
                  value: histogram[i],
                  color: histogram[i] >= 0 ? '#26a69a' : '#ef5350',
                })
              }
            }

            macdLineRef.current?.setData(macdLineData)
            signalLineRef.current?.setData(signalLineData)
            macdHistRef.current?.setData(histData)
            macdChartRef.current?.timeScale().fitContent()
          }

          chartRef.current?.timeScale().fitContent()
        }
      } catch (err) {
        console.error('Failed to load chart data:', err)
      }
    }

    fetchData()
    const interval = setInterval(fetchData, 30000) // Refresh every 30 seconds
    return () => clearInterval(interval)
  }, [activeSymbol, timeframe, showIndicators])

  const toggleIndicator = (key: keyof typeof showIndicators) => {
    setShowIndicators(prev => ({ ...prev, [key]: !prev[key] }))
  }

  return (
    <div ref={containerRef} className="h-full w-full flex flex-col bg-sterling-panel overflow-hidden">
      {/* Header with timeframe and indicator buttons */}
      <div className="flex items-center justify-between px-2 py-1 bg-sterling-header border-b border-sterling-border flex-shrink-0">
        <div className="flex items-center gap-2">
          <span className="font-bold text-xs text-sterling-text">CHART</span>
          <span className="font-bold text-xs text-accent-primary">{activeSymbol}</span>
        </div>

        {/* Timeframe buttons */}
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

        {/* Indicator toggles */}
        <div className="flex gap-1">
          <button
            onClick={() => toggleIndicator('ema9')}
            className={`px-1.5 py-0.5 rounded text-xxs font-bold ${
              showIndicators.ema9
                ? 'bg-[#2196f3] text-white'
                : 'bg-sterling-bg text-sterling-muted hover:bg-sterling-highlight'
            }`}
            title="9 EMA"
          >
            9
          </button>
          <button
            onClick={() => toggleIndicator('ema20')}
            className={`px-1.5 py-0.5 rounded text-xxs font-bold ${
              showIndicators.ema20
                ? 'bg-[#00bcd4] text-black'
                : 'bg-sterling-bg text-sterling-muted hover:bg-sterling-highlight'
            }`}
            title="20 EMA"
          >
            20
          </button>
          <button
            onClick={() => toggleIndicator('vwap')}
            className={`px-1.5 py-0.5 rounded text-xxs font-bold ${
              showIndicators.vwap
                ? 'bg-[#ffeb3b] text-black'
                : 'bg-sterling-bg text-sterling-muted hover:bg-sterling-highlight'
            }`}
            title="VWAP"
          >
            V
          </button>
          <button
            onClick={() => toggleIndicator('macd')}
            className={`px-1.5 py-0.5 rounded text-xxs font-bold ${
              showIndicators.macd
                ? 'bg-[#ef5350] text-white'
                : 'bg-sterling-bg text-sterling-muted hover:bg-sterling-highlight'
            }`}
            title="MACD"
          >
            M
          </button>
        </div>
      </div>

      {/* Main chart container */}
      <div
        ref={chartContainerRef}
        className="w-full min-h-0"
        style={{ position: 'relative', flex: showIndicators.macd ? '3' : '1' }}
      />

      {/* MACD chart container */}
      {showIndicators.macd && (
        <div
          ref={macdContainerRef}
          className="w-full border-t border-sterling-border"
          style={{ position: 'relative', flex: '1', minHeight: '80px' }}
        />
      )}

      {/* Legend */}
      <div className="flex items-center gap-3 px-2 py-0.5 bg-sterling-header border-t border-sterling-border text-xxs flex-shrink-0">
        <span className="text-sterling-muted">Vol</span>
        {showIndicators.ema9 && <span className="text-[#2196f3]">EMA9</span>}
        {showIndicators.ema20 && <span className="text-[#00bcd4]">EMA20</span>}
        {showIndicators.vwap && <span className="text-[#ffeb3b]">VWAP</span>}
        {showIndicators.macd && (
          <>
            <span className="text-[#ef5350]">MACD</span>
            <span className="text-[#ff9800]">Signal</span>
          </>
        )}
      </div>
    </div>
  )
}
