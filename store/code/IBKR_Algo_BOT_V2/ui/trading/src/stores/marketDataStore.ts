import { create } from 'zustand'

export interface Quote {
  symbol: string
  price?: number
  last?: number // alias for price
  bid?: number
  ask?: number
  bidSize?: number
  askSize?: number
  volume?: number
  change?: number
  changePercent?: number
  high?: number
  low?: number
  open?: number
  timestamp?: string
}

export interface Level2Entry {
  price: number
  size: number
  mmid?: string
}

export interface Level2Data {
  bids: Level2Entry[]
  asks: Level2Entry[]
}

export interface TimeSalesEntry {
  time: string
  price: number
  size: number
  side: 'BUY' | 'SELL' | 'UNKNOWN'
}

interface MarketDataState {
  quotes: Record<string, Quote>
  level2: Record<string, Level2Data>
  timeSales: Record<string, TimeSalesEntry[]>

  updateQuote: (symbol: string, quote: Partial<Quote>) => void
  setQuote: (symbol: string, quote: Quote) => void // alias for updateQuote
  updateLevel2: (symbol: string, data: Level2Data) => void
  addTimeSale: (symbol: string, entry: TimeSalesEntry) => void
  clearTimeSales: (symbol: string) => void
}

export const useMarketDataStore = create<MarketDataState>((set) => ({
  quotes: {},
  level2: {},
  timeSales: {},

  updateQuote: (symbol, quote) => {
    set((state) => ({
      quotes: {
        ...state.quotes,
        [symbol]: { ...state.quotes[symbol], ...quote, symbol } as Quote
      }
    }))
  },

  setQuote: (symbol, quote) => {
    set((state) => ({
      quotes: {
        ...state.quotes,
        [symbol]: quote
      }
    }))
  },

  updateLevel2: (symbol, data) => {
    set((state) => ({
      level2: { ...state.level2, [symbol]: data }
    }))
  },

  addTimeSale: (symbol, entry) => {
    set((state) => {
      const existing = state.timeSales[symbol] || []
      return {
        timeSales: {
          ...state.timeSales,
          [symbol]: [entry, ...existing].slice(0, 100) // Keep last 100
        }
      }
    })
  },

  clearTimeSales: (symbol) => {
    set((state) => ({
      timeSales: { ...state.timeSales, [symbol]: [] }
    }))
  },
}))
