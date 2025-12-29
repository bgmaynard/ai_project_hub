import { create } from 'zustand'

// BroadcastChannel for cross-window sync
const channel = typeof window !== 'undefined' ? new BroadcastChannel('morpheus_trading') : null

// Auto-subscribe to Polygon when symbol changes
async function subscribeToPolygon(symbol: string) {
  try {
    await fetch('/api/polygon/stream/subscribe', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ symbol, data_type: 'trades' })
    })
    console.log(`[SymbolStore] Subscribed ${symbol} to Polygon`)
  } catch (err) {
    console.warn(`[SymbolStore] Failed to subscribe ${symbol} to Polygon:`, err)
  }
}

interface SymbolState {
  activeSymbol: string
  symbolHistory: string[]
  newsFilterSymbol: string | null  // Filter Breaking News to this symbol
  setActiveSymbol: (symbol: string) => void
  addToHistory: (symbol: string) => void
  setNewsFilter: (symbol: string | null) => void
}

export const useSymbolStore = create<SymbolState>((set, get) => ({
  activeSymbol: '',  // Start empty, let user select
  symbolHistory: [],
  newsFilterSymbol: null,

  setActiveSymbol: (symbol: string) => {
    const upperSymbol = symbol.toUpperCase()
    set({ activeSymbol: upperSymbol })
    get().addToHistory(upperSymbol)

    // Auto-subscribe to Polygon for real-time T/S data
    subscribeToPolygon(upperSymbol)

    // Broadcast to other windows
    channel?.postMessage({ type: 'SYMBOL_CHANGE', symbol: upperSymbol })
  },

  addToHistory: (symbol: string) => {
    set((state) => {
      const filtered = state.symbolHistory.filter((s) => s !== symbol)
      return { symbolHistory: [symbol, ...filtered].slice(0, 20) }
    })
  },

  setNewsFilter: (symbol: string | null) => {
    set({ newsFilterSymbol: symbol })
  },
}))

// Listen for cross-window messages
if (channel) {
  channel.onmessage = (event) => {
    if (event.data.type === 'SYMBOL_CHANGE') {
      useSymbolStore.setState({ activeSymbol: event.data.symbol })
    }
  }
}
