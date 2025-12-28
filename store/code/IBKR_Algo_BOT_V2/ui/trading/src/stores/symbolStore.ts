import { create } from 'zustand'

// BroadcastChannel for cross-window sync
const channel = typeof window !== 'undefined' ? new BroadcastChannel('morpheus_trading') : null

interface SymbolState {
  activeSymbol: string
  symbolHistory: string[]
  newsFilterSymbol: string | null  // Filter Breaking News to this symbol
  setActiveSymbol: (symbol: string) => void
  addToHistory: (symbol: string) => void
  setNewsFilter: (symbol: string | null) => void
}

export const useSymbolStore = create<SymbolState>((set, get) => ({
  activeSymbol: 'TSLA',
  symbolHistory: [],
  newsFilterSymbol: null,

  setActiveSymbol: (symbol: string) => {
    const upperSymbol = symbol.toUpperCase()
    set({ activeSymbol: upperSymbol })
    get().addToHistory(upperSymbol)

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
