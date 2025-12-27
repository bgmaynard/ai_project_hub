import { create } from 'zustand'

export interface WatchlistItem {
  symbol: string
  price: number
  change: number
  changePercent: number
  volume: number
  rvol: number
  float: number // in millions
  momentum: 'UP' | 'DOWN' | 'FLAT'
  aiScore: number // 0-100
  hasNews: boolean
  lastNews?: string
  fsmState?: string
}

interface WatchlistState {
  items: WatchlistItem[]
  isLoading: boolean
  lastUpdated: string | null

  setItems: (items: WatchlistItem[]) => void
  updateItem: (symbol: string, updates: Partial<WatchlistItem>) => void
  addItem: (item: WatchlistItem) => void
  removeItem: (symbol: string) => void
  setLoading: (loading: boolean) => void
}

export const useWatchlistStore = create<WatchlistState>((set) => ({
  items: [],
  isLoading: false,
  lastUpdated: null,

  setItems: (items) => set({ items, lastUpdated: new Date().toISOString() }),

  updateItem: (symbol, updates) => {
    set((state) => ({
      items: state.items.map((item) =>
        item.symbol === symbol ? { ...item, ...updates } : item
      )
    }))
  },

  addItem: (item) => {
    set((state) => {
      if (state.items.find((i) => i.symbol === item.symbol)) {
        return state // Already exists
      }
      return { items: [...state.items, item] }
    })
  },

  removeItem: (symbol) => {
    set((state) => ({
      items: state.items.filter((item) => item.symbol !== symbol)
    }))
  },

  setLoading: (isLoading) => set({ isLoading }),
}))
