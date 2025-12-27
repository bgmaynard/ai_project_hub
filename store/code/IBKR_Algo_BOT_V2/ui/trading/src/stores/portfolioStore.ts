import { create } from 'zustand'

export interface Position {
  symbol: string
  quantity: number
  avgPrice?: number
  avgCost?: number // alias for avgPrice
  currentPrice?: number
  marketValue?: number
  unrealizedPnL?: number
  unrealizedPnl?: number // alias (lowercase)
  unrealizedPnLPercent?: number
  unrealizedPnlPercent?: number // alias (lowercase)
  dayPnL?: number
}

export interface Order {
  orderId?: string
  id?: string // alias for orderId
  symbol: string
  side: 'BUY' | 'SELL'
  quantity: number
  orderType?: string
  price?: number
  stopPrice?: number
  status: string
  filledQuantity?: number
  filledQty?: number // alias for filledQuantity
  timestamp?: string
  submittedAt?: string
}

export interface AccountInfo {
  accountId?: string
  accountType?: string
  cashBalance?: number
  settledCash?: number // For cash accounts - settled funds available
  buyingPower?: number
  equity?: number
  netLiquidation?: number // alias for equity
  dayPnL?: number
  dayPnl?: number // alias (lowercase)
  unrealizedPnL?: number
  totalPnl?: number // alias for unrealizedPnL
  marginUsed?: number
}

interface PortfolioState {
  positions: Position[]
  orders: Order[]
  account: AccountInfo | null
  isLoading: boolean

  setPositions: (positions: Position[]) => void
  setOrders: (orders: Order[]) => void
  setAccount: (account: AccountInfo) => void
  updatePosition: (symbol: string, updates: Partial<Position>) => void
  updateOrder: (orderId: string, updates: Partial<Order>) => void
  setLoading: (loading: boolean) => void
}

export const usePortfolioStore = create<PortfolioState>((set) => ({
  positions: [],
  orders: [],
  account: null,
  isLoading: false,

  setPositions: (positions) => set({ positions }),
  setOrders: (orders) => set({ orders }),
  setAccount: (account) => set({ account }),
  setLoading: (isLoading) => set({ isLoading }),

  updatePosition: (symbol, updates) => {
    set((state) => ({
      positions: state.positions.map((p) =>
        p.symbol === symbol ? { ...p, ...updates } : p
      )
    }))
  },

  updateOrder: (orderId, updates) => {
    set((state) => ({
      orders: state.orders.map((o) =>
        o.orderId === orderId ? { ...o, ...updates } : o
      )
    }))
  },
}))
