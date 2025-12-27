import { create } from 'zustand'

export interface ScannerResult {
  symbol: string
  price: number
  changePercent: number
  direction: 'BREAKING' | 'TESTING' | 'REJECTING' | 'UP' | 'DOWN'
  volume: number
  float?: number
  lastNews?: string
  grade?: 'A' | 'B' | 'C'
  score?: number
}

export type ScannerType = 'hod' | 'gappers' | 'gainers'

interface ScannerState {
  activeScanner: ScannerType
  results: Record<ScannerType, ScannerResult[]>
  isLoading: Record<ScannerType, boolean>
  lastUpdated: Record<ScannerType, string | null>

  setActiveScanner: (scanner: ScannerType) => void
  setResults: (scanner: ScannerType, results: ScannerResult[]) => void
  setLoading: (scanner: ScannerType, loading: boolean) => void
}

export const useScannerStore = create<ScannerState>((set) => ({
  activeScanner: 'hod',
  results: {
    hod: [],
    gappers: [],
    gainers: [],
  },
  isLoading: {
    hod: false,
    gappers: false,
    gainers: false,
  },
  lastUpdated: {
    hod: null,
    gappers: null,
    gainers: null,
  },

  setActiveScanner: (scanner) => set({ activeScanner: scanner }),

  setResults: (scanner, results) => {
    set((state) => ({
      results: { ...state.results, [scanner]: results },
      lastUpdated: { ...state.lastUpdated, [scanner]: new Date().toISOString() },
    }))
  },

  setLoading: (scanner, loading) => {
    set((state) => ({
      isLoading: { ...state.isLoading, [scanner]: loading },
    }))
  },
}))
