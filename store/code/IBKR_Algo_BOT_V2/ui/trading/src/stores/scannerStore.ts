import { create } from 'zustand'

export interface ScannerResult {
  symbol: string
  price: number
  changePercent: number
  direction: 'BREAKING' | 'TESTING' | 'REJECTING' | 'UP' | 'DOWN'
  volume: number
  relVol?: number      // Relative volume (e.g., 2.5x = 250% of avg)
  dayHigh?: number     // HOD scanner
  gapPct?: number      // Gap scanner
  float?: number
  lastNews?: string
  grade?: 'A' | 'B' | 'C'     // Setup quality grade (Task E)
  score?: number              // Setup score 0-100 (Task E)
  execStatus?: 'YES' | 'NO'   // Execution permission (Task E)
  execReason?: string         // Why YES or NO (Task E)
  scanner?: 'GAPPER' | 'GAINER' | 'HOD' | 'SCHWAB' | 'FINVIZ' | string  // Source scanner
}

export type ScannerType = 'hod' | 'gappers' | 'gainers' | 'premarket'

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
    premarket: [],
  },
  isLoading: {
    hod: false,
    gappers: false,
    gainers: false,
    premarket: false,
  },
  lastUpdated: {
    hod: null,
    gappers: null,
    gainers: null,
    premarket: null,
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
