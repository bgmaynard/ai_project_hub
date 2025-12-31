const API_BASE = '/api'

class ApiService {
  private async fetch<T>(endpoint: string, options?: RequestInit): Promise<T> {
    const response = await fetch(`${API_BASE}${endpoint}`, {
      headers: {
        'Content-Type': 'application/json',
        ...options?.headers,
      },
      ...options,
    })

    if (!response.ok) {
      throw new Error(`API Error: ${response.status} ${response.statusText}`)
    }

    return response.json()
  }

  // System
  async getStatus() {
    return this.fetch('/status')
  }

  // Account & Portfolio
  async getAccount() {
    return this.fetch('/account')
  }

  async getAccounts(): Promise<{ accounts: Array<{ accountNumber: string; accountType: string; selected?: boolean }>; selected?: string }> {
    return this.fetch('/accounts')
  }

  async selectAccount(accountNumber: string): Promise<{ success: boolean; selected?: string; error?: string }> {
    return this.fetch(`/accounts/select/${accountNumber}`, { method: 'POST' })
  }

  async getPositions() {
    return this.fetch('/positions')
  }

  async getOrders() {
    return this.fetch('/orders')
  }

  // Market Data
  async getQuote(symbol: string) {
    return this.fetch(`/price/${symbol}`)
  }

  async getLevel2(symbol: string) {
    return this.fetch(`/level2/${symbol}`)
  }

  async getTimeSales(symbol: string) {
    return this.fetch(`/timesales/${symbol}`)
  }

  // Worklist
  async getWorklist() {
    return this.fetch('/worklist')
  }

  async addToWorklist(symbol: string) {
    return this.fetch('/worklist/add', {
      method: 'POST',
      body: JSON.stringify({ symbol }),
    })
  }

  async removeFromWorklist(symbol: string) {
    return this.fetch(`/worklist/${symbol}`, { method: 'DELETE' })
  }

  // =========================================================================
  // MOMENTUM WATCHLIST (Session-scoped, ranked view)
  // =========================================================================

  async getWatchlistStatus() {
    return this.fetch('/watchlist/status')
  }

  async refreshWatchlist() {
    return this.fetch('/watchlist/refresh', { method: 'POST' })
  }

  async purgeWatchlist() {
    return this.fetch('/watchlist/purge', { method: 'POST' })
  }

  async deleteWatchlistSymbol(symbol: string) {
    return this.fetch(`/watchlist/${symbol}`, { method: 'DELETE' })
  }

  async getWatchlistConfig() {
    return this.fetch('/watchlist/config')
  }

  // =========================================================================
  // TASK QUEUE (Discovery Pipeline)
  // =========================================================================

  async runDiscovery() {
    return this.fetch('/task-queue/run', { method: 'POST' })
  }

  async getTaskQueueStatus() {
    return this.fetch('/task-queue/status')
  }

  // =========================================================================
  // WARRIOR TRADING SCANNERS (Schwab-only, time-based)
  // =========================================================================

  async getScannerStatus() {
    return this.fetch('/scanners/status')
  }

  async getScannerConfig() {
    return this.fetch('/scanners/config')
  }

  async getHODScanner() {
    // HOD scanner - active 09:15+ ET
    return this.fetch('/scanners/candidates/hod')
  }

  async getGappersScanner() {
    // Gap scanner - active 04:00-09:15 ET
    return this.fetch('/scanners/candidates/gap')
  }

  async getGainersScanner() {
    // Gainer scanner - active 07:00-09:30 ET
    return this.fetch('/scanners/candidates/gainer')
  }

  async getPreMarketScanner() {
    // Use premarket scanner with news catalysts (fallback)
    return this.fetch('/scanner/premarket/watchlist')
  }

  async getAllScannerCandidates() {
    return this.fetch('/scanners/candidates')
  }

  async runScannerScan(symbols: string[]) {
    return this.fetch('/scanners/scan', {
      method: 'POST',
      body: JSON.stringify({ symbols }),
    })
  }

  async feedScannersToWatchlist() {
    return this.fetch('/scanners/feed-watchlist', { method: 'POST' })
  }

  async discoverCandidates() {
    // Auto-discover from Schwab movers
    return this.fetch('/scanners/discover')
  }

  async discoverAndScan() {
    // Discover movers and add to watchlist
    return this.fetch('/scanners/discover/scan', { method: 'POST' })
  }

  // =========================================================================
  // FINVIZ SCANNER (Free backup when Schwab empty)
  // =========================================================================

  async getFinvizGainers(minChange = 5, maxPrice = 20) {
    return this.fetch(`/scanner/finviz/gainers?min_change=${minChange}&max_price=${maxPrice}`)
  }

  async getFinvizLowFloat(maxFloat = 20) {
    return this.fetch(`/scanner/finviz/low-float?max_float=${maxFloat}`)
  }

  async getFinvizBreakouts() {
    return this.fetch('/scanner/finviz/breakouts')
  }

  async getFinvizAll() {
    return this.fetch('/scanner/finviz/scan-all')
  }

  async syncFinvizToWatchlist(minChange = 5, maxCount = 10) {
    return this.fetch(`/scanner/finviz/sync-to-watchlist?min_change=${minChange}&max_count=${maxCount}`, {
      method: 'POST'
    })
  }

  // News
  async getBreakingNews() {
    return this.fetch('/news-log/today')
  }

  async getNewsForSymbol(symbol: string) {
    return this.fetch(`/news-log/symbol/${symbol}`)
  }

  // AI Predictions
  async getAIPrediction(symbol: string) {
    return this.fetch(`/stock/ai-prediction/${symbol}`)
  }

  async getAICompare(symbol: string) {
    return this.fetch(`/ai/compare/${symbol}`)
  }

  // Scalper
  async getScalperStatus() {
    return this.fetch('/scanner/scalper/status')
  }

  async getScalperStats() {
    return this.fetch('/scanner/scalper/stats')
  }

  // Orders
  async placeOrder(order: {
    symbol: string
    side: 'BUY' | 'SELL'
    quantity: number
    orderType: string
    price?: number
    timeInForce?: string
    extendedHours?: boolean
    accountNumber?: string
  }) {
    return this.fetch('/order', {
      method: 'POST',
      body: JSON.stringify(order),
    })
  }

  async cancelOrder(orderId: string) {
    return this.fetch(`/order/cancel`, {
      method: 'POST',
      body: JSON.stringify({ order_id: orderId }),
    })
  }

  async cancelAllOrders() {
    return this.fetch('/orders/cancel-all', {
      method: 'POST',
    })
  }
}

export const api = new ApiService()
export default api
