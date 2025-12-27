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

  async getAccounts(): Promise<{ accounts: Array<{ accountNumber: string; accountType: string }> }> {
    return this.fetch('/accounts')
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

  // Scanners
  async getHODScanner() {
    return this.fetch('/scanner/hod/alerts')
  }

  async getGappersScanner() {
    return this.fetch('/scanner/gappers/results')
  }

  async getGainersScanner() {
    return this.fetch('/market/movers?direction=up')
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
