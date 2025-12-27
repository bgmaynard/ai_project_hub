import { useEffect, useState, useCallback } from 'react'
import { usePortfolioStore, Order } from '../stores/portfolioStore'
import { useSymbolStore } from '../stores/symbolStore'
import api from '../services/api'

type OrderTab = 'working' | 'open' | 'filled' | 'closed'

const STATUS_FILTERS: Record<OrderTab, string[]> = {
  working: ['WORKING', 'PENDING_ACTIVATION', 'QUEUED', 'ACCEPTED', 'new', 'accepted', 'pending_new'],
  open: ['WORKING', 'PENDING_ACTIVATION', 'QUEUED', 'new', 'accepted'],
  filled: ['FILLED', 'filled'],
  closed: ['CANCELED', 'EXPIRED', 'REJECTED', 'REPLACED', 'canceled', 'expired', 'rejected'],
}

export default function Orders() {
  const { orders, setOrders, setLoading } = usePortfolioStore()
  const { setActiveSymbol } = useSymbolStore()
  const [activeTab, setActiveTab] = useState<OrderTab>('working')
  const [isRefreshing, setIsRefreshing] = useState(false)
  const [lastSync, setLastSync] = useState<Date | null>(null)

  const fetchOrders = useCallback(async (manual = false) => {
    if (manual) setIsRefreshing(true)
    setLoading(true)
    try {
      const response = await api.getOrders() as any
      // Handle both { orders: [...] } and direct array response
      const data = response?.orders || response || []

      if (Array.isArray(data)) {
        const mapped: Order[] = data.map((o: any) => ({
          id: o.order_id || o.orderId || o.id || '',
          orderId: o.order_id || o.orderId || o.id || '',
          symbol: o.symbol || '',
          side: (o.side || 'BUY').toUpperCase() as 'BUY' | 'SELL',
          quantity: o.quantity || o.qty || 0,
          filledQty: o.filled_qty || o.filledQuantity || o.filledQty || 0,
          price: o.price || o.limit_price || o.limitPrice || 0,
          orderType: (o.order_type || o.orderType || o.type || 'MARKET').toUpperCase(),
          status: (o.status || 'UNKNOWN').toUpperCase(),
          submittedAt: o.entered_time || o.enteredTime || o.submitted_at || o.submittedAt || '',
        }))
        setOrders(mapped)
        setLastSync(new Date())
      }
    } catch (err) {
      console.error('Failed to load orders:', err)
    }
    setLoading(false)
    if (manual) setIsRefreshing(false)
  }, [setOrders, setLoading])

  useEffect(() => {
    fetchOrders()
    const interval = setInterval(() => fetchOrders(false), 3000)
    return () => clearInterval(interval)
  }, [fetchOrders])

  const cancelOrder = async (orderId: string, e: React.MouseEvent) => {
    e.stopPropagation()
    try {
      await api.cancelOrder(orderId)
    } catch (err) {
      console.error('Failed to cancel order:', err)
    }
  }

  const filteredOrders = orders.filter((o) =>
    STATUS_FILTERS[activeTab].some((s) =>
      o.status.toUpperCase().includes(s.toUpperCase())
    )
  )

  return (
    <div className="h-full flex flex-col bg-sterling-panel text-xs">
      {/* Header with tabs */}
      <div className="flex items-center justify-between px-2 py-1 bg-sterling-header border-b border-sterling-border">
        <div className="flex items-center gap-1">
          {(['working', 'open', 'filled', 'closed'] as OrderTab[]).map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`px-2 py-0.5 rounded text-xxs font-bold uppercase ${
                activeTab === tab
                  ? 'bg-accent-primary text-white'
                  : 'bg-sterling-bg text-sterling-muted hover:bg-sterling-highlight'
              }`}
            >
              {tab}
            </button>
          ))}
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => fetchOrders(true)}
            disabled={isRefreshing}
            className={`px-1.5 py-0.5 bg-[#1e3a5f] text-accent-primary text-xxs rounded-sm hover:brightness-110 disabled:opacity-50 ${
              isRefreshing ? 'animate-pulse' : ''
            }`}
            title="Sync from Schwab"
          >
            {isRefreshing ? '...' : '🔄'}
          </button>
          <span className="text-sterling-muted text-xxs">{filteredOrders.length}</span>
          {lastSync && (
            <span className="text-sterling-muted text-xxs">
              {lastSync.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
            </span>
          )}
        </div>
      </div>

      {/* Table */}
      <div className="flex-1 overflow-auto">
        <table className="w-full">
          <thead className="sticky top-0 bg-sterling-header">
            <tr className="text-sterling-muted">
              <th className="px-1 py-1 text-left">Sym</th>
              <th className="px-1 py-1 text-center">Side</th>
              <th className="px-1 py-1 text-right">Qty</th>
              <th className="px-1 py-1 text-right">Price</th>
              <th className="px-1 py-1 text-center">Type</th>
              <th className="px-1 py-1 text-center">Status</th>
              <th className="px-1 py-1 text-center">Action</th>
            </tr>
          </thead>
          <tbody>
            {filteredOrders.length === 0 ? (
              <tr>
                <td colSpan={7} className="text-center py-4 text-sterling-muted">
                  No {activeTab} orders
                </td>
              </tr>
            ) : (
              filteredOrders.map((order) => (
                <tr
                  key={order.id}
                  className="hover:bg-sterling-highlight cursor-pointer border-b border-sterling-border"
                  onClick={() => setActiveSymbol(order.symbol)}
                >
                  <td className="px-1 py-1 font-bold text-accent-primary">
                    {order.symbol}
                  </td>
                  <td className="px-1 py-1 text-center">
                    <span
                      className={`px-1 rounded text-xxs font-bold ${
                        order.side === 'BUY'
                          ? 'bg-buy text-white'
                          : 'bg-sell text-white'
                      }`}
                    >
                      {order.side}
                    </span>
                  </td>
                  <td className="px-1 py-1 text-right">
                    {order.filledQty}/{order.quantity}
                  </td>
                  <td className="px-1 py-1 text-right">
                    {(order.price ?? 0) > 0 ? `$${(order.price ?? 0).toFixed(2)}` : 'MKT'}
                  </td>
                  <td className="px-1 py-1 text-center text-sterling-muted">
                    {order.orderType}
                  </td>
                  <td className="px-1 py-1 text-center">
                    <span
                      className={`text-xxs ${
                        order.status === 'FILLED'
                          ? 'text-up'
                          : order.status.includes('CANCEL') ||
                            order.status.includes('REJECT')
                          ? 'text-down'
                          : 'text-warning'
                      }`}
                    >
                      {order.status}
                    </span>
                  </td>
                  <td className="px-1 py-1 text-center">
                    {(activeTab === 'working' || activeTab === 'open') && (order.id || order.orderId) && (
                      <button
                        onClick={(e) => cancelOrder(order.id || order.orderId || '', e)}
                        className="px-1 py-0.5 bg-sell text-white rounded text-xxs hover:brightness-110"
                      >
                        X
                      </button>
                    )}
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}
