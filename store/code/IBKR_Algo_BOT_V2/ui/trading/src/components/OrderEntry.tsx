import { useState, useEffect } from 'react'
import { useSymbolStore } from '../stores/symbolStore'
import { useMarketDataStore } from '../stores/marketDataStore'
import api from '../services/api'

type OrderType = 'MARKET' | 'LIMIT' | 'STOP' | 'STOP_LIMIT'
type TimeInForce = 'DAY' | 'GTC' | 'IOC' | 'FOK'

interface Account {
  accountNumber: string
  accountType: string
}

export default function OrderEntry() {
  const { activeSymbol } = useSymbolStore()
  const { quotes } = useMarketDataStore()
  const quote = quotes[activeSymbol]

  const [quantity, setQuantity] = useState<number>(100)
  const [orderType, setOrderType] = useState<OrderType>('LIMIT')
  const [price, setPrice] = useState<string>('')
  const [timeInForce, setTimeInForce] = useState<TimeInForce>('DAY')
  const [extendedHours, setExtendedHours] = useState<boolean>(true)
  const [autoSubmit, setAutoSubmit] = useState<boolean>(false)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [accounts, setAccounts] = useState<Account[]>([])
  const [selectedAccount, setSelectedAccount] = useState<string>('')

  // Load accounts on mount
  useEffect(() => {
    const loadAccounts = async () => {
      try {
        const data = await api.getAccounts()
        if (data?.accounts) {
          setAccounts(data.accounts)
          if (data.accounts.length > 0) {
            setSelectedAccount(data.accounts[0].accountNumber)
          }
        }
      } catch (err) {
        console.error('Failed to load accounts:', err)
      }
    }
    loadAccounts()
  }, [])

  // Calculate estimated cost
  const calcPrice = parseFloat(price) || quote?.last || 0
  const estTotal = quantity * calcPrice

  const adjustPrice = (delta: number) => {
    const current = parseFloat(price) || quote?.last || 0
    setPrice((current + delta).toFixed(2))
  }

  const submitOrder = async (side: 'BUY' | 'SELL', priceType?: 'BID' | 'ASK' | 'MARKET') => {
    if (!activeSymbol || quantity <= 0) return

    let orderPrice = parseFloat(price)
    let type = orderType

    if (priceType === 'MARKET') {
      type = 'MARKET'
      orderPrice = 0
    } else if (priceType === 'BID' && quote) {
      orderPrice = quote.bid || 0
    } else if (priceType === 'ASK' && quote) {
      orderPrice = quote.ask || 0
    }

    setIsSubmitting(true)
    try {
      await api.placeOrder({
        symbol: activeSymbol,
        side,
        quantity,
        orderType: type,
        price: type !== 'MARKET' ? orderPrice : undefined,
        timeInForce,
        extendedHours,
        accountNumber: selectedAccount,
      })
    } catch (err) {
      console.error('Order failed:', err)
    }
    setIsSubmitting(false)
  }

  const cancelAllOrders = async () => {
    try {
      await api.cancelAllOrders()
    } catch (err) {
      console.error('Failed to cancel orders:', err)
    }
  }

  return (
    <div className="h-full flex flex-col bg-sterling-panel overflow-y-auto">
      {/* Header */}
      <div className="flex items-center justify-between px-2 py-1 bg-sterling-header border-b border-sterling-border flex-shrink-0">
        <span className="font-bold text-xs text-sterling-text">ORDER ENTRY</span>
        <span className="font-bold text-xs text-accent-primary">{activeSymbol}</span>
      </div>

      <div className="p-2 space-y-2 text-xs">
        {/* Account Selector */}
        <div>
          <label className="block text-[11px] text-sterling-muted uppercase font-semibold mb-1">
            Trading Account
          </label>
          <select
            value={selectedAccount}
            onChange={(e) => setSelectedAccount(e.target.value)}
            className="w-full px-2 py-1 bg-[#1a1a1a] border border-[#404040] text-white text-[13px] font-mono rounded-sm focus:border-accent-primary focus:outline-none"
          >
            {accounts.map((acc) => (
              <option key={acc.accountNumber} value={acc.accountNumber}>
                {acc.accountNumber} ({acc.accountType})
              </option>
            ))}
          </select>
        </div>

        {/* Quantity & Order Type Grid */}
        <div className="grid grid-cols-2 gap-1">
          <div>
            <label className="block text-[11px] text-sterling-muted uppercase font-semibold mb-1">
              Quantity
            </label>
            <input
              type="number"
              value={quantity}
              onChange={(e) => setQuantity(parseInt(e.target.value) || 0)}
              className="w-full px-2 py-1 bg-[#252525] border border-[#404040] text-white text-[13px] font-mono rounded-sm focus:border-accent-primary focus:outline-none"
            />
          </div>
          <div>
            <label className="block text-[11px] text-sterling-muted uppercase font-semibold mb-1">
              Order Type
            </label>
            <select
              value={orderType}
              onChange={(e) => setOrderType(e.target.value as OrderType)}
              className="w-full px-2 py-1 bg-[#252525] border border-[#404040] text-white text-[13px] font-mono rounded-sm focus:border-accent-primary focus:outline-none"
            >
              <option value="MARKET">Market</option>
              <option value="LIMIT">Limit</option>
              <option value="STOP">Stop</option>
              <option value="STOP_LIMIT">Stop Limit</option>
            </select>
          </div>
        </div>

        {/* Price Controls */}
        {orderType !== 'MARKET' && (
          <div className="grid grid-cols-[30px_1fr_30px] gap-0.5">
            <button
              onClick={() => adjustPrice(-0.01)}
              className="px-1 py-1 bg-[#252525] border border-[#404040] text-accent-primary font-bold text-sm rounded-sm hover:bg-accent-primary hover:text-white"
            >
              −
            </button>
            <div>
              <label className="block text-[11px] text-sterling-muted uppercase font-semibold mb-0.5">
                Limit Price
              </label>
              <input
                type="text"
                value={price}
                onChange={(e) => setPrice(e.target.value)}
                placeholder="Enter limit price"
                className="w-full px-2 py-1 bg-[#252525] border border-[#404040] text-white text-[13px] font-mono rounded-sm focus:border-accent-primary focus:outline-none"
              />
            </div>
            <button
              onClick={() => adjustPrice(0.01)}
              className="px-1 py-1 bg-[#252525] border border-[#404040] text-accent-primary font-bold text-sm rounded-sm hover:bg-accent-primary hover:text-white"
            >
              +
            </button>
          </div>
        )}

        {/* Time In Force */}
        <div>
          <label className="block text-[11px] text-sterling-muted uppercase font-semibold mb-1">
            Time In Force
          </label>
          <select
            value={timeInForce}
            onChange={(e) => setTimeInForce(e.target.value as TimeInForce)}
            className="w-full px-2 py-1 bg-[#252525] border border-[#404040] text-white text-[13px] font-mono rounded-sm focus:border-accent-primary focus:outline-none"
          >
            <option value="DAY">Day + Ext Hrs</option>
            <option value="GTC">GTC</option>
            <option value="IOC">IOC</option>
            <option value="FOK">FOK</option>
          </select>
        </div>

        {/* Checkboxes */}
        <div className="grid grid-cols-2 gap-3">
          <label className="flex items-center gap-1.5 cursor-pointer">
            <input
              type="checkbox"
              checked={extendedHours}
              onChange={(e) => setExtendedHours(e.target.checked)}
              className="w-3 h-3"
            />
            <span className="text-[13px] text-up">Ext Hours</span>
          </label>
          <label className="flex items-center gap-1.5 cursor-pointer">
            <input
              type="checkbox"
              checked={autoSubmit}
              onChange={(e) => setAutoSubmit(e.target.checked)}
              className="w-3 h-3"
            />
            <span className="text-[13px] text-warning">Auto-Submit</span>
          </label>
        </div>

        {/* Order Cost Calculator */}
        <div className="p-2.5 bg-[#1a1a1a] border border-[#333] rounded">
          <div className="text-[11px] text-sterling-muted uppercase tracking-wide mb-1.5">
            Estimated Cost
          </div>
          <div className="grid grid-cols-3 gap-2 text-center">
            <div>
              <div className="text-[10px] text-[#666] mb-0.5">Shares</div>
              <div className="text-[15px] font-semibold text-white">{quantity}</div>
            </div>
            <div>
              <div className="text-[10px] text-[#666] mb-0.5">Price</div>
              <div className="text-[15px] font-semibold text-white">${calcPrice.toFixed(2)}</div>
            </div>
            <div>
              <div className="text-[10px] text-[#666] mb-0.5">Est. Total</div>
              <div className="text-[15px] font-semibold text-up">${estTotal.toFixed(2)}</div>
            </div>
          </div>
        </div>

        {/* Order Buttons */}
        <div className="grid grid-cols-2 gap-2">
          {/* BUY SIDE */}
          <div className="space-y-1.5">
            <button
              onClick={() => submitOrder('BUY')}
              disabled={isSubmitting}
              className="w-full py-1.5 font-semibold text-white rounded-sm"
              style={{
                background: 'linear-gradient(180deg, #0077cc 0%, #004488 100%)',
                border: '1px solid #0088dd',
                textShadow: '0 1px 1px rgba(0,0,0,0.5)',
              }}
            >
              BUY
            </button>
            <button
              onClick={() => submitOrder('BUY', 'MARKET')}
              disabled={isSubmitting}
              className="w-full py-1 text-[12px] bg-[#252525] border border-[#404040] text-sterling-muted rounded-sm hover:bg-[#333]"
            >
              BUY MARKET
            </button>
            <button
              onClick={() => submitOrder('BUY', 'BID')}
              disabled={isSubmitting}
              className="w-full py-1 text-[11px] bg-[#134e4a] text-up border border-transparent rounded-sm hover:brightness-110"
            >
              Buy @ Bid
            </button>
            <button
              onClick={() => submitOrder('BUY', 'ASK')}
              disabled={isSubmitting}
              className="w-full py-1 text-[11px] bg-[#134e4a] text-up border border-transparent rounded-sm hover:brightness-110"
            >
              Buy @ Ask
            </button>
          </div>

          {/* SELL SIDE */}
          <div className="space-y-1.5">
            <button
              onClick={() => submitOrder('SELL')}
              disabled={isSubmitting}
              className="w-full py-1.5 font-semibold text-white rounded-sm"
              style={{
                background: 'linear-gradient(180deg, #cc0000 0%, #880000 100%)',
                border: '1px solid #dd0000',
                textShadow: '0 1px 1px rgba(0,0,0,0.5)',
              }}
            >
              SELL
            </button>
            <button
              onClick={() => submitOrder('SELL', 'MARKET')}
              disabled={isSubmitting}
              className="w-full py-1 text-[12px] bg-[#252525] border border-[#404040] text-sterling-muted rounded-sm hover:bg-[#333]"
            >
              SELL MARKET
            </button>
            <button
              onClick={() => submitOrder('SELL', 'BID')}
              disabled={isSubmitting}
              className="w-full py-1 text-[11px] bg-[#7f1d1d] text-down border border-transparent rounded-sm hover:brightness-110"
            >
              Sell @ Bid
            </button>
            <button
              onClick={() => submitOrder('SELL', 'ASK')}
              disabled={isSubmitting}
              className="w-full py-1 text-[11px] bg-[#7f1d1d] text-down border border-transparent rounded-sm hover:brightness-110"
            >
              Sell @ Ask
            </button>
          </div>
        </div>

        {/* Cancel All */}
        <div className="pt-2 border-t border-[#333]">
          <button
            onClick={cancelAllOrders}
            className="w-full py-1.5 text-[13px] bg-[#7f1d1d] text-down border border-transparent rounded-sm hover:brightness-110"
          >
            CANCEL ALL ORDERS
          </button>
        </div>
      </div>
    </div>
  )
}
