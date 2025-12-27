type MessageHandler = (data: any) => void

interface WebSocketConfig {
  url: string
  reconnectAttempts?: number
  reconnectDelay?: number
}

class WebSocketService {
  private ws: WebSocket | null = null
  private handlers: Map<string, Set<MessageHandler>> = new Map()
  private reconnectAttempts = 0
  private maxReconnectAttempts = 5
  private reconnectDelay = 2000
  private isConnecting = false
  private url: string

  constructor(config: WebSocketConfig) {
    this.url = config.url
    this.maxReconnectAttempts = config.reconnectAttempts ?? 5
    this.reconnectDelay = config.reconnectDelay ?? 2000
  }

  connect(): Promise<void> {
    if (this.isConnecting || this.ws?.readyState === WebSocket.OPEN) {
      return Promise.resolve()
    }

    this.isConnecting = true

    return new Promise((resolve, reject) => {
      try {
        this.ws = new WebSocket(this.url)

        this.ws.onopen = () => {
          console.log('[WS] Connected to', this.url)
          this.isConnecting = false
          this.reconnectAttempts = 0
          resolve()
        }

        this.ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data)
            const type = data.type || 'message'
            const handlers = this.handlers.get(type)
            handlers?.forEach((handler) => handler(data))

            // Also call 'all' handlers
            const allHandlers = this.handlers.get('all')
            allHandlers?.forEach((handler) => handler(data))
          } catch (err) {
            console.error('[WS] Failed to parse message:', err)
          }
        }

        this.ws.onclose = () => {
          console.log('[WS] Disconnected')
          this.isConnecting = false
          this.attemptReconnect()
        }

        this.ws.onerror = (error) => {
          console.error('[WS] Error:', error)
          this.isConnecting = false
          reject(error)
        }
      } catch (err) {
        this.isConnecting = false
        reject(err)
      }
    })
  }

  private attemptReconnect() {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('[WS] Max reconnect attempts reached')
      return
    }

    this.reconnectAttempts++
    const delay = this.reconnectDelay * this.reconnectAttempts

    console.log(`[WS] Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`)

    setTimeout(() => {
      this.connect().catch(console.error)
    }, delay)
  }

  subscribe(type: string, handler: MessageHandler) {
    if (!this.handlers.has(type)) {
      this.handlers.set(type, new Set())
    }
    this.handlers.get(type)!.add(handler)

    // Return unsubscribe function
    return () => {
      this.handlers.get(type)?.delete(handler)
    }
  }

  send(data: any) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data))
    } else {
      console.warn('[WS] Cannot send - not connected')
    }
  }

  disconnect() {
    if (this.ws) {
      this.ws.close()
      this.ws = null
    }
  }

  get isConnected() {
    return this.ws?.readyState === WebSocket.OPEN
  }
}

// Create singleton instances
export const marketWS = new WebSocketService({
  url: `ws://${window.location.host}/ws/market`,
  reconnectAttempts: 5,
  reconnectDelay: 2000,
})

export const realtimeWS = new WebSocketService({
  url: `ws://${window.location.host}/ws/realtime`,
  reconnectAttempts: 5,
  reconnectDelay: 2000,
})

// Initialize connections
export async function initWebSockets() {
  try {
    await marketWS.connect()
    await realtimeWS.connect()
    console.log('[WS] All WebSocket connections established')
  } catch (err) {
    console.error('[WS] Failed to initialize WebSockets:', err)
  }
}

export default { marketWS, realtimeWS, initWebSockets }
