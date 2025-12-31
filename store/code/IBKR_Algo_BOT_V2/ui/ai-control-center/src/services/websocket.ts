import type { WebSocketMessage } from '../types/models';

type MessageHandler<T> = (data: T) => void;
type ErrorHandler = (error: Event) => void;
type CloseHandler = () => void;

interface WebSocketConfig {
  url: string;
  reconnectDelay?: number;
  maxReconnectAttempts?: number;
  heartbeatInterval?: number;
}

export class WebSocketService {
  private ws: WebSocket | null = null;
  private config: WebSocketConfig;
  private messageHandlers: Map<string, Set<MessageHandler<any>>> = new Map();
  private errorHandlers: Set<ErrorHandler> = new Set();
  private closeHandlers: Set<CloseHandler> = new Set();
  private reconnectAttempts = 0;
  private reconnectTimer: NodeJS.Timeout | null = null;
  private heartbeatTimer: NodeJS.Timeout | null = null;
  private isIntentionallyClosed = false;

  constructor(config: WebSocketConfig) {
    this.config = {
      reconnectDelay: 3000,
      maxReconnectAttempts: 10,
      heartbeatInterval: 30000,
      ...config
    };
  }

  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        this.isIntentionallyClosed = false;
        this.ws = new WebSocket(this.config.url);

        this.ws.onopen = () => {
          console.log('[WebSocket] Connected:', this.config.url);
          this.reconnectAttempts = 0;
          this.startHeartbeat();
          resolve();
        };

        this.ws.onmessage = (event) => {
          try {
            const message: WebSocketMessage<any> = JSON.parse(event.data);
            this.handleMessage(message);
          } catch (error) {
            console.error('[WebSocket] Failed to parse message:', error);
          }
        };

        this.ws.onerror = (error) => {
          console.error('[WebSocket] Error:', error);
          this.errorHandlers.forEach(handler => handler(error));
          reject(error);
        };

        this.ws.onclose = () => {
          console.log('[WebSocket] Connection closed');
          this.stopHeartbeat();
          this.closeHandlers.forEach(handler => handler());

          if (!this.isIntentionallyClosed) {
            this.attemptReconnect();
          }
        };
      } catch (error) {
        console.error('[WebSocket] Connection failed:', error);
        reject(error);
      }
    });
  }

  disconnect(): void {
    this.isIntentionallyClosed = true;
    this.stopHeartbeat();

    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }

    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  private attemptReconnect(): void {
    if (this.reconnectAttempts >= (this.config.maxReconnectAttempts || 10)) {
      console.error('[WebSocket] Max reconnect attempts reached');
      return;
    }

    this.reconnectAttempts++;
    console.log(`[WebSocket] Reconnecting... Attempt ${this.reconnectAttempts}`);

    this.reconnectTimer = setTimeout(() => {
      this.connect().catch((error) => {
        console.error('[WebSocket] Reconnect failed:', error);
      });
    }, this.config.reconnectDelay);
  }

  private startHeartbeat(): void {
    this.stopHeartbeat();

    this.heartbeatTimer = setInterval(() => {
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.send({ type: 'ping', data: null, timestamp: new Date().toISOString() });
      }
    }, this.config.heartbeatInterval);
  }

  private stopHeartbeat(): void {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }

  private handleMessage(message: WebSocketMessage<any>): void {
    const { type, data } = message;

    // Handle pong response
    if (type === 'pong') {
      return;
    }

    // Call registered handlers for this message type
    const handlers = this.messageHandlers.get(type);
    if (handlers) {
      handlers.forEach(handler => {
        try {
          handler(data);
        } catch (error) {
          console.error(`[WebSocket] Handler error for type "${type}":`, error);
        }
      });
    }

    // Call handlers registered for all messages
    const allHandlers = this.messageHandlers.get('*');
    if (allHandlers) {
      allHandlers.forEach(handler => {
        try {
          handler(message);
        } catch (error) {
          console.error('[WebSocket] Global handler error:', error);
        }
      });
    }
  }

  send<T>(message: WebSocketMessage<T>): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
    } else {
      console.warn('[WebSocket] Cannot send message - not connected');
    }
  }

  on<T>(type: string, handler: MessageHandler<T>): () => void {
    if (!this.messageHandlers.has(type)) {
      this.messageHandlers.set(type, new Set());
    }
    this.messageHandlers.get(type)!.add(handler);

    // Return unsubscribe function
    return () => {
      const handlers = this.messageHandlers.get(type);
      if (handlers) {
        handlers.delete(handler);
      }
    };
  }

  onError(handler: ErrorHandler): () => void {
    this.errorHandlers.add(handler);

    return () => {
      this.errorHandlers.delete(handler);
    };
  }

  onClose(handler: CloseHandler): () => void {
    this.closeHandlers.add(handler);

    return () => {
      this.closeHandlers.delete(handler);
    };
  }

  isConnected(): boolean {
    return this.ws !== null && this.ws.readyState === WebSocket.OPEN;
  }

  getState(): number {
    return this.ws?.readyState ?? WebSocket.CLOSED;
  }
}

// Factory function for creating WebSocket connections
export function createWebSocket(url: string, config?: Partial<WebSocketConfig>): WebSocketService {
  return new WebSocketService({
    url,
    ...config
  });
}

// Get WebSocket URL based on current location
const getWsUrl = (path: string) => {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${protocol}//${window.location.host}${path}`;
};

// Specific WebSocket connections for different channels
export const trainingProgressWebSocket = (trainingId: string) =>
  createWebSocket(getWsUrl(`/api/ai/models/train/progress/${trainingId}`));

export const livePredictionsWebSocket = () =>
  createWebSocket(getWsUrl('/api/predictions/live'));

export const marketDataWebSocket = () =>
  createWebSocket(getWsUrl('/api/market/live'));

export const alertsWebSocket = () =>
  createWebSocket(getWsUrl('/api/alerts/live'));

// ============================================================================
// Governor WebSocket - Real-time health monitoring with auto-reconnect
// ============================================================================

export interface GovernorStatus {
  safe_activation: any;
  scalp_running: boolean;
  polygon: { connected: boolean; healthy: boolean; trades_received?: number };
  connectivity: { system_state: string };
  server_time: string;
  trading_window: { window: string; detail: string; time_et: string };
}

export interface GovernorMessage {
  type: 'connected' | 'heartbeat' | 'status' | 'alert' | 'pong';
  timestamp: string;
  client_id?: string;
  seq?: number;
  server_time?: string;
  data?: GovernorStatus;
  status?: GovernorStatus;
  level?: 'info' | 'warning' | 'error';
  message?: string;
}

type GovernorStatusHandler = (status: GovernorStatus) => void;
type GovernorConnectionHandler = (connected: boolean, reconnecting: boolean) => void;
type GovernorAlertHandler = (level: string, message: string) => void;

class GovernorWebSocketService {
  private ws: WebSocket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 50; // More attempts for critical connection
  private reconnectDelay = 1000; // Start with 1 second
  private maxReconnectDelay = 10000; // Max 10 seconds
  private reconnectTimer: NodeJS.Timeout | null = null;
  private pingTimer: NodeJS.Timeout | null = null;
  private isIntentionallyClosed = false;
  private lastHeartbeat: Date | null = null;
  private heartbeatCheckTimer: NodeJS.Timeout | null = null;

  // Handlers
  private statusHandlers: Set<GovernorStatusHandler> = new Set();
  private connectionHandlers: Set<GovernorConnectionHandler> = new Set();
  private alertHandlers: Set<GovernorAlertHandler> = new Set();

  // Connection state
  public connected = false;
  public reconnecting = false;
  public clientId: string | null = null;
  public lastStatus: GovernorStatus | null = null;

  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        resolve();
        return;
      }

      try {
        this.isIntentionallyClosed = false;
        const wsUrl = getWsUrl('/ws/governor');
        console.log('[Governor WS] Connecting to:', wsUrl);
        this.ws = new WebSocket(wsUrl);

        this.ws.onopen = () => {
          console.log('[Governor WS] Connected');
          this.connected = true;
          this.reconnecting = false;
          this.reconnectAttempts = 0;
          this.reconnectDelay = 1000;
          this.startHeartbeatCheck();
          this.notifyConnectionHandlers();
          resolve();
        };

        this.ws.onmessage = (event) => {
          try {
            const message: GovernorMessage = JSON.parse(event.data);
            this.handleMessage(message);
          } catch (error) {
            console.error('[Governor WS] Failed to parse message:', error);
          }
        };

        this.ws.onerror = (error) => {
          console.error('[Governor WS] Error:', error);
          reject(error);
        };

        this.ws.onclose = (event) => {
          console.log('[Governor WS] Disconnected, code:', event.code);
          this.connected = false;
          this.stopHeartbeatCheck();
          this.notifyConnectionHandlers();

          if (!this.isIntentionallyClosed) {
            this.attemptReconnect();
          }
        };
      } catch (error) {
        console.error('[Governor WS] Connection failed:', error);
        reject(error);
      }
    });
  }

  disconnect(): void {
    this.isIntentionallyClosed = true;
    this.stopHeartbeatCheck();
    this.clearReconnectTimer();

    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }

    this.connected = false;
    this.reconnecting = false;
    this.notifyConnectionHandlers();
  }

  private attemptReconnect(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('[Governor WS] Max reconnect attempts reached');
      this.reconnecting = false;
      this.notifyConnectionHandlers();
      // Notify with alert
      this.alertHandlers.forEach(h => h('error', 'Connection lost - max reconnect attempts reached'));
      return;
    }

    this.reconnecting = true;
    this.reconnectAttempts++;
    this.notifyConnectionHandlers();

    // Exponential backoff with jitter
    const delay = Math.min(
      this.reconnectDelay * Math.pow(1.5, this.reconnectAttempts - 1) + Math.random() * 500,
      this.maxReconnectDelay
    );

    console.log(`[Governor WS] Reconnecting in ${Math.round(delay)}ms (attempt ${this.reconnectAttempts})`);

    this.reconnectTimer = setTimeout(() => {
      this.connect().catch((error) => {
        console.error('[Governor WS] Reconnect failed:', error);
      });
    }, delay);
  }

  private clearReconnectTimer(): void {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
  }

  private startHeartbeatCheck(): void {
    this.stopHeartbeatCheck();

    // Check heartbeat every 5 seconds - if no heartbeat in 10 seconds, reconnect
    this.heartbeatCheckTimer = setInterval(() => {
      if (this.lastHeartbeat) {
        const age = Date.now() - this.lastHeartbeat.getTime();
        if (age > 10000) {
          console.warn('[Governor WS] Heartbeat timeout - reconnecting');
          this.ws?.close();
        }
      }
    }, 5000);

    // Send ping every 10 seconds
    this.pingTimer = setInterval(() => {
      this.sendCommand('ping', { seq: Date.now() });
    }, 10000);
  }

  private stopHeartbeatCheck(): void {
    if (this.heartbeatCheckTimer) {
      clearInterval(this.heartbeatCheckTimer);
      this.heartbeatCheckTimer = null;
    }
    if (this.pingTimer) {
      clearInterval(this.pingTimer);
      this.pingTimer = null;
    }
  }

  private handleMessage(message: GovernorMessage): void {
    switch (message.type) {
      case 'connected':
        this.clientId = message.client_id || null;
        if (message.status) {
          this.lastStatus = message.status;
          this.notifyStatusHandlers(message.status);
        }
        break;

      case 'heartbeat':
        this.lastHeartbeat = new Date();
        break;

      case 'status':
        if (message.data) {
          this.lastStatus = message.data;
          this.notifyStatusHandlers(message.data);
        }
        break;

      case 'alert':
        if (message.level && message.message) {
          this.alertHandlers.forEach(h => h(message.level!, message.message!));
        }
        break;

      case 'pong':
        // Update heartbeat on pong as well
        this.lastHeartbeat = new Date();
        break;
    }
  }

  private notifyStatusHandlers(status: GovernorStatus): void {
    this.statusHandlers.forEach(h => {
      try {
        h(status);
      } catch (e) {
        console.error('[Governor WS] Status handler error:', e);
      }
    });
  }

  private notifyConnectionHandlers(): void {
    this.connectionHandlers.forEach(h => {
      try {
        h(this.connected, this.reconnecting);
      } catch (e) {
        console.error('[Governor WS] Connection handler error:', e);
      }
    });
  }

  // Public API
  sendCommand(command: string, data?: any): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ command, ...data }));
    }
  }

  requestStatus(): void {
    this.sendCommand('status');
  }

  reconnectFeeds(): void {
    this.sendCommand('reconnect_feeds');
  }

  onStatus(handler: GovernorStatusHandler): () => void {
    this.statusHandlers.add(handler);
    // Send last status immediately if available
    if (this.lastStatus) {
      handler(this.lastStatus);
    }
    return () => this.statusHandlers.delete(handler);
  }

  onConnection(handler: GovernorConnectionHandler): () => void {
    this.connectionHandlers.add(handler);
    // Notify immediately with current state
    handler(this.connected, this.reconnecting);
    return () => this.connectionHandlers.delete(handler);
  }

  onAlert(handler: GovernorAlertHandler): () => void {
    this.alertHandlers.add(handler);
    return () => this.alertHandlers.delete(handler);
  }

  getConnectionInfo(): { connected: boolean; reconnecting: boolean; attempts: number; clientId: string | null } {
    return {
      connected: this.connected,
      reconnecting: this.reconnecting,
      attempts: this.reconnectAttempts,
      clientId: this.clientId
    };
  }
}

// Singleton instance for Governor WebSocket
let governorWsInstance: GovernorWebSocketService | null = null;

export function getGovernorWebSocket(): GovernorWebSocketService {
  if (!governorWsInstance) {
    governorWsInstance = new GovernorWebSocketService();
  }
  return governorWsInstance;
}

export { GovernorWebSocketService };

export default WebSocketService;
