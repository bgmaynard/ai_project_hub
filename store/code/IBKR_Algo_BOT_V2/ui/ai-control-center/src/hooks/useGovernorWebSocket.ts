/**
 * useGovernorWebSocket Hook
 *
 * Real-time WebSocket connection to Governor for:
 * - Heartbeat monitoring (detect disconnects instantly)
 * - Auto-reconnect with exponential backoff
 * - Status updates pushed from server
 * - Trade-safe: prevents slippage from stale data
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { getGovernorWebSocket, GovernorStatus } from '../services/websocket';

export interface ConnectionState {
  connected: boolean;
  reconnecting: boolean;
  attempts: number;
  lastHeartbeat: Date | null;
  latency: number | null; // Round-trip time in ms
}

export interface UseGovernorWebSocketReturn {
  connectionState: ConnectionState;
  status: GovernorStatus | null;
  alerts: Array<{ level: string; message: string; timestamp: Date }>;
  connect: () => Promise<void>;
  disconnect: () => void;
  reconnectFeeds: () => void;
  requestStatus: () => void;
  clearAlerts: () => void;
}

export function useGovernorWebSocket(): UseGovernorWebSocketReturn {
  const [connectionState, setConnectionState] = useState<ConnectionState>({
    connected: false,
    reconnecting: false,
    attempts: 0,
    lastHeartbeat: null,
    latency: null,
  });

  const [status, setStatus] = useState<GovernorStatus | null>(null);
  const [alerts, setAlerts] = useState<Array<{ level: string; message: string; timestamp: Date }>>([]);

  const wsRef = useRef(getGovernorWebSocket());
  const pingTimeRef = useRef<number>(0);

  // Connect on mount
  useEffect(() => {
    const ws = wsRef.current;

    // Set up handlers
    const unsubConnection = ws.onConnection((connected, reconnecting) => {
      setConnectionState(prev => ({
        ...prev,
        connected,
        reconnecting,
        attempts: ws.getConnectionInfo().attempts,
      }));
    });

    const unsubStatus = ws.onStatus((newStatus) => {
      setStatus(newStatus);
      setConnectionState(prev => ({
        ...prev,
        lastHeartbeat: new Date(),
      }));
    });

    const unsubAlert = ws.onAlert((level, message) => {
      setAlerts(prev => [
        { level, message, timestamp: new Date() },
        ...prev.slice(0, 9), // Keep last 10 alerts
      ]);
    });

    // Connect
    ws.connect().catch(err => {
      console.error('[useGovernorWebSocket] Initial connection failed:', err);
    });

    // Cleanup
    return () => {
      unsubConnection();
      unsubStatus();
      unsubAlert();
      // Don't disconnect on unmount - keep connection alive for other components
    };
  }, []);

  // Track heartbeat timestamps for latency calculation
  useEffect(() => {
    const ws = wsRef.current;

    // Override ping to track timing
    const originalSendCommand = ws.sendCommand.bind(ws);
    ws.sendCommand = (command: string, data?: any) => {
      if (command === 'ping') {
        pingTimeRef.current = Date.now();
      }
      originalSendCommand(command, data);
    };

    return () => {
      ws.sendCommand = originalSendCommand;
    };
  }, []);

  const connect = useCallback(async () => {
    await wsRef.current.connect();
  }, []);

  const disconnect = useCallback(() => {
    wsRef.current.disconnect();
  }, []);

  const reconnectFeeds = useCallback(() => {
    wsRef.current.reconnectFeeds();
    setAlerts(prev => [
      { level: 'info', message: 'Reconnecting data feeds...', timestamp: new Date() },
      ...prev.slice(0, 9),
    ]);
  }, []);

  const requestStatus = useCallback(() => {
    wsRef.current.requestStatus();
  }, []);

  const clearAlerts = useCallback(() => {
    setAlerts([]);
  }, []);

  return {
    connectionState,
    status,
    alerts,
    connect,
    disconnect,
    reconnectFeeds,
    requestStatus,
    clearAlerts,
  };
}

export default useGovernorWebSocket;
