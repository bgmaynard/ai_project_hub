import React, { useState, useEffect, useRef } from 'react';

interface PatternAlert {
  id: string;
  timestamp: string;
  symbol: string;
  setup_type: string;
  confidence: number;
  entry_price: number;
  stop_price: number;
  target_2r: number;
  risk_reward_ratio: number;
  timeframe: string;
}

/**
 * Pattern Alerts Component
 *
 * Real-time pattern detection alerts via WebSocket
 */
const PatternAlerts: React.FC = () => {
  const [alerts, setAlerts] = useState<PatternAlert[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const [filter, setFilter] = useState<'all' | 'high' | 'medium'>('all');

  useEffect(() => {
    connectWebSocket();

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  const connectWebSocket = () => {
    try {
      const ws = new WebSocket('ws://localhost:9101/api/warrior/ws/alerts');

      ws.onopen = () => {
        console.log('WebSocket connected');
        setIsConnected(true);
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);

          if (data.type === 'pattern_detected') {
            const alert: PatternAlert = {
              id: `${data.data.symbol}_${Date.now()}`,
              timestamp: data.timestamp,
              symbol: data.data.symbol,
              setup_type: data.data.setup_type,
              confidence: data.data.confidence,
              entry_price: data.data.entry_price,
              stop_price: data.data.stop_price,
              target_2r: data.data.target_2r,
              risk_reward_ratio: data.data.risk_reward_ratio || 2.0,
              timeframe: data.data.timeframe || '5min'
            };

            setAlerts(prev => [alert, ...prev].slice(0, 20)); // Keep last 20
          }
        } catch (err) {
          console.error('Error parsing WebSocket message:', err);
        }
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        setIsConnected(false);
      };

      ws.onclose = () => {
        console.log('WebSocket disconnected');
        setIsConnected(false);

        // Reconnect after 5 seconds
        setTimeout(connectWebSocket, 5000);
      };

      wsRef.current = ws;
    } catch (err) {
      console.error('Error connecting WebSocket:', err);
    }
  };

  const getFilteredAlerts = () => {
    switch (filter) {
      case 'high':
        return alerts.filter(a => a.confidence >= 75);
      case 'medium':
        return alerts.filter(a => a.confidence >= 60 && a.confidence < 75);
      default:
        return alerts;
    }
  };

  const getPatternIcon = (setupType: string) => {
    switch (setupType) {
      case 'BULL_FLAG': return '🚩';
      case 'HOD_BREAKOUT': return '🔝';
      case 'WHOLE_DOLLAR_BREAKOUT': return '💲';
      case 'MICRO_PULLBACK': return '🔄';
      case 'HAMMER_REVERSAL': return '🔨';
      default: return '📊';
    }
  };

  const getPatternName = (setupType: string) => {
    return setupType.split('_').map(word =>
      word.charAt(0) + word.slice(1).toLowerCase()
    ).join(' ');
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 80) return 'text-green-500 bg-green-500/10';
    if (confidence >= 70) return 'text-green-400 bg-green-400/10';
    if (confidence >= 60) return 'text-yellow-500 bg-yellow-500/10';
    return 'text-orange-500 bg-orange-500/10';
  };

  const clearAlerts = () => {
    setAlerts([]);
  };

  return (
    <div className="bg-ibkr-surface rounded-lg border border-ibkr-border h-full flex flex-col">
      {/* Header */}
      <div className="p-4 border-b border-ibkr-border">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-lg font-semibold text-ibkr-text flex items-center">
            <span className="mr-2">🎯</span>
            Pattern Alerts
          </h2>

          <div className="flex items-center space-x-2">
            <div className="flex items-center space-x-1.5">
              <span className={`w-2 h-2 rounded-full ${
                isConnected ? 'bg-green-500 animate-pulse' : 'bg-red-500'
              }`}></span>
              <span className="text-xs text-ibkr-text-secondary">
                {isConnected ? 'Live' : 'Disconnected'}
              </span>
            </div>

            {alerts.length > 0 && (
              <button
                onClick={clearAlerts}
                className="text-xs text-ibkr-text-secondary hover:text-red-500 transition-colors"
                title="Clear all alerts"
              >
                Clear
              </button>
            )}
          </div>
        </div>

        {/* Filters */}
        <div className="flex space-x-1">
          <button
            onClick={() => setFilter('all')}
            className={`flex-1 px-3 py-1.5 text-xs rounded ${
              filter === 'all'
                ? 'bg-ibkr-accent text-white'
                : 'bg-ibkr-bg text-ibkr-text-secondary hover:text-ibkr-text'
            }`}
          >
            All ({alerts.length})
          </button>
          <button
            onClick={() => setFilter('high')}
            className={`flex-1 px-3 py-1.5 text-xs rounded ${
              filter === 'high'
                ? 'bg-ibkr-accent text-white'
                : 'bg-ibkr-bg text-ibkr-text-secondary hover:text-ibkr-text'
            }`}
          >
            High (≥75%)
          </button>
          <button
            onClick={() => setFilter('medium')}
            className={`flex-1 px-3 py-1.5 text-xs rounded ${
              filter === 'medium'
                ? 'bg-ibkr-accent text-white'
                : 'bg-ibkr-bg text-ibkr-text-secondary hover:text-ibkr-text'
            }`}
          >
            Medium (60-75%)
          </button>
        </div>
      </div>

      {/* Alerts List */}
      <div className="flex-1 overflow-y-auto">
        {getFilteredAlerts().length === 0 ? (
          <div className="p-8 text-center">
            <div className="text-4xl mb-3">
              {isConnected ? '📡' : '⚠️'}
            </div>
            <h3 className="text-sm font-medium text-ibkr-text mb-1">
              {isConnected ? 'No Patterns Detected' : 'Disconnected'}
            </h3>
            <p className="text-xs text-ibkr-text-secondary">
              {isConnected
                ? 'Waiting for real-time pattern alerts...'
                : 'Attempting to reconnect...'}
            </p>
          </div>
        ) : (
          <div className="divide-y divide-ibkr-border">
            {getFilteredAlerts().map((alert) => (
              <div
                key={alert.id}
                className="p-4 hover:bg-ibkr-bg transition-colors"
              >
                {/* Header */}
                <div className="flex items-start justify-between mb-2">
                  <div className="flex items-center space-x-2">
                    <span className="text-2xl">{getPatternIcon(alert.setup_type)}</span>
                    <div>
                      <div className="font-semibold text-ibkr-text">
                        {alert.symbol}
                      </div>
                      <div className="text-xs text-ibkr-text-secondary">
                        {getPatternName(alert.setup_type)}
                      </div>
                    </div>
                  </div>

                  <div className={`px-2 py-1 rounded text-xs font-medium ${getConfidenceColor(alert.confidence)}`}>
                    {alert.confidence.toFixed(0)}%
                  </div>
                </div>

                {/* Details Grid */}
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div>
                    <span className="text-ibkr-text-secondary">Entry:</span>
                    <span className="ml-1 text-ibkr-text font-medium">
                      ${alert.entry_price.toFixed(2)}
                    </span>
                  </div>
                  <div>
                    <span className="text-ibkr-text-secondary">Stop:</span>
                    <span className="ml-1 text-red-500 font-medium">
                      ${alert.stop_price.toFixed(2)}
                    </span>
                  </div>
                  <div>
                    <span className="text-ibkr-text-secondary">Target:</span>
                    <span className="ml-1 text-green-500 font-medium">
                      ${alert.target_2r.toFixed(2)}
                    </span>
                  </div>
                  <div>
                    <span className="text-ibkr-text-secondary">R:R:</span>
                    <span className="ml-1 text-ibkr-text font-medium">
                      {alert.risk_reward_ratio.toFixed(1)}:1
                    </span>
                  </div>
                </div>

                {/* Footer */}
                <div className="mt-2 flex items-center justify-between text-xs">
                  <span className="text-ibkr-text-secondary">
                    {alert.timeframe} • {new Date(alert.timestamp).toLocaleTimeString()}
                  </span>

                  <button className="px-2 py-1 bg-green-600 text-white rounded hover:bg-green-700 transition-colors">
                    Trade
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default PatternAlerts;
