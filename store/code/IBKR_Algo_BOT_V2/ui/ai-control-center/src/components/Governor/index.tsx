/**
 * Governor Panel - Main Container
 *
 * System oversight dashboard showing trading status, market context,
 * policies, decisions, and health indicators.
 *
 * NO charts, indicators, or order controls.
 * Optimized for instant clarity and trust.
 */

import React, { useState, useEffect } from 'react';
import { useGovernorData } from '../../hooks/useGovernorData';
import { useGovernorWebSocket } from '../../hooks/useGovernorWebSocket';
import { apiService } from '../../services/api';
import GlobalStatus from './GlobalStatus';
import MarketContext from './MarketContext';
import StrategyPolicies from './StrategyPolicies';
import RecentDecisions from './RecentDecisions';
import SystemHealth from './SystemHealth';

export const Governor: React.FC = () => {
  const { data, isLoading, error, refresh } = useGovernorData();
  const { connectionState, alerts, reconnectFeeds, clearAlerts } = useGovernorWebSocket();
  const [isReconnecting, setIsReconnecting] = useState(false);
  const [reconnectMessage, setReconnectMessage] = useState<string | null>(null);

  // Show alert when WebSocket disconnects
  useEffect(() => {
    if (!connectionState.connected && !connectionState.reconnecting) {
      setReconnectMessage('WebSocket disconnected');
    } else if (connectionState.reconnecting) {
      setReconnectMessage(`Reconnecting... (attempt ${connectionState.attempts})`);
    } else if (connectionState.connected && reconnectMessage?.includes('Reconnect')) {
      setReconnectMessage('Connected');
      setTimeout(() => setReconnectMessage(null), 2000);
    }
  }, [connectionState.connected, connectionState.reconnecting, connectionState.attempts]);

  const handleReconnect = async () => {
    if (data.globalStatus.mode !== 'PAPER') {
      setReconnectMessage('Reconnect only available in PAPER mode');
      setTimeout(() => setReconnectMessage(null), 3000);
      return;
    }

    setIsReconnecting(true);
    setReconnectMessage(null);
    try {
      const response = await apiService.reconnectFeeds(true);
      if (response.success) {
        setReconnectMessage('Feeds reconnected successfully');
        refresh(); // Refresh all data
      } else {
        setReconnectMessage(response.error || 'Reconnect failed');
      }
    } catch (err: any) {
      setReconnectMessage(err.message || 'Reconnect failed');
    } finally {
      setIsReconnecting(false);
      setTimeout(() => setReconnectMessage(null), 5000);
    }
  };

  if (isLoading) {
    return (
      <div className="min-h-screen bg-ibkr-bg flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-ibkr-accent mx-auto"></div>
          <p className="mt-4 text-ibkr-text-secondary">Loading System Governor...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-ibkr-bg">
      {/* Header */}
      <header className="bg-ibkr-surface border-b border-ibkr-border px-6 py-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <h1 className="text-lg font-bold text-ibkr-text">
              MORPHEUS SYSTEM GOVERNOR
            </h1>
            {error && (
              <span className="text-xs text-ibkr-error bg-ibkr-error/10 px-2 py-1 rounded">
                {error}
              </span>
            )}
          </div>

          <div className="flex items-center gap-4">
            {/* Last Fetch */}
            <span className="text-xs text-ibkr-text-secondary">
              Updated: {new Date(data.lastFetch).toLocaleTimeString()}
            </span>

            {/* Refresh Button */}
            <button
              onClick={refresh}
              className="text-xs px-3 py-1.5 bg-ibkr-bg text-ibkr-text-secondary hover:text-ibkr-text rounded border border-ibkr-border hover:border-ibkr-accent transition-colors"
            >
              Refresh
            </button>

            {/* Reconnect Button - Paper Mode Only */}
            {data.globalStatus.mode === 'PAPER' && (
              <button
                onClick={handleReconnect}
                disabled={isReconnecting}
                className={`text-xs px-3 py-1.5 rounded border transition-colors ${
                  isReconnecting
                    ? 'bg-ibkr-warning/20 text-ibkr-warning border-ibkr-warning cursor-wait'
                    : 'bg-ibkr-bg text-ibkr-text-secondary hover:text-ibkr-text border-ibkr-border hover:border-ibkr-warning'
                }`}
                title="Reconnect/Restart data feeds (Paper mode only)"
              >
                {isReconnecting ? 'Reconnecting...' : 'Reconnect Feeds'}
              </button>
            )}

            {/* Reconnect Message */}
            {reconnectMessage && (
              <span className={`text-xs px-2 py-1 rounded ${
                reconnectMessage.includes('success') ? 'bg-ibkr-success/20 text-ibkr-success' : 'bg-ibkr-error/20 text-ibkr-error'
              }`}>
                {reconnectMessage}
              </span>
            )}

            {/* Trading UI Link */}
            <a
              href="/trading-new"
              className="text-xs px-3 py-1.5 bg-ibkr-accent text-white rounded hover:bg-ibkr-accent/80 transition-colors"
            >
              Trading UI
            </a>

            {/* WebSocket Connection Status */}
            <div className="flex items-center gap-2" title={connectionState.connected ? 'WebSocket connected - real-time updates active' : connectionState.reconnecting ? `Reconnecting (attempt ${connectionState.attempts})` : 'WebSocket disconnected'}>
              <span
                className={`w-2 h-2 rounded-full ${
                  connectionState.connected
                    ? 'bg-ibkr-success animate-pulse'
                    : connectionState.reconnecting
                    ? 'bg-ibkr-warning animate-ping'
                    : 'bg-ibkr-error'
                }`}
              />
              <span className="text-xs text-ibkr-text-secondary">
                {connectionState.connected ? 'LIVE' : connectionState.reconnecting ? 'RECONNECTING' : 'OFFLINE'}
              </span>
            </div>

            {/* Manual Reconnect Button - shown when disconnected */}
            {!connectionState.connected && !connectionState.reconnecting && (
              <button
                onClick={reconnectFeeds}
                className="text-xs px-2 py-1 bg-ibkr-error/20 text-ibkr-error rounded border border-ibkr-error hover:bg-ibkr-error/30"
              >
                Reconnect
              </button>
            )}
          </div>
        </div>
      </header>

      {/* Alert Banner - shown when there are alerts */}
      {alerts.length > 0 && (
        <div className="bg-ibkr-warning/10 border-b border-ibkr-warning px-6 py-2">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="text-ibkr-warning">
                {alerts[0].level === 'error' ? '⚠️' : alerts[0].level === 'warning' ? '⚡' : 'ℹ️'}
              </span>
              <span className="text-sm text-ibkr-warning">{alerts[0].message}</span>
              <span className="text-xs text-ibkr-text-secondary">
                {alerts[0].timestamp.toLocaleTimeString()}
              </span>
            </div>
            <button
              onClick={clearAlerts}
              className="text-xs text-ibkr-text-secondary hover:text-ibkr-text"
            >
              Clear
            </button>
          </div>
        </div>
      )}

      {/* Main Content */}
      <main className="p-6 space-y-6">
        {/* Top Row: Global Status + Market Context */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <GlobalStatus status={data.globalStatus} />
          <MarketContext context={data.marketContext} />
        </div>

        {/* Strategy Policies */}
        <StrategyPolicies policies={data.policies} />

        {/* Recent Decisions */}
        <RecentDecisions decisions={data.decisions} />

        {/* System Health */}
        <SystemHealth indicators={data.health} />
      </main>

      {/* Footer */}
      <footer className="fixed bottom-0 left-0 right-0 bg-ibkr-surface border-t border-ibkr-border px-6 py-2">
        <div className="flex items-center justify-between text-xs text-ibkr-text-secondary">
          <span>Morpheus Trading Platform v2.0</span>
          <span>
            Mode: <span className={data.globalStatus.mode === 'PAPER' ? 'text-ibkr-accent' : 'text-ibkr-error'}>{data.globalStatus.mode}</span>
            {' | '}
            Window: <span className={data.globalStatus.tradingWindow === 'OPEN' ? 'text-ibkr-success' : 'text-ibkr-warning'}>{data.globalStatus.tradingWindow}</span>
            {' | '}
            AI: <span style={{ color: data.globalStatus.aiPosture === 'ACTIVE' ? '#4ec9b0' : data.globalStatus.aiPosture === 'LOCKED' ? '#f48771' : '#dcdcaa' }}>{data.globalStatus.aiPosture}</span>
          </span>
        </div>
      </footer>
    </div>
  );
};

export default Governor;
