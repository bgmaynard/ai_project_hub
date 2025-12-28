/**
 * Governor Panel - Main Container
 *
 * System oversight dashboard showing trading status, market context,
 * policies, decisions, and health indicators.
 *
 * NO charts, indicators, or order controls.
 * Optimized for instant clarity and trust.
 */

import React from 'react';
import { useGovernorData } from '../../hooks/useGovernorData';
import GlobalStatus from './GlobalStatus';
import MarketContext from './MarketContext';
import StrategyPolicies from './StrategyPolicies';
import RecentDecisions from './RecentDecisions';
import SystemHealth from './SystemHealth';

export const Governor: React.FC = () => {
  const { data, isLoading, error, refresh } = useGovernorData();

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

            {/* Trading UI Link */}
            <a
              href="/trading-new"
              className="text-xs px-3 py-1.5 bg-ibkr-accent text-white rounded hover:bg-ibkr-accent/80 transition-colors"
            >
              Trading UI
            </a>

            {/* API Status */}
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-ibkr-success animate-pulse"></span>
              <span className="text-xs text-ibkr-text-secondary">API</span>
            </div>
          </div>
        </div>
      </header>

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
            AI: <span style={{ color: data.globalStatus.aiState === 'ENABLED' ? '#4ec9b0' : '#f48771' }}>{data.globalStatus.aiState}</span>
          </span>
        </div>
      </footer>
    </div>
  );
};

export default Governor;
