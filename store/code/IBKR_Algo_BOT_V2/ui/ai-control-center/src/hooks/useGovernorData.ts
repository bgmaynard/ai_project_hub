/**
 * useGovernorData Hook
 *
 * Unified hook for fetching all governor panel data with
 * staggered polling intervals for optimal performance.
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import type {
  GovernorData,
  GlobalStatus,
  MarketContext,
  StrategyPolicy,
  AIDecision,
  HealthIndicator,
  TradingWindow,
  DataFreshness,
} from '../types/governor';
import { POLL_INTERVALS } from '../types/governor';

// ============================================
// Default/Fallback Values
// ============================================

const DEFAULT_GLOBAL_STATUS: GlobalStatus = {
  mode: 'PAPER',
  tradingWindow: 'CLOSED',
  windowTime: '--:-- - --:--',
  aiState: 'DISABLED',
  killSwitch: {
    active: false,
    reason: null,
    cooldownSeconds: 0,
  },
};

const DEFAULT_MARKET_CONTEXT: MarketContext = {
  regime: 'RANGING',
  regimeConfidence: 0,
  volatility: 'NORMAL',
  volatilityPct: 0,
  liquidity: 'ADEQUATE',
  dataAge: 999,
  dataFreshness: 'OFFLINE',
  lastUpdate: new Date().toISOString(),
};

// ============================================
// Trading Window Helper
// ============================================

function getTradingWindow(): { window: TradingWindow; time: string } {
  const now = new Date();
  const hours = now.getHours();
  const minutes = now.getMinutes();
  const totalMinutes = hours * 60 + minutes;

  // Pre-market: 4:00 AM - 9:30 AM
  if (totalMinutes >= 240 && totalMinutes < 570) {
    return { window: 'PRE_MARKET', time: '04:00 - 09:30' };
  }
  // Market open: 9:30 AM - 4:00 PM
  if (totalMinutes >= 570 && totalMinutes < 960) {
    return { window: 'OPEN', time: '09:30 - 16:00' };
  }
  // After hours: 4:00 PM - 8:00 PM
  if (totalMinutes >= 960 && totalMinutes < 1200) {
    return { window: 'AFTER_HOURS', time: '16:00 - 20:00' };
  }
  // Closed
  return { window: 'CLOSED', time: 'Market Closed' };
}

function getDataFreshness(ageSeconds: number): DataFreshness {
  if (ageSeconds < 10) return 'FRESH';
  if (ageSeconds < 60) return 'STALE';
  return 'OFFLINE';
}

// ============================================
// API Fetchers
// ============================================

async function fetchGlobalStatus(): Promise<GlobalStatus> {
  try {
    const [safeRes, scalpRes] = await Promise.all([
      fetch('/api/validation/safe/status').catch(() => null),
      fetch('/api/scalp/status').catch(() => null),
    ]);

    const safeData = safeRes?.ok ? await safeRes.json() : {};
    const scalpData = scalpRes?.ok ? await scalpRes.json() : {};

    const { window: tradingWindow, time: windowTime } = getTradingWindow();

    return {
      mode: scalpData.paper_mode !== false ? 'PAPER' : 'LIVE',
      tradingWindow,
      windowTime,
      aiState: scalpData.running ? 'ENABLED' : (safeData.trading_allowed === false ? 'DISABLED' : 'PAUSED'),
      killSwitch: {
        active: safeData.kill_switch_active || false,
        reason: safeData.kill_switch_reason || null,
        cooldownSeconds: safeData.kill_switch_cooldown_remaining_seconds || 0,
      },
    };
  } catch {
    const { window: tradingWindow, time: windowTime } = getTradingWindow();
    return { ...DEFAULT_GLOBAL_STATUS, tradingWindow, windowTime };
  }
}

async function fetchMarketContext(): Promise<MarketContext> {
  try {
    const [momentumRes, policyRes] = await Promise.all([
      fetch('/api/validation/momentum/states').catch(() => null),
      fetch('/api/strategy/policy').catch(() => null),
    ]);

    const momentumData = momentumRes?.ok ? await momentumRes.json() : {};
    const policyData = policyRes?.ok ? await policyRes.json() : {};

    // Extract regime info from policy or momentum data
    const regime = policyData.overall_regime || momentumData.regime || 'RANGING';
    const confidence = policyData.regime_confidence || momentumData.confidence || 0;
    const volatility = policyData.volatility_level || 'NORMAL';
    const volatilityPct = policyData.volatility_pct || 0;

    // Calculate data age
    const lastUpdateTime = momentumData.last_update || policyData.last_update;
    const dataAge = lastUpdateTime
      ? Math.round((Date.now() - new Date(lastUpdateTime).getTime()) / 1000)
      : 999;

    return {
      regime: regime.toUpperCase().replace(' ', '_'),
      regimeConfidence: Math.round(confidence * 100) / 100,
      volatility,
      volatilityPct: Math.round(volatilityPct * 100) / 100,
      liquidity: 'ADEQUATE',
      dataAge,
      dataFreshness: getDataFreshness(dataAge),
      lastUpdate: new Date().toISOString(),
    };
  } catch {
    return { ...DEFAULT_MARKET_CONTEXT, lastUpdate: new Date().toISOString() };
  }
}

async function fetchPolicies(): Promise<StrategyPolicy[]> {
  try {
    const res = await fetch('/api/strategy/policy');
    if (!res.ok) throw new Error('Failed to fetch policies');

    const data = await res.json();
    const policies = data.policies || {};

    return Object.entries(policies).map(([id, policy]: [string, any]) => ({
      id,
      name: formatPolicyName(id),
      status: policy.enabled ? 'ENABLED' : (policy.cooldown_remaining > 0 ? 'PAUSED' : 'DISABLED'),
      reason: policy.reason || 'No reason provided',
      vetoCount: policy.veto_counts || 0,
      cooldownRemaining: policy.cooldown_remaining || 0,
    }));
  } catch {
    // Return default strategies
    return [
      { id: 'momentum', name: 'Momentum', status: 'PAUSED', reason: 'Waiting for data', vetoCount: 0 },
      { id: 'news', name: 'News Trader', status: 'DISABLED', reason: 'Not configured', vetoCount: 0 },
      { id: 'scalper', name: 'HFT Scalper', status: 'PAUSED', reason: 'Waiting for data', vetoCount: 0 },
      { id: 'gap', name: 'Gap Trader', status: 'DISABLED', reason: 'Pre-market only', vetoCount: 0 },
    ];
  }
}

function formatPolicyName(id: string): string {
  const names: Record<string, string> = {
    momentum: 'Momentum',
    news: 'News Trader',
    scalper: 'HFT Scalper',
    gap: 'Gap Trader',
    warrior: 'Warrior',
  };
  return names[id] || id.charAt(0).toUpperCase() + id.slice(1);
}

async function fetchDecisions(): Promise<AIDecision[]> {
  try {
    const [auditRes, exitRes] = await Promise.all([
      fetch('/api/strategy/policy/audit?limit=10').catch(() => null),
      fetch('/api/validation/exit/log?limit=10').catch(() => null),
    ]);

    const auditData = auditRes?.ok ? await auditRes.json() : [];
    const exitData = exitRes?.ok ? await exitRes.json() : [];

    const decisions: AIDecision[] = [];

    // Process audit log (entry decisions)
    if (Array.isArray(auditData)) {
      for (const entry of auditData.slice(0, 5)) {
        decisions.push({
          timestamp: entry.timestamp || new Date().toISOString(),
          action: entry.approved ? 'APPROVED' : 'VETOED',
          symbol: entry.symbol || 'N/A',
          type: 'entry',
          reasons: entry.reasons || [entry.reason || 'No reason'],
        });
      }
    }

    // Process exit log
    if (Array.isArray(exitData)) {
      for (const entry of exitData.slice(0, 5)) {
        decisions.push({
          timestamp: entry.timestamp || new Date().toISOString(),
          action: 'EXIT',
          symbol: entry.symbol || 'N/A',
          type: 'exit',
          reasons: [entry.exit_reason || 'Manual exit'],
          pnl: entry.pnl,
        });
      }
    }

    // Sort by timestamp descending and take top 10
    return decisions
      .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime())
      .slice(0, 10);
  } catch {
    return [];
  }
}

async function fetchHealth(): Promise<HealthIndicator[]> {
  try {
    const [healthRes, summaryRes] = await Promise.all([
      fetch('/api/health').catch(() => null),
      fetch('/api/validation/export/summary').catch(() => null),
    ]);

    const healthData = healthRes?.ok ? await healthRes.json() : {};
    const summaryData = summaryRes?.ok ? await summaryRes.json() : {};

    const indicators: HealthIndicator[] = [
      {
        name: 'Data Feed',
        status: healthData.status === 'ok' ? 'HEALTHY' : 'ERROR',
        detail: healthData.status === 'ok' ? 'Connected' : 'Disconnected',
      },
      {
        name: 'Broker API',
        status: healthData.broker?.connected ? 'HEALTHY' : 'OFFLINE',
        detail: healthData.broker?.connected ? 'Schwab Connected' : 'Not connected',
      },
      {
        name: 'WebSocket',
        status: healthData.websocket?.connected ? 'HEALTHY' : 'OFFLINE',
        detail: healthData.websocket?.connected ? 'Connected' : 'Disconnected',
      },
      {
        name: 'Chronos',
        status: summaryData.momentum?.active ? 'HEALTHY' : 'OFFLINE',
        detail: summaryData.momentum?.active ? 'Online' : 'Offline',
      },
      {
        name: 'Gating Engine',
        status: 'HEALTHY',
        detail: `${summaryData.policy?.total_vetoes || 0} vetoes`,
      },
      {
        name: 'Policy Engine',
        status: summaryData.policy?.active ? 'HEALTHY' : 'DEGRADED',
        detail: summaryData.policy?.active ? 'Active' : 'Degraded',
      },
      {
        name: 'Kill Switch',
        status: summaryData.safe_activation?.kill_switch_active ? 'ERROR' : 'HEALTHY',
        detail: summaryData.safe_activation?.kill_switch_active ? 'ARMED' : 'Ready',
      },
      {
        name: 'Circuit Breaker',
        status: 'HEALTHY',
        detail: `$${summaryData.safe_activation?.daily_loss_remaining || 100} left`,
      },
    ];

    return indicators;
  } catch {
    return [
      { name: 'Data Feed', status: 'OFFLINE', detail: 'Cannot connect' },
      { name: 'Broker API', status: 'OFFLINE', detail: 'Cannot connect' },
      { name: 'WebSocket', status: 'OFFLINE', detail: 'Cannot connect' },
      { name: 'Chronos', status: 'OFFLINE', detail: 'Cannot connect' },
      { name: 'Gating Engine', status: 'OFFLINE', detail: 'Cannot connect' },
      { name: 'Policy Engine', status: 'OFFLINE', detail: 'Cannot connect' },
      { name: 'Kill Switch', status: 'OFFLINE', detail: 'Cannot connect' },
      { name: 'Circuit Breaker', status: 'OFFLINE', detail: 'Cannot connect' },
    ];
  }
}

// ============================================
// Main Hook
// ============================================

export function useGovernorData() {
  const [data, setData] = useState<GovernorData>({
    globalStatus: DEFAULT_GLOBAL_STATUS,
    marketContext: DEFAULT_MARKET_CONTEXT,
    policies: [],
    decisions: [],
    health: [],
    lastFetch: new Date().toISOString(),
  });

  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Track intervals for cleanup
  const intervalsRef = useRef<NodeJS.Timeout[]>([]);

  // Individual update functions
  const updateGlobalStatus = useCallback(async () => {
    const status = await fetchGlobalStatus();
    setData(prev => ({ ...prev, globalStatus: status, lastFetch: new Date().toISOString() }));
  }, []);

  const updateMarketContext = useCallback(async () => {
    const context = await fetchMarketContext();
    setData(prev => ({ ...prev, marketContext: context }));
  }, []);

  const updatePolicies = useCallback(async () => {
    const policies = await fetchPolicies();
    setData(prev => ({ ...prev, policies }));
  }, []);

  const updateDecisions = useCallback(async () => {
    const decisions = await fetchDecisions();
    setData(prev => ({ ...prev, decisions }));
  }, []);

  const updateHealth = useCallback(async () => {
    const health = await fetchHealth();
    setData(prev => ({ ...prev, health }));
  }, []);

  // Initial fetch all
  const fetchAll = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      const [globalStatus, marketContext, policies, decisions, health] = await Promise.all([
        fetchGlobalStatus(),
        fetchMarketContext(),
        fetchPolicies(),
        fetchDecisions(),
        fetchHealth(),
      ]);

      setData({
        globalStatus,
        marketContext,
        policies,
        decisions,
        health,
        lastFetch: new Date().toISOString(),
      });
    } catch (err) {
      setError('Failed to fetch governor data');
      console.error('Governor data fetch error:', err);
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Set up polling intervals
  useEffect(() => {
    // Initial fetch
    fetchAll();

    // Set up staggered polling
    intervalsRef.current = [
      setInterval(updateGlobalStatus, POLL_INTERVALS.globalStatus),
      setInterval(updateMarketContext, POLL_INTERVALS.marketContext),
      setInterval(updatePolicies, POLL_INTERVALS.policies),
      setInterval(updateDecisions, POLL_INTERVALS.decisions),
      setInterval(updateHealth, POLL_INTERVALS.health),
    ];

    // Cleanup
    return () => {
      intervalsRef.current.forEach(clearInterval);
    };
  }, [fetchAll, updateGlobalStatus, updateMarketContext, updatePolicies, updateDecisions, updateHealth]);

  return {
    data,
    isLoading,
    error,
    refresh: fetchAll,
  };
}

export default useGovernorData;
