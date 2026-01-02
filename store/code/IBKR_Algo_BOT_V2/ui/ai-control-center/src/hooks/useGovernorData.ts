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
  SystemHealth,
  TradingWindow,
  DataFreshness,
  AIPosture,
  SystemSafetyStatus,
  SystemState,
} from '../types/governor';
import { POLL_INTERVALS } from '../types/governor';

// ============================================
// Default/Fallback Values
// ============================================

const DEFAULT_GLOBAL_STATUS: GlobalStatus = {
  mode: 'PAPER',
  tradingWindow: 'CLOSED',
  windowTime: '--:-- - --:--',
  aiPosture: 'NO_TRADE',
  aiPostureReason: 'Market closed',
  killSwitch: {
    active: false,
    reason: null,
    cooldownSeconds: 0,
  },
  systemState: 'SERVICE_NOT_RUNNING',
  systemStateReason: 'Connecting to services...',
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
  aiInterpretation: 'Waiting for market data...',
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
// Fetch with Timeout Helper
// ============================================

const FETCH_TIMEOUT = 3000; // 3 second timeout for faster feedback

// Cache last successful responses to show during failures
const responseCache: Record<string, any> = {};

async function fetchWithTimeout(url: string, timeout: number = FETCH_TIMEOUT): Promise<Response | null> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeout);

  try {
    const response = await fetch(url, { signal: controller.signal });
    clearTimeout(timeoutId);
    return response;
  } catch (error) {
    clearTimeout(timeoutId);
    // Only log timeouts, not expected aborts
    if (error instanceof Error && error.name !== 'AbortError') {
      console.warn(`Fetch error for ${url}:`, error);
    }
    return null;
  }
}

// Helper to get cached data or default
function getCachedOrDefault<T>(key: string, defaultValue: T): T {
  return responseCache[key] || defaultValue;
}

// ============================================
// API Fetchers
// ============================================

function deriveAIPosture(
  tradingWindow: TradingWindow,
  killSwitchActive: boolean,
  isRunning: boolean,
  volatilityLevel?: string,
  healthDegraded?: boolean
): { posture: AIPosture; reason: string } {
  // Priority 1: Kill switch
  if (killSwitchActive) {
    return { posture: 'LOCKED', reason: 'Kill switch engaged - all trading halted' };
  }

  // Priority 2: Outside trading window
  if (tradingWindow === 'CLOSED') {
    return { posture: 'NO_TRADE', reason: 'Market closed - by design' };
  }

  // Priority 3: Defensive conditions
  if (volatilityLevel === 'EXTREME' || volatilityLevel === 'HIGH') {
    return { posture: 'DEFENSIVE', reason: `High volatility (${volatilityLevel}) - cautious mode` };
  }
  if (healthDegraded) {
    return { posture: 'DEFENSIVE', reason: 'System health degraded - cautious mode' };
  }

  // Priority 4: Active or not running
  if (isRunning) {
    return { posture: 'ACTIVE', reason: 'Normal trading operations' };
  }

  // Default: NO_TRADE when not running but market open
  return { posture: 'NO_TRADE', reason: tradingWindow === 'PRE_MARKET' ? 'Pre-market monitoring' : 'Trading not started' };
}

async function fetchGlobalStatus(): Promise<GlobalStatus> {
  try {
    const [safeRes, scalpRes, connectivityRes] = await Promise.all([
      fetchWithTimeout('/api/validation/safe/status'),
      fetchWithTimeout('/api/scalp/status'),
      fetchWithTimeout('/api/validation/connectivity/status'),
    ]);

    // Use cached data if individual requests fail (partial failure resilience)
    const safeData = safeRes?.ok ? await safeRes.json() : (responseCache['safe'] || {});
    const scalpData = scalpRes?.ok ? await scalpRes.json() : (responseCache['scalp'] || {});
    const connectivityData = connectivityRes?.ok ? await connectivityRes.json() : (responseCache['connectivity'] || {});

    // Cache successful responses
    if (safeRes?.ok) responseCache['safe'] = safeData;
    if (scalpRes?.ok) responseCache['scalp'] = scalpData;
    if (connectivityRes?.ok) responseCache['connectivity'] = connectivityData;

    const { window: tradingWindow, time: windowTime } = getTradingWindow();
    const killSwitchActive = safeData.kill_switch_active || false;
    const isRunning = scalpData.running || false;

    const { posture, reason } = deriveAIPosture(
      tradingWindow,
      killSwitchActive,
      isRunning,
      safeData.volatility_level,
      safeData.health_degraded
    );

    // Determine system state from connectivity manager
    let systemState: SystemState = connectivityData.system_state || 'READY';
    let systemStateReason = connectivityData.system_state_reason || 'System ready';

    // Count successful responses - only show SERVICE_NOT_RUNNING if ALL fail
    const successCount = [safeRes?.ok, scalpRes?.ok, connectivityRes?.ok].filter(Boolean).length;

    if (successCount === 0) {
      systemState = 'SERVICE_NOT_RUNNING';
      systemStateReason = 'Cannot connect to API';
    } else if (successCount < 3) {
      // Partial failure - show degraded but not down
      systemState = 'PARTIAL';
      systemStateReason = `${successCount}/3 services responding`;
    } else if (isRunning && tradingWindow === 'OPEN') {
      systemState = 'ACTIVE';
      systemStateReason = 'All systems go, trading enabled';
    }

    return {
      mode: scalpData.paper_mode !== false ? 'PAPER' : 'LIVE',
      tradingWindow,
      windowTime,
      aiPosture: posture,
      aiPostureReason: reason,
      killSwitch: {
        active: killSwitchActive,
        reason: safeData.kill_switch_reason || null,
        cooldownSeconds: safeData.kill_switch_cooldown_remaining_seconds || 0,
      },
      systemState,
      systemStateReason,
    };
  } catch {
    // Total failure - try to use cached data
    const { window: tradingWindow, time: windowTime } = getTradingWindow();
    const cachedConnectivity = responseCache['connectivity'];

    if (cachedConnectivity) {
      // We have cached data - show partial/degraded instead of down
      const { posture, reason } = deriveAIPosture(tradingWindow, false, responseCache['scalp']?.running || false);
      return {
        ...DEFAULT_GLOBAL_STATUS,
        tradingWindow,
        windowTime,
        aiPosture: posture,
        aiPostureReason: reason,
        systemState: 'PARTIAL',
        systemStateReason: 'Using cached data - reconnecting...',
      };
    }

    // No cache - show service not running
    const { posture, reason } = deriveAIPosture(tradingWindow, false, false);
    return {
      ...DEFAULT_GLOBAL_STATUS,
      tradingWindow,
      windowTime,
      aiPosture: posture,
      aiPostureReason: reason,
      systemState: 'SERVICE_NOT_RUNNING',
      systemStateReason: 'Cannot connect to services',
    };
  }
}

function generateAIInterpretation(
  regime: string,
  confidence: number,
  volatility: string,
  liquidity: string
): string {
  const regimeLabel = regime.replace('_', ' ').toLowerCase();
  const confPct = Math.round(confidence * 100);

  if (volatility === 'EXTREME') {
    return `Extreme volatility detected - defensive posture recommended regardless of ${regimeLabel} regime`;
  }
  if (volatility === 'HIGH') {
    return `High volatility with ${regimeLabel} regime (${confPct}% confidence) - proceed with caution`;
  }
  if (regime === 'TRENDING_UP' && confidence > 0.7) {
    return `Strong uptrend (${confPct}% confidence) - favorable for momentum entries`;
  }
  if (regime === 'TRENDING_DOWN') {
    return `Downtrend detected (${confPct}% confidence) - avoid long entries`;
  }
  if (regime === 'VOLATILE') {
    return `Choppy/volatile market - reduce position sizes`;
  }
  if (regime === 'RANGING' && liquidity === 'ADEQUATE') {
    return `Range-bound market with adequate liquidity - scalp opportunities may exist`;
  }
  return `Market ${regimeLabel} (${confPct}% confidence) - normal operations`;
}

async function fetchMarketContext(): Promise<MarketContext> {
  try {
    const [momentumRes, policyRes, polygonRes] = await Promise.all([
      fetchWithTimeout('/api/validation/momentum/states'),
      fetchWithTimeout('/api/strategy/policy'),
      fetchWithTimeout('/api/polygon/stream/status'),
    ]);

    // Use cached data on partial failures
    const momentumData = momentumRes?.ok ? await momentumRes.json() : (responseCache['momentum'] || {});
    const policyData = policyRes?.ok ? await policyRes.json() : (responseCache['policy'] || {});
    const polygonData = polygonRes?.ok ? await polygonRes.json() : (responseCache['polygon'] || {});

    // Cache successful responses
    if (momentumRes?.ok) responseCache['momentum'] = momentumData;
    if (policyRes?.ok) responseCache['policy'] = policyData;
    if (polygonRes?.ok) responseCache['polygon'] = polygonData;

    // Extract regime info from policy or safe_activation data
    const regime = (policyData.overall_regime || 'RANGING').toUpperCase().replace(' ', '_');
    const confidence = policyData.regime_confidence || 0;
    const volatility = policyData.volatility_level || 'NORMAL';
    const volatilityPct = policyData.volatility_pct || 0;
    const liquidity = policyData.liquidity_level || 'ADEQUATE';

    // Calculate data age from Polygon stream's last_message (most reliable)
    let dataAge = 999;
    let dataFreshnessNote = '';
    if (polygonData.last_message) {
      dataAge = Math.round((Date.now() - new Date(polygonData.last_message).getTime()) / 1000);
      // If using cached polygon data and age > 60s, note it
      if (!polygonRes?.ok && dataAge > 60) {
        dataFreshnessNote = ' (cached)';
      }
    } else if (policyData.last_updated) {
      dataAge = Math.round((Date.now() - new Date(policyData.last_updated).getTime()) / 1000);
    }

    // Determine liquidity based on subscriptions and data flow
    const subsCount = (polygonData.trade_subscriptions?.length || 0);
    const actualLiquidity = subsCount > 5 ? 'ADEQUATE' : subsCount > 0 ? 'THIN' : 'POOR';

    return {
      regime,
      regimeConfidence: Math.round(confidence * 100) / 100,
      volatility,
      volatilityPct: Math.round(volatilityPct * 100) / 100,
      liquidity: actualLiquidity,
      dataAge,
      dataFreshness: getDataFreshness(dataAge),
      lastUpdate: new Date().toISOString(),
      aiInterpretation: generateAIInterpretation(regime, confidence, volatility, actualLiquidity),
    };
  } catch {
    return { ...DEFAULT_MARKET_CONTEXT, lastUpdate: new Date().toISOString() };
  }
}

async function fetchPolicies(): Promise<StrategyPolicy[]> {
  try {
    const res = await fetchWithTimeout('/api/strategy/policy');
    if (!res?.ok) throw new Error('Failed to fetch policies');

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

function generatePassiveDecision(reason: string): AIDecision {
  return {
    timestamp: new Date().toISOString(),
    action: 'NO_ACTION',
    symbol: '--',
    type: 'passive',
    reasons: [reason],
  };
}

async function fetchDecisions(): Promise<AIDecision[]> {
  try {
    const [auditRes, exitRes] = await Promise.all([
      fetchWithTimeout('/api/strategy/policy/audit?limit=10'),
      fetchWithTimeout('/api/validation/exit/log?limit=10'),
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
    const sorted = decisions
      .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime())
      .slice(0, 10);

    // If no decisions, generate passive NO_ACTION entries
    if (sorted.length === 0) {
      const { window: tradingWindow } = getTradingWindow();
      if (tradingWindow === 'CLOSED') {
        return [generatePassiveDecision('Market closed - monitoring suspended')];
      }
      if (tradingWindow === 'PRE_MARKET') {
        return [generatePassiveDecision('Pre-market scanning - no triggers yet')];
      }
      return [generatePassiveDecision('Monitoring active - awaiting trade triggers')];
    }

    return sorted;
  } catch {
    return [generatePassiveDecision('Unable to fetch decision history')];
  }
}

function deriveSafetyStatus(indicators: HealthIndicator[], killSwitchActive: boolean): SystemSafetyStatus {
  // HALTED if kill switch is active
  if (killSwitchActive) {
    return 'HALTED';
  }

  // Count unhealthy indicators
  const unhealthyCount = indicators.filter(
    i => i.status === 'ERROR' || i.status === 'OFFLINE'
  ).length;
  const degradedCount = indicators.filter(i => i.status === 'DEGRADED').length;

  // HALTED if critical systems down (>50% offline)
  if (unhealthyCount > indicators.length / 2) {
    return 'HALTED';
  }

  // DEGRADED if any system is unhealthy or degraded
  if (unhealthyCount > 0 || degradedCount > 0) {
    return 'DEGRADED';
  }

  return 'SAFE';
}

async function fetchHealth(): Promise<SystemHealth> {
  try {
    const [healthRes, summaryRes, polygonRes] = await Promise.all([
      fetchWithTimeout('/api/health'),
      fetchWithTimeout('/api/validation/export/summary'),
      fetchWithTimeout('/api/polygon/stream/status'),
    ]);

    const healthData = healthRes?.ok ? await healthRes.json() : {};
    const summaryData = summaryRes?.ok ? await summaryRes.json() : {};
    const polygonData = polygonRes?.ok ? await polygonRes.json() : {};

    const killSwitchActive = summaryData.safe_activation?.kill_switch_active || false;

    // Data Feed: check "healthy" status (API returns "healthy" not "ok")
    const dataFeedHealthy = healthData.status === 'healthy';

    // WebSocket: use Polygon stream status
    const websocketConnected = polygonData.connected === true;
    const websocketSubs = (polygonData.trade_subscriptions?.length || 0) + (polygonData.quote_subscriptions?.length || 0);

    // Chronos/Momentum: check total_symbols > 0 (API returns total_symbols, not active)
    const momentumActive = (summaryData.momentum?.total_symbols || 0) > 0;
    const momentumSymbols = summaryData.momentum?.total_symbols || 0;

    // Policy Engine: check if policies object exists and has entries
    const policyActive = summaryData.policy?.policies && Object.keys(summaryData.policy.policies).length > 0;
    const vetoCount = Object.values(summaryData.policy?.veto_counts || {}).reduce((a: number, b: any) => a + (b || 0), 0);

    const indicators: HealthIndicator[] = [
      {
        name: 'Data Feed',
        status: dataFeedHealthy ? 'HEALTHY' : 'ERROR',
        detail: dataFeedHealthy ? 'Connected' : 'Disconnected',
      },
      {
        name: 'Broker API',
        status: healthData.broker?.connected ? 'HEALTHY' : 'OFFLINE',
        detail: healthData.broker?.connected ? 'Schwab Connected' : 'Not connected',
      },
      {
        name: 'WebSocket',
        status: websocketConnected ? 'HEALTHY' : 'OFFLINE',
        detail: websocketConnected ? `Schwab Stream (${websocketSubs} subs)` : 'Disconnected',
      },
      {
        name: 'Chronos',
        status: momentumActive ? 'HEALTHY' : 'OFFLINE',
        detail: momentumActive ? `${momentumSymbols} symbols` : 'No data',
      },
      {
        name: 'Gating Engine',
        status: 'HEALTHY',
        detail: `${vetoCount} vetoes`,
      },
      {
        name: 'Policy Engine',
        status: policyActive ? 'HEALTHY' : 'DEGRADED',
        detail: policyActive ? 'Active' : 'Degraded',
      },
      {
        name: 'Kill Switch',
        status: killSwitchActive ? 'ERROR' : 'HEALTHY',
        detail: killSwitchActive ? 'ARMED' : 'Ready',
      },
      {
        name: 'Circuit Breaker',
        status: 'HEALTHY',
        detail: `$${summaryData.safe_activation?.daily_loss_remaining || 100} left`,
      },
    ];

    return {
      safetyStatus: deriveSafetyStatus(indicators, killSwitchActive),
      indicators,
    };
  } catch {
    const offlineIndicators: HealthIndicator[] = [
      { name: 'Data Feed', status: 'OFFLINE', detail: 'Cannot connect' },
      { name: 'Broker API', status: 'OFFLINE', detail: 'Cannot connect' },
      { name: 'WebSocket', status: 'OFFLINE', detail: 'Cannot connect' },
      { name: 'Chronos', status: 'OFFLINE', detail: 'Cannot connect' },
      { name: 'Gating Engine', status: 'OFFLINE', detail: 'Cannot connect' },
      { name: 'Policy Engine', status: 'OFFLINE', detail: 'Cannot connect' },
      { name: 'Kill Switch', status: 'OFFLINE', detail: 'Cannot connect' },
      { name: 'Circuit Breaker', status: 'OFFLINE', detail: 'Cannot connect' },
    ];
    return {
      safetyStatus: 'HALTED',
      indicators: offlineIndicators,
    };
  }
}

// ============================================
// Main Hook
// ============================================

const DEFAULT_HEALTH: SystemHealth = {
  safetyStatus: 'DEGRADED',
  indicators: [],
};

export function useGovernorData() {
  const [data, setData] = useState<GovernorData>({
    globalStatus: DEFAULT_GLOBAL_STATUS,
    marketContext: DEFAULT_MARKET_CONTEXT,
    policies: [],
    decisions: [],
    health: DEFAULT_HEALTH,
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
