/**
 * System Governor Panel Type Definitions
 *
 * Types for the governor dashboard focused on system oversight,
 * not trading features.
 */

// ============================================
// Global Trading Status
// ============================================

export type TradingMode = 'PAPER' | 'LIVE';
export type TradingWindow = 'OPEN' | 'CLOSED' | 'PRE_MARKET' | 'AFTER_HOURS';

// AI Posture - explicit system stance
export type AIPosture =
  | 'NO_TRADE'    // Outside trading window (by design)
  | 'DEFENSIVE'   // Cautious mode (high volatility, degraded health)
  | 'ACTIVE'      // Normal trading operations
  | 'LOCKED';     // Kill switch engaged

// System State - for clear UI display of WHY system is idle
export type SystemState =
  | 'ACTIVE'              // All systems go, trading enabled
  | 'READY'               // All connected, trading disabled
  | 'MARKET_CLOSED'       // Calendar: market not open
  | 'DATA_OFFLINE'        // Market open but no data feed
  | 'SERVICE_NOT_RUNNING' // Service process not started
  | 'DISCONNECTED'        // Service started but connection lost
  | 'PARTIAL';            // Some services up, some down

export interface KillSwitchStatus {
  active: boolean;
  reason: string | null;
  cooldownSeconds: number;
}

export interface GlobalStatus {
  mode: TradingMode;
  tradingWindow: TradingWindow;
  windowTime: string;
  aiPosture: AIPosture;
  aiPostureReason: string;  // Plain English explanation
  killSwitch: KillSwitchStatus;
  systemState: SystemState;  // Clear indicator of WHY system is idle
  systemStateReason: string; // Human-readable reason
}

// ============================================
// Market Context
// ============================================

export type MarketRegime = 'TRENDING_UP' | 'TRENDING_DOWN' | 'RANGING' | 'VOLATILE';
export type VolatilityLevel = 'LOW' | 'NORMAL' | 'HIGH' | 'EXTREME';
export type LiquidityLevel = 'ADEQUATE' | 'THIN' | 'POOR';
export type DataFreshness = 'FRESH' | 'STALE' | 'OFFLINE';

export interface MarketContext {
  regime: MarketRegime;
  regimeConfidence: number;
  volatility: VolatilityLevel;
  volatilityPct: number;
  liquidity: LiquidityLevel;
  dataAge: number;
  dataFreshness: DataFreshness;
  lastUpdate: string;
  aiInterpretation: string;  // Plain English policy-generated reason for posture
}

// ============================================
// Strategy Policies
// ============================================

export type PolicyStatus = 'ENABLED' | 'DISABLED' | 'PAUSED';

export interface StrategyPolicy {
  id: string;
  name: string;
  status: PolicyStatus;
  reason: string;
  vetoCount: number;
  cooldownRemaining?: number;
}

// ============================================
// AI Decisions
// ============================================

export type DecisionAction = 'APPROVED' | 'VETOED' | 'EXIT' | 'NO_ACTION';
export type DecisionType = 'entry' | 'exit' | 'passive';

export interface AIDecision {
  timestamp: string;
  action: DecisionAction;
  symbol: string;
  type: DecisionType;
  reasons: string[];
  pnl?: number;
}

// ============================================
// System Health
// ============================================

export type HealthStatus = 'HEALTHY' | 'DEGRADED' | 'ERROR' | 'OFFLINE';
export type SystemSafetyStatus = 'SAFE' | 'DEGRADED' | 'HALTED';

export interface HealthIndicator {
  name: string;
  status: HealthStatus;
  detail: string;
}

export interface SystemHealth {
  safetyStatus: SystemSafetyStatus;
  indicators: HealthIndicator[];
}

// ============================================
// Combined Governor Data
// ============================================

export interface GovernorData {
  globalStatus: GlobalStatus;
  marketContext: MarketContext;
  policies: StrategyPolicy[];
  decisions: AIDecision[];
  health: SystemHealth;
  lastFetch: string;
}

// ============================================
// Polling Configuration
// ============================================

// TODO: [STABILITY 2026-01-08] Consider increasing decisions interval to 5s
// if UI stalling persists. 3s may stack requests on slow API responses.
export const POLL_INTERVALS = {
  globalStatus: 5000,    // 5s - critical
  marketContext: 10000,  // 10s - moderate
  policies: 15000,       // 15s - slow-changing
  decisions: 3000,       // 3s - real-time feel (may cause stacking)
  health: 30000,         // 30s - background
} as const;

// ============================================
// Color Constants
// ============================================

export const GOVERNOR_COLORS = {
  success: '#4ec9b0',   // Green - Healthy, Enabled, Approved
  warning: '#dcdcaa',   // Yellow - Warning, Paused, Cooldown
  error: '#f48771',     // Red - Error, Disabled, Vetoed, Kill Switch
  active: '#007acc',    // Blue - Active, Processing
  neutral: '#888888',   // Gray - Neutral, Standby
} as const;

// ============================================
// Status Color Helpers
// ============================================

export function getStatusColor(status: string): string {
  switch (status) {
    case 'HEALTHY':
    case 'ENABLED':
    case 'APPROVED':
    case 'CONNECTED':
    case 'ONLINE':
    case 'READY':
    case 'FRESH':
    case 'SAFE':
    case 'ACTIVE':
    case 'UP':
      return GOVERNOR_COLORS.success;

    case 'DEGRADED':
    case 'PAUSED':
    case 'WARNING':
    case 'STALE':
    case 'ARMED':
    case 'DEFENSIVE':
    case 'NO_ACTION':
    case 'PARTIAL':
    case 'MARKET_CLOSED':
      return GOVERNOR_COLORS.warning;

    case 'ERROR':
    case 'DISABLED':
    case 'VETOED':
    case 'OFFLINE':
    case 'EXIT':
    case 'HALTED':
    case 'LOCKED':
    case 'NO_TRADE':
    case 'DATA_OFFLINE':
    case 'SERVICE_NOT_RUNNING':
    case 'DISCONNECTED':
    case 'DOWN':
      return GOVERNOR_COLORS.error;

    case 'PROCESSING':
      return GOVERNOR_COLORS.active;

    default:
      return GOVERNOR_COLORS.neutral;
  }
}

export function getRegimeColor(regime: MarketRegime): string {
  switch (regime) {
    case 'TRENDING_UP':
      return GOVERNOR_COLORS.success;
    case 'TRENDING_DOWN':
      return GOVERNOR_COLORS.error;
    case 'RANGING':
      return GOVERNOR_COLORS.warning;
    case 'VOLATILE':
      return GOVERNOR_COLORS.error;
    default:
      return GOVERNOR_COLORS.neutral;
  }
}
