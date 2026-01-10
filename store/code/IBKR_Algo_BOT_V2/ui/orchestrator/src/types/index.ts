// System Status Types
export interface SystemStatus {
  bot_status: 'RUNNING' | 'PAUSED' | 'ERROR';
  mode: 'PAPER' | 'LIVE';
  current_phase: string;
  baseline_profile: string;
  scout_enabled: boolean;
  strategies_enabled: string[];
  strategies_disabled: string[];
  last_phase_change: string;
  timestamp: string;
}

export type MarketPhase =
  | 'PRE_MARKET'
  | 'OPEN_IGNITION'
  | 'STRUCTURED_MOMENTUM'
  | 'MIDDAY_COMPRESSION'
  | 'POWER_HOUR'
  | 'AFTER_HOURS'
  | 'CLOSED';

export type BaselineProfile = 'CONSERVATIVE' | 'NEUTRAL' | 'AGGRESSIVE';

// Funnel Types
export interface FunnelStatus {
  timestamp?: string;
  session_start?: string;
  session_duration_minutes: number;
  stages: Record<string, number>;
  conversion_rates: Record<string, number>;
  veto_reasons: Record<string, number>;
  quality_reject_reasons: Record<string, number>;
  scanner_sources?: Record<string, number>;
  symbols: {
    found: string[];
    injected: string[];
    approved: string[];
    vetoed?: string[];
    traded: string[];
  };
  diagnostic: {
    bottleneck: string;
    health: string;
  };
  scout_metrics?: {
    attempts: number;
    confirmed: number;
    failed: number;
    to_trade: number;
    confirmation_rate: number;
    escalation_rate: number;
    block_reasons?: Record<string, number>;
  };
}

// Phase & Strategy Types
export interface PhaseStatus {
  current_phase: string;
  phase_confidence: number;
  time_in_phase_minutes: number;
  next_phase_in: string;
  strategies_enabled: StrategyInfo[];
  strategies_disabled: StrategyInfo[];
  exploration_policy: ExplorationPolicy;
  phase_lock: PhaseLock;
}

export interface StrategyInfo {
  name: string;
  enabled: boolean;
  reason?: string;
  suppressed_count?: number;
}

export interface ExplorationPolicy {
  level: string;
  scouts_enabled: boolean;
  max_scouts_per_hour: number;
  scout_size_multiplier: number;
}

export interface PhaseLock {
  locked: boolean;
  remaining_seconds: number;
  reason: string;
}

// Gating Decision Types
export interface GatingDecision {
  timestamp: string;
  symbol: string;
  strategy: string;
  decision: 'APPROVED' | 'VETOED';
  primary_reason: string;
  secondary_factors: string[];
  chronos_regime: string;
  chronos_confidence: number;
  ats_state: string;
  micro_override_applied?: boolean;
}

// Symbol Lifecycle Types
export interface SymbolLifecycleEvent {
  timestamp: string;
  event_type: string;
  reason?: string;
  metrics?: Record<string, number | string>;
}

export interface SymbolLifecycle {
  symbol: string;
  events: SymbolLifecycleEvent[];
  current_state: string;
  first_seen: string;
  last_event: string;
}

// Daily Report Types
export interface DailyReport {
  date: string;
  market_context: MarketContext;
  funnel_summary: FunnelSummary;
  strategy_activity: StrategyActivity[];
  phase_transitions: PhaseTransition[];
  top_veto_reasons: VetoReason[];
  trade_summary: TradeSummary;
}

export interface MarketContext {
  market_breadth: number;
  small_cap_participation: number;
  gap_continuation_rate: number;
  dominant_regime: string;
}

export interface FunnelSummary {
  total_discovered: number;
  total_injected: number;
  total_gated: number;
  total_approved: number;
  total_executed: number;
  overall_conversion: number;
}

export interface StrategyActivity {
  strategy: string;
  signals_generated: number;
  signals_approved: number;
  signals_vetoed: number;
  trades_executed: number;
}

export interface PhaseTransition {
  from_phase: string;
  to_phase: string;
  timestamp: string;
  reason: string;
}

export interface VetoReason {
  reason: string;
  count: number;
  percentage: number;
}

export interface TradeSummary {
  total_trades: number;
  wins: number;
  losses: number;
  win_rate: number;
  total_pnl: number;
  avg_win: number;
  avg_loss: number;
}
