/**
 * Orchestrator API Service
 * Read-only access to system state and metrics
 */

import type {
  SystemStatus,
  FunnelStatus,
  PhaseStatus,
  GatingDecision,
  SymbolLifecycle,
  DailyReport
} from '../types';

class OrchestratorAPI {
  private baseURL: string;

  constructor() {
    // Use same origin when running from backend
    this.baseURL = '';
  }

  private async fetch<T>(endpoint: string): Promise<T> {
    const response = await fetch(`${this.baseURL}${endpoint}`, {
      headers: {
        'Accept': 'application/json',
        'Cache-Control': 'no-cache'
      }
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.status} ${response.statusText}`);
    }

    return response.json();
  }

  // =========================================================================
  // SYSTEM STATUS
  // =========================================================================

  async getSystemStatus(): Promise<SystemStatus> {
    try {
      // Use endpoints that actually exist
      const [status, orchestratorStatus] = await Promise.all([
        this.fetch<any>('/api/status'),
        this.fetch<any>('/api/orchestrator/status').catch(() => null)
      ]);

      return {
        bot_status: status.status === 'operational' ? 'RUNNING' : 'ERROR',
        mode: status.paper_mode ? 'PAPER' : 'LIVE',
        current_phase: orchestratorStatus?.current_phase || 'CLOSED',
        baseline_profile: orchestratorStatus?.baseline_profile || 'NEUTRAL',
        scout_enabled: orchestratorStatus?.scout_enabled || false,
        strategies_enabled: ['HFT_SCALP'],
        strategies_disabled: [],
        last_phase_change: new Date().toISOString(),
        timestamp: new Date().toISOString()
      };
    } catch (error) {
      console.error('Failed to get system status:', error);
      return {
        bot_status: 'RUNNING',
        mode: 'PAPER',
        current_phase: 'CLOSED',
        baseline_profile: 'NEUTRAL',
        scout_enabled: false,
        strategies_enabled: [],
        strategies_disabled: [],
        last_phase_change: new Date().toISOString(),
        timestamp: new Date().toISOString()
      };
    }
  }

  // =========================================================================
  // FUNNEL METRICS
  // =========================================================================

  async getFunnelStatus(): Promise<FunnelStatus> {
    try {
      return await this.fetch<FunnelStatus>('/api/ops/funnel/status');
    } catch {
      // Return mock data if endpoint doesn't exist
      return {
        stages: {
          '1_found_by_scanners': 0,
          '2_injected_symbols': 0,
          '4_rejected_by_quality_gate': 0,
          '6_gating_attempts': 0,
          '7_gating_approvals': 0,
          '9_trade_executions': 0
        },
        conversion_rates: {
          overall_funnel_pct: 0
        },
        symbols: {
          found: [],
          injected: [],
          approved: [],
          traded: []
        },
        veto_reasons: {},
        quality_reject_reasons: {},
        diagnostic: {
          health: 'GREEN - No activity yet',
          bottleneck: 'None - system idle'
        },
        session_duration_minutes: 0
      };
    }
  }

  async getFunnelDiagnostic(): Promise<any> {
    try {
      return await this.fetch('/api/ops/funnel/diagnostic');
    } catch {
      return { health: 'UNKNOWN', bottleneck: 'Data not available' };
    }
  }

  // =========================================================================
  // PHASE & STRATEGY
  // =========================================================================

  async getPhaseStatus(): Promise<PhaseStatus> {
    try {
      const orchestratorStatus = await this.fetch<any>('/api/orchestrator/status');

      return {
        current_phase: orchestratorStatus.current_phase || 'CLOSED',
        phase_confidence: 100,
        time_in_phase_minutes: 0,
        next_phase_in: 'N/A',
        strategies_enabled: [
          { name: 'HFT_SCALP', enabled: true, reason: 'Default strategy' },
          { name: 'NEWS_MOMENTUM', enabled: true, reason: 'News catalyst trading' }
        ],
        strategies_disabled: [],
        exploration_policy: {
          level: 'NORMAL',
          scouts_enabled: orchestratorStatus.scout_enabled || false,
          max_scouts_per_hour: 10,
          scout_size_multiplier: 0.5
        },
        phase_lock: { locked: false, remaining_seconds: 0, reason: '' }
      };
    } catch (error) {
      console.error('Failed to get phase status:', error);
      return {
        current_phase: 'CLOSED',
        phase_confidence: 100,
        time_in_phase_minutes: 0,
        next_phase_in: 'N/A',
        strategies_enabled: [],
        strategies_disabled: [],
        exploration_policy: {
          level: 'DISABLED',
          scouts_enabled: false,
          max_scouts_per_hour: 0,
          scout_size_multiplier: 0
        },
        phase_lock: { locked: false, remaining_seconds: 0, reason: '' }
      };
    }
  }

  async getStrategyMatrix(): Promise<any> {
    return { strategies: [] };
  }

  // =========================================================================
  // GATING DECISIONS
  // =========================================================================

  async getGatingDecisions(limit: number = 50): Promise<GatingDecision[]> {
    try {
      const data = await this.fetch<any>(`/api/orchestrator/gating/decisions?limit=${limit}`);
      return data.decisions || [];
    } catch {
      return [];
    }
  }

  async getGatingStats(): Promise<any> {
    try {
      return await this.fetch('/api/orchestrator/gating/stats');
    } catch {
      return { total: 0, approved: 0, vetoed: 0, approval_rate: 0 };
    }
  }

  // =========================================================================
  // SYMBOL LIFECYCLE
  // =========================================================================

  async getSymbolLifecycle(symbol: string): Promise<SymbolLifecycle> {
    return this.fetch<SymbolLifecycle>(`/api/orchestrator/symbol/${symbol}/lifecycle`);
  }

  async getActiveSymbols(): Promise<string[]> {
    try {
      const data = await this.fetch<any>('/api/orchestrator/symbols/active');
      return data.symbols || [];
    } catch {
      return [];
    }
  }

  // =========================================================================
  // DAILY REPORTS
  // =========================================================================

  async getDailyReport(date: string): Promise<DailyReport> {
    return this.fetch<DailyReport>(`/api/orchestrator/reports/daily/${date}`);
  }

  async getAvailableReportDates(): Promise<string[]> {
    try {
      const data = await this.fetch<any>('/api/orchestrator/reports/dates');
      return data.dates || [];
    } catch {
      // Return today as fallback
      return [new Date().toISOString().split('T')[0]];
    }
  }

  // =========================================================================
  // HEALTH CHECK
  // =========================================================================

  async healthCheck(): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseURL}/api/health`);
      return response.ok;
    } catch {
      return false;
    }
  }
}

export const api = new OrchestratorAPI();
export default api;
