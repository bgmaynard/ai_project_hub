import { useQuery } from '@tanstack/react-query'
import api from '../services/api'

// Phase Badge Component
function PhaseBadge({ phase, confidence }: { phase: string; confidence: number }) {
  const phaseColors: Record<string, string> = {
    PRE_MARKET: 'bg-purple-600',
    OPEN_IGNITION: 'bg-orange-600',
    STRUCTURED_MOMENTUM: 'bg-green-600',
    MIDDAY_COMPRESSION: 'bg-gray-600',
    POWER_HOUR: 'bg-blue-600',
    AFTER_HOURS: 'bg-indigo-600',
    CLOSED: 'bg-gray-700'
  }

  const phaseDescriptions: Record<string, string> = {
    PRE_MARKET: 'Pre-market trading (4:00-9:30 AM ET)',
    OPEN_IGNITION: 'High volatility open (9:30-9:45 AM ET)',
    STRUCTURED_MOMENTUM: 'Post-open momentum phase (9:45-11:30 AM ET)',
    MIDDAY_COMPRESSION: 'Low volatility midday (11:30 AM-2:00 PM ET)',
    POWER_HOUR: 'End of day momentum (2:00-4:00 PM ET)',
    AFTER_HOURS: 'After-hours trading (4:00-8:00 PM ET)',
    CLOSED: 'Market closed'
  }

  return (
    <div className={`${phaseColors[phase] || 'bg-gray-600'} rounded-lg p-6 text-white`}>
      <div className="text-sm uppercase tracking-wider opacity-75">Current Phase</div>
      <div className="text-3xl font-bold mt-1">{phase.replace(/_/g, ' ')}</div>
      <div className="text-sm mt-2 opacity-90">{phaseDescriptions[phase]}</div>
      <div className="mt-4 flex items-center gap-2">
        <div className="flex-1 bg-white/20 rounded-full h-2">
          <div
            className="bg-white rounded-full h-2 transition-all"
            style={{ width: `${confidence}%` }}
          />
        </div>
        <span className="text-sm font-medium">{confidence}%</span>
      </div>
      <div className="text-xs mt-1 opacity-75">Phase Confidence</div>
    </div>
  )
}

// Strategy Card Component
function StrategyCard({
  name,
  enabled,
  reason,
  suppressedCount
}: {
  name: string
  enabled: boolean
  reason?: string
  suppressedCount?: number
}) {
  return (
    <div
      className={`rounded-lg p-4 border ${
        enabled
          ? 'bg-green-900/20 border-green-600/50'
          : 'bg-red-900/20 border-red-600/50'
      }`}
    >
      <div className="flex items-center justify-between">
        <span className="font-medium">{name}</span>
        <span
          className={`px-2 py-0.5 rounded text-xs font-bold ${
            enabled ? 'bg-green-600 text-white' : 'bg-red-600 text-white'
          }`}
        >
          {enabled ? 'ON' : 'OFF'}
        </span>
      </div>
      {reason && (
        <div className="mt-2 text-xs text-text-secondary">
          Why: {reason}
        </div>
      )}
      {suppressedCount !== undefined && suppressedCount > 0 && (
        <div className="mt-1 text-xs text-status-yellow">
          {suppressedCount} signals suppressed
        </div>
      )}
    </div>
  )
}

// Exploration Policy Panel
function ExplorationPanel({
  policy
}: {
  policy: {
    level: string
    scouts_enabled: boolean
    max_scouts_per_hour: number
    scout_size_multiplier: number
  }
}) {
  const levelColors: Record<string, string> = {
    DISABLED: 'text-status-red',
    MINIMAL: 'text-status-yellow',
    NORMAL: 'text-status-green',
    AGGRESSIVE: 'text-blue-400'
  }

  return (
    <div className="bg-bg-secondary rounded-lg p-6">
      <h3 className="text-lg font-bold mb-4">Exploration Policy</h3>
      <div className="grid grid-cols-2 gap-4">
        <div>
          <div className="text-sm text-text-secondary">Level</div>
          <div className={`text-xl font-bold ${levelColors[policy.level] || 'text-text-primary'}`}>
            {policy.level}
          </div>
        </div>
        <div>
          <div className="text-sm text-text-secondary">Scouts</div>
          <div className={`text-xl font-bold ${policy.scouts_enabled ? 'text-status-green' : 'text-status-red'}`}>
            {policy.scouts_enabled ? 'ENABLED' : 'DISABLED'}
          </div>
        </div>
        <div>
          <div className="text-sm text-text-secondary">Max Scouts/Hour</div>
          <div className="text-xl font-bold">{policy.max_scouts_per_hour}</div>
        </div>
        <div>
          <div className="text-sm text-text-secondary">Scout Size</div>
          <div className="text-xl font-bold">{Math.round(policy.scout_size_multiplier * 100)}%</div>
        </div>
      </div>
    </div>
  )
}

// Phase Lock Indicator
function PhaseLockIndicator({
  lock
}: {
  lock: { locked: boolean; remaining_seconds: number; reason: string }
}) {
  if (!lock.locked) {
    return (
      <div className="bg-bg-tertiary rounded-lg p-4 text-center">
        <span className="text-status-green">Phase unlocked - can transition</span>
      </div>
    )
  }

  const minutes = Math.floor(lock.remaining_seconds / 60)
  const seconds = lock.remaining_seconds % 60

  return (
    <div className="bg-status-yellow/20 border border-status-yellow/50 rounded-lg p-4">
      <div className="flex items-center justify-between">
        <div>
          <div className="text-sm text-status-yellow font-medium">Phase Locked</div>
          <div className="text-xs text-text-secondary mt-1">{lock.reason}</div>
        </div>
        <div className="text-right">
          <div className="text-2xl font-bold text-status-yellow">
            {minutes}:{seconds.toString().padStart(2, '0')}
          </div>
          <div className="text-xs text-text-secondary">remaining</div>
        </div>
      </div>
    </div>
  )
}

// Main Phase & Strategy Component
export default function PhaseStrategy() {
  const { data: phaseStatus, isLoading, error } = useQuery({
    queryKey: ['phaseStatus'],
    queryFn: () => api.getPhaseStatus(),
    refetchInterval: 3000
  })

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-status-yellow"></div>
      </div>
    )
  }

  if (error || !phaseStatus) {
    return (
      <div className="bg-bg-secondary rounded-lg p-6 text-center">
        <p className="text-status-red">Failed to load phase data</p>
        <p className="text-text-secondary text-sm mt-2">
          Make sure /api/phase/current is available
        </p>
      </div>
    )
  }

  return (
    <div className="max-w-5xl mx-auto space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-2xl font-bold mb-2">Market Phase & Strategy Routing</h2>
        <p className="text-text-secondary">
          Which strategies are allowed right now - and why?
        </p>
      </div>

      {/* Phase & Lock Status */}
      <div className="grid grid-cols-3 gap-6">
        <div className="col-span-2">
          <PhaseBadge
            phase={phaseStatus.current_phase}
            confidence={phaseStatus.phase_confidence}
          />
        </div>
        <div className="flex flex-col justify-center">
          <PhaseLockIndicator lock={phaseStatus.phase_lock} />
        </div>
      </div>

      {/* Strategies Grid */}
      <div className="grid grid-cols-2 gap-6">
        {/* Enabled Strategies */}
        <div className="bg-bg-secondary rounded-lg p-6">
          <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
            <span className="w-3 h-3 bg-status-green rounded-full"></span>
            Enabled Strategies
          </h3>
          {phaseStatus.strategies_enabled.length === 0 ? (
            <p className="text-text-secondary text-center py-4">No strategies enabled</p>
          ) : (
            <div className="space-y-3">
              {phaseStatus.strategies_enabled.map((strategy: any) => (
                <StrategyCard
                  key={typeof strategy === 'string' ? strategy : strategy.name}
                  name={typeof strategy === 'string' ? strategy : strategy.name}
                  enabled={true}
                  reason="Allowed in current phase"
                />
              ))}
            </div>
          )}
        </div>

        {/* Disabled Strategies */}
        <div className="bg-bg-secondary rounded-lg p-6">
          <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
            <span className="w-3 h-3 bg-status-red rounded-full"></span>
            Disabled Strategies
          </h3>
          {phaseStatus.strategies_disabled.length === 0 ? (
            <p className="text-text-secondary text-center py-4">No strategies disabled</p>
          ) : (
            <div className="space-y-3">
              {phaseStatus.strategies_disabled.map((strategy: any) => (
                <StrategyCard
                  key={typeof strategy === 'string' ? strategy : strategy.name}
                  name={typeof strategy === 'string' ? strategy : strategy.name}
                  enabled={false}
                  reason={typeof strategy === 'object' ? strategy.reason : 'Not allowed in current phase'}
                  suppressedCount={typeof strategy === 'object' ? strategy.suppressed_count : undefined}
                />
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Exploration Policy */}
      <ExplorationPanel policy={phaseStatus.exploration_policy} />

      {/* Phase Schedule Reference */}
      <div className="bg-bg-secondary rounded-lg p-6">
        <h3 className="text-lg font-bold mb-4">Phase Schedule Reference</h3>
        <div className="grid grid-cols-7 gap-2 text-xs">
          {[
            { phase: 'PRE_MARKET', time: '4:00-9:30', color: 'bg-purple-600' },
            { phase: 'OPEN_IGNITION', time: '9:30-9:45', color: 'bg-orange-600' },
            { phase: 'STRUCTURED', time: '9:45-11:30', color: 'bg-green-600' },
            { phase: 'MIDDAY', time: '11:30-14:00', color: 'bg-gray-600' },
            { phase: 'POWER_HOUR', time: '14:00-16:00', color: 'bg-blue-600' },
            { phase: 'AFTER_HOURS', time: '16:00-20:00', color: 'bg-indigo-600' },
            { phase: 'CLOSED', time: '20:00-4:00', color: 'bg-gray-700' }
          ].map((p) => (
            <div
              key={p.phase}
              className={`${p.color} rounded p-2 text-center ${
                phaseStatus.current_phase === p.phase.replace('STRUCTURED', 'STRUCTURED_MOMENTUM').replace('MIDDAY', 'MIDDAY_COMPRESSION')
                  ? 'ring-2 ring-white'
                  : 'opacity-60'
              }`}
            >
              <div className="font-medium truncate">{p.phase.replace('_', ' ')}</div>
              <div className="opacity-75">{p.time}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
