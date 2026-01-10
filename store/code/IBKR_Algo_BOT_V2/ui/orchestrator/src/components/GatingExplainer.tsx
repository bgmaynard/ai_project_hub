import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import api from '../services/api'
import type { GatingDecision } from '../types'

// Decision Row Component
function DecisionRow({ decision }: { decision: GatingDecision }) {
  const [expanded, setExpanded] = useState(false)

  return (
    <div
      className={`border rounded-lg overflow-hidden ${
        decision.decision === 'APPROVED'
          ? 'border-green-600/50 bg-green-900/10'
          : 'border-red-600/50 bg-red-900/10'
      }`}
    >
      {/* Main Row */}
      <div
        className="p-4 cursor-pointer hover:bg-bg-tertiary/30 transition-colors"
        onClick={() => setExpanded(!expanded)}
      >
        <div className="flex items-center gap-4">
          {/* Decision Badge */}
          <span
            className={`px-2 py-1 rounded text-xs font-bold ${
              decision.decision === 'APPROVED'
                ? 'bg-green-600 text-white'
                : 'bg-red-600 text-white'
            }`}
          >
            {decision.decision}
          </span>

          {/* Symbol */}
          <span className="font-bold text-lg w-16">{decision.symbol}</span>

          {/* Strategy */}
          <span className="text-text-secondary text-sm w-32">{decision.strategy}</span>

          {/* Primary Reason */}
          <span className="flex-1 text-sm truncate">
            {decision.primary_reason}
          </span>

          {/* Timestamp */}
          <span className="text-xs text-text-secondary">
            {new Date(decision.timestamp).toLocaleTimeString()}
          </span>

          {/* Expand Icon */}
          <span className="text-text-secondary">{expanded ? '▼' : '▶'}</span>
        </div>
      </div>

      {/* Expanded Details */}
      {expanded && (
        <div className="px-4 pb-4 border-t border-border-color pt-4 bg-bg-tertiary/20">
          <div className="grid grid-cols-3 gap-4 text-sm">
            {/* Left: Chronos Info */}
            <div>
              <div className="text-text-secondary text-xs mb-1">Chronos Context</div>
              <div className="space-y-1">
                <div>
                  Regime:{' '}
                  <span className={`font-medium ${
                    decision.chronos_regime === 'TRENDING_UP' ? 'text-status-green' :
                    decision.chronos_regime === 'TRENDING_DOWN' ? 'text-status-red' :
                    'text-status-yellow'
                  }`}>
                    {decision.chronos_regime}
                  </span>
                </div>
                <div>
                  Confidence:{' '}
                  <span className="font-medium">
                    {Math.round(decision.chronos_confidence * 100)}%
                  </span>
                </div>
              </div>
            </div>

            {/* Middle: ATS Info */}
            <div>
              <div className="text-text-secondary text-xs mb-1">ATS State</div>
              <div className={`font-medium ${
                decision.ats_state === 'ACTIVE' || decision.ats_state === 'CONFIRMED'
                  ? 'text-status-green'
                  : decision.ats_state === 'IGNITING'
                  ? 'text-status-yellow'
                  : 'text-status-red'
              }`}>
                {decision.ats_state}
              </div>
              {decision.micro_override_applied && (
                <div className="text-xs text-status-yellow mt-1">
                  Micro-override applied
                </div>
              )}
            </div>

            {/* Right: Secondary Factors */}
            <div>
              <div className="text-text-secondary text-xs mb-1">Secondary Factors</div>
              {decision.secondary_factors.length === 0 ? (
                <div className="text-text-secondary">None</div>
              ) : (
                <ul className="list-disc list-inside text-xs space-y-0.5">
                  {decision.secondary_factors.map((factor, i) => (
                    <li key={i}>{factor}</li>
                  ))}
                </ul>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

// Filter Controls Component
function FilterControls({
  filters,
  setFilters,
  symbols,
  strategies
}: {
  filters: {
    symbol: string
    strategy: string
    reason: string
    decision: string
  }
  setFilters: (f: any) => void
  symbols: string[]
  strategies: string[]
}) {
  return (
    <div className="bg-bg-secondary rounded-lg p-4 mb-6">
      <div className="flex items-center gap-4 flex-wrap">
        {/* Symbol Filter */}
        <div>
          <label className="text-xs text-text-secondary block mb-1">Symbol</label>
          <select
            value={filters.symbol}
            onChange={(e) => setFilters({ ...filters, symbol: e.target.value })}
            className="bg-bg-tertiary border border-border-color rounded px-3 py-1.5 text-sm"
          >
            <option value="">All Symbols</option>
            {symbols.map((s) => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
        </div>

        {/* Strategy Filter */}
        <div>
          <label className="text-xs text-text-secondary block mb-1">Strategy</label>
          <select
            value={filters.strategy}
            onChange={(e) => setFilters({ ...filters, strategy: e.target.value })}
            className="bg-bg-tertiary border border-border-color rounded px-3 py-1.5 text-sm"
          >
            <option value="">All Strategies</option>
            {strategies.map((s) => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
        </div>

        {/* Decision Filter */}
        <div>
          <label className="text-xs text-text-secondary block mb-1">Decision</label>
          <select
            value={filters.decision}
            onChange={(e) => setFilters({ ...filters, decision: e.target.value })}
            className="bg-bg-tertiary border border-border-color rounded px-3 py-1.5 text-sm"
          >
            <option value="">All Decisions</option>
            <option value="APPROVED">Approved</option>
            <option value="VETOED">Vetoed</option>
          </select>
        </div>

        {/* Reason Filter */}
        <div className="flex-1">
          <label className="text-xs text-text-secondary block mb-1">Reason Contains</label>
          <input
            type="text"
            value={filters.reason}
            onChange={(e) => setFilters({ ...filters, reason: e.target.value })}
            placeholder="Filter by reason..."
            className="w-full bg-bg-tertiary border border-border-color rounded px-3 py-1.5 text-sm"
          />
        </div>

        {/* Clear Filters */}
        <div className="self-end">
          <button
            onClick={() => setFilters({ symbol: '', strategy: '', reason: '', decision: '' })}
            className="text-sm text-text-secondary hover:text-text-primary"
          >
            Clear Filters
          </button>
        </div>
      </div>
    </div>
  )
}

// Stats Summary Component
function StatsSummary({ decisions }: { decisions: GatingDecision[] }) {
  const approved = decisions.filter((d) => d.decision === 'APPROVED').length
  const vetoed = decisions.filter((d) => d.decision === 'VETOED').length
  const total = decisions.length

  // Group by reason
  const reasonCounts: Record<string, number> = {}
  decisions
    .filter((d) => d.decision === 'VETOED')
    .forEach((d) => {
      reasonCounts[d.primary_reason] = (reasonCounts[d.primary_reason] || 0) + 1
    })

  const topReasons = Object.entries(reasonCounts)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 3)

  return (
    <div className="bg-bg-secondary rounded-lg p-6 mb-6">
      <div className="grid grid-cols-4 gap-6">
        <div className="text-center">
          <div className="text-3xl font-bold">{total}</div>
          <div className="text-sm text-text-secondary">Total Decisions</div>
        </div>
        <div className="text-center">
          <div className="text-3xl font-bold text-status-green">{approved}</div>
          <div className="text-sm text-text-secondary">Approved</div>
        </div>
        <div className="text-center">
          <div className="text-3xl font-bold text-status-red">{vetoed}</div>
          <div className="text-sm text-text-secondary">Vetoed</div>
        </div>
        <div className="text-center">
          <div className="text-3xl font-bold text-status-yellow">
            {total > 0 ? Math.round((approved / total) * 100) : 0}%
          </div>
          <div className="text-sm text-text-secondary">Approval Rate</div>
        </div>
      </div>

      {topReasons.length > 0 && (
        <div className="mt-4 pt-4 border-t border-border-color">
          <div className="text-sm text-text-secondary mb-2">Top Veto Reasons:</div>
          <div className="flex gap-4">
            {topReasons.map(([reason, count]) => (
              <div key={reason} className="text-xs">
                <span className="text-status-red font-medium">{count}x</span>{' '}
                <span className="text-text-secondary">{reason}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

// Main Gating Explainer Component
export default function GatingExplainer() {
  const [filters, setFilters] = useState({
    symbol: '',
    strategy: '',
    reason: '',
    decision: ''
  })

  const { data: decisions, isLoading, error } = useQuery({
    queryKey: ['gatingDecisions'],
    queryFn: () => api.getGatingDecisions(100),
    refetchInterval: 5000
  })

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-status-yellow"></div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="bg-bg-secondary rounded-lg p-6 text-center">
        <p className="text-status-red">Failed to load gating decisions</p>
        <p className="text-text-secondary text-sm mt-2">
          Make sure the gating endpoints are available
        </p>
      </div>
    )
  }

  const allDecisions = decisions || []

  // Extract unique values for filters
  const uniqueSymbols = [...new Set(allDecisions.map((d) => d.symbol))]
  const uniqueStrategies = [...new Set(allDecisions.map((d) => d.strategy))]

  // Apply filters
  const filteredDecisions = allDecisions.filter((d) => {
    if (filters.symbol && d.symbol !== filters.symbol) return false
    if (filters.strategy && d.strategy !== filters.strategy) return false
    if (filters.decision && d.decision !== filters.decision) return false
    if (filters.reason && !d.primary_reason.toLowerCase().includes(filters.reason.toLowerCase())) return false
    return true
  })

  return (
    <div className="max-w-6xl mx-auto">
      {/* Header */}
      <div className="mb-6">
        <h2 className="text-2xl font-bold mb-2">Gating Decision Explainer</h2>
        <p className="text-text-secondary">
          Why did the bot say NO? Click any decision to see full context.
        </p>
      </div>

      {/* Stats Summary */}
      <StatsSummary decisions={allDecisions} />

      {/* Filters */}
      <FilterControls
        filters={filters}
        setFilters={setFilters}
        symbols={uniqueSymbols}
        strategies={uniqueStrategies}
      />

      {/* Decision List */}
      <div className="space-y-2">
        {filteredDecisions.length === 0 ? (
          <div className="bg-bg-secondary rounded-lg p-8 text-center">
            <p className="text-text-secondary">No gating decisions match your filters</p>
          </div>
        ) : (
          filteredDecisions.map((decision, index) => (
            <DecisionRow key={`${decision.timestamp}-${decision.symbol}-${index}`} decision={decision} />
          ))
        )}
      </div>

      {/* Load More / Summary */}
      {filteredDecisions.length > 0 && (
        <div className="mt-4 text-center text-sm text-text-secondary">
          Showing {filteredDecisions.length} of {allDecisions.length} decisions
        </div>
      )}
    </div>
  )
}
