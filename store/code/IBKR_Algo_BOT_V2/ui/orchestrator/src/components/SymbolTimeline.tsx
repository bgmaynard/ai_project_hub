import { useState, useEffect } from 'react'
import { useQuery } from '@tanstack/react-query'
import { useSearchParams } from 'react-router-dom'
import api from '../services/api'
import type { SymbolLifecycle, SymbolLifecycleEvent } from '../types'

// Event Type Icons & Colors
const eventConfig: Record<string, { icon: string; color: string; label: string }> = {
  DISCOVERED: { icon: '🔍', color: 'bg-blue-600', label: 'Discovered' },
  INJECTED: { icon: '💉', color: 'bg-indigo-600', label: 'Injected' },
  QUALITY_PASSED: { icon: '✓', color: 'bg-green-500', label: 'Quality Passed' },
  QUALITY_REJECTED: { icon: '✗', color: 'bg-red-500', label: 'Quality Rejected' },
  SCOUT_ATTEMPT: { icon: '🎯', color: 'bg-purple-600', label: 'Scout Attempt' },
  SCOUT_CONFIRMED: { icon: '✓', color: 'bg-green-600', label: 'Scout Confirmed' },
  SCOUT_STOPPED: { icon: '⛔', color: 'bg-red-600', label: 'Scout Stopped' },
  GATING_ATTEMPT: { icon: '🚦', color: 'bg-yellow-600', label: 'Gating Attempt' },
  GATING_APPROVED: { icon: '✓', color: 'bg-green-600', label: 'Approved' },
  GATING_VETOED: { icon: '🚫', color: 'bg-red-600', label: 'Vetoed' },
  TRADE_EXECUTED: { icon: '💰', color: 'bg-emerald-600', label: 'Executed' },
  TRADE_CLOSED: { icon: '📊', color: 'bg-teal-600', label: 'Closed' },
  PHASE_CHANGE: { icon: '⏱️', color: 'bg-orange-600', label: 'Phase Change' },
  COOLDOWN: { icon: '❄️', color: 'bg-cyan-600', label: 'Cooldown' },
  HANDOFF: { icon: '🔄', color: 'bg-violet-600', label: 'Handoff' }
}

// Timeline Event Component
function TimelineEvent({ event, isLast }: { event: SymbolLifecycleEvent; isLast: boolean }) {
  const [expanded, setExpanded] = useState(false)
  const config = eventConfig[event.event_type] || {
    icon: '📌',
    color: 'bg-gray-600',
    label: event.event_type
  }

  return (
    <div className="relative pl-8">
      {/* Connector Line */}
      {!isLast && (
        <div className="absolute left-3 top-8 bottom-0 w-0.5 bg-border-color"></div>
      )}

      {/* Event Dot */}
      <div
        className={`absolute left-0 top-1 w-6 h-6 rounded-full ${config.color} flex items-center justify-center text-xs`}
      >
        {config.icon}
      </div>

      {/* Event Content */}
      <div
        className="bg-bg-secondary rounded-lg p-4 mb-4 cursor-pointer hover:bg-bg-tertiary transition-colors"
        onClick={() => setExpanded(!expanded)}
      >
        <div className="flex items-center justify-between">
          <div>
            <span className="font-medium">{config.label}</span>
            {event.reason && (
              <span className="text-text-secondary text-sm ml-2">- {event.reason}</span>
            )}
          </div>
          <span className="text-xs text-text-secondary">
            {new Date(event.timestamp).toLocaleTimeString()}
          </span>
        </div>

        {/* Expanded Metrics */}
        {expanded && event.metrics && Object.keys(event.metrics).length > 0 && (
          <div className="mt-3 pt-3 border-t border-border-color">
            <div className="text-xs text-text-secondary mb-2">Metrics at this moment:</div>
            <div className="grid grid-cols-3 gap-2 text-xs">
              {Object.entries(event.metrics).map(([key, value]) => (
                <div key={key} className="bg-bg-tertiary rounded px-2 py-1">
                  <span className="text-text-secondary">{key}: </span>
                  <span className="font-medium">{String(value)}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

// Symbol Selector Component
function SymbolSelector({
  symbols,
  selectedSymbol,
  onSelect
}: {
  symbols: string[]
  selectedSymbol: string
  onSelect: (symbol: string) => void
}) {
  return (
    <div className="bg-bg-secondary rounded-lg p-4 mb-6">
      <div className="flex items-center gap-4">
        <label className="text-sm text-text-secondary">Select Symbol:</label>
        <select
          value={selectedSymbol}
          onChange={(e) => onSelect(e.target.value)}
          className="bg-bg-tertiary border border-border-color rounded px-4 py-2 flex-1 max-w-xs"
        >
          <option value="">Choose a symbol...</option>
          {symbols.map((s) => (
            <option key={s} value={s}>{s}</option>
          ))}
        </select>
        <span className="text-xs text-text-secondary">
          {symbols.length} symbols in pipeline today
        </span>
      </div>
    </div>
  )
}

// Symbol Summary Card
function SymbolSummaryCard({ lifecycle }: { lifecycle: SymbolLifecycle }) {
  const eventCounts = lifecycle.events.reduce((acc, e) => {
    acc[e.event_type] = (acc[e.event_type] || 0) + 1
    return acc
  }, {} as Record<string, number>)

  const wasTraded = eventCounts['TRADE_EXECUTED'] > 0
  const wasVetoed = eventCounts['GATING_VETOED'] > 0

  return (
    <div className="bg-bg-secondary rounded-lg p-6 mb-6">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-2xl font-bold">{lifecycle.symbol}</h3>
          <div className="text-sm text-text-secondary mt-1">
            First seen: {new Date(lifecycle.first_seen).toLocaleTimeString()} |
            Last event: {new Date(lifecycle.last_event).toLocaleTimeString()}
          </div>
        </div>
        <div className="text-right">
          <div className={`text-lg font-bold ${
            wasTraded ? 'text-status-green' :
            wasVetoed ? 'text-status-red' :
            'text-status-yellow'
          }`}>
            {lifecycle.current_state}
          </div>
          <div className="text-sm text-text-secondary">
            {lifecycle.events.length} events recorded
          </div>
        </div>
      </div>

      {/* Quick Stats */}
      <div className="mt-4 pt-4 border-t border-border-color grid grid-cols-5 gap-4">
        <div className="text-center">
          <div className="text-lg font-bold text-blue-400">
            {eventCounts['SCOUT_ATTEMPT'] || 0}
          </div>
          <div className="text-xs text-text-secondary">Scout Attempts</div>
        </div>
        <div className="text-center">
          <div className="text-lg font-bold text-green-400">
            {eventCounts['SCOUT_CONFIRMED'] || 0}
          </div>
          <div className="text-xs text-text-secondary">Confirmed</div>
        </div>
        <div className="text-center">
          <div className="text-lg font-bold text-yellow-400">
            {eventCounts['GATING_ATTEMPT'] || 0}
          </div>
          <div className="text-xs text-text-secondary">Gate Checks</div>
        </div>
        <div className="text-center">
          <div className="text-lg font-bold text-red-400">
            {eventCounts['GATING_VETOED'] || 0}
          </div>
          <div className="text-xs text-text-secondary">Vetoes</div>
        </div>
        <div className="text-center">
          <div className="text-lg font-bold text-emerald-400">
            {eventCounts['TRADE_EXECUTED'] || 0}
          </div>
          <div className="text-xs text-text-secondary">Trades</div>
        </div>
      </div>
    </div>
  )
}

// Empty State
function EmptyState() {
  return (
    <div className="bg-bg-secondary rounded-lg p-12 text-center">
      <div className="text-4xl mb-4">📈</div>
      <h3 className="text-xl font-bold mb-2">Select a Symbol</h3>
      <p className="text-text-secondary">
        Choose a symbol from the dropdown to see its complete lifecycle.
        <br />
        You can also click any symbol from the Funnel view to jump here.
      </p>
    </div>
  )
}

// Main Symbol Timeline Component
export default function SymbolTimeline() {
  const [searchParams, setSearchParams] = useSearchParams()
  const [selectedSymbol, setSelectedSymbol] = useState(searchParams.get('symbol') || '')

  // Get active symbols from funnel
  const { data: activeSymbols } = useQuery({
    queryKey: ['activeSymbols'],
    queryFn: () => api.getActiveSymbols(),
    refetchInterval: 10000
  })

  // Get lifecycle for selected symbol
  const { data: lifecycle, isLoading, error } = useQuery({
    queryKey: ['symbolLifecycle', selectedSymbol],
    queryFn: () => api.getSymbolLifecycle(selectedSymbol),
    enabled: !!selectedSymbol,
    refetchInterval: 5000
  })

  // Update URL when symbol changes
  useEffect(() => {
    if (selectedSymbol) {
      setSearchParams({ symbol: selectedSymbol })
    } else {
      setSearchParams({})
    }
  }, [selectedSymbol, setSearchParams])

  const handleSymbolSelect = (symbol: string) => {
    setSelectedSymbol(symbol)
  }

  return (
    <div className="max-w-4xl mx-auto">
      {/* Header */}
      <div className="mb-6">
        <h2 className="text-2xl font-bold mb-2">Symbol Lifecycle Timeline</h2>
        <p className="text-text-secondary">
          What happened to this stock end-to-end? Click events to see metrics.
        </p>
      </div>

      {/* Symbol Selector */}
      <SymbolSelector
        symbols={activeSymbols || []}
        selectedSymbol={selectedSymbol}
        onSelect={handleSymbolSelect}
      />

      {/* Content */}
      {!selectedSymbol ? (
        <EmptyState />
      ) : isLoading ? (
        <div className="flex items-center justify-center h-64">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-status-yellow"></div>
        </div>
      ) : error || !lifecycle ? (
        <div className="bg-bg-secondary rounded-lg p-6 text-center">
          <p className="text-status-yellow">No lifecycle data for {selectedSymbol}</p>
          <p className="text-text-secondary text-sm mt-2">
            This symbol may not have been processed today yet.
          </p>
        </div>
      ) : (
        <>
          {/* Summary Card */}
          <SymbolSummaryCard lifecycle={lifecycle} />

          {/* Timeline */}
          <div className="bg-bg-secondary rounded-lg p-6">
            <h4 className="text-lg font-bold mb-4">Event Timeline</h4>
            <div>
              {lifecycle.events.map((event, index) => (
                <TimelineEvent
                  key={`${event.timestamp}-${index}`}
                  event={event}
                  isLast={index === lifecycle.events.length - 1}
                />
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  )
}
