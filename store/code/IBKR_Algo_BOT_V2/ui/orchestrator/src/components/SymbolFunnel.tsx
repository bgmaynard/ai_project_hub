import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import api from '../services/api'

// Funnel Stage Component
interface FunnelStageProps {
  name: string
  count: number
  prevCount: number
  dropReason?: string
  dropCount?: number
  symbols?: string[]
  color: string
  isFirst?: boolean
  onClick?: () => void
}

function FunnelStage({
  name,
  count,
  prevCount,
  dropReason,
  dropCount,
  symbols,
  color,
  isFirst,
  onClick
}: FunnelStageProps) {
  const dropPct = isFirst ? 0 : prevCount > 0 ? Math.round((1 - count / prevCount) * 100) : 0
  const widthPct = isFirst ? 100 : prevCount > 0 ? Math.max(20, (count / prevCount) * 100) : 20

  return (
    <div className="funnel-stage mb-2" onClick={onClick}>
      {/* Drop-off indicator between stages */}
      {!isFirst && dropPct > 0 && (
        <div className="flex items-center justify-center py-2 text-xs text-text-secondary">
          <span className="text-status-red">-{dropPct}%</span>
          {dropReason && (
            <span className="ml-2 text-text-secondary">
              (Top reason: {dropReason} - {dropCount})
            </span>
          )}
        </div>
      )}

      {/* Stage bar */}
      <div
        className={`${color} rounded-lg p-4 cursor-pointer hover:opacity-90 transition-opacity`}
        style={{ width: `${widthPct}%`, marginLeft: `${(100 - widthPct) / 2}%` }}
      >
        <div className="flex items-center justify-between">
          <span className="font-medium text-white">{name}</span>
          <span className="text-2xl font-bold text-white">{count}</span>
        </div>
        {symbols && symbols.length > 0 && (
          <div className="mt-2 text-xs text-white/70 truncate">
            {symbols.slice(0, 5).join(', ')}
            {symbols.length > 5 && ` +${symbols.length - 5} more`}
          </div>
        )}
      </div>
    </div>
  )
}

// Symbol List Modal
interface SymbolListModalProps {
  stage: string
  symbols: string[]
  onClose: () => void
  onSymbolClick: (symbol: string) => void
}

function SymbolListModal({ stage, symbols, onClose, onSymbolClick }: SymbolListModalProps) {
  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50" onClick={onClose}>
      <div
        className="bg-bg-secondary rounded-lg p-6 max-w-md w-full max-h-[80vh] overflow-auto"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-bold">{stage} Symbols</h3>
          <button onClick={onClose} className="text-text-secondary hover:text-text-primary">
            ✕
          </button>
        </div>

        {symbols.length === 0 ? (
          <p className="text-text-secondary text-center py-4">No symbols at this stage</p>
        ) : (
          <div className="grid grid-cols-3 gap-2">
            {symbols.map((symbol) => (
              <button
                key={symbol}
                onClick={() => onSymbolClick(symbol)}
                className="bg-bg-tertiary px-3 py-2 rounded text-sm hover:bg-border-color transition-colors"
              >
                {symbol}
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

// Main Symbol Funnel Component
export default function SymbolFunnel() {
  const [selectedStage, setSelectedStage] = useState<string | null>(null)
  const [selectedSymbols, setSelectedSymbols] = useState<string[]>([])

  const { data: funnel, isLoading, error } = useQuery({
    queryKey: ['funnelStatus'],
    queryFn: () => api.getFunnelStatus(),
    refetchInterval: 5000
  })

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-status-yellow"></div>
      </div>
    )
  }

  if (error || !funnel) {
    return (
      <div className="bg-bg-secondary rounded-lg p-6 text-center">
        <p className="text-status-red">Failed to load funnel data</p>
        <p className="text-text-secondary text-sm mt-2">
          Make sure /api/ops/funnel/status is available
        </p>
      </div>
    )
  }

  const stages = funnel.stages
  const topVetoReason = Object.entries(funnel.veto_reasons).sort((a, b) => b[1] - a[1])[0]
  const topQualityReason = Object.entries(funnel.quality_reject_reasons).sort((a, b) => b[1] - a[1])[0]

  const handleStageClick = (stage: string, symbols: string[]) => {
    setSelectedStage(stage)
    setSelectedSymbols(symbols)
  }

  const handleSymbolClick = (symbol: string) => {
    // Navigate to timeline view for this symbol
    window.location.href = `/orchestrator/timeline?symbol=${symbol}`
  }

  return (
    <div className="max-w-4xl mx-auto">
      {/* Header */}
      <div className="mb-6">
        <h2 className="text-2xl font-bold mb-2">Symbol Flow Funnel</h2>
        <p className="text-text-secondary">
          Where are symbols dying in the pipeline? Click any stage to see symbols.
        </p>
        <div className="mt-2 text-sm text-text-secondary">
          Session duration: {Math.round(funnel.session_duration_minutes)} minutes |
          Overall conversion: <span className="text-status-yellow font-bold">
            {funnel.conversion_rates.overall_funnel_pct}%
          </span>
        </div>
      </div>

      {/* Funnel Visualization */}
      <div className="bg-bg-secondary rounded-lg p-6 mb-6">
        <FunnelStage
          name="DISCOVERED"
          count={stages['1_found_by_scanners']}
          prevCount={stages['1_found_by_scanners']}
          symbols={funnel.symbols.found}
          color="bg-blue-600"
          isFirst
          onClick={() => handleStageClick('Discovered', funnel.symbols.found)}
        />

        <FunnelStage
          name="INJECTED"
          count={stages['2_injected_symbols']}
          prevCount={stages['1_found_by_scanners']}
          symbols={funnel.symbols.injected}
          color="bg-blue-500"
          onClick={() => handleStageClick('Injected', funnel.symbols.injected)}
        />

        <FunnelStage
          name="QUALITY PASSED"
          count={stages['2_injected_symbols'] - stages['4_rejected_by_quality_gate']}
          prevCount={stages['2_injected_symbols']}
          dropReason={topQualityReason?.[0]}
          dropCount={topQualityReason?.[1]}
          color="bg-indigo-500"
          onClick={() => handleStageClick('Quality Passed', [])}
        />

        <FunnelStage
          name="GATING ATTEMPTED"
          count={stages['6_gating_attempts']}
          prevCount={stages['2_injected_symbols'] - stages['4_rejected_by_quality_gate']}
          color="bg-purple-500"
          onClick={() => handleStageClick('Gating Attempted', [])}
        />

        <FunnelStage
          name="APPROVED"
          count={stages['7_gating_approvals']}
          prevCount={stages['6_gating_attempts']}
          symbols={funnel.symbols.approved}
          dropReason={topVetoReason?.[0]}
          dropCount={topVetoReason?.[1]}
          color="bg-green-600"
          onClick={() => handleStageClick('Approved', funnel.symbols.approved)}
        />

        <FunnelStage
          name="EXECUTED"
          count={stages['9_trade_executions']}
          prevCount={stages['7_gating_approvals']}
          symbols={funnel.symbols.traded}
          color="bg-emerald-600"
          onClick={() => handleStageClick('Executed', funnel.symbols.traded)}
        />
      </div>

      {/* Diagnostic Panel */}
      <div className="grid grid-cols-2 gap-6">
        {/* Bottleneck Indicator */}
        <div className="bg-bg-secondary rounded-lg p-6">
          <h3 className="text-lg font-bold mb-3">Bottleneck Analysis</h3>
          <div className={`p-4 rounded-lg ${
            funnel.diagnostic.health.includes('GREEN') ? 'bg-green-900/30 border border-green-600' :
            funnel.diagnostic.health.includes('YELLOW') ? 'bg-yellow-900/30 border border-yellow-600' :
            funnel.diagnostic.health.includes('ORANGE') ? 'bg-orange-900/30 border border-orange-600' :
            'bg-red-900/30 border border-red-600'
          }`}>
            <div className="font-medium mb-2">{funnel.diagnostic.health}</div>
            <div className="text-sm text-text-secondary">{funnel.diagnostic.bottleneck}</div>
          </div>
        </div>

        {/* Top Veto Reasons */}
        <div className="bg-bg-secondary rounded-lg p-6">
          <h3 className="text-lg font-bold mb-3">Top Veto Reasons</h3>
          {Object.entries(funnel.veto_reasons).length === 0 ? (
            <p className="text-text-secondary">No vetoes recorded</p>
          ) : (
            <div className="space-y-2">
              {Object.entries(funnel.veto_reasons)
                .sort((a, b) => b[1] - a[1])
                .slice(0, 5)
                .map(([reason, count]) => (
                  <div key={reason} className="flex items-center justify-between">
                    <span className="text-sm truncate flex-1 mr-2">{reason}</span>
                    <span className="text-status-red font-bold">{count}</span>
                  </div>
                ))}
            </div>
          )}
        </div>
      </div>

      {/* Scout Metrics Panel */}
      {funnel.scout_metrics && (
        <div className="bg-bg-secondary rounded-lg p-6 mt-6">
          <h3 className="text-lg font-bold mb-3">Scout Metrics</h3>
          <div className="grid grid-cols-4 gap-4">
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-400">{funnel.scout_metrics.attempts}</div>
              <div className="text-sm text-text-secondary">Attempts</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-status-green">{funnel.scout_metrics.confirmed}</div>
              <div className="text-sm text-text-secondary">Confirmed</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-status-red">{funnel.scout_metrics.failed}</div>
              <div className="text-sm text-text-secondary">Failed</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-status-yellow">{funnel.scout_metrics.to_trade}</div>
              <div className="text-sm text-text-secondary">To Trade</div>
            </div>
          </div>
          <div className="mt-4 flex items-center justify-center gap-8 text-sm">
            <span>
              Confirmation Rate:{' '}
              <span className="font-bold text-status-green">
                {funnel.scout_metrics.confirmation_rate}%
              </span>
            </span>
            <span>
              Escalation Rate:{' '}
              <span className="font-bold text-status-yellow">
                {funnel.scout_metrics.escalation_rate}%
              </span>
            </span>
          </div>
        </div>
      )}

      {/* Symbol List Modal */}
      {selectedStage && (
        <SymbolListModal
          stage={selectedStage}
          symbols={selectedSymbols}
          onClose={() => setSelectedStage(null)}
          onSymbolClick={handleSymbolClick}
        />
      )}
    </div>
  )
}
