import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import api from '../services/api'
import type { DailyReport as DailyReportType } from '../types'

// Date Selector Component
function DateSelector({
  dates,
  selectedDate,
  onSelect
}: {
  dates: string[]
  selectedDate: string
  onSelect: (date: string) => void
}) {
  // Default to today
  const today = new Date().toISOString().split('T')[0]

  return (
    <div className="bg-bg-secondary rounded-lg p-4 mb-6">
      <div className="flex items-center gap-4">
        <label className="text-sm text-text-secondary">Report Date:</label>
        <select
          value={selectedDate || today}
          onChange={(e) => onSelect(e.target.value)}
          className="bg-bg-tertiary border border-border-color rounded px-4 py-2"
        >
          {dates.length === 0 ? (
            <option value={today}>{today} (Today)</option>
          ) : (
            dates.map((date) => (
              <option key={date} value={date}>
                {date} {date === today ? '(Today)' : ''}
              </option>
            ))
          )}
        </select>
        <span className="text-xs text-text-secondary ml-auto">
          Daily reports are generated at market close
        </span>
      </div>
    </div>
  )
}

// Market Context Card
function MarketContextCard({ context }: { context: DailyReportType['market_context'] }) {
  const getHealthColor = (value: number, threshold: number) => {
    if (value >= threshold) return 'text-status-green'
    if (value >= threshold * 0.7) return 'text-status-yellow'
    return 'text-status-red'
  }

  return (
    <div className="bg-bg-secondary rounded-lg p-6">
      <h3 className="text-lg font-bold mb-4">Market Context</h3>
      <div className="grid grid-cols-2 gap-4">
        <div>
          <div className="text-sm text-text-secondary">Market Breadth</div>
          <div className={`text-2xl font-bold ${getHealthColor(context.market_breadth, 50)}`}>
            {context.market_breadth}%
          </div>
        </div>
        <div>
          <div className="text-sm text-text-secondary">Small-Cap Participation</div>
          <div className={`text-2xl font-bold ${getHealthColor(context.small_cap_participation, 50)}`}>
            {context.small_cap_participation}%
          </div>
        </div>
        <div>
          <div className="text-sm text-text-secondary">Gap Continuation Rate</div>
          <div className={`text-2xl font-bold ${getHealthColor(context.gap_continuation_rate, 60)}`}>
            {context.gap_continuation_rate}%
          </div>
        </div>
        <div>
          <div className="text-sm text-text-secondary">Dominant Regime</div>
          <div className={`text-2xl font-bold ${
            context.dominant_regime === 'TRENDING_UP' ? 'text-status-green' :
            context.dominant_regime === 'TRENDING_DOWN' ? 'text-status-red' :
            'text-status-yellow'
          }`}>
            {context.dominant_regime.replace(/_/g, ' ')}
          </div>
        </div>
      </div>
    </div>
  )
}

// Funnel Summary Card
function FunnelSummaryCard({ funnel }: { funnel: DailyReportType['funnel_summary'] }) {
  return (
    <div className="bg-bg-secondary rounded-lg p-6">
      <h3 className="text-lg font-bold mb-4">Funnel Summary</h3>

      {/* Visual Funnel */}
      <div className="space-y-2 mb-4">
        {[
          { label: 'Discovered', value: funnel.total_discovered, color: 'bg-blue-600' },
          { label: 'Injected', value: funnel.total_injected, color: 'bg-blue-500' },
          { label: 'Gated', value: funnel.total_gated, color: 'bg-purple-500' },
          { label: 'Approved', value: funnel.total_approved, color: 'bg-green-500' },
          { label: 'Executed', value: funnel.total_executed, color: 'bg-emerald-600' }
        ].map((stage) => {
          const widthPct = funnel.total_discovered > 0
            ? Math.max(20, (stage.value / funnel.total_discovered) * 100)
            : 20

          return (
            <div key={stage.label} className="flex items-center gap-3">
              <span className="text-xs text-text-secondary w-20">{stage.label}</span>
              <div
                className={`${stage.color} h-6 rounded flex items-center px-2`}
                style={{ width: `${widthPct}%` }}
              >
                <span className="text-xs font-bold text-white">{stage.value}</span>
              </div>
            </div>
          )
        })}
      </div>

      <div className="pt-4 border-t border-border-color text-center">
        <span className="text-text-secondary">Overall Conversion: </span>
        <span className="text-2xl font-bold text-status-yellow">
          {funnel.overall_conversion}%
        </span>
      </div>
    </div>
  )
}

// Strategy Activity Table
function StrategyActivityTable({ activities }: { activities: DailyReportType['strategy_activity'] }) {
  if (activities.length === 0) {
    return (
      <div className="bg-bg-secondary rounded-lg p-6 text-center">
        <p className="text-text-secondary">No strategy activity recorded</p>
      </div>
    )
  }

  return (
    <div className="bg-bg-secondary rounded-lg p-6">
      <h3 className="text-lg font-bold mb-4">Strategy Activity</h3>
      <table className="w-full">
        <thead>
          <tr className="text-left text-text-secondary text-sm border-b border-border-color">
            <th className="pb-2">Strategy</th>
            <th className="pb-2 text-right">Signals</th>
            <th className="pb-2 text-right">Approved</th>
            <th className="pb-2 text-right">Vetoed</th>
            <th className="pb-2 text-right">Traded</th>
            <th className="pb-2 text-right">Rate</th>
          </tr>
        </thead>
        <tbody>
          {activities.map((a) => {
            const rate = a.signals_generated > 0
              ? Math.round((a.trades_executed / a.signals_generated) * 100)
              : 0

            return (
              <tr key={a.strategy} className="border-b border-border-color/50">
                <td className="py-2 font-medium">{a.strategy}</td>
                <td className="py-2 text-right">{a.signals_generated}</td>
                <td className="py-2 text-right text-status-green">{a.signals_approved}</td>
                <td className="py-2 text-right text-status-red">{a.signals_vetoed}</td>
                <td className="py-2 text-right text-status-yellow">{a.trades_executed}</td>
                <td className="py-2 text-right">
                  <span className={rate >= 50 ? 'text-status-green' : rate >= 25 ? 'text-status-yellow' : 'text-status-red'}>
                    {rate}%
                  </span>
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

// Veto Reasons Chart
function VetoReasonsChart({ reasons }: { reasons: DailyReportType['top_veto_reasons'] }) {
  if (reasons.length === 0) {
    return (
      <div className="bg-bg-secondary rounded-lg p-6 text-center">
        <p className="text-text-secondary">No vetoes recorded</p>
      </div>
    )
  }

  const maxCount = Math.max(...reasons.map(r => r.count))

  return (
    <div className="bg-bg-secondary rounded-lg p-6">
      <h3 className="text-lg font-bold mb-4">Top Veto Reasons</h3>
      <div className="space-y-3">
        {reasons.map((r) => (
          <div key={r.reason}>
            <div className="flex items-center justify-between text-sm mb-1">
              <span className="truncate flex-1 mr-2">{r.reason}</span>
              <span className="text-text-secondary">
                {r.count} ({r.percentage}%)
              </span>
            </div>
            <div className="bg-bg-tertiary rounded-full h-2">
              <div
                className="bg-status-red rounded-full h-2 transition-all"
                style={{ width: `${(r.count / maxCount) * 100}%` }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

// Phase Transitions Timeline
function PhaseTransitionsCard({ transitions }: { transitions: DailyReportType['phase_transitions'] }) {
  if (transitions.length === 0) {
    return (
      <div className="bg-bg-secondary rounded-lg p-6 text-center">
        <p className="text-text-secondary">No phase transitions recorded</p>
      </div>
    )
  }

  return (
    <div className="bg-bg-secondary rounded-lg p-6">
      <h3 className="text-lg font-bold mb-4">Phase Transitions</h3>
      <div className="space-y-2">
        {transitions.map((t, i) => (
          <div key={i} className="flex items-center gap-3 text-sm">
            <span className="text-text-secondary w-16">
              {new Date(t.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
            </span>
            <span className="text-status-gray">{t.from_phase}</span>
            <span className="text-text-secondary">→</span>
            <span className="text-status-yellow">{t.to_phase}</span>
            <span className="text-xs text-text-secondary ml-auto">{t.reason}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

// Trade Summary Card
function TradeSummaryCard({ summary }: { summary: DailyReportType['trade_summary'] }) {
  const pnlColor = summary.total_pnl >= 0 ? 'text-status-green' : 'text-status-red'

  return (
    <div className="bg-bg-secondary rounded-lg p-6">
      <h3 className="text-lg font-bold mb-4">Trade Summary</h3>
      <div className="grid grid-cols-4 gap-4 mb-4">
        <div className="text-center">
          <div className="text-2xl font-bold">{summary.total_trades}</div>
          <div className="text-xs text-text-secondary">Total Trades</div>
        </div>
        <div className="text-center">
          <div className="text-2xl font-bold text-status-green">{summary.wins}</div>
          <div className="text-xs text-text-secondary">Wins</div>
        </div>
        <div className="text-center">
          <div className="text-2xl font-bold text-status-red">{summary.losses}</div>
          <div className="text-xs text-text-secondary">Losses</div>
        </div>
        <div className="text-center">
          <div className="text-2xl font-bold text-status-yellow">{summary.win_rate}%</div>
          <div className="text-xs text-text-secondary">Win Rate</div>
        </div>
      </div>
      <div className="pt-4 border-t border-border-color">
        <div className="flex items-center justify-between">
          <div>
            <div className="text-sm text-text-secondary">Total P&L</div>
            <div className={`text-3xl font-bold ${pnlColor}`}>
              ${summary.total_pnl.toFixed(2)}
            </div>
          </div>
          <div className="text-right text-sm">
            <div>Avg Win: <span className="text-status-green">${summary.avg_win.toFixed(2)}</span></div>
            <div>Avg Loss: <span className="text-status-red">${summary.avg_loss.toFixed(2)}</span></div>
          </div>
        </div>
      </div>
    </div>
  )
}

// No Report State
function NoReportState({ date }: { date: string }) {
  return (
    <div className="bg-bg-secondary rounded-lg p-12 text-center">
      <div className="text-4xl mb-4">📋</div>
      <h3 className="text-xl font-bold mb-2">No Report Available</h3>
      <p className="text-text-secondary">
        No baseline report found for {date}.
        <br />
        Reports are generated at market close (4:00 PM ET).
      </p>
    </div>
  )
}

// Main Daily Report Component
export default function DailyReport() {
  const [selectedDate, setSelectedDate] = useState('')

  // Get available dates
  const { data: dates } = useQuery({
    queryKey: ['reportDates'],
    queryFn: () => api.getAvailableReportDates()
  })

  // Get report for selected date
  const today = new Date().toISOString().split('T')[0]
  const dateToFetch = selectedDate || today

  const { data: report, isLoading, error } = useQuery({
    queryKey: ['dailyReport', dateToFetch],
    queryFn: () => api.getDailyReport(dateToFetch),
    retry: 1
  })

  return (
    <div className="max-w-6xl mx-auto">
      {/* Header */}
      <div className="mb-6">
        <h2 className="text-2xl font-bold mb-2">Daily Baseline Report</h2>
        <p className="text-text-secondary">
          How did the system behave today? Visual breakdown of daily metrics.
        </p>
      </div>

      {/* Date Selector */}
      <DateSelector
        dates={dates || [today]}
        selectedDate={selectedDate || today}
        onSelect={setSelectedDate}
      />

      {/* Content */}
      {isLoading ? (
        <div className="flex items-center justify-center h-64">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-status-yellow"></div>
        </div>
      ) : error || !report ? (
        <NoReportState date={dateToFetch} />
      ) : (
        <div className="space-y-6">
          {/* Top Row: Context + Funnel */}
          <div className="grid grid-cols-2 gap-6">
            <MarketContextCard context={report.market_context} />
            <FunnelSummaryCard funnel={report.funnel_summary} />
          </div>

          {/* Trade Summary (full width) */}
          <TradeSummaryCard summary={report.trade_summary} />

          {/* Middle Row: Strategy + Vetoes */}
          <div className="grid grid-cols-2 gap-6">
            <StrategyActivityTable activities={report.strategy_activity} />
            <VetoReasonsChart reasons={report.top_veto_reasons} />
          </div>

          {/* Phase Transitions */}
          <PhaseTransitionsCard transitions={report.phase_transitions} />
        </div>
      )}
    </div>
  )
}
