import { Routes, Route, NavLink } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import api from './services/api'

// Screen Components
import SymbolFunnel from './components/SymbolFunnel'
import PhaseStrategy from './components/PhaseStrategy'
import GatingExplainer from './components/GatingExplainer'
import SymbolTimeline from './components/SymbolTimeline'
import DailyReport from './components/DailyReport'

// System Overview Bar Component
function SystemOverviewBar() {
  const { data: status, isLoading } = useQuery({
    queryKey: ['systemStatus'],
    queryFn: () => api.getSystemStatus(),
    refetchInterval: 3000
  })

  if (isLoading || !status) {
    return (
      <div className="bg-bg-secondary border-b border-border-color px-6 py-3">
        <div className="animate-pulse flex items-center gap-8">
          <div className="h-4 w-24 bg-bg-tertiary rounded"></div>
          <div className="h-4 w-32 bg-bg-tertiary rounded"></div>
          <div className="h-4 w-28 bg-bg-tertiary rounded"></div>
        </div>
      </div>
    )
  }

  const statusColor = status.bot_status === 'RUNNING' ? 'bg-status-green' : 'bg-status-red'
  const modeColor = status.mode === 'PAPER' ? 'text-status-yellow' : 'text-status-green'

  return (
    <div className="bg-bg-secondary border-b border-border-color px-6 py-3">
      <div className="flex items-center justify-between">
        {/* Left: Core Status */}
        <div className="flex items-center gap-8">
          {/* Bot Status */}
          <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${statusColor} status-pulse`}></div>
            <span className="text-sm font-medium">{status.bot_status}</span>
          </div>

          {/* Mode */}
          <div className="flex items-center gap-2">
            <span className="text-text-secondary text-sm">Mode:</span>
            <span className={`text-sm font-bold ${modeColor}`}>{status.mode}</span>
          </div>

          {/* Current Phase */}
          <div className="flex items-center gap-2">
            <span className="text-text-secondary text-sm">Phase:</span>
            <span className="text-sm font-medium text-status-yellow">
              {status.current_phase.replace(/_/g, ' ')}
            </span>
          </div>

          {/* Baseline Profile */}
          <div className="flex items-center gap-2">
            <span className="text-text-secondary text-sm">Profile:</span>
            <span className="text-sm font-medium">{status.baseline_profile}</span>
          </div>

          {/* Scout Mode */}
          <div className="flex items-center gap-2">
            <span className="text-text-secondary text-sm">Scouts:</span>
            <span className={`text-sm font-medium ${status.scout_enabled ? 'text-status-green' : 'text-status-red'}`}>
              {status.scout_enabled ? 'ON' : 'OFF'}
            </span>
          </div>
        </div>

        {/* Right: Strategy Summary */}
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <span className="text-status-green text-sm">
              {status.strategies_enabled.length} enabled
            </span>
            <span className="text-text-secondary text-sm">/</span>
            <span className="text-status-red text-sm">
              {status.strategies_disabled.length} disabled
            </span>
          </div>
          <div className="text-xs text-text-secondary">
            Updated: {new Date(status.timestamp).toLocaleTimeString()}
          </div>
        </div>
      </div>
    </div>
  )
}

// Navigation Component
function Navigation() {
  const navItems = [
    { path: '/', label: 'Symbol Funnel', icon: '📊' },
    { path: '/phase', label: 'Phase & Strategy', icon: '⏱️' },
    { path: '/gating', label: 'Gating Decisions', icon: '🚦' },
    { path: '/timeline', label: 'Symbol Timeline', icon: '📈' },
    { path: '/reports', label: 'Daily Reports', icon: '📋' }
  ]

  return (
    <nav className="bg-bg-secondary border-b border-border-color px-6 py-2">
      <div className="flex items-center gap-1">
        {navItems.map((item) => (
          <NavLink
            key={item.path}
            to={item.path}
            className={({ isActive }) =>
              `px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                isActive
                  ? 'bg-bg-tertiary text-text-primary'
                  : 'text-text-secondary hover:text-text-primary hover:bg-bg-tertiary/50'
              }`
            }
          >
            <span className="mr-2">{item.icon}</span>
            {item.label}
          </NavLink>
        ))}
      </div>
    </nav>
  )
}

// Main App Component
function App() {
  return (
    <div className="min-h-screen bg-bg-primary text-text-primary flex flex-col">
      {/* Header */}
      <header className="bg-bg-secondary border-b border-border-color px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-xl font-bold">MORPHEUS</h1>
            <p className="text-sm text-text-secondary">Orchestration & Observer Console</p>
          </div>
          <div className="text-right">
            <div className="text-xs text-text-secondary uppercase tracking-wider">Read-Only Mode</div>
            <div className="text-xs text-status-yellow">No execution actions available</div>
          </div>
        </div>
      </header>

      {/* System Overview Bar */}
      <SystemOverviewBar />

      {/* Navigation */}
      <Navigation />

      {/* Main Content */}
      <main className="flex-1 overflow-auto p-6">
        <Routes>
          <Route path="/" element={<SymbolFunnel />} />
          <Route path="/phase" element={<PhaseStrategy />} />
          <Route path="/gating" element={<GatingExplainer />} />
          <Route path="/timeline" element={<SymbolTimeline />} />
          <Route path="/reports" element={<DailyReport />} />
        </Routes>
      </main>

      {/* Footer */}
      <footer className="bg-bg-secondary border-t border-border-color px-6 py-2 text-center">
        <span className="text-xs text-text-secondary">
          MORPHEUS Orchestrator v1.0 | Observer Mode | All data refreshes automatically
        </span>
      </footer>
    </div>
  )
}

export default App
