import { useState } from 'react'

interface MenuProps {
  activeSymbol: string
  onResetLayout: () => void
  onOpenAIScreen: () => void
  onAddPanel?: (panelType: string) => void
}

interface MenuItemProps {
  label: string
  items: { label: string; onClick: () => void; shortcut?: string }[]
}

function MenuItem({ label, items }: MenuItemProps) {
  const [isOpen, setIsOpen] = useState(false)

  return (
    <div
      className="relative"
      onMouseEnter={() => setIsOpen(true)}
      onMouseLeave={() => setIsOpen(false)}
    >
      <button className="px-3 py-1 hover:bg-sterling-header text-xs">
        {label}
      </button>
      {isOpen && (
        <div className="absolute left-0 top-full z-50 min-w-[180px] bg-sterling-panel border border-sterling-border shadow-lg">
          {items.map((item, i) => (
            <button
              key={i}
              className="w-full px-3 py-1.5 text-left text-xs hover:bg-sterling-header flex justify-between items-center"
              onClick={() => {
                item.onClick()
                setIsOpen(false)
              }}
            >
              <span>{item.label}</span>
              {item.shortcut && (
                <span className="text-sterling-muted ml-4">{item.shortcut}</span>
              )}
            </button>
          ))}
        </div>
      )}
    </div>
  )
}

export default function Menu({ activeSymbol, onResetLayout, onOpenAIScreen, onAddPanel }: MenuProps) {
  const [symbolInput, setSymbolInput] = useState('')

  const handleSymbolSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (symbolInput.trim()) {
      // Import and use the symbol store
      import('../stores/symbolStore').then(({ useSymbolStore }) => {
        useSymbolStore.getState().setActiveSymbol(symbolInput.trim())
      })
      setSymbolInput('')
    }
  }

  // Available panels that can be added
  const panelOptions = [
    { label: 'Quote', type: 'Quote' },
    { label: 'Positions', type: 'Positions' },
    { label: 'Orders', type: 'Orders' },
    { label: 'Account', type: 'Account' },
    { label: 'Order Entry', type: 'OrderEntry' },
    { label: 'Level 2', type: 'Level2' },
    { label: 'Time & Sales', type: 'TimeSales' },
    { label: 'Worklist', type: 'Worklist' },
    { label: 'Breaking News', type: 'BreakingNews' },
    { label: 'Scanners', type: 'ScannerPanel' },
    { label: 'Chart', type: 'Chart' },
  ]

  return (
    <div className="h-10 bg-sterling-header border-b border-sterling-border flex items-center justify-between px-2">
      {/* Left side - Menu items */}
      <div className="flex items-center">
        <MenuItem
          label="File"
          items={[
            { label: 'Save Layout', onClick: () => {}, shortcut: 'Ctrl+S' },
            { label: 'Load Layout', onClick: () => {}, shortcut: 'Ctrl+O' },
            { label: 'Reset Layout', onClick: onResetLayout },
          ]}
        />
        <MenuItem
          label="Panels"
          items={panelOptions.map(p => ({
            label: `Add ${p.label}`,
            onClick: () => onAddPanel?.(p.type)
          }))}
        />
        <MenuItem
          label="Layouts"
          items={[
            { label: 'Default Trading', onClick: onResetLayout },
            { label: 'Scalping', onClick: () => {} },
            { label: 'Analysis', onClick: () => {} },
            { label: 'Save Current...', onClick: () => {} },
          ]}
        />
        <MenuItem
          label="Scanners"
          items={[
            { label: 'HOD Scanner', onClick: () => {} },
            { label: 'Gappers', onClick: () => {} },
            { label: 'Top Gainers', onClick: () => {} },
            { label: 'Pre-Market', onClick: () => {} },
          ]}
        />
        <MenuItem
          label="AI"
          items={[
            { label: 'Open AI Screen', onClick: onOpenAIScreen, shortcut: 'Ctrl+I' },
            { label: 'AI Signals', onClick: () => {} },
            { label: 'Model Status', onClick: () => {} },
            { label: 'Start Scalper', onClick: () => {} },
          ]}
        />
        <MenuItem
          label="Tools"
          items={[
            { label: 'Settings', onClick: () => {}, shortcut: 'Ctrl+,' },
            { label: 'Keyboard Shortcuts', onClick: () => {} },
            { label: 'Connection Status', onClick: () => {} },
          ]}
        />
        <MenuItem
          label="Schwab"
          items={[
            { label: 'Account Status', onClick: () => {} },
            { label: 'Select Account', onClick: () => {} },
            { label: 'Refresh Connection', onClick: () => {} },
          ]}
        />
      </div>

      {/* Center - Symbol input */}
      <div className="flex items-center gap-4">
        <form onSubmit={handleSymbolSubmit} className="flex items-center gap-2">
          <input
            type="text"
            value={symbolInput}
            onChange={(e) => setSymbolInput(e.target.value.toUpperCase())}
            placeholder="Symbol"
            className="w-20 px-2 py-1 bg-sterling-bg border border-sterling-border text-xs uppercase"
          />
          <button
            type="submit"
            className="px-2 py-1 bg-accent-primary text-white text-xs rounded hover:brightness-110"
          >
            Go
          </button>
        </form>
        <div className="text-xs">
          <span className="text-sterling-muted">Active: </span>
          <span className="text-accent-primary font-bold">{activeSymbol}</span>
        </div>
      </div>

      {/* Right side - Status & AI Screen button */}
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-2 text-xxs">
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-up"></span>
            <span className="text-sterling-muted">Schwab</span>
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-up"></span>
            <span className="text-sterling-muted">Market</span>
          </span>
        </div>
        <button
          onClick={onOpenAIScreen}
          className="px-3 py-1 bg-accent-primary text-white text-xs rounded hover:brightness-110"
        >
          AI UI →
        </button>
      </div>
    </div>
  )
}
