import { useState } from 'react'
import { SavedLayout } from '../stores/layoutStore'

interface MenuProps {
  activeSymbol: string
  savedLayouts: SavedLayout[]
  activeLayoutId: string | null
  onLoadLayout: (layoutId: string) => void
  onResetLayout: () => void
  onSaveLayout: (name: string) => string | null
  onDeleteLayout: (layoutId: string) => void
  onOpenAIScreen: () => void
  onAddPanel?: (panelType: string) => void
}

interface MenuItemDef {
  label: string
  onClick: () => void
  shortcut?: string
  active?: boolean
  danger?: boolean
}

interface MenuItemProps {
  label: string
  items: MenuItemDef[]
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
        <div className="absolute left-0 top-full z-50 min-w-[200px] bg-sterling-panel border border-sterling-border shadow-lg">
          {items.map((item, i) => (
            <button
              key={i}
              className={`w-full px-3 py-1.5 text-left text-xs hover:bg-sterling-header flex justify-between items-center ${
                item.active ? 'bg-sterling-highlight text-accent-primary' : ''
              } ${item.danger ? 'text-down hover:bg-red-900/30' : ''}`}
              onClick={() => {
                item.onClick()
                setIsOpen(false)
              }}
            >
              <span className="flex items-center gap-2">
                {item.active && <span className="text-accent-primary">✓</span>}
                {item.label}
              </span>
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

// Save Layout Dialog
function SaveLayoutDialog({ onSave, onCancel }: { onSave: (name: string) => void; onCancel: () => void }) {
  const [name, setName] = useState('')

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (name.trim()) {
      onSave(name.trim())
    }
  }

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-[100]">
      <div className="bg-sterling-panel border border-sterling-border rounded shadow-xl p-4 min-w-[300px]">
        <h3 className="text-sm font-bold text-sterling-text mb-3">Save Layout</h3>
        <form onSubmit={handleSubmit}>
          <input
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="Layout name..."
            className="w-full px-3 py-2 bg-sterling-bg border border-sterling-border text-xs mb-3"
            autoFocus
          />
          <div className="flex justify-end gap-2">
            <button
              type="button"
              onClick={onCancel}
              className="px-3 py-1.5 bg-sterling-bg text-sterling-text text-xs rounded hover:bg-sterling-highlight"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={!name.trim()}
              className="px-3 py-1.5 bg-accent-primary text-white text-xs rounded hover:brightness-110 disabled:opacity-50"
            >
              Save
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}

// Confirm Delete Dialog
function ConfirmDeleteDialog({ layoutName, onConfirm, onCancel }: { layoutName: string; onConfirm: () => void; onCancel: () => void }) {
  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-[100]">
      <div className="bg-sterling-panel border border-sterling-border rounded shadow-xl p-4 min-w-[300px]">
        <h3 className="text-sm font-bold text-sterling-text mb-3">Delete Layout</h3>
        <p className="text-xs text-sterling-muted mb-4">
          Are you sure you want to delete "{layoutName}"? This cannot be undone.
        </p>
        <div className="flex justify-end gap-2">
          <button
            type="button"
            onClick={onCancel}
            className="px-3 py-1.5 bg-sterling-bg text-sterling-text text-xs rounded hover:bg-sterling-highlight"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={onConfirm}
            className="px-3 py-1.5 bg-down text-white text-xs rounded hover:brightness-110"
          >
            Delete
          </button>
        </div>
      </div>
    </div>
  )
}

export default function Menu({
  activeSymbol,
  savedLayouts,
  activeLayoutId,
  onLoadLayout,
  onResetLayout,
  onSaveLayout,
  onDeleteLayout,
  onOpenAIScreen,
  onAddPanel
}: MenuProps) {
  const [symbolInput, setSymbolInput] = useState('')
  const [showSaveDialog, setShowSaveDialog] = useState(false)
  const [showDeleteDialog, setShowDeleteDialog] = useState<string | null>(null)

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

  const handleSaveLayout = (name: string) => {
    onSaveLayout(name)
    setShowSaveDialog(false)
  }

  const handleDeleteLayout = () => {
    if (showDeleteDialog) {
      onDeleteLayout(showDeleteDialog)
      setShowDeleteDialog(null)
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

  // Build layout menu - layouts at top, actions at bottom
  const activeLayout = savedLayouts.find(l => l.id === activeLayoutId)

  const layoutMenuItems: MenuItemDef[] = [
    // Layout list - clicking selects/loads
    ...savedLayouts.map(layout => ({
      label: layout.name,
      onClick: () => onLoadLayout(layout.id),
      active: activeLayoutId === layout.id,
    })),
    // Separator
    { label: '──────────────', onClick: () => {} },
    // Actions that operate on the active layout
    { label: 'Load', onClick: () => activeLayoutId && onLoadLayout(activeLayoutId) },
    { label: 'Save', onClick: () => setShowSaveDialog(true) },
    { label: 'Delete', onClick: () => activeLayoutId && setShowDeleteDialog(activeLayoutId), danger: true },
  ]

  return (
    <>
      <div className="h-10 bg-sterling-header border-b border-sterling-border flex items-center justify-between px-2">
        {/* Left side - Menu items */}
        <div className="flex items-center">
          <MenuItem
            label="File"
            items={[
              { label: 'Save Layout...', onClick: () => setShowSaveDialog(true), shortcut: 'Ctrl+S' },
              { label: 'Reset to Default', onClick: onResetLayout },
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
            items={layoutMenuItems}
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

        {/* Right side - Layout indicator, Status & AI Screen button */}
        <div className="flex items-center gap-4">
          {activeLayout && (
            <div className="text-xxs px-2 py-0.5 bg-sterling-bg rounded border border-sterling-border">
              <span className="text-sterling-muted">Layout: </span>
              <span className="text-sterling-text">{activeLayout.name}</span>
            </div>
          )}
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
            AI UI
          </button>
        </div>
      </div>

      {/* Dialogs */}
      {showSaveDialog && (
        <SaveLayoutDialog
          onSave={handleSaveLayout}
          onCancel={() => setShowSaveDialog(false)}
        />
      )}
      {showDeleteDialog && (
        <ConfirmDeleteDialog
          layoutName={savedLayouts.find(l => l.id === showDeleteDialog)?.name || ''}
          onConfirm={handleDeleteLayout}
          onCancel={() => setShowDeleteDialog(null)}
        />
      )}
    </>
  )
}
