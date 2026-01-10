import { useEffect, useRef, useCallback } from 'react'
import { GoldenLayout, LayoutConfig, ComponentContainer, ResolvedLayoutConfig } from 'golden-layout'
import Menu from './components/Menu'
import { useSymbolStore } from './stores/symbolStore'
import { useLayoutStore } from './stores/layoutStore'

// Import panel components
import Positions from './components/Positions'
import Orders from './components/Orders'
import Account from './components/Account'
import OrderEntry from './components/OrderEntry'
import Level2 from './components/Level2'
import TimeSales from './components/TimeSales'
import Worklist from './components/Worklist'
import BreakingNews from './components/BreakingNews'
import ScannerPanel from './components/ScannerPanel'
import Chart from './components/Chart'
import Quote from './components/Quote'

// Component registry for Golden Layout
const componentRegistry: Record<string, React.ComponentType<any>> = {
  Positions,
  Orders,
  Account,
  OrderEntry,
  Level2,
  TimeSales,
  Worklist,
  BreakingNews,
  ScannerPanel,
  Chart,
  Quote,
}

// Convert ResolvedLayoutConfig back to LayoutConfig using Golden Layout's built-in method
function convertToLayoutConfig(config: any): LayoutConfig {
  if (!config) return config

  // If it's a ResolvedLayoutConfig (has 'resolved' property or numeric sizes), convert it
  try {
    // Use Golden Layout's built-in conversion
    if (config.resolved === true || (config.root && typeof config.root.size === 'number')) {
      return LayoutConfig.fromResolved(config as ResolvedLayoutConfig)
    }
  } catch (e) {
    console.warn('Failed to convert resolved config, using manual sanitization:', e)
  }

  // Fallback: manual sanitization for corrupted configs
  const sanitizeNode = (node: any): any => {
    if (!node || typeof node !== 'object') return node

    const result: any = {
      type: node.type,
      content: Array.isArray(node.content) ? node.content.map(sanitizeNode) : undefined
    }

    // Preserve essential properties
    if (node.componentType) result.componentType = node.componentType
    if (node.title) result.title = node.title
    if (node.width !== undefined && typeof node.width === 'number') result.width = node.width
    if (node.height !== undefined && typeof node.height === 'number') result.height = node.height

    // Remove undefined content
    if (!result.content) delete result.content

    return result
  }

  return {
    settings: { reorderEnabled: true },
    root: config.root ? sanitizeNode(config.root) : undefined
  } as LayoutConfig
}

function App() {
  const containerRef = useRef<HTMLDivElement>(null)
  const layoutRef = useRef<GoldenLayout | null>(null)
  const isInitializedRef = useRef(false)
  const activeSymbol = useSymbolStore((state) => state.activeSymbol)
  const {
    savedLayouts,
    activeLayoutId,
    loadSavedLayouts,
    saveLayout,
    deleteLayout,
    setActiveLayout,
    getLayout
  } = useLayoutStore()

  // Initialize layout store on mount
  useEffect(() => {
    loadSavedLayouts()
  }, [loadSavedLayouts])

  // Initialize Golden Layout (only once when savedLayouts first loads)
  // Use savedLayouts.length as trigger - only initialize when layouts first become available
  const layoutsLoaded = savedLayouts.length > 0

  useEffect(() => {
    if (!containerRef.current || !layoutsLoaded || isInitializedRef.current) return

    // Get active layout config using current store values
    const currentActiveLayoutId = useLayoutStore.getState().activeLayoutId
    const currentLayouts = useLayoutStore.getState().savedLayouts
    const activeLayout = currentLayouts.find(l => l.id === (currentActiveLayoutId || 'preset_trading'))
    const layoutConfig = activeLayout?.config || currentLayouts[0]?.config

    if (!layoutConfig) return

    isInitializedRef.current = true

    // Create Golden Layout instance
    const layout = new GoldenLayout(containerRef.current)
    layoutRef.current = layout

    // Register all components
    Object.entries(componentRegistry).forEach(([name, Component]) => {
      layout.registerComponentFactoryFunction(name, (container: ComponentContainer) => {
        // Create a wrapper div for React to render into
        const element = document.createElement('div')
        element.style.height = '100%'
        element.style.overflow = 'auto'
        container.element.appendChild(element)

        // Render React component
        import('react-dom/client').then(({ createRoot }) => {
          const root = createRoot(element)
          root.render(<Component />)

          // Cleanup on destroy
          container.on('beforeComponentRelease', () => {
            root.unmount()
          })
        })
      })
    })

    // Load layout with error handling (cast to LayoutConfig to handle ResolvedLayoutConfig)
    try {
      const sanitizedConfig = convertToLayoutConfig(layoutConfig)
      layout.loadLayout(sanitizedConfig as LayoutConfig)
    } catch (e) {
      console.error('Failed to load layout, trying default:', e)
      // Clear corrupted localStorage
      localStorage.removeItem('morpheus_saved_layouts')
      localStorage.removeItem('morpheus_active_layout_id')
      localStorage.removeItem('morpheus_current_layout')

      const defaultLayout = currentLayouts.find(l => l.id === 'preset_trading')
      if (defaultLayout) {
        try {
          const sanitizedDefault = convertToLayoutConfig(defaultLayout.config)
          layout.loadLayout(sanitizedDefault as LayoutConfig)
          setActiveLayout('preset_trading')
        } catch (e2) {
          console.error('Failed to load default layout:', e2)
        }
      }
    }

    // Auto-save current layout changes
    let saveTimeout: ReturnType<typeof setTimeout> | null = null
    layout.on('stateChanged', () => {
      // Debounce saves
      if (saveTimeout) clearTimeout(saveTimeout)
      saveTimeout = setTimeout(() => {
        const config = layout.saveLayout()
        // Save to localStorage as current working state
        localStorage.setItem('morpheus_current_layout', JSON.stringify(config))
      }, 500)
    })

    // Handle window resize
    const handleResize = () => {
      layout.setSize(window.innerWidth, window.innerHeight - 40) // Account for menu
    }
    window.addEventListener('resize', handleResize)
    handleResize()

    // Cleanup only on unmount, not on state changes
    return () => {
      if (saveTimeout) clearTimeout(saveTimeout)
      window.removeEventListener('resize', handleResize)
      layout.destroy()
      layoutRef.current = null
      isInitializedRef.current = false
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [layoutsLoaded]) // Only depend on whether layouts are loaded, not on the layouts themselves

  // Load a specific layout by ID - clears and reloads
  const handleLoadLayout = useCallback((layoutId: string) => {
    const layoutData = getLayout(layoutId)
    if (!layoutData || !layoutRef.current) return

    try {
      // Clear existing layout first
      layoutRef.current.clear()

      // Load the new layout with sanitization
      const sanitizedConfig = convertToLayoutConfig(layoutData.config)
      layoutRef.current.loadLayout(sanitizedConfig as LayoutConfig)
      setActiveLayout(layoutId)

      // Trigger resize to ensure proper sizing
      layoutRef.current.setSize(window.innerWidth, window.innerHeight - 40)
    } catch (e) {
      console.error('Failed to load layout:', e)
      // Clear corrupted localStorage
      localStorage.removeItem('morpheus_saved_layouts')
      localStorage.removeItem('morpheus_active_layout_id')
      localStorage.removeItem('morpheus_current_layout')

      // Try to recover by loading default
      const defaultLayout = getLayout('preset_trading')
      if (defaultLayout) {
        try {
          layoutRef.current.clear()
          const sanitizedDefault = convertToLayoutConfig(defaultLayout.config)
          layoutRef.current.loadLayout(sanitizedDefault as LayoutConfig)
          setActiveLayout('preset_trading')
        } catch (e2) {
          console.error('Failed to recover with default layout:', e2)
        }
      }
    }
  }, [getLayout, setActiveLayout])

  // Reset to default trading layout
  const handleResetLayout = useCallback(() => {
    handleLoadLayout('preset_trading')
  }, [handleLoadLayout])

  // Save current layout with a name
  const handleSaveCurrentLayout = useCallback((name: string) => {
    if (!layoutRef.current) return null
    const config = layoutRef.current.saveLayout()
    const id = saveLayout(name, config)
    return id
  }, [saveLayout])

  // Delete a layout
  const handleDeleteLayout = useCallback((layoutId: string) => {
    deleteLayout(layoutId)
  }, [deleteLayout])

  const handleOpenAIScreen = () => {
    window.open('/governor', 'ai_screen', 'width=1920,height=1080')
  }

  const handleAddPanel = useCallback((panelType: string) => {
    if (!layoutRef.current) return

    const panelTitles: Record<string, string> = {
      Positions: 'Positions',
      Orders: 'Orders',
      Account: 'Account',
      OrderEntry: 'Order Entry',
      Level2: 'Level 2',
      TimeSales: 'Time & Sales',
      Worklist: 'Worklist',
      BreakingNews: 'Breaking News',
      ScannerPanel: 'Scanners',
      Chart: 'Chart',
      Quote: 'Quote',
    }

    // Add new component to root
    layoutRef.current.addComponent(panelType, undefined, panelTitles[panelType] || panelType)
  }, [])

  // Show loading state if layouts haven't loaded yet
  if (savedLayouts.length === 0) {
    return (
      <div className="h-screen w-screen flex items-center justify-center bg-sterling-bg text-sterling-text">
        Loading layouts...
      </div>
    )
  }

  return (
    <div className="h-screen w-screen overflow-hidden bg-sterling-bg">
      <Menu
        activeSymbol={activeSymbol}
        savedLayouts={savedLayouts}
        activeLayoutId={activeLayoutId}
        onLoadLayout={handleLoadLayout}
        onResetLayout={handleResetLayout}
        onSaveLayout={handleSaveCurrentLayout}
        onDeleteLayout={handleDeleteLayout}
        onOpenAIScreen={handleOpenAIScreen}
        onAddPanel={handleAddPanel}
      />
      <div
        ref={containerRef}
        className="w-full"
        style={{ height: 'calc(100vh - 40px)' }}
      />
    </div>
  )
}

export default App
