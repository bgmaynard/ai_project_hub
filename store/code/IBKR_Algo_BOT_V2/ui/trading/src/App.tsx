import { useEffect, useRef } from 'react'
import { GoldenLayout, LayoutConfig, ComponentContainer } from 'golden-layout'
import Menu from './components/Menu'
import { useSymbolStore } from './stores/symbolStore'

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

// Default layout configuration - ALL components wrapped in stacks for draggable tabs
const defaultLayout: LayoutConfig = {
  settings: {
    reorderEnabled: true,
  },
  header: {
    popout: false,
  },
  root: {
    type: 'row',
    content: [
      {
        type: 'column',
        width: 25,
        content: [
          {
            type: 'stack',
            height: 80,
            content: [
              { type: 'component', componentType: 'Positions', title: 'Positions' },
              { type: 'component', componentType: 'Orders', title: 'Orders' },
            ]
          },
          {
            type: 'stack',
            height: 20,
            content: [
              { type: 'component', componentType: 'Account', title: 'Account' }
            ]
          }
        ]
      },
      {
        type: 'column',
        width: 50,
        content: [
          {
            type: 'stack',
            height: 40,
            content: [
              { type: 'component', componentType: 'Chart', title: 'Chart' }
            ]
          },
          {
            type: 'row',
            height: 35,
            content: [
              {
                type: 'stack',
                content: [
                  { type: 'component', componentType: 'Worklist', title: 'Worklist' }
                ]
              },
              {
                type: 'stack',
                content: [
                  { type: 'component', componentType: 'ScannerPanel', title: 'Scanners' }
                ]
              }
            ]
          },
          {
            type: 'row',
            height: 25,
            content: [
              {
                type: 'stack',
                content: [
                  { type: 'component', componentType: 'Level2', title: 'Level 2' }
                ]
              },
              {
                type: 'stack',
                content: [
                  { type: 'component', componentType: 'TimeSales', title: 'Time & Sales' }
                ]
              }
            ]
          }
        ]
      },
      {
        type: 'column',
        width: 25,
        content: [
          {
            type: 'stack',
            height: 30,
            content: [
              { type: 'component', componentType: 'OrderEntry', title: 'Order Entry' }
            ]
          },
          {
            type: 'stack',
            content: [
              { type: 'component', componentType: 'BreakingNews', title: 'Breaking News' }
            ]
          }
        ]
      }
    ]
  }
}

function App() {
  const containerRef = useRef<HTMLDivElement>(null)
  const layoutRef = useRef<GoldenLayout | null>(null)
  const activeSymbol = useSymbolStore((state) => state.activeSymbol)

  useEffect(() => {
    if (!containerRef.current) return

    // Try to load saved layout with error handling
    let layoutConfig = defaultLayout
    try {
      const savedLayoutStr = localStorage.getItem('morpheus_trading_layout')
      if (savedLayoutStr) {
        const parsed = JSON.parse(savedLayoutStr)
        // Validate it has a root property
        if (parsed && parsed.root) {
          layoutConfig = parsed
        }
      }
    } catch (e) {
      console.warn('Failed to load saved layout, using default:', e)
      localStorage.removeItem('morpheus_trading_layout')
    }

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

    // Load layout with error handling
    try {
      layout.loadLayout(layoutConfig)
    } catch (e) {
      console.error('Failed to load layout, trying default:', e)
      localStorage.removeItem('morpheus_trading_layout')
      try {
        layout.loadLayout(defaultLayout)
      } catch (e2) {
        console.error('Failed to load default layout:', e2)
      }
    }

    // Save layout on state change
    layout.on('stateChanged', () => {
      const config = layout.saveLayout()
      localStorage.setItem('morpheus_trading_layout', JSON.stringify(config))
    })

    // Handle window resize
    const handleResize = () => {
      layout.setSize(window.innerWidth, window.innerHeight - 40) // Account for menu
    }
    window.addEventListener('resize', handleResize)
    handleResize()

    return () => {
      window.removeEventListener('resize', handleResize)
      layout.destroy()
    }
  }, [])

  const handleResetLayout = () => {
    // Clear stored layout first
    localStorage.removeItem('morpheus_trading_layout')
    if (layoutRef.current) {
      layoutRef.current.loadLayout(defaultLayout)
    }
  }

  const handleOpenAIScreen = () => {
    window.open('/ai-control-center/', 'ai_screen', 'width=1920,height=1080')
  }

  const handleAddPanel = (panelType: string) => {
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
  }

  return (
    <div className="h-screen w-screen overflow-hidden bg-sterling-bg">
      <Menu
        activeSymbol={activeSymbol}
        onResetLayout={handleResetLayout}
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
