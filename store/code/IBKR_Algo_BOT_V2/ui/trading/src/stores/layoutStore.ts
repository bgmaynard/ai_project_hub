import { create } from 'zustand'
import { LayoutConfig, ResolvedLayoutConfig } from 'golden-layout'

// Store layouts as any to handle both LayoutConfig and ResolvedLayoutConfig
export interface SavedLayout {
  id: string
  name: string
  config: LayoutConfig | ResolvedLayoutConfig
  createdAt: string
  updatedAt: string
  isPreset?: boolean
}

interface LayoutState {
  savedLayouts: SavedLayout[]
  activeLayoutId: string | null

  // Actions
  loadSavedLayouts: () => void
  saveLayout: (name: string, config: LayoutConfig | ResolvedLayoutConfig) => string
  updateLayout: (id: string, config: LayoutConfig | ResolvedLayoutConfig) => void
  deleteLayout: (id: string) => void
  renameLayout: (id: string, newName: string) => void
  duplicateLayout: (id: string, newName: string) => string | null
  setActiveLayout: (id: string | null) => void
  getLayout: (id: string) => SavedLayout | undefined
  getLayoutByName: (name: string) => SavedLayout | undefined
}

const STORAGE_KEY = 'morpheus_saved_layouts'
const ACTIVE_LAYOUT_KEY = 'morpheus_active_layout_id'

// Generate unique ID
const generateId = () => `layout_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`

// Default preset layouts
const createPresetLayouts = (): SavedLayout[] => {
  const now = new Date().toISOString()

  return [
    {
      id: 'preset_trading',
      name: 'Trading (Default)',
      isPreset: true,
      createdAt: now,
      updatedAt: now,
      config: {
        settings: { reorderEnabled: true },
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
    },
    {
      id: 'preset_scalping',
      name: 'Scalping',
      isPreset: true,
      createdAt: now,
      updatedAt: now,
      config: {
        settings: { reorderEnabled: true },
        root: {
          type: 'row',
          content: [
            {
              type: 'column',
              width: 20,
              content: [
                {
                  type: 'stack',
                  height: 50,
                  content: [
                    { type: 'component', componentType: 'Level2', title: 'Level 2' }
                  ]
                },
                {
                  type: 'stack',
                  height: 50,
                  content: [
                    { type: 'component', componentType: 'TimeSales', title: 'Time & Sales' }
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
                  height: 60,
                  content: [
                    { type: 'component', componentType: 'Chart', title: 'Chart' }
                  ]
                },
                {
                  type: 'row',
                  height: 40,
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
                }
              ]
            },
            {
              type: 'column',
              width: 30,
              content: [
                {
                  type: 'stack',
                  height: 25,
                  content: [
                    { type: 'component', componentType: 'Quote', title: 'Quote' }
                  ]
                },
                {
                  type: 'stack',
                  height: 35,
                  content: [
                    { type: 'component', componentType: 'OrderEntry', title: 'Order Entry' }
                  ]
                },
                {
                  type: 'stack',
                  height: 20,
                  content: [
                    { type: 'component', componentType: 'Positions', title: 'Positions' }
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
            }
          ]
        }
      }
    },
    {
      id: 'preset_analysis',
      name: 'Analysis',
      isPreset: true,
      createdAt: now,
      updatedAt: now,
      config: {
        settings: { reorderEnabled: true },
        root: {
          type: 'row',
          content: [
            {
              type: 'column',
              width: 30,
              content: [
                {
                  type: 'stack',
                  content: [
                    { type: 'component', componentType: 'Worklist', title: 'Worklist' },
                    { type: 'component', componentType: 'ScannerPanel', title: 'Scanners' }
                  ]
                }
              ]
            },
            {
              type: 'column',
              width: 70,
              content: [
                {
                  type: 'stack',
                  height: 70,
                  content: [
                    { type: 'component', componentType: 'Chart', title: 'Chart' }
                  ]
                },
                {
                  type: 'row',
                  height: 30,
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
          ]
        }
      }
    }
  ]
}

export const useLayoutStore = create<LayoutState>((set, get) => ({
  savedLayouts: [],
  activeLayoutId: null,

  loadSavedLayouts: () => {
    try {
      // Load saved layouts from localStorage
      const savedStr = localStorage.getItem(STORAGE_KEY)
      const activeId = localStorage.getItem(ACTIVE_LAYOUT_KEY)

      let layouts: SavedLayout[] = []

      if (savedStr) {
        layouts = JSON.parse(savedStr)
      }

      // Ensure presets exist
      const presets = createPresetLayouts()
      const presetIds = presets.map(p => p.id)

      // Remove any outdated presets and add fresh ones
      layouts = layouts.filter(l => !l.isPreset || !presetIds.includes(l.id))
      layouts = [...presets, ...layouts]

      // Save back with updated presets
      localStorage.setItem(STORAGE_KEY, JSON.stringify(layouts))

      set({
        savedLayouts: layouts,
        activeLayoutId: activeId || 'preset_trading'
      })
    } catch (e) {
      console.error('Failed to load saved layouts:', e)
      const presets = createPresetLayouts()
      localStorage.setItem(STORAGE_KEY, JSON.stringify(presets))
      set({ savedLayouts: presets, activeLayoutId: 'preset_trading' })
    }
  },

  saveLayout: (name: string, config: LayoutConfig | ResolvedLayoutConfig) => {
    const id = generateId()
    const now = new Date().toISOString()

    const newLayout: SavedLayout = {
      id,
      name,
      config,
      createdAt: now,
      updatedAt: now,
      isPreset: false
    }

    const layouts = [...get().savedLayouts, newLayout]
    localStorage.setItem(STORAGE_KEY, JSON.stringify(layouts))
    localStorage.setItem(ACTIVE_LAYOUT_KEY, id)

    set({ savedLayouts: layouts, activeLayoutId: id })
    return id
  },

  updateLayout: (id: string, config: LayoutConfig | ResolvedLayoutConfig) => {
    const layouts = get().savedLayouts.map(l => {
      if (l.id === id) {
        return {
          ...l,
          config,
          updatedAt: new Date().toISOString()
        }
      }
      return l
    })

    localStorage.setItem(STORAGE_KEY, JSON.stringify(layouts))
    set({ savedLayouts: layouts })
  },

  deleteLayout: (id: string) => {
    // Allow deleting any layout, including presets
    const layouts = get().savedLayouts.filter(l => l.id !== id)
    localStorage.setItem(STORAGE_KEY, JSON.stringify(layouts))

    // If deleting active layout, switch to first available
    const activeId = get().activeLayoutId
    if (activeId === id) {
      const newActiveId = layouts.length > 0 ? layouts[0].id : null
      if (newActiveId) {
        localStorage.setItem(ACTIVE_LAYOUT_KEY, newActiveId)
      } else {
        localStorage.removeItem(ACTIVE_LAYOUT_KEY)
      }
      set({ savedLayouts: layouts, activeLayoutId: newActiveId })
    } else {
      set({ savedLayouts: layouts })
    }
  },

  renameLayout: (id: string, newName: string) => {
    const layout = get().getLayout(id)
    if (layout?.isPreset) {
      console.warn('Cannot rename preset layouts')
      return
    }

    const layouts = get().savedLayouts.map(l => {
      if (l.id === id) {
        return { ...l, name: newName, updatedAt: new Date().toISOString() }
      }
      return l
    })

    localStorage.setItem(STORAGE_KEY, JSON.stringify(layouts))
    set({ savedLayouts: layouts })
  },

  duplicateLayout: (id: string, newName: string) => {
    const sourceLayout = get().getLayout(id)
    if (!sourceLayout) return null

    const newId = generateId()
    const now = new Date().toISOString()

    // Deep clone the config to avoid reference issues
    const clonedConfig = JSON.parse(JSON.stringify(sourceLayout.config))

    const newLayout: SavedLayout = {
      id: newId,
      name: newName,
      config: clonedConfig,
      createdAt: now,
      updatedAt: now,
      isPreset: false  // Duplicates are always custom
    }

    const layouts = [...get().savedLayouts, newLayout]
    localStorage.setItem(STORAGE_KEY, JSON.stringify(layouts))
    localStorage.setItem(ACTIVE_LAYOUT_KEY, newId)

    set({ savedLayouts: layouts, activeLayoutId: newId })
    return newId
  },

  setActiveLayout: (id: string | null) => {
    if (id) {
      localStorage.setItem(ACTIVE_LAYOUT_KEY, id)
    } else {
      localStorage.removeItem(ACTIVE_LAYOUT_KEY)
    }
    set({ activeLayoutId: id })
  },

  getLayout: (id: string) => {
    return get().savedLayouts.find(l => l.id === id)
  },

  getLayoutByName: (name: string) => {
    return get().savedLayouts.find(l => l.name === name)
  }
}))
