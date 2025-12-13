/**
 * monitor-main.js
 * Main Application Controller
 * Handles initialization, menu interactions, and global functions
 */

// Global state
const AppState = {
    currentLayout: null,
    menuOpen: null
};

/**
 * Initialize the application
 */
async function init() {
    console.log('Initializing Warrior Trading Monitor...');

    // Initialize API connection
    await MonitorAPI.init();

    // Initialize drag-and-drop
    DragDropManager.init();

    // Load saved layout or default (using UI-specific key)
    const storageKey = `${UI_TYPE}_dashboard-layout`;
    const savedLayout = localStorage.getItem(storageKey);
    if (savedLayout) {
        DragDropManager.loadLayout();
    } else {
        DragDropManager.loadDefaultLayout();
    }

    // Set up clock
    updateClock();
    setInterval(updateClock, 1000);

    // Set up WebSocket listeners
    setupWebSocketListeners();

    // Set up keyboard shortcuts
    setupKeyboardShortcuts();

    // Close menus when clicking outside
    document.addEventListener('click', (e) => {
        if (!e.target.closest('.menu-item')) {
            closeAllMenus();
        }
    });

    console.log('Warrior Trading Monitor initialized');
}

/**
 * Update clock display
 */
function updateClock() {
    const now = new Date();
    const timeString = now.toLocaleTimeString('en-US', { hour12: false });
    document.getElementById('clock').textContent = timeString;
}

/**
 * Set up WebSocket event listeners
 */
function setupWebSocketListeners() {
    window.addEventListener('ws-trade-update', (e) => {
        console.log('Trade update received:', e.detail);
        // Refresh active trades widget
        document.querySelectorAll('[data-type="active-trades"]').forEach(widget => {
            WidgetManager.initializeWidget(widget.id, 'active-trades');
        });
    });

    window.addEventListener('ws-error-alert', (e) => {
        console.log('Error alert received:', e.detail);
        // Refresh errors widget
        document.querySelectorAll('[data-type="errors"]').forEach(widget => {
            WidgetManager.initializeWidget(widget.id, 'errors');
        });
    });
}

/**
 * Set up keyboard shortcuts
 */
function setupKeyboardShortcuts() {
    document.addEventListener('keydown', (e) => {
        // Ctrl+N: New Dashboard
        if (e.ctrlKey && e.key === 'n') {
            e.preventDefault();
            newDashboard();
        }

        // Ctrl+S: Save Layout
        if (e.ctrlKey && e.key === 's') {
            e.preventDefault();
            saveLayout();
        }

        // Ctrl+O: Load Layout
        if (e.ctrlKey && e.key === 'o') {
            e.preventDefault();
            loadLayout();
        }

        // F11: Toggle Fullscreen
        if (e.key === 'F11') {
            e.preventDefault();
            toggleFullscreen();
        }

        // Ctrl+R: Reset Layout
        if (e.ctrlKey && e.key === 'r') {
            e.preventDefault();
            resetLayout();
        }
    });
}

// ==================== MENU FUNCTIONS ====================

/**
 * Open a menu dropdown
 */
function openMenu(menuId) {
    closeAllMenus();

    const menu = document.getElementById(`${menuId}-menu`);
    if (menu) {
        menu.classList.add('show');
        AppState.menuOpen = menuId;
    }
}

/**
 * Close all menus
 */
function closeAllMenus() {
    document.querySelectorAll('.dropdown').forEach(menu => {
        menu.classList.remove('show');
    });
    AppState.menuOpen = null;
}

// ==================== FILE MENU FUNCTIONS ====================

/**
 * Create new dashboard
 */
function newDashboard() {
    if (confirm('Create a new dashboard? This will clear the current layout.')) {
        document.querySelectorAll('.widget').forEach(w => w.remove());
        WidgetManager.widgets.clear();
        const storageKey = `${UI_TYPE}_dashboard-layout`;
        localStorage.removeItem(storageKey);
        console.log('New dashboard created');
    }
}

/**
 * Load saved layout
 */
async function loadLayout() {
    const data = await MonitorAPI.getLayouts(UI_TYPE);

    if (!data.success || !data.layouts.length) {
        alert(`No saved layouts found for ${UI_TYPE} dashboard`);
        return;
    }

    // Show layout selector dialog
    const layoutNames = data.layouts.map(l => l.layout_name).join('\n');
    const selected = prompt(`Select a layout:\n\n${layoutNames}\n\nEnter layout name:`);

    if (selected) {
        await DragDropManager.loadLayout(selected);
        console.log(`Layout "${selected}" loaded for ${UI_TYPE}`);
    }
}

/**
 * Save current layout
 */
async function saveLayout() {
    const layoutName = prompt(`Enter a name for this ${UI_TYPE} layout:`);

    if (!layoutName) return;

    const isDefault = confirm(`Set as default ${UI_TYPE} layout?`);

    // Get current layout
    const widgets = document.querySelectorAll('.widget');
    const layout = [];

    widgets.forEach((widget, index) => {
        const widgetData = WidgetManager.widgets.get(widget.id);
        if (widgetData) {
            layout.push({
                id: widget.id,
                type: widgetData.type,
                order: index,
                gridColumn: widget.style.gridColumn,
                gridRow: widget.style.gridRow
            });
        }
    });

    try {
        await MonitorAPI.saveLayout(layoutName, layout, isDefault, UI_TYPE);
        alert(`Layout "${layoutName}" saved successfully for ${UI_TYPE} dashboard`);
    } catch (error) {
        alert('Failed to save layout: ' + error.message);
    }
}

/**
 * Export data
 */
function exportData() {
    alert('Export functionality coming soon!');
    // TODO: Implement data export (trades, errors, performance to CSV/JSON)
}

// ==================== WIDGETS MENU FUNCTIONS ====================

/**
 * Add widget to dashboard
 */
function addWidget(type) {
    const size = prompt('Widget size (sm/md/lg/xl/full):', 'md');
    if (size) {
        WidgetManager.addWidget(type, size);
    }
    closeAllMenus();
}

// ==================== VIEW MENU FUNCTIONS ====================

/**
 * Toggle fullscreen mode
 */
function toggleFullscreen() {
    if (!document.fullscreenElement) {
        document.documentElement.requestFullscreen();
    } else {
        if (document.exitFullscreen) {
            document.exitFullscreen();
        }
    }
}

/**
 * Reset layout to default
 */
function resetLayout() {
    DragDropManager.resetLayout();
}

/**
 * Toggle grid lines
 */
function toggleGrid() {
    const grid = document.getElementById('dashboard-grid');
    grid.classList.toggle('show-grid');
}

// ==================== TOOLS MENU FUNCTIONS ====================

/**
 * Open custom monitor builder
 */
function openBuilder() {
    alert('Custom Monitor Builder coming soon!');
    // TODO: Implement visual builder interface
}

/**
 * Open settings dialog
 */
function openSettings() {
    alert('Settings dialog coming soon!');
    // TODO: Implement settings (refresh intervals, API endpoint, theme, etc.)
}

/**
 * Manage saved layouts
 */
async function manageLayouts() {
    const data = await MonitorAPI.getLayouts();

    if (!data.success || !data.layouts.length) {
        alert('No saved layouts found');
        return;
    }

    const layoutList = data.layouts.map((l, i) =>
        `${i + 1}. ${l.layout_name}${l.is_default ? ' (default)' : ''}`
    ).join('\n');

    alert(`Saved Layouts:\n\n${layoutList}\n\nUse File > Load Layout to load a saved layout.`);
    // TODO: Implement full layout management (rename, delete, set default)
}

// ==================== HELP MENU FUNCTIONS ====================

/**
 * Show documentation
 */
function showDocs() {
    window.open('/docs', '_blank');
}

/**
 * Show keyboard shortcuts
 */
function showShortcuts() {
    const shortcuts = `
Keyboard Shortcuts:

Ctrl+N - New Dashboard
Ctrl+S - Save Layout
Ctrl+O - Load Layout
Ctrl+R - Reset Layout
F11    - Toggle Fullscreen

Widget Controls:
- Drag widget header to move
- Drag bottom-right corner to resize
- Click minimize to collapse
- Click X to remove
    `.trim();

    alert(shortcuts);
}

/**
 * Show about dialog
 */
function showAbout() {
    alert(`
Warrior Trading Monitor
Version 1.0.0

Multi-widget dashboard for tracking trades, errors, and performance.

Features:
- Real-time trade monitoring
- Error tracking and troubleshooting
- Performance analytics
- Custom layouts
- Multi-monitor support
- WebSocket real-time updates

Built with FastAPI, Chart.js, and vanilla JavaScript.
    `.trim());
}

// ==================== FILTER FUNCTIONS ====================

/**
 * Apply filters to trade history
 */
async function applyFilters() {
    const symbol = document.getElementById('filter-symbol').value.trim().toUpperCase();
    const status = document.getElementById('filter-status').value;

    const filters = {};
    if (symbol) filters.symbol = symbol;
    if (status) filters.status = status;

    // Update all trade-history widgets
    document.querySelectorAll('[data-type="trade-history"]').forEach(async widget => {
        const tbody = widget.querySelector('#trade-history-tbody');
        const data = await MonitorAPI.getTrades(filters);

        if (!data.success || !data.trades.length) {
            tbody.innerHTML = '<tr><td colspan="9" style="text-align: center; color: var(--text-muted);">No trades found</td></tr>';
            return;
        }

        tbody.innerHTML = data.trades.map(trade => {
            const pnlClass = trade.pnl > 0 ? 'positive' : trade.pnl < 0 ? 'negative' : 'neutral';
            const entryTime = new Date(trade.entry_time).toLocaleString();

            return `
                <tr>
                    <td>${entryTime}</td>
                    <td><strong>${trade.symbol}</strong></td>
                    <td>${trade.side.toUpperCase()}</td>
                    <td>${trade.shares}</td>
                    <td>$${trade.entry_price.toFixed(2)}</td>
                    <td>$${trade.exit_price ? trade.exit_price.toFixed(2) : '--'}</td>
                    <td class="${pnlClass}">$${trade.pnl ? trade.pnl.toFixed(2) : '--'}</td>
                    <td>${trade.r_multiple ? trade.r_multiple.toFixed(2) + 'R' : '--'}</td>
                    <td><span class="status-badge ${trade.status}">${trade.status}</span></td>
                </tr>
            `;
        }).join('');
    });
}

// ==================== INITIALIZATION ====================

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}

// Handle page unload
window.addEventListener('beforeunload', () => {
    // Save current layout
    DragDropManager.saveCurrentLayout();
});
