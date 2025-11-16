/**
 * monitor-dragdrop.js
 * Drag-and-Drop Functionality for Widgets
 * Enables widget repositioning and resizing
 */

const DragDropManager = {
    draggedElement: null,
    dragStartX: 0,
    dragStartY: 0,
    resizing: false,
    resizeElement: null,
    originalWidth: 0,
    originalHeight: 0,

    /**
     * Initialize drag-and-drop for all widgets
     */
    init() {
        this.setupDragEvents();
        this.setupResizeEvents();
    },

    /**
     * Set up drag event listeners
     */
    setupDragEvents() {
        document.addEventListener('dragstart', (e) => {
            if (e.target.classList.contains('widget')) {
                this.handleDragStart(e);
            }
        });

        document.addEventListener('dragend', (e) => {
            if (e.target.classList.contains('widget')) {
                this.handleDragEnd(e);
            }
        });

        document.addEventListener('dragover', (e) => {
            e.preventDefault();
        });

        document.addEventListener('drop', (e) => {
            e.preventDefault();
            this.handleDrop(e);
        });
    },

    /**
     * Set up resize event listeners
     */
    setupResizeEvents() {
        document.addEventListener('mousedown', (e) => {
            if (e.target.classList.contains('widget-resize-handle')) {
                this.handleResizeStart(e);
            }
        });

        document.addEventListener('mousemove', (e) => {
            if (this.resizing) {
                this.handleResize(e);
            }
        });

        document.addEventListener('mouseup', (e) => {
            if (this.resizing) {
                this.handleResizeEnd(e);
            }
        });
    },

    /**
     * Handle drag start
     */
    handleDragStart(e) {
        this.draggedElement = e.target;
        this.draggedElement.classList.add('dragging');

        // Store original grid position
        const style = window.getComputedStyle(e.target);
        e.dataTransfer.effectAllowed = 'move';
    },

    /**
     * Handle drag end
     */
    handleDragEnd(e) {
        if (this.draggedElement) {
            this.draggedElement.classList.remove('dragging');
            this.draggedElement = null;
        }
    },

    /**
     * Handle drop
     */
    handleDrop(e) {
        if (!this.draggedElement) return;

        const grid = document.getElementById('dashboard-grid');
        const dropTarget = e.target.closest('.widget');

        if (dropTarget && dropTarget !== this.draggedElement) {
            // Swap positions
            const parent = dropTarget.parentNode;
            const draggedNextSibling = this.draggedElement.nextSibling;
            const dropNextSibling = dropTarget.nextSibling;

            parent.insertBefore(this.draggedElement, dropTarget);
            if (draggedNextSibling !== dropTarget) {
                parent.insertBefore(dropTarget, draggedNextSibling);
            }
        }

        // Save layout
        this.saveCurrentLayout();
    },

    /**
     * Handle resize start
     */
    handleResizeStart(e) {
        e.preventDefault();
        this.resizing = true;
        this.resizeElement = e.target.closest('.widget');
        this.dragStartX = e.clientX;
        this.dragStartY = e.clientY;

        const rect = this.resizeElement.getBoundingClientRect();
        this.originalWidth = rect.width;
        this.originalHeight = rect.height;

        document.body.style.cursor = 'nwse-resize';
    },

    /**
     * Handle resize
     */
    handleResize(e) {
        if (!this.resizing || !this.resizeElement) return;

        const deltaX = e.clientX - this.dragStartX;
        const deltaY = e.clientY - this.dragStartY;

        const newWidth = this.originalWidth + deltaX;
        const newHeight = this.originalHeight + deltaY;

        // Calculate grid spans (assuming 12-column grid)
        const gridWidth = document.getElementById('dashboard-grid').offsetWidth;
        const columnWidth = gridWidth / 12;
        const rowHeight = 100; // Approximate row height

        const colSpan = Math.max(1, Math.round(newWidth / columnWidth));
        const rowSpan = Math.max(1, Math.round(newHeight / rowHeight));

        // Update grid spans
        this.resizeElement.style.gridColumn = `span ${colSpan}`;
        this.resizeElement.style.gridRow = `span ${rowSpan}`;
    },

    /**
     * Handle resize end
     */
    handleResizeEnd(e) {
        this.resizing = false;
        this.resizeElement = null;
        document.body.style.cursor = '';

        // Save layout
        this.saveCurrentLayout();
    },

    /**
     * Save current layout configuration
     */
    saveCurrentLayout() {
        const widgets = document.querySelectorAll('.widget');
        const layout = [];

        widgets.forEach((widget, index) => {
            const widgetData = WidgetManager.widgets.get(widget.id);
            if (widgetData) {
                const rect = widget.getBoundingClientRect();
                layout.push({
                    id: widget.id,
                    type: widgetData.type,
                    order: index,
                    gridColumn: widget.style.gridColumn,
                    gridRow: widget.style.gridRow
                });
            }
        });

        // Store in localStorage
        localStorage.setItem('dashboard-layout', JSON.stringify(layout));

        console.log('Layout saved:', layout);
    },

    /**
     * Load layout configuration
     */
    async loadLayout(layoutName = null) {
        let layout;

        if (layoutName) {
            // Load from server
            const data = await MonitorAPI.getLayouts();
            const savedLayout = data.layouts.find(l => l.layout_name === layoutName);
            if (savedLayout) {
                layout = savedLayout.layout_config;
            }
        } else {
            // Load from localStorage
            const stored = localStorage.getItem('dashboard-layout');
            if (stored) {
                layout = JSON.parse(stored);
            }
        }

        if (!layout) {
            console.log('No saved layout found');
            return;
        }

        // Clear current widgets
        document.querySelectorAll('.widget').forEach(w => w.remove());
        WidgetManager.widgets.clear();

        // Recreate widgets
        layout.forEach(item => {
            const widgetId = WidgetManager.addWidget(item.type, 'md');
            const widget = document.getElementById(widgetId);

            if (item.gridColumn) {
                widget.style.gridColumn = item.gridColumn;
            }
            if (item.gridRow) {
                widget.style.gridRow = item.gridRow;
            }
        });

        console.log('Layout loaded:', layout);
    },

    /**
     * Reset to default layout
     */
    resetLayout() {
        if (confirm('Reset to default layout? This will remove all widgets.')) {
            localStorage.removeItem('dashboard-layout');
            document.querySelectorAll('.widget').forEach(w => w.remove());
            WidgetManager.widgets.clear();

            // Add default widgets
            this.loadDefaultLayout();
        }
    },

    /**
     * Load default layout
     */
    loadDefaultLayout() {
        // Add default widgets
        const defaults = [
            { type: 'active-trades', size: 'lg' },
            { type: 'performance', size: 'md' },
            { type: 'pnl-chart', size: 'lg' },
            { type: 'errors', size: 'md' }
        ];

        defaults.forEach(item => {
            WidgetManager.addWidget(item.type, item.size);
        });
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = DragDropManager;
}
