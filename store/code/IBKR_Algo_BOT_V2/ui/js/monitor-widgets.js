/**
 * monitor-widgets.js
 * Widget Rendering and Data Management
 * Handles all widget types and their data display
 */

const WidgetManager = {
    widgets: new Map(),
    refreshIntervals: new Map(),

    /**
     * Add a widget to the dashboard
     */
    addWidget(type, size = 'md') {
        const widgetId = `widget-${type}-${Date.now()}`;
        const widget = this.createWidgetElement(widgetId, type, size);

        document.getElementById('dashboard-grid').appendChild(widget);
        this.widgets.set(widgetId, { type, size, element: widget });

        // Initialize widget data
        this.initializeWidget(widgetId, type);

        // Set up auto-refresh
        this.setupAutoRefresh(widgetId, type);

        return widgetId;
    },

    /**
     * Create widget DOM element
     */
    createWidgetElement(widgetId, type, size) {
        const template = document.getElementById('widget-template');
        const widget = template.content.cloneNode(true).querySelector('.widget');

        widget.id = widgetId;
        widget.classList.add(`widget-${size}`);
        widget.dataset.type = type;

        // Set widget header
        const config = this.getWidgetConfig(type);
        widget.querySelector('.widget-icon').className = `widget-icon ${config.icon}`;
        widget.querySelector('.widget-title-text').textContent = config.title;

        // Set widget content
        const contentTemplate = document.getElementById(`${type}-template`);
        if (contentTemplate) {
            const content = contentTemplate.content.cloneNode(true);
            widget.querySelector('.widget-content').appendChild(content);
        }

        return widget;
    },

    /**
     * Get widget configuration
     */
    getWidgetConfig(type) {
        const configs = {
            'active-trades': {
                title: 'Active Trades',
                icon: 'fas fa-list',
                refreshInterval: 2000
            },
            'trade-history': {
                title: 'Trade History',
                icon: 'fas fa-history',
                refreshInterval: 5000
            },
            'performance': {
                title: 'Performance Metrics',
                icon: 'fas fa-chart-bar',
                refreshInterval: 10000
            },
            'pnl-chart': {
                title: 'P&L Chart',
                icon: 'fas fa-chart-area',
                refreshInterval: 10000
            },
            'errors': {
                title: 'Error Log',
                icon: 'fas fa-exclamation-triangle',
                refreshInterval: 5000
            },
            'slippage': {
                title: 'Slippage Monitor',
                icon: 'fas fa-tachometer-alt',
                refreshInterval: 5000
            },
            'watch-list': {
                title: 'Watch List',
                icon: 'fas fa-eye',
                refreshInterval: 3000
            },
            'alerts': {
                title: 'Alerts',
                icon: 'fas fa-bell',
                refreshInterval: 3000
            }
        };

        return configs[type] || { title: type, icon: 'fas fa-widget', refreshInterval: 5000 };
    },

    /**
     * Initialize widget with data
     */
    async initializeWidget(widgetId, type) {
        try {
            switch (type) {
                case 'active-trades':
                    await this.updateActiveTrades(widgetId);
                    break;
                case 'trade-history':
                    await this.updateTradeHistory(widgetId);
                    break;
                case 'performance':
                    await this.updatePerformance(widgetId);
                    break;
                case 'pnl-chart':
                    await this.updatePnLChart(widgetId);
                    break;
                case 'errors':
                    await this.updateErrors(widgetId);
                    break;
                case 'slippage':
                    await this.updateSlippage(widgetId);
                    break;
                case 'alerts':
                    await this.updateAlerts(widgetId);
                    break;
            }
        } catch (error) {
            console.error(`Failed to initialize widget ${widgetId}:`, error);
        }
    },

    /**
     * Set up auto-refresh for widget
     */
    setupAutoRefresh(widgetId, type) {
        const config = this.getWidgetConfig(type);
        const interval = setInterval(() => {
            this.initializeWidget(widgetId, type);
        }, config.refreshInterval);

        this.refreshIntervals.set(widgetId, interval);
    },

    /**
     * Update Active Trades widget
     */
    async updateActiveTrades(widgetId) {
        const widget = document.getElementById(widgetId);
        const tbody = widget.querySelector('#active-trades-tbody');

        const data = await MonitorAPI.getActiveTrades();

        if (!data.success || !data.active_trades.length) {
            tbody.innerHTML = '<tr><td colspan="8" style="text-align: center; color: var(--text-muted);">No active trades</td></tr>';
            return;
        }

        tbody.innerHTML = data.active_trades.map(trade => `
            <tr>
                <td><strong>${trade.symbol}</strong></td>
                <td><span class="status-badge ${trade.side}">${trade.side.toUpperCase()}</span></td>
                <td>${trade.shares}</td>
                <td>$${trade.entry_price.toFixed(2)}</td>
                <td>$--</td>
                <td class="neutral">$--</td>
                <td>${trade.pattern_type || '--'}</td>
                <td>
                    <button class="widget-btn" onclick="exitTrade('${trade.trade_id}')" title="Exit">
                        <i class="fas fa-sign-out-alt"></i>
                    </button>
                </td>
            </tr>
        `).join('');
    },

    /**
     * Update Trade History widget
     */
    async updateTradeHistory(widgetId) {
        const widget = document.getElementById(widgetId);
        const tbody = widget.querySelector('#trade-history-tbody');

        const data = await MonitorAPI.getTrades({ limit: 50 });

        if (!data.success || !data.trades.length) {
            tbody.innerHTML = '<tr><td colspan="9" style="text-align: center; color: var(--text-muted);">No trade history</td></tr>';
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
    },

    /**
     * Update Performance Metrics widget
     */
    async updatePerformance(widgetId) {
        const widget = document.getElementById(widgetId);

        const data = await MonitorAPI.getTradeSummary(null, 30);

        if (!data.success || !data.summary) {
            return;
        }

        const summary = data.summary;

        // Update metric values
        widget.querySelector('#metric-win-rate').textContent = `${summary.win_rate}%`;
        widget.querySelector('#metric-win-rate').className = `metric-value ${summary.win_rate >= 50 ? 'positive' : 'negative'}`;

        widget.querySelector('#metric-total-pnl').textContent = `$${summary.total_pnl.toFixed(2)}`;
        widget.querySelector('#metric-total-pnl').className = `metric-value ${summary.total_pnl >= 0 ? 'positive' : 'negative'}`;

        widget.querySelector('#metric-profit-factor').textContent = summary.profit_factor.toFixed(2);
        widget.querySelector('#metric-profit-factor').className = `metric-value ${summary.profit_factor >= 1.5 ? 'positive' : 'negative'}`;

        widget.querySelector('#metric-avg-r').textContent = `${summary.avg_r_multiple.toFixed(2)}R`;
        widget.querySelector('#metric-avg-r').className = `metric-value ${summary.avg_r_multiple >= 1 ? 'positive' : 'negative'}`;

        widget.querySelector('#metric-total-trades').textContent = summary.total_trades;

        widget.querySelector('#metric-best-trade').textContent = `$${summary.largest_win.toFixed(2)}`;
        widget.querySelector('#metric-best-trade').className = 'metric-value positive';
    },

    /**
     * Update P&L Chart widget
     */
    async updatePnLChart(widgetId) {
        const widget = document.getElementById(widgetId);
        const canvas = widget.querySelector('#pnl-chart');

        const data = await MonitorAPI.getDailyPerformance(30);

        if (!data.success || !data.metrics.length) {
            return;
        }

        // Prepare chart data
        const labels = data.metrics.map(m => m.date).reverse();
        const pnlData = data.metrics.map(m => m.total_pnl).reverse();

        // Calculate cumulative P&L
        let cumulative = 0;
        const cumulativePnL = pnlData.map(pnl => {
            cumulative += pnl;
            return cumulative;
        });

        // Destroy existing chart if any
        if (widget.chartInstance) {
            widget.chartInstance.destroy();
        }

        // Create new chart
        widget.chartInstance = new Chart(canvas, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Cumulative P&L',
                    data: cumulativePnL,
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        grid: {
                            color: '#2d3748'
                        },
                        ticks: {
                            color: '#a0a7b8',
                            callback: (value) => '$' + value.toFixed(0)
                        }
                    },
                    x: {
                        grid: {
                            color: '#2d3748'
                        },
                        ticks: {
                            color: '#a0a7b8'
                        }
                    }
                }
            }
        });
    },

    /**
     * Update Errors widget
     */
    async updateErrors(widgetId) {
        const widget = document.getElementById(widgetId);
        const tbody = widget.querySelector('#errors-tbody');

        const data = await MonitorAPI.getErrors({ resolved: false, limit: 20 });

        if (!data.success || !data.errors.length) {
            tbody.innerHTML = '<tr><td colspan="6" style="text-align: center; color: var(--text-muted);">No errors</td></tr>';
            return;
        }

        tbody.innerHTML = data.errors.map(error => `
            <tr>
                <td>${new Date(error.timestamp).toLocaleString()}</td>
                <td><span class="status-badge ${error.severity}">${error.severity}</span></td>
                <td>${error.module}</td>
                <td>${error.error_message}</td>
                <td>${error.resolved ? '✓ Resolved' : 'Open'}</td>
                <td>
                    ${!error.resolved ? `
                        <button class="widget-btn" onclick="resolveErrorDialog('${error.error_id}')" title="Resolve">
                            <i class="fas fa-check"></i>
                        </button>
                    ` : ''}
                </td>
            </tr>
        `).join('');
    },

    /**
     * Update Slippage widget
     */
    async updateSlippage(widgetId) {
        const widget = document.getElementById(widgetId);

        const data = await MonitorAPI.getSlippageStats(null, 7);

        if (!data.success || !data.stats) {
            return;
        }

        const stats = data.stats;

        widget.querySelector('#slip-total').textContent = stats.total_executions || 0;
        widget.querySelector('#slip-avg').textContent = stats.avg_slippage ? (stats.avg_slippage * 100).toFixed(3) + '%' : '--';
        widget.querySelector('#slip-acceptable').textContent = stats.acceptable_count || 0;
        widget.querySelector('#slip-warning').textContent = stats.warning_count || 0;
        widget.querySelector('#slip-critical').textContent = stats.critical_count || 0;
        widget.querySelector('#slip-cost').textContent = stats.total_cost ? '$' + stats.total_cost.toFixed(2) : '$0.00';
    },

    /**
     * Update Alerts widget
     */
    async updateAlerts(widgetId) {
        const widget = document.getElementById(widgetId);
        const alertsList = widget.querySelector('#alerts-list');

        // Mock alerts data (would come from API in production)
        const alerts = [
            { type: 'info', title: 'System Started', message: 'Trading bot initialized successfully', time: new Date() },
            { type: 'warning', title: 'High Volatility', message: 'SPY volatility above 20%', time: new Date() }
        ];

        if (!alerts.length) {
            alertsList.innerHTML = '<p style="text-align: center; color: var(--text-muted);">No alerts</p>';
            return;
        }

        alertsList.innerHTML = alerts.map(alert => `
            <div class="alert-item ${alert.type}">
                <div class="alert-header">
                    <span class="alert-title">${alert.title}</span>
                    <span class="alert-time">${alert.time.toLocaleTimeString()}</span>
                </div>
                <div class="alert-message">${alert.message}</div>
            </div>
        `).join('');
    },

    /**
     * Remove widget
     */
    removeWidget(widgetId) {
        const interval = this.refreshIntervals.get(widgetId);
        if (interval) {
            clearInterval(interval);
            this.refreshIntervals.delete(widgetId);
        }

        this.widgets.delete(widgetId);

        const element = document.getElementById(widgetId);
        if (element) {
            element.remove();
        }
    }
};

// Global helper functions called from HTML
function refreshWidget(button) {
    const widget = button.closest('.widget');
    const widgetData = WidgetManager.widgets.get(widget.id);
    if (widgetData) {
        WidgetManager.initializeWidget(widget.id, widgetData.type);
    }
}

function toggleWidget(button) {
    const widget = button.closest('.widget');
    widget.classList.toggle('minimized');

    const icon = button.querySelector('i');
    icon.className = widget.classList.contains('minimized') ? 'fas fa-plus' : 'fas fa-minus';
}

function removeWidget(button) {
    const widget = button.closest('.widget');
    WidgetManager.removeWidget(widget.id);
}

async function resolveErrorDialog(errorId) {
    const notes = prompt('Resolution notes (optional):');
    if (notes !== null) {
        await MonitorAPI.resolveError(errorId, notes);
        // Refresh errors widget
        document.querySelectorAll('[data-type="errors"]').forEach(widget => {
            WidgetManager.initializeWidget(widget.id, 'errors');
        });
    }
}

function exitTrade(tradeId) {
    if (confirm('Exit this trade?')) {
        console.log('Exiting trade:', tradeId);
        // Would call trade exit API
    }
}
