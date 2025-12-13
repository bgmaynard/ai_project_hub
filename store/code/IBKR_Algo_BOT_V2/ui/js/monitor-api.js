/**
 * monitor-api.js
 * API Communication Layer for Warrior Trading Monitor
 * Handles all backend communication and WebSocket connections
 */

const MonitorAPI = {
    baseURL: 'http://localhost:8000',
    ws: null,
    wsReconnectAttempts: 0,
    maxReconnectAttempts: 5,
    reconnectDelay: 2000,

    /**
     * Initialize API connection and WebSocket
     */
    async init() {
        await this.connectWebSocket();
        await this.checkHealth();
    },

    /**
     * Check API health status
     */
    async checkHealth() {
        try {
            const response = await this.get('/api/monitoring/health');
            if (response.success) {
                updateConnectionStatus('connected', 'Connected');
                return true;
            }
        } catch (error) {
            console.error('Health check failed:', error);
            updateConnectionStatus('disconnected', 'Disconnected');
            return false;
        }
    },

    /**
     * Generic GET request
     */
    async get(endpoint, params = {}) {
        const queryString = new URLSearchParams(params).toString();
        const url = `${this.baseURL}${endpoint}${queryString ? '?' + queryString : ''}`;

        const response = await fetch(url, {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json'
            }
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        return await response.json();
    },

    /**
     * Generic POST request
     */
    async post(endpoint, data = {}) {
        const url = `${this.baseURL}${endpoint}`;

        const response = await fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        return await response.json();
    },

    /**
     * Connect to WebSocket for real-time updates
     */
    async connectWebSocket() {
        const wsURL = this.baseURL.replace('http', 'ws') + '/api/monitoring/stream';

        try {
            this.ws = new WebSocket(wsURL);

            this.ws.onopen = () => {
                console.log('WebSocket connected');
                this.wsReconnectAttempts = 0;
                updateConnectionStatus('connected', 'Connected');
            };

            this.ws.onmessage = (event) => {
                const message = JSON.parse(event.data);
                this.handleWebSocketMessage(message);
            };

            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
            };

            this.ws.onclose = () => {
                console.log('WebSocket disconnected');
                updateConnectionStatus('connecting', 'Reconnecting...');
                this.reconnectWebSocket();
            };

        } catch (error) {
            console.error('Failed to connect WebSocket:', error);
            this.reconnectWebSocket();
        }
    },

    /**
     * Reconnect WebSocket with exponential backoff
     */
    reconnectWebSocket() {
        if (this.wsReconnectAttempts >= this.maxReconnectAttempts) {
            console.error('Max reconnect attempts reached');
            updateConnectionStatus('disconnected', 'Connection Failed');
            return;
        }

        this.wsReconnectAttempts++;
        const delay = this.reconnectDelay * this.wsReconnectAttempts;

        setTimeout(() => {
            console.log(`Reconnecting... (attempt ${this.wsReconnectAttempts})`);
            this.connectWebSocket();
        }, delay);
    },

    /**
     * Handle incoming WebSocket messages
     */
    handleWebSocketMessage(message) {
        switch (message.type) {
            case 'initial_data':
                console.log('Received initial data:', message.data);
                // Trigger widget updates
                window.dispatchEvent(new CustomEvent('ws-initial-data', { detail: message.data }));
                break;

            case 'trade_update':
                console.log('Trade update:', message.data);
                window.dispatchEvent(new CustomEvent('ws-trade-update', { detail: message.data }));
                break;

            case 'error_alert':
                console.log('Error alert:', message.data);
                window.dispatchEvent(new CustomEvent('ws-error-alert', { detail: message.data }));
                showAlert('error', message.data.error_message);
                break;

            case 'heartbeat':
                // Keep-alive heartbeat
                break;

            case 'pong':
                // Response to ping
                break;

            default:
                console.log('Unknown message type:', message.type);
        }
    },

    // ==================== TRADE ENDPOINTS ====================

    /**
     * Get trades with filters
     */
    async getTrades(filters = {}) {
        return await this.get('/api/monitoring/trades', filters);
    },

    /**
     * Get active trades
     */
    async getActiveTrades() {
        return await this.get('/api/monitoring/active-trades');
    },

    /**
     * Get trade summary statistics
     */
    async getTradeSummary(symbol = null, days = 30) {
        return await this.get('/api/monitoring/trades/summary', { symbol, days });
    },

    // ==================== ERROR ENDPOINTS ====================

    /**
     * Get error logs with filters
     */
    async getErrors(filters = {}) {
        return await this.get('/api/monitoring/errors', filters);
    },

    /**
     * Resolve an error
     */
    async resolveError(errorId, notes = '') {
        return await this.post('/api/monitoring/errors/resolve', {
            error_id: errorId,
            resolution_notes: notes
        });
    },

    /**
     * Get error statistics
     */
    async getErrorStats(days = 7) {
        return await this.get('/api/monitoring/errors/stats', { days });
    },

    // ==================== PERFORMANCE ENDPOINTS ====================

    /**
     * Get daily performance metrics
     */
    async getDailyPerformance(days = 30) {
        return await this.get('/api/monitoring/performance/daily', { days });
    },

    /**
     * Calculate performance for a specific date
     */
    async calculatePerformance(targetDate = null) {
        return await this.post('/api/monitoring/performance/calculate', { target_date: targetDate });
    },

    /**
     * Get slippage statistics
     */
    async getSlippageStats(symbol = null, days = 7) {
        return await this.get('/api/monitoring/slippage/stats', { symbol, days });
    },

    // ==================== LAYOUT ENDPOINTS ====================

    /**
     * Save dashboard layout
     */
    async saveLayout(layoutName, layoutConfig, isDefault = false, uiType = 'monitor') {
        return await this.post('/api/monitoring/layouts/save', {
            layout_name: layoutName,
            layout_config: layoutConfig,
            is_default: isDefault,
            ui_type: uiType
        });
    },

    /**
     * Get all saved layouts for a specific UI type
     */
    async getLayouts(uiType = 'monitor') {
        return await this.get('/api/monitoring/layouts', { ui_type: uiType });
    },

    /**
     * Get default layout for a specific UI type
     */
    async getDefaultLayout(uiType = 'monitor') {
        return await this.get('/api/monitoring/layouts/default', { ui_type: uiType });
    }
};

/**
 * Update connection status indicator
 */
function updateConnectionStatus(status, text) {
    const indicator = document.getElementById('connection-status');
    if (!indicator) return;

    indicator.className = `status-indicator ${status}`;
    document.getElementById('status-text').textContent = text;
}

/**
 * Show alert notification
 */
function showAlert(type, message) {
    // Create toast notification
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    document.body.appendChild(toast);

    setTimeout(() => {
        toast.classList.add('show');
    }, 100);

    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => toast.remove(), 300);
    }, 5000);
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = MonitorAPI;
}
