import axios, { AxiosInstance } from 'axios';
import type {
  TrainingConfig,
  ModelPerformance,
  BacktestConfig,
  BacktestResponse,
  BacktestResults,
  AlertConfig,
  TradingViewPush,
  ClaudeInsights,
  ApiResponse,
  Prediction
} from '../types/models';

class ApiService {
  private client: AxiosInstance;
  private baseURL: string;

  constructor() {
    // Use relative URL to work with the same origin (port 9100)
    this.baseURL = '';
    this.client = axios.create({
      baseURL: this.baseURL,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json'
      }
    });

    // Request interceptor
    this.client.interceptors.request.use(
      (config) => {
        // Add auth token if available
        const token = localStorage.getItem('auth_token');
        if (token) {
          config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
      },
      (error) => Promise.reject(error)
    );

    // Response interceptor
    this.client.interceptors.response.use(
      (response) => response,
      (error) => {
        console.error('API Error:', error.response?.data || error.message);
        return Promise.reject(error);
      }
    );
  }

  // Model Management APIs
  async trainModel(config: TrainingConfig): Promise<ApiResponse<{ training_id: string; websocket_url: string }>> {
    const response = await this.client.post('/api/ai/models/train', config);
    return response.data;
  }

  async getTrainingInsights(trainingId: string): Promise<ApiResponse<ClaudeInsights>> {
    const response = await this.client.get(`/api/ai/models/train/${trainingId}/insights`);
    return response.data;
  }

  async getModelPerformance(): Promise<ApiResponse<ModelPerformance[]>> {
    const response = await this.client.get('/api/ai/models/performance');
    return response.data;
  }

  async compareModels(timeframe: string = '30d'): Promise<ApiResponse<any>> {
    const response = await this.client.get('/api/ai/models/compare', {
      params: { timeframe }
    });
    return response.data;
  }

  async getModelInsights(): Promise<ApiResponse<ClaudeInsights>> {
    const response = await this.client.get('/api/ai/models/insights');
    return response.data;
  }

  async getExperiments(): Promise<ApiResponse<any[]>> {
    const response = await this.client.get('/api/ai/models/experiments');
    return response.data;
  }

  async createExperiment(config: any): Promise<ApiResponse<{ experiment_id: string }>> {
    const response = await this.client.post('/api/ai/models/experiments/create', config);
    return response.data;
  }

  async promoteExperiment(experimentId: string): Promise<ApiResponse<any>> {
    const response = await this.client.post(`/api/ai/models/experiments/${experimentId}/promote`);
    return response.data;
  }

  async pauseExperiment(experimentId: string): Promise<ApiResponse<any>> {
    const response = await this.client.post(`/api/ai/models/experiments/${experimentId}/pause`);
    return response.data;
  }

  // Backtesting APIs
  async runBacktest(config: BacktestConfig): Promise<ApiResponse<{ backtest_id: string }>> {
    const response = await this.client.post('/api/backtest/run', config);
    return response.data;
  }

  async getBacktestResults(backtestId: string): Promise<ApiResponse<BacktestResponse>> {
    const response = await this.client.get(`/api/backtest/${backtestId}/results`);
    return response.data;
  }

  async stopBacktest(backtestId: string): Promise<ApiResponse<any>> {
    const response = await this.client.post(`/api/backtest/${backtestId}/stop`);
    return response.data;
  }

  async getBacktestAnalysis(backtestId: string): Promise<ApiResponse<ClaudeInsights>> {
    const response = await this.client.get(`/api/backtest/${backtestId}/claude-analysis`);
    return response.data;
  }

  async exportBacktest(backtestId: string, format: 'pdf' | 'csv' | 'json'): Promise<Blob> {
    const response = await this.client.post(`/api/backtest/${backtestId}/export`,
      { format },
      { responseType: 'blob' }
    );
    return response.data;
  }

  async getTradeAnalysis(tradeId: string): Promise<ApiResponse<any>> {
    const response = await this.client.get(`/api/backtest/trade/${tradeId}/analysis`);
    return response.data;
  }

  async getSimilarTrades(tradeId: string): Promise<ApiResponse<any>> {
    const response = await this.client.get(`/api/backtest/trade/${tradeId}/similar`);
    return response.data;
  }

  async findSimilarTrades(tradeId: string): Promise<ApiResponse<any>> {
    return this.getSimilarTrades(tradeId);
  }

  async optimizeStrategy(config: any): Promise<ApiResponse<ClaudeInsights>> {
    const response = await this.client.post('/api/claude/optimize-strategy', config);
    return response.data;
  }

  async getStrategies(): Promise<ApiResponse<any[]>> {
    const response = await this.client.get('/api/strategies');
    return response.data;
  }

  async saveStrategy(strategy: any): Promise<ApiResponse<{ strategy_id: string }>> {
    const response = await this.client.post('/api/strategies/save', strategy);
    return response.data;
  }

  // Live Predictions APIs
  async getLivePredictions(): Promise<ApiResponse<Prediction[]>> {
    const response = await this.client.get('/api/predictions/live');
    return response.data;
  }

  async getPredictionStats(): Promise<ApiResponse<any>> {
    const response = await this.client.get('/api/predictions/stats');
    return response.data;
  }

  async getClaudeCommentary(): Promise<ApiResponse<{ commentary: string }>> {
    const response = await this.client.get('/api/predictions/claude-commentary');
    return response.data;
  }

  async getPredictionAccuracy(): Promise<ApiResponse<any>> {
    const response = await this.client.get('/api/predictions/accuracy');
    return response.data;
  }

  async getPredictionDetails(symbol: string): Promise<ApiResponse<any>> {
    const response = await this.client.get(`/api/predictions/${symbol}/details`);
    return response.data;
  }

  async configureAlert(config: AlertConfig): Promise<ApiResponse<{ alert_id: string }>> {
    const response = await this.client.post('/api/predictions/alerts/configure', config);
    return response.data;
  }

  async getAlerts(): Promise<ApiResponse<AlertConfig[]>> {
    const response = await this.client.get('/api/predictions/alerts');
    return response.data;
  }

  async createAlert(config: AlertConfig): Promise<ApiResponse<{ alert_id: string }>> {
    const response = await this.client.post('/api/predictions/alerts/create', config);
    return response.data;
  }

  async toggleAlert(alertId: string, enabled: boolean): Promise<ApiResponse<any>> {
    const response = await this.client.put(`/api/predictions/alerts/${alertId}/toggle`, { enabled });
    return response.data;
  }

  async deleteAlert(alertId: string): Promise<ApiResponse<any>> {
    const response = await this.client.delete(`/api/predictions/alerts/${alertId}`);
    return response.data;
  }

  async getAlertHistory(): Promise<ApiResponse<any[]>> {
    const response = await this.client.get('/api/predictions/alerts/history');
    return response.data;
  }

  // TradingView APIs
  async pushToTradingView(data: TradingViewPush): Promise<ApiResponse<any>> {
    const response = await this.client.post('/api/tradingview/push', data);
    return response.data;
  }

  async getTradingViewStatus(): Promise<ApiResponse<any>> {
    const response = await this.client.get('/api/tradingview/status');
    return response.data;
  }

  async setTradingViewAutoSync(enabled: boolean): Promise<ApiResponse<any>> {
    const response = await this.client.post('/api/tradingview/auto-sync', { enabled });
    return response.data;
  }

  async getWebhookConfig(): Promise<ApiResponse<any>> {
    const response = await this.client.get('/api/tradingview/webhook/config');
    return response.data;
  }

  async updateWebhookConfig(config: any): Promise<ApiResponse<any>> {
    const response = await this.client.post('/api/tradingview/webhook/config', config);
    return response.data;
  }

  async getWebhookSettings(): Promise<ApiResponse<any>> {
    const response = await this.client.get('/api/tradingview/webhook/settings');
    return response.data;
  }

  async updateWebhookSettings(settings: any): Promise<ApiResponse<any>> {
    const response = await this.client.put('/api/tradingview/webhook/settings', settings);
    return response.data;
  }

  async updateWebhookFilters(filters: any): Promise<ApiResponse<any>> {
    const response = await this.client.put('/api/tradingview/webhook/filters', filters);
    return response.data;
  }

  async regenerateWebhookApiKey(): Promise<ApiResponse<any>> {
    const response = await this.client.post('/api/tradingview/webhook/regenerate-key');
    return response.data;
  }

  async testWebhook(): Promise<ApiResponse<any>> {
    const response = await this.client.post('/api/tradingview/webhook/test');
    return response.data;
  }

  async getWebhookHistory(): Promise<ApiResponse<any[]>> {
    const response = await this.client.get('/api/tradingview/webhook/history');
    return response.data;
  }

  async generatePineScript(strategy: string): Promise<ApiResponse<{ script: string }>> {
    const response = await this.client.post('/api/tradingview/pine-script/generate', { strategy });
    return response.data;
  }

  async generateIndicator(config: any): Promise<ApiResponse<{ script: string }>> {
    const response = await this.client.post('/api/tradingview/indicators/generate', config);
    return response.data;
  }

  async getSavedIndicators(): Promise<ApiResponse<any[]>> {
    const response = await this.client.get('/api/tradingview/indicators/saved');
    return response.data;
  }

  async saveIndicator(indicator: any): Promise<ApiResponse<{ indicator_id: string }>> {
    const response = await this.client.post('/api/tradingview/indicators/save', indicator);
    return response.data;
  }

  async deleteIndicator(indicatorId: string): Promise<ApiResponse<any>> {
    const response = await this.client.delete(`/api/tradingview/indicators/${indicatorId}`);
    return response.data;
  }

  // Claude Orchestrator APIs
  async claudeQuery(query: string, context?: any): Promise<ApiResponse<{ response: string; conversation_id?: string }>> {
    const response = await this.client.post('/api/claude/query', { query, context });
    return response.data;
  }

  async getDailyReview(date?: string): Promise<ApiResponse<any>> {
    const response = await this.client.get('/api/claude/daily-review', {
      params: date ? { date } : {}
    });
    return response.data;
  }

  async getDailyReviewHistory(): Promise<ApiResponse<any[]>> {
    const response = await this.client.get('/api/claude/daily-review/history');
    return response.data;
  }

  async exportDailyReview(date: string, format: 'pdf' | 'json'): Promise<Blob> {
    const response = await this.client.post('/api/claude/daily-review/export',
      { date, format },
      { responseType: 'blob' }
    );
    return response.data;
  }

  async scanOptimizations(): Promise<ApiResponse<any[]>> {
    const response = await this.client.post('/api/claude/scan-optimizations');
    return response.data;
  }

  async applyOptimization(optimizationId: string): Promise<ApiResponse<any>> {
    const response = await this.client.post('/api/claude/apply-optimization', { optimization_id: optimizationId });
    return response.data;
  }

  async revertOptimization(optimizationId: string): Promise<ApiResponse<any>> {
    const response = await this.client.post('/api/claude/revert-optimization', { optimization_id: optimizationId });
    return response.data;
  }

  async getOptimizationHistory(): Promise<ApiResponse<any[]>> {
    const response = await this.client.get('/api/claude/optimization-history');
    return response.data;
  }

  async getConversations(): Promise<ApiResponse<any[]>> {
    const response = await this.client.get('/api/claude/conversations');
    return response.data;
  }

  async getConversation(conversationId: string): Promise<ApiResponse<any>> {
    const response = await this.client.get(`/api/claude/conversation/${conversationId}`);
    return response.data;
  }

  // Worklist APIs
  async getWorklist(): Promise<ApiResponse<any[]>> {
    const response = await this.client.get('/api/worklist');
    return response.data;
  }

  async addToWorklist(item: { symbol: string; exchange?: string; notes?: string }): Promise<ApiResponse<any>> {
    const response = await this.client.post('/api/worklist/add', item);
    return response.data;
  }

  async removeFromWorklist(symbol: string): Promise<ApiResponse<any>> {
    const response = await this.client.delete(`/api/worklist/${symbol}`);
    return response.data;
  }

  async updateWorklistNotes(symbol: string, data: { notes: string }): Promise<ApiResponse<any>> {
    const response = await this.client.put(`/api/worklist/${symbol}/notes`, data);
    return response.data;
  }

  async clearWorklist(): Promise<ApiResponse<any>> {
    const response = await this.client.delete('/api/worklist/clear');
    return response.data;
  }

  // Scanner APIs
  async getScannerPresets(): Promise<ApiResponse<any[]>> {
    const response = await this.client.get('/api/scanner/ibkr/presets');
    return response.data;
  }

  async runScanner(config: any): Promise<ApiResponse<any>> {
    const response = await this.client.post('/api/scanner/ibkr/scan', config);
    return response.data;
  }

  async addScannerToWorklist(data: { symbols: string[] }): Promise<ApiResponse<any>> {
    const response = await this.client.post('/api/scanner/ibkr/add-to-worklist', data);
    return response.data;
  }

  // Connectivity / Governor APIs
  async getConnectivityStatus(): Promise<ApiResponse<any>> {
    const response = await this.client.get('/api/validation/connectivity/status');
    return response.data;
  }

  async runConnectivitySelfTest(): Promise<ApiResponse<any>> {
    const response = await this.client.post('/api/validation/connectivity/self-test');
    return response.data;
  }

  async reconnectFeeds(paperMode: boolean = true): Promise<ApiResponse<any>> {
    const response = await this.client.post('/api/validation/connectivity/reconnect', { paper_mode: paperMode });
    return response.data;
  }

  async getConnectivityReport(): Promise<ApiResponse<any>> {
    const response = await this.client.get('/api/validation/connectivity/report');
    return response.data;
  }

  async getTimeStatus(): Promise<ApiResponse<any>> {
    const response = await this.client.get('/api/validation/time/status');
    return response.data;
  }
}

// Export singleton instance
export const apiService = new ApiService();
export default apiService;
