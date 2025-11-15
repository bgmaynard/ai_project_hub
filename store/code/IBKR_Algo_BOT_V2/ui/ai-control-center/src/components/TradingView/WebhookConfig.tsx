import React, { useState, useEffect } from 'react';
import apiService from '../../services/api';

interface WebhookSettings {
  webhook_url: string;
  api_key: string;
  ip_whitelist: string[];
  enabled: boolean;
}

interface AlertFilter {
  symbols: string[];
  signal_types: ('BUY' | 'SELL' | 'ANY')[];
  min_confidence: number;
}

interface WebhookHistoryItem {
  id: string;
  timestamp: string;
  symbol: string;
  signal: 'BUY' | 'SELL';
  price: number;
  confidence: number;
  status: 'success' | 'failed' | 'pending';
  response_time_ms: number;
  error?: string;
}

export const WebhookConfig: React.FC = () => {
  // State Management
  const [settings, setSettings] = useState<WebhookSettings>({
    webhook_url: 'https://api.tradingview.com/webhooks/v1/your-webhook-id',
    api_key: '',
    ip_whitelist: [],
    enabled: true
  });

  const [filters, setFilters] = useState<AlertFilter>({
    symbols: ['AAPL', 'MSFT', 'GOOGL'],
    signal_types: ['BUY', 'SELL'],
    min_confidence: 0.75
  });

  const [history, setHistory] = useState<WebhookHistoryItem[]>([]);
  const [newIp, setNewIp] = useState('');
  const [newSymbol, setNewSymbol] = useState('');
  const [copied, setCopied] = useState(false);
  const [testing, setTesting] = useState(false);
  const [testResult, setTestResult] = useState<{status: 'success' | 'error', message: string} | null>(null);
  const [regeneratingKey, setRegeneratingKey] = useState(false);
  const [loading, setLoading] = useState(true);

  // Load Initial Data
  useEffect(() => {
    loadWebhookSettings();
    loadWebhookHistory();
  }, []);

  const loadWebhookSettings = async () => {
    setLoading(true);
    try {
      const response = await apiService.getWebhookSettings();
      if (response.data) {
        setSettings(response.data.settings);
        setFilters(response.data.filters);
      }
    } catch (error) {
      console.error('Failed to load webhook settings:', error);
    } finally {
      setLoading(false);
    }
  };

  const loadWebhookHistory = async () => {
    try {
      const response = await apiService.getWebhookHistory();
      if (response.data) {
        setHistory(response.data);
      }
    } catch (error) {
      console.error('Failed to load webhook history:', error);
    }
  };

  // Webhook URL Operations
  const copyWebhookUrl = () => {
    navigator.clipboard.writeText(settings.webhook_url);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  // API Key Management
  const regenerateApiKey = async () => {
    setRegeneratingKey(true);
    try {
      const response = await apiService.regenerateWebhookApiKey();
      if (response.data) {
        setSettings({ ...settings, api_key: response.data.api_key });
      }
    } catch (error) {
      console.error('Failed to regenerate API key:', error);
    } finally {
      setRegeneratingKey(false);
    }
  };

  const toggleWebhook = async () => {
    const newEnabled = !settings.enabled;
    try {
      await apiService.updateWebhookSettings({ ...settings, enabled: newEnabled });
      setSettings({ ...settings, enabled: newEnabled });
    } catch (error) {
      console.error('Failed to toggle webhook:', error);
    }
  };

  // IP Whitelist Management
  const addIpAddress = () => {
    if (newIp && !settings.ip_whitelist.includes(newIp)) {
      const updatedWhitelist = [...settings.ip_whitelist, newIp];
      setSettings({ ...settings, ip_whitelist: updatedWhitelist });
      setNewIp('');
      saveSettings({ ...settings, ip_whitelist: updatedWhitelist });
    }
  };

  const removeIpAddress = (ip: string) => {
    const updatedWhitelist = settings.ip_whitelist.filter(item => item !== ip);
    setSettings({ ...settings, ip_whitelist: updatedWhitelist });
    saveSettings({ ...settings, ip_whitelist: updatedWhitelist });
  };

  // Symbol Filter Management
  const addSymbol = () => {
    if (newSymbol && !filters.symbols.includes(newSymbol.toUpperCase())) {
      const updatedSymbols = [...filters.symbols, newSymbol.toUpperCase()];
      setFilters({ ...filters, symbols: updatedSymbols });
      setNewSymbol('');
      saveFilters({ ...filters, symbols: updatedSymbols });
    }
  };

  const removeSymbol = (symbol: string) => {
    const updatedSymbols = filters.symbols.filter(s => s !== symbol);
    setFilters({ ...filters, symbols: updatedSymbols });
    saveFilters({ ...filters, symbols: updatedSymbols });
  };

  const toggleSignalType = (signalType: 'BUY' | 'SELL' | 'ANY') => {
    const updatedTypes = filters.signal_types.includes(signalType)
      ? filters.signal_types.filter(t => t !== signalType)
      : [...filters.signal_types, signalType];
    setFilters({ ...filters, signal_types: updatedTypes });
    saveFilters({ ...filters, signal_types: updatedTypes });
  };

  // Save Operations
  const saveSettings = async (newSettings: WebhookSettings) => {
    try {
      await apiService.updateWebhookSettings(newSettings);
    } catch (error) {
      console.error('Failed to save settings:', error);
    }
  };

  const saveFilters = async (newFilters: AlertFilter) => {
    try {
      await apiService.updateWebhookFilters(newFilters);
    } catch (error) {
      console.error('Failed to save filters:', error);
    }
  };

  // Test Webhook
  const testWebhook = async () => {
    setTesting(true);
    setTestResult(null);
    try {
      const response = await apiService.testWebhook();
      if (response.success) {
        setTestResult({
          status: 'success',
          message: `Webhook test successful! Response time: ${response.data?.response_time_ms}ms`
        });
      } else {
        setTestResult({
          status: 'error',
          message: response.error || 'Webhook test failed'
        });
      }
    } catch (error: any) {
      setTestResult({
        status: 'error',
        message: error.message || 'Failed to send test webhook'
      });
    } finally {
      setTesting(false);
      loadWebhookHistory(); // Refresh history after test
    }
  };

  // Format timestamp
  const formatTimestamp = (timestamp: string) => {
    return new Date(timestamp).toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    });
  };

  if (loading) {
    return (
      <div className="px-6 py-12 text-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-ibkr-accent mx-auto"></div>
        <p className="mt-4 text-sm text-ibkr-text-secondary">Loading webhook configuration...</p>
      </div>
    );
  }

  return (
    <div className="px-6 space-y-6">
      {/* Webhook URL Section */}
      <div className="bg-ibkr-surface border border-ibkr-border rounded-lg p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-base font-semibold text-ibkr-text">Webhook URL</h2>
          <button
            onClick={toggleWebhook}
            className={`px-4 py-1.5 text-xs font-medium rounded transition-colors ${
              settings.enabled
                ? 'bg-green-500/20 text-green-400 hover:bg-green-500/30'
                : 'bg-red-500/20 text-red-400 hover:bg-red-500/30'
            }`}
          >
            {settings.enabled ? '✓ Enabled' : '✗ Disabled'}
          </button>
        </div>

        <div className="flex items-center space-x-2">
          <input
            type="text"
            value={settings.webhook_url}
            readOnly
            className="flex-1 bg-ibkr-bg border border-ibkr-border rounded px-3 py-2 text-xs text-ibkr-text font-mono"
          />
          <button
            onClick={copyWebhookUrl}
            className="px-4 py-2 bg-ibkr-accent hover:bg-ibkr-accent-hover text-white text-xs rounded transition-colors"
          >
            {copied ? '✓ Copied!' : '📋 Copy'}
          </button>
        </div>
        <p className="mt-2 text-xs text-ibkr-text-secondary">
          Use this URL in your TradingView alerts to send signals to the AI trading system.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Authentication Settings */}
        <div className="bg-ibkr-surface border border-ibkr-border rounded-lg p-6">
          <h2 className="text-base font-semibold text-ibkr-text mb-4">Authentication</h2>

          {/* API Key */}
          <div className="mb-4">
            <label className="block text-xs font-medium text-ibkr-text-secondary mb-2">
              API Key
            </label>
            <div className="flex items-center space-x-2">
              <input
                type="password"
                value={settings.api_key || '••••••••••••••••'}
                readOnly
                className="flex-1 bg-ibkr-bg border border-ibkr-border rounded px-3 py-2 text-xs text-ibkr-text font-mono"
              />
              <button
                onClick={regenerateApiKey}
                disabled={regeneratingKey}
                className="px-4 py-2 bg-orange-500 hover:bg-orange-600 text-white text-xs rounded transition-colors disabled:opacity-50"
              >
                {regeneratingKey ? '⟳ Regenerating...' : '🔄 Regenerate'}
              </button>
            </div>
            <p className="mt-1 text-xs text-orange-400">
              ⚠️ Regenerating will invalidate the current key
            </p>
          </div>

          {/* IP Whitelist */}
          <div>
            <label className="block text-xs font-medium text-ibkr-text-secondary mb-2">
              IP Whitelist
            </label>
            <div className="flex items-center space-x-2 mb-2">
              <input
                type="text"
                value={newIp}
                onChange={(e) => setNewIp(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && addIpAddress()}
                placeholder="Enter IP address (e.g., 192.168.1.1)"
                className="flex-1 bg-ibkr-bg border border-ibkr-border rounded px-3 py-2 text-xs text-ibkr-text placeholder-gray-500"
              />
              <button
                onClick={addIpAddress}
                className="px-4 py-2 bg-ibkr-accent hover:bg-ibkr-accent-hover text-white text-xs rounded transition-colors"
              >
                + Add
              </button>
            </div>

            {settings.ip_whitelist.length === 0 ? (
              <p className="text-xs text-ibkr-text-secondary italic">
                No IP restrictions - accepts requests from any IP
              </p>
            ) : (
              <div className="space-y-1.5 max-h-32 overflow-y-auto">
                {settings.ip_whitelist.map((ip) => (
                  <div
                    key={ip}
                    className="flex items-center justify-between bg-ibkr-bg border border-ibkr-border rounded px-3 py-1.5"
                  >
                    <span className="text-xs text-ibkr-text font-mono">{ip}</span>
                    <button
                      onClick={() => removeIpAddress(ip)}
                      className="text-red-400 hover:text-red-300 text-xs"
                    >
                      ✕
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Alert Configuration */}
        <div className="bg-ibkr-surface border border-ibkr-border rounded-lg p-6">
          <h2 className="text-base font-semibold text-ibkr-text mb-4">Alert Filters</h2>

          {/* Symbol Filters */}
          <div className="mb-4">
            <label className="block text-xs font-medium text-ibkr-text-secondary mb-2">
              Symbol Filters
            </label>
            <div className="flex items-center space-x-2 mb-2">
              <input
                type="text"
                value={newSymbol}
                onChange={(e) => setNewSymbol(e.target.value.toUpperCase())}
                onKeyPress={(e) => e.key === 'Enter' && addSymbol()}
                placeholder="Enter symbol (e.g., AAPL)"
                className="flex-1 bg-ibkr-bg border border-ibkr-border rounded px-3 py-2 text-xs text-ibkr-text placeholder-gray-500"
              />
              <button
                onClick={addSymbol}
                className="px-4 py-2 bg-ibkr-accent hover:bg-ibkr-accent-hover text-white text-xs rounded transition-colors"
              >
                + Add
              </button>
            </div>

            <div className="flex flex-wrap gap-1.5">
              {filters.symbols.map((symbol) => (
                <span
                  key={symbol}
                  className="inline-flex items-center bg-ibkr-accent/20 text-ibkr-accent px-2 py-1 rounded text-xs"
                >
                  {symbol}
                  <button
                    onClick={() => removeSymbol(symbol)}
                    className="ml-1.5 text-ibkr-accent hover:text-white"
                  >
                    ✕
                  </button>
                </span>
              ))}
            </div>
          </div>

          {/* Signal Type Filters */}
          <div className="mb-4">
            <label className="block text-xs font-medium text-ibkr-text-secondary mb-2">
              Signal Types
            </label>
            <div className="flex space-x-2">
              {(['BUY', 'SELL', 'ANY'] as const).map((type) => (
                <button
                  key={type}
                  onClick={() => toggleSignalType(type)}
                  className={`flex-1 px-3 py-2 text-xs font-medium rounded transition-colors ${
                    filters.signal_types.includes(type)
                      ? type === 'BUY'
                        ? 'bg-green-500/20 text-green-400 border border-green-500/50'
                        : type === 'SELL'
                        ? 'bg-red-500/20 text-red-400 border border-red-500/50'
                        : 'bg-blue-500/20 text-blue-400 border border-blue-500/50'
                      : 'bg-ibkr-bg text-ibkr-text-secondary border border-ibkr-border'
                  }`}
                >
                  {type}
                </button>
              ))}
            </div>
          </div>

          {/* Confidence Threshold */}
          <div>
            <label className="block text-xs font-medium text-ibkr-text-secondary mb-2">
              Minimum Confidence: {(filters.min_confidence * 100).toFixed(0)}%
            </label>
            <input
              type="range"
              min="0"
              max="100"
              value={filters.min_confidence * 100}
              onChange={(e) => {
                const newFilters = { ...filters, min_confidence: parseInt(e.target.value) / 100 };
                setFilters(newFilters);
                saveFilters(newFilters);
              }}
              className="w-full h-2 bg-ibkr-bg rounded-lg appearance-none cursor-pointer accent-ibkr-accent"
            />
            <div className="flex justify-between text-xs text-ibkr-text-secondary mt-1">
              <span>0%</span>
              <span>50%</span>
              <span>100%</span>
            </div>
          </div>
        </div>
      </div>

      {/* Test Webhook */}
      <div className="bg-ibkr-surface border border-ibkr-border rounded-lg p-6">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-base font-semibold text-ibkr-text">Test Webhook</h2>
            <p className="text-xs text-ibkr-text-secondary mt-1">
              Send a test signal to verify your webhook configuration
            </p>
          </div>
          <button
            onClick={testWebhook}
            disabled={testing || !settings.enabled}
            className="px-6 py-2 bg-ibkr-accent hover:bg-ibkr-accent-hover text-white text-sm rounded transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {testing ? '⟳ Testing...' : '🧪 Send Test'}
          </button>
        </div>

        {testResult && (
          <div
            className={`p-3 rounded border ${
              testResult.status === 'success'
                ? 'bg-green-500/10 border-green-500/50 text-green-400'
                : 'bg-red-500/10 border-red-500/50 text-red-400'
            }`}
          >
            <p className="text-xs">{testResult.message}</p>
          </div>
        )}
      </div>

      {/* Webhook History */}
      <div className="bg-ibkr-surface border border-ibkr-border rounded-lg p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-base font-semibold text-ibkr-text">
            Webhook History
            <span className="ml-2 text-xs text-ibkr-text-secondary font-normal">
              (Last 100 webhooks)
            </span>
          </h2>
          <button
            onClick={loadWebhookHistory}
            className="text-xs text-ibkr-accent hover:text-ibkr-accent-hover"
          >
            🔄 Refresh
          </button>
        </div>

        <div className="overflow-hidden border border-ibkr-border rounded">
          <div className="overflow-x-auto max-h-96">
            <table className="w-full text-xs">
              <thead className="bg-ibkr-bg sticky top-0">
                <tr>
                  <th className="px-3 py-2 text-left text-ibkr-text-secondary font-medium">Time</th>
                  <th className="px-3 py-2 text-left text-ibkr-text-secondary font-medium">Symbol</th>
                  <th className="px-3 py-2 text-left text-ibkr-text-secondary font-medium">Signal</th>
                  <th className="px-3 py-2 text-right text-ibkr-text-secondary font-medium">Price</th>
                  <th className="px-3 py-2 text-right text-ibkr-text-secondary font-medium">Confidence</th>
                  <th className="px-3 py-2 text-center text-ibkr-text-secondary font-medium">Status</th>
                  <th className="px-3 py-2 text-right text-ibkr-text-secondary font-medium">Response Time</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-ibkr-border">
                {history.length === 0 ? (
                  <tr>
                    <td colSpan={7} className="px-3 py-8 text-center text-ibkr-text-secondary">
                      No webhook history yet. Send a test webhook to get started.
                    </td>
                  </tr>
                ) : (
                  history.map((item) => (
                    <tr key={item.id} className="hover:bg-ibkr-bg transition-colors">
                      <td className="px-3 py-2 text-ibkr-text whitespace-nowrap">
                        {formatTimestamp(item.timestamp)}
                      </td>
                      <td className="px-3 py-2 text-ibkr-text font-medium">{item.symbol}</td>
                      <td className="px-3 py-2">
                        <span
                          className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${
                            item.signal === 'BUY'
                              ? 'bg-green-500/20 text-green-400'
                              : 'bg-red-500/20 text-red-400'
                          }`}
                        >
                          {item.signal}
                        </span>
                      </td>
                      <td className="px-3 py-2 text-right text-ibkr-text font-mono">
                        ${item.price.toFixed(2)}
                      </td>
                      <td className="px-3 py-2 text-right text-ibkr-text">
                        {(item.confidence * 100).toFixed(0)}%
                      </td>
                      <td className="px-3 py-2 text-center">
                        <span
                          className={`inline-flex items-center px-2 py-0.5 rounded text-xs ${
                            item.status === 'success'
                              ? 'bg-green-500/20 text-green-400'
                              : item.status === 'failed'
                              ? 'bg-red-500/20 text-red-400'
                              : 'bg-yellow-500/20 text-yellow-400'
                          }`}
                        >
                          {item.status === 'success' ? '✓' : item.status === 'failed' ? '✗' : '⟳'}
                        </span>
                      </td>
                      <td className="px-3 py-2 text-right text-ibkr-text-secondary">
                        {item.response_time_ms}ms
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
};

export default WebhookConfig;
