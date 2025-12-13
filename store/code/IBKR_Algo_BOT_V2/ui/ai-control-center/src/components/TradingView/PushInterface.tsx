import React, { useState, useEffect } from 'react';
import { apiService } from '../../services/api';
import type { TradingViewPush, TradingViewElements, ColorScheme, SizeStyle } from '../../types/models';

interface PushHistory {
  timestamp: string;
  symbol: string;
  elements_pushed: string[];
  status: 'success' | 'failed';
  error?: string;
}

export const PushInterface: React.FC = () => {
  const [selectedSymbol, setSelectedSymbol] = useState('AAPL');
  const [pushAll, setPushAll] = useState(false);
  const [autoSync, setAutoSync] = useState(false);
  const [syncInterval, setSyncInterval] = useState(60);
  const [pushHistory, setPushHistory] = useState<PushHistory[]>([]);
  const [isPushing, setIsPushing] = useState(false);

  // Push elements state
  const [elements, setElements] = useState<TradingViewElements>({
    trade_markers: true,
    ai_predictions: true,
    support_resistance: true,
    strategy_signals: true,
    risk_zones: true,
    volume_profile: false
  });

  // Color scheme state
  const [colorScheme, setColorScheme] = useState<ColorScheme>({
    buy_color: '#00FF00',
    sell_color: '#FF0000',
    support_color: '#00BFFF',
    resistance_color: '#FF6347',
    stop_loss_color: '#FF4500',
    target_color: '#32CD32'
  });

  // Size/style state
  const [sizeStyle, setSizeStyle] = useState<SizeStyle>({
    marker_size: 'medium',
    line_width: 2,
    label_size: 'medium'
  });

  const [labelVisibility, setLabelVisibility] = useState(true);

  const symbols = ['AAPL', 'TSLA', 'NVDA', 'MSFT', 'GOOGL', 'AMZN', 'META', 'SPY', 'QQQ', 'AMD'];

  useEffect(() => {
    fetchPushHistory();
    fetchAutoSyncStatus();
  }, []);

  const fetchPushHistory = async () => {
    try {
      const response = await apiService.getWebhookHistory();
      if (response.success && response.data) {
        setPushHistory(response.data.slice(0, 20));
      }
    } catch (error) {
      console.error('Error fetching push history:', error);
    }
  };

  const fetchAutoSyncStatus = async () => {
    try {
      const response = await apiService.getTradingViewStatus();
      if (response.success && response.data) {
        setAutoSync(response.data.auto_sync_enabled || false);
      }
    } catch (error) {
      console.error('Error fetching auto-sync status:', error);
    }
  };

  const handlePushToTradingView = async () => {
    setIsPushing(true);
    try {
      const pushData: TradingViewPush = {
        symbol: pushAll ? 'ALL' : selectedSymbol,
        push_all: pushAll,
        elements,
        config: {
          color_scheme: colorScheme,
          size_style: sizeStyle,
          label_visibility: labelVisibility
        }
      };

      const response = await apiService.pushToTradingView(pushData);

      if (response.success) {
        // Add to history
        const newHistoryItem: PushHistory = {
          timestamp: new Date().toISOString(),
          symbol: pushAll ? 'ALL' : selectedSymbol,
          elements_pushed: Object.entries(elements)
            .filter(([_, enabled]) => enabled)
            .map(([key]) => key),
          status: 'success'
        };
        setPushHistory([newHistoryItem, ...pushHistory.slice(0, 19)]);
      }
    } catch (error: any) {
      console.error('Error pushing to TradingView:', error);
      const errorHistoryItem: PushHistory = {
        timestamp: new Date().toISOString(),
        symbol: pushAll ? 'ALL' : selectedSymbol,
        elements_pushed: [],
        status: 'failed',
        error: error.message
      };
      setPushHistory([errorHistoryItem, ...pushHistory.slice(0, 19)]);
    } finally {
      setIsPushing(false);
    }
  };

  const handleToggleAutoSync = async () => {
    try {
      await apiService.setTradingViewAutoSync(!autoSync);
      setAutoSync(!autoSync);
    } catch (error) {
      console.error('Error toggling auto-sync:', error);
    }
  };

  const handleElementToggle = (key: keyof TradingViewElements) => {
    setElements(prev => ({
      ...prev,
      [key]: !prev[key]
    }));
  };

  const handleColorChange = (key: keyof ColorScheme, value: string) => {
    setColorScheme(prev => ({
      ...prev,
      [key]: value
    }));
  };

  const selectedElementsCount = Object.values(elements).filter(Boolean).length;

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold text-ibkr-text mb-1">TradingView Push Interface</h1>
          <p className="text-sm text-ibkr-text-secondary">
            Push AI predictions and analysis to TradingView charts
          </p>
        </div>
        <div className="flex items-center space-x-3">
          <div className="flex items-center space-x-2 bg-ibkr-surface border border-ibkr-border rounded px-3 py-2">
            <span className="text-xs text-ibkr-text-secondary">Auto-Sync:</span>
            <button
              onClick={handleToggleAutoSync}
              className={`px-3 py-1 text-xs font-medium rounded transition-colors ${
                autoSync
                  ? 'bg-ibkr-success text-white'
                  : 'bg-ibkr-bg text-ibkr-text border border-ibkr-border'
              }`}
            >
              {autoSync ? 'ON' : 'OFF'}
            </button>
          </div>
          <button
            onClick={handlePushToTradingView}
            disabled={isPushing || selectedElementsCount === 0}
            className="px-4 py-2 bg-ibkr-accent text-white text-sm font-medium rounded hover:bg-opacity-90 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isPushing ? '📤 Pushing...' : '📤 Push to TradingView'}
          </button>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-6">
        {/* Left Panel: Symbol & Elements */}
        <div className="space-y-6">
          {/* Symbol Selector */}
          <div className="bg-ibkr-surface border border-ibkr-border rounded p-4">
            <h3 className="text-sm font-semibold text-ibkr-text mb-3">Symbol Selection</h3>
            <div className="space-y-3">
              <div>
                <label className="flex items-center space-x-2 mb-3">
                  <input
                    type="checkbox"
                    checked={pushAll}
                    onChange={(e) => setPushAll(e.target.checked)}
                    className="rounded border-ibkr-border"
                  />
                  <span className="text-sm text-ibkr-text font-medium">Push to All Symbols</span>
                </label>
              </div>

              {!pushAll && (
                <div>
                  <label className="block text-xs text-ibkr-text-secondary mb-2">
                    Select Symbol
                  </label>
                  <select
                    value={selectedSymbol}
                    onChange={(e) => setSelectedSymbol(e.target.value)}
                    className="w-full px-3 py-2 text-sm bg-ibkr-bg border border-ibkr-border rounded text-ibkr-text focus:outline-none focus:border-ibkr-accent"
                  >
                    {symbols.map(symbol => (
                      <option key={symbol} value={symbol}>{symbol}</option>
                    ))}
                  </select>
                </div>
              )}

              {pushAll && (
                <div className="text-xs text-ibkr-warning bg-yellow-900 bg-opacity-20 border border-yellow-700 rounded p-2">
                  ⚠️ This will push to all {symbols.length} symbols
                </div>
              )}
            </div>
          </div>

          {/* Push Elements Checklist */}
          <div className="bg-ibkr-surface border border-ibkr-border rounded p-4">
            <h3 className="text-sm font-semibold text-ibkr-text mb-3">
              Push Elements ({selectedElementsCount}/6)
            </h3>
            <div className="space-y-2">
              <label className="flex items-center space-x-2 py-1.5 hover:bg-ibkr-bg px-2 rounded transition-colors cursor-pointer">
                <input
                  type="checkbox"
                  checked={elements.trade_markers}
                  onChange={() => handleElementToggle('trade_markers')}
                  className="rounded border-ibkr-border"
                />
                <span className="text-sm text-ibkr-text flex-1">Trade Markers</span>
                <span className="text-xs text-ibkr-text-secondary">📍</span>
              </label>

              <label className="flex items-center space-x-2 py-1.5 hover:bg-ibkr-bg px-2 rounded transition-colors cursor-pointer">
                <input
                  type="checkbox"
                  checked={elements.ai_predictions}
                  onChange={() => handleElementToggle('ai_predictions')}
                  className="rounded border-ibkr-border"
                />
                <span className="text-sm text-ibkr-text flex-1">AI Predictions</span>
                <span className="text-xs text-ibkr-text-secondary">🧠</span>
              </label>

              <label className="flex items-center space-x-2 py-1.5 hover:bg-ibkr-bg px-2 rounded transition-colors cursor-pointer">
                <input
                  type="checkbox"
                  checked={elements.support_resistance}
                  onChange={() => handleElementToggle('support_resistance')}
                  className="rounded border-ibkr-border"
                />
                <span className="text-sm text-ibkr-text flex-1">Support/Resistance</span>
                <span className="text-xs text-ibkr-text-secondary">📊</span>
              </label>

              <label className="flex items-center space-x-2 py-1.5 hover:bg-ibkr-bg px-2 rounded transition-colors cursor-pointer">
                <input
                  type="checkbox"
                  checked={elements.strategy_signals}
                  onChange={() => handleElementToggle('strategy_signals')}
                  className="rounded border-ibkr-border"
                />
                <span className="text-sm text-ibkr-text flex-1">Strategy Signals</span>
                <span className="text-xs text-ibkr-text-secondary">🎯</span>
              </label>

              <label className="flex items-center space-x-2 py-1.5 hover:bg-ibkr-bg px-2 rounded transition-colors cursor-pointer">
                <input
                  type="checkbox"
                  checked={elements.risk_zones}
                  onChange={() => handleElementToggle('risk_zones')}
                  className="rounded border-ibkr-border"
                />
                <span className="text-sm text-ibkr-text flex-1">Risk Zones</span>
                <span className="text-xs text-ibkr-text-secondary">⚠️</span>
              </label>

              <label className="flex items-center space-x-2 py-1.5 hover:bg-ibkr-bg px-2 rounded transition-colors cursor-pointer">
                <input
                  type="checkbox"
                  checked={elements.volume_profile}
                  onChange={() => handleElementToggle('volume_profile')}
                  className="rounded border-ibkr-border"
                />
                <span className="text-sm text-ibkr-text flex-1">Volume Profile</span>
                <span className="text-xs text-ibkr-text-secondary">📈</span>
              </label>
            </div>
          </div>
        </div>

        {/* Middle Panel: Customization */}
        <div className="space-y-6">
          {/* Color Scheme */}
          <div className="bg-ibkr-surface border border-ibkr-border rounded p-4">
            <h3 className="text-sm font-semibold text-ibkr-text mb-3">Color Scheme</h3>
            <div className="space-y-3">
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-xs text-ibkr-text-secondary mb-1">Buy Signals</label>
                  <div className="flex items-center space-x-2">
                    <input
                      type="color"
                      value={colorScheme.buy_color}
                      onChange={(e) => handleColorChange('buy_color', e.target.value)}
                      className="h-8 w-12 rounded border border-ibkr-border cursor-pointer"
                    />
                    <input
                      type="text"
                      value={colorScheme.buy_color}
                      onChange={(e) => handleColorChange('buy_color', e.target.value)}
                      className="flex-1 px-2 py-1 text-xs bg-ibkr-bg border border-ibkr-border rounded text-ibkr-text font-mono"
                    />
                  </div>
                </div>

                <div>
                  <label className="block text-xs text-ibkr-text-secondary mb-1">Sell Signals</label>
                  <div className="flex items-center space-x-2">
                    <input
                      type="color"
                      value={colorScheme.sell_color}
                      onChange={(e) => handleColorChange('sell_color', e.target.value)}
                      className="h-8 w-12 rounded border border-ibkr-border cursor-pointer"
                    />
                    <input
                      type="text"
                      value={colorScheme.sell_color}
                      onChange={(e) => handleColorChange('sell_color', e.target.value)}
                      className="flex-1 px-2 py-1 text-xs bg-ibkr-bg border border-ibkr-border rounded text-ibkr-text font-mono"
                    />
                  </div>
                </div>

                <div>
                  <label className="block text-xs text-ibkr-text-secondary mb-1">Support</label>
                  <div className="flex items-center space-x-2">
                    <input
                      type="color"
                      value={colorScheme.support_color}
                      onChange={(e) => handleColorChange('support_color', e.target.value)}
                      className="h-8 w-12 rounded border border-ibkr-border cursor-pointer"
                    />
                    <input
                      type="text"
                      value={colorScheme.support_color}
                      onChange={(e) => handleColorChange('support_color', e.target.value)}
                      className="flex-1 px-2 py-1 text-xs bg-ibkr-bg border border-ibkr-border rounded text-ibkr-text font-mono"
                    />
                  </div>
                </div>

                <div>
                  <label className="block text-xs text-ibkr-text-secondary mb-1">Resistance</label>
                  <div className="flex items-center space-x-2">
                    <input
                      type="color"
                      value={colorScheme.resistance_color}
                      onChange={(e) => handleColorChange('resistance_color', e.target.value)}
                      className="h-8 w-12 rounded border border-ibkr-border cursor-pointer"
                    />
                    <input
                      type="text"
                      value={colorScheme.resistance_color}
                      onChange={(e) => handleColorChange('resistance_color', e.target.value)}
                      className="flex-1 px-2 py-1 text-xs bg-ibkr-bg border border-ibkr-border rounded text-ibkr-text font-mono"
                    />
                  </div>
                </div>

                <div>
                  <label className="block text-xs text-ibkr-text-secondary mb-1">Stop Loss</label>
                  <div className="flex items-center space-x-2">
                    <input
                      type="color"
                      value={colorScheme.stop_loss_color}
                      onChange={(e) => handleColorChange('stop_loss_color', e.target.value)}
                      className="h-8 w-12 rounded border border-ibkr-border cursor-pointer"
                    />
                    <input
                      type="text"
                      value={colorScheme.stop_loss_color}
                      onChange={(e) => handleColorChange('stop_loss_color', e.target.value)}
                      className="flex-1 px-2 py-1 text-xs bg-ibkr-bg border border-ibkr-border rounded text-ibkr-text font-mono"
                    />
                  </div>
                </div>

                <div>
                  <label className="block text-xs text-ibkr-text-secondary mb-1">Target</label>
                  <div className="flex items-center space-x-2">
                    <input
                      type="color"
                      value={colorScheme.target_color}
                      onChange={(e) => handleColorChange('target_color', e.target.value)}
                      className="h-8 w-12 rounded border border-ibkr-border cursor-pointer"
                    />
                    <input
                      type="text"
                      value={colorScheme.target_color}
                      onChange={(e) => handleColorChange('target_color', e.target.value)}
                      className="flex-1 px-2 py-1 text-xs bg-ibkr-bg border border-ibkr-border rounded text-ibkr-text font-mono"
                    />
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Size & Style */}
          <div className="bg-ibkr-surface border border-ibkr-border rounded p-4">
            <h3 className="text-sm font-semibold text-ibkr-text mb-3">Size & Style</h3>
            <div className="space-y-3">
              <div>
                <label className="block text-xs text-ibkr-text-secondary mb-1">Marker Size</label>
                <select
                  value={sizeStyle.marker_size}
                  onChange={(e) => setSizeStyle(prev => ({
                    ...prev,
                    marker_size: e.target.value as 'small' | 'medium' | 'large'
                  }))}
                  className="w-full px-3 py-2 text-sm bg-ibkr-bg border border-ibkr-border rounded text-ibkr-text focus:outline-none focus:border-ibkr-accent"
                >
                  <option value="small">Small</option>
                  <option value="medium">Medium</option>
                  <option value="large">Large</option>
                </select>
              </div>

              <div>
                <label className="block text-xs text-ibkr-text-secondary mb-1">
                  Line Width: {sizeStyle.line_width}px
                </label>
                <input
                  type="range"
                  min="1"
                  max="5"
                  step="1"
                  value={sizeStyle.line_width}
                  onChange={(e) => setSizeStyle(prev => ({
                    ...prev,
                    line_width: parseInt(e.target.value)
                  }))}
                  className="w-full"
                />
                <div className="flex justify-between text-xs text-ibkr-text-secondary mt-1">
                  <span>1px</span>
                  <span>5px</span>
                </div>
              </div>

              <div>
                <label className="block text-xs text-ibkr-text-secondary mb-1">Label Size</label>
                <select
                  value={sizeStyle.label_size}
                  onChange={(e) => setSizeStyle(prev => ({
                    ...prev,
                    label_size: e.target.value as 'small' | 'medium' | 'large'
                  }))}
                  className="w-full px-3 py-2 text-sm bg-ibkr-bg border border-ibkr-border rounded text-ibkr-text focus:outline-none focus:border-ibkr-accent"
                >
                  <option value="small">Small</option>
                  <option value="medium">Medium</option>
                  <option value="large">Large</option>
                </select>
              </div>

              <div>
                <label className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    checked={labelVisibility}
                    onChange={(e) => setLabelVisibility(e.target.checked)}
                    className="rounded border-ibkr-border"
                  />
                  <span className="text-sm text-ibkr-text">Show Labels</span>
                </label>
              </div>
            </div>
          </div>

          {/* Auto-Sync Configuration */}
          {autoSync && (
            <div className="bg-ibkr-surface border border-ibkr-border rounded p-4">
              <h3 className="text-sm font-semibold text-ibkr-text mb-3">Auto-Sync Settings</h3>
              <div>
                <label className="block text-xs text-ibkr-text-secondary mb-1">
                  Sync Interval: {syncInterval} seconds
                </label>
                <input
                  type="range"
                  min="15"
                  max="300"
                  step="15"
                  value={syncInterval}
                  onChange={(e) => setSyncInterval(parseInt(e.target.value))}
                  className="w-full"
                />
                <div className="flex justify-between text-xs text-ibkr-text-secondary mt-1">
                  <span>15s</span>
                  <span>5min</span>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Right Panel: Preview & History */}
        <div className="space-y-6">
          {/* Preview */}
          <div className="bg-ibkr-surface border border-ibkr-border rounded p-4">
            <h3 className="text-sm font-semibold text-ibkr-text mb-3">Chart Preview</h3>
            <div className="bg-ibkr-bg border border-ibkr-border rounded p-4 h-64 flex items-center justify-center">
              <div className="text-center space-y-2">
                <div className="text-4xl">📊</div>
                <p className="text-sm text-ibkr-text-secondary">
                  {selectedElementsCount === 0
                    ? 'Select elements to preview'
                    : `${selectedElementsCount} element(s) selected`
                  }
                </p>
                {selectedElementsCount > 0 && (
                  <div className="text-xs text-ibkr-text-secondary space-y-1">
                    {elements.trade_markers && <div>✓ Trade Markers</div>}
                    {elements.ai_predictions && <div>✓ AI Predictions</div>}
                    {elements.support_resistance && <div>✓ Support/Resistance</div>}
                    {elements.strategy_signals && <div>✓ Strategy Signals</div>}
                    {elements.risk_zones && <div>✓ Risk Zones</div>}
                    {elements.volume_profile && <div>✓ Volume Profile</div>}
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Push History */}
          <div className="bg-ibkr-surface border border-ibkr-border rounded p-4">
            <h3 className="text-sm font-semibold text-ibkr-text mb-3">
              Push History ({pushHistory.length})
            </h3>
            <div className="space-y-2 max-h-80 overflow-y-auto">
              {pushHistory.length === 0 ? (
                <div className="text-center py-8 text-xs text-ibkr-text-secondary">
                  No push history yet
                </div>
              ) : (
                pushHistory.map((item, index) => (
                  <div
                    key={index}
                    className={`p-3 rounded border ${
                      item.status === 'success'
                        ? 'bg-green-900 bg-opacity-10 border-green-700'
                        : 'bg-red-900 bg-opacity-10 border-red-700'
                    }`}
                  >
                    <div className="flex items-start justify-between mb-1">
                      <div className="flex items-center space-x-2">
                        <span className={`text-xs font-semibold ${
                          item.status === 'success' ? 'text-green-400' : 'text-red-400'
                        }`}>
                          {item.symbol}
                        </span>
                        <span className={`px-1.5 py-0.5 rounded text-xs ${
                          item.status === 'success'
                            ? 'bg-green-700 text-green-100'
                            : 'bg-red-700 text-red-100'
                        }`}>
                          {item.status}
                        </span>
                      </div>
                      <span className="text-xs text-ibkr-text-secondary">
                        {new Date(item.timestamp).toLocaleTimeString()}
                      </span>
                    </div>
                    {item.status === 'success' && item.elements_pushed.length > 0 && (
                      <div className="text-xs text-ibkr-text-secondary">
                        {item.elements_pushed.length} element(s) pushed
                      </div>
                    )}
                    {item.status === 'failed' && item.error && (
                      <div className="text-xs text-red-400 mt-1">
                        {item.error}
                      </div>
                    )}
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PushInterface;
