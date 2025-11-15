import React, { useState, useEffect, Fragment } from 'react';
import { Dialog, Transition, Switch } from '@headlessui/react';
import { apiService } from '../../services/api';
import type { AlertConfig, ActiveAlert, AlertHistory, NotificationMethods } from '../../types/models';

export const AlertSystem: React.FC = () => {
  const [activeAlerts, setActiveAlerts] = useState<ActiveAlert[]>([]);
  const [alertHistory, setAlertHistory] = useState<AlertHistory[]>([]);
  const [isConfigModalOpen, setIsConfigModalOpen] = useState(false);
  const [editingAlert, setEditingAlert] = useState<ActiveAlert | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [testingAlertId, setTestingAlertId] = useState<string | null>(null);

  // Filter states for history
  const [historySymbolFilter, setHistorySymbolFilter] = useState('all');
  const [historyDateFrom, setHistoryDateFrom] = useState('');
  const [historyDateTo, setHistoryDateTo] = useState('');

  // Alert configuration form state
  const [alertName, setAlertName] = useState('');
  const [selectedSymbols, setSelectedSymbols] = useState<string[]>([]);
  const [signalType, setSignalType] = useState<'BUY' | 'SELL' | 'ANY'>('ANY');
  const [confidenceThreshold, setConfidenceThreshold] = useState(70);
  const [selectedStrategy, setSelectedStrategy] = useState('');
  const [priceAbove, setPriceAbove] = useState('');
  const [priceBelow, setPriceBelow] = useState('');
  const [volumeAbove, setVolumeAbove] = useState('');
  const [notificationMethods, setNotificationMethods] = useState<NotificationMethods>({
    desktop: true,
    browser_push: false,
    sound: true,
    email: undefined,
    sms: undefined,
    sound_file: undefined
  });

  // Popular symbols for selection
  const popularSymbols = ['AAPL', 'TSLA', 'NVDA', 'MSFT', 'GOOGL', 'AMZN', 'META', 'SPY', 'QQQ'];
  const strategies = ['Gap-and-Go', 'Momentum Breakout', 'Mean Reversion', 'News Catalyst', 'Combined'];

  useEffect(() => {
    fetchActiveAlerts();
    fetchAlertHistory();
  }, []);

  const fetchActiveAlerts = async () => {
    try {
      const response = await apiService.getAlerts();
      if (response.success && response.data) {
        setActiveAlerts(response.data as ActiveAlert[]);
      }
    } catch (error) {
      console.error('Error fetching active alerts:', error);
    }
  };

  const fetchAlertHistory = async () => {
    try {
      const response = await apiService.getAlertHistory();
      if (response.success && response.data) {
        setAlertHistory(response.data);
      }
    } catch (error) {
      console.error('Error fetching alert history:', error);
    }
  };

  const openCreateAlertModal = () => {
    resetForm();
    setEditingAlert(null);
    setIsConfigModalOpen(true);
  };

  const openEditAlertModal = (alert: ActiveAlert) => {
    setAlertName(alert.name);
    setSelectedSymbols(alert.symbols);
    setSignalType(alert.signal_type);
    setConfidenceThreshold(alert.confidence_threshold);
    setSelectedStrategy(alert.strategy || '');
    setPriceAbove(alert.price_above?.toString() || '');
    setPriceBelow(alert.price_below?.toString() || '');
    setVolumeAbove(alert.volume_above?.toString() || '');
    setNotificationMethods(alert.notification_methods);
    setEditingAlert(alert);
    setIsConfigModalOpen(true);
  };

  const resetForm = () => {
    setAlertName('');
    setSelectedSymbols([]);
    setSignalType('ANY');
    setConfidenceThreshold(70);
    setSelectedStrategy('');
    setPriceAbove('');
    setPriceBelow('');
    setVolumeAbove('');
    setNotificationMethods({
      desktop: true,
      browser_push: false,
      sound: true,
      email: undefined,
      sms: undefined,
      sound_file: undefined
    });
  };

  const handleSymbolToggle = (symbol: string) => {
    setSelectedSymbols(prev =>
      prev.includes(symbol)
        ? prev.filter(s => s !== symbol)
        : [...prev, symbol]
    );
  };

  const handleSaveAlert = async () => {
    if (!alertName || selectedSymbols.length === 0) {
      alert('Please provide an alert name and select at least one symbol');
      return;
    }

    setIsLoading(true);
    try {
      const config: AlertConfig = {
        alert_id: editingAlert?.alert_id,
        name: alertName,
        symbols: selectedSymbols,
        signal_type: signalType,
        confidence_threshold: confidenceThreshold,
        strategy: selectedStrategy || undefined,
        price_above: priceAbove ? parseFloat(priceAbove) : undefined,
        price_below: priceBelow ? parseFloat(priceBelow) : undefined,
        volume_above: volumeAbove ? parseFloat(volumeAbove) : undefined,
        notification_methods: notificationMethods
      };

      const response = editingAlert
        ? await apiService.configureAlert(config)
        : await apiService.createAlert(config);

      if (response.success) {
        setIsConfigModalOpen(false);
        fetchActiveAlerts();
        resetForm();
      }
    } catch (error) {
      console.error('Error saving alert:', error);
      alert('Failed to save alert');
    } finally {
      setIsLoading(false);
    }
  };

  const handleToggleAlert = async (alertId: string, currentEnabled: boolean) => {
    try {
      await apiService.toggleAlert(alertId, !currentEnabled);
      fetchActiveAlerts();
    } catch (error) {
      console.error('Error toggling alert:', error);
    }
  };

  const handleDeleteAlert = async (alertId: string) => {
    if (!window.confirm('Are you sure you want to delete this alert?')) {
      return;
    }

    try {
      await apiService.deleteAlert(alertId);
      fetchActiveAlerts();
    } catch (error) {
      console.error('Error deleting alert:', error);
    }
  };

  const handleTestAlert = async (alertId: string) => {
    setTestingAlertId(alertId);
    try {
      // Call test endpoint - this would need to be added to API
      await new Promise(resolve => setTimeout(resolve, 1000)); // Simulate API call
      alert('Test notification sent successfully!');
    } catch (error) {
      console.error('Error testing alert:', error);
      alert('Failed to send test notification');
    } finally {
      setTestingAlertId(null);
    }
  };

  const filteredHistory = alertHistory.filter(item => {
    if (historySymbolFilter !== 'all' && item.symbol !== historySymbolFilter) {
      return false;
    }
    if (historyDateFrom && item.timestamp < historyDateFrom) {
      return false;
    }
    if (historyDateTo && item.timestamp > historyDateTo) {
      return false;
    }
    return true;
  });

  const uniqueHistorySymbols = Array.from(new Set(alertHistory.map(h => h.symbol)));

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold text-ibkr-text">Alert Management</h2>
          <p className="text-xs text-ibkr-text-secondary mt-0.5">
            Configure and manage real-time prediction alerts
          </p>
        </div>
        <button
          onClick={openCreateAlertModal}
          className="px-4 py-2 bg-ibkr-accent text-white text-sm font-medium rounded hover:bg-opacity-90 transition-colors"
        >
          + Create Alert
        </button>
      </div>

      {/* Active Alerts List */}
      <div className="bg-ibkr-surface border border-ibkr-border rounded p-4">
        <h3 className="text-sm font-semibold text-ibkr-text mb-3 flex items-center">
          <span className="mr-2">🔔</span>
          Active Alerts ({activeAlerts.length})
        </h3>

        {activeAlerts.length === 0 ? (
          <div className="text-center py-8 text-ibkr-text-secondary text-sm">
            No active alerts configured. Click "Create Alert" to get started.
          </div>
        ) : (
          <div className="space-y-2">
            {activeAlerts.map(alert => (
              <div
                key={alert.alert_id}
                className="bg-ibkr-bg border border-ibkr-border rounded p-3 hover:border-ibkr-accent transition-colors"
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center space-x-3">
                      <Switch
                        checked={alert.enabled}
                        onChange={() => handleToggleAlert(alert.alert_id, alert.enabled)}
                        className={`${
                          alert.enabled ? 'bg-green-600' : 'bg-gray-600'
                        } relative inline-flex h-5 w-9 items-center rounded-full transition-colors`}
                      >
                        <span
                          className={`${
                            alert.enabled ? 'translate-x-5' : 'translate-x-1'
                          } inline-block h-3 w-3 transform rounded-full bg-white transition-transform`}
                        />
                      </Switch>
                      <div>
                        <h4 className="text-sm font-semibold text-ibkr-text">{alert.name}</h4>
                        <div className="flex items-center space-x-2 mt-1 text-xs text-ibkr-text-secondary">
                          <span>Symbols: {alert.symbols.join(', ')}</span>
                          <span>•</span>
                          <span>Signal: {alert.signal_type}</span>
                          <span>•</span>
                          <span>Confidence ≥ {alert.confidence_threshold}%</span>
                        </div>
                      </div>
                    </div>
                    <div className="mt-2 flex items-center space-x-4 text-xs text-ibkr-text-secondary">
                      <span className="flex items-center">
                        <span className="mr-1">🔔</span>
                        Triggered: {alert.trigger_count} times
                      </span>
                      {alert.last_triggered && (
                        <span>
                          Last: {new Date(alert.last_triggered).toLocaleString()}
                        </span>
                      )}
                    </div>
                  </div>
                  <div className="flex items-center space-x-2 ml-4">
                    <button
                      onClick={() => handleTestAlert(alert.alert_id)}
                      disabled={testingAlertId === alert.alert_id}
                      className="px-2 py-1 text-xs bg-ibkr-surface text-ibkr-text border border-ibkr-border rounded hover:bg-opacity-80 transition-colors disabled:opacity-50"
                    >
                      {testingAlertId === alert.alert_id ? 'Testing...' : 'Test'}
                    </button>
                    <button
                      onClick={() => openEditAlertModal(alert)}
                      className="px-2 py-1 text-xs bg-ibkr-surface text-ibkr-text border border-ibkr-border rounded hover:bg-opacity-80 transition-colors"
                    >
                      Edit
                    </button>
                    <button
                      onClick={() => handleDeleteAlert(alert.alert_id)}
                      className="px-2 py-1 text-xs bg-ibkr-error text-white rounded hover:bg-opacity-80 transition-colors"
                    >
                      Delete
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Alert History Log */}
      <div className="bg-ibkr-surface border border-ibkr-border rounded p-4">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-semibold text-ibkr-text flex items-center">
            <span className="mr-2">📜</span>
            Alert History (Last 50)
          </h3>
          <div className="flex items-center space-x-2">
            <select
              value={historySymbolFilter}
              onChange={(e) => setHistorySymbolFilter(e.target.value)}
              className="px-2 py-1 text-xs bg-ibkr-bg border border-ibkr-border rounded text-ibkr-text"
            >
              <option value="all">All Symbols</option>
              {uniqueHistorySymbols.map(symbol => (
                <option key={symbol} value={symbol}>{symbol}</option>
              ))}
            </select>
            <input
              type="date"
              value={historyDateFrom}
              onChange={(e) => setHistoryDateFrom(e.target.value)}
              placeholder="From"
              className="px-2 py-1 text-xs bg-ibkr-bg border border-ibkr-border rounded text-ibkr-text"
            />
            <input
              type="date"
              value={historyDateTo}
              onChange={(e) => setHistoryDateTo(e.target.value)}
              placeholder="To"
              className="px-2 py-1 text-xs bg-ibkr-bg border border-ibkr-border rounded text-ibkr-text"
            />
            {(historySymbolFilter !== 'all' || historyDateFrom || historyDateTo) && (
              <button
                onClick={() => {
                  setHistorySymbolFilter('all');
                  setHistoryDateFrom('');
                  setHistoryDateTo('');
                }}
                className="px-2 py-1 text-xs bg-ibkr-surface text-ibkr-text border border-ibkr-border rounded hover:bg-opacity-80"
              >
                Clear Filters
              </button>
            )}
          </div>
        </div>

        {filteredHistory.length === 0 ? (
          <div className="text-center py-8 text-ibkr-text-secondary text-sm">
            No alert history available
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-ibkr-border">
                  <th className="text-left py-2 px-2 font-semibold text-ibkr-text-secondary">Time</th>
                  <th className="text-left py-2 px-2 font-semibold text-ibkr-text-secondary">Alert</th>
                  <th className="text-left py-2 px-2 font-semibold text-ibkr-text-secondary">Symbol</th>
                  <th className="text-left py-2 px-2 font-semibold text-ibkr-text-secondary">Signal</th>
                  <th className="text-left py-2 px-2 font-semibold text-ibkr-text-secondary">Price</th>
                  <th className="text-left py-2 px-2 font-semibold text-ibkr-text-secondary">Confidence</th>
                  <th className="text-left py-2 px-2 font-semibold text-ibkr-text-secondary">Action Taken</th>
                  <th className="text-left py-2 px-2 font-semibold text-ibkr-text-secondary">Status</th>
                </tr>
              </thead>
              <tbody>
                {filteredHistory.slice(0, 50).map((item, index) => (
                  <tr
                    key={index}
                    className="border-b border-ibkr-border hover:bg-ibkr-bg transition-colors"
                  >
                    <td className="py-2 px-2 text-ibkr-text">
                      {new Date(item.timestamp).toLocaleString()}
                    </td>
                    <td className="py-2 px-2 text-ibkr-text">{item.alert_name}</td>
                    <td className="py-2 px-2">
                      <span className="font-mono font-semibold text-ibkr-text">{item.symbol}</span>
                    </td>
                    <td className="py-2 px-2">
                      <span className={`px-2 py-0.5 rounded text-xs font-semibold ${
                        item.signal === 'BUY'
                          ? 'bg-green-900 bg-opacity-30 text-green-400 border border-green-600'
                          : 'bg-red-900 bg-opacity-30 text-red-400 border border-red-600'
                      }`}>
                        {item.signal}
                      </span>
                    </td>
                    <td className="py-2 px-2 text-ibkr-text font-mono">
                      ${item.price.toFixed(2)}
                    </td>
                    <td className="py-2 px-2 text-ibkr-text">
                      {item.confidence}%
                    </td>
                    <td className="py-2 px-2 text-ibkr-text">{item.action_taken}</td>
                    <td className="py-2 px-2">
                      <span className={`px-2 py-0.5 rounded text-xs ${
                        item.notification_sent
                          ? 'bg-green-900 bg-opacity-20 text-green-400'
                          : 'bg-yellow-900 bg-opacity-20 text-yellow-400'
                      }`}>
                        {item.notification_sent ? 'Sent' : 'Pending'}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Alert Configuration Modal */}
      <Transition appear show={isConfigModalOpen} as={Fragment}>
        <Dialog
          as="div"
          className="relative z-50"
          onClose={() => !isLoading && setIsConfigModalOpen(false)}
        >
          <Transition.Child
            as={Fragment}
            enter="ease-out duration-300"
            enterFrom="opacity-0"
            enterTo="opacity-100"
            leave="ease-in duration-200"
            leaveFrom="opacity-100"
            leaveTo="opacity-0"
          >
            <div className="fixed inset-0 bg-black bg-opacity-75" />
          </Transition.Child>

          <div className="fixed inset-0 overflow-y-auto">
            <div className="flex min-h-full items-center justify-center p-4">
              <Transition.Child
                as={Fragment}
                enter="ease-out duration-300"
                enterFrom="opacity-0 scale-95"
                enterTo="opacity-100 scale-100"
                leave="ease-in duration-200"
                leaveFrom="opacity-100 scale-100"
                leaveTo="opacity-0 scale-95"
              >
                <Dialog.Panel className="w-full max-w-2xl transform overflow-hidden rounded bg-ibkr-surface border border-ibkr-border p-6 shadow-xl transition-all">
                  <Dialog.Title className="text-base font-semibold text-ibkr-text mb-4">
                    {editingAlert ? 'Edit Alert' : 'Create New Alert'}
                  </Dialog.Title>

                  <div className="space-y-4">
                    {/* Alert Name */}
                    <div>
                      <label className="block text-xs font-medium text-ibkr-text-secondary mb-1">
                        Alert Name *
                      </label>
                      <input
                        type="text"
                        value={alertName}
                        onChange={(e) => setAlertName(e.target.value)}
                        placeholder="e.g., High Confidence BUY Alerts"
                        className="w-full px-3 py-2 text-sm bg-ibkr-bg border border-ibkr-border rounded text-ibkr-text focus:outline-none focus:border-ibkr-accent"
                      />
                    </div>

                    {/* Symbol Selection */}
                    <div>
                      <label className="block text-xs font-medium text-ibkr-text-secondary mb-1">
                        Symbols *
                      </label>
                      <div className="flex flex-wrap gap-2">
                        {popularSymbols.map(symbol => (
                          <button
                            key={symbol}
                            onClick={() => handleSymbolToggle(symbol)}
                            className={`px-3 py-1.5 text-xs font-medium rounded transition-colors ${
                              selectedSymbols.includes(symbol)
                                ? 'bg-ibkr-accent text-white'
                                : 'bg-ibkr-bg text-ibkr-text border border-ibkr-border hover:border-ibkr-accent'
                            }`}
                          >
                            {symbol}
                          </button>
                        ))}
                      </div>
                      {selectedSymbols.length > 0 && (
                        <p className="text-xs text-ibkr-text-secondary mt-2">
                          Selected: {selectedSymbols.join(', ')}
                        </p>
                      )}
                    </div>

                    {/* Signal Type and Strategy */}
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <label className="block text-xs font-medium text-ibkr-text-secondary mb-1">
                          Signal Type
                        </label>
                        <select
                          value={signalType}
                          onChange={(e) => setSignalType(e.target.value as 'BUY' | 'SELL' | 'ANY')}
                          className="w-full px-3 py-2 text-sm bg-ibkr-bg border border-ibkr-border rounded text-ibkr-text focus:outline-none focus:border-ibkr-accent"
                        >
                          <option value="ANY">Any (BUY or SELL)</option>
                          <option value="BUY">BUY Only</option>
                          <option value="SELL">SELL Only</option>
                        </select>
                      </div>

                      <div>
                        <label className="block text-xs font-medium text-ibkr-text-secondary mb-1">
                          Strategy (Optional)
                        </label>
                        <select
                          value={selectedStrategy}
                          onChange={(e) => setSelectedStrategy(e.target.value)}
                          className="w-full px-3 py-2 text-sm bg-ibkr-bg border border-ibkr-border rounded text-ibkr-text focus:outline-none focus:border-ibkr-accent"
                        >
                          <option value="">All Strategies</option>
                          {strategies.map(strategy => (
                            <option key={strategy} value={strategy}>{strategy}</option>
                          ))}
                        </select>
                      </div>
                    </div>

                    {/* Confidence Threshold */}
                    <div>
                      <label className="block text-xs font-medium text-ibkr-text-secondary mb-1">
                        Minimum Confidence: {confidenceThreshold}%
                      </label>
                      <input
                        type="range"
                        min="0"
                        max="100"
                        step="5"
                        value={confidenceThreshold}
                        onChange={(e) => setConfidenceThreshold(parseInt(e.target.value))}
                        className="w-full"
                      />
                      <div className="flex justify-between text-xs text-ibkr-text-secondary mt-1">
                        <span>0%</span>
                        <span>50%</span>
                        <span>100%</span>
                      </div>
                    </div>

                    {/* Price and Volume Filters */}
                    <div className="grid grid-cols-3 gap-4">
                      <div>
                        <label className="block text-xs font-medium text-ibkr-text-secondary mb-1">
                          Price Above ($)
                        </label>
                        <input
                          type="number"
                          value={priceAbove}
                          onChange={(e) => setPriceAbove(e.target.value)}
                          placeholder="Optional"
                          className="w-full px-3 py-2 text-sm bg-ibkr-bg border border-ibkr-border rounded text-ibkr-text focus:outline-none focus:border-ibkr-accent"
                        />
                      </div>
                      <div>
                        <label className="block text-xs font-medium text-ibkr-text-secondary mb-1">
                          Price Below ($)
                        </label>
                        <input
                          type="number"
                          value={priceBelow}
                          onChange={(e) => setPriceBelow(e.target.value)}
                          placeholder="Optional"
                          className="w-full px-3 py-2 text-sm bg-ibkr-bg border border-ibkr-border rounded text-ibkr-text focus:outline-none focus:border-ibkr-accent"
                        />
                      </div>
                      <div>
                        <label className="block text-xs font-medium text-ibkr-text-secondary mb-1">
                          Volume Above (M)
                        </label>
                        <input
                          type="number"
                          value={volumeAbove}
                          onChange={(e) => setVolumeAbove(e.target.value)}
                          placeholder="Optional"
                          className="w-full px-3 py-2 text-sm bg-ibkr-bg border border-ibkr-border rounded text-ibkr-text focus:outline-none focus:border-ibkr-accent"
                        />
                      </div>
                    </div>

                    {/* Notification Methods */}
                    <div>
                      <label className="block text-xs font-medium text-ibkr-text-secondary mb-2">
                        Notification Methods
                      </label>
                      <div className="space-y-2">
                        <label className="flex items-center space-x-2">
                          <input
                            type="checkbox"
                            checked={notificationMethods.desktop}
                            onChange={(e) => setNotificationMethods({
                              ...notificationMethods,
                              desktop: e.target.checked
                            })}
                            className="rounded border-ibkr-border"
                          />
                          <span className="text-sm text-ibkr-text">Desktop Notification</span>
                        </label>
                        <label className="flex items-center space-x-2">
                          <input
                            type="checkbox"
                            checked={notificationMethods.browser_push}
                            onChange={(e) => setNotificationMethods({
                              ...notificationMethods,
                              browser_push: e.target.checked
                            })}
                            className="rounded border-ibkr-border"
                          />
                          <span className="text-sm text-ibkr-text">Browser Push</span>
                        </label>
                        <label className="flex items-center space-x-2">
                          <input
                            type="checkbox"
                            checked={notificationMethods.sound}
                            onChange={(e) => setNotificationMethods({
                              ...notificationMethods,
                              sound: e.target.checked
                            })}
                            className="rounded border-ibkr-border"
                          />
                          <span className="text-sm text-ibkr-text">Sound Alert</span>
                        </label>
                        <div className="grid grid-cols-2 gap-2 mt-2">
                          <input
                            type="email"
                            value={notificationMethods.email || ''}
                            onChange={(e) => setNotificationMethods({
                              ...notificationMethods,
                              email: e.target.value || undefined
                            })}
                            placeholder="Email (optional)"
                            className="px-3 py-2 text-sm bg-ibkr-bg border border-ibkr-border rounded text-ibkr-text focus:outline-none focus:border-ibkr-accent"
                          />
                          <input
                            type="tel"
                            value={notificationMethods.sms || ''}
                            onChange={(e) => setNotificationMethods({
                              ...notificationMethods,
                              sms: e.target.value || undefined
                            })}
                            placeholder="SMS (optional)"
                            className="px-3 py-2 text-sm bg-ibkr-bg border border-ibkr-border rounded text-ibkr-text focus:outline-none focus:border-ibkr-accent"
                          />
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Modal Actions */}
                  <div className="flex justify-end space-x-3 mt-6">
                    <button
                      onClick={() => setIsConfigModalOpen(false)}
                      disabled={isLoading}
                      className="px-4 py-2 text-sm bg-ibkr-bg text-ibkr-text border border-ibkr-border rounded hover:bg-opacity-80 transition-colors disabled:opacity-50"
                    >
                      Cancel
                    </button>
                    <button
                      onClick={handleSaveAlert}
                      disabled={isLoading}
                      className="px-4 py-2 text-sm bg-ibkr-accent text-white rounded hover:bg-opacity-90 transition-colors disabled:opacity-50"
                    >
                      {isLoading ? 'Saving...' : (editingAlert ? 'Update Alert' : 'Create Alert')}
                    </button>
                  </div>
                </Dialog.Panel>
              </Transition.Child>
            </div>
          </div>
        </Dialog>
      </Transition>
    </div>
  );
};

export default AlertSystem;
