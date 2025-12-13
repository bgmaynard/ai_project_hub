import React, { useState } from 'react';
import PushInterface from './PushInterface';
import WebhookConfig from './WebhookConfig';
import IndicatorBuilder from './IndicatorBuilder';

export const TradingViewHub: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'push' | 'webhook' | 'indicators'>('push');

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="p-6 pb-0">
        <h1 className="text-xl font-bold text-ibkr-text mb-1">TradingView Integration Hub</h1>
        <p className="text-sm text-ibkr-text-secondary">
          Sync AI predictions and custom indicators with TradingView charts
        </p>
      </div>

      {/* Tab Navigation */}
      <div className="px-6">
        <div className="flex space-x-1 bg-ibkr-surface border border-ibkr-border rounded-lg p-1">
          <button
            onClick={() => setActiveTab('push')}
            className={`flex-1 px-4 py-2 text-sm font-medium rounded transition-colors ${
              activeTab === 'push'
                ? 'bg-ibkr-accent text-white'
                : 'text-ibkr-text hover:bg-ibkr-bg'
            }`}
          >
            📤 Push Interface
          </button>
          <button
            onClick={() => setActiveTab('webhook')}
            className={`flex-1 px-4 py-2 text-sm font-medium rounded transition-colors ${
              activeTab === 'webhook'
                ? 'bg-ibkr-accent text-white'
                : 'text-ibkr-text hover:bg-ibkr-bg'
            }`}
          >
            🔗 Webhook Config
          </button>
          <button
            onClick={() => setActiveTab('indicators')}
            className={`flex-1 px-4 py-2 text-sm font-medium rounded transition-colors ${
              activeTab === 'indicators'
                ? 'bg-ibkr-accent text-white'
                : 'text-ibkr-text hover:bg-ibkr-bg'
            }`}
          >
            📊 Custom Indicators
          </button>
        </div>
      </div>

      {/* Tab Content */}
      {activeTab === 'push' && <PushInterface />}

      {activeTab === 'webhook' && <WebhookConfig />}

      {activeTab === 'indicators' && <IndicatorBuilder />}
    </div>
  );
};

export default TradingViewHub;
