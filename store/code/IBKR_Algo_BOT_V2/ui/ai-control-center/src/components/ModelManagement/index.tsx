import React, { useState } from 'react';
import TrainingInterface from './TrainingInterface';
import PerformanceDashboard from './PerformanceDashboard';
import ABTesting from './ABTesting';

type Tab = 'training' | 'performance' | 'ab-testing';

export const ModelManagement: React.FC = () => {
  const [activeTab, setActiveTab] = useState<Tab>('training');

  const tabs: { id: Tab; label: string; icon: string }[] = [
    { id: 'training', label: 'Model Training', icon: '⚙️' },
    { id: 'performance', label: 'Performance Dashboard', icon: '📊' },
    { id: 'ab-testing', label: 'A/B Testing', icon: '🧪' }
  ];

  return (
    <div className="p-6">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-xl font-bold text-ibkr-text">Model Management</h1>
      </div>

      {/* Tabs */}
      <div className="border-b border-ibkr-border mb-6">
        <div className="flex space-x-1">
          {tabs.map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`px-4 py-2 text-sm font-medium transition-colors ${
                activeTab === tab.id
                  ? 'text-ibkr-text border-b-2 border-ibkr-accent'
                  : 'text-ibkr-text-secondary hover:text-ibkr-text'
              }`}
            >
              <span className="mr-1.5">{tab.icon}</span>
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      {/* Tab Content */}
      <div>
        {activeTab === 'training' && <TrainingInterface />}

        {activeTab === 'performance' && <PerformanceDashboard />}

        {activeTab === 'ab-testing' && <ABTesting />}
      </div>
    </div>
  );
};

export default ModelManagement;
