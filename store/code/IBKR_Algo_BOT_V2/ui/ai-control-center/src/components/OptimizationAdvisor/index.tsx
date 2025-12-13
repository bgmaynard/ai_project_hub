import React, { useState, useEffect } from 'react';
import apiService from '../../services/api';

interface Optimization {
  id: string;
  title: string;
  description: string;
  category: 'risk_management' | 'strategy' | 'execution' | 'parameters' | 'other';
  priority: 'high' | 'medium' | 'low';
  impact: {
    estimated_improvement: string;
    confidence: number;
    affected_metrics: string[];
  };
  current_state: any;
  proposed_changes: any;
  status: 'pending' | 'applied' | 'reverted' | 'testing';
  created_at: string;
  applied_at?: string;
  reverted_at?: string;
}

interface OptimizationHistory {
  id: string;
  optimization_id: string;
  title: string;
  action: 'applied' | 'reverted';
  timestamp: string;
  impact_observed?: string;
}

const CATEGORY_ICONS = {
  risk_management: '=á',
  strategy: '<¯',
  execution: '¡',
  parameters: '',
  other: '=¡'
};

const PRIORITY_COLORS = {
  high: 'text-red-500 bg-red-500/10 border-red-500/30',
  medium: 'text-yellow-500 bg-yellow-500/10 border-yellow-500/30',
  low: 'text-blue-500 bg-blue-500/10 border-blue-500/30'
};

export const OptimizationAdvisor: React.FC = () => {
  const [optimizations, setOptimizations] = useState<Optimization[]>([]);
  const [history, setHistory] = useState<OptimizationHistory[]>([]);
  const [scanning, setScanning] = useState(false);
  const [selectedOptimization, setSelectedOptimization] = useState<Optimization | null>(null);
  const [activeTab, setActiveTab] = useState<'suggestions' | 'history'>('suggestions');
  const [applying, setApplying] = useState<string | null>(null);
  const [reverting, setReverting] = useState<string | null>(null);

  useEffect(() => {
    loadOptimizations();
    loadHistory();
  }, []);

  const loadOptimizations = async () => {
    try {
      const response = await apiService.scanOptimizations();
      if (response.success && response.data) {
        setOptimizations(response.data);
      }
    } catch (error) {
      console.error('Failed to load optimizations:', error);
    }
  };

  const loadHistory = async () => {
    try {
      const response = await apiService.getOptimizationHistory();
      if (response.success && response.data) {
        setHistory(response.data);
      }
    } catch (error) {
      console.error('Failed to load optimization history:', error);
    }
  };

  const scanForOptimizations = async () => {
    setScanning(true);
    try {
      const response = await apiService.scanOptimizations();
      if (response.success && response.data) {
        setOptimizations(response.data);
      }
    } catch (error) {
      console.error('Failed to scan optimizations:', error);
    } finally {
      setScanning(false);
    }
  };

  const applyOptimization = async (optimizationId: string) => {
    setApplying(optimizationId);
    try {
      const response = await apiService.applyOptimization(optimizationId);
      if (response.success) {
        await loadOptimizations();
        await loadHistory();
      }
    } catch (error) {
      console.error('Failed to apply optimization:', error);
    } finally {
      setApplying(null);
    }
  };

  const revertOptimization = async (optimizationId: string) => {
    setReverting(optimizationId);
    try {
      const response = await apiService.revertOptimization(optimizationId);
      if (response.success) {
        await loadOptimizations();
        await loadHistory();
      }
    } catch (error) {
      console.error('Failed to revert optimization:', error);
    } finally {
      setReverting(null);
    }
  };

  const groupedOptimizations = optimizations.reduce((acc, opt) => {
    const category = opt.category;
    if (!acc[category]) {
      acc[category] = [];
    }
    acc[category].push(opt);
    return acc;
  }, {} as Record<string, Optimization[]>);

  const pendingCount = optimizations.filter(o => o.status === 'pending').length;
  const appliedCount = optimizations.filter(o => o.status === 'applied').length;

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold text-ibkr-text">Performance Optimization Advisor</h1>
          <p className="text-sm text-ibkr-text-secondary mt-1">
            AI-powered suggestions to improve your trading performance
          </p>
        </div>
        <button
          onClick={scanForOptimizations}
          disabled={scanning}
          className="px-4 py-2 bg-ibkr-accent hover:bg-opacity-90 text-white rounded transition-colors disabled:opacity-50 flex items-center space-x-2"
        >
          <span>{scanning ? "🔍" : "⚡"}</span>
          <span>{scanning ? 'Scanning...' : 'Scan for Optimizations'}</span>
        </button>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-ibkr-surface border border-ibkr-border rounded-lg p-4">
          <div className="text-xs text-ibkr-text-secondary mb-1">Pending Suggestions</div>
          <div className="text-2xl font-bold text-ibkr-text">{pendingCount}</div>
          <div className="text-xs text-ibkr-text-secondary mt-1">Awaiting review</div>
        </div>

        <div className="bg-ibkr-surface border border-ibkr-border rounded-lg p-4">
          <div className="text-xs text-ibkr-text-secondary mb-1">Applied Optimizations</div>
          <div className="text-2xl font-bold text-green-500">{appliedCount}</div>
          <div className="text-xs text-ibkr-text-secondary mt-1">Currently active</div>
        </div>

        <div className="bg-ibkr-surface border border-ibkr-border rounded-lg p-4">
          <div className="text-xs text-ibkr-text-secondary mb-1">Total Actions</div>
          <div className="text-2xl font-bold text-ibkr-text">{history.length}</div>
          <div className="text-xs text-ibkr-text-secondary mt-1">Historical changes</div>
        </div>
      </div>

      {/* Tab Navigation */}
      <div className="flex space-x-1 bg-ibkr-surface border border-ibkr-border rounded-lg p-1">
        <button
          onClick={() => setActiveTab('suggestions')}
          className={`flex-1 px-4 py-2 text-sm font-medium rounded transition-colors ${
            activeTab === 'suggestions'
              ? 'bg-ibkr-accent text-white'
              : 'text-ibkr-text hover:bg-ibkr-bg'
          }`}
        >
          =¡ Suggestions ({pendingCount})
        </button>
        <button
          onClick={() => setActiveTab('history')}
          className={`flex-1 px-4 py-2 text-sm font-medium rounded transition-colors ${
            activeTab === 'history'
              ? 'bg-ibkr-accent text-white'
              : 'text-ibkr-text hover:bg-ibkr-bg'
          }`}
        >
          =Ü History ({history.length})
        </button>
      </div>

      {/* Tab Content */}
      {activeTab === 'suggestions' ? (
        <div className="space-y-6">
          {Object.keys(groupedOptimizations).length === 0 ? (
            <div className="bg-ibkr-surface border border-ibkr-border rounded-lg p-12 text-center">
              <div className="text-6xl mb-4">(</div>
              <h3 className="text-lg font-semibold text-ibkr-text mb-2">
                No Optimizations Found
              </h3>
              <p className="text-sm text-ibkr-text-secondary mb-4">
                Your system is running optimally. Click "Scan for Optimizations" to check again.
              </p>
            </div>
          ) : (
            Object.entries(groupedOptimizations).map(([category, opts]) => (
              <div key={category}>
                <h3 className="text-sm font-semibold text-ibkr-text mb-3 flex items-center space-x-2">
                  <span>{CATEGORY_ICONS[category as keyof typeof CATEGORY_ICONS]}</span>
                  <span className="capitalize">{category.replace('_', ' ')}</span>
                  <span className="text-ibkr-text-secondary">({opts.length})</span>
                </h3>

                <div className="space-y-3">
                  {opts.map(optimization => (
                    <div
                      key={optimization.id}
                      className="bg-ibkr-surface border border-ibkr-border rounded-lg p-4 hover:border-ibkr-accent transition-colors"
                    >
                      <div className="flex items-start justify-between mb-3">
                        <div className="flex-1">
                          <div className="flex items-center space-x-3 mb-2">
                            <h4 className="text-base font-semibold text-ibkr-text">
                              {optimization.title}
                            </h4>
                            <span
                              className={`px-2 py-0.5 text-xs font-medium rounded border ${
                                PRIORITY_COLORS[optimization.priority]
                              }`}
                            >
                              {optimization.priority.toUpperCase()}
                            </span>
                            {optimization.status !== 'pending' && (
                              <span className="px-2 py-0.5 text-xs font-medium rounded border border-green-500/30 bg-green-500/10 text-green-500">
                                {optimization.status.toUpperCase()}
                              </span>
                            )}
                          </div>
                          <p className="text-sm text-ibkr-text-secondary mb-3">
                            {optimization.description}
                          </p>
                        </div>
                      </div>

                      {/* Impact Metrics */}
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mb-4">
                        <div className="bg-ibkr-bg rounded p-3">
                          <div className="text-xs text-ibkr-text-secondary mb-1">
                            Estimated Improvement
                          </div>
                          <div className="text-sm font-semibold text-green-500">
                            {optimization.impact.estimated_improvement}
                          </div>
                        </div>
                        <div className="bg-ibkr-bg rounded p-3">
                          <div className="text-xs text-ibkr-text-secondary mb-1">
                            Confidence Level
                          </div>
                          <div className="text-sm font-semibold text-ibkr-text">
                            {(optimization.impact.confidence * 100).toFixed(0)}%
                          </div>
                        </div>
                        <div className="bg-ibkr-bg rounded p-3">
                          <div className="text-xs text-ibkr-text-secondary mb-1">
                            Affected Metrics
                          </div>
                          <div className="text-sm font-semibold text-ibkr-text">
                            {optimization.impact.affected_metrics.length} metrics
                          </div>
                        </div>
                      </div>

                      {/* Affected Metrics Tags */}
                      <div className="flex flex-wrap gap-2 mb-4">
                        {optimization.impact.affected_metrics.map((metric, idx) => (
                          <span
                            key={idx}
                            className="px-2 py-1 bg-ibkr-bg text-xs text-ibkr-text-secondary rounded"
                          >
                            {metric}
                          </span>
                        ))}
                      </div>

                      {/* Actions */}
                      <div className="flex items-center space-x-3 pt-3 border-t border-ibkr-border">
                        {optimization.status === 'pending' && (
                          <>
                            <button
                              onClick={() => applyOptimization(optimization.id)}
                              disabled={applying === optimization.id}
                              className="px-4 py-2 bg-green-600 hover:bg-green-700 text-white text-sm rounded transition-colors disabled:opacity-50 flex items-center space-x-2"
                            >
                              <span>{applying === optimization.id ? 'ó' : ''}</span>
                              <span>{applying === optimization.id ? 'Applying...' : 'Apply'}</span>
                            </button>
                            <button
                              onClick={() => setSelectedOptimization(optimization)}
                              className="px-4 py-2 bg-ibkr-bg hover:bg-ibkr-border text-ibkr-text text-sm rounded transition-colors"
                            >
                              =Ë View Details
                            </button>
                          </>
                        )}
                        {optimization.status === 'applied' && (
                          <>
                            <button
                              onClick={() => revertOptimization(optimization.id)}
                              disabled={reverting === optimization.id}
                              className="px-4 py-2 bg-red-600 hover:bg-red-700 text-white text-sm rounded transition-colors disabled:opacity-50 flex items-center space-x-2"
                            >
                              <span>{reverting === optimization.id ? 'ó' : '©'}</span>
                              <span>{reverting === optimization.id ? 'Reverting...' : 'Revert'}</span>
                            </button>
                            <div className="text-xs text-ibkr-text-secondary">
                              Applied {new Date(optimization.applied_at!).toLocaleString()}
                            </div>
                          </>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ))
          )}
        </div>
      ) : (
        // History Tab
        <div className="bg-ibkr-surface border border-ibkr-border rounded-lg">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-ibkr-border">
                  <th className="text-left p-4 text-sm text-ibkr-text-secondary">Timestamp</th>
                  <th className="text-left p-4 text-sm text-ibkr-text-secondary">Optimization</th>
                  <th className="text-left p-4 text-sm text-ibkr-text-secondary">Action</th>
                  <th className="text-left p-4 text-sm text-ibkr-text-secondary">Impact Observed</th>
                </tr>
              </thead>
              <tbody>
                {history.length === 0 ? (
                  <tr>
                    <td colSpan={4} className="p-8 text-center text-ibkr-text-secondary">
                      No optimization history yet
                    </td>
                  </tr>
                ) : (
                  history.map(item => (
                    <tr key={item.id} className="border-b border-ibkr-border hover:bg-ibkr-bg">
                      <td className="p-4 text-sm text-ibkr-text">
                        {new Date(item.timestamp).toLocaleString()}
                      </td>
                      <td className="p-4 text-sm text-ibkr-text font-medium">
                        {item.title}
                      </td>
                      <td className="p-4">
                        <span
                          className={`px-2 py-1 text-xs font-medium rounded ${
                            item.action === 'applied'
                              ? 'bg-green-500/10 text-green-500 border border-green-500/30'
                              : 'bg-red-500/10 text-red-500 border border-red-500/30'
                          }`}
                        >
                          {item.action.toUpperCase()}
                        </span>
                      </td>
                      <td className="p-4 text-sm text-ibkr-text-secondary">
                        {item.impact_observed || 'Pending measurement'}
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Details Modal */}
      {selectedOptimization && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
          <div className="bg-ibkr-surface border border-ibkr-border rounded-lg max-w-2xl w-full max-h-[80vh] overflow-y-auto">
            <div className="p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-bold text-ibkr-text">
                  {selectedOptimization.title}
                </h3>
                <button
                  onClick={() => setSelectedOptimization(null)}
                  className="text-ibkr-text-secondary hover:text-ibkr-text"
                >
                  
                </button>
              </div>

              <div className="space-y-4">
                <div>
                  <h4 className="text-sm font-semibold text-ibkr-text mb-2">Description</h4>
                  <p className="text-sm text-ibkr-text-secondary">
                    {selectedOptimization.description}
                  </p>
                </div>

                <div>
                  <h4 className="text-sm font-semibold text-ibkr-text mb-2">Current State</h4>
                  <pre className="bg-ibkr-bg p-3 rounded text-xs text-ibkr-text overflow-x-auto">
                    {JSON.stringify(selectedOptimization.current_state, null, 2)}
                  </pre>
                </div>

                <div>
                  <h4 className="text-sm font-semibold text-ibkr-text mb-2">Proposed Changes</h4>
                  <pre className="bg-ibkr-bg p-3 rounded text-xs text-ibkr-text overflow-x-auto">
                    {JSON.stringify(selectedOptimization.proposed_changes, null, 2)}
                  </pre>
                </div>
              </div>

              <div className="flex justify-end space-x-3 mt-6 pt-4 border-t border-ibkr-border">
                <button
                  onClick={() => setSelectedOptimization(null)}
                  className="px-4 py-2 bg-ibkr-bg hover:bg-ibkr-border text-ibkr-text text-sm rounded transition-colors"
                >
                  Close
                </button>
                {selectedOptimization.status === 'pending' && (
                  <button
                    onClick={() => {
                      applyOptimization(selectedOptimization.id);
                      setSelectedOptimization(null);
                    }}
                    className="px-4 py-2 bg-green-600 hover:bg-green-700 text-white text-sm rounded transition-colors"
                  >
                    Apply Optimization
                  </button>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default OptimizationAdvisor;
