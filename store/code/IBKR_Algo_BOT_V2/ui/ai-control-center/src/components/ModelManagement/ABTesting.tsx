import React, { useState, useEffect } from 'react';
import { Dialog } from '@headlessui/react';
import apiService from '../../services/api';
import type { ClaudeInsights } from '../../types/models';
import { formatPercentage, formatCurrency, getStatusColor, getStatusIcon } from '../../utils/helpers';

interface Experiment {
  id: string;
  name: string;
  status: 'running' | 'paused' | 'complete';
  modelA: string;
  modelB: string;
  trafficSplit: number;
  duration: number;
  elapsed: number;
  results: ExperimentResults;
}

interface ExperimentResults {
  modelA: ModelResults;
  modelB: ModelResults;
  winner?: 'A' | 'B' | null;
  significance: number;
  claudeVerdict?: ClaudeInsights;
}

interface ModelResults {
  winRate: number;
  totalProfit: number;
  sharpeRatio: number;
  maxDrawdown: number;
  totalTrades: number;
}

export const ABTesting: React.FC = () => {
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [loading, setLoading] = useState(true);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [selectedExperiment, setSelectedExperiment] = useState<Experiment | null>(null);
  const [detailsModalOpen, setDetailsModalOpen] = useState(false);

  // New experiment form
  const [newExperiment, setNewExperiment] = useState({
    name: '',
    modelA: 'Ensemble',
    modelB: 'Random Forest',
    trafficSplit: 50,
    duration: 7
  });

  useEffect(() => {
    fetchExperiments();
  }, []);

  const fetchExperiments = async () => {
    try {
      setLoading(true);
      const response = await apiService.getExperiments();
      if (response.success && response.data) {
        setExperiments(response.data);
      } else {
        // Mock data for demonstration
        setExperiments([
          {
            id: '1',
            name: 'Ensemble vs XGBoost',
            status: 'running',
            modelA: 'Ensemble',
            modelB: 'XGBoost',
            trafficSplit: 50,
            duration: 7,
            elapsed: 4,
            results: {
              modelA: {
                winRate: 0.72,
                totalProfit: 4250.50,
                sharpeRatio: 1.85,
                maxDrawdown: 0.08,
                totalTrades: 125
              },
              modelB: {
                winRate: 0.68,
                totalProfit: 3890.20,
                sharpeRatio: 1.62,
                maxDrawdown: 0.12,
                totalTrades: 118
              },
              winner: 'A',
              significance: 87
            }
          },
          {
            id: '2',
            name: 'Random Forest v2 vs v1',
            status: 'complete',
            modelA: 'Random Forest v2',
            modelB: 'Random Forest v1',
            trafficSplit: 70,
            duration: 14,
            elapsed: 14,
            results: {
              modelA: {
                winRate: 0.76,
                totalProfit: 8920.75,
                sharpeRatio: 2.12,
                maxDrawdown: 0.06,
                totalTrades: 245
              },
              modelB: {
                winRate: 0.71,
                totalProfit: 7150.30,
                sharpeRatio: 1.88,
                maxDrawdown: 0.09,
                totalTrades: 198
              },
              winner: 'A',
              significance: 95
            }
          }
        ]);
      }
    } catch (error) {
      console.error('Error fetching experiments:', error);
    } finally {
      setLoading(false);
    }
  };

  const handlePromote = async (experimentId: string) => {
    if (!window.confirm('Promote the winning model to production?')) return;

    try {
      await apiService.promoteExperiment(experimentId);
      alert('Model promoted successfully!');
      fetchExperiments();
    } catch (error) {
      console.error('Error promoting experiment:', error);
      alert('Failed to promote model');
    }
  };

  const handlePause = async (experimentId: string) => {
    try {
      await apiService.pauseExperiment(experimentId);
      fetchExperiments();
    } catch (error) {
      console.error('Error pausing experiment:', error);
    }
  };

  const handleCreateExperiment = async () => {
    try {
      const response = await apiService.createExperiment(newExperiment);
      if (response.success) {
        setIsModalOpen(false);
        setNewExperiment({
          name: '',
          modelA: 'Ensemble',
          modelB: 'Random Forest',
          trafficSplit: 50,
          duration: 7
        });
        fetchExperiments();
      }
    } catch (error) {
      console.error('Error creating experiment:', error);
      alert('Failed to create experiment');
    }
  };

  const getProgressPercentage = (experiment: Experiment) => {
    return (experiment.elapsed / experiment.duration) * 100;
  };

  const getClaudeVerdict = (results: ExperimentResults): { text: string; recommendation: string; color: string } => {
    if (!results.winner) {
      return {
        text: 'Results are too close to call. Continue testing.',
        recommendation: 'Continue',
        color: '#dcdcaa'
      };
    }

    const winnerModel = results.winner === 'A' ? 'Model A' : 'Model B';
    const confidence = results.significance >= 95 ? 'high' : results.significance >= 80 ? 'moderate' : 'low';

    if (results.significance >= 95) {
      return {
        text: `${winnerModel} is the clear winner with ${results.significance}% statistical significance. Strong recommendation to promote.`,
        recommendation: 'Promote',
        color: '#4ec9b0'
      };
    } else if (results.significance >= 80) {
      return {
        text: `${winnerModel} shows better performance with ${results.significance}% confidence. Consider promoting after more data.`,
        recommendation: 'Continue',
        color: '#007acc'
      };
    } else {
      return {
        text: `${winnerModel} has a slight edge, but ${results.significance}% significance is too low. Keep testing.`,
        recommendation: 'Continue',
        color: '#dcdcaa'
      };
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-bold text-ibkr-text">A/B Testing</h2>
        <button
          onClick={() => setIsModalOpen(true)}
          className="px-4 py-2 bg-ibkr-accent text-white text-xs rounded hover:bg-blue-600 transition-colors"
        >
          + New Experiment
        </button>
      </div>

      {/* Active Experiments List */}
      <div className="space-y-4">
        {loading ? (
          <div className="bg-ibkr-surface p-8 rounded border border-ibkr-border text-center">
            <div className="text-ibkr-text-secondary text-xs">Loading experiments...</div>
          </div>
        ) : experiments.length === 0 ? (
          <div className="bg-ibkr-surface p-8 rounded border border-ibkr-border text-center">
            <div className="text-4xl mb-4">🧪</div>
            <h3 className="text-sm font-bold text-ibkr-text mb-2">No Active Experiments</h3>
            <p className="text-xs text-ibkr-text-secondary mb-4">
              Create an A/B test to compare model performance
            </p>
            <button
              onClick={() => setIsModalOpen(true)}
              className="px-4 py-2 bg-ibkr-accent text-white text-xs rounded hover:bg-blue-600 transition-colors"
            >
              Create First Experiment
            </button>
          </div>
        ) : (
          experiments.map(experiment => {
            const verdict = getClaudeVerdict(experiment.results);

            return (
              <div
                key={experiment.id}
                className="bg-ibkr-surface rounded border border-ibkr-border overflow-hidden"
              >
                {/* Experiment Header */}
                <div className="px-4 py-3 border-b border-ibkr-border flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    <h3 className="text-sm font-bold text-ibkr-text">{experiment.name}</h3>
                    <span
                      className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium"
                      style={{
                        backgroundColor: getStatusColor(experiment.status) + '20',
                        color: getStatusColor(experiment.status)
                      }}
                    >
                      <span className="mr-1">{getStatusIcon(experiment.status)}</span>
                      {experiment.status}
                    </span>
                  </div>

                  <div className="flex items-center space-x-2">
                    <span className="text-xs text-ibkr-text-secondary">
                      Day {experiment.elapsed} of {experiment.duration}
                    </span>
                  </div>
                </div>

                {/* Progress Bar */}
                <div className="px-4 py-2 bg-ibkr-bg">
                  <div className="w-full bg-ibkr-surface rounded-full h-1.5">
                    <div
                      className="bg-ibkr-accent h-1.5 rounded-full transition-all"
                      style={{ width: `${getProgressPercentage(experiment)}%` }}
                    />
                  </div>
                </div>

                {/* Traffic Split Visualization */}
                <div className="px-4 py-3 border-b border-ibkr-border">
                  <div className="flex items-center space-x-2 mb-2">
                    <span className="text-xs text-ibkr-text-secondary">Traffic Split:</span>
                    <div className="flex-1 flex h-6 rounded overflow-hidden">
                      <div
                        className="bg-ibkr-accent flex items-center justify-center"
                        style={{ width: `${experiment.trafficSplit}%` }}
                      >
                        <span className="text-xs text-white font-bold">{experiment.trafficSplit}%</span>
                      </div>
                      <div
                        className="bg-ibkr-warning flex items-center justify-center"
                        style={{ width: `${100 - experiment.trafficSplit}%` }}
                      >
                        <span className="text-xs text-ibkr-bg font-bold">{100 - experiment.trafficSplit}%</span>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Results Table */}
                <div className="p-4">
                  <div className="overflow-x-auto">
                    <table className="w-full text-xs">
                      <thead>
                        <tr className="text-ibkr-text-secondary border-b border-ibkr-border">
                          <th className="px-2 py-2 text-left">Model</th>
                          <th className="px-2 py-2 text-right">Win Rate</th>
                          <th className="px-2 py-2 text-right">Total Profit</th>
                          <th className="px-2 py-2 text-right">Sharpe</th>
                          <th className="px-2 py-2 text-right">Max DD</th>
                          <th className="px-2 py-2 text-right">Trades</th>
                          <th className="px-2 py-2 text-center">Winner</th>
                        </tr>
                      </thead>
                      <tbody>
                        <tr className="border-b border-ibkr-border">
                          <td className="px-2 py-3 font-medium text-ibkr-text">{experiment.modelA}</td>
                          <td className="px-2 py-3 text-right text-ibkr-text">
                            {formatPercentage(experiment.results.modelA.winRate)}
                          </td>
                          <td className="px-2 py-3 text-right text-ibkr-success">
                            {formatCurrency(experiment.results.modelA.totalProfit)}
                          </td>
                          <td className="px-2 py-3 text-right text-ibkr-text">
                            {experiment.results.modelA.sharpeRatio.toFixed(2)}
                          </td>
                          <td className="px-2 py-3 text-right text-ibkr-error">
                            {formatPercentage(experiment.results.modelA.maxDrawdown)}
                          </td>
                          <td className="px-2 py-3 text-right text-ibkr-text">
                            {experiment.results.modelA.totalTrades}
                          </td>
                          <td className="px-2 py-3 text-center">
                            {experiment.results.winner === 'A' && (
                              <span className="text-ibkr-warning text-lg">🏆</span>
                            )}
                          </td>
                        </tr>
                        <tr>
                          <td className="px-2 py-3 font-medium text-ibkr-text">{experiment.modelB}</td>
                          <td className="px-2 py-3 text-right text-ibkr-text">
                            {formatPercentage(experiment.results.modelB.winRate)}
                          </td>
                          <td className="px-2 py-3 text-right text-ibkr-success">
                            {formatCurrency(experiment.results.modelB.totalProfit)}
                          </td>
                          <td className="px-2 py-3 text-right text-ibkr-text">
                            {experiment.results.modelB.sharpeRatio.toFixed(2)}
                          </td>
                          <td className="px-2 py-3 text-right text-ibkr-error">
                            {formatPercentage(experiment.results.modelB.maxDrawdown)}
                          </td>
                          <td className="px-2 py-3 text-right text-ibkr-text">
                            {experiment.results.modelB.totalTrades}
                          </td>
                          <td className="px-2 py-3 text-center">
                            {experiment.results.winner === 'B' && (
                              <span className="text-ibkr-warning text-lg">🏆</span>
                            )}
                          </td>
                        </tr>
                      </tbody>
                    </table>
                  </div>

                  {/* Statistical Significance */}
                  <div className="mt-3 p-3 bg-ibkr-bg rounded">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-xs text-ibkr-text-secondary">Statistical Significance:</span>
                      <span className="text-xs font-bold text-ibkr-text">
                        {experiment.results.significance}%
                      </span>
                    </div>
                    <div className="w-full bg-ibkr-surface rounded-full h-1.5">
                      <div
                        className="h-1.5 rounded-full"
                        style={{
                          width: `${experiment.results.significance}%`,
                          backgroundColor:
                            experiment.results.significance >= 95
                              ? '#4ec9b0'
                              : experiment.results.significance >= 80
                              ? '#007acc'
                              : '#dcdcaa'
                        }}
                      />
                    </div>
                  </div>
                </div>

                {/* Claude's Verdict */}
                <div className="px-4 py-3 border-t border-ibkr-border bg-ibkr-bg">
                  <div className="flex items-start space-x-3">
                    <span className="text-lg">🧠</span>
                    <div className="flex-1">
                      <div className="text-xs font-bold text-ibkr-text mb-1">Claude's Verdict:</div>
                      <p className="text-xs text-ibkr-text-secondary mb-2">{verdict.text}</p>
                      <div className="flex items-center space-x-2">
                        <span className="text-xs text-ibkr-text-secondary">Recommendation:</span>
                        <span
                          className="px-2 py-0.5 rounded text-xs font-bold"
                          style={{
                            backgroundColor: verdict.color + '20',
                            color: verdict.color
                          }}
                        >
                          {verdict.recommendation}
                        </span>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Actions */}
                <div className="px-4 py-3 border-t border-ibkr-border flex items-center justify-end space-x-2">
                  {experiment.results.winner && experiment.results.significance >= 80 && (
                    <button
                      onClick={() => handlePromote(experiment.id)}
                      className="px-3 py-1.5 bg-ibkr-success text-white text-xs rounded hover:bg-green-600 transition-colors"
                    >
                      Promote Winner
                    </button>
                  )}
                  {experiment.status === 'running' && (
                    <button
                      onClick={() => handlePause(experiment.id)}
                      className="px-3 py-1.5 bg-ibkr-bg text-ibkr-text text-xs rounded border border-ibkr-border hover:border-ibkr-accent transition-colors"
                    >
                      Pause Test
                    </button>
                  )}
                  <button
                    onClick={() => {
                      setSelectedExperiment(experiment);
                      setDetailsModalOpen(true);
                    }}
                    className="px-3 py-1.5 bg-ibkr-accent text-white text-xs rounded hover:bg-blue-600 transition-colors"
                  >
                    View Details
                  </button>
                </div>
              </div>
            );
          })
        )}
      </div>

      {/* New Experiment Modal */}
      <Dialog
        open={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        className="relative z-50"
      >
        <div className="fixed inset-0 bg-black/60" aria-hidden="true" />

        <div className="fixed inset-0 flex items-center justify-center p-4">
          <Dialog.Panel className="bg-ibkr-surface rounded border border-ibkr-border p-6 max-w-md w-full">
            <Dialog.Title className="text-sm font-bold text-ibkr-text mb-4">
              Create New A/B Test
            </Dialog.Title>

            <div className="space-y-4">
              <div>
                <label className="block text-xs text-ibkr-text-secondary mb-1">
                  Experiment Name
                </label>
                <input
                  type="text"
                  value={newExperiment.name}
                  onChange={(e) => setNewExperiment({ ...newExperiment, name: e.target.value })}
                  placeholder="e.g., Ensemble vs XGBoost"
                  className="w-full bg-ibkr-bg text-ibkr-text text-xs px-3 py-2 rounded border border-ibkr-border focus:outline-none focus:border-ibkr-accent"
                />
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-xs text-ibkr-text-secondary mb-1">
                    Model A
                  </label>
                  <select
                    value={newExperiment.modelA}
                    onChange={(e) => setNewExperiment({ ...newExperiment, modelA: e.target.value })}
                    className="w-full bg-ibkr-bg text-ibkr-text text-xs px-3 py-2 rounded border border-ibkr-border focus:outline-none focus:border-ibkr-accent"
                  >
                    <option>Ensemble</option>
                    <option>Random Forest</option>
                    <option>XGBoost</option>
                    <option>LightGBM</option>
                  </select>
                </div>

                <div>
                  <label className="block text-xs text-ibkr-text-secondary mb-1">
                    Model B
                  </label>
                  <select
                    value={newExperiment.modelB}
                    onChange={(e) => setNewExperiment({ ...newExperiment, modelB: e.target.value })}
                    className="w-full bg-ibkr-bg text-ibkr-text text-xs px-3 py-2 rounded border border-ibkr-border focus:outline-none focus:border-ibkr-accent"
                  >
                    <option>Ensemble</option>
                    <option>Random Forest</option>
                    <option>XGBoost</option>
                    <option>LightGBM</option>
                  </select>
                </div>
              </div>

              <div>
                <label className="block text-xs text-ibkr-text-secondary mb-1">
                  Traffic Split (Model A / Model B): {newExperiment.trafficSplit}% / {100 - newExperiment.trafficSplit}%
                </label>
                <input
                  type="range"
                  min="10"
                  max="90"
                  step="10"
                  value={newExperiment.trafficSplit}
                  onChange={(e) => setNewExperiment({ ...newExperiment, trafficSplit: parseInt(e.target.value) })}
                  className="w-full"
                />
                <div className="flex h-4 rounded overflow-hidden mt-2">
                  <div
                    className="bg-ibkr-accent"
                    style={{ width: `${newExperiment.trafficSplit}%` }}
                  />
                  <div
                    className="bg-ibkr-warning"
                    style={{ width: `${100 - newExperiment.trafficSplit}%` }}
                  />
                </div>
              </div>

              <div>
                <label className="block text-xs text-ibkr-text-secondary mb-1">
                  Duration (days)
                </label>
                <input
                  type="number"
                  min="1"
                  max="30"
                  value={newExperiment.duration}
                  onChange={(e) => setNewExperiment({ ...newExperiment, duration: parseInt(e.target.value) })}
                  className="w-full bg-ibkr-bg text-ibkr-text text-xs px-3 py-2 rounded border border-ibkr-border focus:outline-none focus:border-ibkr-accent"
                />
              </div>
            </div>

            <div className="flex items-center justify-end space-x-2 mt-6">
              <button
                onClick={() => setIsModalOpen(false)}
                className="px-4 py-2 bg-ibkr-bg text-ibkr-text text-xs rounded border border-ibkr-border hover:border-ibkr-accent transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={handleCreateExperiment}
                disabled={!newExperiment.name || newExperiment.modelA === newExperiment.modelB}
                className="px-4 py-2 bg-ibkr-accent text-white text-xs rounded hover:bg-blue-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Start Experiment
              </button>
            </div>
          </Dialog.Panel>
        </div>
      </Dialog>

      {/* Details Modal */}
      <Dialog
        open={detailsModalOpen}
        onClose={() => setDetailsModalOpen(false)}
        className="relative z-50"
      >
        <div className="fixed inset-0 bg-black/60" aria-hidden="true" />

        <div className="fixed inset-0 flex items-center justify-center p-4">
          <Dialog.Panel className="bg-ibkr-surface rounded border border-ibkr-border p-6 max-w-2xl w-full max-h-[80vh] overflow-y-auto">
            {selectedExperiment && (
              <>
                <Dialog.Title className="text-sm font-bold text-ibkr-text mb-4">
                  {selectedExperiment.name} - Detailed Results
                </Dialog.Title>

                <div className="space-y-4 text-xs">
                  <div className="bg-ibkr-bg p-4 rounded">
                    <p className="text-ibkr-text-secondary">
                      Detailed experiment statistics and analysis would be displayed here, including:
                    </p>
                    <ul className="mt-2 space-y-1 text-ibkr-text-secondary">
                      <li>• Trade-by-trade comparison</li>
                      <li>• Performance over time charts</li>
                      <li>• Risk-adjusted metrics</li>
                      <li>• Confidence intervals</li>
                      <li>• Statistical test results</li>
                    </ul>
                  </div>
                </div>

                <div className="flex justify-end mt-6">
                  <button
                    onClick={() => setDetailsModalOpen(false)}
                    className="px-4 py-2 bg-ibkr-accent text-white text-xs rounded hover:bg-blue-600 transition-colors"
                  >
                    Close
                  </button>
                </div>
              </>
            )}
          </Dialog.Panel>
        </div>
      </Dialog>
    </div>
  );
};

export default ABTesting;
