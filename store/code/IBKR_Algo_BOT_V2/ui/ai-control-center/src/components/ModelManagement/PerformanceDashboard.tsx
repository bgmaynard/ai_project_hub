import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import apiService from '../../services/api';
import type { ModelPerformance, ClaudeInsights } from '../../types/models';
import { formatPercentage, getStatusColor, getStatusIcon, getColorForValue } from '../../utils/helpers';

export const PerformanceDashboard: React.FC = () => {
  const [models, setModels] = useState<ModelPerformance[]>([]);
  const [comparisonData, setComparisonData] = useState<any[]>([]);
  const [claudeAnalysis, setClaudeAnalysis] = useState<ClaudeInsights | null>(null);
  const [loading, setLoading] = useState(true);
  const [sortColumn, setSortColumn] = useState<keyof ModelPerformance>('accuracy');
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('desc');
  const [timeframe, setTimeframe] = useState('30d');

  // Fetch data
  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 30000); // Refresh every 30 seconds
    return () => clearInterval(interval);
  }, [timeframe]);

  const fetchData = async () => {
    try {
      setLoading(true);

      // Fetch model performance
      const perfResponse = await apiService.getModelPerformance();
      if (perfResponse.success && perfResponse.data) {
        setModels(perfResponse.data);
      }

      // Fetch comparison data
      const compResponse = await apiService.compareModels(timeframe);
      if (compResponse.success && compResponse.data) {
        setComparisonData(compResponse.data);
      }

      // Fetch Claude insights
      const insightsResponse = await apiService.getModelInsights();
      if (insightsResponse.success && insightsResponse.data) {
        setClaudeAnalysis(insightsResponse.data);
      }
    } catch (error) {
      console.error('Error fetching performance data:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleSort = (column: keyof ModelPerformance) => {
    if (sortColumn === column) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortColumn(column);
      setSortDirection('desc');
    }
  };

  const sortedModels = [...models].sort((a, b) => {
    const aVal = a[sortColumn];
    const bVal = b[sortColumn];

    if (typeof aVal === 'number' && typeof bVal === 'number') {
      return sortDirection === 'asc' ? aVal - bVal : bVal - aVal;
    }

    return 0;
  });

  // Mock data for demonstration
  const mockTodayStats = {
    correct: 42,
    total: 50,
    winRate: 0.84,
    byStrategy: [
      { name: 'Gap & Go', correct: 15, total: 18, rate: 0.833 },
      { name: 'Momentum', correct: 18, total: 20, rate: 0.900 },
      { name: 'Bull Flag', correct: 9, total: 12, rate: 0.750 }
    ]
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-bold text-ibkr-text">Model Performance Dashboard</h2>
        <button
          onClick={fetchData}
          disabled={loading}
          className="px-3 py-1.5 bg-ibkr-accent text-white text-xs rounded hover:bg-blue-600 transition-colors disabled:opacity-50"
        >
          {loading ? 'Refreshing...' : 'Refresh Now'}
        </button>
      </div>

      {/* Active Models Table */}
      <div className="bg-ibkr-surface rounded border border-ibkr-border overflow-hidden">
        <div className="px-4 py-3 border-b border-ibkr-border">
          <h3 className="text-sm font-bold text-ibkr-text">Active Models</h3>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="bg-ibkr-bg text-ibkr-text-secondary">
                <th className="px-4 py-2 text-left">
                  <button
                    onClick={() => handleSort('model_name')}
                    className="flex items-center hover:text-ibkr-text"
                  >
                    Model Name
                    {sortColumn === 'model_name' && (
                      <span className="ml-1">{sortDirection === 'asc' ? '↑' : '↓'}</span>
                    )}
                  </button>
                </th>
                <th className="px-4 py-2 text-right">
                  <button
                    onClick={() => handleSort('accuracy')}
                    className="flex items-center ml-auto hover:text-ibkr-text"
                  >
                    Accuracy
                    {sortColumn === 'accuracy' && (
                      <span className="ml-1">{sortDirection === 'asc' ? '↑' : '↓'}</span>
                    )}
                  </button>
                </th>
                <th className="px-4 py-2 text-right">
                  <button
                    onClick={() => handleSort('precision')}
                    className="flex items-center ml-auto hover:text-ibkr-text"
                  >
                    Precision
                    {sortColumn === 'precision' && (
                      <span className="ml-1">{sortDirection === 'asc' ? '↑' : '↓'}</span>
                    )}
                  </button>
                </th>
                <th className="px-4 py-2 text-right">
                  <button
                    onClick={() => handleSort('recall')}
                    className="flex items-center ml-auto hover:text-ibkr-text"
                  >
                    Recall
                    {sortColumn === 'recall' && (
                      <span className="ml-1">{sortDirection === 'asc' ? '↑' : '↓'}</span>
                    )}
                  </button>
                </th>
                <th className="px-4 py-2 text-right">
                  <button
                    onClick={() => handleSort('f1_score')}
                    className="flex items-center ml-auto hover:text-ibkr-text"
                  >
                    F1 Score
                    {sortColumn === 'f1_score' && (
                      <span className="ml-1">{sortDirection === 'asc' ? '↑' : '↓'}</span>
                    )}
                  </button>
                </th>
                <th className="px-4 py-2 text-center">Status</th>
              </tr>
            </thead>
            <tbody>
              {sortedModels.length === 0 ? (
                <tr>
                  <td colSpan={6} className="px-4 py-8 text-center text-ibkr-text-secondary">
                    {loading ? 'Loading models...' : 'No models available'}
                  </td>
                </tr>
              ) : (
                sortedModels.map((model, idx) => (
                  <tr
                    key={idx}
                    className="border-t border-ibkr-border hover:bg-ibkr-bg transition-colors"
                  >
                    <td className="px-4 py-3 text-ibkr-text font-medium">{model.model_name}</td>
                    <td className="px-4 py-3 text-right">
                      <span className="font-bold" style={{ color: getColorForValue(model.accuracy - 0.5) }}>
                        {formatPercentage(model.accuracy)}
                      </span>
                      {model.change !== undefined && model.change !== 0 && (
                        <span
                          className="ml-2 text-xs"
                          style={{ color: getColorForValue(model.change) }}
                        >
                          {model.change > 0 ? '↑' : '↓'} {Math.abs(model.change).toFixed(1)}%
                        </span>
                      )}
                    </td>
                    <td className="px-4 py-3 text-right text-ibkr-text">
                      {formatPercentage(model.precision)}
                    </td>
                    <td className="px-4 py-3 text-right text-ibkr-text">
                      {formatPercentage(model.recall)}
                    </td>
                    <td className="px-4 py-3 text-right text-ibkr-text">
                      {formatPercentage(model.f1_score)}
                    </td>
                    <td className="px-4 py-3 text-center">
                      <span
                        className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium"
                        style={{
                          backgroundColor: getStatusColor(model.status) + '20',
                          color: getStatusColor(model.status)
                        }}
                      >
                        <span className="mr-1">{getStatusIcon(model.status)}</span>
                        {model.status}
                      </span>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Model Comparison Chart */}
      <div className="bg-ibkr-surface rounded border border-ibkr-border p-4">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-sm font-bold text-ibkr-text">Model Comparison</h3>
          <div className="flex space-x-2">
            {['7d', '30d', '90d'].map(tf => (
              <button
                key={tf}
                onClick={() => setTimeframe(tf)}
                className={`px-2 py-1 text-xs rounded transition-colors ${
                  timeframe === tf
                    ? 'bg-ibkr-accent text-white'
                    : 'bg-ibkr-bg text-ibkr-text-secondary hover:text-ibkr-text'
                }`}
              >
                {tf}
              </button>
            ))}
          </div>
        </div>

        {comparisonData.length > 0 ? (
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={comparisonData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#3e3e42" />
              <XAxis
                dataKey="date"
                stroke="#888888"
                tick={{ fill: '#888888', fontSize: 10 }}
              />
              <YAxis
                stroke="#888888"
                tick={{ fill: '#888888', fontSize: 10 }}
                domain={[0, 100]}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#252526',
                  border: '1px solid #3e3e42',
                  borderRadius: '4px',
                  fontSize: '10px'
                }}
              />
              <Legend wrapperStyle={{ fontSize: '10px' }} />
              <Line
                type="monotone"
                dataKey="ensemble"
                stroke="#4ec9b0"
                strokeWidth={2}
                dot={{ r: 3 }}
                name="Ensemble"
              />
              <Line
                type="monotone"
                dataKey="random_forest"
                stroke="#007acc"
                strokeWidth={2}
                dot={{ r: 3 }}
                name="Random Forest"
              />
              <Line
                type="monotone"
                dataKey="xgboost"
                stroke="#dcdcaa"
                strokeWidth={2}
                dot={{ r: 3 }}
                name="XGBoost"
              />
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <div className="h-64 flex items-center justify-center text-ibkr-text-secondary text-xs">
            {loading ? 'Loading comparison data...' : 'No comparison data available'}
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Real-time Accuracy Section */}
        <div className="bg-ibkr-surface rounded border border-ibkr-border p-4">
          <h3 className="text-sm font-bold text-ibkr-text mb-4">Today's Predictions</h3>

          <div className="mb-4">
            <div className="flex justify-between text-xs text-ibkr-text-secondary mb-2">
              <span>Overall Win Rate</span>
              <span className="font-bold text-ibkr-text">
                {mockTodayStats.correct}/{mockTodayStats.total} ({formatPercentage(mockTodayStats.winRate)})
              </span>
            </div>
            <div className="w-full bg-ibkr-bg rounded-full h-2">
              <div
                className="h-2 rounded-full transition-all"
                style={{
                  width: `${mockTodayStats.winRate * 100}%`,
                  backgroundColor: mockTodayStats.winRate >= 0.75 ? '#4ec9b0' : '#dcdcaa'
                }}
              />
            </div>
            <div className="flex items-center mt-2 text-xs">
              {mockTodayStats.winRate >= 0.75 && (
                <>
                  <span className="text-ibkr-success mr-1">⭐</span>
                  <span className="text-ibkr-success">Excellent performance today!</span>
                </>
              )}
            </div>
          </div>

          <div className="space-y-2">
            <div className="text-xs text-ibkr-text-secondary mb-2">By Strategy:</div>
            {mockTodayStats.byStrategy.map((strat, idx) => (
              <div key={idx} className="bg-ibkr-bg p-2 rounded">
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-ibkr-text">{strat.name}</span>
                  <span className="text-ibkr-text-secondary">
                    {strat.correct}/{strat.total}
                  </span>
                </div>
                <div className="w-full bg-ibkr-surface rounded-full h-1.5">
                  <div
                    className="h-1.5 rounded-full"
                    style={{
                      width: `${strat.rate * 100}%`,
                      backgroundColor: strat.rate >= 0.75 ? '#4ec9b0' : strat.rate >= 0.5 ? '#dcdcaa' : '#f48771'
                    }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Claude AI Analysis */}
        <div className="bg-ibkr-surface rounded border border-ibkr-border p-4">
          <div className="flex items-center mb-4">
            <span className="mr-2">🧠</span>
            <h3 className="text-sm font-bold text-ibkr-text">Claude AI Analysis</h3>
          </div>

          {claudeAnalysis ? (
            <div className="space-y-3">
              {claudeAnalysis.assessment && (
                <div className="flex items-center space-x-2">
                  <span className="text-xs text-ibkr-text-secondary">Overall:</span>
                  <span
                    className="px-2 py-0.5 rounded text-xs font-bold"
                    style={{
                      backgroundColor:
                        claudeAnalysis.assessment === 'Excellent'
                          ? '#4ec9b020'
                          : claudeAnalysis.assessment === 'Good'
                          ? '#007acc20'
                          : '#dcdcaa20',
                      color:
                        claudeAnalysis.assessment === 'Excellent'
                          ? '#4ec9b0'
                          : claudeAnalysis.assessment === 'Good'
                          ? '#007acc'
                          : '#dcdcaa'
                    }}
                  >
                    {claudeAnalysis.assessment}
                  </span>
                </div>
              )}

              {claudeAnalysis.strengths && claudeAnalysis.strengths.length > 0 && (
                <div>
                  <div className="text-xs font-bold text-ibkr-success mb-1">✓ Strengths:</div>
                  <ul className="space-y-1">
                    {claudeAnalysis.strengths.map((strength, idx) => (
                      <li key={idx} className="text-xs text-ibkr-text-secondary flex items-start">
                        <span className="mr-2">•</span>
                        <span>{strength}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {claudeAnalysis.weaknesses && claudeAnalysis.weaknesses.length > 0 && (
                <div>
                  <div className="text-xs font-bold text-ibkr-warning mb-1">⚠ Weaknesses:</div>
                  <ul className="space-y-1">
                    {claudeAnalysis.weaknesses.map((weakness, idx) => (
                      <li key={idx} className="text-xs text-ibkr-text-secondary flex items-start">
                        <span className="mr-2">•</span>
                        <span>{weakness}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {claudeAnalysis.recommendations && claudeAnalysis.recommendations.length > 0 && (
                <div>
                  <div className="text-xs font-bold text-ibkr-accent mb-1">💡 Recommendations:</div>
                  <ol className="space-y-1">
                    {claudeAnalysis.recommendations.map((rec, idx) => (
                      <li key={idx} className="text-xs text-ibkr-text-secondary flex items-start">
                        <span className="mr-2">{idx + 1}.</span>
                        <span>{rec}</span>
                      </li>
                    ))}
                  </ol>
                </div>
              )}
            </div>
          ) : (
            <div className="text-xs text-ibkr-text-secondary text-center py-8">
              {loading ? 'Loading analysis...' : 'No analysis available'}
            </div>
          )}
        </div>
      </div>

      {/* Action Buttons */}
      <div className="flex space-x-3">
        <button className="px-4 py-2 bg-ibkr-accent text-white text-xs rounded hover:bg-blue-600 transition-colors">
          Export Report
        </button>
        <button className="px-4 py-2 bg-ibkr-bg text-ibkr-text text-xs rounded border border-ibkr-border hover:border-ibkr-accent transition-colors">
          Retrain Selected
        </button>
        <button className="px-4 py-2 bg-ibkr-bg text-ibkr-text text-xs rounded border border-ibkr-border hover:border-ibkr-accent transition-colors">
          Model Settings
        </button>
      </div>
    </div>
  );
};

export default PerformanceDashboard;
