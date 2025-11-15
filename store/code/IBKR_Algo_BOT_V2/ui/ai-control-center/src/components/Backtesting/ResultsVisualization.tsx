import React, { useState, useEffect } from 'react';
import { BacktestResults } from '../../types/models';
import { apiService } from '../../services/api';
import TradeAnalysis from './TradeAnalysis';
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from 'recharts';
import { formatCurrency, formatPercentage, downloadFile } from '../../utils/helpers';

interface ResultsVisualizationProps {
  results: BacktestResults;
}

export const ResultsVisualization: React.FC<ResultsVisualizationProps> = ({ results }) => {
  const [claudeAnalysis, setClaudeAnalysis] = useState<any>(null);
  const [loadingAnalysis, setLoadingAnalysis] = useState(false);
  const [activeTab, setActiveTab] = useState<'overview' | 'equity' | 'trades' | 'detailed' | 'analysis'>('overview');

  useEffect(() => {
    fetchClaudeAnalysis();
  }, [results.backtest_id]);

  const fetchClaudeAnalysis = async () => {
    setLoadingAnalysis(true);
    try {
      const response = await apiService.getBacktestAnalysis(results.backtest_id);
      setClaudeAnalysis(response.data);
    } catch (err) {
      console.error('Error fetching Claude analysis:', err);
    } finally {
      setLoadingAnalysis(false);
    }
  };

  const handleExport = async (format: 'pdf' | 'csv' | 'json') => {
    try {
      const blob = await apiService.exportBacktest(results.backtest_id, format);
      downloadFile(blob, `backtest_${results.backtest_id}.${format}`);
    } catch (err) {
      console.error('Error exporting backtest:', err);
    }
  };

  const { performance } = results;

  // Prepare monthly returns heatmap data
  const monthNames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  const yearlyReturns: { [year: number]: { [month: number]: number } } = {};

  results.monthly_returns.forEach(mr => {
    if (!yearlyReturns[mr.year]) {
      yearlyReturns[mr.year] = {};
    }
    yearlyReturns[mr.year][mr.month] = mr.return_pct;
  });

  // Prepare trade distribution data
  const winningTrades = results.trades.filter(t => t.winner);
  const losingTrades = results.trades.filter(t => !t.winner);

  const tradeDistribution = [
    { name: 'Wins', count: winningTrades.length, fill: '#4ec9b0' },
    { name: 'Losses', count: losingTrades.length, fill: '#f48771' }
  ];

  const tabs = [
    { id: 'overview' as const, label: 'Overview', icon: '📊' },
    { id: 'equity' as const, label: 'Equity Curve', icon: '📈' },
    { id: 'trades' as const, label: 'Trade Summary', icon: '💹' },
    { id: 'detailed' as const, label: 'Trade Details', icon: '🔍' },
    { id: 'analysis' as const, label: 'Claude Insights', icon: '🧠' }
  ];

  return (
    <div className="bg-ibkr-surface border border-ibkr-border rounded">
      {/* Header with Tabs */}
      <div className="border-b border-ibkr-border">
        <div className="flex items-center justify-between p-4">
          <h2 className="text-lg font-bold text-ibkr-text">
            Backtest Results
          </h2>
          <div className="flex items-center space-x-2">
            <button
              onClick={() => handleExport('pdf')}
              className="px-3 py-1.5 text-xs bg-ibkr-bg text-ibkr-text border border-ibkr-border rounded hover:bg-opacity-80 transition-colors"
            >
              Export PDF
            </button>
            <button
              onClick={() => handleExport('csv')}
              className="px-3 py-1.5 text-xs bg-ibkr-bg text-ibkr-text border border-ibkr-border rounded hover:bg-opacity-80 transition-colors"
            >
              Export CSV
            </button>
            <button
              onClick={() => handleExport('json')}
              className="px-3 py-1.5 text-xs bg-ibkr-bg text-ibkr-text border border-ibkr-border rounded hover:bg-opacity-80 transition-colors"
            >
              Export JSON
            </button>
          </div>
        </div>

        {/* Tabs */}
        <div className="flex space-x-1 px-4">
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
      <div className="p-6">
        {/* Overview Tab */}
        {activeTab === 'overview' && (
          <div className="space-y-6">
            {/* Performance Metrics Grid */}
            <div className="grid grid-cols-4 gap-4">
              <div className="bg-ibkr-bg p-4 rounded border border-ibkr-border">
                <p className="text-xs text-ibkr-text-secondary mb-1">Total Return</p>
                <p className={`text-2xl font-bold ${
                  performance.total_return >= 0 ? 'text-ibkr-success' : 'text-ibkr-error'
                }`}>
                  {performance.total_return >= 0 ? '+' : ''}{performance.total_return.toFixed(2)}%
                </p>
              </div>

              <div className="bg-ibkr-bg p-4 rounded border border-ibkr-border">
                <p className="text-xs text-ibkr-text-secondary mb-1">CAGR</p>
                <p className="text-2xl font-bold text-ibkr-text">
                  {performance.cagr.toFixed(2)}%
                </p>
              </div>

              <div className="bg-ibkr-bg p-4 rounded border border-ibkr-border">
                <p className="text-xs text-ibkr-text-secondary mb-1">Sharpe Ratio</p>
                <p className="text-2xl font-bold text-ibkr-text">
                  {performance.sharpe_ratio.toFixed(2)}
                </p>
              </div>

              <div className="bg-ibkr-bg p-4 rounded border border-ibkr-border">
                <p className="text-xs text-ibkr-text-secondary mb-1">Max Drawdown</p>
                <p className="text-2xl font-bold text-ibkr-error">
                  -{performance.max_drawdown.toFixed(2)}%
                </p>
              </div>

              <div className="bg-ibkr-bg p-4 rounded border border-ibkr-border">
                <p className="text-xs text-ibkr-text-secondary mb-1">Win Rate</p>
                <p className="text-2xl font-bold text-ibkr-text">
                  {performance.win_rate.toFixed(1)}%
                </p>
              </div>

              <div className="bg-ibkr-bg p-4 rounded border border-ibkr-border">
                <p className="text-xs text-ibkr-text-secondary mb-1">Profit Factor</p>
                <p className="text-2xl font-bold text-ibkr-text">
                  {performance.profit_factor.toFixed(2)}
                </p>
              </div>

              <div className="bg-ibkr-bg p-4 rounded border border-ibkr-border">
                <p className="text-xs text-ibkr-text-secondary mb-1">Sortino Ratio</p>
                <p className="text-2xl font-bold text-ibkr-text">
                  {performance.sortino_ratio.toFixed(2)}
                </p>
              </div>

              <div className="bg-ibkr-bg p-4 rounded border border-ibkr-border">
                <p className="text-xs text-ibkr-text-secondary mb-1">Total Trades</p>
                <p className="text-2xl font-bold text-ibkr-text">
                  {performance.total_trades}
                </p>
              </div>
            </div>

            {/* Trade Distribution */}
            <div className="bg-ibkr-bg p-4 rounded border border-ibkr-border">
              <h3 className="text-sm font-semibold text-ibkr-text mb-4">Trade Distribution</h3>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={tradeDistribution}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#3e3e42" />
                  <XAxis dataKey="name" stroke="#888888" style={{ fontSize: '11px' }} />
                  <YAxis stroke="#888888" style={{ fontSize: '11px' }} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#252526',
                      border: '1px solid #3e3e42',
                      borderRadius: '4px',
                      fontSize: '11px'
                    }}
                  />
                  <Bar dataKey="count" fill="#007acc" />
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Monthly Returns Heatmap */}
            <div className="bg-ibkr-bg p-4 rounded border border-ibkr-border">
              <h3 className="text-sm font-semibold text-ibkr-text mb-4">Monthly Returns Heatmap</h3>
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr>
                      <th className="text-left text-ibkr-text-secondary p-2">Year</th>
                      {monthNames.map(month => (
                        <th key={month} className="text-center text-ibkr-text-secondary p-2">
                          {month}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {Object.keys(yearlyReturns).sort().reverse().map(year => (
                      <tr key={year}>
                        <td className="text-ibkr-text p-2 font-semibold">{year}</td>
                        {[...Array(12)].map((_, monthIndex) => {
                          const returnValue = yearlyReturns[parseInt(year)]?.[monthIndex + 1];
                          const bgColor = returnValue
                            ? returnValue > 0
                              ? `rgba(78, 201, 176, ${Math.min(Math.abs(returnValue) / 10, 0.8)})`
                              : `rgba(244, 135, 113, ${Math.min(Math.abs(returnValue) / 10, 0.8)})`
                            : 'transparent';

                          return (
                            <td
                              key={monthIndex}
                              className="text-center p-2 border border-ibkr-border"
                              style={{ backgroundColor: bgColor }}
                            >
                              {returnValue ? `${returnValue.toFixed(1)}%` : '-'}
                            </td>
                          );
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* Equity Curve Tab */}
        {activeTab === 'equity' && (
          <div className="space-y-6">
            {/* Equity Curve Chart */}
            <div className="bg-ibkr-bg p-4 rounded border border-ibkr-border">
              <h3 className="text-sm font-semibold text-ibkr-text mb-4">Portfolio Value Over Time</h3>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={results.equity_curve}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#3e3e42" />
                  <XAxis
                    dataKey="timestamp"
                    stroke="#888888"
                    style={{ fontSize: '11px' }}
                    tickFormatter={(value) => new Date(value).toLocaleDateString()}
                  />
                  <YAxis
                    stroke="#888888"
                    style={{ fontSize: '11px' }}
                    tickFormatter={(value) => `$${(value / 1000).toFixed(0)}k`}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#252526',
                      border: '1px solid #3e3e42',
                      borderRadius: '4px',
                      fontSize: '11px'
                    }}
                    labelFormatter={(value) => new Date(value).toLocaleString()}
                    formatter={(value: number) => [`$${value.toFixed(2)}`, 'Portfolio Value']}
                  />
                  <Legend wrapperStyle={{ fontSize: '11px' }} />
                  <Line
                    type="monotone"
                    dataKey="value"
                    stroke="#4ec9b0"
                    strokeWidth={2}
                    dot={false}
                    name="Portfolio Value"
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>

            {/* Drawdown Chart */}
            <div className="bg-ibkr-bg p-4 rounded border border-ibkr-border">
              <h3 className="text-sm font-semibold text-ibkr-text mb-4">Drawdown Analysis</h3>
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={results.drawdown_data}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#3e3e42" />
                  <XAxis
                    dataKey="timestamp"
                    stroke="#888888"
                    style={{ fontSize: '11px' }}
                    tickFormatter={(value) => new Date(value).toLocaleDateString()}
                  />
                  <YAxis
                    stroke="#888888"
                    style={{ fontSize: '11px' }}
                    tickFormatter={(value) => `${value.toFixed(0)}%`}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#252526',
                      border: '1px solid #3e3e42',
                      borderRadius: '4px',
                      fontSize: '11px'
                    }}
                    labelFormatter={(value) => new Date(value).toLocaleString()}
                    formatter={(value: number) => [`${value.toFixed(2)}%`, 'Drawdown']}
                  />
                  <Legend wrapperStyle={{ fontSize: '11px' }} />
                  <Line
                    type="monotone"
                    dataKey="drawdown_pct"
                    stroke="#f48771"
                    strokeWidth={2}
                    dot={false}
                    name="Drawdown %"
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {/* Trade Analysis Tab */}
        {activeTab === 'trades' && (
          <div className="space-y-6">
            {/* Trade Statistics */}
            <div className="grid grid-cols-3 gap-4">
              <div className="bg-ibkr-bg p-4 rounded border border-ibkr-border">
                <p className="text-xs text-ibkr-text-secondary mb-1">Average Win</p>
                <p className="text-xl font-bold text-ibkr-success">
                  {formatCurrency(performance.avg_win)}
                </p>
              </div>

              <div className="bg-ibkr-bg p-4 rounded border border-ibkr-border">
                <p className="text-xs text-ibkr-text-secondary mb-1">Average Loss</p>
                <p className="text-xl font-bold text-ibkr-error">
                  {formatCurrency(performance.avg_loss)}
                </p>
              </div>

              <div className="bg-ibkr-bg p-4 rounded border border-ibkr-border">
                <p className="text-xs text-ibkr-text-secondary mb-1">Expectancy</p>
                <p className="text-xl font-bold text-ibkr-text">
                  {formatCurrency(performance.expectancy)}
                </p>
              </div>
            </div>

            {/* Recent Trades Table */}
            <div className="bg-ibkr-bg p-4 rounded border border-ibkr-border">
              <h3 className="text-sm font-semibold text-ibkr-text mb-4">Recent Trades</h3>
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="border-b border-ibkr-border">
                      <th className="text-left text-ibkr-text-secondary p-2">Date</th>
                      <th className="text-left text-ibkr-text-secondary p-2">Symbol</th>
                      <th className="text-right text-ibkr-text-secondary p-2">Entry</th>
                      <th className="text-right text-ibkr-text-secondary p-2">Exit</th>
                      <th className="text-right text-ibkr-text-secondary p-2">P&L</th>
                      <th className="text-right text-ibkr-text-secondary p-2">P&L %</th>
                      <th className="text-center text-ibkr-text-secondary p-2">Result</th>
                    </tr>
                  </thead>
                  <tbody>
                    {results.trades.slice(0, 20).map((trade) => (
                      <tr key={trade.trade_id} className="border-b border-ibkr-border hover:bg-ibkr-surface">
                        <td className="text-ibkr-text p-2">
                          {new Date(trade.date).toLocaleDateString()}
                        </td>
                        <td className="text-ibkr-text p-2 font-semibold">{trade.symbol}</td>
                        <td className="text-ibkr-text p-2 text-right">
                          ${trade.entry_price.toFixed(2)}
                        </td>
                        <td className="text-ibkr-text p-2 text-right">
                          ${trade.exit_price.toFixed(2)}
                        </td>
                        <td className={`p-2 text-right font-semibold ${
                          trade.pnl >= 0 ? 'text-ibkr-success' : 'text-ibkr-error'
                        }`}>
                          {formatCurrency(trade.pnl)}
                        </td>
                        <td className={`p-2 text-right ${
                          trade.pnl_pct >= 0 ? 'text-ibkr-success' : 'text-ibkr-error'
                        }`}>
                          {trade.pnl_pct >= 0 ? '+' : ''}{trade.pnl_pct.toFixed(2)}%
                        </td>
                        <td className="p-2 text-center">
                          <span className={`px-2 py-1 rounded text-xs ${
                            trade.winner
                              ? 'bg-green-900 bg-opacity-30 text-ibkr-success'
                              : 'bg-red-900 bg-opacity-30 text-ibkr-error'
                          }`}>
                            {trade.winner ? 'Win' : 'Loss'}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* Trade Details Tab */}
        {activeTab === 'detailed' && (
          <TradeAnalysis trades={results.trades} />
        )}

        {/* Claude Analysis Tab */}
        {activeTab === 'analysis' && (
          <div className="space-y-4">
            {loadingAnalysis ? (
              <div className="text-center py-8">
                <div className="animate-spin h-8 w-8 border-4 border-ibkr-accent border-t-transparent rounded-full mx-auto mb-3"></div>
                <p className="text-sm text-ibkr-text-secondary">Analyzing backtest results...</p>
              </div>
            ) : claudeAnalysis ? (
              <>
                {/* Overall Assessment */}
                <div className="bg-ibkr-bg p-4 rounded border border-ibkr-border">
                  <div className="flex items-center justify-between mb-3">
                    <h3 className="text-sm font-semibold text-ibkr-text">Overall Assessment</h3>
                    <span className={`px-3 py-1 rounded text-xs font-semibold ${
                      claudeAnalysis.assessment === 'Excellent' ? 'bg-green-900 bg-opacity-30 text-ibkr-success' :
                      claudeAnalysis.assessment === 'Good' ? 'bg-blue-900 bg-opacity-30 text-blue-300' :
                      claudeAnalysis.assessment === 'Fair' ? 'bg-yellow-900 bg-opacity-30 text-ibkr-warning' :
                      'bg-red-900 bg-opacity-30 text-ibkr-error'
                    }`}>
                      {claudeAnalysis.assessment}
                    </span>
                  </div>
                  <p className="text-sm text-ibkr-text">{claudeAnalysis.summary}</p>
                </div>

                {/* Strengths */}
                <div className="bg-ibkr-bg p-4 rounded border border-ibkr-border">
                  <h3 className="text-sm font-semibold text-ibkr-text mb-3">Strengths</h3>
                  <ul className="space-y-2">
                    {claudeAnalysis.strengths.map((strength: string, index: number) => (
                      <li key={index} className="flex items-start text-sm text-ibkr-text">
                        <span className="text-ibkr-success mr-2">✓</span>
                        <span>{strength}</span>
                      </li>
                    ))}
                  </ul>
                </div>

                {/* Weaknesses */}
                <div className="bg-ibkr-bg p-4 rounded border border-ibkr-border">
                  <h3 className="text-sm font-semibold text-ibkr-text mb-3">Areas for Improvement</h3>
                  <ul className="space-y-2">
                    {claudeAnalysis.weaknesses.map((weakness: string, index: number) => (
                      <li key={index} className="flex items-start text-sm text-ibkr-text">
                        <span className="text-ibkr-warning mr-2">⚠</span>
                        <span>{weakness}</span>
                      </li>
                    ))}
                  </ul>
                </div>

                {/* Recommendations */}
                <div className="bg-ibkr-bg p-4 rounded border border-ibkr-border">
                  <h3 className="text-sm font-semibold text-ibkr-text mb-3">Recommendations</h3>
                  <ol className="space-y-2 list-decimal list-inside">
                    {claudeAnalysis.recommendations.map((rec: string, index: number) => (
                      <li key={index} className="text-sm text-ibkr-text">
                        {rec}
                      </li>
                    ))}
                  </ol>
                </div>
              </>
            ) : (
              <div className="text-center py-8">
                <p className="text-sm text-ibkr-text-secondary">No analysis available</p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default ResultsVisualization;
