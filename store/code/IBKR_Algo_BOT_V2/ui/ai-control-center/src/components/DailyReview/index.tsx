import React, { useState, useEffect } from 'react';
import apiService from '../../services/api';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';

interface DailyReviewData {
  date: string;
  summary: {
    total_trades: number;
    winning_trades: number;
    losing_trades: number;
    total_pnl: number;
    win_rate: number;
    avg_profit: number;
    avg_loss: number;
    largest_win: number;
    largest_loss: number;
    sharpe_ratio: number;
  };
  performance_by_symbol: Array<{
    symbol: string;
    trades: number;
    pnl: number;
    win_rate: number;
  }>;
  performance_by_strategy: Array<{
    strategy: string;
    trades: number;
    pnl: number;
    win_rate: number;
  }>;
  claude_insights: {
    key_findings: string[];
    strengths: string[];
    concerns: string[];
    recommendations: string[];
    market_commentary: string;
  };
}

interface ReviewHistoryItem {
  date: string;
  total_pnl: number;
  win_rate: number;
  total_trades: number;
}

const COLORS = ['#10b981', '#ef4444', '#3b82f6', '#f59e0b', '#8b5cf6', '#ec4899'];

export const DailyReview: React.FC = () => {
  const [reviewData, setReviewData] = useState<DailyReviewData | null>(null);
  const [history, setHistory] = useState<ReviewHistoryItem[]>([]);
  const [selectedDate, setSelectedDate] = useState<string>(new Date().toISOString().split('T')[0]);
  const [loading, setLoading] = useState(false);
  const [exporting, setExporting] = useState(false);

  useEffect(() => {
    loadDailyReview();
    loadReviewHistory();
  }, [selectedDate]);

  const loadDailyReview = async () => {
    setLoading(true);
    try {
      const response = await apiService.getDailyReview(selectedDate);
      if (response.success && response.data) {
        setReviewData(response.data);
      }
    } catch (error) {
      console.error('Failed to load daily review:', error);
    } finally {
      setLoading(false);
    }
  };

  const loadReviewHistory = async () => {
    try {
      const response = await apiService.getDailyReviewHistory();
      if (response.success && response.data) {
        setHistory(response.data.slice(0, 30)); // Last 30 days
      }
    } catch (error) {
      console.error('Failed to load review history:', error);
    }
  };

  const exportReview = async (format: 'pdf' | 'json') => {
    setExporting(true);
    try {
      const blob = await apiService.exportDailyReview(selectedDate, format);
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `daily-review-${selectedDate}.${format}`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Failed to export review:', error);
    } finally {
      setExporting(false);
    }
  };

  if (loading) {
    return (
      <div className="p-6">
        <div className="flex items-center justify-center h-64">
          <div className="text-ibkr-text-secondary">Loading daily review...</div>
        </div>
      </div>
    );
  }

  if (!reviewData) {
    return (
      <div className="p-6">
        <div className="flex items-center justify-center h-64">
          <div className="text-ibkr-text-secondary">No review data available for {selectedDate}</div>
        </div>
      </div>
    );
  }

  const { summary, performance_by_symbol, performance_by_strategy, claude_insights } = reviewData;

  const pieData = [
    { name: 'Winning Trades', value: summary.winning_trades, color: '#10b981' },
    { name: 'Losing Trades', value: summary.losing_trades, color: '#ef4444' }
  ];

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold text-ibkr-text">Daily Performance Review</h1>
          <p className="text-sm text-ibkr-text-secondary mt-1">
            Automated analysis and AI insights for your trading performance
          </p>
        </div>
        <div className="flex items-center space-x-3">
          <input
            type="date"
            value={selectedDate}
            onChange={(e) => setSelectedDate(e.target.value)}
            className="px-3 py-2 bg-ibkr-surface border border-ibkr-border rounded text-ibkr-text focus:outline-none focus:ring-2 focus:ring-ibkr-accent"
          />
          <button
            onClick={() => exportReview('pdf')}
            disabled={exporting}
            className="px-4 py-2 bg-ibkr-accent hover:bg-opacity-90 text-white rounded transition-colors disabled:opacity-50"
          >
            =Ä Export PDF
          </button>
          <button
            onClick={() => exportReview('json')}
            disabled={exporting}
            className="px-4 py-2 bg-ibkr-surface border border-ibkr-border hover:bg-ibkr-bg text-ibkr-text rounded transition-colors disabled:opacity-50"
          >
            =Ê Export JSON
          </button>
        </div>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-ibkr-surface border border-ibkr-border rounded-lg p-4">
          <div className="text-xs text-ibkr-text-secondary mb-1">Total P&L</div>
          <div className={`text-2xl font-bold ${summary.total_pnl >= 0 ? 'text-green-500' : 'text-red-500'}`}>
            ${summary.total_pnl.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
          </div>
          <div className="text-xs text-ibkr-text-secondary mt-1">
            {summary.total_trades} trades
          </div>
        </div>

        <div className="bg-ibkr-surface border border-ibkr-border rounded-lg p-4">
          <div className="text-xs text-ibkr-text-secondary mb-1">Win Rate</div>
          <div className="text-2xl font-bold text-ibkr-text">
            {(summary.win_rate * 100).toFixed(1)}%
          </div>
          <div className="text-xs text-ibkr-text-secondary mt-1">
            {summary.winning_trades}W / {summary.losing_trades}L
          </div>
        </div>

        <div className="bg-ibkr-surface border border-ibkr-border rounded-lg p-4">
          <div className="text-xs text-ibkr-text-secondary mb-1">Avg Profit/Loss</div>
          <div className="text-sm text-ibkr-text">
            <span className="text-green-500">+${summary.avg_profit.toFixed(2)}</span>
            {' / '}
            <span className="text-red-500">-${Math.abs(summary.avg_loss).toFixed(2)}</span>
          </div>
          <div className="text-xs text-ibkr-text-secondary mt-1">
            Profit Factor: {(summary.avg_profit / Math.abs(summary.avg_loss)).toFixed(2)}
          </div>
        </div>

        <div className="bg-ibkr-surface border border-ibkr-border rounded-lg p-4">
          <div className="text-xs text-ibkr-text-secondary mb-1">Sharpe Ratio</div>
          <div className="text-2xl font-bold text-ibkr-text">
            {summary.sharpe_ratio.toFixed(2)}
          </div>
          <div className="text-xs text-ibkr-text-secondary mt-1">
            Risk-adjusted return
          </div>
        </div>
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Win/Loss Distribution */}
        <div className="bg-ibkr-surface border border-ibkr-border rounded-lg p-4">
          <h3 className="text-sm font-semibold text-ibkr-text mb-4">Win/Loss Distribution</h3>
          <ResponsiveContainer width="100%" height={200}>
            <PieChart>
              <Pie
                data={pieData}
                cx="50%"
                cy="50%"
                innerRadius={60}
                outerRadius={80}
                paddingAngle={5}
                dataKey="value"
                label={({ name, value }) => `${name}: ${value}`}
              >
                {pieData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* Performance by Symbol */}
        <div className="bg-ibkr-surface border border-ibkr-border rounded-lg p-4">
          <h3 className="text-sm font-semibold text-ibkr-text mb-4">Performance by Symbol</h3>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={performance_by_symbol.slice(0, 5)}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2a2a2a" />
              <XAxis dataKey="symbol" stroke="#9ca3af" />
              <YAxis stroke="#9ca3af" />
              <Tooltip
                contentStyle={{ backgroundColor: '#2a2a2a', border: '1px solid #3a3a3a' }}
                labelStyle={{ color: '#e5e7eb' }}
              />
              <Bar dataKey="pnl" fill="#10b981" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Claude AI Insights */}
      <div className="bg-gradient-to-br from-ibkr-surface to-ibkr-bg border border-ibkr-border rounded-lg p-6">
        <div className="flex items-center space-x-2 mb-4">
          <div className="text-2xl">{">"}</div>
          <h3 className="text-lg font-semibold text-ibkr-text">Claude AI Insights</h3>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Key Findings */}
          <div>
            <h4 className="text-sm font-semibold text-ibkr-accent mb-2">=
 Key Findings</h4>
            <ul className="space-y-2">
              {claude_insights.key_findings.map((finding, idx) => (
                <li key={idx} className="text-sm text-ibkr-text flex items-start">
                  <span className="text-ibkr-accent mr-2">"</span>
                  <span>{finding}</span>
                </li>
              ))}
            </ul>
          </div>

          {/* Strengths */}
          <div>
            <h4 className="text-sm font-semibold text-green-500 mb-2"> Strengths</h4>
            <ul className="space-y-2">
              {claude_insights.strengths.map((strength, idx) => (
                <li key={idx} className="text-sm text-ibkr-text flex items-start">
                  <span className="text-green-500 mr-2">"</span>
                  <span>{strength}</span>
                </li>
              ))}
            </ul>
          </div>

          {/* Concerns */}
          <div>
            <h4 className="text-sm font-semibold text-yellow-500 mb-2">  Areas of Concern</h4>
            <ul className="space-y-2">
              {claude_insights.concerns.map((concern, idx) => (
                <li key={idx} className="text-sm text-ibkr-text flex items-start">
                  <span className="text-yellow-500 mr-2">"</span>
                  <span>{concern}</span>
                </li>
              ))}
            </ul>
          </div>

          {/* Recommendations */}
          <div>
            <h4 className="text-sm font-semibold text-blue-500 mb-2">=¡ Recommendations</h4>
            <ul className="space-y-2">
              {claude_insights.recommendations.map((rec, idx) => (
                <li key={idx} className="text-sm text-ibkr-text flex items-start">
                  <span className="text-blue-500 mr-2">"</span>
                  <span>{rec}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>

        {/* Market Commentary */}
        {claude_insights.market_commentary && (
          <div className="mt-6 pt-6 border-t border-ibkr-border">
            <h4 className="text-sm font-semibold text-ibkr-text mb-2">=È Market Commentary</h4>
            <p className="text-sm text-ibkr-text-secondary leading-relaxed">
              {claude_insights.market_commentary}
            </p>
          </div>
        )}
      </div>

      {/* Performance Tables */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* By Symbol */}
        <div className="bg-ibkr-surface border border-ibkr-border rounded-lg p-4">
          <h3 className="text-sm font-semibold text-ibkr-text mb-3">Performance by Symbol</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-ibkr-border">
                  <th className="text-left py-2 text-ibkr-text-secondary">Symbol</th>
                  <th className="text-right py-2 text-ibkr-text-secondary">Trades</th>
                  <th className="text-right py-2 text-ibkr-text-secondary">P&L</th>
                  <th className="text-right py-2 text-ibkr-text-secondary">Win Rate</th>
                </tr>
              </thead>
              <tbody>
                {performance_by_symbol.map((item, idx) => (
                  <tr key={idx} className="border-b border-ibkr-border">
                    <td className="py-2 text-ibkr-text font-medium">{item.symbol}</td>
                    <td className="text-right py-2 text-ibkr-text">{item.trades}</td>
                    <td className={`text-right py-2 font-medium ${item.pnl >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                      ${item.pnl.toFixed(2)}
                    </td>
                    <td className="text-right py-2 text-ibkr-text">
                      {(item.win_rate * 100).toFixed(1)}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* By Strategy */}
        <div className="bg-ibkr-surface border border-ibkr-border rounded-lg p-4">
          <h3 className="text-sm font-semibold text-ibkr-text mb-3">Performance by Strategy</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-ibkr-border">
                  <th className="text-left py-2 text-ibkr-text-secondary">Strategy</th>
                  <th className="text-right py-2 text-ibkr-text-secondary">Trades</th>
                  <th className="text-right py-2 text-ibkr-text-secondary">P&L</th>
                  <th className="text-right py-2 text-ibkr-text-secondary">Win Rate</th>
                </tr>
              </thead>
              <tbody>
                {performance_by_strategy.map((item, idx) => (
                  <tr key={idx} className="border-b border-ibkr-border">
                    <td className="py-2 text-ibkr-text font-medium">{item.strategy}</td>
                    <td className="text-right py-2 text-ibkr-text">{item.trades}</td>
                    <td className={`text-right py-2 font-medium ${item.pnl >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                      ${item.pnl.toFixed(2)}
                    </td>
                    <td className="text-right py-2 text-ibkr-text">
                      {(item.win_rate * 100).toFixed(1)}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* History Timeline */}
      {history.length > 0 && (
        <div className="bg-ibkr-surface border border-ibkr-border rounded-lg p-4">
          <h3 className="text-sm font-semibold text-ibkr-text mb-4">30-Day Performance Trend</h3>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={history}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2a2a2a" />
              <XAxis dataKey="date" stroke="#9ca3af" />
              <YAxis stroke="#9ca3af" />
              <Tooltip
                contentStyle={{ backgroundColor: '#2a2a2a', border: '1px solid #3a3a3a' }}
                labelStyle={{ color: '#e5e7eb' }}
              />
              <Bar dataKey="total_pnl" fill="#3b82f6" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
};

export default DailyReview;
