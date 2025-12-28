import React, { useState, useEffect } from 'react';

interface PerformanceData {
  total_trades: number;
  winning_trades: number;
  losing_trades: number;
  win_rate: number;
  net_pnl: number;
  avg_r_multiple: number;
  gross_profit: number;
  gross_loss: number;
  profit_factor: number;
}

/**
 * Performance Charts Component
 *
 * Visual performance metrics and statistics
 */
const PerformanceCharts: React.FC = () => {
  const [performance, setPerformance] = useState<PerformanceData | null>(null);
  const [trades, setTrades] = useState<any[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    fetchPerformance();
  }, []);

  const fetchPerformance = async () => {
    try {
      // Fetch risk status for performance data
      const statusResponse = await fetch('/api/warrior/risk/status');
      const statusData = await statusResponse.json();

      // Fetch trade history
      const tradesResponse = await fetch('/api/warrior/trades/history?status=CLOSED&limit=100');
      const tradesData = await tradesResponse.json();

      const perfData: PerformanceData = {
        total_trades: statusData.total_trades || 0,
        winning_trades: statusData.winning_trades || 0,
        losing_trades: statusData.losing_trades || 0,
        win_rate: statusData.win_rate || 0,
        net_pnl: statusData.current_pnl || 0,
        avg_r_multiple: statusData.avg_r_multiple || 0,
        gross_profit: statusData.avg_win * (statusData.winning_trades || 0),
        gross_loss: statusData.avg_loss * (statusData.losing_trades || 0),
        profit_factor: statusData.avg_win && statusData.avg_loss
          ? Math.abs(statusData.avg_win / statusData.avg_loss)
          : 0
      };

      setPerformance(perfData);
      setTrades(tradesData.trades || []);
      setIsLoading(false);
    } catch (error) {
      console.error('Error fetching performance:', error);
      setIsLoading(false);
    }
  };

  const getPnlColor = (pnl: number) => {
    if (pnl > 0) return 'text-green-500';
    if (pnl < 0) return 'text-red-500';
    return 'text-ibkr-text-secondary';
  };

  const getScoreColor = (score: number) => {
    if (score >= 80) return 'bg-green-500';
    if (score >= 60) return 'bg-yellow-500';
    if (score >= 40) return 'bg-orange-500';
    return 'bg-red-500';
  };

  if (isLoading) {
    return (
      <div className="bg-ibkr-surface rounded-lg border border-ibkr-border p-12 text-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-ibkr-accent mx-auto"></div>
        <p className="mt-4 text-ibkr-text-secondary">Loading performance data...</p>
      </div>
    );
  }

  if (!performance) {
    return (
      <div className="bg-ibkr-surface rounded-lg border border-ibkr-border p-12 text-center">
        <p className="text-ibkr-text-secondary">No performance data available</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-ibkr-surface rounded-lg border border-ibkr-border p-6">
        <h2 className="text-2xl font-bold text-ibkr-text mb-2 flex items-center">
          <span className="mr-2">📈</span>
          Performance Analytics
        </h2>
        <p className="text-ibkr-text-secondary">
          Daily trading performance and statistics
        </p>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-ibkr-surface rounded-lg border border-ibkr-border p-6">
          <div className="text-ibkr-text-secondary text-sm mb-2">Total P&L</div>
          <div className={`text-3xl font-bold ${getPnlColor(performance.net_pnl)}`}>
            {performance.net_pnl > 0 ? '+' : ''}${performance.net_pnl.toFixed(2)}
          </div>
        </div>

        <div className="bg-ibkr-surface rounded-lg border border-ibkr-border p-6">
          <div className="text-ibkr-text-secondary text-sm mb-2">Win Rate</div>
          <div className="text-3xl font-bold text-ibkr-text">
            {performance.win_rate.toFixed(1)}%
          </div>
          <div className="mt-2 w-full bg-ibkr-bg rounded-full h-2">
            <div
              className={`h-2 rounded-full ${getScoreColor(performance.win_rate)}`}
              style={{ width: `${performance.win_rate}%` }}
            ></div>
          </div>
        </div>

        <div className="bg-ibkr-surface rounded-lg border border-ibkr-border p-6">
          <div className="text-ibkr-text-secondary text-sm mb-2">Avg R Multiple</div>
          <div className={`text-3xl font-bold ${getPnlColor(performance.avg_r_multiple)}`}>
            {performance.avg_r_multiple > 0 ? '+' : ''}{performance.avg_r_multiple.toFixed(2)}R
          </div>
        </div>

        <div className="bg-ibkr-surface rounded-lg border border-ibkr-border p-6">
          <div className="text-ibkr-text-secondary text-sm mb-2">Profit Factor</div>
          <div className={`text-3xl font-bold ${
            performance.profit_factor >= 2 ? 'text-green-500' :
            performance.profit_factor >= 1 ? 'text-yellow-500' :
            'text-red-500'
          }`}>
            {performance.profit_factor.toFixed(2)}
          </div>
        </div>
      </div>

      {/* Trade Breakdown */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Win/Loss Breakdown */}
        <div className="bg-ibkr-surface rounded-lg border border-ibkr-border p-6">
          <h3 className="text-lg font-semibold text-ibkr-text mb-4">Trade Breakdown</h3>

          <div className="space-y-4">
            {/* Total Trades */}
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-ibkr-text-secondary">Total Trades</span>
                <span className="text-ibkr-text font-semibold">{performance.total_trades}</span>
              </div>
            </div>

            {/* Winners */}
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-ibkr-text-secondary">Winners</span>
                <span className="text-green-500 font-semibold">{performance.winning_trades}</span>
              </div>
              <div className="w-full bg-ibkr-bg rounded-full h-2">
                <div
                  className="bg-green-500 h-2 rounded-full"
                  style={{ width: `${(performance.winning_trades / (performance.total_trades || 1)) * 100}%` }}
                ></div>
              </div>
            </div>

            {/* Losers */}
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-ibkr-text-secondary">Losers</span>
                <span className="text-red-500 font-semibold">{performance.losing_trades}</span>
              </div>
              <div className="w-full bg-ibkr-bg rounded-full h-2">
                <div
                  className="bg-red-500 h-2 rounded-full"
                  style={{ width: `${(performance.losing_trades / (performance.total_trades || 1)) * 100}%` }}
                ></div>
              </div>
            </div>
          </div>
        </div>

        {/* P&L Breakdown */}
        <div className="bg-ibkr-surface rounded-lg border border-ibkr-border p-6">
          <h3 className="text-lg font-semibold text-ibkr-text mb-4">P&L Breakdown</h3>

          <div className="space-y-4">
            {/* Gross Profit */}
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-ibkr-text-secondary">Gross Profit</span>
                <span className="text-green-500 font-semibold">
                  +${performance.gross_profit.toFixed(2)}
                </span>
              </div>
              <div className="w-full bg-ibkr-bg rounded-full h-2">
                <div
                  className="bg-green-500 h-2 rounded-full"
                  style={{ width: '100%' }}
                ></div>
              </div>
            </div>

            {/* Gross Loss */}
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-ibkr-text-secondary">Gross Loss</span>
                <span className="text-red-500 font-semibold">
                  ${performance.gross_loss.toFixed(2)}
                </span>
              </div>
              <div className="w-full bg-ibkr-bg rounded-full h-2">
                <div
                  className="bg-red-500 h-2 rounded-full"
                  style={{
                    width: `${Math.abs(performance.gross_loss / (performance.gross_profit || 1)) * 100}%`
                  }}
                ></div>
              </div>
            </div>

            {/* Net P&L */}
            <div className="pt-3 border-t border-ibkr-border">
              <div className="flex justify-between">
                <span className="text-ibkr-text font-semibold">Net P&L</span>
                <span className={`text-lg font-bold ${getPnlColor(performance.net_pnl)}`}>
                  {performance.net_pnl > 0 ? '+' : ''}${performance.net_pnl.toFixed(2)}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Recent Trades */}
      <div className="bg-ibkr-surface rounded-lg border border-ibkr-border">
        <div className="p-4 border-b border-ibkr-border">
          <h3 className="text-lg font-semibold text-ibkr-text">Recent Trades</h3>
        </div>

        {trades.length === 0 ? (
          <div className="p-12 text-center">
            <div className="text-5xl mb-4">📊</div>
            <p className="text-ibkr-text-secondary">No trades yet today</p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-ibkr-bg text-ibkr-text-secondary text-xs uppercase tracking-wider">
                <tr>
                  <th className="px-4 py-3 text-left">Symbol</th>
                  <th className="px-4 py-3 text-left">Setup</th>
                  <th className="px-4 py-3 text-right">Entry</th>
                  <th className="px-4 py-3 text-right">Exit</th>
                  <th className="px-4 py-3 text-right">Shares</th>
                  <th className="px-4 py-3 text-right">P&L</th>
                  <th className="px-4 py-3 text-right">R</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-ibkr-border">
                {trades.slice(0, 10).map((trade, index) => (
                  <tr key={index} className="hover:bg-ibkr-bg transition-colors">
                    <td className="px-4 py-3 font-semibold text-ibkr-text">{trade.symbol}</td>
                    <td className="px-4 py-3 text-sm text-ibkr-text-secondary">{trade.setup_type}</td>
                    <td className="px-4 py-3 text-right text-ibkr-text">${trade.entry_price.toFixed(2)}</td>
                    <td className="px-4 py-3 text-right text-ibkr-text">${trade.exit_price?.toFixed(2) || '-'}</td>
                    <td className="px-4 py-3 text-right text-ibkr-text">{trade.shares}</td>
                    <td className={`px-4 py-3 text-right font-semibold ${getPnlColor(trade.pnl)}`}>
                      {trade.pnl > 0 ? '+' : ''}${trade.pnl?.toFixed(2) || '-'}
                    </td>
                    <td className={`px-4 py-3 text-right font-medium ${getPnlColor(trade.pnl)}`}>
                      {trade.r_multiple > 0 ? '+' : ''}{trade.r_multiple?.toFixed(2) || '-'}R
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Performance Guidelines */}
      <div className="bg-gradient-to-r from-blue-600 to-blue-700 rounded-lg p-6 text-white">
        <h3 className="text-lg font-semibold mb-3">Warrior Trading Benchmarks</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
          <div>
            <div className="text-blue-100 mb-1">Target Win Rate</div>
            <div className="font-semibold">50-60%</div>
          </div>
          <div>
            <div className="text-blue-100 mb-1">Target Profit Factor</div>
            <div className="font-semibold">≥ 1.5</div>
          </div>
          <div>
            <div className="text-blue-100 mb-1">Target Avg R</div>
            <div className="font-semibold">≥ 1.5R</div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PerformanceCharts;
