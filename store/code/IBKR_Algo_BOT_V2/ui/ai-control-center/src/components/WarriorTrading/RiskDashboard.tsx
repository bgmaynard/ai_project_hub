import React, { useState, useEffect } from 'react';

interface RiskStatus {
  is_halted: boolean;
  halt_reason: string | null;
  open_positions: number;
  total_trades: number;
  winning_trades: number;
  losing_trades: number;
  win_rate: number;
  current_pnl: number;
  avg_win: number;
  avg_loss: number;
  avg_r_multiple: number;
  consecutive_wins: number;
  consecutive_losses: number;
  distance_to_goal: number;
  best_trade?: {
    symbol: string;
    pnl: number;
    r_multiple: number;
  };
}

/**
 * Risk Dashboard Component
 *
 * Displays risk management status and daily statistics
 */
const RiskDashboard: React.FC = () => {
  const [riskStatus, setRiskStatus] = useState<RiskStatus | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    fetchRiskStatus();
    // Poll every 10 seconds
    const interval = setInterval(fetchRiskStatus, 10000);
    return () => clearInterval(interval);
  }, []);

  const fetchRiskStatus = async () => {
    try {
      const response = await fetch('http://localhost:9101/api/warrior/risk/status');
      const data = await response.json();
      setRiskStatus(data);
      setIsLoading(false);
    } catch (error) {
      console.error('Error fetching risk status:', error);
      setIsLoading(false);
    }
  };

  const getPnlColor = (pnl: number) => {
    if (pnl > 0) return 'text-green-500';
    if (pnl < 0) return 'text-red-500';
    return 'text-ibkr-text-secondary';
  };

  const getProgressColor = (distance: number) => {
    if (distance <= 0) return 'bg-green-500'; // Goal reached
    if (distance <= 100) return 'bg-yellow-500'; // Close to goal
    return 'bg-blue-500'; // Still working
  };

  const getProgressPercentage = (currentPnl: number, goal: number = 200) => {
    return Math.min((currentPnl / goal) * 100, 100);
  };

  if (isLoading) {
    return (
      <div className="bg-ibkr-surface rounded-lg border border-ibkr-border p-8 text-center">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-ibkr-accent mx-auto"></div>
        <p className="mt-2 text-sm text-ibkr-text-secondary">Loading...</p>
      </div>
    );
  }

  if (!riskStatus) {
    return (
      <div className="bg-ibkr-surface rounded-lg border border-ibkr-border p-8 text-center">
        <p className="text-ibkr-text-secondary">No risk data available</p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Risk Manager Status */}
      <div className="bg-ibkr-surface rounded-lg border border-ibkr-border">
        <div className="p-4 border-b border-ibkr-border">
          <h2 className="text-lg font-semibold text-ibkr-text flex items-center">
            <span className="mr-2">⚖️</span>
            Risk Manager
          </h2>
        </div>

        <div className="p-4">
          {/* Trading Status */}
          {riskStatus.is_halted ? (
            <div className="mb-4 p-3 bg-red-500 bg-opacity-10 border border-red-500 rounded">
              <div className="flex items-center space-x-2 text-red-500 font-medium mb-1">
                <span>🛑</span>
                <span>Trading Halted</span>
              </div>
              <p className="text-xs text-red-400">{riskStatus.halt_reason}</p>
            </div>
          ) : (
            <div className="mb-4 p-3 bg-green-500 bg-opacity-10 border border-green-500 rounded">
              <div className="flex items-center space-x-2 text-green-500 font-medium">
                <span>✅</span>
                <span>Trading Active</span>
              </div>
            </div>
          )}

          {/* Daily P&L */}
          <div className="mb-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-ibkr-text-secondary">Daily P&L</span>
              <span className={`text-2xl font-bold ${getPnlColor(riskStatus.current_pnl)}`}>
                ${riskStatus.current_pnl > 0 ? '+' : ''}{riskStatus.current_pnl.toFixed(2)}
              </span>
            </div>

            {/* Progress to Goal */}
            <div className="mb-1">
              <div className="flex justify-between text-xs text-ibkr-text-secondary mb-1">
                <span>Progress to $200 goal</span>
                <span>{getProgressPercentage(riskStatus.current_pnl).toFixed(0)}%</span>
              </div>
              <div className="w-full bg-ibkr-bg rounded-full h-2">
                <div
                  className={`h-2 rounded-full transition-all ${getProgressColor(riskStatus.distance_to_goal)}`}
                  style={{ width: `${getProgressPercentage(riskStatus.current_pnl)}%` }}
                ></div>
              </div>
            </div>

            {riskStatus.distance_to_goal > 0 ? (
              <p className="text-xs text-ibkr-text-secondary mt-1">
                ${riskStatus.distance_to_goal.toFixed(2)} to goal
              </p>
            ) : (
              <p className="text-xs text-green-500 mt-1">
                🎉 Goal reached!
              </p>
            )}
          </div>

          {/* Statistics Grid */}
          <div className="grid grid-cols-2 gap-3">
            <div className="p-3 bg-ibkr-bg rounded">
              <div className="text-xs text-ibkr-text-secondary mb-1">Win Rate</div>
              <div className="text-lg font-semibold text-ibkr-text">
                {riskStatus.win_rate.toFixed(1)}%
              </div>
            </div>

            <div className="p-3 bg-ibkr-bg rounded">
              <div className="text-xs text-ibkr-text-secondary mb-1">Total Trades</div>
              <div className="text-lg font-semibold text-ibkr-text">
                {riskStatus.total_trades}
              </div>
            </div>

            <div className="p-3 bg-ibkr-bg rounded">
              <div className="text-xs text-ibkr-text-secondary mb-1">Avg Win</div>
              <div className="text-lg font-semibold text-green-500">
                +${riskStatus.avg_win.toFixed(2)}
              </div>
            </div>

            <div className="p-3 bg-ibkr-bg rounded">
              <div className="text-xs text-ibkr-text-secondary mb-1">Avg Loss</div>
              <div className="text-lg font-semibold text-red-500">
                ${riskStatus.avg_loss.toFixed(2)}
              </div>
            </div>

            <div className="p-3 bg-ibkr-bg rounded">
              <div className="text-xs text-ibkr-text-secondary mb-1">Avg R</div>
              <div className={`text-lg font-semibold ${getPnlColor(riskStatus.avg_r_multiple)}`}>
                {riskStatus.avg_r_multiple > 0 ? '+' : ''}{riskStatus.avg_r_multiple.toFixed(2)}R
              </div>
            </div>

            <div className="p-3 bg-ibkr-bg rounded">
              <div className="text-xs text-ibkr-text-secondary mb-1">Open Positions</div>
              <div className="text-lg font-semibold text-ibkr-text">
                {riskStatus.open_positions}
              </div>
            </div>
          </div>

          {/* Streaks */}
          <div className="mt-3 grid grid-cols-2 gap-3">
            {riskStatus.consecutive_wins > 0 && (
              <div className="p-2 bg-green-500 bg-opacity-10 border border-green-500 rounded text-center">
                <div className="text-xs text-green-500">Win Streak</div>
                <div className="text-lg font-bold text-green-500">{riskStatus.consecutive_wins}</div>
              </div>
            )}

            {riskStatus.consecutive_losses > 0 && (
              <div className="p-2 bg-red-500 bg-opacity-10 border border-red-500 rounded text-center">
                <div className="text-xs text-red-500">Loss Streak</div>
                <div className="text-lg font-bold text-red-500">{riskStatus.consecutive_losses}</div>
              </div>
            )}
          </div>

          {/* Best Trade */}
          {riskStatus.best_trade && (
            <div className="mt-3 p-3 bg-green-500 bg-opacity-10 border border-green-500 rounded">
              <div className="text-xs text-green-500 mb-1">🏆 Best Trade Today</div>
              <div className="flex items-center justify-between">
                <span className="font-semibold text-ibkr-text">{riskStatus.best_trade.symbol}</span>
                <div className="text-right">
                  <div className="text-sm font-bold text-green-500">
                    +${riskStatus.best_trade.pnl.toFixed(2)}
                  </div>
                  <div className="text-xs text-green-400">
                    {riskStatus.best_trade.r_multiple.toFixed(2)}R
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Warning for consecutive losses */}
          {riskStatus.consecutive_losses >= 3 && (
            <div className="mt-3 p-3 bg-yellow-500 bg-opacity-10 border border-yellow-500 rounded">
              <div className="flex items-center space-x-2 text-yellow-500 text-xs">
                <span>⚠️</span>
                <span>Consider taking a break after 3 consecutive losses</span>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Quick Actions */}
      <div className="bg-ibkr-surface rounded-lg border border-ibkr-border p-4">
        <h3 className="text-sm font-semibold text-ibkr-text mb-3">Quick Actions</h3>
        <div className="space-y-2">
          <button
            className="w-full px-3 py-2 bg-ibkr-bg text-ibkr-text rounded hover:bg-ibkr-border transition-colors text-sm text-left"
            onClick={() => window.location.reload()}
          >
            🔄 Refresh Data
          </button>
          <button
            className="w-full px-3 py-2 bg-yellow-600 text-white rounded hover:bg-yellow-700 transition-colors text-sm text-left"
            onClick={() => {
              if (window.confirm('Reset daily statistics? This will clear all today\'s trades.')) {
                fetch('http://localhost:9101/api/warrior/risk/reset-daily', { method: 'POST' })
                  .then(() => fetchRiskStatus())
                  .catch(err => console.error(err));
              }
            }}
          >
            ⚠️ Reset Daily Stats
          </button>
        </div>
      </div>
    </div>
  );
};

export default RiskDashboard;
