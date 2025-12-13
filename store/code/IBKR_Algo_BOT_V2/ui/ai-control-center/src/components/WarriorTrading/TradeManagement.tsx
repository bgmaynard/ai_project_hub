import React, { useState, useEffect } from 'react';

interface Trade {
  id: number;
  trade_id: string;
  symbol: string;
  setup_type: string;
  entry_time: string;
  entry_price: number;
  shares: number;
  stop_price: number;
  target_price: number;
  exit_time?: string;
  exit_price?: number;
  exit_reason?: string;
  pnl?: number;
  pnl_percent?: number;
  r_multiple?: number;
  status: 'OPEN' | 'CLOSED';
}

/**
 * Trade Management Component
 *
 * View and manage all trades (open and closed)
 */
const TradeManagement: React.FC = () => {
  const [trades, setTrades] = useState<Trade[]>([]);
  const [filter, setFilter] = useState<'all' | 'open' | 'closed'>('all');
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    fetchTrades();
    // Refresh every 30 seconds
    const interval = setInterval(fetchTrades, 30000);
    return () => clearInterval(interval);
  }, [filter]);

  const fetchTrades = async () => {
    try {
      const params = filter !== 'all' ? `?status=${filter.toUpperCase()}` : '';
      const response = await fetch(`http://localhost:9101/api/warrior/trades/history${params}`);
      const data = await response.json();

      if (data.success && data.trades) {
        setTrades(data.trades);
      }
      setIsLoading(false);
    } catch (error) {
      console.error('Error fetching trades:', error);
      setIsLoading(false);
    }
  };

  const getPnlColor = (pnl: number | undefined) => {
    if (!pnl) return 'text-ibkr-text-secondary';
    if (pnl > 0) return 'text-green-500';
    if (pnl < 0) return 'text-red-500';
    return 'text-ibkr-text-secondary';
  };

  const getSetupIcon = (setupType: string) => {
    switch (setupType) {
      case 'BULL_FLAG': return '🚩';
      case 'HOD_BREAKOUT': return '🔝';
      case 'WHOLE_DOLLAR_BREAKOUT': return '💲';
      case 'MICRO_PULLBACK': return '🔄';
      case 'HAMMER_REVERSAL': return '🔨';
      default: return '📊';
    }
  };

  const formatSetupName = (setupType: string) => {
    return setupType.split('_').map(word =>
      word.charAt(0) + word.slice(1).toLowerCase()
    ).join(' ');
  };

  const formatTime = (timeString: string) => {
    const date = new Date(timeString);
    return date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
  };

  const openTrades = trades.filter(t => t.status === 'OPEN');
  const closedTrades = trades.filter(t => t.status === 'CLOSED');

  return (
    <div className="bg-ibkr-surface rounded-lg border border-ibkr-border">
      {/* Header */}
      <div className="p-4 border-b border-ibkr-border">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-lg font-semibold text-ibkr-text flex items-center">
              <span className="mr-2">📊</span>
              Trade Management
            </h2>
            <p className="text-sm text-ibkr-text-secondary mt-1">
              View and manage your trades
            </p>
          </div>

          <button
            onClick={fetchTrades}
            className="px-3 py-1.5 bg-ibkr-bg text-ibkr-text rounded hover:bg-ibkr-border transition-colors text-sm"
          >
            🔄 Refresh
          </button>
        </div>
      </div>

      {/* Filters */}
      <div className="p-3 border-b border-ibkr-border bg-ibkr-bg flex items-center justify-between">
        <div className="flex space-x-1">
          <button
            onClick={() => setFilter('all')}
            className={`px-3 py-1.5 text-xs rounded ${
              filter === 'all'
                ? 'bg-ibkr-accent text-white'
                : 'bg-ibkr-surface text-ibkr-text-secondary hover:text-ibkr-text'
            }`}
          >
            All ({trades.length})
          </button>
          <button
            onClick={() => setFilter('open')}
            className={`px-3 py-1.5 text-xs rounded ${
              filter === 'open'
                ? 'bg-ibkr-accent text-white'
                : 'bg-ibkr-surface text-ibkr-text-secondary hover:text-ibkr-text'
            }`}
          >
            Open ({openTrades.length})
          </button>
          <button
            onClick={() => setFilter('closed')}
            className={`px-3 py-1.5 text-xs rounded ${
              filter === 'closed'
                ? 'bg-ibkr-accent text-white'
                : 'bg-ibkr-surface text-ibkr-text-secondary hover:text-ibkr-text'
            }`}
          >
            Closed ({closedTrades.length})
          </button>
        </div>
      </div>

      {/* Trades List */}
      <div className="max-h-[600px] overflow-y-auto">
        {isLoading ? (
          <div className="p-8 text-center">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-ibkr-accent mx-auto"></div>
            <p className="mt-2 text-sm text-ibkr-text-secondary">Loading trades...</p>
          </div>
        ) : trades.length === 0 ? (
          <div className="p-12 text-center">
            <div className="text-5xl mb-4">📈</div>
            <h3 className="text-lg font-medium text-ibkr-text mb-2">No Trades Yet</h3>
            <p className="text-sm text-ibkr-text-secondary">
              Trades will appear here once they are entered
            </p>
          </div>
        ) : (
          <div className="divide-y divide-ibkr-border">
            {trades.map((trade) => (
              <div
                key={trade.id}
                className="p-4 hover:bg-ibkr-bg transition-colors"
              >
                {/* Trade Header */}
                <div className="flex items-start justify-between mb-3">
                  <div className="flex items-center space-x-2">
                    <span className="text-2xl">{getSetupIcon(trade.setup_type)}</span>
                    <div>
                      <div className="flex items-center space-x-2">
                        <span className="font-bold text-lg text-ibkr-text">{trade.symbol}</span>
                        <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                          trade.status === 'OPEN'
                            ? 'bg-blue-500 bg-opacity-10 text-blue-500'
                            : 'bg-ibkr-bg text-ibkr-text-secondary'
                        }`}>
                          {trade.status}
                        </span>
                      </div>
                      <div className="text-xs text-ibkr-text-secondary">
                        {formatSetupName(trade.setup_type)} • {trade.shares} shares
                      </div>
                    </div>
                  </div>

                  {trade.pnl !== undefined && (
                    <div className="text-right">
                      <div className={`text-lg font-bold ${getPnlColor(trade.pnl)}`}>
                        {trade.pnl > 0 ? '+' : ''}${trade.pnl.toFixed(2)}
                      </div>
                      {trade.r_multiple !== undefined && (
                        <div className={`text-xs ${getPnlColor(trade.pnl)}`}>
                          {trade.r_multiple > 0 ? '+' : ''}{trade.r_multiple.toFixed(2)}R
                        </div>
                      )}
                    </div>
                  )}
                </div>

                {/* Trade Details Grid */}
                <div className="grid grid-cols-2 gap-3 text-sm">
                  <div>
                    <div className="text-xs text-ibkr-text-secondary mb-0.5">Entry</div>
                    <div className="font-medium text-ibkr-text">
                      ${trade.entry_price.toFixed(2)}
                      <span className="text-xs text-ibkr-text-secondary ml-1">
                        @ {formatTime(trade.entry_time)}
                      </span>
                    </div>
                  </div>

                  {trade.exit_price && (
                    <div>
                      <div className="text-xs text-ibkr-text-secondary mb-0.5">Exit</div>
                      <div className="font-medium text-ibkr-text">
                        ${trade.exit_price.toFixed(2)}
                        {trade.exit_time && (
                          <span className="text-xs text-ibkr-text-secondary ml-1">
                            @ {formatTime(trade.exit_time)}
                          </span>
                        )}
                      </div>
                    </div>
                  )}

                  <div>
                    <div className="text-xs text-ibkr-text-secondary mb-0.5">Stop</div>
                    <div className="font-medium text-red-500">
                      ${trade.stop_price.toFixed(2)}
                    </div>
                  </div>

                  <div>
                    <div className="text-xs text-ibkr-text-secondary mb-0.5">Target</div>
                    <div className="font-medium text-green-500">
                      ${trade.target_price.toFixed(2)}
                    </div>
                  </div>
                </div>

                {/* Exit Reason */}
                {trade.exit_reason && (
                  <div className="mt-3 pt-3 border-t border-ibkr-border">
                    <div className="text-xs">
                      <span className="text-ibkr-text-secondary">Exit Reason: </span>
                      <span className={`font-medium ${
                        trade.exit_reason === 'TARGET_HIT' ? 'text-green-500' :
                        trade.exit_reason === 'STOP_HIT' ? 'text-red-500' :
                        'text-ibkr-text'
                      }`}>
                        {trade.exit_reason.replace(/_/g, ' ')}
                      </span>
                    </div>
                  </div>
                )}

                {/* Action for Open Trades */}
                {trade.status === 'OPEN' && (
                  <div className="mt-3 pt-3 border-t border-ibkr-border flex justify-end">
                    <button className="px-3 py-1.5 bg-red-600 text-white rounded hover:bg-red-700 transition-colors text-sm">
                      Close Position
                    </button>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default TradeManagement;
