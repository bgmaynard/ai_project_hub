import React, { useState } from 'react';
import { Dialog, Transition } from '@headlessui/react';
import { Fragment } from 'react';
import { Trade } from '../../types/models';
import { apiService } from '../../services/api';
import { formatCurrency, formatPercentage } from '../../utils/helpers';

interface TradeAnalysisProps {
  trades: Trade[];
}

interface TradeFilters {
  dateFrom: string;
  dateTo: string;
  symbol: string;
  result: 'all' | 'wins' | 'losses';
  minPnL: string;
  maxPnL: string;
}

export const TradeAnalysis: React.FC<TradeAnalysisProps> = ({ trades }) => {
  const [filters, setFilters] = useState<TradeFilters>({
    dateFrom: '',
    dateTo: '',
    symbol: 'all',
    result: 'all',
    minPnL: '',
    maxPnL: ''
  });

  const [currentPage, setCurrentPage] = useState(1);
  const [selectedTrade, setSelectedTrade] = useState<Trade | null>(null);
  const [isDetailModalOpen, setIsDetailModalOpen] = useState(false);
  const [tradeAnalysis, setTradeAnalysis] = useState<any>(null);
  const [similarTrades, setSimilarTrades] = useState<Trade[]>([]);
  const [loadingAnalysis, setLoadingAnalysis] = useState(false);

  const tradesPerPage = 25;

  // Apply filters
  const filteredTrades = trades.filter(trade => {
    if (filters.dateFrom && trade.date < filters.dateFrom) return false;
    if (filters.dateTo && trade.date > filters.dateTo) return false;
    if (filters.symbol !== 'all' && trade.symbol !== filters.symbol) return false;
    if (filters.result === 'wins' && !trade.winner) return false;
    if (filters.result === 'losses' && trade.winner) return false;
    if (filters.minPnL && trade.pnl < parseFloat(filters.minPnL)) return false;
    if (filters.maxPnL && trade.pnl > parseFloat(filters.maxPnL)) return false;
    return true;
  });

  // Pagination
  const totalPages = Math.ceil(filteredTrades.length / tradesPerPage);
  const startIndex = (currentPage - 1) * tradesPerPage;
  const paginatedTrades = filteredTrades.slice(startIndex, startIndex + tradesPerPage);

  // Get unique symbols
  const uniqueSymbols = Array.from(new Set(trades.map(t => t.symbol))).sort();

  const handleTradeClick = async (trade: Trade) => {
    setSelectedTrade(trade);
    setIsDetailModalOpen(true);
    setLoadingAnalysis(true);

    try {
      // Fetch AI analysis for this trade
      const analysisResponse = await apiService.getTradeAnalysis(trade.trade_id);
      setTradeAnalysis(analysisResponse.data);

      // Fetch similar trades
      const similarResponse = await apiService.findSimilarTrades(trade.trade_id);
      setSimilarTrades(similarResponse.data || []);
    } catch (err) {
      console.error('Error fetching trade analysis:', err);
    } finally {
      setLoadingAnalysis(false);
    }
  };

  const handleResetFilters = () => {
    setFilters({
      dateFrom: '',
      dateTo: '',
      symbol: 'all',
      result: 'all',
      minPnL: '',
      maxPnL: ''
    });
    setCurrentPage(1);
  };

  const handleExportTrades = () => {
    const csv = [
      ['Date', 'Symbol', 'Entry', 'Exit', 'P&L', 'P&L %', 'Hold Time', 'Result'].join(','),
      ...filteredTrades.map(t => [
        t.date,
        t.symbol,
        t.entry_price.toFixed(2),
        t.exit_price.toFixed(2),
        t.pnl.toFixed(2),
        t.pnl_pct.toFixed(2),
        `${t.hold_time}m`,
        t.winner ? 'Win' : 'Loss'
      ].join(','))
    ].join('\n');

    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `trades_${new Date().toISOString().split('T')[0]}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-4">
      {/* Filters */}
      <div className="bg-ibkr-surface border border-ibkr-border rounded p-4">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-sm font-semibold text-ibkr-text">Filter Trades</h3>
          <div className="flex items-center space-x-2">
            <button
              onClick={handleResetFilters}
              className="px-3 py-1.5 text-xs bg-ibkr-bg text-ibkr-text border border-ibkr-border rounded hover:bg-opacity-80 transition-colors"
            >
              Reset Filters
            </button>
            <button
              onClick={handleExportTrades}
              className="px-3 py-1.5 text-xs bg-ibkr-accent text-white rounded hover:bg-opacity-80 transition-colors"
            >
              Export CSV
            </button>
          </div>
        </div>

        <div className="grid grid-cols-6 gap-3">
          {/* Date From */}
          <div>
            <label className="block text-xs text-ibkr-text-secondary mb-1">Date From</label>
            <input
              type="date"
              value={filters.dateFrom}
              onChange={(e) => setFilters({ ...filters, dateFrom: e.target.value })}
              className="w-full px-2 py-1.5 text-xs bg-ibkr-bg text-ibkr-text border border-ibkr-border rounded focus:outline-none focus:border-ibkr-accent"
            />
          </div>

          {/* Date To */}
          <div>
            <label className="block text-xs text-ibkr-text-secondary mb-1">Date To</label>
            <input
              type="date"
              value={filters.dateTo}
              onChange={(e) => setFilters({ ...filters, dateTo: e.target.value })}
              className="w-full px-2 py-1.5 text-xs bg-ibkr-bg text-ibkr-text border border-ibkr-border rounded focus:outline-none focus:border-ibkr-accent"
            />
          </div>

          {/* Symbol */}
          <div>
            <label className="block text-xs text-ibkr-text-secondary mb-1">Symbol</label>
            <select
              value={filters.symbol}
              onChange={(e) => setFilters({ ...filters, symbol: e.target.value })}
              className="w-full px-2 py-1.5 text-xs bg-ibkr-bg text-ibkr-text border border-ibkr-border rounded focus:outline-none focus:border-ibkr-accent"
            >
              <option value="all">All Symbols</option>
              {uniqueSymbols.map(symbol => (
                <option key={symbol} value={symbol}>{symbol}</option>
              ))}
            </select>
          </div>

          {/* Result */}
          <div>
            <label className="block text-xs text-ibkr-text-secondary mb-1">Result</label>
            <select
              value={filters.result}
              onChange={(e) => setFilters({ ...filters, result: e.target.value as any })}
              className="w-full px-2 py-1.5 text-xs bg-ibkr-bg text-ibkr-text border border-ibkr-border rounded focus:outline-none focus:border-ibkr-accent"
            >
              <option value="all">All Trades</option>
              <option value="wins">Wins Only</option>
              <option value="losses">Losses Only</option>
            </select>
          </div>

          {/* Min P&L */}
          <div>
            <label className="block text-xs text-ibkr-text-secondary mb-1">Min P&L ($)</label>
            <input
              type="number"
              value={filters.minPnL}
              onChange={(e) => setFilters({ ...filters, minPnL: e.target.value })}
              placeholder="Any"
              className="w-full px-2 py-1.5 text-xs bg-ibkr-bg text-ibkr-text border border-ibkr-border rounded focus:outline-none focus:border-ibkr-accent"
            />
          </div>

          {/* Max P&L */}
          <div>
            <label className="block text-xs text-ibkr-text-secondary mb-1">Max P&L ($)</label>
            <input
              type="number"
              value={filters.maxPnL}
              onChange={(e) => setFilters({ ...filters, maxPnL: e.target.value })}
              placeholder="Any"
              className="w-full px-2 py-1.5 text-xs bg-ibkr-bg text-ibkr-text border border-ibkr-border rounded focus:outline-none focus:border-ibkr-accent"
            />
          </div>
        </div>

        {/* Filter Summary */}
        <div className="mt-3 text-xs text-ibkr-text-secondary">
          Showing {filteredTrades.length} of {trades.length} trades
        </div>
      </div>

      {/* Trades Table */}
      <div className="bg-ibkr-surface border border-ibkr-border rounded">
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-ibkr-border bg-ibkr-bg">
                <th className="text-left text-ibkr-text-secondary p-3 font-semibold">Date</th>
                <th className="text-left text-ibkr-text-secondary p-3 font-semibold">Symbol</th>
                <th className="text-right text-ibkr-text-secondary p-3 font-semibold">Entry</th>
                <th className="text-right text-ibkr-text-secondary p-3 font-semibold">Exit</th>
                <th className="text-right text-ibkr-text-secondary p-3 font-semibold">P&L</th>
                <th className="text-right text-ibkr-text-secondary p-3 font-semibold">P&L %</th>
                <th className="text-center text-ibkr-text-secondary p-3 font-semibold">Hold Time</th>
                <th className="text-center text-ibkr-text-secondary p-3 font-semibold">Gap %</th>
                <th className="text-center text-ibkr-text-secondary p-3 font-semibold">AI Conf.</th>
                <th className="text-center text-ibkr-text-secondary p-3 font-semibold">Result</th>
                <th className="text-center text-ibkr-text-secondary p-3 font-semibold">Actions</th>
              </tr>
            </thead>
            <tbody>
              {paginatedTrades.map((trade) => (
                <tr
                  key={trade.trade_id}
                  className="border-b border-ibkr-border hover:bg-ibkr-bg transition-colors cursor-pointer"
                  onClick={() => handleTradeClick(trade)}
                >
                  <td className="text-ibkr-text p-3">
                    {new Date(trade.date).toLocaleDateString()}
                  </td>
                  <td className="text-ibkr-text p-3 font-semibold">{trade.symbol}</td>
                  <td className="text-ibkr-text p-3 text-right">
                    ${trade.entry_price.toFixed(2)}
                  </td>
                  <td className="text-ibkr-text p-3 text-right">
                    ${trade.exit_price.toFixed(2)}
                  </td>
                  <td className={`p-3 text-right font-semibold ${
                    trade.pnl >= 0 ? 'text-ibkr-success' : 'text-ibkr-error'
                  }`}>
                    {formatCurrency(trade.pnl)}
                  </td>
                  <td className={`p-3 text-right ${
                    trade.pnl_pct >= 0 ? 'text-ibkr-success' : 'text-ibkr-error'
                  }`}>
                    {trade.pnl_pct >= 0 ? '+' : ''}{trade.pnl_pct.toFixed(2)}%
                  </td>
                  <td className="text-ibkr-text p-3 text-center">
                    {Math.floor(trade.hold_time / 60)}h {trade.hold_time % 60}m
                  </td>
                  <td className="text-ibkr-text p-3 text-center">
                    {trade.setup_conditions.gap_pct.toFixed(1)}%
                  </td>
                  <td className="text-ibkr-text p-3 text-center">
                    {trade.setup_conditions.ai_confidence}%
                  </td>
                  <td className="p-3 text-center">
                    <span className={`px-2 py-1 rounded text-xs ${
                      trade.winner
                        ? 'bg-green-900 bg-opacity-30 text-ibkr-success'
                        : 'bg-red-900 bg-opacity-30 text-ibkr-error'
                    }`}>
                      {trade.winner ? 'Win' : 'Loss'}
                    </span>
                  </td>
                  <td className="p-3 text-center">
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        handleTradeClick(trade);
                      }}
                      className="px-2 py-1 text-xs bg-ibkr-accent text-white rounded hover:bg-opacity-80 transition-colors"
                    >
                      Details
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Pagination */}
        {totalPages > 1 && (
          <div className="flex items-center justify-between p-4 border-t border-ibkr-border">
            <div className="text-xs text-ibkr-text-secondary">
              Page {currentPage} of {totalPages}
            </div>
            <div className="flex items-center space-x-2">
              <button
                onClick={() => setCurrentPage(Math.max(1, currentPage - 1))}
                disabled={currentPage === 1}
                className="px-3 py-1.5 text-xs bg-ibkr-bg text-ibkr-text border border-ibkr-border rounded hover:bg-opacity-80 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Previous
              </button>
              <button
                onClick={() => setCurrentPage(Math.min(totalPages, currentPage + 1))}
                disabled={currentPage === totalPages}
                className="px-3 py-1.5 text-xs bg-ibkr-bg text-ibkr-text border border-ibkr-border rounded hover:bg-opacity-80 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Next
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Trade Detail Modal */}
      <Transition appear show={isDetailModalOpen} as={Fragment}>
        <Dialog as="div" className="relative z-50" onClose={() => setIsDetailModalOpen(false)}>
          <Transition.Child
            as={Fragment}
            enter="ease-out duration-300"
            enterFrom="opacity-0"
            enterTo="opacity-100"
            leave="ease-in duration-200"
            leaveFrom="opacity-100"
            leaveTo="opacity-0"
          >
            <div className="fixed inset-0 bg-black bg-opacity-75" />
          </Transition.Child>

          <div className="fixed inset-0 overflow-y-auto">
            <div className="flex min-h-full items-center justify-center p-4">
              <Transition.Child
                as={Fragment}
                enter="ease-out duration-300"
                enterFrom="opacity-0 scale-95"
                enterTo="opacity-100 scale-100"
                leave="ease-in duration-200"
                leaveFrom="opacity-100 scale-100"
                leaveTo="opacity-0 scale-95"
              >
                <Dialog.Panel className="w-full max-w-4xl transform overflow-hidden rounded bg-ibkr-surface border border-ibkr-border shadow-xl transition-all">
                  {selectedTrade && (
                    <>
                      {/* Modal Header */}
                      <div className="flex items-center justify-between p-6 border-b border-ibkr-border">
                        <Dialog.Title className="text-lg font-bold text-ibkr-text">
                          Trade Details: {selectedTrade.symbol} - {new Date(selectedTrade.date).toLocaleDateString()}
                        </Dialog.Title>
                        <button
                          onClick={() => setIsDetailModalOpen(false)}
                          className="text-ibkr-text-secondary hover:text-ibkr-text transition-colors"
                        >
                          ✕
                        </button>
                      </div>

                      {/* Modal Content */}
                      <div className="p-6 space-y-6 max-h-[70vh] overflow-y-auto">
                        {/* Trade Summary */}
                        <div className="grid grid-cols-4 gap-4">
                          <div className="bg-ibkr-bg p-4 rounded border border-ibkr-border">
                            <p className="text-xs text-ibkr-text-secondary mb-1">Entry Price</p>
                            <p className="text-xl font-bold text-ibkr-text">
                              ${selectedTrade.entry_price.toFixed(2)}
                            </p>
                          </div>
                          <div className="bg-ibkr-bg p-4 rounded border border-ibkr-border">
                            <p className="text-xs text-ibkr-text-secondary mb-1">Exit Price</p>
                            <p className="text-xl font-bold text-ibkr-text">
                              ${selectedTrade.exit_price.toFixed(2)}
                            </p>
                          </div>
                          <div className="bg-ibkr-bg p-4 rounded border border-ibkr-border">
                            <p className="text-xs text-ibkr-text-secondary mb-1">P&L</p>
                            <p className={`text-xl font-bold ${
                              selectedTrade.pnl >= 0 ? 'text-ibkr-success' : 'text-ibkr-error'
                            }`}>
                              {formatCurrency(selectedTrade.pnl)}
                            </p>
                          </div>
                          <div className="bg-ibkr-bg p-4 rounded border border-ibkr-border">
                            <p className="text-xs text-ibkr-text-secondary mb-1">P&L %</p>
                            <p className={`text-xl font-bold ${
                              selectedTrade.pnl_pct >= 0 ? 'text-ibkr-success' : 'text-ibkr-error'
                            }`}>
                              {selectedTrade.pnl_pct >= 0 ? '+' : ''}{selectedTrade.pnl_pct.toFixed(2)}%
                            </p>
                          </div>
                        </div>

                        {/* Setup Conditions */}
                        <div className="bg-ibkr-bg p-4 rounded border border-ibkr-border">
                          <h4 className="text-sm font-semibold text-ibkr-text mb-3">Setup Conditions</h4>
                          <div className="grid grid-cols-3 gap-4 text-xs">
                            <div>
                              <p className="text-ibkr-text-secondary mb-1">Gap %</p>
                              <p className="text-ibkr-text font-semibold">
                                {selectedTrade.setup_conditions.gap_pct.toFixed(2)}%
                              </p>
                            </div>
                            <div>
                              <p className="text-ibkr-text-secondary mb-1">Volume Ratio</p>
                              <p className="text-ibkr-text font-semibold">
                                {selectedTrade.setup_conditions.volume_ratio.toFixed(2)}x
                              </p>
                            </div>
                            <div>
                              <p className="text-ibkr-text-secondary mb-1">AI Confidence</p>
                              <p className="text-ibkr-text font-semibold">
                                {selectedTrade.setup_conditions.ai_confidence}%
                              </p>
                            </div>
                            {selectedTrade.setup_conditions.news_catalyst && (
                              <div className="col-span-3">
                                <p className="text-ibkr-text-secondary mb-1">News Catalyst</p>
                                <p className="text-ibkr-text">
                                  {selectedTrade.setup_conditions.news_catalyst}
                                </p>
                              </div>
                            )}
                          </div>
                        </div>

                        {/* Claude AI Analysis */}
                        {loadingAnalysis ? (
                          <div className="bg-ibkr-bg p-6 rounded border border-ibkr-border text-center">
                            <div className="animate-spin h-8 w-8 border-4 border-ibkr-accent border-t-transparent rounded-full mx-auto mb-3"></div>
                            <p className="text-sm text-ibkr-text-secondary">Analyzing trade...</p>
                          </div>
                        ) : tradeAnalysis ? (
                          <div className="bg-ibkr-bg p-4 rounded border border-ibkr-border">
                            <h4 className="text-sm font-semibold text-ibkr-text mb-3">
                              🧠 Claude AI Analysis
                            </h4>
                            <div className="space-y-3 text-xs">
                              <div>
                                <p className="text-ibkr-text-secondary mb-1">What Went Well:</p>
                                <p className="text-ibkr-text">{tradeAnalysis.what_went_well}</p>
                              </div>
                              <div>
                                <p className="text-ibkr-text-secondary mb-1">What Could Improve:</p>
                                <p className="text-ibkr-text">{tradeAnalysis.what_could_improve}</p>
                              </div>
                              <div>
                                <p className="text-ibkr-text-secondary mb-1">Recommendation:</p>
                                <p className="text-ibkr-text">{tradeAnalysis.recommendation}</p>
                              </div>
                            </div>
                          </div>
                        ) : null}

                        {/* Similar Trades */}
                        {similarTrades.length > 0 && (
                          <div className="bg-ibkr-bg p-4 rounded border border-ibkr-border">
                            <h4 className="text-sm font-semibold text-ibkr-text mb-3">
                              Similar Setups ({similarTrades.length})
                            </h4>
                            <div className="space-y-2">
                              {similarTrades.slice(0, 5).map((similar) => (
                                <div
                                  key={similar.trade_id}
                                  className="flex items-center justify-between text-xs p-2 bg-ibkr-surface rounded"
                                >
                                  <div>
                                    <span className="text-ibkr-text font-semibold">{similar.symbol}</span>
                                    <span className="text-ibkr-text-secondary ml-2">
                                      {new Date(similar.date).toLocaleDateString()}
                                    </span>
                                  </div>
                                  <div className="flex items-center space-x-3">
                                    <span className="text-ibkr-text-secondary">
                                      Gap: {similar.setup_conditions.gap_pct.toFixed(1)}%
                                    </span>
                                    <span className={`font-semibold ${
                                      similar.pnl >= 0 ? 'text-ibkr-success' : 'text-ibkr-error'
                                    }`}>
                                      {similar.pnl >= 0 ? '+' : ''}{similar.pnl_pct.toFixed(2)}%
                                    </span>
                                  </div>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>

                      {/* Modal Footer */}
                      <div className="flex items-center justify-end space-x-3 p-6 border-t border-ibkr-border">
                        <button
                          onClick={() => setIsDetailModalOpen(false)}
                          className="px-4 py-2 text-sm bg-ibkr-bg text-ibkr-text border border-ibkr-border rounded hover:bg-opacity-80 transition-colors"
                        >
                          Close
                        </button>
                      </div>
                    </>
                  )}
                </Dialog.Panel>
              </Transition.Child>
            </div>
          </div>
        </Dialog>
      </Transition>
    </div>
  );
};

export default TradeAnalysis;
