import React, { useState, useEffect } from 'react';

interface Candidate {
  symbol: string;
  price: number;
  gap_percent: number;
  relative_volume: number;
  float_shares: number;
  pre_market_volume: number;
  catalyst: string;
  daily_chart_signal: string;
  confidence_score: number;
}

/**
 * Pre-Market Scanner Component
 *
 * Displays watchlist candidates and allows running scans
 */
const PreMarketScanner: React.FC = () => {
  const [watchlist, setWatchlist] = useState<Candidate[]>([]);
  const [isScanning, setIsScanning] = useState(false);
  const [lastScanTime, setLastScanTime] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [sortBy, setSortBy] = useState<'confidence' | 'gap' | 'rvol'>('confidence');

  useEffect(() => {
    fetchWatchlist();
    // Refresh every 5 minutes
    const interval = setInterval(fetchWatchlist, 5 * 60 * 1000);
    return () => clearInterval(interval);
  }, []);

  const fetchWatchlist = async () => {
    try {
      const response = await fetch('/api/warrior/watchlist');
      const data = await response.json();

      if (data.watchlist && data.watchlist.length > 0) {
        setWatchlist(data.watchlist);
        setLastScanTime(data.timestamp);
      }
    } catch (err) {
      console.error('Error fetching watchlist:', err);
    }
  };

  const runScan = async () => {
    setIsScanning(true);
    setError(null);

    try {
      const response = await fetch('/api/warrior/scan/premarket', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({})
      });

      const data = await response.json();

      if (data.success) {
        setWatchlist(data.candidates);
        setLastScanTime(data.timestamp);
      } else {
        setError('Scan failed');
      }
    } catch (err) {
      setError('Error running scan');
      console.error('Scan error:', err);
    } finally {
      setIsScanning(false);
    }
  };

  const getSortedWatchlist = () => {
    const sorted = [...watchlist];
    switch (sortBy) {
      case 'confidence':
        return sorted.sort((a, b) => b.confidence_score - a.confidence_score);
      case 'gap':
        return sorted.sort((a, b) => b.gap_percent - a.gap_percent);
      case 'rvol':
        return sorted.sort((a, b) => b.relative_volume - a.relative_volume);
      default:
        return sorted;
    }
  };

  const getSignalColor = (signal: string) => {
    switch (signal) {
      case 'BULLISH': return 'text-green-500';
      case 'BEARISH': return 'text-red-500';
      default: return 'text-yellow-500';
    }
  };

  const getConfidenceColor = (score: number) => {
    if (score >= 80) return 'bg-green-500';
    if (score >= 60) return 'bg-yellow-500';
    return 'bg-orange-500';
  };

  return (
    <div className="bg-ibkr-surface rounded-lg border border-ibkr-border">
      {/* Header */}
      <div className="p-4 border-b border-ibkr-border">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-lg font-semibold text-ibkr-text flex items-center">
              <span className="mr-2">🔍</span>
              Pre-Market Scanner
            </h2>
            <p className="text-sm text-ibkr-text-secondary mt-1">
              High-probability momentum candidates
            </p>
          </div>

          <button
            onClick={runScan}
            disabled={isScanning}
            className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center space-x-2"
          >
            {isScanning ? (
              <>
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                <span>Scanning...</span>
              </>
            ) : (
              <>
                <span>🚀</span>
                <span>Run Scan</span>
              </>
            )}
          </button>
        </div>

        {lastScanTime && (
          <div className="mt-2 text-xs text-ibkr-text-secondary">
            Last scan: {new Date(lastScanTime).toLocaleString()}
          </div>
        )}

        {error && (
          <div className="mt-2 p-2 bg-red-500 bg-opacity-10 border border-red-500 rounded text-sm text-red-500">
            {error}
          </div>
        )}
      </div>

      {/* Sort Controls */}
      <div className="p-3 border-b border-ibkr-border bg-ibkr-bg flex items-center space-x-2">
        <span className="text-sm text-ibkr-text-secondary">Sort by:</span>
        <div className="flex space-x-1">
          <button
            onClick={() => setSortBy('confidence')}
            className={`px-3 py-1 text-xs rounded ${
              sortBy === 'confidence'
                ? 'bg-ibkr-accent text-white'
                : 'bg-ibkr-surface text-ibkr-text-secondary hover:text-ibkr-text'
            }`}
          >
            Confidence
          </button>
          <button
            onClick={() => setSortBy('gap')}
            className={`px-3 py-1 text-xs rounded ${
              sortBy === 'gap'
                ? 'bg-ibkr-accent text-white'
                : 'bg-ibkr-surface text-ibkr-text-secondary hover:text-ibkr-text'
            }`}
          >
            Gap %
          </button>
          <button
            onClick={() => setSortBy('rvol')}
            className={`px-3 py-1 text-xs rounded ${
              sortBy === 'rvol'
                ? 'bg-ibkr-accent text-white'
                : 'bg-ibkr-surface text-ibkr-text-secondary hover:text-ibkr-text'
            }`}
          >
            RVOL
          </button>
        </div>
        <div className="ml-auto text-sm text-ibkr-text-secondary">
          {watchlist.length} candidates
        </div>
      </div>

      {/* Watchlist Table */}
      <div className="overflow-x-auto">
        {watchlist.length === 0 ? (
          <div className="p-12 text-center">
            <div className="text-5xl mb-4">📊</div>
            <h3 className="text-lg font-medium text-ibkr-text mb-2">No Candidates Yet</h3>
            <p className="text-sm text-ibkr-text-secondary mb-4">
              Run a scan to find high-probability momentum stocks
            </p>
            <button
              onClick={runScan}
              className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700"
            >
              Run First Scan
            </button>
          </div>
        ) : (
          <table className="w-full">
            <thead className="bg-ibkr-bg text-ibkr-text-secondary text-xs uppercase tracking-wider">
              <tr>
                <th className="px-4 py-3 text-left">Symbol</th>
                <th className="px-4 py-3 text-right">Price</th>
                <th className="px-4 py-3 text-right">Gap %</th>
                <th className="px-4 py-3 text-right">RVOL</th>
                <th className="px-4 py-3 text-right">Float (M)</th>
                <th className="px-4 py-3 text-left">Catalyst</th>
                <th className="px-4 py-3 text-center">Signal</th>
                <th className="px-4 py-3 text-center">Confidence</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-ibkr-border">
              {getSortedWatchlist().map((candidate, index) => (
                <tr
                  key={candidate.symbol}
                  className="hover:bg-ibkr-bg transition-colors"
                >
                  <td className="px-4 py-3">
                    <div className="flex items-center space-x-2">
                      <span className="text-xs text-ibkr-text-secondary">#{index + 1}</span>
                      <span className="font-semibold text-ibkr-text">{candidate.symbol}</span>
                    </div>
                  </td>
                  <td className="px-4 py-3 text-right text-ibkr-text">
                    ${candidate.price.toFixed(2)}
                  </td>
                  <td className="px-4 py-3 text-right">
                    <span className="text-green-500 font-medium">
                      +{candidate.gap_percent.toFixed(1)}%
                    </span>
                  </td>
                  <td className="px-4 py-3 text-right">
                    <span className={`font-medium ${
                      candidate.relative_volume >= 3 ? 'text-green-500' : 'text-yellow-500'
                    }`}>
                      {candidate.relative_volume.toFixed(1)}x
                    </span>
                  </td>
                  <td className="px-4 py-3 text-right text-ibkr-text">
                    {candidate.float_shares.toFixed(1)}M
                  </td>
                  <td className="px-4 py-3">
                    <div className="text-sm text-ibkr-text max-w-xs truncate" title={candidate.catalyst}>
                      {candidate.catalyst || 'N/A'}
                    </div>
                  </td>
                  <td className="px-4 py-3 text-center">
                    <span className={`text-sm font-medium ${getSignalColor(candidate.daily_chart_signal)}`}>
                      {candidate.daily_chart_signal}
                    </span>
                  </td>
                  <td className="px-4 py-3">
                    <div className="flex items-center justify-center space-x-2">
                      <div className="flex-1 bg-ibkr-bg rounded-full h-2 max-w-[60px]">
                        <div
                          className={`h-2 rounded-full ${getConfidenceColor(candidate.confidence_score)}`}
                          style={{ width: `${candidate.confidence_score}%` }}
                        ></div>
                      </div>
                      <span className="text-xs font-medium text-ibkr-text min-w-[35px] text-right">
                        {candidate.confidence_score.toFixed(0)}%
                      </span>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
};

export default PreMarketScanner;
