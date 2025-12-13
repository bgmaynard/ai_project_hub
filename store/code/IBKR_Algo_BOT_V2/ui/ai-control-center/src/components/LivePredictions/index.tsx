import React, { useState, useEffect } from 'react';
import { Prediction } from '../../types/models';
import { apiService } from '../../services/api';
import { livePredictionsWebSocket } from '../../services/websocket';
import { formatCurrency } from '../../utils/helpers';
import AlertSystem from './AlertSystem';

interface PredictionStats {
  total_predictions: number;
  correct_predictions: number;
  accuracy_rate: number;
  total_profit: number;
  by_strategy: {
    [key: string]: {
      total: number;
      correct: number;
      accuracy: number;
    };
  };
}

export const LivePredictions: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'predictions' | 'alerts'>('predictions');
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [stats, setStats] = useState<PredictionStats | null>(null);
  const [claudeCommentary, setClaudeCommentary] = useState<string>('');
  const [autoUpdate, setAutoUpdate] = useState(true);
  const [filterStrategy, setFilterStrategy] = useState<string>('all');
  const [filterSymbol, setFilterSymbol] = useState<string>('all');
  const [filterConfidence, setFilterConfidence] = useState<number>(0);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetchPredictions();
    fetchStats();
    fetchClaudeCommentary();

    // Set up WebSocket for live updates
    let ws: any;
    if (autoUpdate) {
      ws = livePredictionsWebSocket();
      ws.on('new_prediction', (prediction: Prediction) => {
        setPredictions(prev => [prediction, ...prev].slice(0, 50));
        // Refresh stats when new prediction arrives
        fetchStats();
      });
    }

    return () => {
      if (ws) {
        ws.disconnect();
      }
    };
  }, [autoUpdate]);

  const fetchPredictions = async () => {
    setLoading(true);
    try {
      const response = await apiService.getLivePredictions();
      setPredictions(response.data || []);
    } catch (err) {
      console.error('Error fetching predictions:', err);
    } finally {
      setLoading(false);
    }
  };

  const fetchStats = async () => {
    try {
      const response = await apiService.getPredictionStats();
      setStats(response.data);
    } catch (err) {
      console.error('Error fetching stats:', err);
    }
  };

  const fetchClaudeCommentary = async () => {
    try {
      const response = await apiService.getClaudeCommentary();
      setClaudeCommentary(response.data?.commentary || '');
    } catch (err) {
      console.error('Error fetching Claude commentary:', err);
    }
  };

  // Filter predictions
  const filteredPredictions = predictions.filter(pred => {
    if (filterStrategy !== 'all' && pred.strategy !== filterStrategy) return false;
    if (filterSymbol !== 'all' && pred.symbol !== filterSymbol) return false;
    if (pred.confidence < filterConfidence) return false;
    return true;
  });

  // Get unique strategies and symbols
  const strategies = Array.from(new Set(predictions.map(p => p.strategy)));
  const symbols = Array.from(new Set(predictions.map(p => p.symbol))).sort();

  // Get active predictions (last 10 minutes)
  const now = new Date();
  const tenMinutesAgo = new Date(now.getTime() - 10 * 60 * 1000);
  const activePredictions = predictions.filter(p => new Date(p.timestamp) > tenMinutesAgo);

  const getSignalColor = (signal: 'BUY' | 'SELL') => {
    return signal === 'BUY' ? 'text-ibkr-success' : 'text-ibkr-error';
  };

  const getSignalBg = (signal: 'BUY' | 'SELL') => {
    return signal === 'BUY'
      ? 'bg-green-900 bg-opacity-30 border-green-700'
      : 'bg-red-900 bg-opacity-30 border-red-700';
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 80) return 'text-ibkr-success';
    if (confidence >= 65) return 'text-blue-400';
    return 'text-ibkr-warning';
  };

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold text-ibkr-text mb-1">Live Predictions Monitor</h1>
          <p className="text-sm text-ibkr-text-secondary">
            Real-time AI predictions and performance tracking
          </p>
        </div>
        {activeTab === 'predictions' && (
          <div className="flex items-center space-x-3">
            <button
              onClick={() => setAutoUpdate(!autoUpdate)}
              className={`px-4 py-2 text-sm rounded transition-colors ${
                autoUpdate
                  ? 'bg-ibkr-success text-white'
                  : 'bg-ibkr-surface text-ibkr-text border border-ibkr-border'
              }`}
            >
              {autoUpdate ? '🔴 Live' : '⏸️ Paused'}
            </button>
            <button
              onClick={fetchPredictions}
              disabled={loading}
              className="px-4 py-2 text-sm bg-ibkr-accent text-white rounded hover:bg-opacity-80 transition-colors disabled:opacity-50"
            >
              🔄 Refresh
            </button>
          </div>
        )}
      </div>

      {/* Tab Navigation */}
      <div className="flex space-x-1 bg-ibkr-surface border border-ibkr-border rounded-lg p-1">
        <button
          onClick={() => setActiveTab('predictions')}
          className={`flex-1 px-4 py-2 text-sm font-medium rounded transition-colors ${
            activeTab === 'predictions'
              ? 'bg-ibkr-accent text-white'
              : 'text-ibkr-text hover:bg-ibkr-bg'
          }`}
        >
          📊 Live Predictions
        </button>
        <button
          onClick={() => setActiveTab('alerts')}
          className={`flex-1 px-4 py-2 text-sm font-medium rounded transition-colors ${
            activeTab === 'alerts'
              ? 'bg-ibkr-accent text-white'
              : 'text-ibkr-text hover:bg-ibkr-bg'
          }`}
        >
          🔔 Alert Management
        </button>
      </div>

      {/* Predictions Tab Content */}
      {activeTab === 'predictions' && (
        <>
          {/* Stats Overview */}
          {stats && (
            <div className="grid grid-cols-4 gap-4">
          <div className="bg-ibkr-surface border border-ibkr-border rounded p-4">
            <p className="text-xs text-ibkr-text-secondary mb-1">Today's Predictions</p>
            <p className="text-2xl font-bold text-ibkr-text">{stats.total_predictions || 0}</p>
          </div>
          <div className="bg-ibkr-surface border border-ibkr-border rounded p-4">
            <p className="text-xs text-ibkr-text-secondary mb-1">Correct Predictions</p>
            <p className="text-2xl font-bold text-ibkr-success">{stats.correct_predictions || 0}</p>
          </div>
          <div className="bg-ibkr-surface border border-ibkr-border rounded p-4">
            <p className="text-xs text-ibkr-text-secondary mb-1">Accuracy Rate</p>
            <p className="text-2xl font-bold text-ibkr-text">
              {(stats.accuracy_rate || 0).toFixed(1)}%
            </p>
          </div>
          <div className="bg-ibkr-surface border border-ibkr-border rounded p-4">
            <p className="text-xs text-ibkr-text-secondary mb-1">Total Profit</p>
            <p className={`text-2xl font-bold ${
              (stats.total_profit || 0) >= 0 ? 'text-ibkr-success' : 'text-ibkr-error'
            }`}>
              {formatCurrency(stats.total_profit || 0)}
            </p>
          </div>
        </div>
      )}

      {/* Strategy Performance Breakdown */}
      {stats && stats.by_strategy && Object.keys(stats.by_strategy).length > 0 && (
        <div className="bg-ibkr-surface border border-ibkr-border rounded p-4">
          <h3 className="text-sm font-semibold text-ibkr-text mb-4">Performance by Strategy</h3>
          <div className="grid grid-cols-3 gap-4">
            {Object.entries(stats.by_strategy).map(([strategy, data]) => (
              <div key={strategy} className="bg-ibkr-bg p-3 rounded border border-ibkr-border">
                <p className="text-xs text-ibkr-text-secondary mb-2">{strategy}</p>
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-xs text-ibkr-text">
                      {data.correct}/{data.total}
                    </p>
                  </div>
                  <div>
                    <p className={`text-lg font-bold ${
                      (data.accuracy || 0) >= 70 ? 'text-ibkr-success' :
                      (data.accuracy || 0) >= 50 ? 'text-blue-400' :
                      'text-ibkr-error'
                    }`}>
                      {(data.accuracy || 0).toFixed(0)}%
                    </p>
                  </div>
                </div>
                <div className="mt-2 h-1.5 bg-ibkr-border rounded-full overflow-hidden">
                  <div
                    className={`h-full transition-all ${
                      data.accuracy >= 70 ? 'bg-ibkr-success' :
                      data.accuracy >= 50 ? 'bg-blue-400' :
                      'bg-ibkr-error'
                    }`}
                    style={{ width: `${data.accuracy}%` }}
                  ></div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Claude Commentary */}
      {claudeCommentary && (
        <div className="bg-ibkr-surface border border-ibkr-border rounded p-4">
          <div className="flex items-start">
            <span className="text-2xl mr-3">🧠</span>
            <div className="flex-1">
              <h3 className="text-sm font-semibold text-ibkr-text mb-2">
                Claude's Market Commentary
              </h3>
              <p className="text-sm text-ibkr-text leading-relaxed">{claudeCommentary}</p>
            </div>
          </div>
        </div>
      )}

      {/* Filters */}
      <div className="bg-ibkr-surface border border-ibkr-border rounded p-4">
        <h3 className="text-sm font-semibold text-ibkr-text mb-4">Filters</h3>
        <div className="grid grid-cols-4 gap-3">
          <div>
            <label className="block text-xs text-ibkr-text-secondary mb-1">Strategy</label>
            <select
              value={filterStrategy}
              onChange={(e) => setFilterStrategy(e.target.value)}
              className="w-full px-3 py-2 text-sm bg-ibkr-bg text-ibkr-text border border-ibkr-border rounded focus:outline-none focus:border-ibkr-accent"
            >
              <option value="all">All Strategies</option>
              {strategies.map(strategy => (
                <option key={strategy} value={strategy}>{strategy}</option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-xs text-ibkr-text-secondary mb-1">Symbol</label>
            <select
              value={filterSymbol}
              onChange={(e) => setFilterSymbol(e.target.value)}
              className="w-full px-3 py-2 text-sm bg-ibkr-bg text-ibkr-text border border-ibkr-border rounded focus:outline-none focus:border-ibkr-accent"
            >
              <option value="all">All Symbols</option>
              {symbols.map(symbol => (
                <option key={symbol} value={symbol}>{symbol}</option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-xs text-ibkr-text-secondary mb-1">
              Min Confidence: {filterConfidence}%
            </label>
            <input
              type="range"
              min="0"
              max="100"
              step="5"
              value={filterConfidence}
              onChange={(e) => setFilterConfidence(Number(e.target.value))}
              className="w-full"
            />
          </div>

          <div className="flex items-end">
            <button
              onClick={() => {
                setFilterStrategy('all');
                setFilterSymbol('all');
                setFilterConfidence(0);
              }}
              className="w-full px-3 py-2 text-sm bg-ibkr-bg text-ibkr-text border border-ibkr-border rounded hover:bg-opacity-80 transition-colors"
            >
              Reset Filters
            </button>
          </div>
        </div>
        <p className="mt-3 text-xs text-ibkr-text-secondary">
          Showing {filteredPredictions.length} of {predictions.length} predictions
        </p>
      </div>

      {/* Active Predictions Cards */}
      <div>
        <h3 className="text-sm font-semibold text-ibkr-text mb-4">
          Active Predictions ({activePredictions.length})
        </h3>
        {activePredictions.length > 0 ? (
          <div className="grid grid-cols-3 gap-4">
            {activePredictions.slice(0, 6).map((pred) => (
              <div
                key={`${pred.symbol}-${pred.timestamp}`}
                className={`border-2 rounded p-4 transition-all ${getSignalBg(pred.signal)}`}
              >
                <div className="flex items-center justify-between mb-3">
                  <div>
                    <h4 className="text-lg font-bold text-ibkr-text">{pred.symbol}</h4>
                    <p className="text-xs text-ibkr-text-secondary">
                      {new Date(pred.timestamp).toLocaleTimeString()}
                    </p>
                  </div>
                  <div className={`text-2xl font-bold ${getSignalColor(pred.signal)}`}>
                    {pred.signal}
                  </div>
                </div>

                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-ibkr-text-secondary">Price:</span>
                    <span className="text-ibkr-text font-semibold">
                      ${(pred.price || 0).toFixed(2)}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-ibkr-text-secondary">Confidence:</span>
                    <span className={`font-semibold ${getConfidenceColor(pred.confidence)}`}>
                      {pred.confidence}%
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-ibkr-text-secondary">Strategy:</span>
                    <span className="text-ibkr-text text-xs">{pred.strategy}</span>
                  </div>
                </div>

                {pred.ai_comment && (
                  <div className="mt-3 pt-3 border-t border-ibkr-border">
                    <p className="text-xs text-ibkr-text-secondary italic">
                      "{pred.ai_comment}"
                    </p>
                  </div>
                )}
              </div>
            ))}
          </div>
        ) : (
          <div className="bg-ibkr-surface border border-ibkr-border rounded p-8 text-center">
            <p className="text-sm text-ibkr-text-secondary">
              No active predictions in the last 10 minutes
            </p>
          </div>
        )}
      </div>

      {/* Prediction History Table */}
      <div className="bg-ibkr-surface border border-ibkr-border rounded">
        <div className="p-4 border-b border-ibkr-border">
          <h3 className="text-sm font-semibold text-ibkr-text">Prediction History</h3>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-ibkr-border bg-ibkr-bg">
                <th className="text-left text-ibkr-text-secondary p-3 font-semibold">Time</th>
                <th className="text-left text-ibkr-text-secondary p-3 font-semibold">Symbol</th>
                <th className="text-center text-ibkr-text-secondary p-3 font-semibold">Signal</th>
                <th className="text-right text-ibkr-text-secondary p-3 font-semibold">Price</th>
                <th className="text-center text-ibkr-text-secondary p-3 font-semibold">Confidence</th>
                <th className="text-left text-ibkr-text-secondary p-3 font-semibold">Strategy</th>
                <th className="text-left text-ibkr-text-secondary p-3 font-semibold">AI Comment</th>
              </tr>
            </thead>
            <tbody>
              {filteredPredictions.slice(0, 50).map((pred, index) => (
                <tr
                  key={`${pred.symbol}-${pred.timestamp}-${index}`}
                  className="border-b border-ibkr-border hover:bg-ibkr-bg transition-colors"
                >
                  <td className="text-ibkr-text p-3">
                    {new Date(pred.timestamp).toLocaleTimeString()}
                  </td>
                  <td className="text-ibkr-text p-3 font-semibold">{pred.symbol}</td>
                  <td className="p-3 text-center">
                    <span className={`px-2 py-1 rounded text-xs font-semibold ${
                      pred.signal === 'BUY'
                        ? 'bg-green-900 bg-opacity-30 text-ibkr-success'
                        : 'bg-red-900 bg-opacity-30 text-ibkr-error'
                    }`}>
                      {pred.signal}
                    </span>
                  </td>
                  <td className="text-ibkr-text p-3 text-right">
                    ${(pred.price || 0).toFixed(2)}
                  </td>
                  <td className="p-3 text-center">
                    <span className={`font-semibold ${getConfidenceColor(pred.confidence)}`}>
                      {pred.confidence}%
                    </span>
                  </td>
                  <td className="text-ibkr-text p-3">{pred.strategy}</td>
                  <td className="text-ibkr-text-secondary p-3 max-w-xs truncate">
                    {pred.ai_comment || '-'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
        </>
      )}

      {/* Alerts Tab Content */}
      {activeTab === 'alerts' && (
        <AlertSystem />
      )}
    </div>
  );
};

export default LivePredictions;
