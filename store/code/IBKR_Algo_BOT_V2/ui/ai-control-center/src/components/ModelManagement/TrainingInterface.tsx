import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { trainingProgressWebSocket } from '../../services/websocket';
import apiService from '../../services/api';
import type { TrainingConfig, TrainingMetrics, ClaudeInsights } from '../../types/models';
import { formatDuration } from '../../utils/helpers';

const MODEL_TYPES = [
  'Ensemble',
  'Random Forest',
  'Gradient Boost',
  'XGBoost',
  'LightGBM'
];

const SYMBOLS = [
  'AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'META', 'AMZN', 'SPY', 'QQQ'
];

const TIMEFRAMES = ['1m', '5m', '15m', '30m', '1h', '4h', '1d'];

const FEATURES = [
  'RSI', 'MACD', 'Bollinger Bands', 'Volume', 'Price Action',
  'Moving Averages', 'Stochastic', 'ATR', 'OBV', 'VWAP'
];

export const TrainingInterface: React.FC = () => {
  // Training configuration
  const [config, setConfig] = useState<TrainingConfig>({
    model_type: 'Ensemble',
    symbols: ['AAPL'],
    timeframes: ['5m'],
    features: ['RSI', 'MACD', 'Volume'],
    start_date: '2024-01-01',
    end_date: '2024-11-14',
    train_split: 0.8
  });

  // Training state
  const [isTraining, setIsTraining] = useState(false);
  const [trainingId, setTrainingId] = useState<string | null>(null);
  const [metrics, setMetrics] = useState<TrainingMetrics | null>(null);
  const [lossHistory, setLossHistory] = useState<Array<{ epoch: number; train_loss: number; val_loss: number }>>([]);
  const [claudeInsights, setClaudeInsights] = useState<ClaudeInsights | null>(null);

  // WebSocket connection
  useEffect(() => {
    if (!trainingId || !isTraining) return;

    const ws = trainingProgressWebSocket(trainingId);

    ws.connect().then(() => {
      console.log('Connected to training progress WebSocket');
    }).catch(err => {
      console.error('Failed to connect to training WebSocket:', err);
    });

    ws.on<TrainingMetrics>('training_metrics', (data) => {
      setMetrics(data);

      // Add to loss history
      setLossHistory(prev => [
        ...prev,
        {
          epoch: data.epoch,
          train_loss: data.train_loss,
          val_loss: data.val_loss
        }
      ].slice(-50)); // Keep last 50 epochs

      // Check if training complete
      if (data.status === 'complete') {
        setIsTraining(false);
        ws.disconnect();
      }
    });

    return () => {
      ws.disconnect();
    };
  }, [trainingId, isTraining]);

  // Fetch Claude insights periodically
  useEffect(() => {
    if (!trainingId || !isTraining) return;

    const fetchInsights = async () => {
      try {
        const response = await apiService.getTrainingInsights(trainingId);
        if (response.success && response.data) {
          setClaudeInsights(response.data);
        }
      } catch (error) {
        console.error('Failed to fetch Claude insights:', error);
      }
    };

    // Fetch immediately
    fetchInsights();

    // Fetch every 10 seconds
    const interval = setInterval(fetchInsights, 10000);

    return () => clearInterval(interval);
  }, [trainingId, isTraining]);

  const handleStartTraining = async () => {
    try {
      setIsTraining(true);
      setLossHistory([]);
      setClaudeInsights(null);

      const response = await apiService.trainModel(config);

      if (response.success && response.data) {
        setTrainingId(response.data.training_id);
      } else {
        setIsTraining(false);
        alert('Failed to start training: ' + (response.error || 'Unknown error'));
      }
    } catch (error: any) {
      console.error('Error starting training:', error);
      setIsTraining(false);
      alert('Error starting training: ' + error.message);
    }
  };

  const handleStopTraining = () => {
    setIsTraining(false);
    setTrainingId(null);
  };

  return (
    <div className="space-y-6">
      <h2 className="text-xl font-bold text-ibkr-text">Model Training</h2>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        {/* Left Column: Configuration */}
        <div className="xl:col-span-1">
          <div className="bg-ibkr-surface p-4 rounded border border-ibkr-border space-y-4">
            <h3 className="text-sm font-bold text-ibkr-text border-b border-ibkr-border pb-2">
              Training Configuration
            </h3>

            {/* Model Type */}
            <div>
              <label className="block text-xs text-ibkr-text-secondary mb-1">Model Type</label>
              <select
                value={config.model_type}
                onChange={(e) => setConfig({ ...config, model_type: e.target.value })}
                disabled={isTraining}
                className="w-full bg-ibkr-bg text-ibkr-text text-xs px-2 py-1.5 rounded border border-ibkr-border focus:outline-none focus:border-ibkr-accent disabled:opacity-50"
              >
                {MODEL_TYPES.map(type => (
                  <option key={type} value={type}>{type}</option>
                ))}
              </select>
            </div>

            {/* Symbols */}
            <div>
              <label className="block text-xs text-ibkr-text-secondary mb-1">Symbols</label>
              <div className="bg-ibkr-bg p-2 rounded border border-ibkr-border max-h-32 overflow-y-auto">
                {SYMBOLS.map(symbol => (
                  <label key={symbol} className="flex items-center text-xs text-ibkr-text mb-1 cursor-pointer hover:text-ibkr-accent">
                    <input
                      type="checkbox"
                      checked={config.symbols.includes(symbol)}
                      onChange={(e) => {
                        if (e.target.checked) {
                          setConfig({ ...config, symbols: [...config.symbols, symbol] });
                        } else {
                          setConfig({ ...config, symbols: config.symbols.filter(s => s !== symbol) });
                        }
                      }}
                      disabled={isTraining}
                      className="mr-2"
                    />
                    {symbol}
                  </label>
                ))}
              </div>
            </div>

            {/* Timeframes */}
            <div>
              <label className="block text-xs text-ibkr-text-secondary mb-1">Timeframes</label>
              <div className="flex flex-wrap gap-1">
                {TIMEFRAMES.map(tf => (
                  <button
                    key={tf}
                    onClick={() => {
                      if (config.timeframes.includes(tf)) {
                        setConfig({ ...config, timeframes: config.timeframes.filter(t => t !== tf) });
                      } else {
                        setConfig({ ...config, timeframes: [...config.timeframes, tf] });
                      }
                    }}
                    disabled={isTraining}
                    className={`px-2 py-1 text-xs rounded transition-colors ${
                      config.timeframes.includes(tf)
                        ? 'bg-ibkr-accent text-white'
                        : 'bg-ibkr-bg text-ibkr-text-secondary border border-ibkr-border hover:border-ibkr-accent'
                    } disabled:opacity-50`}
                  >
                    {tf}
                  </button>
                ))}
              </div>
            </div>

            {/* Features */}
            <div>
              <label className="block text-xs text-ibkr-text-secondary mb-1">Features</label>
              <div className="bg-ibkr-bg p-2 rounded border border-ibkr-border max-h-32 overflow-y-auto">
                {FEATURES.map(feature => (
                  <label key={feature} className="flex items-center text-xs text-ibkr-text mb-1 cursor-pointer hover:text-ibkr-accent">
                    <input
                      type="checkbox"
                      checked={config.features.includes(feature)}
                      onChange={(e) => {
                        if (e.target.checked) {
                          setConfig({ ...config, features: [...config.features, feature] });
                        } else {
                          setConfig({ ...config, features: config.features.filter(f => f !== feature) });
                        }
                      }}
                      disabled={isTraining}
                      className="mr-2"
                    />
                    {feature}
                  </label>
                ))}
              </div>
            </div>

            {/* Date Range */}
            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="block text-xs text-ibkr-text-secondary mb-1">Start Date</label>
                <input
                  type="date"
                  value={config.start_date}
                  onChange={(e) => setConfig({ ...config, start_date: e.target.value })}
                  disabled={isTraining}
                  className="w-full bg-ibkr-bg text-ibkr-text text-xs px-2 py-1.5 rounded border border-ibkr-border focus:outline-none focus:border-ibkr-accent disabled:opacity-50"
                />
              </div>
              <div>
                <label className="block text-xs text-ibkr-text-secondary mb-1">End Date</label>
                <input
                  type="date"
                  value={config.end_date}
                  onChange={(e) => setConfig({ ...config, end_date: e.target.value })}
                  disabled={isTraining}
                  className="w-full bg-ibkr-bg text-ibkr-text text-xs px-2 py-1.5 rounded border border-ibkr-border focus:outline-none focus:border-ibkr-accent disabled:opacity-50"
                />
              </div>
            </div>

            {/* Train/Validation Split */}
            <div>
              <label className="block text-xs text-ibkr-text-secondary mb-1">
                Train/Validation Split: {(config.train_split * 100).toFixed(0)}% / {((1 - config.train_split) * 100).toFixed(0)}%
              </label>
              <input
                type="range"
                min="0.6"
                max="0.9"
                step="0.05"
                value={config.train_split}
                onChange={(e) => setConfig({ ...config, train_split: parseFloat(e.target.value) })}
                disabled={isTraining}
                className="w-full disabled:opacity-50"
              />
            </div>

            {/* Start/Stop Button */}
            <button
              onClick={isTraining ? handleStopTraining : handleStartTraining}
              disabled={!isTraining && (config.symbols.length === 0 || config.features.length === 0)}
              className={`w-full py-2 px-4 rounded font-bold text-sm transition-colors ${
                isTraining
                  ? 'bg-ibkr-error text-white hover:bg-red-600'
                  : 'bg-ibkr-accent text-white hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed'
              }`}
            >
              {isTraining ? 'Stop Training' : 'Start Training'}
            </button>
          </div>
        </div>

        {/* Right Column: Progress & Results */}
        <div className="xl:col-span-2 space-y-4">
          {/* Training Progress */}
          {isTraining && metrics && (
            <div className="bg-ibkr-surface p-4 rounded border border-ibkr-border space-y-4">
              <h3 className="text-sm font-bold text-ibkr-text border-b border-ibkr-border pb-2">
                Training Progress
              </h3>

              {/* Progress Bar */}
              <div>
                <div className="flex justify-between text-xs text-ibkr-text-secondary mb-1">
                  <span>Epoch {metrics.epoch} / {metrics.total_epochs}</span>
                  <span>{((metrics.epoch / metrics.total_epochs) * 100).toFixed(1)}%</span>
                </div>
                <div className="w-full bg-ibkr-bg rounded-full h-2">
                  <div
                    className="bg-ibkr-accent h-2 rounded-full transition-all duration-300"
                    style={{ width: `${(metrics.epoch / metrics.total_epochs) * 100}%` }}
                  />
                </div>
              </div>

              {/* Metrics Grid */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="bg-ibkr-bg p-3 rounded">
                  <div className="text-xs text-ibkr-text-secondary mb-1">Train Accuracy</div>
                  <div className="text-lg font-bold text-ibkr-success">
                    {(metrics.train_accuracy * 100).toFixed(2)}%
                  </div>
                </div>
                <div className="bg-ibkr-bg p-3 rounded">
                  <div className="text-xs text-ibkr-text-secondary mb-1">Val Accuracy</div>
                  <div className="text-lg font-bold text-ibkr-accent">
                    {(metrics.val_accuracy * 100).toFixed(2)}%
                  </div>
                </div>
                <div className="bg-ibkr-bg p-3 rounded">
                  <div className="text-xs text-ibkr-text-secondary mb-1">Train Loss</div>
                  <div className="text-lg font-bold text-ibkr-text">
                    {metrics.train_loss.toFixed(4)}
                  </div>
                </div>
                <div className="bg-ibkr-bg p-3 rounded">
                  <div className="text-xs text-ibkr-text-secondary mb-1">Val Loss</div>
                  <div className="text-lg font-bold text-ibkr-text">
                    {metrics.val_loss.toFixed(4)}
                  </div>
                </div>
              </div>

              {/* Time Info */}
              {metrics.time_elapsed !== undefined && (
                <div className="flex justify-between text-xs text-ibkr-text-secondary">
                  <span>Elapsed: {formatDuration(metrics.time_elapsed)}</span>
                  {metrics.eta !== undefined && (
                    <span>ETA: {formatDuration(metrics.eta)}</span>
                  )}
                </div>
              )}

              {/* Loss Curve Chart */}
              {lossHistory.length > 0 && (
                <div>
                  <div className="text-xs text-ibkr-text-secondary mb-2">Loss Curves</div>
                  <ResponsiveContainer width="100%" height={200}>
                    <LineChart data={lossHistory}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#3e3e42" />
                      <XAxis
                        dataKey="epoch"
                        stroke="#888888"
                        tick={{ fill: '#888888', fontSize: 10 }}
                      />
                      <YAxis
                        stroke="#888888"
                        tick={{ fill: '#888888', fontSize: 10 }}
                      />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: '#252526',
                          border: '1px solid #3e3e42',
                          borderRadius: '4px',
                          fontSize: '10px'
                        }}
                      />
                      <Legend
                        wrapperStyle={{ fontSize: '10px' }}
                      />
                      <Line
                        type="monotone"
                        dataKey="train_loss"
                        stroke="#4ec9b0"
                        strokeWidth={2}
                        dot={false}
                        name="Train Loss"
                      />
                      <Line
                        type="monotone"
                        dataKey="val_loss"
                        stroke="#007acc"
                        strokeWidth={2}
                        dot={false}
                        name="Val Loss"
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              )}
            </div>
          )}

          {/* Claude AI Insights */}
          {isTraining && claudeInsights && (
            <div className="bg-ibkr-surface p-4 rounded border border-ibkr-border space-y-3">
              <div className="flex items-center justify-between border-b border-ibkr-border pb-2">
                <h3 className="text-sm font-bold text-ibkr-text flex items-center">
                  <span className="mr-2">🧠</span>
                  Claude AI Insights
                </h3>
                <span className="text-xs text-ibkr-text-secondary">Updates every 10s</span>
              </div>

              {/* Insights */}
              {claudeInsights.insights && (
                <div className="text-xs text-ibkr-text bg-ibkr-bg p-3 rounded">
                  {claudeInsights.insights}
                </div>
              )}

              {/* Concerns */}
              {claudeInsights.concerns && claudeInsights.concerns.length > 0 && (
                <div>
                  <div className="text-xs font-bold text-ibkr-warning mb-2">⚠️ Concerns:</div>
                  <ul className="space-y-1">
                    {claudeInsights.concerns.map((concern, idx) => (
                      <li key={idx} className="text-xs text-ibkr-text-secondary flex items-start">
                        <span className="mr-2">•</span>
                        <span>{concern}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Suggestions */}
              {claudeInsights.suggestions && claudeInsights.suggestions.length > 0 && (
                <div>
                  <div className="text-xs font-bold text-ibkr-success mb-2">💡 Suggestions:</div>
                  <ul className="space-y-1">
                    {claudeInsights.suggestions.map((suggestion, idx) => (
                      <li key={idx} className="text-xs text-ibkr-text-secondary flex items-start">
                        <span className="mr-2">•</span>
                        <span>{suggestion}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}

          {/* Placeholder when not training */}
          {!isTraining && (
            <div className="bg-ibkr-surface p-8 rounded border border-ibkr-border text-center">
              <div className="text-4xl mb-4">🤖</div>
              <h3 className="text-sm font-bold text-ibkr-text mb-2">Ready to Train</h3>
              <p className="text-xs text-ibkr-text-secondary">
                Configure your model settings and click "Start Training" to begin
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default TrainingInterface;
