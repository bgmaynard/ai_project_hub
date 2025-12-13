import React, { useState, useEffect } from 'react';
import apiService from '../../services/api';
import type { BacktestConfig, EntryRules, ExitRules, RiskManagement, ClaudeInsights } from '../../types/models';

const STRATEGIES = [
  { id: 'gap-and-go', name: 'Gap & Go' },
  { id: 'momentum', name: 'Momentum' },
  { id: 'bull-flag', name: 'Bull Flag' },
  { id: 'flat-top-breakout', name: 'Flat Top Breakout' }
];

const SYMBOLS = [
  'AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'META', 'AMZN',
  'SPY', 'QQQ', 'AMD', 'NFLX', 'DIS'
];

const VOLUME_REQUIREMENTS = [
  { value: '1.5x', label: '>1.5x Average' },
  { value: '2x', label: '>2x Average' },
  { value: '3x', label: '>3x Average' },
  { value: '5x', label: '>5x Average' }
];

interface ConfigurationPanelProps {
  onRunBacktest: (config: BacktestConfig) => void;
  isRunning: boolean;
}

export const ConfigurationPanel: React.FC<ConfigurationPanelProps> = ({ onRunBacktest, isRunning }) => {
  const [strategy, setStrategy] = useState('gap-and-go');
  const [symbols, setSymbols] = useState<string[]>(['AAPL', 'TSLA']);
  const [startDate, setStartDate] = useState('2024-01-01');
  const [endDate, setEndDate] = useState('2024-11-14');
  const [initialCapital, setInitialCapital] = useState(100000);

  // Entry Rules
  const [gapThreshold, setGapThreshold] = useState(3.0);
  const [volumeRequirement, setVolumeRequirement] = useState('2x');
  const [aiConfidence, setAiConfidence] = useState(65);
  const [newsCatalyst, setNewsCatalyst] = useState(false);
  const [customRules, setCustomRules] = useState<string[]>([]);
  const [newRule, setNewRule] = useState('');

  // Exit Rules
  const [targetProfit, setTargetProfit] = useState(5.0);
  const [stopLoss, setStopLoss] = useState(2.0);
  const [timeBasedStop, setTimeBasedStop] = useState('15:30');
  const [trailingStop, setTrailingStop] = useState(1.5);
  const [useTrailingStop, setUseTrailingStop] = useState(false);

  // Risk Management
  const [maxPositionSize, setMaxPositionSize] = useState(10);
  const [maxDailyTrades, setMaxDailyTrades] = useState(3);
  const [dailyLossLimit, setDailyLossLimit] = useState(2.0);
  const [maxDrawdown, setMaxDrawdown] = useState(5.0);

  // Claude Optimization
  const [claudeQuery, setClaudeQuery] = useState('');
  const [claudeSuggestions, setClaudeSuggestions] = useState<ClaudeInsights | null>(null);
  const [optimizing, setOptimizing] = useState(false);

  // Saved strategies
  const [savedStrategies, setSavedStrategies] = useState<any[]>([]);

  useEffect(() => {
    loadStrategies();
  }, []);

  const loadStrategies = async () => {
    try {
      const response = await apiService.getStrategies();
      if (response.success && response.data) {
        setSavedStrategies(response.data);
      }
    } catch (error) {
      console.error('Error loading strategies:', error);
    }
  };

  const handleSymbolToggle = (symbol: string) => {
    if (symbols.includes(symbol)) {
      setSymbols(symbols.filter(s => s !== symbol));
    } else {
      setSymbols([...symbols, symbol]);
    }
  };

  const handleAddCustomRule = () => {
    if (newRule.trim()) {
      setCustomRules([...customRules, newRule.trim()]);
      setNewRule('');
    }
  };

  const handleRemoveCustomRule = (index: number) => {
    setCustomRules(customRules.filter((_, i) => i !== index));
  };

  const handleOptimize = async () => {
    if (!claudeQuery.trim()) return;

    setOptimizing(true);
    try {
      const config = buildConfig();
      const response = await apiService.optimizeStrategy({
        ...config,
        query: claudeQuery
      });

      if (response.success && response.data) {
        setClaudeSuggestions(response.data);
      }
    } catch (error) {
      console.error('Error optimizing strategy:', error);
      alert('Failed to get Claude optimization');
    } finally {
      setOptimizing(false);
    }
  };

  const handleSaveStrategy = async () => {
    const strategyName = prompt('Enter a name for this strategy:');
    if (!strategyName) return;

    try {
      const config = buildConfig();
      await apiService.saveStrategy({
        name: strategyName,
        ...config
      });
      alert('Strategy saved successfully!');
      loadStrategies();
    } catch (error) {
      console.error('Error saving strategy:', error);
      alert('Failed to save strategy');
    }
  };

  const handleLoadStrategy = async (strategyId: string) => {
    const selectedStrategy = savedStrategies.find(s => s.id === strategyId);
    if (!selectedStrategy) return;

    // Load strategy parameters
    setStrategy(selectedStrategy.strategy || 'gap-and-go');
    setSymbols(selectedStrategy.symbols || ['AAPL']);
    setGapThreshold(selectedStrategy.entry_rules?.gap_percentage || 3.0);
    setVolumeRequirement(selectedStrategy.entry_rules?.volume_requirement || '2x');
    setAiConfidence(selectedStrategy.entry_rules?.ai_confidence_threshold || 65);
    setNewsCatalyst(selectedStrategy.entry_rules?.news_catalyst || false);
    setTargetProfit(selectedStrategy.exit_rules?.target_profit_pct || 5.0);
    setStopLoss(selectedStrategy.exit_rules?.stop_loss_pct || 2.0);
    setMaxPositionSize(selectedStrategy.risk_management?.max_position_size_pct || 10);
    setMaxDailyTrades(selectedStrategy.risk_management?.max_daily_trades || 3);
  };

  const handleReset = () => {
    if (!window.confirm('Reset all parameters to defaults?')) return;

    setStrategy('gap-and-go');
    setSymbols(['AAPL', 'TSLA']);
    setStartDate('2024-01-01');
    setEndDate('2024-11-14');
    setInitialCapital(100000);
    setGapThreshold(3.0);
    setVolumeRequirement('2x');
    setAiConfidence(65);
    setNewsCatalyst(false);
    setCustomRules([]);
    setTargetProfit(5.0);
    setStopLoss(2.0);
    setTimeBasedStop('15:30');
    setTrailingStop(1.5);
    setUseTrailingStop(false);
    setMaxPositionSize(10);
    setMaxDailyTrades(3);
    setDailyLossLimit(2.0);
    setMaxDrawdown(5.0);
  };

  const buildConfig = (): BacktestConfig => {
    const entryRules: EntryRules = {
      gap_percentage: gapThreshold,
      volume_requirement: volumeRequirement,
      ai_confidence_threshold: aiConfidence,
      news_catalyst: newsCatalyst,
      custom_rules: customRules.length > 0 ? customRules : undefined
    };

    const exitRules: ExitRules = {
      target_profit_pct: targetProfit,
      stop_loss_pct: stopLoss,
      time_based_stop: timeBasedStop,
      trailing_stop_pct: useTrailingStop ? trailingStop : undefined
    };

    const riskManagement: RiskManagement = {
      max_position_size_pct: maxPositionSize,
      max_daily_trades: maxDailyTrades,
      daily_loss_limit_pct: dailyLossLimit,
      max_drawdown_pct: maxDrawdown
    };

    return {
      strategy,
      symbols,
      start_date: startDate,
      end_date: endDate,
      initial_capital: initialCapital,
      entry_rules: entryRules,
      exit_rules: exitRules,
      risk_management: riskManagement
    };
  };

  const handleRunBacktest = () => {
    if (symbols.length === 0) {
      alert('Please select at least one symbol');
      return;
    }

    const config = buildConfig();
    onRunBacktest(config);
  };

  return (
    <div className="space-y-6">
      <h2 className="text-xl font-bold text-ibkr-text">Backtest Configuration</h2>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        {/* Left Column: Basic Settings */}
        <div className="space-y-4">
          {/* Strategy Selection */}
          <div className="bg-ibkr-surface p-4 rounded border border-ibkr-border">
            <h3 className="text-sm font-bold text-ibkr-text mb-3">Strategy</h3>

            <div className="space-y-2">
              {STRATEGIES.map(strat => (
                <label key={strat.id} className="flex items-center cursor-pointer hover:bg-ibkr-bg p-2 rounded transition-colors">
                  <input
                    type="radio"
                    name="strategy"
                    value={strat.id}
                    checked={strategy === strat.id}
                    onChange={(e) => setStrategy(e.target.value)}
                    disabled={isRunning}
                    className="mr-2"
                  />
                  <span className="text-xs text-ibkr-text">{strat.name}</span>
                </label>
              ))}
            </div>

            {savedStrategies.length > 0 && (
              <div className="mt-3 pt-3 border-t border-ibkr-border">
                <label className="block text-xs text-ibkr-text-secondary mb-1">Load Saved</label>
                <select
                  onChange={(e) => handleLoadStrategy(e.target.value)}
                  disabled={isRunning}
                  className="w-full bg-ibkr-bg text-ibkr-text text-xs px-2 py-1.5 rounded border border-ibkr-border focus:outline-none focus:border-ibkr-accent disabled:opacity-50"
                >
                  <option value="">Select...</option>
                  {savedStrategies.map(s => (
                    <option key={s.id} value={s.id}>{s.name}</option>
                  ))}
                </select>
              </div>
            )}
          </div>

          {/* Symbols */}
          <div className="bg-ibkr-surface p-4 rounded border border-ibkr-border">
            <h3 className="text-sm font-bold text-ibkr-text mb-3">Symbols</h3>
            <div className="bg-ibkr-bg p-2 rounded border border-ibkr-border max-h-48 overflow-y-auto">
              {SYMBOLS.map(symbol => (
                <label key={symbol} className="flex items-center text-xs text-ibkr-text mb-1 cursor-pointer hover:text-ibkr-accent">
                  <input
                    type="checkbox"
                    checked={symbols.includes(symbol)}
                    onChange={() => handleSymbolToggle(symbol)}
                    disabled={isRunning}
                    className="mr-2"
                  />
                  {symbol}
                </label>
              ))}
            </div>
            <div className="mt-2 text-xs text-ibkr-text-secondary">
              Selected: {symbols.length} symbol{symbols.length !== 1 ? 's' : ''}
            </div>
          </div>

          {/* Date Range & Capital */}
          <div className="bg-ibkr-surface p-4 rounded border border-ibkr-border space-y-3">
            <h3 className="text-sm font-bold text-ibkr-text">Parameters</h3>

            <div>
              <label className="block text-xs text-ibkr-text-secondary mb-1">Start Date</label>
              <input
                type="date"
                value={startDate}
                onChange={(e) => setStartDate(e.target.value)}
                disabled={isRunning}
                className="w-full bg-ibkr-bg text-ibkr-text text-xs px-2 py-1.5 rounded border border-ibkr-border focus:outline-none focus:border-ibkr-accent disabled:opacity-50"
              />
            </div>

            <div>
              <label className="block text-xs text-ibkr-text-secondary mb-1">End Date</label>
              <input
                type="date"
                value={endDate}
                onChange={(e) => setEndDate(e.target.value)}
                disabled={isRunning}
                className="w-full bg-ibkr-bg text-ibkr-text text-xs px-2 py-1.5 rounded border border-ibkr-border focus:outline-none focus:border-ibkr-accent disabled:opacity-50"
              />
            </div>

            <div>
              <label className="block text-xs text-ibkr-text-secondary mb-1">Initial Capital</label>
              <div className="relative">
                <span className="absolute left-2 top-1.5 text-xs text-ibkr-text-secondary">$</span>
                <input
                  type="number"
                  value={initialCapital}
                  onChange={(e) => setInitialCapital(parseFloat(e.target.value))}
                  disabled={isRunning}
                  min="1000"
                  step="1000"
                  className="w-full bg-ibkr-bg text-ibkr-text text-xs pl-5 pr-2 py-1.5 rounded border border-ibkr-border focus:outline-none focus:border-ibkr-accent disabled:opacity-50"
                />
              </div>
            </div>
          </div>
        </div>

        {/* Middle Column: Entry & Exit Rules */}
        <div className="space-y-4">
          {/* Entry Rules */}
          <div className="bg-ibkr-surface p-4 rounded border border-ibkr-border">
            <h3 className="text-sm font-bold text-ibkr-text mb-3">Entry Rules</h3>

            <div className="space-y-3">
              <div>
                <label className="block text-xs text-ibkr-text-secondary mb-1">
                  Gap Threshold: {gapThreshold.toFixed(1)}%
                </label>
                <input
                  type="range"
                  min="0.5"
                  max="10"
                  step="0.5"
                  value={gapThreshold}
                  onChange={(e) => setGapThreshold(parseFloat(e.target.value))}
                  disabled={isRunning}
                  className="w-full disabled:opacity-50"
                />
              </div>

              <div>
                <label className="block text-xs text-ibkr-text-secondary mb-1">Volume Requirement</label>
                <select
                  value={volumeRequirement}
                  onChange={(e) => setVolumeRequirement(e.target.value)}
                  disabled={isRunning}
                  className="w-full bg-ibkr-bg text-ibkr-text text-xs px-2 py-1.5 rounded border border-ibkr-border focus:outline-none focus:border-ibkr-accent disabled:opacity-50"
                >
                  {VOLUME_REQUIREMENTS.map(v => (
                    <option key={v.value} value={v.value}>{v.label}</option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-xs text-ibkr-text-secondary mb-1">
                  AI Confidence: {aiConfidence}%
                </label>
                <input
                  type="range"
                  min="50"
                  max="100"
                  step="5"
                  value={aiConfidence}
                  onChange={(e) => setAiConfidence(parseInt(e.target.value))}
                  disabled={isRunning}
                  className="w-full disabled:opacity-50"
                />
              </div>

              <label className="flex items-center cursor-pointer">
                <input
                  type="checkbox"
                  checked={newsCatalyst}
                  onChange={(e) => setNewsCatalyst(e.target.checked)}
                  disabled={isRunning}
                  className="mr-2"
                />
                <span className="text-xs text-ibkr-text">Require News Catalyst</span>
              </label>

              {/* Custom Rules */}
              <div>
                <label className="block text-xs text-ibkr-text-secondary mb-1">Custom Rules</label>
                <div className="flex space-x-1">
                  <input
                    type="text"
                    value={newRule}
                    onChange={(e) => setNewRule(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && handleAddCustomRule()}
                    placeholder="Add custom rule..."
                    disabled={isRunning}
                    className="flex-1 bg-ibkr-bg text-ibkr-text text-xs px-2 py-1.5 rounded border border-ibkr-border focus:outline-none focus:border-ibkr-accent disabled:opacity-50"
                  />
                  <button
                    onClick={handleAddCustomRule}
                    disabled={isRunning || !newRule.trim()}
                    className="px-2 py-1.5 bg-ibkr-accent text-white text-xs rounded hover:bg-blue-600 transition-colors disabled:opacity-50"
                  >
                    +
                  </button>
                </div>
                {customRules.length > 0 && (
                  <div className="mt-2 space-y-1">
                    {customRules.map((rule, idx) => (
                      <div key={idx} className="flex items-center justify-between bg-ibkr-bg p-1.5 rounded text-xs">
                        <span className="text-ibkr-text">{rule}</span>
                        <button
                          onClick={() => handleRemoveCustomRule(idx)}
                          disabled={isRunning}
                          className="text-ibkr-error hover:text-red-600 disabled:opacity-50"
                        >
                          ×
                        </button>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Exit Rules */}
          <div className="bg-ibkr-surface p-4 rounded border border-ibkr-border">
            <h3 className="text-sm font-bold text-ibkr-text mb-3">Exit Rules</h3>

            <div className="space-y-3">
              <div>
                <label className="block text-xs text-ibkr-text-secondary mb-1">
                  Target Profit: {targetProfit.toFixed(1)}%
                </label>
                <input
                  type="range"
                  min="1"
                  max="20"
                  step="0.5"
                  value={targetProfit}
                  onChange={(e) => setTargetProfit(parseFloat(e.target.value))}
                  disabled={isRunning}
                  className="w-full disabled:opacity-50"
                />
              </div>

              <div>
                <label className="block text-xs text-ibkr-text-secondary mb-1">
                  Stop Loss: {stopLoss.toFixed(1)}%
                </label>
                <input
                  type="range"
                  min="0.5"
                  max="10"
                  step="0.5"
                  value={stopLoss}
                  onChange={(e) => setStopLoss(parseFloat(e.target.value))}
                  disabled={isRunning}
                  className="w-full disabled:opacity-50"
                />
              </div>

              <div>
                <label className="block text-xs text-ibkr-text-secondary mb-1">Time-Based Stop</label>
                <input
                  type="time"
                  value={timeBasedStop}
                  onChange={(e) => setTimeBasedStop(e.target.value)}
                  disabled={isRunning}
                  className="w-full bg-ibkr-bg text-ibkr-text text-xs px-2 py-1.5 rounded border border-ibkr-border focus:outline-none focus:border-ibkr-accent disabled:opacity-50"
                />
              </div>

              <label className="flex items-center cursor-pointer">
                <input
                  type="checkbox"
                  checked={useTrailingStop}
                  onChange={(e) => setUseTrailingStop(e.target.checked)}
                  disabled={isRunning}
                  className="mr-2"
                />
                <span className="text-xs text-ibkr-text">Use Trailing Stop</span>
              </label>

              {useTrailingStop && (
                <div className="ml-5">
                  <label className="block text-xs text-ibkr-text-secondary mb-1">
                    Trailing Stop: {trailingStop.toFixed(1)}%
                  </label>
                  <input
                    type="range"
                    min="0.5"
                    max="5"
                    step="0.5"
                    value={trailingStop}
                    onChange={(e) => setTrailingStop(parseFloat(e.target.value))}
                    disabled={isRunning}
                    className="w-full disabled:opacity-50"
                  />
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Right Column: Risk Management & Claude */}
        <div className="space-y-4">
          {/* Risk Management */}
          <div className="bg-ibkr-surface p-4 rounded border border-ibkr-border">
            <h3 className="text-sm font-bold text-ibkr-text mb-3">Risk Management</h3>

            <div className="space-y-3">
              <div>
                <label className="block text-xs text-ibkr-text-secondary mb-1">
                  Max Position Size: {maxPositionSize.toFixed(0)}%
                </label>
                <input
                  type="range"
                  min="1"
                  max="50"
                  step="1"
                  value={maxPositionSize}
                  onChange={(e) => setMaxPositionSize(parseFloat(e.target.value))}
                  disabled={isRunning}
                  className="w-full disabled:opacity-50"
                />
              </div>

              <div>
                <label className="block text-xs text-ibkr-text-secondary mb-1">Max Daily Trades</label>
                <input
                  type="number"
                  value={maxDailyTrades}
                  onChange={(e) => setMaxDailyTrades(parseInt(e.target.value))}
                  disabled={isRunning}
                  min="1"
                  max="20"
                  className="w-full bg-ibkr-bg text-ibkr-text text-xs px-2 py-1.5 rounded border border-ibkr-border focus:outline-none focus:border-ibkr-accent disabled:opacity-50"
                />
              </div>

              <div>
                <label className="block text-xs text-ibkr-text-secondary mb-1">
                  Daily Loss Limit: {dailyLossLimit.toFixed(1)}%
                </label>
                <input
                  type="range"
                  min="1"
                  max="10"
                  step="0.5"
                  value={dailyLossLimit}
                  onChange={(e) => setDailyLossLimit(parseFloat(e.target.value))}
                  disabled={isRunning}
                  className="w-full disabled:opacity-50"
                />
              </div>

              <div>
                <label className="block text-xs text-ibkr-text-secondary mb-1">
                  Max Drawdown: {maxDrawdown.toFixed(1)}%
                </label>
                <input
                  type="range"
                  min="1"
                  max="20"
                  step="0.5"
                  value={maxDrawdown}
                  onChange={(e) => setMaxDrawdown(parseFloat(e.target.value))}
                  disabled={isRunning}
                  className="w-full disabled:opacity-50"
                />
              </div>
            </div>
          </div>

          {/* Claude Optimization */}
          <div className="bg-ibkr-surface p-4 rounded border border-ibkr-border">
            <h3 className="text-sm font-bold text-ibkr-text mb-3 flex items-center">
              <span className="mr-2">🧠</span>
              Claude Optimization
            </h3>

            <div className="space-y-3">
              <textarea
                value={claudeQuery}
                onChange={(e) => setClaudeQuery(e.target.value)}
                placeholder="Ask Claude to optimize this strategy..."
                disabled={isRunning || optimizing}
                rows={3}
                className="w-full bg-ibkr-bg text-ibkr-text text-xs px-2 py-1.5 rounded border border-ibkr-border focus:outline-none focus:border-ibkr-accent disabled:opacity-50 resize-none"
              />

              <button
                onClick={handleOptimize}
                disabled={isRunning || optimizing || !claudeQuery.trim()}
                className="w-full px-3 py-1.5 bg-ibkr-accent text-white text-xs rounded hover:bg-blue-600 transition-colors disabled:opacity-50"
              >
                {optimizing ? 'Optimizing...' : 'Get Suggestions'}
              </button>

              {claudeSuggestions && (
                <div className="bg-ibkr-bg p-3 rounded space-y-2">
                  {claudeSuggestions.suggestions && claudeSuggestions.suggestions.length > 0 && (
                    <div>
                      <div className="text-xs font-bold text-ibkr-success mb-1">💡 Suggestions:</div>
                      {claudeSuggestions.suggestions.map((suggestion, idx) => (
                        <div key={idx} className="flex items-start space-x-2 mb-2">
                          <span className="text-xs text-ibkr-text-secondary">{idx + 1}.</span>
                          <div className="flex-1">
                            <p className="text-xs text-ibkr-text mb-1">{suggestion}</p>
                            <button className="text-xs text-ibkr-accent hover:text-blue-400">
                              Quick Apply →
                            </button>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="flex items-center justify-between pt-4 border-t border-ibkr-border">
        <div className="flex space-x-2">
          <button
            onClick={handleSaveStrategy}
            disabled={isRunning}
            className="px-4 py-2 bg-ibkr-bg text-ibkr-text text-xs rounded border border-ibkr-border hover:border-ibkr-accent transition-colors disabled:opacity-50"
          >
            Save Strategy
          </button>
          <button
            onClick={handleReset}
            disabled={isRunning}
            className="px-4 py-2 bg-ibkr-bg text-ibkr-text text-xs rounded border border-ibkr-border hover:border-ibkr-error transition-colors disabled:opacity-50"
          >
            Reset to Defaults
          </button>
        </div>

        <button
          onClick={handleRunBacktest}
          disabled={isRunning || symbols.length === 0}
          className="px-6 py-2 bg-ibkr-accent text-white text-sm font-bold rounded hover:bg-blue-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isRunning ? 'Running Backtest...' : 'Run Backtest'}
        </button>
      </div>
    </div>
  );
};

export default ConfigurationPanel;
