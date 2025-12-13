import React, { useState } from 'react';
import ConfigurationPanel from './ConfigurationPanel';
import ResultsVisualization from './ResultsVisualization';
import { BacktestConfig, BacktestResults } from '../../types/models';
import { apiService } from '../../services/api';

export const Backtesting: React.FC = () => {
  const [isRunning, setIsRunning] = useState(false);
  const [currentBacktest, setCurrentBacktest] = useState<string | null>(null);
  const [results, setResults] = useState<BacktestResults | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleRunBacktest = async (config: BacktestConfig) => {
    try {
      setIsRunning(true);
      setError(null);

      // Start backtest
      const response = await apiService.runBacktest(config);
      if (!response.data) {
        throw new Error('No data received from backtest API');
      }
      const backtestId = response.data.backtest_id;
      setCurrentBacktest(backtestId);

      // Poll for results (in production, this would use WebSocket)
      const pollResults = async () => {
        try {
          const resultsResponse = await apiService.getBacktestResults(backtestId);

          if (!resultsResponse.data) {
            throw new Error('No data received from results API');
          }

          if (resultsResponse.data.status === 'completed' && resultsResponse.data.results) {
            setResults(resultsResponse.data.results);
            setIsRunning(false);
          } else if (resultsResponse.data.status === 'failed') {
            setError('Backtest failed: ' + (resultsResponse.data.error || 'Unknown error'));
            setIsRunning(false);
          } else {
            // Still running, poll again in 2 seconds
            setTimeout(pollResults, 2000);
          }
        } catch (err) {
          console.error('Error polling backtest results:', err);
          setError('Failed to retrieve backtest results');
          setIsRunning(false);
        }
      };

      // Start polling after a brief delay
      setTimeout(pollResults, 2000);

    } catch (err: any) {
      console.error('Error starting backtest:', err);
      setError(err.response?.data?.message || 'Failed to start backtest');
      setIsRunning(false);
    }
  };

  const handleStopBacktest = async () => {
    if (currentBacktest) {
      try {
        await apiService.stopBacktest(currentBacktest);
        setIsRunning(false);
        setCurrentBacktest(null);
      } catch (err) {
        console.error('Error stopping backtest:', err);
      }
    }
  };

  const handleClearResults = () => {
    setResults(null);
    setCurrentBacktest(null);
    setError(null);
  };

  return (
    <div className="p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-xl font-bold text-ibkr-text">Backtesting Laboratory</h1>
        <div className="flex items-center space-x-3">
          {isRunning && (
            <div className="flex items-center space-x-2">
              <div className="animate-spin h-4 w-4 border-2 border-ibkr-accent border-t-transparent rounded-full"></div>
              <span className="text-sm text-ibkr-text-secondary">Running backtest...</span>
            </div>
          )}
          {results && (
            <button
              onClick={handleClearResults}
              className="px-3 py-1.5 text-xs bg-ibkr-surface text-ibkr-text border border-ibkr-border rounded hover:bg-opacity-80 transition-colors"
            >
              Clear Results
            </button>
          )}
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <div className="mb-4 p-4 bg-red-900 bg-opacity-20 border border-red-500 rounded">
          <div className="flex items-start">
            <span className="text-red-400 mr-2">⚠️</span>
            <div>
              <p className="text-sm font-semibold text-red-300 mb-1">Backtest Error</p>
              <p className="text-sm text-red-200">{error}</p>
            </div>
          </div>
        </div>
      )}

      {/* Main Content */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        {/* Left Panel: Configuration */}
        <div>
          <ConfigurationPanel
            onRunBacktest={handleRunBacktest}
            isRunning={isRunning}
          />
        </div>

        {/* Right Panel: Results */}
        <div>
          {isRunning && !results && (
            <div className="bg-ibkr-surface border border-ibkr-border rounded p-8">
              <div className="flex flex-col items-center justify-center text-center space-y-4">
                <div className="animate-spin h-12 w-12 border-4 border-ibkr-accent border-t-transparent rounded-full"></div>
                <div>
                  <p className="text-base font-semibold text-ibkr-text mb-1">
                    Running Backtest
                  </p>
                  <p className="text-sm text-ibkr-text-secondary">
                    Analyzing historical data and executing strategy...
                  </p>
                </div>
                <button
                  onClick={handleStopBacktest}
                  className="mt-4 px-4 py-2 text-sm bg-ibkr-error text-white rounded hover:bg-opacity-80 transition-colors"
                >
                  Stop Backtest
                </button>
              </div>
            </div>
          )}

          {!isRunning && !results && (
            <div className="bg-ibkr-surface border border-ibkr-border rounded p-8">
              <div className="flex flex-col items-center justify-center text-center space-y-3">
                <div className="text-6xl mb-2">📊</div>
                <p className="text-base font-semibold text-ibkr-text">
                  No Backtest Results Yet
                </p>
                <p className="text-sm text-ibkr-text-secondary max-w-md">
                  Configure your strategy parameters on the left and click "Run Backtest"
                  to see detailed performance analysis, equity curves, and trade-by-trade breakdown.
                </p>
              </div>
            </div>
          )}

          {results && (
            <ResultsVisualization results={results} />
          )}
        </div>
      </div>
    </div>
  );
};

export default Backtesting;
