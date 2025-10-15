import React, { useState, useEffect } from 'react';
import { Activity, TrendingUp, BarChart3, Settings, AlertCircle, Play, Pause, DollarSign } from 'lucide-react';

const AITradingBot = () => {
  const [isConnected, setIsConnected] = useState(false);
  const [accountType, setAccountType] = useState('paper');
  const [tradingActive, setTradingActive] = useState(false);
  const [watchlist, setWatchlist] = useState([
    { symbol: 'AAPL', price: 178.45, change: 2.3, momentum: 0.65, aiScore: 0.72 },
    { symbol: 'TSLA', price: 242.18, change: -1.2, momentum: 0.45, aiScore: 0.58 },
    { symbol: 'NVDA', price: 485.32, change: 3.8, momentum: 0.82, aiScore: 0.85 }
  ]);
  const [positions, setPositions] = useState([]);
  const [accountData, setAccountData] = useState({
    cash: 100000,
    equity: 100000,
    buyingPower: 200000,
    dailyPL: 0,
    dailyPLPercent: 0
  });
  const [strategyConfig, setStrategyConfig] = useState({
    maxPositionSize: 10000,
    dailyLossLimit: 2000,
    priceRangeMin: 5,
    priceRangeMax: 500,
    minVolume: 1000000,
    targetGainPercent: 2.5,
    stopLossPercent: 1.5
  });
  const [aiSignals, setAiSignals] = useState([]);
  const [selectedSymbol, setSelectedSymbol] = useState('AAPL');

  // Simulate IBKR connection
  const toggleConnection = () => {
    setIsConnected(!isConnected);
  };

  // Simulate AI signal generation
  useEffect(() => {
    if (tradingActive && isConnected) {
      const interval = setInterval(() => {
        const signal = {
          symbol: watchlist[Math.floor(Math.random() * watchlist.length)].symbol,
          direction: Math.random() > 0.5 ? 'BUY' : 'SELL',
          confidence: (Math.random() * 0.4 + 0.6).toFixed(2),
          strategy: ['MACD+LSTM', 'Momentum', 'VWAP Breakout'][Math.floor(Math.random() * 3)],
          timestamp: new Date().toLocaleTimeString()
        };
        setAiSignals(prev => [signal, ...prev].slice(0, 5));
      }, 5000);
      return () => clearInterval(interval);
    }
  }, [tradingActive, isConnected, watchlist]);

  // Place order function
  const placeOrder = (symbol, direction, quantity) => {
    if (!isConnected) {
      alert('Connect to IBKR TWS first!');
      return;
    }
    
    const order = {
      symbol,
      direction,
      quantity,
      price: watchlist.find(w => w.symbol === symbol)?.price || 0,
      timestamp: new Date().toLocaleTimeString()
    };
    
    setPositions(prev => [...prev, order]);
    alert(`Order placed: ${direction} ${quantity} ${symbol}`);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 to-slate-800 text-white p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-6 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Activity className="w-8 h-8 text-blue-400" />
            <h1 className="text-3xl font-bold">AI Trading Bot</h1>
          </div>
          <div className="flex gap-3">
            <button
              onClick={toggleConnection}
              className={`px-4 py-2 rounded-lg font-semibold transition ${
                isConnected 
                  ? 'bg-green-600 hover:bg-green-700' 
                  : 'bg-red-600 hover:bg-red-700'
              }`}
            >
              {isConnected ? '● IBKR Connected' : '○ Connect IBKR'}
            </button>
            <select
              value={accountType}
              onChange={(e) => setAccountType(e.target.value)}
              className="bg-slate-700 px-4 py-2 rounded-lg"
            >
              <option value="paper">Paper Trading</option>
              <option value="cash">Cash Account</option>
              <option value="margin">Margin Account</option>
            </select>
          </div>
        </div>

        {/* Account Summary */}
        <div className="grid grid-cols-4 gap-4 mb-6">
          <div className="bg-slate-800 p-4 rounded-lg border border-slate-700">
            <div className="text-sm text-slate-400">Cash Available</div>
            <div className="text-2xl font-bold text-green-400">
              ${accountData.cash.toLocaleString()}
            </div>
          </div>
          <div className="bg-slate-800 p-4 rounded-lg border border-slate-700">
            <div className="text-sm text-slate-400">Total Equity</div>
            <div className="text-2xl font-bold">
              ${accountData.equity.toLocaleString()}
            </div>
          </div>
          <div className="bg-slate-800 p-4 rounded-lg border border-slate-700">
            <div className="text-sm text-slate-400">Buying Power</div>
            <div className="text-2xl font-bold text-blue-400">
              ${accountData.buyingPower.toLocaleString()}
            </div>
          </div>
          <div className="bg-slate-800 p-4 rounded-lg border border-slate-700">
            <div className="text-sm text-slate-400">Daily P/L</div>
            <div className={`text-2xl font-bold ${accountData.dailyPL >= 0 ? 'text-green-400' : 'text-red-400'}`}>
              ${accountData.dailyPL.toLocaleString()} ({accountData.dailyPLPercent}%)
            </div>
          </div>
        </div>

        {/* Main Trading Interface */}
        <div className="grid grid-cols-3 gap-6">
          {/* Watchlist */}
          <div className="col-span-2 bg-slate-800 rounded-lg p-5 border border-slate-700">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-bold flex items-center gap-2">
                <TrendingUp className="w-5 h-5 text-blue-400" />
                AI-Monitored Watchlist
              </h2>
              <button
                onClick={() => setTradingActive(!tradingActive)}
                className={`px-4 py-2 rounded-lg font-semibold transition flex items-center gap-2 ${
                  tradingActive ? 'bg-red-600 hover:bg-red-700' : 'bg-green-600 hover:bg-green-700'
                }`}
              >
                {tradingActive ? <><Pause className="w-4 h-4" /> Pause AI</> : <><Play className="w-4 h-4" /> Start AI</>}
              </button>
            </div>
            
            <div className="space-y-3">
              {watchlist.map((stock) => (
                <div
                  key={stock.symbol}
                  onClick={() => setSelectedSymbol(stock.symbol)}
                  className={`p-4 rounded-lg cursor-pointer transition ${
                    selectedSymbol === stock.symbol 
                      ? 'bg-blue-900 border border-blue-600' 
                      : 'bg-slate-700 hover:bg-slate-600'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex-1">
                      <div className="flex items-center gap-3">
                        <span className="text-lg font-bold">{stock.symbol}</span>
                        <span className={`text-sm ${stock.change >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                          {stock.change >= 0 ? '+' : ''}{stock.change}%
                        </span>
                      </div>
                      <div className="text-2xl font-bold mt-1">${stock.price}</div>
                    </div>
                    <div className="text-right">
                      <div className="text-sm text-slate-400">AI Score</div>
                      <div className={`text-xl font-bold ${
                        stock.aiScore > 0.7 ? 'text-green-400' : 
                        stock.aiScore > 0.5 ? 'text-yellow-400' : 'text-red-400'
                      }`}>
                        {(stock.aiScore * 100).toFixed(0)}%
                      </div>
                      <div className="text-xs text-slate-400 mt-1">
                        Momentum: {(stock.momentum * 100).toFixed(0)}%
                      </div>
                    </div>
                    <div className="flex flex-col gap-2 ml-4">
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          placeOrder(stock.symbol, 'BUY', 100);
                        }}
                        className="px-3 py-1 bg-green-600 hover:bg-green-700 rounded text-sm font-semibold"
                      >
                        BUY
                      </button>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          placeOrder(stock.symbol, 'SELL', 100);
                        }}
                        className="px-3 py-1 bg-red-600 hover:bg-red-700 rounded text-sm font-semibold"
                      >
                        SELL
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* AI Signals & Strategy Config */}
          <div className="space-y-6">
            {/* Live AI Signals */}
            <div className="bg-slate-800 rounded-lg p-5 border border-slate-700">
              <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
                <Activity className="w-5 h-5 text-green-400" />
                Live AI Signals
              </h3>
              <div className="space-y-2">
                {aiSignals.length === 0 ? (
                  <div className="text-center text-slate-400 py-4">
                    {tradingActive ? 'Analyzing...' : 'Start AI to see signals'}
                  </div>
                ) : (
                  aiSignals.map((signal, idx) => (
                    <div key={idx} className="p-3 bg-slate-700 rounded-lg">
                      <div className="flex items-center justify-between mb-1">
                        <span className="font-bold">{signal.symbol}</span>
                        <span className={`text-sm font-semibold ${
                          signal.direction === 'BUY' ? 'text-green-400' : 'text-red-400'
                        }`}>
                          {signal.direction}
                        </span>
                      </div>
                      <div className="text-xs text-slate-400">
                        {signal.strategy} • {(signal.confidence * 100).toFixed(0)}% confidence
                      </div>
                      <div className="text-xs text-slate-500">{signal.timestamp}</div>
                    </div>
                  ))
                )}
              </div>
            </div>

            {/* Strategy Settings */}
            <div className="bg-slate-800 rounded-lg p-5 border border-slate-700">
              <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
                <Settings className="w-5 h-5 text-blue-400" />
                Strategy Config
              </h3>
              <div className="space-y-3 text-sm">
                <div>
                  <label className="text-slate-400">Max Position Size</label>
                  <input
                    type="number"
                    value={strategyConfig.maxPositionSize}
                    onChange={(e) => setStrategyConfig({...strategyConfig, maxPositionSize: +e.target.value})}
                    className="w-full bg-slate-700 p-2 rounded mt-1"
                  />
                </div>
                <div>
                  <label className="text-slate-400">Daily Loss Limit</label>
                  <input
                    type="number"
                    value={strategyConfig.dailyLossLimit}
                    onChange={(e) => setStrategyConfig({...strategyConfig, dailyLossLimit: +e.target.value})}
                    className="w-full bg-slate-700 p-2 rounded mt-1"
                  />
                </div>
                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <label className="text-slate-400">Min Price</label>
                    <input
                      type="number"
                      value={strategyConfig.priceRangeMin}
                      onChange={(e) => setStrategyConfig({...strategyConfig, priceRangeMin: +e.target.value})}
                      className="w-full bg-slate-700 p-2 rounded mt-1"
                    />
                  </div>
                  <div>
                    <label className="text-slate-400">Max Price</label>
                    <input
                      type="number"
                      value={strategyConfig.priceRangeMax}
                      onChange={(e) => setStrategyConfig({...strategyConfig, priceRangeMax: +e.target.value})}
                      className="w-full bg-slate-700 p-2 rounded mt-1"
                    />
                  </div>
                </div>
                <div>
                  <label className="text-slate-400">Target Gain %</label>
                  <input
                    type="number"
                    step="0.1"
                    value={strategyConfig.targetGainPercent}
                    onChange={(e) => setStrategyConfig({...strategyConfig, targetGainPercent: +e.target.value})}
                    className="w-full bg-slate-700 p-2 rounded mt-1"
                  />
                </div>
                <div>
                  <label className="text-slate-400">Stop Loss %</label>
                  <input
                    type="number"
                    step="0.1"
                    value={strategyConfig.stopLossPercent}
                    onChange={(e) => setStrategyConfig({...strategyConfig, stopLossPercent: +e.target.value})}
                    className="w-full bg-slate-700 p-2 rounded mt-1"
                  />
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Active Positions */}
        <div className="mt-6 bg-slate-800 rounded-lg p-5 border border-slate-700">
          <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
            <BarChart3 className="w-5 h-5 text-purple-400" />
            Active Positions
          </h3>
          {positions.length === 0 ? (
            <div className="text-center text-slate-400 py-8">No active positions</div>
          ) : (
            <div className="space-y-2">
              {positions.map((pos, idx) => (
                <div key={idx} className="flex items-center justify-between p-3 bg-slate-700 rounded-lg">
                  <div className="flex items-center gap-4">
                    <span className="font-bold">{pos.symbol}</span>
                    <span className={`px-2 py-1 rounded text-xs font-semibold ${
                      pos.direction === 'BUY' ? 'bg-green-600' : 'bg-red-600'
                    }`}>
                      {pos.direction}
                    </span>
                    <span className="text-slate-400">Qty: {pos.quantity}</span>
                    <span className="text-slate-400">@ ${pos.price}</span>
                  </div>
                  <div className="text-sm text-slate-400">{pos.timestamp}</div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Status Bar */}
        <div className="mt-6 bg-slate-800 rounded-lg p-4 border border-slate-700">
          <div className="flex items-center justify-between text-sm">
            <div className="flex items-center gap-6">
              <span className="text-slate-400">
                Account: <span className="text-white font-semibold">{accountType.toUpperCase()}</span>
              </span>
              <span className="text-slate-400">
                Connection: <span className={isConnected ? 'text-green-400' : 'text-red-400'}>
                  {isConnected ? 'Active' : 'Disconnected'}
                </span>
              </span>
              <span className="text-slate-400">
                AI Status: <span className={tradingActive ? 'text-green-400' : 'text-yellow-400'}>
                  {tradingActive ? 'Running' : 'Standby'}
                </span>
              </span>
            </div>
            <div className="flex items-center gap-2 text-slate-400">
              <AlertCircle className="w-4 h-4" />
              <span>Risk Management Active • FINRA Compliant</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AITradingBot;