import React, { useState, useEffect } from 'react';
import { Play, Square, AlertCircle, Activity, TrendingUp, DollarSign, Clock } from 'lucide-react';

// API Service
const API_BASE = 'http://localhost:5000/api';

const api = {
  getStatus: async () => {
    const res = await fetch(`${API_BASE}/status`);
    return res.json();
  },
  startMTF: async () => {
    const res = await fetch(`${API_BASE}/mtf/start`, { method: 'POST' });
    return res.json();
  },
  stopMTF: async () => {
    const res = await fetch(`${API_BASE}/mtf/stop`, { method: 'POST' });
    return res.json();
  },
  startWarrior: async () => {
    const res = await fetch(`${API_BASE}/warrior/start`, { method: 'POST' });
    return res.json();
  },
  stopWarrior: async () => {
    const res = await fetch(`${API_BASE}/warrior/stop`, { method: 'POST' });
    return res.json();
  },
  getPositions: async () => {
    const res = await fetch(`${API_BASE}/positions`);
    return res.json();
  },
  getTrades: async () => {
    const res = await fetch(`${API_BASE}/trades`);
    return res.json();
  },
  getLogs: async () => {
    const res = await fetch(`${API_BASE}/logs`);
    return res.json();
  },
  getPnL: async () => {
    const res = await fetch(`${API_BASE}/pnl`);
    return res.json();
  }
};

// Status Badge Component
const StatusBadge = ({ status }) => {
  const colors = {
    running: 'bg-green-500',
    stopped: 'bg-gray-500',
    error: 'bg-red-500'
  };
  
  return (
    <div className={`inline-flex items-center gap-2 px-3 py-1 rounded-full ${colors[status] || 'bg-gray-500'} text-white text-sm`}>
      <div className="w-2 h-2 bg-white rounded-full animate-pulse"></div>
      {status.toUpperCase()}
    </div>
  );
};

// System Card Component
const SystemCard = ({ title, status, active, pnl, onStart, onStop }) => {
  const isRunning = status === 'running';
  
  return (
    <div className="bg-white rounded-lg shadow-md p-6 border-2 border-gray-200">
      <div className="flex justify-between items-start mb-4">
        <h3 className="text-xl font-bold text-gray-800">{title}</h3>
        <StatusBadge status={status} />
      </div>
      
      <div className="space-y-3 mb-4">
        <div className="flex justify-between text-sm">
          <span className="text-gray-600">Active Positions:</span>
          <span className="font-semibold">{active}</span>
        </div>
        <div className="flex justify-between text-sm">
          <span className="text-gray-600">P&L:</span>
          <span className={`font-semibold ${pnl >= 0 ? 'text-green-600' : 'text-red-600'}`}>
            ${pnl.toFixed(2)}
          </span>
        </div>
      </div>
      
      <button
        onClick={isRunning ? onStop : onStart}
        className={`w-full py-2 rounded-lg font-semibold transition-colors ${
          isRunning 
            ? 'bg-red-500 hover:bg-red-600 text-white' 
            : 'bg-green-500 hover:bg-green-600 text-white'
        }`}
      >
        {isRunning ? (
          <span className="flex items-center justify-center gap-2">
            <Square size={16} /> STOP
          </span>
        ) : (
          <span className="flex items-center justify-center gap-2">
            <Play size={16} /> START
          </span>
        )}
      </button>
    </div>
  );
};

// Position Row Component
const PositionRow = ({ symbol, qty, entry, current, pnl, duration, source }) => {
  return (
    <tr className="border-b border-gray-200 hover:bg-gray-50">
      <td className="py-3 px-4 font-semibold">{symbol}</td>
      <td className="py-3 px-4 text-center">{qty}</td>
      <td className="py-3 px-4 text-right">${entry.toFixed(2)}</td>
      <td className="py-3 px-4 text-right">${current.toFixed(2)}</td>
      <td className={`py-3 px-4 text-right font-semibold ${pnl >= 0 ? 'text-green-600' : 'text-red-600'}`}>
        {pnl >= 0 ? '+' : ''}${pnl.toFixed(2)}
      </td>
      <td className="py-3 px-4 text-center text-gray-600">{duration}</td>
      <td className="py-3 px-4 text-center">
        <span className={`px-2 py-1 rounded text-xs font-semibold ${
          source === 'MTF' ? 'bg-blue-100 text-blue-800' : 'bg-purple-100 text-purple-800'
        }`}>
          {source}
        </span>
      </td>
    </tr>
  );
};

// Trade Row Component
const TradeRow = ({ time, symbol, action, qty, price, pnl, strategy }) => {
  return (
    <tr className="border-b border-gray-200 hover:bg-gray-50">
      <td className="py-2 px-4 text-sm text-gray-600">{time}</td>
      <td className="py-2 px-4 font-semibold">{symbol}</td>
      <td className="py-2 px-4">
        <span className={`px-2 py-1 rounded text-xs font-semibold ${
          action === 'BUY' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
        }`}>
          {action}
        </span>
      </td>
      <td className="py-2 px-4 text-center">{qty}</td>
      <td className="py-2 px-4 text-right">${price.toFixed(2)}</td>
      <td className={`py-2 px-4 text-right font-semibold ${pnl >= 0 ? 'text-green-600' : 'text-red-600'}`}>
        {pnl >= 0 ? '+' : ''}${pnl.toFixed(2)}
      </td>
      <td className="py-2 px-4 text-center">
        <span className={`px-2 py-1 rounded text-xs font-semibold ${
          strategy === 'MTF' ? 'bg-blue-100 text-blue-800' : 'bg-purple-100 text-purple-800'
        }`}>
          {strategy}
        </span>
      </td>
    </tr>
  );
};

// Log Entry Component
const LogEntry = ({ timestamp, level, source, message }) => {
  const levelColors = {
    info: 'text-blue-600',
    success: 'text-green-600',
    warning: 'text-yellow-600',
    error: 'text-red-600'
  };
  
  const sourceColors = {
    mtf: 'bg-blue-100 text-blue-800',
    warrior: 'bg-purple-100 text-purple-800',
    ibkr: 'bg-gray-100 text-gray-800',
    system: 'bg-orange-100 text-orange-800'
  };
  
  return (
    <div className="py-2 px-4 hover:bg-gray-50 border-b border-gray-100">
      <div className="flex items-start gap-3">
        <span className="text-xs text-gray-500 font-mono">{new Date(timestamp).toLocaleTimeString()}</span>
        <span className={`px-2 py-0.5 rounded text-xs font-semibold ${sourceColors[source] || 'bg-gray-100'}`}>
          {source.toUpperCase()}
        </span>
        <span className={`text-sm ${levelColors[level] || 'text-gray-600'}`}>{message}</span>
      </div>
    </div>
  );
};

// Main Dashboard Component
const TradingDashboard = () => {
  const [systemStatus, setSystemStatus] = useState({
    mtf: { status: 'stopped', active_positions: [], pnl: 0 },
    warrior: { status: 'stopped', active_positions: [], pnl: 0 },
    ibkr: { connected: false, account_value: 0 }
  });
  
  const [positions, setPositions] = useState([]);
  const [trades, setTrades] = useState([]);
  const [logs, setLogs] = useState([]);
  const [pnlData, setPnlData] = useState({ total: 0, mtf: 0, warrior: 0 });

  // Fetch data on mount and set up polling
  useEffect(() => {
    fetchAllData();
    const interval = setInterval(fetchAllData, 5000); // Update every 5 seconds
    return () => clearInterval(interval);
  }, []);

  const fetchAllData = async () => {
    try {
      const [status, pos, trd, lg, pnl] = await Promise.all([
        api.getStatus(),
        api.getPositions(),
        api.getTrades(),
        api.getLogs(),
        api.getPnL()
      ]);
      
      setSystemStatus(status);
      setPositions(pos.positions || []);
      setTrades(trd.trades || []);
      setLogs(lg.logs || []);
      setPnlData(pnl);
    } catch (error) {
      console.error('Error fetching data:', error);
    }
  };

  const handleStartMTF = async () => {
    await api.startMTF();
    fetchAllData();
  };

  const handleStopMTF = async () => {
    await api.stopMTF();
    fetchAllData();
  };

  const handleStartWarrior = async () => {
    await api.startWarrior();
    fetchAllData();
  };

  const handleStopWarrior = async () => {
    await api.stopWarrior();
    fetchAllData();
  };

  return (
    <div className="min-h-screen bg-gray-100 p-6">
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-3xl font-bold text-gray-800 flex items-center gap-3">
          <Activity className="text-blue-600" />
          Trading Bot Dashboard
        </h1>
        <p className="text-gray-600 mt-1">Monitor and control your trading systems</p>
      </div>

      {/* System Status Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        <SystemCard
          title="MTF Swing Trading"
          status={systemStatus.mtf.status}
          active={systemStatus.mtf.active_positions?.length || 0}
          pnl={pnlData.mtf?.current || 0}
          onStart={handleStartMTF}
          onStop={handleStopMTF}
        />
        
        <SystemCard
          title="Warrior Momentum"
          status={systemStatus.warrior.status}
          active={systemStatus.warrior.active_positions?.length || 0}
          pnl={pnlData.warrior?.current || 0}
          onStart={handleStartWarrior}
          onStop={handleStopWarrior}
        />
        
        <div className="bg-white rounded-lg shadow-md p-6 border-2 border-gray-200">
          <h3 className="text-xl font-bold text-gray-800 mb-4">IBKR Status</h3>
          <div className="space-y-3">
            <div className="flex justify-between text-sm">
              <span className="text-gray-600">Connection:</span>
              <span className={`font-semibold ${systemStatus.ibkr.connected ? 'text-green-600' : 'text-red-600'}`}>
                {systemStatus.ibkr.connected ? '✓ Connected' : '✗ Disconnected'}
              </span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-gray-600">Port:</span>
              <span className="font-semibold">{systemStatus.ibkr.port}</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-gray-600">Account:</span>
              <span className="font-semibold">${(systemStatus.ibkr.account_value || 0).toLocaleString()}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Total P&L Banner */}
      <div className="bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg shadow-lg p-6 mb-6 text-white">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-blue-100 text-sm">Total Profit & Loss</p>
            <p className="text-4xl font-bold">${pnlData.total.toFixed(2)}</p>
          </div>
          <DollarSign size={48} className="opacity-50" />
        </div>
      </div>

      {/* Current Positions */}
      <div className="bg-white rounded-lg shadow-md mb-6">
        <div className="p-6 border-b border-gray-200">
          <h2 className="text-xl font-bold text-gray-800 flex items-center gap-2">
            <TrendingUp className="text-blue-600" />
            Current Positions
          </h2>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gray-50">
              <tr>
                <th className="py-3 px-4 text-left font-semibold text-gray-700">Symbol</th>
                <th className="py-3 px-4 text-center font-semibold text-gray-700">Qty</th>
                <th className="py-3 px-4 text-right font-semibold text-gray-700">Entry</th>
                <th className="py-3 px-4 text-right font-semibold text-gray-700">Current</th>
                <th className="py-3 px-4 text-right font-semibold text-gray-700">P&L</th>
                <th className="py-3 px-4 text-center font-semibold text-gray-700">Duration</th>
                <th className="py-3 px-4 text-center font-semibold text-gray-700">Source</th>
              </tr>
            </thead>
            <tbody>
              {positions.length === 0 ? (
                <tr>
                  <td colSpan="7" className="py-8 text-center text-gray-500">
                    No open positions
                  </td>
                </tr>
              ) : (
                positions.map((pos, idx) => (
                  <PositionRow key={idx} {...pos} />
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Recent Trades */}
      <div className="bg-white rounded-lg shadow-md mb-6">
        <div className="p-6 border-b border-gray-200">
          <h2 className="text-xl font-bold text-gray-800 flex items-center gap-2">
            <Clock className="text-blue-600" />
            Recent Trades (Last 10)
          </h2>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gray-50">
              <tr>
                <th className="py-3 px-4 text-left font-semibold text-gray-700">Time</th>
                <th className="py-3 px-4 text-left font-semibold text-gray-700">Symbol</th>
                <th className="py-3 px-4 text-left font-semibold text-gray-700">Action</th>
                <th className="py-3 px-4 text-center font-semibold text-gray-700">Qty</th>
                <th className="py-3 px-4 text-right font-semibold text-gray-700">Price</th>
                <th className="py-3 px-4 text-right font-semibold text-gray-700">P&L</th>
                <th className="py-3 px-4 text-center font-semibold text-gray-700">Strategy</th>
              </tr>
            </thead>
            <tbody>
              {trades.slice(0, 10).length === 0 ? (
                <tr>
                  <td colSpan="7" className="py-8 text-center text-gray-500">
                    No recent trades
                  </td>
                </tr>
              ) : (
                trades.slice(0, 10).map((trade, idx) => (
                  <TradeRow key={idx} {...trade} />
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Activity Log */}
      <div className="bg-white rounded-lg shadow-md">
        <div className="p-6 border-b border-gray-200">
          <h2 className="text-xl font-bold text-gray-800 flex items-center gap-2">
            <AlertCircle className="text-blue-600" />
            Live Activity Log
          </h2>
        </div>
        <div className="max-h-96 overflow-y-auto">
          {logs.length === 0 ? (
            <div className="py-8 text-center text-gray-500">
              No activity yet
            </div>
          ) : (
            logs.map((log, idx) => (
              <LogEntry key={idx} {...log} />
            ))
          )}
        </div>
      </div>
    </div>
  );
};

export default TradingDashboard;