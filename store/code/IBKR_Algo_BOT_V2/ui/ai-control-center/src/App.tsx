import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link, useLocation, Navigate } from 'react-router-dom';
import ModelManagement from './components/ModelManagement';
import Backtesting from './components/Backtesting';
import LivePredictions from './components/LivePredictions';
import TradingViewHub from './components/TradingView';
import ClaudeOrchestrator from './components/ClaudeOrchestrator';
import DailyReview from './components/DailyReview';
import OptimizationAdvisor from './components/OptimizationAdvisor';
import Worklist from './components/Worklist';
import WarriorTrading from './components/WarriorTrading';

// Navigation component
const Navigation: React.FC = () => {
  const location = useLocation();

  const navItems = [
    { path: '/warrior', label: 'Warrior Trading', icon: '⚡' },
    { path: '/worklist', label: 'Worklist', icon: '📋' },
    { path: '/models', label: 'Model Management', icon: '🤖' },
    { path: '/backtest', label: 'Backtesting Lab', icon: '🔬' },
    { path: '/predictions', label: 'Live Predictions', icon: '📡' },
    { path: '/tradingview', label: 'TradingView Hub', icon: '📈' },
    { path: '/claude', label: 'Claude AI', icon: '🧠' },
    { path: '/daily-review', label: 'Daily Review', icon: '📊' },
    { path: '/optimizer', label: 'Optimizer', icon: '🔧' },
  ];

  const externalLink = {
    url: '/complete_platform.html',
    label: 'Trading Platform',
    icon: '💹'
  };

  return (
    <nav className="bg-ibkr-surface border-b border-ibkr-border px-4 py-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-6">
          <Link to="/models" className="flex items-center space-x-2">
            <span className="text-xl">🤖</span>
            <h1 className="text-lg font-bold text-ibkr-text">AI Control Center</h1>
          </Link>

          <div className="flex space-x-1">
            {navItems.map((item) => {
              const isActive = location.pathname === item.path;
              return (
                <Link
                  key={item.path}
                  to={item.path}
                  className={`px-3 py-2 rounded text-sm transition-colors ${
                    isActive
                      ? 'bg-ibkr-accent text-white'
                      : 'text-ibkr-text-secondary hover:bg-ibkr-bg hover:text-ibkr-text'
                  }`}
                >
                  <span className="mr-1.5">{item.icon}</span>
                  {item.label}
                </Link>
              );
            })}
          </div>
        </div>

        <div className="flex items-center space-x-4">
          <a
            href={externalLink.url}
            target="_blank"
            rel="noopener noreferrer"
            className="px-3 py-2 rounded text-sm bg-ibkr-accent text-white hover:bg-opacity-90 transition-colors flex items-center space-x-1.5"
          >
            <span>{externalLink.icon}</span>
            <span>{externalLink.label}</span>
            <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
            </svg>
          </a>
          <div className="flex items-center space-x-2 text-xs text-ibkr-text-secondary">
            <span className="w-2 h-2 bg-ibkr-success rounded-full animate-pulse"></span>
            <span>API Connected</span>
          </div>
        </div>
      </div>
    </nav>
  );
};

// Main App Layout
const AppLayout: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  return (
    <div className="min-h-screen bg-ibkr-bg">
      <Navigation />
      <main className="h-[calc(100vh-56px)] overflow-auto">
        {children}
      </main>
    </div>
  );
};

// Main App Component
function App() {
  return (
    <Router>
      <AppLayout>
        <Routes>
          <Route path="/" element={<Navigate to="/warrior" replace />} />
          <Route path="/warrior" element={<WarriorTrading />} />
          <Route path="/worklist" element={<Worklist />} />
          <Route path="/models" element={<ModelManagement />} />
          <Route path="/backtest" element={<Backtesting />} />
          <Route path="/predictions" element={<LivePredictions />} />
          <Route path="/tradingview" element={<TradingViewHub />} />
          <Route path="/claude" element={<ClaudeOrchestrator />} />
          <Route path="/daily-review" element={<DailyReview />} />
          <Route path="/optimizer" element={<OptimizationAdvisor />} />
          <Route path="*" element={<Navigate to="/warrior" replace />} />
        </Routes>
      </AppLayout>
    </Router>
  );
}

export default App;
