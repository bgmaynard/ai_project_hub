import React, { useState, useEffect } from 'react';

interface PatternFlag {
  id: string;
  severity: string;
  description: string;
  suggested_action: string;
}

interface DeroStatus {
  status: string;
  latest_daily_report: string | null;
  current_time: {
    current_time_et: string;
    current_window: string;
    is_market_hours: boolean;
  };
}

interface DailyReport {
  date: string;
  data_health: string;
  market?: {
    regime: string;
    confidence: number;
  };
  outcomes?: {
    total_trades: number;
    win_rate_pct: number;
    total_pnl: number;
  };
  patterns?: {
    pattern_flags: PatternFlag[];
  };
}

const DeroPanel: React.FC = () => {
  const [status, setStatus] = useState<DeroStatus | null>(null);
  const [latestReport, setLatestReport] = useState<DailyReport | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchData = async () => {
    try {
      // Fetch DERO status
      const statusRes = await fetch('/api/reports/status');
      if (statusRes.ok) {
        const statusData = await statusRes.json();
        setStatus(statusData);
      }

      // Fetch latest daily report
      const reportRes = await fetch('/api/reports/daily/latest');
      if (reportRes.ok) {
        const reportData = await reportRes.json();
        setLatestReport(reportData);
      }

      setError(null);
    } catch (e) {
      setError('Failed to load DERO data');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 60000); // Refresh every minute
    return () => clearInterval(interval);
  }, []);

  const getHealthColor = (health: string): string => {
    switch (health) {
      case 'GREEN': return 'text-green-400';
      case 'YELLOW': return 'text-yellow-400';
      case 'RED': return 'text-red-400';
      default: return 'text-gray-400';
    }
  };

  const getHealthEmoji = (health: string): string => {
    switch (health) {
      case 'GREEN': return '🟢';
      case 'YELLOW': return '🟡';
      case 'RED': return '🔴';
      default: return '⚪';
    }
  };

  const getRegimeEmoji = (regime: string): string => {
    switch (regime) {
      case 'TREND': return '📈';
      case 'CHOP': return '↔️';
      case 'NEWS': return '📰';
      case 'DEAD': return '💤';
      default: return '❓';
    }
  };

  const getSeverityBadge = (severity: string): string => {
    switch (severity) {
      case 'CONFIRMED': return 'bg-red-600';
      case 'POSSIBLE': return 'bg-yellow-600';
      default: return 'bg-gray-600';
    }
  };

  if (loading) {
    return (
      <div className="bg-gray-800 rounded-lg p-4">
        <h3 className="text-white font-semibold mb-2">Daily Evaluation</h3>
        <div className="text-gray-400 text-sm">Loading...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-gray-800 rounded-lg p-4">
        <h3 className="text-white font-semibold mb-2">Daily Evaluation</h3>
        <div className="text-red-400 text-sm">{error}</div>
      </div>
    );
  }

  const health = latestReport?.data_health || 'UNKNOWN';
  const regime = latestReport?.market?.regime || 'UNKNOWN';
  const patterns = latestReport?.patterns?.pattern_flags?.slice(0, 3) || [];
  const outcomes = latestReport?.outcomes;

  return (
    <div className="bg-gray-800 rounded-lg p-4">
      <div className="flex justify-between items-center mb-3">
        <h3 className="text-white font-semibold">Daily Evaluation</h3>
        <span className="text-xs text-gray-400">
          {latestReport?.date || 'No report'}
        </span>
      </div>

      {/* Status Row */}
      <div className="grid grid-cols-2 gap-2 mb-3">
        <div className="bg-gray-700 rounded p-2">
          <div className="text-xs text-gray-400">Data Health</div>
          <div className={`font-semibold ${getHealthColor(health)}`}>
            {getHealthEmoji(health)} {health}
          </div>
        </div>
        <div className="bg-gray-700 rounded p-2">
          <div className="text-xs text-gray-400">Market Regime</div>
          <div className="text-white font-semibold">
            {getRegimeEmoji(regime)} {regime}
          </div>
        </div>
      </div>

      {/* Outcomes */}
      {outcomes && (
        <div className="bg-gray-700 rounded p-2 mb-3">
          <div className="flex justify-between text-sm">
            <span className="text-gray-400">Trades:</span>
            <span className="text-white">{outcomes.total_trades}</span>
          </div>
          <div className="flex justify-between text-sm">
            <span className="text-gray-400">Win Rate:</span>
            <span className="text-white">{outcomes.win_rate_pct.toFixed(1)}%</span>
          </div>
          <div className="flex justify-between text-sm">
            <span className="text-gray-400">P&L:</span>
            <span className={outcomes.total_pnl >= 0 ? 'text-green-400' : 'text-red-400'}>
              ${outcomes.total_pnl.toFixed(2)}
            </span>
          </div>
        </div>
      )}

      {/* Pattern Flags */}
      <div className="mb-3">
        <div className="text-xs text-gray-400 mb-1">Pattern Flags</div>
        {patterns.length > 0 ? (
          <div className="space-y-1">
            {patterns.map((p, i) => (
              <div key={i} className="flex items-center gap-2 text-xs">
                <span className={`px-1 rounded ${getSeverityBadge(p.severity)}`}>
                  {p.severity}
                </span>
                <span className="text-white truncate">{p.id}</span>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-green-400 text-sm">✅ No patterns detected</div>
        )}
      </div>

      {/* Current Window */}
      {status?.current_time && (
        <div className="text-xs text-gray-400 border-t border-gray-700 pt-2">
          <span className="text-white">{status.current_time.current_window}</span>
          {' '}-{' '}
          {status.current_time.is_market_hours ? (
            <span className="text-green-400">Market Open</span>
          ) : (
            <span className="text-gray-400">Market Closed</span>
          )}
        </div>
      )}

      {/* Open Report Link */}
      <div className="mt-3 text-center">
        <a
          href={`/reports/daily/daily_eval_${latestReport?.date || 'latest'}.md`}
          target="_blank"
          rel="noopener noreferrer"
          className="text-blue-400 text-xs hover:underline"
        >
          Open Full Report
        </a>
      </div>
    </div>
  );
};

export default DeroPanel;
