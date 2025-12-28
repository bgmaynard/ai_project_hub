/**
 * RecentDecisions Component
 *
 * Displays the last 10 AI decisions with timestamps and reasons.
 */

import React from 'react';
import type { AIDecision } from '../../types/governor';
import { getStatusColor, GOVERNOR_COLORS } from '../../types/governor';

interface Props {
  decisions: AIDecision[];
}

export const RecentDecisions: React.FC<Props> = ({ decisions }) => {
  const formatTime = (isoString: string) => {
    const date = new Date(isoString);
    return date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  };

  const getActionIcon = (action: string) => {
    switch (action) {
      case 'APPROVED': return '+';
      case 'VETOED': return 'X';
      case 'EXIT': return '-';
      default: return '?';
    }
  };

  return (
    <div className="bg-ibkr-surface rounded-lg border border-ibkr-border">
      <div className="p-4 border-b border-ibkr-border flex justify-between items-center">
        <h2 className="text-xs font-bold uppercase tracking-wider text-ibkr-text-secondary">
          Recent AI Decisions
        </h2>
        <span className="text-xs text-ibkr-text-secondary">
          Last {decisions.length}
        </span>
      </div>

      <div className="max-h-[250px] overflow-y-auto">
        {decisions.length === 0 ? (
          <div className="p-4 text-center text-ibkr-text-secondary text-sm">
            No recent decisions
          </div>
        ) : (
          <div className="divide-y divide-ibkr-border">
            {decisions.map((decision, index) => (
              <div
                key={`${decision.timestamp}-${index}`}
                className="p-3 hover:bg-ibkr-bg transition-colors"
              >
                <div className="flex items-start gap-3">
                  {/* Timestamp */}
                  <span className="text-xs font-mono text-ibkr-text-secondary whitespace-nowrap">
                    {formatTime(decision.timestamp)}
                  </span>

                  {/* Action Badge */}
                  <span
                    className="flex items-center justify-center w-20 px-2 py-0.5 rounded text-xs font-bold"
                    style={{
                      backgroundColor: `${getStatusColor(decision.action)}22`,
                      color: getStatusColor(decision.action),
                    }}
                  >
                    <span className="mr-1 font-mono">{getActionIcon(decision.action)}</span>
                    {decision.action}
                  </span>

                  {/* Symbol */}
                  <span className="text-sm font-bold text-ibkr-text min-w-[50px]">
                    {decision.symbol}
                  </span>

                  {/* Type */}
                  <span className="text-xs text-ibkr-text-secondary">
                    {decision.type}
                  </span>

                  {/* P&L (for exits) */}
                  {decision.pnl !== undefined && (
                    <span
                      className="text-xs font-bold ml-auto"
                      style={{
                        color: decision.pnl >= 0 ? GOVERNOR_COLORS.success : GOVERNOR_COLORS.error,
                      }}
                    >
                      {decision.pnl >= 0 ? '+' : ''}${decision.pnl.toFixed(2)}
                    </span>
                  )}
                </div>

                {/* Reasons */}
                <div className="mt-1 ml-[76px] text-xs text-ibkr-text-secondary truncate">
                  {decision.reasons.join(', ')}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default RecentDecisions;
