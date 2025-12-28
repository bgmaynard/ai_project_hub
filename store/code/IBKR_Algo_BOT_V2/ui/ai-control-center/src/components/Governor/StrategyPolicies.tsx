/**
 * StrategyPolicies Component
 *
 * Displays strategy enablement table with status and reasons.
 */

import React from 'react';
import type { StrategyPolicy } from '../../types/governor';
import { getStatusColor, GOVERNOR_COLORS } from '../../types/governor';

interface Props {
  policies: StrategyPolicy[];
}

export const StrategyPolicies: React.FC<Props> = ({ policies }) => {
  const formatCooldown = (seconds?: number) => {
    if (!seconds || seconds <= 0) return '';
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${String(secs).padStart(2, '0')}`;
  };

  return (
    <div className="bg-ibkr-surface rounded-lg border border-ibkr-border">
      <div className="p-4 border-b border-ibkr-border">
        <h2 className="text-xs font-bold uppercase tracking-wider text-ibkr-text-secondary">
          Strategy Policies
        </h2>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="bg-ibkr-bg text-ibkr-text-secondary text-xs uppercase">
              <th className="text-left p-3 font-medium">Strategy</th>
              <th className="text-center p-3 font-medium">Status</th>
              <th className="text-left p-3 font-medium">Reason</th>
              <th className="text-center p-3 font-medium">Vetoes</th>
            </tr>
          </thead>
          <tbody>
            {policies.length === 0 ? (
              <tr>
                <td colSpan={4} className="p-4 text-center text-ibkr-text-secondary">
                  No strategies configured
                </td>
              </tr>
            ) : (
              policies.map((policy) => (
                <tr
                  key={policy.id}
                  className="border-t border-ibkr-border hover:bg-ibkr-bg transition-colors"
                >
                  <td className="p-3 font-medium text-ibkr-text">
                    {policy.name}
                  </td>
                  <td className="p-3 text-center">
                    <span
                      className="inline-flex items-center px-2 py-0.5 rounded text-xs font-bold"
                      style={{
                        backgroundColor: `${getStatusColor(policy.status)}22`,
                        color: getStatusColor(policy.status),
                      }}
                    >
                      {policy.status}
                      {policy.status === 'PAUSED' && policy.cooldownRemaining && policy.cooldownRemaining > 0 && (
                        <span className="ml-1 font-normal">
                          {formatCooldown(policy.cooldownRemaining)}
                        </span>
                      )}
                    </span>
                  </td>
                  <td className="p-3 text-ibkr-text-secondary text-xs max-w-xs truncate">
                    {policy.reason}
                  </td>
                  <td className="p-3 text-center">
                    {policy.vetoCount > 0 ? (
                      <span
                        className="inline-block min-w-[24px] text-center px-1.5 py-0.5 rounded text-xs font-bold"
                        style={{
                          backgroundColor: policy.vetoCount >= 5 ? '#f4877133' : '#dcdcaa33',
                          color: policy.vetoCount >= 5 ? GOVERNOR_COLORS.error : GOVERNOR_COLORS.warning,
                        }}
                      >
                        {policy.vetoCount}
                      </span>
                    ) : (
                      <span className="text-ibkr-text-secondary">0</span>
                    )}
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default StrategyPolicies;
