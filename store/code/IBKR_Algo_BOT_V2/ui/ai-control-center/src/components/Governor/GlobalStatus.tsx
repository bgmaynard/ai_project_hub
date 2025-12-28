/**
 * GlobalStatus Component
 *
 * Displays trading mode, window, AI state, and kill switch status.
 */

import React from 'react';
import type { GlobalStatus as GlobalStatusType } from '../../types/governor';
import { getStatusColor, GOVERNOR_COLORS } from '../../types/governor';

interface Props {
  status: GlobalStatusType;
}

export const GlobalStatus: React.FC<Props> = ({ status }) => {
  const { mode, tradingWindow, windowTime, aiState, killSwitch } = status;

  return (
    <div className="bg-ibkr-surface rounded-lg border border-ibkr-border p-4">
      <h2 className="text-xs font-bold uppercase tracking-wider text-ibkr-text-secondary mb-4">
        Global Trading Status
      </h2>

      <div className="space-y-3">
        {/* Mode */}
        <div className="flex justify-between items-center">
          <span className="text-sm text-ibkr-text-secondary">Mode</span>
          <span
            className="text-sm font-bold px-2 py-0.5 rounded"
            style={{
              backgroundColor: mode === 'PAPER' ? '#007acc33' : '#f4877133',
              color: mode === 'PAPER' ? GOVERNOR_COLORS.active : GOVERNOR_COLORS.error,
            }}
          >
            {mode}
          </span>
        </div>

        {/* Trading Window */}
        <div className="flex justify-between items-center">
          <span className="text-sm text-ibkr-text-secondary">Window</span>
          <div className="text-right">
            <span
              className="text-sm font-semibold"
              style={{ color: getStatusColor(tradingWindow === 'OPEN' ? 'HEALTHY' : 'WARNING') }}
            >
              {tradingWindow.replace('_', ' ')}
            </span>
            <div className="text-xs text-ibkr-text-secondary">{windowTime}</div>
          </div>
        </div>

        {/* AI State */}
        <div className="flex justify-between items-center">
          <span className="text-sm text-ibkr-text-secondary">AI State</span>
          <span
            className="text-sm font-semibold"
            style={{ color: getStatusColor(aiState) }}
          >
            {aiState}
          </span>
        </div>

        {/* Kill Switch */}
        <div
          className="mt-4 p-3 rounded border text-center"
          style={{
            backgroundColor: killSwitch.active ? '#f4877122' : '#4ec9b022',
            borderColor: killSwitch.active ? GOVERNOR_COLORS.error : GOVERNOR_COLORS.success,
          }}
        >
          <div
            className="text-sm font-bold uppercase"
            style={{ color: killSwitch.active ? GOVERNOR_COLORS.error : GOVERNOR_COLORS.success }}
          >
            Kill Switch: {killSwitch.active ? 'ACTIVE' : 'OFF'}
          </div>
          {killSwitch.active && killSwitch.reason && (
            <div className="text-xs mt-1" style={{ color: GOVERNOR_COLORS.error }}>
              {killSwitch.reason}
            </div>
          )}
          {killSwitch.cooldownSeconds > 0 && (
            <div className="text-xs mt-1 text-ibkr-text-secondary">
              Cooldown: {Math.floor(killSwitch.cooldownSeconds / 60)}:{String(Math.floor(killSwitch.cooldownSeconds % 60)).padStart(2, '0')}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default GlobalStatus;
