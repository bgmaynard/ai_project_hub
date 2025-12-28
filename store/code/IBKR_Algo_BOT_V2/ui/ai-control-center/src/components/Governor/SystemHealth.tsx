/**
 * SystemHealth Component
 *
 * Displays system safety status and health indicators in a compact grid.
 */

import React from 'react';
import type { SystemHealth as SystemHealthType } from '../../types/governor';
import { getStatusColor, GOVERNOR_COLORS } from '../../types/governor';

interface Props {
  indicators: SystemHealthType;
}

const SAFETY_COLORS: Record<string, string> = {
  SAFE: GOVERNOR_COLORS.success,
  DEGRADED: GOVERNOR_COLORS.warning,
  HALTED: GOVERNOR_COLORS.error,
};

export const SystemHealth: React.FC<Props> = ({ indicators }) => {
  const { safetyStatus, indicators: healthIndicators } = indicators;

  const getStatusDot = (status: string) => {
    const color = getStatusColor(status);
    const isAnimated = status === 'HEALTHY' || status === 'CONNECTED';

    return (
      <span
        className={`inline-block w-2 h-2 rounded-full ${isAnimated ? 'animate-pulse' : ''}`}
        style={{ backgroundColor: color }}
      />
    );
  };

  return (
    <div className="bg-ibkr-surface rounded-lg border border-ibkr-border">
      <div className="p-4 border-b border-ibkr-border">
        <h2 className="text-xs font-bold uppercase tracking-wider text-ibkr-text-secondary">
          System Health
        </h2>
      </div>

      <div className="p-4">
        {/* System Safety Status - Top Line */}
        <div
          className="mb-4 p-3 rounded text-center font-bold uppercase"
          style={{
            backgroundColor: `${SAFETY_COLORS[safetyStatus]}22`,
            color: SAFETY_COLORS[safetyStatus],
            border: `1px solid ${SAFETY_COLORS[safetyStatus]}`,
          }}
        >
          SYSTEM SAFETY STATUS: {safetyStatus}
        </div>

        {/* Health Indicators Grid */}
        <div className="grid grid-cols-2 gap-3">
          {healthIndicators.map((indicator) => (
            <div
              key={indicator.name}
              className="flex items-center justify-between p-2 rounded bg-ibkr-bg"
            >
              <div className="flex items-center gap-2">
                {getStatusDot(indicator.status)}
                <span className="text-xs font-medium text-ibkr-text">
                  {indicator.name}
                </span>
              </div>
              <span
                className="text-xs"
                style={{ color: getStatusColor(indicator.status) }}
              >
                {indicator.detail}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default SystemHealth;
