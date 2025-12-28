/**
 * SystemHealth Component
 *
 * Displays health indicators in a compact grid.
 */

import React from 'react';
import type { HealthIndicator } from '../../types/governor';
import { getStatusColor } from '../../types/governor';

interface Props {
  indicators: HealthIndicator[];
}

export const SystemHealth: React.FC<Props> = ({ indicators }) => {
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
        <div className="grid grid-cols-2 gap-3">
          {indicators.map((indicator) => (
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
