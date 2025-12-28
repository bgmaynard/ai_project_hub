/**
 * MarketContext Component
 *
 * Displays market regime, volatility, liquidity, and data freshness.
 */

import React from 'react';
import type { MarketContext as MarketContextType } from '../../types/governor';
import { getRegimeColor, getStatusColor, GOVERNOR_COLORS } from '../../types/governor';

interface Props {
  context: MarketContextType;
}

export const MarketContext: React.FC<Props> = ({ context }) => {
  const {
    regime,
    regimeConfidence,
    volatility,
    volatilityPct,
    liquidity,
    dataAge,
    dataFreshness,
    lastUpdate,
    aiInterpretation,
  } = context;

  const formatTime = (isoString: string) => {
    const date = new Date(isoString);
    return date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  };

  const formatDataAge = (seconds: number) => {
    if (seconds < 60) return `${seconds}s`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m`;
    return `${Math.floor(seconds / 3600)}h`;
  };

  return (
    <div className="bg-ibkr-surface rounded-lg border border-ibkr-border p-4">
      <h2 className="text-xs font-bold uppercase tracking-wider text-ibkr-text-secondary mb-4">
        Market Context
      </h2>

      <div className="space-y-3">
        {/* Regime */}
        <div className="flex justify-between items-center">
          <span className="text-sm text-ibkr-text-secondary">Regime</span>
          <div className="text-right">
            <span
              className="text-sm font-bold"
              style={{ color: getRegimeColor(regime) }}
            >
              {regime.replace('_', ' ')}
            </span>
            <span className="text-xs text-ibkr-text-secondary ml-2">
              ({regimeConfidence}%)
            </span>
          </div>
        </div>

        {/* Volatility */}
        <div className="flex justify-between items-center">
          <span className="text-sm text-ibkr-text-secondary">Volatility</span>
          <div className="text-right">
            <span
              className="text-sm font-semibold"
              style={{
                color: volatility === 'LOW' ? GOVERNOR_COLORS.success :
                       volatility === 'NORMAL' ? GOVERNOR_COLORS.neutral :
                       volatility === 'HIGH' ? GOVERNOR_COLORS.warning :
                       GOVERNOR_COLORS.error
              }}
            >
              {volatility}
            </span>
            <span className="text-xs text-ibkr-text-secondary ml-2">
              ({volatilityPct.toFixed(1)}%)
            </span>
          </div>
        </div>

        {/* Liquidity */}
        <div className="flex justify-between items-center">
          <span className="text-sm text-ibkr-text-secondary">Liquidity</span>
          <span
            className="text-sm font-semibold"
            style={{
              color: liquidity === 'ADEQUATE' ? GOVERNOR_COLORS.success :
                     liquidity === 'THIN' ? GOVERNOR_COLORS.warning :
                     GOVERNOR_COLORS.error
            }}
          >
            {liquidity}
          </span>
        </div>

        {/* Data Age */}
        <div className="flex justify-between items-center">
          <span className="text-sm text-ibkr-text-secondary">Data Age</span>
          <div className="text-right">
            <span
              className="text-sm font-semibold"
              style={{ color: getStatusColor(dataFreshness) }}
            >
              {formatDataAge(dataAge)}
            </span>
            <span
              className="text-xs ml-2"
              style={{ color: getStatusColor(dataFreshness) }}
            >
              ({dataFreshness})
            </span>
          </div>
        </div>

        {/* AI Interpretation */}
        {aiInterpretation && (
          <div className="pt-3 mt-3 border-t border-ibkr-border">
            <div className="text-xs text-ibkr-text-secondary mb-1">AI Interpretation:</div>
            <div className="text-sm text-ibkr-text italic">
              {aiInterpretation}
            </div>
          </div>
        )}

        {/* Last Update */}
        <div className="pt-2 border-t border-ibkr-border">
          <div className="text-xs text-ibkr-text-secondary text-center">
            Last Update: {formatTime(lastUpdate)}
          </div>
        </div>
      </div>
    </div>
  );
};

export default MarketContext;
