"""
Chronos Adapter - Regime Context Provider
==========================================
Wraps Chronos to provide ONLY market context and regime classification.

ARCHITECTURAL PRINCIPLE:
Chronos MUST NOT:
- Place trades
- Suggest trade actions (BUY/SELL)
- Create or modify SignalContracts
- Make any trading decisions

Chronos MUST ONLY:
- Classify current market regime
- Provide confidence levels
- Provide probabilistic context (for informational purposes)
- Output ChronosContext for the gating engine

This adapter enforces these constraints by wrapping the underlying
Chronos predictor and exposing ONLY context-related methods.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List
from datetime import datetime
import logging

from ai.signal_contract import ChronosContext, MarketRegime

logger = logging.getLogger(__name__)


class ChronosAdapter:
    """
    Adapter that restricts Chronos to CONTEXT-ONLY output.

    This adapter:
    1. Wraps the underlying ChronosPredictor
    2. Strips out any trading signals/actions
    3. Returns ONLY ChronosContext for the gating engine

    The separation ensures Chronos cannot influence trading decisions
    beyond providing regime context for signal validation.
    """

    def __init__(self):
        self.chronos_predictor = None
        self.available = False
        self._init_chronos()

        # Regime detection configuration
        self.volatility_threshold = 1.5  # 1.5x avg = VOLATILE regime
        self.trend_adx_threshold = 25    # ADX > 25 = trending

        logger.info("ChronosAdapter initialized (context-only mode)")

    def _init_chronos(self):
        """Initialize underlying Chronos predictor."""
        try:
            from ai.chronos_predictor import get_chronos_predictor
            self.chronos_predictor = get_chronos_predictor()
            self.available = True
            logger.info("Chronos backend loaded successfully")
        except Exception as e:
            logger.warning(f"Chronos not available: {e}")
            self.available = False

    def get_context(
        self,
        symbol: str,
        df: pd.DataFrame = None,
        horizon: int = 5
    ) -> ChronosContext:
        """
        Get market context from Chronos.

        IMPORTANT: This returns CONTEXT ONLY, not trading signals.
        The returned ChronosContext is used by the GatingEngine
        to validate SignalContracts.

        Args:
            symbol: Stock symbol
            df: Optional price DataFrame. If None, will fetch from yfinance.
            horizon: Forecast horizon in days

        Returns:
            ChronosContext with regime, confidence, and probabilistic info
        """
        context = ChronosContext()

        try:
            # Get Chronos forecast
            if self.chronos_predictor and self.available:
                result = self.chronos_predictor.predict(symbol, horizon=horizon)

                if "error" not in result:
                    # Extract ONLY context information
                    context.prob_up = result.get("probabilities", {}).get("prob_up", 0.5)
                    context.prob_down = 1 - context.prob_up
                    context.expected_return_5d = result.get("expected_return_pct", 0) / 100

                    # Map Chronos signal to regime (informational only)
                    chronos_signal = result.get("signal", "NEUTRAL")
                    context.regime_confidence = result.get("confidence", 0.5)

                    # Derive regime from Chronos output
                    if chronos_signal in ["STRONG_BULLISH", "BULLISH"]:
                        context.market_regime = MarketRegime.TRENDING_UP.value
                    elif chronos_signal in ["STRONG_BEARISH", "BEARISH"]:
                        context.market_regime = MarketRegime.TRENDING_DOWN.value
                    else:
                        context.market_regime = MarketRegime.RANGING.value

            # Enhance with technical analysis if DataFrame provided
            if df is not None and len(df) >= 50:
                context = self._enhance_with_technicals(context, df)

        except Exception as e:
            logger.warning(f"Chronos context extraction failed: {e}")
            context.market_regime = MarketRegime.UNKNOWN.value
            context.regime_confidence = 0.0

        context.computed_at = datetime.now().isoformat()
        return context

    def _enhance_with_technicals(
        self,
        context: ChronosContext,
        df: pd.DataFrame
    ) -> ChronosContext:
        """
        Enhance context with technical analysis.

        This provides additional regime context beyond Chronos forecast.
        """
        try:
            import ta

            close = df['Close']
            high = df['High']
            low = df['Low']

            # Volatility
            returns = close.pct_change().dropna()
            current_vol = returns.std() * np.sqrt(252)
            avg_vol = 0.25  # Baseline ~25% annualized

            context.current_volatility = float(current_vol)

            # Calculate volatility percentile over rolling window
            rolling_vol = returns.rolling(20).std() * np.sqrt(252)
            if len(rolling_vol.dropna()) > 0:
                context.volatility_percentile = float(
                    (rolling_vol.iloc[-1] <= rolling_vol).mean()
                )

            # Check for VOLATILE regime override
            if current_vol > avg_vol * self.volatility_threshold:
                context.market_regime = MarketRegime.VOLATILE.value
                context.regime_confidence = min(current_vol / avg_vol - 0.5, 1.0)

            # Trend analysis with ADX
            try:
                adx = ta.trend.ADXIndicator(high, low, close, window=14)
                adx_value = adx.adx().iloc[-1]
                adx_pos = adx.adx_pos().iloc[-1]
                adx_neg = adx.adx_neg().iloc[-1]

                context.trend_strength = min(adx_value / 50, 1.0)

                if adx_value > self.trend_adx_threshold:
                    if adx_pos > adx_neg:
                        context.trend_direction = 1
                        if context.market_regime != MarketRegime.VOLATILE.value:
                            context.market_regime = MarketRegime.TRENDING_UP.value
                    else:
                        context.trend_direction = -1
                        if context.market_regime != MarketRegime.VOLATILE.value:
                            context.market_regime = MarketRegime.TRENDING_DOWN.value
                else:
                    context.trend_direction = 0
                    if context.market_regime not in [
                        MarketRegime.VOLATILE.value,
                        MarketRegime.TRENDING_UP.value,
                        MarketRegime.TRENDING_DOWN.value
                    ]:
                        context.market_regime = MarketRegime.RANGING.value

            except Exception:
                pass

        except Exception as e:
            logger.debug(f"Technical enhancement failed: {e}")

        return context

    def get_batch_context(
        self,
        symbols: List[str],
        horizon: int = 5
    ) -> Dict[str, ChronosContext]:
        """
        Get context for multiple symbols.

        Args:
            symbols: List of stock symbols
            horizon: Forecast horizon

        Returns:
            Dict mapping symbol -> ChronosContext
        """
        contexts = {}
        for symbol in symbols:
            contexts[symbol] = self.get_context(symbol, horizon=horizon)
        return contexts

    def is_regime_favorable(
        self,
        context: ChronosContext,
        valid_regimes: List[str],
        invalid_regimes: List[str] = None
    ) -> bool:
        """
        Check if current regime is favorable.

        This is a HELPER method for quick regime checking.
        The actual gating decision is made by the GatingEngine.

        Args:
            context: Current ChronosContext
            valid_regimes: List of acceptable regimes
            invalid_regimes: List of regimes to avoid

        Returns:
            True if regime is favorable
        """
        if invalid_regimes and context.market_regime in invalid_regimes:
            return False
        if valid_regimes and context.market_regime not in valid_regimes:
            return False
        return True

    def get_regime_summary(self, context: ChronosContext) -> Dict[str, Any]:
        """
        Get a summary of regime context.

        This is for display/logging purposes only.
        """
        return {
            "regime": context.market_regime,
            "confidence": f"{context.regime_confidence:.1%}",
            "trend": {
                "direction": "UP" if context.trend_direction > 0 else (
                    "DOWN" if context.trend_direction < 0 else "NEUTRAL"
                ),
                "strength": f"{context.trend_strength:.1%}"
            },
            "volatility": {
                "current": f"{context.current_volatility:.1%}",
                "percentile": f"{context.volatility_percentile:.0%}"
            },
            "forecast": {
                "prob_up": f"{context.prob_up:.1%}",
                "expected_return": f"{context.expected_return_5d:.2%}"
            }
        }


# Singleton adapter
_chronos_adapter: Optional[ChronosAdapter] = None


def get_chronos_adapter() -> ChronosAdapter:
    """Get the global Chronos adapter (context-only mode)."""
    global _chronos_adapter
    if _chronos_adapter is None:
        _chronos_adapter = ChronosAdapter()
    return _chronos_adapter


def get_market_context(symbol: str, df: pd.DataFrame = None) -> ChronosContext:
    """
    Convenience function to get market context for a symbol.

    This is the PRIMARY entry point for getting Chronos context.
    """
    adapter = get_chronos_adapter()
    return adapter.get_context(symbol, df)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("CHRONOS ADAPTER TEST (Context-Only Mode)")
    print("=" * 60)

    adapter = get_chronos_adapter()

    symbols = ["AAPL", "TSLA", "SPY"]

    for symbol in symbols:
        print(f"\n{symbol}:")
        context = adapter.get_context(symbol)

        summary = adapter.get_regime_summary(context)
        print(f"  Regime: {summary['regime']} ({summary['confidence']} conf)")
        print(f"  Trend: {summary['trend']['direction']} ({summary['trend']['strength']} strength)")
        print(f"  Volatility: {summary['volatility']['current']} (percentile: {summary['volatility']['percentile']})")
        print(f"  Forecast: {summary['forecast']['prob_up']} up, {summary['forecast']['expected_return']} exp ret")

        # Test regime check
        is_favorable = adapter.is_regime_favorable(
            context,
            valid_regimes=["TRENDING_UP", "RANGING"],
            invalid_regimes=["VOLATILE"]
        )
        print(f"  Favorable for LONG scalp: {is_favorable}")
