"""
Chronos Price Forecasting Module
================================
Uses Amazon's Chronos foundation model for probabilistic time series forecasting.
Chronos is a pretrained transformer model for zero-shot time series prediction.

Key advantages:
- No training required - works out of the box
- Probabilistic forecasts with confidence intervals
- Handles various time series patterns automatically
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yfinance as yf

logger = logging.getLogger(__name__)

# Lazy load Chronos to avoid import overhead
_pipeline = None
_model_name = "amazon/chronos-t5-small"  # Options: tiny, mini, small, base, large


def get_chronos_pipeline():
    """Lazy load the Chronos pipeline."""
    global _pipeline
    if _pipeline is None:
        try:
            from chronos import ChronosPipeline

            logger.info(f"Loading Chronos model: {_model_name}")
            _pipeline = ChronosPipeline.from_pretrained(
                _model_name,
                device_map="cpu",  # Use CPU for reliability
                torch_dtype=torch.float32,
            )
            logger.info("Chronos model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Chronos: {e}")
            raise
    return _pipeline


class ChronosPredictor:
    """
    Price predictor using Amazon Chronos foundation model.

    Chronos provides:
    - Zero-shot forecasting (no training needed)
    - Probabilistic predictions with quantiles
    - Multiple forecast horizons
    """

    def __init__(self, model_name: str = "amazon/chronos-t5-small"):
        """
        Initialize Chronos predictor.

        Args:
            model_name: Chronos model to use:
                - amazon/chronos-t5-tiny (8M params, fastest)
                - amazon/chronos-t5-mini (20M params)
                - amazon/chronos-t5-small (46M params, balanced)
                - amazon/chronos-t5-base (200M params)
                - amazon/chronos-t5-large (710M params, most accurate)
        """
        global _model_name
        _model_name = model_name
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)

    def _fetch_price_data(self, symbol: str, period: str = "3mo") -> pd.DataFrame:
        """Fetch historical price data from yfinance."""
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period)
        if df.empty:
            raise ValueError(f"No data available for {symbol}")
        return df

    def _prepare_context(
        self, df: pd.DataFrame, use_returns: bool = False
    ) -> torch.Tensor:
        """
        Prepare price series as context for Chronos.

        Args:
            df: DataFrame with OHLCV data
            use_returns: If True, use log returns instead of raw prices
        """
        if use_returns:
            # Use log returns for stationarity
            prices = df["Close"].values
            returns = np.log(prices[1:] / prices[:-1])
            context = torch.tensor(returns, dtype=torch.float32)
        else:
            # Use raw closing prices
            context = torch.tensor(df["Close"].values, dtype=torch.float32)

        return context.unsqueeze(0)  # Add batch dimension

    def predict(
        self, symbol: str, horizon: int = 5, num_samples: int = 20, period: str = "3mo"
    ) -> Dict:
        """
        Generate probabilistic price forecast for a symbol.

        Args:
            symbol: Stock ticker symbol
            horizon: Number of periods to forecast (days)
            num_samples: Number of sample paths for probabilistic forecast
            period: Historical data period to use as context

        Returns:
            Dict with forecast, confidence intervals, and trading signal
        """
        try:
            pipeline = get_chronos_pipeline()

            # Fetch and prepare data
            df = self._fetch_price_data(symbol, period)
            last_price = float(df["Close"].iloc[-1])
            last_date = df.index[-1]

            # Prepare context (use returns for better forecasting)
            context = self._prepare_context(df, use_returns=True)

            # Generate forecast samples
            forecast = pipeline.predict(
                context,
                prediction_length=horizon,
                num_samples=num_samples,
            )

            # Convert log returns back to prices
            forecast_returns = forecast[0].numpy()  # Shape: (num_samples, horizon)

            # Build price paths from returns
            price_paths = np.zeros((num_samples, horizon + 1))
            price_paths[:, 0] = last_price

            for t in range(horizon):
                price_paths[:, t + 1] = price_paths[:, t] * np.exp(
                    forecast_returns[:, t]
                )

            # Remove the starting price
            forecast_prices = price_paths[:, 1:]

            # Calculate statistics
            median_forecast = np.median(forecast_prices, axis=0)
            mean_forecast = np.mean(forecast_prices, axis=0)
            std_forecast = np.std(forecast_prices, axis=0)

            # Quantiles for confidence intervals
            q10 = np.percentile(forecast_prices, 10, axis=0)
            q25 = np.percentile(forecast_prices, 25, axis=0)
            q75 = np.percentile(forecast_prices, 75, axis=0)
            q90 = np.percentile(forecast_prices, 90, axis=0)

            # Calculate expected return and probability of up move
            final_prices = forecast_prices[:, -1]
            expected_return = (np.mean(final_prices) / last_price - 1) * 100
            prob_up = np.mean(final_prices > last_price)
            prob_up_1pct = np.mean(final_prices > last_price * 1.01)
            prob_down_1pct = np.mean(final_prices < last_price * 0.99)

            # Generate trading signal
            if prob_up > 0.7 and expected_return > 1:
                signal = "STRONG_BULLISH"
            elif prob_up > 0.55 and expected_return > 0.5:
                signal = "BULLISH"
            elif prob_up < 0.3 and expected_return < -1:
                signal = "STRONG_BEARISH"
            elif prob_up < 0.45 and expected_return < -0.5:
                signal = "BEARISH"
            else:
                signal = "NEUTRAL"

            # Confidence based on forecast spread
            forecast_spread = (q90[-1] - q10[-1]) / last_price
            confidence = max(
                0, 1 - forecast_spread
            )  # Tighter spread = higher confidence

            return {
                "symbol": symbol,
                "model": self.model_name,
                "signal": signal,
                "last_price": round(last_price, 4),
                "horizon_days": horizon,
                "forecast": {
                    "median": [round(p, 4) for p in median_forecast.tolist()],
                    "mean": [round(p, 4) for p in mean_forecast.tolist()],
                    "std": [round(s, 4) for s in std_forecast.tolist()],
                },
                "confidence_intervals": {
                    "q10": [round(p, 4) for p in q10.tolist()],
                    "q25": [round(p, 4) for p in q25.tolist()],
                    "q75": [round(p, 4) for p in q75.tolist()],
                    "q90": [round(p, 4) for p in q90.tolist()],
                },
                "probabilities": {
                    "prob_up": round(prob_up, 4),
                    "prob_up_1pct": round(prob_up_1pct, 4),
                    "prob_down_1pct": round(prob_down_1pct, 4),
                },
                "expected_return_pct": round(expected_return, 2),
                "confidence": round(confidence, 4),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Chronos prediction failed for {symbol}: {e}")
            return {
                "symbol": symbol,
                "error": str(e),
                "signal": "ERROR",
                "timestamp": datetime.now().isoformat(),
            }

    def predict_intraday(
        self,
        prices: List[float],
        horizon: int = 10,
        num_samples: int = 20,
    ) -> Dict:
        """
        Generate intraday forecast from minute-bar prices.

        Args:
            prices: List of recent prices (e.g., last 60 minute bars)
            horizon: Number of bars to forecast
            num_samples: Number of sample paths

        Returns:
            Dict with forecast and trading signal
        """
        try:
            pipeline = get_chronos_pipeline()

            if len(prices) < 30:
                return {"error": "Need at least 30 price bars for context"}

            last_price = float(prices[-1])

            # Use log returns
            prices_arr = np.array(prices)
            returns = np.log(prices_arr[1:] / prices_arr[:-1])
            context = torch.tensor(returns, dtype=torch.float32).unsqueeze(0)

            # Generate forecast
            forecast = pipeline.predict(
                context,
                prediction_length=horizon,
                num_samples=num_samples,
            )

            forecast_returns = forecast[0].numpy()

            # Build price paths
            price_paths = np.zeros((num_samples, horizon + 1))
            price_paths[:, 0] = last_price

            for t in range(horizon):
                price_paths[:, t + 1] = price_paths[:, t] * np.exp(
                    forecast_returns[:, t]
                )

            forecast_prices = price_paths[:, 1:]

            # Statistics
            median_forecast = np.median(forecast_prices, axis=0)
            final_prices = forecast_prices[:, -1]
            prob_up = np.mean(final_prices > last_price)
            expected_return = (np.mean(final_prices) / last_price - 1) * 100

            # Signal for scalping
            if prob_up > 0.65 and expected_return > 0.3:
                signal = "BUY"
            elif prob_up < 0.35 and expected_return < -0.3:
                signal = "SELL"
            else:
                signal = "HOLD"

            return {
                "signal": signal,
                "prob_up": round(prob_up, 4),
                "expected_return_pct": round(expected_return, 3),
                "forecast_prices": [round(p, 4) for p in median_forecast.tolist()],
                "last_price": round(last_price, 4),
                "horizon_bars": horizon,
            }

        except Exception as e:
            return {"error": str(e), "signal": "HOLD"}


# Singleton instance
_chronos_predictor = None


def get_chronos_predictor() -> ChronosPredictor:
    """Get or create the Chronos predictor singleton."""
    global _chronos_predictor
    if _chronos_predictor is None:
        _chronos_predictor = ChronosPredictor()
    return _chronos_predictor


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("CHRONOS PRICE FORECASTING TEST")
    print("=" * 60)

    predictor = ChronosPredictor()

    # Test on a few symbols
    for symbol in ["SPY", "AAPL", "TSLA"]:
        print(f"\n{symbol}:")
        result = predictor.predict(symbol, horizon=5)

        if "error" not in result:
            print(f"  Signal: {result['signal']}")
            print(f"  Last Price: ${result['last_price']}")
            print(f"  Expected Return: {result['expected_return_pct']}%")
            print(f"  Prob Up: {result['probabilities']['prob_up']:.1%}")
            print(f"  Confidence: {result['confidence']:.1%}")
            print(f"  5-day Forecast: ${result['forecast']['median'][-1]:.2f}")
        else:
            print(f"  Error: {result['error']}")
