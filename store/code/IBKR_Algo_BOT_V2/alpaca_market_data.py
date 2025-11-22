"""
Alpaca Market Data Provider
Provides historical and real-time market data for AI predictions
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Dict
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import (
    StockBarsRequest,
    StockLatestQuoteRequest,
    StockLatestBarRequest,
    StockSnapshotRequest
)
from alpaca.data.timeframe import TimeFrame
from config.broker_config import get_broker_config
import logging

logger = logging.getLogger(__name__)


class AlpacaMarketData:
    """Alpaca market data provider"""

    def __init__(self):
        """Initialize Alpaca market data client"""
        config = get_broker_config()

        if not config.is_alpaca():
            raise ValueError("Broker configuration is not set to Alpaca")

        self.data_client = StockHistoricalDataClient(
            config.alpaca.api_key,
            config.alpaca.secret_key
        )

    def get_historical_bars(
        self,
        symbol: str,
        timeframe: str = "1Day",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Get historical OHLCV bars

        Args:
            symbol: Stock symbol
            timeframe: Bar timeframe (1Min, 5Min, 15Min, 1Hour, 1Day)
            start: Start date (defaults to 2 years ago)
            end: End date (defaults to now)
            limit: Max number of bars to return

        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Parse timeframe
            tf_map = {
                "1Min": TimeFrame.Minute,
                "5Min": TimeFrame(5, TimeFrame.Unit.Minute),
                "15Min": TimeFrame(15, TimeFrame.Unit.Minute),
                "1Hour": TimeFrame.Hour,
                "1Day": TimeFrame.Day,
                "1D": TimeFrame.Day,
                "5m": TimeFrame(5, TimeFrame.Unit.Minute),
                "15m": TimeFrame(15, TimeFrame.Unit.Minute),
                "1h": TimeFrame.Hour,
                "1d": TimeFrame.Day
            }

            alpaca_timeframe = tf_map.get(timeframe, TimeFrame.Day)

            # Default time range: 2 years
            if start is None:
                start = datetime.now() - timedelta(days=730)
            if end is None:
                end = datetime.now()

            # Create request
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=alpaca_timeframe,
                start=start,
                end=end,
                limit=limit
            )

            # Fetch data
            bars = self.data_client.get_stock_bars(request)

            # Convert to DataFrame
            if symbol in bars:
                df = bars[symbol].df

                # Rename columns to match yfinance format
                df = df.rename(columns={
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low',
                    'close': 'Close',
                    'volume': 'Volume',
                    'vwap': 'VWAP'
                })

                # Reset index to make timestamp a column
                df = df.reset_index()
                df = df.rename(columns={'timestamp': 'Date'})
                df = df.set_index('Date')

                logger.info(f"Fetched {len(df)} bars for {symbol}")
                return df
            else:
                logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return pd.DataFrame()

    def get_latest_quote(self, symbol: str) -> Optional[Dict]:
        """
        Get latest quote for a symbol

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary with quote data or None
        """
        try:
            request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            quotes = self.data_client.get_stock_latest_quote(request)

            if symbol in quotes:
                quote = quotes[symbol]
                return {
                    "symbol": symbol,
                    "bid": float(quote.bid_price),
                    "ask": float(quote.ask_price),
                    "bid_size": int(quote.bid_size),
                    "ask_size": int(quote.ask_size),
                    "last": (float(quote.bid_price) + float(quote.ask_price)) / 2,
                    "timestamp": quote.timestamp
                }
            else:
                return None

        except Exception as e:
            logger.error(f"Error fetching quote for {symbol}: {e}")
            return None

    def get_latest_bar(self, symbol: str) -> Optional[Dict]:
        """
        Get latest bar for a symbol

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary with bar data or None
        """
        try:
            request = StockLatestBarRequest(symbol_or_symbols=symbol)
            bars = self.data_client.get_stock_latest_bar(request)

            if symbol in bars:
                bar = bars[symbol]
                return {
                    "symbol": symbol,
                    "open": float(bar.open),
                    "high": float(bar.high),
                    "low": float(bar.low),
                    "close": float(bar.close),
                    "volume": int(bar.volume),
                    "vwap": float(bar.vwap) if bar.vwap else None,
                    "timestamp": bar.timestamp
                }
            else:
                return None

        except Exception as e:
            logger.error(f"Error fetching latest bar for {symbol}: {e}")
            return None

    def get_snapshot(self, symbol: str) -> Optional[Dict]:
        """
        Get market snapshot for a symbol (combines quote and bar data)

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary with snapshot data or None
        """
        try:
            request = StockSnapshotRequest(symbol_or_symbols=symbol)
            snapshots = self.data_client.get_stock_snapshot(request)

            if symbol in snapshots:
                snapshot = snapshots[symbol]

                # Combine latest quote and bar
                result = {
                    "symbol": symbol,
                    "timestamp": datetime.now().isoformat()
                }

                # Add quote data
                if snapshot.latest_quote:
                    result.update({
                        "bid": float(snapshot.latest_quote.bid_price),
                        "ask": float(snapshot.latest_quote.ask_price),
                        "bid_size": int(snapshot.latest_quote.bid_size),
                        "ask_size": int(snapshot.latest_quote.ask_size)
                    })

                # Add bar data
                if snapshot.latest_trade:
                    result["last_price"] = float(snapshot.latest_trade.price)
                    result["last_size"] = int(snapshot.latest_trade.size)

                # Add daily bar
                if snapshot.daily_bar:
                    result.update({
                        "open": float(snapshot.daily_bar.open),
                        "high": float(snapshot.daily_bar.high),
                        "low": float(snapshot.daily_bar.low),
                        "close": float(snapshot.daily_bar.close),
                        "volume": int(snapshot.daily_bar.volume)
                    })

                return result
            else:
                return None

        except Exception as e:
            logger.error(f"Error fetching snapshot for {symbol}: {e}")
            return None

    def get_multiple_quotes(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        Get quotes for multiple symbols

        Args:
            symbols: List of stock symbols

        Returns:
            Dictionary mapping symbols to quote data
        """
        results = {}

        try:
            request = StockLatestQuoteRequest(symbol_or_symbols=symbols)
            quotes = self.data_client.get_stock_latest_quote(request)

            for symbol, quote in quotes.items():
                results[symbol] = {
                    "symbol": symbol,
                    "bid": float(quote.bid_price),
                    "ask": float(quote.ask_price),
                    "bid_size": int(quote.bid_size),
                    "ask_size": int(quote.ask_size),
                    "last": (float(quote.bid_price) + float(quote.ask_price)) / 2,
                    "timestamp": quote.timestamp
                }

        except Exception as e:
            logger.error(f"Error fetching multiple quotes: {e}")

        return results


# Global instance
_market_data_instance: Optional[AlpacaMarketData] = None


def get_alpaca_market_data() -> AlpacaMarketData:
    """
    Get or create the global Alpaca market data instance

    Returns:
        AlpacaMarketData instance
    """
    global _market_data_instance

    if _market_data_instance is None:
        _market_data_instance = AlpacaMarketData()

    return _market_data_instance
