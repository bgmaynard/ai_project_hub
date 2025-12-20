"""
PyBroker Schwab Data Source
===========================
Custom data source for PyBroker that fetches minute-bar data from Schwab API.
Enables walkforward analysis with actual intraday data.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class SchwabDataSource:
    """
    Custom data source for PyBroker using Schwab API.

    Provides minute-bar data for backtesting and walkforward analysis.

    Usage:
        from ai.pybroker_schwab_data import SchwabDataSource

        schwab_data = SchwabDataSource()
        df = schwab_data.query(symbols=['TSLA', 'AAPL'], start_date='2024-12-01', end_date='2024-12-18')
    """

    def __init__(self):
        """Initialize Schwab data source"""
        self._schwab = None
        self._cache: Dict[str, pd.DataFrame] = {}

    def _get_schwab(self):
        """Lazy load Schwab market data connection"""
        if self._schwab is None:
            try:
                from schwab_market_data import get_schwab_market_data
                self._schwab = get_schwab_market_data()
            except Exception as e:
                logger.error(f"Failed to initialize Schwab connection: {e}")
        return self._schwab

    def fetch_minute_bars(
        self,
        symbol: str,
        days: int = 10
    ) -> Optional[pd.DataFrame]:
        """
        Fetch minute bars for a symbol from Schwab.

        Args:
            symbol: Stock ticker
            days: Number of days (max 10 for minute data)

        Returns:
            DataFrame with OHLCV data or None
        """
        schwab = self._get_schwab()
        if not schwab:
            logger.warning("Schwab connection not available")
            return None

        symbol = symbol.upper()

        try:
            # Schwab limits minute data to 10 days per request
            days = min(days, 10)

            data = schwab.get_price_history(
                symbol=symbol,
                period_type="day",
                period=days,
                frequency_type="minute",
                frequency=1
            )

            if not data or "candles" not in data:
                logger.warning(f"No minute data returned for {symbol}")
                return None

            candles = data["candles"]
            if not candles:
                return None

            # Convert to DataFrame
            df = pd.DataFrame(candles)

            # Rename columns to standard format
            df = df.rename(columns={
                "datetime": "date",
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "volume"
            })

            # Convert timestamp to datetime
            df["date"] = pd.to_datetime(df["date"], unit="ms")

            # Add symbol column
            df["symbol"] = symbol

            # Set date as index
            df = df.set_index("date")

            # Ensure numeric types
            for col in ["open", "high", "low", "close"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype(int)

            logger.info(f"Fetched {len(df)} minute bars for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Error fetching minute bars for {symbol}: {e}")
            return None

    def fetch_multiple_periods(
        self,
        symbol: str,
        total_days: int = 30
    ) -> Optional[pd.DataFrame]:
        """
        Fetch extended minute data by making multiple API calls.

        Schwab limits minute data to 10 days per request, so this
        makes multiple calls to get up to 6 months of data.

        Args:
            symbol: Stock ticker
            total_days: Total days to fetch (max ~120 trading days)

        Returns:
            Combined DataFrame with all data
        """
        all_data = []
        symbol = symbol.upper()

        # Calculate number of 10-day chunks needed
        chunks = (total_days + 9) // 10

        for i in range(chunks):
            # For now, we can only get the most recent 10 days in one call
            # Schwab API doesn't support arbitrary date ranges for minute data
            if i == 0:
                df = self.fetch_minute_bars(symbol, days=min(10, total_days))
                if df is not None:
                    all_data.append(df)
            # Note: Additional historical data would require different API approach

        if not all_data:
            return None

        combined = pd.concat(all_data)
        combined = combined[~combined.index.duplicated(keep="first")]
        combined = combined.sort_index()

        return combined

    def query(
        self,
        symbols: List[str],
        start_date: str = None,
        end_date: str = None,
        days: int = 10
    ) -> pd.DataFrame:
        """
        Query minute bar data for multiple symbols.

        This is the main interface compatible with PyBroker's data source pattern.

        Args:
            symbols: List of stock tickers
            start_date: Start date (YYYY-MM-DD) - used for filtering
            end_date: End date (YYYY-MM-DD) - used for filtering
            days: Days of data to fetch per symbol

        Returns:
            DataFrame with columns: symbol, date, open, high, low, close, volume
        """
        all_data = []

        for symbol in symbols:
            df = self.fetch_minute_bars(symbol, days)
            if df is not None:
                # Reset index to make date a column
                df_copy = df.reset_index()
                all_data.append(df_copy)

        if not all_data:
            return pd.DataFrame(columns=["symbol", "date", "open", "high", "low", "close", "volume"])

        combined = pd.concat(all_data, ignore_index=True)

        # Filter by date range if provided
        if start_date:
            start = pd.to_datetime(start_date)
            combined = combined[combined["date"] >= start]
        if end_date:
            end = pd.to_datetime(end_date) + timedelta(days=1)  # Include end date
            combined = combined[combined["date"] < end]

        # Sort by symbol and date
        combined = combined.sort_values(["symbol", "date"])

        return combined

    def get_pybroker_dataframe(
        self,
        symbols: List[str],
        days: int = 10
    ) -> pd.DataFrame:
        """
        Get data in format directly usable by PyBroker Strategy.

        Returns DataFrame with MultiIndex (date, symbol) for PyBroker compatibility.

        Args:
            symbols: List of stock tickers
            days: Days of data to fetch

        Returns:
            DataFrame formatted for PyBroker
        """
        df = self.query(symbols, days=days)

        if df.empty:
            return df

        # PyBroker expects date as index, symbol as column
        # Or MultiIndex format
        df = df.set_index(["date", "symbol"])

        return df


def create_pybroker_data(
    symbols: List[str],
    days: int = 10
) -> pd.DataFrame:
    """
    Convenience function to create PyBroker-ready data from Schwab.

    Args:
        symbols: List of stock tickers
        days: Days of minute data to fetch

    Returns:
        DataFrame ready for PyBroker backtesting
    """
    source = SchwabDataSource()
    return source.get_pybroker_dataframe(symbols, days)


# Store data in unified collector for later use
def fetch_and_store(symbols: List[str], days: int = 10) -> Dict:
    """
    Fetch data from Schwab and store in unified collector.

    Args:
        symbols: List of stock tickers
        days: Days to fetch

    Returns:
        Dict with fetch results
    """
    from ai.unified_data_collector import get_data_collector

    source = SchwabDataSource()
    collector = get_data_collector()
    results = {}

    for symbol in symbols:
        df = source.fetch_minute_bars(symbol, days)
        if df is not None:
            # Convert to list of dicts for storage
            bars = df.reset_index().to_dict("records")
            # Convert Timestamps to strings
            for bar in bars:
                if isinstance(bar.get("date"), pd.Timestamp):
                    bar["timestamp"] = bar["date"].isoformat()
                    del bar["date"]

            stored = collector.store_minute_bars(symbol, bars, source="schwab")
            results[symbol] = {
                "fetched": len(bars),
                "stored": stored
            }
        else:
            results[symbol] = {"error": "No data returned"}

    return results


# Test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Testing Schwab Data Source for PyBroker...")

    source = SchwabDataSource()

    # Test single symbol
    df = source.fetch_minute_bars("SPY", days=2)
    if df is not None:
        print(f"\nSPY minute bars: {len(df)} rows")
        print(df.head())
        print(df.tail())

    # Test query interface
    df = source.query(["SPY", "AAPL"], days=2)
    print(f"\nMulti-symbol query: {len(df)} rows")
    print(f"Symbols: {df['symbol'].unique()}")

    # Test PyBroker format
    df = source.get_pybroker_dataframe(["SPY"], days=2)
    print(f"\nPyBroker format: {len(df)} rows")
    print(df.head())
