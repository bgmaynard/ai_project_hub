"""
IBKR Historical Data Fetcher
Downloads historical bars from Interactive Brokers for LSTM training
"""

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
import pandas as pd
import time
import threading
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IBKRDataFetcher(EWrapper, EClient):
    """
    Fetch historical data from IBKR for training
    """
    
    def __init__(self):
        EClient.__init__(self, self)
        self.historical_data = []
        self.data_received = threading.Event()
        self.error_occurred = False
        self.request_id = 1
        
    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        """Handle errors"""
        logger.error(f"Error {errorCode}: {errorString}")
        if errorCode in [162, 200, 354]:  # Historical data errors
            self.error_occurred = True
            self.data_received.set()
    
    def historicalData(self, reqId, bar):
        """Receive historical bar data"""
        self.historical_data.append({
            'datetime': bar.date,
            'open': bar.open,
            'high': bar.high,
            'low': bar.low,
            'close': bar.close,
            'volume': bar.volume
        })
    
    def historicalDataEnd(self, reqId, start, end):
        """Historical data complete"""
        logger.info(f"Historical data complete: {len(self.historical_data)} bars")
        self.data_received.set()
    
    def connect_to_ibkr(self, host='127.0.0.1', port=7497, client_id=1):
        """
        Connect to IBKR TWS or Gateway
        
        Args:
            host: TWS host
            port: 7497 (TWS Paper), 7496 (TWS Live), 4002 (Gateway Paper)
            client_id: Unique client ID
        """
        logger.info(f"Connecting to IBKR at {host}:{port}...")
        self.connect(host, port, client_id)
        
        # Start message processing thread
        thread = threading.Thread(target=self.run, daemon=True)
        thread.start()
        
        # Wait for connection
        time.sleep(2)
        
        if self.isConnected():
            logger.info("✓ Connected to IBKR")
            return True
        else:
            logger.error("✗ Failed to connect to IBKR")
            return False
    
    def create_stock_contract(self, symbol, exchange='SMART', currency='USD'):
        """Create stock contract"""
        contract = Contract()
        contract.symbol = symbol
        contract.secType = 'STK'
        contract.exchange = exchange
        contract.currency = currency
        return contract
    
    def fetch_historical_data(
        self,
        symbol,
        duration='2 Y',
        bar_size='1 hour',
        what_to_show='TRADES',
        use_rth=True
    ):
        """
        Fetch historical data from IBKR
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            duration: How far back (e.g., '2 Y', '6 M', '30 D')
            bar_size: Bar size (e.g., '1 hour', '5 mins', '1 day')
            what_to_show: Data type ('TRADES', 'MIDPOINT', 'BID', 'ASK')
            use_rth: Use regular trading hours only
            
        Returns:
            DataFrame with OHLCV data
        """
        logger.info(f"Fetching {duration} of {bar_size} data for {symbol}...")
        
        # Reset data
        self.historical_data = []
        self.data_received.clear()
        self.error_occurred = False
        
        # Create contract
        contract = self.create_stock_contract(symbol)
        
        # Request historical data
        end_datetime = datetime.now().strftime('%Y%m%d %H:%M:%S')
        
        self.reqHistoricalData(
            reqId=self.request_id,
            contract=contract,
            endDateTime=end_datetime,
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow=what_to_show,
            useRTH=1 if use_rth else 0,
            formatDate=1,
            keepUpToDate=False,
            chartOptions=[]
        )
        
        self.request_id += 1
        
        # Wait for data (max 30 seconds)
        self.data_received.wait(timeout=30)
        
        if self.error_occurred:
            logger.error(f"Error fetching data for {symbol}")
            return None
        
        if not self.historical_data:
            logger.warning(f"No data received for {symbol}")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(self.historical_data)
        
        # Parse datetime
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        
        logger.info(f"✓ Fetched {len(df)} bars for {symbol}")
        logger.info(f"  Date range: {df.index[0]} to {df.index[-1]}")
        
        return df
    
    def fetch_multiple_symbols(
        self,
        symbols,
        duration='2 Y',
        bar_size='1 hour',
        save_to_csv=True,
        output_dir='data/historical'
    ):
        """
        Fetch data for multiple symbols
        
        Args:
            symbols: List of symbols
            duration: How far back
            bar_size: Bar size
            save_to_csv: Save to CSV files
            output_dir: Output directory
            
        Returns:
            Dictionary of symbol -> DataFrame
        """
        import os
        
        if save_to_csv:
            os.makedirs(output_dir, exist_ok=True)
        
        results = {}
        
        for symbol in symbols:
            logger.info(f"\n{'='*60}")
            logger.info(f"Fetching {symbol}")
            logger.info(f"{'='*60}")
            
            df = self.fetch_historical_data(symbol, duration, bar_size)
            
            if df is not None:
                results[symbol] = df
                
                if save_to_csv:
                    filename = f"{output_dir}/{symbol}_{bar_size.replace(' ', '_')}.csv"
                    df.to_csv(filename)
                    logger.info(f"✓ Saved to {filename}")
            
            # Wait between requests to avoid pacing violations
            time.sleep(2)
        
        return results
    
    def disconnect_from_ibkr(self):
        """Disconnect from IBKR"""
        self.disconnect()
        logger.info("Disconnected from IBKR")


def fetch_data_for_training(
    symbols=['AAPL', 'TSLA'],
    duration='2 Y',
    bar_size='1 hour',
    host='127.0.0.1',
    port=7497
):
    """
    Convenience function to fetch training data
    
    Args:
        symbols: List of symbols
        duration: How far back (e.g., '2 Y', '6 M')
        bar_size: Bar size (e.g., '1 hour', '5 mins', '1 day')
        host: TWS host
        port: TWS port (7497=Paper, 7496=Live)
    """
    print("\n" + "="*70)
    print("IBKR HISTORICAL DATA FETCHER")
    print("="*70)
    print(f"\nSymbols: {', '.join(symbols)}")
    print(f"Duration: {duration}")
    print(f"Bar Size: {bar_size}")
    print(f"Connection: {host}:{port}")
    print("="*70)
    
    # Create fetcher
    fetcher = IBKRDataFetcher()
    
    # Connect to IBKR
    if not fetcher.connect_to_ibkr(host, port):
        print("\n✗ Failed to connect to IBKR")
        print("\nMake sure:")
        print("  1. TWS or Gateway is running")
        print("  2. API connections are enabled (Settings → API)")
        print("  3. Port is correct (7497=Paper, 7496=Live)")
        return None
    
    # Fetch data
    results = fetcher.fetch_multiple_symbols(
        symbols=symbols,
        duration=duration,
        bar_size=bar_size,
        save_to_csv=True,
        output_dir='data/historical'
    )
    
    # Disconnect
    fetcher.disconnect_from_ibkr()
    
    # Summary
    print("\n" + "="*70)
    print("FETCH COMPLETE")
    print("="*70)
    
    for symbol, df in results.items():
        print(f"\n{symbol}:")
        print(f"  Bars: {len(df)}")
        print(f"  Date Range: {df.index[0].date()} to {df.index[-1].date()}")
        print(f"  File: data/historical/{symbol}_{bar_size.replace(' ', '_')}.csv")
    
    print("\n✓ Data ready for training!")
    print("\nNext step:")
    print("  python train_with_ibkr_data.py")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Fetch historical data from IBKR')
    parser.add_argument('--symbols', nargs='+', default=['AAPL', 'TSLA'],
                       help='Symbols to fetch')
    parser.add_argument('--duration', default='2 Y',
                       help='Duration (e.g., "2 Y", "6 M", "30 D")')
    parser.add_argument('--bar-size', default='1 hour',
                       help='Bar size (e.g., "1 hour", "5 mins", "1 day")')
    parser.add_argument('--port', type=int, default=7497,
                       help='TWS port (7497=Paper, 7496=Live)')
    
    args = parser.parse_args()
    
    # Fetch data
    fetch_data_for_training(
        symbols=args.symbols,
        duration=args.duration,
        bar_size=args.bar_size,
        port=args.port
    )
