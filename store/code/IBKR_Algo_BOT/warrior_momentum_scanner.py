"""
Warrior Trading Momentum Scanner
=================================

Pre-market gap-and-go momentum scanner based on Ross Cameron's methodology.

Key Features:
- Pre-market gap detection (7:00 AM focus)
- High relative volume scanning
- Small-cap momentum plays
- Breakout detection
- Ross Cameron style entry/exit

Target: 70%+ win rate with tight risk control

Usage:
    python warrior_momentum_scanner.py --start-time 07:00
"""

import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.scanner import ScannerSubscription
import threading
import time as time_module
import logging
from collections import defaultdict
import yfinance as yf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WarriorScanner(EWrapper, EClient):
    """
    Warrior Trading style momentum scanner for IBKR
    
    Scans for:
    - Pre-market gappers (2%+ gap)
    - High relative volume (3x+ average)
    - Small caps ($50M - $2B market cap)
    - Clean breakout patterns
    """
    
    def __init__(self):
        EClient.__init__(self, self)
        
        self.connected = False
        self.next_req_id = 1
        
        # Scanner data
        self.gap_candidates = {}  # {symbol: gap_data}
        self.market_data = {}  # {symbol: real-time data}
        self.scanner_complete = False
        
        # Warrior parameters
        self.min_gap_pct = 2.0  # Minimum 2% gap
        self.min_rvol = 3.0  # 3x relative volume
        self.min_price = 1.0  # Min $1
        self.max_price = 20.0  # Max $20 (small caps)
        self.max_spread_pct = 2.0  # Max 2% spread
        
    def nextValidId(self, orderId: int):
        """Connection established"""
        super().nextValidId(orderId)
        self.next_req_id = orderId
        self.connected = True
        logger.info("✓ Connected to IBKR")
    
    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        """Error handler"""
        if errorCode in [2104, 2106, 2158]:
            pass  # Info messages
        elif errorCode >= 2000:
            logger.info(f"Info: {errorString}")
        else:
            logger.error(f"Error {errorCode}: {errorString}")
    
    def scannerData(self, reqId, rank, contractDetails, distance, benchmark, projection, legsStr):
        """Scanner results"""
        symbol = contractDetails.contract.symbol
        
        if symbol not in self.gap_candidates:
            self.gap_candidates[symbol] = {
                'rank': rank,
                'contract': contractDetails.contract,
                'symbol': symbol,
                'exchange': contractDetails.contract.exchange
            }
            
            logger.info(f"Found: {symbol} (rank {rank})")
    
    def scannerDataEnd(self, reqId):
        """Scanner complete"""
        self.scanner_complete = True
        logger.info(f"✓ Scanner found {len(self.gap_candidates)} candidates")
    
    def tickPrice(self, reqId, tickType, price, attrib):
        """Market data updates"""
        if reqId not in self.market_data:
            self.market_data[reqId] = {}
        
        tick_map = {1: 'bid', 2: 'ask', 4: 'last', 6: 'high', 7: 'low', 9: 'close'}
        if tickType in tick_map:
            self.market_data[reqId][tick_map[tickType]] = price
    
    def tickSize(self, reqId, tickType, size):
        """Size updates"""
        if reqId not in self.market_data:
            self.market_data[reqId] = {}
        
        size_map = {0: 'bid_size', 3: 'ask_size', 5: 'last_size', 8: 'volume'}
        if tickType in size_map:
            self.market_data[reqId][size_map[tickType]] = size


class WarriorMomentumTrader:
    """
    Complete Warrior Trading momentum system
    
    Implements Ross Cameron's gap-and-go methodology:
    1. Scan for pre-market gappers
    2. Calculate Warrior-style features
    3. Identify high-probability setups
    4. Execute with tight risk control
    """
    
    def __init__(self):
        self.scanner = WarriorScanner()
        
        # Trading parameters (Warrior style)
        self.min_gap_pct = 2.0  # 2%+ gap up
        self.min_rvol = 3.0  # 3x relative volume
        self.min_price = 1.0  # $1 minimum
        self.max_price = 20.0  # $20 maximum (small caps)
        self.max_position_size = 5000  # $5k per position
        self.target_win_rate = 0.70  # Target 70%+ win rate
        
        # Risk management
        self.risk_per_trade = 0.02  # 2% risk
        self.profit_target_r = 2.0  # 2R profit target
        self.max_hold_minutes = 45  # Max 45 min hold
        
        # Active positions
        self.positions = {}
        self.daily_pnl = 0
        
    def connect(self, host='127.0.0.1', port=7497):
        """Connect to IBKR"""
        logger.info(f"Connecting to IBKR at {host}:{port}...")
        
        self.scanner.connect(host, port, clientId=3)
        
        api_thread = threading.Thread(target=self.scanner.run, daemon=True)
        api_thread.start()
        
        time_module.sleep(2)
        
        if not self.scanner.connected:
            raise ConnectionError("Failed to connect to IBKR")
        
        logger.info("✓ Connected")
    
    def scan_for_gappers(self):
        """
        Scan for pre-market gappers
        
        IBKR Scanner filters:
        - Price: $1 - $20
        - Volume: High relative volume
        - Market cap: Small to mid cap
        - Gap: Pre-market movers
        """
        logger.info("\n" + "="*70)
        logger.info("SCANNING FOR PRE-MARKET GAPPERS")
        logger.info("="*70 + "\n")
        
        # Create scanner subscription
        scanner_sub = ScannerSubscription()
        scanner_sub.instrument = "STK"
        scanner_sub.locationCode = "STK.US.MAJOR"
        scanner_sub.scanCode = "TOP_PERC_GAIN"  # Top % gainers
        
        # Filters
        scanner_sub.abovePrice = self.min_price
        scanner_sub.belowPrice = self.max_price
        scanner_sub.aboveVolume = 100000  # Min 100k shares
        scanner_sub.marketCapAbove = 50000000  # $50M+ market cap
        scanner_sub.marketCapBelow = 2000000000  # $2B max (small caps)
        
        # Request scan
        req_id = self.scanner.next_req_id
        self.scanner.next_req_id += 1
        
        self.scanner.reqScannerSubscription(req_id, scanner_sub, [], [])
        
        # Wait for results
        logger.info("Scanning... (this takes 5-10 seconds)")
        timeout = 15
        start = time_module.time()
        
        while not self.scanner.scanner_complete:
            if time_module.time() - start > timeout:
                logger.error("Scanner timeout")
                break
            time_module.sleep(0.5)
        
        # Cancel scanner
        self.scanner.cancelScannerSubscription(req_id)
        
        return self.scanner.gap_candidates
    
    def calculate_warrior_features(self, symbol, market_data, prev_close):
        """
        Calculate Warrior Trading features
        
        Features:
        - Gap percentage
        - Relative volume
        - Price vs pre-market high
        - Momentum indicators
        - Breakout signals
        """
        
        current_price = market_data.get('last', market_data.get('bid', 0))
        if current_price == 0:
            return None
        
        # Gap calculation
        gap_pct = ((current_price - prev_close) / prev_close) * 100
        
        # Volume (would need historical average for true RVOL)
        volume = market_data.get('volume', 0)
        
        # Spread
        bid = market_data.get('bid', 0)
        ask = market_data.get('ask', 0)
        if bid > 0 and ask > 0:
            spread_pct = ((ask - bid) / current_price) * 100
        else:
            spread_pct = 999  # Invalid
        
        # Get intraday high/low
        high = market_data.get('high', current_price)
        low = market_data.get('low', current_price)
        
        # Price position in range
        if high > low:
            price_position = (current_price - low) / (high - low)
        else:
            price_position = 0.5
        
        features = {
            'symbol': symbol,
            'current_price': current_price,
            'prev_close': prev_close,
            'gap_pct': gap_pct,
            'gap_dollars': current_price - prev_close,
            'is_gap_up': gap_pct > 0,
            'volume': volume,
            'spread_pct': spread_pct,
            'high': high,
            'low': low,
            'price_position': price_position,  # 0 = at low, 1 = at high
            'above_premarket_high': current_price > high * 0.99,
            'bid': bid,
            'ask': ask,
            'timestamp': datetime.now()
        }
        
        return features
    
    def score_warrior_setup(self, features):
        """
        Score setup using Warrior Trading criteria
        
        Returns:
            score (0-100): Higher = better setup
            reasons: List of why this is good/bad
        """
        
        score = 0
        reasons = []
        
        # Gap percentage (0-30 points)
        gap = features['gap_pct']
        if gap >= 5:
            score += 30
            reasons.append(f"✅ Strong gap: {gap:.1f}%")
        elif gap >= 3:
            score += 20
            reasons.append(f"✓ Good gap: {gap:.1f}%")
        elif gap >= 2:
            score += 10
            reasons.append(f"↗ Decent gap: {gap:.1f}%")
        else:
            reasons.append(f"⚠️ Weak gap: {gap:.1f}%")
        
        # Price position (0-20 points)
        pos = features['price_position']
        if pos > 0.9:
            score += 20
            reasons.append("✅ Near high of day")
        elif pos > 0.7:
            score += 15
            reasons.append("✓ Upper range")
        elif pos > 0.5:
            score += 10
            reasons.append("↗ Above midpoint")
        else:
            reasons.append("⚠️ Lower range")
        
        # Spread (0-20 points)
        spread = features['spread_pct']
        if spread < 0.5:
            score += 20
            reasons.append(f"✅ Tight spread: {spread:.2f}%")
        elif spread < 1.0:
            score += 15
            reasons.append(f"✓ Good spread: {spread:.2f}%")
        elif spread < 2.0:
            score += 5
            reasons.append(f"↗ Acceptable spread: {spread:.2f}%")
        else:
            reasons.append(f"❌ Wide spread: {spread:.2f}%")
        
        # Price level (0-15 points)
        price = features['current_price']
        if 2 <= price <= 10:
            score += 15
            reasons.append(f"✅ Ideal price range: ${price:.2f}")
        elif 1 <= price <= 20:
            score += 10
            reasons.append(f"✓ Good price: ${price:.2f}")
        else:
            reasons.append(f"⚠️ Price outside ideal range: ${price:.2f}")
        
        # Breakout (0-15 points)
        if features['above_premarket_high']:
            score += 15
            reasons.append("✅ Above pre-market high (breakout!)")
        
        return score, reasons
    
    def calculate_position_size(self, price, stop_distance):
        """
        Calculate position size using Warrior-style risk management
        
        Risk 2% of account on each trade
        Position size = (Account * Risk%) / Stop Distance
        """
        
        account_size = 100000  # Would get from IBKR account data
        risk_amount = account_size * self.risk_per_trade
        
        # Calculate shares
        shares = int(risk_amount / stop_distance)
        
        # Cap at max position value
        max_shares = int(self.max_position_size / price)
        shares = min(shares, max_shares)
        
        # Minimum 10 shares
        shares = max(10, shares)
        
        return shares
    
    def generate_entry_signal(self, features, score, reasons):
        """
        Generate entry signal with stop and target
        
        Returns:
            signal dict or None
        """
        
        # Minimum score threshold
        if score < 60:
            logger.info(f"{features['symbol']}: Score too low ({score}/100)")
            return None
        
        # Price validation
        price = features['current_price']
        if price < self.min_price or price > self.max_price:
            return None
        
        # Spread validation
        if features['spread_pct'] > self.max_spread_pct:
            logger.info(f"{features['symbol']}: Spread too wide")
            return None
        
        # Calculate stop loss (Warrior style: below pre-market low or recent support)
        low = features['low']
        stop_price = low * 0.98  # 2% below low for safety
        stop_distance = price - stop_price
        
        # Stop distance validation (shouldn't be > 5%)
        stop_distance_pct = (stop_distance / price) * 100
        if stop_distance_pct > 5:
            logger.info(f"{features['symbol']}: Stop too far ({stop_distance_pct:.1f}%)")
            return None
        
        # Calculate target (2R)
        target_price = price + (stop_distance * self.profit_target_r)
        
        # Calculate position size
        shares = self.calculate_position_size(price, stop_distance)
        
        signal = {
            'symbol': features['symbol'],
            'action': 'BUY',
            'entry_price': price,
            'stop_price': stop_price,
            'target_price': target_price,
            'shares': shares,
            'score': score,
            'reasons': reasons,
            'risk_dollars': stop_distance * shares,
            'reward_dollars': (target_price - price) * shares,
            'r_multiple': self.profit_target_r,
            'timestamp': datetime.now()
        }
        
        return signal
    
    def display_watchlist(self, candidates):
        """Display formatted watchlist"""
        
        print("\n" + "="*80)
        print("WARRIOR TRADING WATCHLIST")
        print("="*80)
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Candidates found: {len(candidates)}")
        print("="*80 + "\n")
        
        if not candidates:
            print("No candidates found matching criteria")
            return
        
        # Header
        print(f"{'Rank':<6} {'Symbol':<8} {'Price':<10} {'Gap%':<10} {'Score':<8} {'Status':<15}")
        print("-" * 80)
        
        for i, (symbol, data) in enumerate(sorted(candidates.items(), 
                                                   key=lambda x: x[1].get('score', 0), 
                                                   reverse=True), 1):
            price = data.get('current_price', 0)
            gap = data.get('gap_pct', 0)
            score = data.get('score', 0)
            
            if score >= 80:
                status = "🔥 HOT"
            elif score >= 70:
                status = "✅ STRONG"
            elif score >= 60:
                status = "✓ GOOD"
            else:
                status = "↗ WATCH"
            
            print(f"{i:<6} {symbol:<8} ${price:<9.2f} {gap:+9.1f}% {score:<8} {status:<15}")
        
        print("\n" + "="*80 + "\n")
    
    def run_warrior_scan(self):
        """
        Complete Warrior Trading scan workflow
        
        1. Scan for gappers
        2. Get real-time data
        3. Calculate features
        4. Score setups
        5. Generate signals
        """
        
        print("\n" + "🔥"*40)
        print("WARRIOR TRADING MOMENTUM SCANNER")
        print("🔥"*40 + "\n")
        
        print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
        print(f"Target: Pre-market gappers (2%+ gap, high volume)")
        print(f"Style: Ross Cameron gap-and-go\n")
        
        # Step 1: Scan
        candidates = self.scan_for_gappers()
        
        if not candidates:
            logger.info("No candidates found")
            return
        
        logger.info(f"\nAnalyzing {len(candidates)} candidates...")
        
        # Step 2: Analyze each candidate
        analyzed = {}
        
        for symbol, data in candidates.items():
            # Get previous close (would use IBKR historical data in production)
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period='5d')
                if len(hist) > 0:
                    prev_close = hist['Close'].iloc[-1]
                else:
                    continue
            except:
                continue
            
            # Subscribe to real-time data
            req_id = self.scanner.next_req_id
            self.scanner.next_req_id += 1
            
            contract = data['contract']
            self.scanner.reqMktData(req_id, contract, '', False, False, [])
            
            time_module.sleep(0.5)
            
            # Get market data
            if req_id in self.scanner.market_data:
                market_data = self.scanner.market_data[req_id]
                
                # Calculate features
                features = self.calculate_warrior_features(symbol, market_data, prev_close)
                
                if features:
                    # Score setup
                    score, reasons = self.score_warrior_setup(features)
                    
                    features['score'] = score
                    features['reasons'] = reasons
                    
                    analyzed[symbol] = features
                    
                    # Generate signal if score high enough
                    signal = self.generate_entry_signal(features, score, reasons)
                    if signal:
                        features['signal'] = signal
                        
                        logger.info(f"\n{'='*70}")
                        logger.info(f"TRADE SIGNAL: {symbol}")
                        logger.info(f"{'='*70}")
                        logger.info(f"Score: {score}/100")
                        logger.info(f"Entry: ${signal['entry_price']:.2f}")
                        logger.info(f"Stop: ${signal['stop_price']:.2f}")
                        logger.info(f"Target: ${signal['target_price']:.2f}")
                        logger.info(f"Shares: {signal['shares']}")
                        logger.info(f"Risk: ${signal['risk_dollars']:.2f}")
                        logger.info(f"Reward: ${signal['reward_dollars']:.2f}")
                        logger.info("\nReasons:")
                        for reason in reasons:
                            logger.info(f"  {reason}")
                        logger.info(f"{'='*70}\n")
            
            # Cancel market data
            self.scanner.cancelMktData(req_id)
        
        # Step 3: Display watchlist
        self.display_watchlist(analyzed)
        
        return analyzed


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Warrior Trading Momentum Scanner')
    parser.add_argument('--start-time', type=str, default='07:00',
                       help='Start scanning at this time (HH:MM)')
    parser.add_argument('--port', type=int, default=7497,
                       help='IBKR port (7497=paper, 7496=live)')
    parser.add_argument('--continuous', action='store_true',
                       help='Continuous scanning mode')
    parser.add_argument('--interval', type=int, default=300,
                       help='Scan interval in seconds (continuous mode)')
    
    args = parser.parse_args()
    
    # Create trader
    trader = WarriorMomentumTrader()
    
    # Connect
    trader.connect(port=args.port)
    
    if args.continuous:
        # Continuous mode
        print(f"\n🔄 Continuous scanning mode")
        print(f"   Interval: {args.interval} seconds\n")
        
        try:
            iteration = 0
            while True:
                iteration += 1
                print(f"\n--- Scan #{iteration} @ {datetime.now().strftime('%H:%M:%S')} ---\n")
                
                trader.run_warrior_scan()
                
                print(f"\nNext scan in {args.interval} seconds...")
                time_module.sleep(args.interval)
                
        except KeyboardInterrupt:
            print("\n\nStopped by user")
    
    else:
        # Single scan
        trader.run_warrior_scan()
    
    # Disconnect
    trader.scanner.disconnect()


if __name__ == "__main__":
    main()
