"""
News Triggered Trader
=====================
Automatically trades on breaking news from multiple sources.

SPEED IS CRITICAL:
- News detected -> Order placed in <500ms
- Uses both Alpaca WebSocket + Benzinga RSS
- Scalps the first pop, exits fast

RISK MANAGEMENT:
- Max position size per trade
- Stop loss on every entry
- Time-based exits (don't hold news plays)
"""

import requests
import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import os
import sys

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ai.benzinga_fast_news import get_fast_news, start_fast_news, BenzingaFastNews, BreakingNews

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class NewsTriggeredTrader:
    """
    Trades automatically based on breaking news signals.
    """

    def __init__(self):
        self.api_url = "http://localhost:9100/api/alpaca"

        # Position management
        self.max_position_value = 500      # Max $ per news trade
        self.stop_loss_pct = 0.02          # 2% stop loss
        self.take_profit_pct = 0.05        # 5% take profit
        self.max_hold_minutes = 5          # Exit after 5 mins regardless

        # Track active news positions
        self.news_positions: Dict[str, Dict] = {}
        self.trade_history: List[Dict] = []

        # News scanner
        self.news_scanner: Optional[BenzingaFastNews] = None

        # Control
        self.is_running = False
        self._monitor_thread: Optional[threading.Thread] = None

        # Stats
        self.trades_executed = 0
        self.wins = 0
        self.losses = 0
        self.total_pnl = 0.0

        logger.info("NewsTriggeredTrader initialized")

    def start(self, watchlist: List[str] = None):
        """Start the news trader"""
        if self.is_running:
            logger.warning("Already running")
            return

        self.is_running = True

        # Start news scanner with our callbacks
        self.news_scanner = start_fast_news(
            watchlist=watchlist,
            on_buy=self._handle_buy_signal,
            on_sell=self._handle_sell_signal
        )

        # Start position monitor
        self._monitor_thread = threading.Thread(target=self._monitor_positions, daemon=True)
        self._monitor_thread.start()

        logger.info("=" * 60)
        logger.info("NEWS TRIGGERED TRADER STARTED")
        logger.info(f"Max position: ${self.max_position_value}")
        logger.info(f"Stop loss: {self.stop_loss_pct:.0%} | Take profit: {self.take_profit_pct:.0%}")
        logger.info(f"Max hold time: {self.max_hold_minutes} minutes")
        logger.info("=" * 60)

    def stop(self):
        """Stop the trader"""
        self.is_running = False

        if self.news_scanner:
            self.news_scanner.stop()

        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)

        # Close any open news positions
        self._close_all_positions("Trader stopped")

        logger.info("NewsTriggeredTrader stopped")
        self._print_stats()

    def _handle_buy_signal(self, alert: BreakingNews):
        """Handle a buy signal from news"""
        logger.info(f"\n>>> BUY SIGNAL: {alert.symbols}")
        logger.info(f"    {alert.headline[:80]}...")
        logger.info(f"    Catalyst: {alert.catalyst_type} | Confidence: {alert.confidence:.0%}")

        for symbol in alert.symbols:
            if symbol in self.news_positions:
                logger.info(f"    Skip {symbol} - already in position")
                continue

            # Execute buy
            self._execute_buy(symbol, alert)

    def _handle_sell_signal(self, alert: BreakingNews):
        """Handle a sell signal from news"""
        logger.info(f"\n<<< SELL SIGNAL: {alert.symbols}")
        logger.info(f"    {alert.headline[:80]}...")

        for symbol in alert.symbols:
            if symbol in self.news_positions:
                # Close existing position
                self._close_position(symbol, "Bearish news")
            else:
                # Could short here if enabled
                logger.info(f"    No position in {symbol} to close")

    def _execute_buy(self, symbol: str, alert: BreakingNews):
        """Execute a buy order"""
        try:
            # Get current price
            quote = self._get_quote(symbol)
            if not quote:
                logger.error(f"    Could not get quote for {symbol}")
                return

            price = quote.get('ask', quote.get('last', 0))
            if price <= 0:
                logger.error(f"    Invalid price for {symbol}")
                return

            # Calculate quantity
            qty = max(1, int(self.max_position_value / price))

            # Place order
            order = {
                'symbol': symbol,
                'quantity': qty,
                'action': 'buy',
                'order_type': 'market',
                'time_in_force': 'day',
                'extended_hours': True
            }

            logger.info(f"    Placing BUY: {symbol} x{qty} @ ~${price:.2f}")

            response = requests.post(f"{self.api_url}/place-order", json=order, timeout=5)
            result = response.json()

            if result.get('success'):
                logger.info(f"    ORDER FILLED!")

                # Track position
                self.news_positions[symbol] = {
                    'entry_price': price,
                    'quantity': qty,
                    'entry_time': datetime.now(),
                    'catalyst': alert.catalyst_type,
                    'headline': alert.headline[:100],
                    'stop_loss': price * (1 - self.stop_loss_pct),
                    'take_profit': price * (1 + self.take_profit_pct)
                }

                self.trades_executed += 1

            else:
                logger.error(f"    Order failed: {result.get('detail', result)}")

        except Exception as e:
            logger.error(f"    Buy error: {e}")

    def _close_position(self, symbol: str, reason: str):
        """Close a news position"""
        if symbol not in self.news_positions:
            return

        pos = self.news_positions[symbol]

        try:
            quote = self._get_quote(symbol)
            current_price = quote.get('bid', quote.get('last', 0)) if quote else 0

            order = {
                'symbol': symbol,
                'quantity': pos['quantity'],
                'action': 'sell',
                'order_type': 'market',
                'time_in_force': 'day',
                'extended_hours': True
            }

            logger.info(f"    Closing {symbol}: {reason}")

            response = requests.post(f"{self.api_url}/place-order", json=order, timeout=5)
            result = response.json()

            if result.get('success'):
                # Calculate P&L
                pnl = (current_price - pos['entry_price']) * pos['quantity']
                pnl_pct = (current_price - pos['entry_price']) / pos['entry_price'] * 100

                if pnl >= 0:
                    self.wins += 1
                else:
                    self.losses += 1
                self.total_pnl += pnl

                logger.info(f"    CLOSED: P/L ${pnl:.2f} ({pnl_pct:+.1f}%)")

                # Record trade
                self.trade_history.append({
                    'symbol': symbol,
                    'entry_price': pos['entry_price'],
                    'exit_price': current_price,
                    'quantity': pos['quantity'],
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'catalyst': pos['catalyst'],
                    'reason': reason,
                    'entry_time': pos['entry_time'].isoformat(),
                    'exit_time': datetime.now().isoformat()
                })

                del self.news_positions[symbol]

        except Exception as e:
            logger.error(f"    Close error: {e}")

    def _close_all_positions(self, reason: str):
        """Close all news positions"""
        symbols = list(self.news_positions.keys())
        for symbol in symbols:
            self._close_position(symbol, reason)

    def _monitor_positions(self):
        """Monitor positions for stops and time exits"""
        while self.is_running:
            try:
                now = datetime.now()

                for symbol in list(self.news_positions.keys()):
                    pos = self.news_positions.get(symbol)
                    if not pos:
                        continue

                    # Get current price
                    quote = self._get_quote(symbol)
                    if not quote:
                        continue

                    current_price = quote.get('last', 0)
                    if current_price <= 0:
                        continue

                    # Check stop loss
                    if current_price <= pos['stop_loss']:
                        logger.warning(f"STOP LOSS HIT: {symbol} @ ${current_price:.2f}")
                        self._close_position(symbol, "Stop loss")
                        continue

                    # Check take profit
                    if current_price >= pos['take_profit']:
                        logger.info(f"TAKE PROFIT HIT: {symbol} @ ${current_price:.2f}")
                        self._close_position(symbol, "Take profit")
                        continue

                    # Check time limit
                    hold_time = (now - pos['entry_time']).total_seconds() / 60
                    if hold_time >= self.max_hold_minutes:
                        logger.info(f"TIME EXIT: {symbol} after {hold_time:.1f} minutes")
                        self._close_position(symbol, "Time limit")
                        continue

            except Exception as e:
                logger.error(f"Monitor error: {e}")

            time.sleep(1)  # Check every second

    def _get_quote(self, symbol: str) -> Optional[Dict]:
        """Get current quote for symbol"""
        try:
            r = requests.get(f"{self.api_url}/quote/{symbol}", timeout=2)
            if r.status_code == 200:
                return r.json()
        except:
            pass
        return None

    def _print_stats(self):
        """Print trading statistics"""
        logger.info("\n" + "=" * 60)
        logger.info("NEWS TRADING SESSION STATS")
        logger.info("=" * 60)
        logger.info(f"Trades executed: {self.trades_executed}")
        logger.info(f"Wins: {self.wins} | Losses: {self.losses}")

        if self.trades_executed > 0:
            win_rate = self.wins / self.trades_executed * 100
            logger.info(f"Win rate: {win_rate:.1f}%")

        logger.info(f"Total P/L: ${self.total_pnl:.2f}")
        logger.info("=" * 60)

    def get_status(self) -> Dict:
        """Get trader status"""
        return {
            'is_running': self.is_running,
            'active_positions': len(self.news_positions),
            'positions': {
                s: {
                    'entry': p['entry_price'],
                    'catalyst': p['catalyst'],
                    'since': p['entry_time'].isoformat()
                }
                for s, p in self.news_positions.items()
            },
            'trades_executed': self.trades_executed,
            'wins': self.wins,
            'losses': self.losses,
            'total_pnl': round(self.total_pnl, 2),
            'news_scanner': self.news_scanner.get_status() if self.news_scanner else None
        }


def main():
    """Run the news triggered trader"""
    trader = NewsTriggeredTrader()

    # Default watchlist - common momentum stocks
    watchlist = [
        'AAPL', 'TSLA', 'NVDA', 'AMD', 'META', 'GOOGL', 'AMZN', 'MSFT',
        'SPY', 'QQQ', 'PLTR', 'SOFI', 'NIO', 'RIVN', 'LCID'
    ]

    print("\n" + "=" * 60)
    print("NEWS TRIGGERED TRADER")
    print("=" * 60)
    print(f"Watchlist: {', '.join(watchlist[:10])}...")
    print("Press Ctrl+C to stop")
    print("=" * 60 + "\n")

    trader.start(watchlist)

    try:
        while True:
            time.sleep(30)
            status = trader.get_status()
            logger.info(
                f"[STATUS] Positions: {status['active_positions']} | "
                f"Trades: {status['trades_executed']} | "
                f"P/L: ${status['total_pnl']:.2f}"
            )
    except KeyboardInterrupt:
        print("\nStopping...")
        trader.stop()


if __name__ == "__main__":
    main()
