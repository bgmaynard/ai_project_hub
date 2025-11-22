"""
ALPACA AI AUTO-TRADER
Connects your AI signal generator to Alpaca for automatic trading
"""
import asyncio
import logging
import os
import requests
from datetime import datetime
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AlpacaAITrader:
    def __init__(self):
        # Alpaca client
        self.client = TradingClient(
            os.getenv('ALPACA_API_KEY'),
            os.getenv('ALPACA_SECRET_KEY'),
            paper=True
        )
        
        # Your AI API
        self.ai_base_url = "http://localhost:9101"
        
        # Configuration
        self.confidence_threshold = 0.15  # 15% minimum confidence
        self.max_positions = 3
        self.max_daily_trades = 10
        self.position_size = 1  # Number of shares per trade
        
        # Watchlist (approved symbols)
        self.symbols = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
            "NVDA", "META", "SPY", "QQQ", "AMD"
        ]
        
        # State
        self.running = False
        self.trades_today = 0
        
    def get_account_info(self):
        """Get Alpaca account information"""
        return self.client.get_account()
    
    def get_positions(self):
        """Get current open positions"""
        return self.client.get_all_positions()
    
    def get_ai_prediction(self, symbol):
        """Get AI prediction from your signal generator"""
        try:
            response = requests.post(
                f"{self.ai_base_url}/api/ai/predict",
                json={"symbol": symbol, "timeframe": "5m"},
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"AI prediction failed for {symbol}: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting AI prediction for {symbol}: {e}")
            return None
    
    def place_market_order(self, symbol, side, qty):
        """Place market order on Alpaca"""
        try:
            order_request = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side,
                time_in_force=TimeInForce.DAY
            )
            
            order = self.client.submit_order(order_request)
            
            logger.info(f"✅ ORDER SUBMITTED!")
            logger.info(f"   Symbol: {symbol}")
            logger.info(f"   Side: {side}")
            logger.info(f"   Quantity: {qty}")
            logger.info(f"   Order ID: {order.id}")
            logger.info(f"   Status: {order.status}")
            
            return order
            
        except Exception as e:
            logger.error(f"❌ Order placement failed: {e}")
            return None
    
    async def trading_cycle(self):
        """Main trading logic - runs every cycle"""
        
        # Check daily trade limit
        if self.trades_today >= self.max_daily_trades:
            logger.info(f"📊 Daily trade limit reached ({self.trades_today}/{self.max_daily_trades})")
            return
        
        # Get current positions
        positions = self.get_positions()
        current_position_count = len(positions)
        
        if current_position_count >= self.max_positions:
            logger.info(f"📊 Max positions reached ({current_position_count}/{self.max_positions})")
            return
        
        # Get symbols we don't already own
        held_symbols = [pos.symbol for pos in positions]
        available_symbols = [s for s in self.symbols if s not in held_symbols]
        
        if not available_symbols:
            logger.info("📊 All watchlist symbols already held")
            return
        
        logger.info(f"🔍 Scanning {len(available_symbols)} symbols for signals...")
        
        # Scan all available symbols
        signals = []
        for symbol in available_symbols:
            prediction = self.get_ai_prediction(symbol)
            
            if prediction:
                confidence = prediction.get('confidence', 0)
                action = prediction.get('action', 'HOLD')
                
                if confidence > self.confidence_threshold:
                    signals.append({
                        'symbol': symbol,
                        'confidence': confidence,
                        'action': action
                    })
                    logger.info(f"   📈 {symbol}: {confidence*100:.2f}% confidence - {action}")
        
        if not signals:
            logger.info("   No signals above threshold")
            return
        
        # Sort by confidence and take the best
        signals.sort(key=lambda x: x['confidence'], reverse=True)
        best_signal = signals[0]
        
        logger.info(f"")
        logger.info(f"⚡ BEST SIGNAL: {best_signal['symbol']}")
        logger.info(f"   Confidence: {best_signal['confidence']*100:.2f}%")
        logger.info(f"   Action: {best_signal['action']}")
        
        # Execute the trade
        side = OrderSide.BUY if best_signal['action'] == 'BUY' else OrderSide.SELL
        
        order = self.place_market_order(
            symbol=best_signal['symbol'],
            side=side,
            qty=self.position_size
        )
        
        if order:
            self.trades_today += 1
            logger.info(f"")
            logger.info(f"✅ TRADE EXECUTED! ({self.trades_today}/{self.max_daily_trades} today)")
    
    async def start(self):
        """Start the auto-trading bot"""
        self.running = True
        
        # Display startup info
        account = self.get_account_info()
        
        logger.info("")
        logger.info("=" * 70)
        logger.info("🤖 ALPACA AI AUTO-TRADER - STARTED")
        logger.info("=" * 70)
        logger.info(f"Account Number: {account.account_number}")
        logger.info(f"Account Status: {account.status}")
        logger.info(f"Cash: ${float(account.cash):,.2f}")
        logger.info(f"Buying Power: ${float(account.buying_power):,.2f}")
        logger.info(f"Pattern Day Trader: {account.pattern_day_trader}")
        logger.info("")
        logger.info(f"Configuration:")
        logger.info(f"  • AI Confidence Threshold: {self.confidence_threshold * 100}%")
        logger.info(f"  • Max Positions: {self.max_positions}")
        logger.info(f"  • Max Daily Trades: {self.max_daily_trades}")
        logger.info(f"  • Position Size: {self.position_size} shares")
        logger.info(f"  • Watchlist: {len(self.symbols)} symbols")
        logger.info("=" * 70)
        logger.info("")
        
        # Main trading loop
        while self.running:
            try:
                logger.info(f"[{datetime.now().strftime('%I:%M:%S %p')}] Starting trading cycle...")
                
                await self.trading_cycle()
                
                logger.info(f"⏳ Waiting 2 minutes before next cycle...\n")
                await asyncio.sleep(120)  # Check every 2 minutes
                
            except KeyboardInterrupt:
                logger.info("\n🛑 Keyboard interrupt - stopping...")
                break
            except Exception as e:
                logger.error(f"❌ Error in trading loop: {e}")
                logger.info("⏳ Waiting 1 minute before retry...")
                await asyncio.sleep(60)
        
        logger.info("\n" + "=" * 70)
        logger.info(f"🛑 BOT STOPPED - Trades today: {self.trades_today}")
        logger.info("=" * 70)
    
    def stop(self):
        """Stop the bot"""
        self.running = False

async def main():
    trader = AlpacaAITrader()
    
    try:
        await trader.start()
    except KeyboardInterrupt:
        trader.stop()

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  ALPACA AI AUTO-TRADER")
    print("  Connecting your AI predictions to live trading")
    print("  Paper Trading Mode - Safe Testing")
    print("  Press Ctrl+C to stop")
    print("=" * 70 + "\n")
    
    asyncio.run(main())
