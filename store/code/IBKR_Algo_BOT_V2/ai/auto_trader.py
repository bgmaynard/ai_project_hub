"""
Automated Trading Bot
AI-driven with risk management
"""
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List
import json
from pathlib import Path

class TradingConfig:
    def __init__(self):
        # AI thresholds
        self.min_confidence = float(os.getenv("AUTO_TRADE_MIN_CONFIDENCE", "0.65"))
        self.min_prob_up = float(os.getenv("AUTO_TRADE_MIN_PROB", "0.60"))
        
        # Risk management
        self.max_position_size = int(os.getenv("AUTO_TRADE_MAX_SHARES", "10"))
        self.max_daily_trades = int(os.getenv("AUTO_TRADE_MAX_DAILY", "5"))
        self.max_daily_loss = float(os.getenv("AUTO_TRADE_MAX_LOSS", "500.0"))
        self.stop_loss_pct = float(os.getenv("AUTO_TRADE_STOP_LOSS", "0.05"))
        
        # Trading enabled
        self.enabled = os.getenv("AUTO_TRADE_ENABLED", "false").lower() == "true"
        
    def __str__(self):
        return f"AutoTrader(enabled={self.enabled}, min_conf={self.min_confidence}, max_pos={self.max_position_size})"

class TradeLogger:
    def __init__(self, log_file="trades.json"):
        self.log_file = Path("logs") / log_file
        self.log_file.parent.mkdir(exist_ok=True)
        
    def log_trade(self, trade_data: Dict[str, Any]):
        trades = self.get_all_trades()
        trades.append(trade_data)
        with open(self.log_file, 'w') as f:
            json.dump(trades, f, indent=2)
    
    def get_all_trades(self) -> List[Dict[str, Any]]:
        if not self.log_file.exists():
            return []
        with open(self.log_file, 'r') as f:
            return json.load(f)
    
    def get_today_trades(self) -> List[Dict[str, Any]]:
        all_trades = self.get_all_trades()
        today = datetime.utcnow().date()
        return [t for t in all_trades if datetime.fromisoformat(t['timestamp']).date() == today]

class AutoTrader:
    def __init__(self, config: TradingConfig, predictor, adapter, logger: TradeLogger):
        self.config = config
        self.predictor = predictor
        self.adapter = adapter
        self.logger = logger
        
    def should_trade(self, symbol: str) -> Dict[str, Any]:
        """Determine if we should trade based on AI and risk rules"""
        result = {
            "should_trade": False,
            "reason": "",
            "action": None,
            "quantity": 0,
            "prediction": None
        }
        
        # Check if auto-trading is enabled
        if not self.config.enabled:
            result["reason"] = "Auto-trading is disabled"
            return result
        
        # Check daily trade limit
        today_trades = self.logger.get_today_trades()
        if len(today_trades) >= self.config.max_daily_trades:
            result["reason"] = f"Daily trade limit reached ({self.config.max_daily_trades})"
            return result
        
        # Check daily loss limit
        today_pnl = sum(t.get('pnl', 0) for t in today_trades)
        if today_pnl < -self.config.max_daily_loss:
            result["reason"] = f"Daily loss limit reached (${today_pnl:.2f})"
            return result
        
        # Get AI prediction
        try:
            prediction = self.predictor.predict(symbol)
        except Exception as e:
            result["reason"] = f"Prediction failed: {e}"
            return result
        
        result["prediction"] = prediction
        
        # Check for errors
        if "error" in prediction:
            result["reason"] = f"Prediction error: {prediction['error']}"
            return result
        
        # Check confidence threshold
        if prediction['confidence'] < self.config.min_confidence:
            result["reason"] = f"Confidence too low ({prediction['confidence']:.2%} < {self.config.min_confidence:.2%})"
            return result
        
        # Check probability threshold
        if prediction['prob_up'] < self.config.min_prob_up:
            result["reason"] = f"Probability too low ({prediction['prob_up']:.2%} < {self.config.min_prob_up:.2%})"
            return result
        
        # Bullish signal - BUY
        if prediction['prediction'] == 1:
            result["should_trade"] = True
            result["action"] = "BUY"
            result["quantity"] = self.config.max_position_size
            result["reason"] = f"BULLISH: {prediction['confidence']:.2%} confidence, {prediction['signal_detail']}"
            return result
        
        # Bearish - could add short selling logic here
        result["reason"] = f"BEARISH signal - not trading"
        return result
    
    def execute_trade(self, symbol: str) -> Dict[str, Any]:
        """
        Execute trade if conditions are met.

        GATING ENFORCEMENT: All trades MUST go through Signal Gating Engine.
        """
        decision = self.should_trade(symbol)

        if not decision['should_trade']:
            return {
                "executed": False,
                "reason": decision['reason'],
                "timestamp": datetime.utcnow().isoformat()
            }

        # GATING ENFORCEMENT: Route through Signal Gating Engine first
        try:
            from ai.gated_trading import get_gated_trading_manager
            manager = get_gated_trading_manager()

            current_price = decision['prediction'].get('current_price', 0)
            approved, exec_request, reason = manager.gate_trade_attempt(
                symbol=symbol,
                trigger_type="auto_trader",
                quote={"price": current_price}
            )

            if not approved:
                return {
                    "executed": False,
                    "gating_vetoed": True,
                    "reason": f"GATING VETOED: {reason}",
                    "timestamp": datetime.utcnow().isoformat()
                }

            gating_token = f"GATED_{symbol}_{datetime.utcnow().strftime('%H%M%S')}"

        except Exception as e:
            # Fail-closed: on gating error, reject trade
            return {
                "executed": False,
                "gating_error": True,
                "reason": f"GATING ERROR (fail-closed): {e}",
                "timestamp": datetime.utcnow().isoformat()
            }

        # Gating approved - proceed with execution
        current_price = decision['prediction'].get('current_price', 0)
        limit_price = current_price * 1.01  # 1% above current

        # Place order
        try:
            order_result = self.adapter.place_limit_order_sync(
                symbol=symbol,
                qty=decision['quantity'],
                price=limit_price,
                action=decision['action']
            )

            # Log the trade
            trade_log = {
                "timestamp": datetime.utcnow().isoformat(),
                "symbol": symbol,
                "action": decision['action'],
                "quantity": decision['quantity'],
                "limit_price": limit_price,
                "current_price": current_price,
                "order_id": order_result.get('orderId'),
                "gating_token": gating_token,
                "prediction": decision['prediction'],
                "reason": decision['reason'],
                "pnl": 0  # Will update later when closed
            }
            self.logger.log_trade(trade_log)

            return {
                "executed": True,
                "gating_token": gating_token,
                "order": order_result,
                "trade_log": trade_log,
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            return {
                "executed": False,
                "gating_token": gating_token,
                "reason": f"Order failed: {e}",
                "timestamp": datetime.utcnow().isoformat()
            }

def create_auto_trader(predictor, adapter):
    """Factory function to create auto trader"""
    config = TradingConfig()
    logger = TradeLogger()
    return AutoTrader(config, predictor, adapter, logger)

