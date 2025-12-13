from dataclasses import dataclass
from typing import Optional
import time

@dataclass
class RiskConfig:
    max_position_usd: float = 25000
    max_daily_loss_usd: float = 1500
    per_trade_risk_usd: float = 300
    atr_mult_trailing: float = 2.0

class RiskState:
    def __init__(self):
        self.daily_pnl = 0.0
        self.last_reset_day = time.strftime("%Y-%m-%d")

    def update_pnl(self, delta: float):
        today = time.strftime("%Y-%m-%d")
        if today != self.last_reset_day:
            self.daily_pnl = 0.0
            self.last_reset_day = today
        self.daily_pnl += delta

class RiskManager:
    def __init__(self, cfg: RiskConfig):
        self.cfg = cfg
        self.state = RiskState()

    def can_open(self, symbol: str, price: float, intended_qty: float) -> tuple[bool,str]:
        position_val = abs(price * intended_qty)
        if position_val > self.cfg.max_position_usd:
            return False, f"position {position_val:.2f} exceeds max_position_usd"
        if self.state.daily_pnl < -abs(self.cfg.max_daily_loss_usd):
            return False, f"daily loss {self.state.daily_pnl:.2f} exceeds limit"
        # per-trade risk rough check: notional vs risk budget (improve with ATR/stop distance)
        if position_val > self.cfg.per_trade_risk_usd * 10:
            return False, f"trade notional {position_val:.2f} too large for per_trade_risk_usd"
        return True, "ok"

    def trail_stop_price(self, entry: float, atr: Optional[float]) -> Optional[float]:
        if atr is None: return None
        return entry - self.cfg.atr_mult_trailing * atr
