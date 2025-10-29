"""Analytics Engine for Trading Bot"""
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List
from pathlib import Path

class AnalyticsEngine:
    def __init__(self, trade_log_file="logs/trades.json"):
        self.trade_log_file = Path(trade_log_file)
        
    def get_all_trades(self):
        if not self.trade_log_file.exists():
            return []
        with open(self.trade_log_file, 'r') as f:
            return json.load(f)
    
    def get_performance_summary(self):
        trades = self.get_all_trades()
        if not trades:
            return {"total_trades":0,"total_pnl":0.0,"win_rate":0.0,"avg_win":0.0,"avg_loss":0.0,"largest_win":0.0,"largest_loss":0.0}
        total_pnl = sum(t.get('pnl', 0) for t in trades)
        winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in trades if t.get('pnl', 0) < 0]
        return {
            "total_trades":len(trades),"winning_trades":len(winning_trades),"losing_trades":len(losing_trades),
            "total_pnl":total_pnl,"win_rate":len(winning_trades)/len(trades) if trades else 0,
            "avg_win":sum(t['pnl'] for t in winning_trades)/len(winning_trades) if winning_trades else 0,
            "avg_loss":sum(t['pnl'] for t in losing_trades)/len(losing_trades) if losing_trades else 0,
            "largest_win":max((t.get('pnl',0) for t in trades),default=0),
            "largest_loss":min((t.get('pnl',0) for t in trades),default=0)
        }
    
    def get_daily_pnl(self, days=30):
        trades = self.get_all_trades()
        if not trades:
            return []
        daily = {}
        for trade in trades:
            date = datetime.fromisoformat(trade['timestamp']).date().isoformat()
            if date not in daily:
                daily[date] = {"date":date,"pnl":0,"trades":0}
            daily[date]["pnl"] += trade.get('pnl',0)
            daily[date]["trades"] += 1
        result = sorted(daily.values(), key=lambda x: x['date'])
        return result[-days:]
    
    def get_symbol_performance(self):
        trades = self.get_all_trades()
        if not trades:
            return []
        symbols = {}
        for trade in trades:
            symbol = trade.get('symbol','UNKNOWN')
            if symbol not in symbols:
                symbols[symbol] = {"symbol":symbol,"total_trades":0,"total_pnl":0,"wins":0,"losses":0}
            symbols[symbol]["total_trades"] += 1
            pnl = trade.get('pnl',0)
            symbols[symbol]["total_pnl"] += pnl
            if pnl > 0:
                symbols[symbol]["wins"] += 1
            elif pnl < 0:
                symbols[symbol]["losses"] += 1
        result = list(symbols.values())
        result.sort(key=lambda x: x['total_pnl'], reverse=True)
        return result
    
    def get_recent_activity(self, limit=10):
        trades = self.get_all_trades()
        trades.sort(key=lambda x: x.get('timestamp',''), reverse=True)
        return trades[:limit]

def create_analytics_engine():
    return AnalyticsEngine()
