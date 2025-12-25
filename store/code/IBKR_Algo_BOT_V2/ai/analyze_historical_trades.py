"""
Analyze Historical Trades vs Technical Signals
===============================================
Loads actual scalper trades and correlates with technical signals
to see if the signal strategy would have improved results.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import httpx
from datetime import datetime, timedelta
from collections import defaultdict
from typing import List, Dict
import pandas as pd
import numpy as np

from ai.technical_signals import TechnicalSignalAnalyzer, SignalState

API_BASE = "http://localhost:9100"

def load_trades() -> List[Dict]:
    """Load historical trades from scalper_trades.json"""
    trades_path = os.path.join(os.path.dirname(__file__), "scalper_trades.json")

    with open(trades_path, 'r') as f:
        data = json.load(f)

    return data.get("trades", [])


def fetch_ohlc_for_trade(client: httpx.Client, symbol: str, trade_time: str) -> List[Dict]:
    """Fetch OHLC data around the trade time"""
    try:
        # Get 5 days of 5m data around the trade
        resp = client.get(
            f"{API_BASE}/api/charts/ohlc/{symbol}",
            params={"timeframe": "5m", "days": 5}
        )

        if resp.status_code != 200:
            return []

        data = resp.json()
        if not data.get("success"):
            return []

        ohlc = data.get("data", [])
        volume_data = data.get("volume", [])

        # Merge volume
        vol_by_time = {v["time"]: v["value"] for v in volume_data}
        for bar in ohlc:
            bar["volume"] = vol_by_time.get(bar["time"], 0)

        return ohlc

    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return []


def analyze_trade_with_signals(trade: Dict, ohlc: List[Dict], analyzer: TechnicalSignalAnalyzer) -> Dict:
    """Analyze a single trade with technical signals"""

    if not ohlc or len(ohlc) < 50:
        return None

    # Parse trade entry time
    entry_time_str = trade.get("entry_time", "")
    try:
        entry_dt = datetime.fromisoformat(entry_time_str.replace("Z", "+00:00"))
        entry_unix = int(entry_dt.timestamp())
    except:
        return None

    # Find bars before entry time
    df = pd.DataFrame(ohlc)
    df = df[df['time'] <= entry_unix].tail(200)

    if len(df) < 50:
        return None

    # Get signal state at entry
    window = df.to_dict('records')
    signal = analyzer.analyze(trade["symbol"], window, "5m")

    if not signal:
        return None

    # Determine if trade was a winner
    pnl = trade.get("pnl", 0)
    is_winner = pnl > 0

    return {
        "trade_id": trade.get("trade_id"),
        "symbol": trade.get("symbol"),
        "entry_time": entry_time_str,
        "pnl": pnl,
        "pnl_percent": trade.get("pnl_percent", 0),
        "exit_reason": trade.get("exit_reason"),
        "is_winner": is_winner,
        # Signal state at entry
        "confluence_score": signal.confluence_score,
        "signal_bias": signal.signal_bias,
        "ema_bullish": signal.ema_bullish,
        "macd_bullish": signal.macd_bullish,
        "price_above_vwap": signal.price_above_vwap,
        "ema_crossover": signal.ema_crossover,
        "macd_crossover": signal.macd_crossover,
        "vwap_crossover": signal.vwap_crossover,
        "candle_momentum": signal.candle_momentum,
        # Would signal have approved this trade?
        "signal_approved": signal.confluence_score >= 70 and (
            signal.ema_bullish and signal.macd_bullish
        )
    }


def run_analysis():
    """Main analysis function"""

    print("="*70)
    print("HISTORICAL TRADES vs TECHNICAL SIGNALS ANALYSIS")
    print("="*70)

    # Load trades
    trades = load_trades()
    print(f"\nLoaded {len(trades)} historical trades")

    # Filter to closed trades
    closed_trades = [t for t in trades if t.get("status") == "closed"]
    print(f"Closed trades: {len(closed_trades)}")

    # Get unique symbols
    symbols = list(set(t.get("symbol") for t in closed_trades))
    print(f"Unique symbols: {len(symbols)}")

    # Analyze trades
    analyzer = TechnicalSignalAnalyzer()
    client = httpx.Client(timeout=30.0)

    analyzed = []
    ohlc_cache = {}

    # Process each trade
    for i, trade in enumerate(closed_trades):
        symbol = trade.get("symbol")

        if i % 20 == 0:
            print(f"Processing trade {i+1}/{len(closed_trades)}...")

        # Cache OHLC data per symbol
        if symbol not in ohlc_cache:
            ohlc_cache[symbol] = fetch_ohlc_for_trade(client, symbol, trade.get("entry_time"))

        ohlc = ohlc_cache.get(symbol, [])
        if not ohlc:
            continue

        result = analyze_trade_with_signals(trade, ohlc, analyzer)
        if result:
            analyzed.append(result)

    client.close()

    if not analyzed:
        print("\nNo trades could be analyzed (no OHLC data available)")
        return

    print(f"\nAnalyzed {len(analyzed)} trades with signals")

    # Calculate statistics
    df = pd.DataFrame(analyzed)

    # Overall stats
    total_trades = len(df)
    winners = df[df['is_winner'] == True]
    losers = df[df['is_winner'] == False]

    overall_win_rate = len(winners) / total_trades * 100 if total_trades > 0 else 0
    overall_pnl = df['pnl'].sum()

    print(f"\n[OVERALL PERFORMANCE]")
    print(f"   Total Trades: {total_trades}")
    print(f"   Winners: {len(winners)}")
    print(f"   Losers: {len(losers)}")
    print(f"   Win Rate: {overall_win_rate:.1f}%")
    print(f"   Total P&L: ${overall_pnl:.2f}")

    # Signal-approved trades
    approved = df[df['signal_approved'] == True]
    rejected = df[df['signal_approved'] == False]

    if len(approved) > 0:
        approved_winners = approved[approved['is_winner'] == True]
        approved_win_rate = len(approved_winners) / len(approved) * 100
        approved_pnl = approved['pnl'].sum()
    else:
        approved_win_rate = 0
        approved_pnl = 0

    if len(rejected) > 0:
        rejected_winners = rejected[rejected['is_winner'] == True]
        rejected_win_rate = len(rejected_winners) / len(rejected) * 100
        rejected_pnl = rejected['pnl'].sum()
    else:
        rejected_win_rate = 0
        rejected_pnl = 0

    print(f"\n[SIGNAL FILTER COMPARISON]")
    print(f"   Signal-Approved Trades (conf>=70, EMA+MACD bullish):")
    print(f"      Count: {len(approved)}")
    print(f"      Win Rate: {approved_win_rate:.1f}%")
    print(f"      P&L: ${approved_pnl:.2f}")
    print(f"\n   Signal-Rejected Trades:")
    print(f"      Count: {len(rejected)}")
    print(f"      Win Rate: {rejected_win_rate:.1f}%")
    print(f"      P&L: ${rejected_pnl:.2f}")

    improvement = approved_win_rate - overall_win_rate
    print(f"\n   >>> Win Rate Improvement: {improvement:+.1f}%")

    # Confluence bucket analysis
    print(f"\n[CONFLUENCE SCORE ANALYSIS]")
    buckets = {
        "0-40%": df[df['confluence_score'] < 40],
        "40-60%": df[(df['confluence_score'] >= 40) & (df['confluence_score'] < 60)],
        "60-80%": df[(df['confluence_score'] >= 60) & (df['confluence_score'] < 80)],
        "80-100%": df[df['confluence_score'] >= 80]
    }

    for bucket_name, bucket_df in buckets.items():
        if len(bucket_df) == 0:
            continue
        bucket_winners = bucket_df[bucket_df['is_winner'] == True]
        bucket_wr = len(bucket_winners) / len(bucket_df) * 100
        bucket_pnl = bucket_df['pnl'].sum()
        print(f"   {bucket_name}: {len(bucket_df)} trades, {bucket_wr:.1f}% win rate, ${bucket_pnl:.2f} P&L")

    # Signal combination analysis
    print(f"\n[SIGNAL COMBINATIONS]")

    # Group by signal combinations
    def get_combo(row):
        signals = []
        if row['ema_bullish']:
            signals.append("EMA")
        if row['macd_bullish']:
            signals.append("MACD")
        if row['price_above_vwap']:
            signals.append("VWAP")
        if row['candle_momentum'] == "BUILDING":
            signals.append("MOM")
        return "+".join(sorted(signals)) if signals else "NONE"

    df['combo'] = df.apply(get_combo, axis=1)

    combo_stats = []
    for combo, group in df.groupby('combo'):
        if len(group) < 3:
            continue
        combo_winners = group[group['is_winner'] == True]
        combo_stats.append({
            'combo': combo,
            'count': len(group),
            'win_rate': len(combo_winners) / len(group) * 100,
            'pnl': group['pnl'].sum()
        })

    combo_stats.sort(key=lambda x: x['win_rate'], reverse=True)

    for stat in combo_stats[:7]:
        print(f"   {stat['combo']}: {stat['count']} trades, {stat['win_rate']:.1f}% win rate, ${stat['pnl']:.2f}")

    # Correlation analysis
    print(f"\n[CORRELATION WITH WINS]")

    corr_df = pd.DataFrame({
        'win': df['is_winner'].astype(int),
        'confluence': df['confluence_score'],
        'ema_bullish': df['ema_bullish'].astype(int),
        'macd_bullish': df['macd_bullish'].astype(int),
        'above_vwap': df['price_above_vwap'].astype(int),
        'momentum_building': (df['candle_momentum'] == 'BUILDING').astype(int)
    })

    correlations = corr_df.corr()['win'].drop('win')

    for signal, corr in correlations.sort_values(ascending=False).items():
        if abs(corr) > 0.05:
            direction = "+" if corr > 0 else ""
            print(f"   {signal}: {direction}{corr:.3f}")

    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "total_trades_analyzed": len(analyzed),
        "overall": {
            "trades": total_trades,
            "win_rate": round(overall_win_rate, 1),
            "pnl": round(overall_pnl, 2)
        },
        "signal_approved": {
            "trades": len(approved),
            "win_rate": round(approved_win_rate, 1),
            "pnl": round(approved_pnl, 2)
        },
        "signal_rejected": {
            "trades": len(rejected),
            "win_rate": round(rejected_win_rate, 1),
            "pnl": round(rejected_pnl, 2)
        },
        "improvement": round(improvement, 1),
        "trades": analyzed[-50:]  # Last 50 for reference
    }

    output_path = os.path.join(os.path.dirname(__file__), "historical_signal_analysis.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n[OK] Results saved to {output_path}")

    return output


if __name__ == "__main__":
    run_analysis()
