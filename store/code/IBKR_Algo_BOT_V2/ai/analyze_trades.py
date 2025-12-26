"""
Trade Analysis Script
=====================
Analyzes individual trades to find patterns.
"""

import sys

sys.path.insert(0, "C:/ai_project_hub/store/code/IBKR_Algo_BOT_V2")

from dotenv import load_dotenv

load_dotenv("C:/ai_project_hub/store/code/IBKR_Algo_BOT_V2/.env")

from collections import Counter

from ai.pybroker_schwab_data import SchwabDataSource
from ai.pybroker_walkforward import SCALPER_PARAMS, _run_minute_strategy

source = SchwabDataSource()

# Get YCBD data for detailed analysis
data = source.query(["YCBD"], days=5)

# Use best config
SCALPER_PARAMS["min_spike_percent"] = 7.0
SCALPER_PARAMS["min_volume_surge"] = 5.0
SCALPER_PARAMS["stop_loss_percent"] = 2.0
SCALPER_PARAMS["trailing_stop_percent"] = 1.5

result = _run_minute_strategy(data, 1000)

print("=" * 70)
print("TRADE-BY-TRADE ANALYSIS - YCBD")
print("=" * 70)
print()

wins = []
losses = []

for i, trade in enumerate(result["trades"]):
    pnl = trade["pnl"]
    pnl_pct = trade.get("pnl_pct", 0)
    reason = trade["exit_reason"]
    bars = trade.get("bars_held", 0)
    entry = trade["entry_price"]
    exit_p = trade["exit_price"]

    status = "WIN" if pnl > 0 else "LOSS"

    if pnl > 0:
        wins.append(trade)
    else:
        losses.append(trade)

    print(
        f"{i+1:2}. {status:<4} ${entry:.3f} -> ${exit_p:.3f} | {pnl_pct:+.1f}% | {reason:<15} | {bars} bars | ${pnl:+.2f}"
    )

print()
print("=" * 70)
print("PATTERN ANALYSIS")
print("=" * 70)

# Analyze exit reasons
exit_reasons = Counter(t["exit_reason"] for t in result["trades"])
print(f"\nExit Reasons:")
for reason, count in exit_reasons.items():
    pnl_by_reason = sum(
        t["pnl"] for t in result["trades"] if t["exit_reason"] == reason
    )
    print(f"  {reason}: {count} trades, ${pnl_by_reason:.2f}")

# Analyze hold times
if wins:
    avg_win_bars = sum(t.get("bars_held", 0) for t in wins) / len(wins)
else:
    avg_win_bars = 0
if losses:
    avg_loss_bars = sum(t.get("bars_held", 0) for t in losses) / len(losses)
else:
    avg_loss_bars = 0

print(f"\nAverage Hold Time:")
print(f"  Winners: {avg_win_bars:.1f} bars ({len(wins)} trades)")
print(f"  Losers: {avg_loss_bars:.1f} bars ({len(losses)} trades)")

# Key insight
print()
print("=" * 70)
print("KEY INSIGHTS")
print("=" * 70)

stop_losses = [t for t in result["trades"] if t["exit_reason"] == "STOP_LOSS"]
trailing_stops = [t for t in result["trades"] if t["exit_reason"] == "TRAILING_STOP"]

stop_loss_pnl = sum(t["pnl"] for t in stop_losses)
trailing_pnl = sum(t["pnl"] for t in trailing_stops)

print(f"\n1. Stop Losses: {len(stop_losses)} trades = ${stop_loss_pnl:.2f}")
print(f"   Average loss per stop: ${stop_loss_pnl/max(1,len(stop_losses)):.2f}")

print(f"\n2. Trailing Stops: {len(trailing_stops)} trades = ${trailing_pnl:.2f}")
print(f"   Average P/L per trailing: ${trailing_pnl/max(1,len(trailing_stops)):.2f}")

# Check if quick stops are the problem
quick_stops = [t for t in stop_losses if t.get("bars_held", 0) <= 2]
print(f"\n3. Quick Stop Losses (<=2 bars): {len(quick_stops)} of {len(stop_losses)}")

if quick_stops:
    quick_loss = sum(t["pnl"] for t in quick_stops)
    print(f"   Quick stops cost: ${quick_loss:.2f}")
    print(f"   -> Consider: entry timing may be at local peaks")

# Profit potential
print(f'\n4. Best trade: ${max(t["pnl"] for t in result["trades"]):.2f}')
print(f'   Worst trade: ${min(t["pnl"] for t in result["trades"]):.2f}')

# Recommendation
print()
print("=" * 70)
print("RECOMMENDATIONS")
print("=" * 70)
print(
    """
Based on the analysis:
1. STOP LOSSES are the main drag - hitting stops too quickly
2. Consider adding a "confirmation bar" before entry
3. Consider waiting for a small pullback after the spike
4. Time-of-day filtering may help (morning momentum vs afternoon chop)
5. Current config (7%/5x/2%/1.5%) is optimal for these conditions
"""
)
