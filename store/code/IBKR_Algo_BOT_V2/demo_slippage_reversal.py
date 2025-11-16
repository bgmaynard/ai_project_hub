"""
Demo: Slippage Monitor & Reversal Detector
Shows how to use the new risk management features
"""

from ai.warrior_slippage_monitor import get_slippage_monitor
from ai.warrior_reversal_detector import get_reversal_detector

print("=" * 70)
print("SLIPPAGE & REVERSAL MONITORING DEMO")
print("=" * 70)

# Initialize monitors
slippage_monitor = get_slippage_monitor()
reversal_detector = get_reversal_detector()

# ===== SLIPPAGE MONITORING =====
print("\n1. SLIPPAGE MONITORING")
print("-" * 70)

# Good fill (minimal slippage)
print("\nScenario 1: Good Fill")
exec1 = slippage_monitor.record_execution("AAPL", "buy", 150.00, 150.05, 100)
print(f"  Expected: $150.00, Actual: $150.05")
print(f"  Slippage: {exec1.slippage_pct:.3%} - {exec1.slippage_level.value}")

# Moderate slippage
print("\nScenario 2: Moderate Slippage")
exec2 = slippage_monitor.record_execution("TSLA", "buy", 245.00, 245.40, 50)
print(f"  Expected: $245.00, Actual: $245.40")
print(f"  Slippage: {exec2.slippage_pct:.3%} - {exec2.slippage_level.value}")

# Critical slippage
print("\nScenario 3: CRITICAL Slippage")
exec3 = slippage_monitor.record_execution("AAPL", "sell", 151.00, 150.20, 100)
print(f"  Expected: $151.00, Actual: $150.20")
print(f"  Slippage: {exec3.slippage_pct:.3%} - {exec3.slippage_level.value}")

# Get stats
print("\nSlippage Statistics:")
stats = slippage_monitor.get_stats()
print(f"  Total executions: {stats['total']}")
print(f"  Average slippage: {stats['avg_slippage']:.3%}")
print(f"  Critical fills: {stats['critical_count']}")

# ===== REVERSAL DETECTION =====
print("\n\n2. JACKNIFE REVERSAL DETECTION")
print("-" * 70)

# Simulate jacknife pattern
print("\nScenario: Stock went up 3% then crashed 2%")
recent_prices = [244.0, 245.5, 247.0, 251.5, 246.0]  # Up then sudden drop
reversal = reversal_detector.detect_jacknife(
    symbol="TSLA",
    current_price=246.0,
    entry_price=244.0,
    recent_prices=recent_prices,
    direction='long'
)

if reversal:
    print(f"  REVERSAL DETECTED!")
    print(f"  Type: {reversal.reversal_type.value}")
    print(f"  Severity: {reversal.severity.value}")
    print(f"  Recommendation: {reversal.recommendation}")
    print(f"  Fast Exit Required: {reversal_detector.should_exit_fast(reversal)}")
else:
    print("  No reversal detected")

# ===== INTEGRATION EXAMPLE =====
print("\n\n3. INTEGRATION WITH TRADING")
print("-" * 70)
print("""
Example: Monitor slippage and reversals in real-time

# During order execution:
execution = slippage_monitor.record_execution(
    symbol="AAPL",
    side="buy",
    expected_price=limit_price,
    actual_price=fill_price,
    shares=shares
)

if execution.slippage_level == SlippageLevel.CRITICAL:
    alert("High slippage detected!")

# During position monitoring:
reversal = reversal_detector.detect_jacknife(
    symbol="AAPL",
    current_price=latest_price,
    entry_price=entry_price,
    recent_prices=price_history[-10:],
    direction='long'
)

if reversal and reversal_detector.should_exit_fast(reversal):
    # Execute emergency exit
    exit_position_immediately(symbol)
    alert(f"Jacknife reversal - exited {symbol}")
""")

print("\n" + "=" * 70)
print("Demo complete!")
print("=" * 70)
