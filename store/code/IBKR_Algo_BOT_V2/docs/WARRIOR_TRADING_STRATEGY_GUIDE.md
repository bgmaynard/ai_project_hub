# Warrior Trading Strategy Guide
## Ross Cameron Momentum Day Trading - Complete Implementation Reference

---

## Table of Contents
1. [Core Philosophy](#core-philosophy)
2. [Stock Selection Criteria](#stock-selection-criteria)
3. [Position Sizing & Risk Management](#position-sizing--risk-management)
4. [Technical Setups](#technical-setups)
5. [Entry Mechanics](#entry-mechanics)
6. [Exit Strategies](#exit-strategies)
7. [Tape Reading](#tape-reading)
8. [Halt Trading](#halt-trading)
9. [Implementation Priorities](#implementation-priorities)

---

## Core Philosophy

### The "Front Runner" Concept
> "Is this stock THE ONE? The obvious stock that most active day traders will be watching?"

- Focus on **1-3 leading percentage gainers** each day
- Better follow-through when stock has market's attention
- Multiple stocks moving = no clear leader = AVOID

### Trading Window
| Session | Time (ET) | Strategy |
|---------|-----------|----------|
| Pre-Market | 4:00 AM - 9:30 AM | Gap scanner, news catalyst hunting |
| Market Open | 9:30 AM - 10:00 AM | **Watch only** - too chaotic |
| Mid-Morning | 10:00 AM - 11:30 AM | Primary trading window |
| Midday | 11:30 AM - 2:00 PM | Reduced activity, tighter filters |
| Power Hour | 3:00 PM - 4:00 PM | Selective continuation plays |

---

## Stock Selection Criteria

### Ross Cameron's 5 Criteria (Grade A = 5/5, Grade B = 4/5)

```python
ROSS_CAMERON_CRITERIA = {
    '1_relative_volume': {
        'threshold': 5.0,  # 5x average volume
        'weight': 'CRITICAL',
        'description': 'Shows unusual interest/activity'
    },
    '2_percent_change': {
        'threshold': 10.0,  # Already up 10%+
        'weight': 'CRITICAL',
        'description': 'Momentum already proven'
    },
    '3_news_catalyst': {
        'types': ['FDA', 'EARNINGS', 'CONTRACT', 'MERGER', 'UPGRADE'],
        'weight': 'HIGH',
        'description': 'Fundamental reason for move'
    },
    '4_price_range': {
        'min': 1.00,
        'max': 20.00,
        'weight': 'HIGH',
        'description': 'Sweet spot for retail momentum'
    },
    '5_float': {
        'max': 10_000_000,  # 10 million shares
        'weight': 'HIGH',
        'description': 'Low supply = explosive moves'
    }
}

POSITION_SIZING_BY_GRADE = {
    'A': 1.0,    # Full position (5/5 criteria)
    'B': 0.5,    # Half position (4/5 criteria)
    'C': 0.25,   # Quarter position / Scalp only (3/5 or less)
}
```

### Additional Filters

| Filter | Condition | Rationale |
|--------|-----------|-----------|
| Spread | < 1% of price | Liquidity requirement |
| Volume | > 500K today | Ensures fills |
| Exchange | NYSE/NASDAQ preferred | Better price action |
| Short Interest | Check HTB status | Squeeze potential |

---

## Position Sizing & Risk Management

### Risk Per Trade
```python
RISK_CONFIG = {
    'max_risk_per_trade': 0.01,  # 1% of account
    'max_daily_loss': 0.03,      # 3% daily stop
    'max_concurrent': 2,          # Positions at once
    'scale_in_tranches': 3,       # Entry in 3 parts
}

def calculate_position_size(account_size, entry_price, stop_price, risk_percent=0.01):
    """Risk-based position sizing"""
    risk_amount = account_size * risk_percent
    risk_per_share = abs(entry_price - stop_price)
    shares = int(risk_amount / risk_per_share)
    return shares
```

### Stop Loss Standards

| Method | Description | Use Case |
|--------|-------------|----------|
| **Low of Last 1-min** | Stop at candle low | Standard for all setups |
| **Arbitrary 5-10-20 cents** | Fixed dollar stops | Quick scalps |
| **25-50 cents** | Wider stops | Parabolic/dip trades |
| **VWAP** | Stop below VWAP | VWAP breakout plays |
| **Half/Whole Dollar** | Psychological levels | Near key levels |

---

## Technical Setups

### Setup 1: Bull Flag / Flat Top Breakout
```
Pattern Recognition:
- Consolidation after initial surge (3-7 candles)
- Declining volume during consolidation
- Ascending support line connecting lows
- Resistance at prior high (flat top) or slightly descending

Entry Trigger:
- Break above consolidation high
- Volume surge on break (2x+ consolidation volume)
- Green candle close above resistance

Target: Prior high + flag height
Stop: Low of consolidation
```

### Setup 2: ABCD Pattern
```
4-Point Structure:
Point A (1): Initial low / start of move
Point A (2): Failed breakout / first high
Point B (3): Higher low (above Point 1)
Point B (4): Breakout above Point 2

Entry Options:
1. Early entry: Buy at Point 3 (higher low)
2. Safe entry: Buy at Point 4 break (confirmation)

Target: If near HOD → 25-50 cents, else → HOD first
Stop: Below Point 3
Risk: Double top potential - requires trapped shorts for fuel
```

### Setup 3: Micro Pullback (ADVANCED)
```
Prerequisites:
- Stock up 10-20%+ in 5-10 minutes
- Near circuit breaker halt levels
- Low float (< 10M shares)
- High volume (float rotation)

Entry:
- Within 1-minute candle as stock dips and surges
- Watch tape for green orders (buying/covering)
- Large seller thinning: 25k → 19k → 15k → 11k → 5k

Use 15-second chart for precision
Stop: Low of micro pullback
Target: 10-20-40 cents, HOD, or halt level
```

### Setup 4: High of Day (HOD) Break
```
Mechanics:
- Short sellers set stops at HOD
- Break triggers stop-covering cascade
- Creates rapid squeeze

Entry:
- At HOD break
- Or few cents below if seeing green tape (anticipation)

5-Second Rule: If doesn't break within 5 seconds of test, wait
Breakout or Bailout: 5 minute timeout rule

Target: 10-20-40 cents above HOD
Stop: Low of last 1-min candle
```

### Setup 5: VWAP Breakout (Red-to-Green)
```
Why It Works:
- Red-to-green creates 2-3x buying pressure:
  1. New long buyers
  2. Swing shorts covering
  3. Day trade shorts covering

Entry Options (safest to riskiest):
1. SAFEST: After break, wait for retest to hold VWAP as support
2. MODERATE: On the break itself
3. RISKY: Below VWAP anticipating break

Stop: Just below VWAP (tight)
Target: Always HOD first
Note: Can become parabolic on low float stocks
```

### Setup 6: Buying Into/Out of Halt
```
Circuit Breaker Halts (LULD):
- Requires 15 seconds at halt level to trigger
- FALSE HALT: Pinned 10-12 sec but drops before 15 sec

Gap Up Probability:
- Stocks halting UP typically gap higher on resumption
- Order imbalance bids up the stock
- Shorts get squeezed + FOMO buyers enter

Entry Strategy:
1. Take micro pullback before halt (safest)
2. Add into halt if already have starter position
3. Buy on resumption if gap up

Halt Resumption Rules:
- Halt DOWN opens FLAT → GO LONG (expected to gap lower)
- Halt UP opens FLAT (after 2-3 halts) → SHORT BIAS (exhaustion)

Stop: 25-50+ cents (bigger due to volatility)
Target: Next halt level
```

### Setup 7: Dip Buy (EXPERT LEVEL)
```
Prerequisites:
- NEVER first trade on a stock
- Must have prior profits (cushion)
- Stock is parabolic with big range
- Dips historically get bought up

Entry Trigger:
- Tape is RED, dropping on HIGH VOLUME
- See FIRST GREEN PRINT → BUY (shorts covering)
- "Irrational" flash drops of 50c-$1.00 → buy immediately

AVOID When:
- Drop due to BREAKING NEWS (will gap lower if halts)
- Stock below VWAP (weak)

OK to Trade:
- Panic drop (no news) on stock still above VWAP

Position Size: Start 1/10th of full size
Stop: 25-50 cents, or at the low of the dip
Target: Bounce 50% back to highs, or add for full squeeze
```

---

## Entry Mechanics

### Tape Reading Signals

```python
TAPE_SIGNALS = {
    'green_orders': {
        'meaning': 'Buying at ASK (aggressive buyers) or shorts covering',
        'action': 'BULLISH - confirms entry'
    },
    'red_orders': {
        'meaning': 'Selling at BID (aggressive sellers)',
        'action': 'BEARISH - wait for exhaustion'
    },
    'large_seller_thinning': {
        'pattern': 'Size decreasing: 25k → 19k → 15k → 11k → 5k',
        'meaning': 'Seller being absorbed, about to break',
        'action': 'PREPARE TO BUY'
    },
    'print_size_surge': {
        'meaning': 'Large block trades',
        'action': 'Institutional interest - follow the flow'
    }
}
```

### Entry Confirmation Checklist

```python
def confirm_entry(symbol, setup_type, quote, tape_data, candles):
    """Universal entry confirmation"""
    confirmations = []

    # 1. Price confirmation
    if setup_type in ['breakout', 'hod_break']:
        if quote['last'] > quote['day_high']:
            confirmations.append('PRICE_BREAK')

    # 2. Volume confirmation
    current_vol = candles[-1]['volume']
    avg_vol = sum(c['volume'] for c in candles[-5:-1]) / 4
    if current_vol > avg_vol * 1.5:
        confirmations.append('VOLUME_SURGE')

    # 3. Tape confirmation
    green_volume = sum(t['size'] for t in tape_data[-20:] if t['side'] == 'buy')
    red_volume = sum(t['size'] for t in tape_data[-20:] if t['side'] == 'sell')
    if green_volume > red_volume:
        confirmations.append('TAPE_GREEN')

    # 4. Spread confirmation
    spread_pct = (quote['ask'] - quote['bid']) / quote['bid'] * 100
    if spread_pct < 1.0:
        confirmations.append('SPREAD_OK')

    return {
        'confirmed': len(confirmations) >= 3,
        'signals': confirmations,
        'missing': 4 - len(confirmations)
    }
```

---

## Exit Strategies

### Exit Decision Tree

```
                    POSITION OPEN
                         │
                    ┌────┴────┐
                    │         │
              PROFITABLE?   LOSING?
                    │         │
            ┌───────┴───────┐ │
            │               │ │
      HIT TARGET?     TRAILING  STOP HIT?
            │               │         │
           YES             NO        YES
            │               │         │
    SCALE OUT 50%    ACTIVATE    EXIT 100%
            │         TRAIL
            │           │
    TRAIL REST    1.5% FROM HIGH
            │           │
    EXIT ON REVERSAL    │
            │           │
            └───────────┘
```

### Exit Rules by Setup

| Setup | Target 1 | Target 2 | Stop Logic |
|-------|----------|----------|------------|
| Bull Flag | HOD | HOD + flag height | Low of flag |
| ABCD | 25-50c or HOD | Next resistance | Below Point 3 |
| Micro Pullback | 10-20-40c | Halt level | Low of pullback |
| HOD Break | 10-20-40c | Halt level | Low of last candle |
| VWAP Break | HOD | New highs | Below VWAP |
| Halt Trade | Gap continuation | Next halt | 25-50c |
| Dip Buy | 50% retrace | Full retrace | Low of dip |

### Failed Momentum Exit (Ross Cameron Rule)

```python
def check_failed_momentum(entry_time, entry_price, current_price, elapsed_seconds=30):
    """
    If stock doesn't gain 0.5% within 30 seconds of entry → EXIT
    Momentum has failed - don't wait for stop
    """
    expected_gain = 0.005  # 0.5%
    actual_gain = (current_price - entry_price) / entry_price

    if elapsed_seconds >= 30 and actual_gain < expected_gain:
        return {
            'exit': True,
            'reason': 'FAILED_MOMENTUM',
            'note': 'No follow-through in 30 seconds'
        }

    return {'exit': False}
```

### Breakout or Bailout Rule

```python
def check_breakout_or_bailout(entry_time, entry_price, current_price, elapsed_minutes=5):
    """
    5 minute timeout: If no breakout after 5 minutes → EXIT
    Don't let a trade become a bag hold
    """
    if elapsed_minutes >= 5:
        pnl_pct = (current_price - entry_price) / entry_price * 100

        if pnl_pct < 1.0:  # Less than 1% gain
            return {
                'exit': True,
                'reason': 'TIMEOUT_NO_BREAKOUT',
                'note': '5 minutes elapsed without breakout'
            }

    return {'exit': False}
```

---

## Tape Reading

### Level 2 Analysis

```python
def analyze_level2(level2_data):
    """
    Analyze bid/ask stacking for entry/exit decisions
    """
    bid_total = sum(level['size'] for level in level2_data['bids'][:5])
    ask_total = sum(level['size'] for level in level2_data['asks'][:5])

    imbalance = bid_total / (bid_total + ask_total)

    if imbalance > 0.6:
        return {'bias': 'BULLISH', 'strength': imbalance}
    elif imbalance < 0.4:
        return {'bias': 'BEARISH', 'strength': 1 - imbalance}
    else:
        return {'bias': 'NEUTRAL', 'strength': 0.5}

def detect_large_seller_thinning(level2_history, ask_level):
    """
    Watch for large seller being absorbed
    Pattern: 25k → 19k → 15k → 11k → 5k → GONE
    """
    sizes = [snap['asks'][0]['size'] for snap in level2_history[-5:]]

    # Check if consistently decreasing
    if all(sizes[i] > sizes[i+1] for i in range(len(sizes)-1)):
        if sizes[-1] < sizes[0] * 0.3:  # Down 70%+
            return {
                'signal': 'SELLER_THINNING',
                'action': 'PREPARE_TO_BUY',
                'original_size': sizes[0],
                'current_size': sizes[-1]
            }

    return {'signal': None}
```

---

## Halt Trading

### LULD Halt Level Calculation

```python
def calculate_halt_levels(reference_price, tier):
    """
    LULD (Limit Up/Limit Down) bands
    Tier 1: S&P 500, Russell 1000, some ETFs
    Tier 2: Everything else
    """
    if tier == 1:
        band = 0.05 if reference_price > 3.00 else 0.10
    else:  # Tier 2
        if reference_price > 3.00:
            band = 0.10
        elif reference_price >= 0.75:
            band = 0.20
        else:
            band = min(0.75, reference_price * 0.20)

    return {
        'upper_halt': reference_price * (1 + band),
        'lower_halt': reference_price * (1 - band),
        'band_pct': band * 100
    }

def detect_approaching_halt(current_price, halt_level, direction='UP'):
    """
    Alert when approaching halt level
    """
    distance_pct = abs(halt_level - current_price) / current_price * 100

    if distance_pct < 2:
        return {
            'alert': True,
            'distance_pct': distance_pct,
            'halt_level': halt_level,
            'direction': direction,
            'action': 'PREPARE_FOR_HALT'
        }

    return {'alert': False, 'distance_pct': distance_pct}
```

### False Halt Detection

```python
def detect_false_halt(price_history, halt_level, tolerance=0.005):
    """
    False halt: Price pins near halt for 10-12 seconds but not full 15s
    Watch for sudden drop on false halt UP (large sellers)
    """
    pinned_seconds = 0

    for i, price in enumerate(price_history[-20:]):  # Last 20 seconds
        if abs(price - halt_level) / halt_level < tolerance:
            pinned_seconds += 1
        else:
            pinned_seconds = 0  # Reset if breaks away

    if 10 <= pinned_seconds < 15:
        return {
            'status': 'FALSE_HALT_RISK',
            'pinned_seconds': pinned_seconds,
            'action': 'REDUCE_SIZE_OR_WAIT'
        }
    elif pinned_seconds >= 15:
        return {'status': 'HALTED'}

    return {'status': 'NORMAL'}
```

---

## Implementation Priorities

### Phase 1: Core Scanner Enhancement
1. **Ross Cameron 5-Criteria Grading**
   - Integrate into `top_gappers_scanner.py`
   - Add grade display to dashboard
   - Filter watchlist by grade

2. **Pattern Detection**
   - Bull flag detection
   - ABCD pattern detection
   - HOD break detection

### Phase 2: Entry Signal Generation
1. **Tape Analysis Integration**
   - Green/red flow ratio
   - Large seller thinning detector
   - First green print trigger

2. **Setup-Specific Entries**
   - VWAP breakout detector
   - Micro pullback detector (15-sec chart)
   - HOD approach alerts

### Phase 3: Exit Management
1. **Failed Momentum Exit**
   - 30-second momentum check
   - Breakout or bailout (5-min timeout)

2. **Dynamic Targets**
   - Setup-specific target calculation
   - Half/whole dollar awareness

### Phase 4: Halt Trading
1. **LULD Level Calculator**
   - Real-time halt level display
   - Approaching halt alerts

2. **Halt Resumption Handler**
   - Gap prediction
   - Flat open detection (contrarian signal)

### Phase 5: Advanced Features
1. **Dip Buy Scanner**
   - Irrational flush detection
   - Support level confluence

2. **Market Condition Filter**
   - Halt reversal counter
   - "Front runner" identification

---

## Quick Reference: Setup Cheat Sheet

| If You See... | Setup | Entry | Stop | Target |
|---------------|-------|-------|------|--------|
| Flag consolidation after spike | Bull Flag | Break of high | Flag low | HOD + height |
| Failed break, higher low | ABCD | At higher low or break | Below HL | HOD |
| Parabolic with tiny dip | Micro PB | On tape green | Dip low | 10-40c |
| Testing HOD | HOD Break | At/before break | Last candle | 10-40c |
| Below VWAP going green | VWAP Break | Retest hold | Below VWAP | HOD |
| Squeezing into halt | Halt Trade | Into halt or resume | 25-50c | Next halt |
| Flash crash on strong stock | Dip Buy | First green print | Dip low | 50% retrace |

---

## Files to Update

| File | Enhancement |
|------|-------------|
| `ai/top_gappers_scanner.py` | Already has criteria - enhance with setup detection |
| `ai/hft_scalper.py` | Add setup-specific entries |
| `ai/hod_momentum_scanner.py` | Add HOD break detection |
| `ai/halt_detector.py` | Add LULD calculation, false halt detection |
| `ai/chronos_exit_manager.py` | Add failed momentum, breakout or bailout |
| `ai/tape_analyzer.py` | NEW - tape reading module |
| `ai/pattern_detector.py` | NEW - flag, ABCD, micro PB detection |
| `ai/setup_classifier.py` | NEW - classify current setup type |

---

*Generated from Warrior Trading Course Analysis - December 2024*
