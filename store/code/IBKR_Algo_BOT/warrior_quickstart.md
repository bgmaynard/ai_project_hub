# 🔥 Warrior Trading Scanner - Quick Start

**Pre-Market Gap-and-Go Momentum Trading**  
**Based on Ross Cameron's Methodology**

---

## 🎯 What This Does

Scans for **pre-market gappers** (small caps with 2%+ gaps) around **7:00 AM ET** and identifies high-probability Ross Cameron style setups for:

- Gap-and-go plays
- Breakout entries
- Momentum continuation
- 70%+ target win rate

**This is completely different from your MTF swing trading!**

---

## ⚡ Quick Start (3 Steps)

### Step 1: Run the Scanner (7:00 AM ET)

```bash
python warrior_momentum_scanner.py --start-time 07:00 --continuous
```

### Step 2: Review Watchlist

Scanner displays:
```
╔══════════════════════════════════════════════════════════════╗
║                  WARRIOR TRADING WATCHLIST                    ║
╚══════════════════════════════════════════════════════════════╝

Rank   Symbol   Price      Gap%       Score    Status
----------------------------------------------------------
1      ABCD     $8.45      +5.3%      85       🔥 HOT
2      WXYZ     $12.30     +4.2%      78       ✅ STRONG
3      DEFG     $6.75      +3.8%      72       ✅ STRONG
4      HIJK     $15.20     +2.9%      65       ✓ GOOD
```

### Step 3: Trade the Best Setups

Focus on **score 80+** (🔥 HOT)

Scanner shows entry/stop/target:
```
TRADE SIGNAL: ABCD
=====================================
Score: 85/100
Entry: $8.45
Stop: $8.20 (2.9% risk)
Target: $8.95 (2R profit)
Shares: 243
Risk: $60.75
Reward: $121.50

Reasons:
  ✅ Strong gap: 5.3%
  ✅ Near high of day
  ✅ Tight spread: 0.3%
  ✅ Above pre-market high (breakout!)
  ✅ Ideal price range: $8.45
```

---

## 📊 Warrior vs MTF Trading

| Feature | Warrior Momentum | MTF Swing |
|---------|------------------|-----------|
| **Time** | 7:00-10:30 AM | 9:30 AM - 4:00 PM |
| **Symbols** | Small caps ($1-$20) | Large caps (AAPL, TSLA) |
| **Holding** | 5-45 minutes | 2-24 hours |
| **Trades/Day** | 5-15 | 2-6 |
| **Win Rate** | 70%+ target | 60%+ |
| **Style** | Scalp/Momentum | Position/Swing |
| **Edge** | Pre-market gaps | MTF alignment |

**You can run BOTH!** Different times = no conflict!

---

## 🔥 Warrior Trading Rules (Ross Cameron Style)

### Entry Criteria

**All must be TRUE:**
1. ✅ Gap 2%+ (preferably 3-5%+)
2. ✅ Relative volume 3x+ average
3. ✅ Price $1-$20 (small caps)
4. ✅ Tight spread (<2%)
5. ✅ Breaking above pre-market high
6. ✅ Strong score (80+ for best setups)

### Position Sizing

```python
Risk 2% per trade
Position Size = (Account × 2%) / Stop Distance

Example:
Account: $100,000
Risk: $2,000 (2%)
Stop Distance: $0.25
Shares: 8,000 shares (but capped at max position size)
```

### Stop Loss

**Tight stops!** (Ross Cameron emphasizes this)

- Below pre-market low
- Or below recent support
- Typically 2-5% risk
- **Cut losses immediately if stop hit!**

### Profit Target

**2R minimum** (Ross style)
- If risk = $0.25, target = $0.50 gain
- Take partial profits at 1R
- Let remainder run to 2-3R

### Max Hold Time

**45 minutes maximum**
- Momentum fades after opening hour
- If not working in 45 min → Exit
- Most Warrior trades done by 10:30 AM

---

## 📈 Expected Performance (Ross Cameron Stats)

**Ross Cameron's public results:**
- **Win Rate:** 69% (100 trades in one month)
- **Account Growth:** $583 → $12.6M (2017-2024)
- **Style:** High win rate, tight risk control
- **Average:** Losses slightly larger than wins (need high win rate!)

**Your target with this scanner:**
- Win Rate: 70%+
- Risk/Reward: 1:2 (2R targets)
- Trades: 5-15 per day
- Focus: First 90 minutes of trading

---

## ⏰ Daily Trading Schedule

### 6:45 AM ET
- Open TWS/Gateway
- Start scanner in continuous mode

### 7:00 AM ET
- Scanner finds gappers
- Review watchlist
- Identify 3-5 best setups

### 7:00-9:30 AM (Pre-Market)
- Watch pre-market action
- Look for volume confirmation
- Prepare entry orders

### 9:30 AM (Market Open)
- **GO TIME!**
- Watch for breakouts above pre-market highs
- Enter on breakout with volume
- Set stops immediately

### 9:30-10:30 AM (Power Hour)
- Most active trading
- Take profits at targets
- Cut losses at stops
- Most trades complete here

### 10:30 AM
- Close remaining positions
- Review results
- **Done for the day!**

---

## 🎯 Warrior Scanner Features

### Automatic Gap Detection

Finds stocks with:
- 2%+ pre-market gap
- High relative volume
- Small-mid cap ($50M - $2B)
- Clean price action

### Warrior Scoring System

**Score Components (0-100):**

1. **Gap Strength (0-30 points)**
   - 5%+ gap = 30 points
   - 3-5% gap = 20 points
   - 2-3% gap = 10 points

2. **Price Position (0-20 points)**
   - Near HOD (high of day) = 20 points
   - Upper range = 15 points
   - Mid-range = 10 points

3. **Spread Quality (0-20 points)**
   - <0.5% spread = 20 points
   - <1% spread = 15 points
   - <2% spread = 5 points

4. **Price Level (0-15 points)**
   - $2-$10 ideal = 15 points
   - $1-$20 acceptable = 10 points

5. **Breakout Status (0-15 points)**
   - Above pre-market high = 15 points

**Total Score:**
- 80-100 = 🔥 HOT (trade these!)
- 70-79 = ✅ STRONG (good setups)
- 60-69 = ✓ GOOD (watch)
- <60 = Skip

### Auto-Calculated Entry/Stop/Target

For each signal:
```
Entry: Current ask price
Stop: Below pre-market low (-2% safety buffer)
Target: Entry + (2 × stop distance)
Shares: Risk $2k / stop distance
```

---

## 🛠️ Command Line Options

### Basic Scan (Single Run)
```bash
python warrior_momentum_scanner.py
```

### Continuous Scanning
```bash
python warrior_momentum_scanner.py --continuous --interval 180
```
Rescans every 3 minutes (180 seconds)

### Start at Specific Time
```bash
python warrior_momentum_scanner.py --start-time 07:00
```

### Custom Port (Live Trading)
```bash
python warrior_momentum_scanner.py --port 7496
```

### All Options Combined
```bash
python warrior_momentum_scanner.py \
    --start-time 07:00 \
    --continuous \
    --interval 180 \
    --port 7497
```

---

## 📋 Pre-Flight Checklist

Before trading Warrior style:

### Scanner Setup
- [ ] TWS/Gateway running (7:00 AM)
- [ ] Scanner connected to IBKR
- [ ] Continuous mode enabled
- [ ] Watchlist updating

### Risk Management
- [ ] Max risk 2% per trade
- [ ] Position sizing calculator ready
- [ ] Stops calculated for each trade
- [ ] Max 3-5 positions simultaneously

### Execution Ready
- [ ] Hot keys set in TWS
- [ ] Level 2 window open
- [ ] Time & Sales visible
- [ ] Ready to execute fast

### Mental Preparation
- [ ] Trade only 80+ score setups
- [ ] Cut losses immediately at stop
- [ ] Take profits at 2R target
- [ ] Done by 10:30 AM

---

## ⚠️ Warrior Trading Warnings

### 1. Pre-Market Spreads
Spreads are WIDE pre-market!
- Check spread before every trade
- Use limit orders, not market
- Factor spread into risk calculation

### 2. Momentum Fades Fast
- Best setups: 9:30-10:00 AM
- Good setups: 10:00-10:30 AM
- Avoid after 10:30 AM

### 3. Small Caps Are Volatile
- Price can move 10%+ in minutes
- Stops are CRITICAL
- Never hold overnight
- Risk management is everything

### 4. High Win Rate Required
Ross's stats: Losses > Wins in size
- NEED 70%+ win rate to profit
- Can't afford 50% win rate
- Discipline on entries is key

### 5. Execution Speed Matters
- Warrior trading is fast-paced
- Need to act quickly
- Practice with paper trading first
- Hot keys are essential

---

## 💡 Pro Tips (Ross Cameron Style)

### 1. Wait for the Breakout
Don't chase! Wait for:
- Clear break above pre-market high
- Volume confirmation (spike)
- Clean candle (not wick/doji)

### 2. Take Partials
Ross's approach:
- Sell 1/2 at 1R (lock profit)
- Move stop to breakeven
- Let 1/2 run to 2-3R

### 3. Cut Losses Fast
If trade goes against you:
- Don't hope it comes back
- Hit the stop immediately
- Move to next setup

### 4. Focus on Best Setups
Don't trade everything:
- Trade only 80+ scores
- Skip marginal setups
- Quality > Quantity

### 5. Review Daily
After each session:
- What worked?
- What didn't?
- Win rate for day?
- Adjust for tomorrow

---

## 🎓 Learning Resources

### Ross Cameron / Warrior Trading
- **YouTube:** Warrior Trading channel
- **Website:** warriortrading.com
- **Courses:** Day Trading Course, Simulator

### Your Scanner
- **Code:** `warrior_momentum_scanner.py`
- **Docs:** This guide
- **Support:** Built by your AI assistant

---

## 🚀 Ready to Scan?

### Option 1: Run Scanner Now
```bash
python warrior_momentum_scanner.py --continuous
```

### Option 2: Use Strategy Selector
```bash
python strategy_selector.py --strategy warrior
```

### Option 3: Run Both Strategies
**Terminal 1 (7:00 AM):**
```bash
python strategy_selector.py --strategy warrior
```

**Terminal 2 (9:30 AM):**
```bash
python strategy_selector.py --strategy mtf
```

---

## 📊 Sample Output

```
🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥
WARRIOR TRADING MOMENTUM SCANNER
🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥

Time: 07:15:22
Target: Pre-market gappers (2%+ gap, high volume)
Style: Ross Cameron gap-and-go

==============================================
SCANNING FOR PRE-MARKET GAPPERS
==============================================

Scanning... (this takes 5-10 seconds)
Found: ABCD (rank 1)
Found: WXYZ (rank 2)
Found: DEFG (rank 3)
Found: HIJK (rank 4)
✓ Scanner found 4 candidates

Analyzing 4 candidates...

======================================
TRADE SIGNAL: ABCD
======================================
Score: 85/100
Entry: $8.45
Stop: $8.20
Target: $8.95
Shares: 243
Risk: $60.75
Reward: $121.50

Reasons:
  ✅ Strong gap: 5.3%
  ✅ Near high of day
  ✅ Tight spread: 0.3%
  ✅ Above pre-market high (breakout!)
  ✅ Ideal price range: $8.45
======================================

══════════════════════════════════════════════════════════
                  WARRIOR TRADING WATCHLIST
══════════════════════════════════════════════════════════
Time: 2025-10-12 07:15:45
Candidates found: 4
══════════════════════════════════════════════════════════

Rank   Symbol   Price      Gap%       Score    Status
--------------------------------------------------------------
1      ABCD     $8.45      +5.3%      85       🔥 HOT
2      WXYZ     $12.30     +4.2%      78       ✅ STRONG
3      DEFG     $6.75      +3.8%      72       ✅ STRONG
4      HIJK     $15.20     +2.9%      65       ✓ GOOD

══════════════════════════════════════════════════════════

Next scan in 180 seconds...
```

---

## ✅ Success Checklist

After 2 weeks of Warrior trading:

- [ ] Win rate > 70%
- [ ] Average R:R > 1:1.5
- [ ] Most trades done by 10:30 AM
- [ ] Comfortable with execution speed
- [ ] Following stops discipline
- [ ] Taking profits at targets
- [ ] Only trading 80+ scores

**All checked → You're a Warrior Trader!** 🔥

---

**Good hunting!** 🎯📈💰

*Remember: Ross turned $583 into $12.6M with this style. The edge is real!*
