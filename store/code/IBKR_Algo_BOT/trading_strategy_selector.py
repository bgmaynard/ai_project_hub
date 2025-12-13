"""
Trading Strategy Selector
=========================

Easily switch between different trading strategies:
1. MTF Swing Trading (AAPL/TSLA, 9:30 AM - 4:00 PM)
2. Warrior Momentum (Small caps, 7:00 AM - 10:30 AM)

Usage:
    python strategy_selector.py --strategy warrior
    python strategy_selector.py --strategy mtf
    python strategy_selector.py --list
"""

import argparse
import subprocess


STRATEGIES = {
    'warrior': {
        'name': 'Warrior Trading Momentum',
        'description': 'Pre-market gap-and-go, small caps, Ross Cameron style',
        'hours': '7:00 AM - 10:30 AM ET',
        'symbols': 'Scanner finds gappers (2%+ gap, high volume)',
        'style': 'Scalp/Day trade (5-45 min holds)',
        'win_rate_target': '70%+',
        'script': 'warrior_momentum_scanner.py',
        'default_args': '--start-time 07:00 --continuous --interval 180'
    },
    'mtf': {
        'name': 'MTF Swing Trading',
        'description': 'Multi-timeframe LSTM models, large caps',
        'hours': '9:30 AM - 4:00 PM ET',
        'symbols': 'AAPL, TSLA (configurable)',
        'style': 'Position trade (2-24 hour holds)',
        'win_rate_target': '60%+',
        'script': 'ibkr_live_trading_connector.py',
        'default_args': '--mode paper --symbols AAPL TSLA --interval 300'
    }
}


def print_strategies():
    """Print all available strategies"""
    
    print("\n" + "="*80)
    print("AVAILABLE TRADING STRATEGIES")
    print("="*80 + "\n")
    
    for key, strategy in STRATEGIES.items():
        print(f"Strategy: {key.upper()}")
        print("-" * 80)
        print(f"Name:        {strategy['name']}")
        print(f"Description: {strategy['description']}")
        print(f"Hours:       {strategy['hours']}")
        print(f"Symbols:     {strategy['symbols']}")
        print(f"Style:       {strategy['style']}")
        print(f"Win Rate:    {strategy['win_rate_target']}")
        print(f"Script:      {strategy['script']}")
        print("\nTo run:")
        print(f"  python strategy_selector.py --strategy {key}")
        print("\n" + "="*80 + "\n")


def get_strategy_comparison():
    """Return detailed comparison"""
    
    comparison = """
    
╔══════════════════════════════════════════════════════════════════════════════╗
║                         STRATEGY COMPARISON                                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

┌──────────────────────┬─────────────────────────┬─────────────────────────────┐
│ Feature              │ Warrior Momentum        │ MTF Swing Trading           │
├──────────────────────┼─────────────────────────┼─────────────────────────────┤
│ Trading Hours        │ 7:00 AM - 10:30 AM      │ 9:30 AM - 4:00 PM           │
│ Market Session       │ Pre-market + Opening    │ Regular hours only          │
│ Symbols              │ Small caps (<$20)       │ Large caps (AAPL, TSLA)     │
│ Position Duration    │ 5-45 minutes            │ 2-24 hours                  │
│ Trades per Day       │ 5-15 trades             │ 2-6 trades                  │
│ Win Rate Target      │ 70%+                    │ 60%+                        │
│ Risk per Trade       │ 2% (tight stops)        │ 1-2% (wider stops)          │
│ Profit Target        │ 2R (quick scalps)       │ 1.5-3R (let it run)         │
│ Entry Trigger        │ Gap + Breakout          │ LSTM confidence > 52%       │
│ Style                │ Momentum/Scalp          │ Trend/Position              │
│ Edge                 │ Pre-market gaps         │ Multi-timeframe alignment   │
│ Backtested          │ Ross Cameron: 69% WR    │ Your models: 60-62% WR      │
│ Account Type         │ Margin preferred        │ Cash or Margin OK           │
│ Watching Required    │ CONSTANT (first hour)   │ Periodic checks             │
└──────────────────────┴─────────────────────────┴─────────────────────────────┘


📊 WHICH TO CHOOSE?

Use WARRIOR if you:
✅ Can trade 7:00-10:30 AM ET actively
✅ Like fast-paced, active trading
✅ Want more trades per day
✅ Comfortable with small caps
✅ Can watch screens constantly
✅ Prefer tight stops, quick exits

Use MTF if you:
✅ Trade regular market hours (9:30 AM - 4:00 PM)
✅ Prefer position trading over scalping
✅ Like large, liquid stocks
✅ Want to check trades periodically, not constantly
✅ Prefer letting winners run
✅ Have day job (can check every 30-60 min)


💡 RECOMMENDATION:

Run BOTH strategies simultaneously:
- 7:00-10:30 AM: Warrior momentum scanner
- 9:30 AM-4:00 PM: MTF swing trading

Different timeframes = No conflicts!
Diversified approach = Better overall returns!

"""
    
    return comparison


def run_strategy(strategy_name, custom_args=None):
    """Run selected strategy"""
    
    if strategy_name not in STRATEGIES:
        print(f"Error: Unknown strategy '{strategy_name}'")
        print("Available strategies: " + ", ".join(STRATEGIES.keys()))
        return
    
    strategy = STRATEGIES[strategy_name]
    
    print("\n" + "="*80)
    print(f"LAUNCHING: {strategy['name'].upper()}")
    print("="*80)
    print(f"Description: {strategy['description']}")
    print(f"Hours:       {strategy['hours']}")
    print(f"Target:      {strategy['win_rate_target']} win rate")
    print("="*80 + "\n")
    
    # Build command
    script = strategy['script']
    
    if custom_args:
        args = custom_args
    else:
        args = strategy['default_args']
    
    command = f"python {script} {args}"
    
    print(f"Command: {command}\n")
    print("Starting in 3 seconds...")
    print("Press Ctrl+C to stop\n")
    
    import time
    time.sleep(3)
    
    # Run strategy
    try:
        subprocess.run(command, shell=True)
    except KeyboardInterrupt:
        print("\n\nStrategy stopped by user")


def run_both_strategies():
    """Run both strategies simultaneously"""
    
    print("\n" + "="*80)
    print("RUNNING BOTH STRATEGIES SIMULTANEOUSLY")
    print("="*80)
    print("\nWarrior Momentum: 7:00 AM - 10:30 AM (Terminal 1)")
    print("MTF Swing:        9:30 AM - 4:00 PM (Terminal 2)")
    print("\nYou need to run each in a separate terminal window:")
    print("\nTerminal 1:")
    print("  python strategy_selector.py --strategy warrior")
    print("\nTerminal 2:")
    print("  python strategy_selector.py --strategy mtf")
    print("\n" + "="*80 + "\n")


def main():
    """Main selector"""
    
    parser = argparse.ArgumentParser(
        description='Select and run trading strategy',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--strategy', type=str, choices=['warrior', 'mtf', 'both'],
                       help='Strategy to run')
    parser.add_argument('--list', action='store_true',
                       help='List all available strategies')
    parser.add_argument('--compare', action='store_true',
                       help='Compare strategies')
    parser.add_argument('--args', type=str, default=None,
                       help='Custom arguments for strategy')
    
    args = parser.parse_args()
    
    # Show header
    print("\n" + "🎯"*40)
    print("TRADING STRATEGY SELECTOR")
    print("🎯"*40)
    
    if args.list:
        print_strategies()
    
    elif args.compare:
        print(get_strategy_comparison())
    
    elif args.strategy:
        if args.strategy == 'both':
            run_both_strategies()
        else:
            run_strategy(args.strategy, args.args)
    
    else:
        # Interactive mode
        print("\n" + "="*80)
        print("SELECT STRATEGY")
        print("="*80 + "\n")
        
        print("Available strategies:\n")
        for i, (key, strategy) in enumerate(STRATEGIES.items(), 1):
            print(f"{i}. {key.upper()}: {strategy['name']}")
            print(f"   {strategy['description']}")
            print(f"   Hours: {strategy['hours']}\n")
        
        print("3. BOTH: Run both strategies (separate terminals)")
        print("4. COMPARE: Show detailed comparison")
        print("5. EXIT\n")
        
        choice = input("Enter choice (1-5): ")
        
        if choice == '1':
            run_strategy('warrior')
        elif choice == '2':
            run_strategy('mtf')
        elif choice == '3':
            run_both_strategies()
        elif choice == '4':
            print(get_strategy_comparison())
        elif choice == '5':
            print("Exiting...")
        else:
            print("Invalid choice")


if __name__ == "__main__":
    main()
