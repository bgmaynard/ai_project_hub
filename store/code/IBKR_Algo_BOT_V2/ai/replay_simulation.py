"""
Replay Simulation - Test Different Filter Settings on Historical Data
======================================================================
Simulates what trades would have triggered with different filter configurations.
"""

import json
import os
from datetime import datetime
from typing import Dict, List
import yfinance as yf

# Add parent path for imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_news_log(filepath: str = None) -> List[Dict]:
    """Load news log"""
    if filepath is None:
        filepath = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                "store/scanner/news_log.json")

    with open(filepath, 'r') as f:
        data = json.load(f)
    return data.get('news', [])


def get_price_data(symbol: str, date: str) -> Dict:
    """Get price data for symbol on date using daily bars"""
    try:
        ticker = yf.Ticker(symbol)
        # Use 5 day history to get prior close for gap calculation
        from datetime import datetime, timedelta
        dt = datetime.strptime(date, '%Y-%m-%d')
        start = (dt - timedelta(days=5)).strftime('%Y-%m-%d')
        end = (dt + timedelta(days=1)).strftime('%Y-%m-%d')

        hist = ticker.history(start=start, end=end, interval="1d")

        if hist.empty or len(hist) < 1:
            return {}

        # Find the row for our target date
        target_row = None
        prior_row = None
        for i, (idx, row) in enumerate(hist.iterrows()):
            row_date = idx.strftime('%Y-%m-%d')
            if row_date == date:
                target_row = row
                if i > 0:
                    prior_row = hist.iloc[i-1]
                break

        if target_row is None:
            # Use most recent if exact date not found
            target_row = hist.iloc[-1]
            if len(hist) > 1:
                prior_row = hist.iloc[-2]

        prior_close = prior_row['Close'] if prior_row is not None else target_row['Open']
        change_pct = ((target_row['Close'] - prior_close) / prior_close) * 100 if prior_close > 0 else 0

        return {
            'open': target_row['Open'],
            'high': target_row['High'],
            'low': target_row['Low'],
            'close': target_row['Close'],
            'prior_close': prior_close,
            'volume': int(target_row['Volume']),
            'change_pct': change_pct
        }
    except Exception as e:
        return {}


def simulate_gap_grade(symbol: str, price_data: Dict, headline: str = "") -> Dict:
    """Simulate gap grading"""
    from ai.gap_grader import get_gap_grader

    grader = get_gap_grader()

    current_price = price_data.get('close', 0)
    if current_price <= 0:
        return None

    # Use prior close if available, otherwise estimate from change percent
    gap_pct = price_data.get('change_pct', 0)
    prior_close = price_data.get('prior_close')
    if not prior_close:
        prior_close = current_price / (1 + gap_pct/100) if gap_pct else current_price

    graded = grader.grade_gap(
        symbol=symbol,
        gap_percent=gap_pct,
        current_price=current_price,
        prior_close=prior_close,
        premarket_volume=price_data.get('volume', 0),
        catalyst_headline=headline
    )

    has_catalyst = graded.catalyst_type.value != "NONE"
    return {
        'symbol': symbol,
        'grade': graded.grade.value,
        'score': graded.score.total,
        'gap_pct': gap_pct,
        'catalyst': graded.catalyst_type.value,
        'has_catalyst': has_catalyst,
        'price': current_price,
        'warnings': graded.warnings
    }


def run_replay(date: str = None, min_change_pct: float = 3.0):
    """
    Replay news data and simulate what would have triggered.

    Args:
        date: Date to replay (YYYY-MM-DD), defaults to today
        min_change_pct: Minimum price change to consider
    """
    from ai.low_float_momentum import check_low_float_momentum, MomentumSignal

    if date is None:
        date = datetime.now().strftime('%Y-%m-%d')

    news_entries = load_news_log()

    # Filter to date
    day_news = [n for n in news_entries if n.get('date') == date]
    print(f"\n{'='*60}")
    print(f"REPLAY SIMULATION - {date}")
    print(f"{'='*60}")
    print(f"News entries: {len(day_news)}")

    # Get unique symbols with headlines
    symbols_with_news = {}
    for n in day_news:
        sym = n.get('symbol')
        headline = n.get('headline', '')
        if sym and headline:
            if sym not in symbols_with_news:
                symbols_with_news[sym] = []
            symbols_with_news[sym].append(headline)

    print(f"Unique symbols with news: {len(symbols_with_news)}")

    # Analyze each symbol
    results = {
        'A_grade': [],
        'B_grade': [],
        'C_grade': [],
        'D_grade': [],
        'F_grade': [],
        'low_float_bypass': [],  # Low-float momentum plays that bypass filters
        'no_data': []
    }

    print(f"\nAnalyzing price data...")

    for i, (symbol, headlines) in enumerate(symbols_with_news.items()):
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(symbols_with_news)}...")

        price_data = get_price_data(symbol, date)

        if not price_data:
            results['no_data'].append(symbol)
            continue

        # Skip if change too small
        if abs(price_data.get('change_pct', 0)) < min_change_pct:
            continue

        # Check for low-float momentum bypass FIRST
        lf_analysis = check_low_float_momentum(
            symbol,
            current_price=price_data.get('close'),
            gap_percent=price_data.get('change_pct', 0)
        )

        if lf_analysis.signal in [MomentumSignal.STRONG, MomentumSignal.MODERATE]:
            # This is a low-float momentum play - add to bypass list
            results['low_float_bypass'].append({
                'symbol': symbol,
                'signal': lf_analysis.signal.value,
                'gap_pct': price_data.get('change_pct', 0),
                'float_m': lf_analysis.float_shares / 1e6,
                'volume_ratio': lf_analysis.volume_ratio,
                'float_rotation': lf_analysis.float_rotation,
                'price': price_data.get('close', 0),
                'reason': lf_analysis.reason
            })
            continue  # Skip gap grading for low-float momentum

        # Get best headline
        best_headline = max(headlines, key=len) if headlines else ""

        # Grade the gap
        graded = simulate_gap_grade(symbol, price_data, best_headline)

        if graded:
            grade = graded['grade']
            results[f'{grade}_grade'].append(graded)

    # Print results
    print(f"\n{'='*60}")
    print("RESULTS BY GRADE")
    print(f"{'='*60}")

    for grade in ['A', 'B', 'C', 'D', 'F']:
        grade_results = results[f'{grade}_grade']
        if grade_results:
            print(f"\n{grade} GRADE ({len(grade_results)} stocks):")
            # Sort by score descending
            grade_results.sort(key=lambda x: x['score'], reverse=True)
            for r in grade_results[:5]:  # Top 5
                catalyst_str = f" [{r['catalyst']}]" if r['has_catalyst'] else ""
                print(f"  {r['symbol']:6s} - Score: {r['score']:3d}, Gap: {r['gap_pct']:+6.1f}%{catalyst_str}")

    # Print low-float momentum bypass results
    if results['low_float_bypass']:
        print(f"\nLOW-FLOAT MOMENTUM BYPASS ({len(results['low_float_bypass'])} stocks):")
        # Sort by float rotation descending
        results['low_float_bypass'].sort(key=lambda x: x['float_rotation'], reverse=True)
        for r in results['low_float_bypass']:
            print(f"  {r['symbol']:6s} - {r['signal']:8s} | Gap {r['gap_pct']:+6.1f}% | "
                  f"Float {r['float_m']:.1f}M | {r['volume_ratio']:.0f}x vol | "
                  f"Rotation {r['float_rotation']:.0f}%")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"LOW-FLOAT BYPASS: {len(results['low_float_bypass'])}")
    print(f"A Grade (tradeable): {len(results['A_grade'])}")
    print(f"B Grade (tradeable): {len(results['B_grade'])}")
    print(f"C Grade (marginal):  {len(results['C_grade'])}")
    print(f"D Grade (watch only): {len(results['D_grade'])}")
    print(f"F Grade (avoid):     {len(results['F_grade'])}")
    print(f"No price data:       {len(results['no_data'])}")

    # Would-have-traded analysis
    print(f"\n{'='*60}")
    print("WOULD HAVE TRADED (with current filters + low-float bypass)")
    print(f"{'='*60}")

    # Low-float momentum bypasses ALWAYS trade
    if results['low_float_bypass']:
        print("\n  LOW-FLOAT MOMENTUM (bypasses all AI filters):")
        for r in results['low_float_bypass']:
            print(f"    {r['symbol']:6s} @ ${r['price']:.2f} | {r['signal']} | Gap {r['gap_pct']:+.1f}% | "
                  f"Float {r['float_m']:.1f}M | Rotation {r['float_rotation']:.0f}%")

    # A/B grades (with catalyst requirement)
    ab_trades = results['A_grade'] + results['B_grade']
    if ab_trades:
        print("\n  A/B GRADE (with catalyst):")
        ab_trades.sort(key=lambda x: x['score'], reverse=True)
        for r in ab_trades:
            catalyst_str = f"[{r['catalyst']}]" if r['has_catalyst'] else "[NO CATALYST]"
            print(f"    {r['symbol']:6s} @ ${r['price']:.2f} | Grade {r['grade']} ({r['score']}/100) | Gap {r['gap_pct']:+.1f}% | {catalyst_str}")

    if not results['low_float_bypass'] and not ab_trades:
        print("  No trades would have triggered")
        print("  (This is expected if no stocks had momentum or news catalysts)")

    return results


def simulate_state_machine(symbols: list = None, verbose: bool = True):
    """
    Simulate the momentum state machine on a list of symbols.
    Tests the new state machine architecture with real market data.

    Args:
        symbols: List of symbols to test (defaults to scalper watchlist)
        verbose: Print detailed output
    """
    from ai.momentum_state_machine import get_state_machine, MomentumState
    from ai.momentum_score import MomentumScorer

    if symbols is None:
        # Load from scalper config
        config_path = os.path.join(os.path.dirname(__file__), "scalper_config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            symbols = config.get('watchlist', [])[:5]  # Top 5 for speed
        else:
            symbols = ["AAPL", "TSLA", "NVDA"]

    print(f"\n{'='*60}")
    print("STATE MACHINE SIMULATION")
    print(f"{'='*60}")
    print(f"Testing symbols: {symbols}")

    sm = get_state_machine()
    scorer = MomentumScorer()

    results = {
        'tested': 0,
        'reached_ignition': [],
        'stuck_in_attention': [],
        'stuck_in_setup': [],
        'no_momentum': []
    }

    for symbol in symbols:
        print(f"\n--- {symbol} ---")
        results['tested'] += 1

        try:
            # Get historical prices for scoring
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1d", interval="1m")

            if len(hist) < 30:
                print(f"  Insufficient data ({len(hist)} bars)")
                results['no_momentum'].append(symbol)
                continue

            # Get current quote
            current_price = hist['Close'].iloc[-1]
            spread_pct = 0.1  # Estimated for simulation

            # Calculate momentum score
            prices_30s = hist['Close'].iloc[-6:].tolist()  # ~30s of 5s bars
            prices_60s = hist['Close'].iloc[-12:].tolist()
            prices_5m = hist['Close'].iloc[-60:].tolist() if len(hist) >= 60 else hist['Close'].tolist()
            current_volume = int(hist['Volume'].iloc[-1])

            score_result = scorer.calculate(
                symbol=symbol,
                current_price=current_price,
                prices_30s=prices_30s,
                prices_60s=prices_60s,
                prices_5m=prices_5m,
                current_volume=current_volume,
                spread_pct=spread_pct,
                buy_pressure=0.55,  # Neutral for simulation
                tape_signal="NEUTRAL"
            )

            if verbose:
                print(f"  Price: ${current_price:.2f}")
                print(f"  Momentum Score: {score_result.score}/100 (Grade {score_result.grade.value})")
                print(f"    - Price Urgency: {score_result.price_urgency.total:.0f}/40")
                print(f"    - Participation: {score_result.participation.total:.0f}/35")
                print(f"    - Liquidity: {score_result.liquidity.total:.0f}/25")
                print(f"  Tradeable: {score_result.is_tradeable}, Ignition Ready: {score_result.ignition_ready}")

            # Simulate state machine transitions
            details = {
                "grade": score_result.grade.value,
                "price_urgency": score_result.price_urgency.total,
                "participation": score_result.participation.total,
                "liquidity": score_result.liquidity.total,
                "tradeable": score_result.is_tradeable,
                "ignition_ready": score_result.ignition_ready
            }

            new_state = sm.update_momentum(symbol, score_result.score, details)

            if new_state:
                state_obj = sm.get_state(symbol)
                final_state = state_obj.state if state_obj else MomentumState.IDLE
                print(f"  State: {final_state.value}")

                if final_state == MomentumState.IGNITION:
                    results['reached_ignition'].append(symbol)
                    print(f"  --> IGNITION TRIGGERED! Ready to enter.")
                elif final_state == MomentumState.SETUP:
                    results['stuck_in_setup'].append(symbol)
                elif final_state == MomentumState.ATTENTION:
                    results['stuck_in_attention'].append(symbol)
                else:
                    results['no_momentum'].append(symbol)
            else:
                results['no_momentum'].append(symbol)
                print(f"  State: IDLE (no momentum)")

        except Exception as e:
            print(f"  Error: {e}")
            results['no_momentum'].append(symbol)

    # Summary
    print(f"\n{'='*60}")
    print("STATE MACHINE SIMULATION SUMMARY")
    print(f"{'='*60}")
    print(f"Symbols tested: {results['tested']}")
    print(f"Reached IGNITION: {len(results['reached_ignition'])} - {results['reached_ignition']}")
    print(f"In SETUP: {len(results['stuck_in_setup'])} - {results['stuck_in_setup']}")
    print(f"In ATTENTION: {len(results['stuck_in_attention'])} - {results['stuck_in_attention']}")
    print(f"No momentum: {len(results['no_momentum'])} - {results['no_momentum']}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Replay simulation")
    parser.add_argument("--date", "-d", default=None,
                        help="Date to replay (YYYY-MM-DD)")
    parser.add_argument("--min-change", "-m", type=float, default=3.0,
                        help="Minimum price change percent")
    parser.add_argument("--state-machine", "-s", action="store_true",
                        help="Run state machine simulation instead")
    parser.add_argument("--symbols", nargs="*", default=None,
                        help="Symbols to test (for state machine mode)")

    args = parser.parse_args()

    if args.state_machine:
        simulate_state_machine(symbols=args.symbols)
    else:
        run_replay(date=args.date, min_change_pct=args.min_change)
