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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Replay simulation")
    parser.add_argument("--date", "-d", default=None,
                        help="Date to replay (YYYY-MM-DD)")
    parser.add_argument("--min-change", "-m", type=float, default=3.0,
                        help="Minimum price change percent")

    args = parser.parse_args()

    run_replay(date=args.date, min_change_pct=args.min_change)
