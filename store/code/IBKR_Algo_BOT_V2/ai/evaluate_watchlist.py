"""
Evaluate Watchlist Candidates with Chronos
==========================================
Scans candidates and recommends only bullish stocks for tomorrow.
"""

import sys

sys.path.insert(0, "C:/ai_project_hub/store/code/IBKR_Algo_BOT_V2")

from dotenv import load_dotenv

load_dotenv("C:/ai_project_hub/store/code/IBKR_Algo_BOT_V2/.env")

import json
from pathlib import Path

import yfinance as yf
# Import Chronos
from ai.chronos_predictor import ChronosPredictor


def main():
    print("=" * 80)
    print("WATCHLIST EVALUATION FOR TOMORROW")
    print("=" * 80)

    # Candidates: recent movers + penny stocks with volume
    candidates = [
        # Recent movers from today/this week
        "YCBD",
        "EVTV",
        "LHAI",
        "ATHA",
        "ADTX",
        # Popular penny/momentum stocks
        "FFIE",
        "MULN",
        "SOUN",
        "PLTR",
        "SOFI",
        # Quantum/Tech movers
        "IONQ",
        "RGTI",
        "QBTS",
        "RKLB",
        # Crypto miners
        "MARA",
        "RIOT",
        "CLSK",
        "BITF",
        # EV plays
        "LCID",
        "RIVN",
        "NKLA",
        "GEVO",
        # Other momentum
        "NVDA",
        "TSLA",
        "AMD",
        "SMCI",
    ]

    print(f"Evaluating {len(candidates)} candidates...")
    print()

    chronos = ChronosPredictor()
    results = []

    for symbol in candidates:
        try:
            # Get basic info from yfinance
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="5d")

            if hist.empty:
                print(f"{symbol:<6} No data")
                continue

            price = hist["Close"].iloc[-1]
            volume = hist["Volume"].iloc[-1]

            # Get info for float
            info = ticker.info
            float_shares = info.get("floatShares", 0)
            avg_volume = info.get("averageVolume", volume)

            # Get Chronos prediction
            pred = chronos.predict(symbol, horizon=1)
            signal = pred.get("signal", "ERROR")
            prob_up = pred.get("probabilities", {}).get("prob_up", 0.5)
            exp_ret = pred.get("expected_return_pct", 0)

            # Calculate relative volume
            rel_vol = volume / avg_volume if avg_volume > 0 else 0

            # Float in millions
            float_m = float_shares / 1_000_000 if float_shares else 0

            results.append(
                {
                    "symbol": symbol,
                    "price": price,
                    "volume": volume,
                    "rel_vol": rel_vol,
                    "float_m": float_m,
                    "signal": signal,
                    "prob_up": prob_up,
                    "exp_ret": exp_ret,
                }
            )

            status = (
                "TRADE"
                if signal in ["BULLISH", "STRONG_BULLISH"]
                else "SKIP" if signal in ["BEARISH", "STRONG_BEARISH"] else "WATCH"
            )
            print(
                f"{symbol:<6} ${price:>8.2f} | {signal:<16} | prob={prob_up:>5.0%} | exp_ret={exp_ret:>+5.1f}% | {status}"
            )

        except Exception as e:
            print(f"{symbol:<6} ERROR: {str(e)[:50]}")

    # Sort by probability and expected return
    results.sort(key=lambda x: (x["prob_up"], x["exp_ret"]), reverse=True)

    print()
    print("=" * 80)
    print("RECOMMENDED WATCHLIST FOR TOMORROW")
    print("=" * 80)
    print()

    bullish = [r for r in results if r["signal"] in ["BULLISH", "STRONG_BULLISH"]]
    neutral = [r for r in results if r["signal"] == "NEUTRAL"]
    bearish = [r for r in results if r["signal"] in ["BEARISH", "STRONG_BEARISH"]]

    print(
        f"{'Symbol':<8} {'Price':>8} {'Signal':<16} {'Prob Up':>8} {'Exp Ret':>8} {'Float':>8}"
    )
    print("-" * 65)

    if bullish:
        print("--- TRADE THESE (Chronos Bullish) ---")
        for r in bullish[:10]:
            float_str = f"{r['float_m']:.1f}M" if r["float_m"] > 0 else "N/A"
            print(
                f"{r['symbol']:<8} ${r['price']:>7.2f} {r['signal']:<16} {r['prob_up']:>7.0%} {r['exp_ret']:>+7.1f}% {float_str:>8}"
            )
    else:
        print("No bullish candidates found!")

    if neutral:
        print()
        print("--- WATCH ONLY (Neutral) ---")
        for r in neutral[:5]:
            float_str = f"{r['float_m']:.1f}M" if r["float_m"] > 0 else "N/A"
            print(
                f"{r['symbol']:<8} ${r['price']:>7.2f} {r['signal']:<16} {r['prob_up']:>7.0%} {r['exp_ret']:>+7.1f}% {float_str:>8}"
            )

    print()
    print("--- AVOID (Bearish) ---")
    for r in bearish[:5]:
        print(f"{r['symbol']:<8} {r['signal']:<16} prob_up={r['prob_up']:>5.0%}")

    # Build final watchlist
    watch_symbols = [r["symbol"] for r in bullish[:8]]

    # Add some neutral with high prob if not enough bullish
    if len(watch_symbols) < 5:
        for r in neutral:
            if r["prob_up"] >= 0.6 and len(watch_symbols) < 8:
                watch_symbols.append(r["symbol"])

    print()
    print("=" * 80)
    print(f"FINAL WATCHLIST ({len(watch_symbols)} symbols):")
    print(watch_symbols)
    print("=" * 80)

    # Save to file
    watchlist_data = {
        "symbols": watch_symbols,
        "generated": "chronos_filtered",
        "bullish_count": len(bullish),
        "neutral_count": len(neutral),
        "bearish_count": len(bearish),
    }

    with open("ai/recommended_watchlist.json", "w") as f:
        json.dump(watchlist_data, f, indent=2)

    print(f"\nSaved to ai/recommended_watchlist.json")

    return watch_symbols


if __name__ == "__main__":
    watchlist = main()
