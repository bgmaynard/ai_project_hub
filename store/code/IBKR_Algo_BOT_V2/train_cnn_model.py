#!/usr/bin/env python
"""
CNN Stock Predictor Training Script
====================================
Train the CNN model for stock profile prediction.

Usage:
    python train_cnn_model.py --symbols SPY,QQQ,AAPL,MSFT --days 365 --epochs 50
    python train_cnn_model.py --momentum  # Train on momentum stocks
    python train_cnn_model.py --all       # Train on full universe
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("cnn_training.log"),
    ],
)
logger = logging.getLogger(__name__)


# Default symbol lists
MOMENTUM_STOCKS = [
    "SPY",
    "QQQ",
    "IWM",  # ETFs
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "NVDA",
    "META",
    "TSLA",  # Tech giants
    "AMD",
    "NFLX",
    "CRM",
    "ADBE",
    "INTC",  # More tech
    "JPM",
    "BAC",
    "GS",
    "V",
    "MA",  # Financials
]

SMALL_CAP_MOMENTUM = [
    "PLTR",
    "SOFI",
    "LCID",
    "RIVN",
    "HOOD",
    "UPST",
    "AFRM",
    "COIN",
    "MARA",
    "RIOT",
    "IONQ",
    "QUBT",
    "RGTI",  # Quantum
]

FULL_UNIVERSE = (
    MOMENTUM_STOCKS
    + SMALL_CAP_MOMENTUM
    + [
        "XOM",
        "CVX",
        "COP",  # Energy
        "PFE",
        "JNJ",
        "UNH",
        "MRNA",
        "ABBV",  # Healthcare
        "WMT",
        "COST",
        "TGT",
        "HD",
        "LOW",  # Retail
        "DIS",
        "CMCSA",
        "T",
        "VZ",  # Media/Telecom
        "CAT",
        "DE",
        "BA",
        "LMT",
        "RTX",  # Industrial
    ]
)


def train_model(symbols: list, days: int, epochs: int, batch_size: int):
    """Train the CNN model"""
    from ai.cnn_stock_predictor import get_cnn_predictor

    logger.info("=" * 60)
    logger.info("CNN STOCK PREDICTOR TRAINING")
    logger.info("=" * 60)
    logger.info(f"Symbols: {len(symbols)}")
    logger.info(f"Days of data: {days}")
    logger.info(f"Epochs: {epochs}")
    logger.info(f"Batch size: {batch_size}")
    logger.info("=" * 60)

    predictor = get_cnn_predictor()

    # Train
    start_time = datetime.now()
    results = predictor.train(
        symbols=symbols, days=days, epochs=epochs, batch_size=batch_size
    )
    duration = (datetime.now() - start_time).total_seconds()

    # Print results
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Duration: {duration:.1f} seconds")
    logger.info(f"Epochs Trained: {results['epochs_trained']}")
    logger.info(f"Best Validation Loss: {results['best_val_loss']:.4f}")

    if results["metrics"]:
        logger.info("\nMETRICS:")
        logger.info(f"  Accuracy: {results['metrics']['accuracy']:.4f}")
        logger.info(f"  Precision: {results['metrics']['precision']:.4f}")
        logger.info(f"  Recall: {results['metrics']['recall']:.4f}")
        logger.info(f"  F1 Score: {results['metrics']['f1_score']:.4f}")
        logger.info(
            f"  Directional Accuracy: {results['metrics']['directional_accuracy']:.4f}"
        )
        logger.info(f"  Brier Score: {results['metrics']['brier_score']:.4f}")

    return results


def backtest_model(symbols: list, days: int):
    """Run backtest on trained model"""
    from ai.cnn_stock_predictor import get_cnn_predictor

    logger.info("\n" + "=" * 60)
    logger.info("BACKTESTING")
    logger.info("=" * 60)

    predictor = get_cnn_predictor()

    all_results = []
    for symbol in symbols[:5]:  # Test on first 5 symbols
        try:
            result = predictor.backtest(symbol, days=days)
            all_results.append(result)

            logger.info(f"\n{symbol}:")
            logger.info(f"  Return: {result['total_return_pct']:.2f}%")
            logger.info(f"  Win Rate: {result['win_rate']*100:.1f}%")
            logger.info(f"  Profit Factor: {result['profit_factor']:.2f}")
            logger.info(f"  Max Drawdown: {result['max_drawdown']:.2f}%")
            logger.info(f"  Sharpe: {result['sharpe_ratio']:.2f}")

        except Exception as e:
            logger.warning(f"Backtest failed for {symbol}: {e}")

    # Aggregate results
    if all_results:
        avg_return = sum(r["total_return_pct"] for r in all_results) / len(all_results)
        avg_winrate = sum(r["win_rate"] for r in all_results) / len(all_results)
        avg_sharpe = sum(r["sharpe_ratio"] for r in all_results) / len(all_results)

        logger.info("\n" + "=" * 60)
        logger.info("AGGREGATE BACKTEST RESULTS")
        logger.info("=" * 60)
        logger.info(f"Average Return: {avg_return:.2f}%")
        logger.info(f"Average Win Rate: {avg_winrate*100:.1f}%")
        logger.info(f"Average Sharpe: {avg_sharpe:.2f}")

    return all_results


def compare_with_existing():
    """Compare CNN with existing LightGBM predictor"""
    from ai.alpaca_ai_predictor import get_alpaca_predictor
    from ai.cnn_stock_predictor import get_cnn_predictor

    logger.info("\n" + "=" * 60)
    logger.info("MODEL COMPARISON: CNN vs LightGBM")
    logger.info("=" * 60)

    cnn = get_cnn_predictor()
    lgb = get_alpaca_predictor()

    test_symbols = ["SPY", "AAPL", "TSLA", "NVDA", "AMD"]

    for symbol in test_symbols:
        try:
            cnn_pred = cnn.predict(symbol)
            lgb_pred = lgb.predict(symbol)

            logger.info(f"\n{symbol}:")
            logger.info(
                f"  CNN:     {cnn_pred['action']:5} (conf: {cnn_pred['confidence']:.2f}, momentum: {cnn_pred.get('momentum_score', 0):.2f})"
            )
            logger.info(
                f"  LightGBM: {lgb_pred['action']:5} (conf: {lgb_pred['confidence']:.2f})"
            )

            if cnn_pred["action"] == lgb_pred["action"]:
                logger.info("  Agreement: YES")
            else:
                logger.info("  Agreement: NO")

        except Exception as e:
            logger.warning(f"Comparison failed for {symbol}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Train CNN Stock Predictor")

    parser.add_argument(
        "--symbols", type=str, default=None, help="Comma-separated list of symbols"
    )
    parser.add_argument(
        "--momentum", action="store_true", help="Use momentum stock list"
    )
    parser.add_argument(
        "--smallcap", action="store_true", help="Use small cap momentum list"
    )
    parser.add_argument("--all", action="store_true", help="Use full universe")
    parser.add_argument("--days", type=int, default=365, help="Days of historical data")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--backtest", action="store_true", help="Run backtest after training"
    )
    parser.add_argument(
        "--compare", action="store_true", help="Compare with LightGBM predictor"
    )

    args = parser.parse_args()

    # Determine symbols
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
    elif args.smallcap:
        symbols = SMALL_CAP_MOMENTUM
    elif args.all:
        symbols = FULL_UNIVERSE
    else:
        symbols = MOMENTUM_STOCKS

    # Train
    results = train_model(
        symbols=symbols, days=args.days, epochs=args.epochs, batch_size=args.batch_size
    )

    # Optional: Backtest
    if args.backtest:
        backtest_model(symbols, args.days)

    # Optional: Compare
    if args.compare:
        compare_with_existing()

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING SESSION COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
