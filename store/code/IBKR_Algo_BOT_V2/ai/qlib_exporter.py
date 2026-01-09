"""
Qlib Signal Contract Exporter
==============================
Exports SignalContracts from offline Qlib research.

ARCHITECTURAL PRINCIPLE:
- Qlib MUST NOT perform live inference
- Qlib is for OFFLINE research only
- This exporter runs during research phase (not live trading)
- It generates static SignalContracts based on backtest results
- These contracts are then loaded by the live system

Workflow:
1. Run Qlib analysis on historical data (offline)
2. Identify profitable signal patterns
3. Export SignalContracts with validated parameters
4. Save to signal_contracts.json
5. Live system loads and uses these contracts

The live system NEVER runs Qlib inference - it only validates
incoming triggers against pre-approved SignalContracts.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from ai.signal_contract import (MarketRegime, SignalContract,
                                SignalContractRepository, SignalDirection,
                                SignalHorizon)

logger = logging.getLogger(__name__)


class QlibResearchResult:
    """
    Results from offline Qlib research.

    This is the intermediate format between Qlib analysis
    and SignalContract generation.
    """

    def __init__(
        self,
        symbol: str,
        strategy_type: str,
        backtest_period: str,
        total_trades: int,
        win_rate: float,
        profit_factor: float,
        sharpe_ratio: float,
        max_drawdown: float,
        expected_return: float,
        top_features: List[str],
        valid_regimes: List[str],
        confidence_threshold: float,
    ):
        self.symbol = symbol
        self.strategy_type = strategy_type
        self.backtest_period = backtest_period
        self.total_trades = total_trades
        self.win_rate = win_rate
        self.profit_factor = profit_factor
        self.sharpe_ratio = sharpe_ratio
        self.max_drawdown = max_drawdown
        self.expected_return = expected_return
        self.top_features = top_features
        self.valid_regimes = valid_regimes
        self.confidence_threshold = confidence_threshold


class QlibExporter:
    """
    Exports SignalContracts from Qlib research results.

    This runs OFFLINE during research phase. It:
    1. Analyzes historical data with Qlib
    2. Backtests various signal configurations
    3. Exports validated SignalContracts

    The exported contracts are the ONLY Qlib output used in live trading.
    """

    def __init__(self, output_path: str = "ai/signal_contracts.json"):
        self.output_path = output_path
        self.contracts: List[SignalContract] = []

        # Minimum requirements for contract generation
        self.min_trades = 20  # Minimum backtest trades
        self.min_win_rate = 0.45  # Minimum win rate
        self.min_profit_factor = 1.0  # Minimum profit factor
        self.min_sharpe = 0.5  # Minimum Sharpe ratio

        logger.info("QlibExporter initialized")

    def research_symbol(
        self, symbol: str, strategy_type: str = "SCALP", period: str = "1y"
    ) -> Optional[QlibResearchResult]:
        """
        Run Qlib research on a single symbol.

        This is the OFFLINE research phase where we analyze
        historical patterns and determine valid signal parameters.

        Args:
            symbol: Stock symbol
            strategy_type: SCALP, INTRADAY, SWING, POSITION
            period: Historical period to analyze

        Returns:
            QlibResearchResult if successful, None otherwise
        """
        try:
            import yfinance as yf
            from ai.qlib_predictor import get_qlib_predictor

            predictor = get_qlib_predictor()

            # Fetch historical data
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period)

            if len(df) < 100:
                logger.warning(f"Insufficient data for {symbol}: {len(df)} bars")
                return None

            # Compute Qlib features
            features = predictor.calculator.compute_features(df)
            if features.empty:
                return None

            # Simulate backtest: when would our features have predicted up moves?
            returns = df["Close"].pct_change().shift(-1)

            # Get predictions for each day
            predictions = []
            actuals = []

            for i in range(60, len(features) - 1):
                try:
                    score = predictor.compute_score(features.iloc[: i + 1])
                    predictions.append(score)
                    actuals.append(returns.iloc[i])
                except:
                    continue

            if len(predictions) < self.min_trades:
                logger.warning(
                    f"Insufficient predictions for {symbol}: {len(predictions)}"
                )
                return None

            predictions = np.array(predictions)
            actuals = np.array(actuals)

            # Find optimal threshold
            best_threshold = 0.55
            best_pf = 0

            for threshold in np.arange(0.50, 0.75, 0.05):
                signals = predictions > threshold
                if signals.sum() < 10:
                    continue

                signal_returns = actuals[signals]
                wins = (signal_returns > 0).sum()
                losses = (signal_returns <= 0).sum()

                if losses == 0:
                    continue

                gross_profit = signal_returns[signal_returns > 0].sum()
                gross_loss = abs(signal_returns[signal_returns <= 0].sum())

                pf = gross_profit / (gross_loss + 1e-8)

                if pf > best_pf:
                    best_pf = pf
                    best_threshold = threshold

            # Calculate final metrics with best threshold
            signals = predictions > best_threshold
            signal_returns = actuals[signals]

            if len(signal_returns) < self.min_trades:
                return None

            wins = (signal_returns > 0).sum()
            total = len(signal_returns)
            win_rate = wins / total

            gross_profit = signal_returns[signal_returns > 0].sum()
            gross_loss = abs(signal_returns[signal_returns <= 0].sum())
            profit_factor = gross_profit / (gross_loss + 1e-8)

            expected_return = signal_returns.mean()
            max_drawdown = self._calculate_max_drawdown(signal_returns)

            # Simplified Sharpe
            sharpe = (signal_returns.mean() / (signal_returns.std() + 1e-8)) * np.sqrt(
                252
            )

            # Determine valid regimes from feature analysis
            valid_regimes = self._determine_valid_regimes(features, signals, df)

            # Get top features
            top_features = self._get_top_features(predictor, features)

            return QlibResearchResult(
                symbol=symbol,
                strategy_type=strategy_type,
                backtest_period=f"{df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}",
                total_trades=total,
                win_rate=win_rate,
                profit_factor=profit_factor,
                sharpe_ratio=sharpe,
                max_drawdown=max_drawdown,
                expected_return=expected_return,
                top_features=top_features,
                valid_regimes=valid_regimes,
                confidence_threshold=best_threshold,
            )

        except Exception as e:
            logger.error(f"Research failed for {symbol}: {e}")
            return None

    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown from returns."""
        cumulative = (1 + returns).cumprod()
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        return abs(drawdowns.min()) if len(drawdowns) > 0 else 0.0

    def _determine_valid_regimes(
        self, features: pd.DataFrame, signals: np.ndarray, df: pd.DataFrame
    ) -> List[str]:
        """
        Determine which market regimes had successful signals.

        Analyzes when signals were profitable vs unprofitable
        and maps to regime types.
        """
        valid = []

        try:
            import ta

            close = df["Close"]

            # Compute ADX for trend detection
            adx = ta.trend.ADXIndicator(df["High"], df["Low"], df["Close"], window=14)
            adx_values = adx.adx()
            adx_pos = adx.adx_pos()
            adx_neg = adx.adx_neg()

            # Compute volatility
            returns = close.pct_change()
            volatility = returns.rolling(20).std() * np.sqrt(252)
            avg_vol = 0.25

            # Classify each bar's regime
            regimes = []
            for i in range(len(df)):
                if i < 20:
                    regimes.append("UNKNOWN")
                    continue

                vol = volatility.iloc[i] if i < len(volatility) else 0.25
                adx_val = adx_values.iloc[i] if i < len(adx_values) else 20

                if vol > avg_vol * 1.5:
                    regimes.append("VOLATILE")
                elif adx_val > 25:
                    if adx_pos.iloc[i] > adx_neg.iloc[i]:
                        regimes.append("TRENDING_UP")
                    else:
                        regimes.append("TRENDING_DOWN")
                else:
                    regimes.append("RANGING")

            # Find which regimes had profitable signals
            regimes = np.array(regimes[-len(signals) :])
            returns_arr = df["Close"].pct_change().shift(-1).values[-len(signals) :]

            regime_performance = {}
            for regime in ["TRENDING_UP", "TRENDING_DOWN", "RANGING", "VOLATILE"]:
                mask = (regimes == regime) & signals
                if mask.sum() < 5:
                    continue

                regime_returns = returns_arr[mask]
                win_rate = (regime_returns > 0).mean()

                if win_rate >= 0.5:
                    valid.append(regime)
                    regime_performance[regime] = win_rate

            logger.debug(f"Regime performance: {regime_performance}")

        except Exception as e:
            logger.debug(f"Regime analysis failed: {e}")
            valid = ["TRENDING_UP", "RANGING"]  # Default safe regimes

        if not valid:
            valid = ["TRENDING_UP", "RANGING"]

        return valid

    def _get_top_features(self, predictor, features: pd.DataFrame) -> List[str]:
        """Get top contributing features."""
        if predictor.is_trained and predictor.feature_importance:
            sorted_features = sorted(
                predictor.feature_importance.items(), key=lambda x: x[1], reverse=True
            )
            return [f[0] for f in sorted_features[:10]]
        else:
            # Default important features
            return [
                "ROC_5",
                "ROC_10",
                "MA_5",
                "MA_10",
                "RSV_20",
                "RANK_20",
                "VMA_5",
                "VWAP_RATIO",
                "KMID",
                "CORR_5",
            ]

    def create_contract_from_research(
        self, result: QlibResearchResult, expiry_days: int = 30
    ) -> Optional[SignalContract]:
        """
        Create a SignalContract from research results.

        This applies quality filters and creates the immutable
        contract that will be used in live trading.
        """
        # Apply quality filters
        if result.total_trades < self.min_trades:
            logger.info(
                f"Skipping {result.symbol}: insufficient trades ({result.total_trades})"
            )
            return None

        if result.win_rate < self.min_win_rate:
            logger.info(
                f"Skipping {result.symbol}: low win rate ({result.win_rate:.1%})"
            )
            return None

        if result.profit_factor < self.min_profit_factor:
            logger.info(
                f"Skipping {result.symbol}: low profit factor ({result.profit_factor:.2f})"
            )
            return None

        if result.sharpe_ratio < self.min_sharpe:
            logger.info(
                f"Skipping {result.symbol}: low Sharpe ({result.sharpe_ratio:.2f})"
            )
            return None

        # Determine invalid regimes (inverse of valid)
        all_regimes = ["TRENDING_UP", "TRENDING_DOWN", "RANGING", "VOLATILE"]
        invalid_regimes = [r for r in all_regimes if r not in result.valid_regimes]

        # For scalping, VOLATILE is always risky
        if "VOLATILE" not in invalid_regimes and result.strategy_type == "SCALP":
            invalid_regimes.append("VOLATILE")

        contract = SignalContract(
            symbol=result.symbol,
            direction="LONG",  # Default to long (no shorting per user preference)
            horizon=result.strategy_type,
            features=result.top_features,
            confidence_required=result.confidence_threshold,
            valid_regimes=result.valid_regimes,
            invalid_regimes=invalid_regimes,
            max_drawdown_allowed=max(result.max_drawdown * 1.5, 0.03),  # Buffer
            max_position_pct=0.02,
            expected_return=result.expected_return,
            historical_win_rate=result.win_rate,
            profit_factor=result.profit_factor,
            source="QLIB",
            expires_at=(datetime.now() + timedelta(days=expiry_days)).isoformat(),
            backtest_period=result.backtest_period,
            sample_size=result.total_trades,
            sharpe_ratio=result.sharpe_ratio,
        )

        return contract

    def export_symbols(
        self, symbols: List[str], strategy_type: str = "SCALP", period: str = "1y"
    ) -> Dict[str, Any]:
        """
        Run research on multiple symbols and export valid contracts.

        This is the main entry point for offline research.

        Args:
            symbols: List of symbols to research
            strategy_type: Type of strategy (SCALP, INTRADAY, etc.)
            period: Historical period to analyze

        Returns:
            Export summary with statistics
        """
        logger.info(f"Starting Qlib research on {len(symbols)} symbols")

        self.contracts = []
        results = {
            "symbols_analyzed": 0,
            "contracts_generated": 0,
            "skipped": [],
            "exported": [],
        }

        for symbol in symbols:
            results["symbols_analyzed"] += 1
            logger.info(f"Researching {symbol}...")

            research_result = self.research_symbol(symbol, strategy_type, period)

            if research_result is None:
                results["skipped"].append(
                    {"symbol": symbol, "reason": "Research failed"}
                )
                continue

            contract = self.create_contract_from_research(research_result)

            if contract is None:
                results["skipped"].append(
                    {"symbol": symbol, "reason": "Quality filters not met"}
                )
                continue

            self.contracts.append(contract)
            results["contracts_generated"] += 1
            results["exported"].append(
                {
                    "symbol": symbol,
                    "win_rate": research_result.win_rate,
                    "profit_factor": research_result.profit_factor,
                    "sharpe": research_result.sharpe_ratio,
                    "trades": research_result.total_trades,
                }
            )

            logger.info(
                f"  {symbol}: WR={research_result.win_rate:.1%}, "
                f"PF={research_result.profit_factor:.2f}, "
                f"Sharpe={research_result.sharpe_ratio:.2f}"
            )

        # Save contracts
        self.save_contracts()

        logger.info(
            f"Export complete: {results['contracts_generated']}/{results['symbols_analyzed']} "
            f"contracts generated"
        )

        return results

    def save_contracts(self):
        """Save all contracts to JSON file."""
        data = {
            "contracts": [c.to_dict() for c in self.contracts],
            "exported_at": datetime.now().isoformat(),
            "count": len(self.contracts),
            "source": "QLIB_EXPORTER",
        }

        with open(self.output_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved {len(self.contracts)} contracts to {self.output_path}")

    def load_contracts(self) -> List[SignalContract]:
        """Load existing contracts from file."""
        try:
            with open(self.output_path, "r") as f:
                data = json.load(f)

            self.contracts = [
                SignalContract.from_dict(c) for c in data.get("contracts", [])
            ]

            logger.info(
                f"Loaded {len(self.contracts)} contracts from {self.output_path}"
            )
            return self.contracts

        except FileNotFoundError:
            logger.warning(f"No contracts file found at {self.output_path}")
            return []
        except Exception as e:
            logger.error(f"Failed to load contracts: {e}")
            return []


# Singleton exporter
_exporter: Optional[QlibExporter] = None


def get_qlib_exporter() -> QlibExporter:
    """Get the global Qlib exporter."""
    global _exporter
    if _exporter is None:
        _exporter = QlibExporter()
    return _exporter


def run_research_export(symbols: List[str] = None) -> Dict[str, Any]:
    """
    Convenience function to run full research export.

    This is the PRIMARY entry point for offline research.
    """
    if symbols is None:
        # Default momentum watchlist
        symbols = [
            "AAPL",
            "TSLA",
            "NVDA",
            "AMD",
            "MSFT",
            "META",
            "GOOGL",
            "AMZN",
            "SPY",
            "QQQ",
        ]

    exporter = get_qlib_exporter()
    return exporter.export_symbols(symbols)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    print("=" * 60)
    print("QLIB SIGNAL CONTRACT EXPORTER")
    print("=" * 60)
    print("\nThis is an OFFLINE research tool.")
    print("It generates SignalContracts from backtest results.")
    print("These contracts are then used by the live trading system.\n")

    # Run research on test symbols
    symbols = ["AAPL", "TSLA", "NVDA", "SPY"]

    print(f"Researching {len(symbols)} symbols...")
    results = run_research_export(symbols)

    print("\n" + "=" * 60)
    print("EXPORT SUMMARY")
    print("=" * 60)
    print(f"Symbols analyzed: {results['symbols_analyzed']}")
    print(f"Contracts generated: {results['contracts_generated']}")

    if results["exported"]:
        print("\nExported contracts:")
        for item in results["exported"]:
            print(
                f"  {item['symbol']}: WR={item['win_rate']:.1%}, "
                f"PF={item['profit_factor']:.2f}, Sharpe={item['sharpe']:.2f}"
            )

    if results["skipped"]:
        print("\nSkipped symbols:")
        for item in results["skipped"]:
            print(f"  {item['symbol']}: {item['reason']}")
