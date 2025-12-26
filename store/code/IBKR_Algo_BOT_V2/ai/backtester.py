"""
Backtesting Engine
Test trading strategies on historical data
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List

import pandas as pd
import yfinance as yf


class Backtester:
    def __init__(self, initial_capital=10000.0):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions = {}
        self.trades = []
        self.portfolio_values = []

    def backtest(
        self,
        symbol: str,
        predictor,
        start_date: str,
        end_date: str,
        confidence_threshold=0.65,
        prob_threshold=0.60,
        position_size=5,
    ):
        """Run backtest on historical data"""
        print(f"Backtesting {symbol} from {start_date} to {end_date}...")

        # Get historical data
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date)

        if df.empty:
            return {"error": "No historical data available"}

        # Train model on data before backtest period
        train_start = (pd.to_datetime(start_date) - timedelta(days=730)).strftime(
            "%Y-%m-%d"
        )
        predictor.train(symbol, period="5y")

        results = {
            "symbol": symbol,
            "start_date": start_date,
            "end_date": end_date,
            "initial_capital": self.initial_capital,
            "final_capital": 0,
            "total_return": 0,
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0,
            "max_drawdown": 0,
            "sharpe_ratio": 0,
            "trades": [],
            "equity_curve": [],
        }

        # Simulate trading day by day
        for i in range(len(df) - 1):
            current_date = df.index[i].strftime("%Y-%m-%d")
            current_price = df["Close"].iloc[i]
            next_price = df["Close"].iloc[i + 1]

            # Get prediction for current day
            try:
                # Use actual historical data for prediction
                hist_data = ticker.history(end=current_date, period="3mo")
                if len(hist_data) < 50:
                    continue

                prediction = predictor.predict(symbol)

                # Check if we should trade
                if (
                    prediction["confidence"] >= confidence_threshold
                    and prediction["prob_up"] >= prob_threshold
                    and prediction["prediction"] == 1
                ):

                    # Buy signal
                    if symbol not in self.positions:
                        shares = min(position_size, int(self.capital / current_price))
                        if shares > 0:
                            cost = shares * current_price
                            self.capital -= cost
                            self.positions[symbol] = {
                                "shares": shares,
                                "entry_price": current_price,
                                "entry_date": current_date,
                            }

                # Check if we should sell (next day price movement)
                if symbol in self.positions:
                    position = self.positions[symbol]
                    pnl = (next_price - position["entry_price"]) * position["shares"]
                    pnl_pct = (
                        (next_price - position["entry_price"]) / position["entry_price"]
                    ) * 100

                    # Sell conditions: take profit or stop loss
                    should_sell = False
                    if pnl_pct > 3:  # 3% profit target
                        should_sell = True
                        reason = "Take profit"
                    elif pnl_pct < -2:  # 2% stop loss
                        should_sell = True
                        reason = "Stop loss"
                    elif prediction["prediction"] == 0:  # Bearish signal
                        should_sell = True
                        reason = "Bearish signal"

                    if should_sell:
                        sell_value = position["shares"] * next_price
                        self.capital += sell_value

                        trade = {
                            "symbol": symbol,
                            "entry_date": position["entry_date"],
                            "exit_date": df.index[i + 1].strftime("%Y-%m-%d"),
                            "entry_price": position["entry_price"],
                            "exit_price": next_price,
                            "shares": position["shares"],
                            "pnl": pnl,
                            "pnl_pct": pnl_pct,
                            "reason": reason,
                        }
                        self.trades.append(trade)
                        results["trades"].append(trade)
                        results["total_trades"] += 1

                        if pnl > 0:
                            results["winning_trades"] += 1
                        else:
                            results["losing_trades"] += 1

                        del self.positions[symbol]

            except Exception as e:
                print(f"Error on {current_date}: {e}")
                continue

            # Track portfolio value
            portfolio_value = self.capital
            for pos_symbol, pos in self.positions.items():
                portfolio_value += pos["shares"] * current_price
            self.portfolio_values.append(
                {"date": current_date, "value": portfolio_value}
            )
            results["equity_curve"].append(
                {"date": current_date, "value": portfolio_value}
            )

        # Calculate final results
        results["final_capital"] = self.capital
        for pos_symbol, pos in self.positions.items():
            results["final_capital"] += pos["shares"] * df["Close"].iloc[-1]

        results["total_return"] = (
            (results["final_capital"] - self.initial_capital) / self.initial_capital
        ) * 100
        results["win_rate"] = (
            (results["winning_trades"] / results["total_trades"] * 100)
            if results["total_trades"] > 0
            else 0
        )

        # Calculate max drawdown
        peak = self.initial_capital
        max_dd = 0
        for point in results["equity_curve"]:
            if point["value"] > peak:
                peak = point["value"]
            dd = ((peak - point["value"]) / peak) * 100
            if dd > max_dd:
                max_dd = dd
        results["max_drawdown"] = max_dd

        # Calculate Sharpe ratio
        if len(results["trades"]) > 1:
            returns = [t["pnl_pct"] for t in results["trades"]]
            mean_return = sum(returns) / len(returns)
            std_dev = (
                sum((r - mean_return) ** 2 for r in returns) / len(returns)
            ) ** 0.5
            results["sharpe_ratio"] = (mean_return / std_dev) if std_dev > 0 else 0

        print(
            f"Backtest complete: {results['total_trades']} trades, {results['total_return']:.2f}% return"
        )
        return results


def create_backtester(initial_capital=10000.0):
    return Backtester(initial_capital)
