"""
Backtesting Module
Simple backtester for AI trading strategies using historical data
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
import yfinance as yf

logger = logging.getLogger(__name__)


@dataclass
class BacktestTrade:
    """Record of a simulated trade"""
    symbol: str
    entry_date: str
    entry_price: float
    exit_date: str
    exit_price: float
    side: str
    quantity: int
    pnl: float
    pnl_percent: float
    signal: str
    confidence: float


@dataclass
class BacktestResult:
    """Results of a backtest run"""
    start_date: str
    end_date: str
    symbols: List[str]
    initial_capital: float
    final_capital: float
    total_return: float
    total_return_percent: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    max_drawdown: float
    max_drawdown_percent: float
    sharpe_ratio: float
    avg_trade_pnl: float
    avg_win: float
    avg_loss: float
    trades: List[dict]
    equity_curve: List[float] = None  # Daily equity values for charting
    drawdown_curve: List[float] = None  # Daily drawdown percentages


class Backtester:
    """
    Simple backtester that simulates AI trading strategy on historical data
    """

    def __init__(self):
        self.predictor = None

    def _get_predictor(self):
        """Lazy load the predictor"""
        if self.predictor is None:
            from ai.alpaca_ai_predictor import get_alpaca_predictor
            self.predictor = get_alpaca_predictor()
        return self.predictor

    def _get_historical_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Get historical data for backtesting"""
        try:
            df = yf.download(
                symbol,
                start=start_date,
                end=end_date,
                progress=False
            )

            # Flatten column index if multi-level
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)

            return df
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {e}")
            return pd.DataFrame()

    def _calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical features for prediction"""
        predictor = self._get_predictor()
        return predictor.calculate_features(df)

    def _generate_signals(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Generate AI signals for historical data"""
        predictor = self._get_predictor()

        if predictor.model is None:
            raise ValueError("AI model not trained - train the model first")

        # Calculate features
        df = self._calculate_features(df)

        # Get feature columns
        available_features = [f for f in predictor.feature_names if f in df.columns]

        if len(available_features) < 10:
            raise ValueError(f"Not enough features available for {symbol}")

        signals = []
        confidences = []
        actions = []

        threshold = getattr(predictor, 'optimal_threshold', 0.5)

        for i in range(len(df)):
            if i < 50:  # Need warmup period for features
                signals.append("NEUTRAL")
                confidences.append(0.0)
                actions.append("HOLD")
                continue

            try:
                row = df[available_features].iloc[i:i+1].values

                # Pad if needed
                if len(available_features) < len(predictor.feature_names):
                    padding = np.zeros((1, len(predictor.feature_names) - len(available_features)))
                    row = np.concatenate([row, padding], axis=1)

                prob = predictor.model.predict(row)[0]
                confidence = abs(prob - threshold) * 2

                # Determine signal
                bull_strong = threshold + 0.2
                bull_mild = threshold + 0.05
                bear_strong = threshold - 0.2
                bear_mild = threshold - 0.05

                if prob > bull_strong:
                    signal = "STRONG_BULLISH"
                    action = "BUY"
                elif prob > bull_mild:
                    signal = "BULLISH"
                    action = "BUY"
                elif prob < bear_strong:
                    signal = "STRONG_BEARISH"
                    action = "SELL"
                elif prob < bear_mild:
                    signal = "BEARISH"
                    action = "SELL"
                else:
                    signal = "NEUTRAL"
                    action = "HOLD"

                signals.append(signal)
                confidences.append(confidence)
                actions.append(action)

            except Exception as e:
                signals.append("NEUTRAL")
                confidences.append(0.0)
                actions.append("HOLD")

        df['signal'] = signals
        df['confidence'] = confidences
        df['action'] = actions

        return df

    def run_backtest(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        initial_capital: float = 100000,
        position_size_pct: float = 0.1,
        confidence_threshold: float = 0.15,
        max_positions: int = 5,
        holding_period: int = 5
    ) -> BacktestResult:
        """
        Run a backtest simulation

        Args:
            symbols: List of symbols to trade
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            initial_capital: Starting capital
            position_size_pct: Position size as % of capital (0.1 = 10%)
            confidence_threshold: Minimum confidence to trade
            max_positions: Maximum concurrent positions
            holding_period: Days to hold each position

        Returns:
            BacktestResult with performance metrics
        """
        logger.info(f"Running backtest: {symbols} from {start_date} to {end_date}")

        trades: List[BacktestTrade] = []
        capital = initial_capital
        equity_curve = [capital]
        peak_equity = capital
        max_drawdown = 0
        open_positions: Dict[str, dict] = {}

        # Process each symbol
        all_signals = {}
        for symbol in symbols:
            df = self._get_historical_data(symbol, start_date, end_date)
            if df.empty:
                logger.warning(f"No data for {symbol}")
                continue

            try:
                df = self._generate_signals(df, symbol)
                all_signals[symbol] = df
            except Exception as e:
                logger.warning(f"Could not generate signals for {symbol}: {e}")

        if not all_signals:
            raise ValueError("No valid data for any symbols")

        # Get all trading days
        all_dates = set()
        for df in all_signals.values():
            all_dates.update(df.index.tolist())
        trading_days = sorted(all_dates)

        # Simulate trading day by day
        for day in trading_days:
            # Check for exits (holding period expired)
            positions_to_close = []
            for symbol, pos in open_positions.items():
                days_held = (day - pd.Timestamp(pos['entry_date'])).days
                if days_held >= holding_period:
                    positions_to_close.append(symbol)

            # Close positions
            for symbol in positions_to_close:
                pos = open_positions.pop(symbol)
                if symbol in all_signals and day in all_signals[symbol].index:
                    exit_price = all_signals[symbol].loc[day, 'Close']

                    if pos['side'] == 'BUY':
                        pnl = (exit_price - pos['entry_price']) * pos['quantity']
                    else:
                        pnl = (pos['entry_price'] - exit_price) * pos['quantity']

                    pnl_percent = (pnl / (pos['entry_price'] * pos['quantity'])) * 100
                    capital += pnl

                    trade = BacktestTrade(
                        symbol=symbol,
                        entry_date=pos['entry_date'].isoformat(),
                        entry_price=pos['entry_price'],
                        exit_date=day.isoformat() if hasattr(day, 'isoformat') else str(day),
                        exit_price=exit_price,
                        side=pos['side'],
                        quantity=pos['quantity'],
                        pnl=pnl,
                        pnl_percent=pnl_percent,
                        signal=pos['signal'],
                        confidence=pos['confidence']
                    )
                    trades.append(trade)

            # Check for new entries
            if len(open_positions) < max_positions:
                for symbol, df in all_signals.items():
                    if symbol in open_positions:
                        continue
                    if day not in df.index:
                        continue

                    row = df.loc[day]
                    action = row.get('action', 'HOLD')
                    confidence = row.get('confidence', 0)

                    if action in ['BUY', 'SELL'] and confidence >= confidence_threshold:
                        # Calculate position size
                        position_value = capital * position_size_pct
                        price = row['Close']
                        quantity = int(position_value / price)

                        if quantity > 0 and len(open_positions) < max_positions:
                            open_positions[symbol] = {
                                'entry_date': day,
                                'entry_price': price,
                                'side': action,
                                'quantity': quantity,
                                'signal': row.get('signal', 'NEUTRAL'),
                                'confidence': confidence
                            }

            # Update equity curve
            portfolio_value = capital
            for symbol, pos in open_positions.items():
                if symbol in all_signals and day in all_signals[symbol].index:
                    current_price = all_signals[symbol].loc[day, 'Close']
                    if pos['side'] == 'BUY':
                        unrealized = (current_price - pos['entry_price']) * pos['quantity']
                    else:
                        unrealized = (pos['entry_price'] - current_price) * pos['quantity']
                    portfolio_value += unrealized

            equity_curve.append(portfolio_value)

            # Track drawdown
            if portfolio_value > peak_equity:
                peak_equity = portfolio_value
            drawdown = (peak_equity - portfolio_value) / peak_equity
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        # Close remaining positions at end
        for symbol, pos in open_positions.items():
            if symbol in all_signals:
                df = all_signals[symbol]
                if len(df) > 0:
                    exit_price = df.iloc[-1]['Close']
                    exit_date = df.index[-1]

                    if pos['side'] == 'BUY':
                        pnl = (exit_price - pos['entry_price']) * pos['quantity']
                    else:
                        pnl = (pos['entry_price'] - exit_price) * pos['quantity']

                    pnl_percent = (pnl / (pos['entry_price'] * pos['quantity'])) * 100
                    capital += pnl

                    trade = BacktestTrade(
                        symbol=symbol,
                        entry_date=pos['entry_date'].isoformat(),
                        entry_price=pos['entry_price'],
                        exit_date=exit_date.isoformat() if hasattr(exit_date, 'isoformat') else str(exit_date),
                        exit_price=exit_price,
                        side=pos['side'],
                        quantity=pos['quantity'],
                        pnl=pnl,
                        pnl_percent=pnl_percent,
                        signal=pos['signal'],
                        confidence=pos['confidence']
                    )
                    trades.append(trade)

        # Calculate metrics
        total_trades = len(trades)
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl <= 0]

        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
        gross_profit = sum(t.pnl for t in winning_trades)
        gross_loss = abs(sum(t.pnl for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        avg_win = gross_profit / len(winning_trades) if winning_trades else 0
        avg_loss = gross_loss / len(losing_trades) if losing_trades else 0
        avg_trade_pnl = sum(t.pnl for t in trades) / total_trades if total_trades > 0 else 0

        # Sharpe ratio (simplified - using daily returns)
        returns = pd.Series(equity_curve).pct_change().dropna()
        sharpe_ratio = (returns.mean() / returns.std() * np.sqrt(252)) if len(returns) > 1 and returns.std() > 0 else 0

        total_return = capital - initial_capital
        total_return_percent = (total_return / initial_capital) * 100

        # Calculate drawdown curve
        equity_series = pd.Series(equity_curve)
        rolling_peak = equity_series.expanding().max()
        drawdown_curve = ((rolling_peak - equity_series) / rolling_peak * 100).tolist()

        # Sample equity curve if too large (keep ~50 points for charting)
        sampled_equity = equity_curve
        sampled_drawdown = drawdown_curve
        if len(equity_curve) > 60:
            step = len(equity_curve) // 50
            sampled_equity = [equity_curve[i] for i in range(0, len(equity_curve), step)]
            sampled_drawdown = [drawdown_curve[i] for i in range(0, len(drawdown_curve), step)]

        return BacktestResult(
            start_date=start_date,
            end_date=end_date,
            symbols=symbols,
            initial_capital=initial_capital,
            final_capital=round(capital, 2),
            total_return=round(total_return, 2),
            total_return_percent=round(total_return_percent, 2),
            total_trades=total_trades,
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            win_rate=round(win_rate, 2),
            profit_factor=round(profit_factor, 2) if profit_factor != float('inf') else 999.99,
            max_drawdown=round(max_drawdown * initial_capital, 2),
            max_drawdown_percent=round(max_drawdown * 100, 2),
            sharpe_ratio=round(sharpe_ratio, 2),
            avg_trade_pnl=round(avg_trade_pnl, 2),
            avg_win=round(avg_win, 2),
            avg_loss=round(avg_loss, 2),
            trades=[asdict(t) for t in trades],
            equity_curve=[round(e, 2) for e in sampled_equity],
            drawdown_curve=[round(d, 2) for d in sampled_drawdown]
        )


# Singleton instance
_backtester: Optional[Backtester] = None


def get_backtester() -> Backtester:
    """Get or create the backtester singleton"""
    global _backtester
    if _backtester is None:
        _backtester = Backtester()
    return _backtester
