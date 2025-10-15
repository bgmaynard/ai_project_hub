"""
Improved Backtest Module with Confidence Filtering
Fixes the over-trading issue by only executing high-confidence trades
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class ImprovedBacktester:
    """
    Enhanced backtesting with confidence filtering and proper transaction costs
    """
    
    def __init__(
        self,
        confidence_threshold: float = 0.65,
        transaction_cost: float = 0.001,
        min_hold_periods: int = 3,
        max_position_size: float = 1.0
    ):
        """
        Args:
            confidence_threshold: Only trade when prediction confidence > this (0-1)
            transaction_cost: Cost per trade as fraction (0.001 = 0.1%)
            min_hold_periods: Minimum bars to hold position
            max_position_size: Maximum position as fraction of capital
        """
        self.confidence_threshold = confidence_threshold
        self.transaction_cost = transaction_cost
        self.min_hold_periods = min_hold_periods
        self.max_position_size = max_position_size
        
    def calculate_confidence(self, predictions: np.ndarray) -> np.ndarray:
        """
        Calculate prediction confidence (distance from 0.5)
        
        Returns:
            Array of confidence scores [0-1]
        """
        # Distance from neutral (0.5)
        confidence = np.abs(predictions - 0.5) * 2
        return confidence
    
    def generate_signals(
        self,
        predictions: np.ndarray,
        confidence: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate trading signals with confidence filtering
        
        Returns:
            position: 1 (long), 0 (neutral), -1 (short)
            signal_strength: 0-1 for position sizing
        """
        # Initialize position array
        position = np.zeros(len(predictions), dtype=int)
        signal_strength = np.zeros(len(predictions))
        
        # Long signals: High confidence bullish
        long_mask = (predictions > 0.5) & (confidence > self.confidence_threshold)
        position[long_mask] = 1
        signal_strength[long_mask] = confidence[long_mask]
        
        # Short signals: High confidence bearish
        short_mask = (predictions < 0.5) & (confidence > self.confidence_threshold)
        position[short_mask] = -1
        signal_strength[short_mask] = confidence[short_mask]
        
        # Enforce minimum hold period
        position = self._enforce_min_hold(position)
        
        return position, signal_strength
    
    def _enforce_min_hold(self, position: np.ndarray) -> np.ndarray:
        """
        Prevent rapid position changes - hold for minimum periods
        """
        if self.min_hold_periods <= 1:
            return position
            
        cleaned = position.copy()
        current_pos = 0
        hold_counter = 0
        
        for i in range(len(position)):
            if position[i] != current_pos:
                if hold_counter >= self.min_hold_periods:
                    current_pos = position[i]
                    hold_counter = 0
                else:
                    cleaned[i] = current_pos
                    hold_counter += 1
            else:
                hold_counter += 1
                
        return cleaned
    
    def backtest_strategy(
        self,
        model,
        df: pd.DataFrame,
        initial_capital: float = 100000
    ) -> Dict:
        """
        Run complete backtest with improved logic
        
        Args:
            model: Trained LSTM model
            df: DataFrame with OHLCV data
            initial_capital: Starting capital
            
        Returns:
            Dictionary with performance metrics
        """
        logger.info(f"Running backtest with confidence threshold: {self.confidence_threshold}")
        
        # Get predictions
        predictions = model.predict_batch(df)
        confidence = self.calculate_confidence(predictions)
        
        # Align with data
        sequence_length = model.sequence_length
        start_idx = sequence_length - 1
        end_idx = start_idx + len(predictions)
        
        df_aligned = df.iloc[start_idx:end_idx].copy()
        
        # Handle length mismatches
        if len(predictions) > len(df_aligned):
            predictions = predictions[:len(df_aligned)]
            confidence = confidence[:len(df_aligned)]
        elif len(predictions) < len(df_aligned):
            df_aligned = df_aligned.iloc[:len(predictions)].copy()
        
        # Add predictions and confidence
        df_aligned['prediction'] = predictions
        df_aligned['confidence'] = confidence
        
        # Generate trading signals
        position, signal_strength = self.generate_signals(predictions, confidence)
        df_aligned['position'] = position
        df_aligned['signal_strength'] = signal_strength
        
        # Calculate returns
        df_aligned['returns'] = df_aligned['close'].pct_change()
        
        # Calculate strategy performance
        results = self._calculate_performance(df_aligned, initial_capital)
        
        # Add diagnostic info
        results['total_predictions'] = len(predictions)
        results['high_confidence_trades'] = (confidence > self.confidence_threshold).sum()
        results['trade_frequency'] = results['total_trades'] / len(predictions) if len(predictions) > 0 else 0
        
        return results
    
    def _calculate_performance(
        self,
        df: pd.DataFrame,
        initial_capital: float
    ) -> Dict:
        """
        Calculate detailed performance metrics
        """
        # Track portfolio value
        capital = initial_capital
        position_size = 0
        entry_price = 0
        trades = []
        equity_curve = [initial_capital]
        
        for i in range(len(df)):
            row = df.iloc[i]
            current_price = row['close']
            
            # Check for position changes
            if i == 0:
                prev_position = 0
            else:
                prev_position = df.iloc[i-1]['position']
            
            current_position = row['position']
            
            # Position change detected
            if current_position != prev_position:
                # Close existing position
                if position_size != 0:
                    pnl = position_size * (current_price - entry_price)
                    capital += pnl - (abs(position_size) * self.transaction_cost)
                    
                    trades.append({
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'pnl': pnl,
                        'return': (current_price - entry_price) / entry_price,
                        'direction': 'long' if position_size > 0 else 'short'
                    })
                    position_size = 0
                
                # Open new position
                if current_position != 0:
                    # Size position based on signal strength
                    size_fraction = row['signal_strength'] * self.max_position_size
                    position_size = (capital * size_fraction / current_price) * current_position
                    entry_price = current_price
                    capital -= abs(position_size * current_price * self.transaction_cost)
            
            # Calculate current equity
            if position_size != 0:
                current_equity = capital + position_size * (current_price - entry_price)
            else:
                current_equity = capital
                
            equity_curve.append(current_equity)
        
        # Close final position
        if position_size != 0:
            final_price = df.iloc[-1]['close']
            pnl = position_size * (final_price - entry_price)
            capital += pnl - (abs(position_size) * self.transaction_cost)
            
            trades.append({
                'entry_price': entry_price,
                'exit_price': final_price,
                'pnl': pnl,
                'return': (final_price - entry_price) / entry_price,
                'direction': 'long' if position_size > 0 else 'short'
            })
        
        # Calculate metrics
        if len(trades) > 0:
            trade_returns = [t['return'] for t in trades]
            winning_trades = [t for t in trades if t['pnl'] > 0]
            
            total_return = (capital - initial_capital) / initial_capital
            win_rate = len(winning_trades) / len(trades)
            avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t['pnl'] for t in trades if t['pnl'] < 0])
            
            # Calculate Sharpe ratio (annualized)
            returns_array = np.array(trade_returns)
            sharpe = np.sqrt(252) * returns_array.mean() / returns_array.std() if returns_array.std() > 0 else 0
            
            # Calculate max drawdown
            equity_array = np.array(equity_curve)
            running_max = np.maximum.accumulate(equity_array)
            drawdown = (equity_array - running_max) / running_max
            max_drawdown = drawdown.min()
            
        else:
            total_return = 0
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            sharpe = 0
            max_drawdown = 0
        
        return {
            'total_return': total_return,
            'final_capital': capital,
            'total_trades': len(trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'equity_curve': equity_curve,
            'trades': trades
        }
    
    def print_results(self, results: Dict):
        """Print backtest results in readable format"""
        print("\n" + "="*60)
        print("BACKTEST RESULTS")
        print("="*60)
        print(f"Total Return:        {results['total_return']*100:>8.2f}%")
        print(f"Final Capital:       ${results['final_capital']:>12,.2f}")
        print(f"Total Trades:        {results['total_trades']:>8}")
        print(f"Trade Frequency:     {results['trade_frequency']*100:>8.2f}%")
        print(f"Win Rate:            {results['win_rate']*100:>8.2f}%")
        print(f"Avg Win:             ${results['avg_win']:>12,.2f}")
        print(f"Avg Loss:            ${results['avg_loss']:>12,.2f}")
        print(f"Sharpe Ratio:        {results['sharpe_ratio']:>8.2f}")
        print(f"Max Drawdown:        {results['max_drawdown']*100:>8.2f}%")
        print(f"Confidence Thresh:   {self.confidence_threshold*100:>8.1f}%")
        print("="*60)
        
        if results['total_trades'] > 0:
            print(f"\nHigh Confidence Predictions: {results['high_confidence_trades']}/{results['total_predictions']}")
            print(f"({results['high_confidence_trades']/results['total_predictions']*100:.1f}% of predictions)")


# Example usage function
def run_improved_backtest(model, df, confidence_threshold=0.65):
    """
    Convenience function to run backtest
    """
    backtester = ImprovedBacktester(
        confidence_threshold=confidence_threshold,
        transaction_cost=0.001,  # 0.1%
        min_hold_periods=3,      # Hold at least 3 bars
        max_position_size=0.95   # Use up to 95% of capital
    )
    
    results = backtester.backtest_strategy(model, df)
    backtester.print_results(results)
    
    return results


if __name__ == "__main__":
    print("Improved Backtest Module")
    print("=" * 60)
    print("\nKey Features:")
    print("✓ Confidence threshold filtering")
    print("✓ Proper transaction cost modeling")
    print("✓ Minimum hold period enforcement")
    print("✓ Position sizing based on signal strength")
    print("✓ Detailed performance metrics")
    print("\nUsage:")
    print("  from improved_backtest import run_improved_backtest")
    print("  results = run_improved_backtest(model, df, confidence_threshold=0.65)")
