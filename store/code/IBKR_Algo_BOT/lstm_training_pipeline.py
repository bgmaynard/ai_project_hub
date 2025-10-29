"""
LSTM Training Pipeline - Complete Workflow
Handles data loading, preprocessing, training, evaluation, and deployment
"""

import pandas as pd
import numpy as np
from datetime import datetime
import yfinance as yf
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LSTMTrainingPipeline:
    """
    Complete training pipeline for LSTM trading models
    """
    
    def __init__(self, output_dir="models/lstm_pipeline"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.models = {}
        self.training_history = []
        
    def fetch_historical_data(self, symbol, period="2y", interval="5m"):
        """
        Fetch historical data from Yahoo Finance
        """
        logger.info(f"Fetching data for {symbol}...")
        
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            
            if df.empty:
                logger.error(f"No data retrieved for {symbol}")
                return None
            
            # Rename columns to lowercase
            df.columns = df.columns.str.lower()
            
            logger.info(f"Retrieved {len(df)} bars for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    def load_csv_data(self, filepath):
        """
        Load data from CSV file
        Expected columns: timestamp, open, high, low, close, volume
        """
        logger.info(f"Loading data from {filepath}...")
        
        df = pd.read_csv(filepath)
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        elif 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        
        # Ensure required columns exist
        required = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required):
            logger.error(f"Missing required columns. Found: {df.columns.tolist()}")
            return None
        
        logger.info(f"Loaded {len(df)} bars from CSV")
        return df
    
    def train_model(self, symbol, df, model_config=None):
        """
        Train LSTM model for a specific symbol
        """
        from lstm_model_complete import LSTMTradingModel
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Training LSTM model for {symbol}")
        logger.info(f"{'='*60}")
        
        # Default config
        if model_config is None:
            model_config = {
                'sequence_length': 60,
                'prediction_horizon': 5,
                'lstm_units': [64, 32],
                'dropout_rate': 0.2,
                'learning_rate': 0.001,
                'epochs': 50,
                'batch_size': 32
            }
        
        # Initialize model
        model = LSTMTradingModel(
            sequence_length=model_config['sequence_length'],
            prediction_horizon=model_config['prediction_horizon']
        )
        
        # Prepare data FIRST to determine feature count
        logger.info("Preparing training data...")
        X, y = model.prepare_sequences(df)
        
        if len(X) == 0:
            logger.error(f"No training data created for {symbol}")
            return None, None, None
        
        logger.info(f"Created {len(X)} training sequences with {model.feature_count} features")
        
        # NOW build model with known feature count
        model.build_model(
            lstm_units=model_config['lstm_units'],
            dropout_rate=model_config['dropout_rate'],
            learning_rate=model_config['learning_rate']
        )
        
        # Train (but don't call prepare_sequences again)
        logger.info("Training model...")
        from sklearn.model_selection import train_test_split
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        
        logger.info(f"Training on {len(X_train)} sequences, validating on {len(X_val)}")
        logger.info(f"Target distribution - Up: {y.sum()}, Down: {len(y) - y.sum()}")
        
        # Callbacks
        from tensorflow.keras import callbacks
        
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001,
            verbose=1
        )
        
        checkpoint = callbacks.ModelCheckpoint(
            str(model.model_path / 'best_model.h5'),
            monitor='val_auc',
            mode='max',
            save_best_only=True,
            verbose=1
        )
        
        # Train
        history = model.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=model_config['epochs'],
            batch_size=model_config['batch_size'],
            callbacks=[early_stop, reduce_lr, checkpoint],
            verbose=1
        )
        
        # Evaluate
        logger.info("Evaluating model...")
        train_metrics = model.model.evaluate(X_train, y_train, verbose=0)
        val_metrics = model.model.evaluate(X_val, y_val, verbose=0)
        
        results = {
            'train_loss': float(train_metrics[0]),
            'train_accuracy': float(train_metrics[1]),
            'train_auc': float(train_metrics[2]),
            'val_loss': float(val_metrics[0]),
            'val_accuracy': float(val_metrics[1]),
            'val_auc': float(val_metrics[2]),
            'epochs_trained': len(history.history['loss']),
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Training complete - Val Accuracy: {results['val_accuracy']:.3f}, Val AUC: {results['val_auc']:.3f}")
        
        # Save model
        model.save(f"{symbol}_lstm")
        self.models[symbol] = model
        
        # Store training record
        training_record = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'config': model_config,
            'results': results,
            'data_points': len(df)
        }
        self.training_history.append(training_record)
        
        return model, history, results
    
    def train_ensemble(self, symbol, df):
        """
        Train ensemble of models with different configurations
        """
        from lstm_model_complete import EnsembleLSTM
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Training Ensemble Models for {symbol}")
        logger.info(f"{'='*60}")
        
        ensemble = EnsembleLSTM(
            n_models=3,
            sequence_length=60,
            prediction_horizon=5
        )
        
        results = ensemble.train_ensemble(df, epochs=30)
        
        # Save ensemble
        for i, model in enumerate(ensemble.models):
            model.save(f"{symbol}_ensemble_{i}")
        
        logger.info(f"Ensemble training complete for {symbol}")
        return ensemble, results
    
    def backtest_model(self, model, df, transaction_cost=0.001):
        """
        Backtest model on historical data
        """
        logger.info("Running backtest...")
        
        # Generate predictions
        predictions = model.predict_batch(df)
        
        # Align predictions with data - handle length mismatch
        sequence_length = model.sequence_length
        prediction_horizon = model.prediction_horizon
        
        # Predictions start at sequence_length-1 and end before last prediction_horizon bars
        start_idx = sequence_length - 1
        end_idx = start_idx + len(predictions)
        
        df_aligned = df.iloc[start_idx:end_idx].copy()
        
        # Ensure lengths match
        if len(predictions) > len(df_aligned):
            predictions = predictions[:len(df_aligned)]
        elif len(predictions) < len(df_aligned):
            df_aligned = df_aligned.iloc[:len(predictions)].copy()
        
        df_aligned['prediction'] = predictions
        df_aligned['signal'] = (predictions > 0.5).astype(int)
        
        # Calculate returns
        df_aligned['actual_return'] = df_aligned['close'].pct_change().shift(-model.prediction_horizon)
        df_aligned['actual_direction'] = (df_aligned['actual_return'] > 0).astype(int)
        
        # Strategy returns - only trade when signal says BUY (hold cash when signal=0)
        df_aligned['strategy_return'] = 0.0
        df_aligned.loc[df_aligned['signal'] == 1, 'strategy_return'] = df_aligned.loc[df_aligned['signal'] == 1, 'actual_return']
        
        # Subtract transaction costs only on trades
        df_aligned.loc[df_aligned['signal'] == 1, 'strategy_return'] -= transaction_cost
        
        # Calculate metrics
        df_clean = df_aligned.dropna()
        
        if len(df_clean) == 0:
            logger.error("No valid backtest data")
            return None, None
        
        # Accuracy - when we predicted UP, was it actually UP?
        traded_df = df_clean[df_clean['signal'] == 1]
        if len(traded_df) > 0:
            accuracy = (traded_df['signal'] == traded_df['actual_direction']).mean()
        else:
            accuracy = 0.5
        
        # Total return - compound returns
        total_return = (1 + df_clean['strategy_return']).prod() - 1
        
        # Sharpe ratio - annualized
        if df_clean['strategy_return'].std() > 0:
            sharpe = np.sqrt(252 * 78) * df_clean['strategy_return'].mean() / df_clean['strategy_return'].std()  # 78 5-min bars per day
        else:
            sharpe = 0
        
        # Win rate - of actual trades
        if len(traded_df) > 0:
            profitable_trades = (traded_df['strategy_return'] > 0).sum()
            total_trades = len(traded_df)
            win_rate = profitable_trades / total_trades
        else:
            profitable_trades = 0
            total_trades = 0
            win_rate = 0
        
        # Max drawdown
        cumulative = (1 + df_clean['strategy_return']).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        backtest_results = {
            'accuracy': float(accuracy),
            'total_return': float(total_return),
            'sharpe_ratio': float(sharpe),
            'win_rate': float(win_rate),
            'max_drawdown': float(max_drawdown),
            'total_trades': int(total_trades),
            'profitable_trades': int(profitable_trades)
        }
        
        logger.info("\nBacktest Results:")
        logger.info(f"  Accuracy: {accuracy:.3f}")
        logger.info(f"  Total Return: {total_return:.2%}")
        logger.info(f"  Sharpe Ratio: {sharpe:.2f}")
        logger.info(f"  Win Rate: {win_rate:.2%}")
        logger.info(f"  Max Drawdown: {max_drawdown:.2%}")
        
        return backtest_results, df_aligned
    
    def evaluate_model(self, model, df):
        """
        Comprehensive model evaluation with visualizations
        """
        logger.info("Evaluating model performance...")
        
        # Generate predictions
        predictions = model.predict_batch(df)
        
        # Prepare data
        sequence_length = model.sequence_length
        df_eval = df.iloc[sequence_length-1:-model.prediction_horizon].copy()
        df_eval['prediction'] = predictions[:len(df_eval)]
        
        # True labels
        future_return = df['close'].shift(-model.prediction_horizon) / df['close'] - 1
        df_eval['true_label'] = (future_return.iloc[sequence_length-1:-model.prediction_horizon] > 0).astype(int)
        
        df_eval = df_eval.dropna()
        
        # Binary predictions
        y_true = df_eval['true_label'].values
        y_pred = (df_eval['prediction'].values > 0.5).astype(int)
        y_prob = df_eval['prediction'].values
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Classification report
        report = classification_report(y_true, y_pred, output_dict=True)
        
        # ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        # Create visualizations
        self._plot_evaluation(cm, fpr, tpr, roc_auc, df_eval)
        
        evaluation_results = {
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'roc_auc': float(roc_auc),
            'accuracy': float(report['accuracy'])
        }
        
        return evaluation_results
    
    def _plot_evaluation(self, cm, fpr, tpr, roc_auc, df_eval):
        """
        Create evaluation plots
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Confusion Matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_ylabel('True Label')
        axes[0, 0].set_xlabel('Predicted Label')
        
        # ROC Curve
        axes[0, 1].plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})')
        axes[0, 1].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('ROC Curve')
        axes[0, 1].legend()
        
        # Prediction Distribution
        axes[1, 0].hist(df_eval['prediction'], bins=50, alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(x=0.5, color='r', linestyle='--', label='Decision Threshold')
        axes[1, 0].set_xlabel('Predicted Probability')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Prediction Distribution')
        axes[1, 0].legend()
        
        # Cumulative Returns
        df_eval['strategy_return'] = df_eval['prediction'].apply(
            lambda x: 1 if x > 0.5 else 0
        ) * df_eval['close'].pct_change().shift(-5)
        
        cumulative_returns = (1 + df_eval['strategy_return'].fillna(0)).cumprod()
        axes[1, 1].plot(cumulative_returns.index, cumulative_returns.values)
        axes[1, 1].set_xlabel('Time')
        axes[1, 1].set_ylabel('Cumulative Returns')
        axes[1, 1].set_title('Strategy Performance')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / 'evaluation_plots.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        logger.info(f"Evaluation plots saved to {plot_path}")
        plt.close()
    
    def save_pipeline_summary(self):
        """
        Save training pipeline summary
        """
        summary = {
            'pipeline_run': datetime.now().isoformat(),
            'models_trained': len(self.models),
            'training_history': self.training_history
        }
        
        summary_path = self.output_dir / 'pipeline_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Pipeline summary saved to {summary_path}")
        
    def run_full_pipeline(self, symbols, data_source='yahoo', **kwargs):
        """
        Run complete training pipeline for multiple symbols
        """
        logger.info(f"\n{'='*60}")
        logger.info("LSTM TRAINING PIPELINE - FULL RUN")
        logger.info(f"{'='*60}\n")
        
        results = {}
        
        for symbol in symbols:
            logger.info(f"\nProcessing {symbol}...")
            
            # Load data
            if data_source == 'yahoo':
                df = self.fetch_historical_data(symbol, **kwargs)
            elif data_source == 'csv':
                df = self.load_csv_data(kwargs.get('filepath'))
            else:
                logger.error(f"Unknown data source: {data_source}")
                continue
            
            if df is None:
                continue
            
            # Train model
            model, history, train_results = self.train_model(symbol, df)
            
            # Backtest
            backtest_results, backtest_df = self.backtest_model(model, df)
            
            # Evaluate
            eval_results = self.evaluate_model(model, df)
            
            # Store results
            results[symbol] = {
                'training': train_results,
                'backtest': backtest_results,
                'evaluation': eval_results
            }
            
            logger.info(f"\n{symbol} - Training Complete!")
            logger.info(f"  Val Accuracy: {train_results['val_accuracy']:.3f}")
            logger.info(f"  Backtest Return: {backtest_results['total_return']:.2%}")
            logger.info(f"  Sharpe Ratio: {backtest_results['sharpe_ratio']:.2f}")
        
        # Save summary
        self.save_pipeline_summary()
        
        logger.info(f"\n{'='*60}")
        logger.info("PIPELINE COMPLETE!")
        logger.info(f"{'='*60}\n")
        
        return results


# Main execution
if __name__ == "__main__":
    print("\nLSTM Training Pipeline - Complete Workflow")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = LSTMTrainingPipeline(output_dir="models/lstm_pipeline")
    
    # Define symbols to train on
    symbols = ['AAPL', 'TSLA', 'NVDA']
    
    print(f"\nTraining models for: {', '.join(symbols)}")
    print("This may take several minutes...\n")
    
    # Run full pipeline
    results = pipeline.run_full_pipeline(
        symbols=symbols,
        data_source='yahoo',
        period='1y',
        interval='5m'
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    
    for symbol, result in results.items():
        print(f"\n{symbol}:")
        print(f"  Training Accuracy: {result['training']['val_accuracy']:.3f}")
        print(f"  Training AUC: {result['training']['val_auc']:.3f}")
        print(f"  Backtest Return: {result['backtest']['total_return']:.2%}")
        print(f"  Sharpe Ratio: {result['backtest']['sharpe_ratio']:.2f}")
        print(f"  Win Rate: {result['backtest']['win_rate']:.2%}")
        print(f"  Max Drawdown: {result['backtest']['max_drawdown']:.2%}")
    
    print("\n" + "=" * 60)
    print("Models ready for deployment!")
    print("=" * 60)
