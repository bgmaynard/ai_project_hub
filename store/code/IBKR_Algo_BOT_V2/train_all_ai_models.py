"""
Train All AI Models
===================
Comprehensive training script for all AI enhancement modules:
1. Multi-Armed Bandit - Initialize with simulated trade history
2. Adaptive Weights - Bootstrap with historical predictions
3. Self-Play Optimizer - Train entry/exit agents
4. RL Trading Agent - Train DQN agent
5. Ensemble Predictor - Verify integration

Run this script to initialize all AI models with baseline performance data.
"""

import sys
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Symbols to train on
TRAINING_SYMBOLS = ['AAPL', 'MSFT', 'NVDA', 'TSLA', 'AMD', 'GOOGL', 'AMZN', 'META']
REGIMES = ['TRENDING_UP', 'TRENDING_DOWN', 'RANGING', 'VOLATILE']


def download_data(symbol: str, period: str = "1y") -> pd.DataFrame:
    """Download historical data for a symbol"""
    import yfinance as yf

    logger.info(f"Downloading {symbol} data ({period})...")
    df = yf.download(symbol, period=period, progress=False)

    if df.empty:
        logger.warning(f"No data for {symbol}")
        return pd.DataFrame()

    # Handle multi-index columns
    if hasattr(df.columns, 'levels'):
        df.columns = df.columns.droplevel(1)

    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators to dataframe"""
    import ta

    if df.empty:
        return df

    df = df.copy()

    # RSI
    df['rsi'] = ta.momentum.rsi(df['Close'], window=14)

    # MACD
    macd = ta.trend.MACD(df['Close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df['Close'])
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()
    df['bb_width'] = bb.bollinger_wband()

    # ADX
    adx = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'])
    df['adx'] = adx.adx()

    # ATR
    df['atr'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])

    # Volume ratio
    df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()

    # Trend
    df['sma_20'] = df['Close'].rolling(20).mean()
    df['trend'] = (df['Close'] - df['sma_20']) / df['sma_20']

    # Volatility
    df['volatility'] = df['Close'].pct_change().rolling(20).std()

    # Returns
    df['return_1d'] = df['Close'].pct_change(1)
    df['return_5d'] = df['Close'].pct_change(5)

    return df.dropna()


def detect_regime(df: pd.DataFrame) -> str:
    """Detect market regime from recent data"""
    if df.empty or len(df) < 20:
        return 'RANGING'

    adx = df['adx'].iloc[-1] if 'adx' in df.columns else 20
    volatility = df['volatility'].iloc[-1] if 'volatility' in df.columns else 0.02
    trend = df['trend'].iloc[-1] if 'trend' in df.columns else 0

    # High volatility
    if volatility > 0.03:
        return 'VOLATILE'

    # Strong trend
    if adx > 25:
        if trend > 0.02:
            return 'TRENDING_UP'
        elif trend < -0.02:
            return 'TRENDING_DOWN'

    return 'RANGING'


def train_bandit():
    """Train Multi-Armed Bandit with simulated historical trades"""
    logger.info("\n" + "="*60)
    logger.info("TRAINING MULTI-ARMED BANDIT")
    logger.info("="*60)

    from ai.bandit_selector import get_bandit, ArmType

    bandit = get_bandit()

    # Simulate historical trade outcomes based on actual price movements
    models = ['lightgbm', 'ensemble', 'heuristic', 'momentum', 'rl_agent']
    strategies = ['momentum_breakout', 'trend_following', 'mean_reversion', 'scalping', 'warrior_small_cap']

    total_trades = 0

    for symbol in TRAINING_SYMBOLS:
        df = download_data(symbol, "6mo")
        if df.empty:
            continue

        df = add_indicators(df)

        # Simulate trades at various points
        for i in range(50, len(df) - 5, 10):  # Every 10 bars
            regime = detect_regime(df.iloc[max(0, i-50):i])

            # Calculate actual return over next 5 bars
            entry_price = df['Close'].iloc[i]
            exit_price = df['Close'].iloc[i + 5]
            actual_pnl = (exit_price - entry_price) / entry_price * 100

            # Assign credit to models based on regime performance
            # (Models that work better in certain regimes)
            model_bonuses = {
                'TRENDING_UP': {'momentum': 0.3, 'trend_following': 0.2, 'lightgbm': 0.1},
                'TRENDING_DOWN': {'mean_reversion': 0.2, 'heuristic': 0.15},
                'RANGING': {'mean_reversion': 0.25, 'scalping': 0.2},
                'VOLATILE': {'heuristic': 0.2, 'scalping': 0.15}
            }

            for model in models:
                # Add regime-appropriate bonus
                bonus = model_bonuses.get(regime, {}).get(model, 0)
                adjusted_pnl = actual_pnl + np.random.normal(bonus * 2, 1)  # Add some noise

                strategy = np.random.choice(strategies)
                bandit.record_trade_outcome(symbol, model, strategy, adjusted_pnl, regime)
                total_trades += 1

    bandit.save()
    logger.info(f"Bandit trained with {total_trades} simulated trades")

    # Show rankings
    logger.info("\nSymbol Rankings:")
    for r in bandit.get_rankings(ArmType.SYMBOL)[:5]:
        logger.info(f"  {r['name']}: mean={r['mean']:.3f}, pulls={r['pulls']}")

    logger.info("\nModel Rankings:")
    for r in bandit.get_rankings(ArmType.MODEL):
        logger.info(f"  {r['name']}: mean={r['mean']:.3f}, pulls={r['pulls']}")

    return True


def train_adaptive_weights():
    """Initialize Adaptive Weights with prediction outcomes"""
    logger.info("\n" + "="*60)
    logger.info("TRAINING ADAPTIVE WEIGHTS")
    logger.info("="*60)

    from ai.adaptive_weights import get_weight_manager

    manager = get_weight_manager()

    total_predictions = 0

    for symbol in TRAINING_SYMBOLS[:4]:  # Use subset for speed
        df = download_data(symbol, "6mo")
        if df.empty:
            continue

        df = add_indicators(df)

        # Simulate predictions
        for i in range(50, len(df) - 5, 5):
            regime = detect_regime(df.iloc[max(0, i-50):i])

            # Actual outcome (1 if price went up, 0 if down)
            actual = 1 if df['Close'].iloc[i + 5] > df['Close'].iloc[i] else 0

            # Simulate model predictions with realistic accuracy
            model_accuracy = {
                'lightgbm': 0.58,
                'ensemble': 0.55,
                'heuristic': 0.52,
                'momentum': 0.54,
                'rl_agent': 0.51,
                'sentiment': 0.50
            }

            for model, base_acc in model_accuracy.items():
                # Higher accuracy in trending markets for momentum models
                if regime in ['TRENDING_UP', 'TRENDING_DOWN'] and model in ['momentum', 'ensemble']:
                    acc = base_acc + 0.05
                else:
                    acc = base_acc

                # Generate prediction based on accuracy
                if np.random.random() < acc:
                    prediction = actual  # Correct prediction
                else:
                    prediction = 1 - actual  # Wrong prediction

                manager.record_prediction(model, prediction, actual, regime)
                total_predictions += 1

    # Force rebalance
    manager.force_rebalance()

    logger.info(f"Adaptive weights trained with {total_predictions} predictions")

    # Show results
    logger.info("\nModel Rankings:")
    for r in manager.get_ranking()[:6]:
        logger.info(f"  {r['name']}: weight={r['weight']:.3f}, acc={r['accuracy']:.3f}")

    return True


def train_selfplay():
    """Train Self-Play Entry/Exit agents"""
    logger.info("\n" + "="*60)
    logger.info("TRAINING SELF-PLAY AGENTS")
    logger.info("="*60)

    from ai.self_play_optimizer import get_selfplay_trainer

    trainer = get_selfplay_trainer()

    # Train on multiple symbols
    symbols_to_train = ['SPY', 'AAPL', 'TSLA']
    episodes_per_symbol = 10

    all_results = []

    for symbol in symbols_to_train:
        logger.info(f"\nTraining on {symbol}...")

        df = download_data(symbol, "1y")
        if df.empty:
            continue

        df = add_indicators(df)
        prices = df['Close'].values

        indicators = {
            'rsi': float(df['rsi'].iloc[-1]) if 'rsi' in df.columns else 50,
            'macd': float(df['macd'].iloc[-1]) if 'macd' in df.columns else 0,
            'volatility': float(df['volatility'].iloc[-1]) if 'volatility' in df.columns else 0.02
        }

        for ep in range(episodes_per_symbol):
            result = trainer.train_episode(prices, indicators)
            all_results.append(result)

            if (ep + 1) % 5 == 0:
                logger.info(f"  Episode {ep+1}: PnL={result['total_pnl']:.2f}, "
                          f"trades={result['trades']}, win_rate={result['win_rate']:.1%}")

    # Final stats
    stats = trainer.get_stats()
    logger.info(f"\nSelf-Play Training Complete:")
    logger.info(f"  Total Episodes: {stats['episodes']}")
    logger.info(f"  Entry Agent ELO: {stats['entry_elo']:.0f}")
    logger.info(f"  Exit Agent ELO: {stats['exit_elo']:.0f}")
    logger.info(f"  Best PnL: {stats['best_pnl']:.2f}")

    return True


def train_rl_agent():
    """Train RL Trading Agent"""
    logger.info("\n" + "="*60)
    logger.info("TRAINING RL TRADING AGENT")
    logger.info("="*60)

    from ai.rl_trading_agent import get_rl_agent, get_rl_environment

    agent = get_rl_agent()
    env = get_rl_environment()

    # Train on SPY for stability
    df = download_data('SPY', "1y")
    if df.empty:
        logger.error("Could not download SPY data")
        return False

    df = add_indicators(df)

    episodes = 20
    results = []

    logger.info(f"Training RL agent for {episodes} episodes on SPY...")

    for ep in range(episodes):
        metrics = agent.train_episode(env, df)
        results.append(metrics)

        if (ep + 1) % 5 == 0:
            logger.info(f"  Episode {metrics.episode}: reward={metrics.total_reward:.2f}, "
                      f"epsilon={metrics.epsilon:.3f}, trades={metrics.trades}, "
                      f"win_rate={metrics.win_rate:.1%}")

    # Save model
    agent.save_model()

    logger.info(f"\nRL Agent Training Complete:")
    logger.info(f"  Total Episodes: {agent.episode}")
    logger.info(f"  Final Epsilon: {agent.epsilon:.4f}")

    return True


def verify_models():
    """Verify all models are working"""
    logger.info("\n" + "="*60)
    logger.info("VERIFYING ALL MODELS")
    logger.info("="*60)

    # Test Bandit
    logger.info("\n1. Testing Bandit Selector...")
    try:
        from ai.bandit_selector import get_symbol_selector, get_model_selector

        symbol_selector = get_symbol_selector()
        symbols = symbol_selector.select_symbols(n=3)
        logger.info(f"   Top 3 symbols: {symbols}")

        model_selector = get_model_selector()
        model = model_selector.select_model(regime='TRENDING_UP')
        logger.info(f"   Best model for TRENDING_UP: {model}")
    except Exception as e:
        logger.error(f"   Bandit error: {e}")

    # Test Adaptive Weights
    logger.info("\n2. Testing Adaptive Weights...")
    try:
        from ai.adaptive_weights import get_weight_manager

        manager = get_weight_manager()
        weights = manager.get_weights()
        logger.info(f"   Current weights: {weights}")
        logger.info(f"   Best model: {manager.get_best_model()}")
    except Exception as e:
        logger.error(f"   Adaptive weights error: {e}")

    # Test Self-Play
    logger.info("\n3. Testing Self-Play Optimizer...")
    try:
        from ai.self_play_optimizer import get_selfplay_trainer

        trainer = get_selfplay_trainer()
        stats = trainer.get_stats()
        logger.info(f"   Episodes: {stats['episodes']}")
        logger.info(f"   Entry ELO: {stats['entry_elo']:.0f}")
        logger.info(f"   Exit ELO: {stats['exit_elo']:.0f}")
    except Exception as e:
        logger.error(f"   Self-play error: {e}")

    # Test RL Agent
    logger.info("\n4. Testing RL Agent...")
    try:
        from ai.rl_trading_agent import get_rl_agent

        agent = get_rl_agent()
        logger.info(f"   Episodes: {agent.episode}")
        logger.info(f"   Epsilon: {agent.epsilon:.4f}")
    except Exception as e:
        logger.error(f"   RL agent error: {e}")

    # Test Ensemble
    logger.info("\n5. Testing Ensemble Predictor...")
    try:
        from ai.ensemble_predictor import get_ensemble_predictor

        predictor = get_ensemble_predictor()
        df = download_data('AAPL', '3mo')
        if not df.empty:
            df = add_indicators(df)
            result = predictor.predict('AAPL', df)
            logger.info(f"   AAPL prediction: {'BULLISH' if result.prediction == 1 else 'BEARISH'}")
            logger.info(f"   Confidence: {result.confidence:.1%}")
    except Exception as e:
        logger.error(f"   Ensemble error: {e}")

    logger.info("\n" + "="*60)
    logger.info("VERIFICATION COMPLETE")
    logger.info("="*60)

    return True


def main():
    """Main training function"""
    logger.info("="*60)
    logger.info("AI MODEL TRAINING SCRIPT")
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*60)

    # Create store directories
    Path("store/models").mkdir(parents=True, exist_ok=True)

    success = True

    # 1. Train Bandit
    try:
        train_bandit()
    except Exception as e:
        logger.error(f"Bandit training failed: {e}")
        success = False

    # 2. Train Adaptive Weights
    try:
        train_adaptive_weights()
    except Exception as e:
        logger.error(f"Adaptive weights training failed: {e}")
        success = False

    # 3. Train Self-Play
    try:
        train_selfplay()
    except Exception as e:
        logger.error(f"Self-play training failed: {e}")
        import traceback
        traceback.print_exc()
        success = False

    # 4. Train RL Agent
    try:
        train_rl_agent()
    except Exception as e:
        logger.error(f"RL agent training failed: {e}")
        success = False

    # 5. Verify
    try:
        verify_models()
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        success = False

    logger.info("\n" + "="*60)
    if success:
        logger.info("ALL TRAINING COMPLETE!")
    else:
        logger.info("TRAINING COMPLETED WITH SOME ERRORS")
    logger.info(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*60)

    return success


if __name__ == "__main__":
    main()
