"""
AI Router Module for IBKR Algo Bot
Provides AI prediction endpoints with CSV logging
FULLY WIRED to EnhancedAIPredictor and Backtester
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
import csv
import sys
from pathlib import Path

router = APIRouter()

# Add ai directory to path
ai_dir = Path(__file__).parent.parent / "ai"
sys.path.insert(0, str(ai_dir))

try:
    from ai_predictor import EnhancedAIPredictor
    ai_predictor = EnhancedAIPredictor()
    print("[OK] EnhancedAIPredictor loaded")
except Exception as e:
    print(f"[WARNING] Could not load EnhancedAIPredictor: {e}")
    ai_predictor = None

try:
    from backtester import Backtester
    backtester = Backtester()
    print("[OK] Backtester loaded")
except Exception as e:
    print(f"[WARNING] Could not load Backtester: {e}")
    backtester = None


# Data directory for CSV storage
DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)
PREDICTIONS_CSV = DATA_DIR / "predictions.csv"

# Initialize CSV file with headers if it doesn't exist
if not PREDICTIONS_CSV.exists():
    with open(PREDICTIONS_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'symbol', 'prediction', 'confidence', 'signal', 'features'])

# Pydantic models
class PredictionRequest(BaseModel):
    symbol: str
    timeframe: Optional[str] = "5min"
    features: Optional[dict] = None
    bidSize: Optional[float] = None
    askSize: Optional[float] = None
    sentiment: Optional[float] = None

class TrainRequest(BaseModel):
    symbol: str
    period: str = "2y"
    retrain: bool = False

class BacktestRequest(BaseModel):
    symbol: str
    period: str = "6mo"
    initial_capital: float = 10000.0
    commission: float = 0.001

class PredictionResponse(BaseModel):
    timestamp: str
    symbol: str
    prediction: str
    confidence: float
    signal: str
    explanation: str
    prob_up: Optional[float] = None
    fallback: bool = False

class PredictionHistory(BaseModel):
    timestamp: str
    symbol: str
    prediction: str
    confidence: float
    signal: str

# Helper functions
def calculate_fallback_prediction(symbol: str, bidSize: float = None, askSize: float = None, sentiment: float = None):
    """Fallback prediction using order book imbalance"""
    prob_up = 0.5

    if bidSize is not None and askSize is not None and (bidSize + askSize) > 0:
        imbalance = (bidSize - askSize) / (bidSize + askSize)
        prob_up = 0.5 + (imbalance * 0.3)

    if sentiment is not None:
        prob_up = prob_up * 0.7 + sentiment * 0.3

    prob_up = max(0.0, min(1.0, prob_up))

    if prob_up > 0.55:
        signal = "bullish"
        prediction = "BUY"
    elif prob_up < 0.45:
        signal = "bearish"
        prediction = "SELL"
    else:
        signal = "neutral"
        prediction = "HOLD"

    confidence = abs(prob_up - 0.5) * 2

    return {
        "symbol": symbol,
        "signal": signal,
        "prediction": prediction,
        "prob_up": round(prob_up, 4),
        "confidence": round(max(0.65, confidence), 2),
        "fallback": True
    }

# ==================== PREDICTION ENDPOINTS ====================

@router.post("/predict", response_model=PredictionResponse)
async def make_prediction(request: PredictionRequest):
    """
    Make a trading prediction using AI model
    Returns prediction with confidence score and trading signal
    """
    try:
        result = None

        # Try to use trained AI predictor
        if ai_predictor:
            try:
                result = ai_predictor.predict(
                    symbol=request.symbol,
                    bidSize=request.bidSize,
                    askSize=request.askSize,
                    sentiment=request.sentiment
                )
                result["fallback"] = False
            except Exception as e:
                print(f"AI predictor failed ({e}), using fallback")
                result = None

        # Fallback if AI prediction failed or unavailable
        if not result:
            result = calculate_fallback_prediction(
                request.symbol,
                request.bidSize,
                request.askSize,
                request.sentiment
            )

        # Map to standard format
        if "prediction" not in result:
            result["prediction"] = result.get("signal", "HOLD").upper()

        # Create response
        response = PredictionResponse(
            timestamp=datetime.now().isoformat(),
            symbol=result["symbol"],
            prediction=result["prediction"],
            confidence=result["confidence"],
            signal=result.get("signal", "neutral"),
            prob_up=result.get("prob_up", 0.5),
            fallback=result.get("fallback", False),
            explanation=f"AI predicts {result.get('signal', 'neutral')} with {result['confidence']*100:.0f}% confidence"
        )

        # Log to CSV
        try:
            with open(PREDICTIONS_CSV, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    response.timestamp,
                    response.symbol,
                    response.prediction,
                    response.confidence,
                    response.signal,
                    str(request.features) if request.features else ""
                ])
        except Exception as e:
            print(f"Warning: Failed to log prediction to CSV: {e}")

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@router.get("/predict/last")
async def get_last_prediction(symbol: Optional[str] = None):
    """Get the most recent prediction"""
    if not PREDICTIONS_CSV.exists():
        raise HTTPException(status_code=404, detail="No predictions found")

    try:
        with open(PREDICTIONS_CSV, 'r') as f:
            reader = csv.DictReader(f)
            predictions = list(reader)

        if not predictions:
            raise HTTPException(status_code=404, detail="No predictions found")

        if symbol:
            predictions = [p for p in predictions if p['symbol'].upper() == symbol.upper()]
            if not predictions:
                raise HTTPException(status_code=404, detail=f"No predictions found for {symbol}")

        last = predictions[-1]
        return {
            "timestamp": last['timestamp'],
            "symbol": last['symbol'],
            "prediction": last['prediction'],
            "confidence": float(last['confidence']),
            "signal": last['signal']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading predictions: {str(e)}")

@router.get("/predict/history")
async def get_prediction_history(
    limit: int = Query(default=20, ge=1, le=100),
    symbol: Optional[str] = None
):
    """Get prediction history"""
    if not PREDICTIONS_CSV.exists():
        return {
            "predictions": [],
            "count": 0,
            "message": "No predictions yet"
        }

    try:
        with open(PREDICTIONS_CSV, 'r') as f:
            reader = csv.DictReader(f)
            predictions = list(reader)

        if symbol:
            predictions = [p for p in predictions if p['symbol'].upper() == symbol.upper()]

        predictions = predictions[-limit:]

        history = [
            {
                "timestamp": p['timestamp'],
                "symbol": p['symbol'],
                "prediction": p['prediction'],
                "confidence": float(p['confidence']),
                "signal": p['signal']
            }
            for p in predictions
        ]

        return {
            "predictions": history,
            "count": len(history),
            "symbol_filter": symbol.upper() if symbol else None
        }
    except Exception as e:
       import traceback
       print(f"Prediction error: {e}")
       print("Full traceback:")
       traceback.print_exc()
       raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@router.get("/status")
async def get_ai_status():
    """Get AI model status and statistics"""
    stats = {
        "status": "operational",
        "model_version": "1.0.0",
        "ai_predictor": {
            "available": ai_predictor is not None,
            "model_trained": False,
            "model_info": {}
        },
        "backtester": {
            "available": backtester is not None
        },
        "predictions_logged": 0,
        "csv_path": str(PREDICTIONS_CSV),
        "last_prediction": None
    }

    # Check if model is trained
    if ai_predictor:
        try:
            model_info = ai_predictor.get_model_info()
            stats["ai_predictor"]["model_trained"] = model_info.get("trained", False)
            stats["ai_predictor"]["model_info"] = model_info
        except Exception:
            pass

    # Count predictions in CSV
    if PREDICTIONS_CSV.exists():
        try:
            with open(PREDICTIONS_CSV, 'r') as f:
                reader = csv.DictReader(f)
                predictions = list(reader)
                stats["predictions_logged"] = len(predictions)

                if predictions:
                    last = predictions[-1]
                    stats["last_prediction"] = {
                        "timestamp": last['timestamp'],
                        "symbol": last['symbol'],
                        "prediction": last['prediction'],
                        "confidence": float(last['confidence'])
                    }
        except Exception as e:
            stats["error"] = f"Error reading CSV: {str(e)}"

    return stats

# ==================== TRAINING ENDPOINT (WIRED) ====================

@router.post("/train")
async def train_model(request: TrainRequest):
    """
    Train the AI model on historical data
    WIRED to EnhancedAIPredictor
    """
    try:
        if not ai_predictor:
            raise HTTPException(
                status_code=503,
                detail="AI Predictor not available. Check server logs."
            )

        print(f"Training model on {request.symbol} ({request.period})")

        # Train the model
        result = ai_predictor.train(
            symbol=request.symbol,
            period=request.period
        )

        if result.get("success"):
            print("Training completed successfully")
            return {
                "status": "success",
                "message": f"Model trained on {request.symbol}",
                "symbol": request.symbol,
                "period": request.period,
                "metrics": result.get("metrics", {}),
                "training_samples": result.get("samples", 0),
                "model_path": result.get("model_path", ""),
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(
                status_code=500,
                detail=result.get("error", "Training failed")
            )

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"Training error: {e}")
        print("Full traceback:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Training error: {str(e)}")# ==================== BACKTEST ENDPOINT (WIRED) ====================

@router.post("/backtest")
async def run_backtest(request: BacktestRequest):
    """
    Run backtest on historical data
    WIRED to Backtester
    """
    try:
        if not backtester:
            raise HTTPException(
                status_code=503,
                detail="Backtester not available. Check server logs."
            )

        if not ai_predictor:
            raise HTTPException(
                status_code=503,
                detail="AI Predictor not available. Train a model first."
            )

        print(f"Running backtest for {request.symbol} ({request.period})")

        # Run backtest
        result = backtester.run(
            symbol=request.symbol,
            period=request.period,
            initial_capital=request.initial_capital,
            commission=request.commission,
            predictor=ai_predictor
        )

        if result.get("success"):
            print("Backtest completed successfully")

            metrics = result.get("metrics", {})

            return {
                "status": "success",
                "symbol": request.symbol,
                "period": request.period,
                "initial_capital": request.initial_capital,
                "final_capital": metrics.get("final_capital", 0),
                "total_return": metrics.get("total_return_pct", 0),
                "sharpe_ratio": metrics.get("sharpe_ratio", 0),
                "max_drawdown": metrics.get("max_drawdown_pct", 0),
                "win_rate": metrics.get("win_rate", 0),
                "total_trades": metrics.get("total_trades", 0),
                "profitable_trades": metrics.get("profitable_trades", 0),
                "avg_profit": metrics.get("avg_profit", 0),
                "avg_loss": metrics.get("avg_loss", 0),
                "timestamp": datetime.now().isoformat(),
                "trades": result.get("trades", [])[:100]
            }
        else:
            raise HTTPException(
                status_code=500,
                detail=result.get("error", "Backtest failed")
            )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Backtest error: {e}")
        raise HTTPException(status_code=500, detail=f"Backtest error: {str(e)}")

