"""
AI Router Module for IBKR Algo Bot
Provides AI prediction endpoints with CSV logging
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
import csv
import os
from pathlib import Path

router = APIRouter()

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

class PredictionResponse(BaseModel):
    timestamp: str
    symbol: str
    prediction: str
    confidence: float
    signal: str
    explanation: str

class PredictionHistory(BaseModel):
    timestamp: str
    symbol: str
    prediction: str
    confidence: float
    signal: str

# ==================== PREDICTION ENDPOINTS ====================

@router.post("/predict", response_model=PredictionResponse)
async def make_prediction(request: PredictionRequest):
    """
    Make a trading prediction using AI model
    Returns prediction with confidence score and trading signal
    """
    import random
    
    # Mock prediction logic - replace with actual AI model
    predictions = ["bullish", "bearish", "neutral"]
    prediction = random.choice(predictions)
    confidence = round(random.uniform(0.6, 0.95), 2)
    
    # Generate trading signal
    if prediction == "bullish" and confidence > 0.75:
        signal = "BUY"
    elif prediction == "bearish" and confidence > 0.75:
        signal = "SELL"
    else:
        signal = "HOLD"
    
    # Create response
    response = PredictionResponse(
        timestamp=datetime.now().isoformat(),
        symbol=request.symbol.upper(),
        prediction=prediction,
        confidence=confidence,
        signal=signal,
        explanation=f"AI model predicts {prediction} trend with {confidence*100:.0f}% confidence"
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

@router.get("/predict/last")
async def get_last_prediction(symbol: Optional[str] = None):
    """
    Get the most recent prediction
    Optionally filter by symbol
    """
    if not PREDICTIONS_CSV.exists():
        raise HTTPException(status_code=404, detail="No predictions found")
    
    try:
        with open(PREDICTIONS_CSV, 'r') as f:
            reader = csv.DictReader(f)
            predictions = list(reader)
        
        if not predictions:
            raise HTTPException(status_code=404, detail="No predictions found")
        
        # Filter by symbol if provided
        if symbol:
            predictions = [p for p in predictions if p['symbol'].upper() == symbol.upper()]
            if not predictions:
                raise HTTPException(status_code=404, detail=f"No predictions found for {symbol}")
        
        # Return the last prediction
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
    """
    Get prediction history
    Optionally filter by symbol and limit results
    """
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
        
        # Filter by symbol if provided
        if symbol:
            predictions = [p for p in predictions if p['symbol'].upper() == symbol.upper()]
        
        # Get the last N predictions
        predictions = predictions[-limit:]
        
        # Convert to proper format
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
        raise HTTPException(status_code=500, detail=f"Error reading predictions: {str(e)}")

@router.get("/status")
async def get_ai_status():
    """
    Get AI model status and statistics
    """
    stats = {
        "status": "operational",
        "model_version": "1.0.0-mock",
        "model_type": "EnhancedAIPredictor",
        "predictions_logged": 0,
        "csv_path": str(PREDICTIONS_CSV),
        "last_prediction": None
    }
    
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

# ==================== TRAINING ENDPOINTS (PLACEHOLDER) ====================

@router.post("/train")
async def train_model(training_data: dict):
    """
    Train/retrain the AI model
    TODO: Integrate with EnhancedAIPredictor
    """
    return {
        "status": "pending",
        "message": "Training endpoint - integrate with EnhancedAIPredictor",
        "note": "This is a placeholder. Wire to your actual training logic."
    }

@router.post("/backtest")
async def run_backtest(backtest_config: dict):
    """
    Run backtest on historical data
    TODO: Integrate with backtesting engine
    """
    return {
        "status": "pending",
        "message": "Backtest endpoint - integrate with backtesting engine",
        "note": "This is a placeholder. Wire to your actual backtest logic."
    }
