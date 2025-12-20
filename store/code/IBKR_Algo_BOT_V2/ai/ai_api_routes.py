"""
AI API Routes
==============
FastAPI routes for AI prediction endpoints.

Provides access to:
- Chronos (Amazon foundation model)
- LightGBM predictor
- Ensemble predictor (combines all models)
- Model comparison and benchmarking
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List
import logging
import yfinance as yf
import pandas as pd

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/api/ai/status")
async def get_ai_status():
    """Get status of all AI models."""
    status = {
        "lightgbm": {"available": False, "trained": False},
        "chronos": {"available": False, "model": None},
        "ensemble": {"available": False, "components": []},
    }

    # Check LightGBM
    try:
        from ai.ai_predictor import get_predictor
        predictor = get_predictor()
        status["lightgbm"]["available"] = True
        status["lightgbm"]["trained"] = predictor.model is not None
        if predictor.model:
            status["lightgbm"]["accuracy"] = predictor.accuracy
            status["lightgbm"]["training_date"] = predictor.training_date
    except Exception as e:
        status["lightgbm"]["error"] = str(e)

    # Check Chronos
    try:
        from ai.chronos_predictor import get_chronos_predictor
        chronos = get_chronos_predictor()
        status["chronos"]["available"] = True
        status["chronos"]["model"] = chronos.model_name
    except Exception as e:
        status["chronos"]["error"] = str(e)

    # Check Ensemble
    try:
        from ai.ensemble_predictor import get_ensemble_predictor
        ensemble = get_ensemble_predictor()
        status["ensemble"]["available"] = True
        components = []
        if ensemble.lgb_predictor and ensemble.lgb_predictor.model:
            components.append("lightgbm")
        if ensemble.chronos_scorer.available:
            components.append("chronos")
        components.extend(["heuristic", "momentum"])
        if ensemble.rl_agent:
            components.append("rl_agent")
        status["ensemble"]["components"] = components
        status["ensemble"]["weights"] = ensemble.base_weights
    except Exception as e:
        status["ensemble"]["error"] = str(e)

    return status


@router.get("/api/ai/predict/chronos/{symbol}")
async def predict_chronos(
    symbol: str,
    horizon: int = Query(default=5, ge=1, le=20, description="Forecast horizon in days"),
    num_samples: int = Query(default=20, ge=5, le=100, description="Number of sample paths")
):
    """
    Get Chronos foundation model prediction for a symbol.

    Uses Amazon's Chronos transformer model for zero-shot time series forecasting.
    Returns probabilistic forecast with confidence intervals.
    """
    try:
        from ai.chronos_predictor import get_chronos_predictor
        predictor = get_chronos_predictor()
        result = predictor.predict(symbol.upper(), horizon=horizon, num_samples=num_samples)
        return result
    except Exception as e:
        logger.error(f"Chronos prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/ai/predict/lightgbm/{symbol}")
async def predict_lightgbm(symbol: str):
    """
    Get LightGBM prediction for a symbol.

    Uses technical indicators to predict if price goes up 1% in next 3 days.
    """
    try:
        from ai.ai_predictor import get_predictor
        predictor = get_predictor()

        if predictor.model is None:
            raise HTTPException(status_code=400, detail="LightGBM model not trained")

        result = predictor.predict(symbol.upper())
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"LightGBM prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/ai/predict/ensemble/{symbol}")
async def predict_ensemble(
    symbol: str,
    horizon: int = Query(default=5, ge=1, le=20, description="Forecast horizon in days")
):
    """
    Get ensemble prediction combining all AI models.

    Combines:
    - LightGBM (technical indicators)
    - Chronos (foundation model)
    - Heuristics (pattern recognition)
    - Momentum scoring

    Returns weighted consensus signal.
    """
    try:
        from ai.ensemble_predictor import get_ensemble_predictor
        predictor = get_ensemble_predictor()

        # Fetch data
        ticker = yf.Ticker(symbol.upper())
        df = ticker.history(period="6mo")

        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data for {symbol}")

        # Handle multi-index columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

        result = predictor.predict(symbol.upper(), df)
        return result.to_dict()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ensemble prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/ai/predict/batch")
async def predict_batch(
    symbols: List[str],
    model: str = Query(default="ensemble", description="Model to use: ensemble, chronos, lightgbm")
):
    """
    Get predictions for multiple symbols.
    """
    results = []

    for symbol in symbols[:20]:  # Limit to 20 symbols
        try:
            if model == "chronos":
                from ai.chronos_predictor import get_chronos_predictor
                predictor = get_chronos_predictor()
                result = predictor.predict(symbol.upper())
            elif model == "lightgbm":
                from ai.ai_predictor import get_predictor
                predictor = get_predictor()
                result = predictor.predict(symbol.upper())
            else:  # ensemble
                from ai.ensemble_predictor import get_ensemble_predictor
                predictor = get_ensemble_predictor()
                ticker = yf.Ticker(symbol.upper())
                df = ticker.history(period="6mo")
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(1)
                result = predictor.predict(symbol.upper(), df).to_dict()

            results.append(result)
        except Exception as e:
            results.append({"symbol": symbol, "error": str(e)})

    return {"model": model, "predictions": results}


@router.get("/api/ai/compare/{symbol}")
async def compare_models(symbol: str):
    """
    Compare predictions from all available models for a symbol.

    Useful for evaluating model agreement and identifying opportunities.
    """
    comparison = {
        "symbol": symbol.upper(),
        "models": {},
        "consensus": None,
        "recommendation": None,
    }

    # Chronos
    try:
        from ai.chronos_predictor import get_chronos_predictor
        predictor = get_chronos_predictor()
        result = predictor.predict(symbol.upper())
        comparison["models"]["chronos"] = {
            "signal": result.get("signal"),
            "prob_up": result.get("probabilities", {}).get("prob_up"),
            "expected_return": result.get("expected_return_pct"),
            "confidence": result.get("confidence"),
        }
    except Exception as e:
        comparison["models"]["chronos"] = {"error": str(e)}

    # LightGBM
    try:
        from ai.ai_predictor import get_predictor
        predictor = get_predictor()
        if predictor.model:
            result = predictor.predict(symbol.upper())
            comparison["models"]["lightgbm"] = {
                "signal": result.get("signal"),
                "prob_up": result.get("prob_up"),
                "confidence": result.get("confidence"),
            }
        else:
            comparison["models"]["lightgbm"] = {"error": "Model not trained"}
    except Exception as e:
        comparison["models"]["lightgbm"] = {"error": str(e)}

    # Ensemble
    try:
        from ai.ensemble_predictor import get_ensemble_predictor
        predictor = get_ensemble_predictor()
        ticker = yf.Ticker(symbol.upper())
        df = ticker.history(period="6mo")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        result = predictor.predict(symbol.upper(), df)
        comparison["models"]["ensemble"] = {
            "prediction": "BULLISH" if result.prediction == 1 else "BEARISH",
            "confidence": result.confidence,
            "regime": result.market_regime,
        }
    except Exception as e:
        comparison["models"]["ensemble"] = {"error": str(e)}

    # Calculate consensus
    bullish_count = 0
    bearish_count = 0
    total = 0

    for model, data in comparison["models"].items():
        if "error" in data:
            continue
        total += 1
        signal = data.get("signal") or data.get("prediction", "")
        if "BULLISH" in str(signal).upper() or "BUY" in str(signal).upper():
            bullish_count += 1
        elif "BEARISH" in str(signal).upper() or "SELL" in str(signal).upper():
            bearish_count += 1

    if total > 0:
        if bullish_count == total:
            comparison["consensus"] = "UNANIMOUS_BULLISH"
            comparison["recommendation"] = "STRONG_BUY"
        elif bearish_count == total:
            comparison["consensus"] = "UNANIMOUS_BEARISH"
            comparison["recommendation"] = "STRONG_SELL"
        elif bullish_count > bearish_count:
            comparison["consensus"] = "MAJORITY_BULLISH"
            comparison["recommendation"] = "BUY"
        elif bearish_count > bullish_count:
            comparison["consensus"] = "MAJORITY_BEARISH"
            comparison["recommendation"] = "SELL"
        else:
            comparison["consensus"] = "MIXED"
            comparison["recommendation"] = "HOLD"

    return comparison


@router.post("/api/ai/train/lightgbm")
async def train_lightgbm(
    symbol: str = Query(default="SPY", description="Symbol to train on"),
    period: str = Query(default="2y", description="Training data period")
):
    """
    Train or retrain the LightGBM model.
    """
    try:
        from ai.ai_predictor import EnhancedAIPredictor
        predictor = EnhancedAIPredictor()
        result = predictor.train(symbol.upper(), period=period)
        return result
    except Exception as e:
        logger.error(f"LightGBM training failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/ai/train/lightgbm/multi")
async def train_lightgbm_multi(
    symbols: List[str] = Query(
        default=["SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA"],
        description="Symbols to train on"
    ),
    period: str = Query(default="2y", description="Training data period")
):
    """
    Train LightGBM model on multiple symbols for better generalization.
    """
    try:
        from ai.ai_predictor import EnhancedAIPredictor
        predictor = EnhancedAIPredictor()
        result = predictor.train_multi([s.upper() for s in symbols], period=period)
        return result
    except Exception as e:
        logger.error(f"LightGBM multi-training failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
