# store/code/IBKR_Algo_BOT/ai/alpha_fusion.py
from __future__ import annotations

from dataclasses import dataclass
from math import exp


@dataclass
class L1:
    bid: float
    ask: float
    bidSize: float | int = 0
    askSize: float | int = 0
    vwap: float | None = None


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + exp(-z))


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def predict_one(l1: L1, sentiment: float | None = None) -> dict:
    """
    Minimal AlphaFusion: uses book imbalance, spread and a sentiment hint.
    This is intentionally tiny so it runs without extra deps; we can swap in the
    full model later. See docs for the complete design.
    """
    mid = (l1.bid + l1.ask) / 2.0 if (l1.bid and l1.ask) else None
    spr = (l1.ask - l1.bid) / mid if (mid and l1.ask > 0 and l1.bid > 0) else 0.0
    # top-of-book imbalance ~ [-1,1]
    den = (float(l1.bidSize) + float(l1.askSize)) or 1.0
    imb = (float(l1.bidSize) - float(l1.askSize)) / den

    s = 0.0 if sentiment is None else max(-1.0, min(1.0, float(sentiment)))

    # Tiny logistic: β0 + β1*imb + β2*(-spr) + β3*sentiment
    z = 0.0 + 1.2 * imb + 2.0 * (-spr) + 0.8 * s
    p_up = _clip(_sigmoid(z))
    # quick reliability proxy (higher spread -> lower reliability)
    reliability = _clip(1.0 - min(spr / 0.01, 1.0))  # 1 inside ~1c spread on $100

    return {
        "p_up": p_up,
        "reliability": reliability,
        "features": {"imb": imb, "spr": spr, "sentiment": s},
        "mid": mid,
    }
