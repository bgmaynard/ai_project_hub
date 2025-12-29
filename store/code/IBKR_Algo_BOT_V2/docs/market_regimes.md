# Market Regime Detection

## Overview

Market regime detection classifies current market conditions to filter trades. The system uses Amazon Chronos for zero-shot time series forecasting combined with technical indicators.

## Regime Types

| Regime | Description | Trading Implication |
|--------|-------------|---------------------|
| `TRENDING_UP` | Strong bullish momentum | Favorable for LONG entries |
| `TRENDING_DOWN` | Strong bearish momentum | Avoid LONG entries |
| `RANGING` | Sideways, low directional movement | Favor mean reversion |
| `VOLATILE` | High volatility, erratic moves | Reduce position size or avoid |

## Detection Method

### Chronos Adapter (`ai/chronos_adapter.py`)

```python
@dataclass
class ChronosContext:
    market_regime: str      # TRENDING_UP, TRENDING_DOWN, RANGING, VOLATILE
    regime_confidence: float  # 0.0 - 1.0
    prob_up: float          # Probability of upward move (informational)
    expected_return_5d: float
    current_volatility: float  # Annualized volatility
    trend_strength: float   # ADX-based, 0-100
    trend_direction: int    # -1 (down), 0 (neutral), 1 (up)
```

### Classification Logic

```python
def classify_regime(volatility, trend_strength, trend_direction, prob_up):
    # High volatility overrides everything
    if volatility > 0.5:  # 50% annualized
        return "VOLATILE"

    # Strong trend detection
    if trend_strength > 0.4:  # ADX > 40
        if trend_direction > 0:
            return "TRENDING_UP"
        elif trend_direction < 0:
            return "TRENDING_DOWN"

    # Weak trend = ranging
    return "RANGING"
```

## Indicators Used

### 1. Volatility (Annualized)
- Calculated from 20-day rolling standard deviation of returns
- Scaled to annual basis (×√252)
- Thresholds:
  - LOW: < 20%
  - NORMAL: 20-40%
  - HIGH: 40-60%
  - EXTREME: > 60%

### 2. Trend Strength (ADX-based)
- Average Directional Index normalized to 0-1
- `< 0.25` = Weak/No trend
- `0.25-0.40` = Moderate trend
- `> 0.40` = Strong trend

### 3. Trend Direction
- Based on 20-period price change
- Confirmed by EMA alignment (9/20/50)
- Values: -1, 0, +1

### 4. Probability of Up Move
- Chronos zero-shot forecast
- Uses last 100 price points
- Outputs probability distribution

## Integration with Signal Gating

```python
# In signal_gating_engine.py

def gate_signal(signal_contract, chronos_context):
    # Check if current regime is valid for this signal
    if chronos_context.market_regime in signal_contract.invalid_regimes:
        return VETO, "INVALID_REGIME"

    if chronos_context.market_regime not in signal_contract.valid_regimes:
        return VETO, "REGIME_MISMATCH"

    if chronos_context.regime_confidence < signal_contract.confidence_required:
        return VETO, "CONFIDENCE_LOW"

    return APPROVED, None
```

## Signal Contract Regime Rules

Each SignalContract specifies:
- `valid_regimes`: List of regimes where signal is allowed
- `invalid_regimes`: List of regimes that veto the signal

Example:
```json
{
  "signal_id": "AAPL_MOMENTUM_LONG",
  "valid_regimes": ["TRENDING_UP", "RANGING"],
  "invalid_regimes": ["TRENDING_DOWN", "VOLATILE"],
  "confidence_required": 0.6
}
```

## Exit Manager Integration

The Chronos Exit Manager (`ai/chronos_exit_manager.py`) uses regimes for smart exits:

| Exit Trigger | Condition |
|--------------|-----------|
| `REGIME_SHIFT` | Market moved from favorable to unfavorable regime |
| `CONFIDENCE_LOW` | Regime confidence dropped below 40% |
| `VOLATILITY_SPIKE` | Volatility exceeded 50% |
| `TREND_WEAK` | Trend strength fell below 30% |

## API Endpoints

```
GET /api/ai/regime/{symbol}
Response:
{
  "symbol": "AAPL",
  "regime": "TRENDING_UP",
  "confidence": 0.75,
  "volatility": 0.28,
  "trend_strength": 0.52,
  "trend_direction": 1,
  "prob_up": 0.62
}
```

## Configuration

```json
// chronos_exit_config.json
{
  "exit_on_regime_change": true,
  "favorable_regimes": ["TRENDING_UP", "RANGING"],
  "danger_regimes": ["TRENDING_DOWN", "VOLATILE"],
  "min_confidence": 0.4,
  "min_trend_strength": 0.3,
  "max_volatility": 0.5
}
```
