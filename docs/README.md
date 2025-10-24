# AI Momentum Trading Bot - Documentation

## Architecture Documents

### [AI Momentum Trading Architecture (PDF)](AI_Momentum_Trading_Architecture.pdf)
**Last Updated:** 2025-10-24

Complete system architecture for the AI Momentum Trading Bot with IBKR TWS integration.

**Key Components:**
1. **Data Pipeline** - Real-time market data, order book, sentiment ingestion
2. **Feature Engineering** - Technical indicators, microstructure, sentiment analysis
3. **Model Architecture** - Fast lane (ML) + Slow lane (RL) models
4. **QOS Scorer** - Timeframe-weighted signal aggregator
5. **Slippage Feedback** - Adaptive order execution
6. **Risk Management** - PDT, margin, exposure controls
7. **Monitoring & UI** - Real-time visualization and alerts

**Tech Stack:**
- IBKR API (ib_insync, FastAPI)
- ML: LightGBM, PyTorch, ONNX Runtime
- NLP: FinBERT, DistilBERT
- Storage: DuckDB, Redis
- Visualization: Grafana, Custom UI

## Current Implementation Status

✅ **Completed:**
- IBKR TWS connection via ib_insync
- Real-time order placement (Market & Limit orders)
- Account monitoring dashboard
- Fast momentum trading interface
- AI prediction engine (12 technical indicators)
- Analytics and backtesting framework
- Order aggregation and management

🚧 **In Progress:**
- Extended hours trading support
- Slippage detection and feedback loop
- Sentiment analysis integration
- QOS scorer implementation
- Reinforcement learning policy model

## System Requirements

- Python 3.11+
- Interactive Brokers TWS/Gateway
- 8GB+ RAM recommended
- Windows 10/11

## Quick Start
```powershell
cd C:\ai_project_hub

# Create docs directory if it doesn't exist
New-Item -ItemType Directory -Path "docs" -Force | Out-Null

# Copy the PDF to docs
Copy-Item "C:\Users\bgmay\Downloads\AI_Momentum_Trading_Architecture.pdf" -Destination "docs\AI_Momentum_Trading_Architecture.pdf" -Force

# Create a README for the docs
@'
# AI Momentum Trading Bot - Documentation

## Architecture Documents

### [AI Momentum Trading Architecture (PDF)](AI_Momentum_Trading_Architecture.pdf)
**Last Updated:** 2025-10-24

Complete system architecture for the AI Momentum Trading Bot with IBKR TWS integration.

**Key Components:**
1. **Data Pipeline** - Real-time market data, order book, sentiment ingestion
2. **Feature Engineering** - Technical indicators, microstructure, sentiment analysis
3. **Model Architecture** - Fast lane (ML) + Slow lane (RL) models
4. **QOS Scorer** - Timeframe-weighted signal aggregator
5. **Slippage Feedback** - Adaptive order execution
6. **Risk Management** - PDT, margin, exposure controls
7. **Monitoring & UI** - Real-time visualization and alerts

**Tech Stack:**
- IBKR API (ib_insync, FastAPI)
- ML: LightGBM, PyTorch, ONNX Runtime
- NLP: FinBERT, DistilBERT
- Storage: DuckDB, Redis
- Visualization: Grafana, Custom UI

## Current Implementation Status

✅ **Completed:**
- IBKR TWS connection via ib_insync
- Real-time order placement (Market & Limit orders)
- Account monitoring dashboard
- Fast momentum trading interface
- AI prediction engine (12 technical indicators)
- Analytics and backtesting framework
- Order aggregation and management

🚧 **In Progress:**
- Extended hours trading support
- Slippage detection and feedback loop
- Sentiment analysis integration
- QOS scorer implementation
- Reinforcement learning policy model

## System Requirements

- Python 3.11+
- Interactive Brokers TWS/Gateway
- 8GB+ RAM recommended
- Windows 10/11

## Quick Start
```powershell
cd C:\ai_project_hub
.\.venv\Scripts\Activate.ps1
.\scripts\bootstrap.ps1
```

## API Endpoints

- **Trading:** http://127.0.0.1:9101/ui/trading.html
- **Account:** http://127.0.0.1:9101/ui/account.html
- **Analytics:** http://127.0.0.1:9101/ui/analytics.html
- **API Status:** http://127.0.0.1:9101/api/status

## Contact & Support

For questions or issues, refer to the architecture document or check the API documentation at `/api/docs`.
