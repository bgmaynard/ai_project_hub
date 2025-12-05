"""
LLM Trade Reasoner - AI-Powered Trade Explanations
===================================================
Uses Claude/GPT to generate natural language explanations for trades.

WHAT THIS DOES:
- Takes trade context (price, indicators, news, history)
- Generates human-readable trade thesis
- Explains risk/reward in plain English
- Creates trade journal entries automatically

EXAMPLE OUTPUT:
"BUY AAPL at $175.50 (100 shares)

Thesis: Apple is showing strong momentum with MACD crossing bullish after
a pullback to the 20-day SMA. Volume is 1.5x average, confirming buyer interest.
RSI at 45 gives room to run. Earnings next week could be a catalyst.

Risk: Stop at $172 (-2%). Below the 20 SMA invalidates the setup.
Target: $182 (+3.7%) at previous resistance.
R:R Ratio: 1.85:1"
"""

import logging
import os
import json
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import pytz

logger = logging.getLogger(__name__)

# Try to import Anthropic SDK
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logger.warning("Anthropic SDK not installed. Install with: pip install anthropic")


@dataclass
class TradeReasoning:
    """Complete trade reasoning"""
    symbol: str
    action: str  # BUY, SELL, HOLD
    price: float
    timestamp: str

    # The reasoning
    thesis: str  # Main trade thesis
    technical_summary: str  # Technical analysis
    risk_analysis: str  # Risk factors
    target_price: float
    stop_price: float
    risk_reward_ratio: float

    # Confidence and warnings
    confidence_level: str  # HIGH, MEDIUM, LOW
    warnings: List[str]
    catalysts: List[str]

    # Full narrative
    full_narrative: str

    def to_dict(self) -> Dict:
        return asdict(self)

    def to_journal_entry(self) -> str:
        """Format as trade journal entry"""
        return f"""
═══════════════════════════════════════════════════════════
TRADE: {self.action} {self.symbol} @ ${self.price:.2f}
TIME: {self.timestamp}
═══════════════════════════════════════════════════════════

THESIS:
{self.thesis}

TECHNICAL ANALYSIS:
{self.technical_summary}

RISK MANAGEMENT:
• Stop Loss: ${self.stop_price:.2f} ({((self.stop_price - self.price) / self.price * 100):.1f}%)
• Target: ${self.target_price:.2f} ({((self.target_price - self.price) / self.price * 100):+.1f}%)
• Risk/Reward: {self.risk_reward_ratio:.2f}:1

CONFIDENCE: {self.confidence_level}

CATALYSTS:
{chr(10).join(f'• {c}' for c in self.catalysts) if self.catalysts else '• None identified'}

WARNINGS:
{chr(10).join(f'⚠️ {w}' for w in self.warnings) if self.warnings else '• None'}

═══════════════════════════════════════════════════════════
"""


class LLMTradeReasoner:
    """
    Generates natural language trade explanations using LLMs.

    Supports:
    - Claude (Anthropic) - Primary
    - GPT-4 (OpenAI) - Fallback
    - Rule-based - No API fallback
    """

    def __init__(self):
        self.et_tz = pytz.timezone('US/Eastern')

        # API keys
        self.anthropic_key = os.environ.get('ANTHROPIC_API_KEY', '')
        self.openai_key = os.environ.get('OPENAI_API_KEY', '')

        # Initialize client
        self.client = None
        if ANTHROPIC_AVAILABLE and self.anthropic_key:
            try:
                self.client = anthropic.Anthropic(api_key=self.anthropic_key)
                logger.info("LLM Trade Reasoner initialized with Claude")
            except Exception as e:
                logger.error(f"Failed to initialize Anthropic client: {e}")

        if not self.client:
            logger.warning("No LLM API available - using rule-based reasoning")

        # Reasoning cache
        self.reasoning_cache: Dict[str, TradeReasoning] = {}

        # Trade journal storage
        self.journal_path = os.path.join(
            os.path.dirname(__file__), "..", "store", "trade_journal.json"
        )

    def generate_trade_reasoning(self,
                                  symbol: str,
                                  action: str,
                                  price: float,
                                  indicators: Dict,
                                  history: Dict = None,
                                  news: List[str] = None) -> TradeReasoning:
        """
        Generate comprehensive trade reasoning.

        Args:
            symbol: Stock symbol
            action: BUY, SELL, or HOLD
            price: Current price
            indicators: Dict of technical indicators
            history: Optional historical performance for this symbol
            news: Optional recent news headlines

        Returns:
            TradeReasoning with full explanation
        """
        symbol = symbol.upper()
        action = action.upper()

        # Try LLM first, fall back to rules
        if self.client:
            reasoning = self._llm_reasoning(symbol, action, price, indicators, history, news)
        else:
            reasoning = self._rule_based_reasoning(symbol, action, price, indicators, history, news)

        # Cache and save to journal
        self.reasoning_cache[f"{symbol}_{action}"] = reasoning
        self._save_to_journal(reasoning)

        return reasoning

    def _llm_reasoning(self,
                       symbol: str,
                       action: str,
                       price: float,
                       indicators: Dict,
                       history: Dict = None,
                       news: List[str] = None) -> TradeReasoning:
        """Generate reasoning using Claude"""
        try:
            # Build context prompt
            prompt = self._build_prompt(symbol, action, price, indicators, history, news)

            # Call Claude
            response = self.client.messages.create(
                model="claude-3-haiku-20240307",  # Fast and cheap
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )

            # Parse response
            text = response.content[0].text
            return self._parse_llm_response(symbol, action, price, text, indicators)

        except Exception as e:
            logger.error(f"LLM reasoning failed: {e}")
            return self._rule_based_reasoning(symbol, action, price, indicators, history, news)

    def _build_prompt(self,
                      symbol: str,
                      action: str,
                      price: float,
                      indicators: Dict,
                      history: Dict = None,
                      news: List[str] = None) -> str:
        """Build prompt for LLM"""
        # Format indicators
        ind_str = "\n".join([f"- {k}: {v}" for k, v in indicators.items()])

        # Format history if available
        hist_str = ""
        if history:
            hist_str = f"""
Historical Performance:
- Win rate: {history.get('win_rate', 'N/A')}%
- Avg P&L: ${history.get('avg_pnl', 'N/A')}
- Last trade: {history.get('last_trade', 'N/A')}
"""

        # Format news if available
        news_str = ""
        if news:
            news_str = "Recent News:\n" + "\n".join([f"- {n}" for n in news[:5]])

        return f"""You are a professional day trader writing a trade journal entry.
Generate a concise trade reasoning for this setup:

Symbol: {symbol}
Action: {action}
Current Price: ${price:.2f}

Technical Indicators:
{ind_str}
{hist_str}
{news_str}

Respond in this exact JSON format:
{{
    "thesis": "2-3 sentence main trade thesis",
    "technical_summary": "Key technical factors supporting the trade",
    "risk_analysis": "Main risks to watch",
    "target_price": <number>,
    "stop_price": <number>,
    "confidence_level": "HIGH/MEDIUM/LOW",
    "warnings": ["warning1", "warning2"],
    "catalysts": ["catalyst1", "catalyst2"]
}}

Be specific and actionable. Use the actual indicator values in your reasoning.
For {action}, set target {'above' if action == 'BUY' else 'below'} current price.
Stop should limit loss to 2-3% max."""

    def _parse_llm_response(self,
                            symbol: str,
                            action: str,
                            price: float,
                            text: str,
                            indicators: Dict) -> TradeReasoning:
        """Parse LLM response into TradeReasoning"""
        try:
            # Extract JSON from response
            import re
            json_match = re.search(r'\{[\s\S]*\}', text)
            if json_match:
                data = json.loads(json_match.group())
            else:
                raise ValueError("No JSON found in response")

            # Calculate R:R
            target = float(data.get('target_price', price * 1.03))
            stop = float(data.get('stop_price', price * 0.97))

            if action == 'BUY':
                reward = target - price
                risk = price - stop
            else:
                reward = price - target
                risk = stop - price

            rr_ratio = reward / risk if risk > 0 else 0

            return TradeReasoning(
                symbol=symbol,
                action=action,
                price=price,
                timestamp=datetime.now(self.et_tz).isoformat(),
                thesis=data.get('thesis', ''),
                technical_summary=data.get('technical_summary', ''),
                risk_analysis=data.get('risk_analysis', ''),
                target_price=target,
                stop_price=stop,
                risk_reward_ratio=round(rr_ratio, 2),
                confidence_level=data.get('confidence_level', 'MEDIUM'),
                warnings=data.get('warnings', []),
                catalysts=data.get('catalysts', []),
                full_narrative=text
            )

        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            return self._rule_based_reasoning(symbol, action, price, indicators, None, None)

    def _rule_based_reasoning(self,
                              symbol: str,
                              action: str,
                              price: float,
                              indicators: Dict,
                              history: Dict = None,
                              news: List[str] = None) -> TradeReasoning:
        """Generate reasoning using rules (no LLM)"""
        # Extract key indicators
        rsi = indicators.get('rsi', 50)
        macd = indicators.get('macd', 0)
        macd_signal = indicators.get('macd_signal', 0)
        macd_hist = indicators.get('macd_histogram', 0)
        volume_ratio = indicators.get('volume_ratio', 1.0)
        trend = indicators.get('trend', 0)

        # Build thesis
        factors = []

        if action == 'BUY':
            if macd > macd_signal:
                factors.append("MACD bullish crossover")
            if rsi < 40:
                factors.append(f"RSI oversold at {rsi:.0f}")
            if volume_ratio > 1.5:
                factors.append(f"Volume surge ({volume_ratio:.1f}x average)")
            if trend > 0:
                factors.append("Uptrend confirmed")

            thesis = f"{symbol} showing bullish momentum. " + ", ".join(factors) + "."
            target = price * 1.03  # 3% target
            stop = price * 0.98  # 2% stop
        else:
            if macd < macd_signal:
                factors.append("MACD bearish crossover")
            if rsi > 60:
                factors.append(f"RSI overbought at {rsi:.0f}")
            if trend < 0:
                factors.append("Downtrend confirmed")

            thesis = f"{symbol} showing bearish signals. " + ", ".join(factors) + "."
            target = price * 0.97  # 3% target
            stop = price * 1.02  # 2% stop

        # Technical summary
        tech_summary = f"RSI: {rsi:.0f}, MACD Histogram: {macd_hist:.4f}, Volume: {volume_ratio:.1f}x"

        # Risk analysis
        warnings = []
        if volume_ratio < 1.0:
            warnings.append("Low volume - less conviction")
        if abs(macd_hist) < 0.01:
            warnings.append("Weak MACD histogram")
        if 45 < rsi < 55:
            warnings.append("RSI in neutral zone")

        # Confidence
        if len(factors) >= 3:
            confidence = "HIGH"
        elif len(factors) >= 2:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        # R:R calculation
        if action == 'BUY':
            rr = (target - price) / (price - stop) if (price - stop) > 0 else 0
        else:
            rr = (price - target) / (stop - price) if (stop - price) > 0 else 0

        return TradeReasoning(
            symbol=symbol,
            action=action,
            price=price,
            timestamp=datetime.now(self.et_tz).isoformat(),
            thesis=thesis,
            technical_summary=tech_summary,
            risk_analysis="Manage risk with stop loss. Scale out at target.",
            target_price=round(target, 2),
            stop_price=round(stop, 2),
            risk_reward_ratio=round(rr, 2),
            confidence_level=confidence,
            warnings=warnings,
            catalysts=factors[:3],
            full_narrative=thesis + "\n\n" + tech_summary
        )

    def _save_to_journal(self, reasoning: TradeReasoning):
        """Save reasoning to trade journal"""
        try:
            # Load existing journal
            journal = []
            if os.path.exists(self.journal_path):
                with open(self.journal_path, 'r') as f:
                    journal = json.load(f)

            # Add new entry
            journal.append(reasoning.to_dict())

            # Keep last 500 entries
            journal = journal[-500:]

            # Ensure directory exists
            os.makedirs(os.path.dirname(self.journal_path), exist_ok=True)

            # Save
            with open(self.journal_path, 'w') as f:
                json.dump(journal, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save to journal: {e}")

    def get_journal_entries(self, symbol: str = None, limit: int = 20) -> List[Dict]:
        """Get journal entries, optionally filtered by symbol"""
        try:
            if not os.path.exists(self.journal_path):
                return []

            with open(self.journal_path, 'r') as f:
                journal = json.load(f)

            if symbol:
                journal = [e for e in journal if e.get('symbol') == symbol.upper()]

            return journal[-limit:]

        except Exception as e:
            logger.error(f"Failed to load journal: {e}")
            return []

    def explain_exit(self,
                     symbol: str,
                     entry_price: float,
                     exit_price: float,
                     pnl: float,
                     reason: str) -> str:
        """Generate explanation for why a trade was exited"""
        symbol = symbol.upper()
        pnl_pct = ((exit_price - entry_price) / entry_price) * 100

        if pnl > 0:
            result = "WIN"
            emoji = "✅"
        else:
            result = "LOSS"
            emoji = "❌"

        explanation = f"""
{emoji} {result}: {symbol}
Entry: ${entry_price:.2f} → Exit: ${exit_price:.2f}
P&L: ${pnl:.2f} ({pnl_pct:+.2f}%)

Exit Reason: {reason}
"""
        return explanation


# Singleton instance
_trade_reasoner: Optional[LLMTradeReasoner] = None


def get_trade_reasoner() -> LLMTradeReasoner:
    """Get or create the trade reasoner singleton"""
    global _trade_reasoner
    if _trade_reasoner is None:
        _trade_reasoner = LLMTradeReasoner()
    return _trade_reasoner
