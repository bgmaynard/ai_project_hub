"""
Claude AI Integration Module

Provides centralized Claude API access for all AI-powered features:
- Performance analysis
- Strategy optimization
- Market regime detection
- Error diagnosis and recovery
- Real-time insights generation

Features:
- Rate limiting and cost tracking
- Request caching for efficiency
- Automatic retry logic
- Comprehensive error handling
"""

import hashlib
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from anthropic import Anthropic, APIConnectionError, APIError, RateLimitError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ClaudeRequest:
    """Represents a Claude API request"""

    request_type: str
    prompt: str
    max_tokens: int = 4096
    temperature: float = 0.7
    system_prompt: Optional[str] = None


@dataclass
class ClaudeResponse:
    """Represents a Claude API response"""

    content: str
    tokens_used: int
    cost_usd: float
    response_time_ms: int
    success: bool
    error: Optional[str] = None


@dataclass
class UsageStats:
    """Tracks API usage and costs"""

    total_requests: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    last_request_time: Optional[datetime] = None
    daily_requests: int = 0
    daily_cost_usd: float = 0.0
    last_reset_date: Optional[str] = None


class CostLimitExceeded(Exception):
    """Raised when cost limits are exceeded"""

    pass


class ClaudeIntegration:
    """
    Centralized Claude API integration for Warrior Trading system

    Handles all interactions with Claude API including:
    - Request management with rate limiting
    - Cost tracking and budget enforcement
    - Response caching
    - Error handling and retries
    """

    # Pricing (per million tokens) for Claude Sonnet 4.5
    INPUT_PRICE_PER_MTK = 3.00  # $3 per million input tokens
    OUTPUT_PRICE_PER_MTK = 15.00  # $15 per million output tokens

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_requests_per_minute: int = 50,
        daily_cost_limit: float = 10.00,
        monthly_cost_limit: float = 200.00,
        cache_enabled: bool = True,
        cache_ttl_seconds: int = 300,
    ):
        """
        Initialize Claude integration

        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            max_requests_per_minute: Rate limit for API calls
            daily_cost_limit: Maximum daily spend in USD
            monthly_cost_limit: Maximum monthly spend in USD
            cache_enabled: Whether to cache responses
            cache_ttl_seconds: Cache time-to-live in seconds
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            logger.warning(
                "No Anthropic API key found - Claude features will be disabled"
            )
            self.client = None
        else:
            self.client = Anthropic(api_key=self.api_key)

        self.max_requests_per_minute = max_requests_per_minute
        self.daily_cost_limit = daily_cost_limit
        self.monthly_cost_limit = monthly_cost_limit
        self.cache_enabled = cache_enabled
        self.cache_ttl_seconds = cache_ttl_seconds

        # Request tracking for rate limiting
        self.request_times: List[float] = []

        # Response cache {hash: (response, timestamp)}
        self.response_cache: Dict[str, tuple] = {}

        # Usage statistics
        self.usage_stats = UsageStats()
        self._load_usage_stats()

        # Model configuration
        self.model = "claude-sonnet-4-5-20250929"

        logger.info(f"Claude integration initialized (model: {self.model})")

    def _load_usage_stats(self):
        """Load usage statistics from file"""
        stats_file = Path("data/claude_usage_stats.json")
        if stats_file.exists():
            try:
                with open(stats_file, "r") as f:
                    data = json.load(f)
                    self.usage_stats = UsageStats(**data)

                # Reset daily stats if it's a new day
                today = datetime.now().strftime("%Y-%m-%d")
                if self.usage_stats.last_reset_date != today:
                    self.usage_stats.daily_requests = 0
                    self.usage_stats.daily_cost_usd = 0.0
                    self.usage_stats.last_reset_date = today
                    self._save_usage_stats()

                logger.info(
                    f"Loaded usage stats: {self.usage_stats.total_requests} requests, ${self.usage_stats.total_cost_usd:.2f} total"
                )
            except Exception as e:
                logger.error(f"Error loading usage stats: {e}")

    def _save_usage_stats(self):
        """Save usage statistics to file"""
        stats_file = Path("data/claude_usage_stats.json")
        stats_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(stats_file, "w") as f:
                json.dump(asdict(self.usage_stats), f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving usage stats: {e}")

    def _check_rate_limit(self):
        """Check if we're within rate limits"""
        current_time = time.time()

        # Remove requests older than 1 minute
        self.request_times = [t for t in self.request_times if current_time - t < 60]

        # Check if we've hit the limit
        if len(self.request_times) >= self.max_requests_per_minute:
            wait_time = 60 - (current_time - self.request_times[0])
            logger.warning(f"Rate limit reached, waiting {wait_time:.1f}s")
            time.sleep(wait_time)
            self.request_times = []

        self.request_times.append(current_time)

    def _check_cost_limits(self, estimated_cost: float = 0.01):
        """
        Check if request would exceed cost limits

        Args:
            estimated_cost: Estimated cost of the request in USD

        Raises:
            CostLimitExceeded: If cost limits would be exceeded
        """
        # Check daily limit
        if self.usage_stats.daily_cost_usd + estimated_cost > self.daily_cost_limit:
            raise CostLimitExceeded(
                f"Daily cost limit of ${self.daily_cost_limit} would be exceeded "
                f"(current: ${self.usage_stats.daily_cost_usd:.2f})"
            )

        # Check monthly limit (approximate)
        if self.usage_stats.total_cost_usd + estimated_cost > self.monthly_cost_limit:
            raise CostLimitExceeded(
                f"Monthly cost limit of ${self.monthly_cost_limit} would be exceeded "
                f"(current: ${self.usage_stats.total_cost_usd:.2f})"
            )

    def _get_cache_key(self, request: ClaudeRequest) -> str:
        """Generate cache key for request"""
        key_data = f"{request.request_type}:{request.prompt}:{request.temperature}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _get_cached_response(self, cache_key: str) -> Optional[ClaudeResponse]:
        """Retrieve cached response if available and not expired"""
        if not self.cache_enabled:
            return None

        if cache_key in self.response_cache:
            response, timestamp = self.response_cache[cache_key]
            age = (datetime.now() - timestamp).total_seconds()

            if age < self.cache_ttl_seconds:
                logger.info(f"Cache hit (age: {age:.1f}s)")
                return response
            else:
                # Remove expired cache entry
                del self.response_cache[cache_key]

        return None

    def _cache_response(self, cache_key: str, response: ClaudeResponse):
        """Cache a response"""
        if self.cache_enabled:
            self.response_cache[cache_key] = (response, datetime.now())

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Calculate cost in USD for a request

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Cost in USD
        """
        input_cost = (input_tokens / 1_000_000) * self.INPUT_PRICE_PER_MTK
        output_cost = (output_tokens / 1_000_000) * self.OUTPUT_PRICE_PER_MTK
        return input_cost + output_cost

    def _update_usage_stats(self, response: ClaudeResponse):
        """Update usage statistics"""
        self.usage_stats.total_requests += 1
        self.usage_stats.total_tokens += response.tokens_used
        self.usage_stats.total_cost_usd += response.cost_usd
        self.usage_stats.daily_requests += 1
        self.usage_stats.daily_cost_usd += response.cost_usd
        self.usage_stats.last_request_time = datetime.now()

        today = datetime.now().strftime("%Y-%m-%d")
        if self.usage_stats.last_reset_date != today:
            self.usage_stats.daily_requests = 1
            self.usage_stats.daily_cost_usd = response.cost_usd
            self.usage_stats.last_reset_date = today

        self._save_usage_stats()

    def request(
        self, request: ClaudeRequest, use_cache: bool = True, max_retries: int = 3
    ) -> ClaudeResponse:
        """
        Make a request to Claude API

        Args:
            request: ClaudeRequest object
            use_cache: Whether to use cached responses
            max_retries: Maximum number of retry attempts

        Returns:
            ClaudeResponse object

        Raises:
            CostLimitExceeded: If cost limits would be exceeded
            APIError: If API request fails after retries
        """
        if not self.client:
            return ClaudeResponse(
                content="Claude API not configured - please set ANTHROPIC_API_KEY",
                tokens_used=0,
                cost_usd=0.0,
                response_time_ms=0,
                success=False,
                error="API key not configured",
            )

        # Check cache first
        if use_cache:
            cache_key = self._get_cache_key(request)
            cached = self._get_cached_response(cache_key)
            if cached:
                return cached

        # Check cost limits
        self._check_cost_limits()

        # Rate limiting
        self._check_rate_limit()

        # Make request with retries
        last_error = None
        for attempt in range(max_retries):
            try:
                start_time = time.time()

                # Prepare messages
                messages = [{"role": "user", "content": request.prompt}]

                # Make API call
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    system=request.system_prompt if request.system_prompt else "",
                    messages=messages,
                )

                # Calculate metrics
                response_time_ms = int((time.time() - start_time) * 1000)
                input_tokens = response.usage.input_tokens
                output_tokens = response.usage.output_tokens
                total_tokens = input_tokens + output_tokens
                cost = self._calculate_cost(input_tokens, output_tokens)

                # Extract content
                content = response.content[0].text if response.content else ""

                # Create response object
                claude_response = ClaudeResponse(
                    content=content,
                    tokens_used=total_tokens,
                    cost_usd=cost,
                    response_time_ms=response_time_ms,
                    success=True,
                )

                # Update usage stats
                self._update_usage_stats(claude_response)

                # Cache response
                if use_cache:
                    self._cache_response(cache_key, claude_response)

                logger.info(
                    f"Claude request successful: {total_tokens} tokens, "
                    f"${cost:.4f}, {response_time_ms}ms"
                )

                return claude_response

            except RateLimitError as e:
                logger.warning(
                    f"Rate limit hit, waiting before retry {attempt + 1}/{max_retries}"
                )
                time.sleep(5 * (attempt + 1))  # Exponential backoff
                last_error = e

            except (APIError, APIConnectionError) as e:
                logger.error(f"API error on attempt {attempt + 1}/{max_retries}: {e}")
                last_error = e
                if attempt < max_retries - 1:
                    time.sleep(2 * (attempt + 1))

            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                last_error = e
                break

        # All retries failed
        return ClaudeResponse(
            content="",
            tokens_used=0,
            cost_usd=0.0,
            response_time_ms=0,
            success=False,
            error=str(last_error),
        )

    def analyze_performance(
        self, trades: List[Dict[str, Any]], stats: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze trading performance and provide insights

        Args:
            trades: List of trade dictionaries
            stats: Dictionary of performance statistics

        Returns:
            Analysis with insights and recommendations
        """
        # Build context
        context = {
            "total_trades": len(trades),
            "stats": stats,
            "recent_trades": trades[-10:] if len(trades) > 10 else trades,
        }

        prompt = f"""Analyze this Warrior Trading day trading performance:

STATISTICS:
- Win Rate: {stats.get('win_rate', 0):.1f}%
- Total Trades: {stats.get('total_trades', 0)}
- Net P&L: ${stats.get('net_pnl', 0):.2f}
- Avg R-Multiple: {stats.get('avg_r_multiple', 0):.2f}R
- Best Trade: ${stats.get('best_trade_pnl', 0):.2f}
- Worst Trade: ${stats.get('worst_trade_pnl', 0):.2f}

RECENT TRADES:
{json.dumps(context['recent_trades'], indent=2)}

Provide a comprehensive analysis including:
1. Overall performance assessment
2. Pattern effectiveness (which setups worked best)
3. Entry/exit quality
4. Risk management effectiveness
5. Key strengths to maintain
6. Specific areas for improvement
7. Actionable recommendations for tomorrow

Format as JSON with keys: summary, strengths, weaknesses, recommendations, tomorrow_plan"""

        request = ClaudeRequest(
            request_type="performance_analysis",
            prompt=prompt,
            system_prompt="You are an expert day trading coach analyzing Warrior Trading strategy performance. Provide actionable, specific feedback.",
        )

        response = self.request(request)

        if response.success:
            try:
                return json.loads(response.content)
            except json.JSONDecodeError:
                return {"raw_analysis": response.content}
        else:
            return {"error": response.error}

    def detect_market_regime(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect current market regime and suggest adjustments

        Args:
            market_data: Dictionary containing market indicators

        Returns:
            Regime classification and recommended adjustments
        """
        prompt = f"""Analyze current market conditions and determine the trading regime:

MARKET DATA:
{json.dumps(market_data, indent=2)}

Classify into one of these regimes:
- TRENDING_BULL: Strong upward momentum
- TRENDING_BEAR: Downward pressure
- CHOPPY: Range-bound, low conviction
- HIGH_VOLATILITY: Large swings, news-driven
- LOW_VOLATILITY: Tight ranges

Provide JSON response with:
{{
  "regime": "REGIME_TYPE",
  "confidence": 0.0-1.0,
  "reasoning": "why this classification",
  "recommended_adjustments": {{
    "position_size_multiplier": 0.6-1.2,
    "min_confidence_threshold": 0.60-0.80,
    "max_daily_trades": 3-6,
    "preferred_patterns": ["PATTERN1", "PATTERN2"]
  }},
  "warnings": ["any specific risks or cautions"]
}}"""

        request = ClaudeRequest(
            request_type="market_regime",
            prompt=prompt,
            system_prompt="You are a market structure expert. Analyze conditions and provide regime classification.",
        )

        response = self.request(request)

        if response.success:
            try:
                return json.loads(response.content)
            except json.JSONDecodeError:
                return {"raw_response": response.content}
        else:
            return {"error": response.error}

    def suggest_optimizations(
        self, current_config: Dict[str, Any], performance_history: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Suggest parameter optimizations based on performance

        Args:
            current_config: Current strategy configuration
            performance_history: Historical performance data

        Returns:
            List of optimization suggestions
        """
        prompt = f"""Analyze trading performance and suggest parameter optimizations:

CURRENT CONFIGURATION:
{json.dumps(current_config, indent=2)}

PERFORMANCE HISTORY:
{json.dumps(performance_history, indent=2)}

Suggest specific parameter adjustments that could improve performance.
For each suggestion provide:
{{
  "parameter": "parameter_name",
  "current_value": current,
  "suggested_value": new_value,
  "reasoning": "why this change",
  "expected_impact": "predicted outcome",
  "confidence": 0.0-1.0
}}

Return as JSON array of suggestions."""

        request = ClaudeRequest(
            request_type="optimization_suggestions",
            prompt=prompt,
            system_prompt="You are a quantitative trading strategist. Suggest data-driven optimizations.",
        )

        response = self.request(request)

        if response.success:
            try:
                return json.loads(response.content)
            except json.JSONDecodeError:
                return [{"raw_response": response.content}]
        else:
            return [{"error": response.error}]

    def diagnose_error(self, error_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Diagnose system error and suggest recovery

        Args:
            error_context: Error information and context

        Returns:
            Diagnosis and recovery suggestions
        """
        prompt = f"""Diagnose this trading system error and suggest recovery:

ERROR CONTEXT:
{json.dumps(error_context, indent=2)}

Provide JSON response with:
{{
  "diagnosis": "what went wrong",
  "severity": "LOW|MEDIUM|HIGH|CRITICAL",
  "root_cause": "likely cause",
  "recovery_steps": ["step1", "step2", ...],
  "preventive_measures": ["measure1", "measure2", ...],
  "requires_manual_intervention": true|false
}}"""

        request = ClaudeRequest(
            request_type="error_diagnosis",
            prompt=prompt,
            temperature=0.3,  # Lower temperature for more focused diagnosis
            system_prompt="You are a system reliability engineer. Diagnose errors and provide recovery steps.",
        )

        response = self.request(request, use_cache=False)  # Don't cache error diagnoses

        if response.success:
            try:
                return json.loads(response.content)
            except json.JSONDecodeError:
                return {"raw_diagnosis": response.content}
        else:
            return {"error": response.error}

    def generate_insights(
        self, context: Dict[str, Any], insight_type: str = "general"
    ) -> List[str]:
        """
        Generate real-time trading insights

        Args:
            context: Current trading context
            insight_type: Type of insights to generate

        Returns:
            List of insight strings
        """
        prompts = {
            "general": "Provide 3-5 actionable insights based on current context",
            "risk": "Analyze risk factors and provide warnings",
            "opportunity": "Identify potential trading opportunities",
            "improvement": "Suggest specific improvements to trading approach",
        }

        prompt = f"""{prompts.get(insight_type, prompts['general'])}:

CONTEXT:
{json.dumps(context, indent=2)}

Return as JSON array of insight strings. Each should be specific and actionable."""

        request = ClaudeRequest(
            request_type=f"insights_{insight_type}",
            prompt=prompt,
            max_tokens=1024,
            system_prompt="You are a day trading mentor. Provide concise, actionable insights.",
        )

        response = self.request(request)

        if response.success:
            try:
                return json.loads(response.content)
            except json.JSONDecodeError:
                # Try to extract insights from raw text
                return [
                    line.strip()
                    for line in response.content.split("\n")
                    if line.strip()
                ]
        else:
            return [f"Error generating insights: {response.error}"]

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get current usage statistics"""
        return {
            "total_requests": self.usage_stats.total_requests,
            "total_tokens": self.usage_stats.total_tokens,
            "total_cost_usd": round(self.usage_stats.total_cost_usd, 2),
            "daily_requests": self.usage_stats.daily_requests,
            "daily_cost_usd": round(self.usage_stats.daily_cost_usd, 2),
            "daily_limit_usd": self.daily_cost_limit,
            "monthly_limit_usd": self.monthly_cost_limit,
            "cache_size": len(self.response_cache),
            "last_request": (
                self.usage_stats.last_request_time.isoformat()
                if self.usage_stats.last_request_time
                else None
            ),
        }

    def reset_daily_stats(self):
        """Reset daily statistics"""
        self.usage_stats.daily_requests = 0
        self.usage_stats.daily_cost_usd = 0.0
        self.usage_stats.last_reset_date = datetime.now().strftime("%Y-%m-%d")
        self._save_usage_stats()
        logger.info("Daily statistics reset")

    def clear_cache(self):
        """Clear response cache"""
        self.response_cache.clear()
        logger.info("Response cache cleared")


# Global instance
_claude_integration: Optional[ClaudeIntegration] = None


def get_claude_integration() -> ClaudeIntegration:
    """Get or create global ClaudeIntegration instance"""
    global _claude_integration
    if _claude_integration is None:
        _claude_integration = ClaudeIntegration()
    return _claude_integration


if __name__ == "__main__":
    # Test the integration
    claude = ClaudeIntegration()

    print("Claude Integration Test")
    print("=" * 50)
    print(f"Model: {claude.model}")
    print(f"Rate Limit: {claude.max_requests_per_minute} req/min")
    print(f"Daily Limit: ${claude.daily_cost_limit}")
    print(f"Cache Enabled: {claude.cache_enabled}")
    print("\nUsage Stats:")
    print(json.dumps(claude.get_usage_stats(), indent=2))

    # Test request (if API key is configured)
    if claude.client:
        print("\nTesting simple request...")
        request = ClaudeRequest(
            request_type="test",
            prompt="Explain the Warrior Trading Bull Flag pattern in 2 sentences.",
            max_tokens=256,
        )
        response = claude.request(request)
        print(f"Success: {response.success}")
        print(f"Tokens: {response.tokens_used}")
        print(f"Cost: ${response.cost_usd:.4f}")
        print(f"Time: {response.response_time_ms}ms")
        print(f"Content: {response.content[:200]}...")
    else:
        print("\n⚠️  No API key configured - skipping request test")
