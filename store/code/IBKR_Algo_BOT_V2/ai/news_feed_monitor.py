"""
Breaking News Feed Monitor
==========================
Real-time news monitoring with AI-powered sentiment analysis and trading triggers.
Monitors multiple news sources for market-moving events.
"""

import asyncio
import html
import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)


class NewsImpact(Enum):
    """News impact level on market"""

    CRITICAL = "critical"  # Major market-moving (Fed, earnings miss/beat >20%)
    HIGH = "high"  # Significant (analyst upgrades, M&A rumors)
    MEDIUM = "medium"  # Moderate (product launches, partnerships)
    LOW = "low"  # Minor (routine updates)
    NEUTRAL = "neutral"  # No expected impact


class NewsSentiment(Enum):
    """Sentiment classification"""

    VERY_BULLISH = "very_bullish"
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"
    VERY_BEARISH = "very_bearish"


class NewsCategory(Enum):
    """News categories"""

    EARNINGS = "earnings"
    FDA_APPROVAL = "fda_approval"
    MERGER_ACQUISITION = "merger_acquisition"
    ANALYST_RATING = "analyst_rating"
    INSIDER_TRADING = "insider_trading"
    ECONOMIC_DATA = "economic_data"
    FED_ANNOUNCEMENT = "fed_announcement"
    GUIDANCE = "guidance"
    PRODUCT_LAUNCH = "product_launch"
    LEGAL_REGULATORY = "legal_regulatory"
    MANAGEMENT_CHANGE = "management_change"
    DIVIDEND = "dividend"
    STOCK_SPLIT = "stock_split"
    SECTOR_NEWS = "sector_news"
    GEOPOLITICAL = "geopolitical"
    OTHER = "other"


@dataclass
class NewsItem:
    """Individual news item"""

    id: str
    headline: str
    summary: str
    source: str
    published_at: datetime
    symbols: List[str]
    category: NewsCategory
    sentiment: NewsSentiment
    impact: NewsImpact
    sentiment_score: float  # -1.0 to 1.0
    keywords: List[str]
    url: Optional[str] = None
    raw_data: Optional[Dict] = None
    ai_analysis: Optional[str] = None
    triggered_actions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            **asdict(self),
            "published_at": (
                self.published_at.isoformat() if self.published_at else None
            ),
            "category": self.category.value,
            "sentiment": self.sentiment.value,
            "impact": self.impact.value,
        }


@dataclass
class NewsTrigger:
    """Trigger rule for news events"""

    id: str
    name: str
    enabled: bool
    category_filters: List[NewsCategory]
    impact_threshold: NewsImpact
    sentiment_threshold: Optional[NewsSentiment]
    symbol_filters: List[str]  # Empty = all symbols
    keyword_filters: List[str]
    action: str  # "alert", "pause_trading", "close_positions", "switch_strategy"
    action_params: Dict[str, Any]
    cooldown_minutes: int = 5
    last_triggered: Optional[datetime] = None


# Keyword patterns for category detection
CATEGORY_PATTERNS = {
    NewsCategory.EARNINGS: [
        r"\bearnings\b",
        r"\bEPS\b",
        r"\brevenue\b",
        r"\bquarterly results?\b",
        r"\bbeat estimates?\b",
        r"\bmiss estimates?\b",
        r"\bguidance\b",
    ],
    NewsCategory.FDA_APPROVAL: [
        r"\bFDA\b",
        r"\bapproval\b",
        r"\brejection\b",
        r"\bclinical trial\b",
        r"\bphase [123]\b",
        r"\bdrug approval\b",
    ],
    NewsCategory.MERGER_ACQUISITION: [
        r"\bmerger\b",
        r"\bacquisition\b",
        r"\bbuyout\b",
        r"\btakeover\b",
        r"\bM&A\b",
        r"\bacquire[sd]?\b",
    ],
    NewsCategory.ANALYST_RATING: [
        r"\bupgrade[sd]?\b",
        r"\bdowngrade[sd]?\b",
        r"\bprice target\b",
        r"\banalyst\b",
        r"\brating\b",
        r"\bbuy rating\b",
        r"\bsell rating\b",
    ],
    NewsCategory.INSIDER_TRADING: [
        r"\binsider\b",
        r"\bCEO (buy|sell|purchase)\b",
        r"\bexecutive (buy|sell)\b",
        r"\bForm 4\b",
        r"\binsider trading\b",
    ],
    NewsCategory.ECONOMIC_DATA: [
        r"\bCPI\b",
        r"\binflation\b",
        r"\bunemployment\b",
        r"\bjobs report\b",
        r"\bGDP\b",
        r"\bretail sales\b",
        r"\bhousing\b",
        r"\bPMI\b",
    ],
    NewsCategory.FED_ANNOUNCEMENT: [
        r"\bFed\b",
        r"\bFOMC\b",
        r"\binterest rate\b",
        r"\brate (hike|cut)\b",
        r"\bPowell\b",
        r"\bFederal Reserve\b",
        r"\bmonetary policy\b",
    ],
    NewsCategory.GUIDANCE: [
        r"\bguidance\b",
        r"\boutlook\b",
        r"\bforecast\b",
        r"\bexpectations?\b",
        r"\braise[sd]? guidance\b",
        r"\blower[ed]? guidance\b",
    ],
    NewsCategory.PRODUCT_LAUNCH: [
        r"\blaunch\b",
        r"\bnew product\b",
        r"\bannounce[sd]?\b",
        r"\brelease\b",
        r"\bunveil\b",
        r"\bintroduc\b",
    ],
    NewsCategory.LEGAL_REGULATORY: [
        r"\blawsuit\b",
        r"\bSEC\b",
        r"\bregulator\b",
        r"\binvestigation\b",
        r"\bsubpoena\b",
        r"\bsettlement\b",
        r"\bantitrust\b",
    ],
    NewsCategory.MANAGEMENT_CHANGE: [
        r"\bCEO\b",
        r"\bCFO\b",
        r"\bresign\b",
        r"\bappoint\b",
        r"\bstep down\b",
        r"\bleadership change\b",
        r"\bexecutive\b",
    ],
    NewsCategory.DIVIDEND: [
        r"\bdividend\b",
        r"\byield\b",
        r"\bpayout\b",
        r"\bdistribution\b",
    ],
    NewsCategory.STOCK_SPLIT: [
        r"\bstock split\b",
        r"\breverse split\b",
        r"\bshare split\b",
    ],
    NewsCategory.GEOPOLITICAL: [
        r"\btariff\b",
        r"\btrade war\b",
        r"\bsanction\b",
        r"\bgeopolitical\b",
        r"\bwar\b",
        r"\bconflict\b",
        r"\bChina\b.*\btrade\b",
    ],
}

# Sentiment keywords
BULLISH_KEYWORDS = [
    "beat",
    "exceed",
    "surge",
    "soar",
    "rally",
    "upgrade",
    "breakthrough",
    "approval",
    "growth",
    "record",
    "strong",
    "bullish",
    "outperform",
    "positive",
    "upside",
    "raise",
    "higher",
    "accelerate",
    "expand",
]

BEARISH_KEYWORDS = [
    "miss",
    "decline",
    "plunge",
    "crash",
    "downgrade",
    "rejection",
    "layoff",
    "weak",
    "bearish",
    "underperform",
    "negative",
    "downside",
    "lower",
    "cut",
    "warning",
    "concern",
    "risk",
    "lawsuit",
    "investigation",
]


class NewsFeedMonitor:
    """
    Real-time news monitoring with AI-powered analysis and trading triggers.
    Uses Benzinga as primary news source (Schwab-compatible).
    """

    def __init__(self):
        # Benzinga API for news (primary source)
        self.benzinga_api_key = os.getenv(
            "BENZINGA_API_KEY", "bz.MUTADSLMPPPHDWEGOYUMSFHUGH5TS7TD"
        )
        self.base_url = "https://api.benzinga.com/api/v2/news"

        self.news_cache: List[NewsItem] = []
        self.max_cache_size = 500
        self.triggers: Dict[str, NewsTrigger] = {}
        self.callbacks: List[Callable] = []
        self.is_monitoring = False
        self.monitor_task: Optional[asyncio.Task] = None
        self.poll_interval = 30  # seconds
        self.watched_symbols: List[str] = []

        # Claude AI integration for deep analysis
        self.claude_api_key = os.getenv("ANTHROPIC_API_KEY")

        # Initialize default triggers
        self._setup_default_triggers()

        logger.info("NewsFeedMonitor initialized")

    def _setup_default_triggers(self):
        """Setup default news triggers"""
        default_triggers = [
            NewsTrigger(
                id="fed_announcement",
                name="Federal Reserve Announcement",
                enabled=True,
                category_filters=[NewsCategory.FED_ANNOUNCEMENT],
                impact_threshold=NewsImpact.HIGH,
                sentiment_threshold=None,
                symbol_filters=[],
                keyword_filters=["FOMC", "rate", "Powell"],
                action="alert",
                action_params={"priority": "high", "pause_new_trades": True},
                cooldown_minutes=30,
            ),
            NewsTrigger(
                id="earnings_surprise",
                name="Major Earnings Surprise",
                enabled=True,
                category_filters=[NewsCategory.EARNINGS],
                impact_threshold=NewsImpact.HIGH,
                sentiment_threshold=None,
                symbol_filters=[],
                keyword_filters=["beat", "miss", "surprise"],
                action="alert",
                action_params={"priority": "high"},
                cooldown_minutes=5,
            ),
            NewsTrigger(
                id="fda_decision",
                name="FDA Decision",
                enabled=True,
                category_filters=[NewsCategory.FDA_APPROVAL],
                impact_threshold=NewsImpact.CRITICAL,
                sentiment_threshold=None,
                symbol_filters=[],
                keyword_filters=["FDA", "approval", "rejection"],
                action="close_positions",
                action_params={"affected_symbols_only": True},
                cooldown_minutes=60,
            ),
            NewsTrigger(
                id="merger_news",
                name="M&A Announcement",
                enabled=True,
                category_filters=[NewsCategory.MERGER_ACQUISITION],
                impact_threshold=NewsImpact.HIGH,
                sentiment_threshold=None,
                symbol_filters=[],
                keyword_filters=["merger", "acquisition", "buyout"],
                action="alert",
                action_params={"priority": "high"},
                cooldown_minutes=15,
            ),
            NewsTrigger(
                id="negative_sentiment_spike",
                name="Very Bearish News",
                enabled=True,
                category_filters=[],
                impact_threshold=NewsImpact.HIGH,
                sentiment_threshold=NewsSentiment.VERY_BEARISH,
                symbol_filters=[],
                keyword_filters=[],
                action="switch_strategy",
                action_params={"strategy": "mean_reversion"},
                cooldown_minutes=30,
            ),
        ]

        for trigger in default_triggers:
            self.triggers[trigger.id] = trigger

    def _extract_summary(self, article: dict) -> str:
        """Extract clean summary from article, stripping HTML tags and entities"""

        def clean_text(text: str) -> str:
            """Clean HTML and fix encoding issues"""
            # Strip HTML tags
            text = re.sub(r"<[^>]+>", " ", text)
            # Decode HTML entities
            text = html.unescape(text)
            # Fix double-encoded UTF-8 (common Benzinga issue)
            # \u00c2\u00a0 -> space, \u00c2\u00ad -> nothing
            text = text.replace("\u00c2\u00a0", " ")
            text = text.replace("\u00c2\u00ad", "")
            text = text.replace("\u00c2", "")
            # Normalize whitespace
            text = re.sub(r"\s+", " ", text).strip()
            return text

        # Try teaser first
        teaser = article.get("teaser", "").strip()
        if teaser and len(teaser) > 10:
            return clean_text(teaser)[:300]

        # Fall back to body - extract first paragraph
        body = article.get("body", "")
        if body:
            text = clean_text(body)
            # Get first 300 chars, break at sentence if possible
            if len(text) > 300:
                cut_point = text[:300].rfind(".")
                if cut_point > 100:
                    return text[: cut_point + 1]
                return text[:300] + "..."
            return text

        return ""

    def _detect_category(self, text: str) -> NewsCategory:
        """Detect news category from text"""
        text_lower = text.lower()

        for category, patterns in CATEGORY_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    return category

        return NewsCategory.OTHER

    def _analyze_sentiment(self, text: str) -> tuple[NewsSentiment, float]:
        """Analyze sentiment from text"""
        text_lower = text.lower()

        bullish_count = sum(1 for kw in BULLISH_KEYWORDS if kw in text_lower)
        bearish_count = sum(1 for kw in BEARISH_KEYWORDS if kw in text_lower)

        # Calculate score (-1 to 1)
        total = bullish_count + bearish_count
        if total == 0:
            return NewsSentiment.NEUTRAL, 0.0

        score = (bullish_count - bearish_count) / total

        # Classify sentiment
        if score >= 0.6:
            sentiment = NewsSentiment.VERY_BULLISH
        elif score >= 0.2:
            sentiment = NewsSentiment.BULLISH
        elif score <= -0.6:
            sentiment = NewsSentiment.VERY_BEARISH
        elif score <= -0.2:
            sentiment = NewsSentiment.BEARISH
        else:
            sentiment = NewsSentiment.NEUTRAL

        return sentiment, score

    def _assess_impact(
        self, category: NewsCategory, sentiment_score: float, keywords: List[str]
    ) -> NewsImpact:
        """Assess the market impact of news"""
        # Critical categories
        if category in [NewsCategory.FED_ANNOUNCEMENT, NewsCategory.FDA_APPROVAL]:
            return NewsImpact.CRITICAL

        # High impact categories
        if category in [
            NewsCategory.EARNINGS,
            NewsCategory.MERGER_ACQUISITION,
            NewsCategory.ECONOMIC_DATA,
        ]:
            if abs(sentiment_score) > 0.5:
                return NewsImpact.HIGH
            return NewsImpact.MEDIUM

        # Check for extreme sentiment
        if abs(sentiment_score) > 0.7:
            return NewsImpact.HIGH

        # Analyst ratings and guidance
        if category in [NewsCategory.ANALYST_RATING, NewsCategory.GUIDANCE]:
            return NewsImpact.MEDIUM

        # Check keywords for urgency indicators
        urgent_keywords = ["breaking", "urgent", "halt", "crash", "surge", "plunge"]
        if any(kw in " ".join(keywords).lower() for kw in urgent_keywords):
            return NewsImpact.HIGH

        return NewsImpact.LOW

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text"""
        # Common financial terms to look for
        keywords = []

        # Check for ticker symbols (1-5 uppercase letters)
        tickers = re.findall(r"\b[A-Z]{1,5}\b", text)
        keywords.extend([t for t in tickers if len(t) >= 2])

        # Check for percentages
        percentages = re.findall(r"\d+\.?\d*%", text)
        keywords.extend(percentages)

        # Check for dollar amounts
        amounts = re.findall(r"\$[\d,]+\.?\d*[BMK]?", text)
        keywords.extend(amounts)

        # Add category-specific keywords found
        for category, patterns in CATEGORY_PATTERNS.items():
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                keywords.extend(matches)

        return list(set(keywords))[:20]

    async def fetch_news(
        self, symbols: List[str] = None, limit: int = 50
    ) -> List[NewsItem]:
        """Fetch news from Benzinga API"""
        if not self.benzinga_api_key:
            logger.warning("Benzinga API key not configured, using mock data")
            return self._get_mock_news(symbols, limit)

        # Benzinga API returns XML by default, we need to parse it
        params = {
            "token": self.benzinga_api_key,
            "pageSize": limit,
            "displayOutput": "full",
            "sort": "created:desc",
        }
        if symbols:
            params["tickers"] = ",".join(symbols[:10])

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.base_url,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as response:
                    if response.status == 200:
                        text = await response.text()
                        # Check if response is XML or JSON
                        if text.strip().startswith("<?xml") or text.strip().startswith(
                            "<result"
                        ):
                            return self._parse_benzinga_xml(text)
                        else:
                            data = await response.json()
                            return self._parse_benzinga_news(data)
                    else:
                        error = await response.text()
                        logger.error(f"Benzinga API error: {response.status} - {error}")
                        return self._get_mock_news(symbols, limit)
        except Exception as e:
            logger.error(f"Error fetching news: {e}")
            return self._get_mock_news(symbols, limit)

    def _parse_benzinga_xml(self, xml_text: str) -> List[NewsItem]:
        """Parse Benzinga XML response into NewsItem objects"""
        import xml.etree.ElementTree as ET

        items = []

        try:
            root = ET.fromstring(xml_text)

            for item_elem in root.findall(".//item"):
                try:
                    headline = item_elem.findtext("title", "")
                    url = item_elem.findtext("url", "")
                    created = item_elem.findtext("created", "")
                    teaser = item_elem.findtext("teaser", "")
                    body = item_elem.findtext("body", "")
                    news_id = item_elem.findtext("id", "")

                    # Extract symbols from stocks
                    symbols = []
                    stocks_elem = item_elem.find("stocks")
                    if stocks_elem is not None:
                        for stock in stocks_elem.findall("item"):
                            name = stock.findtext("name", "")
                            if name:
                                symbols.append(name)

                    summary = teaser or body[:200] if body else ""
                    full_text = f"{headline} {summary}"

                    category = self._detect_category(full_text)
                    sentiment, score = self._analyze_sentiment(full_text)
                    keywords = self._extract_keywords(full_text)
                    impact = self._assess_impact(category, score, keywords)

                    # Parse published time
                    published_at = None
                    if created:
                        try:
                            # Format: "Mon, 29 Dec 2025 07:26:02 -0400"
                            from email.utils import parsedate_to_datetime

                            published_at = parsedate_to_datetime(created)
                        except Exception:
                            pass

                    news_item = NewsItem(
                        id=news_id or f"bz_{hash(headline)}",
                        headline=headline,
                        summary=summary,
                        source="Benzinga",
                        url=url,
                        published_at=published_at,
                        symbols=symbols,
                        category=category,
                        sentiment=sentiment,
                        sentiment_score=score,
                        impact=impact,
                        keywords=keywords,
                    )
                    items.append(news_item)
                    logger.debug(f"Parsed news: {symbols} - {headline[:50]}")

                except Exception as e:
                    logger.error(f"Error parsing XML news item: {e}")
                    continue

        except ET.ParseError as e:
            logger.error(f"XML parse error: {e}")

        logger.info(f"Parsed {len(items)} news items from Benzinga XML")
        return items

    def _parse_benzinga_news(self, data: dict) -> List[NewsItem]:
        """Parse Benzinga API response into NewsItem objects"""
        items = []

        # Benzinga returns list directly or in 'data' key
        articles = data if isinstance(data, list) else data.get("data", [])

        for article in articles:
            try:
                headline = article.get("title", article.get("headline", ""))
                summary = self._extract_summary(article)
                full_text = f"{headline} {summary}"

                category = self._detect_category(full_text)
                sentiment, score = self._analyze_sentiment(full_text)
                keywords = self._extract_keywords(full_text)
                impact = self._assess_impact(category, score, keywords)

                # Extract symbols from Benzinga response
                symbols = []
                stocks = article.get("stocks", [])
                if stocks:
                    symbols = [s.get("name", "") for s in stocks if s.get("name")]

                # Parse published time
                created = article.get("created", article.get("updated", ""))
                try:
                    if created:
                        published_at = datetime.fromisoformat(
                            created.replace("Z", "+00:00")
                        )
                    else:
                        published_at = datetime.now()
                except:
                    published_at = datetime.now()

                news_item = NewsItem(
                    id=str(article.get("id", datetime.now().timestamp())),
                    headline=headline,
                    summary=summary,
                    source="Benzinga",
                    published_at=published_at,
                    symbols=symbols,
                    category=category,
                    sentiment=sentiment,
                    impact=impact,
                    sentiment_score=score,
                    keywords=keywords,
                    url=article.get("url"),
                    raw_data=article,
                )
                items.append(news_item)

            except Exception as e:
                logger.error(f"Error parsing Benzinga news item: {e}")
                continue

        return items

    def _get_mock_news(
        self, symbols: List[str] = None, limit: int = 10
    ) -> List[NewsItem]:
        """Generate mock news for testing"""
        mock_items = [
            {
                "headline": "Federal Reserve Holds Rates Steady, Signals Patience on Future Moves",
                "summary": "The Fed maintained interest rates and indicated it will be patient before making changes.",
                "source": "Reuters",
                "symbols": ["SPY", "QQQ"],
                "category": NewsCategory.FED_ANNOUNCEMENT,
            },
            {
                "headline": "NVIDIA Beats Q3 Earnings Estimates, Data Center Revenue Surges 200%",
                "summary": "NVIDIA reported earnings that significantly exceeded analyst expectations driven by AI demand.",
                "source": "Bloomberg",
                "symbols": ["NVDA"],
                "category": NewsCategory.EARNINGS,
            },
            {
                "headline": "Tesla Faces Downgrade from Major Analyst Amid Competition Concerns",
                "summary": "A prominent Wall Street analyst downgraded Tesla stock citing increasing EV competition.",
                "source": "CNBC",
                "symbols": ["TSLA"],
                "category": NewsCategory.ANALYST_RATING,
            },
            {
                "headline": "Apple Announces New AI Features Coming to iPhone",
                "summary": "Apple unveiled new artificial intelligence capabilities for its next iPhone generation.",
                "source": "WSJ",
                "symbols": ["AAPL"],
                "category": NewsCategory.PRODUCT_LAUNCH,
            },
            {
                "headline": "Biotech Company Receives FDA Approval for Cancer Treatment",
                "summary": "XYZ Pharma received FDA approval for its breakthrough cancer treatment drug.",
                "source": "BioPharma Dive",
                "symbols": ["XBI"],
                "category": NewsCategory.FDA_APPROVAL,
            },
        ]

        items = []
        for i, mock in enumerate(mock_items[:limit]):
            full_text = f"{mock['headline']} {mock['summary']}"
            sentiment, score = self._analyze_sentiment(full_text)
            keywords = self._extract_keywords(full_text)
            impact = self._assess_impact(mock["category"], score, keywords)

            # Filter by symbols if provided
            if symbols and not any(s in mock["symbols"] for s in symbols):
                continue

            items.append(
                NewsItem(
                    id=f"mock_{i}_{datetime.now().timestamp()}",
                    headline=mock["headline"],
                    summary=mock["summary"],
                    source=mock["source"],
                    published_at=datetime.now() - timedelta(minutes=i * 15),
                    symbols=mock["symbols"],
                    category=mock["category"],
                    sentiment=sentiment,
                    impact=impact,
                    sentiment_score=score,
                    keywords=keywords,
                )
            )

        return items

    async def analyze_with_claude(self, news_item: NewsItem) -> str:
        """Get deep analysis from Claude AI"""
        if not self.claude_api_key:
            return "Claude API not configured"

        try:
            import anthropic

            client = anthropic.Anthropic(api_key=self.claude_api_key)

            prompt = f"""Analyze this market news for trading implications:

HEADLINE: {news_item.headline}
SUMMARY: {news_item.summary}
SYMBOLS: {', '.join(news_item.symbols)}
CATEGORY: {news_item.category.value}
INITIAL SENTIMENT: {news_item.sentiment.value} (score: {news_item.sentiment_score:.2f})

Provide a brief trading analysis:
1. Key implications for the affected stocks
2. Potential price impact (short-term and medium-term)
3. Recommended trading action (if any)
4. Risk factors to consider
5. Related sectors/stocks that might be affected

Keep response concise (under 200 words)."""

            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
            )

            analysis = response.content[0].text
            news_item.ai_analysis = analysis
            return analysis

        except Exception as e:
            logger.error(f"Claude analysis error: {e}")
            return f"Analysis unavailable: {str(e)}"

    def check_triggers(self, news_item: NewsItem) -> List[Dict]:
        """Check if news item triggers any rules"""
        triggered = []
        now = datetime.now()

        for trigger_id, trigger in self.triggers.items():
            if not trigger.enabled:
                continue

            # Check cooldown
            if trigger.last_triggered:
                cooldown_end = trigger.last_triggered + timedelta(
                    minutes=trigger.cooldown_minutes
                )
                if now < cooldown_end:
                    continue

            # Check category filter
            if (
                trigger.category_filters
                and news_item.category not in trigger.category_filters
            ):
                continue

            # Check impact threshold
            impact_order = [
                NewsImpact.LOW,
                NewsImpact.MEDIUM,
                NewsImpact.HIGH,
                NewsImpact.CRITICAL,
            ]
            if impact_order.index(news_item.impact) < impact_order.index(
                trigger.impact_threshold
            ):
                continue

            # Check sentiment threshold
            if trigger.sentiment_threshold:
                sentiment_order = [
                    NewsSentiment.VERY_BEARISH,
                    NewsSentiment.BEARISH,
                    NewsSentiment.NEUTRAL,
                    NewsSentiment.BULLISH,
                    NewsSentiment.VERY_BULLISH,
                ]
                if news_item.sentiment != trigger.sentiment_threshold:
                    # Check if sentiment is more extreme in the expected direction
                    if trigger.sentiment_threshold in [
                        NewsSentiment.BEARISH,
                        NewsSentiment.VERY_BEARISH,
                    ]:
                        if sentiment_order.index(
                            news_item.sentiment
                        ) > sentiment_order.index(trigger.sentiment_threshold):
                            continue
                    else:
                        if sentiment_order.index(
                            news_item.sentiment
                        ) < sentiment_order.index(trigger.sentiment_threshold):
                            continue

            # Check symbol filter
            if trigger.symbol_filters:
                if not any(s in news_item.symbols for s in trigger.symbol_filters):
                    continue

            # Check keyword filter
            if trigger.keyword_filters:
                text = f"{news_item.headline} {news_item.summary}".lower()
                if not any(kw.lower() in text for kw in trigger.keyword_filters):
                    continue

            # Trigger matched!
            trigger.last_triggered = now
            news_item.triggered_actions.append(trigger.name)

            triggered.append(
                {
                    "trigger_id": trigger_id,
                    "trigger_name": trigger.name,
                    "action": trigger.action,
                    "action_params": trigger.action_params,
                    "news_item": news_item.to_dict(),
                }
            )

            logger.info(
                f"News trigger activated: {trigger.name} for {news_item.headline[:50]}..."
            )

        return triggered

    async def start_monitoring(
        self, symbols: List[str] = None, poll_interval: int = 30
    ):
        """Start continuous news monitoring"""
        self.is_monitoring = True
        self.watched_symbols = symbols or []
        self.poll_interval = poll_interval

        logger.info(f"Starting news monitoring for symbols: {symbols or 'ALL'}")

        while self.is_monitoring:
            try:
                # Fetch latest news
                news_items = await self.fetch_news(self.watched_symbols, limit=20)

                for item in news_items:
                    # Skip if already in cache
                    if any(cached.id == item.id for cached in self.news_cache):
                        continue

                    # Add to cache
                    self.news_cache.insert(0, item)

                    # Trim cache if needed
                    if len(self.news_cache) > self.max_cache_size:
                        self.news_cache = self.news_cache[: self.max_cache_size]

                    # Check triggers
                    triggered = self.check_triggers(item)

                    # Notify callbacks
                    for callback in self.callbacks:
                        try:
                            if asyncio.iscoroutinefunction(callback):
                                await callback(item, triggered)
                            else:
                                callback(item, triggered)
                        except Exception as e:
                            logger.error(f"Callback error: {e}")

                    # For high-impact news, get Claude analysis
                    if item.impact in [NewsImpact.HIGH, NewsImpact.CRITICAL]:
                        await self.analyze_with_claude(item)

                await asyncio.sleep(self.poll_interval)

            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(5)

    def stop_monitoring(self):
        """Stop news monitoring"""
        self.is_monitoring = False
        logger.info("News monitoring stopped")

    def add_callback(self, callback: Callable):
        """Add callback for news events"""
        self.callbacks.append(callback)

    def add_trigger(self, trigger: NewsTrigger):
        """Add a new trigger"""
        self.triggers[trigger.id] = trigger
        logger.info(f"Added trigger: {trigger.name}")

    def remove_trigger(self, trigger_id: str):
        """Remove a trigger"""
        if trigger_id in self.triggers:
            del self.triggers[trigger_id]
            logger.info(f"Removed trigger: {trigger_id}")

    def get_recent_news(
        self,
        limit: int = 20,
        category: NewsCategory = None,
        impact: NewsImpact = None,
        symbols: List[str] = None,
    ) -> List[NewsItem]:
        """Get recent news with optional filters"""
        filtered = self.news_cache

        if category:
            filtered = [n for n in filtered if n.category == category]

        if impact:
            impact_order = [
                NewsImpact.LOW,
                NewsImpact.MEDIUM,
                NewsImpact.HIGH,
                NewsImpact.CRITICAL,
            ]
            min_index = impact_order.index(impact)
            filtered = [
                n for n in filtered if impact_order.index(n.impact) >= min_index
            ]

        if symbols:
            filtered = [n for n in filtered if any(s in n.symbols for s in symbols)]

        return filtered[:limit]

    def get_sentiment_summary(self, symbols: List[str] = None, hours: int = 24) -> Dict:
        """Get aggregated sentiment summary"""
        cutoff = datetime.now() - timedelta(hours=hours)

        relevant_news = [
            n
            for n in self.news_cache
            if n.published_at > cutoff
            and (not symbols or any(s in n.symbols for s in symbols))
        ]

        if not relevant_news:
            return {
                "overall_sentiment": "neutral",
                "sentiment_score": 0.0,
                "news_count": 0,
                "bullish_count": 0,
                "bearish_count": 0,
                "high_impact_count": 0,
            }

        scores = [n.sentiment_score for n in relevant_news]
        avg_score = sum(scores) / len(scores)

        bullish = len([n for n in relevant_news if n.sentiment_score > 0.2])
        bearish = len([n for n in relevant_news if n.sentiment_score < -0.2])
        high_impact = len(
            [
                n
                for n in relevant_news
                if n.impact in [NewsImpact.HIGH, NewsImpact.CRITICAL]
            ]
        )

        if avg_score >= 0.3:
            overall = "bullish"
        elif avg_score <= -0.3:
            overall = "bearish"
        else:
            overall = "neutral"

        return {
            "overall_sentiment": overall,
            "sentiment_score": round(avg_score, 3),
            "news_count": len(relevant_news),
            "bullish_count": bullish,
            "bearish_count": bearish,
            "high_impact_count": high_impact,
            "period_hours": hours,
        }


# Global instance
_news_monitor: Optional[NewsFeedMonitor] = None


def get_news_monitor() -> NewsFeedMonitor:
    """Get or create the global news monitor instance"""
    global _news_monitor
    if _news_monitor is None:
        _news_monitor = NewsFeedMonitor()
    return _news_monitor


async def test_news_monitor():
    """Test the news monitor"""
    monitor = get_news_monitor()

    # Fetch some news
    print("Fetching news...")
    news = await monitor.fetch_news(["AAPL", "TSLA", "NVDA"], limit=10)

    print(f"\nFound {len(news)} news items:\n")
    for item in news:
        print(
            f"[{item.impact.value.upper()}] [{item.sentiment.value}] {item.headline[:60]}..."
        )
        print(f"  Category: {item.category.value}, Score: {item.sentiment_score:.2f}")
        print(f"  Symbols: {item.symbols}")
        print()

    # Get sentiment summary
    summary = monitor.get_sentiment_summary()
    print(f"Sentiment Summary: {summary}")


if __name__ == "__main__":
    asyncio.run(test_news_monitor())
