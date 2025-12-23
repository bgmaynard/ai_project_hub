"""
SEC EDGAR Real-Time Filing Monitor
==================================
Monitors SEC EDGAR for new filings (8-K, 10-K, 10-Q, etc.)
Detects material events before mainstream news picks them up.

Uses free SEC API at data.sec.gov (sub-second updates, 10 req/sec limit)
"""

import asyncio
import logging
import json
import os
import re
import httpx
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Callable
from dataclasses import dataclass, asdict
import threading
import time

logger = logging.getLogger(__name__)

# SEC API endpoints
SEC_RECENT_FILINGS = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=8-K&company=&dateb=&owner=include&count=40&output=atom"
SEC_FULL_TEXT_SEARCH = "https://efts.sec.gov/LATEST/search-index"
SEC_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_FILINGS_RSS = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type={form_type}&count=40&output=atom"

# Material 8-K item codes that move stocks
MATERIAL_8K_ITEMS = {
    "1.01": "Entry into Material Definitive Agreement",
    "1.02": "Termination of Material Definitive Agreement",
    "1.03": "Bankruptcy or Receivership",
    "2.01": "Completion of Acquisition or Disposition",
    "2.02": "Results of Operations (EARNINGS)",
    "2.03": "Creation of Direct Financial Obligation",
    "2.04": "Triggering Events (Acceleration of Obligations)",
    "2.05": "Costs for Exit/Disposal Activities",
    "2.06": "Material Impairments",
    "3.01": "Delisting/Transfer/Failure to Meet Listing Standard",
    "3.02": "Unregistered Sales of Equity Securities",
    "3.03": "Material Modification to Rights of Securities",
    "4.01": "Changes in Registrant's Certifying Accountant",
    "4.02": "Non-Reliance on Previously Issued Financials",
    "5.01": "Changes in Control of Registrant",
    "5.02": "Departure/Election of Directors/Officers",
    "5.03": "Amendments to Articles/Bylaws",
    "5.07": "Submission of Matters to Vote of Security Holders",
    "7.01": "Regulation FD Disclosure",
    "8.01": "Other Events",
}

# High priority items that usually move stocks significantly
HIGH_PRIORITY_ITEMS = ["1.01", "1.03", "2.01", "2.02", "2.06", "3.01", "4.02", "5.01"]

CONFIG_FILE = os.path.join(os.path.dirname(__file__), "edgar_config.json")
FILINGS_CACHE_FILE = os.path.join(os.path.dirname(__file__), "edgar_filings_cache.json")


@dataclass
class EdgarFiling:
    """Represents an SEC filing"""
    accession_number: str
    cik: str
    company_name: str
    form_type: str
    filed_date: str
    accepted_time: str
    ticker: Optional[str] = None
    items: List[str] = None  # For 8-K item codes
    description: str = ""
    url: str = ""
    priority: str = "normal"  # high, normal, low

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class EdgarConfig:
    """Configuration for EDGAR monitor"""
    enabled: bool = True
    poll_interval_seconds: int = 5  # How often to check for new filings
    form_types: List[str] = None  # Which form types to monitor
    min_priority: str = "normal"  # Minimum priority to alert on
    auto_add_to_watchlist: bool = True
    notify_on_filing: bool = True

    def __post_init__(self):
        if self.form_types is None:
            self.form_types = ["8-K", "8-K/A", "10-K", "10-Q", "S-1", "S-3", "4"]

    def to_dict(self) -> Dict:
        return asdict(self)


class EdgarMonitor:
    """
    Real-time SEC EDGAR filing monitor.
    Polls SEC for new filings and alerts on material events.
    """

    def __init__(self):
        self.config = EdgarConfig()
        self.is_running = False
        self.seen_filings: Set[str] = set()  # Accession numbers we've already processed
        self.recent_filings: List[EdgarFiling] = []
        self._thread: Optional[threading.Thread] = None
        self._callbacks: List[Callable] = []

        # Ticker lookup cache (CIK -> ticker)
        self._ticker_cache: Dict[str, str] = {}

        self._load_config()
        self._load_cache()

        logger.info("EdgarMonitor initialized")

    def _load_config(self):
        """Load config from file"""
        try:
            if os.path.exists(CONFIG_FILE):
                with open(CONFIG_FILE, 'r') as f:
                    data = json.load(f)
                    for key, value in data.items():
                        if hasattr(self.config, key):
                            setattr(self.config, key, value)
        except Exception as e:
            logger.error(f"Error loading EDGAR config: {e}")

    def _save_config(self):
        """Save config to file"""
        try:
            with open(CONFIG_FILE, 'w') as f:
                json.dump(self.config.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Error saving EDGAR config: {e}")

    def _load_cache(self):
        """Load seen filings cache"""
        try:
            if os.path.exists(FILINGS_CACHE_FILE):
                with open(FILINGS_CACHE_FILE, 'r') as f:
                    data = json.load(f)
                    self.seen_filings = set(data.get('seen', []))
                    # Keep only last 1000 to prevent unbounded growth
                    if len(self.seen_filings) > 1000:
                        self.seen_filings = set(list(self.seen_filings)[-1000:])
        except Exception as e:
            logger.error(f"Error loading EDGAR cache: {e}")

    def _save_cache(self):
        """Save seen filings cache"""
        try:
            with open(FILINGS_CACHE_FILE, 'w') as f:
                json.dump({'seen': list(self.seen_filings)[-1000:]}, f)
        except Exception as e:
            logger.error(f"Error saving EDGAR cache: {e}")

    def register_callback(self, callback: Callable):
        """Register a callback for new filings"""
        if callback not in self._callbacks:
            self._callbacks.append(callback)

    def _notify_callbacks(self, filing: EdgarFiling):
        """Notify all registered callbacks"""
        for callback in self._callbacks:
            try:
                callback(filing)
            except Exception as e:
                logger.error(f"EDGAR callback error: {e}")

    async def _fetch_recent_filings(self, form_type: str = "8-K") -> List[Dict]:
        """Fetch recent filings from SEC"""
        filings = []

        try:
            # Use SEC's ATOM feed for recent filings
            url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type={form_type}&count=40&output=atom"

            headers = {
                "User-Agent": "MorpheusTradingBot contact@example.com",
                "Accept": "application/atom+xml"
            }

            async with httpx.AsyncClient() as client:
                response = await client.get(url, headers=headers, timeout=10)

                if response.status_code == 200:
                    # Parse ATOM feed
                    content = response.text

                    # Simple regex parsing for ATOM entries
                    entries = re.findall(r'<entry>(.*?)</entry>', content, re.DOTALL)

                    for entry in entries:
                        try:
                            # Extract fields
                            title_match = re.search(r'<title[^>]*>(.*?)</title>', entry)
                            link_match = re.search(r'<link[^>]*href="([^"]*)"', entry)
                            updated_match = re.search(r'<updated>(.*?)</updated>', entry)
                            summary_match = re.search(r'<summary[^>]*>(.*?)</summary>', entry, re.DOTALL)

                            if title_match:
                                title = title_match.group(1)

                                # Parse title: "8-K - Company Name (0001234567)"
                                title_parts = title.split(' - ', 1)
                                filing_type = title_parts[0].strip() if title_parts else form_type

                                company_info = title_parts[1] if len(title_parts) > 1 else ""
                                cik_match = re.search(r'\((\d+)\)', company_info)
                                cik = cik_match.group(1) if cik_match else ""
                                company_name = re.sub(r'\s*\(\d+\)\s*$', '', company_info).strip()

                                # Extract accession number from link
                                accession = ""
                                if link_match:
                                    link = link_match.group(1)
                                    acc_match = re.search(r'/(\d{10}-\d{2}-\d{6})', link)
                                    if acc_match:
                                        accession = acc_match.group(1)

                                filings.append({
                                    'accession_number': accession,
                                    'cik': cik.zfill(10),
                                    'company_name': company_name,
                                    'form_type': filing_type,
                                    'filed_date': updated_match.group(1)[:10] if updated_match else "",
                                    'accepted_time': updated_match.group(1) if updated_match else "",
                                    'url': link_match.group(1) if link_match else "",
                                    'summary': summary_match.group(1) if summary_match else ""
                                })
                        except Exception as e:
                            logger.debug(f"Error parsing entry: {e}")
                            continue

        except Exception as e:
            logger.error(f"Error fetching EDGAR filings: {e}")

        return filings

    async def _lookup_ticker(self, cik: str) -> Optional[str]:
        """Look up ticker symbol from CIK"""
        if cik in self._ticker_cache:
            return self._ticker_cache[cik]

        try:
            # Use SEC company tickers endpoint
            url = f"https://data.sec.gov/submissions/CIK{cik.zfill(10)}.json"
            headers = {"User-Agent": "MorpheusTradingBot contact@example.com"}

            async with httpx.AsyncClient() as client:
                response = await client.get(url, headers=headers, timeout=10)

                if response.status_code == 200:
                    data = response.json()
                    tickers = data.get('tickers', [])
                    if tickers:
                        ticker = tickers[0]
                        self._ticker_cache[cik] = ticker
                        return ticker
        except Exception as e:
            logger.debug(f"Error looking up ticker for CIK {cik}: {e}")

        return None

    def _determine_priority(self, filing: Dict) -> str:
        """Determine priority based on filing type and content"""
        form_type = filing.get('form_type', '').upper()
        summary = filing.get('summary', '').lower()

        # 8-K with material items
        if '8-K' in form_type:
            for item_code in HIGH_PRIORITY_ITEMS:
                if item_code in summary or MATERIAL_8K_ITEMS.get(item_code, '').lower() in summary:
                    return "high"

            # Check for keywords
            high_priority_keywords = ['earnings', 'acquisition', 'merger', 'bankruptcy',
                                     'resignation', 'ceo', 'cfo', 'delisting', 'fda']
            for keyword in high_priority_keywords:
                if keyword in summary:
                    return "high"

        # Form 4 (insider trading)
        if form_type == '4':
            return "normal"

        # 10-K/10-Q (earnings related)
        if form_type in ['10-K', '10-Q']:
            return "high"

        return "normal"

    async def _process_filings(self):
        """Process new filings"""
        for form_type in self.config.form_types:
            try:
                filings = await self._fetch_recent_filings(form_type)

                for filing_data in filings:
                    accession = filing_data.get('accession_number', '')

                    # Skip if already seen
                    if not accession or accession in self.seen_filings:
                        continue

                    # Mark as seen
                    self.seen_filings.add(accession)

                    # Look up ticker
                    cik = filing_data.get('cik', '')
                    ticker = await self._lookup_ticker(cik) if cik else None

                    # Determine priority
                    priority = self._determine_priority(filing_data)

                    # Skip low priority if configured
                    if priority == "low" and self.config.min_priority in ["normal", "high"]:
                        continue
                    if priority == "normal" and self.config.min_priority == "high":
                        continue

                    # Create filing object
                    filing = EdgarFiling(
                        accession_number=accession,
                        cik=cik,
                        company_name=filing_data.get('company_name', ''),
                        form_type=filing_data.get('form_type', ''),
                        filed_date=filing_data.get('filed_date', ''),
                        accepted_time=filing_data.get('accepted_time', ''),
                        ticker=ticker,
                        description=filing_data.get('summary', ''),
                        url=filing_data.get('url', ''),
                        priority=priority
                    )

                    # Add to recent filings
                    self.recent_filings.insert(0, filing)
                    self.recent_filings = self.recent_filings[:100]  # Keep last 100

                    # Log it
                    ticker_str = f"[{ticker}]" if ticker else ""
                    logger.info(f"NEW EDGAR FILING: {filing.form_type} {ticker_str} {filing.company_name} - Priority: {priority}")

                    # Notify callbacks
                    self._notify_callbacks(filing)

                    # Auto-add to watchlist if enabled and has ticker
                    if self.config.auto_add_to_watchlist and ticker and priority == "high":
                        await self._add_to_watchlist(ticker)

            except Exception as e:
                logger.error(f"Error processing {form_type} filings: {e}")

        # Save cache periodically
        self._save_cache()

    async def _add_to_watchlist(self, ticker: str):
        """Add ticker to watchlist via API"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "http://localhost:9100/api/worklist/add",
                    json={"symbol": ticker},
                    timeout=5
                )
                if response.status_code == 200:
                    logger.info(f"EDGAR: Added {ticker} to watchlist")
        except Exception as e:
            logger.error(f"Error adding {ticker} to watchlist: {e}")

    async def _monitor_loop(self):
        """Main monitoring loop"""
        logger.info("EDGAR monitor started")

        while self.is_running:
            try:
                await self._process_filings()
            except Exception as e:
                logger.error(f"EDGAR monitor error: {e}")

            # Wait before next poll
            await asyncio.sleep(self.config.poll_interval_seconds)

        logger.info("EDGAR monitor stopped")

    def _thread_func(self):
        """Thread function to run async loop"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._monitor_loop())
        finally:
            loop.close()

    def start(self):
        """Start the EDGAR monitor"""
        if self.is_running:
            logger.info("EDGAR monitor already running")
            return

        self.is_running = True
        self._thread = threading.Thread(target=self._thread_func, daemon=True)
        self._thread.start()
        logger.info("EDGAR monitor thread started")

    def stop(self):
        """Stop the EDGAR monitor"""
        self.is_running = False
        self._save_cache()
        logger.info("EDGAR monitor stopped")

    def get_recent_filings(self, limit: int = 20) -> List[Dict]:
        """Get recent filings"""
        return [f.to_dict() for f in self.recent_filings[:limit]]

    def get_status(self) -> Dict:
        """Get monitor status"""
        return {
            "is_running": self.is_running,
            "config": self.config.to_dict(),
            "filings_seen": len(self.seen_filings),
            "recent_filings_count": len(self.recent_filings),
            "ticker_cache_size": len(self._ticker_cache)
        }

    def update_config(self, **kwargs):
        """Update configuration"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        self._save_config()


# Singleton instance
_edgar_monitor: Optional[EdgarMonitor] = None


def get_edgar_monitor() -> EdgarMonitor:
    """Get the EDGAR monitor singleton"""
    global _edgar_monitor
    if _edgar_monitor is None:
        _edgar_monitor = EdgarMonitor()
    return _edgar_monitor
