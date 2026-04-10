"""Web Search — DuckDuckGo search + page content extraction.

Lightweight internet access for the autonomous research mode.
No API keys needed — uses DuckDuckGo's public search.
"""

import asyncio
import logging
import re
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single search result."""

    title: str
    url: str
    snippet: str


class WebSearcher:
    """Web search and page content extraction."""

    def __init__(self, max_content_chars: int = 3000) -> None:
        self._max_content_chars = max_content_chars

    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        """Search DuckDuckGo and return results."""
        try:
            results = await asyncio.get_event_loop().run_in_executor(
                None, self._search_sync, query, max_results,
            )
            return results
        except Exception:
            logger.warning("Web search failed for query: %s", query, exc_info=True)
            return []

    def _search_sync(self, query: str, max_results: int) -> list[SearchResult]:
        """Synchronous DuckDuckGo search (runs in thread)."""
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS

        results: list[SearchResult] = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append(SearchResult(
                    title=r.get("title", ""),
                    url=r.get("href", ""),
                    snippet=r.get("body", ""),
                ))
        return results

    async def fetch_content(self, url: str) -> str:
        """Fetch a page and extract plain text content.

        Uses Wikipedia API for wikipedia.org URLs (avoids 403).
        Falls back to direct fetch with browser-like headers for other sites.
        Returns extracted text (max self._max_content_chars chars).
        Returns empty string on failure.
        """
        # Wikipedia: use their public API instead of scraping
        if "wikipedia.org/wiki/" in url:
            return await self._fetch_wikipedia(url)

        # Skip known binary/non-text URLs
        lower_url = url.lower()
        if any(lower_url.endswith(ext) for ext in (".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".zip", ".gz", ".tar", ".mp3", ".mp4", ".jpg", ".png", ".gif")):
            logger.debug("Skipping binary URL: %s", url)
            return ""

        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(10.0),
                follow_redirects=True,
                headers=self._browser_headers(),
            ) as client:
                resp = await client.get(url)
                resp.raise_for_status()

                # Check content-type for binary responses
                content_type = resp.headers.get("content-type", "")
                if not any(t in content_type for t in ("text/html", "text/plain", "application/xhtml", "application/xml")):
                    logger.debug("Skipping non-text content-type: %s for %s", content_type, url)
                    return ""

                html = resp.text
        except Exception:
            logger.warning("Page fetch failed: %s", url, exc_info=True)
            return ""

        # Final check: if content starts with PDF magic bytes or binary garbage
        if html.startswith("%PDF") or html.startswith("\x00") or "\ufffd" in html[:200]:
            return ""

        return self._extract_text(html)

    async def _fetch_wikipedia(self, url: str) -> str:
        """Fetch Wikipedia article via their REST API (no 403 issues)."""
        # Extract article title from URL
        try:
            title = url.split("/wiki/")[-1].split("#")[0].split("?")[0]
        except (IndexError, ValueError):
            return ""

        api_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(10.0),
                headers={"User-Agent": "PathosEngine/1.0 (research tool; https://github.com/pathos-engine)"},
            ) as client:
                resp = await client.get(api_url)
                resp.raise_for_status()
                data = resp.json()
                extract = data.get("extract", "")
                if not extract:
                    return ""
                return extract[: self._max_content_chars]
        except Exception:
            logger.warning("Wikipedia API failed for: %s", title, exc_info=True)
            return ""

    @staticmethod
    def _browser_headers() -> dict[str, str]:
        """Return browser-like headers to reduce 403 rejections."""
        return {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }

    def _extract_text(self, html: str) -> str:
        """Strip HTML tags and extract readable text."""
        # Remove script, style, nav, header, footer blocks
        for tag in ("script", "style", "nav", "header", "footer", "aside"):
            html = re.sub(rf"<{tag}[^>]*>.*?</{tag}>", " ", html, flags=re.DOTALL | re.IGNORECASE)

        # Remove all remaining HTML tags
        text = re.sub(r"<[^>]+>", " ", html)

        # Decode HTML entities
        text = re.sub(r"&nbsp;", " ", text)
        text = re.sub(r"&amp;", "&", text)
        text = re.sub(r"&lt;", "<", text)
        text = re.sub(r"&gt;", ">", text)
        text = re.sub(r"&quot;", '"', text)
        text = re.sub(r"&#\d+;", " ", text)

        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text[: self._max_content_chars]
