"""Real web environment integration for DeepResearch."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Literal, Optional, Tuple
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse
from urllib.request import Request, urlopen

from deepresearch.agent.types import PageContent, SearchResult
from deepresearch.envs.base_env import BaseEnv

_READABILITY_JS_URL = "https://cdn.jsdelivr.net/npm/@mozilla/readability@0.5.0/Readability.js"


class RealWebEnv(BaseEnv):
    """Playwright-backed real web environment with API-based search."""

    def __init__(
        self,
        search_api_key: str,
        search_engine: Literal["bing", "serp"] = "bing",
        headless: bool = True,
        request_delay: float = 1.0,
    ) -> None:
        """Initialize runtime settings for search, browser control, and throttling."""
        self.search_api_key = search_api_key
        if search_engine not in {"bing", "serp"}:
            raise ValueError("search_engine must be either 'bing' or 'serp'.")
        self.search_engine: Literal["bing", "serp"] = search_engine
        self.headless = headless
        self.request_delay = max(0.0, request_delay)

        self._playwright: Any = None
        self._browser: Any = None
        self._context: Any = None

        self._rate_limit_lock = asyncio.Lock()
        self._last_fetch_at = 0.0

    async def __aenter__(self) -> "RealWebEnv":
        """Launch the browser and return this environment instance."""
        await self._start_browser()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        """Close Playwright resources when leaving async context."""
        del exc_type, exc, tb
        await self._close_browser()

    async def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Search via Bing v7 API (or SerpAPI fallback) and return ranked results."""
        if not self.search_api_key:
            raise ValueError("search_api_key is required for web search.")

        result_count = max(1, top_k)
        last_error: Exception = RuntimeError("search failed")

        for attempt in range(3):
            try:
                if self.search_engine == "bing":
                    payload = await self._search_bing(query=query, top_k=result_count)
                    raw_results = payload.get("webPages", {}).get("value", [])
                    return [
                        SearchResult(
                            url=str(item.get("url", "")),
                            title=str(item.get("name", "")),
                            snippet=str(item.get("snippet", "")),
                            rank=rank,
                        )
                        for rank, item in enumerate(raw_results[:result_count], start=1)
                    ]

                payload = await self._search_serp(query=query, top_k=result_count)
                raw_results = payload.get("organic_results", [])
                return [
                    SearchResult(
                        url=str(item.get("link", "")),
                        title=str(item.get("title", "")),
                        snippet=str(item.get("snippet", "")),
                        rank=rank,
                    )
                    for rank, item in enumerate(raw_results[:result_count], start=1)
                ]
            except Exception as exc:
                last_error = exc
                if attempt < 2:
                    await asyncio.sleep(3 * (attempt + 1))

        raise last_error

    _SKIP_EXTENSIONS = {
        ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
        ".zip", ".rar", ".gz", ".tar", ".exe", ".dmg",
    }
    # PDFs handled separately via _fetch_pdf (text extraction, no browser).
    _PDF_EXTENSIONS = {".pdf"}

    async def fetch_page(self, url: str) -> PageContent:
        """Open a URL in Playwright, extract readable text, and detect pagination.

        PDF and arXiv PDF URLs are routed to specialized handlers:
        - arXiv /pdf/ URLs auto-rewrite to /abs/ HTML page (faster + has metadata)
        - other PDFs go through pypdf text extraction
        """
        from urllib.parse import urlparse
        parsed = urlparse(url)
        parsed_path = parsed.path.lower()

        if any(parsed_path.endswith(ext) for ext in self._SKIP_EXTENSIONS):
            raise ValueError(f"Skipping non-HTML resource: {url}")

        # arXiv PDF → redirect to abs page (HTML, has title/abstract/links)
        if parsed.netloc.endswith("arxiv.org") and "/pdf/" in parsed_path:
            abs_url = url.replace("/pdf/", "/abs/")
            if abs_url.endswith(".pdf"):
                abs_url = abs_url[:-4]
            return await self._fetch_html_page(abs_url)

        # Other PDFs → pypdf text extraction
        if any(parsed_path.endswith(ext) for ext in self._PDF_EXTENSIONS):
            return await self._fetch_pdf(url)

        return await self._fetch_html_page(url)

    async def _fetch_html_page(self, url: str) -> PageContent:
        """Fetch an HTML page via Playwright + Readability.js (the original path)."""
        await self._ensure_browser()
        await self._apply_fetch_rate_limit()

        if self._context is None:
            raise RuntimeError("Playwright browser context is not initialized.")

        page = await self._context.new_page()
        try:
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=15000)
            except Exception:
                # fallback: accept whatever loaded within the timeout
                pass
            final_url = page.url
            await self._inject_readability(page)
            extracted = await page.evaluate(
                """() => {
                    const links = Array.from(document.querySelectorAll('a[href]'));
                    const nextLink = links.find((node) => {
                        const text = (node.innerText || '').trim();
                        return /(next page|next|下一页|下页|›|»)/i.test(text);
                    });

                    let bodyText = '';
                    try {
                        if (typeof Readability !== 'undefined') {
                            const docClone = document.cloneNode(true);
                            const article = new Readability(docClone).parse();
                            bodyText = (article?.textContent || '').trim();
                        }
                    } catch (error) {
                        void error;
                    }

                    if (!bodyText) {
                        bodyText = (document.body?.innerText || '').trim();
                    }

                    return {
                        title: document.title || '',
                        body: bodyText,
                        links: links.map((node) => node.href).filter(Boolean),
                        next_link: nextLink ? nextLink.href : null,
                    };
                }"""
            )
        finally:
            await page.close()

        title = str(extracted.get("title", "")).strip() or final_url
        body = str(extracted.get("body", "")).strip()
        links = self._unique_links(extracted.get("links", []))
        next_link = extracted.get("next_link")

        current_page = self._extract_current_page(final_url)
        is_paginated = self._detect_pagination(final_url, links, next_link)

        return PageContent(
            url=final_url,
            title=title,
            body=body,
            links=links,
            is_paginated=is_paginated,
            current_page=current_page,
            total_pages=None,
        )

    async def _fetch_pdf(self, url: str) -> PageContent:
        """Download a PDF and extract text via pypdf. Body has page markers."""
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise ValueError(
                f"PDF extraction requires the 'pypdf' package. URL: {url}"
            ) from exc

        await self._apply_fetch_rate_limit()

        def _download_and_extract() -> Tuple[str, str, int]:
            request = Request(url=url, headers={"User-Agent": "Mozilla/5.0 DeepResearch"})
            with urlopen(request, timeout=30) as response:
                data = response.read()
            import io
            reader = PdfReader(io.BytesIO(data))
            metadata = reader.metadata or {}
            title = ""
            try:
                title = str(metadata.get("/Title") or "").strip()
            except Exception:
                title = ""
            pages_text: List[str] = []
            for i, page in enumerate(reader.pages, start=1):
                try:
                    text = (page.extract_text() or "").strip()
                except Exception:
                    text = ""
                if text:
                    pages_text.append(f"[Page {i}]\n{text}")
            return title, "\n\n".join(pages_text), len(reader.pages)

        try:
            title, body, total_pages = await asyncio.to_thread(_download_and_extract)
        except Exception as exc:
            raise ValueError(f"Failed to extract PDF text from {url}: {exc}") from exc

        if not body:
            raise ValueError(f"PDF appears to contain no extractable text: {url}")

        return PageContent(
            url=url,
            title=title or url,
            body=body,
            links=[],
            is_paginated=False,
            current_page=1,
            total_pages=total_pages,
        )

    async def click_link(self, page: PageContent, link_url: str) -> PageContent:
        """Follow a link by URL and return the fetched page content."""
        del page
        return await self.fetch_page(link_url)

    async def get_next_page(self, page: PageContent) -> Optional[PageContent]:
        """Advance pagination by incrementing URL page query param if available."""
        if not page.is_paginated:
            return None

        if page.total_pages is not None and page.current_page >= page.total_pages:
            return None

        next_page_number = page.current_page + 1
        next_url, changed = self._with_incremented_page_param(page.url, next_page_number)
        if not changed:
            return None

        next_page = await self.fetch_page(next_url)
        return next_page.model_copy(
            update={
                "is_paginated": True,
                "current_page": next_page_number,
                "total_pages": page.total_pages,
            }
        )

    async def _ensure_browser(self) -> None:
        """Ensure Playwright browser and context are initialized."""
        if self._context is None:
            await self._start_browser()

    async def _start_browser(self) -> None:
        """Start Playwright and launch Chromium if not already running."""
        if self._context is not None:
            return

        from playwright.async_api import async_playwright

        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(headless=self.headless)
        self._context = await self._browser.new_context(ignore_https_errors=True)

    async def _close_browser(self) -> None:
        """Close browser context, browser, and Playwright runtime safely."""
        if self._context is not None:
            await self._context.close()
            self._context = None

        if self._browser is not None:
            await self._browser.close()
            self._browser = None

        if self._playwright is not None:
            await self._playwright.stop()
            self._playwright = None

    async def _apply_fetch_rate_limit(self) -> None:
        """Throttle fetch requests by delaying calls based on request_delay."""
        if self.request_delay <= 0:
            return

        async with self._rate_limit_lock:
            loop = asyncio.get_running_loop()
            now = loop.time()
            elapsed = now - self._last_fetch_at
            wait_seconds = self.request_delay - elapsed
            if wait_seconds > 0:
                await asyncio.sleep(wait_seconds)
            self._last_fetch_at = loop.time()

    async def _inject_readability(self, page: Any) -> None:
        """Inject Readability.js before extraction; fallback silently if blocked."""
        try:
            await page.add_script_tag(url=_READABILITY_JS_URL)
        except Exception:
            return

    async def _search_bing(self, query: str, top_k: int) -> Dict[str, Any]:
        """Call Bing Search API v7 and return parsed JSON payload."""
        params = urlencode({"q": query, "count": top_k, "responseFilter": "Webpages"})
        url = f"https://api.bing.microsoft.com/v7.0/search?{params}"
        headers = {"Ocp-Apim-Subscription-Key": self.search_api_key}
        return await self._http_get_json(url, headers=headers)

    async def _search_serp(self, query: str, top_k: int) -> Dict[str, Any]:
        """Call SerpAPI and return parsed JSON payload."""
        params = urlencode({"engine": "google", "q": query, "num": top_k, "api_key": self.search_api_key})
        url = f"https://serpapi.com/search.json?{params}"
        return await self._http_get_json(url)

    async def _http_get_json(self, url: str, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Run an HTTP GET request and decode a JSON response body."""

        def _request() -> Dict[str, Any]:
            request = Request(url=url, headers=headers or {}, method="GET")
            with urlopen(request, timeout=30) as response:
                payload = response.read().decode("utf-8")
            parsed = json.loads(payload)
            if not isinstance(parsed, dict):
                raise ValueError("Expected JSON object response.")
            return parsed

        return await asyncio.to_thread(_request)

    @staticmethod
    def _unique_links(raw_links: Any) -> List[str]:
        """Normalize, deduplicate, and preserve order for extracted links."""
        if not isinstance(raw_links, list):
            return []

        seen = set()
        unique: List[str] = []
        for item in raw_links:
            link = str(item).strip()
            if not link or link in seen:
                continue
            seen.add(link)
            unique.append(link)
        return unique

    @staticmethod
    def _extract_current_page(url: str) -> int:
        """Extract current page index from the URL query, defaulting to 1."""
        parsed = urlparse(url)
        query = parse_qs(parsed.query)
        for key in ("page", "p", "pg"):
            value = query.get(key, [None])[0]
            if value is None:
                continue
            try:
                page_number = int(value)
            except (TypeError, ValueError):
                continue
            if page_number >= 1:
                return page_number
        return 1

    @staticmethod
    def _detect_pagination(url: str, links: List[str], next_link: Any) -> bool:
        """Detect pagination hints from URL params, links, or explicit next-page links."""
        if isinstance(next_link, str) and next_link.strip():
            return True

        url_lower = url.lower()
        if "page=" in url_lower:
            return True

        for link in links:
            if "page=" in link.lower():
                return True

        return False

    @staticmethod
    def _with_incremented_page_param(url: str, page_number: int) -> Tuple[str, bool]:
        """Return URL with incremented page query parameter and whether it changed."""
        parsed = urlparse(url)
        query = parse_qs(parsed.query)

        query["page"] = [str(page_number)]
        encoded_query = urlencode(query, doseq=True)
        updated = parsed._replace(query=encoded_query)
        new_url = urlunparse(updated)
        return new_url, new_url != url

