"""Browser module for fetching web content with real browser capabilities."""

import hashlib
import time
import urllib.parse
import urllib.robotparser
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

from playwright.sync_api import sync_playwright, Page, Browser as PlaywrightBrowser
from pydantic import BaseModel, Field


class FetchResult(BaseModel):
    """Result of a fetch operation with visual capabilities."""

    url: str
    content: str
    mime_type: str
    size_bytes: int
    status_code: int
    headers: dict[str, str]
    fetch_time: datetime
    checksum: str = Field(description="SHA-256 checksum of content")
    screenshot_path: Optional[str] = Field(description="Path to screenshot of the page")
    page_title: Optional[str] = Field(description="Page title")
    page_description: Optional[str] = Field(description="Page meta description")
    viewport_width: int = Field(default=1920, description="Viewport width")
    viewport_height: int = Field(default=1080, description="Viewport height")


class Browser:
    """Real browser with visual capabilities, screenshots, and JavaScript support."""

    def __init__(self, store: Any, user_agent: str = "AI-Infant/0.1.0", headless: bool = False):
        """Initialize browser with storage and user agent."""
        self.store = store
        self.user_agent = user_agent
        self.headless = headless
        self.rate_limit_delay = 2.0  # seconds between requests
        self.last_request_time = 0.0
        self.robots_cache: dict[str, urllib.robotparser.RobotFileParser] = {}
        
        # Initialize Playwright
        self.playwright = None
        self.browser = None
        self.page = None
        
        # Screenshot directory
        self.screenshot_dir = Path("data/screenshots")
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        
        self._init_browser()

    def _init_browser(self):
        """Initialize Playwright browser."""
        try:
            self.playwright = sync_playwright().start()
            
            # Launch browser with visual capabilities
            self.browser = self.playwright.chromium.launch(
                headless=self.headless,
                args=[
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-web-security",
                    "--disable-features=VizDisplayCompositor",
                    "--user-agent=" + self.user_agent
                ]
            )
            
            # Create new page with large viewport for better screenshots
            self.page = self.browser.new_page(
                viewport={"width": 1920, "height": 1080},
                user_agent=self.user_agent
            )
            
            # Set extra headers
            self.page.set_extra_http_headers({
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate",
                "DNT": "1",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
            })
            
        except Exception as e:
            print(f"Failed to initialize browser: {e}")
            raise

    def _get_robots_parser(self, url: str) -> urllib.robotparser.RobotFileParser:
        """Get or create robots.txt parser for a domain."""
        parsed_url = urlparse(url)
        domain = f"{parsed_url.scheme}://{parsed_url.netloc}"

        if domain not in self.robots_cache:
            robots_url = f"{domain}/robots.txt"
            parser = urllib.robotparser.RobotFileParser()
            parser.set_url(robots_url)

            try:
                parser.read()
                self.robots_cache[domain] = parser
            except Exception:
                # If robots.txt fails, create a permissive parser
                parser = urllib.robotparser.RobotFileParser()
                # Create a permissive parser by not setting any restrictions
                self.robots_cache[domain] = parser

        return self.robots_cache[domain]

    def _can_fetch(self, url: str) -> bool:
        """Check if URL can be fetched according to robots.txt."""
        parser = self._get_robots_parser(url)
        return parser.can_fetch(self.user_agent, url)

    def _rate_limit(self) -> None:
        """Apply rate limiting between requests."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last)
        self.last_request_time = time.time()

    def _take_screenshot(self, url: str) -> Optional[str]:
        """Take a screenshot of the current page."""
        try:
            # Wait for page to load completely
            self.page.wait_for_load_state("networkidle", timeout=10000)
            
            # Generate screenshot filename
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
            screenshot_filename = f"{timestamp}_{url_hash}.png"
            screenshot_path = self.screenshot_dir / screenshot_filename
            
            # Take full page screenshot
            self.page.screenshot(
                path=str(screenshot_path),
                full_page=True,
                quality=90
            )
            
            return str(screenshot_path)
            
        except Exception as e:
            print(f"Failed to take screenshot: {e}")
            return None

    def _extract_page_metadata(self) -> tuple[Optional[str], Optional[str]]:
        """Extract page title and description."""
        try:
            title = self.page.title()
            
            # Try to get meta description
            description = None
            try:
                description_elem = self.page.query_selector('meta[name="description"]')
                if description_elem:
                    description = description_elem.get_attribute("content")
            except:
                pass
                
            return title, description
            
        except Exception:
            return None, None

    def _log_job(
        self,
        job_type: str,
        input_data: dict[str, Any],
        output_data: Optional[dict[str, Any]] = None,
        error_data: Optional[dict[str, Any]] = None,
    ) -> str:
        """Log a job to the store."""
        job_id = f"{job_type}-{int(time.time() * 1000)}"

        job_data = {
            "id": job_id,
            "type": job_type,
            "status": "failed" if error_data else "completed",
            "created_at": datetime.utcnow().isoformat() + "Z",
            "updated_at": datetime.utcnow().isoformat() + "Z",
            "started_at": datetime.utcnow().isoformat() + "Z",
            "completed_at": datetime.utcnow().isoformat() + "Z",
            "input": input_data,
            "output": output_data,
            "error": error_data,
            "metadata": {
                "version": "0.1.0",
                "priority": 5,
                "retries": 0,
                "max_retries": 3,
                "timeout_seconds": 30,
            },
        }

        self.store.store_job(job_data)
        return job_id

    def fetch(self, url: str) -> Optional[FetchResult]:
        """Fetch content from URL with real browser capabilities."""
        start_time = datetime.utcnow()

        # Check robots.txt
        if not self._can_fetch(url):
            error_data = {
                "type": "robots_forbidden",
                "message": f"URL {url} is forbidden by robots.txt",
                "stack": None,
            }
            self._log_job("fetch", {"url": url}, error_data=error_data)
            return None

        # Apply rate limiting
        self._rate_limit()

        try:
            # Navigate to URL with real browser
            response = self.page.goto(url, wait_until="networkidle", timeout=30000)
            
            if not response:
                error_data = {"type": "no_response", "message": "No response from page", "stack": None}
                self._log_job("fetch", {"url": url}, error_data=error_data)
                return None
            
            # Get page content
            content = self.page.content()
            checksum = hashlib.sha256(content.encode("utf-8")).hexdigest()
            
            # Take screenshot
            screenshot_path = self._take_screenshot(url)
            
            # Extract metadata
            title, description = self._extract_page_metadata()
            
            # Get viewport info
            viewport = self.page.viewport_size
            viewport_width = viewport["width"] if viewport else 1920
            viewport_height = viewport["height"] if viewport else 1080

            result = FetchResult(
                url=url,
                content=content,
                mime_type=response.headers.get("content-type", "text/html"),
                size_bytes=len(content.encode("utf-8")),
                status_code=response.status,
                headers=dict(response.headers),
                fetch_time=start_time,
                checksum=checksum,
                screenshot_path=screenshot_path,
                page_title=title,
                page_description=description,
                viewport_width=viewport_width,
                viewport_height=viewport_height,
            )

            # Log successful job
            output_data = {
                "status_code": response.status,
                "size_bytes": result.size_bytes,
                "checksum": checksum,
                "mime_type": result.mime_type,
                "screenshot_path": screenshot_path,
                "page_title": title,
                "viewport_width": viewport_width,
                "viewport_height": viewport_height,
            }
            self._log_job("fetch", {"url": url}, output_data)

            return result

        except Exception as e:
            error_data = {"type": "browser_error", "message": str(e), "stack": None}
            self._log_job("fetch", {"url": url}, error_data=error_data)
            return None

    def search(self, query: str, max_results: int = 5) -> list[str]:
        """Search for URLs using a search engine."""
        try:
            # Use Google search
            search_url = f"https://www.google.com/search?q={urllib.parse.quote(query)}"
            
            # Navigate to search page
            self.page.goto(search_url, wait_until="networkidle", timeout=30000)
            
            # Extract search results
            urls = []
            try:
                # Look for search result links
                links = self.page.query_selector_all('a[href^="http"]')
                
                for link in links[:max_results * 2]:  # Get more to filter
                    href = link.get_attribute("href")
                    if href and not any(skip in href for skip in ["google.com", "youtube.com", "facebook.com"]):
                        urls.append(href)
                        if len(urls) >= max_results:
                            break
                            
            except Exception as e:
                print(f"Failed to extract search results: {e}")
                
            return urls[:max_results]
            
        except Exception as e:
            print(f"Search failed: {e}")
            return []

    def close(self):
        """Close browser and cleanup resources."""
        try:
            if self.page:
                self.page.close()
            if self.browser:
                self.browser.close()
            if self.playwright:
                self.playwright.stop()
        except Exception as e:
            print(f"Error closing browser: {e}")

    def __del__(self):
        """Cleanup on destruction."""
        self.close()
