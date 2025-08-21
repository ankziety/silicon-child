"""Browser module for fetching web content with real browser capabilities and interactive actions."""

import hashlib
import threading
import time
import urllib.parse
import urllib.request
import urllib.robotparser
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

from playwright.sync_api import (
    ElementHandle,
    sync_playwright,
)
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


class InteractiveElement(BaseModel):
    """Represents an interactive element on the page."""

    tag_name: str
    element_type: str  # button, input, link, etc.
    text_content: Optional[str]
    placeholder: Optional[str]
    value: Optional[str]
    href: Optional[str]
    id: Optional[str]
    class_name: Optional[str]
    aria_label: Optional[str]
    title: Optional[str]
    is_visible: bool
    is_enabled: bool
    bounding_box: Optional[dict[str, int]]
    selector: str


class PageState(BaseModel):
    """Current state of the page including interactive elements."""

    url: str
    title: str
    interactive_elements: list[InteractiveElement]
    form_inputs: list[InteractiveElement]
    buttons: list[InteractiveElement]
    links: list[InteractiveElement]
    page_content: str
    screenshot_path: Optional[str]


class Browser:
    """Real browser with visual capabilities, screenshots, JavaScript support, and interactive actions."""

    def __init__(
        self, store: Any, user_agent: str = "AI-Infant/0.1.0", headless: bool = False
    ):
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

        # Action history for debugging
        self.action_history: list[dict[str, Any]] = []

        # Confidence threshold for action execution
        self.confidence_threshold = 0.7

        # Internal flag to suppress duplicate logging when execute_action
        # delegates to lower-level methods which also log actions.
        self._suppress_action_logging = False

        # Prepare lazy browser initialization primitives; do not start Playwright
        # here to avoid asyncio loop conflicts in test runners. Actual browser
        # will be started on first real operation via _ensure_browser_started().
        self._init_browser()

    def _init_browser(self):
        """Prepare lazy initialization primitives without starting Playwright."""
        # Playwright will be started lazily in a background thread when needed.
        self.playwright = None
        self.browser = None
        self.page = None

        # Threading primitives for lazy start
        self._start_lock = threading.Lock()
        self._init_event = threading.Event()
        self._playwright_thread: Optional[threading.Thread] = None
        self._init_exception: Optional[BaseException] = None

    def _get_robots_parser(self, url: str) -> urllib.robotparser.RobotFileParser:
        """Get or create robots.txt parser for a domain."""
        parsed_url = urlparse(url)
        domain = f"{parsed_url.scheme}://{parsed_url.netloc}"

        if domain not in self.robots_cache:
            robots_url = f"{domain}/robots.txt"
            parser = urllib.robotparser.RobotFileParser()
            parser.set_url(robots_url)

            try:
                # Use a more targeted approach to prevent hanging
                # Set a timeout on the urllib request
                original_timeout = urllib.request.getdefaulttimeout()
                if original_timeout is not None:
                    urllib.request.setdefaulttimeout(2.0)  # 2 second timeout
                    try:
                        parser.read()
                    finally:
                        urllib.request.setdefaulttimeout(original_timeout)
                else:
                    parser.read()
                self.robots_cache[domain] = parser
            except Exception:
                # If robots.txt fails, create a permissive parser
                parser = urllib.robotparser.RobotFileParser()
                # Create a permissive parser by not setting any restrictions
                self.robots_cache[domain] = parser

        return self.robots_cache[domain]

    def _start_playwright_thread(self) -> None:
        """Start Playwright in a dedicated background thread to avoid
        asyncio loop conflicts with the test runner.

        This method is intended to be run once inside a thread. It will set
        `self.page`, `self.browser`, and `self.playwright` on success and
        signal `self._init_event`. Any exception is captured and also
        signaled to the waiting thread.
        """
        try:
            pw = sync_playwright().start()
            browser = pw.chromium.launch(
                headless=self.headless,
                args=[
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-web-security",
                    "--disable-features=VizDisplayCompositor",
                    "--user-agent=" + self.user_agent,
                ],
            )
            page = browser.new_page(
                viewport={"width": 1920, "height": 1080}, user_agent=self.user_agent
            )

            # Best-effort headers/event listeners
            try:
                page.set_extra_http_headers(
                    {
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                        "Accept-Language": "en-US,en;q=0.5",
                    }
                )
            except Exception:
                pass

            self.playwright = pw
            self.browser = browser
            self.page = page
        except BaseException as e:
            # Store exception for the waiting thread to inspect
            self._init_exception = e
        finally:
            self._init_event.set()

    def _ensure_browser_started(self, timeout: float = 5.0) -> None:
        """Ensure the Playwright browser has been started. This will kick off
        the background thread on first call and wait for initialization to
        complete or timeout.
        """
        # Fast path
        if self.page is not None:
            return

        with self._start_lock:
            if self.page is not None:
                return

            # Start the background thread
            self._init_event.clear()
            self._playwright_thread = threading.Thread(
                target=self._start_playwright_thread, daemon=True
            )
            self._playwright_thread.start()

        # Wait for initialization
        initialized = self._init_event.wait(timeout=timeout)
        if not initialized:
            raise RuntimeError("Timed out starting Playwright browser")
        if self._init_exception:
            # Reraise the captured initialization exception
            raise self._init_exception

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
            # Ensure browser is available
            if self.page is None:
                self._ensure_browser_started()

            # Wait for page to load completely
            try:
                self.page.wait_for_load_state("networkidle", timeout=10000)
            except Exception:
                # Some pages may not support this
                pass

            # Generate screenshot filename
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
            screenshot_filename = f"{timestamp}_{url_hash}.png"
            screenshot_path = self.screenshot_dir / screenshot_filename

            # Take full page screenshot
            self.page.screenshot(path=str(screenshot_path), full_page=True)

            return str(screenshot_path)

        except Exception as e:
            print(f"Failed to take screenshot: {e}")
            return None

    def _extract_page_metadata(self) -> tuple[Optional[str], Optional[str]]:
        """Extract page title and description."""
        try:
            if self.page is None:
                self._ensure_browser_started()

            title = self.page.title()

            # Try to get meta description
            description = None
            try:
                description_elem = self.page.query_selector('meta[name="description"]')
                if description_elem:
                    description = description_elem.get_attribute("content")
            except Exception:
                pass

            return title, description

        except Exception:
            return None, None

    def _extract_interactive_elements(self) -> list[InteractiveElement]:
        """Extract all interactive elements from the current page."""
        elements = []

        try:
            if self.page is None:
                self._ensure_browser_started()

            # Wait for page to be ready
            self.page.wait_for_load_state("domcontentloaded", timeout=10000)

            # Get all interactive elements
            selectors = [
                "button",
                "input",
                "select",
                "textarea",
                "a[href]",
                "[role='button']",
                "[role='link']",
                "[role='menuitem']",
                "[onclick]",
                "[tabindex]",
                "[contenteditable='true']",
            ]

            for selector in selectors:
                try:
                    element_handles = self.page.query_selector_all(selector)

                    for handle in element_handles:
                        try:
                            # Get element properties
                            tag_name = handle.evaluate("el => el.tagName.toLowerCase()")

                            # Determine element type
                            element_type = self._determine_element_type(
                                handle, tag_name
                            )

                            # Get text content
                            text_content = handle.evaluate(
                                "el => el.textContent?.trim() || ''"
                            )
                            if not text_content:
                                text_content = handle.evaluate(
                                    "el => el.innerText?.trim() || ''"
                                )

                            # Get other attributes
                            placeholder = handle.get_attribute("placeholder")
                            value = handle.get_attribute("value")
                            href = handle.get_attribute("href")
                            element_id = handle.get_attribute("id")
                            class_name = handle.get_attribute("class")
                            aria_label = handle.get_attribute("aria-label")
                            title_attr = handle.get_attribute("title")

                            # Check visibility and enabled state
                            is_visible = handle.is_visible()
                            is_enabled = handle.is_enabled()

                            # Get bounding box
                            bounding_box = handle.bounding_box()

                            # Generate unique selector
                            selector = self._generate_unique_selector(handle)

                            element = InteractiveElement(
                                tag_name=tag_name,
                                element_type=element_type,
                                text_content=text_content if text_content else None,
                                placeholder=placeholder,
                                value=value,
                                href=href,
                                id=element_id,
                                class_name=class_name,
                                aria_label=aria_label,
                                title=title_attr,
                                is_visible=is_visible,
                                is_enabled=is_enabled,
                                bounding_box=bounding_box,
                                selector=selector,
                            )

                            elements.append(element)

                        except Exception as e:
                            print(f"Error processing element: {e}")
                            continue

                except Exception as e:
                    print(f"Error with selector {selector}: {e}")
                    continue

            # Remove duplicates based on selector
            unique_elements = []
            seen_selectors = set()
            for element in elements:
                if element.selector not in seen_selectors:
                    unique_elements.append(element)
                    seen_selectors.add(element.selector)

            return unique_elements

        except Exception as e:
            print(f"Error extracting interactive elements: {e}")
            return []

    def _determine_element_type(self, handle: ElementHandle, tag_name: str) -> str:
        """Determine the type of an interactive element."""
        try:
            if tag_name == "button":
                return "button"
            elif tag_name == "input":
                input_type = handle.get_attribute("type") or "text"
                return f"input_{input_type}"
            elif tag_name == "select":
                return "select"
            elif tag_name == "textarea":
                return "textarea"
            elif tag_name == "a":
                return "link"
            else:
                # Check for ARIA roles
                role = handle.get_attribute("role")
                if role:
                    return f"role_{role}"

                # Check for onclick handlers
                onclick = handle.get_attribute("onclick")
                if onclick:
                    return "clickable"

                # Check for tabindex
                tabindex = handle.get_attribute("tabindex")
                if tabindex:
                    return "focusable"

                return "interactive"

        except Exception:
            return "unknown"

    def _generate_unique_selector(self, handle: ElementHandle) -> str:
        """Generate a unique CSS selector for an element."""
        try:
            # Try ID first
            element_id = handle.get_attribute("id")
            if element_id:
                return f"#{element_id}"

            # Try data attributes
            data_testid = handle.get_attribute("data-testid")
            if data_testid:
                return f"[data-testid='{data_testid}']"

            data_id = handle.get_attribute("data-id")
            if data_id:
                return f"[data-id='{data_id}']"

            # Try aria-label
            aria_label = handle.get_attribute("aria-label")
            if aria_label:
                return f"[aria-label='{aria_label}']"

            # Try title
            title = handle.get_attribute("title")
            if title:
                return f"[title='{title}']"

            # Try class-based selector
            class_name = handle.get_attribute("class")
            if class_name:
                classes = class_name.split()
                if classes:
                    return f".{classes[0]}"

            # Fallback to tag name with position
            tag_name = handle.evaluate("el => el.tagName.toLowerCase()")
            return f"{tag_name}"

        except Exception:
            return "unknown"

    def get_page_state(self) -> PageState:
        """Get the current state of the page including all interactive elements."""
        try:
            if self.page is None:
                self._ensure_browser_started()

            # Wait for page to be ready
            self.page.wait_for_load_state("domcontentloaded", timeout=10000)

            # Get current URL and title
            url = self.page.url
            title = self.page.title()

            # Extract interactive elements
            all_elements = self._extract_interactive_elements()

            # Categorize elements
            form_inputs = [
                e for e in all_elements if e.element_type.startswith("input_")
            ]
            buttons = [
                e
                for e in all_elements
                if e.element_type == "button" or "button" in e.element_type
            ]
            links = [e for e in all_elements if e.element_type == "link"]

            # Get page content
            page_content = self.page.content()

            # Take screenshot
            screenshot_path = self._take_screenshot(url)

            return PageState(
                url=url,
                title=title,
                interactive_elements=all_elements,
                form_inputs=form_inputs,
                buttons=buttons,
                links=links,
                page_content=page_content,
                screenshot_path=screenshot_path,
            )

        except Exception as e:
            print(f"Error getting page state: {e}")
            return PageState(
                url=self.page.url if self.page else "",
                title="",
                interactive_elements=[],
                form_inputs=[],
                buttons=[],
                links=[],
                page_content="",
                screenshot_path=None,
            )

    def click_element(
        self,
        selector: str,
        wait_for_navigation: bool = True,
        confidence: Optional[float] = None,
    ) -> bool:
        """Click an element on the page.

        Args:
            selector: CSS selector for the element
            wait_for_navigation: Whether to wait for navigation after click
            confidence: Confidence score (0.0 to 1.0). If provided, validates against threshold.

        Returns:
            bool: True if element was clicked successfully
        """
        try:
            if self.page is None:
                self._ensure_browser_started()
            # Validate confidence if provided
            if confidence is not None and confidence < self.confidence_threshold:
                print(
                    f"Click action skipped due to low confidence: {confidence:.2f} < {self.confidence_threshold:.2f}"
                )
                print(f"Selector: {selector}")

                # Log skipped action
                action_data = {
                    "action": "click",
                    "selector": selector,
                    "confidence": confidence,
                    "reason": "low_confidence",
                    "timestamp": datetime.utcnow().isoformat(),
                    "url": self.page.url,
                }
                self.action_history.append(action_data)
                return False

            # Log action unless suppressed by execute_action
            if not getattr(self, "_suppress_action_logging", False):
                action_data = {
                    "action": "click",
                    "selector": selector,
                    "confidence": confidence,
                    "timestamp": datetime.utcnow().isoformat(),
                    "url": self.page.url,
                }
                self.action_history.append(action_data)

            # Wait for element to be visible and clickable
            element = self.page.wait_for_selector(selector, timeout=10000)
            if not element:
                print(f"Element not found: {selector}")
                return False

            # Ensure element is visible and enabled
            if not element.is_visible() or not element.is_enabled():
                print(f"Element not clickable: {selector}")
                return False

            # Click the element
            if wait_for_navigation:
                # Use click with navigation wait
                element.click()
                # Wait for navigation to complete
                self.page.wait_for_load_state("networkidle", timeout=15000)
            else:
                # Click without waiting for navigation
                element.click()

            print(f"Successfully clicked: {selector}")
            return True

        except Exception as e:
            print(f"Error clicking element {selector}: {e}")
            return False

    def fill_form(
        self, form_data: dict[str, str], confidence: Optional[float] = None
    ) -> bool:
        """Fill form inputs with provided data.

        Args:
            form_data: Dictionary mapping field names to values
            confidence: Confidence score (0.0 to 1.0). If provided, validates against threshold.

        Returns:
            bool: True if at least one field was filled successfully
        """
        try:
            if self.page is None:
                self._ensure_browser_started()
            # Validate confidence if provided
            if confidence is not None and confidence < self.confidence_threshold:
                print(
                    f"Fill form action skipped due to low confidence: {confidence:.2f} < {self.confidence_threshold:.2f}"
                )
                print(f"Form data: {form_data}")

                # Log skipped action
                action_data = {
                    "action": "fill_form",
                    "form_data": form_data,
                    "confidence": confidence,
                    "reason": "low_confidence",
                    "timestamp": datetime.utcnow().isoformat(),
                    "url": self.page.url,
                }
                self.action_history.append(action_data)
                return False

            # Log action unless suppressed by execute_action
            if not getattr(self, "_suppress_action_logging", False):
                action_data = {
                    "action": "fill_form",
                    "form_data": form_data,
                    "confidence": confidence,
                    "timestamp": datetime.utcnow().isoformat(),
                    "url": self.page.url,
                }
                self.action_history.append(action_data)

            success_count = 0

            for field_name, value in form_data.items():
                try:
                    # Try multiple selectors for the field
                    selectors = [
                        f"input[name='{field_name}']",
                        f"input[id='{field_name}']",
                        f"input[placeholder*='{field_name}']",
                        f"textarea[name='{field_name}']",
                        f"textarea[id='{field_name}']",
                        f"select[name='{field_name}']",
                        f"select[id='{field_name}']",
                    ]

                    field_filled = False
                    for selector in selectors:
                        try:
                            element = self.page.query_selector(selector)
                            if element and element.is_visible():
                                # Clear existing value
                                element.fill("")
                                # Fill with new value
                                element.fill(value)
                                print(f"Filled {field_name} with: {value}")
                                success_count += 1
                                field_filled = True
                                break
                        except Exception:
                            continue

                    if not field_filled:
                        print(f"Could not find field: {field_name}")

                except Exception as e:
                    print(f"Error filling field {field_name}: {e}")
                    continue

            print(f"Successfully filled {success_count}/{len(form_data)} form fields")
            return success_count > 0

        except Exception as e:
            print(f"Error filling form: {e}")
            return False

    def navigate_to(self, url: str) -> bool:
        """Navigate to a URL."""
        try:
            if self.page is None:
                self._ensure_browser_started()
            # Log action
            action_data = {
                "action": "navigate",
                "url": url,
                "timestamp": datetime.utcnow().isoformat(),
                "previous_url": self.page.url,
            }
            self.action_history.append(action_data)

            # Navigate to URL
            response = self.page.goto(url, wait_until="networkidle", timeout=30000)

            if response and response.status < 400:
                print(f"Successfully navigated to: {url}")
                return True
            else:
                print(f"Navigation failed: {url}")
                return False

        except Exception as e:
            print(f"Error navigating to {url}: {e}")
            return False

    def scroll_to_element(self, selector: str) -> bool:
        """Scroll to make an element visible."""
        try:
            element = self.page.query_selector(selector)
            if element:
                element.scroll_into_view_if_needed()
                print(f"Scrolled to element: {selector}")
                return True
            else:
                print(f"Element not found for scrolling: {selector}")
                return False

        except Exception as e:
            print(f"Error scrolling to element {selector}: {e}")
            return False

    def wait_for_element(self, selector: str, timeout: int = 10000) -> bool:
        """Wait for an element to appear on the page."""
        try:
            if self.page is None:
                self._ensure_browser_started()

            self.page.wait_for_selector(selector, timeout=timeout)
            print(f"Element appeared: {selector}")
            return True
        except Exception:
            print(f"Element did not appear: {selector}")
            return False

    def select_option(self, selector: str, value: str) -> bool:
        """Select an option from a dropdown/select element."""
        try:
            if self.page is None:
                self._ensure_browser_started()
            element = self.page.query_selector(selector)
            if element:
                element.select_option(value)
                print(f"Selected option '{value}' from: {selector}")
                return True
            else:
                print(f"Select element not found: {selector}")
                return False
        except Exception as e:
            print(f"Error selecting option from {selector}: {e}")
            return False

    def hover_element(self, selector: str) -> bool:
        """Hover over an element on the page."""
        try:
            if self.page is None:
                self._ensure_browser_started()
            element = self.page.query_selector(selector)
            if element:
                element.hover()
                print(f"Hovered over element: {selector}")
                return True
            else:
                print(f"Element not found for hover: {selector}")
                return False
        except Exception as e:
            print(f"Error hovering over element {selector}: {e}")
            return False

    def execute_javascript(self, script: str) -> Any:
        """Execute JavaScript code on the page."""
        try:
            if self.page is None:
                self._ensure_browser_started()

            result = self.page.evaluate(script)
            print(f"Executed JavaScript: {script[:50]}...")
            return result
        except Exception as e:
            print(f"Error executing JavaScript: {e}")
            return None

    def get_element_text(self, selector: str) -> Optional[str]:
        """Get the text content of an element."""
        try:
            if self.page is None:
                self._ensure_browser_started()

            element = self.page.query_selector(selector)
            if element:
                return element.text_content()
            return None
        except Exception as e:
            print(f"Error getting element text: {e}")
            return None

    def find_elements_by_text(
        self, text: str, partial_match: bool = True
    ) -> list[InteractiveElement]:
        """Find elements containing specific text."""
        try:
            elements = []
            all_elements = self._extract_interactive_elements()

            for element in all_elements:
                if element.text_content:
                    if partial_match:
                        if text.lower() in element.text_content.lower():
                            elements.append(element)
                    else:
                        if text.lower() == element.text_content.lower():
                            elements.append(element)

            return elements
        except Exception as e:
            print(f"Error finding elements by text: {e}")
            return []

    def click_element_by_text(self, text: str, partial_match: bool = True) -> bool:
        """Click an element containing specific text."""
        try:
            elements = self.find_elements_by_text(text, partial_match)

            for element in elements:
                if element.is_visible and element.is_enabled:
                    return self.click_element(element.selector)

            print(f"No clickable element found with text: {text}")
            return False

        except Exception as e:
            print(f"Error clicking element by text: {e}")
            return False

    def execute_action(
        self, action_type: str, selector: str, confidence: float, **kwargs
    ) -> bool:
        """Execute an action on the page with confidence validation.

        Args:
            action_type: Type of action ('click', 'fill', 'select', 'hover', etc.)
            selector: CSS selector for the element
            confidence: Confidence score (0.0 to 1.0)
            **kwargs: Additional parameters for the action

        Returns:
            bool: True if action was executed, False otherwise
        """
        try:
            # Validate confidence threshold
            if confidence < self.confidence_threshold:
                print(
                    f"Action skipped due to low confidence: {confidence:.2f} < {self.confidence_threshold:.2f}"
                )
                print(f"Action: {action_type} on selector: {selector}")

                # Log skipped action
                action_data = {
                    "action": action_type,
                    "selector": selector,
                    "confidence": confidence,
                    "reason": "low_confidence",
                    "timestamp": datetime.utcnow().isoformat(),
                    "url": self.page.url if self.page else "",
                }
                self.action_history.append(action_data)
                return False

            # Log action
            action_data = {
                "action": action_type,
                "selector": selector,
                "confidence": confidence,
                "timestamp": datetime.utcnow().isoformat(),
                "url": self.page.url if self.page else "",
            }
            self.action_history.append(action_data)

            # Execute action based on type. Suppress underlying method logging
            # to avoid duplicate entries in action_history since execute_action
            # already logs the action.
            self._suppress_action_logging = True
            try:
                if action_type == "click":
                    return self.click_element(selector, confidence=confidence)
                elif action_type == "fill":
                    value = kwargs.get("value", "")
                    if value:
                        # For fill action, we need to find the field name from the selector
                        # This is a simplified approach - in practice you'd want more sophisticated field detection
                        field_name = (
                            selector.replace("#", "")
                            .replace(".", "")
                            .replace("[", "")
                            .replace("]", "")
                        )
                        return self.fill_form(
                            {field_name: value}, confidence=confidence
                        )
                    else:
                        print("Fill action requires 'value' parameter")
                        return False
                elif action_type == "select":
                    value = kwargs.get("value", "")
                    if value:
                        return self.select_option(selector, value)
                    else:
                        print("Select action requires 'value' parameter")
                        return False
                elif action_type == "navigate":
                    # For navigate, selector is actually the URL
                    return self.navigate_to(selector)
                elif action_type == "hover":
                    return self.hover_element(selector)
                elif action_type == "scroll":
                    return self.scroll_to_element(selector)
                elif action_type == "wait":
                    timeout = kwargs.get("timeout", 10000)
                    return self.wait_for_element(selector, timeout)
                else:
                    print(f"Unknown action type: {action_type}")
                    return False
            finally:
                self._suppress_action_logging = False

        except Exception as e:
            print(f"Error executing action {action_type} on {selector}: {e}")
            # Ensure suppression flag is cleared on error
            self._suppress_action_logging = False
            return False

    def execute_action_struct(
        self, action_type: str, selector: str, confidence: float, **kwargs
    ) -> dict:
        """Execute an action and return a structured result dict.

        Returns:
            dict with at least 'ok' (bool). On failure includes 'reason' and 'details'.
        """
        ok = self.execute_action(action_type, selector, confidence, **kwargs)
        if ok:
            return {"ok": True, "result": {}}

        # Find the most recent action history entry for context
        last = self.action_history[-1] if self.action_history else {}
        reason = last.get("reason", "low_confidence")
        return {"ok": False, "reason": reason, "details": last}

    def set_confidence_threshold(self, threshold: float) -> None:
        """Set the confidence threshold for action execution.

        Args:
            threshold: New confidence threshold (0.0 to 1.0)
        """
        if 0.0 <= threshold <= 1.0:
            self.confidence_threshold = threshold
        else:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")

    def get_confidence_threshold(self) -> float:
        """Get the current confidence threshold."""
        return self.confidence_threshold

    def get_action_history(self) -> list[dict[str, Any]]:
        """Get the history of actions performed."""
        return self.action_history.copy()

    def clear_action_history(self) -> None:
        """Clear the action history."""
        self.action_history.clear()

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
                error_data = {
                    "type": "no_response",
                    "message": "No response from page",
                    "stack": None,
                }
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

    def search(self, query: str, max_results: int = 10) -> list[str]:
        """Enhanced search with dynamic link discovery and multiple sources."""
        urls = []
        discovered_urls = set()

        # Multiple search engines for better coverage
        search_engines = [
            f"https://www.google.com/search?q={urllib.parse.quote(query)}",
            f"https://www.bing.com/search?q={urllib.parse.quote(query)}",
            f"https://duckduckgo.com/html/?q={urllib.parse.quote(query)}",
            f"https://en.wikipedia.org/w/index.php?search={urllib.parse.quote(query)}",
            f"https://scholar.google.com/scholar?q={urllib.parse.quote(query)}",
            f"https://www.reddit.com/search?q={urllib.parse.quote(query)}",
            f"https://www.youtube.com/results?search_query={urllib.parse.quote(query)}",
            f"https://www.amazon.com/s?k={urllib.parse.quote(query)}",
            f"https://www.ebay.com/sch/i.html?_nkw={urllib.parse.quote(query)}",
            f"https://www.github.com/search?q={urllib.parse.quote(query)}",
        ]

        print(f"🔍 Searching for: {query}")

        for search_url in search_engines:
            if len(urls) >= max_results:
                break

            try:
                print(f"   Searching: {search_url}")
                self.page.goto(search_url, wait_until="networkidle", timeout=15000)

                # Extract links based on the search engine
                if "duckduckgo.com" in search_url:
                    links = self.page.query_selector_all(
                        'a[data-testid="result-title-a"]'
                    )
                elif "google.com" in search_url:
                    links = self.page.query_selector_all('a[href^="/url?"]')
                elif "bing.com" in search_url:
                    # For Bing, look for actual result links, not redirects
                    links = self.page.query_selector_all(
                        'a[href^="http"]:not([href*="bing.com"]):not([href*="microsoft.com"])'
                    )
                elif "wikipedia.org" in search_url:
                    links = self.page.query_selector_all('a[href^="/wiki/"]')
                    # Convert relative URLs to absolute
                    links = [
                        f"https://en.wikipedia.org{link.get_attribute('href')}"
                        for link in links
                        if link.get_attribute("href")
                    ]
                elif "scholar.google.com" in search_url:
                    links = self.page.query_selector_all('a[href^="http"]')
                else:
                    links = self.page.query_selector_all('a[href^="http"]')

                # Process and filter links
                for link in links:
                    if len(urls) >= max_results:
                        break

                    if hasattr(link, "get_attribute"):
                        href = link.get_attribute("href")
                    else:
                        href = link  # Already a string

                    if href and self._is_relevant_url(href, query):
                        if href not in discovered_urls:  # Avoid duplicates
                            discovered_urls.add(href)
                            urls.append(href)
                            print(f"   Found: {href}")

                            # Follow promising links to discover more content
                            if len(urls) < max_results and self._should_follow_link(
                                href, query
                            ):
                                try:
                                    print(f"   Following: {href}")
                                    self.page.goto(
                                        href, wait_until="networkidle", timeout=10000
                                    )

                                    # Extract links from the page
                                    page_links = self.page.query_selector_all(
                                        'a[href^="http"]'
                                    )
                                    for page_link in page_links[
                                        :5
                                    ]:  # Limit to 5 links per page
                                        if len(urls) >= max_results:
                                            break

                                        page_href = page_link.get_attribute("href")
                                        if page_href and self._is_relevant_url(
                                            page_href, query
                                        ):
                                            if page_href not in discovered_urls:
                                                discovered_urls.add(page_href)
                                                urls.append(page_href)
                                                print(f"     Discovered: {page_href}")

                                    # Go back to search results
                                    self.page.go_back()

                                except Exception as e:
                                    print(f"     Failed to follow {href}: {e}")
                                    continue

            except Exception as e:
                print(f"   Search failed for {search_url}: {e}")
                continue

        # Fallback to reliable sources if needed
        if len(urls) < max_results // 2:
            reliable_sources = [
                "https://en.wikipedia.org/wiki/Main_Page",
                "https://github.com/trending",
                "https://news.ycombinator.com/",
                "https://arxiv.org/",
            ]

            for source in reliable_sources:
                if len(urls) >= max_results:
                    break
                if source not in discovered_urls:
                    urls.append(source)
                    discovered_urls.add(source)

        print(f"📊 Total URLs found: {len(urls)}")
        return urls[:max_results]

    def _is_relevant_url(self, url: str, query: str) -> bool:
        """Check if a URL is relevant to the search query."""
        # Skip common irrelevant domains
        skip_domains = [
            "google.com",
            "youtube.com",
            "facebook.com",
            "twitter.com",
            "instagram.com",
            "linkedin.com",
            "reddit.com",
            "amazon.com",
            "ebay.com",
            "wikipedia.org",
        ]

        # Skip if it's a search engine or social media
        if any(domain in url.lower() for domain in skip_domains):
            return False

        # Skip if it's clearly not content (ads, analytics, etc.)
        skip_patterns = [
            "ads",
            "analytics",
            "tracking",
            "pixel",
            "beacon",
            "cookie",
            "login",
            "signup",
            "register",
            "subscribe",
            "newsletter",
        ]

        if any(pattern in url.lower() for pattern in skip_patterns):
            return False

        # Check if query terms appear in URL (basic relevance)
        query_terms = query.lower().split()
        url_lower = url.lower()

        # If any query term appears in URL, consider it relevant
        if any(term in url_lower for term in query_terms if len(term) > 2):
            return True

        # For academic/research queries, prefer certain domains
        academic_domains = [
            "arxiv.org",
            "researchgate.net",
            "scholar.google.com",
            "ieee.org",
            "acm.org",
            "springer.com",
            "sciencedirect.com",
            "nature.com",
            "science.org",
            "cell.com",
            "jstor.org",
            "pubmed.ncbi.nlm.nih.gov",
        ]

        if any(domain in url for domain in academic_domains):
            return True

        # Default to relevant if it passes basic filters
        return True

    def _should_follow_link(self, url: str, query: str) -> bool:
        """Determine if we should follow a link to discover more content."""
        # Don't follow too many links to avoid getting lost
        if len(url) > 200:  # Very long URLs are often not content pages
            return False

        # Prefer following links from reputable domains
        follow_domains = [
            "wikipedia.org",
            "github.com",
            "stackoverflow.com",
            "medium.com",
            "dev.to",
            "arxiv.org",
            "researchgate.net",
            "ieee.org",
            "acm.org",
        ]

        if any(domain in url for domain in follow_domains):
            return True

        # For technical queries, prefer technical sites
        technical_terms = ["programming", "code", "software", "technology", "computer"]
        if any(term in query.lower() for term in technical_terms):
            tech_domains = ["github.com", "stackoverflow.com", "dev.to", "medium.com"]
            if any(domain in url for domain in tech_domains):
                return True

        return False

    def close(self):
        """Close browser and cleanup resources."""
        try:
            if self.page:
                try:
                    self.page.close()
                except Exception as e:
                    if "Event loop is closed" not in str(
                        e
                    ) and "Target closed" not in str(e):
                        print(f"Error closing page: {e}")

            if self.browser:
                try:
                    self.browser.close()
                except Exception as e:
                    if "Event loop is closed" not in str(
                        e
                    ) and "Target closed" not in str(e):
                        print(f"Error closing browser: {e}")

            if self.playwright:
                try:
                    self.playwright.stop()
                except Exception as e:
                    if "Event loop is closed" not in str(
                        e
                    ) and "Target closed" not in str(e):
                        print(f"Error stopping playwright: {e}")

        except Exception as e:
            # Only log if it's not the expected event loop closure or target closed
            if "Event loop is closed" not in str(e) and "Target closed" not in str(e):
                print(f"Error during browser cleanup: {e}")
            # Don't re-raise - this is cleanup code

    def __del__(self):
        """Cleanup on destruction."""
        self.close()
