from typing import Any, Optional

from .browser import Browser


class BrowserTool:
    """Facade tool that provides a stable, test-friendly interface over
    the Browser implementation. It lazily constructs the Browser and simply
    delegates calls. This lets higher-level components depend on the tool
    interface and makes it straightforward to inject mocks in tests.
    """

    def __init__(self, store: Any, headless: bool = False) -> None:
        self._store = store
        self._headless = headless
        self._browser: Optional[Browser] = None

    def _ensure(self) -> Browser:
        if self._browser is None:
            self._browser = Browser(self._store, headless=self._headless)
        return self._browser

    def fetch(self, url: str):
        return self._ensure().fetch(url)

    def search(self, query: str, max_results: int = 10):
        return self._ensure().search(query, max_results=max_results)

    def execute_action(
        self, action_type: str, selector: str, confidence: float, **kwargs
    ):
        return self._ensure().execute_action(
            action_type, selector, confidence, **kwargs
        )

    def execute_action_struct(
        self, action_type: str, selector: str, confidence: float, **kwargs
    ) -> dict:
        """Execute an action and return a structured result dict.

        This is the preferred API for orchestrators. It returns at least:
        - `ok` (bool)
        - if not ok: `reason` (str) and optional `details` (dict)
        """
        b = self._ensure()
        # If the underlying Browser implements a structured result, use it
        if hasattr(b, "execute_action_struct"):
            return b.execute_action_struct(action_type, selector, confidence, **kwargs)

        # Otherwise, call the boolean API and synthesize a structure
        ok = b.execute_action(action_type, selector, confidence, **kwargs)
        if ok:
            return {"ok": True, "result": {}}
        # Try to find last logged action to extract reason
        history = getattr(b, "action_history", [])
        last = history[-1] if history else {}
        return {"ok": False, "reason": last.get("reason", "unknown"), "details": last}

    def set_confidence_threshold(self, threshold: float) -> None:
        return self._ensure().set_confidence_threshold(threshold)

    def get_confidence_threshold(self) -> float:
        return self._ensure().get_confidence_threshold()

    def get_action_history(self):
        return self._ensure().get_action_history()

    def clear_action_history(self) -> None:
        return self._ensure().clear_action_history()

    def close(self) -> None:
        if self._browser:
            self._browser.close()


def create_browser_tool(store: Any, headless: bool = False) -> BrowserTool:
    return BrowserTool(store, headless=headless)
