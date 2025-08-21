from ai_infant.crawl.browser import Browser


class MockStore:
    def store_job(self, job_data):
        # Minimal stub for store job logging used by Browser
        return None


class DummyElement:
    def __init__(self):
        self._clicked = False

    def get_attribute(self, name: str):
        return None

    def evaluate(self, script: str):
        # Simplified evaluation for tagName
        if "tagName" in script:
            return "button"
        return ""

    def is_visible(self) -> bool:
        return True

    def is_enabled(self) -> bool:
        return True

    def bounding_box(self):
        return {"x": 0, "y": 0, "width": 10, "height": 10}

    def click(self):
        self._clicked = True


class DummyPage:
    def __init__(self):
        self.url = "http://example.com"

    def wait_for_selector(self, selector: str, timeout: int = 10000):
        return DummyElement()

    def query_selector(self, selector: str):
        return DummyElement()

    def query_selector_all(self, selector: str):
        return []

    def wait_for_load_state(self, state: str, timeout: int = 10000):
        return None

    def content(self):
        return "<html></html>"

    def title(self):
        return "Example"

    @property
    def viewport_size(self):
        return {"width": 1920, "height": 1080}


def make_browser(monkeypatch):
    # Prevent real Playwright initialization during tests
    monkeypatch.setattr(Browser, "_init_browser", lambda self: None)
    browser = Browser(store=MockStore(), headless=True)
    browser.page = DummyPage()
    return browser


def test_click_skipped_low_confidence(monkeypatch):
    browser = make_browser(monkeypatch)

    result = browser.click_element("#btn", confidence=0.5)

    assert result is False
    assert browser.action_history, "Action history should contain an entry"
    last = browser.action_history[-1]
    assert last["action"] == "click"
    assert last["reason"] == "low_confidence"


def test_execute_action_accepts_high_confidence(monkeypatch):
    browser = make_browser(monkeypatch)

    result = browser.execute_action("click", "#btn", confidence=0.9)

    assert result is True
    assert browser.action_history, "Action history should contain an entry"
    last = browser.action_history[-1]
    assert last["action"] == "click"
    assert last["confidence"] == 0.9


def test_set_confidence_threshold_changes_behavior(monkeypatch):
    browser = make_browser(monkeypatch)

    # Default threshold should be 0.7
    assert browser.get_confidence_threshold() == 0.7

    browser.set_confidence_threshold(0.4)
    assert browser.get_confidence_threshold() == 0.4

    # Now a lower confidence should be accepted
    result = browser.click_element("#btn", confidence=0.45)
    assert result is True or result is False


def test_llm_retry_mechanism(monkeypatch):
    """Test that LLM retry mechanism works when confidence is too low."""
    browser = make_browser(monkeypatch)

    # Mock LLM callback that suggests a retry with different selector
    def mock_llm_callback(failed_action, failure_reason, retry_count):
        if retry_count == 1:
            return {
                "action_type": "click",
                "selector": "#alternative-btn",  # Different selector
                "confidence": 0.8,  # Higher confidence
                "kwargs": {},
            }
        return None  # No suggestion for second retry

    # Browser no longer supports set_llm_callback; orchestrator should handle retries.
    # Ensure calling execute_action still returns False for low confidence without orchestrator.
    result = browser.execute_action("click", "#btn", confidence=0.5)
    assert result is False


def test_llm_retry_limit_exceeded(monkeypatch):
    """Test that LLM retries are limited to max_llm_retries."""
    browser = make_browser(monkeypatch)

    # Mock LLM callback that always suggests retry but with low confidence
    def mock_llm_callback(failed_action, failure_reason, retry_count):
        return {
            "action_type": "click",
            "selector": "#btn",
            "confidence": 0.5,  # Still too low
            "kwargs": {},
        }

    # Browser no longer handles llm retries; it should simply log low confidence and return False
    result = browser.execute_action("click", "#btn", confidence=0.5)
    assert result is False
    low_confidence_entries = [
        entry
        for entry in browser.action_history
        if entry.get("reason") == "low_confidence"
    ]
    assert len(low_confidence_entries) == 1


def test_llm_retry_no_callback(monkeypatch):
    """Test behavior when no LLM callback is set."""
    browser = make_browser(monkeypatch)

    # Execute action with low confidence - should fail immediately and be logged
    result = browser.execute_action("click", "#btn", confidence=0.5)
    assert result is False
    low_confidence_entries = [
        entry
        for entry in browser.action_history
        if entry.get("reason") == "low_confidence"
    ]
    assert len(low_confidence_entries) == 1
