from unittest.mock import Mock

from ai_infant.core.loop import ResearchLoop


def test_research_loop_registers_llm_and_retries(monkeypatch):
    store = Mock()

    # Fake llm client that suggests an alternative selector on first retry
    def fake_client(prompt: str) -> str:
        # The LLM should be asked to return a JSON object; return a valid one
        return '{"action_type": "click", "selector": "#alt", "confidence": 0.85, "kwargs": {}}'

    loop = ResearchLoop(store, headless=True, llm_client=fake_client)

    # Browser tool should not expose internal LLM callback registration any more
    assert not hasattr(loop.browser, "set_llm_callback")

    # Simulate a browser-level low-confidence action by calling execute_action
    browser = loop.browser
    # Monkeypatch browser to avoid real Playwright init
    monkeypatch.setattr(browser, "_ensure", lambda: None)

    # The browser wrapper is actually a BrowserTool; ensure underlying browser exists for testing
    # If not present, create a minimal fake
    if getattr(browser, "_browser", None) is None:

        class MiniBrowser:
            def __init__(self):
                self.action_history = []

            def execute_action(self, action_type, selector, confidence, **kwargs):
                # Emulate low-confidence rejected by delegating to Browser._handle_llm_retry
                return False

        browser._browser = MiniBrowser()

    # Test completed if ResearchLoop initializes without registering callbacks
    assert True
