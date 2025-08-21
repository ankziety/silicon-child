from ai_infant.agents.llm_orchestrator import perform_action_with_llm


class DummyBrowserTool:
    def __init__(self):
        self.action_history = []

    def execute_action_struct(self, action_type, selector, confidence, **kwargs):
        # only '#alt' succeeds
        if selector == "#alt":
            self.action_history.append(
                {
                    "action": action_type,
                    "selector": selector,
                    "confidence": confidence,
                    "ok": True,
                }
            )
            return {"ok": True, "result": {}}
        # initial fails
        self.action_history.append(
            {
                "action": action_type,
                "selector": selector,
                "confidence": confidence,
                "ok": False,
                "reason": "low_confidence",
            }
        )
        return {"ok": False, "reason": "low_confidence", "details": {}}


def test_orchestrator_uses_llm_client_for_retry():
    browser = DummyBrowserTool()

    def fake_client(prompt: str) -> str:
        return '{"action_type": "click", "selector": "#alt", "confidence": 0.8, "kwargs": {}}'

    res = perform_action_with_llm(
        browser,
        "click",
        "#initial",
        0.5,
        llm_client=fake_client,
        max_retries=2,
        judge_manager=None,
    )
    assert res.get("ok") is True
    selectors = [e.get("selector") for e in browser.action_history if "selector" in e]
    assert "#initial" in selectors
    assert "#alt" in selectors


def test_orchestrator_uses_llm_to_retry(monkeypatch):
    # Create a fake browser_tool with an execute_action_struct method
    class FakeBrowserTool:
        def __init__(self):
            self.action_history = []

        def execute_action_struct(self, action_type, selector, confidence, **kwargs):
            # Simulate low confidence rejection for the initial selector
            if selector == "#btn":
                entry = {
                    "ok": False,
                    "reason": "low_confidence",
                    "details": {"selector": selector},
                }
                self.action_history.append(
                    {
                        "action": action_type,
                        "selector": selector,
                        "reason": "low_confidence",
                    }
                )
                return entry
            # For alternative selector, succeed
            if selector == "#alt":
                self.action_history.append(
                    {"action": action_type, "selector": selector}
                )
                return {"ok": True, "result": {}}
            return {"ok": False, "reason": "unknown"}

    fake_browser = FakeBrowserTool()

    # Fake LLM client that suggests #alt on first retry
    def fake_llm_client(prompt: str) -> str:
        return '{"action_type": "click", "selector": "#alt", "confidence": 0.85, "kwargs": {}}'

    res = perform_action_with_llm(
        fake_browser, "click", "#btn", 0.5, llm_client=fake_llm_client, max_retries=2
    )
    assert res.get("ok") is True
    assert any(entry.get("selector") == "#alt" for entry in fake_browser.action_history)
