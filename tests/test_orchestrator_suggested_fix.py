from ai_infant.agents.llm_orchestrator import perform_action_with_llm


class DummyBrowserTool:
    def __init__(self):
        self.action_history = []

    def execute_action_struct(self, action_type, selector, confidence, **kwargs):
        # If selector indicates suggested_fix applied, return success
        if selector == "#suggested":
            self.action_history.append(
                {
                    "action": action_type,
                    "selector": selector,
                    "confidence": confidence,
                    "ok": True,
                }
            )
            return {"ok": True, "result": {}}

        # Otherwise initial attempt fails with low_confidence
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


class DummyJudgeManager:
    def assess(self, result):
        # Return a draft assessment suggesting a fix to '#suggested'
        return {
            "verdict": "reject",
            "score": 0.2,
            "details": [
                {
                    "judge": "LLMJudge",
                    "out": {
                        "verdict": "revise",
                        "score": 0.3,
                        "suggested_fix": {
                            "action_type": "click",
                            "selector": "#suggested",
                            "confidence": 0.85,
                            "kwargs": {},
                        },
                    },
                }
            ],
        }


def test_orchestrator_applies_suggested_fix():
    browser = DummyBrowserTool()
    judge = DummyJudgeManager()

    res = perform_action_with_llm(
        browser,
        "click",
        "#initial",
        0.5,
        llm_client=None,
        max_retries=2,
        judge_manager=judge,
    )

    assert res.get("ok") is True
    # Ensure judge assessment and applied suggested fix are recorded in history
    # There should be at least two entries: initial failure and applied suggested fix
    selectors = [
        entry.get("selector") for entry in browser.action_history if "selector" in entry
    ]
    assert "#initial" in selectors
    assert "#suggested" in selectors
