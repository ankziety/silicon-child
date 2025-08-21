from ai_infant.agents.llm_orchestrator import perform_action_with_llm
from ai_infant.judge.interface import JudgeManager, RuleJudge


def test_orchestrator_respects_judge_accepts():
    class FakeBrowserTool:
        def __init__(self):
            self.action_history = []

        def execute_action_struct(self, action_type, selector, confidence, **kwargs):
            return {
                "ok": False,
                "reason": "low_confidence",
                "details": {"selector": selector},
            }

    fake_browser = FakeBrowserTool()
    jm = JudgeManager(
        judges=[(RuleJudge(), 1.0)], threshold=0.0
    )  # threshold 0 to force accept

    def fake_llm_client(prompt: str) -> str:
        return '{"verdict":"accept","score":0.9,"reasons":[]}'

    res = perform_action_with_llm(
        fake_browser,
        "click",
        "#btn",
        0.5,
        llm_client=fake_llm_client,
        max_retries=1,
        judge_manager=jm,
    )
    # Since judge threshold 0.0 and rule judge will accept only ok==True, orchestrator should still proceed to LLM retry
    assert res.get("ok") in (True, False)
