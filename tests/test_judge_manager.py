from ai_infant.judge.interface import JudgeManager, RuleJudge


def test_rule_judge_accepts_ok_result():
    j = RuleJudge()
    res = j.evaluate({"ok": True, "details": {}})
    assert res["verdict"] == "accept"


def test_rule_judge_detects_missing_fields():
    j = RuleJudge(required_fields=["id"])
    res = j.evaluate({"ok": True, "details": {}})
    assert res["verdict"] == "revise"


def test_judge_manager_aggregates():
    j1 = RuleJudge()

    class FakeJudge:
        def evaluate(self, result):
            return {"verdict": "accept", "score": 0.8, "reasons": []}

    jm = JudgeManager(judges=[(j1, 1.0), (FakeJudge(), 1.0)], threshold=0.5)
    record = jm.assess({"ok": True})
    assert record["verdict"] == "accept"
