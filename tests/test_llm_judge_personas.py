from ai_infant.judge.interface import LLMJudge


def test_llm_judge_persona_parsing():
    def fake_client(prompt: str) -> str:
        # Return a well-formed JSON matching the extended schema
        return '{"verdict":"revise","score":0.4,"reasons":["missing_field"],"critique":"Please add the required field.","suggested_fix":{"action_type":"fill","selector":"#name","confidence":0.85,"kwargs":{"value":"John"}},"tone":"tough"}'

    judge = LLMJudge(client=fake_client, persona="tough")
    out = judge.evaluate({"ok": False})
    assert out["verdict"] == "revise"
    assert out.get("suggested_fix") is not None
