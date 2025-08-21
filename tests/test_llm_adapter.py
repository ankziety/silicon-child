import json

from ai_infant.crawl.llm_adapter import create_openai_callback


def test_adapter_with_fake_client_returns_valid_action():
    # Fake client returns a JSON suggestion
    def fake_client(prompt: str) -> str:
        return json.dumps(
            {
                "action_type": "click",
                "selector": "#retry-btn",
                "confidence": 0.85,
                "kwargs": {},
            }
        )

    cb = create_openai_callback(client=fake_client)

    suggestion = cb(
        failed_action={
            "action_type": "click",
            "selector": "#btn",
            "confidence": 0.3,
            "kwargs": {},
        },
        failure_reason="low_confidence",
        retry_count=1,
    )

    assert suggestion is not None
    assert suggestion["action_type"] == "click"
    assert suggestion["selector"] == "#retry-btn"
    assert 0.0 <= suggestion["confidence"] <= 1.0


def test_adapter_rejects_malformed_json():
    def fake_client(prompt: str) -> str:
        return "I think you should click something but here's no json"

    cb = create_openai_callback(client=fake_client)
    suggestion = cb(
        failed_action={
            "action_type": "click",
            "selector": "#btn",
            "confidence": 0.3,
            "kwargs": {},
        },
        failure_reason="low_confidence",
        retry_count=1,
    )
    assert suggestion is None
