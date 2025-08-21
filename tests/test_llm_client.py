import types

import pytest

from ai_infant.llm.client import make_llm_client


class DummyChoice:
    def __init__(self, text=None, message=None):
        self.text = text
        self.message = message


class DummyMessage:
    def __init__(self, content: str):
        self.content = content


class DummyOpenAI:
    def __init__(self, responses):
        # responses: list of reply strings or exceptions
        self._responses = list(responses)
        self.api_key = None

    class ChatCompletion:
        pass

    def ChatCompletion_create(self, model, messages, temperature=0.2):
        # compatibility shim if tests call this directly
        return self.ChatCompletion.create(
            model=model, messages=messages, temperature=temperature
        )


def test_mock_client_passthrough():
    def my_client(prompt: str) -> str:
        return "ok: " + prompt

    c = make_llm_client("mock", client_fn=my_client)
    out = c("hello")
    assert out == "ok: hello"


def test_openai_client_retries_and_response(monkeypatch):
    # Build fake openai module with ChatCompletion.create behavior
    calls = {"n": 0}

    class FakeChoice:
        def __init__(self, content):
            self.message = DummyMessage(content)

    class FakeChatCompletion:
        @staticmethod
        def create(model, messages, temperature=0.2):
            calls["n"] += 1
            if calls["n"] < 2:
                raise RuntimeError("transient")
            return types.SimpleNamespace(choices=[FakeChoice("final reply")])

    fake_openai = types.SimpleNamespace(ChatCompletion=FakeChatCompletion, api_key=None)

    client = make_llm_client(
        "openai", openai_module=fake_openai, max_retries=3, backoff_base=0.0
    )

    resp = client("prompt")
    assert resp == "final reply"
    assert calls["n"] == 2


def test_openai_client_failure(monkeypatch):
    class FailingChatCompletion:
        @staticmethod
        def create(model, messages, temperature=0.2):
            raise RuntimeError("permanent")

    fake_openai = types.SimpleNamespace(
        ChatCompletion=FailingChatCompletion, api_key=None
    )
    client = make_llm_client(
        "openai", openai_module=fake_openai, max_retries=2, backoff_base=0.0
    )

    with pytest.raises(RuntimeError):
        client("x")
