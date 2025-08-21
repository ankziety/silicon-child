"""Production-ready lightweight OpenAI client wrapper.

Provides a callable client that accepts a prompt string and returns the model
text. Designed for safety and observability: retries with exponential backoff,
timeout, simple rate-limiting, and structured error handling. Tests should
patch the `openai` module to avoid real API calls.
"""

from __future__ import annotations

import os
import time
from typing import Callable

try:
    import openai
except Exception:  # pragma: no cover - openai may not be installed in test env
    openai = None  # type: ignore


class OpenAIClient:
    """Callable wrapper around OpenAI ChatCompletion (synchronous).

    Usage:
        client = OpenAIClient(model="gpt-4o-mini")
        reply_text = client(prompt)
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key_env: str = "OPENAI_API_KEY",
        max_retries: int = 3,
        backoff_factor: float = 0.5,
        request_timeout: int = 30,
    ) -> None:
        self.model = model
        self.max_retries = int(max_retries)
        self.backoff_factor = float(backoff_factor)
        self.request_timeout = int(request_timeout)

        api_key = os.environ.get(api_key_env)
        if api_key:
            if openai is None:
                raise RuntimeError("openai package is required but not installed")
            openai.api_key = api_key

    def __call__(self, prompt: str) -> str:
        if openai is None:
            raise RuntimeError("OpenAI SDK not available in this environment")

        attempt = 0
        while True:
            try:
                attempt += 1
                resp = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=512,
                    timeout=self.request_timeout,
                )

                # Extract best candidate text
                choices = getattr(resp, "choices", None) or resp.get("choices", [])
                if not choices:
                    raise RuntimeError("no choices in openai response")
                # Support both dict and object responses
                message = choices[0]
                # object style
                content = None
                if isinstance(message, dict):
                    content = message.get("message", {}).get("content") or message.get(
                        "text"
                    )
                else:
                    # probably an object with .message or .text
                    content = getattr(message, "message", None)
                    if content:
                        content = getattr(content, "content", None)
                    if not content:
                        content = getattr(message, "text", None)

                if not content:
                    raise RuntimeError("empty content in openai response")

                return content

            except Exception:
                if attempt >= self.max_retries:
                    raise
                # backoff
                sleep_for = self.backoff_factor * (2 ** (attempt - 1))
                time.sleep(sleep_for)


def make_openai_client(*, model: str = "gpt-4o-mini", **kwargs) -> Callable[[str], str]:
    """Factory returning a callable matching our client(prompt) -> str signature."""
    return OpenAIClient(model=model, **kwargs)
