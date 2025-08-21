from __future__ import annotations

import os
import time
from typing import Callable


def make_llm_client(provider: str = "openai", **opts) -> Callable[[str], str]:
    """Factory that returns a simple sync LLM client callable: client(prompt) -> str.

    Supported providers:
      - "openai": uses `openai.ChatCompletion.create` (lazy import). Pass
        `openai_module` in opts to inject a test double. Options: model, max_retries,
        backoff_base.
      - "mock": pass `client_fn` in opts and it will be returned directly.
    """

    if provider == "anthropic":
        anthropic_module = opts.get("anthropic_module")
        model = opts.get("model", "claude-3-5-sonnet-20240620")
        max_retries = int(opts.get("max_retries", 3))
        backoff_base = float(opts.get("backoff_base", 0.5))
        api_key = opts.get("api_key") or os.environ.get("ANTHROPIC_API_KEY")

        def _client(prompt: str) -> str:
            nonlocal anthropic_module
            if anthropic_module is None:
                try:
                    import anthropic as _anthropic

                    anthropic_module = _anthropic
                except Exception as e:
                    raise RuntimeError("Anthropic SDK not available") from e

            if api_key:
                try:
                    anthropic_module.api_key = api_key
                except Exception:
                    # some test doubles may not support assignment
                    pass

            last_exc: BaseException | None = None
            for attempt in range(1, max_retries + 1):
                try:
                    resp = anthropic_module.messages.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.2,
                    )
                    # Newer SDKs return .choices[0].message.content
                    # Fallback to choices[0].text if necessary
                    choice = resp.choices[0]
                    text = getattr(choice, "message", None)
                    if text is not None:
                        return text.content
                    return getattr(choice, "text", "")
                except Exception as e:
                    last_exc = e
                    if attempt == max_retries:
                        break
                    # backoff
                    sleep_t = backoff_base * (2 ** (attempt - 1))
                    time.sleep(sleep_t)

            raise RuntimeError("LLM provider failure") from last_exc

        return _client

    if provider == "openrouter":
        openrouter_module = opts.get("openrouter_module")
        model = opts.get("model", "gpt-5-mini")
        max_retries = int(opts.get("max_retries", 3))
        backoff_base = float(opts.get("backoff_base", 0.5))
        api_key = opts.get("api_key") or os.environ.get("OPENROUTER_API_KEY")

        def _client(prompt: str) -> str:
            nonlocal openrouter_module
            if openrouter_module is None:
                try:
                    import openrouter as _openrouter

                    openrouter_module = _openrouter
                except Exception as e:
                    raise RuntimeError("OpenRouter SDK not available") from e

            if api_key:
                try:
                    openrouter_module.api_key = api_key
                except Exception:
                    # some test doubles may not support assignment
                    pass

            last_exc: BaseException | None = None
            for attempt in range(1, max_retries + 1):
                try:
                    resp = openrouter_module.ChatCompletion.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.2,
                    )
                    # Newer SDKs return .choices[0].message.content
                    # Fallback to choices[0].text if necessary
                    choice = resp.choices[0]
                    text = getattr(choice, "message", None)

                    if text is not None:
                        return text.content
                    return getattr(choice, "text", "")
                except Exception as e:
                    last_exc = e
                    if attempt == max_retries:
                        break
                    # backoff
                    sleep_t = backoff_base * (2 ** (attempt - 1))
                    time.sleep(sleep_t)

            raise RuntimeError("LLM provider failure") from last_exc

        return _client
    if provider == "mock":
        client_fn = opts.get("client_fn")
        if not callable(client_fn):
            raise ValueError("mock provider requires callable 'client_fn' in opts")
        return client_fn

    if provider == "openai":
        openai_module = opts.get("openai_module")
        model = opts.get("model", "gpt-4o-mini")
        max_retries = int(opts.get("max_retries", 3))
        backoff_base = float(opts.get("backoff_base", 0.5))
        # Allow overriding api_key from opts for tests
        api_key = opts.get("api_key") or os.environ.get("OPENAI_API_KEY")

        def _client(prompt: str) -> str:
            nonlocal openai_module
            # Lazy import
            if openai_module is None:
                try:
                    import openai as _openai

                    openai_module = _openai
                except Exception as e:
                    raise RuntimeError("OpenAI SDK not available") from e

            if api_key:
                try:
                    openai_module.api_key = api_key
                except Exception:
                    # some test doubles may not support assignment
                    pass

            last_exc: BaseException | None = None
            for attempt in range(1, max_retries + 1):
                try:
                    resp = openai_module.ChatCompletion.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.2,
                    )
                    # Newer SDKs return .choices[0].message.content
                    # Fallback to choices[0].text if necessary
                    choice = resp.choices[0]
                    text = getattr(choice, "message", None)
                    if text is not None:
                        return text.content
                    return getattr(choice, "text", "")
                except Exception as e:
                    last_exc = e
                    if attempt == max_retries:
                        break
                    # backoff
                    sleep_t = backoff_base * (2 ** (attempt - 1))
                    time.sleep(sleep_t)

            raise RuntimeError("LLM provider failure") from last_exc

        return _client

    raise ValueError(f"Unsupported provider: {provider}")
