"""Lightweight LLM adapter for producing retry suggestions for low-confidence actions.

This module provides helpers to create a callable that matches the signature
expected by `Browser.set_llm_callback(...)`:

    retry_action = llm_callback(failed_action=..., failure_reason=..., retry_count=...)

The adapter is written to be testable: you may pass a `client` callable that
accepts a prompt string and returns a textual reply (synchronously). A
production-ready client can be an OpenAI wrapper; tests should inject a fake
client.
"""

from __future__ import annotations

import json
import re
from typing import Callable

SUPPORTED_ACTIONS = {"click", "fill", "select", "navigate", "wait", "scroll", "hover"}


def _extract_json_from_text(text: str) -> dict | None:
    """Try to extract the first JSON object from a free-form text reply.

    Returns the parsed dict or None if no valid JSON object is found.
    """
    # Try direct parse first
    try:
        return json.loads(text)
    except Exception:
        pass

    # Find the first {...} balanced braces chunk using a simple regexp and attempt to parse it.
    # This is intentionally conservative and only returns the first object-looking substring.
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        candidate = m.group(0)
        try:
            return json.loads(candidate)
        except Exception:
            pass

    # Fallback: look for any {...} pair (non-greedy)
    m2 = re.search(r"\{[\s\S]*?\}", text)
    if m2:
        try:
            return json.loads(m2.group(0))
        except Exception:
            return None

    return None


def _validate_retry_action(obj: dict) -> bool:
    """Validate the retry-action object returned by the LLM.

    Required fields: action_type (supported), selector (non-empty string), confidence (0..1)
    Optional: kwargs (dict)
    """
    if not isinstance(obj, dict):
        return False
    action_type = obj.get("action_type")
    selector = obj.get("selector")
    confidence = obj.get("confidence")

    if not isinstance(action_type, str) or action_type not in SUPPORTED_ACTIONS:
        return False
    if not isinstance(selector, str) or not selector.strip():
        return False
    try:
        confidence = float(confidence)
    except Exception:
        return False
    if not (0.0 <= confidence <= 1.0):
        return False
    kwargs = obj.get("kwargs")
    if kwargs is not None and not isinstance(kwargs, dict):
        return False
    return True


def create_openai_callback(
    *, client: Callable[[str], str] | None = None, model: str = "gpt-4o-mini"
) -> Callable:
    """Create an LLM-callback function suitable for `Browser.set_llm_callback`.

    Args:
        client: Optional callable that accepts a prompt string and returns the LLM's textual reply.
                If not provided, the adapter will try to import the OpenAI SDK at runtime and call
                `openai.ChatCompletion.create` (this call may raise if OpenAI SDK isn't installed or
                API key isn't configured). Passing `client` is recommended for tests and for more
                controlled integration.
        model: model name for the LLM (used only in prompt metadata if client is the OpenAI SDK).

    Returns:
        Callable matching signature: llm_callback(failed_action=..., failure_reason=..., retry_count=...)
    """

    def _callback(
        *, failed_action: dict, failure_reason: str, retry_count: int
    ) -> dict | None:
        # Build a concise prompt describing the failure and constraints
        prompt = (
            "The automated browser attempted an action and it was rejected due to low confidence.\n"
            f"Failed action: {json.dumps(failed_action)}\n"
            f"Failure reason: {failure_reason}\n"
            f"Retry attempt: {retry_count}\n"
            "You may suggest at most one alternative non-destructive action that is likely to succeed.\n"
            "Return only a JSON object with the following fields: action_type (one of: "
            + ", ".join(sorted(SUPPORTED_ACTIONS))
            + "), selector (string), confidence (0.0-1.0), and optional kwargs (object).\n"
        )

        # Call the provided client or fall back to OpenAI SDK if available
        try:
            if client is not None:
                reply_text = client(prompt)
            else:
                # Lazy import to avoid hard dependency for tests
                import openai

                api_resp = openai.ChatCompletion.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=200,
                )
                # Extract text safely
                reply_text = api_resp.choices[0].message.content

        except Exception as e:
            print(f"LLM client error: {e}")
            return None

        # Parse JSON from reply
        parsed = _extract_json_from_text(reply_text)
        if not parsed:
            print("LLM reply did not contain valid JSON suggestion")
            return None

        if not _validate_retry_action(parsed):
            print(f"LLM suggestion failed validation: {parsed}")
            return None

        # Normalize kwargs
        if "kwargs" not in parsed:
            parsed["kwargs"] = {}

        return parsed

    return _callback
