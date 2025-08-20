"""Pluggable LLM aggregator / router.

This module provides an AggregatorManager that selects among configured
model aggregators (LLMz, OpenRouter, etc.) according to environment
preferences. It intentionally does NOT provide a local fallback when
enforce_no_fallback=True.
"""

import json
import os
from typing import List, Optional

import requests


class LLMAdapter:
    """Base adapter interface."""

    def available(self) -> bool:
        raise NotImplementedError()

    def generate(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError()


class LLMzAdapter(LLMAdapter):
    def __init__(self, api_key: Optional[str] = None, api_url: Optional[str] = None):
        self.api_key = api_key or os.getenv("LLMZ_API_KEY")
        self.api_url = api_url or os.getenv("LLMZ_API_URL")

    def available(self) -> bool:
        return bool(self.api_key and self.api_url)

    def generate(self, prompt: str, **kwargs) -> str:
        if not self.available():
            raise RuntimeError("LLMz adapter not configured")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {"prompt": prompt}
        # Allow caller to add options
        data.update(kwargs.get("options", {}))

        resp = requests.post(self.api_url, headers=headers, json=data, timeout=30)
        resp.raise_for_status()

        # Try JSON response
        try:
            j = resp.json()
            # common fields: text, result, output
            for key in ("text", "result", "output", "response"):
                if key in j:
                    return j[key]
            # fallback: return full json
            return json.dumps(j)
        except Exception:
            return resp.text


class OpenRouterAdapter(LLMAdapter):
    def __init__(self, api_key: Optional[str] = None, api_url: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.api_url = api_url or os.getenv("OPENROUTER_API_URL")

    def available(self) -> bool:
        return bool(self.api_key and self.api_url)

    def generate(self, prompt: str, **kwargs) -> str:
        if not self.available():
            raise RuntimeError("OpenRouter adapter not configured")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {"input": prompt}
        data.update(kwargs.get("options", {}))

        resp = requests.post(self.api_url, headers=headers, json=data, timeout=30)
        resp.raise_for_status()

        try:
            j = resp.json()
            for key in ("text", "result", "output", "response"):
                if key in j:
                    return j[key]
            return json.dumps(j)
        except Exception:
            return resp.text


class AggregatorManager:
    """Aggregator that routes requests to configured adapters.

    If enforce_no_fallback is True, initialization will raise if no adapters
    are configured.
    """

    def __init__(
        self,
        preference: Optional[str] = None,
        enforce_no_fallback: Optional[bool] = None,
    ):
        """Initialize aggregator manager.

        If `enforce_no_fallback` is not provided, derive default from `ENVIRONMENT` env var:
        - production -> enforce_no_fallback = True
        - development -> enforce_no_fallback = False
        """
        self.preference = preference or os.getenv("AGGREGATOR_PREFERENCE", "")
        if enforce_no_fallback is None:
            env = os.getenv("ENVIRONMENT", "production").lower()
            enforce_no_fallback = True if env == "production" else False
        self.adapters: List[LLMAdapter] = []
        # Build adapters according to preference order
        prefs = [p.strip().lower() for p in self.preference.split(",") if p.strip()]

        # instantiate known adapters
        llmz = LLMzAdapter()
        openrouter = OpenRouterAdapter()

        available_map = {
            "llmz": llmz,
            "openrouter": openrouter,
        }

        # If preference provided, honor the order; else include all available in default order
        if prefs:
            for name in prefs:
                adapter = available_map.get(name)
                if adapter and adapter.available():
                    self.adapters.append(adapter)
        else:
            # default order
            for adapter in (llmz, openrouter):
                if adapter.available():
                    self.adapters.append(adapter)

        if enforce_no_fallback and not self.adapters:
            raise RuntimeError(
                "No LLM aggregator adapters configured and fallback is disabled. Set AGGREGATOR_PREFERENCE and provider API keys."
            )

    def is_available(self) -> bool:
        return len(self.adapters) > 0

    def generate(
        self, prompt: str, hint: Optional[str] = None, timeout: int = 30
    ) -> str:
        """Generate text using the first available adapter in preference order.

        Raises RuntimeError if no adapters are available or all adapters fail.
        """
        errors = []
        for adapter in self.adapters:
            try:
                # pass hint as part of options if needed
                options = {"options": {"hint": hint}} if hint else {}
                return adapter.generate(prompt, **options)
            except Exception as e:
                errors.append(str(e))
                continue

        raise RuntimeError(f"All aggregator adapters failed: {errors}")
