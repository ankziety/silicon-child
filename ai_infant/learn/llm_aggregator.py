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


class FallbackAdapter(LLMAdapter):
    """Local lightweight fallback adapter providing classification and sentiment/tone analysis.

    This adapter attempts to use transformers' pipelines if available, otherwise falls
    back to very small heuristic analyzers. It is intended as a deterministic,
    local fallback for basic classification/sentiment tasks when remote aggregators
    are not configured or allowed.
    """
    def __init__(self):
        # Try to initialize local models lazily
        self._init_done = False
        self._sentiment = None
        self._classifier = None

    def available(self) -> bool:
        # Always available as a local heuristic (no external keys required)
        return True

    def _lazy_init(self):
        if self._init_done:
            return
        self._init_done = True
        try:
            from transformers import pipeline
            # sentiment and classification pipelines
            try:
                self._sentiment = pipeline("sentiment-analysis")
            except Exception:
                self._sentiment = None

            try:
                self._classifier = pipeline("text-classification")
            except Exception:
                self._classifier = None

        except Exception:
            # transformers not available; fall back to minimal heuristics
            self._sentiment = None
            self._classifier = None

    def generate(self, prompt: str, **kwargs) -> str:
        """Return a small JSON string with classification and sentiment analysis.

        The return is a JSON string like:
        {"classification": [{"label":..., "score":...}], "sentiment": {"label":..., "score":...}, "text": "..."}
        """
        self._lazy_init()
        out = {"text": prompt}

        # Classification
        try:
            if self._classifier:
                cls = self._classifier(prompt, top_k=3)
                out["classification"] = cls
            else:
                # very small heuristic: look for keywords
                labels = []
                low = prompt.lower()
                if any(w in low for w in ["error", "fail", "exception"]):
                    labels.append({"label": "bug_report", "score": 0.9})
                if any(w in low for w in ["how to", "tutorial", "guide", "example"]):
                    labels.append({"label": "howto", "score": 0.8})
                if not labels:
                    labels.append({"label": "other", "score": 0.6})
                out["classification"] = labels
        except Exception:
            out["classification"] = [{"label": "error", "score": 0.0}]

        # Sentiment / tone
        try:
            if self._sentiment:
                s = self._sentiment(prompt, top_k=1)
                out["sentiment"] = s[0]
            else:
                # very small heuristic sentiment
                low = prompt.lower()
                pos_words = ["good", "great", "success", "positive", "win", "excellent"]
                neg_words = ["bad", "error", "fail", "problem", "issue", "incorrect"]
                score = 0.5
                for w in pos_words:
                    if w in low:
                        score += 0.1
                for w in neg_words:
                    if w in low:
                        score -= 0.1
                label = "neutral"
                if score > 0.55:
                    label = "positive"
                elif score < 0.45:
                    label = "negative"
                out["sentiment"] = {"label": label, "score": round(score, 2)}
        except Exception:
            out["sentiment"] = {"label": "unknown", "score": 0.5}

        return json.dumps(out)


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

        # Always append fallback adapter if none configured and environment allows
        # (AggregatorManager enforces no-fallback based on enforce_no_fallback flag)
        fallback = FallbackAdapter()
        if not self.adapters:
            # Only add fallback as a last resort when not enforcing no-fallback
            if not enforce_no_fallback:
                self.adapters.append(fallback)
        else:
            # Keep fallback available at end of list if present; but do not prefer it
            self.fallback = fallback

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
