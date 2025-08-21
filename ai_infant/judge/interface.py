from __future__ import annotations

import json
from typing import Callable


class Judge:
    """Abstract judge interface.

    Each judge has a human-readable `name` used for logging and audit. Implementations
    must provide `evaluate(result)` returning a dict with keys:
      - verdict: 'accept' | 'reject' | 'revise'
      - score: float (0.0-1.0)
      - reasons: list[str]
    Optional keys include `critique` (str), `suggested_fix` (dict), and `tone` (str).
    """

    def __init__(self, name: str = "Judge") -> None:
        self.name = name

    def evaluate(self, result: dict) -> dict:
        raise NotImplementedError


class RuleJudge(Judge):
    """Simple rule-based judge.

    Current policy: if result.get('ok') is True -> accept score 1.0;
    otherwise -> reject score 0.0. Optionally provide extra rules via
    `required_fields` which must be present in result['details'].
    """

    def __init__(
        self, required_fields: list[str] | None = None, name: str = "Judge Rule"
    ) -> None:
        super().__init__(name=name)
        self.required_fields = required_fields or []

    def evaluate(self, result: dict) -> dict:
        reasons: list[str] = []
        ok = bool(result.get("ok"))
        details = result.get("details", {}) or {}

        if ok:
            # verify required fields
            missing = [f for f in self.required_fields if f not in details]
            if missing:
                reasons.append(f"missing_fields: {missing}")
                return {
                    "verdict": "revise",
                    "score": 0.5,
                    "reasons": reasons,
                    "critique": "Missing required fields.",
                    "tone": "neutral",
                    "suggested_fix": None,
                }
            return {
                "verdict": "accept",
                "score": 1.0,
                "reasons": reasons,
                "critique": "Looks good.",
                "tone": "neutral",
                "suggested_fix": None,
            }

        reasons.append("result_not_ok")
        return {
            "verdict": "reject",
            "score": 0.0,
            "reasons": reasons,
            "critique": "Result indicates failure.",
            "tone": "neutral",
            "suggested_fix": None,
        }


class LLMJudge(Judge):
    """Judge that asks an LLM for a verdict. The provided `client` must be a
    callable taking a prompt string and returning the model's reply text.

    The LLM should return a JSON object like:
      {"verdict": "accept"|"reject"|"revise", "score": 0.0-1.0, "reasons": [..],
       "critique": "...", "suggested_fix": {...} }
    """

    PERSONA_PRESETS = {
        "tough": {
            "name": "Judge Cowl",
            "instruction": "Be direct and critical, focus on correctness and safety.",
        },
        "gentle": {
            "name": "Judge Pelosi",
            "instruction": "Be kind and constructive; highlight positives then suggest improvements.",
        },
        "irb": {
            "name": "Judge Presby",
            "instruction": "Prioritize safety, ethics, and non-destructive behavior.",
        },
    }

    def __init__(
        self,
        client: Callable[[str], str],
        model_name: str = "gpt-4o-mini",
        persona: str | None = None,
        name: str | None = None,
    ) -> None:
        persona = persona or "irb"
        preset = self.PERSONA_PRESETS.get(persona, {})
        given_name = name or preset.get("name", "Judge LLM")
        super().__init__(name=given_name)
        self.client = client
        self.model_name = model_name
        self.persona = persona
        self.persona_instruction = preset.get(
            "instruction", "Be professional and constructive."
        )

    def evaluate(self, result: dict) -> dict:
        prompt = (
            f"You are {self.name}. {self.persona_instruction}\n"
            "You will inspect the tool result and return only a JSON object with the following keys:\n"
            "verdict (one of: 'accept','reject','revise'), score (0.0-1.0), reasons (array of short strings),\n"
            "critique (1-3 sentences), suggested_fix (an object or null), and tone (short label).\n"
            f"Result: {json.dumps(result)}\n"
            "If suggesting a fix, suggested_fix must be a dict with action_type, selector, confidence, and optional kwargs. Confidence must be numeric between 0 and 1."
        )
        try:
            reply = self.client(prompt)
            # Try parsing JSON
            parsed = json.loads(reply)
            verdict = parsed.get("verdict")
            score = float(parsed.get("score", 0.0))
            reasons = parsed.get("reasons", []) or []
            critique = parsed.get("critique")
            suggested_fix = parsed.get("suggested_fix")
            tone = parsed.get("tone") or self.persona
            if verdict not in ("accept", "reject", "revise"):
                return {
                    "verdict": "reject",
                    "score": 0.0,
                    "reasons": ["invalid_verdict"],
                    "critique": None,
                    "suggested_fix": None,
                    "tone": tone,
                }

            out = {
                "verdict": verdict,
                "score": max(0.0, min(1.0, score)),
                "reasons": reasons,
                "critique": critique,
                "tone": tone,
            }
            # Validate suggested_fix shape conservatively
            if isinstance(suggested_fix, dict):
                at = suggested_fix.get("action_type")
                sel = suggested_fix.get("selector")
                conf = suggested_fix.get("confidence")
                kwargs = suggested_fix.get("kwargs", {})
                try:
                    conf = float(conf)
                except Exception:
                    conf = None
                if (
                    isinstance(at, str)
                    and isinstance(sel, str)
                    and conf is not None
                    and 0.0 <= conf <= 1.0
                    and isinstance(kwargs, dict)
                ):
                    out["suggested_fix"] = {
                        "action_type": at,
                        "selector": sel,
                        "confidence": conf,
                        "kwargs": kwargs,
                    }
                else:
                    out["suggested_fix"] = None
            else:
                out["suggested_fix"] = None
            return out
        except Exception:
            return {
                "verdict": "reject",
                "score": 0.0,
                "reasons": ["llm_error"],
                "critique": None,
                "suggested_fix": None,
                "tone": self.persona,
            }


class JudgeManager:
    """Aggregate multiple judges and produce a final decision.

    Judges can be provided with weights. The aggregate score is a weighted
    average. If any judge returns 'reject' the manager can be configured to
    immediately reject or to use scores; by default it uses weighted score.
    """

    def __init__(
        self, judges: list[tuple[Judge, float]] | None = None, threshold: float = 0.7
    ) -> None:
        self.judges = judges or []
        self.threshold = threshold
        self.log: list[dict] = []

    def add(self, judge: Judge, weight: float = 1.0) -> None:
        self.judges.append((judge, float(weight)))

    def assess(self, result: dict) -> dict:
        if not self.judges:
            # default: accept if result says ok
            return {
                "verdict": "accept" if result.get("ok") else "reject",
                "score": 1.0 if result.get("ok") else 0.0,
                "reasons": [],
            }

        total_weight = sum(w for _, w in self.judges)
        if total_weight <= 0:
            total_weight = 1.0

        weighted_sum = 0.0
        details = []
        for judge, weight in self.judges:
            out = judge.evaluate(result)
            details.append({"judge": judge.__class__.__name__, "out": out})
            weighted_sum += out.get("score", 0.0) * weight

        agg_score = weighted_sum / total_weight
        final_verdict = "accept" if agg_score >= self.threshold else "reject"
        record = {
            "aggregate_score": agg_score,
            "verdict": final_verdict,
            "details": details,
        }
        self.log.append(record)
        return record
