import copy
from typing import Any, Callable, Optional

from ..judge.interface import JudgeManager
from ..llm.client import make_llm_client


def perform_action_with_llm(
    browser_tool: Any,
    action_type: str,
    selector: str,
    confidence: float,
    llm_client: Optional[Callable[[str], str]] = None,
    max_retries: int = 2,
    judge_manager: Optional[JudgeManager] = None,
) -> dict:
    """Perform an action via the browser tool, ask for the other LLMs help, to retry on low-confidence.

    Returns a structured dict with at least the keys: `ok` (bool), and on failure
    `reason` (str) and optional `details`.
    """
    # Try to use the browser_tool's structured API when available
    if hasattr(browser_tool, "execute_action_struct"):
        execute_struct = browser_tool.execute_action_struct
    else:
        # Fallback wrapper: call execute_action and synthesize a struct
        def execute_struct(a_type, sel, conf, **kwargs):
            ok = browser_tool.execute_action(a_type, sel, conf, **kwargs)
            if ok:
                return {"ok": True, "result": {}}
            return {"ok": False, "reason": "unknown", "details": {}}

    # Attempt the initial action
    result = execute_struct(action_type, selector, confidence)
    if result.get("ok"):
        return result

    # If a judge_manager is provided, assess the result and log the assessment.
    suggested_from_judges = None
    if judge_manager is not None:
        assessment = judge_manager.assess(result)
        # Store assessment and critique in browser action history if available
        try:
            b = getattr(browser_tool, "_browser", None) or (
                browser_tool._ensure() if hasattr(browser_tool, "_ensure") else None
            )
            if b is not None and hasattr(b, "action_history"):
                # append a deep copy to avoid mutation issues
                b.action_history.append({"judge_assessment": copy.deepcopy(assessment)})
        except Exception:
            pass

        # If judge accepts, return success
        if assessment.get("verdict") == "accept":
            return result

        # Collect any suggested_fix from judges; prefer first valid one
        for d in assessment.get("details", []):
            out = d.get("out", {})
            if out.get("suggested_fix"):
                candidate = out.get("suggested_fix")
                # basic sanitization
                if isinstance(candidate, dict):
                    at = candidate.get("action_type")
                    sel = candidate.get("selector")
                    conf = candidate.get("confidence")
                    if (
                        isinstance(at, str)
                        and isinstance(sel, str)
                        and isinstance(conf, (int, float))
                    ):
                        # enforce confidence bounds
                        if 0.0 <= float(conf) <= 1.0:
                            suggested_from_judges = {
                                "action_type": at,
                                "selector": sel,
                                "confidence": float(conf),
                                "kwargs": candidate.get("kwargs", {}),
                            }
                            break

    # If failure isn't low_confidence we won't ask LLM to suggest a retry
    reason = result.get("reason", "")
    if reason != "low_confidence":
        return result

    # If judges suggested a valid fix, attempt it first
    if suggested_from_judges is not None:
        s = suggested_from_judges
        # enforce non-destructive policy: no navigate with high risk (limit confidence)
        if s.get("action_type") == "navigate" and s.get("confidence", 1.0) > 0.9:
            s["confidence"] = 0.9

        res = execute_struct(
            s.get("action_type"),
            s.get("selector"),
            s.get("confidence", confidence),
            **s.get("kwargs", {}),
        )
        # log into history
        try:
            b = getattr(browser_tool, "_browser", None) or (
                browser_tool._ensure() if hasattr(browser_tool, "_ensure") else None
            )
            if b is not None and hasattr(b, "action_history"):
                b.action_history.append(
                    {"applied_suggested_fix": s, "ok": res.get("ok", False)}
                )
        except Exception:
            pass

        if res.get("ok"):
            return res

    # If failure isn't low_confidence we won't ask LLM to suggest a retry
    reason = result.get("reason", "")
    if reason != "low_confidence":
        return result

    if llm_client is None:
        return result

    # Create a simple client wrapper using factory to ensure consistent retry/backoff
    client = (
        llm_client
        if callable(llm_client)
        else make_llm_client("openai", openai_module=llm_client)
    )

    # Attempt retries with llm client guidance
    attempts = 0
    while attempts < max_retries:
        attempts += 1
        # Build a concise prompt for the client
        prompt = (
            "The automated browser attempted an action and it was rejected due to low confidence.\n"
            f'Failed action: {{"action_type": "{action_type}", "selector": "{selector}", "confidence": {confidence}}}\n'
            f"Failure reason: {reason}\n"
            f"Retry attempt: {attempts}\n"
            "Return only a JSON object with keys: action_type, selector, confidence (0..1), and optional kwargs."
        )

        try:
            reply = client(prompt)
        except Exception:
            break

        # Try to parse JSON conservatively
        try:
            import json as _json

            suggestion = _json.loads(reply)
        except Exception:
            # attempt to extract braced JSON substring
            import re as _re

            m = _re.search(r"\{[\s\S]*\}", reply)
            if not m:
                suggestion = None
            else:
                try:
                    suggestion = _json.loads(m.group(0))
                except Exception:
                    suggestion = None

        if not suggestion:
            break

        at = suggestion.get("action_type")
        sel = suggestion.get("selector")
        conf = suggestion.get("confidence")
        kwargs_s = suggestion.get("kwargs", {}) or {}

        # Basic validation and sanitization
        if not isinstance(at, str) or not isinstance(sel, str):
            continue
        try:
            conf = float(conf)
        except Exception:
            conf = confidence
        if conf < 0.0:
            conf = 0.0
        if conf > 1.0:
            conf = 1.0

        # Enforce non-destructive policies: cap navigate confidence and disallow script execution
        if at == "navigate" and conf > 0.95:
            conf = 0.95

        # Execute suggested action
        result = execute_struct(at, sel, conf, **kwargs_s)

        # Log suggestion into browser action history if possible
        try:
            browser = getattr(browser_tool, "_browser", None) or (
                browser_tool._ensure() if hasattr(browser_tool, "_ensure") else None
            )
            if browser is not None and hasattr(browser, "action_history"):
                browser.action_history.append(
                    {
                        "action": at,
                        "selector": sel,
                        "confidence": conf,
                        "llm_retry": attempts,
                        "ok": result.get("ok", False),
                    }
                )
        except Exception:
            pass

        if result.get("ok"):
            return result

    return {"ok": False, "reason": "llm_retries_exhausted", "details": {}}
