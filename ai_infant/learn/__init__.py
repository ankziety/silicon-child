"""Learning and evaluation module for AI-Infant research agent."""

from .eval import (
    ClaudeHaikuJudge,
    ClaudeSonnetJudge,
    CommandRPlusJudge,
    EvaluationError,
    EvaluationResult,
    GPT4oMiniJudge,
    GPT5Judge,
    LLMJury,
    create_affordable_jury,
    create_diverse_jury,
    create_frontier_jury,
    create_mixed_jury,
    create_specialized_jury,
)

__all__ = [
    "LLMJury",
    "EvaluationResult",
    "EvaluationError",
    "GPT4oMiniJudge",
    "GPT5Judge",
    "ClaudeHaikuJudge",
    "ClaudeSonnetJudge",
    "CommandRPlusJudge",
    "create_frontier_jury",
    "create_diverse_jury",
    "create_affordable_jury",
    "create_specialized_jury",
    "create_mixed_jury",
]
