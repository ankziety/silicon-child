"""LLM Jury evaluation system using multiple frontier models as judges."""

import json
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

import openai
from anthropic import Anthropic
from cohere import Client as CohereClient
from pydantic import BaseModel


class EvaluationResult(BaseModel):
    """Result from a single LLM judge."""
    
    judge_name: str
    score: float
    reasoning: str
    metadata: Optional[Dict[str, Any]] = None


class JuryResult(BaseModel):
    """Aggregated result from LLM jury evaluation."""
    
    candidate_score: float
    judge_results: List[EvaluationResult]
    aggregation_method: str
    seed: int
    metadata: Dict[str, Any]


class EvaluationError(Exception):
    """Raised when evaluation cannot be performed."""
    pass


class LLMJudge(Protocol):
    """Protocol for LLM judge evaluators."""
    
    def __call__(self, prompt: str, response: str, context: Optional[str] = None) -> EvaluationResult:
        """Evaluate a response and return a structured result."""
        ...


class GPT4oMiniJudge:
    """GPT-4o-mini based judge using OpenAI API (most affordable GPT-4 model)."""
    
    def __init__(self, name: str = "gpt4o_mini_judge", evaluation_type: str = "general"):
        self.name = name
        self.evaluation_type = evaluation_type
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EvaluationError("OPENAI_API_KEY environment variable not set")
        
        self.client = openai.OpenAI(api_key=api_key)
        
        self.prompt_templates = {
            "accuracy": """
You are an impartial judge evaluating factual accuracy. Score 0-1 where 1.0 is completely accurate.
Response: {response}
Context: {context}
Provide JSON: {{"score": <0-1>, "reasoning": "<explanation>"}}
""",
            "fluency": """
You are an impartial judge evaluating language fluency. Score 0-1 where 1.0 is perfectly fluent.
Response: {response}
Provide JSON: {{"score": <0-1>, "reasoning": "<explanation>"}}
""",
            "relevance": """
You are an impartial judge evaluating relevance to the prompt. Score 0-1 where 1.0 is completely relevant.
Prompt: {prompt}
Response: {response}
Provide JSON: {{"score": <0-1>, "reasoning": "<explanation>"}}
""",
            "coherence": """
You are an impartial judge evaluating logical coherence. Score 0-1 where 1.0 is perfectly coherent.
Response: {response}
Provide JSON: {{"score": <0-1>, "reasoning": "<explanation>"}}
""",
            "general": """
You are an impartial judge evaluating overall quality. Score 0-1 where 1.0 is excellent.
Prompt: {prompt}
Response: {response}
Provide JSON: {{"score": <0-1>, "reasoning": "<explanation>"}}
"""
        }

    def __call__(self, prompt: str, response: str, context: Optional[str] = None) -> EvaluationResult:
        """Evaluate using GPT-4o-mini."""
        try:
            template = self.prompt_templates[self.evaluation_type]
            formatted_prompt = template.format(
                prompt=prompt,
                response=response,
                context=context or "No additional context"
            )
            
            completion = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": formatted_prompt}],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            
            result = json.loads(completion.choices[0].message.content)
            
            return EvaluationResult(
                judge_name=self.name,
                score=float(result["score"]),
                reasoning=result["reasoning"],
                metadata={"evaluation_type": self.evaluation_type, "model": "gpt-4o-mini"}
            )
            
        except Exception as e:
            raise EvaluationError(f"GPT-4o-mini evaluation failed: {e}")


class GPT5Judge:
    """GPT-5 based judge using OpenAI API (latest and most capable model)."""
    
    def __init__(self, name: str = "gpt5_judge", evaluation_type: str = "general"):
        self.name = name
        self.evaluation_type = evaluation_type
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EvaluationError("OPENAI_API_KEY environment variable not set")
        
        self.client = openai.OpenAI(api_key=api_key)
        
        self.prompt_templates = {
            "accuracy": """
You are an impartial judge evaluating factual accuracy. Score 0-1 where 1.0 is completely accurate.
Response: {response}
Context: {context}
Provide JSON: {{"score": <0-1>, "reasoning": "<explanation>"}}
""",
            "fluency": """
You are an impartial judge evaluating language fluency. Score 0-1 where 1.0 is perfectly fluent.
Response: {response}
Provide JSON: {{"score": <0-1>, "reasoning": "<explanation>"}}
""",
            "relevance": """
You are an impartial judge evaluating relevance to the prompt. Score 0-1 where 1.0 is completely relevant.
Prompt: {prompt}
Response: {response}
Provide JSON: {{"score": <0-1>, "reasoning": "<explanation>"}}
""",
            "coherence": """
You are an impartial judge evaluating logical coherence. Score 0-1 where 1.0 is perfectly coherent.
Response: {response}
Provide JSON: {{"score": <0-1>, "reasoning": "<explanation>"}}
""",
            "general": """
You are an impartial judge evaluating overall quality. Score 0-1 where 1.0 is excellent.
Prompt: {prompt}
Response: {response}
Provide JSON: {{"score": <0-1>, "reasoning": "<explanation>"}}
"""
        }

    def __call__(self, prompt: str, response: str, context: Optional[str] = None) -> EvaluationResult:
        """Evaluate using GPT-5."""
        try:
            template = self.prompt_templates[self.evaluation_type]
            formatted_prompt = template.format(
                prompt=prompt,
                response=response,
                context=context or "No additional context"
            )
            
            completion = self.client.chat.completions.create(
                model="gpt-5",
                messages=[{"role": "user", "content": formatted_prompt}],
                response_format={"type": "json_object"},
                temperature=0.4
            )
            
            result = json.loads(completion.choices[0].message.content)
            
            return EvaluationResult(
                judge_name=self.name,
                score=float(result["score"]),
                reasoning=result["reasoning"],
                metadata={"evaluation_type": self.evaluation_type, "model": "gpt-5"}
            )
            
        except Exception as e:
            raise EvaluationError(f"GPT-5 evaluation failed: {e}")


class ClaudeHaikuJudge:
    """Claude Haiku based judge using Anthropic API (most affordable Claude model)."""
    
    def __init__(self, name: str = "claude_haiku_judge", evaluation_type: str = "general"):
        self.name = name
        self.evaluation_type = evaluation_type
        
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise EvaluationError("ANTHROPIC_API_KEY environment variable not set")
        
        self.client = Anthropic(api_key=api_key)
        
        self.prompt_templates = {
            "accuracy": """
You are an impartial judge evaluating factual accuracy. Score 0-1 where 1.0 is completely accurate.
Response: {response}
Context: {context}
Provide JSON: {{"score": <0-1>, "reasoning": "<explanation>"}}
""",
            "fluency": """
You are an impartial judge evaluating language fluency. Score 0-1 where 1.0 is perfectly fluent.
Response: {response}
Provide JSON: {{"score": <0-1>, "reasoning": "<explanation>"}}
""",
            "relevance": """
You are an impartial judge evaluating relevance to the prompt. Score 0-1 where 1.0 is completely relevant.
Prompt: {prompt}
Response: {response}
Provide JSON: {{"score": <0-1>, "reasoning": "<explanation>"}}
""",
            "coherence": """
You are an impartial judge evaluating logical coherence. Score 0-1 where 1.0 is perfectly coherent.
Response: {response}
Provide JSON: {{"score": <0-1>, "reasoning": "<explanation>"}}
""",
            "general": """
You are an impartial judge evaluating overall quality. Score 0-1 where 1.0 is excellent.
Prompt: {prompt}
Response: {response}
Provide JSON: {{"score": <0-1>, "reasoning": "<explanation>"}}
"""
        }

    def __call__(self, prompt: str, response: str, context: Optional[str] = None) -> EvaluationResult:
        """Evaluate using Claude Haiku."""
        try:
            template = self.prompt_templates[self.evaluation_type]
            formatted_prompt = template.format(
                prompt=prompt,
                response=response,
                context=context or "No additional context"
            )
            
            message = self.client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=1000,
                temperature=0.4,
                messages=[{"role": "user", "content": formatted_prompt}]
            )
            
            # Parse JSON from Claude's response
            content = message.content[0].text
            result = json.loads(content)
            
            return EvaluationResult(
                judge_name=self.name,
                score=float(result["score"]),
                reasoning=result["reasoning"],
                metadata={"evaluation_type": self.evaluation_type, "model": "claude-3-5-haiku"}
            )
            
        except Exception as e:
            raise EvaluationError(f"Claude Haiku evaluation failed: {e}")


class ClaudeSonnetJudge:
    """Claude Sonnet 4 based judge using Anthropic API (high performance Claude model)."""
    
    def __init__(self, name: str = "claude_sonnet_judge", evaluation_type: str = "general"):
        self.name = name
        self.evaluation_type = evaluation_type
        
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise EvaluationError("ANTHROPIC_API_KEY environment variable not set")
        
        self.client = Anthropic(api_key=api_key)
        
        self.prompt_templates = {
            "accuracy": """
You are an impartial judge evaluating factual accuracy. Score 0-1 where 1.0 is completely accurate.
Response: {response}
Context: {context}
Provide JSON: {{"score": <0-1>, "reasoning": "<explanation>"}}
""",
            "fluency": """
You are an impartial judge evaluating language fluency. Score 0-1 where 1.0 is perfectly fluent.
Response: {response}
Provide JSON: {{"score": <0-1>, "reasoning": "<explanation>"}}
""",
            "relevance": """
You are an impartial judge evaluating relevance to the prompt. Score 0-1 where 1.0 is completely relevant.
Prompt: {prompt}
Response: {response}
Provide JSON: {{"score": <0-1>, "reasoning": "<explanation>"}}
""",
            "coherence": """
You are an impartial judge evaluating logical coherence. Score 0-1 where 1.0 is perfectly coherent.
Response: {response}
Provide JSON: {{"score": <0-1>, "reasoning": "<explanation>"}}
""",
            "general": """
You are an impartial judge evaluating overall quality. Score 0-1 where 1.0 is excellent.
Prompt: {prompt}
Response: {response}
Provide JSON: {{"score": <0-1>, "reasoning": "<explanation>"}}
"""
        }

    def __call__(self, prompt: str, response: str, context: Optional[str] = None) -> EvaluationResult:
        """Evaluate using Claude Sonnet 4."""
        try:
            template = self.prompt_templates[self.evaluation_type]
            formatted_prompt = template.format(
                prompt=prompt,
                response=response,
                context=context or "No additional context"
            )
            
            message = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                temperature=0.1,
                messages=[{"role": "user", "content": formatted_prompt}]
            )
            
            # Parse JSON from Claude's response
            content = message.content[0].text
            result = json.loads(content)
            
            return EvaluationResult(
                judge_name=self.name,
                score=float(result["score"]),
                reasoning=result["reasoning"],
                metadata={"evaluation_type": self.evaluation_type, "model": "claude-3-5-sonnet"}
            )
            
        except Exception as e:
            raise EvaluationError(f"Claude Sonnet 4 evaluation failed: {e}")


class CommandRPlusJudge:
    """Command R+ based judge using Cohere API (most current and affordable Cohere model)."""
    
    def __init__(self, name: str = "commandr_plus_judge", evaluation_type: str = "general"):
        self.name = name
        self.evaluation_type = evaluation_type
        
        api_key = os.getenv("COHERE_API_KEY")
        if not api_key:
            raise EvaluationError("COHERE_API_KEY environment variable not set")
        
        self.client = CohereClient(api_key=api_key)
        
        self.prompt_templates = {
            "accuracy": """
You are an impartial judge evaluating factual accuracy. Score 0-1 where 1.0 is completely accurate.
Response: {response}
Context: {context}
Provide JSON: {{"score": <0-1>, "reasoning": "<explanation>"}}
""",
            "fluency": """
You are an impartial judge evaluating language fluency. Score 0-1 where 1.0 is perfectly fluent.
Response: {response}
Provide JSON: {{"score": <0-1>, "reasoning": "<explanation>"}}
""",
            "relevance": """
You are an impartial judge evaluating relevance to the prompt. Score 0-1 where 1.0 is completely relevant.
Prompt: {prompt}
Response: {response}
Provide JSON: {{"score": <0-1>, "reasoning": "<explanation>"}}
""",
            "coherence": """
You are an impartial judge evaluating logical coherence. Score 0-1 where 1.0 is perfectly coherent.
Response: {response}
Provide JSON: {{"score": <0-1>, "reasoning": "<explanation>"}}
""",
            "general": """
You are an impartial judge evaluating overall quality. Score 0-1 where 1.0 is excellent.
Prompt: {prompt}
Response: {response}
Provide JSON: {{"score": <0-1>, "reasoning": "<explanation>"}}
"""
        }

    def __call__(self, prompt: str, response: str, context: Optional[str] = None) -> EvaluationResult:
        """Evaluate using Command R+."""
        try:
            template = self.prompt_templates[self.evaluation_type]
            formatted_prompt = template.format(
                prompt=prompt,
                response=response,
                context=context or "No additional context"
            )
            
            response_cohere = self.client.chat(
                model="command-r-plus",
                message=formatted_prompt,
                temperature=0.1
            )
            
            # Parse JSON from Command R+'s response
            content = response_cohere.text
            result = json.loads(content)
            
            return EvaluationResult(
                judge_name=self.name,
                score=float(result["score"]),
                reasoning=result["reasoning"],
                metadata={"evaluation_type": self.evaluation_type, "model": "command-r-plus"}
            )
            
        except Exception as e:
            raise EvaluationError(f"Command R+ evaluation failed: {e}")


class LLMJury:
    """LLM Jury system using multiple frontier models as judges."""
    
    def __init__(self, judges: List[LLMJudge], aggregation_method: str = "average"):
        """Initialize jury with LLM judges."""
        if len(judges) < 3:
            raise ValueError("LLM Jury must have at least 3 judges")
        self.judges = judges
        self.aggregation_method = aggregation_method
    
    def evaluate(
        self, 
        prompt: str,
        response: str, 
        context: Optional[str] = None,
        seed: Optional[int] = None
    ) -> JuryResult:
        """Run all judges and return aggregated result."""
        if seed is not None:
            random.seed(seed)
        
        results = []
        errors = []
        
        for judge in self.judges:
            try:
                result = judge(prompt, response, context)
                results.append(result)
                
            except Exception as e:
                error_msg = f"Judge {judge.__class__.__name__} failed: {e}"
                errors.append(error_msg)
        
        # If any judges failed, raise an error with details
        if errors:
            raise EvaluationError(f"LLM Jury evaluation failed with errors: {'; '.join(errors)}")
        
        # Aggregate scores based on method
        scores = [r.score for r in results]
        
        if self.aggregation_method == "average":
            final_score = sum(scores) / len(scores)
        elif self.aggregation_method == "median":
            scores.sort()
            final_score = scores[len(scores) // 2]
        elif self.aggregation_method == "weighted_average":
            # Weight by judge reliability (could be configurable)
            weights = [1.0] * len(scores)  # Equal weights for now
            final_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")
        
        return JuryResult(
            candidate_score=final_score,
            judge_results=results,
            aggregation_method=self.aggregation_method,
            seed=seed or int(time.time()),
            metadata={
                "prompt": prompt,
                "response": response,
                "context": context,
                "judge_count": len(self.judges),
                "successful_evaluations": len(results)
            }
        )


def create_frontier_jury() -> LLMJury:
    """Create a jury with frontier model judges."""
    judges = [
        GPT5Judge("gpt5_accuracy", "accuracy"),
        ClaudeSonnetJudge("claude_sonnet_fluency", "fluency"),
        CommandRPlusJudge("commandr_plus_relevance", "relevance"),
        GPT5Judge("gpt5_coherence", "coherence"),
        ClaudeSonnetJudge("claude_sonnet_general", "general")
    ]
    return LLMJury(judges, aggregation_method="average")


def create_diverse_jury() -> LLMJury:
    """Create a diverse jury with different model families."""
    judges = [
        GPT5Judge("gpt5_general", "general"),
        ClaudeSonnetJudge("claude_sonnet_general", "general"),
        CommandRPlusJudge("commandr_plus_general", "general")
    ]
    return LLMJury(judges, aggregation_method="median")


def create_affordable_jury() -> LLMJury:
    """Create an affordable jury with cost-effective models."""
    judges = [
        GPT4oMiniJudge("gpt4o_mini_accuracy", "accuracy"),
        ClaudeHaikuJudge("claude_haiku_fluency", "fluency"),
        CommandRPlusJudge("commandr_plus_relevance", "relevance"),
        GPT4oMiniJudge("gpt4o_mini_coherence", "coherence"),
        ClaudeHaikuJudge("claude_haiku_general", "general")
    ]
    return LLMJury(judges, aggregation_method="average")


def create_specialized_jury() -> LLMJury:
    """Create a specialized jury for specific evaluation criteria."""
    judges = [
        GPT5Judge("gpt5_accuracy", "accuracy"),
        GPT5Judge("gpt5_fluency", "fluency"),
        GPT5Judge("gpt5_relevance", "relevance"),
        GPT5Judge("gpt5_coherence", "coherence")
    ]
    return LLMJury(judges, aggregation_method="weighted_average")


def create_mixed_jury() -> LLMJury:
    """Create a mixed jury with both high-performance and affordable models."""
    judges = [
        GPT5Judge("gpt5_accuracy", "accuracy"),
        ClaudeHaikuJudge("claude_haiku_fluency", "fluency"),
        CommandRPlusJudge("commandr_plus_relevance", "relevance"),
        GPT4oMiniJudge("gpt4o_mini_coherence", "coherence"),
        ClaudeSonnetJudge("claude_sonnet_general", "general")
    ]
    return LLMJury(judges, aggregation_method="average")
