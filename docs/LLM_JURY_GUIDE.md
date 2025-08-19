# LLM Jury Evaluation System - Technical Guide for Developers

## Overview

This document provides technical implementation details for the LLM Jury evaluation system. It's designed for developers who need to understand, maintain, extend, or debug the evaluation system.

## Architecture

### Core Components

```
ai_infant/learn/eval.py
├── LLMJudge (Protocol)
├── Judge Classes
│   ├── GPT4oMiniJudge
│   ├── GPT5Judge
│   ├── ClaudeHaikuJudge
│   ├── ClaudeSonnetJudge
│   └── CommandRPlusJudge
├── LLMJury (orchestration)
└── Factory Functions
    ├── create_frontier_jury()
    ├── create_affordable_jury()
    ├── create_diverse_jury()
    ├── create_specialized_jury()
    └── create_mixed_jury()

scripts/promote.py
├── PromotionManager
├── AdapterInfo
└── Ring Buffer Logic
```

### Data Flow

1. **Input**: `prompt`, `response`, `context`, `seed`
2. **Judge Evaluation**: Each judge calls respective API with structured prompt
3. **Response Parsing**: JSON response parsed to extract score and reasoning
4. **Aggregation**: Scores combined using selected method
5. **Logging**: Results logged to JobV1 schema via PromotionManager
6. **Output**: `JuryResult` with aggregated score and individual judge results

## Implementation Details

### Judge Protocol

All judges implement the `LLMJudge` protocol:

```python
class LLMJudge(Protocol):
    def __call__(self, prompt: str, response: str, context: Optional[str] = None) -> EvaluationResult:
        """Evaluate a response and return a structured result."""
        ...
```

### Judge Class Structure

Each judge class follows this pattern:

```python
class GPT4oMiniJudge:
    def __init__(self, name: str, evaluation_type: str):
        # API key validation
        # Client initialization
        # Prompt template setup
    
    def __call__(self, prompt: str, response: str, context: Optional[str] = None) -> EvaluationResult:
        # Template formatting
        # API call
        # Response parsing
        # Error handling
```

### Error Handling Strategy

**Fail-Hard Principle**: Any judge failure stops the entire evaluation.

```python
# In LLMJury.evaluate()
for judge in self.judges:
    try:
        result = judge(prompt, response, context)
        results.append(result)
    except Exception as e:
        error_msg = f"Judge {judge.__class__.__name__} failed: {e}"
        errors.append(error_msg)

if errors:
    raise EvaluationError(f"LLM Jury evaluation failed with errors: {'; '.join(errors)}")
```

### API Integration Patterns

#### OpenAI Judges
```python
completion = self.client.chat.completions.create(
    model="gpt-4o-mini",  # or "gpt-5"
    messages=[{"role": "user", "content": formatted_prompt}],
    response_format={"type": "json_object"},
    temperature=0.1
)
```

#### Anthropic Judges
```python
message = self.client.messages.create(
    model="claude-3-5-haiku-20241022",  # or "claude-3-5-sonnet-20241022"
    max_tokens=1000,
    temperature=0.1,
    messages=[{"role": "user", "content": formatted_prompt}]
)
```

#### Cohere Judge
```python
response_cohere = self.client.chat(
    model="command-r-plus",
    message=formatted_prompt,
    temperature=0.1
)
```

## Adding New Judges

### Step 1: Create Judge Class

```python
class NewModelJudge:
    def __init__(self, name: str = "new_model_judge", evaluation_type: str = "general"):
        self.name = name
        self.evaluation_type = evaluation_type
        
        api_key = os.getenv("NEW_MODEL_API_KEY")
        if not api_key:
            raise EvaluationError("NEW_MODEL_API_KEY environment variable not set")
        
        self.client = NewModelClient(api_key=api_key)
        
        # Define prompt templates
        self.prompt_templates = {
            "accuracy": """...""",
            "fluency": """...""",
            # ... other templates
        }

    def __call__(self, prompt: str, response: str, context: Optional[str] = None) -> EvaluationResult:
        try:
            template = self.prompt_templates[self.evaluation_type]
            formatted_prompt = template.format(
                prompt=prompt,
                response=response,
                context=context or "No additional context"
            )
            
            # Make API call
            api_response = self.client.evaluate(formatted_prompt)
            
            # Parse response
            result = json.loads(api_response.content)
            
            return EvaluationResult(
                judge_name=self.name,
                score=float(result["score"]),
                reasoning=result["reasoning"],
                metadata={"evaluation_type": self.evaluation_type, "model": "new-model"}
            )
            
        except Exception as e:
            raise EvaluationError(f"New model evaluation failed: {e}")
```

### Step 2: Add to Factory Functions

```python
def create_new_jury() -> LLMJury:
    """Create a jury with the new model."""
    judges = [
        NewModelJudge("new_model_accuracy", "accuracy"),
        NewModelJudge("new_model_fluency", "fluency"),
        # ... other judges
    ]
    return LLMJury(judges, aggregation_method="average")
```

### Step 3: Update Imports

```python
# In ai_infant/learn/__init__.py
from .eval import NewModelJudge, create_new_jury

__all__ = [..., "NewModelJudge", "create_new_jury"]
```

### Step 4: Add Tests

```python
# In tests/test_eval_promote.py
@patch.dict(os.environ, {"NEW_MODEL_API_KEY": "test_key"})
def test_new_model_judge_initialization(self):
    """Test new model judge initialization."""
    judge = NewModelJudge("test_judge", "general")
    assert judge.name == "test_judge"
    assert judge.evaluation_type == "general"
```

## Aggregation Methods

### Current Implementations

```python
# Average
final_score = sum(scores) / len(scores)

# Median
scores.sort()
final_score = scores[len(scores) // 2]

# Weighted Average
weights = [1.0] * len(scores)  # Equal weights for now
final_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
```

### Adding New Aggregation Methods

```python
elif self.aggregation_method == "custom_method":
    # Implement custom aggregation logic
    final_score = custom_aggregation_function(scores)
```

## Promotion System

### Ring Buffer Implementation

```python
# In PromotionManager
self.adapters: deque = deque(maxlen=max_adapters)
```

### Adapter Storage

```python
# File: data/adapters.json
{
    "adapters": [
        {
            "model_path": "model_v1",
            "score": 0.85,
            "timestamp": "2024-01-01T12:00:00Z"
        }
    ],
    "last_updated": "2024-01-01T12:00:00Z"
}
```

### Rollback Logic

```python
def rollback_to_adapter(self, model_path: str) -> bool:
    """Rollback to specific adapter, removing newer ones."""
    # Find adapter in history
    # Remove all adapters after it
    # Update storage
```

## Performance Considerations

### API Rate Limiting

- All judges use `temperature=0.1` for deterministic results
- Consider implementing exponential backoff for rate limits
- Use affordable jury for development/testing

### Cost Optimization

```python
# Cost comparison (per 1K tokens)
COSTS = {
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-5": {"input": 0.005, "output": 0.015},
    "claude-3-5-haiku": {"input": 0.00025, "output": 0.00125},
    "claude-3-5-sonnet": {"input": 0.003, "output": 0.015},
    "command-r-plus": {"input": 0.0005, "output": 0.0015}
}
```

## Debugging

### Common Issues

1. **API Key Issues**
   ```python
   # Check environment variables
   print(os.getenv("OPENAI_API_KEY"))
   ```

2. **JSON Parsing Errors**
   ```python
   # Add debug logging
   print(f"Raw response: {api_response.content}")
   ```

3. **Rate Limiting**
   ```python
   # Implement retry logic
   import time
   time.sleep(1)  # Basic rate limiting
   ```

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable detailed logging
result = jury.evaluate(prompt, response, context)
```

## Job Logging

### JobV1 Schema Integration

```python
# In PromotionManager._log_job()
job_data = {
    "id": job_id,
    "type": "eval",  # or "promote"
    "status": "completed",  # or "failed"
    "input": input_data,
    "output": output_data,
    "error": error_data,
    "metadata": {...}
}
```

### Querying Evaluation History

```python
# Get all evaluation jobs
jobs = store.conn.execute("SELECT * FROM jobs WHERE type = 'eval'").fetchall()

# Get failed evaluations
failed_jobs = store.conn.execute(
    "SELECT * FROM jobs WHERE type = 'eval' AND status = 'failed'"
).fetchall()
```

## Future Enhancements

### Planned Features

1. **Confidence Scoring**: Add confidence metrics to judge responses
2. **Adaptive Aggregation**: Dynamic weight adjustment based on judge reliability
3. **Batch Processing**: Parallel evaluation of multiple responses
4. **Cost Monitoring**: Real-time cost tracking and alerts
5. **Model Performance Tracking**: Track judge accuracy over time

### Extension Points

1. **New Evaluation Criteria**: Extend prompt templates
2. **Custom Aggregation**: Implement new aggregation methods
3. **Storage Backends**: Support different databases
4. **API Providers**: Add new LLM providers
5. **Evaluation Metrics**: Add custom evaluation dimensions

## Dependencies

### Required Packages

```toml
# pyproject.toml
dependencies = [
    "openai>=1.0.0",
    "anthropic>=0.25.0", 
    "cohere>=4.0.0",
    "pydantic>=2.0.0",
    "duckdb>=0.9.0"
]
```

### Environment Variables

```bash
export OPENAI_API_KEY="your_openai_key"
export ANTHROPIC_API_KEY="your_anthropic_key"
export COHERE_API_KEY="your_cohere_key"
```

## References

- [LLM Juries for Evaluation](https://www.comet.com/site/blog/llm-juries-for-evaluation/)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Anthropic API Documentation](https://docs.anthropic.com/)
- [Cohere API Documentation](https://docs.cohere.com/)
- [JobV1 Schema](../schemas/job.v1.json)
